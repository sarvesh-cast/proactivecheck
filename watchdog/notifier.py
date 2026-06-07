"""Notifier module — posts findings to Slack.

Formats evaluation results into Slack Block Kit messages and posts via
incoming webhook. Implements deduplication (same category + workload
within 30 minutes) to prevent alert fatigue.

Handles: webhook failures (log + retry next cycle), daily summary
posting at 08:00 UTC, and message truncation for Slack's limits.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import httpx

from .config import WatchdogConfig
from .models import (
    DedupKey,
    EvaluationResult,
    Finding,
    FindingCategory,
    Severity,
    Verdict,
)
from .state import StateManager

logger = logging.getLogger("watchdog.notifier")

# Slack message size limit (3000 chars for text blocks)
SLACK_TEXT_LIMIT = 2900

# Verdict → emoji mapping
VERDICT_EMOJI = {
    Verdict.CRITICAL: "\U0001f6a8",   # rotating light
    Verdict.WARNING: "⚠️",  # warning sign
    Verdict.HEALTHY: "✅",        # green check
}

SEVERITY_EMOJI = {
    Severity.CRITICAL: "\U0001f534",  # red circle
    Severity.WARNING: "\U0001f7e0",   # orange circle
    Severity.INFO: "\U0001f535",      # blue circle
}

# Human-readable one-liner for each finding category shown in Slack alerts
# Static fallback — used only if _category_description() isn't called.
CATEGORY_DESCRIPTION: dict[str, str] = {}


class Notifier:
    """Posts evaluation results to Slack with deduplication."""

    def __init__(self, config: WatchdogConfig, state: StateManager) -> None:
        self.config = config
        self.state = state
        self.webhook_url = config.slack.webhook_url
        self.dedup_minutes = config.slack.dedup_window_minutes
        # Build config-aware descriptions so alerts explain the exact trigger condition
        self._build_category_descriptions()

    def _build_category_descriptions(self) -> None:
        """Build alert descriptions that include the exact trigger condition."""
        t = self.config.thresholds
        self._cat_descs: dict[str, str] = {
            FindingCategory.OOMKILL.value: (
                f"Triggered when a pod is OOMKilled ≥{t.oomkill_critical_per_hour} times "
                f"within the last hour"
            ),
            FindingCategory.CRASHLOOP.value: (
                "Triggered when a pod is stuck in CrashLoopBackOff"
            ),
            FindingCategory.MISMATCH.value: (
                f"Triggered when WOOP recommendation differs from applied resources "
                f"by >{t.recommendation_mismatch_pct:.0f}%"
            ),
            FindingCategory.UNSCHEDULABLE.value: (
                f"Triggered when a pod is stuck in Pending for >{t.pending_pod_minutes} min"
            ),
            FindingCategory.AGENT.value: (
                "Triggered when CAST AI agent pod is not Running or heartbeat is stale (>10 min)"
            ),
            FindingCategory.DATA_GAP.value: (
                f"Triggered when a workload has optimization enabled but no recommendation "
                f"for >{t.data_gap_hours}h"
            ),
            FindingCategory.MEMORY_LEAK.value: (
                "Triggered when memory usage is monotonically increasing across "
                "3+ consecutive snapshots (>5% growth)"
            ),
            FindingCategory.ABSURD_RECOMMENDATION.value: (
                f"Triggered when WOOP recommendation exceeds {t.absurd_memory_gib} GiB memory "
                f"or {t.absurd_cpu_cores} CPU cores"
            ),
            FindingCategory.AGENT_RESTART.value: (
                f"Triggered when CAST AI agent pod restarts "
                f">{t.agent_restart_critical_per_hour} times/hour"
            ),
            FindingCategory.WEBHOOK_FAILURE.value: (
                "Triggered when workload autoscaler admission webhook is not responding"
            ),
            FindingCategory.CASCADING_SCALING.value: (
                "Triggered when node or pod count spikes >50% in 30 min "
                "without a matching deployment change"
            ),
            FindingCategory.UNHEALTHY_DEPLOYMENT.value: (
                "Triggered when a deployment has pods in CrashLoopBackOff or failing readiness"
            ),
        }
        # Update the module-level dict for backward compatibility
        CATEGORY_DESCRIPTION.update(self._cat_descs)

    def _cat_desc(self, category: str) -> str:
        """Get category description with trigger condition."""
        return self._cat_descs.get(category, "")

    async def notify(self, result: EvaluationResult, dry_run: bool = False) -> None:
        """Process evaluation result and post to Slack if warranted.

        Rules:
        - HEALTHY → silence (no post except daily summary)
        - WARNING/CRITICAL → post, but skip if deduplicated
        - Daily summary at 08:00 UTC regardless of verdict
        """
        if not self.webhook_url and not dry_run:
            logger.warning("No Slack webhook URL configured, skipping notification")
            return

        # Check if it's time for daily summary
        now = datetime.now(timezone.utc)
        if now.hour == self.config.slack.daily_summary_hour_utc and now.minute < 6:
            await self._post_daily_summary(result, dry_run)

        # Only post for actionable findings
        if not result.has_actionable_findings():
            logger.info("Verdict: HEALTHY — no notification needed")
            return

        # Filter findings through dedup
        new_findings = []
        for finding in result.findings:
            if finding.severity == Severity.INFO:
                continue  # don't post info-level findings

            key = DedupKey.from_finding(finding)
            if self.state.should_notify(key, self.dedup_minutes):
                new_findings.append(finding)
                self.state.record_notification(key)
            else:
                logger.info("Deduplicated: %s on %s", finding.category, finding.workload)

        if not new_findings:
            logger.info("All findings deduplicated, skipping notification")
            return

        # Build and send the message
        message = self._format_message(result.verdict, result.summary, new_findings, result.evaluated_at)

        if dry_run:
            logger.info("[DRY RUN] Would post to Slack:\n%s", json.dumps(message, indent=2))
            return

        await self._post_to_slack(message)

    def _cluster_header(self) -> str:
        """Return a formatted cluster identifier: name + short ID."""
        name = self.config.cluster.cluster_name
        cid = self.config.cluster.cluster_id
        short_id = cid[:8] if cid else "unknown"
        if name and name != cid[:8]:
            return f"{name} (`{short_id}`)"
        return f"`{cid}`"

    def _console_link(self) -> str:
        """Return a CAST AI console link for the cluster."""
        cid = self.config.cluster.cluster_id
        org = self.config.castai.organization_id
        if cid and org:
            return f"https://console.cast.ai/external-clusters/{cid}/overview?organizationId={org}"
        return ""

    async def _post_daily_summary(
        self, result: EvaluationResult, dry_run: bool
    ) -> None:
        """Post a daily summary at 08:00 UTC regardless of verdict."""
        emoji = VERDICT_EMOJI[result.verdict]
        m = result.metrics
        cluster = self._cluster_header()
        console = self._console_link()

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} Daily Summary — {self.config.cluster.cluster_name or self.config.cluster.cluster_id[:8]}"}
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Cluster:* {cluster}\n"
                        f"*Verdict:* {result.verdict.value}\n"
                        f"*Summary:* {result.summary}"
                    ),
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Pods:* {m.total_pods} total"},
                    {"type": "mrkdwn", "text": f"*Nodes:* {m.node_count} ({m.node_count_delta_pct:+.1f}%)"},
                    {"type": "mrkdwn", "text": f"*OOMKilled:* {m.oomkilled_pods}"},
                    {"type": "mrkdwn", "text": f"*Pending:* {m.pending_pods}"},
                    {"type": "mrkdwn", "text": f"*CrashLoop:* {m.crashloop_pods}"},
                    {"type": "mrkdwn", "text": f"*Agent restarts:* {m.agent_restarts_last_hour}/hr"},
                    {"type": "mrkdwn", "text": f"*WOOP mismatches:* {m.recommendation_mismatches}"},
                    {"type": "mrkdwn", "text": f"*Absurd recs:* {m.absurd_recommendations}"},
                ],
            },
        ]

        if result.findings:
            finding_lines = []
            for f in result.findings[:5]:
                finding_lines.append(f"{SEVERITY_EMOJI[f.severity]} `{f.category.value}` — {f.what}")
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Active findings ({len(result.findings)}):*\n" + "\n".join(finding_lines)},
            })

        if console:
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"<{console}|Open in CAST AI Console> · {result.evaluated_at}"}],
            })
        else:
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"Snapshot at {result.evaluated_at}"}],
            })

        message = {"blocks": blocks, "text": f"{emoji} Daily Summary — {self.config.cluster.cluster_name}"}

        if dry_run:
            logger.info("[DRY RUN] Daily summary:\n%s", json.dumps(message, indent=2))
            return

        await self._post_to_slack(message)

    def _format_message(
        self,
        verdict: Verdict,
        summary: str,
        findings: list[Finding],
        evaluated_at: str = "",
    ) -> dict:
        """Format findings into a compact Slack Block Kit message with tabular findings."""
        emoji = VERDICT_EMOJI[verdict]
        cluster = self._cluster_header()
        console = self._console_link()
        cluster_label = self.config.cluster.cluster_name or self.config.cluster.cluster_id[:8]

        # Format timestamp as short UTC time (e.g. "15:14 UTC")
        ts_short = ""
        if evaluated_at:
            try:
                dt = datetime.fromisoformat(evaluated_at.replace("Z", "+00:00"))
                ts_short = dt.strftime("%H:%M UTC")
            except (ValueError, AttributeError):
                ts_short = ""

        title = f"{emoji} {verdict.value} — {cluster_label}"
        if ts_short:
            title += f" · {ts_short}"

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": title},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Cluster:* {cluster}\n{summary}"},
            },
            {"type": "divider"},
        ]

        # Sort findings: CRITICAL first, then WARNING; within each severity,
        # use a fixed category priority so the most actionable alerts appear first.
        _SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        _CATEGORY_ORDER = {
            # Critical-tier (appear first)
            FindingCategory.OOMKILL.value: 0,
            FindingCategory.CRASHLOOP.value: 1,
            FindingCategory.ABSURD_RECOMMENDATION.value: 2,
            FindingCategory.MISMATCH.value: 3,
            FindingCategory.CASCADING_SCALING.value: 4,
            FindingCategory.AGENT.value: 5,
            FindingCategory.AGENT_RESTART.value: 6,
            FindingCategory.UNSCHEDULABLE.value: 7,
            FindingCategory.WEBHOOK_FAILURE.value: 8,
            # Warning-tier
            FindingCategory.UNHEALTHY_DEPLOYMENT.value: 9,
            FindingCategory.MEMORY_LEAK.value: 10,
            FindingCategory.DATA_GAP.value: 11,
        }
        sorted_findings = sorted(
            [f for f in findings if f.severity != Severity.INFO],
            key=lambda f: (
                _SEVERITY_ORDER.get(f.severity, 9),
                _CATEGORY_ORDER.get(f.category.value, 99),
            ),
        )

        # Group consecutive findings by (category, severity) for compact display
        groups: list[tuple[str, str, list[Finding]]] = []
        for finding in sorted_findings:
            cat_key = finding.category.value
            sev_key = finding.severity.value
            if groups and groups[-1][0] == cat_key and groups[-1][1] == sev_key:
                groups[-1][2].append(finding)
            else:
                groups.append((cat_key, sev_key, [finding]))

        rows = []
        cluster_name = self.config.cluster.cluster_name or self.config.cluster.cluster_id[:8]
        for cat_key, sev_key, group_findings in groups:
            sev_emoji = SEVERITY_EMOJI[Severity(sev_key)]

            # Route each category to its dedicated table renderer
            if cat_key in (
                FindingCategory.ABSURD_RECOMMENDATION.value,
                FindingCategory.MISMATCH.value,
            ):
                cat_desc = CATEGORY_DESCRIPTION.get(cat_key, "")
                desc_line = f"\n_{cat_desc}_" if cat_desc else ""
                rows.extend(
                    self._format_woop_table(
                        cat_key, sev_emoji, desc_line, group_findings
                    )
                )
            elif cat_key == FindingCategory.OOMKILL.value:
                rows.append(self._render_oomkill_table(sev_emoji, cluster_name, group_findings))
            elif cat_key == FindingCategory.CRASHLOOP.value:
                rows.append(self._render_crashloop_table(sev_emoji, cluster_name, group_findings))
            elif cat_key == FindingCategory.UNSCHEDULABLE.value:
                rows.append(self._render_unschedulable_table(sev_emoji, cluster_name, group_findings))
            elif cat_key in (FindingCategory.AGENT.value, FindingCategory.AGENT_RESTART.value):
                rows.append(self._render_agent_table(sev_emoji, cluster_name, cat_key, group_findings))
            elif cat_key == FindingCategory.DATA_GAP.value:
                rows.append(self._render_data_gap_table(sev_emoji, cluster_name, group_findings))
            elif cat_key == FindingCategory.MEMORY_LEAK.value:
                rows.append(self._render_memory_leak_table(sev_emoji, cluster_name, group_findings))
            elif cat_key == FindingCategory.CASCADING_SCALING.value:
                rows.append(self._render_cascading_table(sev_emoji, cluster_name, group_findings))
            elif cat_key == FindingCategory.WEBHOOK_FAILURE.value:
                rows.append(self._render_webhook_table(sev_emoji, cluster_name, group_findings))
            elif cat_key == FindingCategory.UNHEALTHY_DEPLOYMENT.value:
                rows.append(self._render_unhealthy_table(sev_emoji, cluster_name, group_findings))
            else:
                rows.append(self._render_generic_table(sev_emoji, cluster_name, cat_key, group_findings))

        if rows:
            # Split into chunks to stay under Slack's 3000-char section limit
            for i in range(0, len(rows)):
                text = rows[i]
                if len(text) > SLACK_TEXT_LIMIT:
                    text = text[:SLACK_TEXT_LIMIT - 20] + "\n_...truncated_"
                blocks.append({
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": text},
                })

        # Footer
        footer_parts = []
        if console:
            footer_parts.append(f"<{console}|Console>")
        footer_parts.append(datetime.now(timezone.utc).strftime("%H:%M UTC"))
        footer_parts.append("Next in 5m")

        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": " · ".join(footer_parts)}],
        })

        return {
            "blocks": blocks,
            "text": f"{emoji} {verdict.value} — {cluster_label}: {summary}",
        }

    @staticmethod
    def _extract_affected_workloads(evidence: str) -> str:
        """Extract workload names from evidence JSON for cluster-level findings."""
        try:
            data = json.loads(evidence)
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(data, list):
            return ""
        names = []
        for item in data[:5]:
            if isinstance(item, dict):
                # Try common keys: workload, namespace/name
                wl = item.get("workload", "")
                if not wl:
                    ns = item.get("namespace", "")
                    name = item.get("name", "")
                    wl = f"{ns}/{name}" if ns and name else name or ns
                if wl:
                    names.append(f"`{wl}`")
        if not names:
            return ""
        result = ", ".join(names)
        total = len(data)
        if total > 5:
            result += f" _+{total - 5} more_"
        return result

    @staticmethod
    def _fmt_gib(gib: float) -> str:
        """Format GiB value as human-readable GiB or MiB."""
        if gib >= 1.0:
            return f"{round(gib, 1)} GiB"
        return f"{round(gib * 1024, 1)} MiB"

    @staticmethod
    def _make_table(headers: list[str], rows: list[list[str]], max_col: int = 120) -> str:
        """Build a monospace-aligned table inside a Slack code block.

        Each column is auto-sized to fit the widest value (capped at max_col).
        Returns a string wrapped in triple backticks.
        """
        if not rows:
            return "```\n(no data)\n```"

        # Ensure all rows have the same number of columns as headers
        n = len(headers)
        safe_rows = []
        for r in rows:
            padded = list(r) + [""] * (n - len(r))
            safe_rows.append(padded[:n])

        # Calculate column widths — auto-fit to content
        widths = [min(max(len(headers[i]), max(len(r[i]) for r in safe_rows)), max_col)
                  for i in range(n)]

        def _row(cells: list[str]) -> str:
            parts = []
            for i, cell in enumerate(cells):
                text = cell[:widths[i]]
                parts.append(text.ljust(widths[i]))
            return " │ ".join(parts)

        separator = "─┼─".join("─" * w for w in widths)
        lines = [_row(headers), separator]
        for r in safe_rows:
            lines.append(_row(r))
        return "```\n" + "\n".join(lines) + "\n```"

    @staticmethod
    def _trunc(text: str, max_len: int = 40) -> str:
        """Truncate a string for table cells."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 2] + ".."

    @staticmethod
    def _workload_name(full: str) -> str:
        """Extract workload name from 'namespace/name', returns name part."""
        return full.split("/", 1)[1] if "/" in full else full

    @staticmethod
    def _workload_ns(full: str) -> str:
        """Extract namespace from 'namespace/name'."""
        return full.split("/", 1)[0] if "/" in full else ""

    @staticmethod
    def _fmt_ns_list(namespaces: list[str], limit: int = 5) -> str:
        """Format namespace list with overflow (for mrkdwn outside code blocks)."""
        shown = namespaces[:limit]
        text = ", ".join(f"`{ns}`" for ns in shown)
        if len(namespaces) > limit:
            text += f" _+{len(namespaces) - limit} more_"
        return text

    @staticmethod
    def _fmt_ns_col(namespaces: list[str], limit: int = 3) -> str:
        """Format namespace list for a monospace table column.

        Shows up to `limit` names comma-separated, with +N more overflow.
        """
        if not namespaces:
            return "?"
        shown = namespaces[:limit]
        text = ", ".join(shown)
        if len(namespaces) > limit:
            text += f" +{len(namespaces) - limit}"
        return text

    def _format_woop_table(
        self,
        cat_key: str,
        sev_emoji: str,
        desc_line: str,
        findings: list[Finding],
    ) -> list[str]:
        """Render absurd/mismatch findings aggregated by workload type.

        Instead of listing every namespace/workload individually (417 lines),
        groups by workload name across namespaces and shows:
          - Workload type, how many namespaces affected
          - Key metrics (rec, applied, ratio, divergence)
          - List of affected namespaces
        """
        cluster_name = (
            self.config.cluster.cluster_name
            or self.config.cluster.cluster_id[:8]
        )

        # Parse evidence JSON from each finding and ensure display fields exist
        parsed: list[tuple[Finding, dict]] = []
        for f in findings:
            try:
                ev = json.loads(f.evidence) if f.evidence else {}
            except (json.JSONDecodeError, TypeError):
                ev = {}
            # Backfill rec_display / applied_display from raw fields if missing
            if "rec_display" not in ev:
                if ev.get("recommended_memory_gib"):
                    ev["rec_display"] = self._fmt_gib(ev["recommended_memory_gib"])
                elif ev.get("recommended_cpu_cores"):
                    ev["rec_display"] = f"{ev['recommended_cpu_cores']} CPU"
            if "applied_display" not in ev:
                if ev.get("actual_memory_gib"):
                    ev["applied_display"] = self._fmt_gib(ev["actual_memory_gib"])
                elif ev.get("applied_memory_gib"):
                    ev["applied_display"] = self._fmt_gib(ev["applied_memory_gib"])
                elif ev.get("actual_cpu_cores"):
                    ev["applied_display"] = f"{ev['actual_cpu_cores']} CPU"
                elif ev.get("applied_cpu_cores"):
                    ev["applied_display"] = f"{ev['applied_cpu_cores']} CPU"
            parsed.append((f, ev))

        rows: list[str] = []

        if cat_key == FindingCategory.ABSURD_RECOMMENDATION.value:
            cap_findings = [(f, ev) for f, ev in parsed if ev.get("sub_type") != "ratio_breach"]
            ratio_findings = [(f, ev) for f, ev in parsed if ev.get("sub_type") == "ratio_breach"]

            if cap_findings:
                rows.append(self._render_cap_breaches(sev_emoji, cluster_name, cap_findings))
            if ratio_findings:
                rows.append(self._render_ratio_breaches(sev_emoji, cluster_name, ratio_findings))

        elif cat_key == FindingCategory.MISMATCH.value:
            rows.append(self._render_mismatches(sev_emoji, cluster_name, parsed))

        return rows

    def _render_cap_breaches(
        self, sev_emoji: str, cluster: str,
        items: list[tuple[Finding, dict]],
    ) -> str:
        header = (
            f"{sev_emoji} *Absolute Cap Breaches* "
            f"(≥ {self.config.thresholds.absurd_memory_gib:.0f} GiB) "
            f"— {len(items)} finding(s) on `{cluster}`"
            f"\n_Recommendation or applied value is unreasonably large_\n"
        )
        groups = self._group_by_workload_name(items)
        rows = []
        for wl_name, group in list(groups.items())[:10]:
            nss = sorted(set(self._workload_ns(f.workload) for f, _ in group))
            ns_str = self._fmt_ns_col(nss)
            sample_ev = group[0][1]
            rec_d = sample_ev.get("rec_display", "?")
            app_d = sample_ev.get("applied_display", "?")
            woop = sample_ev.get("woop", "")
            rows.append([
                wl_name,
                ns_str,
                rec_d,
                app_d,
                woop,
            ])
        table = self._make_table(
            ["Workload", "Namespace(s)", "Recommended", "Applied", "WOOP"],
            rows,
        )
        remaining = len(groups) - 10
        overflow = f"\n_+{remaining} more workload types_" if remaining > 0 else ""
        action = (
            "→ Investigate why WOOP is generating an extreme recommendation. "
            "Check if an outlier pod is driving this via the Max Usage strategy. "
            "Consider setting a hard cap on recommendations for this workload."
        )
        return f"{header}{table}{overflow}\n_{action}_"

    def _render_ratio_breaches(
        self, sev_emoji: str, cluster: str,
        items: list[tuple[Finding, dict]],
    ) -> str:
        header = (
            f"{sev_emoji} *Ratio Breaches* "
            f"(recommendation ≥ {self.config.thresholds.outlier_median_ratio:.0f}x current request) "
            f"— {len(items)} finding(s) on `{cluster}`"
            f"\n_WOOP recommending far more than the pod currently requests_\n"
        )
        groups = self._group_by_workload_name(items)
        rows = []
        for wl_name, group in list(groups.items())[:10]:
            nss = sorted(set(self._workload_ns(f.workload) for f, _ in group))
            ns_str = self._fmt_ns_col(nss)
            ratios = [ev.get("limit_request_ratio", 0) for _, ev in group]
            min_r, max_r = min(ratios), max(ratios)
            ratio_str = f"{min_r}x" if min_r == max_r else f"{min_r}-{max_r}x"
            sample_ev = group[0][1]
            rec_d = sample_ev.get("rec_display", "?")
            app_d = sample_ev.get("applied_display", "?")
            rows.append([
                wl_name,
                ns_str,
                rec_d,
                app_d,
                ratio_str,
            ])
        table = self._make_table(
            ["Workload", "Namespace(s)", "Recommended", "Request", "Ratio"],
            rows,
        )
        remaining = len(groups) - 10
        overflow = f"\n_+{remaining} more workload types_" if remaining > 0 else ""
        action = (
            "→ A large ratio means WOOP thinks the workload needs much more than it currently requests. "
            "Check workload metrics in CAST AI Console → WOOP → select the workload. "
            "If the spike is from an outlier pod, consider switching from Max Usage to a percentile-based strategy."
        )
        return f"{header}{table}{overflow}\n_{action}_"

    def _render_mismatches(
        self, sev_emoji: str, cluster: str,
        items: list[tuple[Finding, dict]],
    ) -> str:
        pct_thresh = self.config.thresholds.recommendation_mismatch_pct
        header = (
            f"{sev_emoji} *Unapplied Recommendations* "
            f"(>{pct_thresh:.0f}% gap) "
            f"— {len(items)} finding(s) on `{cluster}`"
            f"\n_WOOP computed new resource values but the pod is still running "
            f"the old ones — rollout may be pending or stuck_\n"
        )
        groups = self._group_by_workload_name(items)
        rows = []
        for wl_name, group in list(groups.items())[:10]:
            nss = sorted(set(self._workload_ns(f.workload) for f, _ in group))
            ns_str = self._fmt_ns_col(nss)
            pcts = [ev.get("diff_pct", 0) for _, ev in group]
            min_p, max_p = min(pcts), max(pcts)
            pct_str = f"{min_p}%" if min_p == max_p else f"{min_p}-{max_p}%"
            sample_ev = group[0][1]
            rec_d = sample_ev.get("rec_display", "?")
            app_d = sample_ev.get("applied_display", "?")
            apply_type = sample_ev.get("apply_type", "")
            apply_str = apply_type if apply_type else "-"
            rows.append([
                wl_name,
                ns_str,
                rec_d,
                app_d,
                pct_str,
                apply_str,
            ])
        table = self._make_table(
            ["Workload", "Namespace(s)", "Recommended", "Running", "Gap%", "Apply"],
            rows,
        )
        remaining = len(groups) - 10
        overflow = f"\n_+{remaining} more workload types_" if remaining > 0 else ""
        action = (
            "→ *What this means:* WOOP has a new recommendation but the pod hasn't picked it up yet. "
            "Common causes: rollout strategy is set to one-by-one and hasn't cycled this pod yet, "
            "the workload has a PDB blocking restarts, or apply mode is paused.\n"
            "→ *Action:* In CAST AI Console → WOOP → select the workload → check rollout status "
            "and apply type. If IMMEDIATE, the pod should restart soon. If DEFERRED, the recommendation "
            "applies on next natural restart."
        )
        return f"{header}{table}{overflow}\n{action}"

    # ── Per-category table renderers ────────────────────────────────
    # Each parses evidence JSON (enriched from snapshot in evaluator)
    # and renders a code-block table. LLM `what` serves as cause below the table.

    def _parse_evidence(self, finding: Finding) -> dict | list:
        """Parse evidence JSON from a finding, returning dict/list or empty dict."""
        try:
            return json.loads(finding.evidence) if finding.evidence else {}
        except (json.JSONDecodeError, TypeError):
            return {}

    @staticmethod
    def _fmt_bytes(val, field_name: str = "") -> str:
        """Format a byte or MiB value to human-readable string."""
        if not val and val != 0:
            return "?"
        if isinstance(val, (int, float)) and val > 1_000_000:
            return f"{round(val / 1048576)} MiB"
        if isinstance(val, (int, float)):
            return f"{val} MiB"
        return str(val)

    def _render_oomkill_table(
        self, sev_emoji: str, cluster: str, findings: list[Finding],
    ) -> str:
        desc = CATEGORY_DESCRIPTION[FindingCategory.OOMKILL.value]
        header = (
            f"{sev_emoji} *OOMKill Spiral* — {len(findings)} finding(s) on `{cluster}`\n"
            f"_{desc}_\n"
        )
        rows = []
        causes = []
        for f in findings[:15]:
            ev = self._parse_evidence(f)
            if isinstance(ev, list):
                ev = ev[0] if ev else {}

            # OOM rate: prefer _oom_rate_1h (computed by evaluator),
            # then oom_events_1h (WOOP events API), then restart_count (raw)
            oom_rate = ev.get("_oom_rate_1h")
            oom_1h = ev.get("oom_events_1h")
            if oom_rate is not None:
                restarts = f"~{oom_rate}"
            elif oom_1h is not None:
                restarts = str(oom_1h)
            else:
                restarts = str(ev.get("restart_count", "?"))

            # Memory limit from pod spec
            mem_limit = ev.get("mem_limit", "")
            # WOOP recommendation
            woop_rec_mem = ev.get("woop_rec_mem")
            if woop_rec_mem and isinstance(woop_rec_mem, (int, float)):
                if woop_rec_mem >= 1.0:
                    rec_str = f"{round(woop_rec_mem, 1)} GiB"
                else:
                    rec_str = f"{round(woop_rec_mem * 1024)} MiB"
            else:
                rec_str = ""

            # Build context column: "limit 128Mi → rec 512 MiB"
            context_parts = []
            if mem_limit:
                context_parts.append(f"limit {mem_limit}")
            if rec_str:
                context_parts.append(f"rec {rec_str}")
            context = " → ".join(context_parts) if context_parts else "-"

            rows.append([
                f.workload,
                ev.get("container", "-"),
                restarts,
                context,
            ])
            causes.append(f"• `{f.workload}`: {f.what}")

        table = self._make_table(
            ["Workload", "Container", "OOMs/1h", "Limit → Rec"],
            rows,
        )
        if len(findings) > 15:
            causes.append(f"_+{len(findings) - 15} more_")
        cause_text = "\n".join(causes[:5])
        if len(causes) > 5:
            cause_text += f"\n_+{len(causes) - 5} more causes_"
        return f"{header}{table}\n*Analysis:*\n{cause_text}\n_→ {findings[0].suggested_action}_"

    def _render_crashloop_table(
        self, sev_emoji: str, cluster: str, findings: list[Finding],
    ) -> str:
        desc = CATEGORY_DESCRIPTION[FindingCategory.CRASHLOOP.value]
        header = (
            f"{sev_emoji} *CrashLoopBackOff* — {len(findings)} finding(s) on `{cluster}`\n"
            f"_{desc}_\n"
        )
        rows = []
        for f in findings[:15]:
            ev = self._parse_evidence(f)
            if isinstance(ev, dict) and ev:
                container = ev.get("container", "-")
                restarts = str(ev.get("restart_count", "?"))
            else:
                container = "-"
                restarts = "?"
            rows.append([f.workload, container, restarts])

        table = self._make_table(["Workload", "Container", "Restarts"], rows)
        extra = f"\n_+{len(findings) - 15} more_" if len(findings) > 15 else ""
        return f"{header}{table}{extra}\n_→ {findings[0].suggested_action}_"

    def _render_unschedulable_table(
        self, sev_emoji: str, cluster: str, findings: list[Finding],
    ) -> str:
        desc = CATEGORY_DESCRIPTION[FindingCategory.UNSCHEDULABLE.value]
        header = (
            f"{sev_emoji} *Unschedulable Pods* — {len(findings)} finding(s) on `{cluster}`\n"
            f"_{desc}_\n"
        )
        rows = []
        causes = []
        for f in findings[:15]:
            ev = self._parse_evidence(f)
            if isinstance(ev, list):
                ev = ev[0] if ev else {}
            # Evidence is a single pending pod dict: {namespace, name, reason}
            ns = ev.get("namespace", "")
            name = ev.get("name", "")
            pod_label = f"{ns}/{name}" if ns and name else f.workload
            reason = ev.get("reason", ev.get("message", ""))
            rows.append([pod_label, reason or "(no reason recorded)"])
            causes.append(f"• `{f.workload}`: {f.what}")

        table = self._make_table(["Pod", "Reason"], rows)
        cause_text = "\n".join(causes[:5])
        return f"{header}{table}\n*Analysis:*\n{cause_text}\n_→ {findings[0].suggested_action}_"

    def _render_agent_table(
        self, sev_emoji: str, cluster: str, cat_key: str, findings: list[Finding],
    ) -> str:
        is_restart = cat_key == FindingCategory.AGENT_RESTART.value
        title = "Agent Restart Loop" if is_restart else "Agent Down"
        desc = CATEGORY_DESCRIPTION.get(cat_key, "")
        header = (
            f"{sev_emoji} *{title}* — {len(findings)} finding(s) on `{cluster}`\n"
            f"_{desc}_\n"
        )
        rows = []
        causes = []
        seen_pods = set()
        for f in findings[:10]:
            ev = self._parse_evidence(f)
            if isinstance(ev, list):
                ev = {"agent_pods": ev}
            agent_pods = ev.get("agent_pods", [])
            restarts_hr = str(ev.get("restarts_last_hour", "?"))
            if agent_pods:
                for pod in agent_pods[:5]:
                    name = pod.get("name", "?")
                    if name in seen_pods:
                        continue
                    seen_pods.add(name)
                    phase = pod.get("phase", "?")
                    rc = str(pod.get("restart_count", 0))
                    rows.append([name, phase, rc, f"{restarts_hr}/hr"])
            else:
                rows.append([f.workload, "?", "?", f"{restarts_hr}/hr"])
            causes.append(f"• {f.what}")

        table = self._make_table(["Pod", "Phase", "Restarts", "Rate"], rows)
        cause_text = "\n".join(causes[:3])
        return f"{header}{table}\n*Analysis:*\n{cause_text}\n_→ {findings[0].suggested_action}_"

    def _render_data_gap_table(
        self, sev_emoji: str, cluster: str, findings: list[Finding],
    ) -> str:
        desc = CATEGORY_DESCRIPTION[FindingCategory.DATA_GAP.value]
        header = (
            f"{sev_emoji} *Data Gaps* — {len(findings)} finding(s) on `{cluster}`\n"
            f"_{desc}_\n"
        )
        rows = []
        seen_wl = set()
        for f in findings[:15]:
            ev = self._parse_evidence(f)
            if isinstance(ev, list):
                items = ev[:5]
            else:
                items = [ev]
            for g in items:
                wl = g.get("workload", f.workload)
                if wl in seen_wl:
                    continue
                seen_wl.add(wl)
                kind = g.get("kind", "-")
                pods = str(g.get("pod_count", "-"))
                rec_st = g.get("rec_status", "-")
                # Clean up STATUS_ prefix for readability
                if isinstance(rec_st, str) and rec_st.startswith("STATUS_"):
                    rec_st = rec_st[7:]  # e.g. STATUS_NO_RECOMMENDATION → NO_RECOMMENDATION
                age = g.get("age_hours")
                age_str = f"{age}h" if age else "new"
                # Show when WOOP was enabled for this workload
                enabled = g.get("enabled_since", "")
                if enabled and len(enabled) > 10:
                    enabled = enabled[:10]  # just the date
                enabled_str = enabled if enabled else ">7d"
                rows.append([wl, kind, pods, rec_st, age_str, enabled_str])

        table = self._make_table(["Workload", "Kind", "Pods", "Rec Status", "Age", "Enabled"], rows)
        if len(findings) > 15:
            table += f"\n_+{len(findings) - 15} more_"
        return f"{header}{table}\n_→ {findings[0].suggested_action}_"

    def _render_memory_leak_table(
        self, sev_emoji: str, cluster: str, findings: list[Finding],
    ) -> str:
        desc = CATEGORY_DESCRIPTION[FindingCategory.MEMORY_LEAK.value]
        header = (
            f"{sev_emoji} *Memory Leak Suspected* — {len(findings)} finding(s) on `{cluster}`\n"
            f"_{desc}_\n"
        )
        rows = []
        causes = []
        for f in findings[:15]:
            ev = self._parse_evidence(f)
            if isinstance(ev, list):
                ev = ev[0] if ev else {}
            wl = f.workload
            container = ev.get("container", "-")
            usage = ev.get("usage_bytes") or ev.get("usage_mib") or ev.get("request_mem_mib", "")
            request = ev.get("request_bytes") or ev.get("request_mem_mib", "")
            rows.append([
                wl,
                container,
                self._fmt_bytes(usage),
                self._fmt_bytes(request),
            ])
            causes.append(f"• `{wl}`: {f.what}")

        table = self._make_table(["Workload", "Container", "Usage", "Request"], rows)
        cause_text = "\n".join(causes[:5])
        return f"{header}{table}\n*Analysis:*\n{cause_text}\n_→ {findings[0].suggested_action}_"

    def _render_cascading_table(
        self, sev_emoji: str, cluster: str, findings: list[Finding],
    ) -> str:
        desc = CATEGORY_DESCRIPTION[FindingCategory.CASCADING_SCALING.value]
        header = (
            f"{sev_emoji} *Cascading Scaling* on `{cluster}`\n"
            f"_{desc}_\n"
        )
        rows = []
        causes = []
        for f in findings[:5]:
            ev = self._parse_evidence(f)
            rows.append([
                str(ev.get("node_count", "?")),
                f"{ev.get('node_count_delta_pct', '?')}%",
                str(ev.get("total_pods", "?")),
                f"{ev.get('pod_count_delta_pct', '?')}%",
            ])
            causes.append(f"• {f.what}")

        table = self._make_table(["Nodes", "Node Δ%", "Pods", "Pod Δ%"], rows)
        cause_text = "\n".join(causes[:3])
        return f"{header}{table}\n*Analysis:*\n{cause_text}\n_→ {findings[0].suggested_action}_"

    def _render_webhook_table(
        self, sev_emoji: str, cluster: str, findings: list[Finding],
    ) -> str:
        desc = CATEGORY_DESCRIPTION[FindingCategory.WEBHOOK_FAILURE.value]
        header = (
            f"{sev_emoji} *Webhook / Exporter Failure* — {len(findings)} finding(s) on `{cluster}`\n"
            f"_{desc}_\n"
        )
        rows = []
        causes = []
        for f in findings[:10]:
            ev = self._parse_evidence(f)
            if isinstance(ev, list):
                for sig in ev[:5]:
                    signal = sig.get("signal", "?")
                    detail = sig.get("message", sig.get("detail", "-"))
                    rows.append([signal, str(detail)])
            elif isinstance(ev, dict) and ev:
                # WOOP workload error: {"workload": "...", "error": "..."}
                detail = ev.get("error", ev.get("message", "-"))
                rows.append([f.workload, str(detail)[:80]])
            else:
                rows.append([f.workload, "-"])
            causes.append(f"• {f.what}")

        table = self._make_table(["Signal", "Detail"], rows)
        cause_text = "\n".join(causes[:3])
        return f"{header}{table}\n*Analysis:*\n{cause_text}\n_→ {findings[0].suggested_action}_"

    def _render_unhealthy_table(
        self, sev_emoji: str, cluster: str, findings: list[Finding],
    ) -> str:
        desc = CATEGORY_DESCRIPTION[FindingCategory.UNHEALTHY_DEPLOYMENT.value]
        header = (
            f"{sev_emoji} *Unhealthy Deployments* — {len(findings)} finding(s) on `{cluster}`\n"
            f"_{desc}_\n"
        )
        rows = []
        causes = []
        for f in findings[:15]:
            ev = self._parse_evidence(f)
            if isinstance(ev, dict) and ev:
                ns = ev.get("namespace", "")
                name = ev.get("name", "")
                label = f"{ns}/{name}" if ns and name else f.workload
                desired = ev.get("desired", "?")
                ready = ev.get("ready", ev.get("readyReplicas", "?"))
                avail = ev.get("available", ev.get("availableReplicas", "?"))
                rows.append([label, str(desired), str(ready), str(avail)])
            elif isinstance(ev, list):
                for p in ev[:5]:
                    ns = p.get("namespace", "")
                    name = p.get("name", "")
                    label = f"{ns}/{name}" if ns and name else f.workload
                    rows.append([label, str(p.get("desired", "?")), str(p.get("ready", "?")), str(p.get("available", "?"))])
            else:
                rows.append([f.workload, "?", "?", "?"])
            causes.append(f"• `{f.workload}`: {f.what}")

        table = self._make_table(["Deployment", "Desired", "Ready", "Available"], rows)
        if len(findings) > 15:
            causes.append(f"_+{len(findings) - 15} more_")
        cause_text = "\n".join(causes[:5])
        if len(causes) > 5:
            cause_text += f"\n_+{len(causes) - 5} more_"
        return f"{header}{table}\n*Analysis:*\n{cause_text}\n_→ {findings[0].suggested_action}_"

    def _render_generic_table(
        self, sev_emoji: str, cluster: str, cat_key: str, findings: list[Finding],
    ) -> str:
        """Fallback table renderer for CONFIG, OTHER, and any future categories."""
        desc = CATEGORY_DESCRIPTION.get(cat_key, "")
        title = cat_key.replace("_", " ").title()
        header = (
            f"{sev_emoji} *{title}* — {len(findings)} finding(s) on `{cluster}`"
            + (f"\n_{desc}_" if desc else "")
            + "\n"
        )
        rows = []
        for f in findings[:15]:
            rows.append([f.workload, f.what])
        table = self._make_table(["Workload", "Issue"], rows)
        action = findings[0].suggested_action
        overflow = f"\n_+{len(findings) - 15} more_" if len(findings) > 15 else ""
        return f"{header}{table}{overflow}\n_→ {action}_"

    @staticmethod
    def _group_by_workload_name(
        items: list[tuple[Finding, dict]],
    ) -> dict[str, list[tuple[Finding, dict]]]:
        """Group findings by workload name (stripping namespace).

        Returns ordered dict sorted by group size descending.
        """
        groups: dict[str, list[tuple[Finding, dict]]] = {}
        for f, ev in items:
            wl_name = f.workload.split("/", 1)[1] if "/" in f.workload else f.workload
            groups.setdefault(wl_name, []).append((f, ev))
        # Sort by count descending so worst offenders show first
        return dict(sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True))

    async def _post_to_slack(self, message: dict) -> None:
        """POST message to Slack webhook with error handling."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self.webhook_url, json=message)
                if resp.status_code != 200:
                    logger.error(
                        "Slack webhook returned %d: %s",
                        resp.status_code,
                        resp.text[:200],
                    )
                else:
                    logger.info("Posted to Slack successfully")
        except httpx.TimeoutException:
            logger.error("Slack webhook timed out — will retry next cycle")
        except Exception as e:
            logger.error("Failed to post to Slack: %s", e)

    # ── App execution failure alerts (admin channel) ─────────────

    async def notify_app_error(
        self,
        phase: str,
        error: Exception | str,
        *,
        cluster_id: str = "",
        cluster_name: str = "",
        context: str = "",
        dry_run: bool = False,
    ) -> None:
        """Post an app execution failure to the admin Slack channel.

        Called when the watchdog pipeline itself fails — collector crash,
        evaluator exception, LLM unavailability, auth expiry, etc.
        These go to a separate admin-only channel so operators see them
        without cluttering the general findings channel.

        Args:
            phase: Pipeline phase that failed (e.g. "collector", "evaluator", "notifier")
            error: The exception or error message
            cluster_id: Cluster ID (for context)
            cluster_name: Cluster name (for context)
            context: Additional context string (optional)
            dry_run: If True, log instead of posting
        """
        # Dedup admin alerts: same phase + cluster within dedup window
        dedup_key = DedupKey(category=f"APP_ERROR_{phase.upper()}", workload=cluster_id or "global")
        if not self.state.should_notify(dedup_key, self.dedup_minutes):
            logger.info("Admin alert deduplicated: %s on %s", phase, cluster_id or "global")
            return
        self.state.record_notification(dedup_key)

        admin_url = self.config.slack.admin_webhook_url
        if not admin_url and not dry_run:
            logger.warning("No admin Slack webhook configured, skipping app error notification")
            return

        cluster_label = cluster_name or cluster_id[:8] if cluster_id else "unknown"
        error_str = str(error)
        # Truncate very long tracebacks
        if len(error_str) > 1500:
            error_str = error_str[:1500] + "… (truncated)"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"\U0001f6a8 Watchdog App Failure — {phase.upper()}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Cluster:*\n{cluster_label}"},
                    {"type": "mrkdwn", "text": f"*Phase:*\n`{phase}`"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Time:*\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error:*\n```{error_str}```",
                },
            },
        ]

        if context:
            if len(context) > 500:
                context = context[:500] + "… (truncated)"
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Context:*\n{context}"},
            })

        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "This alert is from the watchdog application itself, not a cluster finding. "
                            "Check watchdog pod logs for full traceback.",
                },
            ],
        })

        message = {"blocks": blocks}

        if dry_run:
            logger.info("[DRY RUN] Would post app error to admin Slack:\n%s", json.dumps(message, indent=2))
            return

        await self._post_to_admin_slack(message)

    async def _post_to_admin_slack(self, message: dict) -> None:
        """POST message to admin Slack webhook with error handling."""
        admin_url = self.config.slack.admin_webhook_url
        if not admin_url:
            return
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(admin_url, json=message)
                if resp.status_code != 200:
                    logger.error(
                        "Admin Slack webhook returned %d: %s",
                        resp.status_code,
                        resp.text[:200],
                    )
                else:
                    logger.info("Posted app error to admin Slack successfully")
        except httpx.TimeoutException:
            logger.error("Admin Slack webhook timed out")
        except Exception as e:
            logger.error("Failed to post to admin Slack: %s", e)
