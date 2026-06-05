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
CATEGORY_DESCRIPTION: dict[str, str] = {
    FindingCategory.OOMKILL.value: "Pod killed for exceeding memory limit — restarting in a loop",
    FindingCategory.MISMATCH.value: "WOOP recommendation differs from what the pod is actually running",
    FindingCategory.UNSCHEDULABLE.value: "Pod stuck in Pending — not enough resources to schedule it",
    FindingCategory.AGENT.value: "CAST AI agent pod is not running or heartbeat is stale",
    FindingCategory.DATA_GAP.value: "Workload has optimization enabled but no active recommendation",
    FindingCategory.MEMORY_LEAK.value: "Memory usage trending up across snapshots without stabilizing",
    FindingCategory.ABSURD_RECOMMENDATION.value: "WOOP recommendation exceeds absolute cap (≥100 GiB) or is ≥10x the current request",
    FindingCategory.AGENT_RESTART.value: "CAST AI agent pod is restart-looping (>3 restarts/hour)",
    FindingCategory.WEBHOOK_FAILURE.value: "Workload autoscaler admission webhook is not responding",
    FindingCategory.CASCADING_SCALING.value: "Rapid node/pod count spike (>50% in 30 min) without a matching deployment",
    FindingCategory.UNHEALTHY_DEPLOYMENT.value: "Deployment has pods in CrashLoopBackOff or failing readiness",
}


class Notifier:
    """Posts evaluation results to Slack with deduplication."""

    def __init__(self, config: WatchdogConfig, state: StateManager) -> None:
        self.config = config
        self.state = state
        self.webhook_url = config.slack.webhook_url
        self.dedup_minutes = config.slack.dedup_window_minutes

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
        message = self._format_message(result.verdict, result.summary, new_findings)

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
    ) -> dict:
        """Format findings into a compact Slack Block Kit message with tabular findings."""
        emoji = VERDICT_EMOJI[verdict]
        cluster = self._cluster_header()
        console = self._console_link()
        cluster_label = self.config.cluster.cluster_name or self.config.cluster.cluster_id[:8]

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} {verdict.value} — {cluster_label}"},
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
            FindingCategory.ABSURD_RECOMMENDATION.value: 1,
            FindingCategory.MISMATCH.value: 2,
            FindingCategory.CASCADING_SCALING.value: 3,
            FindingCategory.AGENT.value: 4,
            FindingCategory.AGENT_RESTART.value: 5,
            FindingCategory.UNSCHEDULABLE.value: 6,
            FindingCategory.WEBHOOK_FAILURE.value: 7,
            # Warning-tier
            FindingCategory.UNHEALTHY_DEPLOYMENT.value: 8,
            FindingCategory.MEMORY_LEAK.value: 9,
            FindingCategory.DATA_GAP.value: 10,
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
        """Format namespace list with overflow."""
        shown = namespaces[:limit]
        text = ", ".join(f"`{ns}`" for ns in shown)
        if len(namespaces) > limit:
            text += f" _+{len(namespaces) - limit} more_"
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
            nss = [self._workload_ns(f.workload) for f, _ in group]
            sample_ev = group[0][1]
            rec_d = sample_ev.get("rec_display", "?")
            app_d = sample_ev.get("applied_display", "?")
            woop = sample_ev.get("woop", "")
            rows.append([
                wl_name,
                str(len(nss)),
                rec_d,
                app_d,
                woop,
            ])
        table = self._make_table(
            ["Workload", "NS#", "Recommended", "Applied", "WOOP"],
            rows,
        )
        remaining = len(groups) - 10
        overflow = f"\n_+{remaining} more workload types_" if remaining > 0 else ""
        return f"{header}{table}{overflow}\n_→ {items[0][0].suggested_action}_"

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
            nss = [self._workload_ns(f.workload) for f, _ in group]
            ratios = [ev.get("limit_request_ratio", 0) for _, ev in group]
            min_r, max_r = min(ratios), max(ratios)
            ratio_str = f"{min_r}x" if min_r == max_r else f"{min_r}-{max_r}x"
            sample_ev = group[0][1]
            rec_d = sample_ev.get("rec_display", "?")
            app_d = sample_ev.get("applied_display", "?")
            rows.append([
                wl_name,
                str(len(nss)),
                rec_d,
                app_d,
                ratio_str,
            ])
        table = self._make_table(
            ["Workload", "NS#", "Recommended", "Request", "Ratio"],
            rows,
        )
        remaining = len(groups) - 10
        overflow = f"\n_+{remaining} more workload types_" if remaining > 0 else ""
        return f"{header}{table}{overflow}\n_→ {items[0][0].suggested_action}_"

    def _render_mismatches(
        self, sev_emoji: str, cluster: str,
        items: list[tuple[Finding, dict]],
    ) -> str:
        header = (
            f"{sev_emoji} *Recommendation Mismatches* "
            f"(>{self.config.thresholds.recommendation_mismatch_pct:.0f}% divergence) "
            f"— {len(items)} finding(s) on `{cluster}`"
            f"\n_WOOP recommendation differs from what the pod is actually running_\n"
        )
        groups = self._group_by_workload_name(items)
        rows = []
        for wl_name, group in list(groups.items())[:10]:
            nss = [self._workload_ns(f.workload) for f, _ in group]
            pcts = [ev.get("diff_pct", 0) for _, ev in group]
            min_p, max_p = min(pcts), max(pcts)
            pct_str = f"{min_p}%" if min_p == max_p else f"{min_p}-{max_p}%"
            sample_ev = group[0][1]
            rec_d = sample_ev.get("rec_display", "?")
            app_d = sample_ev.get("applied_display", "?")
            woop = sample_ev.get("woop", "")
            rows.append([
                wl_name,
                str(len(nss)),
                rec_d,
                app_d,
                pct_str,
                woop,
            ])
        table = self._make_table(
            ["Workload", "NS#", "Recommended", "Applied", "Diff%", "WOOP"],
            rows,
        )
        remaining = len(groups) - 10
        overflow = f"\n_+{remaining} more workload types_" if remaining > 0 else ""
        return f"{header}{table}{overflow}\n_→ {items[0][0].suggested_action}_"

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
            restarts = str(ev.get("restart_count", "?"))
            container = ev.get("container", "-")
            last_oom = ev.get("last_oomkill_time", "")
            if last_oom and len(last_oom) > 16:
                last_oom = last_oom[:16]
            rows.append([
                f.workload,
                container,
                restarts,
                last_oom or "-",
            ])
            causes.append(f"• `{f.workload}`: {f.what}")

        table = self._make_table(
            ["Workload", "Container", "Restarts", "Last OOM"],
            rows,
        )
        if len(findings) > 15:
            causes.append(f"_+{len(findings) - 15} more_")
        cause_text = "\n".join(causes[:5])
        if len(causes) > 5:
            cause_text += f"\n_+{len(causes) - 5} more causes_"
        return f"{header}{table}\n*Analysis:*\n{cause_text}\n_→ {findings[0].suggested_action}_"

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
        causes = []
        for f in findings[:15]:
            ev = self._parse_evidence(f)
            if isinstance(ev, list):
                for g in ev[:5]:
                    wl = g.get("workload", f.workload)
                    container = g.get("container", "-")
                    woop = g.get("woop", "-")
                    rows.append([wl, container, woop])
            else:
                wl = ev.get("workload", f.workload)
                container = ev.get("container", "-")
                woop = ev.get("woop", "-")
                rows.append([wl, container, woop])
            causes.append(f"• `{f.workload}`: {f.what}")

        table = self._make_table(["Workload", "Container", "WOOP"], rows)
        if len(findings) > 15:
            causes.append(f"_+{len(findings) - 15} more_")
        cause_text = "\n".join(causes[:5])
        if len(causes) > 5:
            cause_text += f"\n_+{len(causes) - 5} more_"
        return f"{header}{table}\n*Analysis:*\n{cause_text}\n_→ {findings[0].suggested_action}_"

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
            if isinstance(ev, list):
                for p in ev[:5]:
                    ns = p.get("namespace", "")
                    name = p.get("name", "")
                    container = p.get("container", "-")
                    restarts = str(p.get("restart_count", "?"))
                    rows.append([f"{ns}/{name}", container, restarts])
            elif isinstance(ev, dict) and ev:
                # log_signal evidence: {namespace, name, desired, ...}
                ns = ev.get("namespace", "")
                name = ev.get("name", "")
                label = f"{ns}/{name}" if ns and name else f.workload
                desired = ev.get("desired", "?")
                ready = ev.get("ready", ev.get("readyReplicas", "?"))
                rows.append([label, f"desired={desired}", f"ready={ready}"])
            else:
                rows.append([f.workload, "-", "?"])
            causes.append(f"• `{f.workload}`: {f.what}")

        table = self._make_table(["Pod", "Container", "Restarts"], rows)
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
