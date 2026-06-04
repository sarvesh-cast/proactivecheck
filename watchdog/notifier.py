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

        # Group findings by (category, severity) for compact display
        groups: list[tuple[str, str, list[Finding]]] = []
        for finding in findings:
            if finding.severity == Severity.INFO:
                continue  # skip info in alert messages
            cat_key = finding.category.value
            sev_key = finding.severity.value
            if groups and groups[-1][0] == cat_key and groups[-1][1] == sev_key:
                groups[-1][2].append(finding)
            else:
                groups.append((cat_key, sev_key, [finding]))

        rows = []
        for cat_key, sev_key, group_findings in groups:
            sev_emoji = SEVERITY_EMOJI[Severity(sev_key)]
            if len(group_findings) <= 3:
                # Few findings — show each individually
                for f in group_findings:
                    wl = f.workload
                    affected = ""
                    if wl == "cluster-level":
                        affected = self._extract_affected_workloads(f.evidence)
                    rows.append(
                        f"{sev_emoji} `{cat_key}` | `{wl}`\n"
                        + (f"     Affected: {affected}\n" if affected else "")
                        + f"     {f.what}\n"
                        f"     _→ {f.suggested_action}_"
                    )
            else:
                # Many findings of same type — compact list
                header = f"{sev_emoji} `{cat_key}` — {len(group_findings)} workload(s):"
                wl_lines = []
                for f in group_findings[:20]:
                    wl_lines.append(f"  • `{f.workload}` — {f.what}")
                if len(group_findings) > 20:
                    wl_lines.append(f"  _+{len(group_findings) - 20} more_")
                action = group_findings[0].suggested_action
                rows.append(
                    header + "\n" + "\n".join(wl_lines) + f"\n     _→ {action}_"
                )

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
