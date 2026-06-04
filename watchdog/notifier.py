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

    async def _post_daily_summary(
        self, result: EvaluationResult, dry_run: bool
    ) -> None:
        """Post a daily summary at 08:00 UTC regardless of verdict."""
        emoji = VERDICT_EMOJI[result.verdict]
        m = result.metrics

        summary_text = (
            f"{emoji} *Daily Summary — {self.config.cluster.cluster_name}*\n\n"
            f"*Verdict:* {result.verdict.value}\n"
            f"*Summary:* {result.summary}\n\n"
            f"*Metrics:*\n"
            f"- Pods: {m.total_pods} total, {m.oomkilled_pods} OOMKilled, "
            f"{m.pending_pods} Pending, {m.crashloop_pods} CrashLoop\n"
            f"- Nodes: {m.node_count} (delta: {m.node_count_delta_pct:+.1f}%)\n"
            f"- WOOP: {m.recommendation_mismatches} mismatches, "
            f"{m.absurd_recommendations} absurd recs\n"
            f"- Agent restarts: {m.agent_restarts_last_hour} in last hour\n"
        )

        if result.findings:
            summary_text += f"\n*Active findings:* {len(result.findings)}\n"
            for f in result.findings[:5]:
                summary_text += f"  {SEVERITY_EMOJI[f.severity]} {f.category.value}: {f.what}\n"

        summary_text += f"\n_Snapshot at {result.evaluated_at}_"

        message = {"text": summary_text[:SLACK_TEXT_LIMIT]}

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
        """Format findings into a Slack message matching the design doc template."""
        emoji = VERDICT_EMOJI[verdict]
        header = f"{emoji} *{verdict.value} — {self.config.cluster.cluster_name}*"

        sections = [header, ""]

        for finding in findings:
            sev_emoji = SEVERITY_EMOJI[finding.severity]
            sections.append(
                f"{sev_emoji} *{finding.what}* on `{finding.workload}`"
            )

            # Evidence as bullet points
            evidence_lines = finding.evidence.split(";") if ";" in finding.evidence else [finding.evidence]
            for line in evidence_lines[:5]:
                line = line.strip()
                if line:
                    sections.append(f"  • {line}")

            # Suggested action
            if finding.suggested_action:
                sections.append(f"\n*Suggested action:* {finding.suggested_action}")

            sections.append("")  # blank line between findings

        # Timestamp footer
        sections.append(
            f"_Snapshot at {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f" — next check in 5 min_"
        )

        text = "\n".join(sections)

        # Truncate if too long for Slack
        if len(text) > SLACK_TEXT_LIMIT:
            text = text[:SLACK_TEXT_LIMIT - 50] + "\n\n_...truncated (too many findings)_"

        return {"text": text}

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
