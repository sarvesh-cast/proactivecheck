"""Main orchestrator — runs the 3-phase watchdog pipeline on a schedule.

Usage:
    # Full pipeline — single run (for testing / cron)
    python -m watchdog --once

    # Full pipeline — continuous loop (every 5 minutes)
    python -m watchdog

    # Dry-run mode (evaluate but don't post to Slack)
    WATCHDOG_DRY_RUN=true python -m watchdog --once

    # ── Per-module test flags ─────────────────────────────────────
    # Collector only — fetch data, print JSON, skip evaluate/notify
    python -m watchdog --collector-only

    # Evaluator only — load last snapshot from state, run LLM, print result
    python -m watchdog --evaluator-only

    # Evaluator with raw-metrics fallback (no LLM call)
    python -m watchdog --evaluator-only --raw-fallback

    # Notifier only — load last evaluation from state, format & post (dry-run)
    WATCHDOG_DRY_RUN=true python -m watchdog --notifier-only

    # State dump — print rolling window and dedup log
    python -m watchdog --state-dump

    # Collect + evaluate but skip notify
    python -m watchdog --skip-notify --once

    # Use a fixture file instead of live API (offline testing)
    python -m watchdog --evaluator-only --fixture snapshot_fixture.json

Environment variables:
    CASTAI_API_KEY or CASTAI_JWT_TOKEN  — CAST AI authentication
    CASTAI_ORG_ID                       — Organization ID
    ANTHROPIC_API_KEY                   — Anthropic API key for evaluation
    SLACK_WEBHOOK_URL                   — Slack incoming webhook
    WATCHDOG_CLUSTER_ID                 — Target cluster ID
    WATCHDOG_DRY_RUN=true               — Evaluate but don't post
    WATCHDOG_STATE_FILE                 — Path to state file
    WATCHDOG_LOG_LEVEL                  — Logging level (DEBUG/INFO/WARNING)
    WATCHDOG_MODEL                      — Anthropic model (default: claude-haiku-4-5)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .collector import Collector
from .config import WatchdogConfig
from .evaluator import Evaluator
from .models import EvaluationResult, Finding, Severity, FindingCategory, SnapshotData, Verdict
from .notifier import Notifier
from .state import StateManager

logger = logging.getLogger("watchdog")


class Watchdog:
    """Orchestrates the collect → evaluate → notify pipeline."""

    def __init__(self, config: WatchdogConfig | None = None) -> None:
        self.config = config or WatchdogConfig()
        self.collector = Collector(self.config)
        self.evaluator = Evaluator(self.config)
        self.state = StateManager(
            self.config.state_file, self.config.rolling_window_size
        )
        self.notifier = Notifier(self.config, self.state)
        self._shutdown = False

    async def run_once(self) -> None:
        """Execute a single collect → evaluate → notify cycle."""
        cycle_start = time.monotonic()
        logger.info("=== Watchdog cycle starting at %s ===", datetime.now(timezone.utc).isoformat())

        # ── Phase 1: Collect ──────────────────────────────────────
        try:
            snapshot = await self.collector.collect()
            logger.info(
                "Collected: %d pods (%d OOMKilled, %d Pending, %d CrashLoop), "
                "%d nodes, %d collection errors",
                snapshot.total_pods,
                len(snapshot.oomkilled_pods),
                snapshot.pending_pods,
                snapshot.crashloop_pods,
                snapshot.node_count,
                len(snapshot.collection_errors),
            )
        except Exception as e:
            logger.error("Collection phase failed entirely: %s", e, exc_info=True)
            return  # Can't evaluate without data

        # Store snapshot in rolling window
        self.state.push_snapshot(snapshot.to_dict_compact())

        # Track data gap durations (updates first_seen timestamps)
        self.state.update_data_gaps(snapshot.data_gaps)

        # ── Phase 2: Evaluate ─────────────────────────────────────
        try:
            history = self.state.get_snapshots()
            node_delta_pct = self.state.compute_node_count_delta_pct()
            pod_delta_pct = self.state.compute_pod_count_delta_pct()
            agent_restart_delta = self.state.compute_agent_restarts_last_hour()
            memory_leaks = self.state.detect_memory_leaks()
            mature_data_gaps = self.state.get_mature_data_gaps(min_hours=2.0)

            result = await self.evaluator.evaluate(
                snapshot, history, node_delta_pct,
                pod_count_delta_pct=pod_delta_pct,
                agent_restarts_last_hour=agent_restart_delta,
                memory_leaks=memory_leaks,
                mature_data_gaps=mature_data_gaps,
            )
            logger.info(
                "Evaluation: verdict=%s, findings=%d, model=%s",
                result.verdict.value,
                len(result.findings),
                result.model_used,
            )
        except Exception as e:
            logger.error("Evaluation phase failed: %s", e, exc_info=True)
            # Still save state even if evaluation fails
            self.state.save()
            return

        # ── Phase 3: Notify ───────────────────────────────────────
        try:
            await self.notifier.notify(result, dry_run=self.config.dry_run)
        except Exception as e:
            logger.error("Notification phase failed: %s", e, exc_info=True)
            # Non-fatal — we still save state

        # ── Housekeeping ──────────────────────────────────────────
        self.state.cleanup_stale_dedup_entries(max_age_hours=24)
        self.state.save()

        elapsed = time.monotonic() - cycle_start
        logger.info(
            "=== Cycle complete in %.1fs | verdict=%s | next in %ds ===",
            elapsed,
            result.verdict.value,
            self.config.run_interval_seconds,
        )

    async def run_loop(self) -> None:
        """Run the watchdog continuously on a 5-minute schedule.

        Handles SIGINT/SIGTERM for graceful shutdown.
        """
        logger.info(
            "Watchdog starting in %s mode | cluster=%s | interval=%ds",
            "DRY-RUN" if self.config.dry_run else "LIVE",
            self.config.cluster.cluster_id,
            self.config.run_interval_seconds,
        )

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_shutdown)

        while not self._shutdown:
            try:
                await self.run_once()
            except Exception as e:
                logger.error("Unhandled error in watchdog cycle: %s", e, exc_info=True)

            # Sleep in small increments to allow graceful shutdown
            for _ in range(self.config.run_interval_seconds):
                if self._shutdown:
                    break
                await asyncio.sleep(1)

        logger.info("Watchdog shutting down gracefully")

    def _handle_shutdown(self) -> None:
        logger.info("Shutdown signal received")
        self._shutdown = True


    # ── Per-module test methods ─────────────────────────────────────

    async def run_collector_only(self) -> None:
        """Run only the collector, print snapshot JSON to stdout."""
        logger.info("Running collector only against cluster %s", self.config.cluster.cluster_id)
        snapshot = await self.collector.collect()
        self.state.push_snapshot(snapshot.to_dict_compact())
        self.state.save()

        print(snapshot.to_json())

        # ── Summary block ────────────────────────────────────────
        print("\n" + "=" * 70, file=sys.stderr)
        print("  COLLECTOR SUMMARY", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        print(f"  Cluster:      {self.config.cluster.cluster_id}", file=sys.stderr)
        print(f"  Timestamp:    {snapshot.timestamp}", file=sys.stderr)
        print(f"  Pods:         {snapshot.total_pods} total | "
              f"{snapshot.running_pods} running | "
              f"{snapshot.pending_pods} pending | "
              f"{snapshot.crashloop_pods} crashloop", file=sys.stderr)
        print(f"  OOMKilled:    {len(snapshot.oomkilled_pods)}", file=sys.stderr)
        print(f"  Nodes:        {snapshot.node_count}", file=sys.stderr)
        print(f"  WOOP:         {len(snapshot.woop_workloads)} workloads | "
              f"{len(snapshot.recommendation_mismatches)} mismatches | "
              f"{len(snapshot.absurd_recommendations)} absurd recs", file=sys.stderr)
        print(f"  Agents:       {len(snapshot.agent_pods)} pods | "
              f"{snapshot.agent_restarts_last_hour} restarts (lifetime)", file=sys.stderr)
        print(f"  Memory track: {len(snapshot.workload_memory_usage)} workloads tracked", file=sys.stderr)
        print(f"  Log signals:  {len(snapshot.log_signals)}", file=sys.stderr)
        print(f"  Data gaps:    {len(snapshot.data_gaps)}", file=sys.stderr)

        if snapshot.collection_errors:
            print("-" * 70, file=sys.stderr)
            print(f"  ERRORS ({len(snapshot.collection_errors)}):", file=sys.stderr)
            print("-" * 70, file=sys.stderr)
            for i, err in enumerate(snapshot.collection_errors, 1):
                print(f"  {i}. {err}", file=sys.stderr)
        else:
            print("-" * 70, file=sys.stderr)
            print("  No errors — all collectors succeeded", file=sys.stderr)

        print("=" * 70 + "\n", file=sys.stderr)

    async def run_evaluator_only(
        self, fixture_path: str | None = None, raw_fallback: bool = False,
    ) -> None:
        """Run only the evaluator against the last snapshot or a fixture file.

        --fixture <path>   Load snapshot from a JSON file instead of state
        --raw-fallback     Skip LLM, use deterministic threshold checks
        """
        # Load snapshot data
        if fixture_path:
            logger.info("Loading fixture from %s", fixture_path)
            data = json.loads(Path(fixture_path).read_text())
            snapshot = SnapshotData(**{
                k: v for k, v in data.items()
                if k in SnapshotData.__dataclass_fields__
            })
        else:
            snapshots = self.state.get_snapshots()
            if not snapshots:
                logger.error("No snapshots in state file. Run --collector-only first.")
                sys.exit(1)
            latest = snapshots[-1]
            snapshot = SnapshotData(**{
                k: v for k, v in latest.items()
                if k in SnapshotData.__dataclass_fields__
            })
            logger.info("Loaded latest snapshot from state (%s)", snapshot.timestamp)

        history = self.state.get_snapshots()
        node_delta = self.state.compute_node_count_delta_pct()
        pod_delta = self.state.compute_pod_count_delta_pct()
        agent_restart_delta = self.state.compute_agent_restarts_last_hour()
        memory_leaks = self.state.detect_memory_leaks()
        mature_data_gaps = self.state.get_mature_data_gaps(min_hours=2.0)

        if raw_fallback:
            logger.info("Using raw-metrics fallback (no LLM call)")
            result = self.evaluator._raw_metrics_fallback(
                snapshot, node_delta, pod_delta, agent_restart_delta,
                memory_leaks, mature_data_gaps,
            )
        else:
            result = await self.evaluator.evaluate(
                snapshot, history, node_delta,
                pod_count_delta_pct=pod_delta,
                agent_restarts_last_hour=agent_restart_delta,
                memory_leaks=memory_leaks,
                mature_data_gaps=mature_data_gaps,
            )

        # Print structured result to stdout
        print(json.dumps(result.to_dict(), indent=2, default=str))

        logger.info(
            "Evaluation: verdict=%s | %d findings | model=%s",
            result.verdict.value, len(result.findings), result.model_used,
        )
        for f in result.findings:
            logger.info("  [%s] %s: %s — %s", f.severity.value, f.category.value, f.workload, f.what)

    async def run_notifier_only(self) -> None:
        """Run the notifier with a synthetic test evaluation.

        Always runs in dry-run mode to prevent accidental posts.
        Uses a realistic CRITICAL finding so you can see the full
        Slack message format.
        """
        logger.info("Running notifier test (dry-run)")
        test_result = EvaluationResult(
            verdict=Verdict.CRITICAL,
            summary="[TEST] Synthetic evaluation for notifier testing",
            findings=[
                Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.OOMKILL,
                    workload="cengage/discovery-puller",
                    what="OOMKill spiral detected (test)",
                    evidence="6 OOMKills in the last hour (3x threshold); "
                             "Memory limit is 128 MiB but workload needs ~1.2 GiB",
                    suggested_action="Disable WOOP for this workload immediately",
                ),
                Finding(
                    severity=Severity.WARNING,
                    category=FindingCategory.AGENT_RESTART,
                    workload="castai-agent/castai-agent-xyz",
                    what="CAST AI agent restart count elevated (test)",
                    evidence="restartCount=2 in last hour",
                    suggested_action="Monitor — will escalate to CRITICAL at 3/hour",
                ),
            ],
        )
        await self.notifier.notify(test_result, dry_run=True)

    def dump_state(self) -> None:
        """Print the current state file contents."""
        snapshots = self.state.get_snapshots()
        print(json.dumps({
            "snapshot_count": len(snapshots),
            "window_size": self.config.rolling_window_size,
            "snapshots": [
                {
                    "timestamp": s.get("timestamp", "?"),
                    "total_pods": s.get("total_pods", 0),
                    "oomkilled": len(s.get("oomkilled_pods", [])),
                    "pending": s.get("pending_pods", 0),
                    "node_count": s.get("node_count", 0),
                    "errors": len(s.get("collection_errors", [])),
                }
                for s in snapshots
            ],
            "node_count_trend": self.state.get_node_count_trend(),
            "node_count_delta_pct": round(self.state.compute_node_count_delta_pct(), 2),
            "pod_count_trend": self.state.get_pod_count_trend(),
            "pod_count_delta_pct": round(self.state.compute_pod_count_delta_pct(), 2),
            "oomkill_trend": self.state.get_oomkill_trend(),
            "agent_restarts_last_hour": self.state.compute_agent_restarts_last_hour(),
            "memory_leaks": self.state.detect_memory_leaks(),
            "dedup_entries": len(self.state._state.get("dedup_log", {})),
        }, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )


def _validate_config(config: WatchdogConfig, mode: str) -> list[str]:
    """Validate config based on which mode we're running."""
    errors = []

    needs_castai = mode in ("full", "collector-only")
    needs_anthropic = mode in ("full", "evaluator-only")
    needs_slack = mode in ("full",) and not config.dry_run

    if needs_castai and not config.castai.api_key and not config.castai.jwt_token:
        errors.append("Missing CAST AI credentials: set CASTAI_API_KEY or CASTAI_JWT_TOKEN")

    if needs_anthropic and not config.anthropic.api_key:
        errors.append("Missing ANTHROPIC_API_KEY (not needed with --raw-fallback)")

    if needs_slack and not config.slack.webhook_url:
        errors.append("Missing SLACK_WEBHOOK_URL (set WATCHDOG_DRY_RUN=true to skip)")

    if not config.cluster.cluster_id:
        errors.append("Missing WATCHDOG_CLUSTER_ID")

    return errors


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="watchdog",
        description="Grip Security Cluster Watchdog — proactive monitoring agent",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true",
                       help="Run full pipeline once then exit")
    mode.add_argument("--collector-only", action="store_true",
                       help="Run collector, print snapshot JSON, exit")
    mode.add_argument("--evaluator-only", action="store_true",
                       help="Run evaluator on last snapshot from state file")
    mode.add_argument("--notifier-only", action="store_true",
                       help="Send a test notification (always dry-run)")
    mode.add_argument("--state-dump", action="store_true",
                       help="Print rolling window state and exit")

    p.add_argument("--skip-notify", action="store_true",
                    help="Run collect + evaluate but skip Slack posting")
    p.add_argument("--raw-fallback", action="store_true",
                    help="With --evaluator-only: skip LLM, use threshold checks")
    p.add_argument("--fixture", type=str, default=None,
                    help="With --evaluator-only: load snapshot from JSON file")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = WatchdogConfig()
    _setup_logging(config.log_level)

    # Determine mode for validation
    if args.collector_only:
        mode = "collector-only"
    elif args.evaluator_only:
        mode = "evaluator-only" if not args.raw_fallback else "state-dump"
    elif args.notifier_only or args.state_dump:
        mode = "state-dump"
    else:
        mode = "full"

    errors = _validate_config(config, mode)
    if errors:
        for err in errors:
            logger.error("Config error: %s", err)
        sys.exit(1)

    watchdog = Watchdog(config)

    # Override dry_run for --skip-notify
    if args.skip_notify:
        config = WatchdogConfig(
            castai=config.castai, anthropic=config.anthropic,
            slack=config.slack, cluster=config.cluster,
            thresholds=config.thresholds,
            run_interval_seconds=config.run_interval_seconds,
            rolling_window_size=config.rolling_window_size,
            state_file=config.state_file, dry_run=True,
            log_level=config.log_level,
        )
        watchdog = Watchdog(config)

    # Dispatch
    if args.state_dump:
        watchdog.dump_state()
    elif args.collector_only:
        asyncio.run(watchdog.run_collector_only())
    elif args.evaluator_only:
        asyncio.run(watchdog.run_evaluator_only(
            fixture_path=args.fixture,
            raw_fallback=args.raw_fallback,
        ))
    elif args.notifier_only:
        asyncio.run(watchdog.run_notifier_only())
    elif args.once:
        asyncio.run(watchdog.run_once())
    else:
        asyncio.run(watchdog.run_loop())


if __name__ == "__main__":
    main()
