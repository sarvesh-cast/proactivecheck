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

    # ── Backtest mode (historical snapshot analysis) ─────────────
    # Collect + evaluate at specific historical timestamps
    python -m watchdog --backtest 2026-05-21T05:00:00Z

    # Multiple timestamps (builds rolling window across them)
    python -m watchdog --backtest 2026-05-21T05:00:00Z 2026-06-01T18:30:00Z

    # Single snapshot at a specific time (collector-only)
    python -m watchdog --collector-only --snapshot-time 2026-05-21T05:00:00Z

    # ── Tier testing flags (hybrid flow verification) ────────────
    # Force a specific collector tier — no cascade
    python -m watchdog --collector-only --force-tier=hybrid
    python -m watchdog --collector-only --force-tier=mcp
    python -m watchdog --collector-only --force-tier=api

    # Print tier diagnostics (which tier ran, timing, field coverage)
    python -m watchdog --collector-only --tier-report
    python -m watchdog --collector-only --force-tier=hybrid --tier-report

    # Compare all available tiers side-by-side
    python -m watchdog --compare-tiers

    # Full pipeline with tier report (useful for debugging cascade)
    python -m watchdog --once --force-tier=hybrid --tier-report

Environment variables:
    CASTAI_API_KEY or CASTAI_JWT_TOKEN  — CAST AI authentication
    CASTAI_ORG_ID                       — Organization ID
    LLM_API_KEY                         — LLM API key (falls back to CASTAI_API_KEY)
    LLM_BASE_URL                        — OpenAI-compatible endpoint (default: https://llm.kimchi.dev/openai/v1)
    SLACK_WEBHOOK_URL                   — Slack incoming webhook (findings channel)
    SLACK_ADMIN_WEBHOOK_URL             — Slack webhook for app execution failures (admin channel)
    WATCHDOG_CLUSTER_ID                 — Target cluster ID
    WATCHDOG_DRY_RUN=true               — Evaluate but don't post
    WATCHDOG_STATE_FILE                 — Path to state file
    WATCHDOG_LOG_LEVEL                  — Logging level (DEBUG/INFO/WARNING)
    WATCHDOG_MODEL                      — LLM model (default: kimi-k2.6)
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

import httpx

from .collector import Collector
from .config import WatchdogConfig
from .evaluator import Evaluator
from .models import EvaluationResult, Finding, Severity, FindingCategory, SnapshotData, Verdict
from .notifier import Notifier
from .state import StateManager

logger = logging.getLogger("watchdog")


async def _fetch_cluster_name_map(config: WatchdogConfig) -> dict[str, str]:
    """Fetch {cluster_id: cluster_name} from list_clusters.

    Tries the public API first (fast, uses API key), then MCP as fallback.
    Returns empty dict on failure so callers fall back gracefully.
    """
    # ── Try public API first (no MCP session overhead, works with API key) ──
    if config.castai.api_key:
        try:
            async with httpx.AsyncClient(
                headers=config.castai.auth_headers,
                timeout=10,
            ) as client:
                resp = await client.get(f"{config.castai.api_url}/v1/kubernetes/external-clusters")
                resp.raise_for_status()
                items = resp.json().get("items", [])
                return {cl["id"]: cl.get("name", "") for cl in items if cl.get("id")}
        except Exception as e:
            logger.debug("API cluster name lookup failed: %s", e)

    # ── Fallback: MCP list_clusters (requires valid JWT) ──
    if not config.castai.mcp_url:
        return {}
    from .mcp_client import MCPClient
    mcp = MCPClient(
        mcp_url=config.castai.mcp_url,
        jwt_token=config.castai.jwt_token or None,
        organization_id=config.castai.organization_id or None,
    )
    try:
        await mcp.initialize()
        call_args = {}
        if config.castai.organization_id:
            call_args["organization_id"] = config.castai.organization_id
        data = await mcp.call_tool("list_clusters", call_args)
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict) and "result" in data:
            inner = data["result"]
            if isinstance(inner, str):
                inner = json.loads(inner)
            data = inner
        clusters = data if isinstance(data, list) else (data.get("clusters") or data.get("items") or [])
        return {cl["id"]: cl.get("name", "") for cl in clusters if isinstance(cl, dict) and cl.get("id")}
    except Exception as e:
        logger.warning("Could not fetch cluster names: %s", e)
        return {}


async def _resolve_cluster_ids(config: WatchdogConfig) -> list[dict]:
    """Resolve the list of cluster IDs to monitor.

    Returns list of {"id": ..., "name": ...} dicts.

    Priority:
      1. WATCHDOG_CLUSTER_IDS env var (comma-separated, or "auto")
      2. WATCHDOG_CLUSTER_ID (single cluster, default)

    "auto" discovers all clusters from the org via list_clusters MCP call.
    """
    # Explicit comma-separated list takes priority.
    # Resolve names via list_clusters so alerts show "prod-us-3", not "d3b400e6".
    ids = config.cluster.cluster_ids
    if ids:
        name_map = await _fetch_cluster_name_map(config)
        return [{"id": cid, "name": name_map.get(cid, "")} for cid in ids]

    # Single cluster ID set — use it
    if config.cluster.cluster_id:
        name = config.cluster.cluster_name
        if not name:
            name_map = await _fetch_cluster_name_map(config)
            name = name_map.get(config.cluster.cluster_id, "")
        return [{"id": config.cluster.cluster_id, "name": name}]

    # Nothing set — auto-discover all clusters from org.
    # Try public API first (works with API key, no MCP needed), then MCP fallback.
    logger.info("No cluster IDs configured, discovering from org...")

    # ── Try public API first ──
    if config.castai.api_key:
        try:
            async with httpx.AsyncClient(
                headers=config.castai.auth_headers,
                timeout=10,
            ) as client:
                resp = await client.get(f"{config.castai.api_url}/v1/kubernetes/external-clusters")
                resp.raise_for_status()
                items = resp.json().get("items", [])
                discovered = [
                    {"id": cl["id"], "name": cl.get("name", "")}
                    for cl in items
                    if isinstance(cl, dict) and cl.get("id")
                ]
                labels = [f"{c['name'] or '?'} ({c['id'][:8]})" for c in discovered]
                logger.info("Auto-discovered %d clusters via API: %s", len(discovered), ", ".join(labels))
                return discovered
        except Exception as e:
            logger.warning("API cluster auto-discovery failed: %s — trying MCP", e)

    # ── Fallback: MCP list_clusters ──
    if not config.castai.mcp_url:
        logger.error("Cannot auto-discover clusters: no API key and CASTAI_MCP_URL not set")
        return []
    from .mcp_client import MCPClient
    mcp = MCPClient(
        mcp_url=config.castai.mcp_url,
        jwt_token=config.castai.jwt_token or None,
        organization_id=config.castai.organization_id or None,
    )
    try:
        await mcp.initialize()
        call_args = {}
        if config.castai.organization_id:
            call_args["organization_id"] = config.castai.organization_id
        data = await mcp.call_tool("list_clusters", call_args)
        logger.debug("list_clusters raw response: type=%s repr=%.500s", type(data).__name__, repr(data))

        if data is None:
            logger.error("list_clusters returned None — MCP call may have failed silently")
            return []

        # Unwrap layers until we get to the items list
        # The response can arrive in several shapes depending on the MCP transport path:
        #   1. Already parsed dict: {"items": [...]}
        #   2. JSON string: '{"items": [...]}'
        #   3. Wrapped: {"result": '{"items": [...]}'}
        #   4. Double-wrapped: '{"result": "{\"items\": [...]}"}'
        #   5. Just a list: [{"id": ...}, ...]

        # Step 1: parse string to dict/list
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                logger.error("list_clusters returned unparseable string: %.200s", data)
                return []

        # Step 2: unwrap {"result": ...} wrapper (may be nested)
        for _ in range(3):  # max 3 unwrap attempts
            if isinstance(data, dict) and "result" in data and len(data) <= 3:
                inner = data["result"]
                if isinstance(inner, str):
                    try:
                        data = json.loads(inner)
                    except json.JSONDecodeError:
                        break
                else:
                    data = inner
            else:
                break

        # Step 3: extract items (MCP returns "clusters" key, API returns "items")
        if isinstance(data, dict):
            clusters = data.get("clusters") or data.get("items") or []
        elif isinstance(data, list):
            clusters = data
        else:
            logger.error("list_clusters returned unexpected type after unwrap: %s", type(data).__name__)
            return []

        if not isinstance(clusters, list):
            logger.error("Expected list of clusters, got %s", type(clusters).__name__)
            return []

        discovered = [
            {"id": cl["id"], "name": cl.get("name", "")}
            for cl in clusters
            if isinstance(cl, dict) and cl.get("id")
        ]
        labels = [f"{c['name'] or '?'} ({c['id'][:8]})" for c in discovered]
        logger.info("Auto-discovered %d clusters via MCP: %s", len(discovered), ", ".join(labels))
        return discovered
    except Exception as e:
        logger.error("Cluster auto-discovery failed: %s", e, exc_info=True)
    return []


def _config_for_cluster(base_config: WatchdogConfig, cluster_id: str, cluster_name: str = "") -> WatchdogConfig:
    """Create a per-cluster config clone with its own cluster_id, name, and state file."""
    from dataclasses import replace
    cluster = replace(base_config.cluster, cluster_id=cluster_id, cluster_name=cluster_name or cluster_id[:8])
    # Per-cluster state file: watchdog_state_<first8chars>.json
    base = base_config.state_file.replace(".json", "")
    state_file = f"{base}_{cluster_id[:8]}.json" if len(cluster_id) > 8 else f"{base}_{cluster_id}.json"
    return replace(base_config, cluster=cluster, state_file=state_file)


class Watchdog:
    """Orchestrates the collect → evaluate → notify pipeline."""

    def __init__(
        self,
        config: WatchdogConfig | None = None,
        snapshot_time: str | None = None,
        force_tier: str | None = None,
    ) -> None:
        self.config = config or WatchdogConfig()
        self.snapshot_time = snapshot_time
        self.force_tier = force_tier
        self.collector = Collector(self.config, snapshot_time=snapshot_time, force_tier=force_tier)
        self.evaluator = Evaluator(self.config)
        self.state = StateManager(
            self.config.state_file, self.config.rolling_window_size
        )
        self.notifier = Notifier(self.config, self.state)
        self._shutdown = False

    async def run_once(self, tier_report: bool = False) -> None:
        """Execute a single collect → evaluate → notify cycle."""
        cycle_start = time.monotonic()
        logger.info("=== Watchdog cycle starting at %s ===", datetime.now(timezone.utc).isoformat())

        # ── Phase 1: Collect ──────────────────────────────────────
        try:
            snapshot = await self.collector.collect()
            if tier_report:
                print(self.collector.tier_report(snapshot), file=sys.stderr)
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
            await self.notifier.notify_app_error(
                "collector", e,
                cluster_id=self.config.cluster.cluster_id,
                cluster_name=self.config.cluster.cluster_name,
                context=f"Failed to collect snapshot for cluster {self.config.cluster.cluster_name}",
                dry_run=self.config.dry_run,
            )
            return  # Can't evaluate without data

        # Alert admin on partial collection failures (e.g. MCP timeout, API 403)
        if snapshot.collection_errors:
            await self.notifier.notify_app_error(
                "collector", f"{len(snapshot.collection_errors)} partial collection error(s)",
                cluster_id=self.config.cluster.cluster_id,
                cluster_name=self.config.cluster.cluster_name,
                context="; ".join(snapshot.collection_errors[:5]),
                dry_run=self.config.dry_run,
            )

        # Store snapshot in rolling window
        self.state.push_snapshot(snapshot.to_dict_compact())

        # Compute per-interval OOMKill deltas (cross-snapshot comparison).
        # Must run after push_snapshot so the previous snapshot is available.
        snapshot.oomkilled_pods = self.state.compute_oomkill_deltas(
            list(snapshot.oomkilled_pods)
        )

        # Track durations (updates first_seen timestamps)
        self.state.update_data_gaps(snapshot.data_gaps)
        self.state.update_pending_pods(snapshot.pending_pods_detail)

        # ── Phase 2: Evaluate ─────────────────────────────────────
        try:
            history = self.state.get_snapshots()
            node_delta_pct = self.state.compute_node_count_delta_pct()
            pod_delta_pct = self.state.compute_pod_count_delta_pct()
            agent_restart_delta = self.state.compute_agent_restarts_last_hour()
            memory_leaks = self.state.detect_memory_leaks()
            # Suppress data_gap on cold start — need ≥3 snapshots to establish baseline
            mature_data_gaps = self.state.get_mature_data_gaps(min_hours=self.config.thresholds.data_gap_hours) if len(history) >= 3 else []
            mature_pending = self.state.get_mature_pending_pods(min_minutes=float(self.config.thresholds.pending_pod_minutes))
            oomkill_trend = self.state.get_oomkill_trend()

            result = await self.evaluator.evaluate(
                snapshot, history, node_delta_pct,
                pod_count_delta_pct=pod_delta_pct,
                agent_restarts_last_hour=agent_restart_delta,
                memory_leaks=memory_leaks,
                mature_data_gaps=mature_data_gaps,
                mature_pending_pods=mature_pending,
                oomkill_trend=oomkill_trend,
            )
            logger.info(
                "Evaluation: verdict=%s, findings=%d, model=%s",
                result.verdict.value,
                len(result.findings),
                result.model_used,
            )

            # Alert admin if LLM is completely down (findings still work via raw metrics)
            if result.llm_failed:
                await self.notifier.notify_app_error(
                    "llm", "All LLM models failed after retries — using raw metrics summary only",
                    cluster_id=self.config.cluster.cluster_id,
                    cluster_name=self.config.cluster.cluster_name,
                    context=f"Models tried: {self.evaluator.model}, {self.evaluator.fallback_model}. "
                            f"Raw metrics produced {len(result.findings)} findings.",
                    dry_run=self.config.dry_run,
                )
        except Exception as e:
            logger.error("Evaluation phase failed: %s", e, exc_info=True)
            await self.notifier.notify_app_error(
                "evaluator", e,
                cluster_id=self.config.cluster.cluster_id,
                cluster_name=self.config.cluster.cluster_name,
                context=f"Snapshot had {snapshot.total_pods} pods, {snapshot.node_count} nodes, "
                        f"{len(snapshot.collection_errors)} collection errors",
                dry_run=self.config.dry_run,
            )
            # Still save state even if evaluation fails
            self.state.save()
            return

        # ── Phase 3: Notify ───────────────────────────────────────
        try:
            await self.notifier.notify(result, dry_run=self.config.dry_run)
        except Exception as e:
            logger.error("Notification phase failed: %s", e, exc_info=True)
            # Notify admin about notification failure (ironic but important —
            # uses separate admin webhook so a findings-webhook outage is visible)
            await self.notifier.notify_app_error(
                "notifier", e,
                cluster_id=self.config.cluster.cluster_id,
                cluster_name=self.config.cluster.cluster_name,
                context=f"Failed to post {len(result.findings)} findings (verdict={result.verdict.value})",
                dry_run=self.config.dry_run,
            )
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

    async def run_backtest(self, timestamps: list[str]) -> None:
        """Run collect + evaluate at each historical timestamp.

        Simulates the rolling window by collecting each timestamp in order
        and building up state, then evaluating. Always dry-run (no Slack).

        Usage:
            python -m watchdog --backtest 2026-05-21T05:00:00Z 2026-06-01T18:30:00Z
        """
        logger.info("=== BACKTEST MODE: %d timestamps for cluster %s ===",
                     len(timestamps), self.config.cluster.cluster_id)

        for i, ts in enumerate(timestamps, 1):
            logger.info("── Backtest %d/%d: snapshot_time=%s ──", i, len(timestamps), ts)

            # Swap the collector's snapshot_time for this iteration
            self.collector.snapshot_time = ts

            try:
                snapshot = await self.collector.collect()
                logger.info(
                    "Collected @ %s: %d pods (%d OOM, %d Pending, %d CrashLoop), "
                    "%d nodes, %d errors",
                    ts, snapshot.total_pods, len(snapshot.oomkilled_pods),
                    snapshot.pending_pods, snapshot.crashloop_pods,
                    snapshot.node_count, len(snapshot.collection_errors),
                )
            except Exception as e:
                logger.error("Collection failed for %s: %s", ts, e, exc_info=True)
                continue

            # Push to rolling window (builds up state across timestamps)
            self.state.push_snapshot(snapshot.to_dict_compact())
            self.state.update_data_gaps(snapshot.data_gaps)
            self.state.update_pending_pods(snapshot.pending_pods_detail)

            # Evaluate
            try:
                history = self.state.get_snapshots()
                node_delta = self.state.compute_node_count_delta_pct()
                pod_delta = self.state.compute_pod_count_delta_pct()
                agent_restart_delta = self.state.compute_agent_restarts_last_hour()
                memory_leaks = self.state.detect_memory_leaks()
                mature_data_gaps = self.state.get_mature_data_gaps(min_hours=self.config.thresholds.data_gap_hours) if len(history) >= 3 else []
                mature_pending = self.state.get_mature_pending_pods(min_minutes=float(self.config.thresholds.pending_pod_minutes))
                oomkill_trend = self.state.get_oomkill_trend()

                result = await self.evaluator.evaluate(
                    snapshot, history, node_delta,
                    pod_count_delta_pct=pod_delta,
                    agent_restarts_last_hour=agent_restart_delta,
                    memory_leaks=memory_leaks,
                    mature_data_gaps=mature_data_gaps,
                    mature_pending_pods=mature_pending,
                    oomkill_trend=oomkill_trend,
                )

                # Print results
                print(f"\n{'='*70}")
                print(f"  BACKTEST RESULT @ {ts}")
                print(f"{'='*70}")
                print(f"  Verdict: {result.verdict.value}")
                print(f"  Findings: {len(result.findings)}")
                for f in result.findings:
                    print(f"    [{f.severity.value:8s}] {f.category.value:25s} {f.workload}")
                    print(f"             {f.what}")
                    print(f"             Evidence: {f.evidence[:200]}")
                print(f"{'='*70}\n")

            except Exception as e:
                logger.error("Evaluation failed for %s: %s", ts, e, exc_info=True)

        # Save state at the end (useful for inspection)
        self.state.save()
        logger.info("=== BACKTEST COMPLETE: %d timestamps processed ===", len(timestamps))

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
                try:
                    await self.notifier.notify_app_error(
                        "run_cycle", e,
                        cluster_id=self.config.cluster.cluster_id,
                        cluster_name=self.config.cluster.cluster_name,
                        context="Unhandled exception escaped run_once — full cycle failed",
                        dry_run=self.config.dry_run,
                    )
                except Exception:
                    logger.error("Failed to send app error alert for unhandled exception")

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

    async def run_collector_only(self, tier_report: bool = False) -> None:
        """Run only the collector, print snapshot JSON to stdout."""
        logger.info("Running collector only against cluster %s", self.config.cluster.cluster_id)
        snapshot = await self.collector.collect()

        if tier_report:
            print(self.collector.tier_report(snapshot), file=sys.stderr)
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

    async def run_compare_tiers(self) -> None:
        """Run all available tiers sequentially and compare field coverage.

        Useful for verifying that hybrid produces the same signals as MCP.
        Each tier runs in isolation (fresh SnapshotData) — no cascade.
        """
        import time as _time
        from dataclasses import fields as dc_fields

        tiers_to_test = []
        if self.collector.mcp:
            tiers_to_test.append("mcp")
        if self.collector.snap:
            tiers_to_test.append("hybrid")
        tiers_to_test.append("api")  # always available

        if len(tiers_to_test) < 2:
            print("Only 1 tier available — nothing to compare. "
                  "Need at least MCP or hybrid + API.", file=sys.stderr)
            return

        results: dict[str, dict] = {}
        for tier in tiers_to_test:
            logger.info("── compare-tiers: running tier '%s' ──", tier)
            collector = Collector(
                self.config, snapshot_time=self.snapshot_time, force_tier=tier,
            )
            try:
                snapshot = await collector.collect()
                results[tier] = {
                    "total_pods": snapshot.total_pods,
                    "running_pods": snapshot.running_pods,
                    "pending_pods": snapshot.pending_pods,
                    "crashloop_pods": snapshot.crashloop_pods,
                    "oomkilled_pods": len(snapshot.oomkilled_pods),
                    "node_count": snapshot.node_count,
                    "nodes_detail": len(snapshot.nodes),
                    "agent_pods": len(snapshot.agent_pods),
                    "agent_restarts": snapshot.agent_restarts_last_hour,
                    "mismatches": len(snapshot.recommendation_mismatches),
                    "absurd_recs": len(snapshot.absurd_recommendations),
                    "data_gaps": len(snapshot.data_gaps),
                    "memory_usage": len(snapshot.workload_memory_usage),
                    "log_signals": len(snapshot.log_signals),
                    "errors": len(snapshot.collection_errors),
                    "duration_ms": collector.tier_duration_ms,
                }
            except Exception as e:
                logger.error("Tier '%s' failed: %s", tier, e)
                results[tier] = {"error": str(e)}

        # Print comparison table
        print("\n" + "=" * 80, file=sys.stderr)
        print("  TIER COMPARISON", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        # Header
        tier_names = list(results.keys())
        header = f"  {'Field':<25}"
        for t in tier_names:
            header += f" {t:>10}"
        print(header, file=sys.stderr)
        print("  " + "-" * (25 + 11 * len(tier_names)), file=sys.stderr)

        # Check if any tier errored
        for t in tier_names:
            if "error" in results[t]:
                print(f"  {t} FAILED: {results[t]['error']}", file=sys.stderr)

        # Rows
        all_fields = [
            "total_pods", "running_pods", "pending_pods", "crashloop_pods",
            "oomkilled_pods", "node_count", "nodes_detail", "agent_pods",
            "agent_restarts", "mismatches", "absurd_recs", "data_gaps",
            "memory_usage", "log_signals", "errors", "duration_ms",
        ]
        for field in all_fields:
            row = f"  {field:<25}"
            values = []
            for t in tier_names:
                val = results[t].get(field, "—")
                if field == "duration_ms" and isinstance(val, (int, float)):
                    row += f" {val:>9.0f}ms"
                else:
                    row += f" {val:>10}"
                values.append(val)
            # Mark divergences
            numeric = [v for v in values if isinstance(v, (int, float))]
            if len(set(numeric)) > 1 and field != "duration_ms" and field != "errors" and field != "log_signals":
                row += "  ← differs"
            print(row, file=sys.stderr)

        print("=" * 80 + "\n", file=sys.stderr)

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

        # Ensure duration trackers are populated (needed for --evaluator-only with --fixture)
        self.state.update_data_gaps(snapshot.data_gaps)
        self.state.update_pending_pods(snapshot.pending_pods_detail)

        history = self.state.get_snapshots()
        node_delta = self.state.compute_node_count_delta_pct()
        pod_delta = self.state.compute_pod_count_delta_pct()
        agent_restart_delta = self.state.compute_agent_restarts_last_hour()
        memory_leaks = self.state.detect_memory_leaks()
        mature_data_gaps = self.state.get_mature_data_gaps(min_hours=self.config.thresholds.data_gap_hours) if len(history) >= 3 else []
        mature_pending = self.state.get_mature_pending_pods(min_minutes=float(self.config.thresholds.pending_pod_minutes))
        oomkill_trend = self.state.get_oomkill_trend()

        if raw_fallback:
            logger.info("Using raw-metrics fallback (no LLM call)")
            result = self.evaluator._raw_metrics_fallback(
                snapshot, node_delta, pod_delta, agent_restart_delta,
                memory_leaks, mature_data_gaps, oomkill_trend,
                mature_pending_pods=mature_pending,
            )
        else:
            result = await self.evaluator.evaluate(
                snapshot, history, node_delta,
                pod_count_delta_pct=pod_delta,
                agent_restarts_last_hour=agent_restart_delta,
                memory_leaks=memory_leaks,
                mature_data_gaps=mature_data_gaps,
                mature_pending_pods=mature_pending,
                oomkill_trend=oomkill_trend,
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
        cluster_id = self.config.cluster.cluster_id or "test-cluster"
        test_result = EvaluationResult(
            verdict=Verdict.CRITICAL,
            summary=f"[TEST] Synthetic evaluation for notifier testing ({cluster_id[:8]})",
            findings=[
                Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.OOMKILL,
                    workload="test-namespace/test-workload",
                    what="OOMKill spiral detected (test)",
                    evidence="6 OOMKills in the last hour (3x threshold); "
                             "Memory limit is 128 MiB but workload needs ~1.2 GiB",
                    suggested_action="Disable WOOP for this workload immediately",
                ),
                Finding(
                    severity=Severity.WARNING,
                    category=FindingCategory.AGENT_RESTART,
                    workload="castai-agent/castai-agent-pod-abc",
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
    needs_llm = mode in ("full", "evaluator-only")
    needs_slack = mode in ("full",) and not config.dry_run

    if needs_castai and not config.castai.api_key and not config.castai.jwt_token:
        errors.append("Missing CAST AI credentials: set CASTAI_API_KEY or CASTAI_JWT_TOKEN")

    if needs_llm and not config.llm.api_key:
        errors.append("Missing LLM_API_KEY or CASTAI_API_KEY (not needed with --raw-fallback)")

    if needs_slack and not config.slack.webhook_url:
        errors.append("Missing SLACK_WEBHOOK_URL (set WATCHDOG_DRY_RUN=true to skip)")

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
    p.add_argument("--snapshot-time", type=str, default=None,
                    help="ISO 8601 timestamp for historical snapshot (backtest mode). "
                         "e.g. 2026-05-21T05:00:00Z")
    p.add_argument("--backtest", type=str, nargs="+", default=None,
                    metavar="TIMESTAMP",
                    help="Run collect+evaluate at one or more historical timestamps. "
                         "e.g. --backtest 2026-05-21T05:00:00Z 2026-06-01T18:30:00Z")

    # ── Tier testing flags ──
    p.add_argument("--force-tier", type=str, default=None,
                    choices=["mcp", "hybrid", "api"],
                    help="Force collector to use only this tier (no cascade). "
                         "mcp = MCP server only, hybrid = snapshot-cli + API, "
                         "api = public API only. Errors if prerequisites not met.")
    p.add_argument("--tier-report", action="store_true",
                    help="After collection, print a diagnostic report showing which "
                         "tier ran, timing, field coverage, and errors. "
                         "Combine with --collector-only for quick testing.")
    p.add_argument("--compare-tiers", action="store_true",
                    help="Run all available tiers sequentially and print a side-by-side "
                         "comparison of field coverage. Use with --collector-only.")

    return p


async def _run_multi_cluster(base_config: WatchdogConfig, args) -> None:
    """Resolve cluster IDs then run the pipeline for each cluster."""
    clusters = await _resolve_cluster_ids(base_config)
    if not clusters:
        logger.error("No cluster IDs to monitor")
        sys.exit(1)

    is_multi = len(clusters) > 1
    if is_multi:
        logger.info("Multi-cluster mode: %d clusters", len(clusters))

    for i, cl in enumerate(clusters):
        cid, cname = cl["id"], cl["name"]
        # Always build per-cluster config so cluster_id is set correctly,
        # even when WATCHDOG_CLUSTER_IDS contains a single entry (is_multi=False)
        # and WATCHDOG_CLUSTER_ID (singular) is unset.
        cfg = _config_for_cluster(base_config, cid, cname)
        if args.skip_notify:
            from dataclasses import replace as _replace
            cfg = _replace(cfg, dry_run=True)

        if is_multi:
            logger.info("── Cluster %d/%d: %s (%s) ──", i + 1, len(clusters), cname or cid[:8], cid)

        snapshot_time = getattr(args, "snapshot_time", None)
        force_tier = getattr(args, "force_tier", None)
        watchdog = Watchdog(cfg, snapshot_time=snapshot_time, force_tier=force_tier)

        if getattr(args, "backtest", None):
            await watchdog.run_backtest(args.backtest)
        elif args.state_dump:
            watchdog.dump_state()
        elif getattr(args, "compare_tiers", False):
            await watchdog.run_compare_tiers()
        elif args.collector_only:
            await watchdog.run_collector_only(
                tier_report=getattr(args, "tier_report", False),
            )
        elif args.evaluator_only:
            await watchdog.run_evaluator_only(
                fixture_path=args.fixture,
                raw_fallback=args.raw_fallback,
            )
        elif args.notifier_only:
            await watchdog.run_notifier_only()
        elif args.once:
            await watchdog.run_once(
                tier_report=getattr(args, "tier_report", False),
            )
        else:
            # Continuous loop — run all clusters each cycle
            await _run_multi_loop(base_config, clusters, args)
            return  # loop handles its own exit


async def _run_multi_loop(base_config: WatchdogConfig, clusters: list[dict], args) -> None:
    """Continuous loop that processes all clusters each cycle."""
    shutdown = False

    def _handle_shutdown():
        nonlocal shutdown
        shutdown = True

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_shutdown)

    is_multi = len(clusters) > 1
    logger.info(
        "Watchdog starting | %d cluster(s) | interval=%ds | %s mode",
        len(clusters),
        base_config.run_interval_seconds,
        "DRY-RUN" if base_config.dry_run else "LIVE",
    )

    while not shutdown:
        for i, cl in enumerate(clusters):
            if shutdown:
                break
            cid, cname = cl["id"], cl["name"]
            cfg = _config_for_cluster(base_config, cid, cname) if is_multi else base_config
            if is_multi:
                logger.info("── Cluster %d/%d: %s (%s) ──", i + 1, len(clusters), cname or cid[:8], cid)
            force_tier = getattr(args, "force_tier", None)
            watchdog = Watchdog(cfg, force_tier=force_tier)
            try:
                await watchdog.run_once(
                    tier_report=getattr(args, "tier_report", False),
                )
            except Exception as e:
                logger.error("Error on cluster %s: %s", cid, e, exc_info=True)
                try:
                    await watchdog.notifier.notify_app_error(
                        "run_cycle", e,
                        cluster_id=cid,
                        cluster_name=cname,
                        context="Unhandled exception in multi-cluster loop",
                        dry_run=base_config.dry_run,
                    )
                except Exception:
                    logger.error("Failed to send app error alert for cluster %s", cid)

        for _ in range(base_config.run_interval_seconds):
            if shutdown:
                break
            await asyncio.sleep(1)

    logger.info("Watchdog shutting down gracefully")


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

    # Cluster IDs are resolved at runtime (explicit list, single ID, or auto-discovery)
    # so skip the cluster_id check here — _resolve_cluster_ids handles it
    errors = _validate_config(config, mode)
    if errors:
        for err in errors:
            logger.error("Config error: %s", err)
        sys.exit(1)

    asyncio.run(_run_multi_cluster(config, args))


if __name__ == "__main__":
    main()
