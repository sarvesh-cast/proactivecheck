"""Collector module — gathers data via CAST AI MCP server.

MCP-first design: when CASTAI_MCP_URL is set, ALL data flows through the
MCP server which provides access to:

  Snapshot tools (GCS-backed):
    - analyze_snapshot_with_code    → pod health, OOMKills, agent status
    - get_cluster_snapshot_summary  → node count, cluster metadata

  WOOP tools:
    - woop_analyze_with_code        → server-side detection (absurd, mismatch, gaps)
    - woop_get_workload_metrics     → historical per-workload metrics (backtest)
    - woop_get_oom_summary          → OOM kills ranked by workload
    - woop_get_workload_resource_ratios → limit/request ratio checks

  Log tools (Grafana Loki-backed):
    - loki_query                    → evictor, WA mutation, safety logs

  Cluster tools:
    - get_cluster_details           → cluster metadata, agent status
    - list_clusters                 → org-wide enumeration

If CASTAI_MCP_URL is not set, falls back to direct public API calls
(limited: no snapshot pod data, no Loki).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from .config import WatchdogConfig
from .models import SnapshotData
from .snapshot_analyzer import SnapshotAnalyzer, SnapshotCLIError, CLINotFoundError
from .snapshot_health import (
    analyze_pod_health,
    analyze_node_health,
    analyze_deployment_health,
    analyze_recommendations,
    analyze_pod_metrics,
)

logger = logging.getLogger("watchdog.collector")


class Collector:
    """Collects cluster state via MCP server (preferred) or public API."""

    # Valid values for force_tier
    VALID_TIERS = ("mcp", "hybrid", "api")

    def __init__(
        self,
        config: WatchdogConfig,
        snapshot_time: str | None = None,
        force_tier: str | None = None,
    ) -> None:
        self.config = config
        self.api_url = config.castai.api_url.rstrip("/")
        self.timeout = config.castai.request_timeout
        self.cluster_id = config.cluster.cluster_id
        self.org_id = config.castai.organization_id
        self.snapshot_time = snapshot_time  # ISO 8601 for historical backtest
        self.force_tier = force_tier  # "mcp" | "hybrid" | "api" | None (cascade)
        self.tier_used: str = ""  # set after collect() — which tier actually ran
        self.tier_duration_ms: float = 0.0  # wall clock for the collect phase
        self.mcp = None

        if config.castai.mcp_url:
            from .mcp_client import MCPClient
            self.mcp = MCPClient(
                mcp_url=config.castai.mcp_url,
                jwt_token=config.castai.jwt_token or None,
                # Don't pass iap_token string — MCPClient loads from
                # ~/.castai/iap_token.json to preserve the full cookie name
                iap_token=None,
                organization_id=config.castai.organization_id or None,
                timeout=60,
            )

        # Tier 2: Snapshot analyzer (GCS direct via snapshot-cli)
        self.snap = None
        if config.snapshot.enabled and not self.snapshot_time:
            # snapshot_time (backtest mode) uses MCP's historical snapshot access
            try:
                self.snap = SnapshotAnalyzer(
                    cluster_id=self.cluster_id,
                    gcs_bucket=config.snapshot.gcs_bucket,
                    cli_path=config.snapshot.cli_path or None,
                )
            except CLINotFoundError:
                logger.info("snapshot-cli not found — Tier 2 fallback disabled")
            except Exception as e:
                logger.warning("SnapshotAnalyzer init failed: %s", e)

    async def collect(self) -> SnapshotData:
        """Run all collectors concurrently. MCP-first with retry, public API fallback.

        When force_tier is set, skips the cascade and runs only the requested tier.
        Raises RuntimeError if the forced tier's prerequisites aren't met.
        """
        import time as _time
        t0 = _time.monotonic()

        # Use snapshot_time for backtest mode, otherwise current UTC
        ts = self.snapshot_time if self.snapshot_time else datetime.now(timezone.utc).isoformat()
        snapshot = SnapshotData(
            timestamp=ts,
            cluster_id=self.cluster_id,
        )
        if self.snapshot_time:
            logger.info("BACKTEST MODE: collecting snapshot at %s", self.snapshot_time)

        # ── Force-tier mode: bypass cascade, run exactly one tier ──
        if self.force_tier:
            result = await self._collect_forced(snapshot)
            self.tier_duration_ms = (_time.monotonic() - t0) * 1000
            return result

        # ── Normal cascade: MCP → Hybrid → API ──

        # Try MCP path first, with session re-init on failure
        if self.mcp:
            for attempt in range(1, 3):  # 2 attempts: init → collect → (re-init → collect)
                mcp_ok = await self.mcp.initialize()
                if not mcp_ok:
                    if attempt < 2:
                        logger.warning("MCP init failed (attempt %d/2), retrying in 3s", attempt)
                        await asyncio.sleep(3)
                        self.mcp._session_id = None  # force new session
                        continue
                    logger.warning("MCP init failed after 2 attempts — falling back to public API")
                    snapshot.collection_errors.append(
                        "MCP server connection failed after retries, using public API fallback"
                    )
                    break

                try:
                    self.tier_used = "mcp"
                    result = await self._collect_via_mcp(snapshot)
                    self.tier_duration_ms = (_time.monotonic() - t0) * 1000
                    return result
                except Exception as e:
                    if attempt < 2:
                        logger.warning("MCP collection failed (attempt %d/2): %s, re-initializing", attempt, e)
                        self.mcp._session_id = None
                        await asyncio.sleep(3)
                        continue
                    logger.error("MCP collection failed after 2 attempts: %s", e)
                    snapshot.collection_errors.append(f"MCP collection failed: {e}")
                    break

        # Tier 2: Snapshot hybrid (GCS direct + lightweight WOOP API)
        if self.snap:
            try:
                logger.info("Trying Tier 2: snapshot-cli + public API hybrid")
                self.tier_used = "hybrid"
                result = await self._collect_via_snapshot_hybrid(snapshot)
                self.tier_duration_ms = (_time.monotonic() - t0) * 1000
                return result
            except Exception as e:
                logger.error("Snapshot hybrid fallback failed: %s", e)
                snapshot.collection_errors.append(f"Snapshot hybrid failed: {e}")

        # Tier 3: Pure public API (degraded, last resort)
        self.tier_used = "api"
        result = await self._collect_via_public_api(snapshot)
        self.tier_duration_ms = (_time.monotonic() - t0) * 1000
        return result

    async def _collect_forced(self, snapshot: SnapshotData) -> SnapshotData:
        """Run exactly one tier (no cascade). Used by --force-tier flag."""
        tier = self.force_tier
        logger.info("FORCE-TIER MODE: running tier '%s' only (no cascade)", tier)

        if tier == "mcp":
            if not self.mcp:
                raise RuntimeError(
                    "--force-tier=mcp but CASTAI_MCP_URL is not set. "
                    "Set CASTAI_MCP_URL to use MCP tier."
                )
            ok = await self.mcp.initialize()
            if not ok:
                raise RuntimeError("--force-tier=mcp but MCP server init failed")
            self.tier_used = "mcp"
            return await self._collect_via_mcp(snapshot)

        if tier == "hybrid":
            if not self.snap:
                raise RuntimeError(
                    "--force-tier=hybrid but snapshot-cli is not available. "
                    "Install snapshot-cli or set SNAPSHOT_CLI_PATH."
                )
            self.tier_used = "hybrid"
            return await self._collect_via_snapshot_hybrid(snapshot)

        if tier == "api":
            self.tier_used = "api"
            return await self._collect_via_public_api(snapshot)

        raise ValueError(f"Unknown tier '{tier}'. Valid: {self.VALID_TIERS}")

    def tier_report(self, snapshot: SnapshotData) -> str:
        """Generate a diagnostic report about which tier ran and what it produced."""
        lines = [
            f"{'═' * 60}",
            f"  TIER REPORT",
            f"{'═' * 60}",
            f"  Tier used:       {self.tier_used or 'unknown'}",
            f"  Force-tier:      {self.force_tier or 'off (cascade)'}",
            f"  Duration:        {self.tier_duration_ms:.0f} ms",
            f"  Cluster:         {self.cluster_id}",
            f"{'─' * 60}",
            f"  FIELD COVERAGE",
            f"{'─' * 60}",
            f"  total_pods:              {snapshot.total_pods}",
            f"  running_pods:            {snapshot.running_pods}",
            f"  pending_pods:            {snapshot.pending_pods} (detail: {len(snapshot.pending_pods_detail)})",
            f"  crashloop_pods:          {snapshot.crashloop_pods} (detail: {len(snapshot.crashloop_pods_detail)})",
            f"  oomkilled_pods:          {len(snapshot.oomkilled_pods)}",
            f"  node_count:              {snapshot.node_count} (detail: {len(snapshot.nodes)})",
            f"  agent_pods:              {len(snapshot.agent_pods)}",
            f"  agent_restarts_1h:       {snapshot.agent_restarts_last_hour}",
            f"  rec_mismatches:          {len(snapshot.recommendation_mismatches)} (total: {snapshot.recommendation_mismatches_total})",
            f"  absurd_recs:             {len(snapshot.absurd_recommendations)} (total: {snapshot.absurd_recommendations_total})",
            f"  data_gaps:               {len(snapshot.data_gaps)} (total: {snapshot.data_gaps_total})",
            f"  workload_memory_usage:   {len(snapshot.workload_memory_usage)}",
            f"  log_signals:             {len(snapshot.log_signals)}",
            f"  collection_errors:       {len(snapshot.collection_errors)}",
        ]

        # Field population score
        populated = sum(1 for v in [
            snapshot.total_pods, snapshot.running_pods, snapshot.pending_pods,
            snapshot.crashloop_pods, snapshot.oomkilled_pods, snapshot.node_count,
            snapshot.nodes, snapshot.agent_pods, snapshot.agent_restarts_last_hour,
            snapshot.recommendation_mismatches, snapshot.absurd_recommendations,
            snapshot.data_gaps, snapshot.workload_memory_usage, snapshot.log_signals,
        ] if v)
        lines.append(f"{'─' * 60}")
        lines.append(f"  Fields populated: {populated}/14")

        if snapshot.collection_errors:
            lines.append(f"{'─' * 60}")
            lines.append(f"  ERRORS")
            for err in snapshot.collection_errors:
                lines.append(f"    • {err[:120]}")

        # Log signal summary
        if snapshot.log_signals:
            lines.append(f"{'─' * 60}")
            lines.append(f"  LOG SIGNALS")
            for sig in snapshot.log_signals[:10]:
                lines.append(f"    • {sig.get('signal', 'unknown')}")

        lines.append(f"{'═' * 60}")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════
    #  MCP PATH — full fidelity, all tools available
    # ══════════════════════════════════════════════════════════════════

    async def _collect_via_mcp(self, snapshot: SnapshotData) -> SnapshotData:
        """Collect all data through the MCP server.

        Per design doc, these run concurrently:
          1. analyze_snapshot_with_code  → pods, nodes, deployments, agents
          2. woop_get_workloads          → recommendations, mismatches, absurd
          3. woop_get_oom_summary        → OOM kills by workload (last hour)
          4. woop_get_workload_resource_ratios → tight limit detection
          5. get_cluster_details + get_cluster_snapshot_summary → metadata
          6. loki_query                  → mutation errors, evictor, webhooks
          7. list_clusters               → org-wide agent health check

        In backtest mode (snapshot_time set), WOOP-based collectors swap to
        historical event reconstruction since woop_get_workloads has no
        snapshot_time parameter.
        """
        # In backtest mode, use historical WOOP event reconstruction
        if self.snapshot_time:
            woop_fn = self._mcp_woop_backtest(snapshot)
            oom_fn = self._mcp_oom_summary_backtest(snapshot)
            oom_events_fn = asyncio.sleep(0)  # no-op in backtest
            config_events_fn = asyncio.sleep(0)
        else:
            woop_fn = self._mcp_woop_workloads(snapshot)
            oom_fn = self._mcp_oom_summary(snapshot)
            oom_events_fn = self._mcp_oom_events(snapshot)
            config_events_fn = self._mcp_config_change_events(snapshot)

        results = await asyncio.gather(
            self._mcp_snapshot_health(snapshot),
            woop_fn,
            oom_fn,
            oom_events_fn,
            config_events_fn,
            self._mcp_resource_ratios(snapshot),
            self._mcp_cluster_details(snapshot),
            self._mcp_loki_signals(snapshot),
            self._mcp_org_agent_health(snapshot),
            return_exceptions=True,
        )

        names = [
            "snapshot_health", "woop_workloads", "oom_summary",
            "oom_events", "config_events",
            "resource_ratios", "cluster_details", "loki_signals",
            "org_agent_health",
        ]
        for name, result in zip(names, results):
            if isinstance(result, Exception):
                err = f"{name}: {type(result).__name__}: {result}"
                logger.error("MCP collector failed: %s", err)
                snapshot.collection_errors.append(err)

        # If snapshot_health failed, try extract_fields_from_all as fallback
        if any("snapshot_health" in e for e in snapshot.collection_errors):
            if not snapshot.total_pods:
                logger.info("Trying extract_fields_from_all fallback...")
                try:
                    await self._mcp_snapshot_fallback(snapshot)
                except Exception as e:
                    snapshot.collection_errors.append(
                        f"snapshot_fallback: {type(e).__name__}: {e}"
                    )

        # ── Post-collection enrichment ─────────────────────────────────
        # All parallel calls are done; enrich entries with cross-source data.
        self._enrich_oom_entries(snapshot)
        self._enrich_data_gaps(snapshot)

        # Drop ephemeral log signals that were only needed for enrichment.
        # These are bulk dicts (rec_lookup can be 2000+ entries) that should
        # not be persisted in rolling state snapshots.
        ephemeral = {"woop_rec_lookup", "oom_events_1h", "woop_config_changes"}
        snapshot.log_signals = [
            s for s in snapshot.log_signals if s.get("signal") not in ephemeral
        ]

        return snapshot

    def _enrich_oom_entries(self, snapshot: SnapshotData) -> None:
        """Enrich OOM entries with event counts and WOOP recommendations.

        Called after all parallel collectors complete.  Adds:
          - oom_events_1h: exact OOM event count from get_workload_events
          - woop_rec_mem / woop_rec_cpu: WOOP recommendation (from rec_lookup)
        All fields are additive — missing data leaves defaults.
        """
        # Extract oom_events_1h counts from log_signals
        oom_counts: dict[str, int] = {}
        for sig in snapshot.log_signals:
            if sig.get("signal") == "oom_events_1h":
                oom_counts = sig.get("counts", {})
                break

        # Extract rec_lookup from log_signals (populated by woop_analyze_with_code)
        rec_lookup: dict[str, dict] = {}
        for sig in snapshot.log_signals:
            if sig.get("signal") == "woop_rec_lookup":
                rec_lookup = sig.get("lookup", {})
                break

        if not oom_counts and not rec_lookup:
            return

        for o in snapshot.oomkilled_pods:
            ns = o.get("namespace", "")
            pod_name = o.get("name", "")
            # Derive workload name from pod name for matching
            parts = pod_name.rsplit("-", 2)
            wl_name = parts[0] if len(parts) >= 3 else pod_name
            wl_key = f"{ns}/{wl_name}"

            # Also try exact name (for WOOP-sourced entries where name IS the workload)
            exact_key = f"{ns}/{pod_name}"

            # OOM event count enrichment
            if oom_counts:
                count = oom_counts.get(wl_key) or oom_counts.get(exact_key) or 0
                if count > 0:
                    o["oom_events_1h"] = count

            # WOOP recommendation enrichment
            if rec_lookup:
                rec = rec_lookup.get(wl_key) or rec_lookup.get(exact_key)
                if rec:
                    o["woop_rec_mem"] = rec.get("rec_mem", "")
                    o["woop_rec_cpu"] = rec.get("rec_cpu", "")

    def _enrich_data_gaps(self, snapshot: SnapshotData) -> None:
        """Enrich data gap entries with onboarding time from config change events.

        Adds 'enabled_since' field so alerts distinguish recent onboarding
        (warming up) from long-standing gaps (broken).
        """
        # Extract config change lookup from log_signals
        enabled_since: dict[str, str] = {}
        for sig in snapshot.log_signals:
            if sig.get("signal") == "woop_config_changes":
                enabled_since = sig.get("enabled_since", {})
                break

        if not enabled_since:
            return

        for gap in snapshot.data_gaps:
            wl_key = gap.get("workload", "")
            config_time = enabled_since.get(wl_key)
            if config_time:
                gap["enabled_since"] = config_time

    def _unwrap_snapshot_result(self, data: Any) -> dict:
        """Unwrap and validate analyze_snapshot_with_code response."""
        if not data:
            raise RuntimeError("analyze_snapshot_with_code returned empty")
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict) and "error" in data and "error_type" in data:
            raise RuntimeError(
                f"analyze_snapshot code error: {data['error']} "
                f"(hint: {data.get('hint', 'none')})"
            )
        if isinstance(data, dict) and "result" in data and data.get("success"):
            data = data["result"]
        return data

    async def _mcp_snapshot_health(self, snapshot: SnapshotData) -> None:
        """analyze_snapshot_with_code: pod health, OOMKills, agent pods,
        node capacity, pending pod events, deployment replicas.

        NOTE: RestrictedPython — no tuple unpacking, no generators in
        builtins (any/sum/all), no lambda with method calls.
        """
        code = '''
pods = snapshot.get_pods()
nodes = snapshot.get_nodes()
deployments = snapshot.get_deployments()

running = 0
pending = 0
crashloop = 0
oomkilled = []
crashloop_details = []
agent_pods = []
agent_total_restarts = 0
pending_details = []

for pod in pods:
    md = pod.get("metadata", {})
    st = pod.get("status", {})
    ns = md.get("namespace", "")
    name = md.get("name", "")
    phase = st.get("phase", "Unknown")

    if phase == "Running":
        running = running + 1
    elif phase == "Pending":
        pending = pending + 1
        # Only record as unschedulable if PodScheduled is False.
        # Pods that are scheduled but Pending for other reasons
        # (ImagePullBackOff, init containers, etc.) are not unschedulable.
        conds = st.get("conditions", [])
        is_unschedulable = False
        reason = ""
        pod_scheduled_seen = False
        for c in conds:
            if c.get("type") == "PodScheduled":
                pod_scheduled_seen = True
                if c.get("status") == "False":
                    is_unschedulable = True
                    reason = c.get("message", c.get("reason", ""))
        # No PodScheduled condition yet = just created, treat as potentially unschedulable
        if not pod_scheduled_seen:
            is_unschedulable = True
        if is_unschedulable:
            pending_details.append({
                "namespace": ns,
                "name": name,
                "reason": reason,
            })

    # Only track the castai-agent deployment in castai-agent namespace.
    # Pod names follow "deployment-name-replicaset-pod" pattern.
    is_agent = False
    if ns == "castai-agent":
        parts = name.split("-")
        dep_prefix = "-".join(parts[:-2]) if len(parts) > 2 else name
        if dep_prefix == "castai-agent":
            is_agent = True

    # Build container spec lookup for resource limits/requests
    spec_containers = pod.get("spec", {}).get("containers", [])
    container_res = {}
    for sc in spec_containers:
        sc_name = sc.get("name", "")
        sc_res = sc.get("resources", {})
        sc_lim = sc_res.get("limits", {})
        sc_req = sc_res.get("requests", {})
        container_res[sc_name] = {
            "mem_limit": sc_lim.get("memory", ""),
            "mem_request": sc_req.get("memory", ""),
            "cpu_limit": sc_lim.get("cpu", ""),
            "cpu_request": sc_req.get("cpu", ""),
        }

    pod_restarts = 0
    exit0 = False

    for cs in st.get("containerStatuses", []):
        rc = cs.get("restartCount", 0)
        pod_restarts = pod_restarts + rc
        last_state = cs.get("lastState", {})
        term = last_state.get("terminated", {})
        cur_state = cs.get("state", {})
        wait = cur_state.get("waiting", {})

        if term.get("reason") == "OOMKilled":
            oom_time = term.get("finishedAt", "")
            c_name = cs.get("name", "")
            c_res = container_res.get(c_name, {})
            pod_created = md.get("creationTimestamp", "")
            oomkilled.append({
                "namespace": ns,
                "name": name,
                "container": c_name,
                "restart_count": rc,
                "last_oomkill_time": oom_time,
                "pod_created_at": pod_created,
                "mem_limit": c_res.get("mem_limit", ""),
                "mem_request": c_res.get("mem_request", ""),
                "source": "snapshot_lastState",
            })
        if wait.get("reason") == "CrashLoopBackOff":
            crashloop = crashloop + 1
            crashloop_details.append({
                "namespace": ns,
                "name": name,
                "container": cs.get("name", ""),
                "restart_count": rc,
            })
        if term.get("exitCode") == 0 and term.get("reason"):
            exit0 = True

    if is_agent:
        agent_pods.append({
            "name": name,
            "namespace": ns,
            "phase": phase,
            "restart_count": pod_restarts,
            "exit_code_zero_history": exit0,
        })
        agent_total_restarts = agent_total_restarts + pod_restarts

# ── Node capacity & utilization ──
node_summary = []
total_alloc_cpu = 0
total_alloc_mem = 0
total_cap_cpu = 0
total_cap_mem = 0

for node in nodes:
    n_md = node.get("metadata", {})
    n_st = node.get("status", {})
    alloc = n_st.get("allocatable", {})
    cap = n_st.get("capacity", {})
    a_cpu = parse_cpu_millicores(alloc.get("cpu", "0"))
    a_mem = parse_memory_bytes(alloc.get("memory", "0"))
    c_cpu = parse_cpu_millicores(cap.get("cpu", "0"))
    c_mem = parse_memory_bytes(cap.get("memory", "0"))
    total_alloc_cpu = total_alloc_cpu + a_cpu
    total_alloc_mem = total_alloc_mem + a_mem
    total_cap_cpu = total_cap_cpu + c_cpu
    total_cap_mem = total_cap_mem + c_mem

    # Check for NotReady or taints
    ready = "Unknown"
    for cond in n_st.get("conditions", []):
        if cond.get("type") == "Ready":
            ready = cond.get("status", "Unknown")

    node_summary.append({
        "name": n_md.get("name", ""),
        "ready": ready,
        "alloc_cpu_m": a_cpu,
        "alloc_mem_bytes": a_mem,
        "cap_cpu_m": c_cpu,
        "cap_mem_bytes": c_mem,
        "max_pods": int(alloc.get("pods", "0")),
    })

# ── Deployment health (all-replica CrashLoop detection) ──
unhealthy_deployments = []
for dep in deployments:
    d_md = dep.get("metadata", {})
    d_st = dep.get("status", {})
    desired = dep.get("spec", {}).get("replicas", 0)
    ready_reps = d_st.get("readyReplicas", 0)
    avail = d_st.get("availableReplicas", 0)
    if desired > 0 and ready_reps == 0:
        unhealthy_deployments.append({
            "namespace": d_md.get("namespace", ""),
            "name": d_md.get("name", ""),
            "desired": desired,
            "ready": ready_reps,
            "available": avail,
        })

result = {
    "total_pods": len(pods),
    "running": running,
    "pending": pending,
    "crashloop": crashloop,
    "crashloop_details": crashloop_details[:30],
    "oomkilled": oomkilled[:50],
    "pending_details": pending_details[:20],
    "node_count": len(nodes),
    "node_summary": node_summary,
    "total_alloc_cpu_m": total_alloc_cpu,
    "total_alloc_mem_bytes": total_alloc_mem,
    "total_cap_cpu_m": total_cap_cpu,
    "total_cap_mem_bytes": total_cap_mem,
    "agent_pods": agent_pods,
    "agent_total_restarts": agent_total_restarts,
    "unhealthy_deployments": unhealthy_deployments[:20],
}
'''
        data = self._unwrap_snapshot_result(
            await self.mcp.analyze_snapshot(
                self.cluster_id, code,
                "Pod health, node capacity, deployment health, agent status",
                snapshot_time=self.snapshot_time,
            )
        )

        snapshot.total_pods = data.get("total_pods", 0)
        snapshot.running_pods = data.get("running", 0)
        snapshot.pending_pods = data.get("pending", 0)
        snapshot.crashloop_pods = data.get("crashloop", 0)
        snapshot.crashloop_pods_detail = data.get("crashloop_details", [])
        snapshot.pending_pods_detail = data.get("pending_details", [])
        snapshot.node_count = data.get("node_count", 0)
        snapshot.agent_pods = data.get("agent_pods", [])

        # Filter OOMKills: only keep those from the last hour.
        # lastState.terminated.finishedAt can be days old; we need recency.
        # In backtest mode, use snapshot_time as reference instead of now.
        if self.snapshot_time:
            ref_time = datetime.fromisoformat(self.snapshot_time.replace("Z", "+00:00"))
        else:
            ref_time = datetime.now(timezone.utc)
        cutoff = (ref_time - timedelta(hours=1)).isoformat()
        raw_oom = data.get("oomkilled", [])
        recent_oom = []
        for o in raw_oom:
            oom_time = o.get("last_oomkill_time", "")
            if oom_time and oom_time >= cutoff:
                recent_oom.append(o)
            elif not oom_time:
                # No timestamp — keep but mark as unknown
                o["last_oomkill_time"] = "unknown"
                recent_oom.append(o)
        if len(raw_oom) != len(recent_oom):
            logger.info(
                "OOMKill filter: %d total, %d recent (within 1h), %d stale dropped",
                len(raw_oom), len(recent_oom), len(raw_oom) - len(recent_oom),
            )
        snapshot.oomkilled_pods = recent_oom
        snapshot.agent_restarts_last_hour = data.get("agent_total_restarts", 0)

        # Rich node data
        snapshot.nodes = data.get("node_summary", [])

        # Stash extras for the evaluator
        if data.get("pending_details"):
            snapshot.log_signals.append({
                "signal": "pending_pod_details",
                "count": len(data["pending_details"]),
                "sample": data["pending_details"][:5],
            })
        if data.get("unhealthy_deployments"):
            snapshot.log_signals.append({
                "signal": "unhealthy_deployments",
                "count": len(data["unhealthy_deployments"]),
                "sample": data["unhealthy_deployments"][:5],
            })

        logger.info(
            "Snapshot: %d pods (%d OOM, %d Pending, %d CrashLoop), "
            "%d nodes, %d agent pods, %d unhealthy deploys",
            snapshot.total_pods, len(snapshot.oomkilled_pods),
            snapshot.pending_pods, snapshot.crashloop_pods,
            snapshot.node_count, len(snapshot.agent_pods),
            len(data.get("unhealthy_deployments", [])),
        )

    async def _mcp_woop_workloads(self, snapshot: SnapshotData) -> None:
        """woop_analyze_with_code: run all WOOP detection server-side in one call.

        Instead of fetching all workloads over the wire and iterating locally,
        we send Python code to the server that scans all workloads and returns
        only the findings (mismatches, absurd, data_gaps, summary).
        """
        t = self.config.thresholds
        known_s2z = self.config.cluster.known_scale_to_zero_workloads
        # Serialize thresholds and config into the code string
        known_s2z_json = json.dumps(known_s2z)

        analysis_code = f'''
AM = {t.absurd_memory_gib}
AC = {t.absurd_cpu_cores}
MP = {t.recommendation_mismatch_pct}
MMG = {t.mismatch_min_memory_gib}
MMC = {t.mismatch_min_cpu_cores}
OR = {t.outlier_median_ratio}
S2Z = {known_s2z_json}

def fm(g):
    return str(round(g, 1)) + " GiB" if g >= 1.0 else str(round(g * 1024, 1)) + " MiB"

def fc(c):
    return str(round(c, 1)) + " CPU"

def mk(wk, cn, wt, x):
    d = {{"workload": wk, "container": cn, "woop": wt}}
    for k in x:
        d[k] = x[k]
    return d

wls = get_workloads()
mm = []
ab = []
dg = []
dgs = set()
rl = {{}}
su = {{"managed": 0, "read_only": 0, "undefined": 0, "total": len(wls)}}

for wl in wls:
    wn = wl.get("name", "?")
    ns = wl.get("namespace", "?")
    wk = ns + "/" + wn
    vc = (wl.get("workloadConfigV2") or {{}}).get("vpaConfig") or {{}}
    mg = vc.get("managementOption", "UNDEFINED")
    at = vc.get("applyType", "")
    wt = mg + ("/" + at if at else "")
    su["managed" if mg == "MANAGED" else "read_only" if mg == "READ_ONLY" else "undefined"] += 1
    skip = False
    for p in S2Z:
        if p in wn:
            skip = True
            break
    if skip:
        continue
    kd = wl.get("kind", "")
    if kd in ("Job", "CronJob"):
        continue
    ctrs = wl.get("containers") or []
    rs = wl.get("recommendationStatus", "")
    hr = False
    for chk in ctrs:
        if chk.get("recommendation"):
            hr = True
            break
    pc = wl.get("podCount", 0) or 0
    if mg == "MANAGED" and not hr:
        if pc > 0 and wk not in dgs and rs not in ("STATUS_APPLIED", "STATUS_WAITING"):
            dgs.add(wk)
            dg.append({{"workload": wk, "woop": wt, "kind": kd, "pod_count": pc, "rec_status": rs or "NONE", "reason": "No active recommendation"}})
        continue
    if mg != "MANAGED" or not hr or not ctrs:
        continue
    for ct in ctrs:
        cn = ct.get("containerName", ct.get("name", ""))
        rq = (ct.get("resources") or {{}}).get("requests") or {{}}
        am = rq.get("memoryGib", 0) or 0
        ac = rq.get("cpuCores", 0) or 0
        rr = (ct.get("recommendation") or {{}}).get("requests") or {{}}
        rm = rr.get("memoryGib", 0) or 0
        rc = rr.get("cpuCores", 0) or 0
        oq = (ct.get("originalResources") or {{}}).get("requests") or {{}}
        om = oq.get("memoryGib", 0) or 0
        oc = oq.get("cpuCores", 0) or 0
        if wk not in rl and (rm or rc):
            rl[wk] = {{"rec_mem": rm, "rec_cpu": rc, "applied_mem": am, "applied_cpu": ac, "apply_type": at}}
        if rm > AM:
            ab.append(mk(wk, cn, wt, {{"sub_type": "cap_breach", "recommended_memory_gib": round(rm, 1), "applied_memory_gib": round(am, 1), "rec_display": fm(rm), "applied_display": fm(am), "reason": "WOOP recommends " + fm(rm)}}))
        if am > AM:
            ab.append(mk(wk, cn, wt, {{"sub_type": "cap_breach", "applied_memory_gib": round(am, 1), "recommended_memory_gib": round(rm, 1), "rec_display": fm(rm), "applied_display": fm(am), "reason": "Applied " + fm(am) + " (rec " + fm(rm) + ")"}}))
        if am > 0 and rm >= am * OR:
            r = round(rm / am, 1)
            ab.append(mk(wk, cn, wt, {{"sub_type": "ratio_breach", "recommended_memory_gib": round(rm, 1), "applied_memory_gib": round(am, 1), "rec_display": fm(rm), "applied_display": fm(am), "limit_request_ratio": r, "reason": "Rec " + fm(rm) + " is " + str(r) + "x current " + fm(am)}}))
        if ac > 0 and rc >= ac * OR:
            r = round(rc / ac, 1)
            ab.append(mk(wk, cn, wt, {{"sub_type": "ratio_breach", "recommended_cpu_cores": round(rc, 1), "applied_cpu_cores": round(ac, 1), "rec_display": fc(rc), "applied_display": fc(ac), "limit_request_ratio": r, "reason": "Rec " + fc(rc) + " is " + str(r) + "x current " + fc(ac)}}))
        if om > 0 and am >= om * OR:
            r = round(am / om, 1)
            ab.append(mk(wk, cn, wt, {{"sub_type": "baseline_ratio_breach", "recommended_memory_gib": round(rm, 1), "applied_memory_gib": round(am, 1), "original_memory_gib": round(om, 3), "rec_display": fm(am), "applied_display": fm(om), "limit_request_ratio": r, "reason": "Current " + fm(am) + " is " + str(r) + "x baseline " + fm(om)}}))
        if oc > 0 and ac >= oc * OR:
            r = round(ac / oc, 1)
            ab.append(mk(wk, cn, wt, {{"sub_type": "baseline_ratio_breach", "recommended_cpu_cores": round(rc, 1), "applied_cpu_cores": round(ac, 1), "original_cpu_cores": round(oc, 3), "rec_display": fc(ac), "applied_display": fc(oc), "limit_request_ratio": r, "reason": "Current " + fc(ac) + " is " + str(r) + "x baseline " + fc(oc)}}))
        if pc <= 0:
            continue
        if am and rm:
            p = abs(rm - am) / am * 100
            if p > MP and abs(rm - am) >= MMG:
                mm.append(mk(wk, cn, wt, {{"apply_type": at, "recommended_memory_gib": round(rm, 1), "actual_memory_gib": round(am, 1), "rec_display": fm(rm), "applied_display": fm(am), "diff_pct": round(p, 1)}}))
        if ac and rc:
            p = abs(rc - ac) / ac * 100
            if p > MP and abs(rc - ac) >= MMC:
                mm.append(mk(wk, cn, wt, {{"apply_type": at, "recommended_cpu_cores": round(rc, 1), "actual_cpu_cores": round(ac, 1), "rec_display": fc(rc), "applied_display": fc(ac), "diff_pct": round(p, 1)}}))

result = {{"mismatches": mm, "absurd": ab, "data_gaps": dg, "summary": su, "rec_lookup": rl}}
'''

        woop_args = {
            "cluster_id_or_name": self.cluster_id,
            "analysis_code": analysis_code,
            "description": "WOOP watchdog: detect absurd recs, mismatches, data gaps, broken multipliers",
            "timeout": 45,
        }
        if self.org_id:
            woop_args["organization_id"] = self.org_id

        data = await self.mcp.call_tool("woop_analyze_with_code", woop_args)
        if not data:
            raise RuntimeError("woop_analyze_with_code returned empty")
        if isinstance(data, str):
            data = json.loads(data)

        # Handle the server response envelope
        if isinstance(data, dict) and "error" in data and "error_type" in data:
            raise RuntimeError(
                f"woop_analyze_with_code error: {data['error']} "
                f"(hint: {data.get('hint', 'none')})"
            )
        if isinstance(data, dict) and "result" in data and data.get("success"):
            data = data["result"]

        mismatches = data.get("mismatches", [])
        absurd = data.get("absurd", [])
        data_gaps = data.get("data_gaps", [])
        woop_summary = data.get("summary", {})

        # We don't get full workloads back (by design — saves bandwidth).
        # Store an empty list; evaluator uses the findings directly.
        snapshot.woop_workloads = []
        snapshot.recommendation_mismatches = mismatches
        snapshot.absurd_recommendations = absurd
        snapshot.data_gaps = data_gaps
        snapshot.data_gaps_total = len(data_gaps)
        snapshot.recommendation_mismatches_total = len(mismatches)
        snapshot.absurd_recommendations_total = len(absurd)

        snapshot.log_signals.append({
            "signal": "woop_management_summary",
            "managed": woop_summary.get("managed", 0),
            "read_only": woop_summary.get("read_only", 0),
            "undefined": woop_summary.get("undefined", 0),
            "total": woop_summary.get("total", 0),
        })

        # Store rec_lookup for OOM enrichment (joined in _enrich_oom_entries)
        rec_lookup = data.get("rec_lookup", {})
        if rec_lookup:
            snapshot.log_signals.append({
                "signal": "woop_rec_lookup",
                "lookup": rec_lookup,
            })

        logger.info(
            "WOOP (analyze_with_code): %d workloads (managed=%d, read_only=%d, "
            "undefined=%d), %d mismatches, %d absurd, %d data gaps",
            woop_summary.get("total", 0), woop_summary.get("managed", 0),
            woop_summary.get("read_only", 0), woop_summary.get("undefined", 0),
            len(mismatches), len(absurd), len(data_gaps),
        )

    async def _mcp_oom_summary(self, snapshot: SnapshotData) -> None:
        """woop_get_oom_summary: OOM kills ranked by workload over last hour."""
        now = datetime.now(timezone.utc)
        oom_args = {
            "cluster_id_or_name": self.cluster_id,
            "from_date": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to_date": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 20,
        }
        if self.org_id:
            oom_args["organization_id"] = self.org_id
        data = await self.mcp.call_tool("woop_get_oom_summary", oom_args)
        if not data:
            return  # non-fatal, snapshot_health already has OOM data
        if isinstance(data, str):
            data = json.loads(data)

        oom_workloads = data if isinstance(data, list) else data.get("workloads", [])

        # woop_get_oom_summary is the authoritative, time-windowed source.
        # Replace any stale snapshot-derived OOM data with this.
        woop_oom = []
        for oom in oom_workloads:
            oom_count = oom.get("oom_count", 0)
            last_oom = oom.get("last_oom", "")
            woop_oom.append({
                "namespace": oom.get("namespace", ""),
                "name": oom.get("workload_name", ""),
                "restart_count": oom_count,
                "container": oom.get("container_name", ""),
                "last_oomkill_time": last_oom,
                "oom_count": oom_count,
                "first_oom": oom.get("first_oom", ""),
                "last_oom": last_oom,
                "source": "woop_events",
            })

        # Merge strategy: snapshot pods have richer data (container, timestamp,
        # actual restart count). WOOP has authoritative oom_count but at workload
        # level (no container/pod detail). Keep snapshot entries first, then add
        # WOOP entries only for workloads not already covered by snapshot data.
        # Match at workload level: snapshot pod names contain the workload name.
        snapshot_workload_keys = set()
        for o in snapshot.oomkilled_pods:
            ns = o.get("namespace", "")
            pod_name = o.get("name", "")
            # Derive workload name from pod name (strip replicaset/pod hash suffix)
            # e.g. alerts-service-5cf45448d9-xggdt → alerts-service
            parts = pod_name.rsplit("-", 2)
            wl_name = parts[0] if len(parts) >= 3 else pod_name
            snapshot_workload_keys.add(f"{ns}/{wl_name}")

        kept_woop = [
            o for o in woop_oom
            if f"{o['namespace']}/{o['name']}" not in snapshot_workload_keys
        ]
        snapshot_count = len(snapshot.oomkilled_pods)
        snapshot.oomkilled_pods = list(snapshot.oomkilled_pods) + kept_woop

        logger.info(
            "OOM summary: %d from WOOP, %d from snapshot, %d total",
            len(woop_oom), snapshot_count, len(snapshot.oomkilled_pods),
        )

    async def _mcp_oom_events(self, snapshot: SnapshotData) -> None:
        """get_workload_events(OOM_KILL): authoritative, time-windowed OOM event counts.

        Unlike snapshot lastState (shows only most recent OOM per container)
        or restartCount deltas (approximation), this returns every individual
        OOM event in the time window.  Stores a lookup on the snapshot so
        post-collection enrichment can add oom_events_1h to each OOM entry.
        """
        now = datetime.now(timezone.utc)
        args = {
            "cluster_id_or_name": self.cluster_id,
            "from_date": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to_date": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_types": "OOM_KILL",
            "include_event_data": False,
        }
        if self.org_id:
            args["organization_id"] = self.org_id

        data = await self.mcp.call_tool("woop_get_workload_events", args)
        if not data:
            return
        if isinstance(data, str):
            data = json.loads(data)

        items = data.get("items") or data.get("events") or (data if isinstance(data, list) else [])

        # Aggregate: count events per workload (namespace/name)
        counts: dict[str, int] = {}
        for event in items:
            # Event may have top-level workload fields or nested workloads list
            workloads = event.get("workloads", [])
            if not workloads:
                wl_ns = event.get("workloadNamespace", event.get("namespace", ""))
                wl_name = event.get("workloadName", event.get("name", ""))
                if wl_ns or wl_name:
                    workloads = [{"namespace": wl_ns, "name": wl_name}]
            for wl in workloads:
                key = f"{wl.get('namespace', '')}/{wl.get('name', '')}"
                counts[key] = counts.get(key, 0) + 1

        # Store as log signal for post-collection enrichment
        snapshot.log_signals.append({
            "signal": "oom_events_1h",
            "counts": counts,
        })

        logger.info(
            "OOM events (1h): %d events across %d workloads",
            sum(counts.values()), len(counts),
        )

    async def _mcp_config_change_events(self, snapshot: SnapshotData) -> None:
        """get_workload_events(CONFIGURATION_CHANGEDV2): when workloads were onboarded.

        Enriches data_gap entries with 'enabled_since' so alerts can distinguish
        "enabled 30 min ago (warming up)" from "enabled 3 days ago (broken)".
        Fetches config changes from the last 7 days — if a workload has no event,
        it was enabled long ago.
        """
        now = datetime.now(timezone.utc)
        args = {
            "cluster_id_or_name": self.cluster_id,
            "from_date": (now - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to_date": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event_types": "CONFIGURATION_CHANGEDV2",
            "include_event_data": False,
        }
        if self.org_id:
            args["organization_id"] = self.org_id

        data = await self.mcp.call_tool("woop_get_workload_events", args)
        if not data:
            return
        if isinstance(data, str):
            data = json.loads(data)

        items = data.get("items") or data.get("events") or (data if isinstance(data, list) else [])

        # Build lookup: workload key → earliest config change (= onboarding time)
        enabled_since: dict[str, str] = {}
        for event in items:
            event_time = event.get("occurredAt") or event.get("createdAt") or ""
            workloads = event.get("workloads", [])
            if not workloads:
                wl_ns = event.get("workloadNamespace", event.get("namespace", ""))
                wl_name = event.get("workloadName", event.get("name", ""))
                if wl_ns or wl_name:
                    workloads = [{"namespace": wl_ns, "name": wl_name}]
            for wl in workloads:
                key = f"{wl.get('namespace', '')}/{wl.get('name', '')}"
                # Keep the most recent config change (latest onboarding/re-enable)
                existing = enabled_since.get(key, "")
                if event_time > existing:
                    enabled_since[key] = event_time

        if enabled_since:
            snapshot.log_signals.append({
                "signal": "woop_config_changes",
                "enabled_since": enabled_since,
            })
            logger.info(
                "Config change events (7d): %d workloads with recent config changes",
                len(enabled_since),
            )

    # ── Backtest-mode WOOP reconstruction ──────────────────────────────

    async def _mcp_woop_backtest(self, snapshot: SnapshotData) -> None:
        """Reconstruct WOOP state at snapshot_time using historical metrics.

        Strategy (metrics-based, since woop_get_workload_events has no
        RECOMMENDED_REQUESTS_CHANGED retention):
          Phase 1: woop_analyze_with_code (server-side) → screen candidates
                   where current rec or applied exceeds 85% of absurd threshold.
                   Returns top 30 candidates sorted by max resource value.
          Phase 2: woop_get_workload_metrics (historical) → for each candidate,
                   get actual recommended + applied values at snapshot_time.
                   Then run absurd/mismatch detection on historical metrics.
        """
        snap_dt = datetime.fromisoformat(self.snapshot_time.replace("Z", "+00:00"))
        # 24-hour window ending at snapshot_time for metrics.
        # Narrow windows (1h) often return empty containers if the workload
        # had no active metric collection during that period.
        from_str = (snap_dt - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
        to_str = snap_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        t = self.config.thresholds
        known_s2z = self.config.cluster.known_scale_to_zero_workloads
        known_s2z_json = json.dumps(known_s2z)
        screen_mem_gib = t.absurd_memory_gib * 0.85
        screen_cpu = t.absurd_cpu_cores * 0.85

        # ── Phase 1: Server-side candidate screening ──
        screen_code = f'''
SCREEN_MEM_GIB = {screen_mem_gib}
SCREEN_CPU = {screen_cpu}
OUTLIER_RATIO = {t.outlier_median_ratio}
KNOWN_S2Z = {known_s2z_json}

workloads = get_workloads()

candidates = []
data_gaps = []
data_gap_seen = set()
summary = {{"managed": 0, "read_only": 0, "undefined": 0, "total": len(workloads)}}

for wl in workloads:
    wl_name = wl.get("name", "unknown")
    wl_ns = wl.get("namespace", "unknown")
    wl_key = wl_ns + "/" + wl_name
    wl_id = wl.get("id", "")

    wl_config = wl.get("workloadConfigV2") or {{}}
    vpa_cfg = wl_config.get("vpaConfig") or {{}}
    mgmt = vpa_cfg.get("managementOption", "UNDEFINED")
    apply_type = vpa_cfg.get("applyType", "")
    woop_tag = mgmt
    if apply_type:
        woop_tag = woop_tag + "/" + apply_type

    if mgmt == "MANAGED":
        summary["managed"] += 1
    elif mgmt == "READ_ONLY":
        summary["read_only"] += 1
    else:
        summary["undefined"] += 1

    skip = False
    for pat in KNOWN_S2Z:
        if pat in wl_name:
            skip = True
            break
    if skip:
        continue

    # Jobs/CronJobs run to completion — no steady-state to optimize
    wl_kind = wl.get("kind", "")
    if wl_kind in ("Job", "CronJob"):
        continue

    containers = wl.get("containers") or []
    rec_status = wl.get("recommendationStatus", "")

    has_rec = False
    for ctr in containers:
        if ctr.get("recommendation"):
            has_rec = True
            break

    pod_count = wl.get("podCount", 0) or 0

    if mgmt == "MANAGED" and not has_rec and not containers:
        if pod_count > 0 and wl_key not in data_gap_seen:
            if rec_status not in ("STATUS_APPLIED", "STATUS_WAITING"):
                data_gap_seen.add(wl_key)
                data_gaps.append({{"workload": wl_key, "woop": woop_tag,
                                  "kind": wl_kind,
                                  "pod_count": pod_count,
                                  "rec_status": rec_status or "NONE",
                                  "reason": "No recommendation or containers"}})
        continue

    # Only screen MANAGED workloads — READ_ONLY are informational
    if mgmt != "MANAGED":
        continue

    flagged = False
    max_res = 0.0
    for ctr in containers:
        res = ctr.get("resources") or {{}}
        ar = res.get("requests") or {{}}
        al = res.get("limits") or {{}}
        act_mem = ar.get("memoryGib", 0) or 0
        act_cpu = ar.get("cpuCores", 0) or 0
        act_lim = al.get("memoryGib", 0) or 0
        rec_data = ctr.get("recommendation") or {{}}
        rr = rec_data.get("requests") or {{}}
        rec_mem = rr.get("memoryGib", 0) or 0
        rec_cpu = rr.get("cpuCores", 0) or 0
        local_max = max(act_mem, act_lim, rec_mem, act_cpu, rec_cpu)
        if local_max > max_res:
            max_res = local_max
        if act_mem > SCREEN_MEM_GIB or act_cpu > SCREEN_CPU or act_lim > SCREEN_MEM_GIB:
            flagged = True
        if rec_mem > SCREEN_MEM_GIB or rec_cpu > SCREEN_CPU:
            flagged = True
        if act_mem and act_lim and (act_lim / act_mem) > OUTLIER_RATIO:
            flagged = True

    if flagged and wl_id:
        candidates.append({{"wl_key": wl_key, "wl_id": wl_id, "woop_tag": woop_tag, "max_res": max_res}})

# Sort by max resource descending, cap at 30
candidates.sort(key=lambda c: c["max_res"], reverse=True)
candidates = candidates[:30]
# Strip max_res before returning (not needed downstream)
for c in candidates:
    c.pop("max_res", None)

result = {{"candidates": candidates, "data_gaps": data_gaps, "summary": summary}}
'''

        screen_args = {
            "cluster_id_or_name": self.cluster_id,
            "analysis_code": screen_code,
            "description": "WOOP backtest: screen candidates for historical metrics verification",
            "timeout": 45,
        }
        if self.org_id:
            screen_args["organization_id"] = self.org_id

        screen_data = await self.mcp.call_tool("woop_analyze_with_code", screen_args)
        if not screen_data:
            raise RuntimeError("woop_analyze_with_code (backtest screen) returned empty")
        if isinstance(screen_data, str):
            screen_data = json.loads(screen_data)
        if isinstance(screen_data, dict) and "error" in screen_data and "error_type" in screen_data:
            raise RuntimeError(
                f"woop_analyze_with_code error: {screen_data['error']} "
                f"(hint: {screen_data.get('hint', 'none')})"
            )
        if isinstance(screen_data, dict) and "result" in screen_data and screen_data.get("success"):
            screen_data = screen_data["result"]

        candidates = screen_data.get("candidates", [])
        data_gaps = screen_data.get("data_gaps", [])
        woop_summary = screen_data.get("summary", {})

        total_workloads = woop_summary.get("total", 0)
        logger.info(
            "BACKTEST WOOP: %d workloads, %d candidates flagged for "
            "historical metrics verification (server-side screening)",
            total_workloads, len(candidates),
        )

        # ── Phase 2: Fetch historical metrics for candidates ──
        mismatches = []
        absurd = []
        metrics_fetched = 0

        if candidates:
            # Fetch metrics concurrently in batches of 20
            batch_size = 20
            for i in range(0, len(candidates), batch_size):
                batch = candidates[i:i + batch_size]
                metric_args_list = []
                for c in batch:
                    m_args = {
                        "cluster_id_or_name": self.cluster_id,
                        "workload_id": c["wl_id"],
                        "from_time": from_str,
                        "to_time": to_str,
                        "detail": "analysis",
                    }
                    if self.org_id:
                        m_args["organization_id"] = self.org_id
                    metric_args_list.append(m_args)

                results = await asyncio.gather(
                    *[self.mcp.call_tool("woop_get_workload_metrics", args)
                      for args in metric_args_list],
                    return_exceptions=True,
                )

                for cand, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.warning(
                            "BACKTEST metrics failed for %s: %s",
                            cand["wl_key"], result,
                        )
                        continue

                    metrics_fetched += 1
                    if isinstance(result, str):
                        try:
                            result = json.loads(result)
                        except json.JSONDecodeError:
                            continue
                    if not isinstance(result, dict):
                        continue

                    wl_key = cand["wl_key"]
                    woop_tag = cand["woop_tag"]

                    for ctr_m in result.get("containers", []):
                        c_name = ctr_m.get("name", "")
                        mem = ctr_m.get("memory_gib", {})
                        cpu = ctr_m.get("cpu_cores", {})

                        rec_mem_gib = mem.get("recommended", 0) or 0
                        req_mem_gib = mem.get("requested", 0) or 0
                        rec_cpu_cores = cpu.get("recommended", 0) or 0
                        req_cpu_cores = cpu.get("requested", 0) or 0

                        # Absurd recommendation checks
                        if rec_mem_gib > t.absurd_memory_gib:
                            absurd.append({
                                "workload": wl_key, "container": c_name,
                                "woop": woop_tag,
                                "recommended_memory_gib": round(rec_mem_gib, 1),
                                "applied_memory_gib": round(req_mem_gib, 1),
                                "reason": f"Historical WOOP recommended "
                                          f"{rec_mem_gib:.1f} GiB memory "
                                          f"(applied {req_mem_gib:.1f} GiB)",
                                "source": "woop_metrics_backtest",
                            })
                        if rec_cpu_cores > t.absurd_cpu_cores:
                            absurd.append({
                                "workload": wl_key, "container": c_name,
                                "woop": woop_tag,
                                "recommended_cpu_cores": round(rec_cpu_cores, 1),
                                "applied_cpu_cores": round(req_cpu_cores, 1),
                                "reason": f"Historical WOOP recommended "
                                          f"{rec_cpu_cores:.1f} CPU cores "
                                          f"(applied {req_cpu_cores:.1f})",
                                "source": "woop_metrics_backtest",
                            })

                        # Also flag absurd applied values
                        if req_mem_gib > t.absurd_memory_gib:
                            absurd.append({
                                "workload": wl_key, "container": c_name,
                                "woop": woop_tag,
                                "applied_memory_gib": round(req_mem_gib, 1),
                                "recommended_memory_gib": round(rec_mem_gib, 1),
                                "reason": f"Applied {req_mem_gib:.1f} GiB memory "
                                          f"(recommended {rec_mem_gib:.1f} GiB)",
                                "source": "woop_metrics_backtest",
                            })
                        if req_cpu_cores > t.absurd_cpu_cores:
                            absurd.append({
                                "workload": wl_key, "container": c_name,
                                "woop": woop_tag,
                                "applied_cpu_cores": round(req_cpu_cores, 1),
                                "recommended_cpu_cores": round(rec_cpu_cores, 1),
                                "reason": f"Applied {req_cpu_cores:.1f} CPU cores "
                                          f"(recommended {rec_cpu_cores:.1f})",
                                "source": "woop_metrics_backtest",
                            })

                        # Mismatch checks (requires both % AND absolute delta)
                        if req_mem_gib and rec_mem_gib:
                            pct = abs(rec_mem_gib - req_mem_gib) / req_mem_gib * 100
                            abs_delta = abs(rec_mem_gib - req_mem_gib)
                            if pct > t.recommendation_mismatch_pct and abs_delta >= t.mismatch_min_memory_gib:
                                mismatches.append({
                                    "workload": wl_key, "container": c_name,
                                    "woop": woop_tag,
                                    "recommended_memory_gib": round(rec_mem_gib, 1),
                                    "actual_memory_gib": round(req_mem_gib, 1),
                                    "diff_pct": round(pct, 1),
                                    "source": "woop_metrics_backtest",
                                })
                        if req_cpu_cores and rec_cpu_cores:
                            pct = abs(rec_cpu_cores - req_cpu_cores) / req_cpu_cores * 100
                            abs_delta = abs(rec_cpu_cores - req_cpu_cores)
                            if pct > t.recommendation_mismatch_pct and abs_delta >= t.mismatch_min_cpu_cores:
                                mismatches.append({
                                    "workload": wl_key, "container": c_name,
                                    "woop": woop_tag,
                                    "recommended_cpu_cores": round(rec_cpu_cores, 1),
                                    "actual_cpu_cores": round(req_cpu_cores, 1),
                                    "diff_pct": round(pct, 1),
                                    "source": "woop_metrics_backtest",
                                })

        snapshot.woop_workloads = []
        snapshot.recommendation_mismatches = mismatches
        snapshot.absurd_recommendations = absurd
        snapshot.data_gaps = data_gaps
        snapshot.data_gaps_total = len(data_gaps)
        snapshot.recommendation_mismatches_total = len(mismatches)
        snapshot.absurd_recommendations_total = len(absurd)

        snapshot.log_signals.append({
            "signal": "woop_management_summary",
            "managed": woop_summary.get("managed", 0),
            "read_only": woop_summary.get("read_only", 0),
            "undefined": woop_summary.get("undefined", 0),
            "total": total_workloads,
            "backtest": True,
            "candidates_screened": len(candidates),
            "metrics_fetched": metrics_fetched,
        })

        logger.info(
            "BACKTEST WOOP: %d workloads, %d candidates, %d metrics fetched, "
            "%d mismatches, %d absurd, %d data gaps",
            total_workloads, len(candidates), metrics_fetched,
            len(mismatches), len(absurd), len(data_gaps),
        )

    async def _mcp_oom_summary_backtest(self, snapshot: SnapshotData) -> None:
        """woop_get_oom_summary with historical time range for backtest mode."""
        snap_dt = datetime.fromisoformat(self.snapshot_time.replace("Z", "+00:00"))
        from_dt = snap_dt - timedelta(hours=1)

        bt_oom_args = {
            "cluster_id_or_name": self.cluster_id,
            "from_date": from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to_date": snap_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 20,
        }
        if self.org_id:
            bt_oom_args["organization_id"] = self.org_id
        data = await self.mcp.call_tool("woop_get_oom_summary", bt_oom_args)
        if not data:
            logger.info("BACKTEST OOM: no data returned for window ending %s", self.snapshot_time)
            return
        if isinstance(data, str):
            data = json.loads(data)

        oom_workloads = data if isinstance(data, list) else data.get("workloads", [])

        woop_oom = []
        for oom in oom_workloads:
            oom_count = oom.get("oom_count", 0)
            last_oom = oom.get("last_oom", "")
            woop_oom.append({
                "namespace": oom.get("namespace", ""),
                "name": oom.get("workload_name", ""),
                "restart_count": oom_count,
                "container": "",
                "last_oomkill_time": last_oom,
                "oom_count": oom_count,
                "first_oom": oom.get("first_oom", ""),
                "last_oom": last_oom,
                "source": "woop_events_backtest",
            })

        # Prefer snapshot entries (richer: container, timestamp, pod name).
        # Only add WOOP entries for workloads not already in snapshot data.
        snapshot_workload_keys = set()
        for o in snapshot.oomkilled_pods:
            ns = o.get("namespace", "")
            pod_name = o.get("name", "")
            parts = pod_name.rsplit("-", 2)
            wl_name = parts[0] if len(parts) >= 3 else pod_name
            snapshot_workload_keys.add(f"{ns}/{wl_name}")

        kept_woop = [
            o for o in woop_oom
            if f"{o['namespace']}/{o['name']}" not in snapshot_workload_keys
        ]
        snapshot_count = len(snapshot.oomkilled_pods)
        snapshot.oomkilled_pods = list(snapshot.oomkilled_pods) + kept_woop

        logger.info(
            "BACKTEST OOM: %d from WOOP events, %d from snapshot, %d total "
            "(window: %s to %s)",
            len(woop_oom), snapshot_count, len(snapshot.oomkilled_pods),
            from_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), self.snapshot_time,
        )

    # ── End backtest methods ──────────────────────────────────────────

    async def _mcp_resource_ratios(self, snapshot: SnapshotData) -> None:
        """woop_get_workload_resource_ratios: flag tight limits + collect memory usage."""
        ratio_args = {"cluster_id_or_name": self.cluster_id}
        if self.org_id:
            ratio_args["organization_id"] = self.org_id
        data = await self.mcp.call_tool("woop_get_workload_resource_ratios", ratio_args)
        if not data:
            return
        if isinstance(data, str):
            data = json.loads(data)

        ratios = data if isinstance(data, list) else data.get("containers", [])

        # Flag containers where memory limit ≈ request (ratio < 1.1)
        tight_limits = [
            r for r in ratios
            if r.get("mem_ratio") is not None and r["mem_ratio"] < 1.1
            and r.get("request_mem_mib", 0) > 50
        ]

        if tight_limits:
            logger.info(
                "Resource ratios: %d containers with tight memory limits (<1.1x)",
                len(tight_limits),
            )
            snapshot.log_signals.append({
                "signal": "tight_memory_limits",
                "count": len(tight_limits),
                "sample": [
                    f"{r.get('namespace')}/{r.get('workload_name')}: "
                    f"ratio={r.get('mem_ratio'):.2f}"
                    for r in tight_limits[:5]
                ],
            })

        # Collect top memory consumers for leak detection across snapshots.
        # Sort by request_mem_mib descending, keep top 30 workloads.
        mem_entries = [
            r for r in ratios
            if r.get("request_mem_mib") is not None and r["request_mem_mib"] > 0
        ]
        mem_entries.sort(key=lambda r: r.get("request_mem_mib", 0), reverse=True)

        for r in mem_entries[:30]:
            snapshot.workload_memory_usage.append({
                "namespace": r.get("namespace", ""),
                "workload": r.get("workload_name", ""),
                "container": r.get("container_name", ""),
                "request_mem_mib": r.get("request_mem_mib", 0),
                "limit_mem_mib": r.get("limit_mem_mib", 0),
                "mem_ratio": r.get("mem_ratio", 0),
            })

    async def _mcp_cluster_details(self, snapshot: SnapshotData) -> None:
        """get_cluster_details + get_cluster_snapshot_summary."""
        # Cluster details (requires organization_id with JWT auth)
        args = {"cluster_id_or_name": self.cluster_id}
        if self.org_id:
            args["organization_id"] = self.org_id
        details = await self.mcp.call_tool("get_cluster_details", args)
        if details:
            if isinstance(details, str):
                details = json.loads(details)
            region = details.get("region", {})
            region_name = region.get("name", "unknown") if isinstance(region, dict) else str(region)
            # Store cluster metadata without overwriting node_summary from snapshot
            snapshot.log_signals.append({
                "signal": "cluster_metadata",
                "count": 1,
                "sample": [{
                    "status": details.get("status", "unknown"),
                    "provider": details.get("providerType", "unknown"),
                    "region": region_name,
                    "k8s_version": details.get("kubernetesVersion", "unknown"),
                    "agent_status": details.get("agentStatus", "unknown"),
                }],
            })

        # Snapshot summary for node count cross-check
        summary = await self.mcp.get_snapshot_summary(self.cluster_id)
        if summary:
            if isinstance(summary, str):
                summary = json.loads(summary)
            counts = summary.get("counts", {})
            # Actual keys: nodeList, podList (not nodes, pods)
            node_count = counts.get("nodeList", 0) or counts.get("nodes", 0)
            pod_count = counts.get("podList", 0) or counts.get("pods", 0)
            if node_count:
                snapshot.node_count = node_count
            if pod_count and not snapshot.total_pods:
                snapshot.total_pods = pod_count

    async def _mcp_loki_signals(self, snapshot: SnapshotData) -> None:
        """loki_query: evictor blocks, WA mutations, safety mechanism events.

        Uses raw LogQL because the structured log tools (search_logs,
        get_logs_by_cluster_and_service) don't support service="workload-autoscaler"
        — they only map known service names like "autoscaler", "agent", "evictor".
        """
        queries = [
            ("wa_mutation_errors",
             '{app="workload-autoscaler"} '
             f'|= "{self.cluster_id}" '
             '|~ "overflow|integer|mutation.*error|webhook.*fail"'),
            ("oomkill_safety",
             '{app="workload-autoscaler"} '
             f'|= "{self.cluster_id}" '
             '|~ "oom.*kill|safety.*mechanism|disable.*after"'),
            ("evictor_blocked",
             '{app="evictor"} '
             f'|= "{self.cluster_id}" '
             '|~ "blocked|cannot.*evict|skip"'),
            ("webhook_exporter_failure",
             '{app=~"workload-autoscaler|workload-autoscaler-exporter"} '
             f'|= "{self.cluster_id}" '
             '|~ "webhook.*fail|webhook.*timeout|metric.*limit|'
             'exporter.*error|collection.*limit"'),
        ]

        for name, query in queries:
            try:
                resp = await self.mcp.loki_query(query, start="now-15m", limit=50)
                if not resp:
                    continue
                if isinstance(resp, str):
                    resp = json.loads(resp)

                entries = []
                result_data = resp.get("data", resp) if isinstance(resp, dict) else {}
                for stream in result_data.get("result", []):
                    for val in stream.get("values", []):
                        entries.append(val[1] if len(val) > 1 else str(val))

                if entries:
                    snapshot.log_signals.append({
                        "signal": name,
                        "count": len(entries),
                        "sample": entries[:3],
                    })
            except Exception as e:
                logger.warning("Loki '%s' failed: %s", name, e)

        if snapshot.log_signals:
            logger.info("Loki: %s", ", ".join(
                f"{s.get('signal', '?')}={s.get('count', '-')}"
                for s in snapshot.log_signals
                if s.get("signal", "").startswith(("webhook", "wa_", "evictor"))
                or s.get("count") is not None
            ))

    async def _mcp_org_agent_health(self, snapshot: SnapshotData) -> None:
        """list_clusters: org-wide check for agent-offline clusters."""
        if not self.org_id:
            return
        args = {"organization_id": self.org_id}
        data = await self.mcp.call_tool("list_clusters", args)
        if not data:
            return
        if isinstance(data, str):
            data = json.loads(data)

        clusters = data.get("clusters") or data.get("items") or (data if isinstance(data, list) else [])
        if not isinstance(clusters, list):
            return

        offline = []
        for cl in clusters:
            agent_status = cl.get("agentStatus", "unknown")
            if agent_status != "online":
                offline.append({
                    "cluster_id": cl.get("id", ""),
                    "name": cl.get("name", ""),
                    "agent_status": agent_status,
                })

        if offline:
            snapshot.log_signals.append({
                "signal": "org_agent_offline",
                "count": len(offline),
                "sample": offline[:5],
            })
            logger.info("Org health: %d clusters with agent not online", len(offline))

    async def _mcp_snapshot_fallback(self, snapshot: SnapshotData) -> None:
        """query_snapshot_pods: fallback if analyze_snapshot_with_code fails.

        Fetches pods via query_snapshot_pods (max 1000) and processes locally.
        Less efficient than analyze_snapshot_with_code but doesn't require
        RestrictedPython code execution on the server.
        """
        args = {"cluster_id_or_name": self.cluster_id, "max_results": 1000}
        if self.snapshot_time:
            args["snapshot_time"] = self.snapshot_time
        data = await self.mcp.call_tool("query_snapshot_pods", args)
        if not data:
            raise RuntimeError("query_snapshot_pods returned empty")
        if isinstance(data, str):
            data = json.loads(data)

        items = data if isinstance(data, list) else data.get("pods", data.get("items", []))
        running = pending = crashloop = 0
        oomkilled = []
        crashloop_details = []
        pending_details = []

        for item in items:
            md = item.get("metadata", {})
            st = item.get("status", {})
            ns = md.get("namespace", "")
            name = md.get("name", "")
            phase = st.get("phase", "Unknown")

            if phase == "Running":
                running += 1
            elif phase == "Pending":
                pending += 1
                conds = st.get("conditions", [])
                sched_false = False
                sched_seen = False
                pend_reason = ""
                for cond in conds:
                    if isinstance(cond, dict) and cond.get("type") == "PodScheduled":
                        sched_seen = True
                        if cond.get("status") == "False":
                            sched_false = True
                            pend_reason = cond.get("message", cond.get("reason", ""))
                if not sched_seen or sched_false:
                    pending_details.append({"namespace": ns, "name": name, "reason": pend_reason})

            for cs in st.get("containerStatuses", []):
                if isinstance(cs, dict):
                    term = cs.get("lastState", {}).get("terminated", {})
                    if term.get("reason") == "OOMKilled":
                        oomkilled.append({
                            "namespace": ns, "name": name,
                            "container": cs.get("name", ""),
                            "restart_count": cs.get("restartCount", 0),
                        })
                    wait = cs.get("state", {}).get("waiting", {})
                    if wait.get("reason") == "CrashLoopBackOff":
                        crashloop += 1
                        crashloop_details.append({
                            "namespace": ns, "name": name,
                            "container": cs.get("name", ""),
                            "restart_count": cs.get("restartCount", 0),
                        })

        snapshot.total_pods = len(items)
        snapshot.running_pods = running
        snapshot.pending_pods = pending
        snapshot.pending_pods_detail = pending_details
        snapshot.crashloop_pods = crashloop
        snapshot.crashloop_pods_detail = crashloop_details
        snapshot.oomkilled_pods = oomkilled

        logger.info(
            "Fallback (query_snapshot_pods): %d pods (%d OOM, %d Pending, %d CrashLoop)",
            len(items), len(oomkilled), pending, crashloop,
        )

    # ══════════════════════════════════════════════════════════════════
    #  TIER 2: SNAPSHOT HYBRID — GCS direct + lightweight WOOP API
    # ══════════════════════════════════════════════════════════════════

    async def _collect_via_snapshot_hybrid(self, snapshot: SnapshotData) -> SnapshotData:
        """Tier 2 fallback: snapshot-cli reads GCS directly + public API for WOOP.

        Two parallel tracks:
          Left:  snapshot-cli → podList, nodeList, deploymentList,
                 recommendationList, podmetricsList, eventList
          Right: public API → WOOP managementOption, OOM events,
                 erroring workloads, cluster details

        Merge: snapshot recs × WOOP managementOption → mismatches + absurd + data_gaps.

        Advantages over pure API fallback:
          - Full pod data (OOM from containerStatuses, agent restarts, pending details)
          - Full node data (conditions, capacity, ready status)
          - Deployment health (zero-ready detection)
          - Memory metrics from podmetricsList (leak trend detection)
          - No RestrictedPython sandbox — native Python analysis
        """
        # ── Left track: GCS snapshot sections (single CLI call for all 6) ──
        snap_task = self.snap.get_all_watchdog_sections()

        # ── Right track: 4 lightweight API calls ──
        async with httpx.AsyncClient(
            headers=self.config.castai.auth_headers,
            timeout=self.timeout,
        ) as client:
            api_results = await asyncio.gather(
                snap_task,
                self._api_cluster_details(client, snapshot),
                self._api_woop_mgmt_map(client),
                self._api_oom_events(client, snapshot),
                self._api_erroring_workloads(client, snapshot),
                return_exceptions=True,
            )

        names = [
            "snapshot_sections", "cluster_details", "woop_mgmt_map",
            "oom_events", "erroring_workloads",
        ]
        snap_data = None
        woop_mgmt_map: dict[str, dict] = {}

        for name, result in zip(names, api_results):
            if isinstance(result, Exception):
                err = f"{name}: {type(result).__name__}: {result}"
                logger.error("Snapshot hybrid failed: %s", err)
                snapshot.collection_errors.append(err)
                if name == "snapshot_sections":
                    # If GCS fails, the whole tier fails — let caller fall through
                    raise result
            elif name == "snapshot_sections":
                snap_data = result
            elif name == "woop_mgmt_map":
                woop_mgmt_map = result or {}

        if not snap_data:
            raise RuntimeError("snapshot_sections returned empty")

        # ── Analyze snapshot data locally (native Python, no sandbox) ──
        ref_time = datetime.now(timezone.utc)

        # Pod health
        pod_result = analyze_pod_health(snap_data["pods"], ref_time=ref_time)
        snapshot.total_pods = pod_result["total_pods"]
        snapshot.running_pods = pod_result["running"]
        snapshot.pending_pods = pod_result["pending"]
        snapshot.crashloop_pods = pod_result["crashloop"]
        snapshot.crashloop_pods_detail = pod_result["crashloop_details"]
        snapshot.pending_pods_detail = pod_result["pending_details"]
        snapshot.oomkilled_pods = pod_result["oomkilled"]
        snapshot.agent_pods = pod_result["agent_pods"]
        snapshot.agent_restarts_last_hour = pod_result["agent_total_restarts"]

        # Node health
        node_result = analyze_node_health(snap_data["nodes"])
        snapshot.node_count = node_result["node_count"]
        snapshot.nodes = node_result["node_summary"]

        # Deployment health
        unhealthy_deps = analyze_deployment_health(snap_data["deployments"])
        if unhealthy_deps:
            snapshot.log_signals.append({
                "signal": "unhealthy_deployments",
                "count": len(unhealthy_deps),
                "sample": unhealthy_deps[:5],
            })

        # Pending pod details signal
        if pod_result["pending_details"]:
            snapshot.log_signals.append({
                "signal": "pending_pod_details",
                "count": len(pod_result["pending_details"]),
                "sample": pod_result["pending_details"][:5],
            })

        # ── Merge snapshot recs × WOOP managementOption ──
        gcs_recs = snap_data["recommendations"]
        logger.info(
            "Hybrid rec data: GCS recommendations=%d, WOOP workloads=%d",
            len(gcs_recs), len(woop_mgmt_map),
        )
        if woop_mgmt_map:
            rec_result = analyze_recommendations(
                snapshot_recs=gcs_recs,
                woop_mgmt_map=woop_mgmt_map,
                thresholds=self.config.thresholds,
                known_s2z=self.config.cluster.known_scale_to_zero_workloads,
            )
            snapshot.recommendation_mismatches = rec_result["mismatches"]
            snapshot.absurd_recommendations = rec_result["absurd"]
            snapshot.data_gaps = rec_result["data_gaps"]
            snapshot.data_gaps_total = len(rec_result["data_gaps"])
            snapshot.recommendation_mismatches_total = len(rec_result["mismatches"])
            snapshot.absurd_recommendations_total = len(rec_result["absurd"])
            snapshot.woop_workloads = []  # stripped — evaluator uses findings

            woop_summary = rec_result["summary"]
            snapshot.log_signals.append({
                "signal": "woop_management_summary",
                "managed": woop_summary.get("managed", 0),
                "read_only": woop_summary.get("read_only", 0),
                "undefined": woop_summary.get("undefined", 0),
                "total": woop_summary.get("total", 0),
            })

            # Rec lookup for OOM enrichment
            rec_lookup = rec_result.get("rec_lookup", {})
            if rec_lookup:
                snapshot.log_signals.append({
                    "signal": "woop_rec_lookup",
                    "lookup": rec_lookup,
                })
        else:
            snapshot.collection_errors.append(
                "WOOP managementOption unavailable — mismatch/absurd detection skipped"
            )

        # ── Memory metrics from podmetricsList ──
        mem_usage = analyze_pod_metrics(snap_data["pod_metrics"])
        snapshot.workload_memory_usage = [
            {
                "namespace": m["namespace"],
                "workload": m["workload"],
                "container": m["container"],
                "usage_bytes": m["usage_bytes"],
            }
            for m in mem_usage
        ]

        # ── Post-collection enrichment (same as MCP path) ──
        self._enrich_oom_entries(snapshot)
        self._enrich_data_gaps(snapshot)

        # Drop ephemeral log signals
        ephemeral = {"woop_rec_lookup", "oom_events_1h", "woop_config_changes"}
        snapshot.log_signals = [
            s for s in snapshot.log_signals if s.get("signal") not in ephemeral
        ]

        if not self.force_tier:
            snapshot.collection_errors.append(
                "info: Running via Tier 2 (snapshot-cli + public API). "
                "No Loki logs. Set CASTAI_MCP_URL for full fidelity."
            )

        logger.info(
            "Snapshot hybrid: %d pods (%d OOM, %d Pending, %d CrashLoop), "
            "%d nodes, %d mismatches, %d absurd, %d data gaps",
            snapshot.total_pods, len(snapshot.oomkilled_pods),
            snapshot.pending_pods, snapshot.crashloop_pods,
            snapshot.node_count,
            len(snapshot.recommendation_mismatches),
            len(snapshot.absurd_recommendations),
            len(snapshot.data_gaps),
        )
        return snapshot

    async def _api_woop_mgmt_map(
        self, client: httpx.AsyncClient,
    ) -> dict[str, dict]:
        """WOOP API call: fetch managementOption + recommendations per workload.

        Returns dict of workload_key → {mgmt, apply_type, pod_count, rec_status,
        kind, containers} for merging with snapshot recommendations.

        Includes recommendations so analyze_recommendations can fall back to
        API rec values when the GCS snapshot's recommendationList is empty
        (e.g. section name mismatch or missing section).
        """
        # containers for actual resource values, recommendations as GCS fallback
        params = {"includeRecommendations": "true", "includeContainers": "true"}
        workloads = await self._paginated_get(
            client,
            f"/v1/workload-autoscaling/clusters/{self.cluster_id}/workloads",
            params=params, items_key="workloads",
        )

        mgmt_map: dict[str, dict] = {}
        for wl in workloads:
            wl_name = wl.get("workloadName") or wl.get("name") or "unknown"
            wl_ns = wl.get("workloadNamespace") or wl.get("namespace") or "unknown"
            wl_key = f"{wl_ns}/{wl_name}"

            wl_config = wl.get("workloadConfigV2") or {}
            vpa_cfg = wl_config.get("vpaConfig") or {}
            mgmt = vpa_cfg.get("managementOption", "UNDEFINED")
            apply_type = vpa_cfg.get("applyType", "")

            raw_rec_status = wl.get("recommendationStatus", "")
            rec_status = (
                raw_rec_status.get("type", "") if isinstance(raw_rec_status, dict)
                else str(raw_rec_status)
            )

            mgmt_map[wl_key] = {
                "mgmt": mgmt,
                "apply_type": apply_type,
                "pod_count": wl.get("podCount", 0) or 0,
                "rec_status": rec_status,
                "kind": wl.get("kind") or wl.get("workloadKind") or "",
                "containers": wl.get("containers", []),
            }

        logger.info("WOOP mgmt map: %d workloads fetched via API", len(mgmt_map))
        return mgmt_map

    # ══════════════════════════════════════════════════════════════════
    #  TIER 3: PUBLIC API FALLBACK — limited data, no snapshot/Loki
    # ══════════════════════════════════════════════════════════════════

    async def _collect_via_public_api(self, snapshot: SnapshotData) -> SnapshotData:
        """Fallback: collect what we can from the public REST API."""
        async with httpx.AsyncClient(
            headers=self.config.castai.auth_headers,
            timeout=self.timeout,
        ) as client:
            results = await asyncio.gather(
                self._api_cluster_details(client, snapshot),
                self._api_woop_workloads(client, snapshot),
                self._api_oom_events(client, snapshot),
                self._api_erroring_workloads(client, snapshot),
                return_exceptions=True,
            )
            names = ["cluster_details", "woop_workloads", "oom_events", "erroring_workloads"]
            for name, result in zip(names, results):
                if isinstance(result, Exception):
                    err = f"{name}: {type(result).__name__}: {result}"
                    logger.error("API collector failed: %s", err)
                    snapshot.collection_errors.append(err)

        if not self.force_tier:
            snapshot.collection_errors.append(
                "info: Running without MCP server. Set CASTAI_MCP_URL for "
                "snapshot pod/node detail, Loki logs, and resource ratio checks."
            )
        return snapshot

    async def _api_cluster_details(
        self, client: httpx.AsyncClient, snapshot: SnapshotData
    ) -> None:
        resp = await self._api_get(
            client, f"/v1/kubernetes/external-clusters/{self.cluster_id}",
        )
        if resp:
            snapshot.node_count = resp.get("nodeCount", 0) or resp.get("node_count", 0)
            agent_status = resp.get("agentStatus", "unknown")

            # Detect stale agent heartbeat — agentSnapshotReceivedAt > 10 min ago
            agent_phase = "Running" if agent_status in ("online", "connected") else agent_status
            heartbeat_str = resp.get("agentSnapshotReceivedAt", "")
            stale_minutes = 0.0
            if heartbeat_str:
                try:
                    hb_time = datetime.fromisoformat(heartbeat_str.replace("Z", "+00:00"))
                    stale_minutes = (datetime.now(timezone.utc) - hb_time).total_seconds() / 60
                    if stale_minutes > 10:
                        agent_phase = f"StaleHeartbeat ({stale_minutes:.0f}m)"
                        snapshot.log_signals.append({
                            "signal": "agent_stale_heartbeat",
                            "last_heartbeat": heartbeat_str,
                            "stale_minutes": round(stale_minutes, 1),
                        })
                except (ValueError, TypeError):
                    pass

            snapshot.agent_pods = [{
                "name": "castai-agent", "namespace": "castai-agent",
                "phase": agent_phase,
                "restart_count": 0, "source": "public-api",
            }]

    async def _api_woop_workloads(
        self, client: httpx.AsyncClient, snapshot: SnapshotData
    ) -> None:
        """Same WOOP analysis but via direct API call."""
        params = {"includeRecommendations": "true", "includeContainers": "true"}
        workloads = await self._paginated_get(
            client,
            f"/v1/workload-autoscaling/clusters/{self.cluster_id}/workloads",
            params=params, items_key="workloads",
        )

        # Reuse the same mismatch/absurd logic
        mismatches, absurd, data_gaps = [], [], []
        data_gap_seen: set[str] = set()
        woop_summary = {"managed": 0, "read_only": 0, "undefined": 0}
        t = self.config.thresholds
        known_s2z = self.config.cluster.known_scale_to_zero_workloads

        for wl in workloads:
            wl_name = wl.get("workloadName") or wl.get("name") or "unknown"
            wl_ns = wl.get("workloadNamespace") or wl.get("namespace") or "unknown"
            wl_key = f"{wl_ns}/{wl_name}"

            # Extract WOOP config from the correct nested path
            wl_config = wl.get("workloadConfigV2", {})
            vpa_cfg = wl_config.get("vpaConfig", {})
            mgmt = vpa_cfg.get("managementOption", "UNDEFINED")
            apply_type = vpa_cfg.get("applyType", "")
            mem_limit_mult = vpa_cfg.get("memory", {}).get("limit", {}).get("multiplier", 0)

            woop_tag = f"{mgmt}"
            if apply_type:
                woop_tag += f"/{apply_type}"

            if mgmt == "MANAGED":
                woop_summary["managed"] += 1
            elif mgmt == "READ_ONLY":
                woop_summary["read_only"] += 1
            else:
                woop_summary["undefined"] += 1

            if any(pat in wl_name for pat in known_s2z):
                continue

            # Jobs/CronJobs run to completion — no steady-state to optimize
            wl_kind = wl.get("kind") or wl.get("workloadKind") or ""
            if wl_kind in ("Job", "CronJob"):
                continue

            containers = wl.get("containers", [])
            # API returns recommendationStatus as {"type": "STATUS_APPLIED", ...}
            raw_rec_status = wl.get("recommendationStatus", "")
            rec_status = (
                raw_rec_status.get("type", "") if isinstance(raw_rec_status, dict)
                else str(raw_rec_status)
            )

            # Check if any container has a recommendation
            has_rec = False
            for ctr in containers:
                if ctr.get("recommendation"):
                    has_rec = True
                    break

            pod_count = wl.get("podCount", 0) or 0

            if mgmt == "MANAGED" and not has_rec:
                if pod_count > 0 and wl_key not in data_gap_seen:
                    # STATUS_APPLIED/STATUS_WAITING = lookback pending, not a real gap
                    if rec_status not in ("STATUS_APPLIED", "STATUS_WAITING"):
                        data_gap_seen.add(wl_key)
                        data_gaps.append({
                            "workload": wl_key, "woop": woop_tag,
                            "kind": wl_kind,
                            "pod_count": pod_count,
                            "rec_status": rec_status or "NONE",
                            "reason": "No active recommendation",
                        })
                continue
            # Only flag MANAGED workloads for mismatch/absurd
            if mgmt != "MANAGED":
                continue
            if not has_rec or not containers:
                continue

            for ctr in containers:
                c_name = ctr.get("containerName", ctr.get("name", ""))
                # API returns values in GiB and cores
                res = ctr.get("resources") or {}
                ar = res.get("requests") or {}
                al = res.get("limits") or {}
                act_mem_gib = ar.get("memoryGib", 0) or 0
                act_cpu_cores = ar.get("cpuCores", 0) or 0
                act_lim_mem_gib = al.get("memoryGib", 0) or 0

                orig_res = ctr.get("originalResources") or {}
                orig_req = orig_res.get("requests") or {}
                orig_mem_gib = orig_req.get("memoryGib", 0) or 0
                orig_cpu_cores = orig_req.get("cpuCores", 0) or 0

                rec_data = ctr.get("recommendation") or {}
                rr = rec_data.get("requests") or {}
                rec_mem_gib = rr.get("memoryGib", 0) or 0
                rec_cpu_cores = rr.get("cpuCores", 0) or 0

                if rec_mem_gib > t.absurd_memory_gib:
                    absurd.append({"workload": wl_key, "container": c_name,
                                   "woop": woop_tag,
                                   "recommended_memory_gib": round(rec_mem_gib, 1),
                                   "reason": f"WOOP recommends {rec_mem_gib:.0f} GiB"})

                # Ratio breach: recommendation >= 10x current request (memory)
                if act_mem_gib > 0 and rec_mem_gib >= act_mem_gib * t.outlier_median_ratio:
                    rec_ratio = round(rec_mem_gib / act_mem_gib, 1)
                    absurd.append({
                        "workload": wl_key, "container": c_name,
                        "woop": woop_tag,
                        "sub_type": "ratio_breach",
                        "recommended_memory_gib": round(rec_mem_gib, 1),
                        "applied_memory_gib": round(act_mem_gib, 1),
                        "limit_request_ratio": rec_ratio,
                        "reason": f"Rec {rec_mem_gib:.1f} GiB is {rec_ratio}x "
                                  f"the current request {act_mem_gib:.1f} GiB",
                    })
                # Ratio breach: recommendation >= 10x current request (CPU)
                if act_cpu_cores > 0 and rec_cpu_cores >= act_cpu_cores * t.outlier_median_ratio:
                    cpu_ratio = round(rec_cpu_cores / act_cpu_cores, 1)
                    absurd.append({
                        "workload": wl_key, "container": c_name,
                        "woop": woop_tag,
                        "sub_type": "ratio_breach",
                        "recommended_cpu_cores": round(rec_cpu_cores, 1),
                        "applied_cpu_cores": round(act_cpu_cores, 1),
                        "limit_request_ratio": cpu_ratio,
                        "reason": f"Rec {rec_cpu_cores:.1f} CPU is {cpu_ratio}x "
                                  f"the current request {act_cpu_cores:.1f} CPU",
                    })

                # Baseline ratio breach: current applied >= 10x original baseline (memory)
                if orig_mem_gib > 0 and act_mem_gib >= orig_mem_gib * t.outlier_median_ratio:
                    base_ratio = round(act_mem_gib / orig_mem_gib, 1)
                    absurd.append({
                        "workload": wl_key, "container": c_name,
                        "woop": woop_tag,
                        "sub_type": "baseline_ratio_breach",
                        "recommended_memory_gib": round(rec_mem_gib, 1),
                        "applied_memory_gib": round(act_mem_gib, 1),
                        "original_memory_gib": round(orig_mem_gib, 3),
                        "limit_request_ratio": base_ratio,
                        "reason": f"Current {act_mem_gib:.1f} GiB is {base_ratio}x "
                                  f"the original baseline {orig_mem_gib:.3f} GiB",
                    })
                # Baseline ratio breach: current applied >= 10x original baseline (CPU)
                if orig_cpu_cores > 0 and act_cpu_cores >= orig_cpu_cores * t.outlier_median_ratio:
                    base_ratio = round(act_cpu_cores / orig_cpu_cores, 1)
                    absurd.append({
                        "workload": wl_key, "container": c_name,
                        "woop": woop_tag,
                        "sub_type": "baseline_ratio_breach",
                        "recommended_cpu_cores": round(rec_cpu_cores, 1),
                        "applied_cpu_cores": round(act_cpu_cores, 1),
                        "original_cpu_cores": round(orig_cpu_cores, 3),
                        "limit_request_ratio": base_ratio,
                        "reason": f"Current {act_cpu_cores:.1f} CPU is {base_ratio}x "
                                  f"the original baseline {orig_cpu_cores:.3f} CPU",
                    })

                # Skip mismatch for workloads with 0 running pods — no pods
                # means recommendation can't be applied; mismatch is expected.
                if pod_count <= 0:
                    continue

                if act_mem_gib and rec_mem_gib:
                    # Divide by actual (applied) so over-recommendations can exceed 100%
                    pct = abs(rec_mem_gib - act_mem_gib) / act_mem_gib * 100
                    abs_delta = abs(rec_mem_gib - act_mem_gib)
                    if pct > t.recommendation_mismatch_pct and abs_delta >= t.mismatch_min_memory_gib:
                        mismatches.append({"workload": wl_key, "container": c_name,
                                           "woop": woop_tag, "diff_pct": round(pct, 1),
                                           "pod_count": pod_count})
                if act_cpu_cores and rec_cpu_cores:
                    pct = abs(rec_cpu_cores - act_cpu_cores) / act_cpu_cores * 100
                    abs_delta = abs(rec_cpu_cores - act_cpu_cores)
                    if pct > t.recommendation_mismatch_pct and abs_delta >= t.mismatch_min_cpu_cores:
                        mismatches.append({"workload": wl_key, "container": c_name,
                                           "woop": woop_tag, "diff_pct": round(pct, 1),
                                           "pod_count": pod_count})

        # Populate workload_memory_usage for memory leak trend detection.
        # In MCP mode this comes from _mcp_resource_ratios(); in API mode we
        # extract memory request from each workload's first container.
        mem_usage = []
        for wl in workloads:
            wl_ns = wl.get("workloadNamespace") or wl.get("namespace") or "unknown"
            wl_name = wl.get("workloadName") or wl.get("name") or "unknown"
            for ctr in wl.get("containers", []):
                res = ctr.get("resources") or {}
                req = res.get("requests") or {}
                lim = res.get("limits") or {}
                req_mem_gib = req.get("memoryGib", 0) or 0
                lim_mem_gib = lim.get("memoryGib", 0) or 0
                if req_mem_gib > 0:
                    mem_usage.append({
                        "namespace": wl_ns,
                        "workload": wl_name,
                        "container": ctr.get("containerName", ctr.get("name", "")),
                        "request_mem_mib": round(req_mem_gib * 1024, 1),
                        "limit_mem_mib": round(lim_mem_gib * 1024, 1),
                    })
                break  # first container only
        snapshot.workload_memory_usage = mem_usage

        # Sum podCount across all workloads as approximate total_pods
        total_pods = sum(wl.get("podCount", 0) or 0 for wl in workloads)
        if total_pods > 0:
            snapshot.total_pods = total_pods

        # Strip workloads to minimal footprint for downstream use.
        # Evaluator only needs namespace + name + first container's resources
        # for CONFIG/OTHER finding validation (woop_resource_map).
        # Full workload blob is 9+ MB per cluster — stripped version < 200 KB.
        stripped = []
        for wl in workloads:
            first_ctr = {}
            for ctr in wl.get("containers", []):
                res = ctr.get("resources") or {}
                req = res.get("requests") or {}
                lim = res.get("limits") or {}
                first_ctr = {
                    "name": ctr.get("containerName", ctr.get("name", "")),
                    "resources": {
                        "requests": req,
                        "limits": lim,
                    },
                }
                break  # first container only
            stripped.append({
                "namespace": wl.get("workloadNamespace", wl.get("namespace", "")),
                "workloadName": wl.get("workloadName", wl.get("name", "")),
                "containers": [first_ctr] if first_ctr else [],
            })
        snapshot.woop_workloads = stripped
        snapshot.recommendation_mismatches = mismatches
        snapshot.absurd_recommendations = absurd
        snapshot.data_gaps = data_gaps
        snapshot.data_gaps_total = len(data_gaps)
        snapshot.recommendation_mismatches_total = len(mismatches)
        snapshot.absurd_recommendations_total = len(absurd)

        snapshot.log_signals.append({
            "signal": "woop_management_summary",
            "managed": woop_summary["managed"],
            "read_only": woop_summary["read_only"],
            "undefined": woop_summary["undefined"],
            "total": len(workloads),
        })

    async def _api_oom_events(
        self, client: httpx.AsyncClient, snapshot: SnapshotData
    ) -> None:
        """Fetch OOM_KILL and STARTUP_FAILURE events from last hour."""
        now = datetime.now(timezone.utc)
        from_time = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        to_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Fetch both event types concurrently
        # API uses "type" param (not "eventTypes") with EVENT_TYPE_ prefix,
        # and returns items under "items" key (not "events").
        oom_task = self._paginated_get(
            client,
            f"/v1/workload-autoscaling/clusters/{self.cluster_id}/workload-events",
            params={"type": "EVENT_TYPE_OOM_KILL",
                    "fromDate": from_time, "toDate": to_time},
            items_key="items",
        )
        startup_task = self._paginated_get(
            client,
            f"/v1/workload-autoscaling/clusters/{self.cluster_id}/workload-events",
            params={"type": "EVENT_TYPE_STARTUP_FAILURE",
                    "fromDate": from_time, "toDate": to_time},
            items_key="items",
        )
        oom_events, startup_events = await asyncio.gather(
            oom_task, startup_task, return_exceptions=True,
        )

        # Process OOM events
        # Field names must match what evaluator reads: restart_count, container
        seen = {}
        if not isinstance(oom_events, Exception):
            for event in oom_events:
                event_time = event.get("occurredAt") or event.get("createdAt") or ""
                for wl in event.get("workloads", []):
                    wl_ns = wl.get("namespace", "")
                    wl_name = wl.get("name", "")
                    key = f"{wl_ns}/{wl_name}"
                    if key not in seen:
                        seen[key] = {
                            "namespace": wl_ns,
                            "name": wl_name,
                            "restart_count": 0,
                            "container": wl.get("containerName", ""),
                            "last_oomkill_time": event_time,
                            "source": "workload-events-api",
                        }
                    seen[key]["restart_count"] += 1
                    # Keep the most recent event time
                    if event_time > (seen[key].get("last_oomkill_time") or ""):
                        seen[key]["last_oomkill_time"] = event_time
        snapshot.oomkilled_pods = sorted(
            seen.values(), key=lambda x: x["restart_count"], reverse=True,
        )

        # Process STARTUP_FAILURE events → log signal for evaluator
        if not isinstance(startup_events, Exception) and startup_events:
            startup_workloads = {}
            for event in startup_events:
                for wl in event.get("workloads", []):
                    key = f"{wl.get('namespace', '')}/{wl.get('name', '')}"
                    startup_workloads[key] = startup_workloads.get(key, 0) + 1
            if startup_workloads:
                snapshot.log_signals.append({
                    "signal": "startup_failures",
                    "workloads": [
                        {"workload": k, "count": v}
                        for k, v in sorted(startup_workloads.items(),
                                           key=lambda x: x[1], reverse=True)
                    ],
                })

    async def _api_erroring_workloads(
        self, client: httpx.AsyncClient, snapshot: SnapshotData
    ) -> None:
        """Fetch workloads with errors — tiny response (~4 results).

        Detects webhook failures, cluster-controller issues, and other
        WOOP-reported errors without needing Loki logs.
        """
        resp = await self._api_get(
            client,
            f"/v1/workload-autoscaling/clusters/{self.cluster_id}/workloads",
            params={"workloadHasError": "true", "page.limit": "50"},
        )
        if not resp:
            return

        erroring = resp.get("workloads", [])
        if erroring:
            errors = []
            for wl in erroring:
                wl_ns = wl.get("workloadNamespace") or wl.get("namespace") or ""
                wl_name = wl.get("workloadName") or wl.get("name") or ""
                wl_key = f"{wl_ns}/{wl_name}"
                error_msg = wl.get("error", wl.get("errorMessage", "unknown error"))
                errors.append({"workload": wl_key, "error": str(error_msg)[:200]})
            snapshot.log_signals.append({
                "signal": "woop_workload_errors",
                "count": len(errors),
                "workloads": errors[:20],  # cap at 20
            })

    # ── HTTP helpers ──────────────────────────────────────────────────

    async def _api_get(
        self, client: httpx.AsyncClient, path: str,
        params: dict | None = None,
    ) -> dict | list | None:
        url = f"{self.api_url}{path}"
        # Only append organizationId when using JWT auth (cross-org).
        # Customer API keys are already org-scoped; adding the param
        # triggers a different auth path that rejects the key with 401.
        if self.org_id and not self.config.castai.uses_api_key:
            params = params or {}
            params.setdefault("organizationId", self.org_id)
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            logger.error("Timeout: %s", path)
            raise
        except httpx.HTTPStatusError as e:
            logger.error("HTTP %d: %s — %s", e.response.status_code, path, e.response.text[:200])
            raise

    async def _paginated_get(
        self, client: httpx.AsyncClient, path: str,
        params: dict | None = None, items_key: str = "items",
        max_pages: int = 20,
    ) -> list:
        params = dict(params or {})
        params["page.limit"] = 100
        all_items = []
        for _ in range(max_pages):
            resp = await self._api_get(client, path, params)
            if not resp:
                break
            all_items.extend(resp.get(items_key, []))
            cursor = resp.get("nextCursor") or resp.get("page", {}).get("nextCursor")
            if not cursor:
                break
            params["page.cursor"] = cursor
        return all_items
