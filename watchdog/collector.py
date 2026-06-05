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

logger = logging.getLogger("watchdog.collector")


class Collector:
    """Collects cluster state via MCP server (preferred) or public API."""

    def __init__(self, config: WatchdogConfig, snapshot_time: str | None = None) -> None:
        self.config = config
        self.api_url = config.castai.api_url.rstrip("/")
        self.timeout = config.castai.request_timeout
        self.cluster_id = config.cluster.cluster_id
        self.org_id = config.castai.organization_id
        self.snapshot_time = snapshot_time  # ISO 8601 for historical backtest
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

    async def collect(self) -> SnapshotData:
        """Run all collectors concurrently. MCP-first with retry, public API fallback."""
        # Use snapshot_time for backtest mode, otherwise current UTC
        ts = self.snapshot_time if self.snapshot_time else datetime.now(timezone.utc).isoformat()
        snapshot = SnapshotData(
            timestamp=ts,
            cluster_id=self.cluster_id,
        )
        if self.snapshot_time:
            logger.info("BACKTEST MODE: collecting snapshot at %s", self.snapshot_time)

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
                    return await self._collect_via_mcp(snapshot)
                except Exception as e:
                    if attempt < 2:
                        logger.warning("MCP collection failed (attempt %d/2): %s, re-initializing", attempt, e)
                        self.mcp._session_id = None
                        await asyncio.sleep(3)
                        continue
                    logger.error("MCP collection failed after 2 attempts: %s", e)
                    snapshot.collection_errors.append(f"MCP collection failed: {e}")
                    break

        return await self._collect_via_public_api(snapshot)

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
        else:
            woop_fn = self._mcp_woop_workloads(snapshot)
            oom_fn = self._mcp_oom_summary(snapshot)

        results = await asyncio.gather(
            self._mcp_snapshot_health(snapshot),
            woop_fn,
            oom_fn,
            self._mcp_resource_ratios(snapshot),
            self._mcp_cluster_details(snapshot),
            self._mcp_loki_signals(snapshot),
            self._mcp_org_agent_health(snapshot),
            return_exceptions=True,
        )

        names = [
            "snapshot_health", "woop_workloads", "oom_summary",
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

        return snapshot

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
        # Capture reason for pending
        conds = st.get("conditions", [])
        reason = ""
        for c in conds:
            if c.get("type") == "PodScheduled" and c.get("status") == "False":
                reason = c.get("message", c.get("reason", ""))
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
            oomkilled.append({
                "namespace": ns,
                "name": name,
                "container": cs.get("name", ""),
                "restart_count": rc,
                "last_oomkill_time": oom_time,
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
ABSURD_MEM_GIB = {t.absurd_memory_gib}
ABSURD_CPU_CORES = {t.absurd_cpu_cores}
MISMATCH_PCT = {t.recommendation_mismatch_pct}
MISMATCH_MIN_MEM_GIB = {t.mismatch_min_memory_gib}
MISMATCH_MIN_CPU_CORES = {t.mismatch_min_cpu_cores}
OUTLIER_RATIO = {t.outlier_median_ratio}
KNOWN_S2Z = {known_s2z_json}

workloads = get_workloads()

mismatches = []
absurd = []
data_gaps = []
summary = {{"managed": 0, "read_only": 0, "undefined": 0, "total": len(workloads)}}

for wl in workloads:
    wl_name = wl.get("name", "unknown")
    wl_ns = wl.get("namespace", "unknown")
    wl_key = wl_ns + "/" + wl_name

    wl_config = wl.get("workloadConfigV2") or {{}}
    vpa_cfg = wl_config.get("vpaConfig") or {{}}
    mem_limit_cfg = vpa_cfg.get("memory") or {{}}
    mem_limit_sub = mem_limit_cfg.get("limit") or {{}}
    mem_limit_mult = mem_limit_sub.get("multiplier", 0) or 0

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

    containers = wl.get("containers") or []

    has_rec = False
    for ctr in containers:
        if ctr.get("recommendation"):
            has_rec = True
            break

    pod_count = wl.get("podCount", 0) or 0

    if mgmt == "MANAGED" and not has_rec:
        if pod_count > 0:
            data_gaps.append({{"workload": wl_key, "woop": woop_tag,
                              "reason": "MANAGED but no active recommendation"}})
        continue

    # Only flag MANAGED workloads for mismatch/absurd — READ_ONLY are informational
    if mgmt != "MANAGED":
        continue

    if not has_rec or not containers:
        continue

    for ctr in containers:
        c_name = ctr.get("containerName", ctr.get("name", ""))
        res = ctr.get("resources") or {{}}
        ar = res.get("requests") or {{}}
        al = res.get("limits") or {{}}
        act_mem_gib = ar.get("memoryGib", 0) or 0
        act_cpu_cores = ar.get("cpuCores", 0) or 0
        act_lim_mem_gib = al.get("memoryGib", 0) or 0

        rec_data = ctr.get("recommendation") or {{}}
        rr = rec_data.get("requests") or {{}}
        rec_mem_gib = rr.get("memoryGib", 0) or 0
        rec_cpu_cores = rr.get("cpuCores", 0) or 0

        # Helper to format GiB/MiB
        def fmt_mem(gib):
            if gib >= 1.0:
                return str(round(gib, 1)) + " GiB"
            return str(round(gib * 1024, 1)) + " MiB"

        # Absurd — recommendation exceeds cap
        if rec_mem_gib > ABSURD_MEM_GIB:
            absurd.append({{"workload": wl_key, "container": c_name, "woop": woop_tag,
                           "sub_type": "cap_breach",
                           "recommended_memory_gib": round(rec_mem_gib, 1),
                           "applied_memory_gib": round(act_mem_gib, 1),
                           "rec_display": fmt_mem(rec_mem_gib),
                           "applied_display": fmt_mem(act_mem_gib),
                           "reason": "WOOP recommends " + fmt_mem(rec_mem_gib)}})
        if rec_cpu_cores > ABSURD_CPU_CORES:
            absurd.append({{"workload": wl_key, "container": c_name, "woop": woop_tag,
                           "sub_type": "cap_breach",
                           "recommended_cpu_cores": round(rec_cpu_cores, 1),
                           "applied_cpu_cores": round(act_cpu_cores, 1),
                           "rec_display": str(round(rec_cpu_cores, 1)) + " CPU",
                           "applied_display": str(round(act_cpu_cores, 1)) + " CPU",
                           "reason": "WOOP recommends " + str(round(rec_cpu_cores)) + " CPU cores"}})

        # Absurd — applied exceeds cap
        if act_mem_gib > ABSURD_MEM_GIB:
            absurd.append({{"workload": wl_key, "container": c_name, "woop": woop_tag,
                           "sub_type": "cap_breach",
                           "applied_memory_gib": round(act_mem_gib, 1),
                           "recommended_memory_gib": round(rec_mem_gib, 1),
                           "rec_display": fmt_mem(rec_mem_gib),
                           "applied_display": fmt_mem(act_mem_gib),
                           "reason": "Applied " + fmt_mem(act_mem_gib) + " (rec " + fmt_mem(rec_mem_gib) + ")"}})
        if act_cpu_cores > ABSURD_CPU_CORES:
            absurd.append({{"workload": wl_key, "container": c_name, "woop": woop_tag,
                           "sub_type": "cap_breach",
                           "applied_cpu_cores": round(act_cpu_cores, 1),
                           "recommended_cpu_cores": round(rec_cpu_cores, 1),
                           "rec_display": str(round(rec_cpu_cores, 1)) + " CPU",
                           "applied_display": str(round(act_cpu_cores, 1)) + " CPU",
                           "reason": "Applied " + str(round(act_cpu_cores)) + " cores (rec " + str(round(rec_cpu_cores, 1)) + ")"}})

        # Ratio breach — recommendation >= 10x current request
        if act_mem_gib > 0 and rec_mem_gib >= act_mem_gib * OUTLIER_RATIO:
            rec_ratio = round(rec_mem_gib / act_mem_gib, 1)
            absurd.append({{"workload": wl_key, "container": c_name, "woop": woop_tag,
                           "sub_type": "ratio_breach",
                           "recommended_memory_gib": round(rec_mem_gib, 1),
                           "applied_memory_gib": round(act_mem_gib, 1),
                           "rec_display": fmt_mem(rec_mem_gib),
                           "applied_display": fmt_mem(act_mem_gib),
                           "limit_request_ratio": rec_ratio,
                           "reason": "Rec " + fmt_mem(rec_mem_gib) + " is " + str(rec_ratio) + "x the current request " + fmt_mem(act_mem_gib)}})

        # Mismatch — rec vs applied divergence (requires both % AND absolute delta)
        if act_mem_gib and rec_mem_gib:
            pct = abs(rec_mem_gib - act_mem_gib) / rec_mem_gib * 100
            abs_delta = abs(rec_mem_gib - act_mem_gib)
            if pct > MISMATCH_PCT and abs_delta >= MISMATCH_MIN_MEM_GIB:
                mismatches.append({{"workload": wl_key, "container": c_name, "woop": woop_tag,
                                   "recommended_memory_gib": round(rec_mem_gib, 1),
                                   "actual_memory_gib": round(act_mem_gib, 1),
                                   "rec_display": fmt_mem(rec_mem_gib),
                                   "applied_display": fmt_mem(act_mem_gib),
                                   "diff_pct": round(pct, 1)}})
        if act_cpu_cores and rec_cpu_cores:
            pct = abs(rec_cpu_cores - act_cpu_cores) / rec_cpu_cores * 100
            abs_delta = abs(rec_cpu_cores - act_cpu_cores)
            if pct > MISMATCH_PCT and abs_delta >= MISMATCH_MIN_CPU_CORES:
                mismatches.append({{"workload": wl_key, "container": c_name, "woop": woop_tag,
                                   "recommended_cpu_cores": round(rec_cpu_cores, 1),
                                   "actual_cpu_cores": round(act_cpu_cores, 1),
                                   "rec_display": str(round(rec_cpu_cores, 2)) + " CPU",
                                   "applied_display": str(round(act_cpu_cores, 2)) + " CPU",
                                   "diff_pct": round(pct, 1)}})

result = {{
    "mismatches": mismatches,
    "absurd": absurd,
    "data_gaps": data_gaps,
    "summary": summary,
}}
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
            woop_oom.append({
                "namespace": oom.get("namespace", ""),
                "name": oom.get("workload_name", ""),
                "oom_count": oom.get("oom_count", 0),
                "first_oom": oom.get("first_oom", ""),
                "last_oom": oom.get("last_oom", ""),
                "source": "woop_events",
            })

        # Keep only snapshot OOM entries that are recent AND not duplicated
        # by WOOP data. WOOP is authoritative for the time window.
        woop_keys = {
            f"{o['namespace']}/{o['name']}" for o in woop_oom
        }
        kept_snapshot = [
            o for o in snapshot.oomkilled_pods
            if o.get("source") == "snapshot_lastState"
            and f"{o.get('namespace')}/{o.get('name')}" not in woop_keys
        ]
        snapshot.oomkilled_pods = woop_oom + kept_snapshot

        logger.info(
            "OOM summary: %d from WOOP, %d recent from snapshot, %d total",
            len(woop_oom), len(kept_snapshot), len(snapshot.oomkilled_pods),
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

    containers = wl.get("containers") or []

    has_rec = False
    for ctr in containers:
        if ctr.get("recommendation"):
            has_rec = True
            break

    pod_count = wl.get("podCount", 0) or 0

    if mgmt == "MANAGED" and not has_rec and not containers:
        if pod_count > 0:
            data_gaps.append({{"workload": wl_key, "woop": woop_tag,
                              "reason": "MANAGED but no recommendation or containers"}})
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
                            pct = abs(rec_mem_gib - req_mem_gib) / rec_mem_gib * 100
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
                            pct = abs(rec_cpu_cores - req_cpu_cores) / rec_cpu_cores * 100
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
            woop_oom.append({
                "namespace": oom.get("namespace", ""),
                "name": oom.get("workload_name", ""),
                "oom_count": oom.get("oom_count", 0),
                "first_oom": oom.get("first_oom", ""),
                "last_oom": oom.get("last_oom", ""),
                "source": "woop_events_backtest",
            })

        woop_keys = {f"{o['namespace']}/{o['name']}" for o in woop_oom}
        kept_snapshot = [
            o for o in snapshot.oomkilled_pods
            if o.get("source") == "snapshot_lastState"
            and f"{o.get('namespace')}/{o.get('name')}" not in woop_keys
        ]
        snapshot.oomkilled_pods = woop_oom + kept_snapshot

        logger.info(
            "BACKTEST OOM: %d from WOOP events, %d from snapshot, %d total "
            "(window: %s to %s)",
            len(woop_oom), len(kept_snapshot), len(snapshot.oomkilled_pods),
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
        """extract_fields_from_all: fallback if analyze_snapshot_with_code fails.

        Extracts basic pod statuses without running arbitrary code.
        """
        data = await self.mcp.call_tool("extract_fields_from_all", {
            "cluster_id_or_name": self.cluster_id,
            "resource_kind": "pods",
            "fields": ["metadata.namespace", "metadata.name",
                        "status.phase", "status.containerStatuses"],
        })
        if not data:
            raise RuntimeError("extract_fields_from_all returned empty")
        if isinstance(data, str):
            data = json.loads(data)

        items = data if isinstance(data, list) else data.get("items", [])
        running = pending = crashloop = 0
        oomkilled = []
        crashloop_details = []
        pending_details = []

        for item in items:
            ns = item.get("metadata.namespace", item.get("namespace", ""))
            name = item.get("metadata.name", item.get("name", ""))
            phase = item.get("status.phase", item.get("phase", "Unknown"))
            if phase == "Running":
                running += 1
            elif phase == "Pending":
                pending += 1
                pending_details.append({"namespace": ns, "name": name, "reason": ""})

            for cs in (item.get("status.containerStatuses") or
                        item.get("containerStatuses") or []):
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
            "Fallback: %d pods (%d OOM, %d Pending, %d CrashLoop)",
            len(items), len(oomkilled), pending, crashloop,
        )

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC API FALLBACK — limited data, no snapshot/Loki access
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
                return_exceptions=True,
            )
            names = ["cluster_details", "woop_workloads", "oom_events"]
            for name, result in zip(names, results):
                if isinstance(result, Exception):
                    err = f"{name}: {type(result).__name__}: {result}"
                    logger.error("API collector failed: %s", err)
                    snapshot.collection_errors.append(err)

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
            snapshot.agent_pods = [{
                "name": "castai-agent", "namespace": "castai-agent",
                "phase": "Running" if agent_status in ("online", "connected") else agent_status,
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
        woop_summary = {"managed": 0, "read_only": 0, "undefined": 0}
        t = self.config.thresholds
        known_s2z = self.config.cluster.known_scale_to_zero_workloads

        for wl in workloads:
            wl_name = wl.get("workloadName", "unknown")
            wl_ns = wl.get("workloadNamespace", "unknown")
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

            containers = wl.get("containers", [])

            # Check if any container has a recommendation
            has_rec = False
            for ctr in containers:
                if ctr.get("recommendation"):
                    has_rec = True
                    break

            pod_count = wl.get("podCount", 0) or 0

            if mgmt == "MANAGED" and not has_rec:
                if pod_count > 0:
                    data_gaps.append({
                        "workload": wl_key, "woop": woop_tag,
                        "reason": "MANAGED but no active recommendation",
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

                rec_data = ctr.get("recommendation") or {}
                rr = rec_data.get("requests") or {}
                rec_mem_gib = rr.get("memoryGib", 0) or 0
                rec_cpu_cores = rr.get("cpuCores", 0) or 0

                if rec_mem_gib > t.absurd_memory_gib:
                    absurd.append({"workload": wl_key, "container": c_name,
                                   "woop": woop_tag,
                                   "recommended_memory_gib": round(rec_mem_gib, 1),
                                   "reason": f"WOOP recommends {rec_mem_gib:.0f} GiB"})
                if rec_cpu_cores > t.absurd_cpu_cores:
                    absurd.append({"workload": wl_key, "container": c_name,
                                   "woop": woop_tag,
                                   "recommended_cpu_cores": round(rec_cpu_cores, 1),
                                   "reason": f"WOOP recommends {rec_cpu_cores:.0f} cores"})

                # Ratio breach: recommendation >= 10x current request
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

                if act_mem_gib and rec_mem_gib:
                    pct = abs(rec_mem_gib - act_mem_gib) / rec_mem_gib * 100
                    abs_delta = abs(rec_mem_gib - act_mem_gib)
                    if pct > t.recommendation_mismatch_pct and abs_delta >= t.mismatch_min_memory_gib:
                        mismatches.append({"workload": wl_key, "container": c_name,
                                           "woop": woop_tag, "diff_pct": round(pct, 1)})
                if act_cpu_cores and rec_cpu_cores:
                    pct = abs(rec_cpu_cores - act_cpu_cores) / rec_cpu_cores * 100
                    abs_delta = abs(rec_cpu_cores - act_cpu_cores)
                    if pct > t.recommendation_mismatch_pct and abs_delta >= t.mismatch_min_cpu_cores:
                        mismatches.append({"workload": wl_key, "container": c_name,
                                           "woop": woop_tag, "diff_pct": round(pct, 1)})

        snapshot.woop_workloads = workloads
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
        now = datetime.now(timezone.utc)
        events = await self._paginated_get(
            client,
            f"/v1/workload-autoscaling/clusters/{self.cluster_id}/workload-events",
            params={
                "eventTypes": "OOM_KILL",
                "fromDate": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "toDate": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            items_key="events",
        )
        seen = {}
        for event in events:
            for wl in event.get("workloads", []):
                key = f"{wl.get('workloadNamespace')}/{wl.get('workloadName')}"
                if key not in seen:
                    seen[key] = {"namespace": wl.get("workloadNamespace", ""),
                                 "name": wl.get("workloadName", ""), "oom_count": 0}
                seen[key]["oom_count"] += 1

        snapshot.oomkilled_pods = sorted(
            seen.values(), key=lambda x: x["oom_count"], reverse=True,
        )

    # ── HTTP helpers ──────────────────────────────────────────────────

    async def _api_get(
        self, client: httpx.AsyncClient, path: str,
        params: dict | None = None,
    ) -> dict | list | None:
        url = f"{self.api_url}{path}"
        if self.org_id:
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
