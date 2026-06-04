"""Collector module — gathers data via CAST AI MCP server.

MCP-first design: when CASTAI_MCP_URL is set, ALL data flows through the
MCP server which provides access to:

  Snapshot tools (GCS-backed):
    - analyze_snapshot_with_code    → pod health, OOMKills, agent status
    - get_cluster_snapshot_summary  → node count, cluster metadata

  WOOP tools:
    - woop_get_workloads            → recommendations vs applied values
    - woop_get_oom_summary          → OOM kills ranked by workload
    - woop_get_workload_resource_ratios → limit/request ratio checks
    - woop_get_workload_events      → OOM_KILL, SURGE, STARTUP_FAILURE events

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

    def __init__(self, config: WatchdogConfig) -> None:
        self.config = config
        self.api_url = config.castai.api_url.rstrip("/")
        self.timeout = config.castai.request_timeout
        self.cluster_id = config.cluster.cluster_id
        self.org_id = config.castai.organization_id
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
        snapshot = SnapshotData(
            timestamp=datetime.now(timezone.utc).isoformat(),
            cluster_id=self.cluster_id,
        )

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
        """
        results = await asyncio.gather(
            self._mcp_snapshot_health(snapshot),
            self._mcp_woop_workloads(snapshot),
            self._mcp_oom_summary(snapshot),
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
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
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
        """woop_get_workloads: recommendation vs applied, mismatches, absurd recs."""
        data = await self.mcp.call_tool("woop_get_workloads", {
            "cluster_id_or_name": self.cluster_id,
            "auto_paginate": True,
            "include_recommendations": True,
            "include_containers": True,
        })
        if not data:
            raise RuntimeError("woop_get_workloads returned empty")
        if isinstance(data, str):
            data = json.loads(data)

        workloads = data.get("workloads", data) if isinstance(data, dict) else data
        if not isinstance(workloads, list):
            workloads = []

        mismatches = []
        absurd = []
        data_gaps = []
        woop_summary = {"managed": 0, "read_only": 0, "undefined": 0}
        t = self.config.thresholds
        known_s2z = self.config.cluster.known_scale_to_zero_workloads

        for wl in workloads:
            wl_name = wl.get("workloadName", wl.get("name", "unknown"))
            wl_ns = wl.get("workloadNamespace", wl.get("namespace", "unknown"))
            wl_key = f"{wl_ns}/{wl_name}"

            # Extract WOOP config from the correct nested path
            wl_config = wl.get("workloadConfigV2", {})
            vpa_cfg = wl_config.get("vpaConfig", {})
            hpa_cfg = wl_config.get("hpaConfig", {})
            rollout = wl_config.get("rolloutBehavior", {})

            mgmt = vpa_cfg.get("managementOption", "UNDEFINED")
            apply_type = vpa_cfg.get("applyType", "")
            downscale_apply = vpa_cfg.get("downscaling", {}).get("applyType", "")
            mem_event_apply = vpa_cfg.get("memoryEvent", {}).get("applyType", "")
            hpa_mgmt = hpa_cfg.get("managementOption", "")
            mem_limit_type = vpa_cfg.get("memory", {}).get("limit", {}).get("type", "")
            mem_limit_mult = vpa_cfg.get("memory", {}).get("limit", {}).get("multiplier", 0)
            rollout_type = rollout.get("type", "")

            # WOOP config tag for findings
            woop_tag = f"{mgmt}"
            if apply_type:
                woop_tag += f"/{apply_type}"

            # Track management stats
            if mgmt == "MANAGED":
                woop_summary["managed"] += 1
            elif mgmt == "READ_ONLY":
                woop_summary["read_only"] += 1
            else:
                woop_summary["undefined"] += 1

            if any(pat in wl_name for pat in known_s2z):
                continue

            rec = wl.get("recommendation", {})
            containers = wl.get("containers", [])

            if mgmt == "MANAGED" and not rec:
                data_gaps.append({
                    "workload": wl_key,
                    "woop": woop_tag,
                    "reason": "MANAGED but no active recommendation",
                })
                continue

            if not rec or not containers:
                continue

            rec_map = {
                c.get("containerName"): c
                for c in rec.get("containers", [])
            }

            for ctr in containers:
                c_name = ctr.get("containerName", "")
                rc = rec_map.get(c_name, {})
                if not rc:
                    continue

                rr = rc.get("requests", {})
                ar = ctr.get("requests", {})
                rec_mem, act_mem = rr.get("memory", 0), ar.get("memory", 0)
                rec_cpu, act_cpu = rr.get("cpu", 0), ar.get("cpu", 0)

                # Absurd checks — on recommendation
                rec_mem_gib = rec_mem / (1024**3) if rec_mem else 0
                rec_cpu_cores = rec_cpu / 1000 if rec_cpu else 0

                if rec_mem_gib > t.absurd_memory_gib:
                    absurd.append({
                        "workload": wl_key, "container": c_name,
                        "woop": woop_tag,
                        "recommended_memory_gib": round(rec_mem_gib, 1),
                        "reason": f"WOOP recommends {rec_mem_gib:.0f} GiB memory",
                    })
                if rec_cpu_cores > t.absurd_cpu_cores:
                    absurd.append({
                        "workload": wl_key, "container": c_name,
                        "woop": woop_tag,
                        "recommended_cpu_cores": round(rec_cpu_cores, 1),
                        "reason": f"WOOP recommends {rec_cpu_cores:.0f} CPU cores",
                    })

                # Absurd checks — on applied (actual) values
                act_mem_gib = act_mem / (1024**3) if act_mem else 0
                act_cpu_cores = act_cpu / 1000 if act_cpu else 0

                if act_mem_gib > t.absurd_memory_gib:
                    absurd.append({
                        "workload": wl_key, "container": c_name,
                        "woop": woop_tag,
                        "applied_memory_gib": round(act_mem_gib, 1),
                        "recommended_memory_gib": round(rec_mem_gib, 1),
                        "reason": f"Applied {act_mem_gib:.0f} GiB memory "
                                  f"(WOOP recommends only {rec_mem_gib:.1f} GiB)",
                    })
                if act_cpu_cores > t.absurd_cpu_cores:
                    absurd.append({
                        "workload": wl_key, "container": c_name,
                        "woop": woop_tag,
                        "applied_cpu_cores": round(act_cpu_cores, 1),
                        "recommended_cpu_cores": round(rec_cpu_cores, 1),
                        "reason": f"Applied {act_cpu_cores:.0f} CPU cores "
                                  f"(WOOP recommends only {rec_cpu_cores:.1f})",
                    })

                # Broken multiplier — limit/request ratio far from expected
                al = ctr.get("limits", {})
                act_limit_mem = al.get("memory", 0)
                if act_mem and act_limit_mem:
                    applied_ratio = act_limit_mem / act_mem
                    expected_mult = mem_limit_mult if mem_limit_mult > 0 else 1.5
                    if applied_ratio > t.outlier_median_ratio:
                        absurd.append({
                            "workload": wl_key, "container": c_name,
                            "woop": woop_tag,
                            "applied_memory_gib": round(act_mem_gib, 1),
                            "applied_limit_memory_gib": round(
                                act_limit_mem / (1024**3), 1),
                            "limit_request_ratio": round(applied_ratio, 1),
                            "expected_multiplier": expected_mult,
                            "reason": f"Limit/request ratio {applied_ratio:.0f}x "
                                      f"(expected ~{expected_mult}x) — broken multiplier",
                        })

                # Mismatch checks
                if act_mem and rec_mem:
                    pct = abs(rec_mem - act_mem) / rec_mem * 100
                    if pct > t.recommendation_mismatch_pct:
                        mismatches.append({
                            "workload": wl_key, "container": c_name,
                            "woop": woop_tag,
                            "recommended_memory": rec_mem,
                            "actual_memory": act_mem, "diff_pct": round(pct, 1),
                        })
                if act_cpu and rec_cpu:
                    pct = abs(rec_cpu - act_cpu) / rec_cpu * 100
                    if pct > t.recommendation_mismatch_pct:
                        mismatches.append({
                            "workload": wl_key, "container": c_name,
                            "woop": woop_tag,
                            "recommended_cpu": rec_cpu,
                            "actual_cpu": act_cpu, "diff_pct": round(pct, 1),
                        })

        snapshot.woop_workloads = workloads
        snapshot.recommendation_mismatches = mismatches
        snapshot.absurd_recommendations = absurd
        snapshot.data_gaps = data_gaps

        # Store WOOP management summary as a log signal for evaluator
        snapshot.log_signals.append({
            "signal": "woop_management_summary",
            "managed": woop_summary["managed"],
            "read_only": woop_summary["read_only"],
            "undefined": woop_summary["undefined"],
            "total": len(workloads),
        })

        logger.info(
            "WOOP: %d workloads (managed=%d, read_only=%d, undefined=%d), "
            "%d mismatches, %d absurd, %d data gaps",
            len(workloads), woop_summary["managed"], woop_summary["read_only"],
            woop_summary["undefined"], len(mismatches), len(absurd), len(data_gaps),
        )

    async def _mcp_oom_summary(self, snapshot: SnapshotData) -> None:
        """woop_get_oom_summary: OOM kills ranked by workload over last hour."""
        now = datetime.now(timezone.utc)
        data = await self.mcp.call_tool("woop_get_oom_summary", {
            "cluster_id_or_name": self.cluster_id,
            "from_date": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to_date": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 20,
        })
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

    async def _mcp_resource_ratios(self, snapshot: SnapshotData) -> None:
        """woop_get_workload_resource_ratios: flag tight limits + collect memory usage."""
        data = await self.mcp.call_tool("woop_get_workload_resource_ratios", {
            "cluster_id_or_name": self.cluster_id,
        })
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
        """loki_query: evictor blocks, WA mutations, safety mechanism events."""
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

            rec = wl.get("recommendation", {})
            containers = wl.get("containers", [])

            if mgmt == "MANAGED" and not rec:
                data_gaps.append({
                    "workload": wl_key, "woop": woop_tag,
                    "reason": "MANAGED but no active recommendation",
                })
                continue
            if not rec or not containers:
                continue

            rec_map = {c.get("containerName"): c for c in rec.get("containers", [])}
            for ctr in containers:
                c_name = ctr.get("containerName", "")
                rc = rec_map.get(c_name, {})
                if not rc:
                    continue
                rr, ar = rc.get("requests", {}), ctr.get("requests", {})
                rec_mem, act_mem = rr.get("memory", 0), ar.get("memory", 0)
                rec_cpu, act_cpu = rr.get("cpu", 0), ar.get("cpu", 0)

                rec_mem_gib = rec_mem / (1024**3) if rec_mem else 0
                rec_cpu_cores = rec_cpu / 1000 if rec_cpu else 0
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

                # Broken multiplier check
                al = ctr.get("limits", {})
                act_limit_mem = al.get("memory", 0)
                if act_mem and act_limit_mem:
                    applied_ratio = act_limit_mem / act_mem
                    expected_mult = mem_limit_mult if mem_limit_mult > 0 else 1.5
                    if applied_ratio > t.outlier_median_ratio:
                        absurd.append({
                            "workload": wl_key, "container": c_name,
                            "woop": woop_tag,
                            "limit_request_ratio": round(applied_ratio, 1),
                            "expected_multiplier": expected_mult,
                            "reason": f"Limit/request ratio {applied_ratio:.0f}x "
                                      f"(expected ~{expected_mult}x)",
                        })

                if act_mem and rec_mem:
                    pct = abs(rec_mem - act_mem) / rec_mem * 100
                    if pct > t.recommendation_mismatch_pct:
                        mismatches.append({"workload": wl_key, "container": c_name,
                                           "woop": woop_tag, "diff_pct": round(pct, 1)})
                if act_cpu and rec_cpu:
                    pct = abs(rec_cpu - act_cpu) / rec_cpu * 100
                    if pct > t.recommendation_mismatch_pct:
                        mismatches.append({"workload": wl_key, "container": c_name,
                                           "woop": woop_tag, "diff_pct": round(pct, 1)})

        snapshot.woop_workloads = workloads
        snapshot.recommendation_mismatches = mismatches
        snapshot.absurd_recommendations = absurd
        snapshot.data_gaps = data_gaps

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
