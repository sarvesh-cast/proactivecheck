"""Local snapshot analysis — native Python equivalents of MCP code strings.

These functions replicate the analysis that runs inside analyze_snapshot_with_code
and woop_analyze_with_code on the MCP server, but execute locally without the
RestrictedPython sandbox.  Used by the Tier 2 (snapshot hybrid) fallback path.

Every function takes raw snapshot data (lists of K8s resource dicts) and returns
a structured result dict matching what the collector expects.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("watchdog.snapshot_health")


def _safe_int(value, default: int = 0) -> int:
    """Parse a value to int, returning default on failure."""
    try:
        return int(value or default)
    except (ValueError, TypeError):
        return default


# ── Unit parsers ─────────────────────────────────────────────────────────

def parse_cpu_millicores(value: str) -> int:
    """Parse K8s CPU string to millicores.  '500m' → 500, '2' → 2000."""
    if not value:
        return 0
    value = str(value).strip()
    if value.endswith("m"):
        try:
            return int(value[:-1])
        except ValueError:
            return 0
    try:
        return int(float(value) * 1000)
    except ValueError:
        return 0


def parse_memory_bytes(value: str) -> int:
    """Parse K8s memory string to bytes.  '128Mi' → 134217728, '1Gi' → 1073741824."""
    if not value:
        return 0
    value = str(value).strip()
    suffixes = {
        "Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
        "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4,
        "k": 1000,
    }
    for suf, mult in suffixes.items():
        if value.endswith(suf):
            try:
                return int(float(value[: -len(suf)]) * mult)
            except ValueError:
                return 0
    try:
        return int(value)
    except ValueError:
        return 0


def parse_memory_gib(value: str) -> float:
    """Parse K8s memory string to GiB float."""
    return parse_memory_bytes(value) / (1024**3)


def parse_cpu_cores(value: str) -> float:
    """Parse K8s CPU string to cores float."""
    return parse_cpu_millicores(value) / 1000.0


def fmt_mem(gib: float) -> str:
    """Format GiB value for display."""
    if gib >= 1.0:
        return f"{round(gib, 1)} GiB"
    return f"{round(gib * 1024, 1)} MiB"


# ── Pod health analysis ─────────────────────────────────────────────────

def analyze_pod_health(
    pods: list[dict],
    ref_time: datetime | None = None,
) -> dict[str, Any]:
    """Analyze pod health from snapshot podList.

    Returns same structure as _mcp_snapshot_health code string result:
      total_pods, running, pending, crashloop, oomkilled, pending_details,
      crashloop_details, agent_pods, agent_total_restarts, unhealthy pods data.
    """
    if ref_time is None:
        ref_time = datetime.now(timezone.utc)
    cutoff = (ref_time - timedelta(hours=1)).isoformat()

    running = 0
    pending = 0
    crashloop = 0
    oomkilled: list[dict] = []
    crashloop_details: list[dict] = []
    pending_details: list[dict] = []
    agent_pods: list[dict] = []
    agent_total_restarts = 0

    for pod in pods:
        md = pod.get("metadata") or {}
        st = pod.get("status") or {}
        ns = md.get("namespace", "")
        name = md.get("name", "")
        phase = st.get("phase", "Unknown")

        if phase == "Running":
            running += 1
        elif phase == "Pending":
            pending += 1
            # Only record as unschedulable if PodScheduled is False
            conds = st.get("conditions") or []
            is_unschedulable = False
            reason = ""
            pod_scheduled_seen = False
            for c in conds:
                if c.get("type") == "PodScheduled":
                    pod_scheduled_seen = True
                    if c.get("status") == "False":
                        is_unschedulable = True
                        reason = c.get("message", c.get("reason", ""))
            if not pod_scheduled_seen:
                is_unschedulable = True
            if is_unschedulable:
                pending_details.append({
                    "namespace": ns, "name": name, "reason": reason,
                })

        # Agent pod detection
        is_agent = False
        if ns == "castai-agent":
            parts = name.split("-")
            dep_prefix = "-".join(parts[:-2]) if len(parts) > 2 else name
            if dep_prefix == "castai-agent":
                is_agent = True

        # Container resource specs
        spec_containers = (pod.get("spec") or {}).get("containers") or []
        container_res: dict[str, dict] = {}
        for sc in spec_containers:
            sc_name = sc.get("name", "")
            sc_res = sc.get("resources") or {}
            sc_lim = sc_res.get("limits") or {}
            sc_req = sc_res.get("requests") or {}
            container_res[sc_name] = {
                "mem_limit": sc_lim.get("memory", ""),
                "mem_request": sc_req.get("memory", ""),
                "cpu_limit": sc_lim.get("cpu", ""),
                "cpu_request": sc_req.get("cpu", ""),
            }

        pod_restarts = 0
        exit0 = False

        for cs in st.get("containerStatuses") or []:
            rc = cs.get("restartCount", 0)
            pod_restarts += rc
            last_state = cs.get("lastState") or {}
            term = last_state.get("terminated") or {}
            cur_state = cs.get("state") or {}
            wait = cur_state.get("waiting") or {}

            if term.get("reason") == "OOMKilled":
                oom_time = term.get("finishedAt", "")
                c_name = cs.get("name", "")
                c_res = container_res.get(c_name, {})
                pod_created = md.get("creationTimestamp", "")
                # Only keep OOM from within the last hour
                if oom_time and oom_time >= cutoff:
                    oomkilled.append({
                        "namespace": ns,
                        "name": name,
                        "container": c_name,
                        "restart_count": rc,
                        "last_oomkill_time": oom_time,
                        "pod_created_at": pod_created,
                        "mem_limit": c_res.get("mem_limit", ""),
                        "mem_request": c_res.get("mem_request", ""),
                        "source": "snapshot_local",
                    })
                elif not oom_time:
                    oomkilled.append({
                        "namespace": ns,
                        "name": name,
                        "container": c_name,
                        "restart_count": rc,
                        "last_oomkill_time": "unknown",
                        "pod_created_at": pod_created,
                        "mem_limit": c_res.get("mem_limit", ""),
                        "mem_request": c_res.get("mem_request", ""),
                        "source": "snapshot_local",
                    })

            if wait.get("reason") == "CrashLoopBackOff":
                crashloop += 1
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
            agent_total_restarts += pod_restarts

    return {
        "total_pods": len(pods),
        "running": running,
        "pending": pending,
        "crashloop": crashloop,
        "crashloop_details": crashloop_details[:30],
        "oomkilled": oomkilled[:50],
        "pending_details": pending_details[:20],
        "agent_pods": agent_pods,
        "agent_total_restarts": agent_total_restarts,
    }


# ── Node health analysis ────────────────────────────────────────────────

def analyze_node_health(nodes: list[dict]) -> dict[str, Any]:
    """Analyze node health from snapshot nodeList.

    Returns: node_count, node_summary, total alloc/cap cpu/mem.
    """
    node_summary: list[dict] = []
    total_alloc_cpu = 0
    total_alloc_mem = 0
    total_cap_cpu = 0
    total_cap_mem = 0

    for node in nodes:
        n_md = node.get("metadata") or {}
        n_st = node.get("status") or {}
        alloc = n_st.get("allocatable") or {}
        cap = n_st.get("capacity") or {}

        a_cpu = parse_cpu_millicores(alloc.get("cpu", "0"))
        a_mem = parse_memory_bytes(alloc.get("memory", "0"))
        c_cpu = parse_cpu_millicores(cap.get("cpu", "0"))
        c_mem = parse_memory_bytes(cap.get("memory", "0"))

        total_alloc_cpu += a_cpu
        total_alloc_mem += a_mem
        total_cap_cpu += c_cpu
        total_cap_mem += c_mem

        ready = "Unknown"
        for cond in n_st.get("conditions") or []:
            if cond.get("type") == "Ready":
                ready = cond.get("status", "Unknown")

        node_summary.append({
            "name": n_md.get("name", ""),
            "ready": ready,
            "alloc_cpu_m": a_cpu,
            "alloc_mem_bytes": a_mem,
            "cap_cpu_m": c_cpu,
            "cap_mem_bytes": c_mem,
            "max_pods": _safe_int(alloc.get("pods", "0")),
        })

    return {
        "node_count": len(nodes),
        "node_summary": node_summary,
        "total_alloc_cpu_m": total_alloc_cpu,
        "total_alloc_mem_bytes": total_alloc_mem,
        "total_cap_cpu_m": total_cap_cpu,
        "total_cap_mem_bytes": total_cap_mem,
    }


# ── Deployment health analysis ──────────────────────────────────────────

def analyze_deployment_health(deployments: list[dict]) -> list[dict]:
    """Find unhealthy deployments (desired > 0 but readyReplicas == 0).

    Returns list of unhealthy deployment dicts.
    """
    unhealthy: list[dict] = []
    for dep in deployments:
        d_md = dep.get("metadata") or {}
        d_st = dep.get("status") or {}
        desired = (dep.get("spec") or {}).get("replicas", 0)
        ready_reps = d_st.get("readyReplicas", 0)
        avail = d_st.get("availableReplicas", 0)
        if desired > 0 and ready_reps == 0:
            unhealthy.append({
                "namespace": d_md.get("namespace", ""),
                "name": d_md.get("name", ""),
                "desired": desired,
                "ready": ready_reps,
                "available": avail,
            })
    return unhealthy[:20]


# ── Recommendation analysis (snapshot recs × WOOP managementOption) ─────

def analyze_recommendations(
    snapshot_recs: list[dict],
    woop_mgmt_map: dict[str, dict],
    thresholds: Any,
    known_s2z: list[str],
) -> dict[str, Any]:
    """Merge snapshot recommendationList with WOOP managementOption map.

    Args:
        snapshot_recs: Raw recommendationList from snapshot (has spec.recommendation
                       per container but NO managementOption).
        woop_mgmt_map: Dict of workload_key → {"mgmt": "MANAGED"|"READ_ONLY"|...,
                       "apply_type": str, "pod_count": int, "rec_status": str,
                       "kind": str} from the lightweight WOOP API call.
        thresholds: Thresholds config object.
        known_s2z: List of scale-to-zero workload name patterns.

    Returns: {"mismatches": [...], "absurd": [...], "data_gaps": [...],
              "summary": {...}, "rec_lookup": {...}}
    """
    mismatches: list[dict] = []
    absurd: list[dict] = []
    data_gaps: list[dict] = []
    data_gap_seen: set[str] = set()
    rec_lookup: dict[str, dict] = {}
    summary = {"managed": 0, "read_only": 0, "undefined": 0, "total": 0}

    # Build recommendation lookup from snapshot: workload_key → list of container recs
    snap_rec_map: dict[str, list[dict]] = {}
    for rec in snapshot_recs:
        md = rec.get("metadata") or {}
        ns = md.get("namespace", "")
        name = md.get("name", "")
        wl_key = f"{ns}/{name}"
        containers = (rec.get("spec") or {}).get("recommendation") or []
        snap_rec_map[wl_key] = containers

    # Iterate WOOP workloads (has managementOption) and cross-reference snapshot recs
    summary["total"] = len(woop_mgmt_map)
    for wl_key, wl_info in woop_mgmt_map.items():
        mgmt = wl_info.get("mgmt", "UNDEFINED")
        apply_type = wl_info.get("apply_type", "")
        pod_count = wl_info.get("pod_count", 0)
        rec_status = wl_info.get("rec_status", "")
        wl_kind = wl_info.get("kind", "")
        wl_name = wl_key.split("/", 1)[1] if "/" in wl_key else wl_key

        woop_tag = mgmt
        if apply_type:
            woop_tag += f"/{apply_type}"

        if mgmt == "MANAGED":
            summary["managed"] += 1
        elif mgmt == "READ_ONLY":
            summary["read_only"] += 1
        else:
            summary["undefined"] += 1

        # Skip known scale-to-zero
        if any(pat in wl_name for pat in known_s2z):
            continue
        # Skip Jobs/CronJobs
        if wl_kind in ("Job", "CronJob"):
            continue

        # Get snapshot recommendation containers for this workload
        snap_containers = snap_rec_map.get(wl_key, [])
        has_gcs_rec = len(snap_containers) > 0

        # Fallback: if GCS recs are empty, check WOOP API container recs
        # (available when _api_woop_mgmt_map uses includeRecommendations=true)
        wl_containers = wl_info.get("containers", [])
        has_api_rec = False
        if not has_gcs_rec:
            for ctr in wl_containers:
                if ctr.get("recommendation"):
                    has_api_rec = True
                    break

        has_rec = has_gcs_rec or has_api_rec

        # Data gap: MANAGED, no recommendation, has running pods
        if mgmt == "MANAGED" and not has_rec:
            if pod_count > 0 and wl_key not in data_gap_seen:
                if rec_status not in ("STATUS_APPLIED", "STATUS_WAITING"):
                    data_gap_seen.add(wl_key)
                    data_gaps.append({
                        "workload": wl_key, "woop": woop_tag,
                        "kind": wl_kind, "pod_count": pod_count,
                        "rec_status": rec_status or "NONE",
                        "reason": "No active recommendation",
                    })
            continue

        # Only MANAGED workloads for mismatch/absurd
        if mgmt != "MANAGED":
            continue
        if not has_rec:
            continue

        # ── Build per-container rec+actual pairs ──
        # Source A: GCS snapshot recs (K8s resource strings)
        # Source B: WOOP API container recs (GiB/cores, used when GCS empty)
        if has_gcs_rec:
            # GCS path: iterate snap_containers, look up actuals from API
            container_pairs = []
            for snap_ctr in snap_containers:
                c_name = snap_ctr.get("containerName", "")
                snap_req = snap_ctr.get("requests") or {}
                rec_mem_gib = parse_memory_gib(snap_req.get("memory", "0"))
                rec_cpu_cores = parse_cpu_cores(snap_req.get("cpu", "0"))
                # Find matching container in WOOP API data for actual + original values
                act_mem_gib = 0.0
                act_cpu_cores = 0.0
                orig_mem_gib = 0.0
                orig_cpu_cores = 0.0
                for wl_ctr in wl_containers:
                    wl_ctr_name = wl_ctr.get("containerName", wl_ctr.get("name", ""))
                    if wl_ctr_name == c_name:
                        wl_res = wl_ctr.get("resources") or {}
                        wl_req = wl_res.get("requests") or {}
                        act_mem_gib = wl_req.get("memoryGib", 0) or 0
                        act_cpu_cores = wl_req.get("cpuCores", 0) or 0
                        wl_orig = wl_ctr.get("originalResources") or {}
                        wl_orig_req = wl_orig.get("requests") or {}
                        orig_mem_gib = wl_orig_req.get("memoryGib", 0) or 0
                        orig_cpu_cores = wl_orig_req.get("cpuCores", 0) or 0
                        break
                container_pairs.append((c_name, rec_mem_gib, rec_cpu_cores, act_mem_gib, act_cpu_cores, orig_mem_gib, orig_cpu_cores))
        else:
            # API fallback path: both rec and actual from WOOP API containers
            container_pairs = []
            for wl_ctr in wl_containers:
                c_name = wl_ctr.get("containerName", wl_ctr.get("name", ""))
                wl_res = wl_ctr.get("resources") or {}
                wl_req = wl_res.get("requests") or {}
                act_mem_gib = wl_req.get("memoryGib", 0) or 0
                act_cpu_cores = wl_req.get("cpuCores", 0) or 0
                wl_orig = wl_ctr.get("originalResources") or {}
                wl_orig_req = wl_orig.get("requests") or {}
                orig_mem_gib = wl_orig_req.get("memoryGib", 0) or 0
                orig_cpu_cores = wl_orig_req.get("cpuCores", 0) or 0
                rec_data = wl_ctr.get("recommendation") or {}
                rr = rec_data.get("requests") or {}
                rec_mem_gib = rr.get("memoryGib", 0) or 0
                rec_cpu_cores = rr.get("cpuCores", 0) or 0
                if rec_mem_gib or rec_cpu_cores:
                    container_pairs.append((c_name, rec_mem_gib, rec_cpu_cores, act_mem_gib, act_cpu_cores, orig_mem_gib, orig_cpu_cores))

        for c_name, rec_mem_gib, rec_cpu_cores, act_mem_gib, act_cpu_cores, orig_mem_gib, orig_cpu_cores in container_pairs:

            # Rec lookup for OOM enrichment
            if wl_key not in rec_lookup and (rec_mem_gib or rec_cpu_cores):
                rec_lookup[wl_key] = {
                    "rec_mem": rec_mem_gib,
                    "rec_cpu": rec_cpu_cores,
                    "applied_mem": act_mem_gib,
                    "applied_cpu": act_cpu_cores,
                    "apply_type": apply_type,
                }

            # ── Absurd: recommendation or applied exceeds cap ──
            if rec_mem_gib > thresholds.absurd_memory_gib:
                absurd.append({
                    "workload": wl_key, "container": c_name, "woop": woop_tag,
                    "sub_type": "cap_breach",
                    "recommended_memory_gib": round(rec_mem_gib, 1),
                    "applied_memory_gib": round(act_mem_gib, 1),
                    "rec_display": fmt_mem(rec_mem_gib),
                    "applied_display": fmt_mem(act_mem_gib),
                    "reason": f"WOOP recommends {fmt_mem(rec_mem_gib)}",
                })
            if act_mem_gib > thresholds.absurd_memory_gib:
                absurd.append({
                    "workload": wl_key, "container": c_name, "woop": woop_tag,
                    "sub_type": "cap_breach",
                    "applied_memory_gib": round(act_mem_gib, 1),
                    "recommended_memory_gib": round(rec_mem_gib, 1),
                    "rec_display": fmt_mem(rec_mem_gib),
                    "applied_display": fmt_mem(act_mem_gib),
                    "reason": f"Applied {fmt_mem(act_mem_gib)} (rec {fmt_mem(rec_mem_gib)})",
                })

            # Ratio breach: rec >= 10x actual (memory)
            if act_mem_gib > 0 and rec_mem_gib >= act_mem_gib * thresholds.outlier_median_ratio:
                rec_ratio = round(rec_mem_gib / act_mem_gib, 1)
                absurd.append({
                    "workload": wl_key, "container": c_name, "woop": woop_tag,
                    "sub_type": "ratio_breach",
                    "recommended_memory_gib": round(rec_mem_gib, 1),
                    "applied_memory_gib": round(act_mem_gib, 1),
                    "rec_display": fmt_mem(rec_mem_gib),
                    "applied_display": fmt_mem(act_mem_gib),
                    "limit_request_ratio": rec_ratio,
                    "reason": f"Rec {fmt_mem(rec_mem_gib)} is {rec_ratio}x the current request {fmt_mem(act_mem_gib)}",
                })

            # Baseline ratio breach: current applied >= 10x original baseline (memory)
            # Catches already-applied absurd recs where rec ≈ current >> original
            if orig_mem_gib > 0 and act_mem_gib >= orig_mem_gib * thresholds.outlier_median_ratio:
                base_ratio = round(act_mem_gib / orig_mem_gib, 1)
                absurd.append({
                    "workload": wl_key, "container": c_name, "woop": woop_tag,
                    "sub_type": "baseline_ratio_breach",
                    "recommended_memory_gib": round(rec_mem_gib, 1),
                    "applied_memory_gib": round(act_mem_gib, 1),
                    "original_memory_gib": round(orig_mem_gib, 3),
                    "rec_display": fmt_mem(act_mem_gib),
                    "applied_display": fmt_mem(orig_mem_gib),
                    "limit_request_ratio": base_ratio,
                    "reason": f"Current {fmt_mem(act_mem_gib)} is {base_ratio}x the original baseline {fmt_mem(orig_mem_gib)}",
                })

            # Skip mismatch for 0-pod workloads
            if pod_count <= 0:
                continue

            # ── Mismatch: rec vs applied divergence ──
            if act_mem_gib and rec_mem_gib:
                pct = abs(rec_mem_gib - act_mem_gib) / act_mem_gib * 100
                abs_delta = abs(rec_mem_gib - act_mem_gib)
                if pct > thresholds.recommendation_mismatch_pct and abs_delta >= thresholds.mismatch_min_memory_gib:
                    mismatches.append({
                        "workload": wl_key, "container": c_name, "woop": woop_tag,
                        "apply_type": apply_type,
                        "recommended_memory_gib": round(rec_mem_gib, 1),
                        "actual_memory_gib": round(act_mem_gib, 1),
                        "rec_display": fmt_mem(rec_mem_gib),
                        "applied_display": fmt_mem(act_mem_gib),
                        "diff_pct": round(pct, 1),
                        "pod_count": pod_count,
                    })

    return {
        "mismatches": mismatches,
        "absurd": absurd,
        "data_gaps": data_gaps,
        "summary": summary,
        "rec_lookup": rec_lookup,
    }


# ── Pod-to-workload owner mapping ─────────────────────────────────────

def build_pod_owner_map(pods: list[dict]) -> dict[str, str]:
    """Build a map of (namespace/pod-name) → workload-name from ownerReferences.

    Walks the ownerReference chain: Pod → ReplicaSet → Deployment.
    For DaemonSet/StatefulSet pods, the owner IS the workload directly.
    For ReplicaSet-owned pods, we strip the RS hash suffix to get the Deployment name.

    Returns: {"namespace/pod-name": "workload-name", ...}
    """
    owner_map: dict[str, str] = {}
    for pod in pods:
        md = pod.get("metadata") or {}
        ns = md.get("namespace", "")
        pod_name = md.get("name", "")
        owners = md.get("ownerReferences") or []

        wl_name = pod_name  # fallback
        for ref in owners:
            kind = ref.get("kind", "")
            ref_name = ref.get("name", "")
            if kind == "ReplicaSet":
                # RS name = <deployment>-<template-hash>; strip the hash
                # Use rsplit to remove the last segment (the template hash)
                parts = ref_name.rsplit("-", 1)
                wl_name = parts[0] if len(parts) == 2 else ref_name
            elif kind in ("DaemonSet", "StatefulSet", "Job"):
                wl_name = ref_name
            else:
                wl_name = ref_name
            break  # first owner only

        owner_map[f"{ns}/{pod_name}"] = wl_name

    return owner_map


def _derive_workload_name(pod_name: str) -> str:
    """Derive workload name from pod name using K8s naming conventions.

    Fallback for when ownerReferences are unavailable.  Handles:
      - Deployment:   <name>-<rs-hash(6-10)>-<pod-hash(5)>
      - DaemonSet:    <name>-<hash(5)>
      - StatefulSet:  <name>-<ordinal>
    """
    # Deployment: strip -<rs-template-hash>-<pod-hash(5)>
    # RS hash must contain at least one digit to avoid matching real name segments
    # like '-exporter-' which are pure alpha.
    m = re.match(r"^(.+)-(?=[a-z0-9]*\d)[a-z0-9]{6,10}-[a-z0-9]{5}$", pod_name)
    if m:
        return m.group(1)
    # StatefulSet: strip -<ordinal>
    m = re.match(r"^(.+)-(\d+)$", pod_name)
    if m:
        return m.group(1)
    # DaemonSet/Job: strip -<hash(5)>
    m = re.match(r"^(.+)-[a-z0-9]{5}$", pod_name)
    if m:
        return m.group(1)
    return pod_name


# ── Pod metrics analysis (memory leak detection) ────────────────────────

def analyze_pod_metrics(
    pod_metrics: list[dict],
    pod_owner_map: dict[str, str] | None = None,
) -> list[dict]:
    """Extract per-workload memory usage from podmetricsList.

    Args:
        pod_metrics: Raw podmetricsList items from the snapshot.
        pod_owner_map: Optional map from build_pod_owner_map() for accurate
                       pod→workload name resolution.  Falls back to regex
                       heuristic if not provided.

    Returns list of {namespace, workload, container, usage_bytes, ...} dicts
    for the evaluator's memory leak trend detection.
    """
    usage: list[dict] = []
    for pm in pod_metrics:
        md = pm.get("metadata") or {}
        ns = md.get("namespace", "")
        pod_name = md.get("name", "")

        # Resolve workload name: prefer owner map, fall back to heuristic
        map_key = f"{ns}/{pod_name}"
        if pod_owner_map and map_key in pod_owner_map:
            wl_name = pod_owner_map[map_key]
        else:
            wl_name = _derive_workload_name(pod_name)

        for ctr in pm.get("containers") or []:
            c_name = ctr.get("name", "")
            mem_usage = ctr.get("usage", {}).get("memory", "0")
            cpu_usage = ctr.get("usage", {}).get("cpu", "0")
            usage.append({
                "namespace": ns,
                "workload": wl_name,
                "container": c_name,
                "usage_bytes": parse_memory_bytes(mem_usage),
                "usage_cpu_m": parse_cpu_millicores(cpu_usage),
            })

    # Aggregate by workload (sum across pods), return top 50 by memory
    agg: dict[str, dict] = {}
    for u in usage:
        key = f"{u['namespace']}/{u['workload']}/{u['container']}"
        if key not in agg:
            agg[key] = {
                "namespace": u["namespace"],
                "workload": u["workload"],
                "container": u["container"],
                "usage_bytes": 0,
                "usage_cpu_m": 0,
                "pod_count": 0,
            }
        agg[key]["usage_bytes"] += u["usage_bytes"]
        agg[key]["usage_cpu_m"] += u["usage_cpu_m"]
        agg[key]["pod_count"] += 1

    sorted_usage = sorted(agg.values(), key=lambda x: x["usage_bytes"], reverse=True)
    return sorted_usage[:50]
