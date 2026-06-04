"""Evaluator module — feeds snapshot data to Claude for anomaly detection.

Sends the collected snapshot + trailing history to Claude Haiku (or Sonnet
for higher fidelity) as a single-shot evaluation. The model acts as an SRE
reviewing the cluster, classifies findings by severity, and outputs
structured JSON.

Handles: LLM API failures (retry once, fallback to raw metrics), malformed
JSON responses (regex extraction fallback), and token budget management.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from .config import WatchdogConfig
from .models import (
    EvaluationMetrics,
    EvaluationResult,
    Finding,
    FindingCategory,
    Severity,
    SnapshotData,
    Verdict,
)

logger = logging.getLogger("watchdog.evaluator")

# ── System prompt (from design doc, page 5-7) ────────────────────────

SYSTEM_PROMPT = """SYSTEM PROMPT FOR CLUSTER WATCHDOG EVALUATION

You are an SRE monitoring agent for CAST AI. You inspect Kubernetes
cluster snapshots every 5 minutes and identify anomalies that need
human attention. Your job is to reduce noise—only flag things that
a senior SRE would actually investigate.

## Cluster Context (Grip Security — prod-us-4)

Known baseline behaviors (DO NOT flag these as anomalies):
- aggregator workloads in every namespace routinely scale to zero
  replicas. Seeing 0/0 ready pods for aggregator is NORMAL.
- discovery-puller workloads have high memory usage (800MB-1.5GB
  per pod is expected). Flag only if limits are set far below this.
- Some namespaces are low-traffic and may show 0 resource usage.
  This is not an incident.

Known configuration:
- WOOP memory limit multiplier: 1.5x (recently corrected from 100x)
- WOOP recommendation strategy: Max Usage (NOTE: this strategy is
  vulnerable to outlier pods driving extreme recommendations —
  flag any recommendation where a single pod's usage is >10x the
  median of its siblings as a potential outlier-driven spike)
- Cluster ID: {cluster_id}
- Multiple customer namespaces (cengage, williamsmullen,
  oscarhealth, ensemblehealthpartners, athenahealth, and others)
- Cilium CNI in use — nodes may briefly show cilium agent-not-ready
  taint during startup. This is normal and transient (<2 min).
- Customer uses KEDA for some workloads — rapid replica changes
  driven by KEDA are expected and should not trigger cascading
  scaling alerts unless node count also spikes disproportionately.

Known fragile components (extra scrutiny):
- castai-agent: has a history of ExitCode:0 silent failures and
  OOM issues. Any restart is worth logging; >3/hour is CRITICAL.
- castai-workload-autoscaler: should be running with 2 replicas.
  If only 1 replica is Running, flag as WARNING.
- castai-workload-autoscaler-exporter: watch for metric limit
  errors in logs (via loki_query). If exporter is unhealthy,
  recommendations will be incomplete.

## Your Task

Analyze the following snapshot and trailing history. Produce a JSON
response with this exact structure:

{{
  "verdict": "HEALTHY" | "WARNING" | "CRITICAL",
  "summary": "<one sentence overall assessment>",
  "findings": [
    {{
      "severity": "critical" | "warning" | "info",
      "category": "<oomkill|mismatch|unschedulable|agent|data_gap|memory_leak|absurd_recommendation|agent_restart|webhook_failure|cascading_scaling|config|other>",
      "workload": "<namespace/workload or cluster-level>",
      "what": "<what is happening>",
      "evidence": "<specific numbers from the snapshot>",
      "suggested_action": "<what the TAM or customer should do>"
    }}
  ],
  "metrics": {{
    "total_pods": <int>,
    "oomkilled_pods": <int>,
    "pending_pods": <int>,
    "crashloop_pods": <int>,
    "oomkill_rate_per_hour": <float>,
    "recommendation_mismatches": <int>,
    "absurd_recommendations": <int>,
    "agent_restarts_last_hour": <int>,
    "node_count": <int>,
    "node_count_delta_pct": <float>
  }}
}}

## Classification Rules

CRITICAL (post immediately):
- Any pod OOMKilled more than 3 times in the last hour
- Any WOOP recommendation where applied value differs from intended by more than 50%
- Any workload with all replicas in CrashLoopBackOff
- CAST AI agent pod not running
- Any pod Pending for more than 15 minutes
- Any WOOP recommendation exceeding 100 GiB memory or 100 CPU cores for a single workload. This is ALWAYS a bug or runaway logic — no discovery-puller pod should ever need 10 TB of RAM. Check woop_get_workloads for the raw recommendation values.
- CAST AI agent pod restarting more than 3 times in the last hour, or showing ExitCode:0 with restartCount > 5. Silent agent failures have caused multi-day outages for this customer.
- Cluster node count or total pod count increased by >50% vs. the trailing 1-hour average without a corresponding deployment event (cascading scaling / runaway autoscaler).

WARNING (post, but less urgent):
- OOMKill count increasing but below critical threshold
- Memory usage trending upward across 3+ consecutive snapshots without leveling off (potential leak)
- Workload with optimization enabled but no recommendation for >2 hours
- Node running at >90% memory utilization
- CAST AI agent pod restartCount between 1-3 in the last hour (early signal before it becomes critical)
- Workload autoscaler webhook not responding or exporter pod not Running (recommendations won't be applied even if correct)
- Any single pod memory usage exceeding 2x its request across 3+ snapshots (potential memory leak in the application itself, which will eventually trigger OOM recovery compounding)

HEALTHY (do not post):
- All pods Running with expected restart counts
- Resource usage within expected ranges
- Aggregator at zero replicas (this is normal)
- Sporadic single OOMKill with no repeat (transient)

## Snapshot Data

{snapshot_json}

## Trailing History (last {window_size} snapshots)

{history_json}

## Collection Errors

{collection_errors}

IMPORTANT: If collection errors exist, note which data sources are missing
and reduce confidence in your assessment accordingly. Do NOT report missing
data as a cluster problem — it's an observability gap."""


class Evaluator:
    """Evaluates cluster snapshots using Claude."""

    def __init__(self, config: WatchdogConfig) -> None:
        self.config = config
        self.api_key = config.anthropic.api_key
        self.model = config.anthropic.model
        self.fallback_model = config.anthropic.fallback_model

    async def evaluate(
        self,
        snapshot: SnapshotData,
        history: list[dict],
        node_count_delta_pct: float = 0.0,
        pod_count_delta_pct: float = 0.0,
        agent_restarts_last_hour: int = 0,
        memory_leaks: list[dict] | None = None,
        mature_data_gaps: list[dict] | None = None,
    ) -> EvaluationResult:
        """Send snapshot to Claude and parse the structured response."""

        # Build the prompt with a compact snapshot (strip bulk WOOP data)
        compact_snap = self._compact_snapshot(snapshot, mature_data_gaps or [])
        prompt = SYSTEM_PROMPT.format(
            cluster_id=self.config.cluster.cluster_id,
            snapshot_json=json.dumps(compact_snap, indent=2, default=str),
            history_json=json.dumps(
                self._compact_history(history), indent=2, default=str
            ),
            window_size=len(history),
            collection_errors=json.dumps(snapshot.collection_errors, indent=2)
            if snapshot.collection_errors
            else "None — all data sources collected successfully.",
        )

        # Try primary model, fall back if needed
        for model in [self.model, self.fallback_model]:
            try:
                result = await self._call_anthropic(prompt, model)
                if result:
                    return result
            except Exception as e:
                logger.warning("Model %s failed: %s", model, e)
                continue

        # Both models failed — return a raw metrics-only evaluation
        logger.error("All LLM evaluations failed, producing raw metrics fallback")
        return self._raw_metrics_fallback(
            snapshot, node_count_delta_pct, pod_count_delta_pct,
            agent_restarts_last_hour, memory_leaks or [],
            mature_data_gaps or [],
        )

    async def _call_anthropic(
        self, prompt: str, model: str
    ) -> EvaluationResult | None:
        """Call the Anthropic Messages API and parse the response."""
        logger.debug(
            "Sending prompt to %s: %d chars (~%d tokens)",
            model, len(prompt), len(prompt) // 4,
        )
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": self.config.anthropic.max_tokens,
                    "temperature": self.config.anthropic.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if resp.status_code != 200:
                logger.error(
                    "Anthropic %s returned %d: %s",
                    model, resp.status_code, resp.text[:500],
                )
            resp.raise_for_status()
            data = resp.json()

        # Extract text content
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block["text"]

        if not text:
            logger.error("Empty response from %s", model)
            return None

        # Parse JSON from the response
        parsed = self._parse_llm_json(text)
        if not parsed:
            logger.error("Failed to parse JSON from %s response", model)
            return None

        return self._build_result(parsed, model, text)

    def _parse_llm_json(self, text: str) -> dict | None:
        """Extract JSON from LLM response, handling markdown fences and preamble."""
        # Try 1: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try 2: extract from markdown code fence
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try 3: find the first { ... } block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _build_result(
        self, parsed: dict, model: str, raw: str
    ) -> EvaluationResult:
        """Convert parsed JSON into typed EvaluationResult."""
        findings = []
        for f in parsed.get("findings", []):
            try:
                findings.append(Finding(
                    severity=Severity(f.get("severity", "info")),
                    category=FindingCategory(f.get("category", "other")),
                    workload=f.get("workload", "unknown"),
                    what=f.get("what", ""),
                    evidence=f.get("evidence", ""),
                    suggested_action=f.get("suggested_action", ""),
                ))
            except (ValueError, KeyError) as e:
                logger.warning("Skipping malformed finding: %s", e)

        m = parsed.get("metrics", {})
        metrics = EvaluationMetrics(
            total_pods=m.get("total_pods", 0),
            oomkilled_pods=m.get("oomkilled_pods", 0),
            pending_pods=m.get("pending_pods", 0),
            crashloop_pods=m.get("crashloop_pods", 0),
            oomkill_rate_per_hour=m.get("oomkill_rate_per_hour", 0.0),
            recommendation_mismatches=m.get("recommendation_mismatches", 0),
            absurd_recommendations=m.get("absurd_recommendations", 0),
            agent_restarts_last_hour=m.get("agent_restarts_last_hour", 0),
            node_count=m.get("node_count", 0),
            node_count_delta_pct=m.get("node_count_delta_pct", 0.0),
        )

        try:
            verdict = Verdict(parsed.get("verdict", "HEALTHY"))
        except ValueError:
            verdict = Verdict.WARNING  # default to caution

        return EvaluationResult(
            verdict=verdict,
            summary=parsed.get("summary", "No summary provided"),
            findings=findings,
            metrics=metrics,
            model_used=model,
            raw_response=raw[:2000],  # truncate for storage
        )

    def _raw_metrics_fallback(
        self,
        snapshot: SnapshotData,
        node_count_delta_pct: float,
        pod_count_delta_pct: float = 0.0,
        agent_restarts_last_hour: int = 0,
        memory_leaks: list[dict] | None = None,
        mature_data_gaps: list[dict] | None = None,
    ) -> EvaluationResult:
        """Produce a deterministic evaluation from raw data when LLM is unavailable.

        This is the safety net — uses hardcoded thresholds from config
        to classify without LLM assistance. Covers all 10 design doc scenarios.
        """
        t = self.config.thresholds
        findings = []
        verdict = Verdict.HEALTHY

        def _escalate(sev: Severity) -> None:
            nonlocal verdict
            if sev == Severity.CRITICAL:
                verdict = Verdict.CRITICAL
            elif sev == Severity.WARNING and verdict == Verdict.HEALTHY:
                verdict = Verdict.WARNING

        # ── 1. OOMKill spiral ────────────────────────────────────────
        if len(snapshot.oomkilled_pods) >= t.oomkill_critical_per_hour:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.OOMKILL,
                workload="cluster-level",
                what=f"{len(snapshot.oomkilled_pods)} pods with OOMKill detected",
                evidence=json.dumps(snapshot.oomkilled_pods[:3]),
                suggested_action="Investigate OOMKill workloads immediately",
            ))
        elif len(snapshot.oomkilled_pods) > 0:
            _escalate(Severity.WARNING)
            findings.append(Finding(
                severity=Severity.WARNING,
                category=FindingCategory.OOMKILL,
                workload="cluster-level",
                what=f"{len(snapshot.oomkilled_pods)} OOMKill(s) — below critical threshold but rising",
                evidence=json.dumps(snapshot.oomkilled_pods[:3]),
                suggested_action="Monitor closely; check if count increases next cycle",
            ))

        # ── 2. Recommendation/actual mismatch ────────────────────────
        for mm in snapshot.recommendation_mismatches:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.MISMATCH,
                workload=mm["workload"],
                what=f"Recommendation/actual mismatch ({mm['diff_pct']}% divergence)",
                evidence=json.dumps(mm),
                suggested_action="Disable WOOP for this workload and investigate",
            ))

        # ── 3. Unschedulable workloads ───────────────────────────────
        if snapshot.pending_pods > 0:
            sev = Severity.CRITICAL if snapshot.pending_pods > 5 else Severity.WARNING
            _escalate(sev)
            findings.append(Finding(
                severity=sev,
                category=FindingCategory.UNSCHEDULABLE,
                workload="cluster-level",
                what=f"{snapshot.pending_pods} pods in Pending state",
                evidence=f"pending_pods={snapshot.pending_pods}",
                suggested_action="Check node capacity and resource requests",
            ))

        # ── 4. Agent down ────────────────────────────────────────────
        for agent in snapshot.agent_pods:
            if agent.get("phase") != "Running":
                _escalate(Severity.CRITICAL)
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.AGENT,
                    workload=f"castai-system/{agent['name']}",
                    what=f"CAST AI agent pod not Running (phase={agent.get('phase')})",
                    evidence=json.dumps(agent),
                    suggested_action="Check agent pod logs and events immediately",
                ))

        # Org-wide agent offline (from log_signals)
        for sig in snapshot.log_signals:
            if sig.get("signal") == "org_agent_offline":
                _escalate(Severity.CRITICAL)
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.AGENT,
                    workload="org-level",
                    what=f"{sig['count']} cluster(s) with agent not online",
                    evidence=json.dumps(sig.get("sample", [])[:3]),
                    suggested_action="Check agent status on offline clusters",
                ))

        # ── 5. Data gap / no recommendation (>2 hours) ──────────────
        mgaps = mature_data_gaps or []
        if mgaps:
            _escalate(Severity.WARNING)
            findings.append(Finding(
                severity=Severity.WARNING,
                category=FindingCategory.DATA_GAP,
                workload="cluster-level",
                what=f"{len(mgaps)} MANAGED workload(s) with no recommendation for >2 hours",
                evidence=json.dumps(mgaps[:5]),
                suggested_action="Check WOOP exporter health; workloads may not be getting optimized",
            ))
        elif snapshot.data_gaps:
            # Fresh gaps (< 2 hours) — info-level only, don't escalate
            findings.append(Finding(
                severity=Severity.INFO,
                category=FindingCategory.DATA_GAP,
                workload="cluster-level",
                what=f"{len(snapshot.data_gaps)} MANAGED workloads with no recommendation (< 2 hours)",
                evidence=json.dumps(snapshot.data_gaps[:5]),
                suggested_action="Monitor — will escalate to WARNING if gap persists beyond 2 hours",
            ))

        # ── 6. Memory leak pattern ───────────────────────────────────
        for leak in (memory_leaks or []):
            _escalate(Severity.WARNING)
            findings.append(Finding(
                severity=Severity.WARNING,
                category=FindingCategory.MEMORY_LEAK,
                workload=f"{leak['namespace']}/{leak['workload']}",
                what=f"Memory request increasing across {len(leak['trend_mib'])} snapshots (+{leak['growth_pct']}%)",
                evidence=f"trend_mib={leak['trend_mib']}",
                suggested_action="Investigate application memory growth; may trigger OOM recovery compounding",
            ))

        # ── 7. Absurd recommendation ─────────────────────────────────
        for ar in snapshot.absurd_recommendations:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.ABSURD_RECOMMENDATION,
                workload=ar["workload"],
                what=ar["reason"],
                evidence=json.dumps(ar),
                suggested_action="Disable WOOP for this workload immediately — likely runaway OOM recovery logic",
            ))

        # ── 8. Agent restart loop ────────────────────────────────────
        if agent_restarts_last_hour > 3:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.AGENT_RESTART,
                workload="castai-system",
                what=f"CAST AI agent(s) restarted {agent_restarts_last_hour} times in the last hour",
                evidence=f"agent_restarts_delta={agent_restarts_last_hour}",
                suggested_action="Check agent logs for crash loops or silent ExitCode:0 failures",
            ))
        elif agent_restarts_last_hour > 0:
            _escalate(Severity.WARNING)
            findings.append(Finding(
                severity=Severity.WARNING,
                category=FindingCategory.AGENT_RESTART,
                workload="castai-system",
                what=f"CAST AI agent(s) restarted {agent_restarts_last_hour} time(s) in the last hour",
                evidence=f"agent_restarts_delta={agent_restarts_last_hour}",
                suggested_action="Monitor; >3/hour escalates to CRITICAL",
            ))

        # ExitCode:0 with high restart count (silent failure pattern)
        for agent in snapshot.agent_pods:
            if agent.get("exit_code_zero_history") and agent.get("restart_count", 0) > 5:
                _escalate(Severity.CRITICAL)
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.AGENT_RESTART,
                    workload=f"castai-system/{agent['name']}",
                    what=f"Agent has ExitCode:0 history with {agent['restart_count']} restarts — silent failure pattern",
                    evidence=json.dumps(agent),
                    suggested_action="Investigate immediately — this pattern caused multi-day outages before",
                ))

        # ── 9. Webhook / exporter failure ────────────────────────────
        for sig in snapshot.log_signals:
            if sig.get("signal") == "webhook_exporter_failure":
                _escalate(Severity.WARNING)
                findings.append(Finding(
                    severity=Severity.WARNING,
                    category=FindingCategory.WEBHOOK_FAILURE,
                    workload="castai-system/workload-autoscaler",
                    what=f"{sig['count']} webhook/exporter error(s) in last 15 min",
                    evidence=json.dumps(sig.get("sample", [])[:2]),
                    suggested_action="Check WA webhook and exporter health; recommendations may not be applied",
                ))
            elif sig.get("signal") == "wa_mutation_errors":
                _escalate(Severity.WARNING)
                findings.append(Finding(
                    severity=Severity.WARNING,
                    category=FindingCategory.WEBHOOK_FAILURE,
                    workload="castai-system/workload-autoscaler",
                    what=f"{sig['count']} WA mutation error(s) — possible overflow or webhook issue",
                    evidence=json.dumps(sig.get("sample", [])[:2]),
                    suggested_action="Check for integer overflow in recommendations or webhook connectivity",
                ))

        # ── 10. Cascading scaling ────────────────────────────────────
        if abs(node_count_delta_pct) > 50:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.CASCADING_SCALING,
                workload="cluster-level",
                what=f"Node count changed by {node_count_delta_pct:+.0f}% vs. trailing average",
                evidence=f"node_count={snapshot.node_count}, delta_pct={node_count_delta_pct:.1f}%",
                suggested_action="Check for runaway autoscaler or cascading scaling event",
            ))
        if abs(pod_count_delta_pct) > 50:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.CASCADING_SCALING,
                workload="cluster-level",
                what=f"Pod count changed by {pod_count_delta_pct:+.0f}% vs. trailing average",
                evidence=f"total_pods={snapshot.total_pods}, delta_pct={pod_count_delta_pct:.1f}%",
                suggested_action="Check for unexpected deployment scaling or pod storm",
            ))

        # ── CrashLoop check (not its own scenario but always important) ──
        if snapshot.crashloop_pods > 0:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.OTHER,
                workload="cluster-level",
                what=f"{snapshot.crashloop_pods} pod(s) in CrashLoopBackOff",
                evidence=f"crashloop_pods={snapshot.crashloop_pods}",
                suggested_action="Investigate CrashLooping pods — all replicas may be down",
            ))

        # ── Unhealthy deployments (from log_signals) ─────────────────
        for sig in snapshot.log_signals:
            if sig.get("signal") == "unhealthy_deployments":
                _escalate(Severity.WARNING)
                findings.append(Finding(
                    severity=Severity.WARNING,
                    category=FindingCategory.OTHER,
                    workload="cluster-level",
                    what=f"{sig['count']} deployment(s) with 0 ready replicas",
                    evidence=json.dumps(sig.get("sample", [])[:3]),
                    suggested_action="Check deployment rollout status and pod events",
                ))

        metrics = EvaluationMetrics(
            total_pods=snapshot.total_pods,
            oomkilled_pods=len(snapshot.oomkilled_pods),
            pending_pods=snapshot.pending_pods,
            crashloop_pods=snapshot.crashloop_pods,
            recommendation_mismatches=len(snapshot.recommendation_mismatches),
            absurd_recommendations=len(snapshot.absurd_recommendations),
            agent_restarts_last_hour=agent_restarts_last_hour,
            node_count=snapshot.node_count,
            node_count_delta_pct=node_count_delta_pct,
        )

        return EvaluationResult(
            verdict=verdict,
            summary=f"[RAW METRICS FALLBACK] {len(findings)} issues detected without LLM evaluation",
            findings=findings,
            metrics=metrics,
            model_used="raw_metrics_fallback",
        )

    def _compact_snapshot(
        self, snapshot: SnapshotData, mature_data_gaps: list[dict],
    ) -> dict:
        """Produce a compact snapshot for the LLM prompt.

        Strips bulk data (woop_workloads = 2K+ objects, full node list)
        and keeps only pre-computed findings and aggregated metrics.
        Target: < 8K tokens (~32K chars).
        """
        return {
            "timestamp": snapshot.timestamp,
            "cluster_id": snapshot.cluster_id,
            # Pod health
            "total_pods": snapshot.total_pods,
            "running_pods": snapshot.running_pods,
            "pending_pods": snapshot.pending_pods,
            "crashloop_pods": snapshot.crashloop_pods,
            "oomkilled_pods": snapshot.oomkilled_pods[:10],
            # Node health (summary, not full list)
            "node_count": snapshot.node_count,
            "nodes_summary": [
                {
                    "name": n.get("name", ""),
                    "ready": n.get("ready", ""),
                    "alloc_cpu_m": n.get("alloc_cpu_m", 0),
                    "alloc_mem_bytes": n.get("alloc_mem_bytes", 0),
                }
                for n in snapshot.nodes[:10]
            ],
            # WOOP findings (pre-computed, not raw workloads)
            "woop_workload_count": len(snapshot.woop_workloads),
            "recommendation_mismatches": snapshot.recommendation_mismatches[:10],
            "absurd_recommendations": snapshot.absurd_recommendations[:10],
            "data_gaps": snapshot.data_gaps[:10],
            "mature_data_gaps": mature_data_gaps[:10],
            # Agent health
            "agent_pods": snapshot.agent_pods,
            "agent_restarts_last_hour": snapshot.agent_restarts_last_hour,
            # Memory consumers (top 10 for leak detection)
            "top_memory_consumers": snapshot.workload_memory_usage[:10],
            # Log signals
            "log_signals": snapshot.log_signals,
        }

    def _compact_history(self, history: list[dict]) -> list[dict]:
        """Produce a compact version of history to fit within token budget.

        Strips raw workload lists and keeps only aggregated metrics
        to avoid blowing the ~4K input token budget.
        """
        compact = []
        for snap in history:
            entry = {
                "timestamp": snap.get("timestamp"),
                "total_pods": snap.get("total_pods", 0),
                "pending_pods": snap.get("pending_pods", 0),
                "crashloop_pods": snap.get("crashloop_pods", 0),
                "oomkilled_count": len(snap.get("oomkilled_pods", [])),
                "oomkilled_workloads": [
                    f"{p.get('namespace')}/{p.get('name')}"
                    for p in snap.get("oomkilled_pods", [])[:10]
                ],
                "node_count": snap.get("node_count", 0),
                "agent_restarts": snap.get("agent_restarts_last_hour", 0),
                "recommendation_mismatches": len(snap.get("recommendation_mismatches", [])),
                "absurd_recommendations": len(snap.get("absurd_recommendations", [])),
                "data_gaps": len(snap.get("data_gaps", [])),
                "collection_errors": snap.get("collection_errors", []),
            }
            # Include top 5 memory consumers for leak trend detection
            mem_usage = snap.get("workload_memory_usage", [])
            if mem_usage:
                entry["top_memory_mib"] = [
                    {"wl": f"{m.get('namespace')}/{m.get('workload')}",
                     "mib": m.get("request_mem_mib", 0)}
                    for m in mem_usage[:5]
                ]
            # Include per-agent restart counts for delta tracking
            agent_pods = snap.get("agent_pods", [])
            if agent_pods:
                entry["agent_pods"] = [
                    {"name": a.get("name", ""), "rc": a.get("restart_count", 0)}
                    for a in agent_pods
                ]
            compact.append(entry)
        return compact
