"""Evaluator module — feeds snapshot data to an LLM for anomaly detection.

Sends the collected snapshot + trailing history to an OpenAI-compatible
LLM endpoint as a single-shot evaluation. The model acts as an SRE
reviewing the cluster, classifies findings by severity, and outputs
structured JSON.

Supports any OpenAI-compatible API (CAST AI LLM proxy, OpenAI, Anthropic
via proxy, local models, etc.) configured via LLM_BASE_URL.

Handles: LLM API failures (retry once, fallback to raw metrics), malformed
JSON responses (regex extraction fallback), and token budget management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

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

# ── System prompt (dynamic — cluster context injected from config) ───

SYSTEM_PROMPT_TEMPLATE = """SYSTEM PROMPT FOR CLUSTER WATCHDOG EVALUATION

You are an SRE monitoring agent for CAST AI. You inspect Kubernetes
cluster snapshots every 5 minutes and identify anomalies that need
human attention. Your job is to reduce noise—only flag things that
a senior SRE would actually investigate.

## Cluster Context ({cluster_name})

{cluster_context}

Known fragile components (extra scrutiny):
- castai-agent: Any restart is worth logging; >{agent_restart_critical}/hour is CRITICAL.
- castai-workload-autoscaler: should be running with 2 replicas.
  If only 1 replica is Running, flag as WARNING.
- castai-workload-autoscaler-exporter: watch for metric limit
  errors in logs (via loki_query). If exporter is unhealthy,
  recommendations will be incomplete.

## Your Task

Analyze the following snapshot and trailing history. Produce a JSON
response with this exact structure:

{{{{
  "verdict": "HEALTHY" | "WARNING" | "CRITICAL",
  "summary": "<one sentence overall assessment>",
  "findings": [
    {{{{
      "severity": "critical" | "warning" | "info",
      "category": "<oomkill|mismatch|unschedulable|agent|data_gap|memory_leak|absurd_recommendation|agent_restart|webhook_failure|cascading_scaling|config|other>",
      "workload": "<namespace/workload or cluster-level>",
      "what": "<what is happening>",
      "evidence": "<specific numbers from the snapshot>",
      "suggested_action": "<what the TAM or customer should do>"
    }}}}
  ],
  "metrics": {{{{
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
  }}}}
}}}}

## Classification Rules

CRITICAL (post immediately):
- Any pod OOMKilled more than {oomkill_critical} times in the last hour
- Any WOOP recommendation where applied value differs from intended by more than {mismatch_pct}%
- Any workload with all replicas in CrashLoopBackOff
- CAST AI agent pod not running
- Any pod Pending for more than {pending_minutes} minutes
- Any WOOP recommendation exceeding {absurd_mem_gib} GiB memory or {absurd_cpu} CPU cores for a single workload. This is ALWAYS a bug or runaway logic.
- CAST AI agent pod restarting more than {agent_restart_critical} times in the last hour, or showing ExitCode:0 with restartCount > {agent_restart_count_critical}.
- Cluster node count or total pod count increased by >{node_spike_pct}% vs. the trailing 1-hour average without a corresponding deployment event (cascading scaling / runaway autoscaler).

WARNING (post, but less urgent):
- Memory usage trending upward across {leak_snapshots}+ consecutive snapshots without leveling off (potential leak)
- Workload with optimization enabled but no recommendation for >{data_gap_hours} hours
- Node running at >{node_mem_pct}% memory utilization
- Workload autoscaler webhook not responding or exporter pod not Running
- Any single pod memory usage exceeding {mem_over_request}x its request across 3+ snapshots (potential memory leak)

HEALTHY (do not post):
- All pods Running with expected restart counts
- Resource usage within expected ranges
- Scale-to-zero workloads at zero replicas (this is normal)
- Sporadic single OOMKill with no repeat (transient)

## Important: Counts and Totals

The snapshot data below may show CAPPED lists (e.g. only 10 data_gaps or
10 recommendation_mismatches). The REAL total count is in the corresponding
`*_total` field. For example:
- `data_gaps_total` = actual number of MANAGED workloads with no recommendation
- `recommendation_mismatches_total` = actual number of mismatches
- `absurd_recommendations_total` = actual number of absurd recommendations

ALWAYS use the `*_total` fields when reporting counts in your findings.
Do NOT count the items in the capped list — that will undercount.

Also: do NOT fabricate resource values. If claiming a specific ratio like
"100x limit" or a specific byte amount, verify the exact numbers from the
snapshot containers or WOOP data. Stale or incorrect numbers erode trust.

## Snapshot Data

{{snapshot_json}}

## Trailing History (last {{window_size}} snapshots)

{{history_json}}

## Collection Errors

{{collection_errors}}

IMPORTANT: If collection errors exist, note which data sources are missing
and reduce confidence in your assessment accordingly. Do NOT report missing
data as a cluster problem — it's an observability gap."""


def _build_cluster_context(config: WatchdogConfig) -> str:
    """Build the cluster context section from config, not hardcoded values."""
    ctx = config.cluster
    lines = []

    # Known baseline behaviors
    lines.append("Known baseline behaviors (DO NOT flag these as anomalies):")
    if ctx.known_scale_to_zero_workloads:
        wl_list = ", ".join(ctx.known_scale_to_zero_workloads)
        lines.append(f"- {wl_list} workloads routinely scale to zero replicas.")
        lines.append("  Seeing 0/0 ready pods for these is NORMAL.")
    lines.append("- Some namespaces are low-traffic and may show 0 resource usage.")
    lines.append("  This is not an incident.")
    lines.append("")

    # Known configuration
    lines.append("Known configuration:")
    lines.append(f"- WOOP memory limit multiplier: {ctx.woop_memory_limit_multiplier}x")
    lines.append(f"- WOOP recommendation strategy: {ctx.woop_recommendation_strategy}")
    if ctx.woop_recommendation_strategy == "Max Usage":
        lines.append("  (NOTE: this strategy is vulnerable to outlier pods driving")
        lines.append("  extreme recommendations — flag any recommendation where a")
        lines.append("  single pod's usage is >10x the median of its siblings)")
    lines.append(f"- Cluster ID: {ctx.cluster_id}")
    if ctx.namespaces:
        ns_list = ", ".join(ctx.namespaces)
        lines.append(f"- Customer namespaces include: {ns_list}")
    if ctx.cni:
        lines.append(f"- {ctx.cni.title()} CNI in use — nodes may briefly show")
        lines.append(f"  {ctx.cni} agent-not-ready taint during startup (normal, <2 min).")
    if ctx.uses_keda:
        lines.append("- KEDA is in use — rapid replica changes driven by KEDA are")
        lines.append("  expected. Only flag if node count also spikes disproportionately.")

    return "\n".join(lines)


def build_system_prompt(config: WatchdogConfig) -> str:
    """Build the full system prompt with cluster context and thresholds from config."""
    t = config.thresholds
    ctx = config.cluster

    # First pass: inject config values into the template
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        cluster_name=ctx.cluster_name or ctx.cluster_id[:12],
        cluster_context=_build_cluster_context(config),
        agent_restart_critical=t.agent_restart_critical_per_hour,
        oomkill_critical=t.oomkill_critical_per_hour,
        mismatch_pct=t.recommendation_mismatch_pct,
        pending_minutes=t.pending_pod_minutes,
        absurd_mem_gib=t.absurd_memory_gib,
        absurd_cpu=t.absurd_cpu_cores,
        agent_restart_count_critical=t.agent_restart_count_critical,
        node_spike_pct=t.node_count_spike_pct,
        leak_snapshots=t.memory_leak_snapshots,
        data_gap_hours=t.data_gap_hours,
        node_mem_pct=t.node_memory_utilization_pct,
        mem_over_request=t.memory_usage_over_request_ratio,
    )

    return prompt


class Evaluator:
    """Evaluates cluster snapshots using an OpenAI-compatible LLM."""

    def __init__(self, config: WatchdogConfig) -> None:
        self.config = config
        self.model = config.llm.model
        self.fallback_model = config.llm.fallback_model
        self.client = AsyncOpenAI(
            base_url=config.llm.base_url,
            api_key=config.llm.api_key,
            timeout=600.0,  # reasoning models need long thinking time
        )

    async def evaluate(
        self,
        snapshot: SnapshotData,
        history: list[dict],
        node_count_delta_pct: float = 0.0,
        pod_count_delta_pct: float = 0.0,
        agent_restarts_last_hour: int = 0,
        memory_leaks: list[dict] | None = None,
        mature_data_gaps: list[dict] | None = None,
        mature_pending_pods: list[dict] | None = None,
        oomkill_trend: list[int] | None = None,
    ) -> EvaluationResult:
        """Send snapshot to Claude and parse the structured response."""

        # Build the prompt with a compact snapshot (strip bulk WOOP data)
        compact_snap = self._compact_snapshot(snapshot, mature_data_gaps or [])
        # Two-stage formatting: build_system_prompt injects config values,
        # then we inject the runtime snapshot/history data.
        base_prompt = build_system_prompt(self.config)
        prompt = base_prompt.format(
            snapshot_json=json.dumps(compact_snap, indent=2, default=str),
            history_json=json.dumps(
                self._compact_history(history), indent=2, default=str
            ),
            window_size=len(history),
            collection_errors=json.dumps(snapshot.collection_errors, indent=2)
            if snapshot.collection_errors
            else "None — all data sources collected successfully.",
        )

        # Build raw metrics result for validation & fallback
        raw_result = self._raw_metrics_fallback(
            snapshot, node_count_delta_pct, pod_count_delta_pct,
            agent_restarts_last_hour, memory_leaks or [],
            mature_data_gaps or [], oomkill_trend or [],
            mature_pending_pods=mature_pending_pods or [],
        )

        # Try primary model with retries, then fallback model with retries
        for model in [self.model, self.fallback_model]:
            for attempt in range(1, 4):  # 3 attempts per model
                try:
                    result = await self._call_llm(prompt, model)
                    if result:
                        # Cross-check LLM findings against raw data
                        validated = self._validate_llm_result(
                            result, raw_result, snapshot,
                            node_count_delta_pct=node_count_delta_pct,
                            pod_count_delta_pct=pod_count_delta_pct,
                            mature_pending_pods=mature_pending_pods or [],
                            mature_data_gaps=mature_data_gaps or [],
                            memory_leaks=memory_leaks or [],
                        )
                        return validated
                    # JSON parse failed — retry with same model before switching
                    if attempt < 3:
                        wait = 2 ** attempt
                        logger.warning(
                            "Model %s returned unparseable response (attempt %d/3), retrying in %ds",
                            model, attempt, wait,
                        )
                        await asyncio.sleep(wait)
                except Exception as e:
                    if attempt < 3:
                        wait = 2 ** attempt
                        logger.warning(
                            "Model %s failed (attempt %d/3): %s, retrying in %ds",
                            model, attempt, e, wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        logger.warning("Model %s failed after 3 attempts: %s", model, e)
                    continue

        # Both models failed — return raw metrics-only evaluation
        logger.error("All LLM evaluations failed, producing raw metrics fallback")
        return raw_result

    async def _call_llm(
        self, prompt: str, model: str
    ) -> EvaluationResult | None:
        """Call an OpenAI-compatible chat completions endpoint and parse the response."""
        logger.debug(
            "Sending prompt to %s (%s): %d chars (~%d tokens)",
            model, self.config.llm.base_url, len(prompt), len(prompt) // 4,
        )
        try:
            completion = await self.client.chat.completions.create(
                model=model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            logger.error("LLM %s request failed: %s", model, e)
            raise

        msg = completion.choices[0].message if completion.choices else None
        text = (msg.content or "") if msg else ""

        # Reasoning models (kimi, deepseek-r1, etc.) may put the answer
        # in reasoning_content when content is empty/null
        if not text and msg:
            rc = getattr(msg, "reasoning_content", None)
            if not rc:
                # Also check provider_specific_fields (raw API passthrough)
                psf = getattr(msg, "provider_specific_fields", None) or {}
                if isinstance(psf, dict):
                    rc = psf.get("reasoning_content") or psf.get("reasoning")
            if rc:
                logger.info("Using reasoning_content as primary response from %s", model)
                text = rc

        if not text:
            logger.error("Empty response from %s", model)
            return None

        # Parse JSON from the response
        parsed = self._parse_llm_json(text)
        if not parsed:
            logger.error(
                "Failed to parse JSON from %s response. First 500 chars: %.500s",
                model, text,
            )
            return None

        return self._build_result(parsed, model, text)

    def _parse_llm_json(self, text: str) -> dict | None:
        """Extract JSON from LLM response, handling markdown fences, preamble, and partial output."""
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

        # Try 3: brace-matched extraction — find the first top-level { } block
        start = text.find("{")
        if start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                c = text[i]
                if escape:
                    escape = False
                    continue  # skip escaped character entirely
                if c == "\\":
                    if in_string:
                        escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break

        # Try 4: greedy regex fallback
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Try 5: truncated JSON — model hit token limit mid-output
        # Find the last complete "findings" entry and close the structure
        if '"findings"' in text and start is not None:
            truncated = text[start:] if start != -1 else text
            # Close any open arrays/objects
            for suffix in [']}]}', '"]}]}', '"}]}', '"]}}']:
                try:
                    return json.loads(truncated + suffix)
                except json.JSONDecodeError:
                    continue

        logger.debug("Unparseable LLM response (first 1000 chars): %.1000s", text)
        return None

    def _build_result(
        self, parsed: dict, model: str, raw: str
    ) -> EvaluationResult:
        """Convert parsed JSON into typed EvaluationResult."""
        findings = []
        for f in parsed.get("findings", []):
            try:
                try:
                    category = FindingCategory(f.get("category", "other"))
                except ValueError:
                    logger.info("Unknown category '%s', mapping to OTHER", f.get("category"))
                    category = FindingCategory.OTHER
                try:
                    severity = Severity(f.get("severity", "info"))
                except ValueError:
                    severity = Severity.INFO
                findings.append(Finding(
                    severity=severity,
                    category=category,
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

    def _validate_llm_result(
        self,
        llm_result: EvaluationResult,
        raw_result: EvaluationResult,
        snapshot: SnapshotData,
        node_count_delta_pct: float = 0.0,
        pod_count_delta_pct: float = 0.0,
        mature_pending_pods: list[dict] | None = None,
        mature_data_gaps: list[dict] | None = None,
        memory_leaks: list[dict] | None = None,
    ) -> EvaluationResult:
        """Cross-check LLM findings against raw snapshot data to eliminate hallucinations.

        For each LLM finding, verify that the underlying data supports it.
        Then merge in any real findings from raw metrics that the LLM missed.
        """
        # Build lookup sets from actual snapshot data for fast validation
        oom_workloads = set()
        for o in snapshot.oomkilled_pods:
            ns = o.get("namespace", "")
            name = o.get("name", o.get("workload_name", ""))
            if ns and name:
                oom_workloads.add(f"{ns}/{name}")
            # Also add partial matches (just the workload name)
            if name:
                oom_workloads.add(name)

        mismatch_workloads = {m.get("workload", "") for m in snapshot.recommendation_mismatches}
        absurd_workloads = {a.get("workload", "") for a in snapshot.absurd_recommendations}
        data_gap_workloads = {d.get("workload", "") for d in snapshot.data_gaps}

        agent_names = {a.get("name", "") for a in snapshot.agent_pods}
        agent_not_running = any(a.get("phase") != "Running" for a in snapshot.agent_pods)

        # Build workload resource lookup for CONFIG/OTHER validation
        # Maps "ns/workload_name" → {mem_request, mem_limit, cpu_request, cpu_limit}
        woop_resource_map: dict[str, dict] = {}
        for w in snapshot.woop_workloads:
            wl_key = f"{w.get('namespace', '')}/{w.get('workloadName', w.get('name', ''))}"
            for c in w.get("containers", []):
                res = c.get("resources", c.get("requests", {}))
                woop_resource_map[wl_key] = {
                    "mem_request": res.get("mem_request", res.get("requestMemMib", 0)),
                    "mem_limit": res.get("mem_limit", res.get("limitMemMib", 0)),
                    "cpu_request": res.get("cpu_request", res.get("requestCpuMillis", 0)),
                    "cpu_limit": res.get("cpu_limit", res.get("limitCpuMillis", 0)),
                }
                break  # first container is representative enough

        # Also index by short name for fuzzy matching
        woop_resource_short: dict[str, dict] = {}
        for key, val in woop_resource_map.items():
            short = key.split("/")[-1] if "/" in key else key
            woop_resource_short[short] = val

        # Validate each LLM finding
        validated = []
        dropped = 0

        for finding in llm_result.findings:
            cat = finding.category
            wl = finding.workload

            # Extract the workload name (last segment after /)
            wl_name = wl.split("/")[-1] if "/" in wl else wl

            keep = False

            if cat == FindingCategory.OOMKILL:
                # Must match a specific OOMKilled pod AND meet restart threshold
                oom_threshold = self.config.thresholds.oomkill_critical_per_hour
                matches = [o for o in snapshot.oomkilled_pods
                           if wl_name in o.get("name", o.get("workload_name", ""))
                           or wl == f"{o.get('namespace', '')}/{o.get('name', '')}"]
                # Sum restarts across matching pods for this workload
                total_restarts = sum(m.get("restart_count", 0) for m in matches)
                if matches and total_restarts >= oom_threshold:
                    keep = True
                    finding.evidence = json.dumps(matches[0])
                elif wl == "cluster-level":
                    # cluster-level: keep only if total OOMKills meet threshold
                    total_oom_restarts = sum(
                        o.get("restart_count", 0) for o in snapshot.oomkilled_pods
                    )
                    if total_oom_restarts >= oom_threshold:
                        keep = True
                        finding.evidence = json.dumps(snapshot.oomkilled_pods[:5])

            elif cat == FindingCategory.MISMATCH:
                # Must find matching evidence — no evidence = no alert
                match = next((m for m in snapshot.recommendation_mismatches
                              if m.get("workload") == wl or wl_name in m.get("workload", "")), None)
                if match:
                    keep = True
                    finding.evidence = json.dumps(match)

            elif cat == FindingCategory.ABSURD_RECOMMENDATION:
                # Must find matching evidence — no evidence = no alert
                match = next((a for a in snapshot.absurd_recommendations
                              if a.get("workload") == wl or wl_name in a.get("workload", "")), None)
                if match:
                    keep = True
                    finding.evidence = json.dumps(match)

            elif cat == FindingCategory.UNSCHEDULABLE:
                # Only keep if the pod has been Pending beyond the maturity threshold (15 min).
                # Fresh Pending pods are noise — they may schedule on the next cycle.
                m_pending = mature_pending_pods or []
                matches = [p for p in m_pending
                           if wl_name in p.get("name", "")
                           or wl == f"{p.get('namespace', '')}/{p.get('name', '')}"]
                keep = len(matches) > 0
                if keep:
                    finding.evidence = json.dumps(matches[0])

            elif cat in (FindingCategory.AGENT, FindingCategory.AGENT_RESTART):
                if cat == FindingCategory.AGENT_RESTART:
                    # Agent restart loop: require ≥3 restarts/hr (matches critical threshold)
                    keep = snapshot.agent_restarts_last_hour >= self.config.thresholds.agent_restart_critical_per_hour
                else:
                    # Agent down: require a non-Running agent pod or no agent pods at all
                    keep = agent_not_running or not snapshot.agent_pods
                # Enrich with agent pod state
                if keep and snapshot.agent_pods:
                    finding.evidence = json.dumps({
                        "agent_pods": snapshot.agent_pods,
                        "restarts_last_hour": snapshot.agent_restarts_last_hour,
                    })

            elif cat == FindingCategory.DATA_GAP:
                # Only keep if the gap is mature (persisted >2h), matching fallback behavior.
                # Fresh gaps (<2h) are INFO-only and handled by expansion from raw_result.
                m_gaps = mature_data_gaps or []
                match = next((d for d in m_gaps
                              if d.get("workload") == wl or wl_name in d.get("workload", "")), None)
                if match:
                    keep = True
                    finding.evidence = json.dumps(match)
                elif wl == "cluster-level" and m_gaps:
                    keep = True
                    finding.evidence = json.dumps(m_gaps[:5])

            elif cat == FindingCategory.MEMORY_LEAK:
                # Only keep if the workload has a trend-detected memory leak
                # (consecutive upward trend across snapshots), not just high current usage.
                m_leaks = memory_leaks or []
                matches = [m for m in m_leaks
                           if wl_name in m.get("workload", "")
                           or wl == f"{m.get('namespace', '')}/{m.get('workload', '')}"]
                if matches:
                    keep = True
                    finding.evidence = json.dumps(matches[0])
                elif wl == "cluster-level" and m_leaks:
                    keep = True
                    finding.evidence = json.dumps(m_leaks[:5])

            elif cat == FindingCategory.CASCADING_SCALING:
                spike_pct = self.config.thresholds.node_count_spike_pct
                keep = node_count_delta_pct > spike_pct or pod_count_delta_pct > spike_pct
                # Enrich with actual delta data
                if keep:
                    finding.evidence = json.dumps({
                        "node_count": snapshot.node_count,
                        "total_pods": snapshot.total_pods,
                        "node_count_delta_pct": round(node_count_delta_pct, 1),
                        "pod_count_delta_pct": round(pod_count_delta_pct, 1),
                    })

            elif cat == FindingCategory.WEBHOOK_FAILURE:
                keep = any(
                    s.get("signal") in ("webhook_exporter_failure", "wa_mutation_errors")
                    for s in snapshot.log_signals
                )
                # Enrich with matching log signals
                if keep:
                    signals = [s for s in snapshot.log_signals
                               if s.get("signal") in ("webhook_exporter_failure", "wa_mutation_errors")]
                    if signals:
                        finding.evidence = json.dumps(signals)

            elif cat == FindingCategory.UNHEALTHY_DEPLOYMENT:
                # Must match a specific workload with crashlooping pods
                matches = [p for p in snapshot.crashloop_pods_detail
                           if wl_name in p.get("name", "")]
                if matches:
                    keep = True
                    finding.evidence = json.dumps(matches[:5])
                elif wl == "cluster-level" and snapshot.crashloop_pods_detail:
                    keep = True
                    finding.evidence = json.dumps(snapshot.crashloop_pods_detail[:5])

            elif cat in (FindingCategory.CONFIG, FindingCategory.OTHER):
                # Validate resource-specific claims against actual container specs.
                # If the finding mentions a specific workload, verify it exists and
                # that any resource ratio claims (e.g. "100x limit") are plausible.
                res = woop_resource_map.get(wl) or woop_resource_short.get(wl_name)
                if res:
                    # Workload exists — check if the finding's claim is plausible
                    what_lower = finding.what.lower()
                    if "100x" in what_lower or "limit" in what_lower and "request" in what_lower:
                        # Finding claims a specific limit:request ratio — verify
                        mem_req = res.get("mem_request", 0)
                        mem_lim = res.get("mem_limit", 0)
                        if mem_req > 0 and mem_lim > 0:
                            ratio = mem_lim / mem_req
                            # Only keep if the ratio is actually extreme (>10x)
                            keep = ratio > 10
                            if not keep:
                                logger.info(
                                    "Dropped stale CONFIG finding for %s: actual ratio %.1fx (not 100x)",
                                    wl, ratio,
                                )
                        else:
                            keep = True  # can't verify, keep conservatively
                    else:
                        keep = True  # non-ratio claim, keep
                elif wl in ("cluster-level", "org-level"):
                    keep = True  # cluster-wide findings are OK
                else:
                    # Workload not found in snapshot at all — drop
                    keep = False

            if keep:
                validated.append(finding)
            else:
                dropped += 1
                logger.warning(
                    "Dropped unconfirmed finding: [%s] %s on %s — no supporting data in snapshot",
                    finding.severity.value, finding.category.value, finding.workload,
                )

        # Replace grouped LLM findings with per-workload raw findings.
        # When the LLM lumps N workloads into one cluster-level finding
        # but the raw fallback has per-workload detail, prefer the raw detail.
        _expandable = {
            FindingCategory.DATA_GAP, FindingCategory.MISMATCH,
            FindingCategory.ABSURD_RECOMMENDATION, FindingCategory.UNSCHEDULABLE,
            FindingCategory.OOMKILL, FindingCategory.OTHER,
        }
        raw_by_cat: dict[str, list[Finding]] = {}
        for rf in raw_result.findings:
            raw_by_cat.setdefault(rf.category.value, []).append(rf)

        # Build set of per-workload names from raw findings for comparison
        raw_workloads_by_cat: dict[str, set[str]] = {}
        for rf in raw_result.findings:
            raw_workloads_by_cat.setdefault(rf.category.value, set()).add(rf.workload)

        expanded = []
        expanded_cats = set()
        for f in validated:
            # Check if this LLM finding is a grouped/summarized one
            # (e.g. "multiple (10 workloads)") rather than per-workload
            raw_wl_names = raw_workloads_by_cat.get(f.category.value, set())
            is_grouped = (
                f.category in _expandable
                and f.category.value in raw_by_cat
                and len(raw_by_cat[f.category.value]) > 1
                and f.workload not in raw_wl_names
                and (
                    any(
                        kw in f.workload.lower()
                        for kw in ("cluster-level", "multiple", "workload")
                    )
                    or f.workload == "cluster-level"
                )
            )
            if is_grouped and f.category.value not in expanded_cats:
                # Replace with per-workload raw findings
                expanded.extend(raw_by_cat[f.category.value])
                expanded_cats.add(f.category.value)
            else:
                expanded.append(f)
        validated = expanded

        # Merge in raw-metrics findings the LLM missed entirely
        llm_categories_workloads = {
            (f.category.value, f.workload) for f in validated
        }
        merged_from_raw = 0
        for raw_finding in raw_result.findings:
            key = (raw_finding.category.value, raw_finding.workload)
            if key not in llm_categories_workloads:
                if raw_finding.workload == "cluster-level":
                    if any(f.category == raw_finding.category for f in validated):
                        continue
                validated.append(raw_finding)
                merged_from_raw += 1

        if dropped or merged_from_raw:
            logger.info(
                "Validation: %d LLM findings kept, %d hallucinations dropped, %d raw findings merged",
                len(validated) - merged_from_raw, dropped, merged_from_raw,
            )

        # Recompute verdict from validated findings
        verdict = Verdict.HEALTHY
        for f in validated:
            if f.severity == Severity.CRITICAL:
                verdict = Verdict.CRITICAL
                break
            if f.severity == Severity.WARNING:
                verdict = Verdict.WARNING

        # Update summary if findings changed significantly
        summary = llm_result.summary
        if dropped > 0:
            summary = f"[Validated: {dropped} unconfirmed finding(s) removed] {summary}"

        return EvaluationResult(
            verdict=verdict,
            summary=summary,
            findings=validated,
            metrics=llm_result.metrics,
            model_used=f"{llm_result.model_used}+validated",
            raw_response=llm_result.raw_response,
        )

    def _raw_metrics_fallback(
        self,
        snapshot: SnapshotData,
        node_count_delta_pct: float,
        pod_count_delta_pct: float = 0.0,
        agent_restarts_last_hour: int = 0,
        memory_leaks: list[dict] | None = None,
        mature_data_gaps: list[dict] | None = None,
        oomkill_trend: list[int] | None = None,
        mature_pending_pods: list[dict] | None = None,
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
        # Only alert if total restarts across OOMKilled pods meet threshold
        total_oom_restarts = sum(o.get("restart_count", 0) for o in snapshot.oomkilled_pods)
        if total_oom_restarts >= t.oomkill_critical_per_hour:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.OOMKILL,
                workload="cluster-level",
                what=f"{len(snapshot.oomkilled_pods)} pods with OOMKill detected ({total_oom_restarts} total restarts)",
                evidence=json.dumps(snapshot.oomkilled_pods[:3]),
                suggested_action="Investigate OOMKill workloads immediately",
            ))

        # OOMKill escalating trend: 3+ consecutive snapshots with increasing OOM count
        trend = oomkill_trend or []
        if len(trend) >= 3:
            recent = trend[-3:]
            if recent[-1] > 0 and all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                _escalate(Severity.CRITICAL)
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.OOMKILL,
                    workload="cluster-level",
                    what=f"OOMKill count escalating across {len(recent)} snapshots: {recent}",
                    evidence=f"oomkill_trend={trend}",
                    suggested_action="OOMKill spiral detected — investigate affected workloads immediately",
                ))

        # ── 2. Recommendation/actual mismatch ────────────────────────
        for mm in snapshot.recommendation_mismatches:
            _escalate(Severity.CRITICAL)
            woop_info = f" [{mm['woop']}]" if mm.get("woop") else ""
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.MISMATCH,
                workload=mm.get("workload", "unknown"),
                what=f"Recommendation/actual mismatch ({mm.get('diff_pct', '?')}% divergence){woop_info}",
                evidence=json.dumps(mm),
                suggested_action="Disable WOOP for this workload and investigate",
            ))

        # ── 3. Unschedulable workloads (>15 min = CRITICAL) ──────────
        m_pending = mature_pending_pods or []
        if m_pending:
            # Pods stuck Pending for >15 minutes — per-pod findings
            _escalate(Severity.CRITICAL)
            for p in m_pending[:10]:
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.UNSCHEDULABLE,
                    workload=f"{p.get('namespace', '')}/{p.get('name', '')}",
                    what=f"Pod Pending for {p.get('age_minutes', 0):.0f} min — {p.get('reason', 'unknown reason')}",
                    evidence=json.dumps(p),
                    suggested_action="Check node capacity and resource requests",
                ))
            remaining = len(m_pending) - 10
            if remaining > 0:
                findings.append(Finding(
                    severity=Severity.INFO,
                    category=FindingCategory.UNSCHEDULABLE,
                    workload="cluster-level",
                    what=f"+{remaining} more pod(s) Pending >15 min",
                    evidence=f"total_mature_pending={len(m_pending)}",
                    suggested_action="Review all stuck Pending pods in console",
                ))
        elif snapshot.pending_pods > 0:
            # Fresh Pending pods (< 15 min) — INFO only, don't escalate
            findings.append(Finding(
                severity=Severity.INFO,
                category=FindingCategory.UNSCHEDULABLE,
                workload="cluster-level",
                what=f"{snapshot.pending_pods} pod(s) in Pending state (< 15 min)",
                evidence=json.dumps(snapshot.pending_pods_detail[:5]),
                suggested_action="Monitor — will escalate to CRITICAL if still Pending after 15 min",
            ))

        # ── 4. Agent down ────────────────────────────────────────────
        for agent in snapshot.agent_pods:
            if agent.get("phase") != "Running":
                _escalate(Severity.CRITICAL)
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.AGENT,
                    workload=f"castai-system/{agent.get('name', 'unknown')}",
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
                    what=f"{sig.get('count', '?')} cluster(s) with agent not online",
                    evidence=json.dumps(sig.get("sample", [])[:3]),
                    suggested_action="Check agent status on offline clusters",
                ))

        # ── 5. Data gap / no recommendation (>2 hours) ──────────────
        mgaps = mature_data_gaps or []
        if mgaps:
            _escalate(Severity.WARNING)
            for g in mgaps[:15]:
                woop_info = f" [{g.get('woop', '')}]" if g.get("woop") else ""
                age = g.get("age_hours", "?")
                findings.append(Finding(
                    severity=Severity.WARNING,
                    category=FindingCategory.DATA_GAP,
                    workload=g.get("workload", "unknown"),
                    what=f"MANAGED{woop_info} — no recommendation for {age}h",
                    evidence=json.dumps(g),
                    suggested_action="Check WOOP exporter health; workload not being optimized",
                ))
            remaining = len(mgaps) - min(len(mgaps), 15)
            if remaining > 0:
                findings.append(Finding(
                    severity=Severity.INFO,
                    category=FindingCategory.DATA_GAP,
                    workload="cluster-level",
                    what=f"+{remaining} more MANAGED workload(s) with no recommendation for >2 hours",
                    evidence=f"total_mature_data_gaps={len(mgaps)}",
                    suggested_action="Review all data gaps in console",
                ))
        elif snapshot.data_gaps:
            # Fresh gaps (< 2 hours) — INFO, show per-workload
            for g in snapshot.data_gaps[:10]:
                woop_info = f" [{g.get('woop', '')}]" if g.get("woop") else ""
                findings.append(Finding(
                    severity=Severity.INFO,
                    category=FindingCategory.DATA_GAP,
                    workload=g.get("workload", "unknown"),
                    what=f"MANAGED{woop_info} — no recommendation (< 2 hours)",
                    evidence=json.dumps(g),
                    suggested_action="Monitor — will escalate to WARNING if gap persists beyond 2 hours",
                ))
            remaining = len(snapshot.data_gaps) - min(len(snapshot.data_gaps), 10)
            if remaining > 0:
                findings.append(Finding(
                    severity=Severity.INFO,
                    category=FindingCategory.DATA_GAP,
                    workload="cluster-level",
                    what=f"+{remaining} more MANAGED workload(s) with no recommendation (< 2 hours)",
                    evidence=f"total_fresh_data_gaps={len(snapshot.data_gaps)}",
                    suggested_action="Monitor — will escalate if gap persists",
                ))

        # ── 6. Memory leak pattern ───────────────────────────────────
        for leak in (memory_leaks or []):
            _escalate(Severity.WARNING)
            findings.append(Finding(
                severity=Severity.WARNING,
                category=FindingCategory.MEMORY_LEAK,
                workload=f"{leak.get('namespace', '?')}/{leak.get('workload', '?')}",
                what=f"Memory request increasing across {len(leak.get('trend_mib', []))} snapshots (+{leak.get('growth_pct', '?')}%)",
                evidence=f"trend_mib={leak.get('trend_mib', [])}",
                suggested_action="Investigate application memory growth; may trigger OOM recovery compounding",
            ))

        # ── 7. Absurd recommendation ─────────────────────────────────
        for ar in snapshot.absurd_recommendations:
            _escalate(Severity.CRITICAL)
            woop_info = f" [{ar['woop']}]" if ar.get("woop") else ""
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.ABSURD_RECOMMENDATION,
                workload=ar.get("workload", "unknown"),
                what=f"{ar.get('reason', 'absurd recommendation')}{woop_info}",
                evidence=json.dumps(ar),
                suggested_action="Disable WOOP for this workload immediately — likely runaway OOM recovery logic",
            ))

        # ── 8. Agent restart loop ────────────────────────────────────
        if agent_restarts_last_hour > t.agent_restart_critical_per_hour:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.AGENT_RESTART,
                workload="castai-system",
                what=f"CAST AI agent(s) restarted {agent_restarts_last_hour} times in the last hour",
                evidence=f"agent_restarts_delta={agent_restarts_last_hour}",
                suggested_action="Check agent logs for crash loops or silent ExitCode:0 failures",
            ))
        # 1-2 restarts/hr is normal operational behavior — don't alert

        # ExitCode:0 with high restart count (silent failure pattern)
        for agent in snapshot.agent_pods:
            if agent.get("exit_code_zero_history") and agent.get("restart_count", 0) > t.agent_restart_count_critical:
                _escalate(Severity.CRITICAL)
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.AGENT_RESTART,
                    workload=f"castai-system/{agent.get('name', 'unknown')}",
                    what=f"Agent has ExitCode:0 history with {agent.get('restart_count', 0)} restarts — silent failure pattern",
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
                    what=f"{sig.get('count', 0)} webhook/exporter error(s) in last 15 min",
                    evidence=json.dumps(sig.get("sample", [])[:2]),
                    suggested_action="Check WA webhook and exporter health; recommendations may not be applied",
                ))
            elif sig.get("signal") == "wa_mutation_errors":
                _escalate(Severity.WARNING)
                findings.append(Finding(
                    severity=Severity.WARNING,
                    category=FindingCategory.WEBHOOK_FAILURE,
                    workload="castai-system/workload-autoscaler",
                    what=f"{sig.get('count', 0)} WA mutation error(s) — possible overflow or webhook issue",
                    evidence=json.dumps(sig.get("sample", [])[:2]),
                    suggested_action="Check for integer overflow in recommendations or webhook connectivity",
                ))

        # ── 10. Cascading scaling (only on scale-UP, not consolidation) ──
        if node_count_delta_pct > t.node_count_spike_pct:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.CASCADING_SCALING,
                workload="cluster-level",
                what=f"Node count spiked {node_count_delta_pct:+.0f}% vs. trailing average",
                evidence=f"node_count={snapshot.node_count}, delta_pct={node_count_delta_pct:.1f}%",
                suggested_action="Check for runaway autoscaler or cascading scaling event",
            ))
        if pod_count_delta_pct > t.node_count_spike_pct:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.CASCADING_SCALING,
                workload="cluster-level",
                what=f"Pod count spiked {pod_count_delta_pct:+.0f}% vs. trailing average",
                evidence=f"total_pods={snapshot.total_pods}, delta_pct={pod_count_delta_pct:.1f}%",
                suggested_action="Check for unexpected deployment scaling or pod storm",
            ))

        # ── CrashLoop check (per-workload detail) ─────────────────────
        if snapshot.crashloop_pods_detail:
            _escalate(Severity.CRITICAL)
            for cl in snapshot.crashloop_pods_detail[:10]:
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.OTHER,
                    workload=f"{cl['namespace']}/{cl['name']}",
                    what=f"Container `{cl.get('container', '?')}` in CrashLoopBackOff (restarts: {cl.get('restart_count', 0)})",
                    evidence=json.dumps(cl),
                    suggested_action="Check pod logs and events — all replicas may be down",
                ))
            remaining = snapshot.crashloop_pods - len(snapshot.crashloop_pods_detail[:10])
            if remaining > 0:
                findings.append(Finding(
                    severity=Severity.INFO,
                    category=FindingCategory.OTHER,
                    workload="cluster-level",
                    what=f"+{remaining} more pod(s) in CrashLoopBackOff",
                    evidence=f"crashloop_pods={snapshot.crashloop_pods}",
                    suggested_action="Review all CrashLooping pods in console",
                ))
        elif snapshot.crashloop_pods > 0:
            # Fallback if detail not available
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
                unhealthy_added = False
                for dep in sig.get("sample", [])[:5]:
                    ns = dep.get("namespace", "")
                    name = dep.get("name", "")
                    desired = dep.get("desired", 0)
                    # Skip scale-to-zero workloads — desired=0 means KEDA/HPA scaled down
                    if desired == 0:
                        continue
                    # Verify workload actually has unhealthy pods in snapshot
                    has_crashloop = any(
                        name in p.get("name", "") for p in snapshot.crashloop_pods_detail
                    )
                    has_pending = any(
                        name in p.get("name", "") for p in snapshot.pending_pods_detail
                    )
                    if not has_crashloop and not has_pending:
                        continue
                    unhealthy_added = True
                    findings.append(Finding(
                        severity=Severity.WARNING,
                        category=FindingCategory.UNHEALTHY_DEPLOYMENT,
                        workload=f"{ns}/{name}",
                        what=f"Deployment has desired={desired} but ready=0, available=0",
                        evidence=json.dumps(dep),
                        suggested_action="Check pod events and logs — may be CrashLoopBackOff or stuck Pending",
                    ))
                if unhealthy_added:
                    _escalate(Severity.WARNING)
                    remaining = sig.get("count", 0) - min(sig.get("count", 0), 5)
                    if remaining > 0:
                        findings.append(Finding(
                            severity=Severity.INFO,
                            category=FindingCategory.UNHEALTHY_DEPLOYMENT,
                            workload="cluster-level",
                            what=f"+{remaining} more unhealthy deployment(s)",
                            evidence=json.dumps(sig.get("sample", [])[5:8]),
                            suggested_action="Review all unhealthy deployments in console",
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
            "pending_pods_detail": snapshot.pending_pods_detail[:10],
            "crashloop_pods": snapshot.crashloop_pods,
            "crashloop_pods_detail": snapshot.crashloop_pods_detail[:10],
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
            "woop_management": next(
                (s for s in snapshot.log_signals if s.get("signal") == "woop_management_summary"),
                {},
            ),
            "recommendation_mismatches": snapshot.recommendation_mismatches[:10],
            "recommendation_mismatches_total": snapshot.recommendation_mismatches_total,
            "absurd_recommendations": snapshot.absurd_recommendations[:10],
            "absurd_recommendations_total": snapshot.absurd_recommendations_total,
            "data_gaps": snapshot.data_gaps[:10],
            "data_gaps_total": snapshot.data_gaps_total,
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
