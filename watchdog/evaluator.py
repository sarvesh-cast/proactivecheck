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
from datetime import datetime, timezone
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
response with this exact structure. Do NOT include any preamble,
explanation, or <think> tags — output ONLY the JSON object:

{{{{
  "verdict": "HEALTHY" | "WARNING" | "CRITICAL",
  "summary": "<2-3 sentence overall assessment for Slack — mention the most important issues by namespace/workload name and severity>",
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

        # ── LLM provides summary only; raw metrics are the single source
        # ── of truth for findings.  This eliminates the entire validate/
        # ── expand/merge pipeline and all duplicate-finding risks.
        llm_summary: str | None = None
        llm_model: str | None = None
        for model in [self.model, self.fallback_model]:
            for attempt in range(1, 4):
                try:
                    result = await self._call_llm(prompt, model)
                    if result:
                        llm_summary = result.summary
                        llm_model = result.model_used
                        break
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
            if llm_summary:
                break

        if llm_summary:
            raw_result.summary = llm_summary
            raw_result.model_used = f"{llm_model}+raw_findings"
            logger.info(
                "Using LLM summary from %s with %d raw findings",
                llm_model, len(raw_result.findings),
            )
        else:
            logger.error("All LLM evaluations failed, using raw metrics summary")
            raw_result.llm_failed = True

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
        # Strip reasoning/thinking tags (minimax, deepseek, etc.)
        # Handle both closed <think>...</think> and unclosed <think>... (truncated)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()

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
        """Convert parsed JSON into typed EvaluationResult.

        The LLM provides verdict, summary, and metrics only.
        Findings are generated separately by _raw_metrics_fallback.
        """
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
            findings=[],  # LLM is summary-only; findings from _raw_metrics_fallback
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
        # Rate calculation priority:
        #   1. oom_events_1h (WOOP events API — authoritative, already time-windowed)
        #   2. delta / snapshot_interval_hours (cross-snapshot comparison from state.py)
        #   3. restart_count / pod_age_hours (cold-start fallback, first snapshot only)
        #
        # Recency gate: regardless of computed rate, skip the pod if
        # last_oomkill_time is >1h ago — the problem has stopped.
        critical_oom = []
        now_utc = datetime.now(timezone.utc)
        for o in snapshot.oomkilled_pods:
            # ── Recency gate ──────────────────────────────────────
            last_oom_str = o.get("last_oomkill_time", "")
            if last_oom_str:
                try:
                    last_oom = datetime.fromisoformat(last_oom_str.replace("Z", "+00:00"))
                    age_hours = (now_utc - last_oom).total_seconds() / 3600.0
                    if age_hours > 1.0:
                        o["_oom_rate_1h"] = 0.0
                        o["_oom_skipped"] = "stale"
                        continue  # last OOM was >1h ago — not active
                except (ValueError, TypeError):
                    pass  # unparseable → don't gate, proceed to rate calc

            # ── Rate calculation ──────────────────────────────────
            oom_events = o.get("oom_events_1h")
            if oom_events:
                # Priority 1: WOOP events API (authoritative)
                oom_rate = oom_events
            else:
                delta = o.get("restart_count", 0)
                interval = o.get("snapshot_interval_hours")
                if interval is not None and interval > 0:
                    # Priority 2: cross-snapshot delta / interval
                    oom_rate = delta / interval
                else:
                    # Priority 3: cold-start fallback (first snapshot, no previous).
                    # Lifetime average is only reliable for young pods (< 1h).
                    # For older pods, skip — accurate delta will be available
                    # on the next run (5 min later).
                    pod_age_hours = self._pod_age_hours(o.get("pod_created_at"))
                    if pod_age_hours is not None and pod_age_hours <= 1.0:
                        oom_rate = delta / max(pod_age_hours, 0.1)
                    else:
                        # Old pod on cold start — can't trust lifetime average
                        o["_oom_rate_1h"] = 0.0
                        o["_oom_skipped"] = "cold_start_old_pod"
                        continue

            # Minimum count gate: a single OOM restart is not a "spiral".
            # Require at least 2 actual restarts before alerting, regardless
            # of extrapolated rate.
            raw_count = o.get("restart_count", 0)
            if raw_count < 2:
                o["_oom_rate_1h"] = round(oom_rate, 1)
                o["_oom_skipped"] = "single_event"
                continue

            o["_oom_rate_1h"] = round(oom_rate, 1)
            if oom_rate >= t.oomkill_critical_per_hour:
                critical_oom.append(o)
        if critical_oom:
            _escalate(Severity.CRITICAL)
            for oom_pod in critical_oom:
                ns = oom_pod.get("namespace", "")
                name = oom_pod.get("name", "unknown")
                wl = f"{ns}/{name}" if ns else name
                oom_rate = oom_pod.get("_oom_rate_1h", 0)
                container = oom_pod.get("container", "")
                mem_limit = oom_pod.get("mem_limit", "")
                limit_info = f" (limit {mem_limit})" if mem_limit else ""
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.OOMKILL,
                    workload=wl,
                    what=f"OOMKilled ~{oom_rate}/h — container {container or 'unknown'}{limit_info}",
                    evidence=json.dumps(oom_pod),
                    suggested_action="Check memory limits and WOOP recommendation for this workload",
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
                    evidence=json.dumps({"oomkill_trend": trend, "recent": recent, "is_trend": True}),
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
                    evidence=json.dumps({"total_mature_pending": len(m_pending)}),
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
                    workload=f"castai-agent/{agent.get('name', 'unknown')}",
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
                    evidence=json.dumps({"total_mature_data_gaps": len(mgaps)}),
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
                    evidence=json.dumps({"total_fresh_data_gaps": len(snapshot.data_gaps)}),
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
                evidence=json.dumps(leak),
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
                workload="castai-agent",
                what=f"CAST AI agent(s) restarted {agent_restarts_last_hour} times in the last hour",
                evidence=json.dumps({"agent_restarts_delta": agent_restarts_last_hour}),
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
                    workload=f"castai-agent/{agent.get('name', 'unknown')}",
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
                    workload="castai-agent/workload-autoscaler",
                    what=f"{sig.get('count', 0)} webhook/exporter error(s) in last 15 min",
                    evidence=json.dumps(sig.get("sample", [])[:2]),
                    suggested_action="Check WA webhook and exporter health; recommendations may not be applied",
                ))
            elif sig.get("signal") == "wa_mutation_errors":
                _escalate(Severity.WARNING)
                findings.append(Finding(
                    severity=Severity.WARNING,
                    category=FindingCategory.WEBHOOK_FAILURE,
                    workload="castai-agent/workload-autoscaler",
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
                evidence=json.dumps({"node_count": snapshot.node_count, "node_count_delta_pct": round(node_count_delta_pct, 1), "total_pods": snapshot.total_pods, "pod_count_delta_pct": round(pod_count_delta_pct, 1)}),
                suggested_action="Check for runaway autoscaler or cascading scaling event",
            ))
        if pod_count_delta_pct > t.node_count_spike_pct:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.CASCADING_SCALING,
                workload="cluster-level",
                what=f"Pod count spiked {pod_count_delta_pct:+.0f}% vs. trailing average",
                evidence=json.dumps({"node_count": snapshot.node_count, "node_count_delta_pct": round(node_count_delta_pct, 1), "total_pods": snapshot.total_pods, "pod_count_delta_pct": round(pod_count_delta_pct, 1)}),
                suggested_action="Check for unexpected deployment scaling or pod storm",
            ))

        # ── CrashLoop check (per-workload detail) ─────────────────────
        if snapshot.crashloop_pods_detail:
            _escalate(Severity.CRITICAL)
            for cl in snapshot.crashloop_pods_detail[:10]:
                findings.append(Finding(
                    severity=Severity.CRITICAL,
                    category=FindingCategory.CRASHLOOP,
                    workload=f"{cl['namespace']}/{cl['name']}",
                    what=f"Container `{cl.get('container', '?')}` in CrashLoopBackOff (restarts: {cl.get('restart_count', 0)})",
                    evidence=json.dumps(cl),
                    suggested_action="Check pod logs and events — all replicas may be down",
                ))
            remaining = snapshot.crashloop_pods - len(snapshot.crashloop_pods_detail[:10])
            if remaining > 0:
                findings.append(Finding(
                    severity=Severity.INFO,
                    category=FindingCategory.CRASHLOOP,
                    workload="cluster-level",
                    what=f"+{remaining} more pod(s) in CrashLoopBackOff",
                    evidence=json.dumps({"crashloop_pods": snapshot.crashloop_pods}),
                    suggested_action="Review all CrashLooping pods in console",
                ))
        elif snapshot.crashloop_pods > 0:
            # Fallback if detail not available
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.CRASHLOOP,
                workload="cluster-level",
                what=f"{snapshot.crashloop_pods} pod(s) in CrashLoopBackOff",
                evidence=json.dumps({"crashloop_pods": snapshot.crashloop_pods}),
                suggested_action="Investigate CrashLooping pods — all replicas may be down",
            ))

        # ── Stale agent heartbeat (from API path) ─────────────────────
        # Only emit if the phase check above didn't already flag this agent.
        # If agent_phase is "StaleHeartbeat (Xm)" the phase check already fires
        # a CRITICAL finding — this block adds detail only when agent shows
        # "online" but heartbeat is actually stale (edge case).
        flagged_agents = {f.workload for f in findings if f.category == FindingCategory.AGENT}
        for sig in snapshot.log_signals:
            if sig.get("signal") == "agent_stale_heartbeat":
                if "castai-agent/castai-agent" in flagged_agents:
                    continue  # already flagged by phase check
                stale_min = sig.get("stale_minutes", 0)
                sev = Severity.CRITICAL if stale_min > 30 else Severity.WARNING
                _escalate(sev)
                findings.append(Finding(
                    severity=sev,
                    category=FindingCategory.AGENT,
                    workload="castai-agent/castai-agent",
                    what=f"Agent heartbeat stale for {stale_min:.0f} minutes",
                    evidence=json.dumps({"last_heartbeat": sig.get("last_heartbeat"), "stale_minutes": stale_min}),
                    suggested_action="Agent may be down or stuck — check castai-agent pod logs",
                ))

        # ── STARTUP_FAILURE events (from API path) ───────────────────
        for sig in snapshot.log_signals:
            if sig.get("signal") == "startup_failures":
                for wl_info in sig.get("workloads", [])[:5]:
                    wl_key = wl_info.get("workload", "unknown")
                    count = wl_info.get("count", 0)
                    _escalate(Severity.WARNING)
                    findings.append(Finding(
                        severity=Severity.WARNING,
                        category=FindingCategory.UNHEALTHY_DEPLOYMENT,
                        workload=wl_key,
                        what=f"{count} STARTUP_FAILURE event(s) in last hour",
                        evidence=json.dumps(wl_info),
                        suggested_action="Container failing to start — check image, resources, and liveness probes",
                    ))

        # ── WOOP workload errors (webhook/controller failures) ───────
        for sig in snapshot.log_signals:
            if sig.get("signal") == "woop_workload_errors":
                for wl_err in sig.get("workloads", [])[:5]:
                    wl_key = wl_err.get("workload", "unknown")
                    err_msg = wl_err.get("error", "unknown")
                    err_lower = err_msg.lower()

                    # Classify error: recommendation timeout vs webhook vs generic
                    if "timed out" in err_lower and "recommendation" in err_lower:
                        sev = Severity.WARNING
                        what = f"Recommendation creation timed out — cluster-controller may be overloaded"
                        action = (
                            "Check cluster-controller pod health and resource usage; "
                            "recommendation will retry on next cycle"
                        )
                    elif "webhook" in err_lower or "admission" in err_lower:
                        sev = Severity.CRITICAL
                        what = f"Admission webhook error: {err_msg[:80]}"
                        action = "Check WA admission webhook health; pods may not get correct resources on restart"
                    elif "cluster-controller" in err_lower:
                        sev = Severity.CRITICAL
                        what = f"Cluster-controller error: {err_msg[:80]}"
                        action = "Check cluster-controller pod health and logs"
                    else:
                        sev = Severity.WARNING
                        what = f"WOOP reports error: {err_msg[:100]}"
                        action = "Check cluster-controller and admission webhook health"

                    _escalate(sev)
                    findings.append(Finding(
                        severity=sev,
                        category=FindingCategory.WEBHOOK_FAILURE,
                        workload=wl_key,
                        what=what,
                        evidence=json.dumps(wl_err),
                        suggested_action=action,
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

    @staticmethod
    def _pod_age_hours(created_at: str | None) -> float | None:
        """Return pod age in hours from an ISO-8601 creationTimestamp.

        Returns None if the timestamp is missing or unparseable.
        """
        if not created_at:
            return None
        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - created
            return max(age.total_seconds() / 3600.0, 0.01)
        except (ValueError, TypeError):
            return None

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
