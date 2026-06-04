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
                        validated = self._validate_llm_result(result, raw_result, snapshot)
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
                # Must match an actual OOMKilled pod
                keep = (
                    wl in oom_workloads
                    or wl_name in oom_workloads
                    or any(wl_name in ow for ow in oom_workloads)
                    or len(snapshot.oomkilled_pods) > 0 and wl == "cluster-level"
                )

            elif cat == FindingCategory.MISMATCH:
                keep = wl in mismatch_workloads or any(wl_name in m for m in mismatch_workloads)

            elif cat == FindingCategory.ABSURD_RECOMMENDATION:
                keep = wl in absurd_workloads or any(wl_name in a for a in absurd_workloads)

            elif cat == FindingCategory.UNSCHEDULABLE:
                keep = snapshot.pending_pods > 0

            elif cat in (FindingCategory.AGENT, FindingCategory.AGENT_RESTART):
                keep = (
                    agent_not_running
                    or snapshot.agent_restarts_last_hour > 0
                    or any(wl_name in an for an in agent_names)
                    or wl in ("cluster-level", "castai-system", "org-level")
                )

            elif cat == FindingCategory.DATA_GAP:
                keep = len(snapshot.data_gaps) > 0 or wl in data_gap_workloads

            elif cat == FindingCategory.MEMORY_LEAK:
                # Trust the LLM if we have memory usage data at all
                keep = len(snapshot.workload_memory_usage) > 0

            elif cat == FindingCategory.CASCADING_SCALING:
                keep = True  # Validated by node/pod delta in raw metrics

            elif cat == FindingCategory.WEBHOOK_FAILURE:
                keep = any(
                    s.get("signal") in ("webhook_exporter_failure", "wa_mutation_errors")
                    for s in snapshot.log_signals
                )

            elif cat in (FindingCategory.UNHEALTHY_DEPLOYMENT, FindingCategory.CONFIG, FindingCategory.OTHER):
                # Generic categories — keep if any log signal or deployment issue exists
                keep = True

            if keep:
                validated.append(finding)
            else:
                dropped += 1
                logger.warning(
                    "Dropped hallucinated finding: [%s] %s on %s — no supporting data in snapshot",
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
            summary = f"[Validated: {dropped} hallucinated finding(s) removed] {summary}"

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
        if node_count_delta_pct > 50:
            _escalate(Severity.CRITICAL)
            findings.append(Finding(
                severity=Severity.CRITICAL,
                category=FindingCategory.CASCADING_SCALING,
                workload="cluster-level",
                what=f"Node count spiked {node_count_delta_pct:+.0f}% vs. trailing average",
                evidence=f"node_count={snapshot.node_count}, delta_pct={node_count_delta_pct:.1f}%",
                suggested_action="Check for runaway autoscaler or cascading scaling event",
            ))
        if pod_count_delta_pct > 50:
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
                _escalate(Severity.WARNING)
                for dep in sig.get("sample", [])[:5]:
                    ns = dep.get("namespace", "")
                    name = dep.get("name", "")
                    desired = dep.get("desired", 0)
                    findings.append(Finding(
                        severity=Severity.WARNING,
                        category=FindingCategory.UNHEALTHY_DEPLOYMENT,
                        workload=f"{ns}/{name}",
                        what=f"Deployment has desired={desired} but ready=0, available=0",
                        evidence=json.dumps(dep),
                        suggested_action="Check pod events and logs — may be CrashLoopBackOff or stuck Pending",
                    ))
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
