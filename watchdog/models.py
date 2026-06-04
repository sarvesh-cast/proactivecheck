"""Data models for the watchdog pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ── Verdict & Category enums ──────────────────────────────────────────

class Verdict(str, Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class FindingCategory(str, Enum):
    OOMKILL = "oomkill"
    MISMATCH = "mismatch"
    UNSCHEDULABLE = "unschedulable"
    AGENT = "agent"
    DATA_GAP = "data_gap"
    MEMORY_LEAK = "memory_leak"
    ABSURD_RECOMMENDATION = "absurd_recommendation"
    AGENT_RESTART = "agent_restart"
    WEBHOOK_FAILURE = "webhook_failure"
    CASCADING_SCALING = "cascading_scaling"
    UNHEALTHY_DEPLOYMENT = "unhealthy_deployment"
    CONFIG = "config"
    OTHER = "other"


class Severity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


# ── Collector output ──────────────────────────────────────────────────

@dataclass
class SnapshotData:
    """Standardized output from the collector module."""

    timestamp: str  # UTC ISO 8601
    cluster_id: str

    # Pod health
    total_pods: int = 0
    running_pods: int = 0
    pending_pods: int = 0
    pending_pods_detail: list[dict] = field(default_factory=list)
    # [{namespace, name, reason}]
    crashloop_pods: int = 0
    crashloop_pods_detail: list[dict] = field(default_factory=list)
    # [{namespace, name, container, restart_count}]
    oomkilled_pods: list[dict] = field(default_factory=list)
    # [{namespace, name, restart_count, container, last_oomkill_time}]

    # Node health
    node_count: int = 0
    nodes: list[dict] = field(default_factory=list)
    # [{name, capacity_cpu, capacity_mem, allocatable_cpu, allocatable_mem, conditions}]

    # WOOP recommendations
    woop_workloads: list[dict] = field(default_factory=list)
    recommendation_mismatches: list[dict] = field(default_factory=list)
    absurd_recommendations: list[dict] = field(default_factory=list)
    data_gaps: list[dict] = field(default_factory=list)

    # CAST AI agent health
    agent_pods: list[dict] = field(default_factory=list)
    agent_restarts_last_hour: int = 0

    # Per-workload memory usage (top consumers, for leak detection)
    workload_memory_usage: list[dict] = field(default_factory=list)
    # [{namespace, workload, container, usage_bytes, request_bytes, limit_bytes}]

    # Loki log signals
    log_signals: list[dict] = field(default_factory=list)

    # Errors during collection (partial failure tracking)
    collection_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_dict_compact(self) -> dict:
        """Compact dict for state storage — strips bulk fields.

        Drops woop_workloads (can be 10M+ chars) and full node details.
        Keeps only pre-computed findings and fields needed for trend detection.
        """
        d = asdict(self)
        # Drop the mega-list; pre-computed findings are in separate fields
        d.pop("woop_workloads", None)
        # Keep node_count but drop per-node detail (nodes list)
        d.pop("nodes", None)
        # Trim collection_errors to just the count
        errors = d.get("collection_errors", [])
        d["collection_errors"] = [f"({len(errors)} errors)"] if errors else []
        # Cap list sizes for safety
        for key in ("oomkilled_pods", "pending_pods_detail",
                     "crashloop_pods_detail", "recommendation_mismatches",
                     "absurd_recommendations", "data_gaps",
                     "workload_memory_usage", "log_signals"):
            if key in d and len(d[key]) > 30:
                d[key] = d[key][:30]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ── Evaluator output ──────────────────────────────────────────────────

@dataclass
class Finding:
    """A single anomaly detected by the evaluator."""

    severity: Severity
    category: FindingCategory
    workload: str  # "namespace/workload" or "cluster-level"
    what: str
    evidence: str
    suggested_action: str


@dataclass
class EvaluationMetrics:
    """Aggregate metrics from the evaluation."""

    total_pods: int = 0
    oomkilled_pods: int = 0
    pending_pods: int = 0
    crashloop_pods: int = 0
    oomkill_rate_per_hour: float = 0.0
    recommendation_mismatches: int = 0
    absurd_recommendations: int = 0
    agent_restarts_last_hour: int = 0
    node_count: int = 0
    node_count_delta_pct: float = 0.0


@dataclass
class EvaluationResult:
    """Complete output from the evaluator module."""

    verdict: Verdict
    summary: str
    findings: list[Finding] = field(default_factory=list)
    metrics: EvaluationMetrics = field(default_factory=EvaluationMetrics)
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    model_used: str = ""
    raw_response: str = ""  # for debugging

    def has_actionable_findings(self) -> bool:
        return self.verdict in (Verdict.WARNING, Verdict.CRITICAL)

    def critical_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == Severity.CRITICAL]

    def to_dict(self) -> dict:
        return asdict(self)


# ── Deduplication key ─────────────────────────────────────────────────

@dataclass(frozen=True)
class DedupKey:
    """Uniquely identifies a finding for deduplication."""

    category: str
    workload: str

    @classmethod
    def from_finding(cls, f: Finding) -> DedupKey:
        return cls(category=f.category.value, workload=f.workload)
