"""Configuration and constants for the Grip Security Cluster Watchdog."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CastAIConfig:
    """CAST AI API connection settings."""

    api_url: str = os.getenv("CASTAI_API_URL", "https://api.cast.ai")
    api_key: str = os.getenv("CASTAI_API_KEY", "")
    jwt_token: str = os.getenv("CASTAI_JWT_TOKEN", "")
    organization_id: str = os.getenv("CASTAI_ORG_ID", "")
    iap_token: str = os.getenv("CASTAI_IAP_TOKEN", "")
    mcp_url: str = os.getenv("CASTAI_MCP_URL", "")  # e.g. https://castai-mcp.prod-master.cast.ai/mcp
    request_timeout: int = 30  # seconds per API call

    @property
    def auth_headers(self) -> dict[str, str]:
        if self.jwt_token:
            return {"Authorization": f"Bearer {self.jwt_token}"}
        if self.api_key:
            return {"X-API-Key": self.api_key}
        raise RuntimeError("No CAST AI credentials configured (set CASTAI_API_KEY or CASTAI_JWT_TOKEN)")


@dataclass(frozen=True)
class AnthropicConfig:
    """Anthropic API settings for the evaluator."""

    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv("WATCHDOG_MODEL", "claude-haiku-4-5")
    fallback_model: str = "claude-sonnet-4-5"
    max_tokens: int = 4096
    temperature: float = 0.0  # deterministic for consistency


@dataclass(frozen=True)
class SlackConfig:
    """Slack notification settings."""

    webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")
    channel: str = os.getenv("SLACK_CHANNEL", "#castai_grip_security_ext")
    dedup_window_minutes: int = 30
    daily_summary_hour_utc: int = 8


@dataclass(frozen=True)
class ClusterContext:
    """Grip Security cluster-specific baseline context.

    This is the critical section for false-positive reduction.
    Must be tuned during the 24-hour dry-run period.
    """

    cluster_id: str = os.getenv(
        "WATCHDOG_CLUSTER_ID", "e9f502e2-46dd-49fb-9310-149d7d8ad0ba"
    )
    cluster_name: str = "Grip Security prod-us-4"
    namespaces: list[str] = field(
        default_factory=lambda: [
            "cengage", "williamsmullen", "oscarhealth",
            "ensemblehealthpartners", "athenahealth",
        ]
    )
    # Known baseline behaviors (DO NOT flag these as anomalies)
    known_scale_to_zero_workloads: list[str] = field(
        default_factory=lambda: ["aggregator"]
    )
    discovery_puller_memory_range_mb: tuple[int, int] = (800, 1500)
    woop_memory_limit_multiplier: float = 1.5
    woop_recommendation_strategy: str = "Max Usage"
    cni: str = "cilium"  # transient agent-not-ready taints are normal
    uses_keda: bool = True  # rapid replica changes are expected


@dataclass(frozen=True)
class Thresholds:
    """Detection thresholds — maps to the classification rules in the design doc."""

    # CRITICAL thresholds
    oomkill_critical_per_hour: int = 3
    recommendation_mismatch_pct: float = 50.0
    absurd_memory_gib: float = 100.0
    absurd_cpu_cores: int = 100
    pending_pod_minutes: int = 15
    agent_restart_critical_per_hour: int = 3
    agent_restart_count_critical: int = 5
    node_count_spike_pct: float = 50.0

    # WARNING thresholds
    memory_leak_snapshots: int = 3  # consecutive upward trend
    memory_usage_over_request_ratio: float = 2.0
    data_gap_hours: float = 2.0
    node_memory_utilization_pct: float = 90.0
    agent_restart_warning_per_hour: int = 1
    workload_autoscaler_replicas_expected: int = 2

    # Outlier detection (Max Usage strategy vulnerability)
    outlier_median_ratio: float = 10.0


@dataclass
class WatchdogConfig:
    """Top-level configuration combining all sub-configs."""

    castai: CastAIConfig = field(default_factory=CastAIConfig)
    anthropic: AnthropicConfig = field(default_factory=AnthropicConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)
    cluster: ClusterContext = field(default_factory=ClusterContext)
    thresholds: Thresholds = field(default_factory=Thresholds)

    run_interval_seconds: int = 300  # 5 minutes
    rolling_window_size: int = 12  # 12 snapshots = 1 hour
    state_file: str = os.getenv("WATCHDOG_STATE_FILE", "watchdog_state.json")
    dry_run: bool = os.getenv("WATCHDOG_DRY_RUN", "false").lower() == "true"
    log_level: str = os.getenv("WATCHDOG_LOG_LEVEL", "INFO")
