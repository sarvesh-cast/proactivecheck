"""Configuration and constants for the Grip Security Cluster Watchdog."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _load_dotenv() -> None:
    """Load .env file from watchdog/ dir or repo root into os.environ (won't override existing vars)."""
    for candidate in [Path(__file__).parent / ".env", Path.cwd() / ".env"]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if not os.environ.get(key):  # don't override existing
                    os.environ[key] = value
            break  # only load the first .env found


_load_dotenv()


def _load_castai_credential(env_var: str, file_path: str, json_key: str) -> str:
    """Load a credential from env var first, then fall back to ~/.castai/ file."""
    val = os.getenv(env_var, "")
    if val:
        return val
    p = Path.home() / ".castai" / file_path
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return data.get(json_key, "")
        except (json.JSONDecodeError, OSError):
            pass
    return ""


@dataclass(frozen=True)
class CastAIConfig:
    """CAST AI API connection settings.

    JWT and IAP tokens are auto-loaded from ~/.castai/ if env vars are not set:
      - ~/.castai/credentials.json  → "token" field → CASTAI_JWT_TOKEN
      - ~/.castai/iap_token.json    → "token" field → CASTAI_IAP_TOKEN
    """

    api_url: str = field(default_factory=lambda: os.getenv("CASTAI_API_URL", "https://api.cast.ai"))
    api_key: str = field(default_factory=lambda: os.getenv("CASTAI_API_KEY", ""))
    jwt_token: str = field(default_factory=lambda: _load_castai_credential("CASTAI_JWT_TOKEN", "credentials.json", "token"))
    organization_id: str = field(default_factory=lambda: os.getenv("CASTAI_ORG_ID", ""))
    iap_token: str = field(default_factory=lambda: _load_castai_credential("CASTAI_IAP_TOKEN", "iap_token.json", "cookie_value"))
    mcp_url: str = field(default_factory=lambda: os.getenv("CASTAI_MCP_URL", ""))
    request_timeout: int = 30  # seconds per API call

    @property
    def auth_headers(self) -> dict[str, str]:
        if self.jwt_token:
            return {"Authorization": f"Bearer {self.jwt_token}"}
        if self.api_key:
            return {"X-API-Key": self.api_key}
        raise RuntimeError("No CAST AI credentials configured (set CASTAI_API_KEY or CASTAI_JWT_TOKEN)")


@dataclass(frozen=True)
class LLMConfig:
    """LLM API settings for the evaluator.

    Supports any OpenAI-compatible endpoint via LLM_BASE_URL.
    Default: CAST AI internal LLM proxy (https://llm.kimchi.dev/openai/v1).
    Auth: uses CASTAI_API_KEY by default, or LLM_API_KEY if set.
    """

    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "") or os.getenv("CASTAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://llm.kimchi.dev/openai/v1"))
    model: str = field(default_factory=lambda: os.getenv("WATCHDOG_MODEL", "minimax-m2.7"))
    fallback_model: str = field(default_factory=lambda: os.getenv("WATCHDOG_FALLBACK_MODEL", "minimax-m2.7"))
    max_tokens: int = 4096
    temperature: float = 0.0  # deterministic for consistency


@dataclass(frozen=True)
class SlackConfig:
    """Slack notification settings."""

    webhook_url: str = field(default_factory=lambda: os.getenv("SLACK_WEBHOOK_URL", ""))
    channel: str = field(default_factory=lambda: os.getenv("SLACK_CHANNEL", "#castai_grip_security_ext"))
    dedup_window_minutes: int = 30
    daily_summary_hour_utc: int = 8


@dataclass(frozen=True)
class ClusterContext:
    """Grip Security cluster-specific baseline context.

    This is the critical section for false-positive reduction.
    Must be tuned during the 24-hour dry-run period.
    """

    cluster_id: str = field(default_factory=lambda: os.getenv(
        "WATCHDOG_CLUSTER_ID", ""
    ))
    # Comma-separated list of cluster IDs for multi-cluster mode.
    # If set, overrides cluster_id. Use "auto" to discover all clusters from org.
    cluster_ids: list[str] = field(default_factory=lambda: [
        cid.strip() for cid in os.getenv("WATCHDOG_CLUSTER_IDS", "").split(",")
        if cid.strip()
    ])
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
    llm: LLMConfig = field(default_factory=LLMConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)
    cluster: ClusterContext = field(default_factory=ClusterContext)
    thresholds: Thresholds = field(default_factory=Thresholds)

    run_interval_seconds: int = 300  # 5 minutes
    rolling_window_size: int = 12  # 12 snapshots = 1 hour
    state_file: str = field(default_factory=lambda: os.getenv("WATCHDOG_STATE_FILE", "watchdog_state.json"))
    dry_run: bool = field(default_factory=lambda: os.getenv("WATCHDOG_DRY_RUN", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("WATCHDOG_LOG_LEVEL", "INFO"))
