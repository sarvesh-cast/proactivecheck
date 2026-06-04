# Cluster Watchdog

Proactive monitoring agent for CAST AI-managed Kubernetes clusters. Runs every 5 minutes, collects cluster state via the CAST AI MCP server, evaluates anomalies with Claude LLM, and posts actionable findings to Slack.

Silence means healthy — notifications only fire when something needs attention.

## Architecture

Each run executes three phases:

```
Collect → Evaluate → Notify
```

**Collect** — Calls the CAST AI MCP server (`analyze_snapshot_with_code`, `woop_get_workloads`, `woop_get_oom_summary`, `woop_get_workload_resource_ratios`, `loki_query`) to gather pod health, node state, WOOP recommendations, agent status, and log signals. No kubectl or kubeconfig required — the MCP handles auth via CAST AI API token.

**Evaluate** — Feeds the snapshot (current + trailing 1-hour window of 12 snapshots) to Claude as a single-shot prompt. The model acts as an SRE, classifies findings by severity, and returns a structured verdict: `HEALTHY`, `WARNING`, or `CRITICAL`. A raw-metrics fallback path covers all scenarios without an LLM call.

**Notify** — Posts formatted Slack messages for `WARNING`/`CRITICAL` verdicts with affected workloads, evidence, and suggested actions. Deduplicates within a 30-minute window. Daily summary at 08:00 UTC regardless of status.

## Detection Scenarios

| # | Scenario | Signal |
|---|----------|--------|
| 1 | OOMKill spiral | Restart count increasing, OOMKilled exit code (time-filtered, WOOP-authoritative) |
| 2 | Recommendation/actual mismatch | WOOP intended vs. applied differs >50% |
| 3 | Unschedulable workloads | Pods in Pending with insufficient resource events |
| 4 | Agent down | CAST AI agent pod not Running or heartbeat >10 min stale |
| 5 | Data gap / no recommendation | Optimization enabled but no recommendation for >2 hours |
| 6 | Memory leak | Memory usage monotonically increasing across 3+ snapshots (>5% growth) |
| 7 | Absurd recommendation | WOOP recommends >100 GiB memory or >100 CPU cores |
| 8 | Agent restart loop | Agent restartCount >3/hour or ExitCode:0 with >5 restarts |
| 9 | Webhook / exporter failure | WA admission webhook non-functional, exporter metric limit hit |
| 10 | Cascading scaling | Node or pod count spikes >50% in 30 min without matching deployment |

## Project Structure

```
watchdog/
├── __main__.py      # Entry point (python -m watchdog)
├── main.py          # Orchestrator — CLI parsing, run loop
├── config.py        # All configuration dataclasses and thresholds
├── collector.py     # Phase 1: MCP data collection
├── evaluator.py     # Phase 2: LLM + raw-metrics evaluation
├── notifier.py      # Phase 3: Slack posting with dedup
├── state.py         # Rolling window, trend detection, dedup log
├── models.py        # Data models (SnapshotData, Finding, EvaluationResult)
└── mcp_client.py    # MCP Streamable HTTP transport (JSON-RPC 2.0)
```

## Setup

### Prerequisites

- Python 3.11+
- CAST AI API credentials (JWT token or API key)
- IAP token for MCP server access (`~/.castai/iap_token.json`)
- LLM API key (CASTAI_API_KEY or LLM_API_KEY)
- Slack incoming webhook URL

### Install dependencies

```bash
pip install httpx openai
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CASTAI_JWT_TOKEN` | Yes* | JWT token from `~/.castai/credentials.json` |
| `CASTAI_API_KEY` | Yes* | Alternative: CAST AI API key |
| `CASTAI_ORG_ID` | Yes | Organization UUID |
| `CASTAI_MCP_URL` | Yes | MCP endpoint, e.g. `https://castai-mcp.prod-master.cast.ai/mcp` |
| `CASTAI_IAP_TOKEN` | Yes | IAP cookie from `~/.castai/iap_token.json` |
| `LLM_API_KEY` | No | LLM API key (falls back to `CASTAI_API_KEY`) |
| `LLM_BASE_URL` | No | OpenAI-compatible endpoint (default: `https://llm.kimchi.dev/openai/v1`) |
| `SLACK_WEBHOOK_URL` | Yes | Slack incoming webhook |
| `WATCHDOG_CLUSTER_ID` | No | Target cluster UUID (default: Grip prod-us-4) |
| `WATCHDOG_MODEL` | No | LLM model (default: `kimi-k2.6`) |
| `WATCHDOG_DRY_RUN` | No | `true` to evaluate without posting to Slack |
| `WATCHDOG_STATE_FILE` | No | State file path (default: `watchdog_state.json`) |
| `WATCHDOG_LOG_LEVEL` | No | `DEBUG` / `INFO` / `WARNING` |
| `SLACK_CHANNEL` | No | Slack channel (default: `#castai_grip_security_ext`) |
| `CASTAI_API_URL` | No | API base URL (default: `https://api.cast.ai`) |

\* One of `CASTAI_JWT_TOKEN` or `CASTAI_API_KEY` is required.

## Usage

```bash
# Continuous loop (every 5 minutes)
python -m watchdog

# Single run
python -m watchdog --once

# Dry-run (evaluate but don't post to Slack)
WATCHDOG_DRY_RUN=true python -m watchdog --once

# Collector only — fetch snapshot, print JSON
python -m watchdog --collector-only

# Evaluator only — run LLM against last snapshot in state
python -m watchdog --evaluator-only

# Evaluator with raw-metrics fallback (no LLM)
python -m watchdog --evaluator-only --raw-fallback

# Notifier only — format + post last evaluation
python -m watchdog --notifier-only

# State dump — print rolling window and dedup log
python -m watchdog --state-dump

# Collect + evaluate, skip notify
python -m watchdog --skip-notify --once

# Offline testing with a fixture file
python -m watchdog --evaluator-only --fixture snapshot_fixture.json
```

## State Management

The watchdog maintains a rolling window of the last 12 snapshots (1 hour) in `watchdog_state.json`. This enables trend detection across runs:

- **OOMKill escalation** — counts across the window detect spiraling restarts
- **Memory leaks** — monotonically increasing memory across 3+ snapshots with >5% growth
- **Cascading scaling** — node/pod count delta vs. trailing average
- **Agent restart rate** — restartCount diffs between first and last snapshot in window
- **Data gap duration** — tracked separately in `data_gap_first_seen`; only escalated to WARNING after 2+ hours

State file corruption is handled gracefully (reset to empty, log warning). Dedup entries expire after 24 hours.

## How the Evaluator Works

The snapshot is compacted before sending to Claude — bulk WOOP workload data (can exceed 10M chars) is stripped to pre-computed findings (~8K chars). The prompt provides cluster baseline context (known scale-to-zero workloads, expected resource ranges, CNI quirks) so the model distinguishes normal behavior from real anomalies.

If the LLM call fails, the raw-metrics fallback covers all 10 scenarios using threshold-based rules defined in `config.py`. Use `--raw-fallback` to force this path.

### Verdicts

- **HEALTHY** — No actionable findings. Nothing posted to Slack.
- **WARNING** — Non-urgent issues detected (e.g. data gap >2h, memory trend, 1-3 agent restarts).
- **CRITICAL** — Immediate attention needed (e.g. OOMKill spiral, absurd recommendation, agent down).

## MCP Transport

The watchdog communicates with CAST AI via Streamable HTTP (JSON-RPC 2.0). Authentication uses JWT + IAP cookie headers. The MCP client in `mcp_client.py` handles session initialization, tool discovery, and tool invocation. Some MCP tools reject the `organization_id` parameter — the client selectively omits it for those tools.
