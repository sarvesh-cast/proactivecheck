"""Rolling state manager — persists snapshot history for trend detection.

Stores the last N snapshots (default 12 = 1 hour) in a local JSON file.
Provides helpers for detecting trends (memory leaks, escalating OOMKills,
node count changes) across the rolling window.

Handles corruption gracefully: if the state file is unreadable, it resets
to empty and logs a warning rather than crashing.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import DedupKey

logger = logging.getLogger("watchdog.state")


class StateManager:
    """Manages rolling snapshot window and dedup state."""

    def __init__(self, state_file: str, window_size: int = 12) -> None:
        self.state_file = Path(state_file)
        self.window_size = window_size
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load state from disk. Reset on corruption."""
        if not self.state_file.exists():
            return self._empty_state()

        try:
            data = json.loads(self.state_file.read_text())
            # Basic shape validation
            if not isinstance(data, dict) or "snapshots" not in data:
                raise ValueError("Invalid state shape")
            # Backfill keys added after initial release
            data.setdefault("dedup_log", {})
            data.setdefault("data_gap_first_seen", {})
            data.setdefault("pending_pod_first_seen", {})
            data.setdefault("last_daily_summary_date", "")
            return data
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("State file corrupted (%s), resetting: %s", self.state_file, e)
            return self._empty_state()

    def _empty_state(self) -> dict[str, Any]:
        return {
            "snapshots": [],
            "dedup_log": {},  # {dedup_key_str: last_posted_iso}
            "data_gap_first_seen": {},  # {"ns/workload": iso_timestamp}
            "pending_pod_first_seen": {},  # {"ns/pod_name": iso_timestamp}
            "last_daily_summary_date": "",  # "YYYY-MM-DD" — prevents double-posting
        }

    def save(self) -> None:
        """Persist state to disk. Atomic write via tmp + rename."""
        tmp = self.state_file.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._state, indent=2, default=str))
            tmp.replace(self.state_file)
        except OSError as e:
            logger.error("Failed to save state: %s", e)

    # ── Snapshot window ───────────────────────────────────────────────

    def push_snapshot(self, snapshot_data: dict) -> None:
        """Add a snapshot to the rolling window, evicting the oldest if full."""
        self._state["snapshots"].append(snapshot_data)
        # Trim to window size
        if len(self._state["snapshots"]) > self.window_size:
            self._state["snapshots"] = self._state["snapshots"][-self.window_size:]

    def get_snapshots(self) -> list[dict]:
        """Return all snapshots in the rolling window (oldest first)."""
        return list(self._state["snapshots"])

    def get_previous_snapshot(self) -> dict | None:
        """Return the most recent snapshot before the current one."""
        snaps = self._state["snapshots"]
        return snaps[-2] if len(snaps) >= 2 else None

    @property
    def snapshot_count(self) -> int:
        return len(self._state["snapshots"])

    # ── Trend detection helpers ───────────────────────────────────────

    def get_oomkill_trend(self) -> list[int]:
        """Return OOMKill counts across the rolling window."""
        return [
            len(s.get("oomkilled_pods", []))
            for s in self._state["snapshots"]
        ]

    def get_node_count_trend(self) -> list[int]:
        """Return node counts across the rolling window."""
        return [s.get("node_count", 0) for s in self._state["snapshots"]]

    def get_workload_memory_trend(self, namespace: str, workload: str) -> list[float]:
        """Return memory (MiB) for a specific workload across snapshots.

        Prefers usage_mib (actual usage from pod metrics, hybrid tier) over
        request_mem_mib (WOOP resource ratios, MCP/API tiers).
        If monotonically increasing across 3+ snapshots, it's a potential leak.
        """
        values = []
        for snap in self._state["snapshots"]:
            found = False
            for entry in snap.get("workload_memory_usage", []):
                if (entry.get("namespace") == namespace
                        and entry.get("workload") == workload):
                    mib = entry.get("usage_mib") or entry.get("request_mem_mib", 0)
                    values.append(mib)
                    found = True
                    break
            if not found:
                values.append(0)
        return values

    def get_agent_restart_trend(self) -> list[dict]:
        """Return per-agent-pod restart counts across the rolling window.

        Returns [{name: str, restarts: [int, ...]}] — one entry per agent.
        The delta between consecutive values gives restarts-in-that-interval.
        """
        agent_history: dict[str, list[int]] = {}
        for snap in self._state["snapshots"]:
            seen_this_snap = set()
            for agent in snap.get("agent_pods", []):
                name = agent.get("name", "")
                rc = agent.get("restart_count", 0)
                if name not in agent_history:
                    agent_history[name] = []
                agent_history[name].append(rc)
                seen_this_snap.add(name)
            # Fill gaps for agents not seen in this snapshot
            for name in agent_history:
                if name not in seen_this_snap:
                    agent_history[name].append(0)
        return [
            {"name": name, "restarts": counts}
            for name, counts in agent_history.items()
        ]

    def compute_agent_restarts_last_hour(self) -> int:
        """Compute total agent restarts in the rolling window by diffing restartCount."""
        total_delta = 0
        for agent in self.get_agent_restart_trend():
            counts = agent["restarts"]
            if len(counts) >= 2:
                delta = counts[-1] - counts[0]
                if delta > 0:
                    total_delta += delta
        return total_delta

    def detect_memory_leaks(self, min_snapshots: int = 3) -> list[dict]:
        """Detect workloads with monotonically increasing memory across snapshots.

        Returns list of {namespace, workload, trend: [mib, ...]} for leaking workloads.
        """
        if len(self._state["snapshots"]) < min_snapshots:
            return []

        # Collect all workloads seen in the latest snapshot
        latest = self._state["snapshots"][-1]
        leaks = []
        for entry in latest.get("workload_memory_usage", []):
            ns = entry.get("namespace", "")
            wl = entry.get("workload", "")
            trend = self.get_workload_memory_trend(ns, wl)
            recent = trend[-min_snapshots:]
            # Check monotonically increasing with meaningful growth (>5% total)
            if len(recent) >= min_snapshots and all(
                b > a for a, b in zip(recent, recent[1:])
            ):
                if recent[0] > 0:
                    growth_pct = ((recent[-1] - recent[0]) / recent[0]) * 100
                    if growth_pct > 5:
                        leaks.append({
                            "namespace": ns,
                            "workload": wl,
                            "trend_mib": recent,
                            "growth_pct": round(growth_pct, 1),
                        })
        return leaks

    def is_monotonically_increasing(self, values: list[float], min_length: int = 3) -> bool:
        """Check if values are monotonically increasing (potential leak)."""
        if len(values) < min_length:
            return False
        recent = values[-min_length:]
        return all(b > a for a, b in zip(recent, recent[1:]))

    def get_pod_count_trend(self) -> list[int]:
        """Return total pod counts across the rolling window."""
        return [s.get("total_pods", 0) for s in self._state["snapshots"]]

    def compute_node_count_delta_pct(self) -> float:
        """Compute node count change vs. trailing 1-hour average."""
        counts = self.get_node_count_trend()
        if len(counts) < 2:
            return 0.0
        current = counts[-1]
        avg = sum(counts[:-1]) / len(counts[:-1])
        if avg == 0:
            return 0.0
        return ((current - avg) / avg) * 100

    def compute_pod_count_delta_pct(self) -> float:
        """Compute pod count change vs. trailing 1-hour average."""
        counts = self.get_pod_count_trend()
        if len(counts) < 2:
            return 0.0
        current = counts[-1]
        avg = sum(counts[:-1]) / len(counts[:-1])
        if avg == 0:
            return 0.0
        return ((current - avg) / avg) * 100

    # ── Pending pod age tracking ────────────────────────────────────

    def update_pending_pods(self, pending_details: list[dict]) -> None:
        """Track how long each pod has been in Pending state.

        Maintains {ns/pod_name: first_seen_iso}. Pods no longer
        Pending are removed (resolved).
        """
        tracker = self._state.setdefault("pending_pod_first_seen", {})
        now_iso = datetime.now(timezone.utc).isoformat()

        current_keys = set()
        for p in pending_details:
            key = f"{p.get('namespace', '')}/{p.get('name', '')}"
            current_keys.add(key)
            if key not in tracker:
                tracker[key] = now_iso

        # Remove resolved (no longer Pending)
        resolved = [k for k in tracker if k not in current_keys]
        for k in resolved:
            del tracker[k]

    def get_mature_pending_pods(self, min_minutes: float = 15.0) -> list[dict]:
        """Return Pending pods that have been Pending for >= min_minutes.

        Each entry gets 'age_minutes' and 'first_seen' fields.
        """
        tracker = self._state.get("pending_pod_first_seen", {})
        now = datetime.now(timezone.utc)

        # Get current pending from latest snapshot
        snaps = self._state["snapshots"]
        if not snaps:
            return []
        latest = snaps[-1]
        current_pending = latest.get("pending_pods_detail", [])

        mature = []
        for p in current_pending:
            key = f"{p.get('namespace', '')}/{p.get('name', '')}"
            first_seen = tracker.get(key)
            if not first_seen:
                continue
            try:
                first_time = datetime.fromisoformat(first_seen)
                age_min = (now - first_time).total_seconds() / 60
                if age_min >= min_minutes:
                    mature.append({
                        **p,
                        "age_minutes": round(age_min, 1),
                        "first_seen": first_seen,
                    })
            except (ValueError, TypeError):
                continue
        return mature

    # ── OOMKill delta computation ───────────────────────────────────

    def compute_oomkill_deltas(self, current_oom: list[dict]) -> list[dict]:
        """Compute per-pod OOMKill restart deltas using cross-snapshot comparison.

        For snapshot-sourced entries, `restart_count` is the pod's lifetime
        restartCount — not per-hour. This method compares against the previous
        snapshot to compute actual restarts since last check.

        For WOOP/API-sourced entries (source != "snapshot_lastState"),
        `restart_count` is already time-windowed, so we pass it through unchanged.

        Returns the same list with:
        - `restart_count` replaced by the delta for snapshot-sourced entries
        - `lifetime_restart_count` storing the original value
        - `snapshot_interval_hours` storing the time between current and previous snapshot
        """
        prev = self.get_previous_snapshot()
        if not prev:
            # First snapshot — can't compute delta.
            # For snapshot-sourced entries, keep restart_count as-is (best effort).
            return current_oom

        # Compute interval between current and previous snapshot
        snaps = self._state["snapshots"]
        current_ts_str = snaps[-1].get("timestamp", "")
        prev_ts_str = prev.get("timestamp", "")
        interval_hours = self._compute_interval_hours(current_ts_str, prev_ts_str)

        # Build lookup: pod_key → restart_count from previous snapshot
        prev_restarts: dict[str, int] = {}
        for o in prev.get("oomkilled_pods", []):
            if o.get("source", "").startswith("snapshot_"):
                ns = o.get("namespace", "")
                name = o.get("name", "")
                key = f"{ns}/{name}"
                prev_restarts[key] = o.get("restart_count", 0)

        result = []
        for o in current_oom:
            if not o.get("source", "").startswith("snapshot_"):
                # WOOP/API-sourced — already time-windowed
                result.append(o)
                continue

            ns = o.get("namespace", "")
            name = o.get("name", "")
            key = f"{ns}/{name}"
            lifetime_count = o.get("restart_count", 0)

            if key in prev_restarts:
                # Pod was in previous snapshot — compute genuine delta
                delta = max(0, lifetime_count - prev_restarts[key])
                enriched = {
                    **o,
                    "restart_count": delta,
                    "lifetime_restart_count": lifetime_count,
                    "snapshot_interval_hours": interval_hours,
                }
            else:
                # Pod is new to OOM tracking — can't compute delta.
                # Do NOT set snapshot_interval_hours so evaluator falls
                # through to the cold-start fallback (pod age < 1h gate).
                enriched = {
                    **o,
                    "restart_count": lifetime_count,
                    "lifetime_restart_count": lifetime_count,
                }
            result.append(enriched)

        return result

    @staticmethod
    def _compute_interval_hours(current_ts: str, prev_ts: str) -> float | None:
        """Compute hours between two ISO-8601 timestamps. Returns None on failure."""
        if not current_ts or not prev_ts:
            return None
        try:
            cur = datetime.fromisoformat(current_ts.replace("Z", "+00:00"))
            prv = datetime.fromisoformat(prev_ts.replace("Z", "+00:00"))
            delta = (cur - prv).total_seconds() / 3600.0
            return max(delta, 0.01)  # floor at ~36 seconds
        except (ValueError, TypeError):
            return None

    # ── Data gap duration tracking ────────────────────────────────────

    def update_data_gaps(self, current_gaps: list[dict]) -> list[dict]:
        """Track data gap durations and return only those exceeding the threshold.

        Maintains a dict of {workload_key: first_seen_iso} in state.
        Workloads no longer in the gap list are removed (gap resolved).
        Returns gaps that have persisted for >= min_hours.
        """
        gap_tracker = self._state.setdefault("data_gap_first_seen", {})
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()

        # Current gap keys
        current_keys = set()
        for g in current_gaps:
            key = g.get("workload", "")
            current_keys.add(key)
            if key not in gap_tracker:
                gap_tracker[key] = now_iso

        # Remove resolved gaps
        resolved = [k for k in gap_tracker if k not in current_keys]
        for k in resolved:
            del gap_tracker[k]

        return current_gaps  # filtering by age is done by get_mature_data_gaps

    def get_mature_data_gaps(self, min_hours: float = 2.0) -> list[dict]:
        """Return data gaps that have persisted for at least min_hours.

        Each entry gets an 'age_hours' field added.
        """
        gap_tracker = self._state.get("data_gap_first_seen", {})
        now = datetime.now(timezone.utc)
        mature = []

        # Get current gaps from latest snapshot
        snaps = self._state["snapshots"]
        if not snaps:
            return []
        latest = snaps[-1]
        current_gaps = latest.get("data_gaps", [])

        for g in current_gaps:
            key = g.get("workload", "")
            first_seen = gap_tracker.get(key)
            if not first_seen:
                continue
            try:
                first_time = datetime.fromisoformat(first_seen)
                age_hours = (now - first_time).total_seconds() / 3600
                if age_hours >= min_hours:
                    mature.append({
                        **g,
                        "age_hours": round(age_hours, 1),
                        "first_seen": first_seen,
                    })
            except (ValueError, TypeError):
                continue

        return mature

    # ── Deduplication ─────────────────────────────────────────────────

    def should_notify(self, key: DedupKey, dedup_window_minutes: int) -> bool:
        """Check if this finding should trigger a notification.

        Returns False if the same category + workload was posted within
        the dedup window.
        """
        key_str = f"{key.category}:{key.workload}"
        last_posted = self._state["dedup_log"].get(key_str)

        if not last_posted:
            return True

        try:
            last_time = datetime.fromisoformat(last_posted)
            now = datetime.now(timezone.utc)
            elapsed = (now - last_time).total_seconds() / 60
            return elapsed >= dedup_window_minutes
        except (ValueError, TypeError):
            return True  # corrupt timestamp → allow notification

    def record_notification(self, key: DedupKey) -> None:
        """Record that a notification was sent for this finding."""
        key_str = f"{key.category}:{key.workload}"
        self._state["dedup_log"][key_str] = datetime.now(timezone.utc).isoformat()

    def cleanup_stale_dedup_entries(self, max_age_hours: int = 24) -> None:
        """Remove dedup entries older than max_age_hours."""
        now = datetime.now(timezone.utc)
        cleaned = {}
        for key_str, ts_str in self._state["dedup_log"].items():
            try:
                ts = datetime.fromisoformat(ts_str)
                if (now - ts).total_seconds() < max_age_hours * 3600:
                    cleaned[key_str] = ts_str
            except (ValueError, TypeError):
                pass  # drop corrupt entries
        self._state["dedup_log"] = cleaned

    def should_post_daily_summary(self) -> bool:
        """Return True if no daily summary has been posted today (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._state.get("last_daily_summary_date", "") != today

    def record_daily_summary(self) -> None:
        """Record that the daily summary was posted today."""
        self._state["last_daily_summary_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
