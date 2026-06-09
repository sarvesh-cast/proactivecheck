"""End-to-end tests for state.py — staleness eviction, rolling window, trends.

Tests cover:
  1. Normal rolling window behavior (no gaps)
  2. Staleness eviction after a gap (2h downtime)
  3. Edge cases: empty window, single entry, all stale, none stale
  4. Missing/unparseable timestamps — kept safely
  5. Trend calculations post-eviction (sparse windows)
  6. Memory leak detection with sparse data
  7. OOMKill delta computation across eviction boundary
  8. Persistence round-trip (save → load → verify)
  9. Pending pod age tracking + data gap tracking
 10. Dedup / daily summary
 11. Naive vs. tz-aware timestamp comparison
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

# Add parent to path so we can import watchdog
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from watchdog.state import StateManager
from watchdog.models import DedupKey


def _ts(dt: datetime) -> str:
    """Format a datetime as ISO 8601 with UTC offset."""
    return dt.isoformat()


def _make_snap(ts: datetime, node_count: int = 5, total_pods: int = 50,
               oomkilled: int = 0, agent_pods: list = None,
               workload_memory: list = None, pending_detail: list = None,
               data_gaps: list = None) -> dict:
    """Build a minimal snapshot dict for testing."""
    snap = {
        "timestamp": _ts(ts),
        "cluster_id": "test-cluster",
        "node_count": node_count,
        "total_pods": total_pods,
        "oomkilled_pods": [{"namespace": "ns", "name": f"pod-{i}", "restart_count": i+1}
                           for i in range(oomkilled)],
        "agent_pods": agent_pods or [],
        "workload_memory_usage": workload_memory or [],
        "pending_pods_detail": pending_detail or [],
        "data_gaps": data_gaps or [],
    }
    return snap


def fresh_sm(window_size=12, interval_seconds=300) -> StateManager:
    """Create a fresh StateManager with a unique temp file."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)  # start with no file
    return StateManager(path, window_size=window_size, interval_seconds=interval_seconds)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Normal rolling window — no gaps, count-based trim only
# ═══════════════════════════════════════════════════════════════════════

def test_normal_rolling_window():
    sm = fresh_sm(window_size=5, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Push 8 snapshots 5 min apart (no gap)
    for i in range(8):
        ts = base + timedelta(minutes=5 * i)
        sm.push_snapshot(_make_snap(ts, node_count=5 + i))

    assert sm.snapshot_count == 5, f"Expected 5 after trim, got {sm.snapshot_count}"
    # Should keep the last 5 (indices 3..7)
    snaps = sm.get_snapshots()
    assert snaps[0]["node_count"] == 8  # i=3 → 5+3=8
    assert snaps[-1]["node_count"] == 12  # i=7 → 5+7=12
    print("✓ Test 1 PASSED: Normal rolling window trims to window_size")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Staleness eviction after a 2-hour gap
# ═══════════════════════════════════════════════════════════════════════

def test_staleness_eviction_after_gap():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Push 5 snapshots at T+0..T+20min
    for i in range(5):
        ts = base + timedelta(minutes=5 * i)
        sm.push_snapshot(_make_snap(ts, node_count=10))

    assert sm.snapshot_count == 5

    # Gap: watchdog down for 2 hours. Next snapshot at T+140min.
    resume_ts = base + timedelta(minutes=140)
    sm.push_snapshot(_make_snap(resume_ts, node_count=15))

    # All 5 old snapshots are > 60min old relative to resume_ts → evicted
    # Only the resume snapshot should remain
    assert sm.snapshot_count == 1, f"Expected 1 after eviction, got {sm.snapshot_count}"
    assert sm.get_snapshots()[0]["node_count"] == 15
    print("✓ Test 2 PASSED: Staleness eviction drops all old snapshots after 2h gap")


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Partial eviction — some stale, some fresh
# ═══════════════════════════════════════════════════════════════════════

def test_partial_eviction():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Push 3 at T+0, T+5, T+10 (old)
    for i in range(3):
        sm.push_snapshot(_make_snap(base + timedelta(minutes=5 * i), node_count=i))

    # Push 2 at T+55, T+60 (within 60min of upcoming T+65)
    sm.push_snapshot(_make_snap(base + timedelta(minutes=55), node_count=100))
    sm.push_snapshot(_make_snap(base + timedelta(minutes=60), node_count=101))

    assert sm.snapshot_count == 5

    # Now push at T+65 — cutoff is T+65 - 60min = T+5
    # T+0 (node=0) → stale (0 < 5), evicted
    # T+5 (node=1) → cutoff is exactly 5min → ts.timestamp() >= cutoff → KEPT
    # T+10 (node=2) → kept
    # T+55 (node=100) → kept
    # T+60 (node=101) → kept
    sm.push_snapshot(_make_snap(base + timedelta(minutes=65), node_count=200))

    # T+0 evicted, T+5 through T+65 kept = 5 snapshots
    assert sm.snapshot_count == 5, f"Expected 5, got {sm.snapshot_count}"
    snaps = sm.get_snapshots()
    assert snaps[0]["node_count"] == 1  # T+5
    assert snaps[-1]["node_count"] == 200  # T+65
    print("✓ Test 3 PASSED: Partial eviction keeps fresh + boundary snapshots")


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Empty window — eviction is no-op
# ═══════════════════════════════════════════════════════════════════════

def test_empty_window():
    sm = fresh_sm()
    ts = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    sm.push_snapshot(_make_snap(ts))
    assert sm.snapshot_count == 1
    print("✓ Test 4 PASSED: Empty window — first push succeeds")


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: No eviction when snapshots are on schedule
# ═══════════════════════════════════════════════════════════════════════

def test_no_eviction_on_schedule():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Push 12 snapshots every 5 min — exactly filling the window
    for i in range(12):
        sm.push_snapshot(_make_snap(base + timedelta(minutes=5 * i), node_count=i))

    assert sm.snapshot_count == 12
    # 13th push: T+60. Cutoff = T+60 - 60min = T+0. T+0 is exactly at cutoff → kept.
    # But count trim will drop it anyway (13 > 12).
    sm.push_snapshot(_make_snap(base + timedelta(minutes=60), node_count=99))
    assert sm.snapshot_count == 12
    snaps = sm.get_snapshots()
    assert snaps[-1]["node_count"] == 99
    assert snaps[0]["node_count"] == 1  # T+5 (T+0 trimmed by count)
    print("✓ Test 5 PASSED: No eviction needed on normal schedule, count trim handles it")


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Missing timestamp on incoming — eviction skipped
# ═══════════════════════════════════════════════════════════════════════

def test_missing_incoming_timestamp():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    sm.push_snapshot(_make_snap(base, node_count=1))
    sm.push_snapshot(_make_snap(base + timedelta(minutes=5), node_count=2))

    # Push snapshot with no timestamp
    sm.push_snapshot({"cluster_id": "test", "node_count": 3})

    # All 3 should be present — eviction skipped due to missing incoming ts
    assert sm.snapshot_count == 3
    print("✓ Test 6 PASSED: Missing incoming timestamp — eviction skipped safely")


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: Unparseable timestamp on existing snap — kept
# ═══════════════════════════════════════════════════════════════════════

def test_unparseable_existing_timestamp():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Manually inject a snapshot with garbage timestamp
    sm._state["snapshots"].append({"timestamp": "not-a-date", "node_count": 999})
    sm._state["snapshots"].append(_make_snap(base, node_count=1))

    # Push something 2h later — the garbage-ts snap should be KEPT
    future = base + timedelta(hours=2)
    sm.push_snapshot(_make_snap(future, node_count=50))

    # garbage-ts snap is kept, base snap is evicted (>60min old), new snap added
    snaps = sm.get_snapshots()
    node_counts = [s["node_count"] for s in snaps]
    assert 999 in node_counts, f"Garbage-ts snap should be kept, got {node_counts}"
    assert 1 not in node_counts, f"Old snap should be evicted, got {node_counts}"
    assert 50 in node_counts
    print("✓ Test 7 PASSED: Unparseable existing timestamp — kept safely")


# ═══════════════════════════════════════════════════════════════════════
# TEST 8: Trend calculations after eviction (sparse window)
# ═══════════════════════════════════════════════════════════════════════

def test_trends_after_eviction():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Push 1 snapshot (post-eviction scenario: only resume snap exists)
    sm.push_snapshot(_make_snap(base, node_count=10, total_pods=50, oomkilled=2))

    # Trend lists should have 1 entry
    assert sm.get_oomkill_trend() == [2]
    assert sm.get_node_count_trend() == [10]
    assert sm.get_pod_count_trend() == [50]

    # Delta computations return 0.0 with < 2 snapshots
    assert sm.compute_node_count_delta_pct() == 0.0
    assert sm.compute_pod_count_delta_pct() == 0.0

    # Memory leak detection returns [] with < 3 snapshots
    assert sm.detect_memory_leaks(min_snapshots=3) == []

    # Agent restarts
    assert sm.compute_agent_restarts_last_hour() == 0

    # get_previous_snapshot returns None with 1 snap
    assert sm.get_previous_snapshot() is None

    print("✓ Test 8 PASSED: All trend methods handle single-snapshot window safely")


# ═══════════════════════════════════════════════════════════════════════
# TEST 9: Memory leak detection with growing usage
# ═══════════════════════════════════════════════════════════════════════

def test_memory_leak_detection():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Push 4 snapshots with monotonically increasing memory for one workload
    for i in range(4):
        ts = base + timedelta(minutes=5 * i)
        mem = [{"namespace": "prod", "workload": "leaky-app",
                "usage_mib": 100 + (i * 20)}]  # 100, 120, 140, 160
        sm.push_snapshot(_make_snap(ts, workload_memory=mem))

    leaks = sm.detect_memory_leaks(min_snapshots=3)
    assert len(leaks) == 1
    assert leaks[0]["namespace"] == "prod"
    assert leaks[0]["workload"] == "leaky-app"
    assert leaks[0]["growth_pct"] > 5
    print("✓ Test 9 PASSED: Memory leak detected with monotonic growth")


# ═══════════════════════════════════════════════════════════════════════
# TEST 10: Memory leak NOT triggered for stable usage
# ═══════════════════════════════════════════════════════════════════════

def test_no_false_positive_memory_leak():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Stable memory — no leak
    for i in range(4):
        ts = base + timedelta(minutes=5 * i)
        mem = [{"namespace": "prod", "workload": "stable-app",
                "usage_mib": 100}]
        sm.push_snapshot(_make_snap(ts, workload_memory=mem))

    leaks = sm.detect_memory_leaks(min_snapshots=3)
    assert len(leaks) == 0
    print("✓ Test 10 PASSED: No false positive for stable memory")


# ═══════════════════════════════════════════════════════════════════════
# TEST 11: OOMKill delta computation
# ═══════════════════════════════════════════════════════════════════════

def test_oomkill_delta():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Snap 1: pod-a has 5 lifetime restarts
    snap1 = _make_snap(base, oomkilled=0)
    snap1["oomkilled_pods"] = [
        {"namespace": "ns", "name": "pod-a", "restart_count": 5,
         "source": "snapshot_lastState"}
    ]
    sm.push_snapshot(snap1)

    # Snap 2: pod-a has 8 lifetime restarts → delta = 3
    snap2 = _make_snap(base + timedelta(minutes=5), oomkilled=0)
    snap2["oomkilled_pods"] = [
        {"namespace": "ns", "name": "pod-a", "restart_count": 8,
         "source": "snapshot_lastState"}
    ]
    sm.push_snapshot(snap2)

    result = sm.compute_oomkill_deltas(snap2["oomkilled_pods"])
    assert len(result) == 1
    assert result[0]["restart_count"] == 3, f"Expected delta 3, got {result[0]['restart_count']}"
    assert result[0]["lifetime_restart_count"] == 8
    print("✓ Test 11 PASSED: OOMKill delta = 3 (8 - 5)")


# ═══════════════════════════════════════════════════════════════════════
# TEST 12: OOMKill delta — new pod (no previous entry)
# ═══════════════════════════════════════════════════════════════════════

def test_oomkill_delta_new_pod():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Snap 1: empty
    sm.push_snapshot(_make_snap(base))

    # Snap 2: new OOM pod appears
    snap2 = _make_snap(base + timedelta(minutes=5))
    snap2["oomkilled_pods"] = [
        {"namespace": "ns", "name": "new-pod", "restart_count": 3,
         "source": "snapshot_lastState"}
    ]
    sm.push_snapshot(snap2)

    result = sm.compute_oomkill_deltas(snap2["oomkilled_pods"])
    # New pod — can't compute delta, pass through lifetime count
    assert result[0]["restart_count"] == 3
    assert "snapshot_interval_hours" not in result[0]  # cold-start marker
    print("✓ Test 12 PASSED: New OOM pod — no delta, no interval (cold-start)")


# ═══════════════════════════════════════════════════════════════════════
# TEST 13: Persistence round-trip
# ═══════════════════════════════════════════════════════════════════════

def test_persistence_roundtrip():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)

    sm1 = StateManager(path, window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    for i in range(3):
        sm1.push_snapshot(_make_snap(base + timedelta(minutes=5 * i), node_count=i))
    sm1.save()

    # Load from same file
    sm2 = StateManager(path, window_size=12, interval_seconds=300)
    assert sm2.snapshot_count == 3
    assert sm2.get_snapshots()[0]["node_count"] == 0
    assert sm2.get_snapshots()[-1]["node_count"] == 2

    # Push more and verify staleness works on reloaded state
    # Push at T+130min (2h10m) — all 3 old ones are > 60min old
    resume = base + timedelta(minutes=130)
    sm2.push_snapshot(_make_snap(resume, node_count=99))
    assert sm2.snapshot_count == 1
    assert sm2.get_snapshots()[0]["node_count"] == 99

    os.unlink(path)
    print("✓ Test 13 PASSED: Persistence round-trip with eviction on reload")


# ═══════════════════════════════════════════════════════════════════════
# TEST 14: Dedup tracking
# ═══════════════════════════════════════════════════════════════════════

def test_dedup():
    sm = fresh_sm()
    key = DedupKey(category="oomkill", workload="ns/pod-a")

    # First notification should be allowed
    assert sm.should_notify(key, dedup_window_minutes=60) is True
    sm.record_notification(key)

    # Immediate re-check should be blocked
    assert sm.should_notify(key, dedup_window_minutes=60) is False

    # With 0-minute window, should always allow
    assert sm.should_notify(key, dedup_window_minutes=0) is True
    print("✓ Test 14 PASSED: Dedup allows first, blocks duplicate within window")


# ═══════════════════════════════════════════════════════════════════════
# TEST 15: Daily summary tracking
# ═══════════════════════════════════════════════════════════════════════

def test_daily_summary():
    sm = fresh_sm()
    assert sm.should_post_daily_summary() is True
    sm.record_daily_summary()
    assert sm.should_post_daily_summary() is False
    print("✓ Test 15 PASSED: Daily summary posted once per day")


# ═══════════════════════════════════════════════════════════════════════
# TEST 16: Agent restart trend + delta computation
# ═══════════════════════════════════════════════════════════════════════

def test_agent_restart_trend():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # 3 snapshots with castai-agent restartCount going from 2 → 4 → 7
    for i, rc in enumerate([2, 4, 7]):
        agents = [{"name": "castai-agent-abc", "restart_count": rc}]
        sm.push_snapshot(_make_snap(base + timedelta(minutes=5 * i), agent_pods=agents))

    trend = sm.get_agent_restart_trend()
    assert len(trend) == 1
    assert trend[0]["name"] == "castai-agent-abc"
    assert trend[0]["restarts"] == [2, 4, 7]

    delta = sm.compute_agent_restarts_last_hour()
    assert delta == 5  # 7 - 2
    print("✓ Test 16 PASSED: Agent restart trend and delta correct")


# ═══════════════════════════════════════════════════════════════════════
# TEST 17: Node/pod count delta percentage
# ═══════════════════════════════════════════════════════════════════════

def test_count_delta_pct():
    sm = fresh_sm(window_size=12, interval_seconds=300)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # 3 snaps: nodes=[10, 10, 20] → avg of first 2 = 10, current = 20 → +100%
    sm.push_snapshot(_make_snap(base, node_count=10, total_pods=100))
    sm.push_snapshot(_make_snap(base + timedelta(minutes=5), node_count=10, total_pods=100))
    sm.push_snapshot(_make_snap(base + timedelta(minutes=10), node_count=20, total_pods=200))

    node_delta = sm.compute_node_count_delta_pct()
    assert abs(node_delta - 100.0) < 0.01, f"Expected ~100%, got {node_delta}"

    pod_delta = sm.compute_pod_count_delta_pct()
    assert abs(pod_delta - 100.0) < 0.01
    print("✓ Test 17 PASSED: Node/pod count delta percentage correct")


# ═══════════════════════════════════════════════════════════════════════
# TEST 18: Corrupted state file → graceful reset
# ═══════════════════════════════════════════════════════════════════════

def test_corrupted_state_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    with open(path, "w") as f:
        f.write("{garbage not json")

    sm = StateManager(path, window_size=12, interval_seconds=300)
    assert sm.snapshot_count == 0  # reset to empty
    print("✓ Test 18 PASSED: Corrupted state file → graceful reset")
    os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════
# TEST 19: Z-suffix timestamp handling
# ═══════════════════════════════════════════════════════════════════════

def test_z_suffix_timestamps():
    sm = fresh_sm(window_size=12, interval_seconds=300)

    # Use Z-suffix timestamps (common from MCP server)
    snap1 = {"timestamp": "2025-06-01T10:00:00Z", "cluster_id": "test", "node_count": 1}
    snap2 = {"timestamp": "2025-06-01T12:30:00Z", "cluster_id": "test", "node_count": 2}

    sm.push_snapshot(snap1)
    sm.push_snapshot(snap2)

    # snap1 is 2.5h before snap2 → stale (> 60min) → evicted
    assert sm.snapshot_count == 1
    assert sm.get_snapshots()[0]["node_count"] == 2
    print("✓ Test 19 PASSED: Z-suffix timestamps handled correctly")


# ═══════════════════════════════════════════════════════════════════════
# TEST 20: Eviction boundary — exactly at cutoff is KEPT
# ═══════════════════════════════════════════════════════════════════════

def test_boundary_exactly_at_cutoff():
    sm = fresh_sm(window_size=12, interval_seconds=300)

    # cutoff = incoming - (12 * 300) = incoming - 3600s
    # snap at exactly cutoff should be KEPT (>=)
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    sm.push_snapshot(_make_snap(base, node_count=1))  # T+0

    incoming = base + timedelta(seconds=3600)  # exactly 60min later
    sm.push_snapshot(_make_snap(incoming, node_count=2))

    # T+0 is exactly at cutoff → kept
    assert sm.snapshot_count == 2, f"Expected 2 (boundary kept), got {sm.snapshot_count}"
    print("✓ Test 20 PASSED: Snapshot exactly at cutoff boundary is kept")


# ═══════════════════════════════════════════════════════════════════════
# TEST 21: Eviction boundary — 1 second before cutoff is EVICTED
# ═══════════════════════════════════════════════════════════════════════

def test_boundary_one_second_before_cutoff():
    sm = fresh_sm(window_size=12, interval_seconds=300)

    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    sm.push_snapshot(_make_snap(base, node_count=1))  # T+0

    # incoming at T+60min+1sec → cutoff = T+1sec → T+0 is below cutoff → evicted
    incoming = base + timedelta(seconds=3601)
    sm.push_snapshot(_make_snap(incoming, node_count=2))

    assert sm.snapshot_count == 1, f"Expected 1 (old evicted), got {sm.snapshot_count}"
    assert sm.get_snapshots()[0]["node_count"] == 2
    print("✓ Test 21 PASSED: Snapshot 1s before cutoff is evicted")


# ═══════════════════════════════════════════════════════════════════════
# TEST 22: Data gap tracking — mature gaps only
# ═══════════════════════════════════════════════════════════════════════

def test_data_gap_tracking():
    sm = fresh_sm()
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    gaps = [{"workload": "ns/app-a"}, {"workload": "ns/app-b"}]
    snap = _make_snap(base, data_gaps=gaps)
    sm.push_snapshot(snap)
    sm.update_data_gaps(gaps)

    # Just created — not mature (< 2h)
    mature = sm.get_mature_data_gaps(min_hours=2.0)
    assert len(mature) == 0

    # Manually backdate first_seen to 3 hours ago
    sm._state["data_gap_first_seen"]["ns/app-a"] = (
        datetime.now(timezone.utc) - timedelta(hours=3)
    ).isoformat()

    mature = sm.get_mature_data_gaps(min_hours=2.0)
    assert len(mature) == 1
    assert mature[0]["workload"] == "ns/app-a"
    assert mature[0]["age_hours"] >= 2.0
    print("✓ Test 22 PASSED: Data gap maturity tracking works")


# ═══════════════════════════════════════════════════════════════════════
# TEST 23: Stale dedup cleanup
# ═══════════════════════════════════════════════════════════════════════

def test_stale_dedup_cleanup():
    sm = fresh_sm()
    key1 = DedupKey(category="oomkill", workload="ns/pod-a")
    key2 = DedupKey(category="mismatch", workload="ns/pod-b")

    sm.record_notification(key1)
    sm.record_notification(key2)

    # Backdate key1 to 25 hours ago
    key1_str = f"{key1.category}:{key1.workload}"
    sm._state["dedup_log"][key1_str] = (
        datetime.now(timezone.utc) - timedelta(hours=25)
    ).isoformat()

    sm.cleanup_stale_dedup_entries(max_age_hours=24)
    assert key1_str not in sm._state["dedup_log"]
    key2_str = f"{key2.category}:{key2.workload}"
    assert key2_str in sm._state["dedup_log"]
    print("✓ Test 23 PASSED: Stale dedup entries cleaned up")


# ═══════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_normal_rolling_window,
        test_staleness_eviction_after_gap,
        test_partial_eviction,
        test_empty_window,
        test_no_eviction_on_schedule,
        test_missing_incoming_timestamp,
        test_unparseable_existing_timestamp,
        test_trends_after_eviction,
        test_memory_leak_detection,
        test_no_false_positive_memory_leak,
        test_oomkill_delta,
        test_oomkill_delta_new_pod,
        test_persistence_roundtrip,
        test_dedup,
        test_daily_summary,
        test_agent_restart_trend,
        test_count_delta_pct,
        test_corrupted_state_file,
        test_z_suffix_timestamps,
        test_boundary_exactly_at_cutoff,
        test_boundary_one_second_before_cutoff,
        test_data_gap_tracking,
        test_stale_dedup_cleanup,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)
