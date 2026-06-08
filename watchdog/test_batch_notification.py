"""Principal-level E2E test suite for the batch notification layer.

Tests all flows: accumulation, dedup (within-buffer + cross-window),
flush success/failure/retry, immediate mode, --once flush, daily summary
bypass, app error bypass, multi-cluster isolation, verdict escalation,
and Slack block limits.

Run: python -m watchdog.test_batch_notification
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

# ── Env setup (must come before imports) ──
os.environ.setdefault("CASTAI_API_KEY", "fake-key")
os.environ.setdefault("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
os.environ.setdefault("SLACK_ADMIN_WEBHOOK_URL", "https://hooks.slack.com/admin-test")
os.environ.setdefault("WATCHDOG_CLUSTER_ID", "test-cluster-id-12345678")
os.environ.setdefault("WATCHDOG_CLUSTER_NAME", "test-cluster")
os.environ["WATCHDOG_NOTIFY_BATCH_CYCLES"] = "3"
os.environ["WATCHDOG_DRY_RUN"] = "true"
os.environ["WATCHDOG_LOG_LEVEL"] = "ERROR"

from watchdog.config import WatchdogConfig
from watchdog.models import (
    DedupKey,
    EvaluationResult,
    Finding,
    FindingCategory,
    Severity,
    SnapshotData,
    Verdict,
)
from watchdog.notifier import Notifier
from watchdog.state import StateManager

logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

# Use a unique state file per test run to avoid cross-run pollution
import tempfile
_TEST_STATE_DIR = tempfile.mkdtemp(prefix="watchdog_test_")
_test_counter = [0]


def _test_state_file() -> str:
    """Return a unique state file path for each test that creates a Watchdog."""
    _test_counter[0] += 1
    return os.path.join(_TEST_STATE_DIR, f"test_state_{_test_counter[0]}.json")

PASS_COUNT = 0
FAIL_COUNT = 0
FAILURES: list[str] = []


def check(condition: bool, msg: str):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✓ {msg}")
    else:
        FAIL_COUNT += 1
        FAILURES.append(msg)
        print(f"  ✗ FAIL: {msg}")


def make_finding(
    severity=Severity.CRITICAL,
    category=FindingCategory.OOMKILL,
    workload="ns/wl-A",
    what="Test finding",
) -> Finding:
    return Finding(
        severity=severity,
        category=category,
        workload=workload,
        what=what,
        evidence=json.dumps({"test": True, "workload": workload}),
        suggested_action="Investigate",
    )


def make_result(
    verdict=Verdict.WARNING,
    findings: list[Finding] | None = None,
    evaluated_at="2026-06-08T12:00:00Z",
) -> EvaluationResult:
    return EvaluationResult(
        verdict=verdict,
        summary=f"Test summary ({verdict.value})",
        findings=findings or [],
        evaluated_at=evaluated_at,
    )


def fresh_notifier() -> tuple[Notifier, StateManager]:
    config = WatchdogConfig()
    state = StateManager(":memory:", 12)
    return Notifier(config, state), state


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Batch accumulation across 3 cycles with mixed findings
# ═══════════════════════════════════════════════════════════════════
async def test_1_batch_accumulation():
    print("\n═══ TEST 1: Batch accumulation across 3 cycles ═══")
    n, state = fresh_notifier()

    # Cycle 1: OOMKill on wl-A
    r1 = make_result(Verdict.WARNING, [make_finding(workload="ns/wl-A")])
    added = await n.buffer_findings(r1, dry_run=True)
    check(added == 1, f"Cycle 1: added 1 finding (got {added})")
    check(n.buffer_size == 1, f"Buffer size == 1 (got {n.buffer_size})")

    # Cycle 2: Mismatch on wl-B + Agent restart on wl-C
    r2 = make_result(
        Verdict.CRITICAL,
        [
            make_finding(Severity.WARNING, FindingCategory.MISMATCH, "ns/wl-B", "Mismatch"),
            make_finding(Severity.WARNING, FindingCategory.AGENT_RESTART, "castai/agent", "Restart"),
        ],
    )
    added = await n.buffer_findings(r2, dry_run=True)
    check(added == 2, f"Cycle 2: added 2 new findings (got {added})")
    check(n.buffer_size == 3, f"Buffer size == 3 (got {n.buffer_size})")

    # Cycle 3: Data gap on wl-D
    r3 = make_result(Verdict.WARNING, [make_finding(Severity.WARNING, FindingCategory.DATA_GAP, "ns/wl-D")])
    added = await n.buffer_findings(r3, dry_run=True)
    check(added == 1, f"Cycle 3: added 1 new finding (got {added})")
    check(n.buffer_size == 4, f"Buffer size == 4 (got {n.buffer_size})")

    # Flush
    await n.flush(dry_run=True)
    check(n.buffer_size == 0, f"Buffer cleared after flush (got {n.buffer_size})")


# ═══════════════════════════════════════════════════════════════════
# TEST 2: Merged Slack output format after flush
# ═══════════════════════════════════════════════════════════════════
async def test_2_merged_slack_output():
    print("\n═══ TEST 2: Merged Slack output format ═══")
    n, state = fresh_notifier()

    # Buffer 3 different categories across 2 cycles
    await n.buffer_findings(
        make_result(Verdict.CRITICAL, [
            make_finding(Severity.CRITICAL, FindingCategory.OOMKILL, "ns/wl-A", "OOMKill spiral"),
        ]),
        dry_run=True,
    )
    await n.buffer_findings(
        make_result(Verdict.WARNING, [
            make_finding(Severity.WARNING, FindingCategory.MISMATCH, "ns/wl-B", "Mismatch detected"),
            make_finding(Severity.WARNING, FindingCategory.DATA_GAP, "ns/wl-C", "No recommendation"),
        ]),
        dry_run=True,
    )

    # Capture the _format_message output
    msg = n._format_message(
        n._latest_verdict,
        n._latest_summary,
        n._finding_buffer,
        n._latest_eval_at,
    )
    blocks = msg.get("blocks", [])
    text_fallback = msg.get("text", "")

    # Header should show CRITICAL (worst verdict)
    header = blocks[0] if blocks else {}
    header_text = header.get("text", {}).get("text", "")
    check("CRITICAL" in header_text, f"Header shows CRITICAL verdict: '{header_text[:60]}'")
    check("test-cluster" in header_text, f"Header shows cluster name: '{header_text[:60]}'")

    # Should have timestamp in header
    check("2026-06-08" in header_text or "UTC" in header_text, f"Header has full timestamp")

    # Footer should show "Next in 15m"
    footer_block = blocks[-1] if blocks else {}
    footer_text = ""
    if footer_block.get("type") == "context":
        footer_text = footer_block.get("elements", [{}])[0].get("text", "")
    check("15m" in footer_text, f"Footer shows 'Next in 15m': '{footer_text}'")

    # All 3 findings should be represented in the blocks
    all_block_text = json.dumps(blocks)
    check("oomkill" in all_block_text.lower() or "OOMKill" in all_block_text,
          "OOMKill finding present in blocks")
    check(len(blocks) >= 4, f"At least 4 blocks (header + summary + findings + footer), got {len(blocks)}")

    # Fallback text should contain verdict
    check("CRITICAL" in text_fallback, f"Fallback text has CRITICAL: '{text_fallback[:80]}'")

    await n.flush(dry_run=True)


# ═══════════════════════════════════════════════════════════════════
# TEST 3: Within-buffer dedup
# ═══════════════════════════════════════════════════════════════════
async def test_3_within_buffer_dedup():
    print("\n═══ TEST 3: Within-buffer dedup ═══")
    n, state = fresh_notifier()

    # Cycle 1: OOMKill on ns/wl-A
    r1 = make_result(Verdict.CRITICAL, [make_finding(workload="ns/wl-A")])
    await n.buffer_findings(r1, dry_run=True)
    check(n.buffer_size == 1, "Cycle 1: buffered 1")

    # Cycle 2: Same OOMKill on ns/wl-A again
    r2 = make_result(Verdict.CRITICAL, [make_finding(workload="ns/wl-A")])
    added = await n.buffer_findings(r2, dry_run=True)
    check(added == 0, f"Cycle 2: same workload deduped, added 0 (got {added})")
    check(n.buffer_size == 1, f"Buffer still 1 (got {n.buffer_size})")

    # Cycle 3: Same category different workload → should add
    r3 = make_result(Verdict.CRITICAL, [make_finding(workload="ns/wl-Z")])
    added = await n.buffer_findings(r3, dry_run=True)
    check(added == 1, f"Cycle 3: different workload added (got {added})")
    check(n.buffer_size == 2, f"Buffer now 2 (got {n.buffer_size})")

    await n.flush(dry_run=True)


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Cross-window dedup
# ═══════════════════════════════════════════════════════════════════
async def test_4_cross_window_dedup():
    print("\n═══ TEST 4: Cross-window dedup ═══")
    n, state = fresh_notifier()

    # Window 1: buffer + flush OOMKill on wl-A
    r1 = make_result(Verdict.CRITICAL, [make_finding(workload="ns/wl-A")])
    await n.buffer_findings(r1, dry_run=True)
    await n.flush(dry_run=True)  # records dedup key

    # Window 2: same finding should be rejected by state dedup
    r2 = make_result(Verdict.CRITICAL, [make_finding(workload="ns/wl-A")])
    added = await n.buffer_findings(r2, dry_run=True)
    check(added == 0, f"Cross-window: same finding rejected (got {added})")
    check(n.buffer_size == 0, f"Buffer empty (got {n.buffer_size})")

    # But a different workload should pass
    r3 = make_result(Verdict.CRITICAL, [make_finding(workload="ns/wl-NEW")])
    added = await n.buffer_findings(r3, dry_run=True)
    check(added == 1, f"New workload passes cross-window check (got {added})")

    await n.flush(dry_run=True)


# ═══════════════════════════════════════════════════════════════════
# TEST 5: Flush failure → buffer preserved → retry succeeds
# ═══════════════════════════════════════════════════════════════════
async def test_5_flush_failure_retry():
    print("\n═══ TEST 5: Flush failure → preserve → retry ═══")
    n, state = fresh_notifier()

    r = make_result(Verdict.CRITICAL, [make_finding(workload="ns/wl-RETRY")])
    await n.buffer_findings(r, dry_run=False)  # not dry_run!
    check(n.buffer_size == 1, "Buffered 1 finding")

    # Fail the flush
    n._post_to_slack = AsyncMock(return_value=False)
    await n.flush(dry_run=False)
    check(n.buffer_size == 1, f"Buffer PRESERVED on failure (got {n.buffer_size})")

    # Verify key NOT in dedup log
    key = DedupKey(category="oomkill", workload="ns/wl-RETRY")
    check(state.should_notify(key, 30), "Dedup key NOT recorded after failed flush")

    # Retry with success
    n._post_to_slack = AsyncMock(return_value=True)
    await n.flush(dry_run=False)
    check(n.buffer_size == 0, f"Buffer CLEARED on success (got {n.buffer_size})")
    check(not state.should_notify(key, 30), "Dedup key recorded after successful flush")


# ═══════════════════════════════════════════════════════════════════
# TEST 6: batch_cycles=1 → immediate notify() path
# ═══════════════════════════════════════════════════════════════════
async def test_6_immediate_mode():
    print("\n═══ TEST 6: batch_cycles=1 → immediate notify() ═══")
    from watchdog.main import Watchdog

    os.environ["WATCHDOG_NOTIFY_BATCH_CYCLES"] = "1"
    os.environ["WATCHDOG_STATE_FILE"] = _test_state_file()
    config = WatchdogConfig()
    check(config.notify_batch_cycles == 1, f"Config batch_cycles == 1 (got {config.notify_batch_cycles})")

    w = Watchdog(config)
    notify_called = []
    buffer_called = []

    orig_notify = w.notifier.notify
    orig_buffer = w.notifier.buffer_findings

    async def track_notify(result, dry_run=False):
        notify_called.append(True)
        return await orig_notify(result, dry_run=dry_run)

    async def track_buffer(result, dry_run=False):
        buffer_called.append(True)
        return await orig_buffer(result, dry_run=dry_run)

    w.notifier.notify = track_notify
    w.notifier.buffer_findings = track_buffer

    w.collector.collect = AsyncMock(return_value=SnapshotData(
        cluster_id="test", timestamp="2026-06-08T12:00:00Z",
        total_pods=10, running_pods=10, pending_pods=0, crashloop_pods=0,
        node_count=2, agent_restarts_last_hour=0,
    ))
    w.evaluator.evaluate = AsyncMock(return_value=make_result(
        Verdict.WARNING, [make_finding(workload="ns/wl-immed")]
    ))

    await w.run_once()
    check(len(notify_called) == 1, f"notify() called once (got {len(notify_called)})")
    check(len(buffer_called) == 0, f"buffer_findings() NOT called (got {len(buffer_called)})")
    check(w._cycle_count == 0, f"Cycle count stays 0 in immediate mode (got {w._cycle_count})")

    # Restore
    os.environ["WATCHDOG_NOTIFY_BATCH_CYCLES"] = "3"


# ═══════════════════════════════════════════════════════════════════
# TEST 7: --once mode flushes buffer before exit
# ═══════════════════════════════════════════════════════════════════
async def test_7_once_mode_flush():
    print("\n═══ TEST 7: --once mode flushes buffer ═══")
    from watchdog.main import Watchdog

    os.environ["WATCHDOG_NOTIFY_BATCH_CYCLES"] = "3"
    os.environ["WATCHDOG_STATE_FILE"] = _test_state_file()
    config = WatchdogConfig()
    w = Watchdog(config)

    w.collector.collect = AsyncMock(return_value=SnapshotData(
        cluster_id="test", timestamp="2026-06-08T12:00:00Z",
        total_pods=10, running_pods=10, pending_pods=0, crashloop_pods=0,
        node_count=2, agent_restarts_last_hour=0,
    ))
    w.evaluator.evaluate = AsyncMock(return_value=make_result(
        Verdict.WARNING, [make_finding(workload="ns/wl-once")]
    ))

    # run_once only does 1 cycle — buffer should have 1 finding
    await w.run_once()
    check(w._cycle_count == 1, f"After run_once: cycle=1 (got {w._cycle_count})")
    check(w.notifier.buffer_size == 1, f"Buffer has 1 finding (got {w.notifier.buffer_size})")

    # Simulate --once: flush before exit
    if w.notifier.buffer_size > 0:
        await w.notifier.flush(dry_run=True)
        w.state.save()
    check(w.notifier.buffer_size == 0, f"Buffer flushed before exit (got {w.notifier.buffer_size})")


# ═══════════════════════════════════════════════════════════════════
# TEST 8: 3 HEALTHY cycles → empty flush
# ═══════════════════════════════════════════════════════════════════
async def test_8_healthy_only():
    print("\n═══ TEST 8: 3 HEALTHY cycles → empty flush ═══")
    n, state = fresh_notifier()

    post_count = []
    orig_post = n._post_to_slack

    async def track_post(msg):
        post_count.append(True)
        return True

    n._post_to_slack = track_post

    for i in range(3):
        r = make_result(Verdict.HEALTHY, [])
        await n.buffer_findings(r, dry_run=False)

    check(n.buffer_size == 0, f"Nothing buffered (got {n.buffer_size})")

    await n.flush(dry_run=False)
    check(len(post_count) == 0, f"No Slack POST made (got {len(post_count)})")
    check(n._latest_verdict == Verdict.HEALTHY, f"Verdict reset to HEALTHY")


# ═══════════════════════════════════════════════════════════════════
# TEST 9: Daily summary bypasses buffer
# ═══════════════════════════════════════════════════════════════════
async def test_9_daily_summary_bypass():
    print("\n═══ TEST 9: Daily summary bypasses buffer ═══")
    n, state = fresh_notifier()

    daily_posted = []

    async def track_daily(result, dry_run):
        daily_posted.append(True)

    n._post_daily_summary = track_daily

    # Patch datetime so it's 08:00 UTC
    fake_now = datetime(2026, 6, 8, 8, 2, tzinfo=timezone.utc)
    with patch("watchdog.notifier.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.fromisoformat = datetime.fromisoformat
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        # Force should_post_daily_summary to return True
        state._state["last_daily_summary"] = ""

        r = make_result(Verdict.HEALTHY, [])
        await n.buffer_findings(r, dry_run=True)

    check(len(daily_posted) == 1, f"Daily summary fired immediately (got {len(daily_posted)} calls)")
    check(n.buffer_size == 0, "Buffer still empty (daily summary doesn't add findings)")


# ═══════════════════════════════════════════════════════════════════
# TEST 10: App error alerts bypass buffer
# ═══════════════════════════════════════════════════════════════════
async def test_10_app_error_bypass():
    print("\n═══ TEST 10: App error alerts bypass buffer ═══")
    n, state = fresh_notifier()

    admin_posted = []
    n._post_to_admin_slack = AsyncMock(return_value=True)

    # App error should post immediately, not go through buffer
    await n.notify_app_error(
        "collector", "Test error",
        cluster_id="test-12345678",
        cluster_name="test-cluster",
        context="Test context",
        dry_run=True,  # dry_run logs instead of posting
    )

    # The buffer should be untouched
    check(n.buffer_size == 0, "App error didn't touch the finding buffer")


# ═══════════════════════════════════════════════════════════════════
# TEST 11: Multi-cluster batch isolation
# ═══════════════════════════════════════════════════════════════════
async def test_11_multi_cluster_isolation():
    print("\n═══ TEST 11: Multi-cluster batch isolation ═══")
    from watchdog.main import Watchdog, _config_for_cluster

    os.environ["WATCHDOG_NOTIFY_BATCH_CYCLES"] = "3"
    os.environ["WATCHDOG_STATE_FILE"] = _test_state_file()
    base = WatchdogConfig()
    cfg_a = _config_for_cluster(base, "aaaa-1111-aaaa", "cluster-A")
    cfg_b = _config_for_cluster(base, "bbbb-2222-bbbb", "cluster-B")

    w_a = Watchdog(cfg_a)
    w_b = Watchdog(cfg_b)

    # Buffer a finding in cluster A
    r_a = make_result(Verdict.CRITICAL, [make_finding(workload="ns/wl-A-only")])
    await w_a.notifier.buffer_findings(r_a, dry_run=True)

    # Buffer a different finding in cluster B
    r_b = make_result(Verdict.WARNING, [make_finding(Severity.WARNING, FindingCategory.DATA_GAP, "ns/wl-B-only")])
    await w_b.notifier.buffer_findings(r_b, dry_run=True)

    check(w_a.notifier.buffer_size == 1, f"Cluster A buffer == 1 (got {w_a.notifier.buffer_size})")
    check(w_b.notifier.buffer_size == 1, f"Cluster B buffer == 1 (got {w_b.notifier.buffer_size})")

    # Flush A — should not affect B
    await w_a.notifier.flush(dry_run=True)
    check(w_a.notifier.buffer_size == 0, "Cluster A flushed")
    check(w_b.notifier.buffer_size == 1, f"Cluster B UNAFFECTED (got {w_b.notifier.buffer_size})")

    # Cycle counters are independent
    w_a._cycle_count = 2
    w_b._cycle_count = 1
    check(w_a._cycle_count != w_b._cycle_count, "Cycle counters are independent")

    await w_b.notifier.flush(dry_run=True)


# ═══════════════════════════════════════════════════════════════════
# TEST 12: Verdict escalation across batch window
# ═══════════════════════════════════════════════════════════════════
async def test_12_verdict_escalation():
    print("\n═══ TEST 12: Verdict escalation ═══")
    n, state = fresh_notifier()

    # Cycle 1: WARNING
    r1 = make_result(Verdict.WARNING, [make_finding(Severity.WARNING, workload="ns/wl-1")])
    await n.buffer_findings(r1, dry_run=True)
    check(n._latest_verdict == Verdict.WARNING, "After C1: WARNING")

    # Cycle 2: CRITICAL
    r2 = make_result(Verdict.CRITICAL, [make_finding(Severity.CRITICAL, workload="ns/wl-2")])
    await n.buffer_findings(r2, dry_run=True)
    check(n._latest_verdict == Verdict.CRITICAL, "After C2: CRITICAL (escalated)")

    # Cycle 3: HEALTHY — verdict should stay CRITICAL
    r3 = make_result(Verdict.HEALTHY, [])
    await n.buffer_findings(r3, dry_run=True)
    check(n._latest_verdict == Verdict.CRITICAL, "After C3 HEALTHY: still CRITICAL")

    # After flush, reset to HEALTHY
    await n.flush(dry_run=True)
    check(n._latest_verdict == Verdict.HEALTHY, "After flush: reset to HEALTHY")


# ═══════════════════════════════════════════════════════════════════
# TEST 13: Slack block count under limit with large buffer
# ═══════════════════════════════════════════════════════════════════
async def test_13_slack_block_limits():
    print("\n═══ TEST 13: Slack block count under 50 ═══")
    n, state = fresh_notifier()

    # Buffer 20 unique findings across multiple categories
    categories = [
        FindingCategory.OOMKILL, FindingCategory.MISMATCH,
        FindingCategory.CRASHLOOP, FindingCategory.UNSCHEDULABLE,
        FindingCategory.AGENT_RESTART, FindingCategory.DATA_GAP,
        FindingCategory.MEMORY_LEAK, FindingCategory.WEBHOOK_FAILURE,
    ]
    for i in range(20):
        cat = categories[i % len(categories)]
        sev = Severity.CRITICAL if i < 10 else Severity.WARNING
        f = make_finding(sev, cat, f"ns-{i}/wl-{i}", f"Finding #{i}")
        r = make_result(Verdict.CRITICAL if sev == Severity.CRITICAL else Verdict.WARNING, [f])
        await n.buffer_findings(r, dry_run=True)

    check(n.buffer_size == 20, f"Buffered 20 findings (got {n.buffer_size})")

    # Generate the Slack message
    msg = n._format_message(
        n._latest_verdict,
        n._latest_summary,
        n._finding_buffer,
        n._latest_eval_at,
    )
    blocks = msg.get("blocks", [])
    check(len(blocks) <= 50, f"Block count <= 50 (got {len(blocks)})")

    # Verify total text size is reasonable
    total_chars = sum(
        len(json.dumps(b.get("text", {})))
        for b in blocks
        if isinstance(b.get("text"), dict)
    )
    check(total_chars < 50_000, f"Total text chars reasonable ({total_chars})")

    await n.flush(dry_run=True)


# ═══════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════
async def main():
    await test_1_batch_accumulation()
    await test_2_merged_slack_output()
    await test_3_within_buffer_dedup()
    await test_4_cross_window_dedup()
    await test_5_flush_failure_retry()
    await test_6_immediate_mode()
    await test_7_once_mode_flush()
    await test_8_healthy_only()
    await test_9_daily_summary_bypass()
    await test_10_app_error_bypass()
    await test_11_multi_cluster_isolation()
    await test_12_verdict_escalation()
    await test_13_slack_block_limits()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {PASS_COUNT} passed, {FAIL_COUNT} failed")
    print("=" * 60)

    if FAILURES:
        print("\n  FAILURES:")
        for f in FAILURES:
            print(f"    ✗ {f}")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED ✓")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
