from datetime import UTC, datetime, timedelta

from sts2_rl.runtime.watchdog import (
    InstanceWatchdogStatus,
    WatchdogPolicy,
    cooldown_remaining_seconds,
    record_watchdog_failure,
    record_watchdog_success,
)


def test_watchdog_transitions_through_degraded_cooling_and_quarantine() -> None:
    policy = WatchdogPolicy(
        failure_degraded_threshold=1,
        failure_cooldown_threshold=2,
        failure_quarantine_threshold=3,
        cooldown_seconds=15.0,
    )
    now = datetime(2026, 4, 12, 12, 0, tzinfo=UTC)
    status = InstanceWatchdogStatus(instance_id="inst-01")

    degraded = record_watchdog_failure(
        status,
        policy=policy,
        failure_kind="task_failed",
        error="first failure",
        now=now,
    )
    assert degraded.state == "degraded"
    assert degraded.total_failures == 1
    assert degraded.consecutive_failures == 1
    assert degraded.cooldown_until_utc is None

    cooling = record_watchdog_failure(
        degraded,
        policy=policy,
        failure_kind="task_failed",
        error="second failure",
        now=now + timedelta(seconds=1),
    )
    assert cooling.state == "cooling_down"
    assert cooling.total_failures == 2
    assert cooling.consecutive_failures == 2
    assert cooldown_remaining_seconds(cooling, now=now + timedelta(seconds=2)) > 0.0

    quarantined = record_watchdog_failure(
        cooling,
        policy=policy,
        failure_kind="task_failed",
        error="third failure",
        now=now + timedelta(seconds=20),
    )
    assert quarantined.state == "quarantined"
    assert quarantined.total_failures == 3
    assert quarantined.quarantine_reason is not None
    assert len(quarantined.transition_history) == 3


def test_watchdog_terminal_failure_recovers_after_success() -> None:
    policy = WatchdogPolicy(success_recovery_threshold=1)
    now = datetime(2026, 4, 12, 12, 0, tzinfo=UTC)
    status = InstanceWatchdogStatus(instance_id="inst-02")

    failed = record_watchdog_failure(
        status,
        policy=policy,
        failure_kind="artifact_missing",
        error="checkpoint missing",
        terminal=True,
        now=now,
    )
    assert failed.state == "failed"
    assert failed.last_failure_kind == "artifact_missing"

    recovered = record_watchdog_success(
        failed,
        policy=policy,
        reason="manual_retry_succeeded",
        now=now + timedelta(seconds=10),
    )
    assert recovered.state == "healthy"
    assert recovered.total_failures == 1
    assert recovered.total_successes == 1
    assert recovered.last_error is None
