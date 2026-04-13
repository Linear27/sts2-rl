from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

WatchdogState = Literal["healthy", "degraded", "cooling_down", "quarantined", "failed"]


class WatchdogPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    health_check_timeout_seconds: float = 2.0
    failure_degraded_threshold: int = 1
    failure_cooldown_threshold: int = 2
    failure_quarantine_threshold: int = 4
    cooldown_seconds: float = 30.0
    success_recovery_threshold: int = 1

    @model_validator(mode="after")
    def validate_thresholds(self) -> WatchdogPolicy:
        if self.health_check_timeout_seconds <= 0.0:
            raise ValueError("health_check_timeout_seconds must be > 0.")
        if self.failure_degraded_threshold < 1:
            raise ValueError("failure_degraded_threshold must be >= 1.")
        if self.failure_cooldown_threshold < self.failure_degraded_threshold:
            raise ValueError("failure_cooldown_threshold must be >= failure_degraded_threshold.")
        if self.failure_quarantine_threshold < self.failure_cooldown_threshold:
            raise ValueError("failure_quarantine_threshold must be >= failure_cooldown_threshold.")
        if self.cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds must be >= 0.")
        if self.success_recovery_threshold < 1:
            raise ValueError("success_recovery_threshold must be >= 1.")
        return self


@dataclass(frozen=True)
class WatchdogTransition:
    timestamp_utc: str
    previous_state: WatchdogState
    new_state: WatchdogState
    reason: str
    detail: str | None = None
    failure_kind: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "reason": self.reason,
            "detail": self.detail,
            "failure_kind": self.failure_kind,
        }


@dataclass
class InstanceWatchdogStatus:
    instance_id: str
    state: WatchdogState = "healthy"
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_failures: int = 0
    total_successes: int = 0
    cooldown_until_utc: str | None = None
    last_error: str | None = None
    last_failure_kind: str | None = None
    quarantine_reason: str | None = None
    transition_history: list[WatchdogTransition] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "state": self.state,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "cooldown_until_utc": self.cooldown_until_utc,
            "last_error": self.last_error,
            "last_failure_kind": self.last_failure_kind,
            "quarantine_reason": self.quarantine_reason,
            "transition_history": [item.as_dict() for item in self.transition_history],
        }


def can_execute_watchdog(status: InstanceWatchdogStatus, *, now: datetime | None = None) -> bool:
    timestamp = _coerce_utc(now)
    if status.state == "quarantined":
        return False
    if status.state != "cooling_down":
        return True
    if status.cooldown_until_utc is None:
        return True
    cooldown_until = datetime.fromisoformat(status.cooldown_until_utc)
    return timestamp >= cooldown_until


def cooldown_remaining_seconds(status: InstanceWatchdogStatus, *, now: datetime | None = None) -> float:
    if status.cooldown_until_utc is None:
        return 0.0
    timestamp = _coerce_utc(now)
    cooldown_until = datetime.fromisoformat(status.cooldown_until_utc)
    remaining = (cooldown_until - timestamp).total_seconds()
    return remaining if remaining > 0.0 else 0.0


def record_watchdog_success(
    status: InstanceWatchdogStatus,
    *,
    policy: WatchdogPolicy,
    reason: str = "execution_succeeded",
    detail: str | None = None,
    now: datetime | None = None,
) -> InstanceWatchdogStatus:
    timestamp = _coerce_utc(now)
    next_status = InstanceWatchdogStatus(
        instance_id=status.instance_id,
        state=status.state,
        consecutive_failures=0,
        consecutive_successes=status.consecutive_successes + 1,
        total_failures=status.total_failures,
        total_successes=status.total_successes + 1,
        cooldown_until_utc=None,
        last_error=None,
        last_failure_kind=None,
        quarantine_reason=None if status.state != "quarantined" else status.quarantine_reason,
        transition_history=list(status.transition_history),
    )
    if next_status.state != "quarantined" and next_status.consecutive_successes >= policy.success_recovery_threshold:
        next_status = _transition(
            next_status,
            new_state="healthy",
            reason=reason,
            detail=detail,
            now=timestamp,
        )
    return next_status


def record_watchdog_failure(
    status: InstanceWatchdogStatus,
    *,
    policy: WatchdogPolicy,
    failure_kind: str,
    error: str | None = None,
    terminal: bool = False,
    now: datetime | None = None,
) -> InstanceWatchdogStatus:
    timestamp = _coerce_utc(now)
    consecutive_failures = status.consecutive_failures + 1
    next_status = InstanceWatchdogStatus(
        instance_id=status.instance_id,
        state=status.state,
        consecutive_failures=consecutive_failures,
        consecutive_successes=0,
        total_failures=status.total_failures + 1,
        total_successes=status.total_successes,
        cooldown_until_utc=status.cooldown_until_utc,
        last_error=error,
        last_failure_kind=failure_kind,
        quarantine_reason=status.quarantine_reason,
        transition_history=list(status.transition_history),
    )

    if consecutive_failures >= policy.failure_quarantine_threshold:
        return _transition(
            next_status,
            new_state="quarantined",
            reason="failure_quarantine_threshold_reached",
            detail=error,
            failure_kind=failure_kind,
            quarantine_reason=f"{failure_kind}:{error}" if error else failure_kind,
            cooldown_until_utc=None,
            now=timestamp,
        )

    if terminal:
        return _transition(
            next_status,
            new_state="failed",
            reason="terminal_failure",
            detail=error,
            failure_kind=failure_kind,
            cooldown_until_utc=None,
            now=timestamp,
        )

    if consecutive_failures >= policy.failure_cooldown_threshold:
        cooldown_until = timestamp + timedelta(seconds=policy.cooldown_seconds)
        return _transition(
            next_status,
            new_state="cooling_down",
            reason="failure_cooldown_threshold_reached",
            detail=error,
            failure_kind=failure_kind,
            cooldown_until_utc=cooldown_until.isoformat(),
            now=timestamp,
        )

    if consecutive_failures >= policy.failure_degraded_threshold:
        return _transition(
            next_status,
            new_state="degraded",
            reason="failure_degraded_threshold_reached",
            detail=error,
            failure_kind=failure_kind,
            cooldown_until_utc=None,
            now=timestamp,
        )

    return next_status


def _transition(
    status: InstanceWatchdogStatus,
    *,
    new_state: WatchdogState,
    reason: str,
    detail: str | None,
    failure_kind: str | None = None,
    quarantine_reason: str | None = None,
    cooldown_until_utc: str | None = None,
    now: datetime,
) -> InstanceWatchdogStatus:
    previous_state = status.state
    transition = WatchdogTransition(
        timestamp_utc=now.isoformat(),
        previous_state=previous_state,
        new_state=new_state,
        reason=reason,
        detail=detail,
        failure_kind=failure_kind,
    )
    return InstanceWatchdogStatus(
        instance_id=status.instance_id,
        state=new_state,
        consecutive_failures=status.consecutive_failures,
        consecutive_successes=status.consecutive_successes,
        total_failures=status.total_failures,
        total_successes=status.total_successes,
        cooldown_until_utc=cooldown_until_utc,
        last_error=status.last_error,
        last_failure_kind=status.last_failure_kind,
        quarantine_reason=quarantine_reason,
        transition_history=[*status.transition_history, transition],
    )


def _coerce_utc(now: datetime | None) -> datetime:
    timestamp = datetime.now(UTC) if now is None else now
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)
