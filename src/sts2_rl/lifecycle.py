from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from time import monotonic
from typing import Any

from sts2_rl.env.types import StepObservation


@dataclass(frozen=True)
class SessionBudgets:
    max_env_steps: int | None = None
    max_runs: int | None = None
    max_combats: int | None = None

    def as_dict(self) -> dict[str, int | None]:
        return {
            "max_env_steps": self.max_env_steps,
            "max_runs": self.max_runs,
            "max_combats": self.max_combats,
        }

    def stop_reason(
        self,
        *,
        env_steps: int,
        completed_run_count: int,
        completed_combat_count: int,
    ) -> str | None:
        if self.max_env_steps is not None and env_steps >= self.max_env_steps:
            return "max_env_steps_reached"
        if self.max_runs is not None and completed_run_count >= self.max_runs:
            return "max_runs_reached"
        if self.max_combats is not None and completed_combat_count >= self.max_combats:
            return "max_combats_reached"
        return None


@dataclass
class ObservationHeartbeat:
    last_progress_at: float = field(default_factory=monotonic)
    last_signature: tuple[Any, ...] | None = None
    idle_polls: int = 0

    def observe(self, observation: StepObservation) -> bool:
        signature = observation_signature(observation)
        if signature == self.last_signature:
            return False
        self.last_signature = signature
        self.mark_progress()
        return True

    def mark_progress(self, observation: StepObservation | None = None) -> None:
        if observation is not None:
            self.last_signature = observation_signature(observation)
        self.idle_polls = 0
        self.last_progress_at = monotonic()

    def note_wait(self) -> None:
        self.idle_polls += 1

    def reached_idle_poll_limit(self, max_idle_polls: int) -> bool:
        return self.idle_polls >= max_idle_polls

    def reached_idle_timeout(self, idle_timeout_seconds: float) -> bool:
        return (monotonic() - self.last_progress_at) >= idle_timeout_seconds


def normalize_optional_limit(limit: int | None) -> int | None:
    if limit is None or limit <= 0:
        return None
    return limit


def observation_signature(observation: StepObservation) -> tuple[Any, ...]:
    game_over = observation.state.game_over
    run = observation.state.run
    return (
        observation.screen_type,
        observation.run_id,
        observation.state.state_version,
        observation.state.turn,
        run.floor if run is not None else None,
        observation.state.session.phase,
        observation.state.session.control_scope,
        game_over.showing_summary if game_over is not None else None,
        game_over.can_continue if game_over is not None else None,
        game_over.can_return_to_main_menu if game_over is not None else None,
        tuple(candidate.action_id for candidate in observation.legal_actions),
    )
