from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Callable, Literal, Protocol

from sts2_rl.collect.policy import PolicyDecision, SimplePolicy
from sts2_rl.env.types import CandidateAction, StepObservation
from sts2_rl.env.wrapper import Sts2Env
from sts2_rl.lifecycle import normalize_optional_limit

NormalizeTarget = Literal["main_menu", "character_select"]


@dataclass(frozen=True)
class RuntimeNormalizationReport:
    base_url: str
    target: NormalizeTarget
    reached_target: bool
    stop_reason: str
    initial_screen: str
    final_screen: str
    initial_run_id: str
    final_run_id: str
    step_count: int
    wait_count: int
    action_sequence: list[str]
    strategy_histogram: dict[str, int]
    final_observation: StepObservation | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "base_url": self.base_url,
            "target": self.target,
            "reached_target": self.reached_target,
            "stop_reason": self.stop_reason,
            "initial_screen": self.initial_screen,
            "final_screen": self.final_screen,
            "initial_run_id": self.initial_run_id,
            "final_run_id": self.final_run_id,
            "step_count": self.step_count,
            "wait_count": self.wait_count,
            "action_sequence": self.action_sequence,
            "strategy_histogram": self.strategy_histogram,
        }


@dataclass(frozen=True)
class _NormalizationChoice:
    action: CandidateAction | None
    strategy: str


class SupportsEnv(Protocol):
    def observe(self): ...

    def step(self, action): ...

    def close(self) -> None: ...


def _default_env_factory(base_url: str, timeout: float) -> SupportsEnv:
    return Sts2Env.from_base_url(base_url, timeout=timeout)


def normalize_runtime_state(
    *,
    base_url: str,
    target: NormalizeTarget = "main_menu",
    poll_interval_seconds: float = 0.25,
    max_idle_polls: int = 40,
    max_steps: int | None = 64,
    request_timeout_seconds: float = 30.0,
    env_factory: Callable[[str, float], SupportsEnv] = _default_env_factory,
    policy: SimplePolicy | None = None,
) -> RuntimeNormalizationReport:
    policy = policy or SimplePolicy()
    env = env_factory(base_url, request_timeout_seconds)
    max_steps = normalize_optional_limit(max_steps)
    action_sequence: list[str] = []
    strategy_histogram: Counter[str] = Counter()
    wait_count = 0
    step_count = 0
    initial_screen = "UNKNOWN"
    final_screen = "UNKNOWN"
    initial_run_id = "run_unknown"
    final_run_id = "run_unknown"
    final_observation: StepObservation | None = None
    stop_reason = "target_reached"

    try:
        while True:
            observation = env.observe()
            final_observation = observation
            final_screen = observation.screen_type
            final_run_id = observation.run_id
            if step_count == 0 and not action_sequence:
                initial_screen = observation.screen_type
                initial_run_id = observation.run_id

            if _target_reached(observation, target):
                break

            if not observation.legal_actions:
                wait_count += 1
                if wait_count >= max_idle_polls:
                    stop_reason = "max_idle_polls_reached"
                    break
                sleep(poll_interval_seconds)
                continue

            choice = _choose_normalization_action(observation, target=target, policy=policy)
            if choice.action is None:
                stop_reason = f"no_normalization_action:{observation.screen_type.lower()}"
                break

            env.step(choice.action)
            action_sequence.append(choice.action.action_id)
            strategy_histogram[choice.strategy] += 1
            step_count += 1
            wait_count = 0
            if max_steps is not None and step_count >= max_steps:
                stop_reason = "max_steps_reached"
                break
    finally:
        env.close()

    return RuntimeNormalizationReport(
        base_url=base_url,
        target=target,
        reached_target=_target_reached(final_observation, target),
        stop_reason=stop_reason if final_observation is None or not _target_reached(final_observation, target) else "target_reached",
        initial_screen=initial_screen,
        final_screen=final_screen,
        initial_run_id=initial_run_id,
        final_run_id=final_run_id,
        step_count=step_count,
        wait_count=wait_count,
        action_sequence=action_sequence,
        strategy_histogram=dict(strategy_histogram),
        final_observation=final_observation,
    )


def write_runtime_normalization_report(
    report: RuntimeNormalizationReport,
    path: str | Path,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **report.as_dict(),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _target_reached(observation: StepObservation | None, target: NormalizeTarget) -> bool:
    if observation is None:
        return False
    if target == "main_menu":
        return observation.screen_type == "MAIN_MENU"
    if target == "character_select":
        return observation.screen_type == "CHARACTER_SELECT"
    return False


def _choose_normalization_action(
    observation: StepObservation,
    *,
    target: NormalizeTarget,
    policy: SimplePolicy,
) -> _NormalizationChoice:
    direct = _choose_direct_action(observation, target)
    if direct is not None:
        return direct

    if target == "character_select" and observation.screen_type == "MAIN_MENU":
        action = _find_first(observation, "open_character_select")
        if action is not None:
            return _NormalizationChoice(action=action, strategy="open_character_select")

    # When there is no direct exit path, use the existing heuristic controller to
    # progress toward a recoverable boundary instead of failing on mid-run screens.
    decision = policy.choose(observation)
    if decision.action is not None:
        return _NormalizationChoice(
            action=decision.action,
            strategy=_decision_strategy(observation, decision),
        )
    return _NormalizationChoice(action=None, strategy="no_action")


def _choose_direct_action(
    observation: StepObservation,
    target: NormalizeTarget,
) -> _NormalizationChoice | None:
    if target == "character_select" and observation.screen_type == "CHARACTER_SELECT":
        return _NormalizationChoice(action=None, strategy="already_at_target")
    if target == "main_menu" and observation.screen_type == "MAIN_MENU":
        return _NormalizationChoice(action=None, strategy="already_at_target")

    if observation.screen_type == "GAME_OVER":
        action = _find_first(observation, "return_to_main_menu", "continue_run")
        if action is not None:
            return _NormalizationChoice(action=action, strategy="game_over_to_main_menu")

    if target == "character_select" and observation.screen_type == "MAIN_MENU":
        action = _find_first(observation, "open_character_select")
        if action is not None:
            return _NormalizationChoice(action=action, strategy="menu_to_character_select")

    if observation.screen_type == "CHARACTER_SELECT":
        if target == "main_menu":
            action = _find_first(observation, "close_main_menu_submenu", "return_to_main_menu")
            if action is not None:
                return _NormalizationChoice(action=action, strategy="close_character_select")

    if observation.screen_type == "TIMELINE":
        action = _find_first(observation, "close_main_menu_submenu", "dismiss_modal", "confirm_modal")
        if action is not None:
            return _NormalizationChoice(action=action, strategy="close_timeline")

    if observation.screen_type == "MODAL":
        action = _find_first(observation, "dismiss_modal", "confirm_modal", "close_main_menu_submenu")
        if action is not None:
            return _NormalizationChoice(action=action, strategy="dismiss_modal")

    # If the runtime exposes an explicit abandon action, prefer that over
    # progressing deeper into the run.
    abandon = _find_first(observation, "abandon_run")
    if abandon is not None:
        return _NormalizationChoice(action=abandon, strategy="abandon_run")

    # Some screens expose a direct exit even when they are not menus.
    action = _find_first(observation, "return_to_main_menu", "close_main_menu_submenu")
    if action is not None:
        return _NormalizationChoice(action=action, strategy="direct_main_menu")

    return None


def _decision_strategy(observation: StepObservation, decision: PolicyDecision) -> str:
    return f"policy:{observation.screen_type.lower()}:{decision.reason}"


def _find_first(observation: StepObservation, *actions: str) -> CandidateAction | None:
    for action_name in actions:
        for candidate in observation.legal_actions:
            if candidate.action == action_name:
                return candidate
    return None
