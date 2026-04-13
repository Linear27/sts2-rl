from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from sts2_rl.env.models import ActionRequest
from sts2_rl.env.types import StepObservation
from sts2_rl.game_run_contract import GameRunContract

from .normalize import normalize_runtime_state


class SupportsEnv(Protocol):
    def observe(self): ...

    def step(self, action): ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CustomRunPreparationReport:
    base_url: str
    run_mode: str
    requested_seed: str | None
    requested_character_id: str | None
    requested_ascension: int | None
    requested_custom_modifiers: tuple[str, ...]
    initial_screen: str
    final_screen: str
    final_run_id: str
    action_sequence: list[str]
    normalization_report: dict[str, Any]
    final_custom_seed: str | None
    final_custom_character_id: str | None
    final_custom_ascension: int | None
    final_custom_modifier_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "run_mode": self.run_mode,
            "requested_seed": self.requested_seed,
            "requested_character_id": self.requested_character_id,
            "requested_ascension": self.requested_ascension,
            "requested_custom_modifiers": list(self.requested_custom_modifiers),
            "initial_screen": self.initial_screen,
            "final_screen": self.final_screen,
            "final_run_id": self.final_run_id,
            "action_sequence": list(self.action_sequence),
            "normalization_report": dict(self.normalization_report),
            "final_custom_seed": self.final_custom_seed,
            "final_custom_character_id": self.final_custom_character_id,
            "final_custom_ascension": self.final_custom_ascension,
            "final_custom_modifier_ids": list(self.final_custom_modifier_ids),
        }


def contract_requires_custom_run_prepare(contract: GameRunContract | None) -> bool:
    return contract is not None and (contract.run_mode or "").lower() == "custom"


def prepare_custom_run_from_contract(
    *,
    base_url: str,
    contract: GameRunContract,
    request_timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.25,
    max_idle_polls: int = 40,
    max_prepare_steps: int | None = None,
    env_factory: Callable[[str, float], SupportsEnv],
) -> CustomRunPreparationReport:
    if not contract_requires_custom_run_prepare(contract):
        raise ValueError("prepare_custom_run_from_contract requires a custom run contract.")

    normalization = normalize_runtime_state(
        base_url=base_url,
        target="main_menu",
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        max_steps=max_prepare_steps,
        request_timeout_seconds=request_timeout_seconds,
        env_factory=env_factory,
    )
    if not normalization.reached_target:
        raise RuntimeError(
            "Failed to normalize runtime before preparing custom run. "
            f"initial={normalization.initial_screen} final={normalization.final_screen} "
            f"reason={normalization.stop_reason}"
        )

    action_sequence: list[str] = list(normalization.action_sequence)
    env = env_factory(base_url, request_timeout_seconds)
    try:
        observation = env.observe()
        observation = _ensure_main_menu_ready_for_custom_run(
            env,
            observation,
            action_sequence=action_sequence,
            max_attempts=8,
        )

        observation = _step_request(
            env,
            observation,
            ActionRequest(action="open_custom_run"),
            action_sequence=action_sequence,
        )
        custom = _require_custom_run(observation)

        requested_character_id = contract.character_id
        if requested_character_id is not None and custom.selected_character_id != requested_character_id:
            _require_action_descriptor(observation, "select_character")
            character_option = next(
                (
                    option
                    for option in custom.characters
                    if option.character_id is not None and option.character_id.upper() == requested_character_id.upper()
                ),
                None,
            )
            if character_option is None:
                available_character_ids = [option.character_id for option in custom.characters]
                raise RuntimeError(
                    "Requested custom-run character is unavailable. "
                    f"requested={requested_character_id} available={available_character_ids}"
                )
            observation = _step_request(
                env,
                observation,
                ActionRequest(action="select_character", option_index=character_option.index),
                action_sequence=action_sequence,
            )
            custom = _require_custom_run(observation)

        if contract.ascension is not None and custom.ascension != contract.ascension:
            if contract.ascension < 0:
                raise RuntimeError(
                    "Requested custom-run ascension is out of range. "
                    f"requested={contract.ascension} max={custom.max_ascension}"
                )
            if custom.max_ascension > 0 and contract.ascension > custom.max_ascension:
                raise RuntimeError(
                    "Requested custom-run ascension is out of range. "
                    f"requested={contract.ascension} max={custom.max_ascension}"
                )
            _require_custom_run_configuration_action(
                observation,
                action_name="set_custom_ascension",
                capability_enabled=custom.can_set_ascension,
            )
            observation = _step_request(
                env,
                observation,
                ActionRequest(action="set_custom_ascension", ascension=contract.ascension),
                action_sequence=action_sequence,
            )
            custom = _require_custom_run(observation)

        if contract.game_seed is not None and custom.seed != contract.game_seed:
            _require_custom_run_configuration_action(
                observation,
                action_name="set_custom_seed",
                capability_enabled=custom.can_set_seed,
            )
            observation = _step_request(
                env,
                observation,
                ActionRequest(action="set_custom_seed", seed=contract.game_seed),
                action_sequence=action_sequence,
            )
            custom = _require_custom_run(observation)

        desired_modifier_indices = _resolve_modifier_indices(
            desired_tokens=contract.custom_modifiers,
            modifiers=custom.modifiers,
        )
        desired_modifier_ids = _modifier_ids_for_indices(custom.modifiers, desired_modifier_indices)
        current_modifier_ids = _selected_modifier_ids(custom)
        if current_modifier_ids != desired_modifier_ids:
            _require_custom_run_configuration_action(
                observation,
                action_name="set_custom_modifiers",
                capability_enabled=custom.can_set_modifiers,
            )
            observation = _step_request(
                env,
                observation,
                ActionRequest(action="set_custom_modifiers", modifier_ids=list(desired_modifier_ids)),
                action_sequence=action_sequence,
            )
            custom = _require_custom_run(observation)
            final_modifier_ids = _selected_modifier_ids(custom)
            if final_modifier_ids != desired_modifier_ids:
                raise RuntimeError(
                    "Custom run modifiers did not match requested configuration after apply. "
                    f"requested={desired_modifier_ids} observed={final_modifier_ids}"
                )

        custom = _require_custom_run(observation)
        if not custom.can_embark:
            raise RuntimeError(
                "Custom run is configured but cannot embark. "
                f"character={custom.selected_character_id} seed={custom.seed} ascension={custom.ascension}"
            )

        configured_custom = _require_custom_run(observation)
        final_observation = _step_request(
            env,
            observation,
            ActionRequest(action="embark"),
            action_sequence=action_sequence,
        )
        return CustomRunPreparationReport(
            base_url=base_url,
            run_mode=contract.run_mode or "custom",
            requested_seed=contract.game_seed,
            requested_character_id=contract.character_id,
            requested_ascension=contract.ascension,
            requested_custom_modifiers=tuple(contract.custom_modifiers),
            initial_screen=normalization.initial_screen,
            final_screen=final_observation.screen_type,
            final_run_id=final_observation.run_id,
            action_sequence=action_sequence,
            normalization_report=normalization.as_dict(),
            final_custom_seed=configured_custom.seed,
            final_custom_character_id=configured_custom.selected_character_id,
            final_custom_ascension=configured_custom.ascension,
            final_custom_modifier_ids=list(configured_custom.modifier_ids),
        )
    finally:
        env.close()


def _step_request(
    env: SupportsEnv,
    observation: StepObservation,
    request: ActionRequest,
    *,
    action_sequence: list[str],
) -> StepObservation:
    result = env.step(request)
    action_sequence.append(_request_action_id(request))
    return result.observation


def _ensure_main_menu_ready_for_custom_run(
    env: SupportsEnv,
    observation: StepObservation,
    *,
    action_sequence: list[str],
    max_attempts: int,
) -> StepObservation:
    current = observation
    for _ in range(max_attempts):
        if current.screen_type == "MAIN_MENU" and _has_action(current, "open_custom_run"):
            return current
        if current.screen_type == "MAIN_MENU" and _has_action(current, "abandon_run"):
            current = _step_request(
                env,
                current,
                ActionRequest(action="abandon_run"),
                action_sequence=action_sequence,
            )
            continue
        if current.screen_type == "MODAL" and _has_action(current, "confirm_modal"):
            current = _step_request(
                env,
                current,
                ActionRequest(action="confirm_modal"),
                action_sequence=action_sequence,
            )
            continue
        if current.screen_type == "MODAL" and _has_action(current, "dismiss_modal"):
            current = _step_request(
                env,
                current,
                ActionRequest(action="dismiss_modal"),
                action_sequence=action_sequence,
            )
            continue
        current = env.observe()
    raise RuntimeError(
        "Main menu is not ready for custom-run opening. "
        f"screen={current.screen_type} actions={[candidate.action for candidate in current.legal_actions]}"
    )


def _require_custom_run(observation: StepObservation):
    custom = observation.state.custom_run
    if observation.screen_type != "CUSTOM_RUN" or custom is None:
        raise RuntimeError(
            "Expected CUSTOM_RUN while preparing fixed-seed run. "
            f"got_screen={observation.screen_type}"
        )
    return custom


def _request_action_id(request: ActionRequest) -> str:
    parts = [request.action]
    if request.option_index is not None:
        parts.append(f"option={request.option_index}")
    if request.modifier_index is not None:
        parts.append(f"modifier={request.modifier_index}")
    if request.enabled is not None:
        parts.append(f"enabled={str(bool(request.enabled)).lower()}")
    if request.ascension is not None:
        parts.append(f"ascension={request.ascension}")
    if request.seed is not None:
        parts.append(f"seed={request.seed}")
    if request.modifier_ids is not None:
        encoded_modifier_ids = ",".join(str(modifier_id) for modifier_id in request.modifier_ids if str(modifier_id))
        parts.append(f"modifiers={encoded_modifier_ids or 'none'}")
    return "|".join(parts)


def _resolve_modifier_indices(*, desired_tokens: tuple[str, ...], modifiers: list[Any]) -> set[int]:
    if not desired_tokens:
        return set()

    by_index = {int(modifier.index): modifier for modifier in modifiers}
    by_id: dict[str, list[int]] = {}
    by_name: dict[str, list[int]] = {}
    for modifier in modifiers:
        by_id.setdefault(_canonicalize_modifier_token(getattr(modifier, "modifier_id", None)), []).append(int(modifier.index))
        by_name.setdefault(_canonicalize_modifier_token(getattr(modifier, "name", None)), []).append(int(modifier.index))

    resolved: set[int] = set()
    for token in desired_tokens:
        index_override = _parse_modifier_index_token(token)
        if index_override is not None:
            if index_override not in by_index:
                raise RuntimeError(
                    f"Requested custom modifier index is out of range: {token}"
                )
            resolved.add(index_override)
            continue

        canonical = _canonicalize_modifier_token(token)
        if not canonical:
            continue

        id_matches = by_id.get(canonical, [])
        name_matches = by_name.get(canonical, [])
        combined = sorted(set(id_matches + name_matches))
        if not combined:
            available = [f"{modifier.index}:{modifier.modifier_id}:{modifier.name}" for modifier in modifiers]
            raise RuntimeError(
                "Requested custom modifier is unavailable. "
                f"requested={token} available={available}"
            )
        if len(combined) > 1:
            options = [f"{by_index[index].index}:{by_index[index].modifier_id}:{by_index[index].name}" for index in combined]
            raise RuntimeError(
                "Requested custom modifier is ambiguous. "
                f"requested={token} matches={options}"
            )
        resolved.add(combined[0])

    return resolved


def _parse_modifier_index_token(value: str) -> int | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    if "#" in normalized:
        prefix, suffix = normalized.rsplit("#", maxsplit=1)
        if prefix.strip():
            return int(suffix)
    if normalized.isdigit():
        return int(normalized)
    return None


def _canonicalize_modifier_token(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    return "".join(character for character in raw.upper() if character.isalnum())


def _modifier_ids_for_indices(modifiers: list[Any], indices: set[int]) -> list[str]:
    target_indices = set(indices)
    resolved_ids: list[str] = []
    for modifier in modifiers:
        if int(modifier.index) not in target_indices:
            continue
        modifier_id = str(getattr(modifier, "modifier_id", "") or "").strip()
        if modifier_id:
            resolved_ids.append(modifier_id)
    return resolved_ids


def _selected_modifier_ids(custom: Any) -> list[str]:
    if getattr(custom, "modifiers", None):
        return [
            str(modifier.modifier_id)
            for modifier in custom.modifiers
            if getattr(modifier, "is_selected", False) and str(getattr(modifier, "modifier_id", "") or "").strip()
        ]
    return [str(modifier_id) for modifier_id in getattr(custom, "modifier_ids", []) if str(modifier_id).strip()]


def _require_custom_run_configuration_action(
    observation: StepObservation,
    *,
    action_name: str,
    capability_enabled: bool,
) -> None:
    if not capability_enabled:
        raise RuntimeError(
            "Custom run configuration action is unavailable on the current runtime state. "
            f"action={action_name} screen={observation.screen_type}"
        )
    _require_action_descriptor(observation, action_name)


def _require_action_descriptor(observation: StepObservation, action_name: str) -> None:
    if _has_action_descriptor(observation, action_name):
        return
    available = [descriptor.name for descriptor in observation.action_descriptors.actions]
    raise RuntimeError(
        "Required runtime action descriptor is unavailable. "
        f"requested={action_name} available={available}"
    )


def _has_action_descriptor(observation: StepObservation, action_name: str) -> bool:
    return any(descriptor.name == action_name for descriptor in observation.action_descriptors.actions)


def _has_action(observation: StepObservation, action_name: str) -> bool:
    return any(candidate.action == action_name for candidate in observation.legal_actions)
