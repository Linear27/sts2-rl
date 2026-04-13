from __future__ import annotations

from dataclasses import dataclass

from sts2_rl.env.types import CandidateAction, StepObservation

MAX_HAND_CARDS = 10
MAX_TARGETS = 4
MAX_POTION_SLOTS = 3
COMBAT_ACTION_SCHEMA_VERSION = 1
COMBAT_ACTION_SPACE_NAME = "combat_v1"

_END_TURN_SLOT = 0
_PLAY_CARD_NO_TARGET_BASE = 1
_PLAY_CARD_TARGET_BASE = _PLAY_CARD_NO_TARGET_BASE + MAX_HAND_CARDS
_USE_POTION_NO_TARGET_BASE = _PLAY_CARD_TARGET_BASE + (MAX_HAND_CARDS * MAX_TARGETS)
_USE_POTION_TARGET_BASE = _USE_POTION_NO_TARGET_BASE + MAX_POTION_SLOTS
_DISCARD_POTION_BASE = _USE_POTION_TARGET_BASE + (MAX_POTION_SLOTS * MAX_TARGETS)
_TOTAL_SLOTS = _DISCARD_POTION_BASE + MAX_POTION_SLOTS


@dataclass(frozen=True)
class CombatActionBinding:
    mask: list[bool]
    candidates: list[CandidateAction | None]

    @property
    def slot_count(self) -> int:
        return len(self.mask)

    @property
    def available_slots(self) -> list[int]:
        return [index for index, enabled in enumerate(self.mask) if enabled]


class CombatActionSpace:
    @property
    def slot_count(self) -> int:
        return _TOTAL_SLOTS

    @property
    def action_schema_version(self) -> int:
        return COMBAT_ACTION_SCHEMA_VERSION

    @property
    def action_space_name(self) -> str:
        return COMBAT_ACTION_SPACE_NAME

    @property
    def slot_labels(self) -> tuple[str, ...]:
        return _SLOT_LABELS

    def bind(self, observation: StepObservation) -> CombatActionBinding:
        mask = [False] * self.slot_count
        candidates: list[CandidateAction | None] = [None] * self.slot_count

        for candidate in observation.legal_actions:
            slot = self._slot_for_candidate(candidate)
            if slot is None:
                continue
            if not mask[slot]:
                mask[slot] = True
                candidates[slot] = candidate

        return CombatActionBinding(mask=mask, candidates=candidates)

    def _slot_for_candidate(self, candidate: CandidateAction) -> int | None:
        request = candidate.request

        if candidate.action == "end_turn":
            return _END_TURN_SLOT

        if candidate.action == "play_card":
            card_index = request.card_index
            if card_index is None or not 0 <= card_index < MAX_HAND_CARDS:
                return None
            if request.target_index is None:
                return _PLAY_CARD_NO_TARGET_BASE + card_index
            if not 0 <= request.target_index < MAX_TARGETS:
                return None
            return _PLAY_CARD_TARGET_BASE + (card_index * MAX_TARGETS) + request.target_index

        if candidate.action == "use_potion":
            option_index = request.option_index
            if option_index is None or not 0 <= option_index < MAX_POTION_SLOTS:
                return None
            if request.target_index is None:
                return _USE_POTION_NO_TARGET_BASE + option_index
            if not 0 <= request.target_index < MAX_TARGETS:
                return None
            return _USE_POTION_TARGET_BASE + (option_index * MAX_TARGETS) + request.target_index

        if candidate.action == "discard_potion":
            option_index = request.option_index
            if option_index is None or not 0 <= option_index < MAX_POTION_SLOTS:
                return None
            return _DISCARD_POTION_BASE + option_index

        return None

    def schema_payload(self) -> dict[str, object]:
        return {
            "action_space_name": self.action_space_name,
            "action_schema_version": self.action_schema_version,
            "slot_count": self.slot_count,
            "slot_labels": list(self.slot_labels),
        }


def _build_slot_labels() -> tuple[str, ...]:
    labels = ["end_turn"]
    labels.extend(f"play_card_no_target_{index}" for index in range(MAX_HAND_CARDS))
    for card_index in range(MAX_HAND_CARDS):
        labels.extend(f"play_card_target_{card_index}_{target_index}" for target_index in range(MAX_TARGETS))
    labels.extend(f"use_potion_no_target_{index}" for index in range(MAX_POTION_SLOTS))
    for option_index in range(MAX_POTION_SLOTS):
        labels.extend(f"use_potion_target_{option_index}_{target_index}" for target_index in range(MAX_TARGETS))
    labels.extend(f"discard_potion_{index}" for index in range(MAX_POTION_SLOTS))
    return tuple(labels)


_SLOT_LABELS = _build_slot_labels()
