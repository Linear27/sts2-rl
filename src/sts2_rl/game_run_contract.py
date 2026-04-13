from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GameRunContract:
    run_mode: str | None = None
    game_seed: str | None = None
    seed_source: str | None = None
    character_id: str | None = None
    ascension: int | None = None
    custom_modifiers: tuple[str, ...] = ()
    progress_profile: str | None = None
    benchmark_contract_id: str | None = None
    strict: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "game_seed": self.game_seed,
            "seed_source": self.seed_source,
            "character_id": self.character_id,
            "ascension": self.ascension,
            "custom_modifiers": list(self.custom_modifiers),
            "progress_profile": self.progress_profile,
            "benchmark_contract_id": self.benchmark_contract_id,
            "strict": self.strict,
        }


@dataclass(frozen=True)
class GameRunContractObservation:
    checked: bool
    matches: bool
    mismatches: tuple[str, ...] = ()
    observed_seed: str | None = None
    observed_character_id: str | None = None
    observed_ascension: int | None = None
    screen_type: str | None = None
    run_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "matches": self.matches,
            "mismatches": list(self.mismatches),
            "observed_seed": self.observed_seed,
            "observed_character_id": self.observed_character_id,
            "observed_ascension": self.observed_ascension,
            "screen_type": self.screen_type,
            "run_id": self.run_id,
        }


def build_game_run_contract(
    *,
    run_mode: str | None = None,
    game_seed: str | None = None,
    seed_source: str | None = None,
    character_id: str | None = None,
    ascension: int | None = None,
    custom_modifiers: list[str] | tuple[str, ...] | None = None,
    progress_profile: str | None = None,
    benchmark_contract_id: str | None = None,
    strict: bool = True,
) -> GameRunContract | None:
    normalized_run_mode = _normalize_optional_text(run_mode, lowercase=True)
    normalized_game_seed = _normalize_optional_text(game_seed)
    normalized_seed_source = _normalize_optional_text(seed_source)
    normalized_character_id = _normalize_optional_text(character_id, uppercase=True)
    normalized_progress_profile = _normalize_optional_text(progress_profile)
    normalized_benchmark_contract_id = _normalize_optional_text(benchmark_contract_id)
    normalized_modifiers = tuple(_normalize_modifier_list(custom_modifiers))
    has_contract = any(
        value is not None
        for value in (
            normalized_run_mode,
            normalized_game_seed,
            normalized_seed_source,
            normalized_character_id,
            ascension,
            normalized_progress_profile,
            normalized_benchmark_contract_id,
        )
    ) or bool(normalized_modifiers)
    if not has_contract:
        return None
    return GameRunContract(
        run_mode=normalized_run_mode,
        game_seed=normalized_game_seed,
        seed_source=normalized_seed_source,
        character_id=normalized_character_id,
        ascension=ascension,
        custom_modifiers=normalized_modifiers,
        progress_profile=normalized_progress_profile,
        benchmark_contract_id=normalized_benchmark_contract_id,
        strict=bool(strict),
    )


def inspect_game_run_contract(
    *,
    contract: GameRunContract | None,
    screen_type: str,
    run_id: str,
    observed_seed: str | None,
    observed_character_id: str | None,
    observed_ascension: int | None,
) -> GameRunContractObservation:
    if contract is None:
        return GameRunContractObservation(
            checked=False,
            matches=True,
            observed_seed=observed_seed,
            observed_character_id=observed_character_id,
            observed_ascension=observed_ascension,
            screen_type=screen_type,
            run_id=run_id,
        )

    checked = any(
        value is not None for value in (observed_seed, observed_character_id, observed_ascension)
    )
    mismatches: list[str] = []
    if checked:
        if contract.game_seed is not None:
            if observed_seed is None:
                mismatches.append("missing_seed")
            elif observed_seed != contract.game_seed:
                mismatches.append("seed_mismatch")
        if contract.character_id is not None:
            if observed_character_id is None:
                mismatches.append("missing_character_id")
            elif observed_character_id != contract.character_id:
                mismatches.append("character_id_mismatch")
        if contract.ascension is not None:
            if observed_ascension is None:
                mismatches.append("missing_ascension")
            elif observed_ascension != contract.ascension:
                mismatches.append("ascension_mismatch")

    return GameRunContractObservation(
        checked=checked,
        matches=checked and not mismatches,
        mismatches=tuple(mismatches),
        observed_seed=observed_seed,
        observed_character_id=observed_character_id,
        observed_ascension=observed_ascension,
        screen_type=screen_type,
        run_id=run_id,
    )


def merge_game_run_contract_config(
    config: dict[str, Any],
    contract: GameRunContract | None,
) -> dict[str, Any]:
    merged = dict(config)
    if contract is not None:
        merged["game_run_contract"] = contract.as_dict()
    return merged


def build_game_run_contract_validation_payload(
    *,
    contract: GameRunContract | None,
    observation_check_count: int,
    observation_match_count: int,
    observation_mismatch_count: int,
    mismatch_histogram: dict[str, int],
    last_mismatches: list[str],
    observed_seed_histogram: dict[str, int],
    observed_character_histogram: dict[str, int],
    observed_ascension_histogram: dict[int, int],
) -> dict[str, Any]:
    if contract is None:
        return {
            "enabled": False,
            "status": "not_configured",
            "strict": False,
        }

    if observation_mismatch_count > 0:
        status = "mismatch"
    elif observation_check_count > 0:
        status = "matched"
    else:
        status = "pending"

    enforced_fields: list[str] = []
    if contract.game_seed is not None:
        enforced_fields.append("game_seed")
    if contract.character_id is not None:
        enforced_fields.append("character_id")
    if contract.ascension is not None:
        enforced_fields.append("ascension")

    unverified_fields: list[str] = []
    if contract.run_mode is not None:
        unverified_fields.append("run_mode")
    if contract.custom_modifiers:
        unverified_fields.append("custom_modifiers")

    return {
        "enabled": True,
        "status": status,
        "strict": contract.strict,
        "contract": contract.as_dict(),
        "observation_check_count": observation_check_count,
        "observation_match_count": observation_match_count,
        "observation_mismatch_count": observation_mismatch_count,
        "mismatch_histogram": dict(mismatch_histogram),
        "last_mismatches": list(last_mismatches),
        "enforced_fields": enforced_fields,
        "unverified_fields": unverified_fields,
        "seed_matches_expected": _histogram_matches_single_expected(
            observed_seed_histogram,
            contract.game_seed,
        ),
        "character_matches_expected": _histogram_matches_single_expected(
            observed_character_histogram,
            contract.character_id,
        ),
        "ascension_matches_expected": _int_histogram_matches_single_expected(
            observed_ascension_histogram,
            contract.ascension,
        ),
    }


def _histogram_matches_single_expected(histogram: dict[str, int], expected: str | None) -> bool | None:
    if expected is None:
        return None
    if not histogram:
        return None
    return set(histogram) == {expected}


def _int_histogram_matches_single_expected(histogram: dict[int, int], expected: int | None) -> bool | None:
    if expected is None:
        return None
    if not histogram:
        return None
    return set(histogram) == {expected}


def _normalize_modifier_list(values: list[str] | tuple[str, ...] | None) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    for value in values:
        item = _normalize_optional_text(value)
        if item is not None:
            normalized.append(item)
    return sorted(set(normalized))


def _normalize_optional_text(
    value: str | None,
    *,
    lowercase: bool = False,
    uppercase: bool = False,
) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if lowercase:
        normalized = normalized.lower()
    if uppercase:
        normalized = normalized.upper()
    return normalized
