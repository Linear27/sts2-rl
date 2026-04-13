from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SHADOW_COMBAT_SCHEMA_VERSION = 1
SHADOW_COMBAT_ENCOUNTERS_FILENAME = "encounters.jsonl"
SHADOW_COMBAT_ENCOUNTERS_TABLE_FILENAME = "encounters.csv"


class ShadowCombatEncounterRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SHADOW_COMBAT_SCHEMA_VERSION
    record_type: Literal["shadow_combat_encounter"] = "shadow_combat_encounter"
    encounter_id: str
    session_name: str
    session_kind: str
    instance_id: str
    run_id: str
    observed_seed: str | None = None
    character_id: str | None = None
    floor: int | None = None
    combat_index: int | None = None
    start_step_index: int | None = None
    finished_step_index: int | None = None
    outcome: str | None = None
    outcome_reason: str | None = None
    enemy_ids: list[str] = Field(default_factory=list)
    encounter_family: str | None = None
    action_trace_ids: list[str] = Field(default_factory=list)
    action_trace_count: int = 0
    unique_action_id_count: int = 0
    action_id_histogram: dict[str, int] = Field(default_factory=dict)
    cumulative_reward: float | None = None
    step_count: int | None = None
    damage_dealt: int | None = None
    damage_taken: int | None = None
    start_player_hp: int | None = None
    end_player_hp: int | None = None
    start_enemy_hp: int | None = None
    end_enemy_hp: int | None = None
    legal_action_count: int | None = None
    legal_action_ids: list[str] = Field(default_factory=list)
    state_summary: dict[str, Any] = Field(default_factory=dict)
    end_state_summary: dict[str, Any] = Field(default_factory=dict)
    action_descriptors: dict[str, Any] = Field(default_factory=dict)
    state: dict[str, Any] = Field(default_factory=dict)
    strategic_context: dict[str, Any] = Field(default_factory=dict)
    state_fingerprint: str | None = None
    action_space_fingerprint: str | None = None
    has_full_snapshot: bool = False
    has_terminal_outcome: bool = False

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def load_shadow_combat_encounter_records(source: str | Path) -> list[ShadowCombatEncounterRecord]:
    source_path = Path(source).expanduser().resolve()
    records_path = (
        source_path / SHADOW_COMBAT_ENCOUNTERS_FILENAME
        if source_path.is_dir()
        else source_path
    )
    if not records_path.exists():
        raise FileNotFoundError(f"Shadow combat encounter records do not exist: {records_path}")
    records: list[ShadowCombatEncounterRecord] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(ShadowCombatEncounterRecord.model_validate(json.loads(line)))
    return records
