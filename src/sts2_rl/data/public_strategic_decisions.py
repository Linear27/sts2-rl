from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PUBLIC_STRATEGIC_DECISIONS_SCHEMA_VERSION = 1
PUBLIC_STRATEGIC_DECISIONS_FILENAME = "strategic-decisions.jsonl"
PUBLIC_STRATEGIC_DECISIONS_TABLE_FILENAME = "strategic-decisions.csv"

PublicStrategicDecisionType = Literal[
    "reward_card_pick",
    "shop_buy",
    "selection_pick",
    "selection_remove",
    "selection_upgrade",
    "selection_transform",
    "event_choice",
    "rest_site_action",
]
PublicStrategicSupportQuality = Literal["full_candidates", "chosen_only"]


class PublicStrategicDecisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PUBLIC_STRATEGIC_DECISIONS_SCHEMA_VERSION
    record_type: Literal["public_strategic_decision"] = "public_strategic_decision"
    decision_id: str
    source_name: str
    snapshot_date: str | None = None
    source_run_id: int
    run_id: str
    character_id: str | None = None
    ascension: int | None = None
    build_id: str | None = None
    game_version: str | None = None
    branch: str | None = None
    content_channel: str | None = None
    game_mode: str | None = None
    platform_type: str | None = None
    run_outcome: Literal["win", "loss"] | None = None
    acts_reached: int | None = None
    act_index: int
    act_id: str
    floor: int | None = None
    floor_within_act: int
    room_type: str
    map_point_type: str | None = None
    model_id: str | None = None
    decision_type: PublicStrategicDecisionType
    support_quality: PublicStrategicSupportQuality
    reconstruction_confidence: float
    source_type: str | None = None
    candidate_actions: list[str] = Field(default_factory=list)
    chosen_action: str
    alternate_actions: list[str] = Field(default_factory=list)
    chosen_present_in_candidates: bool | None = None
    source_record_path: str
    source_record_index: int
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record(self) -> PublicStrategicDecisionRecord:
        self.decision_id = str(self.decision_id).strip()
        self.source_name = str(self.source_name).strip()
        self.run_id = str(self.run_id).strip()
        self.chosen_action = str(self.chosen_action).strip()
        if not self.decision_id:
            raise ValueError("decision_id is required.")
        if not self.source_name:
            raise ValueError("source_name is required.")
        if not self.run_id:
            raise ValueError("run_id is required.")
        if not self.chosen_action:
            raise ValueError("chosen_action is required.")
        if not (0.0 <= float(self.reconstruction_confidence) <= 1.0):
            raise ValueError("reconstruction_confidence must be between 0 and 1.")
        if self.support_quality == "full_candidates" and not self.candidate_actions:
            raise ValueError("full_candidates examples must include candidate_actions.")
        if self.chosen_present_in_candidates is True and self.chosen_action not in self.candidate_actions:
            raise ValueError("chosen_action must appear in candidate_actions when chosen_present_in_candidates is true.")
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def resolve_public_strategic_decisions_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    path = source_path / PUBLIC_STRATEGIC_DECISIONS_FILENAME if source_path.is_dir() else source_path
    if not path.exists():
        raise FileNotFoundError(f"Public strategic decisions do not exist: {path}")
    return path


def load_public_strategic_decision_records(source: str | Path) -> list[PublicStrategicDecisionRecord]:
    path = resolve_public_strategic_decisions_path(source)
    records: list[PublicStrategicDecisionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(PublicStrategicDecisionRecord.model_validate(json.loads(line)))
    return records
