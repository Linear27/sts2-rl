from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PREDICTOR_DATASET_SCHEMA_VERSION = 1
PREDICTOR_MODEL_SCHEMA_VERSION = 1
PREDICTOR_TRAINING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PredictorExample:
    example_id: str
    source_path: str
    source_record_index: int
    session_name: str
    session_kind: str
    instance_id: str
    run_id: str
    floor: int | None
    combat_index: int
    outcome: str
    outcome_win_label: float | None
    reward_label: float
    damage_delta_label: float
    outcome_weight: float
    reward_weight: float
    damage_weight: float
    enemy_ids: list[str]
    feature_map: dict[str, float]
    start_summary: dict[str, Any]
    end_summary: dict[str, Any]
    strategic_context: dict[str, Any]
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREDICTOR_DATASET_SCHEMA_VERSION,
            "example_id": self.example_id,
            "source_path": self.source_path,
            "source_record_index": self.source_record_index,
            "session_name": self.session_name,
            "session_kind": self.session_kind,
            "instance_id": self.instance_id,
            "run_id": self.run_id,
            "floor": self.floor,
            "combat_index": self.combat_index,
            "outcome": self.outcome,
            "outcome_win_label": self.outcome_win_label,
            "reward_label": self.reward_label,
            "damage_delta_label": self.damage_delta_label,
            "outcome_weight": self.outcome_weight,
            "reward_weight": self.reward_weight,
            "damage_weight": self.damage_weight,
            "enemy_ids": self.enemy_ids,
            "feature_map": self.feature_map,
            "start_summary": self.start_summary,
            "end_summary": self.end_summary,
            "strategic_context": self.strategic_context,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictorExample:
        schema_version = int(payload.get("schema_version", PREDICTOR_DATASET_SCHEMA_VERSION))
        if schema_version != PREDICTOR_DATASET_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported predictor example schema_version="
                f"{schema_version}; expected {PREDICTOR_DATASET_SCHEMA_VERSION}."
            )
        return cls(
            example_id=str(payload["example_id"]),
            source_path=str(payload["source_path"]),
            source_record_index=int(payload["source_record_index"]),
            session_name=str(payload.get("session_name", "")),
            session_kind=str(payload.get("session_kind", "")),
            instance_id=str(payload.get("instance_id", "")),
            run_id=str(payload.get("run_id", "run_unknown")),
            floor=int(payload["floor"]) if payload.get("floor") is not None else None,
            combat_index=int(payload.get("combat_index", 0)),
            outcome=str(payload.get("outcome", "interrupted")),
            outcome_win_label=(
                None if payload.get("outcome_win_label") is None else float(payload["outcome_win_label"])
            ),
            reward_label=float(payload.get("reward_label", 0.0)),
            damage_delta_label=float(payload.get("damage_delta_label", 0.0)),
            outcome_weight=float(payload.get("outcome_weight", 0.0)),
            reward_weight=float(payload.get("reward_weight", 0.0)),
            damage_weight=float(payload.get("damage_weight", 0.0)),
            enemy_ids=[str(value) for value in payload.get("enemy_ids", [])],
            feature_map={str(key): float(value) for key, value in payload.get("feature_map", {}).items()},
            start_summary=dict(payload.get("start_summary", {})),
            end_summary=dict(payload.get("end_summary", {})),
            strategic_context=dict(payload.get("strategic_context", {})),
            metadata=dict(payload.get("metadata", {})),
        )
