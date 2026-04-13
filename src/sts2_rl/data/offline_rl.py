from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field

from .trajectory import DataModel

OFFLINE_RL_SCHEMA_VERSION = 1
OFFLINE_RL_TRANSITIONS_FILENAME = "transitions.jsonl"
OFFLINE_RL_EPISODES_FILENAME = "episodes.jsonl"
OFFLINE_RL_TRANSITIONS_TABLE_FILENAME = "transitions.csv"
OFFLINE_RL_FEATURE_STATS_FILENAME = "feature-stats.json"


class OfflineRlTransitionRecord(DataModel):
    schema_version: int = OFFLINE_RL_SCHEMA_VERSION
    record_type: str = "transition"
    transition_id: str
    episode_id: str
    session_name: str
    session_kind: str
    instance_id: str
    run_id: str
    character_id: str | None = None
    floor: int | None = None
    step_index: int
    transition_index: int
    screen_type: str
    decision_stage: str | None = None
    decision_source: str | None = None
    policy_name: str | None = None
    policy_pack: str | None = None
    algorithm: str | None = None
    run_outcome: str | None = None
    run_finish_reason: str | None = None
    action_space_name: str | None = None
    action_schema_version: int | None = None
    feature_space_name: str | None = None
    feature_schema_version: int | None = None
    action_supported: bool = False
    action_index: int | None = None
    chosen_action_id: str | None = None
    chosen_action_label: str | None = None
    chosen_action_source: str | None = None
    legal_action_count: int = 0
    legal_action_ids: list[str] = Field(default_factory=list)
    action_mask: list[bool] = Field(default_factory=list)
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    environment_terminated: bool = False
    environment_truncated: bool = False
    next_transition_id: str | None = None
    next_screen_type: str | None = None
    next_floor: int | None = None
    next_legal_action_count: int | None = None
    next_legal_action_ids: list[str] = Field(default_factory=list)
    next_action_mask: list[bool] | None = None
    feature_vector: list[float] = Field(default_factory=list)
    next_feature_vector: list[float] | None = None
    state_summary: dict[str, Any] = Field(default_factory=dict)
    next_state_summary: dict[str, Any] | None = None
    strategic_context: dict[str, Any] = Field(default_factory=dict)


class OfflineRlEpisodeRecord(DataModel):
    schema_version: int = OFFLINE_RL_SCHEMA_VERSION
    record_type: str = "episode"
    episode_id: str
    session_name: str
    session_kind: str
    instance_id: str
    run_id: str
    character_id: str | None = None
    run_outcome: str | None = None
    run_finish_reason: str | None = None
    first_floor: int | None = None
    last_floor: int | None = None
    transition_count: int = 0
    supported_transition_count: int = 0
    return_value: float = 0.0
    discounted_return: float = 0.0
    mean_reward: float | None = None
    screen_histogram: dict[str, int] = Field(default_factory=dict)
    decision_stage_histogram: dict[str, int] = Field(default_factory=dict)
    decision_source_histogram: dict[str, int] = Field(default_factory=dict)
    action_space_histogram: dict[str, int] = Field(default_factory=dict)
    legal_action_count_stats: dict[str, float | int | None] = Field(default_factory=dict)
    first_transition_id: str | None = None
    last_transition_id: str | None = None


def load_offline_rl_transition_records(source: str | Path) -> list[OfflineRlTransitionRecord]:
    source_path = Path(source).expanduser().resolve()
    if source_path.is_dir():
        source_path = source_path / OFFLINE_RL_TRANSITIONS_FILENAME
    return _load_jsonl_records(source_path, OfflineRlTransitionRecord)


def load_offline_rl_episode_records(source: str | Path) -> list[OfflineRlEpisodeRecord]:
    source_path = Path(source).expanduser().resolve()
    if source_path.is_dir():
        source_path = source_path / OFFLINE_RL_EPISODES_FILENAME
    return _load_jsonl_records(source_path, OfflineRlEpisodeRecord)


def load_offline_rl_feature_stats(source: str | Path) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    if source_path.is_dir():
        source_path = source_path / OFFLINE_RL_FEATURE_STATS_FILENAME
    if not source_path.exists():
        raise FileNotFoundError(f"Offline RL feature stats do not exist: {source_path}")
    return json.loads(source_path.read_text(encoding="utf-8"))


def _load_jsonl_records(path: Path, model_type):
    if not path.exists():
        raise FileNotFoundError(f"Offline RL dataset file does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            records.append(model_type.model_validate(json.loads(line)))
    return records
