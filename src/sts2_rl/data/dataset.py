from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import AvailableActionsPayload, GameStatePayload
from sts2_rl.env.types import CandidateAction, StepObservation
from sts2_rl.predict.features import extract_feature_map_from_summary
from sts2_rl.predict.schema import PredictorExample

from .offline_rl import (
    OFFLINE_RL_EPISODES_FILENAME,
    OFFLINE_RL_FEATURE_STATS_FILENAME,
    OFFLINE_RL_SCHEMA_VERSION,
    OFFLINE_RL_TRANSITIONS_FILENAME,
    OFFLINE_RL_TRANSITIONS_TABLE_FILENAME,
    OfflineRlEpisodeRecord,
    OfflineRlTransitionRecord,
)
from .shadow import (
    SHADOW_COMBAT_ENCOUNTERS_FILENAME,
    SHADOW_COMBAT_ENCOUNTERS_TABLE_FILENAME,
    SHADOW_COMBAT_SCHEMA_VERSION,
    ShadowCombatEncounterRecord,
)
from .public_run_normalized import (
    PUBLIC_RUN_NORMALIZED_FILENAME,
    PUBLIC_RUN_NORMALIZED_SOURCE_MANIFEST_FILENAME,
    PublicRunNormalizedRunRecord,
)
from .public_strategic_decisions import (
    PUBLIC_STRATEGIC_DECISIONS_FILENAME,
    PUBLIC_STRATEGIC_DECISIONS_TABLE_FILENAME,
    PublicStrategicDecisionRecord,
)
from .trajectory import TrajectoryStepRecord

DATASET_MANIFEST_SCHEMA_VERSION = 1
DATASET_SUMMARY_SCHEMA_VERSION = 2
DATASET_MANIFEST_FILENAME = "dataset-manifest.resolved.json"
DATASET_SUMMARY_FILENAME = "dataset-summary.json"
DATASET_SPLIT_NAMES = ("train", "validation", "test")
PREDICTOR_EXAMPLES_FILENAME = "examples.jsonl"
PREDICTOR_FEATURE_TABLE_FILENAME = "feature-table.csv"
TRAJECTORY_STEPS_FILENAME = "steps.jsonl"
TRAJECTORY_STEPS_TABLE_FILENAME = "steps.csv"

DatasetKind = Literal[
    "predictor_combat_outcomes",
    "trajectory_steps",
    "offline_rl_transitions",
    "shadow_combat_encounters",
    "public_strategic_decisions",
]
DatasetSourceKind = Literal["combat_outcomes", "trajectory_log", "public_run_normalized"]
DatasetSplitGroupBy = Literal["record", "run_id", "session_run", "combat_id"]


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetSourceSpec(DatasetModel):
    path: str
    source_kind: DatasetSourceKind
    recursive: bool = True


class DatasetFilterSpec(DatasetModel):
    session_kinds: list[str] = Field(default_factory=list)
    session_names: list[str] = Field(default_factory=list)
    instance_ids: list[str] = Field(default_factory=list)
    character_ids: list[str] = Field(default_factory=list)
    act_ids: list[str] = Field(default_factory=list)
    boss_ids: list[str] = Field(default_factory=list)
    second_boss_ids: list[str] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    screen_types: list[str] = Field(default_factory=list)
    decision_sources: list[str] = Field(default_factory=list)
    decision_stages: list[str] = Field(default_factory=list)
    decision_reasons: list[str] = Field(default_factory=list)
    policy_names: list[str] = Field(default_factory=list)
    policy_packs: list[str] = Field(default_factory=list)
    algorithms: list[str] = Field(default_factory=list)
    planner_names: list[str] = Field(default_factory=list)
    planner_strategies: list[str] = Field(default_factory=list)
    route_reason_tags: list[str] = Field(default_factory=list)
    route_profiles: list[str] = Field(default_factory=list)
    build_ids: list[str] = Field(default_factory=list)
    source_names: list[str] = Field(default_factory=list)
    decision_types: list[str] = Field(default_factory=list)
    support_qualities: list[str] = Field(default_factory=list)
    min_confidence: float | None = None
    min_reward: float | None = None
    max_reward: float | None = None
    min_legal_actions: int | None = None
    min_floor: int | None = None
    max_floor: int | None = None

    @model_validator(mode="after")
    def validate_floor_range(self) -> DatasetFilterSpec:
        if self.min_floor is not None and self.max_floor is not None and self.min_floor > self.max_floor:
            raise ValueError("filters.min_floor must be <= filters.max_floor.")
        if self.min_reward is not None and self.max_reward is not None and self.min_reward > self.max_reward:
            raise ValueError("filters.min_reward must be <= filters.max_reward.")
        if self.min_legal_actions is not None and self.min_legal_actions < 0:
            raise ValueError("filters.min_legal_actions must be non-negative.")
        if self.min_confidence is not None and not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("filters.min_confidence must be between 0 and 1.")
        return self


class DatasetSplitSpec(DatasetModel):
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 0
    group_by: DatasetSplitGroupBy = "session_run"

    @model_validator(mode="after")
    def validate_fractions(self) -> DatasetSplitSpec:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value < 0.0 for value in fractions):
            raise ValueError("split fractions must be non-negative.")
        total = sum(fractions)
        if total <= 0.0:
            raise ValueError("At least one split fraction must be positive.")
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1.0.")
        if self.train_fraction <= 0.0:
            raise ValueError("split.train_fraction must be positive.")
        return self


class DatasetOutputSpec(DatasetModel):
    export_csv: bool = True
    include_top_level_records: bool = True
    write_split_files: bool = True


class DatasetManifest(DatasetModel):
    schema_version: int = DATASET_MANIFEST_SCHEMA_VERSION
    dataset_name: str
    dataset_kind: DatasetKind
    description: str = ""
    sources: list[DatasetSourceSpec]
    filters: DatasetFilterSpec = Field(default_factory=DatasetFilterSpec)
    split: DatasetSplitSpec = Field(default_factory=DatasetSplitSpec)
    output: DatasetOutputSpec = Field(default_factory=DatasetOutputSpec)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sources(self) -> DatasetManifest:
        if not self.sources:
            raise ValueError("Dataset manifest must contain at least one source.")
        return self


@dataclass(frozen=True)
class BuiltDatasetReport:
    output_dir: Path
    manifest_path: Path
    summary_path: Path
    records_path: Path
    split_paths: dict[str, Path]
    dataset_kind: DatasetKind
    record_count: int
    feature_count: int
    source_file_count: int
    source_record_count: int
    filtered_out_count: int
    split_counts: dict[str, int]


@dataclass(frozen=True)
class DatasetValidationReport:
    manifest_path: Path | None
    dataset_name: str
    dataset_kind: DatasetKind
    source_files: tuple[Path, ...]
    filters: dict[str, Any]
    split: dict[str, Any]
    output: dict[str, Any]


@dataclass(frozen=True)
class _DatasetItem:
    record_id: str
    group_key: str
    session_name: str
    session_kind: str
    instance_id: str
    run_id: str
    character_id: str | None
    floor: int | None
    outcome: str | None
    payload: dict[str, Any]
    feature_names: tuple[str, ...] = ()
    screen_type: str | None = None
    decision_source: str | None = None
    decision_stage: str | None = None
    decision_reason: str | None = None
    policy_name: str | None = None
    policy_pack: str | None = None
    algorithm: str | None = None
    reward: float | None = None
    legal_action_count: int | None = None
    combat_id: str | None = None
    act_id: str | None = None
    act_index: int | None = None
    boss_id: str | None = None
    second_boss_id: str | None = None
    planner_name: str | None = None
    planner_strategy: str | None = None
    route_reason_tags: tuple[str, ...] = ()
    route_profile: str | None = None
    build_id: str | None = None
    source_name: str | None = None
    decision_type: str | None = None
    support_quality: str | None = None
    reconstruction_confidence: float | None = None
    strategic_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _TrajectoryRunMetadata:
    outcome: str | None
    finish_reason: str | None


@dataclass(frozen=True)
class _DatasetBuildResult:
    items: list[_DatasetItem]
    source_record_count: int
    filtered_out_count: int
    extras: dict[str, Any]


def load_dataset_manifest(path: str | Path) -> DatasetManifest:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    suffix = manifest_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported dataset manifest format: {manifest_path.suffix}")
    return DatasetManifest.model_validate(payload)


def load_dataset_summary(source: str | Path) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    summary_path = source_path / DATASET_SUMMARY_FILENAME if source_path.is_dir() else source_path
    if not summary_path.exists():
        raise FileNotFoundError(f"Dataset summary does not exist: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def validate_dataset_manifest(manifest: DatasetManifest | str | Path) -> DatasetValidationReport:
    manifest_model, _, manifest_path = _resolve_manifest_input(manifest)
    source_files = resolve_dataset_source_files(manifest)
    return DatasetValidationReport(
        manifest_path=manifest_path,
        dataset_name=manifest_model.dataset_name,
        dataset_kind=manifest_model.dataset_kind,
        source_files=tuple(source_files),
        filters=manifest_model.filters.model_dump(mode="json"),
        split=manifest_model.split.model_dump(mode="json"),
        output=manifest_model.output.model_dump(mode="json"),
    )


def resolve_dataset_source_files(manifest: DatasetManifest | str | Path) -> list[Path]:
    manifest_model, base_dir, _ = _resolve_manifest_input(manifest)
    resolved_sources = _resolve_source_specs(manifest_model.sources, base_dir=base_dir)
    return _discover_source_files(resolved_sources)


def resolve_dataset_split_paths(source: str | Path) -> dict[str, Path]:
    summary_payload = load_dataset_summary(source)
    split_paths = summary_payload.get("split", {}).get("split_paths", {})
    resolved: dict[str, Path] = {}
    for split_name in DATASET_SPLIT_NAMES:
        raw_path = split_paths.get(split_name)
        if raw_path:
            resolved[split_name] = Path(raw_path).expanduser().resolve()
    return resolved


def build_dataset_from_manifest(
    manifest: DatasetManifest | str | Path,
    *,
    output_dir: str | Path,
    replace_existing: bool = False,
) -> BuiltDatasetReport:
    manifest_model, _, _ = _resolve_manifest_input(manifest)
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        if not replace_existing:
            raise FileExistsError(f"Dataset output already exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    source_files = resolve_dataset_source_files(manifest)
    build_result = _build_items(manifest_model, source_files)
    items = build_result.items
    source_record_count = build_result.source_record_count
    filtered_out_count = build_result.filtered_out_count
    split_assignments = _assign_splits(items, manifest_model.split)
    split_counts = {split_name: len(indices) for split_name, indices in split_assignments.items()}
    split_group_counts = {
        split_name: len({items[index].group_key for index in indices}) for split_name, indices in split_assignments.items()
    }

    records_filename = _records_filename(manifest_model.dataset_kind)
    records_path = output_path / records_filename
    if manifest_model.output.include_top_level_records:
        _write_jsonl(records_path, (item.payload for item in items))
    else:
        records_path.write_text("", encoding="utf-8")

    split_paths: dict[str, Path] = {}
    if manifest_model.output.write_split_files:
        for split_name, indices in split_assignments.items():
            split_path = output_path / _split_records_filename(split_name, manifest_model.dataset_kind)
            _write_jsonl(split_path, (items[index].payload for index in indices))
            split_paths[split_name] = split_path

    if manifest_model.dataset_kind == "offline_rl_transitions":
        episode_records = list(build_result.extras.get("episode_records", []))
        if manifest_model.output.include_top_level_records:
            _write_jsonl(output_path / OFFLINE_RL_EPISODES_FILENAME, episode_records)
        else:
            (output_path / OFFLINE_RL_EPISODES_FILENAME).write_text("", encoding="utf-8")
        if manifest_model.output.write_split_files:
            episode_by_group_key = {
                item["episode_id"]: item for item in episode_records if isinstance(item, dict) and item.get("episode_id")
            }
            for split_name, indices in split_assignments.items():
                episode_ids = sorted(
                    {
                        str(items[index].payload.get("episode_id"))
                        for index in indices
                        if isinstance(items[index].payload, dict) and items[index].payload.get("episode_id")
                    }
                )
                split_episode_path = output_path / f"{split_name}.episodes.jsonl"
                _write_jsonl(
                    split_episode_path,
                    (episode_by_group_key[episode_id] for episode_id in episode_ids if episode_id in episode_by_group_key),
                )
        feature_stats_payload = dict(build_result.extras.get("feature_stats", {}))
        (output_path / OFFLINE_RL_FEATURE_STATS_FILENAME).write_text(
            json.dumps(feature_stats_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if manifest_model.output.export_csv:
        if manifest_model.dataset_kind == "predictor_combat_outcomes":
            _write_predictor_feature_table(output_path / PREDICTOR_FEATURE_TABLE_FILENAME, items)
        elif manifest_model.dataset_kind == "trajectory_steps":
            _write_trajectory_steps_table(output_path / TRAJECTORY_STEPS_TABLE_FILENAME, items)
        elif manifest_model.dataset_kind == "shadow_combat_encounters":
            _write_shadow_combat_encounters_table(output_path / SHADOW_COMBAT_ENCOUNTERS_TABLE_FILENAME, items)
        elif manifest_model.dataset_kind == "public_strategic_decisions":
            _write_public_strategic_decisions_table(output_path / PUBLIC_STRATEGIC_DECISIONS_TABLE_FILENAME, items)
        else:
            _write_offline_rl_transition_table(output_path / OFFLINE_RL_TRANSITIONS_TABLE_FILENAME, items)

    resolved_manifest_payload = _resolved_manifest_payload(manifest_model, source_files)
    manifest_path = output_path / DATASET_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(resolved_manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_payload = _dataset_summary_payload(
        manifest=manifest_model,
        output_dir=output_path,
        records_filename=records_filename,
        items=items,
        source_files=source_files,
        source_record_count=source_record_count,
        filtered_out_count=filtered_out_count,
        split_paths=split_paths,
        split_counts=split_counts,
        split_group_counts=split_group_counts,
        extras=build_result.extras,
    )
    summary_path = output_path / DATASET_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return BuiltDatasetReport(
        output_dir=output_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        records_path=records_path,
        split_paths=split_paths,
        dataset_kind=manifest_model.dataset_kind,
        record_count=len(items),
        feature_count=len(summary_payload.get("feature_names", [])),
        source_file_count=len(source_files),
        source_record_count=source_record_count,
        filtered_out_count=filtered_out_count,
        split_counts=split_counts,
    )


def discover_combat_outcome_paths(sources: Sequence[str | Path]) -> list[Path]:
    specs = [DatasetSourceSpec(path=str(Path(source).expanduser()), source_kind="combat_outcomes") for source in sources]
    return _discover_source_files(specs)


def discover_trajectory_log_paths(sources: Sequence[str | Path]) -> list[Path]:
    specs = [DatasetSourceSpec(path=str(Path(source).expanduser()), source_kind="trajectory_log") for source in sources]
    return _discover_source_files(specs)


def discover_public_run_normalized_paths(sources: Sequence[str | Path]) -> list[Path]:
    specs = [DatasetSourceSpec(path=str(Path(source).expanduser()), source_kind="public_run_normalized") for source in sources]
    return _discover_source_files(specs)


def _resolve_manifest_input(
    manifest: DatasetManifest | str | Path,
) -> tuple[DatasetManifest, Path | None, Path | None]:
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).expanduser().resolve()
        return load_dataset_manifest(manifest_path), manifest_path.parent, manifest_path
    return manifest, None, None


def _resolve_source_specs(sources: Sequence[DatasetSourceSpec], *, base_dir: Path | None) -> list[DatasetSourceSpec]:
    resolved_sources: list[DatasetSourceSpec] = []
    for source in sources:
        path = Path(source.path).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = (base_dir / path).resolve()
        else:
            path = path.resolve()
        resolved_sources.append(
            DatasetSourceSpec(
                path=str(path),
                source_kind=source.source_kind,
                recursive=source.recursive,
            )
        )
    return resolved_sources


def _discover_source_files(sources: Sequence[DatasetSourceSpec]) -> list[Path]:
    resolved_files: list[Path] = []
    seen: set[Path] = set()

    for source in sources:
        source_path = Path(source.path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset source does not exist: {source_path}")

        if source.source_kind == "combat_outcomes":
            if source_path.is_dir():
                candidate_paths = source_path.rglob("combat-outcomes.jsonl") if source.recursive else source_path.glob("combat-outcomes.jsonl")
            else:
                candidate_paths = [source_path]
        elif source.source_kind == "public_run_normalized":
            if source_path.is_dir():
                candidate_paths = (
                    source_path.rglob(PUBLIC_RUN_NORMALIZED_FILENAME)
                    if source.recursive
                    else source_path.glob(PUBLIC_RUN_NORMALIZED_FILENAME)
                )
            else:
                candidate_paths = [source_path]
        else:
            if source_path.is_dir():
                candidate_paths = source_path.rglob("*.jsonl") if source.recursive else source_path.glob("*.jsonl")
            else:
                candidate_paths = [source_path]

        for candidate in sorted(path.resolve() for path in candidate_paths):
            if not candidate.is_file() or candidate in seen:
                continue
            resolved_files.append(candidate)
            seen.add(candidate)

    if not resolved_files:
        raise FileNotFoundError("No dataset source files were discovered for the provided manifest.")
    return sorted(resolved_files)


def _build_items(
    manifest: DatasetManifest,
    source_files: Sequence[Path],
) -> _DatasetBuildResult:
    if manifest.dataset_kind == "offline_rl_transitions":
        return _build_offline_rl_items(manifest, source_files)
    if manifest.dataset_kind == "shadow_combat_encounters":
        return _build_shadow_combat_items(manifest, source_files)
    if manifest.dataset_kind == "public_strategic_decisions":
        return _build_public_strategic_items(manifest, source_files)

    items: list[_DatasetItem] = []
    source_record_count = 0
    filtered_out_count = 0

    for source_file in source_files:
        trajectory_run_metadata = (
            _extract_trajectory_run_metadata(source_file) if manifest.dataset_kind == "trajectory_steps" else {}
        )
        for record_index, payload in _iter_jsonl_payloads(source_file):
            source_record_count += 1
            item = _build_item_from_payload(
                dataset_kind=manifest.dataset_kind,
                source_file=source_file,
                record_index=record_index,
                payload=payload,
                group_by=manifest.split.group_by,
                trajectory_run_metadata=trajectory_run_metadata,
            )
            if item is None:
                filtered_out_count += 1
                continue
            if not _item_matches_filters(item, manifest.filters):
                filtered_out_count += 1
                continue
            items.append(item)

    if not items:
        raise ValueError("Dataset build produced zero records after filtering.")
    return _DatasetBuildResult(
        items=items,
        source_record_count=source_record_count,
        filtered_out_count=filtered_out_count,
        extras={},
    )


@dataclass
class _OpenCombatEncounter:
    session_name: str
    session_kind: str
    instance_id: str
    run_id: str
    combat_index: int | None
    floor: int | None
    started_step_index: int | None
    enemy_ids: list[str] = field(default_factory=list)
    start_summary: dict[str, Any] = field(default_factory=dict)
    first_step: TrajectoryStepRecord | None = None
    action_trace_ids: list[str] = field(default_factory=list)


def _build_shadow_combat_items(
    manifest: DatasetManifest,
    source_files: Sequence[Path],
) -> _DatasetBuildResult:
    items: list[_DatasetItem] = []
    source_record_count = 0
    filtered_out_count = 0

    for source_file in source_files:
        open_combat: _OpenCombatEncounter | None = None
        for record_index, payload in _iter_jsonl_payloads(source_file):
            source_record_count += 1
            record_type = payload.get("record_type")
            if record_type == "combat_started":
                if open_combat is not None:
                    item = _shadow_item_from_open_combat(
                        open_combat,
                        finish_payload=None,
                        source_file=source_file,
                        record_index=record_index,
                        group_by=manifest.split.group_by,
                    )
                    if item is not None and _item_matches_filters(item, manifest.filters):
                        items.append(item)
                    else:
                        filtered_out_count += 1
                open_combat = _OpenCombatEncounter(
                    session_name=str(payload.get("session_name", "")),
                    session_kind=str(payload.get("session_kind", "")),
                    instance_id=str(payload.get("instance_id", "")),
                    run_id=str(payload.get("run_id", "")),
                    combat_index=_normalized_optional_int(payload.get("combat_index")),
                    floor=_normalized_optional_int(payload.get("floor")),
                    started_step_index=_normalized_optional_int(payload.get("step_index")),
                    enemy_ids=_normalized_string_list(payload.get("enemy_ids")),
                    start_summary=dict(payload.get("state_summary", {})) if isinstance(payload.get("state_summary"), dict) else {},
                )
                continue
            if record_type == "step":
                record = TrajectoryStepRecord.model_validate(payload)
                if record.screen_type != "COMBAT":
                    filtered_out_count += 1
                    continue
                if open_combat is None or open_combat.run_id != record.run_id:
                    open_combat = _OpenCombatEncounter(
                        session_name=record.session_name,
                        session_kind=record.session_kind,
                        instance_id=record.instance_id,
                        run_id=record.run_id,
                        combat_index=None,
                        floor=record.floor,
                        started_step_index=record.step_index,
                        enemy_ids=_normalized_string_list(dict(record.state_summary.get("combat", {})).get("enemy_ids")),
                        start_summary=dict(record.state_summary),
                    )
                if open_combat.first_step is None:
                    open_combat.first_step = record
                if record.chosen_action_id is not None:
                    open_combat.action_trace_ids.append(str(record.chosen_action_id))
                continue
            if record_type == "combat_finished":
                if open_combat is None:
                    open_combat = _OpenCombatEncounter(
                        session_name=str(payload.get("session_name", "")),
                        session_kind=str(payload.get("session_kind", "")),
                        instance_id=str(payload.get("instance_id", "")),
                        run_id=str(payload.get("run_id", "")),
                        combat_index=_normalized_optional_int(payload.get("combat_index")),
                        floor=_normalized_optional_int(payload.get("floor")),
                        started_step_index=_normalized_optional_int(payload.get("started_step_index")),
                        enemy_ids=_normalized_string_list(payload.get("enemy_ids")),
                        start_summary=dict(payload.get("start_summary", {})) if isinstance(payload.get("start_summary"), dict) else {},
                    )
                item = _shadow_item_from_open_combat(
                    open_combat,
                    finish_payload=payload,
                    source_file=source_file,
                    record_index=record_index,
                    group_by=manifest.split.group_by,
                )
                if item is None or not _item_matches_filters(item, manifest.filters):
                    filtered_out_count += 1
                else:
                    items.append(item)
                open_combat = None
                continue
            filtered_out_count += 1
        if open_combat is not None:
            item = _shadow_item_from_open_combat(
                open_combat,
                finish_payload=None,
                source_file=source_file,
                record_index=source_record_count,
                group_by=manifest.split.group_by,
            )
            if item is None or not _item_matches_filters(item, manifest.filters):
                filtered_out_count += 1
            else:
                items.append(item)

    if not items:
        raise ValueError("Dataset build produced zero records after filtering.")
    return _DatasetBuildResult(
        items=items,
        source_record_count=source_record_count,
        filtered_out_count=filtered_out_count,
        extras={"shadow_schema_version": SHADOW_COMBAT_SCHEMA_VERSION},
    )


def _shadow_item_from_open_combat(
    open_combat: _OpenCombatEncounter,
    *,
    finish_payload: dict[str, Any] | None,
    source_file: Path,
    record_index: int,
    group_by: DatasetSplitGroupBy,
) -> _DatasetItem | None:
    first_step = open_combat.first_step
    finish_summary = dict(finish_payload.get("end_summary", {})) if isinstance(finish_payload, dict) and isinstance(finish_payload.get("end_summary"), dict) else {}
    start_summary = (
        dict(first_step.state_summary)
        if first_step is not None
        else dict(open_combat.start_summary)
    )
    strategic_context = _strategic_context_from_sources(summary=start_summary)
    run_summary = dict(start_summary.get("run", {}))
    combat_summary = dict(start_summary.get("combat", {}))
    enemy_ids = list(open_combat.enemy_ids) or _normalized_string_list(combat_summary.get("enemy_ids"))
    encounter_family = _encounter_family_label(enemy_ids)
    action_id_histogram = dict(Counter(open_combat.action_trace_ids))
    state_payload = dict(first_step.state) if first_step is not None else {}
    action_descriptors_payload = dict(first_step.action_descriptors) if first_step is not None else {}
    legal_action_ids = list(first_step.legal_action_ids) if first_step is not None else []
    legal_action_count = first_step.legal_action_count if first_step is not None else None
    state_fingerprint = _fingerprint_payload(state_payload) if state_payload else None
    action_space_fingerprint = _fingerprint_payload(
        {"legal_action_ids": legal_action_ids, "legal_action_count": legal_action_count}
    ) if legal_action_ids or legal_action_count is not None else None
    combat_index = open_combat.combat_index
    if combat_index is None and finish_payload is not None:
        combat_index = _normalized_optional_int(finish_payload.get("combat_index"))
    encounter_id = (
        f"{open_combat.session_name}:{open_combat.instance_id}:{open_combat.run_id}:{combat_index if combat_index is not None else record_index}"
    )
    payload = ShadowCombatEncounterRecord(
        encounter_id=encounter_id,
        session_name=open_combat.session_name,
        session_kind=open_combat.session_kind,
        instance_id=open_combat.instance_id,
        run_id=open_combat.run_id,
        observed_seed=_normalized_optional_str(start_summary.get("observed_seed") or run_summary.get("seed")),
        character_id=_normalized_optional_str(run_summary.get("character_id")),
        floor=open_combat.floor if open_combat.floor is not None else _normalized_optional_int(run_summary.get("floor")),
        combat_index=combat_index,
        start_step_index=open_combat.started_step_index,
        finished_step_index=_normalized_optional_int(None if finish_payload is None else finish_payload.get("finished_step_index")),
        outcome=_normalized_optional_str(None if finish_payload is None else finish_payload.get("outcome")),
        outcome_reason=_normalized_optional_str(None if finish_payload is None else finish_payload.get("reason")),
        enemy_ids=enemy_ids,
        encounter_family=encounter_family,
        action_trace_ids=list(open_combat.action_trace_ids),
        action_trace_count=len(open_combat.action_trace_ids),
        unique_action_id_count=len(action_id_histogram),
        action_id_histogram=action_id_histogram,
        cumulative_reward=_normalized_optional_float(None if finish_payload is None else finish_payload.get("cumulative_reward")),
        step_count=_normalized_optional_int(None if finish_payload is None else finish_payload.get("step_count")),
        damage_dealt=_normalized_optional_int(None if finish_payload is None else finish_payload.get("damage_dealt")),
        damage_taken=_normalized_optional_int(None if finish_payload is None else finish_payload.get("damage_taken")),
        start_player_hp=_normalized_optional_int(combat_summary.get("player_hp")),
        end_player_hp=_extract_end_player_hp(finish_summary),
        start_enemy_hp=_extract_total_enemy_hp(combat_summary.get("enemy_hp")),
        end_enemy_hp=_extract_end_enemy_hp(finish_summary),
        legal_action_count=legal_action_count,
        legal_action_ids=legal_action_ids,
        state_summary=start_summary,
        end_state_summary=finish_summary,
        action_descriptors=action_descriptors_payload,
        state=state_payload,
        strategic_context=strategic_context,
        state_fingerprint=state_fingerprint,
        action_space_fingerprint=action_space_fingerprint,
        has_full_snapshot=first_step is not None,
        has_terminal_outcome=finish_payload is not None,
    )
    return _DatasetItem(
        record_id=payload.encounter_id,
        group_key=_dataset_group_key(
            group_by=group_by,
            record_id=payload.encounter_id,
            session_name=payload.session_name,
            run_id=payload.run_id,
            combat_id=payload.encounter_id,
        ),
        session_name=payload.session_name,
        session_kind=payload.session_kind,
        instance_id=payload.instance_id,
        run_id=payload.run_id,
        character_id=payload.character_id,
        floor=payload.floor,
        outcome=payload.outcome,
        payload=payload.as_dict(),
        screen_type="COMBAT",
        reward=payload.cumulative_reward,
        legal_action_count=payload.legal_action_count,
        combat_id=payload.encounter_id,
        act_id=_normalized_optional_str(run_summary.get("act_id")),
        act_index=_normalized_optional_int(run_summary.get("act_index")),
        boss_id=_normalized_optional_str(run_summary.get("boss_encounter_id")),
        second_boss_id=_normalized_optional_str(run_summary.get("second_boss_encounter_id")),
        planner_name=_normalized_optional_str(strategic_context.get("planner_name")),
        planner_strategy=_normalized_optional_str(strategic_context.get("planner_strategy")),
        route_reason_tags=tuple(str(tag) for tag in strategic_context.get("route_reason_tags", [])),
        route_profile=_normalized_optional_str(strategic_context.get("route_profile")),
        strategic_context=strategic_context,
    )


def _build_public_strategic_items(
    manifest: DatasetManifest,
    source_files: Sequence[Path],
) -> _DatasetBuildResult:
    items: list[_DatasetItem] = []
    source_record_count = 0
    filtered_out_count = 0

    for source_file in source_files:
        source_context = _load_public_run_normalized_source_context(source_file)
        for record_index, payload in _iter_jsonl_payloads(source_file):
            source_record_count += 1
            run_record = PublicRunNormalizedRunRecord.model_validate(payload)
            decision_records = _extract_public_strategic_decision_records(
                run_record,
                source_file=source_file,
                source_record_index=record_index,
                snapshot_date=source_context.get("snapshot_date"),
            )
            if not decision_records:
                filtered_out_count += 1
                continue
            for decision_record in decision_records:
                item = _public_strategic_item_from_record(
                    decision_record=decision_record,
                    group_by=manifest.split.group_by,
                )
                if not _item_matches_filters(item, manifest.filters):
                    filtered_out_count += 1
                    continue
                items.append(item)

    if not items:
        raise ValueError("Dataset build produced zero records after filtering.")
    return _DatasetBuildResult(
        items=items,
        source_record_count=source_record_count,
        filtered_out_count=filtered_out_count,
        extras={},
    )


def _load_public_run_normalized_source_context(source_file: Path) -> dict[str, Any]:
    manifest_path = source_file.parent / PUBLIC_RUN_NORMALIZED_SOURCE_MANIFEST_FILENAME
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    generated_at_utc = payload.get("generated_at_utc")
    snapshot_date = None
    if isinstance(generated_at_utc, str) and len(generated_at_utc) >= 10:
        snapshot_date = generated_at_utc[:10]
    return {
        "generated_at_utc": generated_at_utc,
        "snapshot_date": snapshot_date,
        "source_manifest_path": str(manifest_path),
    }


def _extract_public_strategic_decision_records(
    run_record: PublicRunNormalizedRunRecord,
    *,
    source_file: Path,
    source_record_index: int,
    snapshot_date: str | None,
) -> list[PublicStrategicDecisionRecord]:
    run_id = str(run_record.provenance.get("dedupe_key") or f"{run_record.source_name}:{run_record.source_run_id}")
    run_outcome = _public_run_outcome_label(run_record)
    absolute_floors = _public_run_absolute_floor_map(run_record)
    records: list[PublicStrategicDecisionRecord] = []
    decision_counter = 0

    for room_index, room in enumerate(run_record.rooms):
        base_payload = {
            "source_name": run_record.source_name,
            "snapshot_date": snapshot_date,
            "source_run_id": run_record.source_run_id,
            "run_id": run_id,
            "character_id": run_record.character_id,
            "ascension": run_record.ascension,
            "build_id": run_record.build_id,
            "game_version": run_record.build_id,
            "branch": None,
            "content_channel": None,
            "game_mode": run_record.game_mode,
            "platform_type": run_record.platform_type,
            "run_outcome": run_outcome,
            "acts_reached": run_record.acts_reached,
            "act_index": room.act_index,
            "act_id": f"ACT_{room.act_index}",
            "floor": absolute_floors.get(room_index),
            "floor_within_act": room.floor_within_act,
            "room_type": room.room_type,
            "map_point_type": room.map_point_type,
            "model_id": room.model_id,
            "source_record_path": str(source_file),
            "source_record_index": source_record_index,
            "provenance": dict(run_record.provenance),
            "metadata": {
                "artifact_family": "public_run_normalized",
                "room_index": room_index,
                "has_detail_payload": bool(run_record.coverage_flags.get("has_detail_payload")),
                "has_room_history": bool(run_record.coverage_flags.get("has_room_history")),
                "room_source_type": room.source_type,
            },
        }

        reward_candidates = _unique_non_empty_strings(room.offered_cards)
        reward_picks = [card_id for card_id in _unique_non_empty_strings(room.picked_cards) if card_id in reward_candidates]
        for card_id in reward_picks:
            records.append(
                PublicStrategicDecisionRecord.model_validate(
                    {
                        **base_payload,
                        "decision_id": _public_decision_id(
                            run_id=run_id,
                            decision_type="reward_card_pick",
                            room_index=room_index,
                            decision_index=decision_counter,
                            chosen_action=card_id,
                        ),
                        "decision_type": "reward_card_pick",
                        "support_quality": "full_candidates",
                        "reconstruction_confidence": 1.0,
                        "source_type": room.source_type or "reward",
                        "candidate_actions": reward_candidates,
                        "chosen_action": card_id,
                        "alternate_actions": [candidate for candidate in reward_candidates if candidate != card_id],
                        "chosen_present_in_candidates": True,
                    }
                )
            )
            decision_counter += 1

        shop_candidates = _unique_non_empty_strings(room.shop_offered_cards)
        shop_purchases = [card_id for card_id in _unique_non_empty_strings(room.shop_purchased_cards) if card_id in shop_candidates]
        for card_id in shop_purchases:
            records.append(
                PublicStrategicDecisionRecord.model_validate(
                    {
                        **base_payload,
                        "decision_id": _public_decision_id(
                            run_id=run_id,
                            decision_type="shop_buy",
                            room_index=room_index,
                            decision_index=decision_counter,
                            chosen_action=card_id,
                        ),
                        "decision_type": "shop_buy",
                        "support_quality": "full_candidates",
                        "reconstruction_confidence": 1.0,
                        "source_type": "shop",
                        "candidate_actions": shop_candidates,
                        "chosen_action": card_id,
                        "alternate_actions": [candidate for candidate in shop_candidates if candidate != card_id],
                        "chosen_present_in_candidates": True,
                    }
                )
            )
            decision_counter += 1

        for card_id in _unique_non_empty_strings(room.cards_removed):
            records.append(
                PublicStrategicDecisionRecord.model_validate(
                    {
                        **base_payload,
                        "decision_id": _public_decision_id(
                            run_id=run_id,
                            decision_type="selection_remove",
                            room_index=room_index,
                            decision_index=decision_counter,
                            chosen_action=card_id,
                        ),
                        "decision_type": "selection_remove",
                        "support_quality": "chosen_only",
                        "reconstruction_confidence": 0.4,
                        "source_type": room.source_type or "selection",
                        "candidate_actions": [],
                        "chosen_action": card_id,
                        "alternate_actions": [],
                        "chosen_present_in_candidates": None,
                    }
                )
            )
            decision_counter += 1

        for choice_key in _unique_non_empty_strings(room.event_choice_keys):
            records.append(
                PublicStrategicDecisionRecord.model_validate(
                    {
                        **base_payload,
                        "decision_id": _public_decision_id(
                            run_id=run_id,
                            decision_type="event_choice",
                            room_index=room_index,
                            decision_index=decision_counter,
                            chosen_action=choice_key,
                        ),
                        "decision_type": "event_choice",
                        "support_quality": "chosen_only",
                        "reconstruction_confidence": 0.35,
                        "source_type": "event",
                        "candidate_actions": [],
                        "chosen_action": choice_key,
                        "alternate_actions": [],
                        "chosen_present_in_candidates": None,
                    }
                )
            )
            decision_counter += 1

        for action_name in _unique_non_empty_strings(room.rest_site_actions):
            records.append(
                PublicStrategicDecisionRecord.model_validate(
                    {
                        **base_payload,
                        "decision_id": _public_decision_id(
                            run_id=run_id,
                            decision_type="rest_site_action",
                            room_index=room_index,
                            decision_index=decision_counter,
                            chosen_action=action_name,
                        ),
                        "decision_type": "rest_site_action",
                        "support_quality": "chosen_only",
                        "reconstruction_confidence": 0.5,
                        "source_type": "rest",
                        "candidate_actions": [],
                        "chosen_action": action_name,
                        "alternate_actions": [],
                        "chosen_present_in_candidates": None,
                    }
                )
            )
            decision_counter += 1

    return records


def _public_strategic_item_from_record(
    *,
    decision_record: PublicStrategicDecisionRecord,
    group_by: DatasetSplitGroupBy,
) -> _DatasetItem:
    decision_stage = _public_decision_stage(decision_record.decision_type)
    strategic_context = {
        "artifact_family": "public_strategic_decisions",
        "acts_reached": decision_record.acts_reached,
        "floor_within_act": decision_record.floor_within_act,
        "candidate_count": len(decision_record.candidate_actions),
        "support_quality": decision_record.support_quality,
    }
    return _DatasetItem(
        record_id=decision_record.decision_id,
        group_key=_dataset_group_key(
            group_by=group_by,
            record_id=decision_record.decision_id,
            session_name=decision_record.source_name,
            run_id=decision_record.run_id,
        ),
        session_name=decision_record.source_name,
        session_kind="public_run",
        instance_id=f"public:{decision_record.source_name}",
        run_id=decision_record.run_id,
        character_id=decision_record.character_id,
        floor=decision_record.floor,
        outcome=decision_record.run_outcome,
        payload=decision_record.as_dict(),
        screen_type="PUBLIC_STRATEGIC",
        decision_source="public_run",
        decision_stage=decision_stage,
        decision_reason=decision_record.decision_type,
        act_id=decision_record.act_id,
        act_index=decision_record.act_index,
        build_id=decision_record.build_id,
        source_name=decision_record.source_name,
        decision_type=decision_record.decision_type,
        support_quality=decision_record.support_quality,
        reconstruction_confidence=decision_record.reconstruction_confidence,
        strategic_context=strategic_context,
    )


def _build_item_from_payload(
    *,
    dataset_kind: DatasetKind,
    source_file: Path,
    record_index: int,
    payload: dict[str, Any],
    group_by: DatasetSplitGroupBy,
    trajectory_run_metadata: dict[str, _TrajectoryRunMetadata] | None = None,
) -> _DatasetItem | None:
    if dataset_kind == "predictor_combat_outcomes":
        if payload.get("record_type") != "combat_finished":
            return None
        example = _predictor_example_from_combat_outcome(
            payload,
            source_path=source_file,
            source_record_index=record_index,
        )
        strategic_context = dict(example.strategic_context)
        run_summary = dict(example.start_summary.get("run", {}))
        return _DatasetItem(
            record_id=example.example_id,
            group_key=_dataset_group_key(
                group_by=group_by,
                record_id=example.example_id,
                session_name=example.session_name,
                run_id=example.run_id,
                combat_id=f"{example.session_name}:{example.instance_id}:{example.run_id}:{example.combat_index}",
            ),
            session_name=example.session_name,
            session_kind=example.session_kind,
            instance_id=example.instance_id,
            run_id=example.run_id,
            character_id=str((example.start_summary.get("run") or {}).get("character_id") or "") or None,
            floor=example.floor,
            outcome=example.outcome,
            payload=example.as_dict(),
            feature_names=tuple(sorted(example.feature_map)),
            reward=float(example.reward_label),
            combat_id=f"{example.session_name}:{example.instance_id}:{example.run_id}:{example.combat_index}",
            act_id=_normalized_optional_str(run_summary.get("act_id")),
            act_index=_normalized_optional_int(run_summary.get("act_index")),
            boss_id=_normalized_optional_str(run_summary.get("boss_encounter_id")),
            second_boss_id=_normalized_optional_str(run_summary.get("second_boss_encounter_id")),
            planner_name=_normalized_optional_str(strategic_context.get("planner_name")),
            planner_strategy=_normalized_optional_str(strategic_context.get("planner_strategy")),
            route_reason_tags=tuple(str(tag) for tag in strategic_context.get("route_reason_tags", [])),
            route_profile=_normalized_optional_str(strategic_context.get("route_profile")),
            strategic_context=strategic_context,
        )

    if payload.get("record_type") != "step":
        return None
    record = TrajectoryStepRecord.model_validate(payload)
    run_metadata = (trajectory_run_metadata or {}).get(record.run_id)
    state_run = record.state_summary.get("run") or {}
    payload_copy = record.model_dump(mode="json")
    strategic_context = _strategic_context_from_sources(
        summary=record.state_summary,
        decision_metadata=record.decision_metadata,
        decision_reason=record.decision_reason,
        planner_name=record.planner_name,
        planner_strategy=record.planner_strategy,
    )
    if run_metadata is not None:
        payload_copy.setdefault("info", {})
        payload_copy["info"]["run_outcome"] = run_metadata.outcome
        payload_copy["info"]["run_finish_reason"] = run_metadata.finish_reason
    payload_copy["strategic_context"] = strategic_context
    return _DatasetItem(
        record_id=f"{record.session_name}:{record.instance_id}:{record.run_id}:{record.step_index}",
        group_key=_dataset_group_key(
            group_by=group_by,
            record_id=f"{record.session_name}:{record.instance_id}:{record.run_id}:{record.step_index}",
            session_name=record.session_name,
            run_id=record.run_id,
            combat_id=(
                f"{record.session_name}:{record.instance_id}:{record.run_id}:combat:{record.floor}"
                if record.screen_type == "COMBAT"
                else None
            ),
        ),
        session_name=record.session_name,
        session_kind=record.session_kind,
        instance_id=record.instance_id,
        run_id=record.run_id,
        character_id=str(state_run.get("character_id") or "") or None,
        floor=record.floor,
        outcome=run_metadata.outcome if run_metadata is not None else None,
        payload=payload_copy,
        screen_type=record.screen_type,
        decision_source=record.decision_source,
        decision_stage=record.decision_stage,
        decision_reason=record.decision_reason,
        policy_name=record.policy_name,
        policy_pack=record.policy_pack,
        algorithm=record.algorithm,
        reward=record.reward,
        legal_action_count=record.legal_action_count,
        combat_id=(
            f"{record.session_name}:{record.instance_id}:{record.run_id}:combat:{record.floor}"
            if record.screen_type == "COMBAT"
            else None
        ),
        act_id=_normalized_optional_str(state_run.get("act_id")),
        act_index=_normalized_optional_int(state_run.get("act_index")),
        boss_id=_normalized_optional_str(state_run.get("boss_encounter_id")),
        second_boss_id=_normalized_optional_str(state_run.get("second_boss_encounter_id")),
        planner_name=_normalized_optional_str(strategic_context.get("planner_name")),
        planner_strategy=_normalized_optional_str(strategic_context.get("planner_strategy")),
        route_reason_tags=tuple(str(tag) for tag in strategic_context.get("route_reason_tags", [])),
        route_profile=_normalized_optional_str(strategic_context.get("route_profile")),
        strategic_context=strategic_context,
    )


def _dataset_group_key(
    *,
    group_by: DatasetSplitGroupBy,
    record_id: str,
    session_name: str,
    run_id: str,
    combat_id: str | None = None,
) -> str:
    if group_by == "record":
        return record_id
    if group_by == "run_id":
        return run_id
    if group_by == "combat_id":
        return combat_id or record_id
    return f"{session_name}:{run_id}"


def _public_run_outcome_label(run_record: PublicRunNormalizedRunRecord) -> str | None:
    if isinstance(run_record.benchmark_slice.get("outcome"), str):
        outcome = str(run_record.benchmark_slice["outcome"]).strip().lower()
        if outcome in {"win", "loss"}:
            return outcome
    if run_record.win is True:
        return "win"
    if run_record.win is False:
        return "loss"
    return None


def _public_run_absolute_floor_map(run_record: PublicRunNormalizedRunRecord) -> dict[int, int]:
    act_lengths: dict[int, int] = {}
    for room in run_record.rooms:
        act_lengths[room.act_index] = max(act_lengths.get(room.act_index, 0), room.floor_within_act)
    offsets: dict[int, int] = {}
    floor_offset = 0
    for act_index in sorted(act_lengths):
        offsets[act_index] = floor_offset
        floor_offset += act_lengths[act_index]
    return {
        room_index: offsets.get(room.act_index, 0) + room.floor_within_act
        for room_index, room in enumerate(run_record.rooms)
    }


def _public_decision_id(
    *,
    run_id: str,
    decision_type: str,
    room_index: int,
    decision_index: int,
    chosen_action: str,
) -> str:
    normalized_action = chosen_action.strip().replace(" ", "_")
    return f"{run_id}:{decision_type}:{room_index}:{decision_index}:{normalized_action}"


def _public_decision_stage(decision_type: str) -> str:
    if decision_type == "reward_card_pick":
        return "reward"
    if decision_type == "shop_buy":
        return "shop"
    if decision_type.startswith("selection_"):
        return "selection"
    if decision_type == "event_choice":
        return "event"
    return "rest"


def _unique_non_empty_strings(values: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values or ():
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _item_matches_filters(item: _DatasetItem, filters: DatasetFilterSpec) -> bool:
    if filters.session_kinds and item.session_kind not in filters.session_kinds:
        return False
    if filters.session_names and item.session_name not in filters.session_names:
        return False
    if filters.instance_ids and item.instance_id not in filters.instance_ids:
        return False
    if filters.character_ids:
        if item.character_id is None or item.character_id not in filters.character_ids:
            return False
    if filters.act_ids:
        if item.act_id is None or item.act_id not in filters.act_ids:
            return False
    if filters.boss_ids:
        if item.boss_id is None or item.boss_id not in filters.boss_ids:
            return False
    if filters.second_boss_ids:
        if item.second_boss_id is None or item.second_boss_id not in filters.second_boss_ids:
            return False
    if filters.outcomes:
        if item.outcome is None or item.outcome not in filters.outcomes:
            return False
    if filters.screen_types:
        if item.screen_type is None or item.screen_type not in filters.screen_types:
            return False
    if filters.decision_sources:
        if item.decision_source is None or item.decision_source not in filters.decision_sources:
            return False
    if filters.decision_stages:
        if item.decision_stage is None or item.decision_stage not in filters.decision_stages:
            return False
    if filters.decision_reasons:
        if item.decision_reason is None or item.decision_reason not in filters.decision_reasons:
            return False
    if filters.policy_names:
        if item.policy_name is None or item.policy_name not in filters.policy_names:
            return False
    if filters.policy_packs:
        if item.policy_pack is None or item.policy_pack not in filters.policy_packs:
            return False
    if filters.algorithms:
        if item.algorithm is None or item.algorithm not in filters.algorithms:
            return False
    if filters.planner_names:
        if item.planner_name is None or item.planner_name not in filters.planner_names:
            return False
    if filters.planner_strategies:
        if item.planner_strategy is None or item.planner_strategy not in filters.planner_strategies:
            return False
    if filters.route_reason_tags:
        if not item.route_reason_tags or not set(filters.route_reason_tags).issubset(set(item.route_reason_tags)):
            return False
    if filters.route_profiles:
        if item.route_profile is None or item.route_profile not in filters.route_profiles:
            return False
    if filters.build_ids:
        if item.build_id is None or item.build_id not in filters.build_ids:
            return False
    if filters.source_names:
        if item.source_name is None or item.source_name not in filters.source_names:
            return False
    if filters.decision_types:
        if item.decision_type is None or item.decision_type not in filters.decision_types:
            return False
    if filters.support_qualities:
        if item.support_quality is None or item.support_quality not in filters.support_qualities:
            return False
    if filters.min_confidence is not None and (
        item.reconstruction_confidence is None or item.reconstruction_confidence < filters.min_confidence
    ):
        return False
    if filters.min_floor is not None and (item.floor is None or item.floor < filters.min_floor):
        return False
    if filters.max_floor is not None and (item.floor is None or item.floor > filters.max_floor):
        return False
    if filters.min_reward is not None and (item.reward is None or item.reward < filters.min_reward):
        return False
    if filters.max_reward is not None and (item.reward is None or item.reward > filters.max_reward):
        return False
    if filters.min_legal_actions is not None and (
        item.legal_action_count is None or item.legal_action_count < filters.min_legal_actions
    ):
        return False
    return True


def _build_offline_rl_items(
    manifest: DatasetManifest,
    source_files: Sequence[Path],
) -> _DatasetBuildResult:
    encoder, action_space = _offline_rl_components()
    items: list[_DatasetItem] = []
    episode_records: list[dict[str, Any]] = []
    feature_stats = _FeatureStatsAccumulator(feature_names=encoder.feature_names)
    source_record_count = 0
    filtered_out_count = 0

    for source_file in source_files:
        run_metadata = _extract_trajectory_run_metadata(source_file)
        steps_by_run: dict[str, list[TrajectoryStepRecord]] = defaultdict(list)
        for _, payload in _iter_jsonl_payloads(source_file):
            source_record_count += 1
            if payload.get("record_type") != "step":
                filtered_out_count += 1
                continue
            record = TrajectoryStepRecord.model_validate(payload)
            steps_by_run[record.run_id].append(record)

        for run_id in sorted(steps_by_run):
            built_items, episode_payload, dropped_count = _build_offline_rl_episode_items(
                run_id=run_id,
                records=steps_by_run[run_id],
                manifest=manifest,
                run_metadata=run_metadata.get(run_id),
                source_file=source_file,
                encoder=encoder,
                action_space=action_space,
                feature_stats=feature_stats,
            )
            filtered_out_count += dropped_count
            items.extend(built_items)
            if episode_payload is not None:
                episode_records.append(episode_payload)

    if not items:
        raise ValueError("Dataset build produced zero records after filtering.")

    return _DatasetBuildResult(
        items=items,
        source_record_count=source_record_count,
        filtered_out_count=filtered_out_count,
        extras={
            "episode_records": episode_records,
            "feature_stats": feature_stats.as_dict(
                action_schema=action_space.schema_payload(),
                feature_schema=encoder.schema_payload(),
            ),
        },
    )


def _build_offline_rl_episode_items(
    *,
    run_id: str,
    records: Sequence[TrajectoryStepRecord],
    manifest: DatasetManifest,
    run_metadata: _TrajectoryRunMetadata | None,
    source_file: Path,
    encoder,
    action_space,
    feature_stats,
) -> tuple[list[_DatasetItem], dict[str, Any] | None, int]:
    sorted_records = sorted(records, key=lambda item: item.step_index)
    built_items: list[_DatasetItem] = []
    dropped_count = 0
    episode_id = _dataset_group_key(
        group_by="session_run",
        record_id=f"{sorted_records[0].session_name}:{run_id}",
        session_name=sorted_records[0].session_name,
        run_id=run_id,
    )

    for transition_index, record in enumerate(sorted_records, start=1):
        next_record = sorted_records[transition_index] if transition_index < len(sorted_records) else None
        item = _build_offline_rl_transition_item(
            record=record,
            next_record=next_record,
            transition_index=transition_index,
            group_by=manifest.split.group_by,
            episode_id=episode_id,
            run_metadata=run_metadata,
            source_file=source_file,
            encoder=encoder,
            action_space=action_space,
            feature_stats=feature_stats,
        )
        if item is None:
            dropped_count += 1
            continue
        if not _item_matches_filters(item, manifest.filters):
            dropped_count += 1
            continue
        built_items.append(item)

    if not built_items:
        return [], None, dropped_count

    episode_payload = _offline_rl_episode_payload(
        episode_id=episode_id,
        items=built_items,
        run_id=run_id,
        run_metadata=run_metadata,
    )
    return built_items, episode_payload, dropped_count


def _build_offline_rl_transition_item(
    *,
    record: TrajectoryStepRecord,
    next_record: TrajectoryStepRecord | None,
    transition_index: int,
    group_by: DatasetSplitGroupBy,
    episode_id: str,
    run_metadata: _TrajectoryRunMetadata | None,
    source_file: Path,
    encoder,
    action_space,
    feature_stats,
) -> _DatasetItem | None:
    current_observation = _step_observation_from_record(record)
    next_observation = _next_step_observation(record, next_record)
    state_run = record.state_summary.get("run") or {}
    strategic_context = _strategic_context_from_sources(
        summary=record.state_summary,
        decision_metadata=record.decision_metadata,
        decision_reason=record.decision_reason,
        planner_name=record.planner_name,
        planner_strategy=record.planner_strategy,
    )
    transition_id = f"{record.session_name}:{record.instance_id}:{record.run_id}:{record.step_index}"
    next_transition_id = (
        f"{next_record.session_name}:{next_record.instance_id}:{next_record.run_id}:{next_record.step_index}"
        if next_record is not None
        else None
    )

    current_mask: list[bool] = []
    current_action_index: int | None = None
    current_legal_action_ids = list(record.legal_action_ids)
    feature_vector: list[float] = []
    action_space_name: str | None = None
    action_schema_version: int | None = None
    feature_space_name: str | None = None
    feature_schema_version: int | None = None

    if current_observation is not None and current_observation.screen_type == "COMBAT":
        binding = action_space.bind(current_observation)
        current_mask = list(binding.mask)
        current_legal_action_ids = [candidate.action_id for candidate in current_observation.legal_actions]
        current_action_index = _action_index_from_binding(binding.candidates, record.chosen_action_id)
        action_space_name = action_space.action_space_name
        action_schema_version = action_space.action_schema_version
        feature_space_name = encoder.feature_space_name
        feature_schema_version = encoder.feature_schema_version
        if current_action_index is not None:
            feature_vector = encoder.encode(current_observation)
            feature_stats.observe(feature_vector)

    next_mask: list[bool] | None = None
    next_feature_vector: list[float] | None = None
    next_legal_action_ids = _next_legal_action_ids(record, next_record)
    if (
        current_action_index is not None
        and next_observation is not None
        and next_observation.screen_type == "COMBAT"
        and not record.terminated
        and not record.truncated
    ):
        next_binding = action_space.bind(next_observation)
        next_mask = list(next_binding.mask)
        next_feature_vector = encoder.encode(next_observation)
        next_legal_action_ids = [candidate.action_id for candidate in next_observation.legal_actions]

    offline_done = bool(
        record.terminated
        or record.truncated
        or next_record is None
        or current_action_index is None
        or next_feature_vector is None
    )
    offline_truncated = bool(record.truncated or (not record.terminated and next_record is None))

    transition = OfflineRlTransitionRecord(
        transition_id=transition_id,
        episode_id=episode_id,
        session_name=record.session_name,
        session_kind=record.session_kind,
        instance_id=record.instance_id,
        run_id=record.run_id,
        character_id=str(state_run.get("character_id") or "") or None,
        floor=record.floor,
        step_index=record.step_index,
        transition_index=transition_index,
        screen_type=record.screen_type,
        decision_stage=record.decision_stage,
        decision_source=record.decision_source,
        policy_name=record.policy_name,
        policy_pack=record.policy_pack,
        algorithm=record.algorithm,
        run_outcome=run_metadata.outcome if run_metadata is not None else _optional_info_value(record, "run_outcome"),
        run_finish_reason=(
            run_metadata.finish_reason if run_metadata is not None else _optional_info_value(record, "run_finish_reason")
        ),
        action_space_name=action_space_name,
        action_schema_version=action_schema_version,
        feature_space_name=feature_space_name,
        feature_schema_version=feature_schema_version,
        action_supported=current_action_index is not None,
        action_index=current_action_index,
        chosen_action_id=record.chosen_action_id,
        chosen_action_label=record.chosen_action_label,
        chosen_action_source=record.chosen_action_source,
        legal_action_count=len(current_legal_action_ids),
        legal_action_ids=current_legal_action_ids,
        action_mask=current_mask,
        reward=record.reward,
        done=offline_done,
        truncated=offline_truncated,
        environment_terminated=record.terminated,
        environment_truncated=record.truncated,
        next_transition_id=next_transition_id,
        next_screen_type=_next_screen_type(record, next_record),
        next_floor=_next_floor(record, next_record),
        next_legal_action_count=len(next_legal_action_ids) if (next_record is not None or next_legal_action_ids) else None,
        next_legal_action_ids=next_legal_action_ids,
        next_action_mask=next_mask,
        feature_vector=feature_vector,
        next_feature_vector=next_feature_vector,
        state_summary=record.state_summary,
        next_state_summary=_next_state_summary(record, next_record),
        strategic_context=strategic_context,
    )
    return _DatasetItem(
        record_id=transition.transition_id,
        group_key=_dataset_group_key(
            group_by=group_by,
            record_id=transition.transition_id,
            session_name=transition.session_name,
            run_id=transition.episode_id if group_by == "record" else transition.run_id,
        ),
        session_name=transition.session_name,
        session_kind=transition.session_kind,
        instance_id=transition.instance_id,
        run_id=transition.run_id,
        character_id=transition.character_id,
        floor=transition.floor,
        outcome=transition.run_outcome,
        payload=transition.model_dump(mode="json"),
        feature_names=tuple(encoder.feature_names) if transition.action_supported else (),
        screen_type=transition.screen_type,
        decision_source=transition.decision_source,
        decision_stage=transition.decision_stage,
        decision_reason=record.decision_reason,
        policy_name=transition.policy_name,
        policy_pack=transition.policy_pack,
        algorithm=transition.algorithm,
        reward=transition.reward,
        legal_action_count=transition.legal_action_count,
        act_id=_normalized_optional_str(state_run.get("act_id")),
        act_index=_normalized_optional_int(state_run.get("act_index")),
        boss_id=_normalized_optional_str(state_run.get("boss_encounter_id")),
        second_boss_id=_normalized_optional_str(state_run.get("second_boss_encounter_id")),
        planner_name=_normalized_optional_str(strategic_context.get("planner_name")),
        planner_strategy=_normalized_optional_str(strategic_context.get("planner_strategy")),
        route_reason_tags=tuple(str(tag) for tag in strategic_context.get("route_reason_tags", [])),
        route_profile=_normalized_optional_str(strategic_context.get("route_profile")),
        strategic_context=strategic_context,
    )


def _offline_rl_episode_payload(
    *,
    episode_id: str,
    items: Sequence[_DatasetItem],
    run_id: str,
    run_metadata: _TrajectoryRunMetadata | None,
) -> dict[str, Any]:
    transition_payloads = [OfflineRlTransitionRecord.model_validate(item.payload) for item in items]
    legal_action_counts = [payload.legal_action_count for payload in transition_payloads]
    rewards = [payload.reward for payload in transition_payloads]
    discounted_return = 0.0
    gamma = 0.97
    for index, reward in enumerate(rewards):
        discounted_return += (gamma**index) * reward
    episode = OfflineRlEpisodeRecord(
        episode_id=episode_id,
        session_name=transition_payloads[0].session_name,
        session_kind=transition_payloads[0].session_kind,
        instance_id=transition_payloads[0].instance_id,
        run_id=run_id,
        character_id=transition_payloads[0].character_id,
        run_outcome=run_metadata.outcome if run_metadata is not None else transition_payloads[-1].run_outcome,
        run_finish_reason=(
            run_metadata.finish_reason if run_metadata is not None else transition_payloads[-1].run_finish_reason
        ),
        first_floor=transition_payloads[0].floor,
        last_floor=transition_payloads[-1].floor,
        transition_count=len(transition_payloads),
        supported_transition_count=sum(1 for payload in transition_payloads if payload.action_supported),
        return_value=sum(rewards),
        discounted_return=discounted_return,
        mean_reward=(sum(rewards) / len(rewards)) if rewards else None,
        screen_histogram=dict(Counter(payload.screen_type for payload in transition_payloads)),
        decision_stage_histogram=dict(
            Counter(payload.decision_stage for payload in transition_payloads if payload.decision_stage is not None)
        ),
        decision_source_histogram=dict(
            Counter(payload.decision_source for payload in transition_payloads if payload.decision_source is not None)
        ),
        action_space_histogram=dict(
            Counter(payload.action_space_name for payload in transition_payloads if payload.action_space_name is not None)
        ),
        legal_action_count_stats=_basic_stats([float(value) for value in legal_action_counts]),
        first_transition_id=transition_payloads[0].transition_id,
        last_transition_id=transition_payloads[-1].transition_id,
    )
    return episode.model_dump(mode="json")


def _step_observation_from_record(record: TrajectoryStepRecord | None) -> StepObservation | None:
    if record is None:
        return None
    state = _safe_game_state(record.state)
    descriptors = _safe_action_descriptors(record.action_descriptors)
    if state is None or descriptors is None:
        return None
    build_result = build_candidate_actions(state, descriptors)
    candidates = list(build_result.candidates)
    candidate_by_id = {candidate.action_id: candidate for candidate in candidates}
    if record.legal_action_ids and all(action_id in candidate_by_id for action_id in record.legal_action_ids):
        ordered_candidates = [candidate_by_id[action_id] for action_id in record.legal_action_ids]
    else:
        ordered_candidates = candidates
    return StepObservation(
        screen_type=state.screen,
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=ordered_candidates,
        build_warnings=list(build_result.unsupported_actions),
    )


def _next_step_observation(
    record: TrajectoryStepRecord,
    next_record: TrajectoryStepRecord | None,
) -> StepObservation | None:
    if next_record is not None:
        return _step_observation_from_record(next_record)
    next_state = _safe_game_state(record.next_state)
    next_descriptors = _safe_action_descriptors(record.next_action_descriptors)
    if next_state is None or next_descriptors is None:
        return None
    build_result = build_candidate_actions(next_state, next_descriptors)
    candidates = list(build_result.candidates)
    candidate_by_id = {candidate.action_id: candidate for candidate in candidates}
    if record.next_legal_action_ids and all(action_id in candidate_by_id for action_id in record.next_legal_action_ids):
        ordered_candidates = [candidate_by_id[action_id] for action_id in record.next_legal_action_ids]
    else:
        ordered_candidates = candidates
    return StepObservation(
        screen_type=next_state.screen,
        run_id=next_state.run_id,
        state=next_state,
        action_descriptors=next_descriptors,
        legal_actions=ordered_candidates,
        build_warnings=list(build_result.unsupported_actions),
    )


def _next_legal_action_ids(record: TrajectoryStepRecord, next_record: TrajectoryStepRecord | None) -> list[str]:
    if next_record is not None:
        return list(next_record.legal_action_ids)
    return list(record.next_legal_action_ids)


def _next_state_summary(
    record: TrajectoryStepRecord,
    next_record: TrajectoryStepRecord | None,
) -> dict[str, Any] | None:
    if next_record is not None:
        return next_record.state_summary
    return record.next_state_summary or None


def _next_screen_type(record: TrajectoryStepRecord, next_record: TrajectoryStepRecord | None) -> str | None:
    if next_record is not None:
        return next_record.screen_type
    return record.next_screen_type


def _next_floor(record: TrajectoryStepRecord, next_record: TrajectoryStepRecord | None) -> int | None:
    if next_record is not None:
        return next_record.floor
    return record.next_floor


def _safe_game_state(payload: dict[str, Any] | None) -> GameStatePayload | None:
    if not isinstance(payload, dict) or not payload:
        return None
    try:
        return GameStatePayload.model_validate(payload)
    except Exception:
        return None


def _safe_action_descriptors(payload: dict[str, Any] | None) -> AvailableActionsPayload | None:
    if not isinstance(payload, dict) or not payload:
        return None
    try:
        return AvailableActionsPayload.model_validate(payload)
    except Exception:
        return None


def _action_index_from_binding(candidates: Sequence[CandidateAction | None], action_id: str | None) -> int | None:
    if action_id is None:
        return None
    for index, candidate in enumerate(candidates):
        if candidate is not None and candidate.action_id == action_id:
            return index
    return None


def _optional_info_value(record: TrajectoryStepRecord, key: str) -> str | None:
    value = (record.info or {}).get(key)
    return str(value) if value is not None else None


def _offline_rl_components():
    from sts2_rl.train.combat_encoder import CombatStateEncoder
    from sts2_rl.train.combat_space import CombatActionSpace

    return CombatStateEncoder(), CombatActionSpace()


class _FeatureStatsAccumulator:
    def __init__(self, *, feature_names: Sequence[str]) -> None:
        self._feature_names = list(feature_names)
        self._count = 0
        self._mins = [float("inf")] * len(self._feature_names)
        self._maxs = [float("-inf")] * len(self._feature_names)
        self._sums = [0.0] * len(self._feature_names)
        self._sum_squares = [0.0] * len(self._feature_names)

    def observe(self, values: Sequence[float]) -> None:
        if len(values) != len(self._feature_names):
            raise ValueError("Feature stats observation length does not match feature schema.")
        self._count += 1
        for index, value in enumerate(values):
            numeric = float(value)
            self._mins[index] = min(self._mins[index], numeric)
            self._maxs[index] = max(self._maxs[index], numeric)
            self._sums[index] += numeric
            self._sum_squares[index] += numeric * numeric

    def as_dict(self, *, action_schema: dict[str, Any], feature_schema: dict[str, Any]) -> dict[str, Any]:
        feature_stats = []
        for index, name in enumerate(self._feature_names):
            if self._count == 0:
                mean = None
                variance = None
                min_value = None
                max_value = None
            else:
                mean = self._sums[index] / self._count
                variance = max(0.0, (self._sum_squares[index] / self._count) - (mean * mean))
                min_value = self._mins[index]
                max_value = self._maxs[index]
            feature_stats.append(
                {
                    "index": index,
                    "name": name,
                    "count": self._count,
                    "min": min_value,
                    "mean": mean,
                    "std": variance**0.5 if variance is not None else None,
                    "max": max_value,
                }
            )
        return {
            "schema_version": OFFLINE_RL_SCHEMA_VERSION,
            "feature_observation_count": self._count,
            "action_schema": action_schema,
            "feature_schema": feature_schema,
            "feature_stats": feature_stats,
        }


def _assign_splits(items: Sequence[_DatasetItem], split: DatasetSplitSpec) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        grouped.setdefault(item.group_key, []).append(index)

    group_keys = sorted(grouped)
    random.Random(split.seed).shuffle(group_keys)
    total_groups = len(group_keys)

    train_count = int(round(total_groups * split.train_fraction))
    validation_count = int(round(total_groups * split.validation_fraction))
    test_count = total_groups - train_count - validation_count

    if train_count <= 0:
        train_count = 1
    while train_count + validation_count + test_count > total_groups:
        if test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            train_count -= 1
    while train_count + validation_count + test_count < total_groups:
        test_count += 1

    split_groups = {
        "train": group_keys[:train_count],
        "validation": group_keys[train_count : train_count + validation_count],
        "test": group_keys[train_count + validation_count : train_count + validation_count + test_count],
    }
    assignments: dict[str, list[int]] = {}
    for split_name, keys in split_groups.items():
        indices: list[int] = []
        for key in keys:
            indices.extend(grouped[key])
        assignments[split_name] = sorted(indices)
    return assignments


def _records_filename(dataset_kind: DatasetKind) -> str:
    if dataset_kind == "predictor_combat_outcomes":
        return PREDICTOR_EXAMPLES_FILENAME
    if dataset_kind == "trajectory_steps":
        return TRAJECTORY_STEPS_FILENAME
    if dataset_kind == "shadow_combat_encounters":
        return SHADOW_COMBAT_ENCOUNTERS_FILENAME
    if dataset_kind == "public_strategic_decisions":
        return PUBLIC_STRATEGIC_DECISIONS_FILENAME
    return OFFLINE_RL_TRANSITIONS_FILENAME


def _split_records_filename(split_name: str, dataset_kind: DatasetKind) -> str:
    if dataset_kind == "predictor_combat_outcomes":
        stem = "examples"
    elif dataset_kind == "trajectory_steps":
        stem = "steps"
    elif dataset_kind == "shadow_combat_encounters":
        stem = "encounters"
    elif dataset_kind == "public_strategic_decisions":
        stem = "strategic-decisions"
    else:
        stem = "transitions"
    return f"{split_name}.{stem}.jsonl"


def _write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def _write_predictor_feature_table(path: Path, items: Sequence[_DatasetItem]) -> None:
    feature_names = sorted({name for item in items for name in item.feature_names})
    fieldnames = [
        "example_id",
        "session_name",
        "session_kind",
        "instance_id",
        "run_id",
        "floor",
        "combat_index",
        "outcome",
        "outcome_win_label",
        "reward_label",
        "damage_delta_label",
    ] + feature_names
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            payload = item.payload
            row = {
                "example_id": payload["example_id"],
                "session_name": payload.get("session_name", ""),
                "session_kind": payload.get("session_kind", ""),
                "instance_id": payload.get("instance_id", ""),
                "run_id": payload.get("run_id", ""),
                "floor": payload.get("floor"),
                "combat_index": payload.get("combat_index"),
                "outcome": payload.get("outcome"),
                "outcome_win_label": payload.get("outcome_win_label"),
                "reward_label": payload.get("reward_label"),
                "damage_delta_label": payload.get("damage_delta_label"),
            }
            feature_map = payload.get("feature_map", {})
            for feature_name in feature_names:
                row[feature_name] = feature_map.get(feature_name, 0.0)
            writer.writerow(row)


def _write_trajectory_steps_table(path: Path, items: Sequence[_DatasetItem]) -> None:
    fieldnames = [
        "session_name",
        "session_kind",
        "instance_id",
        "step_index",
        "run_id",
        "run_outcome",
        "run_finish_reason",
        "screen_type",
        "floor",
        "chosen_action_id",
        "chosen_action_source",
        "policy_pack",
        "policy_handler",
        "decision_source",
        "decision_stage",
        "decision_reason",
        "act_id",
        "boss_id",
        "route_profile",
        "reward",
        "terminated",
        "truncated",
        "planner_name",
        "planner_strategy",
        "route_path_length",
        "selected_route_score",
        "ranked_action_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            payload = item.payload
            writer.writerow(
                {
                    "session_name": payload.get("session_name", ""),
                    "session_kind": payload.get("session_kind", ""),
                    "instance_id": payload.get("instance_id", ""),
                    "step_index": payload.get("step_index"),
                    "run_id": payload.get("run_id", ""),
                    "run_outcome": (payload.get("info") or {}).get("run_outcome"),
                    "run_finish_reason": (payload.get("info") or {}).get("run_finish_reason"),
                    "screen_type": payload.get("screen_type", ""),
                    "floor": payload.get("floor"),
                    "chosen_action_id": payload.get("chosen_action_id"),
                    "chosen_action_source": payload.get("chosen_action_source"),
                    "policy_pack": payload.get("policy_pack"),
                    "policy_handler": payload.get("policy_handler"),
                    "decision_source": payload.get("decision_source"),
                    "decision_stage": payload.get("decision_stage"),
                    "decision_reason": payload.get("decision_reason"),
                    "act_id": (payload.get("strategic_context") or {}).get("act_id"),
                    "boss_id": (payload.get("strategic_context") or {}).get("boss_id"),
                    "route_profile": (payload.get("strategic_context") or {}).get("route_profile"),
                    "reward": payload.get("reward"),
                    "terminated": payload.get("terminated"),
                    "truncated": payload.get("truncated"),
                    "planner_name": payload.get("planner_name"),
                    "planner_strategy": payload.get("planner_strategy"),
                    "route_path_length": (payload.get("strategic_context") or {}).get("route_path_length"),
                    "selected_route_score": (payload.get("strategic_context") or {}).get("selected_route_score"),
                    "ranked_action_count": payload.get("ranked_action_count"),
                }
            )


def _write_shadow_combat_encounters_table(path: Path, items: Sequence[_DatasetItem]) -> None:
    fieldnames = [
        "encounter_id",
        "session_name",
        "session_kind",
        "instance_id",
        "run_id",
        "observed_seed",
        "character_id",
        "floor",
        "combat_index",
        "outcome",
        "outcome_reason",
        "encounter_family",
        "enemy_ids",
        "legal_action_count",
        "action_trace_count",
        "start_player_hp",
        "end_player_hp",
        "start_enemy_hp",
        "end_enemy_hp",
        "damage_dealt",
        "damage_taken",
        "act_id",
        "boss_id",
        "route_profile",
        "planner_name",
        "planner_strategy",
        "selected_route_score",
        "has_full_snapshot",
        "has_terminal_outcome",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            payload = item.payload
            writer.writerow(
                {
                    "encounter_id": payload.get("encounter_id"),
                    "session_name": payload.get("session_name"),
                    "session_kind": payload.get("session_kind"),
                    "instance_id": payload.get("instance_id"),
                    "run_id": payload.get("run_id"),
                    "observed_seed": payload.get("observed_seed"),
                    "character_id": payload.get("character_id"),
                    "floor": payload.get("floor"),
                    "combat_index": payload.get("combat_index"),
                    "outcome": payload.get("outcome"),
                    "outcome_reason": payload.get("outcome_reason"),
                    "encounter_family": payload.get("encounter_family"),
                    "enemy_ids": ",".join(payload.get("enemy_ids", [])),
                    "legal_action_count": payload.get("legal_action_count"),
                    "action_trace_count": payload.get("action_trace_count"),
                    "start_player_hp": payload.get("start_player_hp"),
                    "end_player_hp": payload.get("end_player_hp"),
                    "start_enemy_hp": payload.get("start_enemy_hp"),
                    "end_enemy_hp": payload.get("end_enemy_hp"),
                    "damage_dealt": payload.get("damage_dealt"),
                    "damage_taken": payload.get("damage_taken"),
                    "act_id": (payload.get("strategic_context") or {}).get("act_id"),
                    "boss_id": (payload.get("strategic_context") or {}).get("boss_id"),
                    "route_profile": (payload.get("strategic_context") or {}).get("route_profile"),
                    "planner_name": (payload.get("strategic_context") or {}).get("planner_name"),
                    "planner_strategy": (payload.get("strategic_context") or {}).get("planner_strategy"),
                    "selected_route_score": (payload.get("strategic_context") or {}).get("selected_route_score"),
                    "has_full_snapshot": payload.get("has_full_snapshot"),
                    "has_terminal_outcome": payload.get("has_terminal_outcome"),
                }
            )


def _write_public_strategic_decisions_table(path: Path, items: Sequence[_DatasetItem]) -> None:
    fieldnames = [
        "decision_id",
        "source_name",
        "snapshot_date",
        "source_run_id",
        "run_id",
        "character_id",
        "ascension",
        "build_id",
        "game_version",
        "branch",
        "content_channel",
        "game_mode",
        "platform_type",
        "run_outcome",
        "act_id",
        "act_index",
        "floor",
        "floor_within_act",
        "room_type",
        "map_point_type",
        "model_id",
        "decision_type",
        "decision_stage",
        "support_quality",
        "reconstruction_confidence",
        "source_type",
        "candidate_count",
        "candidate_actions",
        "chosen_action",
        "alternate_actions",
        "chosen_present_in_candidates",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            payload = item.payload
            writer.writerow(
                {
                    "decision_id": payload.get("decision_id"),
                    "source_name": payload.get("source_name"),
                    "snapshot_date": payload.get("snapshot_date"),
                    "source_run_id": payload.get("source_run_id"),
                    "run_id": payload.get("run_id"),
                    "character_id": payload.get("character_id"),
                    "ascension": payload.get("ascension"),
                    "build_id": payload.get("build_id"),
                    "game_version": payload.get("game_version"),
                    "branch": payload.get("branch"),
                    "content_channel": payload.get("content_channel"),
                    "game_mode": payload.get("game_mode"),
                    "platform_type": payload.get("platform_type"),
                    "run_outcome": payload.get("run_outcome"),
                    "act_id": payload.get("act_id"),
                    "act_index": payload.get("act_index"),
                    "floor": payload.get("floor"),
                    "floor_within_act": payload.get("floor_within_act"),
                    "room_type": payload.get("room_type"),
                    "map_point_type": payload.get("map_point_type"),
                    "model_id": payload.get("model_id"),
                    "decision_type": payload.get("decision_type"),
                    "decision_stage": item.decision_stage,
                    "support_quality": payload.get("support_quality"),
                    "reconstruction_confidence": payload.get("reconstruction_confidence"),
                    "source_type": payload.get("source_type"),
                    "candidate_count": len(payload.get("candidate_actions", [])),
                    "candidate_actions": ",".join(payload.get("candidate_actions", [])),
                    "chosen_action": payload.get("chosen_action"),
                    "alternate_actions": ",".join(payload.get("alternate_actions", [])),
                    "chosen_present_in_candidates": payload.get("chosen_present_in_candidates"),
                }
            )


def _write_offline_rl_transition_table(path: Path, items: Sequence[_DatasetItem]) -> None:
    fieldnames = [
        "transition_id",
        "episode_id",
        "session_name",
        "session_kind",
        "instance_id",
        "run_id",
        "character_id",
        "floor",
        "step_index",
        "transition_index",
        "screen_type",
        "decision_stage",
        "decision_source",
        "policy_name",
        "policy_pack",
        "algorithm",
        "act_id",
        "boss_id",
        "route_profile",
        "planner_name",
        "planner_strategy",
        "run_outcome",
        "run_finish_reason",
        "action_space_name",
        "action_index",
        "action_supported",
        "chosen_action_id",
        "legal_action_count",
        "reward",
        "done",
        "truncated",
        "environment_terminated",
        "environment_truncated",
        "next_transition_id",
        "next_screen_type",
        "next_floor",
        "feature_count",
        "next_feature_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            payload = item.payload
            writer.writerow(
                {
                    "transition_id": payload.get("transition_id"),
                    "episode_id": payload.get("episode_id"),
                    "session_name": payload.get("session_name"),
                    "session_kind": payload.get("session_kind"),
                    "instance_id": payload.get("instance_id"),
                    "run_id": payload.get("run_id"),
                    "character_id": payload.get("character_id"),
                    "floor": payload.get("floor"),
                    "step_index": payload.get("step_index"),
                    "transition_index": payload.get("transition_index"),
                    "screen_type": payload.get("screen_type"),
                    "decision_stage": payload.get("decision_stage"),
                    "decision_source": payload.get("decision_source"),
                    "policy_name": payload.get("policy_name"),
                    "policy_pack": payload.get("policy_pack"),
                    "algorithm": payload.get("algorithm"),
                    "act_id": (payload.get("strategic_context") or {}).get("act_id"),
                    "boss_id": (payload.get("strategic_context") or {}).get("boss_id"),
                    "route_profile": (payload.get("strategic_context") or {}).get("route_profile"),
                    "planner_name": (payload.get("strategic_context") or {}).get("planner_name"),
                    "planner_strategy": (payload.get("strategic_context") or {}).get("planner_strategy"),
                    "run_outcome": payload.get("run_outcome"),
                    "run_finish_reason": payload.get("run_finish_reason"),
                    "action_space_name": payload.get("action_space_name"),
                    "action_index": payload.get("action_index"),
                    "action_supported": payload.get("action_supported"),
                    "chosen_action_id": payload.get("chosen_action_id"),
                    "legal_action_count": payload.get("legal_action_count"),
                    "reward": payload.get("reward"),
                    "done": payload.get("done"),
                    "truncated": payload.get("truncated"),
                    "environment_terminated": payload.get("environment_terminated"),
                    "environment_truncated": payload.get("environment_truncated"),
                    "next_transition_id": payload.get("next_transition_id"),
                    "next_screen_type": payload.get("next_screen_type"),
                    "next_floor": payload.get("next_floor"),
                    "feature_count": len(payload.get("feature_vector", [])),
                    "next_feature_count": len(payload.get("next_feature_vector") or []),
                }
            )


def _resolved_manifest_payload(manifest: DatasetManifest, source_files: Sequence[Path]) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    payload["schema_version"] = DATASET_MANIFEST_SCHEMA_VERSION
    payload["resolved_source_files"] = [str(path) for path in source_files]
    return payload


def _dataset_summary_payload(
    *,
    manifest: DatasetManifest,
    output_dir: Path,
    records_filename: str,
    items: Sequence[_DatasetItem],
    source_files: Sequence[Path],
    source_record_count: int,
    filtered_out_count: int,
    split_paths: dict[str, Path],
    split_counts: dict[str, int],
    split_group_counts: dict[str, int],
    extras: dict[str, Any],
) -> dict[str, Any]:
    strategic_payload = _strategic_summary_payload(items)
    payload: dict[str, Any] = {
        "schema_version": DATASET_SUMMARY_SCHEMA_VERSION,
        "dataset_name": manifest.dataset_name,
        "dataset_kind": manifest.dataset_kind,
        "description": manifest.description,
        "output_dir": str(output_dir),
        "records_path": str(output_dir / records_filename),
        "manifest_path": str(output_dir / DATASET_MANIFEST_FILENAME),
        "record_count": len(items),
        "source_file_count": len(source_files),
        "source_record_count": source_record_count,
        "filtered_out_count": filtered_out_count,
        "included_record_count": len(items),
        "source_files": [str(path) for path in source_files],
        "split": {
            "train_fraction": manifest.split.train_fraction,
            "validation_fraction": manifest.split.validation_fraction,
            "test_fraction": manifest.split.test_fraction,
            "seed": manifest.split.seed,
            "group_by": manifest.split.group_by,
            "split_counts": split_counts,
            "split_group_counts": split_group_counts,
            "split_paths": {name: str(path) for name, path in split_paths.items()},
        },
        "filters": manifest.filters.model_dump(mode="json"),
        "output": manifest.output.model_dump(mode="json"),
        "lineage": {
            "source_paths": [source.path for source in manifest.sources],
            "resolved_source_paths": [str(Path(source.path).expanduser().resolve()) for source in manifest.sources],
            "source_kinds": [source.source_kind for source in manifest.sources],
            "resolved_source_files": [str(path) for path in source_files],
        },
        "session_kind_histogram": dict(Counter(item.session_kind for item in items)),
        "instance_histogram": dict(Counter(item.instance_id for item in items)),
        "character_histogram": dict(
            Counter(item.character_id for item in items if item.character_id is not None)
        ),
        "floor_stats": _basic_stats([float(item.floor) for item in items if item.floor is not None]),
        "strategic": strategic_payload,
        "strategic_coverage": dict(strategic_payload.get("coverage", {})),
        "act_histogram": dict(strategic_payload.get("act_histogram", {})),
        "boss_histogram": dict(strategic_payload.get("boss_histogram", {})),
        "second_boss_histogram": dict(strategic_payload.get("second_boss_histogram", {})),
        "planner_name_histogram": dict(strategic_payload.get("planner_name_histogram", {})),
        "planner_strategy_histogram": dict(strategic_payload.get("planner_strategy_histogram", {})),
        "decision_reason_histogram": dict(strategic_payload.get("decision_reason_histogram", {})),
        "route_profile_histogram": dict(strategic_payload.get("route_profile_histogram", {})),
        "route_reason_tag_histogram": dict(strategic_payload.get("route_reason_tag_histogram", {})),
    }

    if manifest.dataset_kind == "predictor_combat_outcomes":
        feature_names = sorted({name for item in items for name in item.feature_names})
        payload.update(
            {
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "outcome_histogram": dict(Counter(item.outcome for item in items if item.outcome is not None)),
                "reward_label_stats": _basic_stats(
                    [float(item.payload.get("reward_label", 0.0)) for item in items]
                ),
                "damage_delta_label_stats": _basic_stats(
                    [float(item.payload.get("damage_delta_label", 0.0)) for item in items]
                ),
                "effective_label_counts": {
                    "outcome": sum(1 for item in items if float(item.payload.get("outcome_weight", 0.0)) > 0.0),
                    "reward": sum(1 for item in items if float(item.payload.get("reward_weight", 0.0)) > 0.0),
                    "damage_delta": sum(1 for item in items if float(item.payload.get("damage_weight", 0.0)) > 0.0),
                },
                "exports": {
                    "feature_table_csv": (
                        str(output_dir / PREDICTOR_FEATURE_TABLE_FILENAME) if manifest.output.export_csv else None
                    ),
                },
            }
        )
    elif manifest.dataset_kind == "trajectory_steps":
        payload.update(
            {
                "screen_histogram": dict(Counter(item.screen_type for item in items if item.screen_type is not None)),
                "decision_source_histogram": dict(
                    Counter(item.decision_source for item in items if item.decision_source is not None)
                ),
                "run_outcome_histogram": dict(Counter(item.outcome for item in items if item.outcome is not None)),
                "exports": {
                    "steps_table_csv": (
                        str(output_dir / TRAJECTORY_STEPS_TABLE_FILENAME) if manifest.output.export_csv else None
                    ),
                },
            }
        )
    elif manifest.dataset_kind == "shadow_combat_encounters":
        payload.update(
            {
                "feature_count": 0,
                "feature_names": [],
                "outcome_histogram": dict(Counter(item.outcome for item in items if item.outcome is not None)),
                "encounter_family_histogram": dict(
                    Counter(item.payload.get("encounter_family") for item in items if item.payload.get("encounter_family") is not None)
                ),
                "screen_histogram": {"COMBAT": len(items)},
                "legal_action_count_stats": _basic_stats(
                    [float(item.legal_action_count) for item in items if item.legal_action_count is not None]
                ),
                "action_trace_count_stats": _basic_stats(
                    [float(item.payload.get("action_trace_count", 0) or 0) for item in items]
                ),
                "player_hp_stats": {
                    "start": _basic_stats(
                        [float(item.payload.get("start_player_hp")) for item in items if item.payload.get("start_player_hp") is not None]
                    ),
                    "end": _basic_stats(
                        [float(item.payload.get("end_player_hp")) for item in items if item.payload.get("end_player_hp") is not None]
                    ),
                },
                "enemy_hp_stats": {
                    "start": _basic_stats(
                        [float(item.payload.get("start_enemy_hp")) for item in items if item.payload.get("start_enemy_hp") is not None]
                    ),
                    "end": _basic_stats(
                        [float(item.payload.get("end_enemy_hp")) for item in items if item.payload.get("end_enemy_hp") is not None]
                    ),
                },
                "snapshot_coverage": {
                    "full_snapshot_count": sum(1 for item in items if item.payload.get("has_full_snapshot")),
                    "terminal_outcome_count": sum(1 for item in items if item.payload.get("has_terminal_outcome")),
                },
                "action_id_histogram": _merge_counter_payload(
                    item.payload.get("action_id_histogram", {}) for item in items
                ),
                "exports": {
                    "encounters_table_csv": (
                        str(output_dir / SHADOW_COMBAT_ENCOUNTERS_TABLE_FILENAME) if manifest.output.export_csv else None
                    ),
                },
            }
        )
    elif manifest.dataset_kind == "public_strategic_decisions":
        decision_payloads = [PublicStrategicDecisionRecord.model_validate(item.payload) for item in items]
        payload.update(
            {
                "feature_count": 0,
                "feature_names": [],
                "outcome_histogram": dict(Counter(item.outcome for item in items if item.outcome is not None)),
                "screen_histogram": {"PUBLIC_STRATEGIC": len(items)},
                "decision_source_histogram": {"public_run": len(items)},
                "decision_type_histogram": dict(
                    Counter(record.decision_type for record in decision_payloads)
                ),
                "support_quality_histogram": dict(
                    Counter(record.support_quality for record in decision_payloads)
                ),
                "source_name_histogram": dict(
                    Counter(record.source_name for record in decision_payloads)
                ),
                "build_id_histogram": dict(
                    Counter(record.build_id for record in decision_payloads if record.build_id is not None)
                ),
                "game_version_histogram": dict(
                    Counter(record.game_version for record in decision_payloads if record.game_version is not None)
                ),
                "branch_histogram": dict(
                    Counter(record.branch for record in decision_payloads if record.branch is not None)
                ),
                "content_channel_histogram": dict(
                    Counter(record.content_channel for record in decision_payloads if record.content_channel is not None)
                ),
                "decision_stage_histogram": dict(
                    Counter(item.decision_stage for item in items if item.decision_stage is not None)
                ),
                "room_type_histogram": dict(
                    Counter(record.room_type for record in decision_payloads)
                ),
                "map_point_type_histogram": dict(
                    Counter(record.map_point_type for record in decision_payloads if record.map_point_type is not None)
                ),
                "source_type_histogram": dict(
                    Counter(record.source_type for record in decision_payloads if record.source_type is not None)
                ),
                "confidence_stats": _basic_stats(
                    [float(record.reconstruction_confidence) for record in decision_payloads]
                ),
                "candidate_count_stats": _basic_stats(
                    [float(len(record.candidate_actions)) for record in decision_payloads]
                ),
                "candidate_coverage": {
                    "full_candidate_count": sum(1 for record in decision_payloads if record.support_quality == "full_candidates"),
                    "chosen_only_count": sum(1 for record in decision_payloads if record.support_quality == "chosen_only"),
                    "candidate_present_count": sum(1 for record in decision_payloads if record.candidate_actions),
                },
                "exports": {
                    "strategic_decisions_table_csv": (
                        str(output_dir / PUBLIC_STRATEGIC_DECISIONS_TABLE_FILENAME)
                        if manifest.output.export_csv
                        else None
                    ),
                },
            }
        )
    else:
        transition_payloads = [OfflineRlTransitionRecord.model_validate(item.payload) for item in items]
        episode_records = [OfflineRlEpisodeRecord.model_validate(item) for item in extras.get("episode_records", [])]
        feature_stats = dict(extras.get("feature_stats", {}))
        supported_transitions = [payload for payload in transition_payloads if payload.action_supported]
        action_support_counter: Counter[str] = Counter()
        chosen_action_counter: Counter[str] = Counter()
        action_schema = feature_stats.get("action_schema", {})
        slot_labels = list(action_schema.get("slot_labels", []))
        for transition in supported_transitions:
            chosen_index = transition.action_index
            if chosen_index is not None and 0 <= chosen_index < len(slot_labels):
                chosen_action_counter[slot_labels[chosen_index]] += 1
            for index, enabled in enumerate(transition.action_mask):
                if enabled and 0 <= index < len(slot_labels):
                    action_support_counter[slot_labels[index]] += 1
        payload.update(
            {
                "feature_count": len(feature_stats.get("feature_schema", {}).get("feature_names", [])),
                "feature_names": list(feature_stats.get("feature_schema", {}).get("feature_names", [])),
                "feature_schema_version": feature_stats.get("feature_schema", {}).get("feature_schema_version"),
                "action_schema_version": feature_stats.get("action_schema", {}).get("action_schema_version"),
                "episode_count": len(episode_records),
                "supported_transition_count": len(supported_transitions),
                "screen_histogram": dict(Counter(item.screen_type for item in items if item.screen_type is not None)),
                "decision_source_histogram": dict(
                    Counter(item.decision_source for item in items if item.decision_source is not None)
                ),
                "decision_stage_histogram": dict(
                    Counter(item.decision_stage for item in items if item.decision_stage is not None)
                ),
                "policy_name_histogram": dict(Counter(item.policy_name for item in items if item.policy_name is not None)),
                "policy_pack_histogram": dict(Counter(item.policy_pack for item in items if item.policy_pack is not None)),
                "algorithm_histogram": dict(Counter(item.algorithm for item in items if item.algorithm is not None)),
                "run_outcome_histogram": dict(Counter(item.outcome for item in items if item.outcome is not None)),
                "reward_stats": _basic_stats([transition.reward for transition in transition_payloads]),
                "return_stats": _basic_stats([episode.return_value for episode in episode_records]),
                "discounted_return_stats": _basic_stats([episode.discounted_return for episode in episode_records]),
                "transition_horizon_stats": _basic_stats([float(episode.transition_count) for episode in episode_records]),
                "legal_action_count_stats": _basic_stats(
                    [float(transition.legal_action_count) for transition in transition_payloads]
                ),
                "action_support_histogram": dict(action_support_counter),
                "chosen_action_histogram": dict(chosen_action_counter),
                "normalization": {
                    "feature_stats_path": str(output_dir / OFFLINE_RL_FEATURE_STATS_FILENAME),
                    "feature_observation_count": feature_stats.get("feature_observation_count", 0),
                    "feature_schema": feature_stats.get("feature_schema", {}),
                    "action_schema": feature_stats.get("action_schema", {}),
                },
                "exports": {
                    "transitions_table_csv": (
                        str(output_dir / OFFLINE_RL_TRANSITIONS_TABLE_FILENAME) if manifest.output.export_csv else None
                    ),
                    "episodes_path": str(output_dir / OFFLINE_RL_EPISODES_FILENAME),
                    "feature_stats_path": str(output_dir / OFFLINE_RL_FEATURE_STATS_FILENAME),
                    "episode_split_paths": {
                        split_name: str(output_dir / f"{split_name}.episodes.jsonl")
                        for split_name in DATASET_SPLIT_NAMES
                        if manifest.output.write_split_files
                    },
                },
            }
        )
    return payload


def _iter_jsonl_payloads(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            yield line_number, json.loads(line)


def _predictor_example_from_combat_outcome(
    payload: dict[str, Any],
    *,
    source_path: Path,
    source_record_index: int,
) -> PredictorExample:
    start_summary = dict(payload.get("start_summary", {}))
    end_summary = dict(payload.get("end_summary", {}))
    outcome = str(payload.get("outcome", "interrupted"))
    outcome_weight = 1.0 if outcome in {"won", "lost"} else 0.0
    outcome_win_label = 1.0 if outcome == "won" else 0.0 if outcome == "lost" else None
    reward_label = float(payload.get("cumulative_reward", 0.0))
    damage_dealt = float(payload.get("damage_dealt", 0.0))
    damage_taken = float(payload.get("damage_taken", 0.0))
    damage_delta_label = damage_dealt - damage_taken
    run_id = str(payload.get("run_id", "run_unknown"))
    combat_index = int(payload.get("combat_index", 0))
    session_name = str(payload.get("session_name", ""))
    instance_id = str(payload.get("instance_id", ""))

    feature_map = extract_feature_map_from_summary(start_summary)
    example_id = f"{session_name}:{instance_id}:{run_id}:{combat_index}:{source_record_index}"
    strategic_context = _strategic_context_from_sources(summary=start_summary)

    metadata = {
        "record_type": payload.get("record_type"),
        "timestamp_utc": payload.get("timestamp_utc"),
        "step_count": int(payload.get("step_count", 0) or 0),
        "started_step_index": int(payload.get("started_step_index", 0) or 0),
        "finished_step_index": int(payload.get("finished_step_index", 0) or 0),
        "reason": payload.get("reason"),
        "damage_dealt": damage_dealt,
        "damage_taken": damage_taken,
        "strategic_context": strategic_context,
    }

    floor = payload.get("floor")
    return PredictorExample(
        example_id=example_id,
        source_path=str(source_path),
        source_record_index=source_record_index,
        session_name=session_name,
        session_kind=str(payload.get("session_kind", "")),
        instance_id=instance_id,
        run_id=run_id,
        floor=int(floor) if floor is not None else None,
        combat_index=combat_index,
        outcome=outcome,
        outcome_win_label=outcome_win_label,
        reward_label=reward_label,
        damage_delta_label=damage_delta_label,
        outcome_weight=outcome_weight,
        reward_weight=outcome_weight,
        damage_weight=outcome_weight,
        enemy_ids=[str(enemy_id) for enemy_id in payload.get("enemy_ids", [])],
        feature_map=feature_map,
        start_summary=start_summary,
        end_summary=end_summary,
        strategic_context=strategic_context,
        metadata=metadata,
    )


def _strategic_context_from_sources(
    *,
    summary: dict[str, Any] | None,
    decision_metadata: dict[str, Any] | None = None,
    decision_reason: str | None = None,
    planner_name: str | None = None,
    planner_strategy: str | None = None,
) -> dict[str, Any]:
    summary_payload = summary if isinstance(summary, dict) else {}
    run_summary = dict(summary_payload.get("run", {}))
    map_summary = dict(summary_payload.get("map", {}))
    route_plan = map_summary.get("route_plan")
    route_plan_payload = dict(route_plan) if isinstance(route_plan, dict) else {}
    decision_metadata_payload = decision_metadata if isinstance(decision_metadata, dict) else {}
    route_planner = decision_metadata_payload.get("route_planner")
    route_planner_payload = dict(route_planner) if isinstance(route_planner, dict) else {}
    selected_route = route_planner_payload.get("selected")
    selected_route_payload = dict(selected_route) if isinstance(selected_route, dict) else route_plan_payload

    planned_node_types = _normalized_string_list(
        map_summary.get("planned_node_types")
        or selected_route_payload.get("path_node_types")
        or route_plan_payload.get("path_node_types")
    )
    selected_path = selected_route_payload.get("path")
    route_path_length = None
    if isinstance(selected_path, list):
        route_path_length = len(selected_path)
    elif planned_node_types:
        route_path_length = len(planned_node_types)

    route_reason_tags = _normalized_string_list(selected_route_payload.get("reason_tags") or route_plan_payload.get("reason_tags"))
    normalized_planner_name = (
        _normalized_optional_str(route_planner_payload.get("planner_name"))
        or _normalized_optional_str(planner_name)
    )
    normalized_planner_strategy = (
        _normalized_optional_str(route_planner_payload.get("planner_strategy"))
        or _normalized_optional_str(planner_strategy)
    )
    normalized_decision_reason = _normalized_optional_str(decision_reason)

    strategic_context = {
        "act_id": _normalized_optional_str(run_summary.get("act_id")),
        "act_index": _normalized_optional_int(run_summary.get("act_index")),
        "boss_id": _normalized_optional_str(run_summary.get("boss_encounter_id")),
        "second_boss_id": _normalized_optional_str(run_summary.get("second_boss_encounter_id")),
        "planner_name": normalized_planner_name,
        "planner_strategy": normalized_planner_strategy,
        "decision_reason": normalized_decision_reason,
        "current_to_boss_distance": _normalized_optional_int(map_summary.get("current_to_boss_distance")),
        "frontier_to_boss_min_distance": _normalized_optional_int(map_summary.get("frontier_to_boss_min_distance")),
        "planned_first_rest_distance": _normalized_optional_int(
            map_summary.get("planned_first_rest_distance") or selected_route_payload.get("first_rest_distance")
        ),
        "planned_first_shop_distance": _normalized_optional_int(
            map_summary.get("planned_first_shop_distance") or selected_route_payload.get("first_shop_distance")
        ),
        "planned_first_elite_distance": _normalized_optional_int(
            map_summary.get("planned_first_elite_distance") or selected_route_payload.get("first_elite_distance")
        ),
        "planned_rest_count": _normalized_optional_int(
            map_summary.get("planned_rest_count") or selected_route_payload.get("rest_count")
        ),
        "planned_shop_count": _normalized_optional_int(
            map_summary.get("planned_shop_count") or selected_route_payload.get("shop_count")
        ),
        "planned_elite_count": _normalized_optional_int(
            map_summary.get("planned_elite_count") or selected_route_payload.get("elite_count")
        ),
        "planned_event_count": _normalized_optional_int(selected_route_payload.get("event_count")),
        "planned_treasure_count": _normalized_optional_int(selected_route_payload.get("treasure_count")),
        "planned_monster_count": _normalized_optional_int(selected_route_payload.get("monster_count")),
        "planned_elites_before_rest": _normalized_optional_int(selected_route_payload.get("elites_before_rest")),
        "remaining_distance_to_boss": _normalized_optional_int(selected_route_payload.get("remaining_distance_to_boss")),
        "selected_route_score": _normalized_optional_float(selected_route_payload.get("score")),
        "route_path_length": route_path_length,
        "route_reason_tags": route_reason_tags,
        "planned_node_types": planned_node_types,
        "route_profile": _route_profile_label(
            reason_tags=route_reason_tags,
            planned_node_types=planned_node_types,
            planner_strategy=normalized_planner_strategy,
        ),
        "has_route_plan": bool(route_plan_payload or planned_node_types),
        "has_route_planner_trace": bool(route_planner_payload),
    }
    return {key: value for key, value in strategic_context.items() if value is not None and value != []}


def _normalized_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalized_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalized_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized_value = _normalized_optional_str(value)
        if normalized_value is None or normalized_value in seen:
            continue
        seen.add(normalized_value)
        normalized.append(normalized_value)
    return normalized


def _route_profile_label(
    *,
    reason_tags: Sequence[str],
    planned_node_types: Sequence[str],
    planner_strategy: str | None,
) -> str | None:
    normalized_reason_tags = sorted({_slug_token(tag) for tag in reason_tags if _slug_token(tag)})
    if normalized_reason_tags:
        return "+".join(normalized_reason_tags)
    normalized_node_types = [_slug_token(node_type) for node_type in planned_node_types[:3] if _slug_token(node_type)]
    if normalized_node_types:
        return "path:" + ">".join(normalized_node_types)
    if planner_strategy is not None:
        return "planner:" + _slug_token(planner_strategy)
    return None


def _slug_token(value: Any) -> str:
    normalized = _normalized_optional_str(value)
    if normalized is None:
        return ""
    return normalized.lower().replace(" ", "_")


def _merge_counter_payload(histograms) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for histogram in histograms:
        if not isinstance(histogram, dict):
            continue
        for key, value in histogram.items():
            merged[str(key)] += int(value)
    return dict(merged)


def _extract_total_enemy_hp(value: Any) -> int | None:
    if isinstance(value, list):
        numeric_values = [_normalized_optional_int(item) for item in value]
        filtered = [item for item in numeric_values if item is not None]
        if not filtered:
            return None
        return sum(filtered)
    return _normalized_optional_int(value)


def _extract_end_player_hp(summary: dict[str, Any]) -> int | None:
    combat_summary = dict(summary.get("combat", {}))
    run_summary = dict(summary.get("run", {}))
    return _normalized_optional_int(combat_summary.get("player_hp") or run_summary.get("current_hp"))


def _extract_end_enemy_hp(summary: dict[str, Any]) -> int | None:
    combat_summary = dict(summary.get("combat", {}))
    return _extract_total_enemy_hp(combat_summary.get("enemy_hp"))


def _encounter_family_label(enemy_ids: Sequence[str]) -> str | None:
    normalized = [enemy_id for enemy_id in _normalized_string_list(list(enemy_ids)) if enemy_id]
    if not normalized:
        return None
    return "+".join(normalized)


def _fingerprint_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _strategic_summary_payload(items: Sequence[_DatasetItem]) -> dict[str, Any]:
    strategic_items = [item for item in items if item.strategic_context]
    route_plan_items = [
        item for item in strategic_items if bool(item.strategic_context.get("has_route_plan"))
    ]
    route_trace_items = [
        item for item in strategic_items if bool(item.strategic_context.get("has_route_planner_trace"))
    ]
    planned_node_type_histogram: Counter[str] = Counter()
    route_reason_tag_histogram: Counter[str] = Counter()
    for item in strategic_items:
        planned_node_type_histogram.update(
            str(node_type) for node_type in item.strategic_context.get("planned_node_types", [])
        )
        route_reason_tag_histogram.update(item.route_reason_tags)

    return {
        "coverage": {
            "strategic_context_count": len(strategic_items),
            "strategic_context_fraction": (len(strategic_items) / len(items)) if items else None,
            "act_id_count": sum(1 for item in items if item.act_id is not None),
            "boss_id_count": sum(1 for item in items if item.boss_id is not None),
            "planner_name_count": sum(1 for item in items if item.planner_name is not None),
            "planner_strategy_count": sum(1 for item in items if item.planner_strategy is not None),
            "route_plan_count": len(route_plan_items),
            "route_plan_fraction": (len(route_plan_items) / len(items)) if items else None,
            "route_planner_trace_count": len(route_trace_items),
            "route_planner_trace_fraction": (len(route_trace_items) / len(items)) if items else None,
        },
        "act_histogram": dict(Counter(item.act_id for item in items if item.act_id is not None)),
        "act_index_histogram": dict(
            Counter(str(item.act_index) for item in items if item.act_index is not None)
        ),
        "boss_histogram": dict(Counter(item.boss_id for item in items if item.boss_id is not None)),
        "second_boss_histogram": dict(
            Counter(item.second_boss_id for item in items if item.second_boss_id is not None)
        ),
        "planner_name_histogram": dict(
            Counter(item.planner_name for item in items if item.planner_name is not None)
        ),
        "planner_strategy_histogram": dict(
            Counter(item.planner_strategy for item in items if item.planner_strategy is not None)
        ),
        "decision_reason_histogram": dict(
            Counter(item.decision_reason for item in items if item.decision_reason is not None)
        ),
        "route_profile_histogram": dict(
            Counter(item.route_profile for item in items if item.route_profile is not None)
        ),
        "route_reason_tag_histogram": dict(route_reason_tag_histogram),
        "planned_node_type_histogram": dict(planned_node_type_histogram),
        "current_to_boss_distance_stats": _basic_stats(
            [
                float(item.strategic_context["current_to_boss_distance"])
                for item in strategic_items
                if item.strategic_context.get("current_to_boss_distance") is not None
            ]
        ),
        "frontier_to_boss_min_distance_stats": _basic_stats(
            [
                float(item.strategic_context["frontier_to_boss_min_distance"])
                for item in strategic_items
                if item.strategic_context.get("frontier_to_boss_min_distance") is not None
            ]
        ),
        "route_path_length_stats": _basic_stats(
            [
                float(item.strategic_context["route_path_length"])
                for item in route_plan_items
                if item.strategic_context.get("route_path_length") is not None
            ]
        ),
        "selected_route_score_stats": _basic_stats(
            [
                float(item.strategic_context["selected_route_score"])
                for item in route_trace_items
                if item.strategic_context.get("selected_route_score") is not None
            ]
        ),
        "planned_rest_count_stats": _basic_stats(
            [
                float(item.strategic_context["planned_rest_count"])
                for item in route_plan_items
                if item.strategic_context.get("planned_rest_count") is not None
            ]
        ),
        "planned_shop_count_stats": _basic_stats(
            [
                float(item.strategic_context["planned_shop_count"])
                for item in route_plan_items
                if item.strategic_context.get("planned_shop_count") is not None
            ]
        ),
        "planned_elite_count_stats": _basic_stats(
            [
                float(item.strategic_context["planned_elite_count"])
                for item in route_plan_items
                if item.strategic_context.get("planned_elite_count") is not None
            ]
        ),
        "planned_first_rest_distance_stats": _basic_stats(
            [
                float(item.strategic_context["planned_first_rest_distance"])
                for item in route_plan_items
                if item.strategic_context.get("planned_first_rest_distance") is not None
            ]
        ),
        "planned_first_shop_distance_stats": _basic_stats(
            [
                float(item.strategic_context["planned_first_shop_distance"])
                for item in route_plan_items
                if item.strategic_context.get("planned_first_shop_distance") is not None
            ]
        ),
        "planned_first_elite_distance_stats": _basic_stats(
            [
                float(item.strategic_context["planned_first_elite_distance"])
                for item in route_plan_items
                if item.strategic_context.get("planned_first_elite_distance") is not None
            ]
        ),
    }


def _basic_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _extract_trajectory_run_metadata(source_file: Path) -> dict[str, _TrajectoryRunMetadata]:
    metadata: dict[str, _TrajectoryRunMetadata] = {}
    for _, payload in _iter_jsonl_payloads(source_file):
        if payload.get("record_type") != "run_finished":
            continue
        run_id = str(payload.get("run_id") or "")
        if not run_id:
            continue
        metadata[run_id] = _TrajectoryRunMetadata(
            outcome=str(payload.get("outcome")) if payload.get("outcome") is not None else None,
            finish_reason=str(payload.get("reason")) if payload.get("reason") is not None else None,
        )
    return metadata
