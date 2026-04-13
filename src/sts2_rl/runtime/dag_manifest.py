from __future__ import annotations

import json
import tomllib
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sts2_rl.data import validate_dataset_manifest
from sts2_rl.predict.reports import PredictorBenchmarkComparisonThresholds, PredictorCalibrationThresholds, PredictorRankingThresholds
from sts2_rl.predict.trainer import CombatOutcomePredictorTrainConfig

from .config import load_instance_config
from .job_manifest import load_runtime_job_manifest
from .manifest import build_instance_specs

if TYPE_CHECKING:
    from sts2_rl.train.behavior_cloning import BehaviorCloningFloorBandWeight, BehaviorCloningTrainConfig
    from sts2_rl.train.offline_cql import OfflineCqlTrainConfig

EXPERIMENT_DAG_SCHEMA_VERSION = 1
EXPERIMENT_DAG_SUMMARY_FILENAME = "dag-summary.json"
EXPERIMENT_DAG_STATE_FILENAME = "dag-state.json"
EXPERIMENT_DAG_LOG_FILENAME = "dag-log.jsonl"
EXPERIMENT_DAG_RESOLVED_MANIFEST_FILENAME = "dag-manifest.resolved.json"


class ExperimentDagModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PredictorTrainConfigSpec(ExperimentDagModel):
    epochs: int = 250
    learning_rate: float = 0.05
    l2: float = 0.0005
    validation_fraction: float = 0.2
    seed: int = 0

    @model_validator(mode="after")
    def validate_predictor_train_config(self) -> PredictorTrainConfigSpec:
        self.to_runtime_config()
        return self

    def to_runtime_config(self) -> CombatOutcomePredictorTrainConfig:
        return CombatOutcomePredictorTrainConfig(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            l2=self.l2,
            validation_fraction=self.validation_fraction,
            seed=self.seed,
        )


class PredictorCalibrationThresholdsSpec(ExperimentDagModel):
    outcome_ece_max: float = 0.12
    outcome_brier_max: float = 0.25
    reward_rmse_max: float = 3.0
    damage_rmse_max: float = 24.0

    def to_runtime_config(self) -> PredictorCalibrationThresholds:
        return PredictorCalibrationThresholds(
            outcome_ece_max=self.outcome_ece_max,
            outcome_brier_max=self.outcome_brier_max,
            reward_rmse_max=self.reward_rmse_max,
            damage_rmse_max=self.damage_rmse_max,
        )


class PredictorRankingThresholdsSpec(ExperimentDagModel):
    outcome_pairwise_accuracy_min: float = 0.58
    reward_pairwise_accuracy_min: float = 0.58
    damage_pairwise_accuracy_min: float = 0.58
    reward_ndcg_at_3_min: float = 0.62
    damage_ndcg_at_3_min: float = 0.62

    def to_runtime_config(self) -> PredictorRankingThresholds:
        return PredictorRankingThresholds(
            outcome_pairwise_accuracy_min=self.outcome_pairwise_accuracy_min,
            reward_pairwise_accuracy_min=self.reward_pairwise_accuracy_min,
            damage_pairwise_accuracy_min=self.damage_pairwise_accuracy_min,
            reward_ndcg_at_3_min=self.reward_ndcg_at_3_min,
            damage_ndcg_at_3_min=self.damage_ndcg_at_3_min,
        )


class PredictorBenchmarkComparisonThresholdsSpec(ExperimentDagModel):
    delta_total_reward_min: float = 0.0
    delta_combat_win_rate_min: float = 0.0

    def to_runtime_config(self) -> PredictorBenchmarkComparisonThresholds:
        return PredictorBenchmarkComparisonThresholds(
            delta_total_reward_min=self.delta_total_reward_min,
            delta_combat_win_rate_min=self.delta_combat_win_rate_min,
        )


class BehaviorCloningFloorBandWeightSpec(ExperimentDagModel):
    min_floor: int | None = None
    max_floor: int | None = None
    weight: float = 1.0

    def to_runtime_config(self) -> BehaviorCloningFloorBandWeight:
        from sts2_rl.train.behavior_cloning import BehaviorCloningFloorBandWeight

        return BehaviorCloningFloorBandWeight(
            min_floor=self.min_floor,
            max_floor=self.max_floor,
            weight=self.weight,
        )


class BehaviorCloningTrainConfigSpec(ExperimentDagModel):
    epochs: int = 40
    learning_rate: float = 0.035
    l2: float = 0.0001
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0
    include_stages: list[str] = Field(default_factory=list)
    include_decision_sources: list[str] = Field(default_factory=list)
    include_policy_packs: list[str] = Field(default_factory=list)
    include_policy_names: list[str] = Field(default_factory=list)
    min_floor: int | None = None
    max_floor: int | None = None
    min_legal_actions: int = 2
    top_k: list[int] = Field(default_factory=lambda: [1, 3])
    stage_weights: dict[str, float] = Field(default_factory=dict)
    decision_source_weights: dict[str, float] = Field(default_factory=dict)
    policy_pack_weights: dict[str, float] = Field(default_factory=dict)
    policy_name_weights: dict[str, float] = Field(default_factory=dict)
    run_outcome_weights: dict[str, float] = Field(default_factory=dict)
    floor_band_weights: list[BehaviorCloningFloorBandWeightSpec] = Field(default_factory=list)
    benchmark_manifest_path: Path | None = None
    live_base_url: str | None = None
    live_eval_max_env_steps: int | None = 0
    live_eval_max_runs: int | None = 1
    live_eval_max_combats: int | None = 0
    live_eval_poll_interval_seconds: float = 0.25
    live_eval_max_idle_polls: int = 40
    live_eval_request_timeout_seconds: float = 30.0

    @model_validator(mode="after")
    def validate_behavior_cloning_config(self) -> BehaviorCloningTrainConfigSpec:
        self.to_runtime_config()
        return self

    def to_runtime_config(self) -> BehaviorCloningTrainConfig:
        from sts2_rl.train.behavior_cloning import BehaviorCloningTrainConfig

        return BehaviorCloningTrainConfig(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            l2=self.l2,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            seed=self.seed,
            include_stages=tuple(self.include_stages),
            include_decision_sources=tuple(self.include_decision_sources),
            include_policy_packs=tuple(self.include_policy_packs),
            include_policy_names=tuple(self.include_policy_names),
            min_floor=self.min_floor,
            max_floor=self.max_floor,
            min_legal_actions=self.min_legal_actions,
            top_k=tuple(self.top_k),
            stage_weights=dict(self.stage_weights),
            decision_source_weights=dict(self.decision_source_weights),
            policy_pack_weights=dict(self.policy_pack_weights),
            policy_name_weights=dict(self.policy_name_weights),
            run_outcome_weights=dict(self.run_outcome_weights),
            floor_band_weights=tuple(item.to_runtime_config() for item in self.floor_band_weights),
            benchmark_manifest_path=None if self.benchmark_manifest_path is None else Path(self.benchmark_manifest_path),
            live_base_url=self.live_base_url,
            live_eval_max_env_steps=self.live_eval_max_env_steps,
            live_eval_max_runs=self.live_eval_max_runs,
            live_eval_max_combats=self.live_eval_max_combats,
            live_eval_poll_interval_seconds=self.live_eval_poll_interval_seconds,
            live_eval_max_idle_polls=self.live_eval_max_idle_polls,
            live_eval_request_timeout_seconds=self.live_eval_request_timeout_seconds,
        )


class OfflineCqlTrainConfigSpec(ExperimentDagModel):
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.97
    huber_delta: float = 1.0
    hidden_sizes: list[int] = Field(default_factory=lambda: [64, 64])
    l2: float = 0.0001
    conservative_alpha: float = 1.0
    conservative_temperature: float = 1.0
    target_sync_interval: int = 50
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0
    early_stopping_patience: int = 8
    include_action_space_names: list[str] = Field(default_factory=lambda: ["combat_v1"])
    min_floor: int | None = None
    max_floor: int | None = None
    min_reward: float | None = None
    max_reward: float | None = None
    live_base_url: str | None = None
    live_eval_max_env_steps: int | None = 0
    live_eval_max_runs: int | None = 1
    live_eval_max_combats: int | None = 0
    live_eval_poll_interval_seconds: float = 0.25
    live_eval_max_idle_polls: int = 40
    live_eval_request_timeout_seconds: float = 30.0
    benchmark_manifest_path: Path | None = None

    @model_validator(mode="after")
    def validate_offline_cql_config(self) -> OfflineCqlTrainConfigSpec:
        self.to_runtime_config()
        return self

    def to_runtime_config(self) -> OfflineCqlTrainConfig:
        from sts2_rl.train.offline_cql import OfflineCqlTrainConfig

        return OfflineCqlTrainConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            huber_delta=self.huber_delta,
            hidden_sizes=tuple(self.hidden_sizes),
            l2=self.l2,
            conservative_alpha=self.conservative_alpha,
            conservative_temperature=self.conservative_temperature,
            target_sync_interval=self.target_sync_interval,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
            seed=self.seed,
            early_stopping_patience=self.early_stopping_patience,
            include_action_space_names=tuple(self.include_action_space_names),
            min_floor=self.min_floor,
            max_floor=self.max_floor,
            min_reward=self.min_reward,
            max_reward=self.max_reward,
            live_base_url=self.live_base_url,
            live_eval_max_env_steps=self.live_eval_max_env_steps,
            live_eval_max_runs=self.live_eval_max_runs,
            live_eval_max_combats=self.live_eval_max_combats,
            live_eval_poll_interval_seconds=self.live_eval_poll_interval_seconds,
            live_eval_max_idle_polls=self.live_eval_max_idle_polls,
            live_eval_request_timeout_seconds=self.live_eval_request_timeout_seconds,
            benchmark_manifest_path=None if self.benchmark_manifest_path is None else Path(self.benchmark_manifest_path),
        )


class DagStageBase(ExperimentDagModel):
    stage_id: str
    description: str = ""
    depends_on: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    max_attempts: int = 1
    cooldown_seconds: float = 0.0

    @model_validator(mode="after")
    def validate_stage_controls(self) -> DagStageBase:
        if not self.stage_id.strip():
            raise ValueError("stage_id must not be empty.")
        duplicate_dependencies = [item for item, count in Counter(self.depends_on).items() if count > 1]
        if duplicate_dependencies:
            raise ValueError(
                f"Stage '{self.stage_id}' contains duplicate dependencies: {', '.join(sorted(duplicate_dependencies))}"
            )
        if self.stage_id in self.depends_on:
            raise ValueError(f"Stage '{self.stage_id}' cannot depend on itself.")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1.")
        if self.cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds must be >= 0.")
        return self


class RuntimeJobDagStageSpec(DagStageBase):
    kind: Literal["runtime_job"] = "runtime_job"
    manifest_path: Path
    config_path: Path
    job_name: str | None = None
    output_root: Path | None = None
    replace_existing: bool = False


class DatasetBuildDagStageSpec(DagStageBase):
    kind: Literal["dataset_build"] = "dataset_build"
    manifest_path: Path
    output_dir: Path
    replace_existing: bool = False


class PredictDatasetExtractDagStageSpec(DagStageBase):
    kind: Literal["predict_dataset_extract"] = "predict_dataset_extract"
    sources: list[Path | str]
    output_dir: Path
    replace_existing: bool = False
    split_seed: int = 0
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    split_group_by: str = "session_run"


class PredictTrainDagStageSpec(DagStageBase):
    kind: Literal["predict_train"] = "predict_train"
    dataset_source: Path | str
    output_root: Path
    session_name: str | None = None
    config: PredictorTrainConfigSpec = Field(default_factory=PredictorTrainConfigSpec)


class PredictReportCalibrationDagStageSpec(DagStageBase):
    kind: Literal["predict_report_calibration"] = "predict_report_calibration"
    model_path: Path | str
    dataset_source: Path | str
    output_root: Path
    session_name: str | None = None
    split: str = "validation"
    bin_count: int = 10
    min_slice_examples: int = 5
    thresholds: PredictorCalibrationThresholdsSpec = Field(default_factory=PredictorCalibrationThresholdsSpec)


class PredictReportRankingDagStageSpec(DagStageBase):
    kind: Literal["predict_report_ranking"] = "predict_report_ranking"
    model_path: Path | str
    dataset_source: Path | str
    output_root: Path
    session_name: str | None = None
    split: str = "validation"
    group_by: list[str] = Field(default_factory=list)
    top_k: int = 3
    min_group_size: int = 2
    thresholds: PredictorRankingThresholdsSpec = Field(default_factory=PredictorRankingThresholdsSpec)


class PredictReportCompareDagStageSpec(DagStageBase):
    kind: Literal["predict_report_compare"] = "predict_report_compare"
    sources: list[Path | str]
    output_root: Path
    session_name: str | None = None
    thresholds: PredictorBenchmarkComparisonThresholdsSpec = Field(
        default_factory=PredictorBenchmarkComparisonThresholdsSpec
    )


class BehaviorCloningTrainDagStageSpec(DagStageBase):
    kind: Literal["behavior_cloning_train"] = "behavior_cloning_train"
    dataset_source: Path | str
    output_root: Path
    session_name: str | None = None
    benchmark_suite_name: str | None = None
    config: BehaviorCloningTrainConfigSpec = Field(default_factory=BehaviorCloningTrainConfigSpec)


class OfflineCqlTrainDagStageSpec(DagStageBase):
    kind: Literal["offline_cql_train"] = "offline_cql_train"
    dataset_source: Path | str
    output_root: Path
    session_name: str | None = None
    benchmark_suite_name: str | None = None
    config: OfflineCqlTrainConfigSpec = Field(default_factory=OfflineCqlTrainConfigSpec)


class RegistryRegisterDagStageSpec(DagStageBase):
    kind: Literal["registry_register"] = "registry_register"
    registry_root: Path = Path("artifacts/registry")
    source: Path | str
    experiment_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    aliases: list[str] = Field(default_factory=list)
    replace_existing: bool = False


class RegistryPromoteDagStageSpec(DagStageBase):
    kind: Literal["registry_promote"] = "registry_promote"
    registry_root: Path = Path("artifacts/registry")
    alias_name: str
    experiment: str | None = None
    family: str | None = None
    tag: str | None = None
    benchmark_suite_name: str | None = None
    artifact_path_key: str | None = None
    reason: str | None = None


class RegistryLeaderboardDagStageSpec(DagStageBase):
    kind: Literal["registry_leaderboard"] = "registry_leaderboard"
    registry_root: Path = Path("artifacts/registry")
    output_root: Path | None = None
    session_name: str | None = None
    family: str | None = None
    tag: str | None = None
    benchmark_suite_name: str | None = None


class RegistryCompareDagStageSpec(DagStageBase):
    kind: Literal["registry_compare"] = "registry_compare"
    registry_root: Path = Path("artifacts/registry")
    experiments: list[str]
    output_root: Path | None = None
    session_name: str | None = None

    @model_validator(mode="after")
    def validate_experiments(self) -> RegistryCompareDagStageSpec:
        if len(self.experiments) < 2:
            raise ValueError("Registry compare stages require at least two experiments.")
        return self


ExperimentDagStageSpec = Annotated[
    RuntimeJobDagStageSpec
    | DatasetBuildDagStageSpec
    | PredictDatasetExtractDagStageSpec
    | PredictTrainDagStageSpec
    | PredictReportCalibrationDagStageSpec
    | PredictReportRankingDagStageSpec
    | PredictReportCompareDagStageSpec
    | BehaviorCloningTrainDagStageSpec
    | OfflineCqlTrainDagStageSpec
    | RegistryRegisterDagStageSpec
    | RegistryPromoteDagStageSpec
    | RegistryLeaderboardDagStageSpec
    | RegistryCompareDagStageSpec,
    Field(discriminator="kind"),
]


class ExperimentDagManifest(ExperimentDagModel):
    schema_version: int = EXPERIMENT_DAG_SCHEMA_VERSION
    dag_name: str
    description: str = ""
    output_root: Path = Path("artifacts/dags")
    lock_root: Path = Path("artifacts/orchestration-locks")
    stages: list[ExperimentDagStageSpec]

    @model_validator(mode="after")
    def validate_stages(self) -> ExperimentDagManifest:
        if not self.dag_name.strip():
            raise ValueError("dag_name must not be empty.")
        if not self.stages:
            raise ValueError("Experiment DAG manifest must include at least one stage.")
        stage_ids = [stage.stage_id for stage in self.stages]
        duplicates = [stage_id for stage_id, count in Counter(stage_ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate stage ids are not allowed: {', '.join(sorted(duplicates))}")
        known_stage_ids = set(stage_ids)
        missing_dependencies = sorted(
            {
                dependency
                for stage in self.stages
                for dependency in stage.depends_on
                if dependency not in known_stage_ids
            }
        )
        if missing_dependencies:
            raise ValueError(
                "Unknown stage dependencies are not allowed: " + ", ".join(missing_dependencies)
            )
        topologically_sort_dag(self)
        return self


@dataclass(frozen=True)
class ExperimentDagValidationReport:
    manifest_path: Path | None
    dag_name: str
    stage_count: int
    stage_order: list[str]
    stage_kind_histogram: dict[str, int]
    stage_resource_hints: dict[str, list[str]]


def load_experiment_dag_manifest(path: str | Path) -> ExperimentDagManifest:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment DAG manifest does not exist: {manifest_path}")
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif manifest_path.suffix.lower() == ".toml":
        payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported experiment DAG manifest format: {manifest_path.suffix}")
    manifest = ExperimentDagManifest.model_validate(payload)
    return _resolve_manifest_paths(manifest, base_dir=manifest_path.parent)


def load_experiment_dag_summary(source: str | Path) -> dict[str, object]:
    source_path = Path(source).expanduser().resolve()
    summary_path = source_path / EXPERIMENT_DAG_SUMMARY_FILENAME if source_path.is_dir() else source_path
    if not summary_path.exists():
        raise FileNotFoundError(f"Experiment DAG summary does not exist: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_experiment_dag_state(source: str | Path) -> dict[str, object]:
    source_path = Path(source).expanduser().resolve()
    state_path = source_path / EXPERIMENT_DAG_STATE_FILENAME if source_path.is_dir() else source_path
    if not state_path.exists():
        raise FileNotFoundError(f"Experiment DAG state does not exist: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def topologically_sort_dag(manifest: ExperimentDagManifest) -> list[str]:
    order_index = {stage.stage_id: index for index, stage in enumerate(manifest.stages)}
    in_degree = {stage.stage_id: len(stage.depends_on) for stage in manifest.stages}
    downstream: dict[str, list[str]] = {stage.stage_id: [] for stage in manifest.stages}
    for stage in manifest.stages:
        for dependency in stage.depends_on:
            downstream[dependency].append(stage.stage_id)

    queue = deque(sorted((stage_id for stage_id, degree in in_degree.items() if degree == 0), key=order_index.get))
    stage_order: list[str] = []
    while queue:
        stage_id = queue.popleft()
        stage_order.append(stage_id)
        for follower in sorted(downstream[stage_id], key=order_index.get):
            in_degree[follower] -= 1
            if in_degree[follower] == 0:
                queue.append(follower)

    if len(stage_order) != len(manifest.stages):
        remaining = sorted(stage_id for stage_id, degree in in_degree.items() if degree > 0)
        raise ValueError("Experiment DAG contains a dependency cycle involving: " + ", ".join(remaining))
    return stage_order


def validate_experiment_dag_manifest(
    manifest: ExperimentDagManifest | str | Path,
    *,
    deep: bool = True,
) -> ExperimentDagValidationReport:
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).expanduser().resolve()
        manifest_model = load_experiment_dag_manifest(manifest_path)
    else:
        manifest_path = None
        manifest_model = manifest
    stage_order = topologically_sort_dag(manifest_model)
    if deep:
        for stage in manifest_model.stages:
            _deep_validate_stage(stage)
    return ExperimentDagValidationReport(
        manifest_path=manifest_path,
        dag_name=manifest_model.dag_name,
        stage_count=len(manifest_model.stages),
        stage_order=stage_order,
        stage_kind_histogram=dict(Counter(stage.kind for stage in manifest_model.stages)),
        stage_resource_hints={stage.stage_id: _stage_resource_hints(stage) for stage in manifest_model.stages},
    )


def _resolve_manifest_paths(manifest: ExperimentDagManifest, *, base_dir: Path) -> ExperimentDagManifest:
    payload = _resolve_path_like_values(manifest.model_dump(mode="python"), base_dir=base_dir)
    return ExperimentDagManifest.model_validate(payload)


def _resolve_path_like_values(value: Any, *, base_dir: Path) -> Any:
    if isinstance(value, Path):
        text = str(value)
        if "${" in text:
            return text
        expanded = value.expanduser()
        return expanded.resolve() if expanded.is_absolute() else (base_dir / expanded).resolve()
    if isinstance(value, list):
        return [_resolve_path_like_values(item, base_dir=base_dir) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_path_like_values(item, base_dir=base_dir) for key, item in value.items()}
    return value


def _deep_validate_stage(stage: ExperimentDagStageSpec) -> None:
    if isinstance(stage, RuntimeJobDagStageSpec):
        if _is_templated_path(stage.manifest_path) or _is_templated_path(stage.config_path):
            return
        config = load_instance_config(stage.config_path)
        known_instance_ids = {spec.instance_id for spec in build_instance_specs(config)}
        load_runtime_job_manifest(stage.manifest_path, known_instance_ids=known_instance_ids)
        return
    if isinstance(stage, DatasetBuildDagStageSpec):
        if _is_templated_path(stage.manifest_path):
            return
        validate_dataset_manifest(stage.manifest_path)


def _stage_resource_hints(stage: ExperimentDagStageSpec) -> list[str]:
    hints = list(stage.resources)
    if isinstance(stage, RuntimeJobDagStageSpec):
        hints.append(f"runtime-config:{_stringify_path(stage.config_path)}")
    if isinstance(stage, DatasetBuildDagStageSpec):
        hints.append(f"output:{_stringify_path(stage.output_dir)}")
    if isinstance(stage, PredictDatasetExtractDagStageSpec):
        hints.append(f"output:{_stringify_path(stage.output_dir)}")
    if isinstance(stage, PredictTrainDagStageSpec):
        hints.append(f"output-root:{_stringify_path(stage.output_root)}")
    if isinstance(
        stage,
        (
            PredictReportCalibrationDagStageSpec,
            PredictReportRankingDagStageSpec,
            PredictReportCompareDagStageSpec,
        ),
    ):
        hints.append(f"output-root:{_stringify_path(stage.output_root)}")
    if isinstance(stage, (BehaviorCloningTrainDagStageSpec, OfflineCqlTrainDagStageSpec)):
        hints.append(f"output-root:{_stringify_path(stage.output_root)}")
    if isinstance(
        stage,
        (
            RegistryRegisterDagStageSpec,
            RegistryPromoteDagStageSpec,
            RegistryLeaderboardDagStageSpec,
            RegistryCompareDagStageSpec,
        ),
    ):
        hints.append(f"registry:{_stringify_path(stage.registry_root)}")
    return sorted(dict.fromkeys(hints))


def _is_templated_path(value: Path | str) -> bool:
    return "${" in str(value)


def _stringify_path(value: Path | str | None) -> str:
    return "-" if value is None else str(value)
