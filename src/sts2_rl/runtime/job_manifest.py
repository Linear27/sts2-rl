from __future__ import annotations

import json
import tomllib
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sts2_rl.collect import CommunityPriorRuntimeConfig
from sts2_rl.predict import PredictorRuntimeConfig

from .watchdog import WatchdogPolicy

RUNTIME_JOB_SCHEMA_VERSION = 1
RUNTIME_JOB_SUMMARY_FILENAME = "job-summary.json"
RUNTIME_JOB_LOG_FILENAME = "job-log.jsonl"

NormalizeTarget = Literal["none", "main_menu", "character_select"]


class RuntimeJobModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class JobPredictorSpec(RuntimeJobModel):
    model_path: Path | None = None
    mode: str = "heuristic_only"
    hooks: list[str] = Field(default_factory=list)
    win_probability_weight: float = 1.50
    reward_weight: float = 0.75
    damage_weight: float = 0.50
    assist_heuristic_weight: float = 1.00
    assist_predictor_weight: float = 0.85
    dominant_heuristic_weight: float = 0.35
    dominant_predictor_weight: float = 1.75

    @model_validator(mode="after")
    def validate_predictor(self) -> JobPredictorSpec:
        self.to_runtime_config()
        return self

    def to_runtime_config(self) -> PredictorRuntimeConfig | None:
        if self.mode == "heuristic_only" and self.model_path is None:
            return None
        return PredictorRuntimeConfig(
            model_path=self.model_path,
            mode=self.mode,
            hooks=tuple(self.hooks) if self.hooks else None,
            win_probability_weight=self.win_probability_weight,
            reward_weight=self.reward_weight,
            damage_weight=self.damage_weight,
            assist_heuristic_weight=self.assist_heuristic_weight,
            assist_predictor_weight=self.assist_predictor_weight,
            dominant_heuristic_weight=self.dominant_heuristic_weight,
            dominant_predictor_weight=self.dominant_predictor_weight,
        )


class JobCommunityPriorSpec(RuntimeJobModel):
    source_path: Path
    route_source_path: Path | None = None
    reward_pick_weight: float = 1.15
    selection_pick_weight: float = 1.05
    selection_upgrade_weight: float = 0.55
    selection_remove_weight: float = 0.95
    shop_buy_weight: float = 1.00
    route_weight: float = 0.90
    reward_pick_neutral_rate: float = 0.33
    shop_buy_neutral_rate: float = 0.10
    route_neutral_win_rate: float = 0.50
    pick_rate_scale: float = 3.0
    buy_rate_scale: float = 5.0
    win_delta_scale: float = 12.0
    route_win_rate_scale: float = 8.0
    min_sample_size: int = 40
    route_min_sample_size: int = 30
    max_confidence_sample_size: int = 1200
    max_source_age_days: int | None = None

    @model_validator(mode="after")
    def validate_community_prior(self) -> JobCommunityPriorSpec:
        self.to_runtime_config()
        return self

    def to_runtime_config(self) -> CommunityPriorRuntimeConfig:
        return CommunityPriorRuntimeConfig(
            source_path=self.source_path,
            route_source_path=self.route_source_path,
            reward_pick_weight=self.reward_pick_weight,
            selection_pick_weight=self.selection_pick_weight,
            selection_upgrade_weight=self.selection_upgrade_weight,
            selection_remove_weight=self.selection_remove_weight,
            shop_buy_weight=self.shop_buy_weight,
            route_weight=self.route_weight,
            reward_pick_neutral_rate=self.reward_pick_neutral_rate,
            shop_buy_neutral_rate=self.shop_buy_neutral_rate,
            route_neutral_win_rate=self.route_neutral_win_rate,
            pick_rate_scale=self.pick_rate_scale,
            buy_rate_scale=self.buy_rate_scale,
            win_delta_scale=self.win_delta_scale,
            route_win_rate_scale=self.route_win_rate_scale,
            min_sample_size=self.min_sample_size,
            route_min_sample_size=self.route_min_sample_size,
            max_confidence_sample_size=self.max_confidence_sample_size,
            max_source_age_days=self.max_source_age_days,
        )


class RuntimeJobTaskBase(RuntimeJobModel):
    task_id: str
    description: str = ""
    instance_ids: list[str] = Field(default_factory=list)
    concurrency_limit: int = 0
    max_attempts: int = 2
    normalize_target: NormalizeTarget = "none"
    normalize_poll_interval_seconds: float = 0.25
    normalize_max_idle_polls: int = 40
    normalize_max_steps: int = 8
    request_timeout_seconds: float = 30.0

    @model_validator(mode="after")
    def validate_runtime_controls(self) -> RuntimeJobTaskBase:
        if self.concurrency_limit < 0:
            raise ValueError("concurrency_limit must be >= 0.")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1.")
        if self.normalize_poll_interval_seconds <= 0.0:
            raise ValueError("normalize_poll_interval_seconds must be > 0.")
        if self.normalize_max_idle_polls < 1:
            raise ValueError("normalize_max_idle_polls must be >= 1.")
        if self.normalize_max_steps < 0:
            raise ValueError("normalize_max_steps must be >= 0.")
        if self.request_timeout_seconds <= 0.0:
            raise ValueError("request_timeout_seconds must be > 0.")
        return self


class CollectJobTaskSpec(RuntimeJobTaskBase):
    kind: Literal["collect"] = "collect"
    policy_profile: str = "baseline"
    predictor: JobPredictorSpec = Field(default_factory=JobPredictorSpec)
    community_prior: JobCommunityPriorSpec | None = None
    max_steps_per_instance: int = 200
    max_runs_per_instance: int = 1
    max_combats_per_instance: int = 0
    poll_interval_seconds: float = 0.25
    idle_timeout_seconds: float = 15.0


class EvalCheckpointJobTaskSpec(RuntimeJobTaskBase):
    kind: Literal["eval_checkpoint"] = "eval_checkpoint"
    checkpoint_path: Path
    policy_profile: str = "baseline"
    predictor: JobPredictorSpec = Field(default_factory=JobPredictorSpec)
    community_prior: JobCommunityPriorSpec | None = None
    max_env_steps: int = 64
    max_runs: int = 1
    max_combats: int = 0
    poll_interval_seconds: float = 0.25
    max_idle_polls: int = 40


class EvalPolicyPackJobTaskSpec(RuntimeJobTaskBase):
    kind: Literal["eval_policy_pack"] = "eval_policy_pack"
    policy_profile: str = "baseline"
    predictor: JobPredictorSpec = Field(default_factory=JobPredictorSpec)
    community_prior: JobCommunityPriorSpec | None = None
    max_env_steps: int = 64
    max_runs: int = 1
    max_combats: int = 0
    poll_interval_seconds: float = 0.25
    max_idle_polls: int = 40


class CompareJobTaskSpec(RuntimeJobTaskBase):
    kind: Literal["compare"] = "compare"
    baseline_checkpoint_path: Path
    candidate_checkpoint_path: Path
    repeats: int = 3
    max_env_steps: int = 0
    max_runs: int = 1
    max_combats: int = 0
    poll_interval_seconds: float = 0.25
    max_idle_polls: int = 40
    prepare_target: NormalizeTarget = "main_menu"
    prepare_max_steps: int = 8
    prepare_max_idle_polls: int = 40


class ReplayJobTaskSpec(RuntimeJobTaskBase):
    kind: Literal["replay"] = "replay"
    checkpoint_path: Path
    repeats: int = 3
    max_env_steps: int = 0
    max_runs: int = 1
    max_combats: int = 0
    poll_interval_seconds: float = 0.25
    max_idle_polls: int = 40
    prepare_target: NormalizeTarget = "main_menu"
    prepare_max_steps: int = 8
    prepare_max_idle_polls: int = 40

    @model_validator(mode="after")
    def validate_repeats(self) -> ReplayJobTaskSpec:
        if self.repeats < 2:
            raise ValueError("Replay tasks require repeats >= 2.")
        return self


class BenchmarkJobTaskSpec(RuntimeJobTaskBase):
    kind: Literal["benchmark"] = "benchmark"
    benchmark_manifest_path: Path
    replace_existing: bool = False


RuntimeJobTaskSpec = Annotated[
    CollectJobTaskSpec
    | EvalCheckpointJobTaskSpec
    | EvalPolicyPackJobTaskSpec
    | CompareJobTaskSpec
    | ReplayJobTaskSpec
    | BenchmarkJobTaskSpec,
    Field(discriminator="kind"),
]


class RuntimeJobManifest(RuntimeJobModel):
    schema_version: int = RUNTIME_JOB_SCHEMA_VERSION
    job_name: str
    description: str = ""
    output_root: Path = Path("artifacts/jobs")
    concurrency_limit: int = 0
    watchdog: WatchdogPolicy = Field(default_factory=WatchdogPolicy)
    tasks: list[RuntimeJobTaskSpec]

    @model_validator(mode="after")
    def validate_tasks(self) -> RuntimeJobManifest:
        if self.concurrency_limit < 0:
            raise ValueError("concurrency_limit must be >= 0.")
        if not self.tasks:
            raise ValueError("Runtime job manifest must include at least one task.")
        task_ids = [task.task_id for task in self.tasks]
        duplicates = [task_id for task_id, count in Counter(task_ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate task ids are not allowed: {', '.join(sorted(duplicates))}")
        return self


def load_runtime_job_manifest(
    path: str | Path,
    *,
    known_instance_ids: set[str] | None = None,
) -> RuntimeJobManifest:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Runtime job manifest does not exist: {manifest_path}")
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif manifest_path.suffix.lower() == ".toml":
        payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported runtime job manifest format: {manifest_path.suffix}")
    manifest = RuntimeJobManifest.model_validate(payload)
    resolved = _resolve_manifest_paths(manifest, base_dir=manifest_path.parent)
    if known_instance_ids is not None:
        _validate_known_instance_ids(resolved, known_instance_ids=known_instance_ids)
    return resolved


def load_runtime_job_summary(source: str | Path) -> dict[str, object]:
    source_path = Path(source).expanduser().resolve()
    summary_path = source_path / RUNTIME_JOB_SUMMARY_FILENAME if source_path.is_dir() else source_path
    if not summary_path.exists():
        raise FileNotFoundError(f"Runtime job summary does not exist: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_manifest_paths(manifest: RuntimeJobManifest, *, base_dir: Path) -> RuntimeJobManifest:
    payload = manifest.model_dump(mode="python")
    payload["output_root"] = _resolve_relative_path(Path(payload["output_root"]), base_dir)
    resolved_tasks: list[dict[str, object]] = []
    for task in payload["tasks"]:
        task_payload = dict(task)
        predictor_payload = task_payload.get("predictor")
        if isinstance(predictor_payload, dict) and predictor_payload.get("model_path") is not None:
            predictor_payload["model_path"] = _resolve_relative_path(Path(predictor_payload["model_path"]), base_dir)
        community_prior_payload = task_payload.get("community_prior")
        if isinstance(community_prior_payload, dict) and community_prior_payload.get("source_path") is not None:
            community_prior_payload["source_path"] = _resolve_relative_path(
                Path(community_prior_payload["source_path"]),
                base_dir,
            )
            if community_prior_payload.get("route_source_path") is not None:
                community_prior_payload["route_source_path"] = _resolve_relative_path(
                    Path(community_prior_payload["route_source_path"]),
                    base_dir,
                )
        for key in ("checkpoint_path", "baseline_checkpoint_path", "candidate_checkpoint_path", "benchmark_manifest_path"):
            value = task_payload.get(key)
            if value is not None:
                task_payload[key] = _resolve_relative_path(Path(value), base_dir)
        resolved_tasks.append(task_payload)
    payload["tasks"] = resolved_tasks
    return RuntimeJobManifest.model_validate(payload)


def _resolve_relative_path(path: Path, base_dir: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base_dir / expanded).resolve()


def _validate_known_instance_ids(manifest: RuntimeJobManifest, *, known_instance_ids: set[str]) -> None:
    requested_ids = {
        instance_id
        for task in manifest.tasks
        for instance_id in task.instance_ids
    }
    unknown_ids = sorted(requested_ids - known_instance_ids)
    if unknown_ids:
        raise ValueError(f"Unknown instance ids in runtime job manifest: {', '.join(unknown_ids)}")
