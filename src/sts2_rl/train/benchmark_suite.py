from __future__ import annotations

import json
import math
import random
import shutil
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sts2_rl.capability import compare_capability_summaries, merge_capability_summaries
from sts2_rl.collect import CommunityCardPriorSource, CommunityPriorRuntimeConfig
from sts2_rl.game_run_contract import build_game_run_contract
from sts2_rl.predict import PredictorRuntimeConfig
from sts2_rl.shadow import load_shadow_combat_report, run_shadow_combat_comparison

from .compare import (
    CombatCheckpointComparisonReport,
    CombatCheckpointEvalIterationReport,
    _aggregate_iterations,
    _delta_metrics,
    _pick_better_checkpoint,
)
from .replay import (
    CombatReplaySuiteReport,
    _default_env_factory,
    _prepare_replay_start,
    run_combat_dqn_replay_suite,
)
from .policy_checkpoint import run_policy_checkpoint_comparison, run_policy_checkpoint_evaluation
from .runner import run_policy_pack_evaluation

BENCHMARK_SUITE_SCHEMA_VERSION = 1
BENCHMARK_SUITE_SUMMARY_SCHEMA_VERSION = 2
BENCHMARK_SUITE_SUMMARY_FILENAME = "benchmark-suite-summary.json"
BENCHMARK_SUITE_LOG_FILENAME = "benchmark-suite-log.jsonl"
BENCHMARK_CASE_SUMMARY_FILENAME = "case-summary.json"
BENCHMARK_CASE_LOG_FILENAME = "case-log.jsonl"

BenchmarkCaseMode = Literal["eval", "compare", "replay"]
PrepareTarget = Literal["none", "main_menu", "character_select"]


class BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkStatsSpec(BenchmarkModel):
    bootstrap_resamples: int = 1000
    confidence_level: float = 0.95
    seed: int = 0

    @model_validator(mode="after")
    def validate_config(self) -> BenchmarkStatsSpec:
        if self.bootstrap_resamples < 100:
            raise ValueError("stats.bootstrap_resamples must be at least 100.")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("stats.confidence_level must be between 0 and 1.")
        return self


class BenchmarkScenarioSpec(BenchmarkModel):
    seed_set: list[str] = Field(default_factory=list)
    floor_min: int | None = None
    floor_max: int | None = None
    combat_scope: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    act_ids: list[str] = Field(default_factory=list)
    boss_ids: list[str] = Field(default_factory=list)
    planner_strategies: list[str] = Field(default_factory=list)
    route_reason_tags: list[str] = Field(default_factory=list)
    route_profiles: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_floor_band(self) -> BenchmarkScenarioSpec:
        if self.floor_min is not None and self.floor_max is not None and self.floor_min > self.floor_max:
            raise ValueError("scenario.floor_min must be <= scenario.floor_max.")
        self.seed_set = _normalized_unique_strings(self.seed_set)
        self.combat_scope = _normalized_unique_strings(self.combat_scope)
        self.tags = _normalized_unique_strings(self.tags)
        self.act_ids = _normalized_unique_strings(self.act_ids)
        self.boss_ids = _normalized_unique_strings(self.boss_ids)
        self.planner_strategies = _normalized_unique_strings(self.planner_strategies)
        self.route_reason_tags = _normalized_unique_strings(self.route_reason_tags)
        self.route_profiles = _normalized_unique_strings(self.route_profiles)
        return self


class BenchmarkGameRunContractSpec(BenchmarkModel):
    run_mode: str | None = None
    game_seed: str | None = None
    seed_source: str | None = None
    character_id: str | None = None
    ascension: int | None = None
    custom_modifiers: list[str] = Field(default_factory=list)
    progress_profile: str | None = None
    benchmark_contract_id: str | None = None
    strict: bool = True

    @model_validator(mode="after")
    def validate_contract(self) -> BenchmarkGameRunContractSpec:
        self.custom_modifiers = _normalized_unique_strings(self.custom_modifiers)
        self.run_mode = _as_optional_str(self.run_mode)
        self.game_seed = _as_optional_str(self.game_seed)
        self.seed_source = _as_optional_str(self.seed_source)
        self.character_id = _as_optional_str(self.character_id)
        self.progress_profile = _as_optional_str(self.progress_profile)
        self.benchmark_contract_id = _as_optional_str(self.benchmark_contract_id)
        return self

    def as_contract(self):
        return build_game_run_contract(
            run_mode=self.run_mode,
            game_seed=self.game_seed,
            seed_source=self.seed_source,
            character_id=self.character_id,
            ascension=self.ascension,
            custom_modifiers=self.custom_modifiers,
            progress_profile=self.progress_profile,
            benchmark_contract_id=self.benchmark_contract_id,
            strict=self.strict,
        )


class StrategicPromotionSpec(BenchmarkModel):
    min_seed_set_coverage: float = 1.0
    min_route_decision_count: int = 1
    min_route_decision_overlap_rate: float = 1.0
    min_delta_total_reward: float = 0.0
    min_delta_combat_win_rate: float = 0.0
    min_delta_route_quality_score: float = 0.0
    min_delta_pre_boss_readiness: float = 0.0
    max_delta_route_risk_score: float = 0.0
    min_shadow_comparable_encounter_count: int | None = None
    min_shadow_candidate_advantage_rate: float | None = None
    min_shadow_delta_first_action_match_rate: float | None = None
    min_shadow_delta_trace_hit_rate: float | None = None
    max_new_non_combat_capability_regressions: int = 0
    max_candidate_non_combat_capability_issues: int | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> StrategicPromotionSpec:
        if not (0.0 <= self.min_seed_set_coverage <= 1.0):
            raise ValueError("promotion.min_seed_set_coverage must be between 0 and 1.")
        if self.min_route_decision_count < 0:
            raise ValueError("promotion.min_route_decision_count must be non-negative.")
        if not (0.0 <= self.min_route_decision_overlap_rate <= 1.0):
            raise ValueError("promotion.min_route_decision_overlap_rate must be between 0 and 1.")
        if self.min_shadow_comparable_encounter_count is not None and self.min_shadow_comparable_encounter_count < 0:
            raise ValueError("promotion.min_shadow_comparable_encounter_count must be non-negative.")
        if self.max_new_non_combat_capability_regressions < 0:
            raise ValueError("promotion.max_new_non_combat_capability_regressions must be non-negative.")
        if self.max_candidate_non_combat_capability_issues is not None and self.max_candidate_non_combat_capability_issues < 0:
            raise ValueError("promotion.max_candidate_non_combat_capability_issues must be non-negative.")
        return self


class ShadowBenchmarkSpec(BenchmarkModel):
    source: str

    @model_validator(mode="after")
    def validate_source(self) -> ShadowBenchmarkSpec:
        source = _as_optional_str(self.source)
        if source is None:
            raise ValueError("shadow.source is required when shadow benchmarking is enabled.")
        self.source = source
        return self


class PredictorBenchmarkSpec(BenchmarkModel):
    model_path: str | None = None
    mode: str = "heuristic_only"
    hooks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_predictor(self) -> PredictorBenchmarkSpec:
        if self.mode != "heuristic_only" and not self.model_path:
            raise ValueError("predictor-guided cases require predictor.model_path when mode is not heuristic_only.")
        PredictorRuntimeConfig(
            model_path=None if self.model_path is None else Path(self.model_path),
            mode=self.mode,
            hooks=tuple(self.hooks) if self.hooks else None,
        )
        return self


class CommunityPriorBenchmarkSpec(BenchmarkModel):
    source_path: str
    route_source_path: str | None = None
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
    def validate_prior(self) -> CommunityPriorBenchmarkSpec:
        source_path = _as_optional_str(self.source_path)
        if source_path is None:
            raise ValueError("community_prior.source_path is required.")
        self.source_path = source_path
        CommunityPriorRuntimeConfig(
            source_path=Path(source_path),
            route_source_path=None if self.route_source_path is None else Path(self.route_source_path),
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
        return self


class BenchmarkCaseBase(BenchmarkModel):
    case_id: str
    description: str = ""
    repeats: int = 3
    max_env_steps: int = 0
    max_runs: int = 1
    max_combats: int = 0
    poll_interval_seconds: float = 0.25
    max_idle_polls: int = 40
    request_timeout_seconds: float = 30.0
    prepare_target: PrepareTarget = "main_menu"
    prepare_max_steps: int = 8
    prepare_max_idle_polls: int = 40
    scenario: BenchmarkScenarioSpec = Field(default_factory=BenchmarkScenarioSpec)
    game_run_contract: BenchmarkGameRunContractSpec = Field(default_factory=BenchmarkGameRunContractSpec)

    @model_validator(mode="after")
    def validate_repeats(self) -> BenchmarkCaseBase:
        if self.repeats < 1:
            raise ValueError("case repeats must be at least 1.")
        contract = self.game_run_contract.as_contract()
        if contract is not None and contract.game_seed is not None and contract.game_seed not in self.scenario.seed_set:
            self.scenario.seed_set = [contract.game_seed, *self.scenario.seed_set]
        return self


class EvalBenchmarkCaseSpec(BenchmarkCaseBase):
    mode: Literal["eval"] = "eval"
    checkpoint_path: str | None = None
    policy_profile: str | None = None
    predictor: PredictorBenchmarkSpec = Field(default_factory=PredictorBenchmarkSpec)
    community_prior: CommunityPriorBenchmarkSpec | None = None

    @model_validator(mode="after")
    def validate_target(self) -> EvalBenchmarkCaseSpec:
        if bool(self.checkpoint_path) == bool(self.policy_profile):
            raise ValueError("eval case requires exactly one of checkpoint_path or policy_profile.")
        return self


class CompareBenchmarkCaseSpec(BenchmarkCaseBase):
    mode: Literal["compare"] = "compare"
    baseline_checkpoint_path: str | None = None
    candidate_checkpoint_path: str | None = None
    baseline_policy_profile: str | None = None
    candidate_policy_profile: str | None = None
    baseline_predictor: PredictorBenchmarkSpec = Field(default_factory=PredictorBenchmarkSpec)
    candidate_predictor: PredictorBenchmarkSpec = Field(default_factory=PredictorBenchmarkSpec)
    baseline_community_prior: CommunityPriorBenchmarkSpec | None = None
    candidate_community_prior: CommunityPriorBenchmarkSpec | None = None
    promotion: StrategicPromotionSpec | None = None
    shadow: ShadowBenchmarkSpec | None = None

    @model_validator(mode="after")
    def validate_compare_targets(self) -> CompareBenchmarkCaseSpec:
        baseline_is_checkpoint = bool(self.baseline_checkpoint_path)
        candidate_is_checkpoint = bool(self.candidate_checkpoint_path)
        baseline_is_policy = bool(self.baseline_policy_profile)
        candidate_is_policy = bool(self.candidate_policy_profile)
        if baseline_is_checkpoint == baseline_is_policy:
            raise ValueError("compare case baseline requires exactly one of baseline_checkpoint_path or baseline_policy_profile.")
        if candidate_is_checkpoint == candidate_is_policy:
            raise ValueError("compare case candidate requires exactly one of candidate_checkpoint_path or candidate_policy_profile.")
        if baseline_is_checkpoint != candidate_is_checkpoint:
            raise ValueError("compare case must compare like-for-like targets: checkpoint vs checkpoint or policy vs policy.")
        if self.shadow is not None and (not baseline_is_policy or not candidate_is_policy):
            raise ValueError("compare case shadow benchmarking requires baseline_policy_profile and candidate_policy_profile.")
        return self


class ReplayBenchmarkCaseSpec(BenchmarkCaseBase):
    mode: Literal["replay"] = "replay"
    checkpoint_path: str

    @model_validator(mode="after")
    def validate_replay_repeats(self) -> ReplayBenchmarkCaseSpec:
        if self.repeats < 2:
            raise ValueError("replay cases require repeats >= 2.")
        return self


BenchmarkCaseSpec = Annotated[
    EvalBenchmarkCaseSpec | CompareBenchmarkCaseSpec | ReplayBenchmarkCaseSpec,
    Field(discriminator="mode"),
]


class BenchmarkSuiteManifest(BenchmarkModel):
    schema_version: int = BENCHMARK_SUITE_SCHEMA_VERSION
    suite_name: str
    description: str = ""
    base_url: str = "http://127.0.0.1:8080"
    stats: BenchmarkStatsSpec = Field(default_factory=BenchmarkStatsSpec)
    cases: list[BenchmarkCaseSpec]

    @model_validator(mode="after")
    def validate_cases(self) -> BenchmarkSuiteManifest:
        if not self.cases:
            raise ValueError("Benchmark suite must include at least one case.")
        case_ids = [case.case_id for case in self.cases]
        duplicates = [case_id for case_id, count in Counter(case_ids).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate case ids are not allowed: {', '.join(sorted(duplicates))}")
        return self


@dataclass(frozen=True)
class BenchmarkCaseReport:
    case_id: str
    mode: BenchmarkCaseMode
    case_dir: Path
    summary_path: Path
    raw_summary_path: Path | None
    primary_metric: str
    primary_estimate: float | None
    primary_ci_low: float | None
    primary_ci_high: float | None


@dataclass(frozen=True)
class BenchmarkSuiteReport:
    suite_name: str
    suite_dir: Path
    summary_path: Path
    log_path: Path
    manifest_path: Path | None
    case_reports: list[BenchmarkCaseReport]


@dataclass(frozen=True)
class _EvalIterationReport:
    iteration_index: int
    session_name: str
    session_dir: Path
    summary_path: Path
    log_path: Path
    combat_outcomes_path: Path
    prepare_target: str
    normalization_report: dict[str, Any]
    start_signature: str
    start_payload: dict[str, Any]
    prepare_action_ids: list[str]
    env_steps: int
    combat_steps: int
    heuristic_steps: int
    total_reward: float
    stop_reason: str
    final_screen: str
    completed_run_count: int
    completed_combat_count: int
    combat_performance: dict[str, Any]
    observed_run_seeds: list[str]
    observed_run_seed_histogram: dict[str, int]
    runs_without_observed_seed: int
    last_observed_seed: str | None


def load_benchmark_suite_manifest(path: str | Path) -> BenchmarkSuiteManifest:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Benchmark suite manifest does not exist: {manifest_path}")
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif manifest_path.suffix.lower() == ".toml":
        payload = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported benchmark suite manifest format: {manifest_path.suffix}")
    manifest = BenchmarkSuiteManifest.model_validate(payload)
    return _resolve_manifest_paths(manifest, base_dir=manifest_path.parent)


def load_benchmark_suite_summary(source: str | Path) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    summary_path = source_path / BENCHMARK_SUITE_SUMMARY_FILENAME if source_path.is_dir() else source_path
    if not summary_path.exists():
        raise FileNotFoundError(f"Benchmark suite summary does not exist: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_benchmark_suite(
    manifest: BenchmarkSuiteManifest | str | Path,
    *,
    output_root: str | Path,
    suite_name: str | None = None,
    replace_existing: bool = False,
    env_factory=_default_env_factory,
    evaluation_fn=run_policy_checkpoint_evaluation,
    policy_evaluation_fn=run_policy_pack_evaluation,
    comparison_fn=run_policy_checkpoint_comparison,
    replay_fn=run_combat_dqn_replay_suite,
) -> BenchmarkSuiteReport:
    manifest_model, manifest_path = _resolve_manifest_input(manifest)
    suite_dir = Path(output_root).expanduser().resolve() / (suite_name or manifest_model.suite_name)
    if suite_dir.exists():
        if not replace_existing:
            raise FileExistsError(f"Benchmark suite output already exists: {suite_dir}")
        shutil.rmtree(suite_dir)
    suite_dir.mkdir(parents=True, exist_ok=True)

    log_path = suite_dir / BENCHMARK_SUITE_LOG_FILENAME
    summary_path = suite_dir / BENCHMARK_SUITE_SUMMARY_FILENAME
    _append_log(
        log_path,
        {
            "record_type": "benchmark_suite_started",
            "suite_name": manifest_model.suite_name,
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "base_url": manifest_model.base_url,
            "case_count": len(manifest_model.cases),
        },
    )

    case_reports: list[BenchmarkCaseReport] = []
    case_payloads: list[dict[str, Any]] = []
    for case in manifest_model.cases:
        case_dir = suite_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=False)
        case_log_path = case_dir / BENCHMARK_CASE_LOG_FILENAME
        _append_log(
            case_log_path,
            {
                "record_type": "benchmark_case_started",
                "case_id": case.case_id,
                "mode": case.mode,
                "scenario": case.scenario.model_dump(mode="json"),
            },
        )
        case_payload = _run_case(
            manifest=manifest_model,
            case=case,
            case_dir=case_dir,
            case_log_path=case_log_path,
            env_factory=env_factory,
            evaluation_fn=evaluation_fn,
            policy_evaluation_fn=policy_evaluation_fn,
            comparison_fn=comparison_fn,
            replay_fn=replay_fn,
        )
        case_payloads.append(case_payload)
        case_reports.append(
            BenchmarkCaseReport(
                case_id=case.case_id,
                mode=case.mode,
                case_dir=case_dir,
                summary_path=case_dir / BENCHMARK_CASE_SUMMARY_FILENAME,
                raw_summary_path=Path(case_payload["artifacts"]["raw_summary_path"]).resolve()
                if case_payload["artifacts"].get("raw_summary_path")
                else None,
                primary_metric=str(case_payload["primary_metric"]["name"]),
                primary_estimate=_as_float(case_payload["primary_metric"].get("estimate")),
                primary_ci_low=_as_float(case_payload["primary_metric"].get("ci_low")),
                primary_ci_high=_as_float(case_payload["primary_metric"].get("ci_high")),
            )
        )
        _append_log(
            log_path,
            {
                "record_type": "benchmark_case_finished",
                "case_id": case.case_id,
                "mode": case.mode,
                "primary_metric": case_payload["primary_metric"],
                "summary_path": str(case_dir / BENCHMARK_CASE_SUMMARY_FILENAME),
            },
        )

    suite_payload = {
        "schema_version": BENCHMARK_SUITE_SUMMARY_SCHEMA_VERSION,
        "suite_name": manifest_model.suite_name,
        "description": manifest_model.description,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "base_url": manifest_model.base_url,
        "suite_dir": str(suite_dir),
        "stats": manifest_model.stats.model_dump(mode="json"),
        "case_count": len(case_payloads),
        "case_mode_histogram": dict(Counter(case_payload["mode"] for case_payload in case_payloads)),
        "scenario_histograms": _suite_scenario_histograms(case_payloads),
        "strategic": _suite_case_strategic_summary(case_payloads),
        "non_combat_capability": _suite_non_combat_capability_summary(case_payloads),
        "community_alignment": _suite_community_alignment_summary(case_payloads),
        "public_sources": _suite_public_source_summary(case_payloads),
        "shadow": _suite_shadow_summary(case_payloads),
        "promotion": _suite_promotion_summary(case_payloads),
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "cases": case_payloads,
    }
    summary_path.write_text(json.dumps(suite_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_log(
        log_path,
        {
            "record_type": "benchmark_suite_finished",
            "summary_path": str(summary_path),
            "case_mode_histogram": suite_payload["case_mode_histogram"],
        },
    )

    return BenchmarkSuiteReport(
        suite_name=manifest_model.suite_name,
        suite_dir=suite_dir,
        summary_path=summary_path,
        log_path=log_path,
        manifest_path=manifest_path,
        case_reports=case_reports,
    )


def _resolve_manifest_input(
    manifest: BenchmarkSuiteManifest | str | Path,
) -> tuple[BenchmarkSuiteManifest, Path | None]:
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).expanduser().resolve()
        return load_benchmark_suite_manifest(manifest_path), manifest_path
    return manifest, None


def _resolve_manifest_paths(manifest: BenchmarkSuiteManifest, *, base_dir: Path) -> BenchmarkSuiteManifest:
    case_payloads: list[dict[str, Any]] = []
    for case in manifest.cases:
        payload = case.model_dump(mode="json")
        if payload.get("checkpoint_path"):
            payload["checkpoint_path"] = str(_resolve_relative_path(payload["checkpoint_path"], base_dir))
        if payload.get("baseline_checkpoint_path"):
            payload["baseline_checkpoint_path"] = str(
                _resolve_relative_path(payload["baseline_checkpoint_path"], base_dir)
            )
        if payload.get("candidate_checkpoint_path"):
            payload["candidate_checkpoint_path"] = str(
                _resolve_relative_path(payload["candidate_checkpoint_path"], base_dir)
            )
        shadow_payload = payload.get("shadow")
        if isinstance(shadow_payload, dict) and shadow_payload.get("source"):
            shadow_payload["source"] = str(_resolve_relative_path(shadow_payload["source"], base_dir))
        predictor_payload = payload.get("predictor")
        if isinstance(predictor_payload, dict) and predictor_payload.get("model_path"):
            predictor_payload["model_path"] = str(_resolve_relative_path(predictor_payload["model_path"], base_dir))
        community_prior_payload = payload.get("community_prior")
        if isinstance(community_prior_payload, dict) and community_prior_payload.get("source_path"):
            community_prior_payload["source_path"] = str(
                _resolve_relative_path(community_prior_payload["source_path"], base_dir)
            )
            if community_prior_payload.get("route_source_path"):
                community_prior_payload["route_source_path"] = str(
                    _resolve_relative_path(community_prior_payload["route_source_path"], base_dir)
                )
        for key in ("baseline_predictor", "candidate_predictor"):
            nested = payload.get(key)
            if isinstance(nested, dict) and nested.get("model_path"):
                nested["model_path"] = str(_resolve_relative_path(nested["model_path"], base_dir))
        for key in ("baseline_community_prior", "candidate_community_prior"):
            nested = payload.get(key)
            if isinstance(nested, dict) and nested.get("source_path"):
                nested["source_path"] = str(_resolve_relative_path(nested["source_path"], base_dir))
                if nested.get("route_source_path"):
                    nested["route_source_path"] = str(_resolve_relative_path(nested["route_source_path"], base_dir))
        case_payloads.append(payload)
    resolved_payload = manifest.model_dump(mode="json")
    resolved_payload["cases"] = case_payloads
    return BenchmarkSuiteManifest.model_validate(resolved_payload)


def _resolve_relative_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _normalized_unique_strings(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = _as_optional_str(raw_value)
        if value is None or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _predictor_runtime_config_from_spec(spec: PredictorBenchmarkSpec) -> PredictorRuntimeConfig | None:
    if spec.mode == "heuristic_only" and not spec.model_path:
        return None
    resolved_model_path = None if spec.model_path is None else Path(spec.model_path).expanduser().resolve()
    return PredictorRuntimeConfig(
        model_path=resolved_model_path,
        mode=spec.mode,
        hooks=tuple(spec.hooks) if spec.hooks else None,
    )


def _community_prior_runtime_config_from_spec(
    spec: CommunityPriorBenchmarkSpec | None,
) -> CommunityPriorRuntimeConfig | None:
    if spec is None:
        return None
    return CommunityPriorRuntimeConfig(
        source_path=Path(spec.source_path).expanduser().resolve(),
        route_source_path=None if spec.route_source_path is None else Path(spec.route_source_path).expanduser().resolve(),
        reward_pick_weight=spec.reward_pick_weight,
        selection_pick_weight=spec.selection_pick_weight,
        selection_upgrade_weight=spec.selection_upgrade_weight,
        selection_remove_weight=spec.selection_remove_weight,
        shop_buy_weight=spec.shop_buy_weight,
        route_weight=spec.route_weight,
        reward_pick_neutral_rate=spec.reward_pick_neutral_rate,
        shop_buy_neutral_rate=spec.shop_buy_neutral_rate,
        route_neutral_win_rate=spec.route_neutral_win_rate,
        pick_rate_scale=spec.pick_rate_scale,
        buy_rate_scale=spec.buy_rate_scale,
        win_delta_scale=spec.win_delta_scale,
        route_win_rate_scale=spec.route_win_rate_scale,
        min_sample_size=spec.min_sample_size,
        route_min_sample_size=spec.route_min_sample_size,
        max_confidence_sample_size=spec.max_confidence_sample_size,
        max_source_age_days=spec.max_source_age_days,
    )


def _public_source_payload_from_prior_config(
    config: CommunityPriorRuntimeConfig | None,
) -> dict[str, Any] | None:
    if config is None:
        return None
    source = CommunityCardPriorSource.from_config(config)
    if source is None:
        return None
    return {
        "community_prior": {
            "config": config.as_dict(),
            "diagnostics": source.diagnostics(),
        }
    }


def _case_game_run_contract(case: BenchmarkCaseBase):
    return case.game_run_contract.as_contract()


def _run_case(
    *,
    manifest: BenchmarkSuiteManifest,
    case: BenchmarkCaseSpec,
    case_dir: Path,
    case_log_path: Path,
    env_factory,
    evaluation_fn,
    policy_evaluation_fn,
    comparison_fn,
    replay_fn,
) -> dict[str, Any]:
    if case.mode == "eval":
        return _run_eval_case(
            manifest=manifest,
            case=case,
            case_dir=case_dir,
            case_log_path=case_log_path,
            env_factory=env_factory,
            evaluation_fn=evaluation_fn,
            policy_evaluation_fn=policy_evaluation_fn,
        )
    if case.mode == "compare":
        return _run_compare_case(
            manifest=manifest,
            case=case,
            case_dir=case_dir,
            case_log_path=case_log_path,
            comparison_fn=comparison_fn,
            env_factory=env_factory,
            evaluation_fn=evaluation_fn,
            policy_evaluation_fn=policy_evaluation_fn,
        )
    return _run_replay_case(
        manifest=manifest,
        case=case,
        case_dir=case_dir,
        case_log_path=case_log_path,
        replay_fn=replay_fn,
        env_factory=env_factory,
        evaluation_fn=evaluation_fn,
    )


def _run_eval_case(
    *,
    manifest: BenchmarkSuiteManifest,
    case: EvalBenchmarkCaseSpec,
    case_dir: Path,
    case_log_path: Path,
    env_factory,
    evaluation_fn,
    policy_evaluation_fn,
) -> dict[str, Any]:
    raw_root = case_dir / "eval"
    raw_root.mkdir(parents=True, exist_ok=True)
    predictor_config = _predictor_runtime_config_from_spec(case.predictor)
    community_prior_config = _community_prior_runtime_config_from_spec(case.community_prior)
    game_run_contract = _case_game_run_contract(case)
    iterations: list[_EvalIterationReport] = []
    for iteration_index in range(1, case.repeats + 1):
        prepare_report = _prepare_replay_start(
            base_url=manifest.base_url,
            request_timeout_seconds=case.request_timeout_seconds,
            poll_interval_seconds=case.poll_interval_seconds,
            max_idle_polls=case.prepare_max_idle_polls,
            max_prepare_steps=case.prepare_max_steps,
            prepare_target=case.prepare_target,
            env_factory=env_factory,
        )
        session_name = f"iteration-{iteration_index:03d}"
        if case.checkpoint_path is not None:
            report = evaluation_fn(
                base_url=manifest.base_url,
                checkpoint_path=case.checkpoint_path,
                output_root=raw_root,
                session_name=session_name,
                max_env_steps=case.max_env_steps,
                max_runs=case.max_runs,
                max_combats=case.max_combats,
                poll_interval_seconds=case.poll_interval_seconds,
                max_idle_polls=case.max_idle_polls,
                request_timeout_seconds=case.request_timeout_seconds,
                policy_profile="baseline",
                predictor_config=predictor_config,
                community_prior_config=community_prior_config,
                game_run_contract=game_run_contract,
                env_factory=env_factory,
            )
        else:
            report = policy_evaluation_fn(
                base_url=manifest.base_url,
                output_root=raw_root,
                session_name=session_name,
                policy_profile=str(case.policy_profile),
                max_env_steps=case.max_env_steps,
                max_runs=case.max_runs,
                max_combats=case.max_combats,
                poll_interval_seconds=case.poll_interval_seconds,
                max_idle_polls=case.max_idle_polls,
                request_timeout_seconds=case.request_timeout_seconds,
                predictor_config=predictor_config,
                community_prior_config=community_prior_config,
                game_run_contract=game_run_contract,
                env_factory=env_factory,
            )
        iteration_summary = load_json(report.summary_path)
        observed_seed_metadata = _observed_seed_metadata(iteration_summary)
        iteration = _EvalIterationReport(
            iteration_index=iteration_index,
            session_name=session_name,
            session_dir=raw_root / session_name,
            summary_path=report.summary_path,
            log_path=report.log_path,
            combat_outcomes_path=report.combat_outcomes_path,
            prepare_target=str(prepare_report["prepare_target"]),
            normalization_report=dict(prepare_report["normalization_report"]),
            start_signature=str(prepare_report["start_signature"]),
            start_payload=dict(prepare_report["start_payload"]),
            prepare_action_ids=list(prepare_report["prepare_action_ids"]),
            env_steps=report.env_steps,
            combat_steps=report.combat_steps,
            heuristic_steps=report.heuristic_steps,
            total_reward=report.total_reward,
            stop_reason=report.stop_reason,
            final_screen=report.final_screen,
            completed_run_count=report.completed_run_count,
            completed_combat_count=report.completed_combat_count,
            combat_performance=dict(report.combat_performance),
            observed_run_seeds=list(observed_seed_metadata["observed_run_seeds"]),
            observed_run_seed_histogram=dict(observed_seed_metadata["observed_run_seed_histogram"]),
            runs_without_observed_seed=int(observed_seed_metadata["runs_without_observed_seed"]),
            last_observed_seed=_as_optional_str(observed_seed_metadata["last_observed_seed"]),
        )
        iterations.append(iteration)
        _append_log(
            case_log_path,
            {
                "record_type": "benchmark_eval_iteration_finished",
                "iteration_index": iteration_index,
                "session_name": session_name,
                "summary_path": str(report.summary_path),
                "prepare_target": iteration.prepare_target,
                "normalization_stop_reason": iteration.normalization_report.get("stop_reason"),
                "total_reward": iteration.total_reward,
            },
        )

    stats_seed = manifest.stats.seed + _stable_seed_offset(case.case_id)
    metrics = {
        "total_reward": _bootstrap_summary(
            [iteration.total_reward for iteration in iterations],
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 1,
        ),
        "combat_win_rate": _bootstrap_summary(
            [
                float(iteration.combat_performance.get("combat_win_rate"))
                for iteration in iterations
                if iteration.combat_performance.get("combat_win_rate") is not None
            ],
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 2,
        ),
        "reward_per_combat": _bootstrap_summary(
            [
                float(iteration.combat_performance.get("reward_per_combat"))
                for iteration in iterations
                if iteration.combat_performance.get("reward_per_combat") is not None
            ],
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 3,
        ),
        "reward_per_combat_step": _bootstrap_summary(
            [
                float(iteration.combat_performance.get("reward_per_combat_step"))
                for iteration in iterations
                if iteration.combat_performance.get("reward_per_combat_step") is not None
            ],
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 4,
        ),
    }
    iteration_summaries = [load_json(iteration.summary_path) for iteration in iterations]
    strategic_summary = _aggregate_strategic_session_summaries(iteration_summaries)
    capability_summary = merge_capability_summaries(
        summary.get("non_combat_capability") for summary in iteration_summaries
    )
    route_diagnostics = _build_iteration_route_diagnostics(iterations)
    community_alignment = _build_iteration_community_alignment(iterations)
    public_sources = _public_source_payload_from_prior_config(community_prior_config)
    strategic_summary.update(_route_summary_to_strategic_overlay(route_diagnostics["summary"]))
    payload = {
        "schema_version": BENCHMARK_SUITE_SUMMARY_SCHEMA_VERSION,
        "case_id": case.case_id,
        "mode": case.mode,
        "description": case.description,
        "scenario": case.scenario.model_dump(mode="json"),
        "config": _case_config_payload(case),
        "checkpoint_paths": {"checkpoint_path": case.checkpoint_path, "policy_profile": case.policy_profile},
        "predictor": case.predictor.model_dump(mode="json"),
        "community_prior": None if case.community_prior is None else case.community_prior.model_dump(mode="json"),
        "public_sources": public_sources,
        "artifacts": {
            "case_dir": str(case_dir),
            "raw_root": str(raw_root),
            "raw_summary_path": None,
        },
        "iteration_count": len(iterations),
        "stop_reason_histogram": dict(Counter(iteration.stop_reason for iteration in iterations)),
        "final_screen_histogram": dict(Counter(iteration.final_screen for iteration in iterations)),
        "normalization_stop_reason_histogram": dict(
            Counter(str(iteration.normalization_report.get("stop_reason", "unknown")) for iteration in iterations)
        ),
        "all_normalization_targets_reached": all(
            bool(iteration.normalization_report.get("reached_target")) for iteration in iterations
        ),
        "observed_run_seeds": _merge_seed_lists(iteration.observed_run_seeds for iteration in iterations),
        "observed_run_seed_histogram": _merge_counter_payload(
            iteration.observed_run_seed_histogram for iteration in iterations
        ),
        "runs_without_observed_seed": sum(iteration.runs_without_observed_seed for iteration in iterations),
        "last_observed_seed": _last_non_empty_seed(iteration.last_observed_seed for iteration in iterations),
        "policy_pack_histogram": _merge_counter_payload(
            summary.get("policy_pack_histogram", {}) for summary in iteration_summaries
        ),
        "policy_handler_histogram": _merge_counter_payload(
            summary.get("policy_handler_histogram", {}) for summary in iteration_summaries
        ),
        "planner_histogram": _merge_counter_payload(
            summary.get("planner_histogram", {}) for summary in iteration_summaries
        ),
        "predictor_mode_histogram": _merge_counter_payload(
            summary.get("predictor_mode_histogram", {}) for summary in iteration_summaries
        ),
        "predictor_domain_histogram": _merge_counter_payload(
            summary.get("predictor_domain_histogram", {}) for summary in iteration_summaries
        ),
        "predictor_model_histogram": _merge_counter_payload(
            summary.get("predictor_model_histogram", {}) for summary in iteration_summaries
        ),
        "predictor_value_estimate_stats": _merge_numeric_summary_stats(
            summary.get("predictor_value_estimate_stats", {}) for summary in iteration_summaries
        ),
        "predictor_outcome_win_probability_stats": _merge_numeric_summary_stats(
            summary.get("predictor_outcome_win_probability_stats", {}) for summary in iteration_summaries
        ),
        "predictor_expected_reward_stats": _merge_numeric_summary_stats(
            summary.get("predictor_expected_reward_stats", {}) for summary in iteration_summaries
        ),
        "predictor_expected_damage_delta_stats": _merge_numeric_summary_stats(
            summary.get("predictor_expected_damage_delta_stats", {}) for summary in iteration_summaries
        ),
        "strategic": strategic_summary,
        "non_combat_capability": capability_summary,
        "boss_histogram": dict(strategic_summary.get("boss_histogram", {})),
        "route_reason_tag_histogram": dict(strategic_summary.get("route_reason_tag_histogram", {})),
        "route_profile_histogram": dict(strategic_summary.get("route_profile_histogram", {})),
        "route_diagnostics": route_diagnostics,
        "community_alignment": community_alignment,
        "seed_set_diagnostics": _seed_set_diagnostics(
            requested_seed_set=case.scenario.seed_set,
            observed_run_seed_histogram=_merge_counter_payload(
                iteration.observed_run_seed_histogram for iteration in iterations
            ),
            runs_without_observed_seed=sum(iteration.runs_without_observed_seed for iteration in iterations),
        ),
        "metrics": {
            **metrics,
            "community_top_choice_match_rate": _bootstrap_summary(
                list(community_alignment["metric_inputs"]["top_choice_match_rates"]),
                resamples=manifest.stats.bootstrap_resamples,
                confidence_level=manifest.stats.confidence_level,
                seed=stats_seed + 5,
            ),
            "community_weighted_top_choice_match_rate": _bootstrap_summary(
                list(community_alignment["metric_inputs"]["weighted_top_choice_match_rates"]),
                resamples=manifest.stats.bootstrap_resamples,
                confidence_level=manifest.stats.confidence_level,
                seed=stats_seed + 6,
            ),
            "community_opportunity_coverage": _bootstrap_summary(
                list(community_alignment["metric_inputs"]["opportunity_coverages"]),
                resamples=manifest.stats.bootstrap_resamples,
                confidence_level=manifest.stats.confidence_level,
                seed=stats_seed + 7,
            ),
            "community_selected_prior_available_rate": _bootstrap_summary(
                list(community_alignment["metric_inputs"]["selected_prior_available_rates"]),
                resamples=manifest.stats.bootstrap_resamples,
                confidence_level=manifest.stats.confidence_level,
                seed=stats_seed + 8,
            ),
            "community_alignment_regret": _bootstrap_summary(
                list(community_alignment["metric_inputs"]["mean_alignment_regrets"]),
                resamples=manifest.stats.bootstrap_resamples,
                confidence_level=manifest.stats.confidence_level,
                seed=stats_seed + 9,
            ),
            "community_selected_score_bonus": _bootstrap_summary(
                list(community_alignment["metric_inputs"]["mean_selected_score_bonus"]),
                resamples=manifest.stats.bootstrap_resamples,
                confidence_level=manifest.stats.confidence_level,
                seed=stats_seed + 10,
            ),
            "community_best_score_bonus": _bootstrap_summary(
                list(community_alignment["metric_inputs"]["mean_best_score_bonus"]),
                resamples=manifest.stats.bootstrap_resamples,
                confidence_level=manifest.stats.confidence_level,
                seed=stats_seed + 11,
            ),
        },
        "primary_metric": _primary_metric_payload("combat_win_rate", metrics["combat_win_rate"]),
        "iterations": [_eval_iteration_payload(iteration) for iteration in iterations],
    }
    summary_path = case_dir / BENCHMARK_CASE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _run_compare_case(
    *,
    manifest: BenchmarkSuiteManifest,
    case: CompareBenchmarkCaseSpec,
    case_dir: Path,
    case_log_path: Path,
    comparison_fn,
    env_factory,
    evaluation_fn,
    policy_evaluation_fn,
) -> dict[str, Any]:
    game_run_contract = _case_game_run_contract(case)
    requires_internal_checkpoint_comparison = case.baseline_checkpoint_path is not None and (
        game_run_contract is not None
        or case.promotion is not None
        or case.baseline_predictor.model_path
        or case.candidate_predictor.model_path
        or case.baseline_predictor.mode != "heuristic_only"
        or case.candidate_predictor.mode != "heuristic_only"
        or case.baseline_community_prior is not None
        or case.candidate_community_prior is not None
    )
    if requires_internal_checkpoint_comparison:
        report = _run_checkpoint_predictor_comparison(
            manifest=manifest,
            case=case,
            case_dir=case_dir,
            env_factory=env_factory,
            evaluation_fn=evaluation_fn,
        )
    elif case.baseline_checkpoint_path is not None:
        report: CombatCheckpointComparisonReport = comparison_fn(
            base_url=manifest.base_url,
            baseline_checkpoint_path=case.baseline_checkpoint_path,
            candidate_checkpoint_path=case.candidate_checkpoint_path,
            output_root=case_dir,
            comparison_name="comparison",
            repeats=case.repeats,
            max_env_steps=case.max_env_steps,
            max_runs=case.max_runs,
            max_combats=case.max_combats,
            poll_interval_seconds=case.poll_interval_seconds,
            max_idle_polls=case.max_idle_polls,
            request_timeout_seconds=case.request_timeout_seconds,
            prepare_main_menu=case.prepare_target == "main_menu",
            prepare_target=case.prepare_target,
            prepare_max_steps=case.prepare_max_steps,
            prepare_max_idle_polls=case.prepare_max_idle_polls,
            env_factory=env_factory,
            evaluation_fn=evaluation_fn,
        )
    else:
        report = _run_policy_profile_comparison(
            manifest=manifest,
            case=case,
            case_dir=case_dir,
            env_factory=env_factory,
            policy_evaluation_fn=policy_evaluation_fn,
        )
    _append_log(
        case_log_path,
        {
            "record_type": "benchmark_compare_finished",
            "summary_path": str(report.summary_path),
            "better_checkpoint_label": report.better_checkpoint_label,
        },
    )
    baseline_by_iteration = {
        iteration.iteration_index: iteration for iteration in report.iterations if iteration.checkpoint_label == "baseline"
    }
    candidate_by_iteration = {
        iteration.iteration_index: iteration for iteration in report.iterations if iteration.checkpoint_label == "candidate"
    }
    paired_indices = sorted(set(baseline_by_iteration) & set(candidate_by_iteration))
    stats_seed = manifest.stats.seed + _stable_seed_offset(case.case_id)
    baseline_iteration_summaries = [load_json(baseline_by_iteration[index].summary_path) for index in paired_indices]
    candidate_iteration_summaries = [load_json(candidate_by_iteration[index].summary_path) for index in paired_indices]
    baseline_route_diagnostics = _build_iteration_route_diagnostics(
        baseline_by_iteration[index] for index in paired_indices
    )
    candidate_route_diagnostics = _build_iteration_route_diagnostics(
        candidate_by_iteration[index] for index in paired_indices
    )
    baseline_community_alignment = _build_iteration_community_alignment(
        baseline_by_iteration[index] for index in paired_indices
    )
    candidate_community_alignment = _build_iteration_community_alignment(
        candidate_by_iteration[index] for index in paired_indices
    )
    community_alignment_comparison = _build_community_alignment_comparison(
        baseline_iterations=[baseline_by_iteration[index] for index in paired_indices],
        candidate_iterations=[candidate_by_iteration[index] for index in paired_indices],
    )
    route_comparison = _build_route_comparison(
        baseline_iterations=[baseline_by_iteration[index] for index in paired_indices],
        candidate_iterations=[candidate_by_iteration[index] for index in paired_indices],
    )
    baseline_capability = merge_capability_summaries(
        summary.get("non_combat_capability") for summary in baseline_iteration_summaries
    )
    candidate_capability = merge_capability_summaries(
        summary.get("non_combat_capability") for summary in candidate_iteration_summaries
    )
    capability_comparison = compare_capability_summaries(
        baseline=baseline_capability,
        candidate=candidate_capability,
    )
    baseline_strategic = _aggregate_strategic_session_summaries(baseline_iteration_summaries)
    baseline_strategic.update(_route_summary_to_strategic_overlay(baseline_route_diagnostics["summary"]))
    candidate_strategic = _aggregate_strategic_session_summaries(candidate_iteration_summaries)
    candidate_strategic.update(_route_summary_to_strategic_overlay(candidate_route_diagnostics["summary"]))
    shadow_payload = _run_shadow_compare_for_case(case=case, case_dir=case_dir)
    baseline_public_sources = _public_source_payload_from_prior_config(
        _community_prior_runtime_config_from_spec(case.baseline_community_prior)
    )
    candidate_public_sources = _public_source_payload_from_prior_config(
        _community_prior_runtime_config_from_spec(case.candidate_community_prior)
    )
    metrics = {
        "baseline_total_reward": _bootstrap_summary(
            [baseline_by_iteration[index].total_reward for index in paired_indices],
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 1,
        ),
        "candidate_total_reward": _bootstrap_summary(
            [candidate_by_iteration[index].total_reward for index in paired_indices],
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 2,
        ),
        "delta_total_reward": _bootstrap_summary(
            [
                candidate_by_iteration[index].total_reward - baseline_by_iteration[index].total_reward
                for index in paired_indices
            ],
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 3,
        ),
        "delta_combat_win_rate": _bootstrap_summary(
            [
                float(candidate_by_iteration[index].combat_performance.get("combat_win_rate", 0.0) or 0.0)
                - float(baseline_by_iteration[index].combat_performance.get("combat_win_rate", 0.0) or 0.0)
                for index in paired_indices
            ],
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 4,
        ),
        "delta_reward_per_combat": _bootstrap_summary(
            [
                float(candidate_by_iteration[index].combat_performance.get("reward_per_combat", 0.0) or 0.0)
                - float(baseline_by_iteration[index].combat_performance.get("reward_per_combat", 0.0) or 0.0)
                for index in paired_indices
            ],
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 5,
        ),
        "route_decision_overlap_rate": _bootstrap_summary(
            list(route_comparison["metric_inputs"]["path_match_indicators"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 6,
        ),
        "first_node_agreement_rate": _bootstrap_summary(
            list(route_comparison["metric_inputs"]["first_node_match_indicators"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 7,
        ),
        "action_id_agreement_rate": _bootstrap_summary(
            list(route_comparison["metric_inputs"]["action_id_match_indicators"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 8,
        ),
        "delta_route_quality_score": _bootstrap_summary(
            list(route_comparison["metric_inputs"]["delta_route_quality_scores"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 9,
        ),
        "delta_pre_boss_readiness": _bootstrap_summary(
            list(route_comparison["metric_inputs"]["delta_pre_boss_readiness"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 10,
        ),
        "delta_route_risk_score": _bootstrap_summary(
            list(route_comparison["metric_inputs"]["delta_route_risk_scores"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 11,
        ),
        "delta_community_top_choice_match_rate": _bootstrap_summary(
            list(community_alignment_comparison["metric_inputs"]["delta_top_choice_match_rates"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 12,
        ),
        "delta_community_weighted_top_choice_match_rate": _bootstrap_summary(
            list(community_alignment_comparison["metric_inputs"]["delta_weighted_top_choice_match_rates"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 13,
        ),
        "delta_community_opportunity_coverage": _bootstrap_summary(
            list(community_alignment_comparison["metric_inputs"]["delta_opportunity_coverages"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 14,
        ),
        "delta_community_selected_prior_available_rate": _bootstrap_summary(
            list(community_alignment_comparison["metric_inputs"]["delta_selected_prior_available_rates"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 15,
        ),
        "delta_community_alignment_regret": _bootstrap_summary(
            list(community_alignment_comparison["metric_inputs"]["delta_alignment_regrets"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 16,
        ),
        "delta_community_selected_score_bonus": _bootstrap_summary(
            list(community_alignment_comparison["metric_inputs"]["delta_selected_score_bonus"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 17,
        ),
        "delta_community_best_score_bonus": _bootstrap_summary(
            list(community_alignment_comparison["metric_inputs"]["delta_best_score_bonus"]),
            resamples=manifest.stats.bootstrap_resamples,
            confidence_level=manifest.stats.confidence_level,
            seed=stats_seed + 18,
        ),
    }
    seed_set_diagnostics = {
        "baseline": _seed_set_diagnostics_from_aggregate(case.scenario.seed_set, report.baseline),
        "candidate": _seed_set_diagnostics_from_aggregate(case.scenario.seed_set, report.candidate),
    }
    promotion_payload = _evaluate_strategic_promotion(
        case=case,
        metrics=metrics,
        seed_set_diagnostics=seed_set_diagnostics,
        route_comparison=route_comparison,
        capability_comparison=capability_comparison,
        shadow_payload=shadow_payload,
    )
    payload = {
        "schema_version": BENCHMARK_SUITE_SUMMARY_SCHEMA_VERSION,
        "case_id": case.case_id,
        "mode": case.mode,
        "description": case.description,
        "scenario": case.scenario.model_dump(mode="json"),
        "config": _case_config_payload(case),
        "checkpoint_paths": {
            "baseline_checkpoint_path": case.baseline_checkpoint_path,
            "candidate_checkpoint_path": case.candidate_checkpoint_path,
            "baseline_policy_profile": case.baseline_policy_profile,
            "candidate_policy_profile": case.candidate_policy_profile,
        },
        "predictor": {
            "baseline": case.baseline_predictor.model_dump(mode="json"),
            "candidate": case.candidate_predictor.model_dump(mode="json"),
        },
        "community_prior": {
            "baseline": (
                None if case.baseline_community_prior is None else case.baseline_community_prior.model_dump(mode="json")
            ),
            "candidate": (
                None if case.candidate_community_prior is None else case.candidate_community_prior.model_dump(mode="json")
            ),
        },
        "public_sources": {
            "baseline": baseline_public_sources,
            "candidate": candidate_public_sources,
        },
        "shadow": shadow_payload,
        "artifacts": {
            "case_dir": str(case_dir),
            "raw_root": str(report.comparison_dir),
            "raw_summary_path": str(report.summary_path),
            "raw_iterations_path": str(report.iterations_path),
            "raw_log_path": str(report.log_path),
            "shadow_summary_path": None if shadow_payload is None else shadow_payload.get("summary_path"),
            "shadow_results_path": None if shadow_payload is None else shadow_payload.get("comparisons_path"),
        },
        "paired_iteration_count": len(paired_indices),
        "better_checkpoint_label": report.better_checkpoint_label,
        "delta_metrics": report.delta_metrics,
        "baseline": report.baseline,
        "candidate": report.candidate,
        "seed_set_diagnostics": seed_set_diagnostics,
        "policy_pack_histogram": {
            "baseline": _merge_counter_payload(
                summary.get("policy_pack_histogram", {}) for summary in baseline_iteration_summaries
            ),
            "candidate": _merge_counter_payload(
                summary.get("policy_pack_histogram", {}) for summary in candidate_iteration_summaries
            ),
        },
        "planner_histogram": {
            "baseline": _merge_counter_payload(
                summary.get("planner_histogram", {}) for summary in baseline_iteration_summaries
            ),
            "candidate": _merge_counter_payload(
                summary.get("planner_histogram", {}) for summary in candidate_iteration_summaries
            ),
        },
        "predictor_mode_histogram": {
            "baseline": _merge_counter_payload(
                summary.get("predictor_mode_histogram", {}) for summary in baseline_iteration_summaries
            ),
            "candidate": _merge_counter_payload(
                summary.get("predictor_mode_histogram", {}) for summary in candidate_iteration_summaries
            ),
        },
        "predictor_domain_histogram": {
            "baseline": _merge_counter_payload(
                summary.get("predictor_domain_histogram", {}) for summary in baseline_iteration_summaries
            ),
            "candidate": _merge_counter_payload(
                summary.get("predictor_domain_histogram", {}) for summary in candidate_iteration_summaries
            ),
        },
        "predictor_model_histogram": {
            "baseline": _merge_counter_payload(
                summary.get("predictor_model_histogram", {}) for summary in baseline_iteration_summaries
            ),
            "candidate": _merge_counter_payload(
                summary.get("predictor_model_histogram", {}) for summary in candidate_iteration_summaries
            ),
        },
        "predictor_value_estimate_stats": {
            "baseline": _merge_numeric_summary_stats(
                summary.get("predictor_value_estimate_stats", {}) for summary in baseline_iteration_summaries
            ),
            "candidate": _merge_numeric_summary_stats(
                summary.get("predictor_value_estimate_stats", {}) for summary in candidate_iteration_summaries
            ),
        },
        "strategic": {
            "baseline": baseline_strategic,
            "candidate": candidate_strategic,
        },
        "non_combat_capability": {
            "baseline": baseline_capability,
            "candidate": candidate_capability,
            "comparison": capability_comparison,
        },
        "boss_histogram": {
            "baseline": dict(baseline_strategic.get("boss_histogram", {})),
            "candidate": dict(candidate_strategic.get("boss_histogram", {})),
        },
        "route_reason_tag_histogram": {
            "baseline": dict(baseline_strategic.get("route_reason_tag_histogram", {})),
            "candidate": dict(candidate_strategic.get("route_reason_tag_histogram", {})),
        },
        "route_profile_histogram": {
            "baseline": dict(baseline_strategic.get("route_profile_histogram", {})),
            "candidate": dict(candidate_strategic.get("route_profile_histogram", {})),
        },
        "route_diagnostics": {
            "baseline": baseline_route_diagnostics,
            "candidate": candidate_route_diagnostics,
            "comparison": route_comparison,
        },
        "community_alignment": {
            "baseline": baseline_community_alignment,
            "candidate": candidate_community_alignment,
            "comparison": community_alignment_comparison,
        },
        "promotion": promotion_payload,
        "metrics": metrics,
        "primary_metric": _primary_metric_payload("delta_combat_win_rate", metrics["delta_combat_win_rate"]),
    }
    summary_path = case_dir / BENCHMARK_CASE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _run_policy_profile_comparison(
    *,
    manifest: BenchmarkSuiteManifest,
    case: CompareBenchmarkCaseSpec,
    case_dir: Path,
    env_factory,
    policy_evaluation_fn,
) -> CombatCheckpointComparisonReport:
    comparison_dir = case_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_path = comparison_dir / "comparison-summary.json"
    iterations_path = comparison_dir / "comparison-iterations.jsonl"
    log_path = comparison_dir / "comparison-log.jsonl"
    game_run_contract = _case_game_run_contract(case)

    iterations: list[CombatCheckpointEvalIterationReport] = []
    for checkpoint_label, policy_profile in (
        ("baseline", str(case.baseline_policy_profile)),
        ("candidate", str(case.candidate_policy_profile)),
    ):
        predictor_config = _predictor_runtime_config_from_spec(
            case.baseline_predictor if checkpoint_label == "baseline" else case.candidate_predictor
        )
        community_prior_config = _community_prior_runtime_config_from_spec(
            case.baseline_community_prior if checkpoint_label == "baseline" else case.candidate_community_prior
        )
        for iteration_index in range(1, case.repeats + 1):
            session_name = f"{checkpoint_label}-iteration-{iteration_index:03d}"
            report = policy_evaluation_fn(
                base_url=manifest.base_url,
                output_root=comparison_dir,
                session_name=session_name,
                policy_profile=policy_profile,
                max_env_steps=case.max_env_steps,
                max_runs=case.max_runs,
                max_combats=case.max_combats,
                poll_interval_seconds=case.poll_interval_seconds,
                max_idle_polls=case.max_idle_polls,
                request_timeout_seconds=case.request_timeout_seconds,
                predictor_config=predictor_config,
                community_prior_config=community_prior_config,
                game_run_contract=game_run_contract,
                env_factory=env_factory,
            )
            iteration_summary = load_json(report.summary_path)
            observed_seed_metadata = _observed_seed_metadata(iteration_summary)
            iteration_payload = {
                "checkpoint_label": checkpoint_label,
                "checkpoint_path": str(report.checkpoint_path),
                "iteration_index": iteration_index,
                "session_name": session_name,
                "session_dir": str(comparison_dir / session_name),
                "summary_path": str(report.summary_path),
                "log_path": str(report.log_path),
                "combat_outcomes_path": str(report.combat_outcomes_path),
                "prepare_target": case.prepare_target,
                "normalization_report": {
                    "target": case.prepare_target,
                    "reached_target": True,
                    "stop_reason": "target_reached",
                    "strategy_histogram": {},
                },
                "start_screen": "UNKNOWN",
                "start_signature": f"{checkpoint_label}:{iteration_index}",
                "start_payload": {"policy_profile": policy_profile},
                "prepare_action_ids": [],
                "env_steps": report.env_steps,
                "combat_steps": report.combat_steps,
                "heuristic_steps": report.heuristic_steps,
                "total_reward": report.total_reward,
                "stop_reason": report.stop_reason,
                "final_screen": report.final_screen,
                "completed_run_count": report.completed_run_count,
                "completed_combat_count": report.completed_combat_count,
                "combat_performance": report.combat_performance,
                "observed_run_seeds": observed_seed_metadata["observed_run_seeds"],
                "observed_run_seed_histogram": observed_seed_metadata["observed_run_seed_histogram"],
                "runs_without_observed_seed": observed_seed_metadata["runs_without_observed_seed"],
                "last_observed_seed": observed_seed_metadata["last_observed_seed"],
            }
            with iterations_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(iteration_payload, ensure_ascii=False))
                handle.write("\n")
            iterations.append(
                CombatCheckpointEvalIterationReport(
                    checkpoint_label=checkpoint_label,
                    checkpoint_path=report.checkpoint_path,
                    iteration_index=iteration_index,
                    session_name=session_name,
                    session_dir=comparison_dir / session_name,
                    summary_path=report.summary_path,
                    log_path=report.log_path,
                    combat_outcomes_path=report.combat_outcomes_path,
                    prepare_target=case.prepare_target,
                    normalization_report=dict(iteration_payload["normalization_report"]),
                    start_screen="UNKNOWN",
                    start_signature=str(iteration_payload["start_signature"]),
                    start_payload=dict(iteration_payload["start_payload"]),
                    runtime_metadata={},
                    runtime_fingerprint=f"{checkpoint_label}:{policy_profile}:{iteration_index}",
                    step_trace_fingerprint=f"{checkpoint_label}:{policy_profile}:{iteration_index}",
                    prepare_action_ids=[],
                    env_steps=report.env_steps,
                    combat_steps=report.combat_steps,
                    heuristic_steps=report.heuristic_steps,
                    total_reward=report.total_reward,
                    stop_reason=report.stop_reason,
                    final_screen=report.final_screen,
                    completed_run_count=report.completed_run_count,
                    completed_combat_count=report.completed_combat_count,
                    combat_performance=dict(report.combat_performance),
                    observed_run_seeds=list(observed_seed_metadata["observed_run_seeds"]),
                    observed_run_seed_histogram=dict(observed_seed_metadata["observed_run_seed_histogram"]),
                    runs_without_observed_seed=int(observed_seed_metadata["runs_without_observed_seed"]),
                    last_observed_seed=_as_optional_str(observed_seed_metadata["last_observed_seed"]),
                )
            )

    baseline_iterations = [item for item in iterations if item.checkpoint_label == "baseline"]
    candidate_iterations = [item for item in iterations if item.checkpoint_label == "candidate"]
    baseline = _aggregate_iterations("baseline", Path(case.baseline_policy_profile or "baseline"), baseline_iterations)
    candidate = _aggregate_iterations("candidate", Path(case.candidate_policy_profile or "candidate"), candidate_iterations)
    delta_metrics = _delta_metrics(baseline, candidate)
    better_checkpoint_label = _pick_better_checkpoint(baseline, candidate)
    summary_payload = {
        "base_url": manifest.base_url,
        "comparison_dir": str(comparison_dir),
        "baseline_checkpoint_path": case.baseline_policy_profile,
        "candidate_checkpoint_path": case.candidate_policy_profile,
        "repeat_count": case.repeats,
        "prepare_target": case.prepare_target,
        "better_checkpoint_label": better_checkpoint_label,
        "delta_metrics": delta_metrics,
        "baseline": baseline,
        "candidate": candidate,
        "summary_path": str(summary_path),
        "iterations_path": str(iterations_path),
        "diagnostics_path": str(comparison_dir / "comparison-diagnostics.jsonl"),
        "log_path": str(log_path),
        "paired_diagnostics": [],
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text("", encoding="utf-8")
    return CombatCheckpointComparisonReport(
        base_url=manifest.base_url,
        comparison_dir=comparison_dir,
        summary_path=summary_path,
        iterations_path=iterations_path,
        log_path=log_path,
        baseline_checkpoint_path=Path(case.baseline_policy_profile or "baseline"),
        candidate_checkpoint_path=Path(case.candidate_policy_profile or "candidate"),
        repeat_count=case.repeats,
        prepare_target=case.prepare_target,
        better_checkpoint_label=better_checkpoint_label,
        delta_metrics=delta_metrics,
        baseline=baseline,
        candidate=candidate,
        iterations=iterations,
        diagnostics_path=comparison_dir / "comparison-diagnostics.jsonl",
        paired_diagnostics=[],
    )


def _run_shadow_compare_for_case(
    *,
    case: CompareBenchmarkCaseSpec,
    case_dir: Path,
) -> dict[str, Any] | None:
    if case.shadow is None:
        return None
    if case.baseline_policy_profile is None or case.candidate_policy_profile is None:
        raise ValueError("Shadow compare requires compare case policy profiles to be configured.")
    report = run_shadow_combat_comparison(
        source=case.shadow.source,
        output_root=case_dir / "shadow",
        session_name="shadow-compare",
        baseline_policy_profile=case.baseline_policy_profile,
        candidate_policy_profile=case.candidate_policy_profile,
        baseline_predictor_config=_predictor_runtime_config_from_spec(case.baseline_predictor),
        candidate_predictor_config=_predictor_runtime_config_from_spec(case.candidate_predictor),
        replace_existing=True,
    )
    summary = load_shadow_combat_report(report.summary_path)
    return {
        "enabled": True,
        "source": case.shadow.source,
        "summary_path": str(report.summary_path),
        "comparisons_path": str(report.comparisons_path),
        "encounter_count": summary.get("encounter_count"),
        "comparable_encounter_count": summary.get("comparable_encounter_count"),
        "agreement_rate": summary.get("agreement_rate"),
        "candidate_advantage_rate": summary.get("candidate_advantage_rate"),
        "delta_metrics": dict(summary.get("delta_metrics", {})),
        "baseline": dict(summary.get("baseline", {})),
        "candidate": dict(summary.get("candidate", {})),
        "comparison_skip_reason_histogram": dict(summary.get("comparison_skip_reason_histogram", {})),
        "boss_histogram": dict(summary.get("boss_histogram", {})),
        "encounter_family_histogram": dict(summary.get("encounter_family_histogram", {})),
        "report": summary,
    }


def _run_checkpoint_predictor_comparison(
    *,
    manifest: BenchmarkSuiteManifest,
    case: CompareBenchmarkCaseSpec,
    case_dir: Path,
    env_factory,
    evaluation_fn,
) -> CombatCheckpointComparisonReport:
    comparison_dir = case_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_path = comparison_dir / "comparison-summary.json"
    iterations_path = comparison_dir / "comparison-iterations.jsonl"
    log_path = comparison_dir / "comparison-log.jsonl"
    game_run_contract = _case_game_run_contract(case)

    iterations: list[CombatCheckpointEvalIterationReport] = []
    for checkpoint_label, checkpoint_path in (
        ("baseline", Path(str(case.baseline_checkpoint_path))),
        ("candidate", Path(str(case.candidate_checkpoint_path))),
    ):
        predictor_config = _predictor_runtime_config_from_spec(
            case.baseline_predictor if checkpoint_label == "baseline" else case.candidate_predictor
        )
        community_prior_config = _community_prior_runtime_config_from_spec(
            case.baseline_community_prior if checkpoint_label == "baseline" else case.candidate_community_prior
        )
        for iteration_index in range(1, case.repeats + 1):
            prepare_report = _prepare_replay_start(
                base_url=manifest.base_url,
                request_timeout_seconds=case.request_timeout_seconds,
                poll_interval_seconds=case.poll_interval_seconds,
                max_idle_polls=case.prepare_max_idle_polls,
                max_prepare_steps=case.prepare_max_steps,
                prepare_target=case.prepare_target,
                env_factory=env_factory,
            )
            session_name = f"{checkpoint_label}-iteration-{iteration_index:03d}"
            report = evaluation_fn(
                base_url=manifest.base_url,
                checkpoint_path=checkpoint_path,
                output_root=comparison_dir,
                session_name=session_name,
                max_env_steps=case.max_env_steps,
                max_runs=case.max_runs,
                max_combats=case.max_combats,
                poll_interval_seconds=case.poll_interval_seconds,
                max_idle_polls=case.max_idle_polls,
                request_timeout_seconds=case.request_timeout_seconds,
                policy_profile="baseline",
                predictor_config=predictor_config,
                community_prior_config=community_prior_config,
                game_run_contract=game_run_contract,
                env_factory=env_factory,
            )
            iteration_summary = load_json(report.summary_path)
            observed_seed_metadata = _observed_seed_metadata(iteration_summary)
            iteration_payload = {
                "checkpoint_label": checkpoint_label,
                "checkpoint_path": str(report.checkpoint_path),
                "iteration_index": iteration_index,
                "session_name": session_name,
                "session_dir": str(comparison_dir / session_name),
                "summary_path": str(report.summary_path),
                "log_path": str(report.log_path),
                "combat_outcomes_path": str(report.combat_outcomes_path),
                "prepare_target": str(prepare_report["prepare_target"]),
                "normalization_report": dict(prepare_report["normalization_report"]),
                "start_screen": str(prepare_report["start_screen"]),
                "start_signature": str(prepare_report["start_signature"]),
                "start_payload": dict(prepare_report["start_payload"]),
                "prepare_action_ids": list(prepare_report["prepare_action_ids"]),
                "env_steps": report.env_steps,
                "combat_steps": report.combat_steps,
                "heuristic_steps": report.heuristic_steps,
                "total_reward": report.total_reward,
                "stop_reason": report.stop_reason,
                "final_screen": report.final_screen,
                "completed_run_count": report.completed_run_count,
                "completed_combat_count": report.completed_combat_count,
                "combat_performance": report.combat_performance,
                "observed_run_seeds": observed_seed_metadata["observed_run_seeds"],
                "observed_run_seed_histogram": observed_seed_metadata["observed_run_seed_histogram"],
                "runs_without_observed_seed": observed_seed_metadata["runs_without_observed_seed"],
                "last_observed_seed": observed_seed_metadata["last_observed_seed"],
            }
            with iterations_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(iteration_payload, ensure_ascii=False))
                handle.write("\n")
            iterations.append(
                CombatCheckpointEvalIterationReport(
                    checkpoint_label=checkpoint_label,
                    checkpoint_path=Path(checkpoint_path),
                    iteration_index=iteration_index,
                    session_name=session_name,
                    session_dir=comparison_dir / session_name,
                    summary_path=report.summary_path,
                    log_path=report.log_path,
                    combat_outcomes_path=report.combat_outcomes_path,
                    prepare_target=str(prepare_report["prepare_target"]),
                    normalization_report=dict(prepare_report["normalization_report"]),
                    start_screen=str(prepare_report["start_screen"]),
                    start_signature=str(prepare_report["start_signature"]),
                    start_payload=dict(prepare_report["start_payload"]),
                    runtime_metadata={},
                    runtime_fingerprint=f"{checkpoint_label}:{iteration_index}",
                    step_trace_fingerprint=f"{checkpoint_label}:{iteration_index}",
                    prepare_action_ids=list(prepare_report["prepare_action_ids"]),
                    env_steps=report.env_steps,
                    combat_steps=report.combat_steps,
                    heuristic_steps=report.heuristic_steps,
                    total_reward=report.total_reward,
                    stop_reason=report.stop_reason,
                    final_screen=report.final_screen,
                    completed_run_count=report.completed_run_count,
                    completed_combat_count=report.completed_combat_count,
                    combat_performance=dict(report.combat_performance),
                    observed_run_seeds=list(observed_seed_metadata["observed_run_seeds"]),
                    observed_run_seed_histogram=dict(observed_seed_metadata["observed_run_seed_histogram"]),
                    runs_without_observed_seed=int(observed_seed_metadata["runs_without_observed_seed"]),
                    last_observed_seed=_as_optional_str(observed_seed_metadata["last_observed_seed"]),
                )
            )

    baseline_iterations = [item for item in iterations if item.checkpoint_label == "baseline"]
    candidate_iterations = [item for item in iterations if item.checkpoint_label == "candidate"]
    baseline = _aggregate_iterations("baseline", Path(str(case.baseline_checkpoint_path)), baseline_iterations)
    candidate = _aggregate_iterations("candidate", Path(str(case.candidate_checkpoint_path)), candidate_iterations)
    delta_metrics = _delta_metrics(baseline, candidate)
    better_checkpoint_label = _pick_better_checkpoint(baseline, candidate)
    summary_payload = {
        "base_url": manifest.base_url,
        "comparison_dir": str(comparison_dir),
        "baseline_checkpoint_path": case.baseline_checkpoint_path,
        "candidate_checkpoint_path": case.candidate_checkpoint_path,
        "repeat_count": case.repeats,
        "prepare_target": case.prepare_target,
        "better_checkpoint_label": better_checkpoint_label,
        "delta_metrics": delta_metrics,
        "baseline": baseline,
        "candidate": candidate,
        "summary_path": str(summary_path),
        "iterations_path": str(iterations_path),
        "diagnostics_path": str(comparison_dir / "comparison-diagnostics.jsonl"),
        "log_path": str(log_path),
        "paired_diagnostics": [],
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text("", encoding="utf-8")
    return CombatCheckpointComparisonReport(
        base_url=manifest.base_url,
        comparison_dir=comparison_dir,
        summary_path=summary_path,
        iterations_path=iterations_path,
        log_path=log_path,
        baseline_checkpoint_path=Path(str(case.baseline_checkpoint_path)),
        candidate_checkpoint_path=Path(str(case.candidate_checkpoint_path)),
        repeat_count=case.repeats,
        prepare_target=case.prepare_target,
        better_checkpoint_label=better_checkpoint_label,
        delta_metrics=delta_metrics,
        baseline=baseline,
        candidate=candidate,
        iterations=iterations,
        diagnostics_path=comparison_dir / "comparison-diagnostics.jsonl",
        paired_diagnostics=[],
    )


def _run_replay_case(
    *,
    manifest: BenchmarkSuiteManifest,
    case: ReplayBenchmarkCaseSpec,
    case_dir: Path,
    case_log_path: Path,
    replay_fn,
    env_factory,
    evaluation_fn,
) -> dict[str, Any]:
    report: CombatReplaySuiteReport = replay_fn(
        base_url=manifest.base_url,
        checkpoint_path=case.checkpoint_path,
        output_root=case_dir,
        suite_name="replay",
        repeats=case.repeats,
        max_env_steps=case.max_env_steps,
        max_runs=case.max_runs,
        max_combats=case.max_combats,
        poll_interval_seconds=case.poll_interval_seconds,
        max_idle_polls=case.max_idle_polls,
        request_timeout_seconds=case.request_timeout_seconds,
        prepare_main_menu=case.prepare_target == "main_menu",
        prepare_target=case.prepare_target,
        prepare_max_steps=case.prepare_max_steps,
        prepare_max_idle_polls=case.prepare_max_idle_polls,
        env_factory=env_factory,
        evaluation_fn=evaluation_fn,
    )
    _append_log(
        case_log_path,
        {
            "record_type": "benchmark_replay_finished",
            "summary_path": str(report.summary_path),
            "exact_match_count": report.exact_match_count,
            "comparison_count": report.comparison_count,
        },
    )
    stats_seed = manifest.stats.seed + _stable_seed_offset(case.case_id)
    exact_match_indicators = [1.0 if comparison.status == "exact_match" else 0.0 for comparison in report.comparisons]
    exact_match_summary = _bootstrap_summary(
        exact_match_indicators,
        resamples=manifest.stats.bootstrap_resamples,
        confidence_level=manifest.stats.confidence_level,
        seed=stats_seed + 1,
    )
    observed_run_seed_histogram = _merge_counter_payload(
        iteration.observed_run_seed_histogram for iteration in report.iterations
    )
    runs_without_observed_seed = sum(iteration.runs_without_observed_seed for iteration in report.iterations)
    payload = {
        "schema_version": BENCHMARK_SUITE_SUMMARY_SCHEMA_VERSION,
        "case_id": case.case_id,
        "mode": case.mode,
        "description": case.description,
        "scenario": case.scenario.model_dump(mode="json"),
        "config": _case_config_payload(case),
        "checkpoint_paths": {"checkpoint_path": case.checkpoint_path},
        "artifacts": {
            "case_dir": str(case_dir),
            "raw_root": str(report.suite_dir),
            "raw_summary_path": str(report.summary_path),
            "raw_comparisons_path": str(report.comparisons_path),
            "raw_log_path": str(report.log_path),
        },
        "comparison_count": report.comparison_count,
        "status_histogram": report.status_histogram,
        "observed_run_seeds": _merge_seed_lists(iteration.observed_run_seeds for iteration in report.iterations),
        "observed_run_seed_histogram": observed_run_seed_histogram,
        "runs_without_observed_seed": runs_without_observed_seed,
        "last_observed_seed": _last_non_empty_seed(iteration.last_observed_seed for iteration in report.iterations),
        "seed_set_diagnostics": _seed_set_diagnostics(
            requested_seed_set=case.scenario.seed_set,
            observed_run_seed_histogram=observed_run_seed_histogram,
            runs_without_observed_seed=runs_without_observed_seed,
        ),
        "metrics": {
            "exact_match_rate": exact_match_summary,
            "start_signature_match_rate": _bootstrap_summary(
                [1.0 if comparison.start_signature_match else 0.0 for comparison in report.comparisons],
                resamples=manifest.stats.bootstrap_resamples,
                confidence_level=manifest.stats.confidence_level,
                seed=stats_seed + 2,
            ),
            "action_sequence_match_rate": _bootstrap_summary(
                [1.0 if comparison.action_sequence_match else 0.0 for comparison in report.comparisons],
                resamples=manifest.stats.bootstrap_resamples,
                confidence_level=manifest.stats.confidence_level,
                seed=stats_seed + 3,
            ),
            "common_action_prefix_length": _bootstrap_summary(
                [float(comparison.common_action_prefix_length) for comparison in report.comparisons],
                resamples=manifest.stats.bootstrap_resamples,
                confidence_level=manifest.stats.confidence_level,
                seed=stats_seed + 4,
            ),
        },
        "primary_metric": _primary_metric_payload("exact_match_rate", exact_match_summary),
        "raw_suite_summary": load_json(report.summary_path),
    }
    summary_path = case_dir / BENCHMARK_CASE_SUMMARY_FILENAME
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _case_config_payload(case: BenchmarkCaseBase) -> dict[str, Any]:
    return {
        "repeats": case.repeats,
        "max_env_steps": case.max_env_steps,
        "max_runs": case.max_runs,
        "max_combats": case.max_combats,
        "poll_interval_seconds": case.poll_interval_seconds,
        "max_idle_polls": case.max_idle_polls,
        "request_timeout_seconds": case.request_timeout_seconds,
        "prepare_target": case.prepare_target,
        "prepare_max_steps": case.prepare_max_steps,
        "prepare_max_idle_polls": case.prepare_max_idle_polls,
        "game_run_contract": case.game_run_contract.model_dump(mode="json"),
    }


def _build_iteration_route_diagnostics(iterations) -> dict[str, Any]:
    iteration_payloads: list[dict[str, Any]] = []
    for iteration in iterations:
        route_trace = _extract_route_trace(iteration.log_path)
        iteration_payloads.append(
            {
                "iteration_index": int(iteration.iteration_index),
                "session_name": str(iteration.session_name),
                "summary_path": str(iteration.summary_path),
                "log_path": str(iteration.log_path),
                "decision_count": int(route_trace["decision_count"]),
                "floors": list(route_trace["floors"]),
                "trace_lines": list(route_trace["trace_lines"]),
                "decisions": list(route_trace["decisions"]),
            }
        )
    return {
        "summary": _route_iteration_summary(iteration_payloads),
        "iterations": iteration_payloads,
    }


def _build_route_comparison(
    *,
    baseline_iterations,
    candidate_iterations,
) -> dict[str, Any]:
    baseline_by_index = {int(iteration.iteration_index): iteration for iteration in baseline_iterations}
    candidate_by_index = {int(iteration.iteration_index): iteration for iteration in candidate_iterations}
    paired_indices = sorted(set(baseline_by_index) & set(candidate_by_index))
    paired_iterations: list[dict[str, Any]] = []
    path_match_indicators: list[float] = []
    first_node_match_indicators: list[float] = []
    action_id_match_indicators: list[float] = []
    delta_route_quality_scores: list[float] = []
    delta_pre_boss_readiness: list[float] = []
    delta_route_risk_scores: list[float] = []
    baseline_only_decision_count = 0
    candidate_only_decision_count = 0
    for iteration_index in paired_indices:
        baseline_trace = _extract_route_trace(baseline_by_index[iteration_index].log_path)
        candidate_trace = _extract_route_trace(candidate_by_index[iteration_index].log_path)
        decision_pairs, pairing_mode = _pair_route_decisions(
            baseline_trace["decisions"],
            candidate_trace["decisions"],
        )
        baseline_only_decision_count += max(0, int(baseline_trace["decision_count"]) - len(decision_pairs))
        candidate_only_decision_count += max(0, int(candidate_trace["decision_count"]) - len(decision_pairs))
        comparison_rows: list[dict[str, Any]] = []
        for pair in decision_pairs:
            baseline_decision = dict(pair["baseline"])
            candidate_decision = dict(pair["candidate"])
            path_match = bool(
                baseline_decision.get("path_signature")
                and baseline_decision.get("path_signature") == candidate_decision.get("path_signature")
            )
            first_node_match = bool(
                baseline_decision.get("first_node")
                and baseline_decision.get("first_node") == candidate_decision.get("first_node")
            )
            action_id_match = bool(
                baseline_decision.get("action_id")
                and baseline_decision.get("action_id") == candidate_decision.get("action_id")
            )
            delta_quality = _pair_delta(candidate_decision.get("route_quality_score"), baseline_decision.get("route_quality_score"))
            delta_readiness = _pair_delta(
                candidate_decision.get("pre_boss_readiness"),
                baseline_decision.get("pre_boss_readiness"),
            )
            delta_risk = _pair_delta(candidate_decision.get("route_risk_score"), baseline_decision.get("route_risk_score"))
            path_match_indicators.append(1.0 if path_match else 0.0)
            first_node_match_indicators.append(1.0 if first_node_match else 0.0)
            action_id_match_indicators.append(1.0 if action_id_match else 0.0)
            if delta_quality is not None:
                delta_route_quality_scores.append(delta_quality)
            if delta_readiness is not None:
                delta_pre_boss_readiness.append(delta_readiness)
            if delta_risk is not None:
                delta_route_risk_scores.append(delta_risk)
            comparison_rows.append(
                {
                    "floor": pair.get("floor"),
                    "pairing_key": pair.get("pairing_key"),
                    "path_match": path_match,
                    "first_node_match": first_node_match,
                    "action_id_match": action_id_match,
                    "delta_route_quality_score": delta_quality,
                    "delta_pre_boss_readiness": delta_readiness,
                    "delta_route_risk_score": delta_risk,
                    "baseline": baseline_decision,
                    "candidate": candidate_decision,
                }
            )
        paired_iterations.append(
            {
                "iteration_index": iteration_index,
                "pairing_mode": pairing_mode,
                "baseline_decision_count": int(baseline_trace["decision_count"]),
                "candidate_decision_count": int(candidate_trace["decision_count"]),
                "paired_decision_count": len(decision_pairs),
                "baseline_trace_lines": list(baseline_trace["trace_lines"]),
                "candidate_trace_lines": list(candidate_trace["trace_lines"]),
                "pairs": comparison_rows,
            }
        )
    pair_count = len(path_match_indicators)
    return {
        "paired_iteration_count": len(paired_indices),
        "route_decision_pair_count": pair_count,
        "baseline_only_decision_count": baseline_only_decision_count,
        "candidate_only_decision_count": candidate_only_decision_count,
        "route_decision_overlap_rate": None if pair_count == 0 else (sum(path_match_indicators) / pair_count),
        "first_node_agreement_rate": None if pair_count == 0 else (sum(first_node_match_indicators) / pair_count),
        "action_id_agreement_rate": None if pair_count == 0 else (sum(action_id_match_indicators) / pair_count),
        "delta_route_quality_score_stats": _numeric_summary(delta_route_quality_scores),
        "delta_pre_boss_readiness_stats": _numeric_summary(delta_pre_boss_readiness),
        "delta_route_risk_score_stats": _numeric_summary(delta_route_risk_scores),
        "paired_iterations": paired_iterations,
        "metric_inputs": {
            "path_match_indicators": path_match_indicators,
            "first_node_match_indicators": first_node_match_indicators,
            "action_id_match_indicators": action_id_match_indicators,
            "delta_route_quality_scores": delta_route_quality_scores,
            "delta_pre_boss_readiness": delta_pre_boss_readiness,
            "delta_route_risk_scores": delta_route_risk_scores,
        },
    }


def _extract_route_trace(log_path: Path) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    with Path(log_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("record_type") != "step":
                continue
            decision = _route_decision_from_step_payload(payload)
            if decision is not None:
                decisions.append(decision)
    return {
        "decision_count": len(decisions),
        "floors": [decision["floor"] for decision in decisions if decision.get("floor") is not None],
        "trace_lines": [_route_trace_line(decision) for decision in decisions],
        "decisions": decisions,
    }


def _route_decision_from_step_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    decision_metadata = payload.get("decision_metadata")
    if not isinstance(decision_metadata, dict):
        return None
    route_planner = decision_metadata.get("route_planner")
    if not isinstance(route_planner, dict):
        return None
    selected = route_planner.get("selected")
    if not isinstance(selected, dict):
        return None
    path = _normalized_string_list(selected.get("path"))
    path_node_types = _normalized_string_list(selected.get("path_node_types"))
    reason_tags = _normalized_string_list(selected.get("reason_tags"))
    planner_strategy = _as_optional_str(route_planner.get("planner_strategy"))
    score = _as_float(selected.get("score"))
    first_rest_distance = _as_optional_int(selected.get("first_rest_distance"))
    first_elite_distance = _as_optional_int(selected.get("first_elite_distance"))
    rest_count = _as_optional_int(selected.get("rest_count"))
    shop_count = _as_optional_int(selected.get("shop_count"))
    elite_count = _as_optional_int(selected.get("elite_count"))
    event_count = _as_optional_int(selected.get("event_count"))
    treasure_count = _as_optional_int(selected.get("treasure_count"))
    monster_count = _as_optional_int(selected.get("monster_count"))
    elites_before_rest = _as_optional_int(selected.get("elites_before_rest"))
    remaining_distance_to_boss = _as_optional_int(selected.get("remaining_distance_to_boss"))
    route_risk_score = _route_risk_score(
        elite_count=elite_count,
        monster_count=monster_count,
        rest_count=rest_count,
        shop_count=shop_count,
        elites_before_rest=elites_before_rest,
        first_rest_distance=first_rest_distance,
    )
    pre_boss_readiness = _pre_boss_readiness(
        selected_route_score=score,
        rest_count=rest_count,
        shop_count=shop_count,
        elite_count=elite_count,
        remaining_distance_to_boss=remaining_distance_to_boss,
        first_rest_distance=first_rest_distance,
    )
    route_quality_score = _route_quality_score(
        selected_route_score=score,
        pre_boss_readiness=pre_boss_readiness,
        route_risk_score=route_risk_score,
    )
    path_signature_parts = path if path else path_node_types
    return {
        "step_index": int(payload.get("step_index", 0) or 0),
        "floor": _as_optional_int(payload.get("floor")),
        "action_id": _as_optional_str(selected.get("action_id")),
        "boss_id": _as_optional_str(route_planner.get("boss_encounter_id")),
        "planner_name": _as_optional_str(route_planner.get("planner_name")),
        "planner_strategy": planner_strategy,
        "reason_tags": reason_tags,
        "route_profile": _route_profile_label(
            reason_tags=reason_tags,
            planned_node_types=path_node_types,
            planner_strategy=planner_strategy,
        ),
        "path": path,
        "path_node_types": path_node_types,
        "path_signature": ">".join(path_signature_parts) if path_signature_parts else None,
        "first_node": _as_optional_str(path_signature_parts[0]) if path_signature_parts else None,
        "first_rest_distance": first_rest_distance,
        "first_elite_distance": first_elite_distance,
        "rest_count": rest_count,
        "shop_count": shop_count,
        "elite_count": elite_count,
        "event_count": event_count,
        "treasure_count": treasure_count,
        "monster_count": monster_count,
        "elites_before_rest": elites_before_rest,
        "remaining_distance_to_boss": remaining_distance_to_boss,
        "selected_route_score": score,
        "route_risk_score": route_risk_score,
        "pre_boss_readiness": pre_boss_readiness,
        "route_quality_score": route_quality_score,
    }


def _route_trace_line(decision: dict[str, Any]) -> str:
    floor = decision.get("floor")
    first_node = decision.get("first_node") or "?"
    reason_tags = ",".join(decision.get("reason_tags", [])) or "none"
    return (
        f"floor={floor if floor is not None else '?'} "
        f"first={first_node} "
        f"score={_format_metric(decision.get('selected_route_score'))} "
        f"quality={_format_metric(decision.get('route_quality_score'))} "
        f"risk={_format_metric(decision.get('route_risk_score'))} "
        f"readiness={_format_metric(decision.get('pre_boss_readiness'))} "
        f"tags={reason_tags}"
    )


def _route_iteration_summary(iteration_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [decision for payload in iteration_payloads for decision in payload.get("decisions", [])]
    return {
        "iterations_with_route_decisions": sum(1 for payload in iteration_payloads if payload.get("decision_count")),
        "route_decision_count": len(decisions),
        "route_decision_count_stats": _numeric_summary(
            [float(payload.get("decision_count", 0) or 0) for payload in iteration_payloads]
        ),
        "floor_histogram": _histogram(decision.get("floor") for decision in decisions),
        "planner_name_histogram": _histogram(decision.get("planner_name") for decision in decisions),
        "planner_strategy_histogram": _histogram(decision.get("planner_strategy") for decision in decisions),
        "route_reason_tag_histogram": _histogram(
            reason_tag
            for decision in decisions
            for reason_tag in decision.get("reason_tags", [])
        ),
        "route_profile_histogram": _histogram(decision.get("route_profile") for decision in decisions),
        "first_rest_distance_stats": _numeric_summary(
            [decision["first_rest_distance"] for decision in decisions if decision.get("first_rest_distance") is not None]
        ),
        "first_elite_distance_stats": _numeric_summary(
            [decision["first_elite_distance"] for decision in decisions if decision.get("first_elite_distance") is not None]
        ),
        "route_risk_score_stats": _numeric_summary(
            [decision["route_risk_score"] for decision in decisions if decision.get("route_risk_score") is not None]
        ),
        "pre_boss_readiness_stats": _numeric_summary(
            [decision["pre_boss_readiness"] for decision in decisions if decision.get("pre_boss_readiness") is not None]
        ),
        "route_quality_score_stats": _numeric_summary(
            [decision["route_quality_score"] for decision in decisions if decision.get("route_quality_score") is not None]
        ),
    }


def _route_summary_to_strategic_overlay(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_decision_count": int(summary.get("route_decision_count", 0) or 0),
        "route_decision_count_stats": dict(summary.get("route_decision_count_stats", {})),
        "first_rest_distance_stats": dict(summary.get("first_rest_distance_stats", {})),
        "first_elite_distance_stats": dict(summary.get("first_elite_distance_stats", {})),
        "route_risk_score_stats": dict(summary.get("route_risk_score_stats", {})),
        "pre_boss_readiness_stats": dict(summary.get("pre_boss_readiness_stats", {})),
        "route_quality_score_stats": dict(summary.get("route_quality_score_stats", {})),
    }


def _build_iteration_community_alignment(iterations) -> dict[str, Any]:
    iteration_payloads: list[dict[str, Any]] = []
    metric_inputs = {
        "top_choice_match_rates": [],
        "weighted_top_choice_match_rates": [],
        "opportunity_coverages": [],
        "selected_prior_available_rates": [],
        "mean_alignment_regrets": [],
        "mean_selected_score_bonus": [],
        "mean_best_score_bonus": [],
    }
    for iteration in iterations:
        community_trace = _extract_community_alignment_trace(iteration.log_path)
        iteration_payload = {
            "iteration_index": int(iteration.iteration_index),
            "session_name": str(iteration.session_name),
            "summary_path": str(iteration.summary_path),
            "log_path": str(iteration.log_path),
            "decision_step_count": int(community_trace["decision_step_count"]),
            "eligible_decision_count": int(community_trace["eligible_decision_count"]),
            "floors": list(community_trace["floors"]),
            "trace_lines": list(community_trace["trace_lines"]),
            "decisions": list(community_trace["decisions"]),
        }
        iteration_summary = _community_alignment_iteration_summary([iteration_payload])
        iteration_payload["summary"] = iteration_summary
        iteration_payloads.append(iteration_payload)
        for key, value in (
            ("top_choice_match_rates", _as_float(iteration_summary.get("top_choice_match_rate"))),
            ("weighted_top_choice_match_rates", _as_float(iteration_summary.get("weighted_top_choice_match_rate"))),
            ("opportunity_coverages", _as_float(iteration_summary.get("opportunity_coverage"))),
            ("selected_prior_available_rates", _as_float(iteration_summary.get("selected_prior_available_rate"))),
            ("mean_alignment_regrets", _metric_mean(iteration_summary.get("alignment_regret_stats"))),
            ("mean_selected_score_bonus", _metric_mean(iteration_summary.get("selected_score_bonus_stats"))),
            ("mean_best_score_bonus", _metric_mean(iteration_summary.get("best_score_bonus_stats"))),
        ):
            if value is not None:
                metric_inputs[key].append(value)
    return {
        "summary": _community_alignment_iteration_summary(iteration_payloads),
        "iterations": iteration_payloads,
        "metric_inputs": metric_inputs,
    }


def _build_community_alignment_comparison(
    *,
    baseline_iterations,
    candidate_iterations,
) -> dict[str, Any]:
    baseline_by_index = {int(iteration.iteration_index): iteration for iteration in baseline_iterations}
    candidate_by_index = {int(iteration.iteration_index): iteration for iteration in candidate_iterations}
    paired_indices = sorted(set(baseline_by_index) & set(candidate_by_index))
    paired_iterations: list[dict[str, Any]] = []
    delta_top_choice_match_rates: list[float] = []
    delta_weighted_top_choice_match_rates: list[float] = []
    delta_opportunity_coverages: list[float] = []
    delta_selected_prior_available_rates: list[float] = []
    delta_alignment_regrets: list[float] = []
    delta_selected_score_bonus: list[float] = []
    delta_best_score_bonus: list[float] = []
    comparable_iteration_count = 0
    baseline_only_eligible_iteration_count = 0
    candidate_only_eligible_iteration_count = 0
    for iteration_index in paired_indices:
        baseline_trace = _extract_community_alignment_trace(baseline_by_index[iteration_index].log_path)
        candidate_trace = _extract_community_alignment_trace(candidate_by_index[iteration_index].log_path)
        baseline_payload = {
            "iteration_index": iteration_index,
            "session_name": str(baseline_by_index[iteration_index].session_name),
            "summary_path": str(baseline_by_index[iteration_index].summary_path),
            "log_path": str(baseline_by_index[iteration_index].log_path),
            "decision_step_count": int(baseline_trace["decision_step_count"]),
            "eligible_decision_count": int(baseline_trace["eligible_decision_count"]),
            "floors": list(baseline_trace["floors"]),
            "trace_lines": list(baseline_trace["trace_lines"]),
            "decisions": list(baseline_trace["decisions"]),
        }
        candidate_payload = {
            "iteration_index": iteration_index,
            "session_name": str(candidate_by_index[iteration_index].session_name),
            "summary_path": str(candidate_by_index[iteration_index].summary_path),
            "log_path": str(candidate_by_index[iteration_index].log_path),
            "decision_step_count": int(candidate_trace["decision_step_count"]),
            "eligible_decision_count": int(candidate_trace["eligible_decision_count"]),
            "floors": list(candidate_trace["floors"]),
            "trace_lines": list(candidate_trace["trace_lines"]),
            "decisions": list(candidate_trace["decisions"]),
        }
        baseline_summary = _community_alignment_iteration_summary([baseline_payload])
        candidate_summary = _community_alignment_iteration_summary([candidate_payload])
        if baseline_summary["eligible_decision_count"] and candidate_summary["eligible_decision_count"]:
            comparable_iteration_count += 1
        elif baseline_summary["eligible_decision_count"]:
            baseline_only_eligible_iteration_count += 1
        elif candidate_summary["eligible_decision_count"]:
            candidate_only_eligible_iteration_count += 1
        delta_top_choice_match_rate = _pair_delta(
            candidate_summary.get("top_choice_match_rate"),
            baseline_summary.get("top_choice_match_rate"),
        )
        delta_weighted_top_choice_match_rate = _pair_delta(
            candidate_summary.get("weighted_top_choice_match_rate"),
            baseline_summary.get("weighted_top_choice_match_rate"),
        )
        delta_opportunity_coverage = _pair_delta(
            candidate_summary.get("opportunity_coverage"),
            baseline_summary.get("opportunity_coverage"),
        )
        delta_selected_prior_available_rate = _pair_delta(
            candidate_summary.get("selected_prior_available_rate"),
            baseline_summary.get("selected_prior_available_rate"),
        )
        delta_alignment_regret = _pair_delta(
            _metric_mean(candidate_summary.get("alignment_regret_stats")),
            _metric_mean(baseline_summary.get("alignment_regret_stats")),
        )
        delta_selected_bonus = _pair_delta(
            _metric_mean(candidate_summary.get("selected_score_bonus_stats")),
            _metric_mean(baseline_summary.get("selected_score_bonus_stats")),
        )
        delta_best_bonus = _pair_delta(
            _metric_mean(candidate_summary.get("best_score_bonus_stats")),
            _metric_mean(baseline_summary.get("best_score_bonus_stats")),
        )
        if delta_top_choice_match_rate is not None:
            delta_top_choice_match_rates.append(delta_top_choice_match_rate)
        if delta_weighted_top_choice_match_rate is not None:
            delta_weighted_top_choice_match_rates.append(delta_weighted_top_choice_match_rate)
        if delta_opportunity_coverage is not None:
            delta_opportunity_coverages.append(delta_opportunity_coverage)
        if delta_selected_prior_available_rate is not None:
            delta_selected_prior_available_rates.append(delta_selected_prior_available_rate)
        if delta_alignment_regret is not None:
            delta_alignment_regrets.append(delta_alignment_regret)
        if delta_selected_bonus is not None:
            delta_selected_score_bonus.append(delta_selected_bonus)
        if delta_best_bonus is not None:
            delta_best_score_bonus.append(delta_best_bonus)
        paired_iterations.append(
            {
                "iteration_index": iteration_index,
                "baseline": baseline_summary,
                "candidate": candidate_summary,
                "baseline_trace_lines": list(baseline_trace["trace_lines"]),
                "candidate_trace_lines": list(candidate_trace["trace_lines"]),
                "delta_top_choice_match_rate": delta_top_choice_match_rate,
                "delta_weighted_top_choice_match_rate": delta_weighted_top_choice_match_rate,
                "delta_opportunity_coverage": delta_opportunity_coverage,
                "delta_selected_prior_available_rate": delta_selected_prior_available_rate,
                "delta_alignment_regret": delta_alignment_regret,
                "delta_selected_score_bonus": delta_selected_bonus,
                "delta_best_score_bonus": delta_best_bonus,
            }
        )
    return {
        "paired_iteration_count": len(paired_indices),
        "comparable_iteration_count": comparable_iteration_count,
        "baseline_only_eligible_iteration_count": baseline_only_eligible_iteration_count,
        "candidate_only_eligible_iteration_count": candidate_only_eligible_iteration_count,
        "delta_top_choice_match_rate_stats": _numeric_summary(delta_top_choice_match_rates),
        "delta_weighted_top_choice_match_rate_stats": _numeric_summary(delta_weighted_top_choice_match_rates),
        "delta_opportunity_coverage_stats": _numeric_summary(delta_opportunity_coverages),
        "delta_selected_prior_available_rate_stats": _numeric_summary(delta_selected_prior_available_rates),
        "delta_alignment_regret_stats": _numeric_summary(delta_alignment_regrets),
        "delta_selected_score_bonus_stats": _numeric_summary(delta_selected_score_bonus),
        "delta_best_score_bonus_stats": _numeric_summary(delta_best_score_bonus),
        "paired_iterations": paired_iterations,
        "metric_inputs": {
            "delta_top_choice_match_rates": delta_top_choice_match_rates,
            "delta_weighted_top_choice_match_rates": delta_weighted_top_choice_match_rates,
            "delta_opportunity_coverages": delta_opportunity_coverages,
            "delta_selected_prior_available_rates": delta_selected_prior_available_rates,
            "delta_alignment_regrets": delta_alignment_regrets,
            "delta_selected_score_bonus": delta_selected_score_bonus,
            "delta_best_score_bonus": delta_best_score_bonus,
        },
    }


def _extract_community_alignment_trace(log_path: Path) -> dict[str, Any]:
    decision_step_count = 0
    decisions: list[dict[str, Any]] = []
    with Path(log_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("record_type") != "step":
                continue
            decision_step_count += 1
            decision = _community_alignment_decision_from_step_payload(payload)
            if decision is not None:
                decisions.append(decision)
    return {
        "decision_step_count": decision_step_count,
        "eligible_decision_count": len(decisions),
        "floors": [decision["floor"] for decision in decisions if decision.get("floor") is not None],
        "trace_lines": [_community_alignment_trace_line(decision) for decision in decisions],
        "decisions": decisions,
    }


def _community_alignment_decision_from_step_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    decision_metadata = payload.get("decision_metadata")
    if not isinstance(decision_metadata, dict):
        return None
    community_prior = decision_metadata.get("community_prior")
    if not isinstance(community_prior, dict):
        return None
    selected = community_prior.get("selected")
    selected_payload = selected if isinstance(selected, dict) else {}
    selected_prior = selected_payload.get("prior")
    selected_prior_payload = selected_prior if isinstance(selected_prior, dict) else None
    ranked_candidates = community_prior.get("ranked_candidates")
    candidates: list[dict[str, Any]] = []
    if isinstance(ranked_candidates, list):
        for item in ranked_candidates:
            if not isinstance(item, dict):
                continue
            prior_payload = item.get("prior")
            if not isinstance(prior_payload, dict):
                continue
            candidate = _community_alignment_candidate_from_payload(item, prior_payload)
            if candidate is not None:
                candidates.append(candidate)
    if not candidates and selected_prior_payload is not None:
        candidate = _community_alignment_candidate_from_payload(selected_payload, selected_prior_payload)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None
    best_candidate = max(candidates, key=_community_alignment_candidate_sort_key)
    selected_action_id = _as_optional_str(selected_payload.get("action_id"))
    selected_score_bonus = _as_float(selected_prior_payload.get("score_bonus")) if selected_prior_payload else None
    selected_confidence = _as_float(selected_prior_payload.get("confidence")) if selected_prior_payload else None
    best_score_bonus = _as_float(best_candidate.get("score_bonus"))
    regret = None
    if selected_score_bonus is not None and best_score_bonus is not None:
        regret = max(0.0, best_score_bonus - selected_score_bonus)
    weight = _as_float(best_candidate.get("confidence"))
    if weight is None:
        weight = 1.0
    weight = max(0.0, weight)
    return {
        "step_index": int(payload.get("step_index", 0) or 0),
        "floor": _as_optional_int(payload.get("floor")),
        "decision_stage": _as_optional_str(payload.get("decision_stage")),
        "selected_action_id": selected_action_id,
        "selected_action": _as_optional_str(selected_payload.get("action")),
        "selected_reason": _as_optional_str(selected_payload.get("reason")),
        "selected_heuristic_score": _as_float(selected_payload.get("heuristic_score")),
        "selected_final_score": _as_float(selected_payload.get("final_score")),
        "selected_prior_available": selected_prior_payload is not None and selected_score_bonus is not None,
        "selected_domain": _as_optional_str(selected_prior_payload.get("domain")) if selected_prior_payload else None,
        "selected_subject_id": (
            _as_optional_str(selected_prior_payload.get("subject_id"))
            or (_as_optional_str(selected_prior_payload.get("card_id")) if selected_prior_payload else None)
        ),
        "selected_card_id": _as_optional_str(selected_prior_payload.get("card_id")) if selected_prior_payload else None,
        "selected_source_name": (
            _as_optional_str(selected_prior_payload.get("source_name")) if selected_prior_payload else None
        ),
        "selected_artifact_family": (
            _as_optional_str(selected_prior_payload.get("artifact_family")) if selected_prior_payload else None
        ),
        "selected_score_bonus": selected_score_bonus,
        "selected_confidence": selected_confidence,
        "best_action_id": _as_optional_str(best_candidate.get("action_id")),
        "best_action": _as_optional_str(best_candidate.get("action")),
        "best_reason": _as_optional_str(best_candidate.get("reason")),
        "best_heuristic_score": _as_float(best_candidate.get("heuristic_score")),
        "best_final_score": _as_float(best_candidate.get("final_score")),
        "best_domain": _as_optional_str(best_candidate.get("domain")),
        "best_subject_id": _as_optional_str(best_candidate.get("subject_id")) or _as_optional_str(best_candidate.get("card_id")),
        "best_card_id": _as_optional_str(best_candidate.get("card_id")),
        "best_source_name": _as_optional_str(best_candidate.get("source_name")),
        "best_artifact_family": _as_optional_str(best_candidate.get("artifact_family")),
        "best_score_bonus": best_score_bonus,
        "best_confidence": _as_float(best_candidate.get("confidence")),
        "domain": _as_optional_str(best_candidate.get("domain"))
        or (_as_optional_str(selected_prior_payload.get("domain")) if selected_prior_payload else None),
        "subject_id": _as_optional_str(best_candidate.get("subject_id"))
        or _as_optional_str(best_candidate.get("card_id"))
        or (_as_optional_str(selected_prior_payload.get("subject_id")) if selected_prior_payload else None)
        or (_as_optional_str(selected_prior_payload.get("card_id")) if selected_prior_payload else None),
        "card_id": _as_optional_str(best_candidate.get("card_id"))
        or (_as_optional_str(selected_prior_payload.get("card_id")) if selected_prior_payload else None),
        "source_name": _as_optional_str(best_candidate.get("source_name"))
        or (_as_optional_str(selected_prior_payload.get("source_name")) if selected_prior_payload else None),
        "artifact_family": _as_optional_str(best_candidate.get("artifact_family"))
        or (_as_optional_str(selected_prior_payload.get("artifact_family")) if selected_prior_payload else None),
        "candidate_count": len(candidates),
        "weight": weight,
        "top_choice_match": bool(selected_action_id and selected_action_id == best_candidate.get("action_id")),
        "alignment_regret": regret,
    }


def _community_alignment_candidate_from_payload(
    candidate_payload: dict[str, Any],
    prior_payload: dict[str, Any],
) -> dict[str, Any] | None:
    score_bonus = _as_float(prior_payload.get("score_bonus"))
    if score_bonus is None:
        return None
    return {
        "action_id": _as_optional_str(candidate_payload.get("action_id")),
        "action": _as_optional_str(candidate_payload.get("action")),
        "reason": _as_optional_str(candidate_payload.get("reason")),
        "heuristic_score": _as_float(candidate_payload.get("heuristic_score")),
        "final_score": _as_float(candidate_payload.get("final_score")),
        "domain": _as_optional_str(prior_payload.get("domain")),
        "subject_id": _as_optional_str(prior_payload.get("subject_id")) or _as_optional_str(prior_payload.get("card_id")),
        "card_id": _as_optional_str(prior_payload.get("card_id")),
        "source_name": _as_optional_str(prior_payload.get("source_name")),
        "artifact_family": _as_optional_str(prior_payload.get("artifact_family")),
        "score_bonus": score_bonus,
        "confidence": _as_float(prior_payload.get("confidence")),
    }


def _community_alignment_candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, str]:
    return (
        float(candidate.get("score_bonus", float("-inf"))),
        float(candidate.get("confidence", float("-inf"))),
        float(candidate.get("final_score", float("-inf"))),
        float(candidate.get("heuristic_score", float("-inf"))),
        str(candidate.get("action_id") or ""),
    )


def _community_alignment_trace_line(decision: dict[str, Any]) -> str:
    floor = decision.get("floor")
    domain = decision.get("domain") or "unknown"
    selected = decision.get("selected_subject_id") or decision.get("selected_card_id") or decision.get("selected_action_id") or "none"
    best = decision.get("best_subject_id") or decision.get("best_card_id") or decision.get("best_action_id") or "none"
    return (
        f"floor={floor if floor is not None else '?'} "
        f"domain={domain} "
        f"selected={selected} "
        f"best={best} "
        f"match={'yes' if decision.get('top_choice_match') else 'no'} "
        f"selected_bonus={_format_metric(decision.get('selected_score_bonus'))} "
        f"best_bonus={_format_metric(decision.get('best_score_bonus'))} "
        f"regret={_format_metric(decision.get('alignment_regret'))}"
    )


def _community_alignment_decision_summary(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    decision_count = len(decisions)
    aligned_decision_count = sum(1 for decision in decisions if decision.get("top_choice_match"))
    selected_prior_available_count = sum(1 for decision in decisions if decision.get("selected_prior_available"))
    weighted_top_choice_match_numerator = 0.0
    weighted_top_choice_match_denominator = 0.0
    for decision in decisions:
        weight = _as_float(decision.get("weight"))
        if weight is None or weight < 0.0:
            continue
        weighted_top_choice_match_denominator += weight
        if decision.get("top_choice_match"):
            weighted_top_choice_match_numerator += weight
    return {
        "decision_count": decision_count,
        "eligible_decision_count": decision_count,
        "aligned_decision_count": aligned_decision_count,
        "selected_prior_available_count": selected_prior_available_count,
        "top_choice_match_rate": None if decision_count == 0 else (aligned_decision_count / decision_count),
        "weighted_top_choice_match_rate": (
            None
            if weighted_top_choice_match_denominator <= 0.0
            else (weighted_top_choice_match_numerator / weighted_top_choice_match_denominator)
        ),
        "weighted_top_choice_match_numerator": weighted_top_choice_match_numerator,
        "weighted_top_choice_match_denominator": weighted_top_choice_match_denominator,
        "selected_prior_available_rate": (
            None if decision_count == 0 else (selected_prior_available_count / decision_count)
        ),
        "source_name_histogram": _histogram(decision.get("source_name") for decision in decisions),
        "artifact_family_histogram": _histogram(decision.get("artifact_family") for decision in decisions),
        "selected_source_name_histogram": _histogram(decision.get("selected_source_name") for decision in decisions),
        "selected_artifact_family_histogram": _histogram(
            decision.get("selected_artifact_family") for decision in decisions
        ),
        "subject_id_histogram": _histogram(decision.get("subject_id") for decision in decisions),
        "selected_subject_id_histogram": _histogram(decision.get("selected_subject_id") for decision in decisions),
        "card_id_histogram": _histogram(decision.get("card_id") for decision in decisions),
        "selected_card_id_histogram": _histogram(decision.get("selected_card_id") for decision in decisions),
        "candidate_count_stats": _numeric_summary(
            [decision["candidate_count"] for decision in decisions if decision.get("candidate_count") is not None]
        ),
        "selected_score_bonus_stats": _numeric_summary(
            [decision["selected_score_bonus"] for decision in decisions if decision.get("selected_score_bonus") is not None]
        ),
        "best_score_bonus_stats": _numeric_summary(
            [decision["best_score_bonus"] for decision in decisions if decision.get("best_score_bonus") is not None]
        ),
        "alignment_regret_stats": _numeric_summary(
            [decision["alignment_regret"] for decision in decisions if decision.get("alignment_regret") is not None]
        ),
        "confidence_stats": _numeric_summary(
            [decision["best_confidence"] for decision in decisions if decision.get("best_confidence") is not None]
        ),
        "selected_confidence_stats": _numeric_summary(
            [decision["selected_confidence"] for decision in decisions if decision.get("selected_confidence") is not None]
        ),
        "best_confidence_stats": _numeric_summary(
            [decision["best_confidence"] for decision in decisions if decision.get("best_confidence") is not None]
        ),
    }


def _community_alignment_iteration_summary(iteration_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [decision for payload in iteration_payloads for decision in payload.get("decisions", [])]
    decision_step_count = sum(int(payload.get("decision_step_count", 0) or 0) for payload in iteration_payloads)
    eligible_counts = [float(payload.get("eligible_decision_count", 0) or 0) for payload in iteration_payloads]
    aligned_counts = [
        float(
            sum(1 for decision in payload.get("decisions", []) if decision.get("top_choice_match"))
        )
        for payload in iteration_payloads
    ]
    summary = _community_alignment_decision_summary(decisions)
    summary.update(
        {
            "iterations_with_alignment_decisions": sum(
                1 for payload in iteration_payloads if int(payload.get("eligible_decision_count", 0) or 0) > 0
            ),
            "decision_step_count": decision_step_count,
            "opportunity_coverage": None if decision_step_count == 0 else (len(decisions) / decision_step_count),
            "decision_step_count_stats": _numeric_summary(
                [float(payload.get("decision_step_count", 0) or 0) for payload in iteration_payloads]
            ),
            "eligible_decision_count_stats": _numeric_summary(eligible_counts),
            "aligned_decision_count_stats": _numeric_summary(aligned_counts),
            "floor_histogram": _histogram(decision.get("floor") for decision in decisions),
            "decision_stage_histogram": _histogram(decision.get("decision_stage") for decision in decisions),
            "domain_histogram": _histogram(decision.get("domain") for decision in decisions),
            "domains": {
                domain: _community_alignment_decision_summary(
                    [decision for decision in decisions if decision.get("domain") == domain]
                )
                for domain in sorted({_as_optional_str(decision.get("domain")) for decision in decisions if decision.get("domain")})
            },
        }
    )
    return summary


def _pair_route_decisions(
    baseline_decisions: list[dict[str, Any]],
    candidate_decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    baseline_floors = [decision.get("floor") for decision in baseline_decisions]
    candidate_floors = [decision.get("floor") for decision in candidate_decisions]
    if _floors_are_pairable(baseline_floors) and _floors_are_pairable(candidate_floors):
        candidate_by_floor = {decision["floor"]: decision for decision in candidate_decisions}
        pairs = [
            {
                "pairing_key": f"floor:{decision['floor']}",
                "floor": decision["floor"],
                "baseline": decision,
                "candidate": candidate_by_floor[decision["floor"]],
            }
            for decision in baseline_decisions
            if decision.get("floor") in candidate_by_floor
        ]
        return pairs, "floor"
    shared_count = min(len(baseline_decisions), len(candidate_decisions))
    return (
        [
            {
                "pairing_key": f"order:{index + 1}",
                "floor": baseline_decisions[index].get("floor") or candidate_decisions[index].get("floor"),
                "baseline": baseline_decisions[index],
                "candidate": candidate_decisions[index],
            }
            for index in range(shared_count)
        ],
        "ordinal",
    )


def _evaluate_strategic_promotion(
    *,
    case: CompareBenchmarkCaseSpec,
    metrics: dict[str, Any],
    seed_set_diagnostics: dict[str, Any],
    route_comparison: dict[str, Any],
    capability_comparison: dict[str, Any],
    shadow_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if case.promotion is None:
        return {
            "enabled": False,
            "passed": None,
            "check_count": 0,
            "checks": [],
            "failed_checks": [],
            "rollback_signals": [],
            "promotion_candidate_count": 0,
            "rollback_signal_count": 0,
        }
    effective_seed_coverage = _minimum_defined(
        dict(seed_set_diagnostics.get("baseline", {})).get("seed_set_coverage"),
        dict(seed_set_diagnostics.get("candidate", {})).get("seed_set_coverage"),
    )
    requested_seed_set = list(dict(seed_set_diagnostics.get("baseline", {})).get("requested_seed_set", []))
    if effective_seed_coverage is None and not requested_seed_set:
        effective_seed_coverage = 1.0
    route_decision_pair_count = _as_float(route_comparison.get("route_decision_pair_count"))
    check_specs = [
        ("seed_set_coverage", effective_seed_coverage, case.promotion.min_seed_set_coverage, ">="),
        (
            "route_decision_count",
            route_decision_pair_count,
            float(case.promotion.min_route_decision_count),
            ">=",
        ),
        ("delta_total_reward", _metric_mean(metrics.get("delta_total_reward")), case.promotion.min_delta_total_reward, ">="),
        (
            "delta_combat_win_rate",
            _metric_mean(metrics.get("delta_combat_win_rate")),
            case.promotion.min_delta_combat_win_rate,
            ">=",
        ),
        (
            "new_non_combat_capability_regressions",
            float(int(capability_comparison.get("new_regression_count", 0) or 0)),
            float(case.promotion.max_new_non_combat_capability_regressions),
            "<=",
        ),
    ]
    if case.promotion.max_candidate_non_combat_capability_issues is not None:
        check_specs.append(
            (
                "candidate_non_combat_capability_issues",
                float(int(capability_comparison.get("candidate_issue_count", 0) or 0)),
                float(case.promotion.max_candidate_non_combat_capability_issues),
                "<=",
            )
        )
    if route_decision_pair_count is not None and route_decision_pair_count > 0:
        check_specs.extend(
            [
                (
                    "route_decision_overlap_rate",
                    _as_float(route_comparison.get("route_decision_overlap_rate")),
                    case.promotion.min_route_decision_overlap_rate,
                    ">=",
                ),
                (
                    "delta_route_quality_score",
                    _metric_mean(metrics.get("delta_route_quality_score")),
                    case.promotion.min_delta_route_quality_score,
                    ">=",
                ),
                (
                    "delta_pre_boss_readiness",
                    _metric_mean(metrics.get("delta_pre_boss_readiness")),
                    case.promotion.min_delta_pre_boss_readiness,
                    ">=",
                ),
                (
                    "delta_route_risk_score",
                    _metric_mean(metrics.get("delta_route_risk_score")),
                    case.promotion.max_delta_route_risk_score,
                    "<=",
                ),
            ]
        )
    if shadow_payload is not None:
        shadow_delta_metrics = dict(shadow_payload.get("delta_metrics", {}))
        if case.promotion.min_shadow_comparable_encounter_count is not None:
            check_specs.append(
                (
                    "shadow_comparable_encounter_count",
                    _as_float(shadow_payload.get("comparable_encounter_count")),
                    float(case.promotion.min_shadow_comparable_encounter_count),
                    ">=",
                )
            )
        if case.promotion.min_shadow_candidate_advantage_rate is not None:
            check_specs.append(
                (
                    "shadow_candidate_advantage_rate",
                    _as_float(shadow_payload.get("candidate_advantage_rate")),
                    case.promotion.min_shadow_candidate_advantage_rate,
                    ">=",
                )
            )
        if case.promotion.min_shadow_delta_first_action_match_rate is not None:
            check_specs.append(
                (
                    "shadow_delta_first_action_match_rate",
                    _as_float(shadow_delta_metrics.get("delta_first_action_match_rate")),
                    case.promotion.min_shadow_delta_first_action_match_rate,
                    ">=",
                )
            )
        if case.promotion.min_shadow_delta_trace_hit_rate is not None:
            check_specs.append(
                (
                    "shadow_delta_trace_hit_rate",
                    _as_float(shadow_delta_metrics.get("delta_trace_hit_rate")),
                    case.promotion.min_shadow_delta_trace_hit_rate,
                    ">=",
                )
            )
    checks: list[dict[str, Any]] = []
    for name, actual, threshold, comparator in check_specs:
        passed = False
        if actual is not None:
            passed = actual >= threshold if comparator == ">=" else actual <= threshold
        checks.append(
            {
                "name": name,
                "actual": actual,
                "threshold": threshold,
                "comparator": comparator,
                "passed": passed,
            }
        )
    failed_checks = [check for check in checks if not check["passed"]]
    rollback_signals = [
        {
            "name": check["name"],
            "reason": f"{check['name']} failed strategic promotion gate.",
            "actual": check["actual"],
            "threshold": check["threshold"],
            "comparator": check["comparator"],
        }
        for check in failed_checks
        if check["name"]
        in {
            "seed_set_coverage",
            "route_decision_count",
            "route_decision_overlap_rate",
            "delta_route_quality_score",
            "delta_pre_boss_readiness",
            "delta_route_risk_score",
            "new_non_combat_capability_regressions",
            "candidate_non_combat_capability_issues",
            "shadow_comparable_encounter_count",
            "shadow_candidate_advantage_rate",
            "shadow_delta_first_action_match_rate",
            "shadow_delta_trace_hit_rate",
        }
    ]
    passed = all(check["passed"] for check in checks)
    return {
        "enabled": True,
        "passed": passed,
        "effective_seed_set_coverage": effective_seed_coverage,
        "route_decision_pair_count": route_comparison.get("route_decision_pair_count"),
        "capability_comparison": capability_comparison,
        "shadow_comparable_encounter_count": None if shadow_payload is None else shadow_payload.get("comparable_encounter_count"),
        "check_count": len(checks),
        "checks": checks,
        "failed_checks": failed_checks,
        "rollback_signals": rollback_signals,
        "promotion_candidate_count": 1 if passed else 0,
        "rollback_signal_count": len(rollback_signals),
    }


def _aggregate_strategic_session_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    route_profile_histogram: Counter[str] = Counter()
    for summary in summaries:
        reason_histogram = summary.get("route_planner_reason_tag_histogram", {})
        if isinstance(reason_histogram, dict) and reason_histogram:
            profile = "+".join(sorted(str(key) for key in reason_histogram))
            route_profile_histogram[profile] += 1
    return {
        "route_planner_step_count": sum(int(summary.get("route_planner_step_count", 0) or 0) for summary in summaries),
        "boss_histogram": _merge_counter_payload(summary.get("route_planner_boss_histogram", {}) for summary in summaries),
        "route_reason_tag_histogram": _merge_counter_payload(
            summary.get("route_planner_reason_tag_histogram", {}) for summary in summaries
        ),
        "route_profile_histogram": dict(route_profile_histogram),
        "route_path_length_stats": _merge_numeric_summary_stats(
            summary.get("route_planner_path_length_stats", {}) for summary in summaries
        ),
        "selected_route_score_stats": _merge_numeric_summary_stats(
            summary.get("route_planner_selected_score_stats", {}) for summary in summaries
        ),
        "route_decision_count": 0,
        "route_decision_count_stats": {"count": 0, "min": None, "mean": None, "max": None},
        "first_rest_distance_stats": {"count": 0, "min": None, "mean": None, "max": None},
        "first_elite_distance_stats": {"count": 0, "min": None, "mean": None, "max": None},
        "route_risk_score_stats": {"count": 0, "min": None, "mean": None, "max": None},
        "pre_boss_readiness_stats": {"count": 0, "min": None, "mean": None, "max": None},
        "route_quality_score_stats": {"count": 0, "min": None, "mean": None, "max": None},
    }


def _suite_scenario_histograms(case_payloads: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "floor_band": _histogram(_scenario_floor_band(dict(case_payload.get("scenario", {}))) for case_payload in case_payloads),
        "boss_ids": _histogram(
            boss_id
            for case_payload in case_payloads
            for boss_id in list(dict(case_payload.get("scenario", {})).get("boss_ids", []))
        ),
        "act_ids": _histogram(
            act_id
            for case_payload in case_payloads
            for act_id in list(dict(case_payload.get("scenario", {})).get("act_ids", []))
        ),
        "planner_strategies": _histogram(
            planner_strategy
            for case_payload in case_payloads
            for planner_strategy in list(dict(case_payload.get("scenario", {})).get("planner_strategies", []))
        ),
        "route_reason_tags": _histogram(
            reason_tag
            for case_payload in case_payloads
            for reason_tag in list(dict(case_payload.get("scenario", {})).get("route_reason_tags", []))
        ),
        "route_profiles": _histogram(
            route_profile
            for case_payload in case_payloads
            for route_profile in list(dict(case_payload.get("scenario", {})).get("route_profiles", []))
        ),
    }


def _suite_case_strategic_summary(case_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    eval_strategic = [
        dict(case_payload.get("strategic", {}))
        for case_payload in case_payloads
        if case_payload.get("mode") == "eval"
    ]
    compare_baseline = [
        dict(dict(case_payload.get("strategic", {})).get("baseline", {}))
        for case_payload in case_payloads
        if case_payload.get("mode") == "compare"
    ]
    compare_candidate = [
        dict(dict(case_payload.get("strategic", {})).get("candidate", {}))
        for case_payload in case_payloads
        if case_payload.get("mode") == "compare"
    ]
    return {
        "eval": _merge_strategic_case_payloads(eval_strategic),
        "compare": {
            "baseline": _merge_strategic_case_payloads(compare_baseline),
            "candidate": _merge_strategic_case_payloads(compare_candidate),
        },
    }


def _suite_non_combat_capability_summary(case_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    eval_payloads = [
        dict(case_payload.get("non_combat_capability", {}))
        for case_payload in case_payloads
        if case_payload.get("mode") == "eval"
    ]
    compare_baseline_payloads = [
        dict(dict(case_payload.get("non_combat_capability", {})).get("baseline", {}))
        for case_payload in case_payloads
        if case_payload.get("mode") == "compare"
    ]
    compare_candidate_payloads = [
        dict(dict(case_payload.get("non_combat_capability", {})).get("candidate", {}))
        for case_payload in case_payloads
        if case_payload.get("mode") == "compare"
    ]
    compare_payloads = [
        dict(dict(case_payload.get("non_combat_capability", {})).get("comparison", {}))
        for case_payload in case_payloads
        if case_payload.get("mode") == "compare"
    ]
    new_regression_histogram: Counter[str] = Counter()
    for payload in compare_payloads:
        new_regression_histogram.update(
            {
                str(key): int(value)
                for key, value in dict(payload.get("new_regression_histogram", {})).items()
            }
        )
    return {
        "eval": merge_capability_summaries(eval_payloads),
        "compare": {
            "baseline": merge_capability_summaries(compare_baseline_payloads),
            "candidate": merge_capability_summaries(compare_candidate_payloads),
            "comparison": {
                "configured_case_count": len(compare_payloads),
                "baseline_issue_count": int(
                    sum(int(payload.get("baseline_issue_count", 0) or 0) for payload in compare_payloads)
                ),
                "candidate_issue_count": int(
                    sum(int(payload.get("candidate_issue_count", 0) or 0) for payload in compare_payloads)
                ),
                "delta_issue_count": int(
                    sum(int(payload.get("delta_issue_count", 0) or 0) for payload in compare_payloads)
                ),
                "new_regression_count": int(
                    sum(int(payload.get("new_regression_count", 0) or 0) for payload in compare_payloads)
                ),
                "new_regression_keys": sorted(new_regression_histogram),
                "new_regression_histogram": dict(new_regression_histogram),
            },
        },
    }


def _suite_community_alignment_summary(case_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    eval_iteration_payloads: list[dict[str, Any]] = []
    compare_baseline_iteration_payloads: list[dict[str, Any]] = []
    compare_candidate_iteration_payloads: list[dict[str, Any]] = []
    compare_payloads: list[dict[str, Any]] = []
    for case_payload in case_payloads:
        community_alignment = case_payload.get("community_alignment")
        if not isinstance(community_alignment, dict):
            continue
        if case_payload.get("mode") == "eval":
            eval_iteration_payloads.extend(
                dict(payload) for payload in community_alignment.get("iterations", []) if isinstance(payload, dict)
            )
            continue
        if case_payload.get("mode") != "compare":
            continue
        baseline_payload = community_alignment.get("baseline")
        candidate_payload = community_alignment.get("candidate")
        comparison_payload = community_alignment.get("comparison")
        if isinstance(baseline_payload, dict):
            compare_baseline_iteration_payloads.extend(
                dict(payload) for payload in baseline_payload.get("iterations", []) if isinstance(payload, dict)
            )
        if isinstance(candidate_payload, dict):
            compare_candidate_iteration_payloads.extend(
                dict(payload) for payload in candidate_payload.get("iterations", []) if isinstance(payload, dict)
            )
        if isinstance(comparison_payload, dict):
            compare_payloads.append(dict(comparison_payload))
    delta_top_choice_match_rates = [
        float(value)
        for payload in compare_payloads
        for value in list(dict(payload.get("metric_inputs", {})).get("delta_top_choice_match_rates", []))
    ]
    delta_weighted_top_choice_match_rates = [
        float(value)
        for payload in compare_payloads
        for value in list(dict(payload.get("metric_inputs", {})).get("delta_weighted_top_choice_match_rates", []))
    ]
    delta_opportunity_coverages = [
        float(value)
        for payload in compare_payloads
        for value in list(dict(payload.get("metric_inputs", {})).get("delta_opportunity_coverages", []))
    ]
    delta_selected_prior_available_rates = [
        float(value)
        for payload in compare_payloads
        for value in list(dict(payload.get("metric_inputs", {})).get("delta_selected_prior_available_rates", []))
    ]
    delta_alignment_regrets = [
        float(value)
        for payload in compare_payloads
        for value in list(dict(payload.get("metric_inputs", {})).get("delta_alignment_regrets", []))
    ]
    delta_selected_score_bonus = [
        float(value)
        for payload in compare_payloads
        for value in list(dict(payload.get("metric_inputs", {})).get("delta_selected_score_bonus", []))
    ]
    delta_best_score_bonus = [
        float(value)
        for payload in compare_payloads
        for value in list(dict(payload.get("metric_inputs", {})).get("delta_best_score_bonus", []))
    ]
    return {
        "eval": _community_alignment_iteration_summary(eval_iteration_payloads),
        "compare": {
            "baseline": _community_alignment_iteration_summary(compare_baseline_iteration_payloads),
            "candidate": _community_alignment_iteration_summary(compare_candidate_iteration_payloads),
            "comparison": {
                "configured_case_count": len(compare_payloads),
                "paired_iteration_count": int(
                    sum(int(payload.get("paired_iteration_count", 0) or 0) for payload in compare_payloads)
                ),
                "comparable_iteration_count": int(
                    sum(int(payload.get("comparable_iteration_count", 0) or 0) for payload in compare_payloads)
                ),
                "baseline_only_eligible_iteration_count": int(
                    sum(int(payload.get("baseline_only_eligible_iteration_count", 0) or 0) for payload in compare_payloads)
                ),
                "candidate_only_eligible_iteration_count": int(
                    sum(int(payload.get("candidate_only_eligible_iteration_count", 0) or 0) for payload in compare_payloads)
                ),
                "delta_top_choice_match_rate_stats": _numeric_summary(delta_top_choice_match_rates),
                "delta_weighted_top_choice_match_rate_stats": _numeric_summary(delta_weighted_top_choice_match_rates),
                "delta_opportunity_coverage_stats": _numeric_summary(delta_opportunity_coverages),
                "delta_selected_prior_available_rate_stats": _numeric_summary(delta_selected_prior_available_rates),
                "delta_alignment_regret_stats": _numeric_summary(delta_alignment_regrets),
                "delta_selected_score_bonus_stats": _numeric_summary(delta_selected_score_bonus),
                "delta_best_score_bonus_stats": _numeric_summary(delta_best_score_bonus),
            },
        },
    }


def _suite_shadow_summary(case_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    shadow_payloads: list[dict[str, Any]] = []
    for case_payload in case_payloads:
        shadow_payload = case_payload.get("shadow")
        if isinstance(shadow_payload, dict) and shadow_payload.get("enabled"):
            shadow_payloads.append(dict(shadow_payload))
    delta_first_action_match_values = [
        _as_float(dict(payload.get("delta_metrics", {})).get("delta_first_action_match_rate"))
        for payload in shadow_payloads
    ]
    delta_trace_hit_values = [
        _as_float(dict(payload.get("delta_metrics", {})).get("delta_trace_hit_rate"))
        for payload in shadow_payloads
    ]
    agreement_values = [_as_float(payload.get("agreement_rate")) for payload in shadow_payloads]
    candidate_advantage_values = [_as_float(payload.get("candidate_advantage_rate")) for payload in shadow_payloads]
    comparable_counts = [_as_float(payload.get("comparable_encounter_count")) for payload in shadow_payloads]
    return {
        "configured_case_count": len(shadow_payloads),
        "comparable_encounter_count": int(
            sum(int(payload.get("comparable_encounter_count", 0) or 0) for payload in shadow_payloads)
        ),
        "encounter_count": int(sum(int(payload.get("encounter_count", 0) or 0) for payload in shadow_payloads)),
        "boss_histogram": _merge_counter_payload(payload.get("boss_histogram", {}) for payload in shadow_payloads),
        "encounter_family_histogram": _merge_counter_payload(
            payload.get("encounter_family_histogram", {}) for payload in shadow_payloads
        ),
        "agreement_rate_stats": _numeric_summary([value for value in agreement_values if value is not None]),
        "candidate_advantage_rate_stats": _numeric_summary(
            [value for value in candidate_advantage_values if value is not None]
        ),
        "delta_first_action_match_rate_stats": _numeric_summary(
            [value for value in delta_first_action_match_values if value is not None]
        ),
        "delta_trace_hit_rate_stats": _numeric_summary([value for value in delta_trace_hit_values if value is not None]),
        "comparable_encounter_count_stats": _numeric_summary([value for value in comparable_counts if value is not None]),
    }


def _merge_strategic_case_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        return {
            "route_planner_step_count": 0,
            "route_decision_count": 0,
            "boss_histogram": {},
            "route_reason_tag_histogram": {},
            "route_profile_histogram": {},
            "route_decision_count_stats": {"count": 0, "min": None, "mean": None, "max": None},
            "route_path_length_stats": {"count": 0, "min": None, "mean": None, "max": None},
            "selected_route_score_stats": {"count": 0, "min": None, "mean": None, "max": None},
            "first_rest_distance_stats": {"count": 0, "min": None, "mean": None, "max": None},
            "first_elite_distance_stats": {"count": 0, "min": None, "mean": None, "max": None},
            "route_risk_score_stats": {"count": 0, "min": None, "mean": None, "max": None},
            "pre_boss_readiness_stats": {"count": 0, "min": None, "mean": None, "max": None},
            "route_quality_score_stats": {"count": 0, "min": None, "mean": None, "max": None},
        }
    return {
        "route_planner_step_count": sum(int(payload.get("route_planner_step_count", 0) or 0) for payload in payloads),
        "route_decision_count": sum(int(payload.get("route_decision_count", 0) or 0) for payload in payloads),
        "boss_histogram": _merge_counter_payload(payload.get("boss_histogram", {}) for payload in payloads),
        "route_reason_tag_histogram": _merge_counter_payload(
            payload.get("route_reason_tag_histogram", {}) for payload in payloads
        ),
        "route_profile_histogram": _merge_counter_payload(
            payload.get("route_profile_histogram", {}) for payload in payloads
        ),
        "route_decision_count_stats": _merge_numeric_summary_stats(
            payload.get("route_decision_count_stats", {}) for payload in payloads
        ),
        "route_path_length_stats": _merge_numeric_summary_stats(
            payload.get("route_path_length_stats", {}) for payload in payloads
        ),
        "selected_route_score_stats": _merge_numeric_summary_stats(
            payload.get("selected_route_score_stats", {}) for payload in payloads
        ),
        "first_rest_distance_stats": _merge_numeric_summary_stats(
            payload.get("first_rest_distance_stats", {}) for payload in payloads
        ),
        "first_elite_distance_stats": _merge_numeric_summary_stats(
            payload.get("first_elite_distance_stats", {}) for payload in payloads
        ),
        "route_risk_score_stats": _merge_numeric_summary_stats(
            payload.get("route_risk_score_stats", {}) for payload in payloads
        ),
        "pre_boss_readiness_stats": _merge_numeric_summary_stats(
            payload.get("pre_boss_readiness_stats", {}) for payload in payloads
        ),
        "route_quality_score_stats": _merge_numeric_summary_stats(
            payload.get("route_quality_score_stats", {}) for payload in payloads
        ),
    }


def _suite_public_source_summary(case_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    for case_payload in case_payloads:
        public_sources = case_payload.get("public_sources")
        if not isinstance(public_sources, dict):
            continue
        for payload in public_sources.values() if {"baseline", "candidate"} & set(public_sources) else [public_sources]:
            if not isinstance(payload, dict):
                continue
            community_prior = payload.get("community_prior")
            if not isinstance(community_prior, dict):
                continue
            items = community_prior.get("diagnostics")
            if not isinstance(items, dict):
                continue
            diagnostics.extend(dict(item) for item in items.values() if isinstance(item, dict))
    return {
        "configured_source_count": len(diagnostics),
        "artifact_family_histogram": _histogram(item.get("artifact_family") for item in diagnostics),
        "source_name_histogram": _histogram(item.get("source_name") for item in diagnostics),
        "stat_family_histogram": _histogram(item.get("stat_family") for item in diagnostics),
        "age_days_stats": _numeric_summary(
            [float(item["age_days"]) for item in diagnostics if item.get("age_days") is not None]
        ),
    }


def _suite_promotion_summary(case_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    promotion_payloads: list[dict[str, Any]] = []
    for case_payload in case_payloads:
        promotion_payload = case_payload.get("promotion")
        if isinstance(promotion_payload, dict) and promotion_payload.get("enabled"):
            promotion_payloads.append(dict(promotion_payload))
    return {
        "configured_case_count": len(promotion_payloads),
        "passed_case_count": sum(1 for payload in promotion_payloads if payload.get("passed")),
        "failed_case_count": sum(1 for payload in promotion_payloads if payload.get("passed") is False),
        "promotion_candidate_count": sum(
            int(payload.get("promotion_candidate_count", 0) or 0) for payload in promotion_payloads
        ),
        "rollback_signal_count": sum(int(payload.get("rollback_signal_count", 0) or 0) for payload in promotion_payloads),
    }


def _scenario_floor_band(scenario: dict[str, Any]) -> str:
    floor_min = scenario.get("floor_min")
    floor_max = scenario.get("floor_max")
    if floor_min is None and floor_max is None:
        return "all"
    return f"{floor_min if floor_min is not None else '*'}-{floor_max if floor_max is not None else '*'}"


def _primary_metric_payload(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "estimate": summary.get("mean"),
        "ci_low": summary.get("ci_low"),
        "ci_high": summary.get("ci_high"),
    }


def _observed_seed_metadata(summary_payload: dict[str, Any]) -> dict[str, Any]:
    histogram = summary_payload.get("observed_run_seed_histogram", {})
    if not isinstance(histogram, dict):
        histogram = {}
    return {
        "observed_run_seeds": _merge_seed_lists([summary_payload.get("observed_run_seeds", [])]),
        "observed_run_seed_histogram": _merge_counter_payload([histogram]),
        "runs_without_observed_seed": int(summary_payload.get("runs_without_observed_seed", 0) or 0),
        "last_observed_seed": summary_payload.get("last_observed_seed"),
    }


def _seed_set_diagnostics_from_aggregate(
    requested_seed_set: list[str],
    aggregate_payload: dict[str, Any],
) -> dict[str, Any]:
    histogram = aggregate_payload.get("observed_run_seed_histogram", {})
    if not isinstance(histogram, dict):
        histogram = {}
    return _seed_set_diagnostics(
        requested_seed_set=requested_seed_set,
        observed_run_seed_histogram=_merge_counter_payload([histogram]),
        runs_without_observed_seed=int(aggregate_payload.get("runs_without_observed_seed", 0) or 0),
    )


def _seed_set_diagnostics(
    *,
    requested_seed_set: list[str],
    observed_run_seed_histogram: dict[str, int],
    runs_without_observed_seed: int,
) -> dict[str, Any]:
    observed_run_seeds = _merge_seed_lists([list(observed_run_seed_histogram)])
    matched_seed_set = [seed for seed in requested_seed_set if seed in observed_run_seed_histogram]
    missing_seed_set = [seed for seed in requested_seed_set if seed not in observed_run_seed_histogram]
    unexpected_seed_set = sorted(seed for seed in observed_run_seeds if seed not in requested_seed_set)
    seed_set_coverage = None
    if requested_seed_set:
        seed_set_coverage = len(matched_seed_set) / len(requested_seed_set)
    return {
        "requested_seed_set": list(requested_seed_set),
        "observed_run_seeds": observed_run_seeds,
        "observed_run_seed_histogram": dict(observed_run_seed_histogram),
        "matched_seed_set": matched_seed_set,
        "missing_seed_set": missing_seed_set,
        "unexpected_seed_set": unexpected_seed_set,
        "seed_set_coverage": seed_set_coverage,
        "runs_without_observed_seed": runs_without_observed_seed,
    }


def _merge_seed_lists(seed_lists) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for seed_list in seed_lists:
        if not isinstance(seed_list, list):
            continue
        for seed in seed_list:
            normalized = _as_optional_str(seed)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return sorted(merged)


def _last_non_empty_seed(values) -> str | None:
    last_seed: str | None = None
    for value in values:
        normalized = _as_optional_str(value)
        if normalized is not None:
            last_seed = normalized
    return last_seed


def _merge_counter_payload(histograms) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for histogram in histograms:
        if not isinstance(histogram, dict):
            continue
        for key, value in histogram.items():
            merged[str(key)] += int(value)
    return dict(merged)


def _histogram(values) -> dict[str, int]:
    histogram: Counter[str] = Counter()
    for value in values:
        normalized = _as_optional_str(value)
        if normalized is None:
            continue
        histogram[normalized] += 1
    return dict(histogram)


def _merge_numeric_summary_stats(stats_iterable) -> dict[str, float | int | None]:
    count = 0
    weighted_sum = 0.0
    min_value: float | None = None
    max_value: float | None = None
    for stats in stats_iterable:
        if not isinstance(stats, dict):
            continue
        item_count = int(stats.get("count", 0) or 0)
        item_mean = stats.get("mean")
        if item_count <= 0 or item_mean is None:
            continue
        item_min = stats.get("min")
        item_max = stats.get("max")
        count += item_count
        weighted_sum += float(item_mean) * item_count
        if item_min is not None:
            min_value = float(item_min) if min_value is None else min(min_value, float(item_min))
        if item_max is not None:
            max_value = float(item_max) if max_value is None else max(max_value, float(item_max))
    if count == 0:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": count,
        "min": min_value,
        "mean": weighted_sum / count,
        "max": max_value,
    }


def _numeric_summary(values: Sequence[float | int]) -> dict[str, float | int | None]:
    filtered = [float(value) for value in values]
    if not filtered:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(filtered),
        "min": min(filtered),
        "mean": sum(filtered) / len(filtered),
        "max": max(filtered),
    }


def _eval_iteration_payload(iteration: _EvalIterationReport) -> dict[str, Any]:
    return {
        "iteration_index": iteration.iteration_index,
        "session_name": iteration.session_name,
        "session_dir": str(iteration.session_dir),
        "summary_path": str(iteration.summary_path),
        "log_path": str(iteration.log_path),
        "combat_outcomes_path": str(iteration.combat_outcomes_path),
        "prepare_target": iteration.prepare_target,
        "normalization_report": iteration.normalization_report,
        "start_signature": iteration.start_signature,
        "start_payload": iteration.start_payload,
        "prepare_action_ids": iteration.prepare_action_ids,
        "env_steps": iteration.env_steps,
        "combat_steps": iteration.combat_steps,
        "heuristic_steps": iteration.heuristic_steps,
        "total_reward": iteration.total_reward,
        "stop_reason": iteration.stop_reason,
        "final_screen": iteration.final_screen,
        "completed_run_count": iteration.completed_run_count,
        "completed_combat_count": iteration.completed_combat_count,
        "combat_performance": iteration.combat_performance,
        "observed_run_seeds": iteration.observed_run_seeds,
        "observed_run_seed_histogram": iteration.observed_run_seed_histogram,
        "runs_without_observed_seed": iteration.runs_without_observed_seed,
        "last_observed_seed": iteration.last_observed_seed,
    }


def _bootstrap_summary(
    values: list[float],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    filtered = [float(value) for value in values]
    if not filtered:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "ci_low": None,
            "ci_high": None,
        }

    value_mean = sum(filtered) / len(filtered)
    if len(filtered) == 1:
        value_std = 0.0
    else:
        variance = sum((value - value_mean) ** 2 for value in filtered) / (len(filtered) - 1)
        value_std = math.sqrt(variance)

    bootstrap_means = _bootstrap_means(filtered, resamples=resamples, seed=seed)
    alpha = 1.0 - confidence_level
    ci_low = _percentile(bootstrap_means, alpha / 2.0)
    ci_high = _percentile(bootstrap_means, 1.0 - (alpha / 2.0))
    return {
        "count": len(filtered),
        "mean": value_mean,
        "std": value_std,
        "min": min(filtered),
        "max": max(filtered),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _bootstrap_means(values: list[float], *, resamples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    sample_count = len(values)
    means: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(sample_count)] for _ in range(sample_count)]
        means.append(sum(sample) / sample_count)
    return sorted(means)


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] + ((values[upper] - values[lower]) * weight)


def _stable_seed_offset(value: str) -> int:
    total = 0
    for character in value:
        total = ((total * 33) + ord(character)) % 1_000_000_007
    return total


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    record = {"timestamp_utc": datetime.now(UTC).isoformat(), **payload}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalized_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized_value = _as_optional_str(value)
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
    normalized = _as_optional_str(value)
    if normalized is None:
        return ""
    return normalized.lower().replace(" ", "_")


def _floors_are_pairable(floors: Sequence[Any]) -> bool:
    normalized = [_as_optional_int(value) for value in floors]
    return all(value is not None for value in normalized) and len(set(normalized)) == len(normalized)


def _pair_delta(candidate_value: Any, baseline_value: Any) -> float | None:
    if candidate_value is None or baseline_value is None:
        return None
    return float(candidate_value) - float(baseline_value)


def _route_risk_score(
    *,
    elite_count: int | None,
    monster_count: int | None,
    rest_count: int | None,
    shop_count: int | None,
    elites_before_rest: int | None,
    first_rest_distance: int | None,
) -> float:
    score = float(elite_count or 0)
    score += 0.35 * float(monster_count or 0)
    score += 0.5 * max(float(elites_before_rest or 0) - 1.0, 0.0)
    score -= 0.6 * float(rest_count or 0)
    score -= 0.2 * float(shop_count or 0)
    if first_rest_distance is not None and first_rest_distance > 2:
        score += 0.15 * float(first_rest_distance - 2)
    return score


def _pre_boss_readiness(
    *,
    selected_route_score: float | None,
    rest_count: int | None,
    shop_count: int | None,
    elite_count: int | None,
    remaining_distance_to_boss: int | None,
    first_rest_distance: int | None,
) -> float:
    score = float(selected_route_score or 0.0)
    score += 0.75 * float(rest_count or 0)
    score += 0.45 * float(shop_count or 0)
    score += 0.2 * float(elite_count or 0)
    score -= 0.2 * float(remaining_distance_to_boss or 0)
    if first_rest_distance is not None and first_rest_distance <= 2:
        score += 0.3
    return score


def _route_quality_score(
    *,
    selected_route_score: float | None,
    pre_boss_readiness: float | None,
    route_risk_score: float | None,
) -> float:
    return float(selected_route_score or 0.0) + (0.25 * float(pre_boss_readiness or 0.0)) - (
        0.25 * float(route_risk_score or 0.0)
    )


def _metric_mean(summary: Any) -> float | None:
    if not isinstance(summary, dict):
        return None
    return _as_float(summary.get("mean"))


def _minimum_defined(*values: Any) -> float | None:
    filtered = [_as_float(value) for value in values if value is not None]
    filtered = [value for value in filtered if value is not None]
    if not filtered:
        return None
    return min(filtered)


def _format_metric(value: Any) -> str:
    numeric = _as_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.3f}"
