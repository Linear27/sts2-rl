from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from sts2_rl.data import (
    load_community_card_stat_records,
    load_dataset_summary,
    load_public_run_normalized_summary,
    load_public_run_strategic_stat_records,
    resolve_dataset_split_paths,
)

from .dataset import load_predictor_examples, resolve_predictor_examples_path
from .model import CombatOutcomePredictor
from .schema import PredictorExample

PREDICTOR_REPORT_SCHEMA_VERSION = 1
BENCHMARK_SUITE_SUMMARY_FILENAME = "benchmark-suite-summary.json"
BENCHMARK_CASE_SUMMARY_FILENAME = "case-summary.json"


@dataclass(frozen=True)
class PredictorCalibrationThresholds:
    outcome_ece_max: float = 0.12
    outcome_brier_max: float = 0.25
    reward_rmse_max: float = 3.00
    damage_rmse_max: float = 24.00

    def as_dict(self) -> dict[str, float]:
        return {
            "outcome_ece_max": self.outcome_ece_max,
            "outcome_brier_max": self.outcome_brier_max,
            "reward_rmse_max": self.reward_rmse_max,
            "damage_rmse_max": self.damage_rmse_max,
        }


@dataclass(frozen=True)
class PredictorRankingThresholds:
    outcome_pairwise_accuracy_min: float = 0.58
    reward_pairwise_accuracy_min: float = 0.58
    damage_pairwise_accuracy_min: float = 0.58
    reward_ndcg_at_3_min: float = 0.62
    damage_ndcg_at_3_min: float = 0.62

    def as_dict(self) -> dict[str, float]:
        return {
            "outcome_pairwise_accuracy_min": self.outcome_pairwise_accuracy_min,
            "reward_pairwise_accuracy_min": self.reward_pairwise_accuracy_min,
            "damage_pairwise_accuracy_min": self.damage_pairwise_accuracy_min,
            "reward_ndcg_at_3_min": self.reward_ndcg_at_3_min,
            "damage_ndcg_at_3_min": self.damage_ndcg_at_3_min,
        }


@dataclass(frozen=True)
class PredictorBenchmarkComparisonThresholds:
    delta_total_reward_min: float = 0.0
    delta_combat_win_rate_min: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "delta_total_reward_min": self.delta_total_reward_min,
            "delta_combat_win_rate_min": self.delta_combat_win_rate_min,
        }


@dataclass(frozen=True)
class PredictorReportArtifacts:
    output_dir: Path
    summary_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class _ScoredExample:
    example: PredictorExample
    slices: dict[str, str]
    outcome_prediction: float | None
    reward_prediction: float | None
    damage_prediction: float | None


@dataclass(frozen=True)
class _RankingGroup:
    group_id: str
    key: tuple[str, ...]
    slice_values: dict[str, str]
    items: tuple[_ScoredExample, ...]


def default_predictor_report_session_name(prefix: str) -> str:
    return datetime.now(UTC).strftime(f"{prefix}-%Y%m%d-%H%M%S")


def build_predictor_calibration_report(
    *,
    model_path: str | Path,
    dataset_source: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    split: str = "validation",
    bin_count: int = 10,
    min_slice_examples: int = 5,
    thresholds: PredictorCalibrationThresholds | None = None,
    public_aggregate_source: str | Path | None = None,
    public_run_source: str | Path | None = None,
) -> PredictorReportArtifacts:
    if bin_count < 2:
        raise ValueError("Calibration reports require bin_count >= 2.")
    threshold_config = thresholds or PredictorCalibrationThresholds()
    predictor = CombatOutcomePredictor.load(model_path)
    dataset_path, examples_path, effective_split, examples = _load_examples_for_reporting(dataset_source, split=split)
    scored_examples = _score_examples(predictor, examples)

    output_dir = _prepare_output_dir(output_root, session_name or default_predictor_report_session_name("predict-calibration"))
    outcome_summary = _binary_calibration_summary(
        scored_examples,
        prediction_getter=lambda item: item.outcome_prediction,
        label_getter=lambda item: item.example.outcome_win_label,
        weight_getter=lambda item: item.example.outcome_weight,
        bin_count=bin_count,
        include_bins=True,
    )
    reward_summary = _regression_calibration_summary(
        scored_examples,
        prediction_getter=lambda item: item.reward_prediction,
        label_getter=lambda item: item.example.reward_label,
        weight_getter=lambda item: item.example.reward_weight,
        bin_count=bin_count,
        include_bins=True,
    )
    damage_summary = _regression_calibration_summary(
        scored_examples,
        prediction_getter=lambda item: item.damage_prediction,
        label_getter=lambda item: item.example.damage_delta_label,
        weight_getter=lambda item: item.example.damage_weight,
        bin_count=bin_count,
        include_bins=True,
    )
    slices_payload = _slice_calibration_summaries(
        scored_examples,
        min_slice_examples=min_slice_examples,
        bin_count=bin_count,
    )
    public_sources = _build_public_source_diagnostics(
        public_aggregate_source=public_aggregate_source,
        public_run_source=public_run_source,
    )

    promotion = _evaluate_thresholds(
        [
            _max_threshold_check("outcome_ece", outcome_summary.get("ece"), threshold_config.outcome_ece_max),
            _max_threshold_check("outcome_brier_score", outcome_summary.get("brier_score"), threshold_config.outcome_brier_max),
            _max_threshold_check("reward_rmse", reward_summary.get("rmse"), threshold_config.reward_rmse_max),
            _max_threshold_check("damage_rmse", damage_summary.get("rmse"), threshold_config.damage_rmse_max),
        ]
    )

    summary_payload = {
        "schema_version": PREDICTOR_REPORT_SCHEMA_VERSION,
        "report_kind": "predictor_calibration",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_path": str(Path(model_path).expanduser().resolve()),
        "model_label": Path(model_path).name,
        "dataset_path": str(dataset_path),
        "examples_path": str(examples_path),
        "split": effective_split,
        "example_count": len(scored_examples),
        "bin_count": bin_count,
        "min_slice_examples": min_slice_examples,
        "thresholds": threshold_config.as_dict(),
        "public_sources": public_sources,
        "promotion": promotion,
        "overall": {
            "outcome_win_probability": outcome_summary,
            "expected_reward": reward_summary,
            "expected_damage_delta": damage_summary,
        },
        "slices": slices_payload,
    }
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_calibration_markdown(summary_payload), encoding="utf-8")
    return PredictorReportArtifacts(output_dir=output_dir, summary_path=summary_path, markdown_path=markdown_path)


def build_predictor_ranking_report(
    *,
    model_path: str | Path,
    dataset_source: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    split: str = "validation",
    group_by: Sequence[str] | None = None,
    top_k: int = 3,
    min_group_size: int = 2,
    thresholds: PredictorRankingThresholds | None = None,
    public_aggregate_source: str | Path | None = None,
    public_run_source: str | Path | None = None,
) -> PredictorReportArtifacts:
    if top_k < 1:
        raise ValueError("Ranking reports require top_k >= 1.")
    if min_group_size < 2:
        raise ValueError("Ranking reports require min_group_size >= 2.")
    ranking_group_by = tuple(group_by or ("character", "floor_band", "encounter_family"))
    threshold_config = thresholds or PredictorRankingThresholds()
    predictor = CombatOutcomePredictor.load(model_path)
    dataset_path, examples_path, effective_split, examples = _load_examples_for_reporting(dataset_source, split=split)
    scored_examples = _score_examples(predictor, examples)
    groups = _build_ranking_groups(scored_examples, group_by=ranking_group_by, min_group_size=min_group_size)

    overall = {
        "outcome_win_probability": _ranking_summary(
            groups,
            score_getter=lambda item: item.outcome_prediction,
            label_getter=lambda item: item.example.outcome_win_label,
            weight_getter=lambda item: item.example.outcome_weight,
            top_k=top_k,
        ),
        "expected_reward": _ranking_summary(
            groups,
            score_getter=lambda item: item.reward_prediction,
            label_getter=lambda item: item.example.reward_label,
            weight_getter=lambda item: item.example.reward_weight,
            top_k=top_k,
        ),
        "expected_damage_delta": _ranking_summary(
            groups,
            score_getter=lambda item: item.damage_prediction,
            label_getter=lambda item: item.example.damage_delta_label,
            weight_getter=lambda item: item.example.damage_weight,
            top_k=top_k,
        ),
    }
    slices_payload = _slice_ranking_summaries(groups, top_k=top_k)
    public_sources = _build_public_source_diagnostics(
        public_aggregate_source=public_aggregate_source,
        public_run_source=public_run_source,
    )

    promotion = _evaluate_thresholds(
        [
            _min_threshold_check(
                "outcome_pairwise_accuracy",
                overall["outcome_win_probability"].get("pairwise_accuracy"),
                threshold_config.outcome_pairwise_accuracy_min,
            ),
            _min_threshold_check(
                "reward_pairwise_accuracy",
                overall["expected_reward"].get("pairwise_accuracy"),
                threshold_config.reward_pairwise_accuracy_min,
            ),
            _min_threshold_check(
                "damage_pairwise_accuracy",
                overall["expected_damage_delta"].get("pairwise_accuracy"),
                threshold_config.damage_pairwise_accuracy_min,
            ),
            _min_threshold_check(
                f"reward_ndcg_at_{top_k}",
                overall["expected_reward"].get("ndcg_at_k"),
                threshold_config.reward_ndcg_at_3_min,
            ),
            _min_threshold_check(
                f"damage_ndcg_at_{top_k}",
                overall["expected_damage_delta"].get("ndcg_at_k"),
                threshold_config.damage_ndcg_at_3_min,
            ),
        ]
    )

    summary_payload = {
        "schema_version": PREDICTOR_REPORT_SCHEMA_VERSION,
        "report_kind": "predictor_ranking",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_path": str(Path(model_path).expanduser().resolve()),
        "model_label": Path(model_path).name,
        "dataset_path": str(dataset_path),
        "examples_path": str(examples_path),
        "split": effective_split,
        "example_count": len(scored_examples),
        "group_count": len(groups),
        "group_by": list(ranking_group_by),
        "top_k": top_k,
        "min_group_size": min_group_size,
        "thresholds": threshold_config.as_dict(),
        "public_sources": public_sources,
        "promotion": promotion,
        "overall": overall,
        "slices": slices_payload,
    }
    output_dir = _prepare_output_dir(output_root, session_name or default_predictor_report_session_name("predict-ranking"))
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_ranking_markdown(summary_payload), encoding="utf-8")
    return PredictorReportArtifacts(output_dir=output_dir, summary_path=summary_path, markdown_path=markdown_path)


def build_predictor_benchmark_comparison_report(
    *,
    sources: Sequence[str | Path],
    output_root: str | Path,
    session_name: str | None = None,
    thresholds: PredictorBenchmarkComparisonThresholds | None = None,
    public_aggregate_source: str | Path | None = None,
    public_run_source: str | Path | None = None,
) -> PredictorReportArtifacts:
    if not sources:
        raise ValueError("Benchmark comparison reports require at least one source.")
    threshold_config = thresholds or PredictorBenchmarkComparisonThresholds()
    case_payloads: list[dict[str, Any]] = []
    for source in sources:
        case_payloads.extend(_load_benchmark_case_payloads(source))

    eval_arms: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for payload in case_payloads:
        mode = str(payload.get("mode", ""))
        scenario = dict(payload.get("scenario", {}))
        if mode == "eval":
            predictor_payload = dict(payload.get("predictor", {}))
            scenario_slices = _scenario_slice_fields(scenario)
            eval_arms.append(
                {
                    "case_id": str(payload.get("case_id", "unknown")),
                    "mode": str(predictor_payload.get("mode", "heuristic_only")),
                    "hooks": list(predictor_payload.get("hooks", [])),
                    "scenario_floor_band": _scenario_floor_band(scenario),
                    **scenario_slices,
                    "metrics": {
                        "total_reward": _metric_summary_mean(payload.get("metrics", {}).get("total_reward")),
                        "combat_win_rate": _metric_summary_mean(payload.get("metrics", {}).get("combat_win_rate")),
                        "reward_per_combat": _metric_summary_mean(payload.get("metrics", {}).get("reward_per_combat")),
                    },
                }
            )
            continue
        if mode != "compare":
            continue
        predictor_payload = dict(payload.get("predictor", {}))
        baseline_predictor = dict(predictor_payload.get("baseline", {}))
        candidate_predictor = dict(predictor_payload.get("candidate", {}))
        metrics = dict(payload.get("metrics", {}))
        scenario_slices = _scenario_slice_fields(scenario)
        comparison_entry = {
            "case_id": str(payload.get("case_id", "unknown")),
            "baseline_mode": str(baseline_predictor.get("mode", "heuristic_only")),
            "candidate_mode": str(candidate_predictor.get("mode", "heuristic_only")),
            "baseline_hooks": list(baseline_predictor.get("hooks", [])),
            "candidate_hooks": list(candidate_predictor.get("hooks", [])),
            "scenario": scenario,
            "scenario_floor_band": _scenario_floor_band(scenario),
            **scenario_slices,
            "better_checkpoint_label": payload.get("better_checkpoint_label"),
            "delta_total_reward": _metric_summary_mean(metrics.get("delta_total_reward"))
            if metrics.get("delta_total_reward") is not None
            else _float_or_none(dict(payload.get("delta_metrics", {})).get("mean_total_reward")),
            "delta_combat_win_rate": _metric_summary_mean(metrics.get("delta_combat_win_rate"))
            if metrics.get("delta_combat_win_rate") is not None
            else _float_or_none(dict(payload.get("delta_metrics", {})).get("combat_win_rate")),
            "delta_reward_per_combat": _metric_summary_mean(metrics.get("delta_reward_per_combat"))
            if metrics.get("delta_reward_per_combat") is not None
            else _float_or_none(dict(payload.get("delta_metrics", {})).get("reward_per_combat")),
            "baseline": {
                "mean_total_reward": _float_or_none(dict(payload.get("baseline", {})).get("mean_total_reward")),
                "combat_win_rate": _float_or_none(dict(payload.get("baseline", {})).get("combat_win_rate")),
                "reward_per_combat": _float_or_none(dict(payload.get("baseline", {})).get("reward_per_combat")),
            },
            "candidate": {
                "mean_total_reward": _float_or_none(dict(payload.get("candidate", {})).get("mean_total_reward")),
                "combat_win_rate": _float_or_none(dict(payload.get("candidate", {})).get("combat_win_rate")),
                "reward_per_combat": _float_or_none(dict(payload.get("candidate", {})).get("reward_per_combat")),
            },
            "baseline_public_sources": _artifact_family_list(
                dict(payload.get("public_sources", {})).get("baseline")
            ),
            "candidate_public_sources": _artifact_family_list(
                dict(payload.get("public_sources", {})).get("candidate")
            ),
            "delta_community_top_choice_match_rate": _metric_summary_mean(
                dict(payload.get("metrics", {})).get("delta_community_top_choice_match_rate")
            ),
        }
        comparison_entry["promotion"] = _evaluate_thresholds(
            [
                _min_threshold_check(
                    "delta_total_reward",
                    comparison_entry.get("delta_total_reward"),
                    threshold_config.delta_total_reward_min,
                ),
                _min_threshold_check(
                    "delta_combat_win_rate",
                    comparison_entry.get("delta_combat_win_rate"),
                    threshold_config.delta_combat_win_rate_min,
                ),
            ]
        )
        comparisons.append(comparison_entry)

    mode_summaries = _mode_eval_summaries(eval_arms)
    mode_pair_summaries = _mode_pair_comparison_summaries(comparisons)
    scenario_floor_band_summaries = _scenario_floor_band_comparison_summaries(comparisons)
    public_sources = _build_public_source_diagnostics(
        public_aggregate_source=public_aggregate_source,
        public_run_source=public_run_source,
    )
    benchmark_public_sources = _benchmark_public_source_summary(case_payloads)
    public_source_comparisons = _public_source_comparison_summaries(comparisons)
    promotion_candidates = [
        {
            "case_id": comparison["case_id"],
            "baseline_mode": comparison["baseline_mode"],
            "candidate_mode": comparison["candidate_mode"],
        }
        for comparison in comparisons
        if comparison["promotion"]["passed"]
    ]
    rollback_signals = [
        {
            "case_id": comparison["case_id"],
            "baseline_mode": comparison["baseline_mode"],
            "candidate_mode": comparison["candidate_mode"],
            "delta_total_reward": comparison["delta_total_reward"],
            "delta_combat_win_rate": comparison["delta_combat_win_rate"],
        }
        for comparison in comparisons
        if not comparison["promotion"]["passed"]
    ]
    summary_payload = {
        "schema_version": PREDICTOR_REPORT_SCHEMA_VERSION,
        "report_kind": "predictor_benchmark_compare",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_paths": [str(Path(source).expanduser().resolve()) for source in sources],
        "case_count": len(case_payloads),
        "eval_case_count": len(eval_arms),
        "compare_case_count": len(comparisons),
        "thresholds": threshold_config.as_dict(),
        "public_sources": public_sources,
        "benchmark_public_sources": benchmark_public_sources,
        "public_source_comparisons": public_source_comparisons,
        "promotion": {
            "passed": bool(promotion_candidates),
            "promotion_candidate_count": len(promotion_candidates),
            "promotion_candidates": promotion_candidates,
            "rollback_signal_count": len(rollback_signals),
            "rollback_signals": rollback_signals,
        },
        "mode_summaries": mode_summaries,
        "mode_pair_summaries": mode_pair_summaries,
        "scenario_floor_band_summaries": scenario_floor_band_summaries,
        "comparisons": comparisons,
    }
    output_dir = _prepare_output_dir(
        output_root,
        session_name or default_predictor_report_session_name("predict-benchmark-compare"),
    )
    summary_path = output_dir / "summary.json"
    markdown_path = output_dir / "summary.md"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_benchmark_compare_markdown(summary_payload), encoding="utf-8")
    return PredictorReportArtifacts(output_dir=output_dir, summary_path=summary_path, markdown_path=markdown_path)


def _prepare_output_dir(output_root: str | Path, session_name: str) -> Path:
    output_dir = Path(output_root).expanduser().resolve() / session_name
    if output_dir.exists():
        raise FileExistsError(f"Predictor report output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _build_public_source_diagnostics(
    *,
    public_aggregate_source: str | Path | None,
    public_run_source: str | Path | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if public_aggregate_source is not None:
        payload["aggregate_card"] = _aggregate_card_source_diagnostics(public_aggregate_source)
    if public_run_source is not None:
        payload["run_strategic"] = _run_strategic_source_diagnostics(public_run_source)
    return payload


def _aggregate_card_source_diagnostics(source: str | Path) -> dict[str, Any]:
    resolved = Path(source).expanduser().resolve()
    records = load_community_card_stat_records(resolved)
    exemplar = records[0] if records else None
    sample_sizes = [
        float(record.offer_count or record.shop_offer_count or record.run_count or record.deck_presence_runs or 0)
        for record in records
        if (record.offer_count or record.shop_offer_count or record.run_count or record.deck_presence_runs) is not None
    ]
    return {
        "source_path": str(resolved),
        "artifact_family": "community_card_stats",
        "source_name": None if exemplar is None else exemplar.source_name,
        "snapshot_date": None if exemplar is None else exemplar.snapshot_date,
        "age_days": _age_days(None if exemplar is None else exemplar.snapshot_date),
        "record_count": len(records),
        "source_type_histogram": _histogram(record.source_type for record in records),
        "character_histogram": _histogram(record.character_id for record in records),
        "sample_size_stats": _value_distribution_summary(sample_sizes),
    }


def _run_strategic_source_diagnostics(source: str | Path) -> dict[str, Any]:
    resolved = Path(source).expanduser().resolve()
    card_records = load_public_run_strategic_stat_records(resolved, stat_family="card")
    route_records: list[Any]
    try:
        route_records = load_public_run_strategic_stat_records(resolved, stat_family="route")
    except FileNotFoundError:
        route_records = []
    try:
        summary = load_public_run_normalized_summary(resolved if resolved.is_dir() else resolved.parent)
    except FileNotFoundError:
        summary = {}
    exemplar = card_records[0] if card_records else (route_records[0] if route_records else None)
    return {
        "source_path": str(resolved),
        "artifact_family": "public_run_strategic_stats",
        "source_name": None if exemplar is None else exemplar.source_name,
        "snapshot_date": None if exemplar is None else exemplar.snapshot_date,
        "generated_at_utc": summary.get("generated_at_utc"),
        "age_days": _age_days((None if exemplar is None else exemplar.snapshot_date) or _as_optional_str(summary.get("generated_at_utc"))),
        "card_record_count": len(card_records),
        "route_record_count": len(route_records),
        "card_source_type_histogram": _histogram(record.source_type for record in card_records),
        "route_subject_histogram": _histogram(record.subject_id for record in route_records),
        "offer_count_stats": _value_distribution_summary(
            [float(record.offer_count) for record in card_records if record.offer_count is not None]
        ),
        "shop_offer_count_stats": _value_distribution_summary(
            [float(record.shop_offer_count) for record in card_records if record.shop_offer_count is not None]
        ),
        "route_seen_count_stats": _value_distribution_summary(
            [float(record.seen_count) for record in route_records if record.seen_count is not None]
        ),
    }


def _benchmark_public_source_summary(case_payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    for payload in case_payloads:
        public_sources = payload.get("public_sources")
        if not isinstance(public_sources, dict):
            continue
        for nested in public_sources.values() if {"baseline", "candidate"} & set(public_sources) else [public_sources]:
            if not isinstance(nested, dict):
                continue
            community_prior = nested.get("community_prior")
            if not isinstance(community_prior, dict):
                continue
            items = community_prior.get("diagnostics")
            if not isinstance(items, dict):
                continue
            diagnostics.extend(dict(item) for item in items.values() if isinstance(item, dict))
    return {
        "configured_source_count": len(diagnostics),
        "artifact_family_histogram": _histogram(item.get("artifact_family") for item in diagnostics),
        "stat_family_histogram": _histogram(item.get("stat_family") for item in diagnostics),
        "source_name_histogram": _histogram(item.get("source_name") for item in diagnostics),
        "age_days_stats": _value_distribution_summary(
            [float(item["age_days"]) for item in diagnostics if item.get("age_days") is not None]
        ),
    }


def _public_source_comparison_summaries(comparisons: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for comparison in comparisons:
        candidate_sources = comparison.get("candidate_public_sources")
        if not isinstance(candidate_sources, list) or not candidate_sources:
            continue
        for artifact_family in candidate_sources:
            grouped.setdefault(str(artifact_family), []).append(comparison)
    rows: list[dict[str, Any]] = []
    for artifact_family, items in grouped.items():
        rows.append(
            {
                "artifact_family": artifact_family,
                "comparison_count": len(items),
                "mode_pair_histogram": _histogram(
                    f"{item['baseline_mode']}->{item['candidate_mode']}" for item in items
                ),
                "delta_total_reward_stats": _value_distribution_summary(
                    [item["delta_total_reward"] for item in items if item.get("delta_total_reward") is not None]
                ),
                "delta_combat_win_rate_stats": _value_distribution_summary(
                    [item["delta_combat_win_rate"] for item in items if item.get("delta_combat_win_rate") is not None]
                ),
                "delta_community_top_choice_match_rate_stats": _value_distribution_summary(
                    [
                        item["delta_community_top_choice_match_rate"]
                        for item in items
                        if item.get("delta_community_top_choice_match_rate") is not None
                    ]
                ),
            }
        )
    return sorted(rows, key=lambda row: str(row["artifact_family"]))


def _load_examples_for_reporting(
    dataset_source: str | Path,
    *,
    split: str,
) -> tuple[Path, Path, str, list[PredictorExample]]:
    dataset_path = Path(dataset_source).expanduser().resolve()
    normalized_split = split.strip().lower()
    if normalized_split not in {"all", "train", "validation", "test"}:
        raise ValueError("split must be one of: all, train, validation, test.")
    if dataset_path.is_dir():
        try:
            summary_payload = load_dataset_summary(dataset_path)
        except FileNotFoundError:
            summary_payload = None
        if (
            summary_payload is not None
            and summary_payload.get("dataset_kind") == "predictor_combat_outcomes"
            and normalized_split != "all"
        ):
            split_paths = resolve_dataset_split_paths(dataset_path)
            split_path = split_paths.get(normalized_split)
            if split_path is not None and split_path.exists():
                return dataset_path, split_path, normalized_split, load_predictor_examples(split_path)
    examples_path = resolve_predictor_examples_path(dataset_path)
    return dataset_path, examples_path, "all", load_predictor_examples(examples_path)


def _score_examples(
    predictor: CombatOutcomePredictor,
    examples: Sequence[PredictorExample],
) -> list[_ScoredExample]:
    scored: list[_ScoredExample] = []
    for example in examples:
        scores = predictor.score_summary(example.start_summary)
        scored.append(
            _ScoredExample(
                example=example,
                slices=_slice_values(example),
                outcome_prediction=scores.outcome_win_probability,
                reward_prediction=scores.expected_reward,
                damage_prediction=scores.expected_damage_delta,
            )
        )
    return scored


def _slice_values(example: PredictorExample) -> dict[str, str]:
    run_summary = dict(example.start_summary.get("run", {}))
    combat_summary = dict(example.start_summary.get("combat", {}))
    strategic_context = dict(example.strategic_context or {})
    floor = example.floor
    if floor is None and run_summary.get("floor") is not None:
        floor = int(run_summary["floor"])
    enemy_ids = [str(enemy_id) for enemy_id in example.enemy_ids]
    if not enemy_ids:
        enemy_ids = [str(enemy_id) for enemy_id in combat_summary.get("enemy_ids", [])]
    encounter_family = "+".join(sorted(enemy_ids)) if enemy_ids else "unknown"
    character = str(run_summary.get("character_id") or run_summary.get("character_name") or "unknown")
    return {
        "character": character,
        "floor_band": _floor_band(floor),
        "act_id": str(run_summary.get("act_id") or "unknown"),
        "boss_id": str(run_summary.get("boss_encounter_id") or "unknown"),
        "encounter_family": encounter_family,
        "combat_type": _combat_type(enemy_ids),
        "route_profile": str(strategic_context.get("route_profile") or "unknown"),
    }


def _floor_band(floor: int | None) -> str:
    if floor is None:
        return "unknown"
    if floor <= 16:
        return "act1"
    if floor <= 33:
        return "act2"
    if floor <= 50:
        return "act3"
    return "act4_plus"


def _combat_type(enemy_ids: Sequence[str]) -> str:
    if not enemy_ids:
        return "unknown"
    return "multi_enemy" if len(enemy_ids) > 1 else "single_enemy"


def _binary_calibration_summary(
    items: Sequence[_ScoredExample],
    *,
    prediction_getter,
    label_getter,
    weight_getter,
    bin_count: int,
    include_bins: bool,
) -> dict[str, Any]:
    rows: list[tuple[float, float, float]] = []
    for item in items:
        prediction = prediction_getter(item)
        label = label_getter(item)
        weight = float(weight_getter(item))
        if prediction is None or label is None or weight <= 0:
            continue
        rows.append((float(prediction), float(label), weight))
    if not rows:
        return _empty_binary_summary()

    total_weight = sum(weight for _, _, weight in rows)
    loss_sum = 0.0
    brier_sum = 0.0
    correct_sum = 0.0
    prediction_sum = 0.0
    label_sum = 0.0
    overconfidence_sum = 0.0
    underconfidence_sum = 0.0
    prediction_values: list[float] = []
    bin_rows: list[list[tuple[float, float, float]]] = [[] for _ in range(bin_count)]

    for prediction, label, weight in rows:
        clipped = min(max(prediction, 1e-6), 1.0 - 1e-6)
        loss_sum += weight * (-(label * math.log(clipped) + ((1.0 - label) * math.log(1.0 - clipped))))
        brier_sum += weight * ((prediction - label) ** 2)
        correct_sum += weight * (1.0 if (prediction >= 0.5) == (label >= 0.5) else 0.0)
        prediction_sum += weight * prediction
        label_sum += weight * label
        overconfidence_sum += weight * max(prediction - label, 0.0)
        underconfidence_sum += weight * max(label - prediction, 0.0)
        prediction_values.append(prediction)
        bin_index = min(bin_count - 1, int(prediction * bin_count))
        bin_rows[bin_index].append((prediction, label, weight))

    bins = [_binary_reliability_bin(index, bin_count, rows) for index, rows in enumerate(bin_rows)]
    populated_bins = [payload for payload in bins if payload["count"] > 0]
    ece = sum((float(payload["total_weight"]) / total_weight) * abs(float(payload["gap"])) for payload in populated_bins)
    mce = max((abs(float(payload["gap"])) for payload in populated_bins), default=None)
    prediction_mean = prediction_sum / total_weight
    label_mean = label_sum / total_weight
    return {
        "count": len(rows),
        "total_weight": total_weight,
        "accuracy": correct_sum / total_weight,
        "label_mean": label_mean,
        "prediction_mean": prediction_mean,
        "prediction_std": _weighted_std(rows, center=prediction_mean),
        "brier_score": brier_sum / total_weight,
        "log_loss": loss_sum / total_weight,
        "ece": ece,
        "mce": mce,
        "overconfidence": overconfidence_sum / total_weight,
        "underconfidence": underconfidence_sum / total_weight,
        "prediction_spread": _value_distribution_summary(prediction_values),
        **({"reliability_bins": bins} if include_bins else {}),
    }


def _binary_reliability_bin(
    index: int,
    bin_count: int,
    rows: Sequence[tuple[float, float, float]],
) -> dict[str, Any]:
    lower = index / bin_count
    upper = (index + 1) / bin_count
    if not rows:
        return {
            "bin_index": index,
            "lower_bound": lower,
            "upper_bound": upper,
            "count": 0,
            "total_weight": 0.0,
            "prediction_mean": None,
            "label_mean": None,
            "gap": None,
        }
    total_weight = sum(weight for _, _, weight in rows)
    prediction_mean = sum(prediction * weight for prediction, _, weight in rows) / total_weight
    label_mean = sum(label * weight for _, label, weight in rows) / total_weight
    return {
        "bin_index": index,
        "lower_bound": lower,
        "upper_bound": upper,
        "count": len(rows),
        "total_weight": total_weight,
        "prediction_mean": prediction_mean,
        "label_mean": label_mean,
        "gap": prediction_mean - label_mean,
    }


def _regression_calibration_summary(
    items: Sequence[_ScoredExample],
    *,
    prediction_getter,
    label_getter,
    weight_getter,
    bin_count: int,
    include_bins: bool,
) -> dict[str, Any]:
    rows: list[tuple[float, float, float]] = []
    for item in items:
        prediction = prediction_getter(item)
        weight = float(weight_getter(item))
        if prediction is None or weight <= 0:
            continue
        rows.append((float(prediction), float(label_getter(item)), weight))
    if not rows:
        return _empty_regression_summary()

    total_weight = sum(weight for _, _, weight in rows)
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    signed_error_sum = 0.0
    prediction_sum = 0.0
    label_sum = 0.0
    over_prediction_sum = 0.0
    under_prediction_sum = 0.0
    prediction_values: list[float] = []
    residual_rows: list[tuple[float, float]] = []
    for prediction, label, weight in rows:
        error = prediction - label
        absolute_error_sum += weight * abs(error)
        squared_error_sum += weight * (error**2)
        signed_error_sum += weight * error
        prediction_sum += weight * prediction
        label_sum += weight * label
        over_prediction_sum += weight * (1.0 if error > 0 else 0.0)
        under_prediction_sum += weight * (1.0 if error < 0 else 0.0)
        prediction_values.append(prediction)
        residual_rows.append((error, weight))

    bins = _regression_reliability_bins(rows, bin_count=bin_count)
    populated_bins = [payload for payload in bins if payload["count"] > 0]
    calibration_error = sum(
        (float(payload["total_weight"]) / total_weight) * abs(float(payload["gap"])) for payload in populated_bins
    )
    max_bin_gap = max((abs(float(payload["gap"])) for payload in populated_bins), default=None)
    prediction_mean = prediction_sum / total_weight
    label_mean = label_sum / total_weight
    mse = squared_error_sum / total_weight
    return {
        "count": len(rows),
        "total_weight": total_weight,
        "rmse": math.sqrt(mse),
        "mae": absolute_error_sum / total_weight,
        "bias": signed_error_sum / total_weight,
        "prediction_mean": prediction_mean,
        "label_mean": label_mean,
        "prediction_std": _weighted_std(rows, center=prediction_mean),
        "residual_std": _weighted_residual_std(residual_rows, center=signed_error_sum / total_weight),
        "calibration_error": calibration_error,
        "max_bin_gap": max_bin_gap,
        "over_prediction_rate": over_prediction_sum / total_weight,
        "under_prediction_rate": under_prediction_sum / total_weight,
        "prediction_spread": _value_distribution_summary(prediction_values),
        **({"reliability_bins": bins} if include_bins else {}),
    }


def _regression_reliability_bins(rows: Sequence[tuple[float, float, float]], *, bin_count: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda row: row[0])
    chunk_size = max(1, math.ceil(len(ordered) / bin_count))
    bins: list[dict[str, Any]] = []
    for index in range(bin_count):
        start = index * chunk_size
        chunk = ordered[start : start + chunk_size]
        if not chunk:
            bins.append(
                {
                    "bin_index": index,
                    "lower_bound": None,
                    "upper_bound": None,
                    "count": 0,
                    "total_weight": 0.0,
                    "prediction_mean": None,
                    "label_mean": None,
                    "gap": None,
                }
            )
            continue
        total_weight = sum(weight for _, _, weight in chunk)
        prediction_mean = sum(prediction * weight for prediction, _, weight in chunk) / total_weight
        label_mean = sum(label * weight for _, label, weight in chunk) / total_weight
        bins.append(
            {
                "bin_index": index,
                "lower_bound": chunk[0][0],
                "upper_bound": chunk[-1][0],
                "count": len(chunk),
                "total_weight": total_weight,
                "prediction_mean": prediction_mean,
                "label_mean": label_mean,
                "gap": prediction_mean - label_mean,
            }
        )
    return bins


def _slice_calibration_summaries(
    items: Sequence[_ScoredExample],
    *,
    min_slice_examples: int,
    bin_count: int,
) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for dimension in ("character", "floor_band", "act_id", "boss_id", "encounter_family", "combat_type", "route_profile"):
        grouped: dict[str, list[_ScoredExample]] = {}
        for item in items:
            grouped.setdefault(item.slices.get(dimension, "unknown"), []).append(item)
        dimension_rows: list[dict[str, Any]] = []
        for slice_value, slice_items in grouped.items():
            if len(slice_items) < min_slice_examples:
                continue
            dimension_rows.append(
                {
                    "slice_value": slice_value,
                    "example_count": len(slice_items),
                    "outcome_win_probability": _binary_calibration_summary(
                        slice_items,
                        prediction_getter=lambda row: row.outcome_prediction,
                        label_getter=lambda row: row.example.outcome_win_label,
                        weight_getter=lambda row: row.example.outcome_weight,
                        bin_count=bin_count,
                        include_bins=False,
                    ),
                    "expected_reward": _regression_calibration_summary(
                        slice_items,
                        prediction_getter=lambda row: row.reward_prediction,
                        label_getter=lambda row: row.example.reward_label,
                        weight_getter=lambda row: row.example.reward_weight,
                        bin_count=bin_count,
                        include_bins=False,
                    ),
                    "expected_damage_delta": _regression_calibration_summary(
                        slice_items,
                        prediction_getter=lambda row: row.damage_prediction,
                        label_getter=lambda row: row.example.damage_delta_label,
                        weight_getter=lambda row: row.example.damage_weight,
                        bin_count=bin_count,
                        include_bins=False,
                    ),
                }
            )
        payload[dimension] = sorted(dimension_rows, key=lambda row: (-int(row["example_count"]), str(row["slice_value"])))
    return payload


def _build_ranking_groups(
    items: Sequence[_ScoredExample],
    *,
    group_by: Sequence[str],
    min_group_size: int,
) -> list[_RankingGroup]:
    grouped: dict[tuple[str, ...], list[_ScoredExample]] = {}
    for item in items:
        key = tuple(item.slices.get(dimension, "unknown") for dimension in group_by)
        grouped.setdefault(key, []).append(item)
    groups: list[_RankingGroup] = []
    for index, (key, group_items) in enumerate(sorted(grouped.items(), key=lambda entry: entry[0])):
        if len(group_items) < min_group_size:
            continue
        groups.append(
            _RankingGroup(
                group_id=f"group-{index + 1:05d}",
                key=key,
                slice_values=_group_slice_values(group_items),
                items=tuple(group_items),
            )
        )
    return groups


def _group_slice_values(items: Sequence[_ScoredExample]) -> dict[str, str]:
    values: dict[str, str] = {}
    for dimension in ("character", "floor_band", "act_id", "boss_id", "encounter_family", "combat_type", "route_profile"):
        distinct = {item.slices.get(dimension, "unknown") for item in items}
        values[dimension] = next(iter(distinct)) if len(distinct) == 1 else "mixed"
    return values


def _ranking_summary(
    groups: Sequence[_RankingGroup],
    *,
    score_getter,
    label_getter,
    weight_getter,
    top_k: int,
) -> dict[str, Any]:
    pairwise_weight_sum = 0.0
    pairwise_correct_sum = 0.0
    ndcg_values: list[float] = []
    top1_hits = 0
    top1_valid_groups = 0
    top1_lifts: list[float] = []
    candidate_counts: list[float] = []
    eligible_group_count = 0

    for group in groups:
        candidates = [
            (
                item,
                score_getter(item),
                label_getter(item),
                float(weight_getter(item)),
            )
            for item in group.items
        ]
        candidates = [candidate for candidate in candidates if candidate[1] is not None and candidate[3] > 0]
        if len(candidates) < 2:
            continue
        eligible_group_count += 1
        candidate_counts.append(float(len(candidates)))

        group_pair_weight, group_pair_correct = _pairwise_accuracy(candidates)
        pairwise_weight_sum += group_pair_weight
        pairwise_correct_sum += group_pair_correct

        ndcg_value = _ndcg_at_k(candidates, top_k=top_k)
        if ndcg_value is not None:
            ndcg_values.append(ndcg_value)

        top1_payload = _top1_metrics(candidates)
        if top1_payload is not None:
            top1_valid_groups += 1
            top1_hits += int(top1_payload["hit"])
            top1_lifts.append(float(top1_payload["lift"]))

    return {
        "group_count": len(groups),
        "eligible_group_count": eligible_group_count,
        "pair_count": pairwise_weight_sum,
        "pairwise_accuracy": None if pairwise_weight_sum <= 0 else (pairwise_correct_sum / pairwise_weight_sum),
        "ndcg_at_k": None if not ndcg_values else sum(ndcg_values) / len(ndcg_values),
        "ndcg_group_count": len(ndcg_values),
        "top1_hit_rate": None if top1_valid_groups == 0 else (top1_hits / top1_valid_groups),
        "top1_lift_mean": None if not top1_lifts else (sum(top1_lifts) / len(top1_lifts)),
        "candidate_count_stats": _value_distribution_summary(candidate_counts),
    }


def _pairwise_accuracy(candidates: Sequence[tuple[Any, float | None, float | None, float]]) -> tuple[float, float]:
    total_weight = 0.0
    correct_weight = 0.0
    for left_index in range(len(candidates)):
        _, left_score, left_label, left_weight = candidates[left_index]
        if left_score is None or left_label is None or left_weight <= 0:
            continue
        for right_index in range(left_index + 1, len(candidates)):
            _, right_score, right_label, right_weight = candidates[right_index]
            if right_score is None or right_label is None or right_weight <= 0:
                continue
            actual_delta = float(left_label) - float(right_label)
            if abs(actual_delta) <= 1e-9:
                continue
            pair_weight = left_weight * right_weight
            predicted_delta = float(left_score) - float(right_score)
            total_weight += pair_weight
            if predicted_delta == 0:
                correct_weight += pair_weight * 0.5
            elif (predicted_delta > 0 and actual_delta > 0) or (predicted_delta < 0 and actual_delta < 0):
                correct_weight += pair_weight
    return total_weight, correct_weight


def _ndcg_at_k(candidates: Sequence[tuple[Any, float | None, float | None, float]], *, top_k: int) -> float | None:
    ranked = [(float(score), float(label)) for _, score, label, _ in candidates if score is not None and label is not None]
    if len(ranked) < 2:
        return None
    labels = [label for _, label in ranked]
    min_label = min(labels)
    max_label = max(labels)
    if math.isclose(min_label, max_label):
        return None
    normalized = [(score, (label - min_label) / (max_label - min_label)) for score, label in ranked]
    predicted_order = [relevance for _, relevance in sorted(normalized, key=lambda row: row[0], reverse=True)]
    ideal_order = sorted((relevance for _, relevance in normalized), reverse=True)
    dcg = _dcg(predicted_order[:top_k])
    ideal_dcg = _dcg(ideal_order[:top_k])
    if ideal_dcg <= 0:
        return None
    return dcg / ideal_dcg


def _dcg(values: Sequence[float]) -> float:
    return sum(value / math.log2(index + 2) for index, value in enumerate(values))


def _top1_metrics(candidates: Sequence[tuple[Any, float | None, float | None, float]]) -> dict[str, float] | None:
    ranked = [(float(score), float(label)) for _, score, label, _ in candidates if score is not None and label is not None]
    if len(ranked) < 2:
        return None
    labels = [label for _, label in ranked]
    if math.isclose(min(labels), max(labels)):
        return None
    _, predicted_top_label = max(ranked, key=lambda row: row[0])
    group_mean = sum(labels) / len(labels)
    group_best = max(labels)
    return {
        "hit": 1.0 if math.isclose(predicted_top_label, group_best) else 0.0,
        "lift": predicted_top_label - group_mean,
    }


def _slice_ranking_summaries(groups: Sequence[_RankingGroup], *, top_k: int) -> dict[str, list[dict[str, Any]]]:
    payload: dict[str, list[dict[str, Any]]] = {}
    for dimension in ("character", "floor_band", "act_id", "boss_id", "encounter_family", "combat_type", "route_profile"):
        grouped: dict[str, list[_RankingGroup]] = {}
        for group in groups:
            grouped.setdefault(group.slice_values.get(dimension, "unknown"), []).append(group)
        rows: list[dict[str, Any]] = []
        for slice_value, slice_groups in grouped.items():
            rows.append(
                {
                    "slice_value": slice_value,
                    "group_count": len(slice_groups),
                    "example_count": sum(len(group.items) for group in slice_groups),
                    "outcome_win_probability": _ranking_summary(
                        slice_groups,
                        score_getter=lambda item: item.outcome_prediction,
                        label_getter=lambda item: item.example.outcome_win_label,
                        weight_getter=lambda item: item.example.outcome_weight,
                        top_k=top_k,
                    ),
                    "expected_reward": _ranking_summary(
                        slice_groups,
                        score_getter=lambda item: item.reward_prediction,
                        label_getter=lambda item: item.example.reward_label,
                        weight_getter=lambda item: item.example.reward_weight,
                        top_k=top_k,
                    ),
                    "expected_damage_delta": _ranking_summary(
                        slice_groups,
                        score_getter=lambda item: item.damage_prediction,
                        label_getter=lambda item: item.example.damage_delta_label,
                        weight_getter=lambda item: item.example.damage_weight,
                        top_k=top_k,
                    ),
                }
            )
        payload[dimension] = sorted(rows, key=lambda row: (-int(row["group_count"]), str(row["slice_value"])))
    return payload


def _mode_eval_summaries(arms: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        grouped.setdefault(str(arm["mode"]), []).append(arm)
    rows: list[dict[str, Any]] = []
    for mode, mode_arms in grouped.items():
        rows.append(
            {
                "mode": mode,
                "case_count": len(mode_arms),
                "scenario_floor_band_histogram": _histogram(str(arm["scenario_floor_band"]) for arm in mode_arms),
                "scenario_boss_histogram": _histogram(str(arm["scenario_boss"]) for arm in mode_arms),
                "scenario_route_profile_histogram": _histogram(
                    str(arm["scenario_route_profile"]) for arm in mode_arms
                ),
                "total_reward_estimate_stats": _value_distribution_summary(
                    [arm["metrics"]["total_reward"] for arm in mode_arms if arm["metrics"]["total_reward"] is not None]
                ),
                "combat_win_rate_estimate_stats": _value_distribution_summary(
                    [arm["metrics"]["combat_win_rate"] for arm in mode_arms if arm["metrics"]["combat_win_rate"] is not None]
                ),
                "reward_per_combat_estimate_stats": _value_distribution_summary(
                    [
                        arm["metrics"]["reward_per_combat"]
                        for arm in mode_arms
                        if arm["metrics"]["reward_per_combat"] is not None
                    ]
                ),
            }
        )
    return sorted(rows, key=lambda row: str(row["mode"]))


def _mode_pair_comparison_summaries(comparisons: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for comparison in comparisons:
        key = (str(comparison["baseline_mode"]), str(comparison["candidate_mode"]))
        grouped.setdefault(key, []).append(comparison)
    rows: list[dict[str, Any]] = []
    for (baseline_mode, candidate_mode), pair_rows in grouped.items():
        rows.append(
            {
                "baseline_mode": baseline_mode,
                "candidate_mode": candidate_mode,
                "comparison_count": len(pair_rows),
                "scenario_floor_band_histogram": _histogram(
                    str(comparison["scenario_floor_band"]) for comparison in pair_rows
                ),
                "scenario_boss_histogram": _histogram(str(comparison["scenario_boss"]) for comparison in pair_rows),
                "scenario_route_profile_histogram": _histogram(
                    str(comparison["scenario_route_profile"]) for comparison in pair_rows
                ),
                "passing_case_count": sum(1 for comparison in pair_rows if comparison["promotion"]["passed"]),
                "delta_total_reward_stats": _value_distribution_summary(
                    [
                        comparison["delta_total_reward"]
                        for comparison in pair_rows
                        if comparison["delta_total_reward"] is not None
                    ]
                ),
                "delta_combat_win_rate_stats": _value_distribution_summary(
                    [
                        comparison["delta_combat_win_rate"]
                        for comparison in pair_rows
                        if comparison["delta_combat_win_rate"] is not None
                    ]
                ),
                "delta_reward_per_combat_stats": _value_distribution_summary(
                    [
                        comparison["delta_reward_per_combat"]
                        for comparison in pair_rows
                        if comparison["delta_reward_per_combat"] is not None
                    ]
                ),
            }
        )
    return sorted(rows, key=lambda row: (str(row["baseline_mode"]), str(row["candidate_mode"])))


def _scenario_floor_band_comparison_summaries(comparisons: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for comparison in comparisons:
        grouped.setdefault(str(comparison["scenario_floor_band"]), []).append(comparison)
    rows: list[dict[str, Any]] = []
    for floor_band, floor_rows in grouped.items():
        rows.append(
            {
                "scenario_floor_band": floor_band,
                "comparison_count": len(floor_rows),
                "mode_pair_histogram": _histogram(
                    f"{comparison['baseline_mode']}->{comparison['candidate_mode']}" for comparison in floor_rows
                ),
                "scenario_boss_histogram": _histogram(str(comparison["scenario_boss"]) for comparison in floor_rows),
                "scenario_route_profile_histogram": _histogram(
                    str(comparison["scenario_route_profile"]) for comparison in floor_rows
                ),
                "passing_case_count": sum(1 for comparison in floor_rows if comparison["promotion"]["passed"]),
                "delta_total_reward_stats": _value_distribution_summary(
                    [
                        comparison["delta_total_reward"]
                        for comparison in floor_rows
                        if comparison["delta_total_reward"] is not None
                    ]
                ),
                "delta_combat_win_rate_stats": _value_distribution_summary(
                    [
                        comparison["delta_combat_win_rate"]
                        for comparison in floor_rows
                        if comparison["delta_combat_win_rate"] is not None
                    ]
                ),
            }
        )
    return sorted(rows, key=lambda row: str(row["scenario_floor_band"]))


def _load_benchmark_case_payloads(source: str | Path) -> list[dict[str, Any]]:
    source_path = Path(source).expanduser().resolve()
    if source_path.is_dir():
        suite_summary = source_path / BENCHMARK_SUITE_SUMMARY_FILENAME
        case_summary = source_path / BENCHMARK_CASE_SUMMARY_FILENAME
        if suite_summary.exists():
            payload = json.loads(suite_summary.read_text(encoding="utf-8"))
            return [dict(item) for item in payload.get("cases", [])]
        if case_summary.exists():
            return [json.loads(case_summary.read_text(encoding="utf-8"))]
        raise FileNotFoundError(f"Benchmark source did not contain a suite or case summary: {source_path}")

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload.get("cases"), list):
        return [dict(item) for item in payload.get("cases", [])]
    if payload.get("case_id") is not None:
        return [payload]
    raise ValueError(f"Benchmark source is not a suite summary or case summary: {source_path}")


def _scenario_floor_band(scenario: dict[str, Any]) -> str:
    floor_min = scenario.get("floor_min")
    floor_max = scenario.get("floor_max")
    if floor_min is None and floor_max is None:
        return "all"
    return f"{floor_min if floor_min is not None else '*'}-{floor_max if floor_max is not None else '*'}"


def _scenario_slice_fields(scenario: dict[str, Any]) -> dict[str, str]:
    return {
        "scenario_boss": _scenario_multi_value_label(scenario.get("boss_ids")),
        "scenario_act": _scenario_multi_value_label(scenario.get("act_ids")),
        "scenario_planner_strategy": _scenario_multi_value_label(scenario.get("planner_strategies")),
        "scenario_route_reason_tag": _scenario_multi_value_label(scenario.get("route_reason_tags")),
        "scenario_route_profile": _scenario_multi_value_label(scenario.get("route_profiles")),
    }


def _scenario_multi_value_label(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "all"
    normalized = sorted(str(value) for value in values if str(value).strip())
    return "|".join(normalized) if normalized else "all"


def _artifact_family_list(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    community_prior = payload.get("community_prior")
    if not isinstance(community_prior, dict):
        return []
    diagnostics = community_prior.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    families = {
        str(item.get("artifact_family"))
        for item in diagnostics.values()
        if isinstance(item, dict) and item.get("artifact_family") is not None
    }
    return sorted(families)


def _metric_summary_mean(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    return _float_or_none(payload.get("mean"))


def _histogram(values: Any) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for value in values:
        histogram[str(value)] = histogram.get(str(value), 0) + 1
    return histogram


def _empty_binary_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "total_weight": 0.0,
        "accuracy": None,
        "label_mean": None,
        "prediction_mean": None,
        "prediction_std": None,
        "brier_score": None,
        "log_loss": None,
        "ece": None,
        "mce": None,
        "overconfidence": None,
        "underconfidence": None,
        "prediction_spread": _value_distribution_summary([]),
    }


def _empty_regression_summary() -> dict[str, Any]:
    return {
        "count": 0,
        "total_weight": 0.0,
        "rmse": None,
        "mae": None,
        "bias": None,
        "prediction_mean": None,
        "label_mean": None,
        "prediction_std": None,
        "residual_std": None,
        "calibration_error": None,
        "max_bin_gap": None,
        "over_prediction_rate": None,
        "under_prediction_rate": None,
        "prediction_spread": _value_distribution_summary([]),
    }


def _weighted_std(rows: Sequence[tuple[float, float, float]], *, center: float) -> float | None:
    total_weight = sum(weight for _, _, weight in rows)
    if total_weight <= 0:
        return None
    variance = sum(weight * ((value - center) ** 2) for value, _, weight in rows) / total_weight
    return math.sqrt(variance)


def _weighted_residual_std(rows: Sequence[tuple[float, float]], *, center: float) -> float | None:
    total_weight = sum(weight for _, weight in rows)
    if total_weight <= 0:
        return None
    variance = sum(weight * ((value - center) ** 2) for value, weight in rows) / total_weight
    return math.sqrt(variance)


def _value_distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    filtered = [float(value) for value in values]
    if not filtered:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(filtered),
        "min": min(filtered),
        "mean": sum(filtered) / len(filtered),
        "max": max(filtered),
    }


def _max_threshold_check(name: str, actual: Any, target: float) -> dict[str, Any]:
    actual_value = _float_or_none(actual)
    passed = actual_value is not None and actual_value <= target
    return {
        "name": name,
        "direction": "max",
        "actual": actual_value,
        "target": float(target),
        "passed": passed,
    }


def _min_threshold_check(name: str, actual: Any, target: float) -> dict[str, Any]:
    actual_value = _float_or_none(actual)
    passed = actual_value is not None and actual_value >= target
    return {
        "name": name,
        "direction": "min",
        "actual": actual_value,
        "target": float(target),
        "passed": passed,
    }


def _evaluate_thresholds(checks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failed_checks = [dict(check) for check in checks if not check["passed"]]
    return {
        "passed": bool(checks) and not failed_checks,
        "check_count": len(checks),
        "checks": [dict(check) for check in checks],
        "failed_checks": failed_checks,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _age_days(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        if "T" in normalized:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00")).date()
        else:
            parsed = datetime.fromisoformat(normalized).date()
    except ValueError:
        return None
    return (datetime.now(UTC).date() - parsed).days


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _render_calibration_markdown(payload: dict[str, Any]) -> str:
    overall = payload["overall"]
    promotion = payload["promotion"]
    return "\n".join(
        [
            "# Predictor Calibration Report",
            "",
            f"- Model: `{payload['model_label']}`",
            f"- Dataset split: `{payload['split']}`",
            f"- Examples: `{payload['example_count']}`",
            f"- Public Sources: `{len(payload.get('public_sources', {}))}`",
            f"- Promotion passed: `{promotion['passed']}`",
            f"- Outcome ECE: `{_fmt(overall['outcome_win_probability'].get('ece'))}`",
            f"- Outcome Brier: `{_fmt(overall['outcome_win_probability'].get('brier_score'))}`",
            f"- Reward RMSE: `{_fmt(overall['expected_reward'].get('rmse'))}`",
            f"- Damage RMSE: `{_fmt(overall['expected_damage_delta'].get('rmse'))}`",
        ]
    )


def _render_ranking_markdown(payload: dict[str, Any]) -> str:
    overall = payload["overall"]
    promotion = payload["promotion"]
    return "\n".join(
        [
            "# Predictor Ranking Report",
            "",
            f"- Model: `{payload['model_label']}`",
            f"- Dataset split: `{payload['split']}`",
            f"- Examples: `{payload['example_count']}`",
            f"- Groups: `{payload['group_count']}`",
            f"- Public Sources: `{len(payload.get('public_sources', {}))}`",
            f"- Promotion passed: `{promotion['passed']}`",
            f"- Outcome pairwise accuracy: `{_fmt(overall['outcome_win_probability'].get('pairwise_accuracy'))}`",
            f"- Reward pairwise accuracy: `{_fmt(overall['expected_reward'].get('pairwise_accuracy'))}`",
            f"- Reward NDCG@{payload['top_k']}: `{_fmt(overall['expected_reward'].get('ndcg_at_k'))}`",
            f"- Damage pairwise accuracy: `{_fmt(overall['expected_damage_delta'].get('pairwise_accuracy'))}`",
            f"- Damage NDCG@{payload['top_k']}: `{_fmt(overall['expected_damage_delta'].get('ndcg_at_k'))}`",
        ]
    )


def _render_benchmark_compare_markdown(payload: dict[str, Any]) -> str:
    promotion = payload["promotion"]
    return "\n".join(
        [
            "# Predictor Benchmark Comparison Report",
            "",
            f"- Sources: `{len(payload['source_paths'])}`",
            f"- Cases: `{payload['case_count']}`",
            f"- Compare cases: `{payload['compare_case_count']}`",
            f"- Benchmark Public Sources: `{payload.get('benchmark_public_sources', {}).get('configured_source_count', 0)}`",
            f"- Promotion candidates: `{promotion['promotion_candidate_count']}`",
            f"- Rollback signals: `{promotion['rollback_signal_count']}`",
            f"- Promotion passed: `{promotion['passed']}`",
        ]
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"
