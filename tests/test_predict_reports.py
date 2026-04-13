import json
from pathlib import Path

from sts2_rl.predict import (
    PredictorBenchmarkComparisonThresholds,
    PredictorCalibrationThresholds,
    PredictorRankingThresholds,
    build_predictor_benchmark_comparison_report,
    build_predictor_calibration_report,
    build_predictor_ranking_report,
)
from tests.predict_report_fixtures import build_synthetic_predictor_fixture


def test_build_predictor_calibration_report_writes_slices_and_thresholds(tmp_path: Path) -> None:
    dataset_dir, model_path = build_synthetic_predictor_fixture(tmp_path)
    aggregate_source, run_source = _write_public_source_fixtures(tmp_path)

    report = build_predictor_calibration_report(
        model_path=model_path,
        dataset_source=dataset_dir,
        output_root=tmp_path / "artifacts" / "predict-reports",
        session_name="calibration-report",
        split="validation",
        bin_count=4,
        min_slice_examples=1,
        thresholds=PredictorCalibrationThresholds(
            outcome_ece_max=1.0,
            outcome_brier_max=1.0,
            reward_rmse_max=100.0,
            damage_rmse_max=100.0,
        ),
        public_aggregate_source=aggregate_source,
        public_run_source=run_source,
    )

    summary_payload = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["report_kind"] == "predictor_calibration"
    assert summary_payload["split"] == "validation"
    assert summary_payload["promotion"]["passed"] is True
    assert summary_payload["public_sources"]["aggregate_card"]["artifact_family"] == "community_card_stats"
    assert summary_payload["public_sources"]["run_strategic"]["artifact_family"] == "public_run_strategic_stats"
    assert len(summary_payload["overall"]["outcome_win_probability"]["reliability_bins"]) == 4
    assert summary_payload["slices"]["character"]
    assert summary_payload["slices"]["floor_band"]
    assert "boss_id" in summary_payload["slices"]
    assert "route_profile" in summary_payload["slices"]
    assert report.markdown_path.exists()


def test_build_predictor_ranking_report_writes_group_metrics(tmp_path: Path) -> None:
    dataset_dir, model_path = build_synthetic_predictor_fixture(tmp_path)
    aggregate_source, run_source = _write_public_source_fixtures(tmp_path)

    report = build_predictor_ranking_report(
        model_path=model_path,
        dataset_source=dataset_dir,
        output_root=tmp_path / "artifacts" / "predict-reports",
        session_name="ranking-report",
        split="all",
        group_by=("character", "floor_band", "encounter_family"),
        top_k=2,
        min_group_size=2,
        thresholds=PredictorRankingThresholds(
            outcome_pairwise_accuracy_min=0.0,
            reward_pairwise_accuracy_min=0.0,
            damage_pairwise_accuracy_min=0.0,
            reward_ndcg_at_3_min=0.0,
            damage_ndcg_at_3_min=0.0,
        ),
        public_aggregate_source=aggregate_source,
        public_run_source=run_source,
    )

    summary_payload = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["report_kind"] == "predictor_ranking"
    assert summary_payload["group_count"] >= 4
    assert summary_payload["promotion"]["passed"] is True
    assert summary_payload["public_sources"]["aggregate_card"]["record_count"] == 1
    assert summary_payload["public_sources"]["run_strategic"]["card_record_count"] == 1
    assert summary_payload["overall"]["expected_reward"]["pairwise_accuracy"] is not None
    assert summary_payload["overall"]["expected_damage_delta"]["ndcg_at_k"] is not None
    assert summary_payload["slices"]["encounter_family"]
    assert summary_payload["slices"]["boss_id"]
    assert summary_payload["slices"]["route_profile"]
    assert report.markdown_path.exists()


def test_build_predictor_benchmark_comparison_report_summarizes_mode_pairs(tmp_path: Path) -> None:
    suite_summary_path = tmp_path / "benchmarks" / "predictor-suite" / "benchmark-suite-summary.json"
    aggregate_source, run_source = _write_public_source_fixtures(tmp_path)
    suite_summary_path.parent.mkdir(parents=True)
    suite_summary_path.write_text(
        json.dumps(
            {
                "suite_name": "predictor-suite",
                "cases": [
                    {
                        "case_id": "eval-heuristic",
                        "mode": "eval",
                        "predictor": {"mode": "heuristic_only", "hooks": []},
                        "scenario": {
                            "floor_min": 1,
                            "floor_max": 16,
                            "boss_ids": ["THE_GUARDIAN"],
                            "route_profiles": ["search_aoe_tools"],
                        },
                        "metrics": {
                            "total_reward": {"mean": 1.0},
                            "combat_win_rate": {"mean": 0.4},
                            "reward_per_combat": {"mean": 0.2},
                        },
                    },
                    {
                        "case_id": "eval-assist",
                        "mode": "eval",
                        "predictor": {"mode": "assist", "hooks": ["combat"]},
                        "scenario": {
                            "floor_min": 17,
                            "floor_max": 33,
                            "boss_ids": ["THE_CHAMP"],
                            "route_profiles": ["prepare_scaling_boss"],
                        },
                        "metrics": {
                            "total_reward": {"mean": 2.0},
                            "combat_win_rate": {"mean": 0.7},
                            "reward_per_combat": {"mean": 0.5},
                        },
                    },
                    {
                        "case_id": "compare-dominant",
                        "mode": "compare",
                        "predictor": {
                            "baseline": {"mode": "heuristic_only", "hooks": []},
                            "candidate": {"mode": "dominant", "hooks": ["combat", "reward"]},
                        },
                        "scenario": {
                            "floor_min": 1,
                            "floor_max": 16,
                            "boss_ids": ["THE_GUARDIAN"],
                            "route_profiles": ["search_aoe_tools"],
                        },
                        "metrics": {
                            "delta_total_reward": {"mean": 1.4},
                            "delta_combat_win_rate": {"mean": 0.2},
                            "delta_reward_per_combat": {"mean": 0.1},
                            "delta_community_top_choice_match_rate": {"mean": 0.15},
                        },
                        "baseline": {"mean_total_reward": 1.0, "combat_win_rate": 0.4, "reward_per_combat": 0.2},
                        "candidate": {"mean_total_reward": 2.4, "combat_win_rate": 0.6, "reward_per_combat": 0.3},
                        "public_sources": {
                            "baseline": {
                                "community_prior": {
                                    "diagnostics": {
                                        "card": {"artifact_family": "community_card_stats", "stat_family": "card", "source_name": "spiremeta", "age_days": 1}
                                    }
                                }
                            },
                            "candidate": {
                                "community_prior": {
                                    "diagnostics": {
                                        "card": {"artifact_family": "public_run_strategic_stats", "stat_family": "card", "source_name": "sts2runs", "age_days": 0},
                                        "route": {"artifact_family": "public_run_strategic_stats", "stat_family": "route", "source_name": "sts2runs", "age_days": 0}
                                    }
                                }
                            },
                        },
                    },
                    {
                        "case_id": "compare-assist",
                        "mode": "compare",
                        "predictor": {
                            "baseline": {"mode": "heuristic_only", "hooks": []},
                            "candidate": {"mode": "assist", "hooks": ["combat"]},
                        },
                        "scenario": {
                            "floor_min": 17,
                            "floor_max": 33,
                            "boss_ids": ["THE_CHAMP"],
                            "route_profiles": ["prepare_scaling_boss"],
                        },
                        "metrics": {
                            "delta_total_reward": {"mean": -0.2},
                            "delta_combat_win_rate": {"mean": -0.05},
                            "delta_reward_per_combat": {"mean": -0.02},
                        },
                        "baseline": {"mean_total_reward": 1.2, "combat_win_rate": 0.5, "reward_per_combat": 0.25},
                        "candidate": {"mean_total_reward": 1.0, "combat_win_rate": 0.45, "reward_per_combat": 0.23},
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = build_predictor_benchmark_comparison_report(
        sources=[suite_summary_path],
        output_root=tmp_path / "artifacts" / "predict-reports",
        session_name="compare-report",
        thresholds=PredictorBenchmarkComparisonThresholds(
            delta_total_reward_min=0.0,
            delta_combat_win_rate_min=0.0,
        ),
        public_aggregate_source=aggregate_source,
        public_run_source=run_source,
    )

    summary_payload = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["report_kind"] == "predictor_benchmark_compare"
    assert summary_payload["promotion"]["passed"] is True
    assert summary_payload["promotion"]["promotion_candidate_count"] == 1
    assert summary_payload["promotion"]["rollback_signal_count"] == 1
    assert summary_payload["public_sources"]["aggregate_card"]["record_count"] == 1
    assert summary_payload["benchmark_public_sources"]["configured_source_count"] == 3
    assert any(row["artifact_family"] == "public_run_strategic_stats" for row in summary_payload["public_source_comparisons"])
    assert any(row["candidate_mode"] == "dominant" for row in summary_payload["mode_pair_summaries"])
    assert any(row["scenario_floor_band"] == "1-16" for row in summary_payload["scenario_floor_band_summaries"])
    assert any(row["scenario_boss_histogram"].get("THE_GUARDIAN") == 1 for row in summary_payload["mode_pair_summaries"])


def _write_public_source_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    aggregate_path = tmp_path / "public" / "community-card-stats.jsonl"
    run_dir = tmp_path / "public-run"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(
        json.dumps(
            {
                "record_type": "community_card_stat",
                "schema_version": 1,
                "source_name": "spiremeta",
                "snapshot_date": "2026-04-13",
                "character_id": "IRONCLAD",
                "source_type": "reward",
                "card_id": "CARD_A",
                "offer_count": 100,
                "pick_count": 50,
                "pick_rate": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "strategic-card-stats.jsonl").write_text(
        json.dumps(
            {
                "record_type": "public_run_strategic_stat",
                "schema_version": 1,
                "stat_family": "card",
                "source_name": "sts2runs",
                "snapshot_date": "2026-04-14",
                "subject_id": "CARD_B",
                "character_id": "IRONCLAD",
                "act_id": "ACT_1",
                "source_type": "reward",
                "run_count": 20,
                "offer_count": 20,
                "pick_count": 10,
                "pick_rate": 0.5,
                "win_count": 10,
                "win_rate": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "strategic-route-stats.jsonl").write_text(
        json.dumps(
            {
                "record_type": "public_run_strategic_stat",
                "schema_version": 1,
                "stat_family": "route",
                "source_name": "sts2runs",
                "snapshot_date": "2026-04-14",
                "subject_id": "shop",
                "character_id": "IRONCLAD",
                "act_id": "ACT_1",
                "room_type": "shop",
                "run_count": 20,
                "seen_count": 20,
                "win_count": 12,
                "win_rate": 0.6,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps({"generated_at_utc": "2026-04-14T00:00:00+00:00"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return aggregate_path, run_dir
