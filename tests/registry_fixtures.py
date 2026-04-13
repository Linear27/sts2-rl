import json
from pathlib import Path


def build_registry_fixture(tmp_path: Path) -> dict[str, Path]:
    dataset_dir = tmp_path / "data" / "trajectory" / "bc-synthetic"
    dataset_dir.mkdir(parents=True)
    dataset_summary_path = dataset_dir / "dataset-summary.json"
    dataset_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "bc-synthetic",
                "dataset_kind": "trajectory_steps",
                "record_count": 8,
                "feature_count": 42,
                "source_file_count": 1,
                "filtered_out_count": 0,
                "lineage": {"source_paths": ["synthetic"], "resolved_source_files": ["synthetic.steps.jsonl"]},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    benchmark_dir = tmp_path / "artifacts" / "benchmarks" / "policy-suite"
    benchmark_dir.mkdir(parents=True)
    benchmark_summary_path = benchmark_dir / "benchmark-suite-summary.json"
    benchmark_summary_path.write_text(
        json.dumps(
            {
                "suite_name": "policy-suite",
                "case_count": 2,
                "promotion": {
                    "configured_case_count": 1,
                    "passed_case_count": 1,
                    "failed_case_count": 0,
                    "promotion_candidate_count": 1,
                    "rollback_signal_count": 0,
                },
                "cases": [
                    {
                        "case_id": "bc-case",
                        "mode": "eval",
                        "primary_metric": {"name": "combat_win_rate", "estimate": 0.78},
                    },
                    {
                        "case_id": "offline-case",
                        "mode": "eval",
                        "primary_metric": {"name": "combat_win_rate", "estimate": 0.71},
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    benchmark_case_dir = benchmark_dir / "strategic-compare"
    benchmark_case_dir.mkdir(parents=True)
    benchmark_case_summary_path = benchmark_case_dir / "case-summary.json"
    benchmark_case_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "case_id": "strategic-compare",
                "mode": "compare",
                "primary_metric": {"name": "delta_combat_win_rate", "estimate": 0.12},
                "paired_iteration_count": 4,
                "metrics": {
                    "route_decision_overlap_rate": {"mean": 0.75},
                    "first_node_agreement_rate": {"mean": 0.88},
                    "delta_route_quality_score": {"mean": 0.42},
                    "delta_pre_boss_readiness": {"mean": 0.31},
                    "delta_route_risk_score": {"mean": -0.2},
                },
                "shadow": {
                    "enabled": True,
                    "comparable_encounter_count": 12,
                    "agreement_rate": 0.66,
                    "candidate_advantage_rate": 0.25,
                    "delta_metrics": {
                        "delta_first_action_match_rate": 0.1,
                        "delta_trace_hit_rate": 0.2,
                    },
                },
                "route_diagnostics": {"comparison": {"route_decision_pair_count": 8}},
                "promotion": {
                    "enabled": True,
                    "passed": True,
                    "check_count": 8,
                    "promotion_candidate_count": 1,
                    "rollback_signal_count": 0,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    bc_dir = tmp_path / "artifacts" / "behavior-cloning" / "bc-live"
    bc_dir.mkdir(parents=True)
    (bc_dir / "behavior-cloning-checkpoint.json").write_text("{}", encoding="utf-8")
    (bc_dir / "behavior-cloning-best.json").write_text("{}", encoding="utf-8")
    (bc_dir / "training-metrics.jsonl").write_text("", encoding="utf-8")
    bc_summary_path = bc_dir / "summary.json"
    bc_summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_path": str(dataset_dir.resolve()),
                "output_dir": str(bc_dir.resolve()),
                "checkpoint_path": str((bc_dir / "behavior-cloning-checkpoint.json").resolve()),
                "best_checkpoint_path": str((bc_dir / "behavior-cloning-best.json").resolve()),
                "metrics_path": str((bc_dir / "training-metrics.jsonl").resolve()),
                "example_count": 8,
                "best_epoch": 12,
                "feature_count": 42,
                "stage_count": 3,
                "config": {"epochs": 20, "learning_rate": 0.05},
                "validation": {
                    "loss": 0.32,
                    "top_k_accuracy": {"1": 0.74, "3": 0.91},
                },
                "test": {
                    "loss": 0.35,
                    "top_k_accuracy": {"1": 0.72, "3": 0.9},
                },
                "dataset_lineage": {"source_paths": ["synthetic"], "resolved_source_files": ["synthetic.steps.jsonl"]},
                "benchmark_summary_path": str(benchmark_summary_path.resolve()),
                "live_eval_summary_path": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    offline_dir = tmp_path / "artifacts" / "offline-cql" / "offline-live"
    offline_dir.mkdir(parents=True)
    (offline_dir / "offline-cql-checkpoint.json").write_text("{}", encoding="utf-8")
    (offline_dir / "offline-cql-best.json").write_text("{}", encoding="utf-8")
    (offline_dir / "offline-cql-dqn-seed.json").write_text("{}", encoding="utf-8")
    (offline_dir / "training-metrics.jsonl").write_text("", encoding="utf-8")
    offline_summary_path = offline_dir / "summary.json"
    offline_summary_path.write_text(
        json.dumps(
            {
                "algorithm": "offline_cql",
                "dataset_path": str(dataset_dir.resolve()),
                "output_dir": str(offline_dir.resolve()),
                "checkpoint_path": str((offline_dir / "offline-cql-checkpoint.json").resolve()),
                "best_checkpoint_path": str((offline_dir / "offline-cql-best.json").resolve()),
                "warmstart_checkpoint_path": str((offline_dir / "offline-cql-dqn-seed.json").resolve()),
                "metrics_path": str((offline_dir / "training-metrics.jsonl").resolve()),
                "best_epoch": 9,
                "config": {"epochs": 15, "learning_rate": 0.001},
                "validation_metrics": {"mean_loss": 0.48, "support_coverage": 1.0},
                "test_metrics": {"mean_loss": 0.51},
                "dataset_summary": {"lineage": {"source_paths": ["synthetic"], "resolved_source_files": ["synthetic.steps.jsonl"]}},
                "benchmark_summary_path": str(benchmark_summary_path.resolve()),
                "live_eval_summary_path": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    predictor_report_dir = tmp_path / "artifacts" / "predict-reports" / "predict-ranking"
    predictor_report_dir.mkdir(parents=True)
    (predictor_report_dir / "combat-outcome-predictor.json").write_text("{}", encoding="utf-8")
    predictor_report_summary_path = predictor_report_dir / "summary.json"
    predictor_report_summary_path.write_text(
        json.dumps(
            {
                "report_kind": "predictor_ranking",
                "model_path": str((predictor_report_dir / "combat-outcome-predictor.json").resolve()),
                "dataset_path": str(dataset_dir.resolve()),
                "split": "validation",
                "group_by": ["character", "floor_band", "encounter_family"],
                "thresholds": {"reward_pairwise_accuracy_min": 0.58},
                "promotion": {"passed": True, "check_count": 3},
                "overall": {
                    "expected_reward": {"pairwise_accuracy": 0.83, "ndcg_at_k": 0.88},
                    "outcome_win_probability": {"pairwise_accuracy": 0.79},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "dataset_dir": dataset_dir,
        "dataset_summary_path": dataset_summary_path,
        "benchmark_dir": benchmark_dir,
        "benchmark_summary_path": benchmark_summary_path,
        "benchmark_case_dir": benchmark_case_dir,
        "benchmark_case_summary_path": benchmark_case_summary_path,
        "bc_dir": bc_dir,
        "bc_summary_path": bc_summary_path,
        "offline_dir": offline_dir,
        "offline_summary_path": offline_summary_path,
        "predictor_report_dir": predictor_report_dir,
        "predictor_report_summary_path": predictor_report_summary_path,
    }
