import json
from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app
from tests.predict_report_fixtures import build_synthetic_predictor_fixture

runner = CliRunner()


def test_predict_report_cli_commands(tmp_path: Path) -> None:
    dataset_dir, model_path = build_synthetic_predictor_fixture(tmp_path)
    suite_summary_path = tmp_path / "benchmarks" / "predictor-suite" / "benchmark-suite-summary.json"
    suite_summary_path.parent.mkdir(parents=True)
    suite_summary_path.write_text(
        json.dumps(
            {
                "suite_name": "predictor-suite",
                "cases": [
                    {
                        "case_id": "compare-dominant",
                        "mode": "compare",
                        "predictor": {
                            "baseline": {"mode": "heuristic_only", "hooks": []},
                            "candidate": {"mode": "dominant", "hooks": ["combat"]},
                        },
                        "scenario": {"floor_min": 1, "floor_max": 16},
                        "metrics": {
                            "delta_total_reward": {"mean": 1.0},
                            "delta_combat_win_rate": {"mean": 0.1},
                            "delta_reward_per_combat": {"mean": 0.05},
                        },
                        "baseline": {"mean_total_reward": 1.0, "combat_win_rate": 0.5, "reward_per_combat": 0.2},
                        "candidate": {"mean_total_reward": 2.0, "combat_win_rate": 0.6, "reward_per_combat": 0.25},
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    calibration_result = runner.invoke(
        app,
        [
            "predict",
            "report",
            "calibration",
            "--model-path",
            str(model_path),
            "--dataset",
            str(dataset_dir),
            "--output-root",
            str(tmp_path / "artifacts" / "reports"),
            "--session-name",
            "cli-calibration",
            "--split",
            "all",
            "--min-slice-examples",
            "1",
            "--outcome-ece-max",
            "1.0",
            "--outcome-brier-max",
            "1.0",
            "--reward-rmse-max",
            "100.0",
            "--damage-rmse-max",
            "100.0",
        ],
    )
    ranking_result = runner.invoke(
        app,
        [
            "predict",
            "report",
            "ranking",
            "--model-path",
            str(model_path),
            "--dataset",
            str(dataset_dir),
            "--output-root",
            str(tmp_path / "artifacts" / "reports"),
            "--session-name",
            "cli-ranking",
            "--split",
            "all",
            "--group-by",
            "character",
            "--group-by",
            "floor_band",
            "--group-by",
            "encounter_family",
            "--reward-ndcg-min",
            "0.0",
            "--damage-ndcg-min",
            "0.0",
            "--outcome-pairwise-accuracy-min",
            "0.0",
            "--reward-pairwise-accuracy-min",
            "0.0",
            "--damage-pairwise-accuracy-min",
            "0.0",
        ],
    )
    compare_result = runner.invoke(
        app,
        [
            "predict",
            "report",
            "compare",
            "--source",
            str(suite_summary_path),
            "--output-root",
            str(tmp_path / "artifacts" / "reports"),
            "--session-name",
            "cli-compare",
        ],
    )

    assert calibration_result.exit_code == 0
    assert "Predictor Calibration" in calibration_result.stdout
    assert ranking_result.exit_code == 0
    assert "Predictor Ranking" in ranking_result.stdout
    assert compare_result.exit_code == 0
    assert "Predictor Benchmark Compare" in compare_result.stdout
    assert (tmp_path / "artifacts" / "reports" / "cli-calibration" / "summary.json").exists()
    assert (tmp_path / "artifacts" / "reports" / "cli-ranking" / "summary.json").exists()
    assert (tmp_path / "artifacts" / "reports" / "cli-compare" / "summary.json").exists()
