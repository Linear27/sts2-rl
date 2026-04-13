import json
from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app

runner = CliRunner()


def test_benchmark_suite_validate_and_summary_commands(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "latest.json").write_text("latest", encoding="utf-8")

    manifest_path = tmp_path / "suite.toml"
    manifest_path.write_text(
        """
schema_version = 1
suite_name = "bench"

[[cases]]
case_id = "eval"
mode = "eval"
checkpoint_path = "./checkpoints/latest.json"
prepare_target = "none"
""".strip(),
        encoding="utf-8",
    )
    summary_path = tmp_path / "benchmark-suite-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "suite_name": "bench",
                "base_url": "http://127.0.0.1:8080",
                "case_count": 1,
                "case_mode_histogram": {"eval": 1},
                "stats": {"bootstrap_resamples": 200, "confidence_level": 0.95, "seed": 11},
                "summary_path": str(summary_path),
                "log_path": str(tmp_path / "benchmark-suite-log.jsonl"),
                "cases": [
                    {
                        "case_id": "eval",
                        "mode": "eval",
                        "primary_metric": {
                            "name": "combat_win_rate",
                            "estimate": 0.5,
                            "ci_low": 0.25,
                            "ci_high": 0.75,
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    validate_result = runner.invoke(app, ["benchmark", "suite", "validate", "--manifest", str(manifest_path)])
    summary_result = runner.invoke(app, ["benchmark", "suite", "summary", "--source", str(summary_path)])

    assert validate_result.exit_code == 0
    assert "bench" in validate_result.stdout
    assert summary_result.exit_code == 0
    assert "combat_win_rate" in summary_result.stdout
