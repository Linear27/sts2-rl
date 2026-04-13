import json
from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app

runner = CliRunner()


def test_eval_divergence_summary_command(tmp_path: Path) -> None:
    summary_path = tmp_path / "replay-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "log_path": str(tmp_path / "replay-suite.jsonl"),
                "comparisons": [
                    {
                        "diagnostic": {
                            "status": "policy_choice_diverged",
                            "family": "policy_choice",
                            "category": "chosen_action_mismatch",
                            "explanation": "action mismatch",
                            "step_index": 2,
                        }
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval", "divergence-summary", "--source", str(summary_path)])

    assert result.exit_code == 0
    assert "policy_choice" in result.stdout
    assert "chosen_action_mismatch" in result.stdout
