import json
from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app

runner = CliRunner()


def test_eval_capability_summary_command(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "session_name": "capability-session",
                "non_combat_capability": {
                    "diagnostic_count": 1,
                    "owner_histogram": {"sts2-rl": 1},
                    "bucket_histogram": {"repo_action_space_gap": 1},
                    "category_histogram": {"unsupported_action_descriptor": 1},
                    "screen_histogram": {"MAP": 1},
                    "descriptor_histogram": {"mystery_map_action": 1},
                    "reason_histogram": {},
                    "regression_key_histogram": {
                        "repo_action_space_gap|unsupported_action_descriptor|MAP|mystery_map_action": 1
                    },
                    "unsupported_descriptor_count": 1,
                    "no_action_timeout_count": 0,
                    "ambiguous_semantic_block_count": 0,
                    "unexpected_runtime_divergence_count": 0,
                    "diagnostics": [
                        {
                            "status": "issue",
                            "bucket": "repo_action_space_gap",
                            "owner": "sts2-rl",
                            "category": "unsupported_action_descriptor",
                            "screen_type": "MAP",
                            "step_index": 0,
                            "descriptor": "mystery_map_action",
                            "decision_reason": None,
                            "stop_reason": None,
                            "explanation": "unsupported",
                            "regression_key": "repo_action_space_gap|unsupported_action_descriptor|MAP|mystery_map_action",
                            "details": {"warning": "Unsupported action descriptor: mystery_map_action"},
                        }
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval", "capability-summary", "--source", str(summary_path)])

    assert result.exit_code == 0
    assert "repo_action_space_gap" in result.stdout
    assert "mystery_map_action" in result.stdout
