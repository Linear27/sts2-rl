from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app
from tests.shadow_fixtures import build_shadow_encounter_fixture

runner = CliRunner()


def test_shadow_cli_commands(tmp_path: Path) -> None:
    source_dir = build_shadow_encounter_fixture(tmp_path)
    output_root = tmp_path / "artifacts" / "shadow"

    eval_result = runner.invoke(
        app,
        [
            "shadow",
            "combat-eval",
            "--source",
            str(source_dir),
            "--output-root",
            str(output_root),
            "--session-name",
            "cli-shadow-eval",
            "--policy-profile",
            "planner",
        ],
    )
    eval_summary_result = runner.invoke(
        app,
        [
            "shadow",
            "summary",
            "--source",
            str(output_root / "cli-shadow-eval"),
        ],
    )
    compare_result = runner.invoke(
        app,
        [
            "shadow",
            "combat-compare",
            "--source",
            str(source_dir),
            "--output-root",
            str(output_root),
            "--session-name",
            "cli-shadow-compare",
            "--baseline-policy-profile",
            "baseline",
            "--candidate-policy-profile",
            "planner",
        ],
    )
    compare_summary_result = runner.invoke(
        app,
        [
            "shadow",
            "summary",
            "--source",
            str(output_root / "cli-shadow-compare"),
        ],
    )

    assert eval_result.exit_code == 0
    assert "Shadow Combat Eval" in eval_result.stdout
    assert eval_summary_result.exit_code == 0
    assert "Shadow Combat Eval Summary" in eval_summary_result.stdout
    assert compare_result.exit_code == 0
    assert "Shadow Combat Compare" in compare_result.stdout
    assert compare_summary_result.exit_code == 0
    assert "Shadow Combat Compare Summary" in compare_summary_result.stdout
    assert (output_root / "cli-shadow-eval" / "summary.json").exists()
    assert (output_root / "cli-shadow-compare" / "summary.json").exists()
