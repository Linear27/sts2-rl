from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app
from tests.registry_fixtures import build_registry_fixture

runner = CliRunner()


def test_registry_cli_flows(tmp_path: Path) -> None:
    fixture = build_registry_fixture(tmp_path)
    registry_root = tmp_path / "artifacts" / "registry"

    init_result = runner.invoke(
        app,
        ["registry", "init", "--root", str(registry_root)],
    )
    register_bc_result = runner.invoke(
        app,
        [
            "registry",
            "register",
            "--root",
            str(registry_root),
            "--source",
            str(fixture["bc_dir"]),
            "--alias",
            "best_bc",
        ],
    )
    register_offline_result = runner.invoke(
        app,
        [
            "registry",
            "register",
            "--root",
            str(registry_root),
            "--source",
            str(fixture["offline_dir"]),
            "--alias",
            "offline_candidate",
        ],
    )
    register_predictor_result = runner.invoke(
        app,
        [
            "registry",
            "register",
            "--root",
            str(registry_root),
            "--source",
            str(fixture["predictor_report_dir"]),
        ],
    )
    list_result = runner.invoke(app, ["registry", "list", "--root", str(registry_root)])
    show_result = runner.invoke(app, ["registry", "show", "--root", str(registry_root), "--experiment", "best_bc"])
    alias_set_result = runner.invoke(
        app,
        [
            "registry",
            "alias",
            "set",
            "--root",
            str(registry_root),
            "--alias-name",
            "recommended_default",
            "--experiment",
            "best_bc",
            "--artifact-path-key",
            "best_checkpoint_path",
        ],
    )
    leaderboard_result = runner.invoke(app, ["registry", "leaderboard", "--root", str(registry_root), "--session-name", "cli-board"])
    compare_result = runner.invoke(
        app,
        [
            "registry",
            "compare",
            "--root",
            str(registry_root),
            "--session-name",
            "cli-compare",
            "--experiment",
            "best_bc",
            "--experiment",
            "offline_candidate",
        ],
    )
    promote_result = runner.invoke(
        app,
        [
            "registry",
            "promote",
            "--root",
            str(registry_root),
            "--alias-name",
            "best_predictor",
            "--family",
            "predictor_report",
        ],
    )
    alias_list_result = runner.invoke(app, ["registry", "alias", "list", "--root", str(registry_root)])

    assert init_result.exit_code == 0
    assert register_bc_result.exit_code == 0
    assert register_offline_result.exit_code == 0
    assert register_predictor_result.exit_code == 0
    assert list_result.exit_code == 0
    assert show_result.exit_code == 0
    assert alias_set_result.exit_code == 0
    assert leaderboard_result.exit_code == 0
    assert compare_result.exit_code == 0
    assert promote_result.exit_code == 0
    assert alias_list_result.exit_code == 0
    assert (registry_root / "reports" / "cli-board" / "leaderboard-summary.json").exists()
    assert (registry_root / "reports" / "cli-compare" / "compare-summary.json").exists()
    assert (registry_root / "experiments").exists()
    assert (registry_root / "aliases.json").exists()
