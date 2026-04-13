from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app
from tests.registry_fixtures import build_registry_fixture

runner = CliRunner()


def test_cli_experiment_dag_flows(tmp_path: Path) -> None:
    fixture = build_registry_fixture(tmp_path)
    manifest_path = tmp_path / "dag.toml"
    manifest_path.write_text(
        f"""
schema_version = 1
dag_name = "cli-dag"
output_root = "{(tmp_path / 'artifacts' / 'dags').as_posix()}"
lock_root = "{(tmp_path / 'artifacts' / 'locks').as_posix()}"

[[stages]]
stage_id = "register"
kind = "registry_register"
registry_root = "{(tmp_path / 'artifacts' / 'registry').as_posix()}"
source = "{fixture['bc_dir'].as_posix()}"
aliases = ["bc_candidate"]

[[stages]]
stage_id = "promote"
kind = "registry_promote"
depends_on = ["register"]
registry_root = "{(tmp_path / 'artifacts' / 'registry').as_posix()}"
alias_name = "best_bc"
experiment = "${{stages.register.outputs.experiment_id}}"
""".strip(),
        encoding="utf-8",
    )

    validate_result = runner.invoke(app, ["instances", "dag", "validate", "--manifest", str(manifest_path)])
    run_result = runner.invoke(app, ["instances", "dag", "run", "--manifest", str(manifest_path)])
    inspect_result = runner.invoke(
        app,
        [
            "instances",
            "dag",
            "inspect",
            "--source",
            str(tmp_path / "artifacts" / "dags" / "cli-dag"),
        ],
    )
    summary_result = runner.invoke(
        app,
        [
            "instances",
            "dag",
            "summary",
            "--source",
            str(tmp_path / "artifacts" / "dags" / "cli-dag"),
        ],
    )

    assert validate_result.exit_code == 0
    assert run_result.exit_code == 0
    assert inspect_result.exit_code == 0
    assert summary_result.exit_code == 0
    assert (tmp_path / "artifacts" / "dags" / "cli-dag" / "dag-summary.json").exists()
    assert (tmp_path / "artifacts" / "registry" / "aliases.json").exists()
