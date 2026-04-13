from pathlib import Path

import pytest

from sts2_rl.runtime import load_experiment_dag_manifest, validate_experiment_dag_manifest


def test_experiment_dag_manifest_resolves_relative_paths_and_reports_stage_order(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    configs_dir = tmp_path / "instances"
    jobs_dir.mkdir()
    configs_dir.mkdir()
    (jobs_dir / "collect.toml").write_text(
        """
schema_version = 1
job_name = "collect"

[[tasks]]
task_id = "collect"
kind = "collect"
""".strip(),
        encoding="utf-8",
    )
    (configs_dir / "local.toml").write_text(
        """
[reference]
game_root = "D:/stub/game"
clean_baseline_root = "D:/stub/clean"

[runtime]
instances_root = "D:/stub/runtime"
first_api_port = 8080
instance_count = 1

[mods]
enable_sts2_agent = true
enable_animation_acceleration = true

[logging]
instance_logs_root = "D:/stub/logs"
""".strip(),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "dag.toml"
    manifest_path.write_text(
        """
schema_version = 1
dag_name = "smoke-dag"
output_root = "./artifacts/dags"
lock_root = "./artifacts/locks"

[[stages]]
stage_id = "collect"
kind = "runtime_job"
manifest_path = "./jobs/collect.toml"
config_path = "./instances/local.toml"

[[stages]]
stage_id = "register"
kind = "registry_register"
depends_on = ["collect"]
registry_root = "./artifacts/registry"
source = "${stages.collect.outputs.job_root}"
""".strip(),
        encoding="utf-8",
    )

    manifest = load_experiment_dag_manifest(manifest_path)
    report = validate_experiment_dag_manifest(manifest, deep=False)

    assert manifest.output_root == (tmp_path / "artifacts" / "dags").resolve()
    assert manifest.lock_root == (tmp_path / "artifacts" / "locks").resolve()
    assert manifest.stages[0].manifest_path == (jobs_dir / "collect.toml").resolve()
    assert manifest.stages[0].config_path == (configs_dir / "local.toml").resolve()
    assert report.stage_order == ["collect", "register"]
    assert report.stage_resource_hints["collect"][0].startswith("runtime-config:")


def test_experiment_dag_manifest_rejects_dependency_cycles(tmp_path: Path) -> None:
    manifest_path = tmp_path / "cycle.toml"
    manifest_path.write_text(
        """
schema_version = 1
dag_name = "cycle"

[[stages]]
stage_id = "a"
kind = "registry_leaderboard"
registry_root = "./registry"
depends_on = ["b"]

[[stages]]
stage_id = "b"
kind = "registry_compare"
registry_root = "./registry"
experiments = ["x", "y"]
depends_on = ["a"]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dependency cycle"):
        load_experiment_dag_manifest(manifest_path)
