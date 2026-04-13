from pathlib import Path

import pytest

from sts2_rl.runtime import load_runtime_job_manifest


def test_runtime_job_manifest_resolves_relative_paths_and_validates_instances(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    models_dir = tmp_path / "models"
    benchmarks_dir = tmp_path / "benchmarks"
    community_dir = tmp_path / "community"
    route_dir = tmp_path / "public-runs"
    checkpoints_dir.mkdir()
    models_dir.mkdir()
    benchmarks_dir.mkdir()
    community_dir.mkdir()
    route_dir.mkdir()
    (checkpoints_dir / "best.json").write_text("{}", encoding="utf-8")
    (checkpoints_dir / "candidate.json").write_text("{}", encoding="utf-8")
    (models_dir / "predictor.json").write_text("{}", encoding="utf-8")
    (community_dir / "community-card-stats.jsonl").write_text("", encoding="utf-8")
    (route_dir / "strategic-route-stats.jsonl").write_text("", encoding="utf-8")
    (benchmarks_dir / "suite.toml").write_text(
        """
schema_version = 1
suite_name = "suite"

[[cases]]
case_id = "eval"
mode = "eval"
policy_profile = "baseline"
prepare_target = "none"
""".strip(),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "job.toml"
    manifest_path.write_text(
        """
schema_version = 1
job_name = "smoke-job"
output_root = "./artifacts/jobs"

[watchdog]
failure_degraded_threshold = 1
failure_cooldown_threshold = 2
failure_quarantine_threshold = 3

[[tasks]]
task_id = "collect-smoke"
kind = "collect"
instance_ids = ["inst-01"]
policy_profile = "planner"
normalize_target = "main_menu"

[tasks.predictor]
model_path = "./models/predictor.json"
mode = "assist"
hooks = ["combat"]

[tasks.community_prior]
source_path = "./community/community-card-stats.jsonl"
route_source_path = "./public-runs/strategic-route-stats.jsonl"
shop_buy_weight = 1.4

[[tasks]]
task_id = "eval-best"
kind = "eval_checkpoint"
instance_ids = ["inst-02"]
checkpoint_path = "./checkpoints/best.json"

[tasks.community_prior]
source_path = "./community/community-card-stats.jsonl"

[[tasks]]
task_id = "compare-best"
kind = "compare"
baseline_checkpoint_path = "./checkpoints/best.json"
candidate_checkpoint_path = "./checkpoints/candidate.json"

[[tasks]]
task_id = "benchmark-smoke"
kind = "benchmark"
benchmark_manifest_path = "./benchmarks/suite.toml"
""".strip(),
        encoding="utf-8",
    )

    manifest = load_runtime_job_manifest(
        manifest_path,
        known_instance_ids={"inst-01", "inst-02"},
    )

    assert manifest.output_root == (tmp_path / "artifacts" / "jobs").resolve()
    assert manifest.tasks[0].predictor.model_path == (models_dir / "predictor.json").resolve()
    assert manifest.tasks[0].community_prior is not None
    assert manifest.tasks[0].community_prior.source_path == (community_dir / "community-card-stats.jsonl").resolve()
    assert manifest.tasks[0].community_prior.route_source_path == (route_dir / "strategic-route-stats.jsonl").resolve()
    assert manifest.tasks[1].checkpoint_path == (checkpoints_dir / "best.json").resolve()
    assert manifest.tasks[1].community_prior is not None
    assert manifest.tasks[1].community_prior.source_path == (community_dir / "community-card-stats.jsonl").resolve()
    assert manifest.tasks[2].candidate_checkpoint_path == (checkpoints_dir / "candidate.json").resolve()
    assert manifest.tasks[3].benchmark_manifest_path == (benchmarks_dir / "suite.toml").resolve()

    with pytest.raises(ValueError, match="Unknown instance ids"):
        load_runtime_job_manifest(manifest_path, known_instance_ids={"inst-01"})
