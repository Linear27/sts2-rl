import json
from pathlib import Path

from sts2_rl.registry import (
    build_registry_leaderboard,
    compare_registry_experiments,
    get_registry_experiment,
    initialize_registry,
    list_registry_experiments,
    load_registry_aliases,
    register_experiment,
    set_registry_alias,
)
from sts2_rl.train import (
    StrategicFinetuneTrainConfig,
    StrategicPretrainTrainConfig,
    train_strategic_finetune_policy,
    train_strategic_pretrain_policy,
)
from tests.registry_fixtures import build_registry_fixture
from tests.test_strategic_finetune import _write_public_dataset, _write_runtime_dataset


def test_registry_registers_experiments_and_preserves_lineage(tmp_path: Path) -> None:
    fixture = build_registry_fixture(tmp_path)
    registry_root = tmp_path / "artifacts" / "registry"
    initialize_registry(registry_root)

    dataset_report = register_experiment(registry_root, source=fixture["dataset_dir"], aliases=["dataset_current"])
    bc_report = register_experiment(registry_root, source=fixture["bc_dir"], aliases=["best_bc"])
    offline_report = register_experiment(registry_root, source=fixture["offline_dir"])
    predictor_report = register_experiment(registry_root, source=fixture["predictor_report_dir"], aliases=["best_predictor"])
    benchmark_report = register_experiment(registry_root, source=fixture["benchmark_dir"])
    benchmark_case_report = register_experiment(registry_root, source=fixture["benchmark_case_dir"])

    assert dataset_report.family == "dataset"
    assert bc_report.family == "behavior_cloning"
    assert offline_report.family == "offline_cql"
    assert predictor_report.family == "predictor_report"
    assert benchmark_report.family == "benchmark_suite"
    assert benchmark_case_report.family == "benchmark_case"

    bc_entry = get_registry_experiment(registry_root, bc_report.experiment_id)
    assert bc_entry["lineage"]["dataset_paths"] == [str(fixture["dataset_dir"].resolve())]
    assert bc_entry["lineage"]["benchmark_summary_paths"] == [str(fixture["benchmark_summary_path"].resolve())]
    assert bc_entry["metrics"]["primary"]["name"] == "combat_win_rate"
    assert bc_entry["aliases"] == ["best_bc"]

    benchmark_entry = get_registry_experiment(registry_root, benchmark_report.experiment_id)
    assert benchmark_entry["metrics"]["snapshot"]["promotion_candidate_count"] == 1.0

    benchmark_case_entry = get_registry_experiment(registry_root, benchmark_case_report.experiment_id)
    assert benchmark_case_entry["metrics"]["snapshot"]["promotion_passed"] == 1.0
    assert benchmark_case_entry["metrics"]["snapshot"]["route_decision_overlap_rate"] == 0.75
    assert benchmark_case_entry["metrics"]["snapshot"]["delta_route_quality_score"] == 0.42
    assert benchmark_case_entry["metrics"]["snapshot"]["delta_route_risk_score"] == -0.2
    assert benchmark_case_entry["metrics"]["snapshot"]["shadow_comparable_encounter_count"] == 12.0
    assert benchmark_case_entry["metrics"]["snapshot"]["shadow_candidate_advantage_rate"] == 0.25
    assert benchmark_case_entry["metrics"]["snapshot"]["shadow_delta_trace_hit_rate"] == 0.2

    aliases = load_registry_aliases(registry_root)
    assert aliases["dataset_current"]["experiment_id"] == dataset_report.experiment_id
    assert aliases["best_predictor"]["experiment_id"] == predictor_report.experiment_id


def test_registry_alias_updates_history_and_leaderboard(tmp_path: Path) -> None:
    fixture = build_registry_fixture(tmp_path)
    registry_root = tmp_path / "artifacts" / "registry"
    initialize_registry(registry_root)

    bc_report = register_experiment(registry_root, source=fixture["bc_dir"])
    offline_report = register_experiment(registry_root, source=fixture["offline_dir"])
    predictor_report = register_experiment(registry_root, source=fixture["predictor_report_dir"])

    alias_report = set_registry_alias(
        registry_root,
        alias_name="recommended_default",
        experiment_id=bc_report.experiment_id,
        artifact_path_key="best_checkpoint_path",
        reason="strongest benchmark",
    )
    assert alias_report.artifact_path is not None

    leaderboard = build_registry_leaderboard(registry_root, family=None, tag=None)
    leaderboard_payload = json.loads(leaderboard.summary_path.read_text(encoding="utf-8"))
    assert leaderboard_payload["row_count"] == 3
    assert leaderboard_payload["rows"][0]["experiment_id"] == predictor_report.experiment_id
    assert leaderboard_payload["rows"][1]["experiment_id"] == bc_report.experiment_id

    compare = compare_registry_experiments(
        registry_root,
        experiment_ids=[bc_report.experiment_id, offline_report.experiment_id],
    )
    compare_payload = json.loads(compare.summary_path.read_text(encoding="utf-8"))
    assert compare_payload["compared_count"] == 2
    metric_names = {row["metric_name"] for row in compare_payload["metric_rows"]}
    assert "benchmark_primary_metric" in metric_names
    assert "validation_loss" in metric_names or "validation_mean_loss" in metric_names

    alias_history = (registry_root / "alias-history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(alias_history) == 1


def test_registry_list_supports_family_tag_and_alias_filters(tmp_path: Path) -> None:
    fixture = build_registry_fixture(tmp_path)
    registry_root = tmp_path / "artifacts" / "registry"
    initialize_registry(registry_root)

    register_experiment(registry_root, source=fixture["bc_dir"], tags=["train", "policy"], aliases=["best_bc"])
    register_experiment(registry_root, source=fixture["offline_dir"], tags=["train", "policy"])
    register_experiment(registry_root, source=fixture["predictor_report_dir"], tags=["report", "predictor"])

    bc_rows = list_registry_experiments(registry_root, family="behavior_cloning")
    policy_rows = list_registry_experiments(registry_root, tag="policy")
    alias_rows = list_registry_experiments(registry_root, alias="best_bc")

    assert len(bc_rows) == 1
    assert len(policy_rows) == 2
    assert len(alias_rows) == 1
    assert alias_rows[0]["family"] == "behavior_cloning"


def test_registry_inspects_strategic_finetune_training_lineage(tmp_path: Path) -> None:
    registry_root = tmp_path / "artifacts" / "registry"
    initialize_registry(registry_root)

    runtime_dataset = _write_runtime_dataset(tmp_path)
    public_dataset = _write_public_dataset(tmp_path)
    pretrain_report = train_strategic_pretrain_policy(
        dataset_source=public_dataset,
        output_root=tmp_path / "artifacts" / "strategic-pretrain",
        session_name="registry-warmstart",
        config=StrategicPretrainTrainConfig(epochs=5),
    )
    finetune_report = train_strategic_finetune_policy(
        runtime_dataset_source=runtime_dataset,
        public_dataset_source=public_dataset,
        output_root=tmp_path / "artifacts" / "strategic-finetune",
        session_name="registry-finetune",
        config=StrategicFinetuneTrainConfig(
            epochs=5,
            runtime_build_id="v0.103.0",
            warmstart_checkpoint_path=pretrain_report.best_checkpoint_path,
        ),
    )

    report = register_experiment(registry_root, source=finetune_report.output_dir, aliases=["best_strategic_ft"])
    entry = get_registry_experiment(registry_root, report.experiment_id)

    assert report.family == "strategic_finetune"
    assert entry["lineage"]["dataset_paths"] == [
        str(runtime_dataset.resolve()),
        str(public_dataset.resolve()),
    ]
    assert entry["references"]["warmstart_checkpoint_path"] == str(pretrain_report.best_checkpoint_path.resolve())
    assert entry["aliases"] == ["best_strategic_ft"]
