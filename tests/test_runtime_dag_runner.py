import json
from pathlib import Path

from sts2_rl.runtime import (
    ExperimentDagManifest,
    PredictDatasetExtractDagStageSpec,
    PredictReportRankingDagStageSpec,
    PredictTrainDagStageSpec,
    RegistryPromoteDagStageSpec,
    RegistryRegisterDagStageSpec,
    RuntimeJobDagStageSpec,
    load_experiment_dag_state,
    load_experiment_dag_summary,
    resume_experiment_dag,
    run_experiment_dag,
)
from sts2_rl.runtime.dag_runner import DagStageResult


def test_run_experiment_dag_retries_stage_attempts(tmp_path: Path) -> None:
    manifest = ExperimentDagManifest(
        dag_name="retry-dag",
        output_root=tmp_path / "artifacts" / "dags",
        lock_root=tmp_path / "artifacts" / "locks",
        stages=[
            RuntimeJobDagStageSpec(
                stage_id="collect",
                manifest_path=tmp_path / "jobs" / "stub.toml",
                config_path=tmp_path / "instances" / "stub.toml",
                max_attempts=2,
            )
        ],
    )

    call_count = {"collect": 0}

    def fake_runtime_job(stage, resolved_inputs, context) -> DagStageResult:
        call_count["collect"] += 1
        if call_count["collect"] == 1:
            raise RuntimeError("transient failure")
        summary_path = context.attempt_root / "summary.json"
        summary_path.write_text(json.dumps({"stage": stage.stage_id}), encoding="utf-8")
        return DagStageResult(
            artifact_root=context.attempt_root,
            summary_path=summary_path,
            metrics={"calls": call_count["collect"]},
            outputs={"job_root": str(context.attempt_root)},
        )

    report = run_experiment_dag(
        manifest,
        executor_overrides={"runtime_job": fake_runtime_job},
    )
    summary_payload = load_experiment_dag_summary(report.summary_path)
    stage_payload = summary_payload["stages"][0]

    assert summary_payload["status"] == "succeeded"
    assert stage_payload["status"] == "succeeded"
    assert stage_payload["attempt_count"] == 2
    assert call_count["collect"] == 2


def test_resume_experiment_dag_reuses_successful_stages_and_reruns_descendants(tmp_path: Path) -> None:
    manifest = ExperimentDagManifest(
        dag_name="resume-dag",
        output_root=tmp_path / "artifacts" / "dags",
        lock_root=tmp_path / "artifacts" / "locks",
        stages=[
            RuntimeJobDagStageSpec(
                stage_id="prepare",
                manifest_path=tmp_path / "jobs" / "prepare.toml",
                config_path=tmp_path / "instances" / "prepare.toml",
            ),
            RuntimeJobDagStageSpec(
                stage_id="train",
                manifest_path=tmp_path / "jobs" / "train.toml",
                config_path=tmp_path / "instances" / "train.toml",
                depends_on=["prepare"],
            ),
            RuntimeJobDagStageSpec(
                stage_id="promote",
                manifest_path=tmp_path / "jobs" / "promote.toml",
                config_path=tmp_path / "instances" / "promote.toml",
                depends_on=["train"],
            ),
        ],
    )

    train_should_fail = {"value": True}
    call_histogram = {"prepare": 0, "train": 0, "promote": 0}

    def fake_runtime_job(stage, resolved_inputs, context) -> DagStageResult:
        call_histogram[stage.stage_id] += 1
        if stage.stage_id == "train" and train_should_fail["value"]:
            raise RuntimeError("first run failure")
        summary_path = context.attempt_root / "summary.json"
        summary_path.write_text(json.dumps({"stage": stage.stage_id}), encoding="utf-8")
        return DagStageResult(
            artifact_root=context.attempt_root,
            summary_path=summary_path,
            outputs={"job_root": str(context.attempt_root), "stage_id": stage.stage_id},
        )

    first_report = run_experiment_dag(
        manifest,
        executor_overrides={"runtime_job": fake_runtime_job},
    )
    first_summary = load_experiment_dag_summary(first_report.summary_path)
    assert first_summary["status"] == "failed"
    assert [stage["status"] for stage in first_summary["stages"]] == ["succeeded", "failed", "blocked"]

    train_should_fail["value"] = False
    resumed_report = resume_experiment_dag(
        first_report.run_root,
        retry_stage_ids=["train"],
        executor_overrides={"runtime_job": fake_runtime_job},
    )
    resumed_summary = load_experiment_dag_summary(resumed_report.summary_path)
    resumed_state = load_experiment_dag_state(resumed_report.state_path)

    assert resumed_summary["status"] == "succeeded"
    assert resumed_summary["reused_stage_count"] == 1
    assert [stage["status"] for stage in resumed_summary["stages"]] == ["succeeded", "succeeded", "succeeded"]
    assert resumed_state["stages"]["prepare"]["reused_from_previous_run"] is True
    assert call_histogram == {"prepare": 1, "train": 2, "promote": 1}


def test_run_experiment_dag_surfaces_resource_lock_conflicts(tmp_path: Path) -> None:
    lock_root = tmp_path / "artifacts" / "locks"
    lock_root.mkdir(parents=True)
    (lock_root / "gpu-0.lock").write_text(json.dumps({"owner": {"run_name": "other"}}), encoding="utf-8")
    manifest = ExperimentDagManifest(
        dag_name="lock-dag",
        output_root=tmp_path / "artifacts" / "dags",
        lock_root=lock_root,
        stages=[
            RuntimeJobDagStageSpec(
                stage_id="collect",
                manifest_path=tmp_path / "jobs" / "stub.toml",
                config_path=tmp_path / "instances" / "stub.toml",
                resources=["gpu:0"],
            )
        ],
    )

    def fake_runtime_job(stage, resolved_inputs, context) -> DagStageResult:
        summary_path = context.attempt_root / "summary.json"
        summary_path.write_text(json.dumps({"stage": stage.stage_id}), encoding="utf-8")
        return DagStageResult(artifact_root=context.attempt_root, summary_path=summary_path)

    report = run_experiment_dag(
        manifest,
        executor_overrides={"runtime_job": fake_runtime_job},
    )
    summary_payload = load_experiment_dag_summary(report.summary_path)

    assert summary_payload["status"] == "failed"
    assert summary_payload["stages"][0]["failure_kind"] == "resource_lock_conflict"


def test_run_experiment_dag_predictor_pipeline_and_registry_handoff(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "session-a"
    artifacts_root.mkdir(parents=True)
    combat_outcomes_path = artifacts_root / "combat-outcomes.jsonl"
    records = [
        _combat_outcome_payload("RUN-001", 3, "won", 72, 120, "SLIME_SMALL", 18),
        _combat_outcome_payload("RUN-002", 4, "lost", 12, 60, "FUNGI_BEAST", 50),
        _combat_outcome_payload("RUN-003", 5, "won", 68, 128, "SLAVER_RED", 20),
        _combat_outcome_payload("RUN-004", 6, "lost", 10, 58, "SENTRY", 52),
        _combat_outcome_payload("RUN-005", 7, "won", 64, 136, "JAW_WORM", 22),
        _combat_outcome_payload("RUN-006", 8, "lost", 8, 54, "ORB_WALKER", 60),
    ]
    with combat_outcomes_path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in records:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    manifest = ExperimentDagManifest(
        dag_name="predictor-dag",
        output_root=tmp_path / "artifacts" / "dags",
        lock_root=tmp_path / "artifacts" / "locks",
        stages=[
            PredictDatasetExtractDagStageSpec(
                stage_id="extract",
                sources=[artifacts_root],
                output_dir=tmp_path / "data" / "predict" / "dag-dataset",
                replace_existing=True,
                split_seed=3,
                train_fraction=0.5,
                validation_fraction=0.25,
                test_fraction=0.25,
                split_group_by="record",
            ),
            PredictTrainDagStageSpec(
                stage_id="train",
                depends_on=["extract"],
                dataset_source="${stages.extract.outputs.output_dir}",
                output_root=tmp_path / "artifacts" / "predict",
                session_name="dag-train",
            ),
            PredictReportRankingDagStageSpec(
                stage_id="rank",
                depends_on=["train"],
                model_path="${stages.train.outputs.model_path}",
                dataset_source="${stages.extract.outputs.output_dir}",
                output_root=tmp_path / "artifacts" / "predict-reports",
                session_name="dag-ranking",
                split="all",
                group_by=["character", "floor_band", "encounter_family"],
                top_k=3,
                min_group_size=2,
            ),
            RegistryRegisterDagStageSpec(
                stage_id="register",
                depends_on=["rank"],
                registry_root=tmp_path / "artifacts" / "registry",
                source="${stages.rank.outputs.output_dir}",
                aliases=["predictor_candidate"],
            ),
            RegistryPromoteDagStageSpec(
                stage_id="promote",
                depends_on=["register"],
                registry_root=tmp_path / "artifacts" / "registry",
                alias_name="best_predictor",
                experiment="${stages.register.outputs.experiment_id}",
            ),
        ],
    )

    report = run_experiment_dag(manifest)
    summary_payload = load_experiment_dag_summary(report.summary_path)
    aliases_payload = json.loads((tmp_path / "artifacts" / "registry" / "aliases.json").read_text(encoding="utf-8"))

    assert summary_payload["status"] == "succeeded"
    assert [stage["status"] for stage in summary_payload["stages"]] == ["succeeded"] * 5
    assert aliases_payload["aliases"]["best_predictor"]["experiment_id"] == summary_payload["stages"][3]["outputs"]["experiment_id"]


def _combat_outcome_payload(
    run_id: str,
    floor: int,
    outcome: str,
    player_hp: int,
    gold: int,
    enemy_id: str,
    enemy_hp: int,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "record_type": "combat_finished",
        "timestamp_utc": "2026-04-12T00:00:00+00:00",
        "session_name": "session-a",
        "session_kind": "train",
        "instance_id": "inst-01",
        "run_id": run_id,
        "floor": floor,
        "combat_index": floor,
        "started_step_index": 0,
        "finished_step_index": 7,
        "outcome": outcome,
        "cumulative_reward": 1.5 if outcome == "won" else -1.0,
        "step_count": 7,
        "enemy_ids": [enemy_id],
        "damage_dealt": 32 if outcome == "won" else 10,
        "damage_taken": 5 if outcome == "won" else 30,
        "start_summary": {
            "screen_type": "COMBAT",
            "run_id": run_id,
            "state_version": 8,
            "turn": 1,
            "in_combat": True,
            "available_action_count": 4,
            "build_warning_count": 0,
            "session_phase": "run",
            "control_scope": "local_player",
            "run": {
                "character_id": "IRONCLAD",
                "character_name": "Ironclad",
                "ascension": 0,
                "floor": floor,
                "current_hp": player_hp,
                "max_hp": 80,
                "gold": gold,
                "max_energy": 3,
                "occupied_potions": 0,
            },
            "combat": {
                "player_hp": player_hp,
                "player_block": 0,
                "energy": 3,
                "stars": 0,
                "focus": 0,
                "enemy_ids": [enemy_id],
                "enemy_hp": [enemy_hp],
                "hand_card_ids": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
                "playable_hand_count": 3,
            },
        },
        "end_summary": {"screen_type": "MAP" if outcome == "won" else "GAME_OVER", "run_id": run_id},
        "reason": "combat_exited",
    }
