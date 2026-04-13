import json
from pathlib import Path

from sts2_rl.runtime import (
    CollectJobTaskSpec,
    RuntimeJobManifest,
    RuntimeNormalizationReport,
    WatchdogPolicy,
    load_instance_config,
    load_runtime_job_summary,
)
from sts2_rl.runtime.job_runner import InstanceProbeResult, TaskDispatchResult, run_runtime_job


def test_run_runtime_job_retries_and_quarantines_instances(tmp_path: Path) -> None:
    config_path = tmp_path / "local.toml"
    config_path.write_text(
        f"""
[reference]
game_root = "{(tmp_path / 'reference').as_posix()}"
clean_baseline_root = "{(tmp_path / 'clean').as_posix()}"

[runtime]
instances_root = "{(tmp_path / 'runtime').as_posix()}"
first_api_port = 8100
instance_count = 2

[mods]
enable_sts2_agent = true
enable_animation_acceleration = true

[logging]
instance_logs_root = "{(tmp_path / 'logs').as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    config = load_instance_config(config_path)
    manifest = RuntimeJobManifest(
        job_name="runtime-job-smoke",
        output_root=tmp_path / "artifacts",
        watchdog=WatchdogPolicy(
            failure_degraded_threshold=1,
            failure_cooldown_threshold=2,
            failure_quarantine_threshold=2,
            cooldown_seconds=0.0,
        ),
        tasks=[
            CollectJobTaskSpec(
                task_id="collect-primary",
                instance_ids=["inst-01", "inst-02"],
                max_attempts=2,
                normalize_target="main_menu",
            ),
            CollectJobTaskSpec(
                task_id="collect-followup",
                instance_ids=["inst-02"],
                max_attempts=1,
            ),
        ],
    )

    dispatch_calls: dict[tuple[str, str], int] = {}

    def fake_probe(_spec, _timeout: float) -> InstanceProbeResult:
        return InstanceProbeResult(
            reachable=True,
            status="ok",
            game_version="v0.103.0",
            mod_version="agent",
            payload={"status": "ok"},
        )

    def fake_normalize(**kwargs) -> RuntimeNormalizationReport:
        return RuntimeNormalizationReport(
            base_url=str(kwargs["base_url"]),
            target=kwargs["target"],
            reached_target=True,
            stop_reason="target_reached",
            initial_screen="MAIN_MENU",
            final_screen="MAIN_MENU",
            initial_run_id="run_unknown",
            final_run_id="run_unknown",
            step_count=0,
            wait_count=0,
            action_sequence=[],
            strategy_histogram={},
            final_observation=None,
        )

    def fake_dispatch(unit, attempt_root: Path) -> TaskDispatchResult:
        key = (unit.task.task_id, unit.spec.instance_id)
        dispatch_calls[key] = dispatch_calls.get(key, 0) + 1
        attempt_root.mkdir(parents=True, exist_ok=True)
        if key == ("collect-primary", "inst-01") and dispatch_calls[key] == 1:
            raise RuntimeError("transient failure")
        if key == ("collect-primary", "inst-02"):
            raise RuntimeError("persistent failure")
        summary_path = attempt_root / "summary.json"
        log_path = attempt_root / "rollout.jsonl"
        combat_outcomes_path = attempt_root / "combat-outcomes.jsonl"
        summary_path.write_text(
            json.dumps({"task_id": unit.task.task_id, "instance_id": unit.spec.instance_id}, ensure_ascii=False),
            encoding="utf-8",
        )
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")
        return TaskDispatchResult(
            artifact_root=attempt_root,
            summary_path=summary_path,
            log_path=log_path,
            combat_outcomes_path=combat_outcomes_path,
            metrics={"calls": dispatch_calls[key]},
            raw_summary={"task_id": unit.task.task_id},
        )

    report = run_runtime_job(
        manifest,
        config=config,
        probe_fn=fake_probe,
        normalize_fn=fake_normalize,
        dispatch_fn=fake_dispatch,
    )

    summary = load_runtime_job_summary(report.summary_path)
    executions = {item["execution_id"]: item for item in summary["executions"]}

    assert summary["execution_status_histogram"] == {"succeeded": 1, "failed": 1, "quarantined": 1}
    assert summary["watchdogs"]["inst-01"]["state"] == "healthy"
    assert summary["watchdogs"]["inst-01"]["total_failures"] == 1
    assert summary["watchdogs"]["inst-02"]["state"] == "quarantined"
    assert executions["collect-primary:inst-01"]["attempt_count"] == 2
    assert executions["collect-primary:inst-01"]["status"] == "succeeded"
    assert executions["collect-primary:inst-02"]["status"] == "failed"
    assert executions["collect-followup:inst-02"]["status"] == "quarantined"
    assert dispatch_calls[("collect-primary", "inst-01")] == 2
    assert dispatch_calls[("collect-primary", "inst-02")] == 2
