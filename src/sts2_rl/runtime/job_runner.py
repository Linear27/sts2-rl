from __future__ import annotations

import json
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any, Callable

import httpx

from sts2_rl.collect.runner import CollectionReport, collect_round_robin
from sts2_rl.train import (
    BenchmarkSuiteReport,
    CombatCheckpointComparisonReport,
    CombatEvaluationReport,
    CombatReplaySuiteReport,
    load_benchmark_suite_manifest,
    run_benchmark_suite,
    run_combat_dqn_checkpoint_comparison,
    run_combat_dqn_evaluation,
    run_combat_dqn_replay_suite,
    run_policy_pack_evaluation,
)

from .config import LocalInstanceConfig, load_instance_config
from .job_manifest import (
    BenchmarkJobTaskSpec,
    CollectJobTaskSpec,
    CompareJobTaskSpec,
    EvalCheckpointJobTaskSpec,
    EvalPolicyPackJobTaskSpec,
    ReplayJobTaskSpec,
    RUNTIME_JOB_LOG_FILENAME,
    RUNTIME_JOB_SUMMARY_FILENAME,
    RuntimeJobManifest,
    RuntimeJobTaskSpec,
    load_runtime_job_manifest,
)
from .manifest import InstanceSpec, build_instance_specs
from .normalize import RuntimeNormalizationReport, normalize_runtime_state
from .watchdog import (
    InstanceWatchdogStatus,
    WatchdogPolicy,
    cooldown_remaining_seconds,
    record_watchdog_failure,
    record_watchdog_success,
)


@dataclass(frozen=True)
class InstanceProbeResult:
    reachable: bool
    status: str | None
    game_version: str | None
    mod_version: str | None
    payload: dict[str, Any]
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "reachable": self.reachable,
            "status": self.status,
            "game_version": self.game_version,
            "mod_version": self.mod_version,
            "payload": self.payload,
            "error": self.error,
        }


@dataclass(frozen=True)
class TaskDispatchResult:
    artifact_root: Path
    summary_path: Path | None
    log_path: Path | None
    combat_outcomes_path: Path | None
    metrics: dict[str, object]
    raw_summary: dict[str, object] | None = None
    extra_paths: dict[str, str] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_root": str(self.artifact_root),
            "summary_path": str(self.summary_path) if self.summary_path is not None else None,
            "log_path": str(self.log_path) if self.log_path is not None else None,
            "combat_outcomes_path": (
                str(self.combat_outcomes_path) if self.combat_outcomes_path is not None else None
            ),
            "metrics": self.metrics,
            "raw_summary": self.raw_summary,
            "extra_paths": self.extra_paths or {},
        }


@dataclass(frozen=True)
class JobExecutionAttempt:
    attempt_index: int
    started_at_utc: str
    finished_at_utc: str
    status: str
    watchdog_state_before: str
    watchdog_state_after: str
    failure_kind: str | None
    retryable: bool
    error: str | None
    health_probe: dict[str, object] | None
    normalization: dict[str, object] | None
    dispatch: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "status": self.status,
            "watchdog_state_before": self.watchdog_state_before,
            "watchdog_state_after": self.watchdog_state_after,
            "failure_kind": self.failure_kind,
            "retryable": self.retryable,
            "error": self.error,
            "health_probe": self.health_probe,
            "normalization": self.normalization,
            "dispatch": self.dispatch,
        }


@dataclass(frozen=True)
class JobExecutionReport:
    execution_index: int
    execution_id: str
    task_id: str
    task_kind: str
    instance_id: str
    base_url: str
    status: str
    attempt_count: int
    max_attempts: int
    final_failure_kind: str | None
    final_error: str | None
    watchdog_state: str
    artifact_root: Path
    summary_path: Path | None
    log_path: Path | None
    combat_outcomes_path: Path | None
    metrics: dict[str, object]
    attempts: list[JobExecutionAttempt]

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_index": self.execution_index,
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "task_kind": self.task_kind,
            "instance_id": self.instance_id,
            "base_url": self.base_url,
            "status": self.status,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "final_failure_kind": self.final_failure_kind,
            "final_error": self.final_error,
            "watchdog_state": self.watchdog_state,
            "artifact_root": str(self.artifact_root),
            "summary_path": str(self.summary_path) if self.summary_path is not None else None,
            "log_path": str(self.log_path) if self.log_path is not None else None,
            "combat_outcomes_path": (
                str(self.combat_outcomes_path) if self.combat_outcomes_path is not None else None
            ),
            "metrics": self.metrics,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
        }


@dataclass(frozen=True)
class RuntimeJobReport:
    job_name: str
    job_root: Path
    summary_path: Path
    log_path: Path
    manifest_path: Path | None
    config_path: Path | None
    execution_reports: list[JobExecutionReport]
    watchdogs: dict[str, InstanceWatchdogStatus]

    def as_dict(self) -> dict[str, object]:
        execution_status_histogram = Counter(report.status for report in self.execution_reports)
        task_kind_histogram = Counter(report.task_kind for report in self.execution_reports)
        return {
            "job_name": self.job_name,
            "job_root": str(self.job_root),
            "summary_path": str(self.summary_path),
            "log_path": str(self.log_path),
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "config_path": str(self.config_path) if self.config_path is not None else None,
            "execution_count": len(self.execution_reports),
            "execution_status_histogram": dict(execution_status_histogram),
            "task_kind_histogram": dict(task_kind_histogram),
            "watchdogs": {
                instance_id: status.as_dict()
                for instance_id, status in sorted(self.watchdogs.items())
            },
            "executions": [report.as_dict() for report in self.execution_reports],
        }


@dataclass(frozen=True)
class _ExecutionUnit:
    execution_index: int
    execution_id: str
    task: RuntimeJobTaskSpec
    spec: InstanceSpec


def run_runtime_job(
    manifest: RuntimeJobManifest | str | Path,
    *,
    config: LocalInstanceConfig | str | Path,
    job_name: str | None = None,
    output_root: str | Path | None = None,
    replace_existing: bool = False,
    probe_fn: Callable[[InstanceSpec, float], InstanceProbeResult] | None = None,
    normalize_fn: Callable[..., RuntimeNormalizationReport] = normalize_runtime_state,
    dispatch_fn: Callable[[_ExecutionUnit, Path], TaskDispatchResult] | None = None,
    sleep_fn: Callable[[float], None] = sleep,
) -> RuntimeJobReport:
    manifest_model, manifest_path = _resolve_job_manifest_input(manifest, config=config)
    config_model, config_path = _resolve_config_input(config)
    specs_by_id = {spec.instance_id: spec for spec in build_instance_specs(config_model)}
    requested_ids = {
        instance_id
        for task in manifest_model.tasks
        for instance_id in (task.instance_ids or list(specs_by_id))
    }
    unknown_ids = sorted(requested_ids - set(specs_by_id))
    if unknown_ids:
        raise ValueError(f"Unknown instance ids requested by runtime job: {', '.join(unknown_ids)}")

    resolved_job_name = job_name or manifest_model.job_name
    resolved_output_root = (
        manifest_model.output_root if output_root is None else Path(output_root).expanduser().resolve()
    )
    job_root = resolved_output_root / resolved_job_name
    if job_root.exists():
        if not replace_existing:
            raise FileExistsError(f"Runtime job output already exists: {job_root}")
        shutil.rmtree(job_root)
    job_root.mkdir(parents=True, exist_ok=True)

    log_path = job_root / RUNTIME_JOB_LOG_FILENAME
    summary_path = job_root / RUNTIME_JOB_SUMMARY_FILENAME
    execution_units = _build_execution_units(manifest_model, specs_by_id=specs_by_id)
    if not execution_units:
        raise ValueError("Runtime job did not resolve any execution units.")

    max_workers = _resolve_concurrency_limit(
        manifest_model.concurrency_limit,
        default_limit=len(execution_units),
    )
    task_semaphores = {
        task.task_id: threading.Semaphore(
            _resolve_concurrency_limit(task.concurrency_limit, default_limit=len(specs_by_id))
        )
        for task in manifest_model.tasks
    }
    instance_locks = {instance_id: threading.Lock() for instance_id in specs_by_id}
    instance_order_locks = {instance_id: threading.Lock() for instance_id in specs_by_id}
    instance_conditions = {
        instance_id: threading.Condition(instance_order_locks[instance_id])
        for instance_id in specs_by_id
    }
    instance_execution_order = {instance_id: [] for instance_id in specs_by_id}
    for unit in execution_units:
        instance_execution_order[unit.spec.instance_id].append(unit.execution_index)
    watchdogs = {
        instance_id: InstanceWatchdogStatus(instance_id=instance_id)
        for instance_id in specs_by_id
    }
    log_lock = threading.Lock()
    resolved_probe_fn = probe_fn or probe_instance_health
    resolved_dispatch_fn = dispatch_fn or _dispatch_execution_unit

    _append_job_log(
        log_path,
        {
            "record_type": "job_started",
            "job_name": resolved_job_name,
            "job_root": str(job_root),
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "config_path": str(config_path) if config_path is not None else None,
            "execution_count": len(execution_units),
            "concurrency_limit": max_workers,
        },
        lock=log_lock,
    )

    execution_reports: list[JobExecutionReport] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _run_execution_unit,
                unit,
                job_root,
                manifest_model.watchdog,
                task_semaphores[unit.task.task_id],
                instance_locks[unit.spec.instance_id],
                instance_conditions[unit.spec.instance_id],
                instance_execution_order[unit.spec.instance_id],
                watchdogs,
                log_path,
                log_lock,
                resolved_probe_fn,
                normalize_fn,
                resolved_dispatch_fn,
                sleep_fn,
            )
            for unit in execution_units
        ]
        for future in as_completed(futures):
            execution_reports.append(future.result())

    execution_reports.sort(key=lambda item: item.execution_index)
    report = RuntimeJobReport(
        job_name=resolved_job_name,
        job_root=job_root,
        summary_path=summary_path,
        log_path=log_path,
        manifest_path=manifest_path,
        config_path=config_path,
        execution_reports=execution_reports,
        watchdogs=watchdogs,
    )
    summary_path.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    _append_job_log(
        log_path,
        {
            "record_type": "job_finished",
            "summary_path": str(summary_path),
            "execution_status_histogram": report.as_dict()["execution_status_histogram"],
        },
        lock=log_lock,
    )
    return report


def probe_instance_health(spec: InstanceSpec, timeout_seconds: float) -> InstanceProbeResult:
    try:
        with httpx.Client(base_url=spec.base_url, timeout=timeout_seconds) as client:
            response = client.get("/health")
            response.raise_for_status()
            payload = response.json()
        data = payload.get("data", {})
        return InstanceProbeResult(
            reachable=True,
            status=data.get("status"),
            game_version=data.get("game_version"),
            mod_version=data.get("mod_version"),
            payload=data if isinstance(data, dict) else {},
        )
    except Exception as exc:
        return InstanceProbeResult(
            reachable=False,
            status=None,
            game_version=None,
            mod_version=None,
            payload={},
            error=str(exc),
        )


def _resolve_job_manifest_input(
    manifest: RuntimeJobManifest | str | Path,
    *,
    config: LocalInstanceConfig | str | Path,
) -> tuple[RuntimeJobManifest, Path | None]:
    known_instance_ids = None
    if isinstance(config, LocalInstanceConfig):
        known_instance_ids = {spec.instance_id for spec in build_instance_specs(config)}
    elif isinstance(config, (str, Path)):
        known_instance_ids = {
            spec.instance_id
            for spec in build_instance_specs(load_instance_config(config))
        }
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).expanduser().resolve()
        return load_runtime_job_manifest(manifest_path, known_instance_ids=known_instance_ids), manifest_path
    return manifest, None


def _resolve_config_input(config: LocalInstanceConfig | str | Path) -> tuple[LocalInstanceConfig, Path | None]:
    if isinstance(config, (str, Path)):
        config_path = Path(config).expanduser().resolve()
        return load_instance_config(config_path), config_path
    return config, None


def _build_execution_units(
    manifest: RuntimeJobManifest,
    *,
    specs_by_id: dict[str, InstanceSpec],
) -> list[_ExecutionUnit]:
    units: list[_ExecutionUnit] = []
    execution_index = 0
    for task in manifest.tasks:
        selected_instance_ids = task.instance_ids or sorted(specs_by_id)
        for instance_id in selected_instance_ids:
            execution_index += 1
            units.append(
                _ExecutionUnit(
                    execution_index=execution_index,
                    execution_id=f"{task.task_id}:{instance_id}",
                    task=task,
                    spec=specs_by_id[instance_id],
                )
            )
    return units


def _run_execution_unit(
    unit: _ExecutionUnit,
    job_root: Path,
    watchdog_policy: WatchdogPolicy,
    task_semaphore: threading.Semaphore,
    instance_lock: threading.Lock,
    instance_condition: threading.Condition,
    instance_execution_order: list[int],
    watchdogs: dict[str, InstanceWatchdogStatus],
    log_path: Path,
    log_lock: threading.Lock,
    probe_fn: Callable[[InstanceSpec, float], InstanceProbeResult],
    normalize_fn: Callable[..., RuntimeNormalizationReport],
    dispatch_fn: Callable[[_ExecutionUnit, Path], TaskDispatchResult],
    sleep_fn: Callable[[float], None],
) -> JobExecutionReport:
    attempts: list[JobExecutionAttempt] = []
    execution_root = job_root / "tasks" / unit.task.task_id / unit.spec.instance_id
    execution_root.mkdir(parents=True, exist_ok=True)

    with instance_condition:
        while instance_execution_order and instance_execution_order[0] != unit.execution_index:
            instance_condition.wait()

    with task_semaphore:
        with instance_lock:
            try:
                watchdog = watchdogs[unit.spec.instance_id]
                if watchdog.state == "quarantined":
                    report = JobExecutionReport(
                        execution_index=unit.execution_index,
                        execution_id=unit.execution_id,
                        task_id=unit.task.task_id,
                        task_kind=unit.task.kind,
                        instance_id=unit.spec.instance_id,
                        base_url=unit.spec.base_url,
                        status="quarantined",
                        attempt_count=0,
                        max_attempts=unit.task.max_attempts,
                        final_failure_kind=watchdog.last_failure_kind,
                        final_error=watchdog.quarantine_reason or watchdog.last_error,
                        watchdog_state=watchdog.state,
                        artifact_root=execution_root,
                        summary_path=None,
                        log_path=None,
                        combat_outcomes_path=None,
                        metrics={},
                        attempts=[],
                    )
                    _append_job_log(
                        log_path,
                        {
                            "record_type": "execution_quarantined",
                            "execution_id": unit.execution_id,
                            "task_id": unit.task.task_id,
                            "instance_id": unit.spec.instance_id,
                            "watchdog_state": watchdog.state,
                            "quarantine_reason": watchdog.quarantine_reason,
                        },
                        lock=log_lock,
                    )
                    return report

                for attempt_index in range(1, unit.task.max_attempts + 1):
                    watchdog = watchdogs[unit.spec.instance_id]
                    remaining = cooldown_remaining_seconds(watchdog)
                    if remaining > 0.0:
                        _append_job_log(
                            log_path,
                            {
                                "record_type": "execution_cooldown_wait",
                                "execution_id": unit.execution_id,
                                "instance_id": unit.spec.instance_id,
                                "remaining_seconds": remaining,
                                "watchdog_state": watchdog.state,
                            },
                            lock=log_lock,
                        )
                        sleep_fn(remaining)

                    started_at = _timestamp_utc()
                    watchdog_before = watchdog.state
                    health_probe = probe_fn(unit.spec, watchdog_policy.health_check_timeout_seconds)
                    normalization_payload: dict[str, object] | None = None
                    dispatch_payload: dict[str, object] | None = None

                    _append_job_log(
                        log_path,
                        {
                            "record_type": "attempt_started",
                            "execution_id": unit.execution_id,
                            "attempt_index": attempt_index,
                            "instance_id": unit.spec.instance_id,
                            "watchdog_state": watchdog_before,
                        },
                        lock=log_lock,
                    )

                    if not health_probe.reachable:
                        retryable = attempt_index < unit.task.max_attempts
                        watchdog = record_watchdog_failure(
                            watchdog,
                            policy=watchdog_policy,
                            failure_kind="health_check_failed",
                            error=health_probe.error,
                            terminal=not retryable,
                        )
                        watchdogs[unit.spec.instance_id] = watchdog
                        attempt = JobExecutionAttempt(
                            attempt_index=attempt_index,
                            started_at_utc=started_at,
                            finished_at_utc=_timestamp_utc(),
                            status="failed",
                            watchdog_state_before=watchdog_before,
                            watchdog_state_after=watchdog.state,
                            failure_kind="health_check_failed",
                            retryable=retryable and watchdog.state != "quarantined",
                            error=health_probe.error,
                            health_probe=health_probe.as_dict(),
                            normalization=None,
                            dispatch=None,
                        )
                        attempts.append(attempt)
                        _append_job_log(
                            log_path,
                            {
                                "record_type": "attempt_finished",
                                "execution_id": unit.execution_id,
                                "attempt_index": attempt_index,
                                "status": attempt.status,
                                "failure_kind": attempt.failure_kind,
                                "retryable": attempt.retryable,
                                "watchdog_state": watchdog.state,
                                "error": attempt.error,
                            },
                            lock=log_lock,
                        )
                        if not attempt.retryable:
                            return _failed_execution_report(
                                unit,
                                execution_root,
                                attempts,
                                watchdog,
                                failure_kind="health_check_failed",
                                error=health_probe.error,
                            )
                        continue

                    if unit.task.normalize_target != "none":
                        try:
                            normalization = normalize_fn(
                                base_url=unit.spec.base_url,
                                target=unit.task.normalize_target,
                                poll_interval_seconds=unit.task.normalize_poll_interval_seconds,
                                max_idle_polls=unit.task.normalize_max_idle_polls,
                                max_steps=unit.task.normalize_max_steps,
                                request_timeout_seconds=unit.task.request_timeout_seconds,
                            )
                            normalization_payload = normalization.as_dict()
                        except Exception as exc:
                            normalization = None
                            normalization_error = str(exc)
                        else:
                            normalization_error = None
                        if normalization_error is not None or normalization is None or not normalization.reached_target:
                            failure_error = normalization_error or (
                                "normalization_target_not_reached:"
                                f"{normalization.final_screen}:{normalization.stop_reason}"
                            )
                            retryable = attempt_index < unit.task.max_attempts
                            watchdog = record_watchdog_failure(
                                watchdogs[unit.spec.instance_id],
                                policy=watchdog_policy,
                                failure_kind="normalization_failed",
                                error=failure_error,
                                terminal=not retryable,
                            )
                            watchdogs[unit.spec.instance_id] = watchdog
                            attempt = JobExecutionAttempt(
                                attempt_index=attempt_index,
                                started_at_utc=started_at,
                                finished_at_utc=_timestamp_utc(),
                                status="failed",
                                watchdog_state_before=watchdog_before,
                                watchdog_state_after=watchdog.state,
                                failure_kind="normalization_failed",
                                retryable=retryable and watchdog.state != "quarantined",
                                error=failure_error,
                                health_probe=health_probe.as_dict(),
                                normalization=normalization_payload,
                                dispatch=None,
                            )
                            attempts.append(attempt)
                            _append_job_log(
                                log_path,
                                {
                                    "record_type": "attempt_finished",
                                    "execution_id": unit.execution_id,
                                    "attempt_index": attempt_index,
                                    "status": attempt.status,
                                    "failure_kind": attempt.failure_kind,
                                    "retryable": attempt.retryable,
                                    "watchdog_state": watchdog.state,
                                    "error": attempt.error,
                                },
                                lock=log_lock,
                            )
                            if not attempt.retryable:
                                return _failed_execution_report(
                                    unit,
                                    execution_root,
                                    attempts,
                                    watchdog,
                                    failure_kind="normalization_failed",
                                    error=failure_error,
                                )
                            continue

                    attempt_root = execution_root / f"attempt-{attempt_index:03d}"
                    attempt_root.mkdir(parents=True, exist_ok=True)
                    try:
                        dispatch_result = dispatch_fn(unit, attempt_root)
                        dispatch_payload = dispatch_result.as_dict()
                    except Exception as exc:
                        failure_kind, retryable_by_kind = _classify_execution_exception(exc)
                        retryable = retryable_by_kind and attempt_index < unit.task.max_attempts
                        watchdog = record_watchdog_failure(
                            watchdogs[unit.spec.instance_id],
                            policy=watchdog_policy,
                            failure_kind=failure_kind,
                            error=str(exc),
                            terminal=not retryable,
                        )
                        watchdogs[unit.spec.instance_id] = watchdog
                        attempt = JobExecutionAttempt(
                            attempt_index=attempt_index,
                            started_at_utc=started_at,
                            finished_at_utc=_timestamp_utc(),
                            status="failed",
                            watchdog_state_before=watchdog_before,
                            watchdog_state_after=watchdog.state,
                            failure_kind=failure_kind,
                            retryable=retryable and watchdog.state != "quarantined",
                            error=str(exc),
                            health_probe=health_probe.as_dict(),
                            normalization=normalization_payload,
                            dispatch=None,
                        )
                        attempts.append(attempt)
                        _append_job_log(
                            log_path,
                            {
                                "record_type": "attempt_finished",
                                "execution_id": unit.execution_id,
                                "attempt_index": attempt_index,
                                "status": attempt.status,
                                "failure_kind": attempt.failure_kind,
                                "retryable": attempt.retryable,
                                "watchdog_state": watchdog.state,
                                "error": attempt.error,
                            },
                            lock=log_lock,
                        )
                        if not attempt.retryable:
                            return _failed_execution_report(
                                unit,
                                execution_root,
                                attempts,
                                watchdog,
                                failure_kind=failure_kind,
                                error=str(exc),
                            )
                        continue

                    watchdog = record_watchdog_success(
                        watchdogs[unit.spec.instance_id],
                        policy=watchdog_policy,
                    )
                    watchdogs[unit.spec.instance_id] = watchdog
                    attempt = JobExecutionAttempt(
                        attempt_index=attempt_index,
                        started_at_utc=started_at,
                        finished_at_utc=_timestamp_utc(),
                        status="succeeded",
                        watchdog_state_before=watchdog_before,
                        watchdog_state_after=watchdog.state,
                        failure_kind=None,
                        retryable=False,
                        error=None,
                        health_probe=health_probe.as_dict(),
                        normalization=normalization_payload,
                        dispatch=dispatch_payload,
                    )
                    attempts.append(attempt)
                    _append_job_log(
                        log_path,
                        {
                            "record_type": "attempt_finished",
                            "execution_id": unit.execution_id,
                            "attempt_index": attempt_index,
                            "status": attempt.status,
                            "watchdog_state": watchdog.state,
                            "summary_path": (
                                str(dispatch_result.summary_path) if dispatch_result.summary_path is not None else None
                            ),
                        },
                        lock=log_lock,
                    )
                    return JobExecutionReport(
                        execution_index=unit.execution_index,
                        execution_id=unit.execution_id,
                        task_id=unit.task.task_id,
                        task_kind=unit.task.kind,
                        instance_id=unit.spec.instance_id,
                        base_url=unit.spec.base_url,
                        status="succeeded",
                        attempt_count=len(attempts),
                        max_attempts=unit.task.max_attempts,
                        final_failure_kind=None,
                        final_error=None,
                        watchdog_state=watchdog.state,
                        artifact_root=dispatch_result.artifact_root,
                        summary_path=dispatch_result.summary_path,
                        log_path=dispatch_result.log_path,
                        combat_outcomes_path=dispatch_result.combat_outcomes_path,
                        metrics=dispatch_result.metrics,
                        attempts=attempts,
                    )
            finally:
                with instance_condition:
                    if instance_execution_order and instance_execution_order[0] == unit.execution_index:
                        instance_execution_order.pop(0)
                    instance_condition.notify_all()

    raise RuntimeError(f"Execution loop exited unexpectedly for {unit.execution_id}")


def _failed_execution_report(
    unit: _ExecutionUnit,
    execution_root: Path,
    attempts: list[JobExecutionAttempt],
    watchdog: InstanceWatchdogStatus,
    *,
    failure_kind: str,
    error: str | None,
) -> JobExecutionReport:
    return JobExecutionReport(
        execution_index=unit.execution_index,
        execution_id=unit.execution_id,
        task_id=unit.task.task_id,
        task_kind=unit.task.kind,
        instance_id=unit.spec.instance_id,
        base_url=unit.spec.base_url,
        status="failed",
        attempt_count=len(attempts),
        max_attempts=unit.task.max_attempts,
        final_failure_kind=failure_kind,
        final_error=error,
        watchdog_state=watchdog.state,
        artifact_root=execution_root,
        summary_path=None,
        log_path=None,
        combat_outcomes_path=None,
        metrics={},
        attempts=attempts,
    )


def _dispatch_execution_unit(unit: _ExecutionUnit, attempt_root: Path) -> TaskDispatchResult:
    task = unit.task
    if isinstance(task, CollectJobTaskSpec):
        return _dispatch_collect(task, unit.spec, attempt_root)
    if isinstance(task, EvalCheckpointJobTaskSpec):
        return _dispatch_eval_checkpoint(task, unit.spec, attempt_root)
    if isinstance(task, EvalPolicyPackJobTaskSpec):
        return _dispatch_eval_policy_pack(task, unit.spec, attempt_root)
    if isinstance(task, CompareJobTaskSpec):
        return _dispatch_compare(task, unit.spec, attempt_root)
    if isinstance(task, ReplayJobTaskSpec):
        return _dispatch_replay(task, unit.spec, attempt_root)
    if isinstance(task, BenchmarkJobTaskSpec):
        return _dispatch_benchmark(task, unit.spec, attempt_root)
    raise TypeError(f"Unsupported runtime job task: {task!r}")


def _dispatch_collect(task: CollectJobTaskSpec, spec: InstanceSpec, attempt_root: Path) -> TaskDispatchResult:
    report: CollectionReport = collect_round_robin(
        [spec],
        output_root=attempt_root,
        policy_profile=task.policy_profile,
        predictor_config=task.predictor.to_runtime_config(),
        community_prior_config=None if task.community_prior is None else task.community_prior.to_runtime_config(),
        max_steps_per_instance=task.max_steps_per_instance,
        max_runs_per_instance=task.max_runs_per_instance,
        max_combats_per_instance=task.max_combats_per_instance,
        poll_interval_seconds=task.poll_interval_seconds,
        idle_timeout_seconds=task.idle_timeout_seconds,
    )[0]
    raw_summary = _load_json_if_exists(report.summary_path)
    return TaskDispatchResult(
        artifact_root=attempt_root,
        summary_path=report.summary_path,
        log_path=report.output_path,
        combat_outcomes_path=report.combat_outcomes_path,
        metrics={
            "step_count": report.step_count,
            "stop_reason": report.stop_reason,
            "completed_run_count": report.completed_run_count,
            "completed_combat_count": report.completed_combat_count,
            "last_screen": report.last_screen,
            "last_run_id": report.last_run_id,
            "error": report.error,
        },
        raw_summary=raw_summary,
    )


def _dispatch_eval_checkpoint(
    task: EvalCheckpointJobTaskSpec,
    spec: InstanceSpec,
    attempt_root: Path,
) -> TaskDispatchResult:
    report: CombatEvaluationReport = run_combat_dqn_evaluation(
        base_url=spec.base_url,
        checkpoint_path=task.checkpoint_path,
        output_root=attempt_root.parent,
        session_name=attempt_root.name,
        max_env_steps=task.max_env_steps,
        max_runs=task.max_runs,
        max_combats=task.max_combats,
        poll_interval_seconds=task.poll_interval_seconds,
        max_idle_polls=task.max_idle_polls,
        request_timeout_seconds=task.request_timeout_seconds,
        policy_profile=task.policy_profile,
        predictor_config=task.predictor.to_runtime_config(),
        community_prior_config=None if task.community_prior is None else task.community_prior.to_runtime_config(),
    )
    return _combat_evaluation_dispatch_result(report)


def _dispatch_eval_policy_pack(
    task: EvalPolicyPackJobTaskSpec,
    spec: InstanceSpec,
    attempt_root: Path,
) -> TaskDispatchResult:
    report: CombatEvaluationReport = run_policy_pack_evaluation(
        base_url=spec.base_url,
        output_root=attempt_root.parent,
        session_name=attempt_root.name,
        policy_profile=task.policy_profile,
        max_env_steps=task.max_env_steps,
        max_runs=task.max_runs,
        max_combats=task.max_combats,
        poll_interval_seconds=task.poll_interval_seconds,
        max_idle_polls=task.max_idle_polls,
        request_timeout_seconds=task.request_timeout_seconds,
        predictor_config=task.predictor.to_runtime_config(),
        community_prior_config=None if task.community_prior is None else task.community_prior.to_runtime_config(),
    )
    return _combat_evaluation_dispatch_result(report)


def _combat_evaluation_dispatch_result(report: CombatEvaluationReport) -> TaskDispatchResult:
    raw_summary = _load_json_if_exists(report.summary_path)
    return TaskDispatchResult(
        artifact_root=report.summary_path.parent,
        summary_path=report.summary_path,
        log_path=report.log_path,
        combat_outcomes_path=report.combat_outcomes_path,
        metrics={
            "env_steps": report.env_steps,
            "combat_steps": report.combat_steps,
            "heuristic_steps": report.heuristic_steps,
            "total_reward": report.total_reward,
            "final_screen": report.final_screen,
            "final_run_id": report.final_run_id,
            "stop_reason": report.stop_reason,
            "completed_run_count": report.completed_run_count,
            "completed_combat_count": report.completed_combat_count,
            "combat_win_rate": report.combat_performance.get("combat_win_rate"),
            "reward_per_combat": report.combat_performance.get("reward_per_combat"),
        },
        raw_summary=raw_summary,
    )


def _dispatch_compare(task: CompareJobTaskSpec, spec: InstanceSpec, attempt_root: Path) -> TaskDispatchResult:
    report: CombatCheckpointComparisonReport = run_combat_dqn_checkpoint_comparison(
        base_url=spec.base_url,
        baseline_checkpoint_path=task.baseline_checkpoint_path,
        candidate_checkpoint_path=task.candidate_checkpoint_path,
        output_root=attempt_root.parent,
        comparison_name=attempt_root.name,
        repeats=task.repeats,
        max_env_steps=task.max_env_steps,
        max_runs=task.max_runs,
        max_combats=task.max_combats,
        poll_interval_seconds=task.poll_interval_seconds,
        max_idle_polls=task.max_idle_polls,
        request_timeout_seconds=task.request_timeout_seconds,
        prepare_target=task.prepare_target,
        prepare_main_menu=task.prepare_target == "main_menu",
        prepare_max_steps=task.prepare_max_steps,
        prepare_max_idle_polls=task.prepare_max_idle_polls,
    )
    raw_summary = _load_json_if_exists(report.summary_path)
    return TaskDispatchResult(
        artifact_root=report.comparison_dir,
        summary_path=report.summary_path,
        log_path=report.log_path,
        combat_outcomes_path=None,
        metrics={
            "repeat_count": report.repeat_count,
            "prepare_target": report.prepare_target,
            "better_checkpoint_label": report.better_checkpoint_label,
            "candidate_reward_delta": report.delta_metrics.get("mean_total_reward"),
            "candidate_win_rate_delta": report.delta_metrics.get("combat_win_rate"),
        },
        raw_summary=raw_summary,
        extra_paths={"iterations_path": str(report.iterations_path)},
    )


def _dispatch_replay(task: ReplayJobTaskSpec, spec: InstanceSpec, attempt_root: Path) -> TaskDispatchResult:
    report: CombatReplaySuiteReport = run_combat_dqn_replay_suite(
        base_url=spec.base_url,
        checkpoint_path=task.checkpoint_path,
        output_root=attempt_root.parent,
        suite_name=attempt_root.name,
        repeats=task.repeats,
        max_env_steps=task.max_env_steps,
        max_runs=task.max_runs,
        max_combats=task.max_combats,
        poll_interval_seconds=task.poll_interval_seconds,
        max_idle_polls=task.max_idle_polls,
        request_timeout_seconds=task.request_timeout_seconds,
        prepare_target=task.prepare_target,
        prepare_main_menu=task.prepare_target == "main_menu",
        prepare_max_steps=task.prepare_max_steps,
        prepare_max_idle_polls=task.prepare_max_idle_polls,
    )
    raw_summary = _load_json_if_exists(report.summary_path)
    return TaskDispatchResult(
        artifact_root=report.suite_dir,
        summary_path=report.summary_path,
        log_path=report.log_path,
        combat_outcomes_path=None,
        metrics={
            "repeat_count": report.repeat_count,
            "comparison_count": report.comparison_count,
            "exact_match_count": report.exact_match_count,
            "divergent_iteration_count": report.divergent_iteration_count,
            "prepare_target": report.prepare_target,
            "status_histogram": report.status_histogram,
        },
        raw_summary=raw_summary,
        extra_paths={"comparisons_path": str(report.comparisons_path)},
    )


def _dispatch_benchmark(task: BenchmarkJobTaskSpec, spec: InstanceSpec, attempt_root: Path) -> TaskDispatchResult:
    manifest = load_benchmark_suite_manifest(task.benchmark_manifest_path)
    bound_manifest = manifest.model_copy(update={"base_url": spec.base_url})
    report: BenchmarkSuiteReport = run_benchmark_suite(
        bound_manifest,
        output_root=attempt_root.parent,
        suite_name=attempt_root.name,
        replace_existing=task.replace_existing,
    )
    raw_summary = _load_json_if_exists(report.summary_path)
    return TaskDispatchResult(
        artifact_root=report.suite_dir,
        summary_path=report.summary_path,
        log_path=report.log_path,
        combat_outcomes_path=None,
        metrics={
            "suite_name": report.suite_name,
            "case_count": len(report.case_reports),
            "case_ids": [case.case_id for case in report.case_reports],
        },
        raw_summary=raw_summary,
    )


def _resolve_concurrency_limit(limit: int, *, default_limit: int) -> int:
    if limit <= 0:
        return max(1, default_limit)
    return max(1, min(limit, default_limit))


def _classify_execution_exception(exc: Exception) -> tuple[str, bool]:
    if isinstance(exc, FileNotFoundError):
        return "artifact_missing", False
    if isinstance(exc, ValueError):
        return "invalid_task_configuration", False
    return "task_failed", True


def _append_job_log(
    path: Path,
    payload: dict[str, object],
    *,
    lock: threading.Lock,
) -> None:
    record = {"timestamp_utc": _timestamp_utc(), **payload}
    with lock:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _timestamp_utc() -> str:
    return datetime.now(UTC).isoformat()


def _load_json_if_exists(path: Path | None) -> dict[str, object] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
