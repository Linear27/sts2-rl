from __future__ import annotations

import json
import os
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any, Callable, Mapping, Sequence

from sts2_rl.data import build_dataset_from_manifest
from sts2_rl.predict.dataset import extract_predictor_dataset
from sts2_rl.predict.reports import (
    build_predictor_benchmark_comparison_report,
    build_predictor_calibration_report,
    build_predictor_ranking_report,
)
from sts2_rl.predict.trainer import train_combat_outcome_predictor
from sts2_rl.registry import (
    build_registry_leaderboard,
    compare_registry_experiments,
    get_registry_experiment,
    register_experiment,
    set_registry_alias,
)

from .config import load_instance_config
from .dag_manifest import (
    EXPERIMENT_DAG_LOG_FILENAME,
    EXPERIMENT_DAG_RESOLVED_MANIFEST_FILENAME,
    EXPERIMENT_DAG_SCHEMA_VERSION,
    EXPERIMENT_DAG_STATE_FILENAME,
    EXPERIMENT_DAG_SUMMARY_FILENAME,
    BehaviorCloningTrainDagStageSpec,
    DatasetBuildDagStageSpec,
    ExperimentDagManifest,
    ExperimentDagStageSpec,
    OfflineCqlTrainDagStageSpec,
    PredictDatasetExtractDagStageSpec,
    PredictReportCalibrationDagStageSpec,
    PredictReportCompareDagStageSpec,
    PredictReportRankingDagStageSpec,
    PredictTrainDagStageSpec,
    RegistryCompareDagStageSpec,
    RegistryLeaderboardDagStageSpec,
    RegistryPromoteDagStageSpec,
    RegistryRegisterDagStageSpec,
    RuntimeJobDagStageSpec,
    load_experiment_dag_manifest,
    load_experiment_dag_state,
    load_experiment_dag_summary,
    topologically_sort_dag,
)
from .job_manifest import load_runtime_job_manifest
from .manifest import build_instance_specs

_TEMPLATE_PATTERN = re.compile(r"\$\{([^}]+)\}")


class ResourceLockError(RuntimeError):
    pass


class InterpolationError(ValueError):
    pass


@dataclass(frozen=True)
class DagStageResult:
    artifact_root: Path | None
    summary_path: Path | None
    metrics: dict[str, object] = field(default_factory=dict)
    outputs: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "artifact_root": None if self.artifact_root is None else str(self.artifact_root),
            "summary_path": None if self.summary_path is None else str(self.summary_path),
            "metrics": _jsonify(self.metrics),
            "outputs": _jsonify(self.outputs),
            "metadata": _jsonify(self.metadata),
        }


@dataclass(frozen=True)
class DagStageAttempt:
    attempt_index: int
    started_at_utc: str
    finished_at_utc: str
    status: str
    failure_kind: str | None
    retryable: bool
    error: str | None
    stage_root: Path
    resolved_inputs: dict[str, object]
    resources: list[str]
    result: dict[str, object] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "status": self.status,
            "failure_kind": self.failure_kind,
            "retryable": self.retryable,
            "error": self.error,
            "stage_root": str(self.stage_root),
            "resolved_inputs": _jsonify(self.resolved_inputs),
            "resources": list(self.resources),
            "result": self.result,
        }


@dataclass(frozen=True)
class ExperimentDagReport:
    dag_name: str
    run_name: str
    run_root: Path
    manifest_path: Path | None
    resolved_manifest_path: Path
    state_path: Path
    summary_path: Path
    log_path: Path
    status: str
    stage_reports: list[dict[str, object]]
    invocation_count: int

    def as_dict(self) -> dict[str, object]:
        return dict(load_experiment_dag_summary(self.summary_path))


@dataclass(frozen=True)
class StageExecutionContext:
    dag_name: str
    run_name: str
    run_root: Path
    manifest_path: Path | None
    resolved_manifest_path: Path
    stage_root: Path
    attempt_root: Path
    stage_state: dict[str, object]
    state_payload: dict[str, object]


StageExecutor = Callable[[ExperimentDagStageSpec, dict[str, object], StageExecutionContext], DagStageResult]


def run_experiment_dag(
    manifest: ExperimentDagManifest | str | Path,
    *,
    run_name: str | None = None,
    output_root: str | Path | None = None,
    replace_existing: bool = False,
    executor_overrides: Mapping[str, StageExecutor] | None = None,
    sleep_fn: Callable[[float], None] = sleep,
) -> ExperimentDagReport:
    manifest_model, manifest_path = _resolve_manifest_input(manifest)
    resolved_run_name = run_name or manifest_model.dag_name
    resolved_output_root = (
        manifest_model.output_root if output_root is None else Path(output_root).expanduser().resolve()
    )
    run_root = resolved_output_root / resolved_run_name
    if run_root.exists():
        if not replace_existing:
            raise FileExistsError(f"Experiment DAG output already exists: {run_root}")
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=False)
    resolved_manifest_path = run_root / EXPERIMENT_DAG_RESOLVED_MANIFEST_FILENAME
    _write_json(resolved_manifest_path, manifest_model.model_dump(mode="json"))
    state_payload = _initialize_run_state(
        manifest_model,
        run_name=resolved_run_name,
        run_root=run_root,
        manifest_path=manifest_path,
        resolved_manifest_path=resolved_manifest_path,
    )
    return _execute_experiment_dag(
        manifest_model,
        state_payload=state_payload,
        executor_overrides=executor_overrides,
        sleep_fn=sleep_fn,
    )


def resume_experiment_dag(
    source: str | Path,
    *,
    retry_stage_ids: Sequence[str] | None = None,
    executor_overrides: Mapping[str, StageExecutor] | None = None,
    sleep_fn: Callable[[float], None] = sleep,
) -> ExperimentDagReport:
    prior_state = load_experiment_dag_state(source)
    resolved_manifest_path = Path(str(prior_state["resolved_manifest_path"])).expanduser().resolve()
    manifest_model = load_experiment_dag_manifest(resolved_manifest_path)
    requested_retry_stage_ids = set(retry_stage_ids or [])
    known_stage_ids = {stage.stage_id for stage in manifest_model.stages}
    unknown_stage_ids = sorted(requested_retry_stage_ids - known_stage_ids)
    if unknown_stage_ids:
        raise ValueError("Unknown retry stage ids: " + ", ".join(unknown_stage_ids))
    state_payload = _build_resume_state(
        manifest_model,
        prior_state=prior_state,
        retry_stage_ids=requested_retry_stage_ids,
    )
    return _execute_experiment_dag(
        manifest_model,
        state_payload=state_payload,
        executor_overrides=executor_overrides,
        sleep_fn=sleep_fn,
    )


def _execute_experiment_dag(
    manifest: ExperimentDagManifest,
    *,
    state_payload: dict[str, object],
    executor_overrides: Mapping[str, StageExecutor] | None,
    sleep_fn: Callable[[float], None],
) -> ExperimentDagReport:
    run_root = Path(str(state_payload["run_root"])).expanduser().resolve()
    state_path = run_root / EXPERIMENT_DAG_STATE_FILENAME
    summary_path = run_root / EXPERIMENT_DAG_SUMMARY_FILENAME
    log_path = run_root / EXPERIMENT_DAG_LOG_FILENAME
    executors = {**_default_stage_executors(), **dict(executor_overrides or {})}
    stage_order = topologically_sort_dag(manifest)
    manifest_path_value = state_payload.get("manifest_path")
    manifest_path = None if manifest_path_value is None else Path(str(manifest_path_value)).expanduser().resolve()
    resolved_manifest_path = Path(str(state_payload["resolved_manifest_path"])).expanduser().resolve()

    state_payload["status"] = "running"
    state_payload["last_started_at_utc"] = _timestamp_utc()
    _write_state(state_path, state_payload)
    _append_log(
        log_path,
        {
            "record_type": "dag_invocation_started",
            "run_name": state_payload["run_name"],
            "invocation_count": state_payload["invocation_count"],
            "run_root": str(run_root),
        },
    )

    stage_specs = {stage.stage_id: stage for stage in manifest.stages}
    for stage_id in stage_order:
        stage = stage_specs[stage_id]
        stage_state = dict(state_payload["stages"][stage_id])
        if stage_state.get("status") == "succeeded":
            state_payload["stages"][stage_id] = {
                **stage_state,
                "reused_from_previous_run": bool(stage_state.get("reused_from_previous_run", False)),
            }
            _append_log(log_path, {"record_type": "stage_reused", "stage_id": stage_id})
            _write_state(state_path, state_payload)
            continue

        dependency_statuses = {
            dependency: str(state_payload["stages"][dependency]["status"])
            for dependency in stage.depends_on
        }
        if any(status != "succeeded" for status in dependency_statuses.values()):
            blocked_by = sorted(
                dependency for dependency, status in dependency_statuses.items() if status != "succeeded"
            )
            state_payload["stages"][stage_id] = {
                **stage_state,
                "status": "blocked",
                "blocked_by": blocked_by,
                "last_error": "Dependencies did not succeed: " + ", ".join(blocked_by),
                "finished_at_utc": _timestamp_utc(),
            }
            _append_log(log_path, {"record_type": "stage_blocked", "stage_id": stage_id, "blocked_by": blocked_by})
            _write_state(state_path, state_payload)
            continue

        resolved_inputs = _resolve_stage_inputs(manifest, stage_id=stage_id, state_payload=state_payload)
        stage_root = run_root / "stages" / stage_id
        stage_root.mkdir(parents=True, exist_ok=True)
        attempts = list(stage_state.get("attempts", []))
        state_payload["stages"][stage_id] = {
            **stage_state,
            "status": "running",
            "started_at_utc": _timestamp_utc(),
            "blocked_by": [],
            "resolved_inputs": _jsonify(resolved_inputs),
        }
        _append_log(log_path, {"record_type": "stage_started", "stage_id": stage_id, "kind": stage.kind})
        _write_state(state_path, state_payload)

        last_failure_kind: str | None = None
        last_error: str | None = None
        resource_names = _resolve_stage_resources(stage, resolved_inputs)
        for offset in range(stage.max_attempts):
            attempt_index = len(attempts) + 1
            attempt_root = stage_root / f"attempt-{attempt_index:03d}"
            attempt_root.mkdir(parents=True, exist_ok=True)
            _write_json(attempt_root / "resolved-inputs.json", _jsonify(resolved_inputs))
            attempt_started_at = _timestamp_utc()
            _append_log(
                log_path,
                {
                    "record_type": "stage_attempt_started",
                    "stage_id": stage_id,
                    "attempt_index": attempt_index,
                    "resources": resource_names,
                },
            )
            try:
                with _ResourceLease(
                    lock_root=Path(str(state_payload["lock_root"])).expanduser().resolve(),
                    run_name=str(state_payload["run_name"]),
                    stage_id=stage_id,
                    attempt_index=attempt_index,
                    resources=resource_names,
                ):
                    executor = executors.get(stage.kind)
                    if executor is None:
                        raise ValueError(f"Unsupported DAG stage kind: {stage.kind}")
                    result = executor(
                        stage,
                        resolved_inputs,
                        StageExecutionContext(
                            dag_name=str(state_payload["dag_name"]),
                            run_name=str(state_payload["run_name"]),
                            run_root=run_root,
                            manifest_path=manifest_path,
                            resolved_manifest_path=resolved_manifest_path,
                            stage_root=stage_root,
                            attempt_root=attempt_root,
                            stage_state=deepcopy(state_payload["stages"][stage_id]),
                            state_payload=deepcopy(state_payload),
                        ),
                    )
                result_payload = result.as_dict()
                _write_json(attempt_root / "stage-result.json", result_payload)
                attempts.append(
                    DagStageAttempt(
                        attempt_index=attempt_index,
                        started_at_utc=attempt_started_at,
                        finished_at_utc=_timestamp_utc(),
                        status="succeeded",
                        failure_kind=None,
                        retryable=False,
                        error=None,
                        stage_root=attempt_root,
                        resolved_inputs=_jsonify(resolved_inputs),
                        resources=resource_names,
                        result=result_payload,
                    ).as_dict()
                )
                state_payload["stages"][stage_id] = {
                    **state_payload["stages"][stage_id],
                    "status": "succeeded",
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "reused_from_previous_run": False,
                    "finished_at_utc": _timestamp_utc(),
                    "failure_kind": None,
                    "last_error": None,
                    "artifact_root": result_payload["artifact_root"],
                    "summary_path": result_payload["summary_path"],
                    "metrics": result_payload["metrics"],
                    "outputs": result_payload["outputs"],
                    "metadata": result_payload["metadata"],
                }
                _append_log(
                    log_path,
                    {
                        "record_type": "stage_finished",
                        "stage_id": stage_id,
                        "status": "succeeded",
                        "summary_path": result_payload["summary_path"],
                    },
                )
                _write_state(state_path, state_payload)
                break
            except Exception as exc:
                last_failure_kind, retryable = _classify_stage_exception(exc)
                last_error = str(exc)
                _write_json(
                    attempt_root / "stage-error.json",
                    {
                        "failure_kind": last_failure_kind,
                        "error": last_error,
                        "retryable": retryable and offset < stage.max_attempts - 1,
                    },
                )
                attempts.append(
                    DagStageAttempt(
                        attempt_index=attempt_index,
                        started_at_utc=attempt_started_at,
                        finished_at_utc=_timestamp_utc(),
                        status="failed",
                        failure_kind=last_failure_kind,
                        retryable=retryable and offset < stage.max_attempts - 1,
                        error=last_error,
                        stage_root=attempt_root,
                        resolved_inputs=_jsonify(resolved_inputs),
                        resources=resource_names,
                        result=None,
                    ).as_dict()
                )
                state_payload["stages"][stage_id] = {
                    **state_payload["stages"][stage_id],
                    "status": "failed",
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "reused_from_previous_run": False,
                    "finished_at_utc": _timestamp_utc(),
                    "failure_kind": last_failure_kind,
                    "last_error": last_error,
                    "artifact_root": None,
                    "summary_path": None,
                    "metrics": {},
                    "outputs": {},
                    "metadata": {},
                }
                _append_log(
                    log_path,
                    {
                        "record_type": "stage_attempt_finished",
                        "stage_id": stage_id,
                        "attempt_index": attempt_index,
                        "status": "failed",
                        "failure_kind": last_failure_kind,
                        "retryable": retryable and offset < stage.max_attempts - 1,
                        "error": last_error,
                    },
                )
                _write_state(state_path, state_payload)
                if not retryable or offset >= stage.max_attempts - 1:
                    break
                if stage.cooldown_seconds > 0.0:
                    sleep_fn(stage.cooldown_seconds)

        if state_payload["stages"][stage_id]["status"] != "succeeded":
            state_payload["stages"][stage_id] = {
                **state_payload["stages"][stage_id],
                "status": "failed",
                "finished_at_utc": _timestamp_utc(),
                "failure_kind": last_failure_kind,
                "last_error": last_error,
            }
            _append_log(
                log_path,
                {
                    "record_type": "stage_finished",
                    "stage_id": stage_id,
                    "status": "failed",
                    "failure_kind": last_failure_kind,
                    "error": last_error,
                },
            )
            _write_state(state_path, state_payload)

    final_status = _final_dag_status(state_payload)
    state_payload["status"] = final_status
    state_payload["finished_at_utc"] = _timestamp_utc()
    _write_state(state_path, state_payload)
    summary_payload = _build_summary_payload(manifest, state_payload)
    _write_json(summary_path, summary_payload)
    _append_log(
        log_path,
        {
            "record_type": "dag_invocation_finished",
            "status": final_status,
            "summary_path": str(summary_path),
        },
    )
    return ExperimentDagReport(
        dag_name=str(state_payload["dag_name"]),
        run_name=str(state_payload["run_name"]),
        run_root=run_root,
        manifest_path=manifest_path,
        resolved_manifest_path=resolved_manifest_path,
        state_path=state_path,
        summary_path=summary_path,
        log_path=log_path,
        status=final_status,
        stage_reports=[dict(state_payload["stages"][stage_key]) for stage_key in stage_order],
        invocation_count=int(state_payload["invocation_count"]),
    )


def _resolve_manifest_input(
    manifest: ExperimentDagManifest | str | Path,
) -> tuple[ExperimentDagManifest, Path | None]:
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).expanduser().resolve()
        return load_experiment_dag_manifest(manifest_path), manifest_path
    return manifest, None


def _initialize_run_state(
    manifest: ExperimentDagManifest,
    *,
    run_name: str,
    run_root: Path,
    manifest_path: Path | None,
    resolved_manifest_path: Path,
) -> dict[str, object]:
    stages_state = {stage.stage_id: _fresh_stage_state(stage) for stage in manifest.stages}
    return {
        "schema_version": EXPERIMENT_DAG_SCHEMA_VERSION,
        "dag_name": manifest.dag_name,
        "run_name": run_name,
        "run_root": str(run_root),
        "output_root": str(manifest.output_root),
        "lock_root": str(manifest.lock_root),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "resolved_manifest_path": str(resolved_manifest_path),
        "started_at_utc": _timestamp_utc(),
        "last_started_at_utc": None,
        "finished_at_utc": None,
        "status": "pending",
        "invocation_count": 1,
        "stage_order": topologically_sort_dag(manifest),
        "stages": stages_state,
    }


def _build_resume_state(
    manifest: ExperimentDagManifest,
    *,
    prior_state: dict[str, object],
    retry_stage_ids: set[str],
) -> dict[str, object]:
    resumed_state = deepcopy(prior_state)
    descendants = _collect_descendants(manifest, retry_stage_ids)
    invalidated_stage_ids = set(retry_stage_ids) | descendants
    if not retry_stage_ids:
        invalidated_stage_ids = {
            stage.stage_id
            for stage in manifest.stages
            if str(dict(prior_state["stages"]).get(stage.stage_id, {}).get("status")) != "succeeded"
        }
    for stage in manifest.stages:
        prior_stage_state = dict(prior_state["stages"].get(stage.stage_id, {}))
        prior_attempts = list(prior_stage_state.get("attempts", []))
        if stage.stage_id not in invalidated_stage_ids and prior_stage_state.get("status") == "succeeded":
            reused_stage_state = deepcopy(prior_stage_state)
            reused_stage_state["reused_from_previous_run"] = True
            resumed_state["stages"][stage.stage_id] = reused_stage_state
            continue
        if retry_stage_ids and stage.stage_id not in invalidated_stage_ids and prior_stage_state.get("status") in {"failed", "blocked"}:
            preserved_failure = deepcopy(prior_stage_state)
            preserved_failure["reused_from_previous_run"] = False
            resumed_state["stages"][stage.stage_id] = preserved_failure
            continue
        refreshed_stage_state = _fresh_stage_state(stage)
        refreshed_stage_state["attempts"] = prior_attempts
        refreshed_stage_state["attempt_count"] = len(prior_attempts)
        refreshed_stage_state["previous_status"] = prior_stage_state.get("status")
        resumed_state["stages"][stage.stage_id] = refreshed_stage_state
    resumed_state["status"] = "pending"
    resumed_state["finished_at_utc"] = None
    resumed_state["invocation_count"] = int(prior_state.get("invocation_count", 1)) + 1
    resumed_state["stage_order"] = topologically_sort_dag(manifest)
    return resumed_state


def _collect_descendants(manifest: ExperimentDagManifest, roots: set[str]) -> set[str]:
    if not roots:
        return set()
    downstream: dict[str, set[str]] = {stage.stage_id: set() for stage in manifest.stages}
    for stage in manifest.stages:
        for dependency in stage.depends_on:
            downstream[dependency].add(stage.stage_id)
    pending = list(roots)
    descendants: set[str] = set()
    while pending:
        stage_id = pending.pop()
        for child in sorted(downstream.get(stage_id, set())):
            if child in descendants:
                continue
            descendants.add(child)
            pending.append(child)
    return descendants


def _fresh_stage_state(stage: ExperimentDagStageSpec) -> dict[str, object]:
    return {
        "stage_id": stage.stage_id,
        "kind": stage.kind,
        "description": stage.description,
        "depends_on": list(stage.depends_on),
        "resources": list(stage.resources),
        "status": "pending",
        "reused_from_previous_run": False,
        "attempt_count": 0,
        "attempts": [],
        "started_at_utc": None,
        "finished_at_utc": None,
        "failure_kind": None,
        "last_error": None,
        "blocked_by": [],
        "resolved_inputs": {},
        "artifact_root": None,
        "summary_path": None,
        "metrics": {},
        "outputs": {},
        "metadata": {},
    }


def _resolve_stage_inputs(
    manifest: ExperimentDagManifest,
    *,
    stage_id: str,
    state_payload: dict[str, object],
) -> dict[str, object]:
    stage = next(candidate for candidate in manifest.stages if candidate.stage_id == stage_id)
    context = {
        "manifest_dir": str(Path(str(state_payload["resolved_manifest_path"])).expanduser().resolve().parent),
        "run": {
            "dag_name": str(state_payload["dag_name"]),
            "run_name": str(state_payload["run_name"]),
            "run_root": str(state_payload["run_root"]),
        },
        "stages": {
            candidate_id: {
                "status": state["status"],
                "artifact_root": state.get("artifact_root"),
                "summary_path": state.get("summary_path"),
                "metrics": state.get("metrics", {}),
                "outputs": state.get("outputs", {}),
            }
            for candidate_id, state in dict(state_payload["stages"]).items()
        },
    }
    return _resolve_value(stage.model_dump(mode="python"), context=context)


def _resolve_value(value: Any, *, context: dict[str, object]) -> Any:
    if isinstance(value, Path):
        return _resolve_string(str(value), context=context)
    if isinstance(value, list):
        resolved_items: list[object] = []
        for item in value:
            resolved = _resolve_value(item, context=context)
            if isinstance(resolved, list):
                resolved_items.extend(resolved)
            else:
                resolved_items.append(resolved)
        return resolved_items
    if isinstance(value, dict):
        return {key: _resolve_value(item, context=context) for key, item in value.items()}
    if isinstance(value, str):
        return _resolve_string(value, context=context)
    return value


def _resolve_string(value: str, *, context: dict[str, object]) -> Any:
    match = _TEMPLATE_PATTERN.fullmatch(value.strip())
    if match is not None:
        return deepcopy(_lookup_context_value(context, match.group(1)))

    def replace(match_object: re.Match[str]) -> str:
        resolved = _lookup_context_value(context, match_object.group(1))
        if isinstance(resolved, (dict, list)):
            raise InterpolationError(
                f"Template '{match_object.group(0)}' resolved to a non-scalar value inside a string context."
            )
        return str(resolved)

    return _TEMPLATE_PATTERN.sub(replace, value)


def _lookup_context_value(context: dict[str, object], expression: str) -> Any:
    current: Any = context
    for segment in expression.split("."):
        if isinstance(current, dict) and segment in current:
            current = current[segment]
            continue
        raise InterpolationError(f"Unknown DAG interpolation path: {expression}")
    return current


def _default_stage_executors() -> dict[str, StageExecutor]:
    return {
        "runtime_job": _execute_runtime_job_stage,
        "dataset_build": _execute_dataset_build_stage,
        "predict_dataset_extract": _execute_predict_dataset_extract_stage,
        "predict_train": _execute_predict_train_stage,
        "predict_report_calibration": _execute_predict_report_calibration_stage,
        "predict_report_ranking": _execute_predict_report_ranking_stage,
        "predict_report_compare": _execute_predict_report_compare_stage,
        "behavior_cloning_train": _execute_behavior_cloning_stage,
        "offline_cql_train": _execute_offline_cql_stage,
        "registry_register": _execute_registry_register_stage,
        "registry_promote": _execute_registry_promote_stage,
        "registry_leaderboard": _execute_registry_leaderboard_stage,
        "registry_compare": _execute_registry_compare_stage,
    }


def _execute_runtime_job_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    from .job_runner import run_runtime_job

    report = run_runtime_job(
        Path(str(resolved_inputs["manifest_path"])).expanduser().resolve(),
        config=Path(str(resolved_inputs["config_path"])).expanduser().resolve(),
        job_name=None if resolved_inputs.get("job_name") is None else str(resolved_inputs["job_name"]),
        output_root=None
        if resolved_inputs.get("output_root") is None
        else Path(str(resolved_inputs["output_root"])).expanduser().resolve(),
        replace_existing=bool(resolved_inputs.get("replace_existing", False)),
    )
    executions = [item.as_dict() for item in report.execution_reports]
    successful = [item for item in executions if item["status"] == "succeeded"]
    outputs: dict[str, object] = {
        "job_root": str(report.job_root),
        "summary_path": str(report.summary_path),
        "log_path": str(report.log_path),
        "execution_ids": [item["execution_id"] for item in executions],
        "successful_execution_ids": [item["execution_id"] for item in successful],
        "artifact_roots": [item["artifact_root"] for item in successful],
        "summary_paths": [item["summary_path"] for item in successful if item["summary_path"] is not None],
        "log_paths": [item["log_path"] for item in successful if item["log_path"] is not None],
        "combat_outcomes_paths": [
            item["combat_outcomes_path"] for item in successful if item["combat_outcomes_path"] is not None
        ],
    }
    if len(successful) == 1:
        successful_item = successful[0]
        outputs["artifact_root"] = successful_item["artifact_root"]
        outputs["execution_summary_path"] = successful_item["summary_path"]
        outputs["execution_log_path"] = successful_item["log_path"]
        outputs["execution_combat_outcomes_path"] = successful_item["combat_outcomes_path"]
    return DagStageResult(
        artifact_root=report.job_root,
        summary_path=report.summary_path,
        metrics={
            "execution_count": len(executions),
            "successful_execution_count": len(successful),
            "failed_execution_count": sum(1 for item in executions if item["status"] == "failed"),
            "quarantined_execution_count": sum(1 for item in executions if item["status"] == "quarantined"),
        },
        outputs=outputs,
        metadata={"watchdogs": {key: value.as_dict() for key, value in sorted(report.watchdogs.items())}},
    )


def _execute_dataset_build_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    report = build_dataset_from_manifest(
        Path(str(resolved_inputs["manifest_path"])).expanduser().resolve(),
        output_dir=Path(str(resolved_inputs["output_dir"])).expanduser().resolve(),
        replace_existing=bool(resolved_inputs.get("replace_existing", False)),
    )
    return DagStageResult(
        artifact_root=report.output_dir,
        summary_path=report.summary_path,
        metrics={
            "record_count": report.record_count,
            "feature_count": report.feature_count,
            "source_file_count": report.source_file_count,
            "filtered_out_count": report.filtered_out_count,
        },
        outputs={
            "output_dir": str(report.output_dir),
            "manifest_path": str(report.manifest_path),
            "summary_path": str(report.summary_path),
            "records_path": str(report.records_path),
            "split_paths": {key: str(value) for key, value in sorted(report.split_paths.items())},
        },
        metadata={"dataset_kind": report.dataset_kind, "split_counts": report.split_counts},
    )


def _execute_predict_dataset_extract_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    report = extract_predictor_dataset(
        _resolve_path_sequence(resolved_inputs.get("sources")),
        output_dir=Path(str(resolved_inputs["output_dir"])).expanduser().resolve(),
        replace_existing=bool(resolved_inputs.get("replace_existing", False)),
        split_seed=int(resolved_inputs.get("split_seed", 0)),
        train_fraction=float(resolved_inputs.get("train_fraction", 0.8)),
        validation_fraction=float(resolved_inputs.get("validation_fraction", 0.1)),
        test_fraction=float(resolved_inputs.get("test_fraction", 0.1)),
        split_group_by=str(resolved_inputs.get("split_group_by", "session_run")),
    )
    return DagStageResult(
        artifact_root=report.output_dir,
        summary_path=report.summary_path,
        metrics={
            "example_count": report.example_count,
            "feature_count": report.feature_count,
            "source_path_count": len(report.source_paths),
        },
        outputs={
            "output_dir": str(report.output_dir),
            "examples_path": str(report.examples_path),
            "summary_path": str(report.summary_path),
            "manifest_path": str(report.manifest_path),
            "source_paths": [str(path) for path in report.source_paths],
            "combat_outcome_paths": [str(path) for path in report.combat_outcome_paths],
        },
        metadata={
            "split_counts": report.split_counts,
            "outcome_histogram": report.outcome_histogram,
            "character_histogram": report.character_histogram,
        },
    )


def _execute_predict_train_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    stage_spec = PredictTrainDagStageSpec.model_validate(resolved_inputs)
    report = train_combat_outcome_predictor(
        dataset_source=Path(str(stage_spec.dataset_source)).expanduser().resolve(),
        output_root=Path(str(stage_spec.output_root)).expanduser().resolve(),
        session_name=stage_spec.session_name,
        config=stage_spec.config.to_runtime_config(),
    )
    return DagStageResult(
        artifact_root=report.output_dir,
        summary_path=report.summary_path,
        metrics={
            "example_count": report.example_count,
            "feature_count": report.feature_count,
            "best_epoch": report.best_epoch,
        },
        outputs={
            "output_dir": str(report.output_dir),
            "model_path": str(report.model_path),
            "metrics_path": str(report.metrics_path),
            "summary_path": str(report.summary_path),
            "dataset_path": str(report.dataset_path),
            "examples_path": None if report.examples_path is None else str(report.examples_path),
            "train_examples_path": None if report.train_examples_path is None else str(report.train_examples_path),
            "validation_examples_path": None
            if report.validation_examples_path is None
            else str(report.validation_examples_path),
        },
        metadata={"split_strategy": report.split_strategy},
    )


def _execute_predict_report_calibration_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    stage_spec = PredictReportCalibrationDagStageSpec.model_validate(resolved_inputs)
    report = build_predictor_calibration_report(
        model_path=Path(str(stage_spec.model_path)).expanduser().resolve(),
        dataset_source=Path(str(stage_spec.dataset_source)).expanduser().resolve(),
        output_root=Path(str(stage_spec.output_root)).expanduser().resolve(),
        session_name=stage_spec.session_name,
        split=stage_spec.split,
        bin_count=stage_spec.bin_count,
        min_slice_examples=stage_spec.min_slice_examples,
        thresholds=stage_spec.thresholds.to_runtime_config(),
    )
    return _predict_report_result(report)


def _execute_predict_report_ranking_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    stage_spec = PredictReportRankingDagStageSpec.model_validate(resolved_inputs)
    report = build_predictor_ranking_report(
        model_path=Path(str(stage_spec.model_path)).expanduser().resolve(),
        dataset_source=Path(str(stage_spec.dataset_source)).expanduser().resolve(),
        output_root=Path(str(stage_spec.output_root)).expanduser().resolve(),
        session_name=stage_spec.session_name,
        split=stage_spec.split,
        group_by=stage_spec.group_by or None,
        top_k=stage_spec.top_k,
        min_group_size=stage_spec.min_group_size,
        thresholds=stage_spec.thresholds.to_runtime_config(),
    )
    return _predict_report_result(report)


def _execute_predict_report_compare_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    stage_spec = PredictReportCompareDagStageSpec.model_validate(resolved_inputs)
    report = build_predictor_benchmark_comparison_report(
        sources=_resolve_path_sequence(stage_spec.sources),
        output_root=Path(str(stage_spec.output_root)).expanduser().resolve(),
        session_name=stage_spec.session_name,
        thresholds=stage_spec.thresholds.to_runtime_config(),
    )
    return _predict_report_result(report)


def _predict_report_result(report: Any) -> DagStageResult:
    return DagStageResult(
        artifact_root=report.output_dir,
        summary_path=report.summary_path,
        outputs={
            "output_dir": str(report.output_dir),
            "summary_path": str(report.summary_path),
            "markdown_path": str(report.markdown_path),
        },
    )


def _execute_behavior_cloning_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    from sts2_rl.train.behavior_cloning import train_behavior_cloning_policy

    stage_spec = BehaviorCloningTrainDagStageSpec.model_validate(resolved_inputs)
    report = train_behavior_cloning_policy(
        dataset_source=Path(str(stage_spec.dataset_source)).expanduser().resolve(),
        output_root=Path(str(stage_spec.output_root)).expanduser().resolve(),
        session_name=stage_spec.session_name,
        config=stage_spec.config.to_runtime_config(),
        benchmark_suite_name=stage_spec.benchmark_suite_name,
    )
    return DagStageResult(
        artifact_root=report.output_dir,
        summary_path=report.summary_path,
        metrics={
            "example_count": report.example_count,
            "feature_count": report.feature_count,
            "stage_count": report.stage_count,
            "best_epoch": report.best_epoch,
        },
        outputs={
            "output_dir": str(report.output_dir),
            "checkpoint_path": str(report.checkpoint_path),
            "best_checkpoint_path": str(report.best_checkpoint_path),
            "metrics_path": str(report.metrics_path),
            "summary_path": str(report.summary_path),
            "dataset_path": str(report.dataset_path),
            "live_eval_summary_path": None
            if report.live_eval_summary_path is None
            else str(report.live_eval_summary_path),
            "benchmark_summary_path": None
            if report.benchmark_summary_path is None
            else str(report.benchmark_summary_path),
        },
        metadata={"split_strategy": report.split_strategy},
    )


def _execute_offline_cql_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    from sts2_rl.train.offline_cql import train_offline_cql_policy

    stage_spec = OfflineCqlTrainDagStageSpec.model_validate(resolved_inputs)
    report = train_offline_cql_policy(
        dataset_source=Path(str(stage_spec.dataset_source)).expanduser().resolve(),
        output_root=Path(str(stage_spec.output_root)).expanduser().resolve(),
        session_name=stage_spec.session_name,
        config=stage_spec.config.to_runtime_config(),
        benchmark_suite_name=stage_spec.benchmark_suite_name,
    )
    return DagStageResult(
        artifact_root=report.output_dir,
        summary_path=report.summary_path,
        metrics={
            "example_count": report.example_count,
            "feature_count": report.feature_count,
            "action_count": report.action_count,
            "best_epoch": report.best_epoch,
        },
        outputs={
            "output_dir": str(report.output_dir),
            "checkpoint_path": str(report.checkpoint_path),
            "best_checkpoint_path": str(report.best_checkpoint_path),
            "warmstart_checkpoint_path": str(report.warmstart_checkpoint_path),
            "metrics_path": str(report.metrics_path),
            "summary_path": str(report.summary_path),
            "dataset_path": str(report.dataset_path),
            "live_eval_summary_path": None
            if report.live_eval_summary_path is None
            else str(report.live_eval_summary_path),
            "benchmark_summary_path": None
            if report.benchmark_summary_path is None
            else str(report.benchmark_summary_path),
        },
        metadata={"split_strategy": report.split_strategy},
    )


def _execute_registry_register_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    report = register_experiment(
        Path(str(resolved_inputs["registry_root"])).expanduser().resolve(),
        source=Path(str(resolved_inputs["source"])).expanduser().resolve(),
        experiment_id=None if resolved_inputs.get("experiment_id") is None else str(resolved_inputs["experiment_id"]),
        tags=[str(item) for item in resolved_inputs.get("tags", [])],
        notes=None if resolved_inputs.get("notes") is None else str(resolved_inputs["notes"]),
        aliases=[str(item) for item in resolved_inputs.get("aliases", [])],
        replace_existing=bool(resolved_inputs.get("replace_existing", False)),
    )
    return DagStageResult(
        artifact_root=report.root_dir,
        summary_path=report.entry_path,
        metrics={"primary_metric_value": report.primary_metric_value},
        outputs={
            "registry_root": str(report.root_dir),
            "experiment_id": report.experiment_id,
            "family": report.family,
            "artifact_kind": report.artifact_kind,
            "display_name": report.display_name,
            "entry_path": str(report.entry_path),
            "source_summary_path": str(report.source_summary_path),
            "primary_metric_name": report.primary_metric_name,
            "primary_metric_value": report.primary_metric_value,
        },
    )


def _execute_registry_promote_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    stage_spec = RegistryPromoteDagStageSpec.model_validate(resolved_inputs)
    registry_root = Path(str(stage_spec.registry_root)).expanduser().resolve()
    if stage_spec.experiment is not None:
        selected = get_registry_experiment(registry_root, stage_spec.experiment)
    else:
        leaderboard = build_registry_leaderboard(
            registry_root,
            family=stage_spec.family,
            tag=stage_spec.tag,
            benchmark_suite_name=stage_spec.benchmark_suite_name,
        )
        payload = json.loads(leaderboard.summary_path.read_text(encoding="utf-8"))
        if not payload["rows"]:
            raise ValueError("Registry promote stage could not find a leaderboard candidate.")
        selected = get_registry_experiment(registry_root, str(payload["rows"][0]["experiment_id"]))
    report = set_registry_alias(
        registry_root,
        alias_name=stage_spec.alias_name,
        experiment_id=str(selected["experiment_id"]),
        artifact_path_key=stage_spec.artifact_path_key,
        reason=stage_spec.reason or "promote",
        updated_by="dag",
    )
    return DagStageResult(
        artifact_root=report.root_dir,
        summary_path=report.aliases_path,
        outputs={
            "registry_root": str(report.root_dir),
            "alias_name": report.alias_name,
            "experiment_id": report.experiment_id,
            "artifact_path_key": report.artifact_path_key,
            "artifact_path": None if report.artifact_path is None else str(report.artifact_path),
            "aliases_path": str(report.aliases_path),
            "alias_history_path": str(report.alias_history_path),
        },
    )


def _execute_registry_leaderboard_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    stage_spec = RegistryLeaderboardDagStageSpec.model_validate(resolved_inputs)
    report = build_registry_leaderboard(
        Path(str(stage_spec.registry_root)).expanduser().resolve(),
        output_root=None if stage_spec.output_root is None else Path(str(stage_spec.output_root)).expanduser().resolve(),
        session_name=stage_spec.session_name,
        family=stage_spec.family,
        tag=stage_spec.tag,
        benchmark_suite_name=stage_spec.benchmark_suite_name,
    )
    return DagStageResult(
        artifact_root=report.output_dir,
        summary_path=report.summary_path,
        metrics={"row_count": report.row_count},
        outputs={
            "output_dir": str(report.output_dir),
            "summary_path": str(report.summary_path),
            "markdown_path": str(report.markdown_path),
        },
    )


def _execute_registry_compare_stage(
    stage: ExperimentDagStageSpec,
    resolved_inputs: dict[str, object],
    _context: StageExecutionContext,
) -> DagStageResult:
    stage_spec = RegistryCompareDagStageSpec.model_validate(resolved_inputs)
    report = compare_registry_experiments(
        Path(str(stage_spec.registry_root)).expanduser().resolve(),
        experiment_ids=stage_spec.experiments,
        output_root=None if stage_spec.output_root is None else Path(str(stage_spec.output_root)).expanduser().resolve(),
        session_name=stage_spec.session_name,
    )
    return DagStageResult(
        artifact_root=report.output_dir,
        summary_path=report.summary_path,
        metrics={"compared_count": report.compared_count},
        outputs={
            "output_dir": str(report.output_dir),
            "summary_path": str(report.summary_path),
            "markdown_path": str(report.markdown_path),
        },
    )


def _resolve_path_sequence(value: Any) -> list[Path]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    return [Path(str(item)).expanduser().resolve() for item in items]


def _resolve_stage_resources(stage: ExperimentDagStageSpec, resolved_inputs: dict[str, object]) -> list[str]:
    resources = [str(item) for item in resolved_inputs.get("resources", stage.resources)]
    if isinstance(stage, RuntimeJobDagStageSpec):
        config_path = Path(str(resolved_inputs["config_path"])).expanduser().resolve()
        resources.append(f"runtime-config:{config_path}")
        try:
            config = load_instance_config(config_path)
            known_ids = {spec.instance_id for spec in build_instance_specs(config)}
            manifest = load_runtime_job_manifest(
                Path(str(resolved_inputs["manifest_path"])).expanduser().resolve(),
                known_instance_ids=known_ids,
            )
            requested_ids = sorted(
                {
                    instance_id
                    for task in manifest.tasks
                    for instance_id in (task.instance_ids or sorted(known_ids))
                }
            )
            resources.extend(f"instance:{config_path}:{instance_id}" for instance_id in requested_ids)
            output_root = manifest.output_root if resolved_inputs.get("output_root") is None else Path(
                str(resolved_inputs["output_root"])
            ).expanduser().resolve()
            job_name = manifest.job_name if resolved_inputs.get("job_name") is None else str(resolved_inputs["job_name"])
            resources.append(f"output:{output_root / job_name}")
        except Exception:
            pass
    elif isinstance(stage, DatasetBuildDagStageSpec):
        resources.append(f"output:{Path(str(resolved_inputs['output_dir'])).expanduser().resolve()}")
    elif isinstance(stage, PredictDatasetExtractDagStageSpec):
        resources.append(f"output:{Path(str(resolved_inputs['output_dir'])).expanduser().resolve()}")
    elif isinstance(
        stage,
        (
            PredictTrainDagStageSpec,
            PredictReportCalibrationDagStageSpec,
            PredictReportRankingDagStageSpec,
            PredictReportCompareDagStageSpec,
            BehaviorCloningTrainDagStageSpec,
            OfflineCqlTrainDagStageSpec,
        ),
    ):
        resources.append(_session_output_resource(resolved_inputs))
    elif isinstance(
        stage,
        (
            RegistryRegisterDagStageSpec,
            RegistryPromoteDagStageSpec,
            RegistryLeaderboardDagStageSpec,
            RegistryCompareDagStageSpec,
        ),
    ):
        registry_root = Path(str(resolved_inputs["registry_root"])).expanduser().resolve()
        resources.append(f"registry:{registry_root}")
    return sorted(dict.fromkeys(resources))


def _session_output_resource(resolved_inputs: dict[str, object]) -> str:
    output_root = Path(str(resolved_inputs["output_root"])).expanduser().resolve()
    session_name = resolved_inputs.get("session_name")
    if session_name is None:
        return f"output-root:{output_root}"
    return f"output:{output_root / str(session_name)}"


class _ResourceLease:
    def __init__(
        self,
        *,
        lock_root: Path,
        run_name: str,
        stage_id: str,
        attempt_index: int,
        resources: Sequence[str],
    ) -> None:
        self._lock_root = lock_root
        self._run_name = run_name
        self._stage_id = stage_id
        self._attempt_index = attempt_index
        self._resources = sorted(dict.fromkeys(resources))
        self._acquired_paths: list[Path] = []

    def __enter__(self) -> _ResourceLease:
        self._lock_root.mkdir(parents=True, exist_ok=True)
        for resource in self._resources:
            resource_path = self._lock_root / f"{_stable_resource_name(resource)}.lock"
            try:
                fd = os.open(resource_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                existing_payload = _load_json_if_exists(resource_path)
                for acquired_path in reversed(self._acquired_paths):
                    if acquired_path.exists():
                        acquired_path.unlink()
                owner = None if existing_payload is None else existing_payload.get("owner")
                raise ResourceLockError(
                    f"Resource '{resource}' is already locked by {owner or resource_path}."
                ) from exc
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    {
                        "resource": resource,
                        "owner": {
                            "run_name": self._run_name,
                            "stage_id": self._stage_id,
                            "attempt_index": self._attempt_index,
                            "pid": os.getpid(),
                            "timestamp_utc": _timestamp_utc(),
                        },
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
            self._acquired_paths.append(resource_path)
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        for resource_path in reversed(self._acquired_paths):
            if resource_path.exists():
                resource_path.unlink()
        self._acquired_paths.clear()


def _classify_stage_exception(exc: Exception) -> tuple[str, bool]:
    if isinstance(exc, ResourceLockError):
        return "resource_lock_conflict", True
    if isinstance(exc, InterpolationError):
        return "invalid_interpolation", False
    if isinstance(exc, FileNotFoundError):
        return "artifact_missing", False
    if isinstance(exc, FileExistsError):
        return "output_conflict", False
    if isinstance(exc, ValueError):
        return "invalid_stage_configuration", False
    return "stage_failed", True


def _final_dag_status(state_payload: dict[str, object]) -> str:
    stage_statuses = [str(stage_state["status"]) for stage_state in dict(state_payload["stages"]).values()]
    if all(status == "succeeded" for status in stage_statuses):
        return "succeeded"
    if any(status == "failed" for status in stage_statuses):
        return "failed"
    if any(status == "blocked" for status in stage_statuses):
        return "failed"
    if any(status == "running" for status in stage_statuses):
        return "running"
    return "pending"


def _build_summary_payload(manifest: ExperimentDagManifest, state_payload: dict[str, object]) -> dict[str, object]:
    stage_order = topologically_sort_dag(manifest)
    stage_reports = [dict(state_payload["stages"][stage_id]) for stage_id in stage_order]
    status_histogram: dict[str, int] = {}
    for report in stage_reports:
        status_histogram[report["status"]] = status_histogram.get(report["status"], 0) + 1
    return {
        "schema_version": EXPERIMENT_DAG_SCHEMA_VERSION,
        "dag_name": state_payload["dag_name"],
        "run_name": state_payload["run_name"],
        "run_root": state_payload["run_root"],
        "manifest_path": state_payload["manifest_path"],
        "resolved_manifest_path": state_payload["resolved_manifest_path"],
        "lock_root": state_payload["lock_root"],
        "status": state_payload["status"],
        "started_at_utc": state_payload["started_at_utc"],
        "finished_at_utc": state_payload["finished_at_utc"],
        "invocation_count": state_payload["invocation_count"],
        "stage_count": len(stage_reports),
        "stage_status_histogram": status_histogram,
        "reused_stage_count": sum(1 for report in stage_reports if report.get("reused_from_previous_run")),
        "critical_path_seconds": _critical_path_seconds(manifest, stage_reports),
        "stages": _jsonify(stage_reports),
    }


def _critical_path_seconds(manifest: ExperimentDagManifest, stage_reports: list[dict[str, object]]) -> float:
    report_by_id = {report["stage_id"]: report for report in stage_reports}
    longest_path: dict[str, float] = {}
    for stage_id in topologically_sort_dag(manifest):
        report = report_by_id[stage_id]
        duration = _stage_duration_seconds(report)
        dependencies = list(report.get("depends_on", []))
        longest_path[stage_id] = duration + max((longest_path[dependency] for dependency in dependencies), default=0.0)
    return max(longest_path.values(), default=0.0)


def _stage_duration_seconds(report: dict[str, object]) -> float:
    started = report.get("started_at_utc")
    finished = report.get("finished_at_utc")
    if started is None or finished is None:
        return 0.0
    try:
        started_dt = datetime.fromisoformat(str(started))
        finished_dt = datetime.fromisoformat(str(finished))
    except ValueError:
        return 0.0
    return max(0.0, (finished_dt - started_dt).total_seconds())


def _write_state(path: Path, payload: dict[str, object]) -> None:
    _write_json(path, payload)


def _append_log(path: Path, payload: dict[str, object]) -> None:
    record = {"timestamp_utc": _timestamp_utc(), **_jsonify(payload)}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_resource_name(resource: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", resource).strip("-")
    return safe or "resource"


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    return value


def _timestamp_utc() -> str:
    return datetime.now(UTC).isoformat()
