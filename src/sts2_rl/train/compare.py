from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from .divergence import (
    DivergenceDiagnosticReport,
    build_iteration_runtime_metadata,
    diagnose_iteration_divergence,
    load_step_trace,
)
from .replay import _default_env_factory, _prepare_replay_start, _resolve_prepare_target
from .runner import CombatEvaluationReport, run_combat_dqn_evaluation


@dataclass(frozen=True)
class CombatCheckpointEvalIterationReport:
    checkpoint_label: str
    checkpoint_path: Path
    iteration_index: int
    session_name: str
    session_dir: Path
    summary_path: Path
    log_path: Path
    combat_outcomes_path: Path
    prepare_target: str
    normalization_report: dict[str, Any]
    start_screen: str
    start_signature: str
    start_payload: dict[str, Any]
    runtime_metadata: dict[str, Any]
    runtime_fingerprint: str
    step_trace_fingerprint: str
    prepare_action_ids: list[str]
    env_steps: int
    combat_steps: int
    heuristic_steps: int
    total_reward: float
    stop_reason: str
    final_screen: str
    completed_run_count: int
    completed_combat_count: int
    combat_performance: dict[str, object]
    observed_run_seeds: list[str] = field(default_factory=list)
    observed_run_seed_histogram: dict[str, int] = field(default_factory=dict)
    runs_without_observed_seed: int = 0
    last_observed_seed: str | None = None


@dataclass(frozen=True)
class CombatCheckpointComparisonReport:
    base_url: str
    comparison_dir: Path
    summary_path: Path
    iterations_path: Path
    log_path: Path
    baseline_checkpoint_path: Path
    candidate_checkpoint_path: Path
    repeat_count: int
    prepare_target: str
    better_checkpoint_label: str | None
    delta_metrics: dict[str, object]
    baseline: dict[str, object]
    candidate: dict[str, object]
    iterations: list[CombatCheckpointEvalIterationReport]
    diagnostics_path: Path
    paired_diagnostics: list[DivergenceDiagnosticReport]


def run_combat_dqn_checkpoint_comparison(
    *,
    base_url: str,
    baseline_checkpoint_path: str | Path,
    candidate_checkpoint_path: str | Path,
    output_root: str | Path,
    comparison_name: str | None = None,
    repeats: int = 3,
    max_env_steps: int | None = 0,
    max_runs: int | None = 1,
    max_combats: int | None = 0,
    poll_interval_seconds: float = 0.25,
    max_idle_polls: int = 40,
    request_timeout_seconds: float = 30.0,
    prepare_main_menu: bool = True,
    prepare_target: str | None = None,
    prepare_max_steps: int = 8,
    prepare_max_idle_polls: int = 40,
    env_factory: Callable = _default_env_factory,
    evaluation_fn: Callable[..., CombatEvaluationReport] = run_combat_dqn_evaluation,
) -> CombatCheckpointComparisonReport:
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    baseline_checkpoint_path = Path(baseline_checkpoint_path)
    candidate_checkpoint_path = Path(candidate_checkpoint_path)
    prepare_target = _resolve_prepare_target(prepare_main_menu=prepare_main_menu, prepare_target=prepare_target)
    comparison_dir = Path(output_root) / (comparison_name or default_checkpoint_comparison_name())
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_path = comparison_dir / "comparison-summary.json"
    iterations_path = comparison_dir / "comparison-iterations.jsonl"
    log_path = comparison_dir / "comparison-log.jsonl"
    diagnostics_path = comparison_dir / "comparison-diagnostics.jsonl"

    _append_log(
        log_path,
        {
            "record_type": "comparison_started",
            "base_url": base_url,
            "baseline_checkpoint_path": str(baseline_checkpoint_path),
            "candidate_checkpoint_path": str(candidate_checkpoint_path),
            "repeats": repeats,
            "prepare_target": prepare_target,
        },
    )

    iterations: list[CombatCheckpointEvalIterationReport] = []
    for checkpoint_label, checkpoint_path in (
        ("baseline", baseline_checkpoint_path),
        ("candidate", candidate_checkpoint_path),
    ):
        for iteration_index in range(1, repeats + 1):
            prepare_report = _prepare_replay_start(
                base_url=base_url,
                request_timeout_seconds=request_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                max_idle_polls=prepare_max_idle_polls,
                max_prepare_steps=prepare_max_steps,
                prepare_target=prepare_target,
                env_factory=env_factory,
            )
            _append_log(
                log_path,
                {
                    "record_type": "iteration_prepared",
                    "checkpoint_label": checkpoint_label,
                    "iteration_index": iteration_index,
                    "prepare_target": prepare_report["prepare_target"],
                    "start_screen": prepare_report["start_screen"],
                    "start_signature": prepare_report["start_signature"],
                    "prepare_action_ids": prepare_report["prepare_action_ids"],
                    "normalization_report": prepare_report["normalization_report"],
                },
            )
            session_name = f"{checkpoint_label}-iteration-{iteration_index:03d}"
            report = evaluation_fn(
                base_url=base_url,
                checkpoint_path=checkpoint_path,
                output_root=comparison_dir,
                session_name=session_name,
                max_env_steps=max_env_steps,
                max_runs=max_runs,
                max_combats=max_combats,
                poll_interval_seconds=poll_interval_seconds,
                max_idle_polls=max_idle_polls,
                request_timeout_seconds=request_timeout_seconds,
                env_factory=env_factory,
            )
            summary_payload = json.loads(report.summary_path.read_text(encoding="utf-8"))
            observed_seed_metadata = _observed_seed_metadata(summary_payload)
            runtime_metadata, runtime_fingerprint = build_iteration_runtime_metadata(
                base_url=base_url,
                prepare_target=str(prepare_report["prepare_target"]),
                summary_payload=summary_payload,
                checkpoint_path=checkpoint_path,
                checkpoint_label=checkpoint_label,
            )
            step_trace = load_step_trace(report.log_path)
            iteration = CombatCheckpointEvalIterationReport(
                checkpoint_label=checkpoint_label,
                checkpoint_path=checkpoint_path,
                iteration_index=iteration_index,
                session_name=session_name,
                session_dir=comparison_dir / session_name,
                summary_path=report.summary_path,
                log_path=report.log_path,
                combat_outcomes_path=report.combat_outcomes_path,
                prepare_target=str(prepare_report["prepare_target"]),
                normalization_report=dict(prepare_report["normalization_report"]),
                start_screen=str(prepare_report["start_screen"]),
                start_signature=str(prepare_report["start_signature"]),
                start_payload=dict(prepare_report["start_payload"]),
                runtime_metadata=runtime_metadata,
                runtime_fingerprint=runtime_fingerprint,
                step_trace_fingerprint=str(step_trace["trace_fingerprint"]),
                prepare_action_ids=list(prepare_report["prepare_action_ids"]),
                env_steps=report.env_steps,
                combat_steps=report.combat_steps,
                heuristic_steps=report.heuristic_steps,
                total_reward=report.total_reward,
                stop_reason=report.stop_reason,
                final_screen=report.final_screen,
                completed_run_count=report.completed_run_count,
                completed_combat_count=report.completed_combat_count,
                combat_performance=dict(report.combat_performance),
                observed_run_seeds=list(observed_seed_metadata["observed_run_seeds"]),
                observed_run_seed_histogram=dict(observed_seed_metadata["observed_run_seed_histogram"]),
                runs_without_observed_seed=int(observed_seed_metadata["runs_without_observed_seed"]),
                last_observed_seed=_as_optional_str(observed_seed_metadata["last_observed_seed"]),
            )
            iterations.append(iteration)
            with iterations_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(_iteration_payload(iteration), ensure_ascii=False))
                handle.write("\n")
            _append_log(
                log_path,
                {
                    "record_type": "iteration_finished",
                    "checkpoint_label": checkpoint_label,
                    "iteration_index": iteration_index,
                    "session_name": session_name,
                    "prepare_target": iteration.prepare_target,
                    "summary_path": str(report.summary_path),
                    "total_reward": report.total_reward,
                    "combat_performance": report.combat_performance,
                    "normalization_stop_reason": iteration.normalization_report.get("stop_reason"),
                },
            )

    baseline_iterations = [item for item in iterations if item.checkpoint_label == "baseline"]
    candidate_iterations = [item for item in iterations if item.checkpoint_label == "candidate"]
    baseline = _aggregate_iterations("baseline", baseline_checkpoint_path, baseline_iterations)
    candidate = _aggregate_iterations("candidate", candidate_checkpoint_path, candidate_iterations)
    delta_metrics = _delta_metrics(baseline, candidate)
    better_checkpoint_label = _pick_better_checkpoint(baseline, candidate)
    paired_diagnostics = _paired_iteration_diagnostics(
        baseline_iterations=baseline_iterations,
        candidate_iterations=candidate_iterations,
        diagnostics_path=diagnostics_path,
    )
    divergence_family_histogram = Counter(item.family for item in paired_diagnostics)
    divergence_category_histogram = Counter(item.category for item in paired_diagnostics)

    summary = {
        "base_url": base_url,
        "comparison_dir": str(comparison_dir),
        "baseline_checkpoint_path": str(baseline_checkpoint_path),
        "candidate_checkpoint_path": str(candidate_checkpoint_path),
        "repeat_count": repeats,
        "prepare_target": prepare_target,
        "max_env_steps": max_env_steps,
        "max_runs": max_runs,
        "max_combats": max_combats,
        "better_checkpoint_label": better_checkpoint_label,
        "delta_metrics": delta_metrics,
        "divergence_family_histogram": dict(divergence_family_histogram),
        "divergence_category_histogram": dict(divergence_category_histogram),
        "baseline": baseline,
        "candidate": candidate,
        "summary_path": str(summary_path),
        "iterations_path": str(iterations_path),
        "diagnostics_path": str(diagnostics_path),
        "log_path": str(log_path),
        "paired_diagnostics": [item.as_dict() for item in paired_diagnostics],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_log(
        log_path,
        {
            "record_type": "comparison_finished",
            "better_checkpoint_label": better_checkpoint_label,
            "delta_metrics": delta_metrics,
            "divergence_family_histogram": dict(divergence_family_histogram),
            "divergence_category_histogram": dict(divergence_category_histogram),
            "summary_path": str(summary_path),
        },
    )

    return CombatCheckpointComparisonReport(
        base_url=base_url,
        comparison_dir=comparison_dir,
        summary_path=summary_path,
        iterations_path=iterations_path,
        log_path=log_path,
        baseline_checkpoint_path=baseline_checkpoint_path,
        candidate_checkpoint_path=candidate_checkpoint_path,
        repeat_count=repeats,
        prepare_target=prepare_target,
        better_checkpoint_label=better_checkpoint_label,
        delta_metrics=delta_metrics,
        baseline=baseline,
        candidate=candidate,
        iterations=iterations,
        diagnostics_path=diagnostics_path,
        paired_diagnostics=paired_diagnostics,
    )


def default_checkpoint_comparison_name() -> str:
    return datetime.now(UTC).strftime("combat-dqn-compare-%Y%m%d-%H%M%S")


def _aggregate_iterations(
    checkpoint_label: str,
    checkpoint_path: Path,
    iterations: list[CombatCheckpointEvalIterationReport],
) -> dict[str, object]:
    reward_sum = sum(item.total_reward for item in iterations)
    env_steps_sum = sum(item.env_steps for item in iterations)
    combat_steps_sum = sum(item.combat_steps for item in iterations)
    completed_runs_sum = sum(item.completed_run_count for item in iterations)
    completed_combats_sum = sum(item.completed_combat_count for item in iterations)
    won_combats_sum = sum(int(item.combat_performance.get("won_combats", 0) or 0) for item in iterations)
    lost_combats_sum = sum(int(item.combat_performance.get("lost_combats", 0) or 0) for item in iterations)
    stop_reason_histogram = Counter(item.stop_reason for item in iterations)
    final_screen_histogram = Counter(item.final_screen for item in iterations)
    start_signature_histogram = Counter(item.start_signature for item in iterations)
    normalization_stop_reason_histogram = Counter(
        str(item.normalization_report.get("stop_reason", "unknown")) for item in iterations
    )
    return {
        "checkpoint_label": checkpoint_label,
        "checkpoint_path": str(checkpoint_path),
        "prepare_target": iterations[0].prepare_target if iterations else "none",
        "iteration_count": len(iterations),
        "mean_total_reward": reward_sum / len(iterations) if iterations else None,
        "mean_env_steps": env_steps_sum / len(iterations) if iterations else None,
        "mean_combat_steps": combat_steps_sum / len(iterations) if iterations else None,
        "mean_completed_runs": completed_runs_sum / len(iterations) if iterations else None,
        "mean_completed_combats": completed_combats_sum / len(iterations) if iterations else None,
        "total_completed_combats": completed_combats_sum,
        "total_won_combats": won_combats_sum,
        "total_lost_combats": lost_combats_sum,
        "combat_win_rate": (won_combats_sum / completed_combats_sum) if completed_combats_sum else None,
        "reward_per_combat": (reward_sum / completed_combats_sum) if completed_combats_sum else None,
        "reward_per_combat_step": (reward_sum / combat_steps_sum) if combat_steps_sum else None,
        "observed_run_seeds": _merge_seed_lists(item.observed_run_seeds for item in iterations),
        "observed_run_seed_histogram": _merge_histograms(item.observed_run_seed_histogram for item in iterations),
        "runs_without_observed_seed": sum(item.runs_without_observed_seed for item in iterations),
        "last_observed_seed": _last_non_empty_seed(item.last_observed_seed for item in iterations),
        "stop_reason_histogram": dict(stop_reason_histogram),
        "final_screen_histogram": dict(final_screen_histogram),
        "start_signature_histogram": dict(start_signature_histogram),
        "normalization_stop_reason_histogram": dict(normalization_stop_reason_histogram),
        "normalization_strategy_histogram": _merge_histograms(
            item.normalization_report.get("strategy_histogram", {}) for item in iterations
        ),
        "all_normalization_targets_reached": all(
            bool(item.normalization_report.get("reached_target")) for item in iterations
        ),
    }


def _delta_metrics(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    deltas: dict[str, object] = {}
    for key in (
        "mean_total_reward",
        "mean_env_steps",
        "mean_combat_steps",
        "mean_completed_runs",
        "mean_completed_combats",
        "combat_win_rate",
        "reward_per_combat",
        "reward_per_combat_step",
    ):
        baseline_value = baseline.get(key)
        candidate_value = candidate.get(key)
        if baseline_value is None or candidate_value is None:
            continue
        deltas[key] = float(candidate_value) - float(baseline_value)
    return deltas


def _pick_better_checkpoint(baseline: dict[str, object], candidate: dict[str, object]) -> str | None:
    def score(payload: dict[str, object]) -> tuple[float, float, float]:
        return (
            float(payload.get("combat_win_rate") or float("-inf")),
            float(payload.get("reward_per_combat") or float("-inf")),
            float(payload.get("mean_total_reward") or float("-inf")),
        )

    baseline_score = score(baseline)
    candidate_score = score(candidate)
    if candidate_score > baseline_score:
        return "candidate"
    if baseline_score > candidate_score:
        return "baseline"
    return None


def _iteration_payload(iteration: CombatCheckpointEvalIterationReport) -> dict[str, Any]:
    return {
        "checkpoint_label": iteration.checkpoint_label,
        "checkpoint_path": str(iteration.checkpoint_path),
        "iteration_index": iteration.iteration_index,
        "session_name": iteration.session_name,
        "session_dir": str(iteration.session_dir),
        "summary_path": str(iteration.summary_path),
        "log_path": str(iteration.log_path),
        "combat_outcomes_path": str(iteration.combat_outcomes_path),
        "prepare_target": iteration.prepare_target,
        "normalization_report": iteration.normalization_report,
        "start_screen": iteration.start_screen,
        "start_signature": iteration.start_signature,
        "start_payload": iteration.start_payload,
        "runtime_metadata": iteration.runtime_metadata,
        "runtime_fingerprint": iteration.runtime_fingerprint,
        "step_trace_fingerprint": iteration.step_trace_fingerprint,
        "prepare_action_ids": iteration.prepare_action_ids,
        "env_steps": iteration.env_steps,
        "combat_steps": iteration.combat_steps,
        "heuristic_steps": iteration.heuristic_steps,
        "total_reward": iteration.total_reward,
        "stop_reason": iteration.stop_reason,
        "final_screen": iteration.final_screen,
        "completed_run_count": iteration.completed_run_count,
        "completed_combat_count": iteration.completed_combat_count,
        "combat_performance": iteration.combat_performance,
        "observed_run_seeds": iteration.observed_run_seeds,
        "observed_run_seed_histogram": iteration.observed_run_seed_histogram,
        "runs_without_observed_seed": iteration.runs_without_observed_seed,
        "last_observed_seed": iteration.last_observed_seed,
    }


def _paired_iteration_diagnostics(
    *,
    baseline_iterations: list[CombatCheckpointEvalIterationReport],
    candidate_iterations: list[CombatCheckpointEvalIterationReport],
    diagnostics_path: Path,
) -> list[DivergenceDiagnosticReport]:
    diagnostics: list[DivergenceDiagnosticReport] = []
    paired_candidate_by_index = {
        item.iteration_index: item
        for item in candidate_iterations
    }
    for baseline in baseline_iterations:
        candidate = paired_candidate_by_index.get(baseline.iteration_index)
        if candidate is None:
            continue
        baseline_payload = _iteration_payload(baseline)
        candidate_payload = _iteration_payload(candidate)
        baseline_payload["step_trace"] = load_step_trace(baseline.log_path)
        candidate_payload["step_trace"] = load_step_trace(candidate.log_path)
        diagnostic = diagnose_iteration_divergence(
            baseline=baseline_payload,
            candidate=candidate_payload,
        )
        diagnostics.append(diagnostic)
        with diagnostics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "iteration_index": baseline.iteration_index,
                        "baseline_checkpoint_label": baseline.checkpoint_label,
                        "candidate_checkpoint_label": candidate.checkpoint_label,
                        "diagnostic": diagnostic.as_dict(),
                    },
                    ensure_ascii=False,
                )
            )
            handle.write("\n")
    return diagnostics


def _merge_histograms(histograms: Any) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for histogram in histograms:
        if not isinstance(histogram, dict):
            continue
        for key, value in histogram.items():
            merged[str(key)] += int(value)
    return dict(merged)


def _merge_seed_lists(seed_lists: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for seed_list in seed_lists:
        if not isinstance(seed_list, list):
            continue
        for seed in seed_list:
            normalized = _as_optional_str(seed)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return sorted(merged)


def _last_non_empty_seed(values: Any) -> str | None:
    last_seed: str | None = None
    for value in values:
        normalized = _as_optional_str(value)
        if normalized is not None:
            last_seed = normalized
    return last_seed


def _observed_seed_metadata(summary_payload: dict[str, Any]) -> dict[str, Any]:
    histogram = summary_payload.get("observed_run_seed_histogram", {})
    if not isinstance(histogram, dict):
        histogram = {}
    return {
        "observed_run_seeds": _merge_seed_lists([summary_payload.get("observed_run_seeds", [])]),
        "observed_run_seed_histogram": _merge_histograms([histogram]),
        "runs_without_observed_seed": int(summary_payload.get("runs_without_observed_seed", 0) or 0),
        "last_observed_seed": summary_payload.get("last_observed_seed"),
    }


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _append_log(path: Path, payload: dict[str, object]) -> None:
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
