from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from sts2_rl.env.wrapper import Sts2Env
from sts2_rl.runtime import build_start_payload, normalize_runtime_state

from .divergence import (
    DivergenceDiagnosticReport,
    build_iteration_runtime_metadata,
    diagnose_iteration_divergence,
    load_step_trace,
)

from .runner import CombatEvaluationReport, run_combat_dqn_evaluation


@dataclass(frozen=True)
class CombatReplayIterationReport:
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
    action_count: int
    action_sequence: list[str]
    action_id_histogram: dict[str, int]
    action_histogram: dict[str, int]
    run_outcome_histogram: dict[str, int]
    observed_run_seeds: list[str] = field(default_factory=list)
    observed_run_seed_histogram: dict[str, int] = field(default_factory=dict)
    runs_without_observed_seed: int = 0
    last_observed_seed: str | None = None


@dataclass(frozen=True)
class CombatReplayComparisonReport:
    baseline_iteration: int
    candidate_iteration: int
    status: str
    start_signature_match: bool
    stop_reason_match: bool
    final_screen_match: bool
    action_sequence_match: bool
    action_histogram_match: bool
    action_id_histogram_match: bool
    run_outcome_histogram_match: bool
    common_action_prefix_length: int
    first_action_divergence_index: int | None
    baseline_action_count: int
    candidate_action_count: int
    metric_differences: dict[str, dict[str, Any]]
    action_histogram_delta: dict[str, dict[str, int]]
    action_id_histogram_delta: dict[str, dict[str, int]]
    diagnostic: DivergenceDiagnosticReport


@dataclass(frozen=True)
class CombatReplaySuiteReport:
    base_url: str
    checkpoint_path: Path
    suite_dir: Path
    summary_path: Path
    comparisons_path: Path
    log_path: Path
    repeat_count: int
    comparison_count: int
    exact_match_count: int
    divergent_iteration_count: int
    status_histogram: dict[str, int]
    prepare_target: str
    iterations: list[CombatReplayIterationReport]
    comparisons: list[CombatReplayComparisonReport]


class SupportsEnv(Protocol):
    def observe(self): ...

    def step(self, action): ...

    def close(self) -> None: ...


def _default_env_factory(base_url: str, timeout: float) -> SupportsEnv:
    return Sts2Env.from_base_url(base_url, timeout=timeout)


def run_combat_dqn_replay_suite(
    *,
    base_url: str,
    checkpoint_path: str | Path,
    output_root: str | Path,
    suite_name: str | None = None,
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
    env_factory: Callable[[str, float], SupportsEnv] = _default_env_factory,
    evaluation_fn: Callable[..., CombatEvaluationReport] = run_combat_dqn_evaluation,
) -> CombatReplaySuiteReport:
    if repeats < 2:
        raise ValueError("repeats must be at least 2 for replay comparison.")

    checkpoint_path = Path(checkpoint_path)
    prepare_target = _resolve_prepare_target(prepare_main_menu=prepare_main_menu, prepare_target=prepare_target)
    suite_dir = Path(output_root) / (suite_name or default_replay_suite_name())
    suite_dir.mkdir(parents=True, exist_ok=True)
    summary_path = suite_dir / "replay-summary.json"
    comparisons_path = suite_dir / "replay-comparisons.jsonl"
    log_path = suite_dir / "replay-suite.jsonl"

    _append_suite_log(
        log_path,
        {
            "record_type": "replay_suite_started",
            "base_url": base_url,
            "checkpoint_path": str(checkpoint_path),
            "repeats": repeats,
            "max_env_steps": max_env_steps,
            "max_runs": max_runs,
            "max_combats": max_combats,
            "prepare_target": prepare_target,
            "prepare_max_steps": prepare_max_steps,
            "prepare_max_idle_polls": prepare_max_idle_polls,
        },
    )

    iterations: list[CombatReplayIterationReport] = []
    comparisons: list[CombatReplayComparisonReport] = []
    baseline: CombatReplayIterationReport | None = None

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
        _append_suite_log(
            log_path,
            {
                "record_type": "replay_iteration_prepared",
                "iteration_index": iteration_index,
                "prepare_target": prepare_report["prepare_target"],
                "start_screen": prepare_report["start_screen"],
                "start_signature": prepare_report["start_signature"],
                "prepare_action_ids": prepare_report["prepare_action_ids"],
                "normalization_report": prepare_report["normalization_report"],
            },
        )

        session_name = f"iteration-{iteration_index:03d}"
        report = evaluation_fn(
            base_url=base_url,
            checkpoint_path=checkpoint_path,
            output_root=suite_dir,
            session_name=session_name,
            max_env_steps=max_env_steps,
            max_runs=max_runs,
            max_combats=max_combats,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=max_idle_polls,
            request_timeout_seconds=request_timeout_seconds,
            env_factory=env_factory,
        )
        iteration = _build_iteration_report(
            iteration_index=iteration_index,
            session_name=session_name,
            session_dir=suite_dir / session_name,
            report=report,
            prepare_report=prepare_report,
        )
        iterations.append(iteration)
        _append_suite_log(
            log_path,
            {
                "record_type": "replay_iteration_finished",
                "iteration_index": iteration_index,
                "session_name": session_name,
                "stop_reason": iteration.stop_reason,
                "final_screen": iteration.final_screen,
                "action_count": iteration.action_count,
                "prepare_target": iteration.prepare_target,
                "normalization_stop_reason": iteration.normalization_report.get("stop_reason"),
                "summary_path": str(iteration.summary_path),
            },
        )

        if baseline is None:
            baseline = iteration
            continue

        comparison = _compare_iterations(baseline=baseline, candidate=iteration)
        comparisons.append(comparison)
        with comparisons_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(_comparison_payload(comparison), ensure_ascii=False))
            handle.write("\n")
        _append_suite_log(
            log_path,
            {
                "record_type": "replay_iteration_compared",
                "baseline_iteration": baseline.iteration_index,
                "candidate_iteration": iteration.iteration_index,
                "status": comparison.status,
                "first_action_divergence_index": comparison.first_action_divergence_index,
            },
        )

    status_histogram = Counter(comparison.status for comparison in comparisons)
    normalization_stop_reason_histogram = Counter(
        str(item.normalization_report.get("stop_reason", "unknown")) for item in iterations
    )
    exact_match_count = status_histogram.get("exact_match", 0)
    divergence_family_histogram = Counter(comparison.diagnostic.family for comparison in comparisons)
    divergence_category_histogram = Counter(comparison.diagnostic.category for comparison in comparisons)
    suite_summary = {
        "base_url": base_url,
        "checkpoint_path": str(checkpoint_path),
        "suite_dir": str(suite_dir),
        "repeat_count": repeats,
        "comparison_count": len(comparisons),
        "prepare_target": prepare_target,
        "prepare_max_steps": prepare_max_steps,
        "prepare_max_idle_polls": prepare_max_idle_polls,
        "max_env_steps": max_env_steps,
        "max_runs": max_runs,
        "max_combats": max_combats,
        "exact_match_count": exact_match_count,
        "divergent_iteration_count": len(comparisons) - exact_match_count,
        "status_histogram": dict(status_histogram),
        "observed_run_seeds": _merge_seed_lists(item.observed_run_seeds for item in iterations),
        "observed_run_seed_histogram": _merge_histograms(item.observed_run_seed_histogram for item in iterations),
        "runs_without_observed_seed": sum(item.runs_without_observed_seed for item in iterations),
        "last_observed_seed": _last_non_empty_seed(item.last_observed_seed for item in iterations),
        "divergence_family_histogram": dict(divergence_family_histogram),
        "divergence_category_histogram": dict(divergence_category_histogram),
        "normalization_stop_reason_histogram": dict(normalization_stop_reason_histogram),
        "normalization_strategy_histogram": _merge_histograms(
            item.normalization_report.get("strategy_histogram", {}) for item in iterations
        ),
        "all_normalization_targets_reached": all(
            bool(item.normalization_report.get("reached_target")) for item in iterations
        ),
        "all_start_signatures_match": all(item.start_signature_match for item in comparisons) if comparisons else True,
        "all_action_sequences_match": all(item.action_sequence_match for item in comparisons) if comparisons else True,
        "all_action_histograms_match": all(item.action_histogram_match for item in comparisons) if comparisons else True,
        "all_run_outcomes_match": all(item.run_outcome_histogram_match for item in comparisons) if comparisons else True,
        "all_exact_match": all(item.status == "exact_match" for item in comparisons) if comparisons else True,
        "summary_path": str(summary_path),
        "comparisons_path": str(comparisons_path),
        "log_path": str(log_path),
        "iterations": [_iteration_payload(item) for item in iterations],
        "comparisons": [_comparison_payload(item) for item in comparisons],
    }
    summary_path.write_text(json.dumps(suite_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_suite_log(
        log_path,
        {
            "record_type": "replay_suite_finished",
            "comparison_count": len(comparisons),
            "exact_match_count": exact_match_count,
            "divergent_iteration_count": len(comparisons) - exact_match_count,
            "status_histogram": dict(status_histogram),
            "divergence_family_histogram": dict(divergence_family_histogram),
            "divergence_category_histogram": dict(divergence_category_histogram),
            "normalization_stop_reason_histogram": dict(normalization_stop_reason_histogram),
            "summary_path": str(summary_path),
            "comparisons_path": str(comparisons_path),
        },
    )

    return CombatReplaySuiteReport(
        base_url=base_url,
        checkpoint_path=checkpoint_path,
        suite_dir=suite_dir,
        summary_path=summary_path,
        comparisons_path=comparisons_path,
        log_path=log_path,
        repeat_count=repeats,
        comparison_count=len(comparisons),
        exact_match_count=exact_match_count,
        divergent_iteration_count=len(comparisons) - exact_match_count,
        status_histogram=dict(status_histogram),
        prepare_target=prepare_target,
        iterations=iterations,
        comparisons=comparisons,
    )


def default_replay_suite_name() -> str:
    return datetime.now(UTC).strftime("combat-dqn-replay-%Y%m%d-%H%M%S")


def _resolve_prepare_target(*, prepare_main_menu: bool, prepare_target: str | None) -> str:
    if prepare_target is None:
        return "main_menu" if prepare_main_menu else "none"
    normalized = prepare_target.lower()
    if normalized not in {"none", "main_menu", "character_select"}:
        raise ValueError("prepare_target must be one of: none, main_menu, character_select.")
    return normalized


def _normalize_target_literal(value: str) -> str:
    if value not in {"main_menu", "character_select"}:
        raise ValueError(f"Unsupported normalization target: {value}")
    return value


def _prepare_replay_start(
    *,
    base_url: str,
    request_timeout_seconds: float,
    poll_interval_seconds: float,
    max_idle_polls: int,
    max_prepare_steps: int,
    prepare_target: str,
    env_factory: Callable[[str, float], SupportsEnv],
) -> dict[str, Any]:
    if prepare_target == "none":
        env = env_factory(base_url, request_timeout_seconds)
        try:
            observation = env.observe()
        finally:
            env.close()
        payload = build_start_payload(observation)
        return {
            "prepare_target": prepare_target,
            "normalization_report": {
                "base_url": base_url,
                "target": prepare_target,
                "reached_target": True,
                "stop_reason": "target_reached",
                "initial_screen": observation.screen_type,
                "final_screen": observation.screen_type,
                "initial_run_id": observation.run_id,
                "final_run_id": observation.run_id,
                "step_count": 0,
                "wait_count": 0,
                "action_sequence": [],
                "strategy_histogram": {},
            },
            "start_screen": observation.screen_type,
            "start_signature": payload["start_signature"],
            "start_payload": payload,
            "prepare_action_ids": [],
        }

    target = _normalize_target_literal(prepare_target)
    normalization = normalize_runtime_state(
        base_url=base_url,
        target=target,
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        max_steps=max_prepare_steps,
        request_timeout_seconds=request_timeout_seconds,
        env_factory=env_factory,
    )
    if not normalization.reached_target or normalization.final_observation is None:
        raise RuntimeError(
            "Failed to normalize replay start. "
            f"target={prepare_target} initial={normalization.initial_screen} final={normalization.final_screen} "
            f"reason={normalization.stop_reason}"
        )

    payload = build_start_payload(normalization.final_observation)
    return {
        "prepare_target": prepare_target,
        "normalization_report": normalization.as_dict(),
        "start_screen": normalization.final_observation.screen_type,
        "start_signature": payload["start_signature"],
        "start_payload": payload,
        "prepare_action_ids": normalization.action_sequence,
    }


def _build_iteration_report(
    *,
    iteration_index: int,
    session_name: str,
    session_dir: Path,
    report: CombatEvaluationReport,
    prepare_report: dict[str, Any],
) -> CombatReplayIterationReport:
    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    step_trace = load_step_trace(report.log_path)
    runtime_metadata, runtime_fingerprint = build_iteration_runtime_metadata(
        base_url=report.base_url,
        prepare_target=str(prepare_report["prepare_target"]),
        summary_payload=summary,
        checkpoint_path=report.checkpoint_path,
    )
    action_id_histogram = dict(Counter(step_trace["action_sequence"]))
    observed_seed_metadata = _observed_seed_metadata(summary)
    return CombatReplayIterationReport(
        iteration_index=iteration_index,
        session_name=session_name,
        session_dir=session_dir,
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
        env_steps=int(report.env_steps),
        combat_steps=int(report.combat_steps),
        heuristic_steps=int(report.heuristic_steps),
        total_reward=float(report.total_reward),
        stop_reason=str(report.stop_reason),
        final_screen=str(report.final_screen),
        completed_run_count=int(report.completed_run_count),
        completed_combat_count=int(report.completed_combat_count),
        action_count=len(step_trace["action_sequence"]),
        action_sequence=step_trace["action_sequence"],
        action_id_histogram=action_id_histogram,
        action_histogram=dict(summary.get("action_histogram", {})),
        run_outcome_histogram=dict(summary.get("run_outcome_histogram", {})),
        observed_run_seeds=list(observed_seed_metadata["observed_run_seeds"]),
        observed_run_seed_histogram=dict(observed_seed_metadata["observed_run_seed_histogram"]),
        runs_without_observed_seed=int(observed_seed_metadata["runs_without_observed_seed"]),
        last_observed_seed=_as_optional_str(observed_seed_metadata["last_observed_seed"]),
    )


def _compare_iterations(
    *,
    baseline: CombatReplayIterationReport,
    candidate: CombatReplayIterationReport,
) -> CombatReplayComparisonReport:
    common_prefix_length = _common_prefix_length(baseline.action_sequence, candidate.action_sequence)
    first_action_divergence_index = _first_divergence_index(
        baseline.action_sequence,
        candidate.action_sequence,
        common_prefix_length=common_prefix_length,
    )

    start_signature_match = baseline.start_signature == candidate.start_signature
    stop_reason_match = baseline.stop_reason == candidate.stop_reason
    final_screen_match = baseline.final_screen == candidate.final_screen
    action_sequence_match = baseline.action_sequence == candidate.action_sequence
    action_histogram_match = baseline.action_histogram == candidate.action_histogram
    action_id_histogram_match = baseline.action_id_histogram == candidate.action_id_histogram
    run_outcome_histogram_match = baseline.run_outcome_histogram == candidate.run_outcome_histogram

    metric_differences: dict[str, dict[str, Any]] = {}
    for key, baseline_value, candidate_value in (
        ("env_steps", baseline.env_steps, candidate.env_steps),
        ("combat_steps", baseline.combat_steps, candidate.combat_steps),
        ("heuristic_steps", baseline.heuristic_steps, candidate.heuristic_steps),
        ("total_reward", baseline.total_reward, candidate.total_reward),
        ("stop_reason", baseline.stop_reason, candidate.stop_reason),
        ("final_screen", baseline.final_screen, candidate.final_screen),
        ("completed_run_count", baseline.completed_run_count, candidate.completed_run_count),
        ("completed_combat_count", baseline.completed_combat_count, candidate.completed_combat_count),
    ):
        if baseline_value != candidate_value:
            metric_differences[key] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
            }

    baseline_payload = _iteration_payload(baseline)
    candidate_payload = _iteration_payload(candidate)
    baseline_payload["step_trace"] = load_step_trace(baseline.log_path)
    candidate_payload["step_trace"] = load_step_trace(candidate.log_path)
    diagnostic = diagnose_iteration_divergence(
        baseline=baseline_payload,
        candidate=candidate_payload,
    )

    return CombatReplayComparisonReport(
        baseline_iteration=baseline.iteration_index,
        candidate_iteration=candidate.iteration_index,
        status=diagnostic.status,
        start_signature_match=start_signature_match,
        stop_reason_match=stop_reason_match,
        final_screen_match=final_screen_match,
        action_sequence_match=action_sequence_match,
        action_histogram_match=action_histogram_match,
        action_id_histogram_match=action_id_histogram_match,
        run_outcome_histogram_match=run_outcome_histogram_match,
        common_action_prefix_length=common_prefix_length,
        first_action_divergence_index=first_action_divergence_index,
        baseline_action_count=baseline.action_count,
        candidate_action_count=candidate.action_count,
        metric_differences=metric_differences,
        action_histogram_delta=_counter_delta(baseline.action_histogram, candidate.action_histogram),
        action_id_histogram_delta=_counter_delta(baseline.action_id_histogram, candidate.action_id_histogram),
        diagnostic=diagnostic,
    )


def _counter_delta(baseline: dict[str, int], candidate: dict[str, int]) -> dict[str, dict[str, int]]:
    delta: dict[str, dict[str, int]] = {}
    keys = sorted(set(baseline) | set(candidate))
    for key in keys:
        baseline_count = int(baseline.get(key, 0))
        candidate_count = int(candidate.get(key, 0))
        if baseline_count == candidate_count:
            continue
        delta[key] = {
            "baseline": baseline_count,
            "candidate": candidate_count,
            "delta": candidate_count - baseline_count,
        }
    return delta


def _common_prefix_length(lhs: list[str], rhs: list[str]) -> int:
    count = 0
    for left, right in zip(lhs, rhs, strict=False):
        if left != right:
            break
        count += 1
    return count


def _first_divergence_index(lhs: list[str], rhs: list[str], *, common_prefix_length: int) -> int | None:
    if lhs == rhs:
        return None
    return common_prefix_length + 1


def _iteration_payload(iteration: CombatReplayIterationReport) -> dict[str, Any]:
    return {
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
        "action_count": iteration.action_count,
        "action_histogram": iteration.action_histogram,
        "action_id_histogram": iteration.action_id_histogram,
        "run_outcome_histogram": iteration.run_outcome_histogram,
        "observed_run_seeds": iteration.observed_run_seeds,
        "observed_run_seed_histogram": iteration.observed_run_seed_histogram,
        "runs_without_observed_seed": iteration.runs_without_observed_seed,
        "last_observed_seed": iteration.last_observed_seed,
    }


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


def _comparison_payload(comparison: CombatReplayComparisonReport) -> dict[str, Any]:
    return {
        "baseline_iteration": comparison.baseline_iteration,
        "candidate_iteration": comparison.candidate_iteration,
        "status": comparison.status,
        "start_signature_match": comparison.start_signature_match,
        "stop_reason_match": comparison.stop_reason_match,
        "final_screen_match": comparison.final_screen_match,
        "action_sequence_match": comparison.action_sequence_match,
        "action_histogram_match": comparison.action_histogram_match,
        "action_id_histogram_match": comparison.action_id_histogram_match,
        "run_outcome_histogram_match": comparison.run_outcome_histogram_match,
        "common_action_prefix_length": comparison.common_action_prefix_length,
        "first_action_divergence_index": comparison.first_action_divergence_index,
        "baseline_action_count": comparison.baseline_action_count,
        "candidate_action_count": comparison.candidate_action_count,
        "metric_differences": comparison.metric_differences,
        "action_histogram_delta": comparison.action_histogram_delta,
        "action_id_histogram_delta": comparison.action_id_histogram_delta,
        "diagnostic": comparison.diagnostic.as_dict(),
    }


def _append_suite_log(path: Path, payload: dict[str, Any]) -> None:
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
