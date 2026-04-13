from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

from .compare import CombatCheckpointComparisonReport, run_combat_dqn_checkpoint_comparison
from .dqn import DqnConfig
from .runner import CombatTrainingReport, run_combat_dqn_training

CheckpointSource = Literal["latest", "best", "best_eval"]
BestEvalFallback = Literal["latest", "best"]


@dataclass(frozen=True)
class CombatCheckpointSelectionReport:
    requested_source: CheckpointSource
    selected_checkpoint_label: str
    selected_checkpoint_path: Path
    selection_mode: str
    comparison_dir: Path | None = None
    comparison_summary_path: Path | None = None
    comparison_iterations_path: Path | None = None
    comparison_log_path: Path | None = None
    comparison_better_checkpoint_label: str | None = None
    comparison_delta_metrics: dict[str, object] = field(default_factory=dict)
    fallback_reason: str | None = None
    comparison_error: str | None = None


@dataclass(frozen=True)
class CombatTrainingScheduleSessionReport:
    session_index: int
    session_name: str
    resume_from: Path | None
    selected_checkpoint_label: str
    selected_checkpoint_for_next_session: Path
    checkpoint_selection: CombatCheckpointSelectionReport
    report: CombatTrainingReport


@dataclass(frozen=True)
class CombatTrainingScheduleReport:
    base_url: str
    schedule_dir: Path
    summary_path: Path
    log_path: Path
    checkpoint_source: CheckpointSource
    promotion_artifacts_root: Path | None
    session_count: int
    total_env_steps: int
    total_rl_steps: int
    total_reward: float
    final_checkpoint_path: Path | None
    sessions: list[CombatTrainingScheduleSessionReport]


def run_combat_dqn_schedule(
    *,
    base_url: str,
    output_root: str | Path,
    schedule_name: str | None = None,
    max_sessions: int = 3,
    session_max_env_steps: int | None = 64,
    session_max_runs: int | None = 1,
    session_max_combats: int | None = None,
    checkpoint_source: CheckpointSource = "latest",
    initial_resume_from: str | Path | None = None,
    poll_interval_seconds: float = 0.25,
    max_idle_polls: int = 40,
    checkpoint_every_rl_steps: int = 25,
    request_timeout_seconds: float = 30.0,
    best_eval_repeats: int = 3,
    best_eval_max_env_steps: int | None = 0,
    best_eval_max_runs: int | None = 1,
    best_eval_max_combats: int | None = 0,
    best_eval_prepare_target: str = "main_menu",
    best_eval_prepare_max_steps: int = 8,
    best_eval_prepare_max_idle_polls: int = 40,
    best_eval_fallback: BestEvalFallback = "latest",
    dqn_config: DqnConfig | None = None,
    learning_rate_override: float | None = None,
    gamma_override: float | None = None,
    epsilon_start_override: float | None = None,
    epsilon_end_override: float | None = None,
    epsilon_decay_steps_override: int | None = None,
    replay_capacity_override: int | None = None,
    batch_size_override: int | None = None,
    min_replay_size_override: int | None = None,
    target_sync_interval_override: int | None = None,
    updates_per_env_step_override: int | None = None,
    huber_delta_override: float | None = None,
    seed_override: int | None = None,
    double_dqn_override: bool | None = None,
    n_step_override: int | None = None,
    prioritized_replay_override: bool | None = None,
    priority_alpha_override: float | None = None,
    priority_beta_start_override: float | None = None,
    priority_beta_end_override: float | None = None,
    priority_beta_decay_steps_override: int | None = None,
    priority_epsilon_override: float | None = None,
    training_fn: Callable[..., CombatTrainingReport] = run_combat_dqn_training,
    comparison_fn: Callable[..., CombatCheckpointComparisonReport] = run_combat_dqn_checkpoint_comparison,
) -> CombatTrainingScheduleReport:
    schedule_dir = Path(output_root) / (schedule_name or default_schedule_name())
    schedule_dir.mkdir(parents=True, exist_ok=True)
    log_path = schedule_dir / "schedule.jsonl"
    summary_path = schedule_dir / "schedule-summary.json"
    promotion_artifacts_root = schedule_dir / "promotions" if checkpoint_source == "best_eval" else None

    sessions: list[CombatTrainingScheduleSessionReport] = []
    total_env_steps = 0
    total_rl_steps = 0
    total_reward = 0.0
    current_resume = Path(initial_resume_from) if initial_resume_from is not None else None

    _append_schedule_log(
        log_path,
        {
            "record_type": "schedule_started",
            "base_url": base_url,
            "max_sessions": max_sessions,
            "session_max_env_steps": session_max_env_steps,
            "session_max_runs": session_max_runs,
            "session_max_combats": session_max_combats,
            "checkpoint_source": checkpoint_source,
            "initial_resume_from": str(current_resume) if current_resume is not None else None,
            "checkpoint_every_rl_steps": checkpoint_every_rl_steps,
            "best_eval": {
                "repeats": best_eval_repeats,
                "max_env_steps": best_eval_max_env_steps,
                "max_runs": best_eval_max_runs,
                "max_combats": best_eval_max_combats,
                "prepare_target": best_eval_prepare_target,
                "prepare_max_steps": best_eval_prepare_max_steps,
                "prepare_max_idle_polls": best_eval_prepare_max_idle_polls,
                "fallback": best_eval_fallback,
                "promotion_artifacts_root": str(promotion_artifacts_root) if promotion_artifacts_root is not None else None,
            },
        },
    )

    for session_index in range(1, max_sessions + 1):
        session_name = f"session-{session_index:03d}"
        session_resume = current_resume

        session_dqn_config: DqnConfig | None = None
        if session_resume is None:
            session_dqn_config = dqn_config or DqnConfig()

        report = training_fn(
            base_url=base_url,
            output_root=schedule_dir,
            session_name=session_name,
            max_env_steps=session_max_env_steps,
            max_runs=session_max_runs,
            max_combats=session_max_combats,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=max_idle_polls,
            dqn_config=session_dqn_config,
            resume_from=session_resume,
            learning_rate_override=learning_rate_override,
            gamma_override=gamma_override,
            epsilon_start_override=epsilon_start_override,
            epsilon_end_override=epsilon_end_override,
            epsilon_decay_steps_override=epsilon_decay_steps_override,
            replay_capacity_override=replay_capacity_override,
            batch_size_override=batch_size_override,
            min_replay_size_override=min_replay_size_override,
            target_sync_interval_override=target_sync_interval_override,
            updates_per_env_step_override=updates_per_env_step_override,
            huber_delta_override=huber_delta_override,
            seed_override=seed_override,
            double_dqn_override=double_dqn_override,
            n_step_override=n_step_override,
            prioritized_replay_override=prioritized_replay_override,
            priority_alpha_override=priority_alpha_override,
            priority_beta_start_override=priority_beta_start_override,
            priority_beta_end_override=priority_beta_end_override,
            priority_beta_decay_steps_override=priority_beta_decay_steps_override,
            priority_epsilon_override=priority_epsilon_override,
            checkpoint_every_rl_steps=checkpoint_every_rl_steps,
            request_timeout_seconds=request_timeout_seconds,
        )

        total_env_steps += report.env_steps
        total_rl_steps += report.rl_steps
        total_reward += report.total_reward

        selection = _select_next_checkpoint(
            base_url=base_url,
            schedule_dir=schedule_dir,
            session_name=session_name,
            report=report,
            checkpoint_source=checkpoint_source,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=max_idle_polls,
            request_timeout_seconds=request_timeout_seconds,
            best_eval_repeats=best_eval_repeats,
            best_eval_max_env_steps=best_eval_max_env_steps,
            best_eval_max_runs=best_eval_max_runs,
            best_eval_max_combats=best_eval_max_combats,
            best_eval_prepare_target=best_eval_prepare_target,
            best_eval_prepare_max_steps=best_eval_prepare_max_steps,
            best_eval_prepare_max_idle_polls=best_eval_prepare_max_idle_polls,
            best_eval_fallback=best_eval_fallback,
            comparison_fn=comparison_fn,
        )
        next_checkpoint = selection.selected_checkpoint_path
        sessions.append(
            CombatTrainingScheduleSessionReport(
                session_index=session_index,
                session_name=session_name,
                resume_from=session_resume,
                selected_checkpoint_label=selection.selected_checkpoint_label,
                selected_checkpoint_for_next_session=next_checkpoint,
                checkpoint_selection=selection,
                report=report,
            )
        )

        _append_schedule_log(
            log_path,
            {
                "record_type": "session_completed",
                "session_index": session_index,
                "session_name": session_name,
                "resume_from": str(session_resume) if session_resume is not None else None,
                "env_steps": report.env_steps,
                "rl_steps": report.rl_steps,
                "heuristic_steps": report.heuristic_steps,
                "update_steps": report.update_steps,
                "completed_run_count": report.completed_run_count,
                "completed_combat_count": report.completed_combat_count,
                "stop_reason": report.stop_reason,
                "total_reward": report.total_reward,
                "final_screen": report.final_screen,
                "final_run_id": report.final_run_id,
                "latest_checkpoint": str(report.checkpoint_path),
                "best_checkpoint": str(report.best_checkpoint_path) if report.best_checkpoint_path is not None else None,
                "selected_checkpoint_label": selection.selected_checkpoint_label,
                "selected_checkpoint_for_next_session": str(next_checkpoint),
                "checkpoint_selection": _checkpoint_selection_payload(selection),
                "learning_metrics": report.learning_metrics,
                "replay_metrics": report.replay_metrics,
                "checkpoint_comparison": report.checkpoint_comparison,
                "summary_path": str(report.summary_path),
            },
        )

        current_resume = next_checkpoint
        if report.env_steps == 0 and report.rl_steps == 0 and report.heuristic_steps == 0:
            _append_schedule_log(
                log_path,
                {
                    "record_type": "schedule_stopped_early",
                    "reason": "no_progress",
                    "session_index": session_index,
                },
            )
            break

    final_checkpoint_path = current_resume
    summary = {
        "base_url": base_url,
        "schedule_dir": str(schedule_dir),
        "session_count": len(sessions),
        "total_env_steps": total_env_steps,
        "total_rl_steps": total_rl_steps,
        "total_reward": total_reward,
        "final_checkpoint_path": str(final_checkpoint_path) if final_checkpoint_path is not None else None,
        "checkpoint_source": checkpoint_source,
        "promotion_artifacts_root": str(promotion_artifacts_root) if promotion_artifacts_root is not None else None,
        "best_eval": {
            "repeats": best_eval_repeats,
            "max_env_steps": best_eval_max_env_steps,
            "max_runs": best_eval_max_runs,
            "max_combats": best_eval_max_combats,
            "prepare_target": best_eval_prepare_target,
            "prepare_max_steps": best_eval_prepare_max_steps,
            "prepare_max_idle_polls": best_eval_prepare_max_idle_polls,
            "fallback": best_eval_fallback,
        },
        "sessions": [
            {
                "session_index": session.session_index,
                "session_name": session.session_name,
                "resume_from": str(session.resume_from) if session.resume_from is not None else None,
                "selected_checkpoint_label": session.selected_checkpoint_label,
                "selected_checkpoint_for_next_session": str(session.selected_checkpoint_for_next_session),
                "env_steps": session.report.env_steps,
                "rl_steps": session.report.rl_steps,
                "heuristic_steps": session.report.heuristic_steps,
                "update_steps": session.report.update_steps,
                "completed_run_count": session.report.completed_run_count,
                "completed_combat_count": session.report.completed_combat_count,
                "stop_reason": session.report.stop_reason,
                "total_reward": session.report.total_reward,
                "final_screen": session.report.final_screen,
                "final_run_id": session.report.final_run_id,
                "latest_checkpoint": str(session.report.checkpoint_path),
                "best_checkpoint": str(session.report.best_checkpoint_path)
                if session.report.best_checkpoint_path is not None
                else None,
                "learning_metrics": session.report.learning_metrics,
                "replay_metrics": session.report.replay_metrics,
                "checkpoint_comparison": session.report.checkpoint_comparison,
                "checkpoint_selection": _checkpoint_selection_payload(session.checkpoint_selection),
                "summary_path": str(session.report.summary_path),
            }
            for session in sessions
        ],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_schedule_log(
        log_path,
        {
            "record_type": "schedule_stopped",
            "session_count": len(sessions),
            "total_env_steps": total_env_steps,
            "total_rl_steps": total_rl_steps,
            "total_reward": total_reward,
            "final_checkpoint_path": str(final_checkpoint_path) if final_checkpoint_path is not None else None,
            "summary_path": str(summary_path),
        },
    )

    return CombatTrainingScheduleReport(
        base_url=base_url,
        schedule_dir=schedule_dir,
        summary_path=summary_path,
        log_path=log_path,
        checkpoint_source=checkpoint_source,
        promotion_artifacts_root=promotion_artifacts_root,
        session_count=len(sessions),
        total_env_steps=total_env_steps,
        total_rl_steps=total_rl_steps,
        total_reward=total_reward,
        final_checkpoint_path=final_checkpoint_path,
        sessions=sessions,
    )


def default_schedule_name() -> str:
    return datetime.now(UTC).strftime("combat-dqn-schedule-%Y%m%d-%H%M%S")


def _select_next_checkpoint(
    *,
    base_url: str,
    schedule_dir: Path,
    session_name: str,
    report: CombatTrainingReport,
    checkpoint_source: CheckpointSource,
    poll_interval_seconds: float,
    max_idle_polls: int,
    request_timeout_seconds: float,
    best_eval_repeats: int,
    best_eval_max_env_steps: int | None,
    best_eval_max_runs: int | None,
    best_eval_max_combats: int | None,
    best_eval_prepare_target: str,
    best_eval_prepare_max_steps: int,
    best_eval_prepare_max_idle_polls: int,
    best_eval_fallback: BestEvalFallback,
    comparison_fn: Callable[..., CombatCheckpointComparisonReport],
) -> CombatCheckpointSelectionReport:
    latest_path = report.checkpoint_path
    best_path = report.best_checkpoint_path

    if checkpoint_source == "latest":
        return CombatCheckpointSelectionReport(
            requested_source=checkpoint_source,
            selected_checkpoint_label="latest",
            selected_checkpoint_path=latest_path,
            selection_mode="direct",
        )

    if checkpoint_source == "best":
        if best_path is not None:
            return CombatCheckpointSelectionReport(
                requested_source=checkpoint_source,
                selected_checkpoint_label="best",
                selected_checkpoint_path=best_path,
                selection_mode="direct",
            )
        return CombatCheckpointSelectionReport(
            requested_source=checkpoint_source,
            selected_checkpoint_label="latest",
            selected_checkpoint_path=latest_path,
            selection_mode="fallback",
            fallback_reason="best_checkpoint_missing",
        )

    if best_path is None:
        fallback_label, fallback_path = _fallback_selection(
            latest_path=latest_path,
            best_path=best_path,
            best_eval_fallback=best_eval_fallback,
        )
        return CombatCheckpointSelectionReport(
            requested_source=checkpoint_source,
            selected_checkpoint_label=fallback_label,
            selected_checkpoint_path=fallback_path,
            selection_mode="best_eval_fallback",
            fallback_reason="best_checkpoint_missing",
        )

    promotion_root = schedule_dir / "promotions"
    promotion_root.mkdir(parents=True, exist_ok=True)
    comparison_name = f"{session_name}-best-eval"

    try:
        comparison = comparison_fn(
            base_url=base_url,
            baseline_checkpoint_path=latest_path,
            candidate_checkpoint_path=best_path,
            output_root=promotion_root,
            comparison_name=comparison_name,
            repeats=best_eval_repeats,
            max_env_steps=best_eval_max_env_steps,
            max_runs=best_eval_max_runs,
            max_combats=best_eval_max_combats,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=max_idle_polls,
            request_timeout_seconds=request_timeout_seconds,
            prepare_target=best_eval_prepare_target,
            prepare_max_steps=best_eval_prepare_max_steps,
            prepare_max_idle_polls=best_eval_prepare_max_idle_polls,
        )
    except Exception as exc:
        fallback_label, fallback_path = _fallback_selection(
            latest_path=latest_path,
            best_path=best_path,
            best_eval_fallback=best_eval_fallback,
        )
        return CombatCheckpointSelectionReport(
            requested_source=checkpoint_source,
            selected_checkpoint_label=fallback_label,
            selected_checkpoint_path=fallback_path,
            selection_mode="best_eval_fallback",
            fallback_reason="comparison_failed",
            comparison_error=f"{type(exc).__name__}: {exc}",
        )

    selected_checkpoint_label: str
    selected_checkpoint_path: Path
    selection_mode = "best_eval"
    fallback_reason: str | None = None

    if comparison.better_checkpoint_label == "candidate":
        selected_checkpoint_label = "best"
        selected_checkpoint_path = best_path
    elif comparison.better_checkpoint_label == "baseline":
        selected_checkpoint_label = "latest"
        selected_checkpoint_path = latest_path
    else:
        selected_checkpoint_label, selected_checkpoint_path = _fallback_selection(
            latest_path=latest_path,
            best_path=best_path,
            best_eval_fallback=best_eval_fallback,
        )
        selection_mode = "best_eval_fallback"
        fallback_reason = "comparison_tie"

    return CombatCheckpointSelectionReport(
        requested_source=checkpoint_source,
        selected_checkpoint_label=selected_checkpoint_label,
        selected_checkpoint_path=selected_checkpoint_path,
        selection_mode=selection_mode,
        comparison_dir=comparison.comparison_dir,
        comparison_summary_path=comparison.summary_path,
        comparison_iterations_path=comparison.iterations_path,
        comparison_log_path=comparison.log_path,
        comparison_better_checkpoint_label=comparison.better_checkpoint_label,
        comparison_delta_metrics=dict(comparison.delta_metrics),
        fallback_reason=fallback_reason,
    )


def _fallback_selection(
    *,
    latest_path: Path,
    best_path: Path | None,
    best_eval_fallback: BestEvalFallback,
) -> tuple[str, Path]:
    if best_eval_fallback == "best" and best_path is not None:
        return "best", best_path
    return "latest", latest_path


def _checkpoint_selection_payload(selection: CombatCheckpointSelectionReport) -> dict[str, object]:
    return {
        "requested_source": selection.requested_source,
        "selected_checkpoint_label": selection.selected_checkpoint_label,
        "selected_checkpoint_path": str(selection.selected_checkpoint_path),
        "selection_mode": selection.selection_mode,
        "comparison_dir": str(selection.comparison_dir) if selection.comparison_dir is not None else None,
        "comparison_summary_path": (
            str(selection.comparison_summary_path) if selection.comparison_summary_path is not None else None
        ),
        "comparison_iterations_path": (
            str(selection.comparison_iterations_path) if selection.comparison_iterations_path is not None else None
        ),
        "comparison_log_path": str(selection.comparison_log_path) if selection.comparison_log_path is not None else None,
        "comparison_better_checkpoint_label": selection.comparison_better_checkpoint_label,
        "comparison_delta_metrics": selection.comparison_delta_metrics,
        "fallback_reason": selection.fallback_reason,
        "comparison_error": selection.comparison_error,
    }


def _append_schedule_log(path: Path, payload: dict[str, object]) -> None:
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
