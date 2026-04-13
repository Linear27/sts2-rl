from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from sts2_rl.collect import CommunityPriorRuntimeConfig, StrategicRuntimeConfig
from sts2_rl.game_run_contract import GameRunContract

from .behavior_cloning import load_policy_checkpoint_metadata, run_behavior_cloning_evaluation
from .offline_cql import run_offline_cql_evaluation
from .compare import (
    CombatCheckpointComparisonReport,
    CombatCheckpointEvalIterationReport,
    _aggregate_iterations,
    _append_log,
    _delta_metrics,
    _pick_better_checkpoint,
    default_checkpoint_comparison_name,
)
from .replay import _default_env_factory, _prepare_replay_start
from .runner import CombatEvaluationReport, run_combat_dqn_evaluation, run_policy_pack_evaluation


def run_policy_checkpoint_evaluation(
    *,
    base_url: str,
    checkpoint_path: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    max_env_steps: int | None = 64,
    max_runs: int | None = 1,
    max_combats: int | None = 0,
    poll_interval_seconds: float = 0.25,
    max_idle_polls: int = 40,
    request_timeout_seconds: float = 30.0,
    env_factory: Callable[[str, float], Any] = _default_env_factory,
    policy_profile: str | None = None,
    predictor_config: Any | None = None,
    community_prior_config: CommunityPriorRuntimeConfig | None = None,
    game_run_contract: GameRunContract | None = None,
) -> CombatEvaluationReport:
    metadata = load_policy_checkpoint_metadata(checkpoint_path)
    algorithm = metadata.get("algorithm")
    if algorithm == "dqn":
        return run_combat_dqn_evaluation(
            base_url=base_url,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            session_name=session_name,
            max_env_steps=max_env_steps,
            max_runs=max_runs,
            max_combats=max_combats,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=max_idle_polls,
            request_timeout_seconds=request_timeout_seconds,
            game_run_contract=game_run_contract,
            env_factory=env_factory,
        )
    if algorithm == "behavior_cloning":
        return run_behavior_cloning_evaluation(
            base_url=base_url,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            session_name=session_name,
            max_env_steps=max_env_steps,
            max_runs=max_runs,
            max_combats=max_combats,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=max_idle_polls,
            request_timeout_seconds=request_timeout_seconds,
            game_run_contract=game_run_contract,
            env_factory=env_factory,
        )
    if algorithm == "offline_cql":
        return run_offline_cql_evaluation(
            base_url=base_url,
            checkpoint_path=checkpoint_path,
            output_root=output_root,
            session_name=session_name,
            max_env_steps=max_env_steps,
            max_runs=max_runs,
            max_combats=max_combats,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=max_idle_polls,
            request_timeout_seconds=request_timeout_seconds,
            policy_profile=policy_profile or "baseline",
            predictor_config=predictor_config,
            community_prior_config=community_prior_config,
            game_run_contract=game_run_contract,
            env_factory=env_factory,
        )
    if algorithm in {"strategic_pretrain", "strategic_finetune"}:
        return _run_strategic_checkpoint_evaluation(
            base_url=base_url,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=metadata,
            output_root=output_root,
            session_name=session_name,
            max_env_steps=max_env_steps,
            max_runs=max_runs,
            max_combats=max_combats,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=max_idle_polls,
            request_timeout_seconds=request_timeout_seconds,
            policy_profile=policy_profile or "baseline",
            predictor_config=predictor_config,
            community_prior_config=community_prior_config,
            game_run_contract=game_run_contract,
            env_factory=env_factory,
        )
    raise ValueError(f"Unsupported checkpoint algorithm: {algorithm}")


def run_policy_checkpoint_comparison(
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
    prepare_target: str = "main_menu",
    prepare_max_steps: int = 8,
    prepare_max_idle_polls: int = 40,
    env_factory: Callable[[str, float], Any] = _default_env_factory,
    evaluation_fn: Callable[..., CombatEvaluationReport] = run_policy_checkpoint_evaluation,
) -> CombatCheckpointComparisonReport:
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    baseline_checkpoint = Path(baseline_checkpoint_path).expanduser().resolve()
    candidate_checkpoint = Path(candidate_checkpoint_path).expanduser().resolve()
    comparison_dir = Path(output_root).expanduser().resolve() / (comparison_name or default_checkpoint_comparison_name())
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_path = comparison_dir / "comparison-summary.json"
    iterations_path = comparison_dir / "comparison-iterations.jsonl"
    log_path = comparison_dir / "comparison-log.jsonl"
    diagnostics_path = comparison_dir / "comparison-diagnostics.jsonl"
    diagnostics_path.write_text("", encoding="utf-8")

    _append_log(
        log_path,
        {
            "record_type": "policy_checkpoint_comparison_started",
            "base_url": base_url,
            "baseline_checkpoint_path": str(baseline_checkpoint),
            "candidate_checkpoint_path": str(candidate_checkpoint),
            "repeats": repeats,
            "prepare_target": prepare_target,
        },
    )

    iterations: list[CombatCheckpointEvalIterationReport] = []
    for checkpoint_label, checkpoint_path in (("baseline", baseline_checkpoint), ("candidate", candidate_checkpoint)):
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
            report = evaluation_fn(
                base_url=base_url,
                checkpoint_path=checkpoint_path,
                output_root=comparison_dir,
                session_name=f"{checkpoint_label}-iteration-{iteration_index:03d}",
                max_env_steps=max_env_steps,
                max_runs=max_runs,
                max_combats=max_combats,
                poll_interval_seconds=poll_interval_seconds,
                max_idle_polls=max_idle_polls,
                request_timeout_seconds=request_timeout_seconds,
                env_factory=env_factory,
            )
            iteration = CombatCheckpointEvalIterationReport(
                checkpoint_label=checkpoint_label,
                checkpoint_path=checkpoint_path,
                iteration_index=iteration_index,
                session_name=f"{checkpoint_label}-iteration-{iteration_index:03d}",
                session_dir=comparison_dir / f"{checkpoint_label}-iteration-{iteration_index:03d}",
                summary_path=report.summary_path,
                log_path=report.log_path,
                combat_outcomes_path=report.combat_outcomes_path,
                prepare_target=str(prepare_report["prepare_target"]),
                normalization_report=dict(prepare_report["normalization_report"]),
                start_screen=str(prepare_report["start_screen"]),
                start_signature=str(prepare_report["start_signature"]),
                start_payload=dict(prepare_report["start_payload"]),
                runtime_metadata={},
                runtime_fingerprint=f"{checkpoint_label}:{iteration_index}",
                step_trace_fingerprint=f"{checkpoint_label}:{iteration_index}",
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
            )
            iterations.append(iteration)
            with iterations_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(_iteration_payload(iteration), ensure_ascii=False))
                handle.write("\n")

    baseline_iterations = [item for item in iterations if item.checkpoint_label == "baseline"]
    candidate_iterations = [item for item in iterations if item.checkpoint_label == "candidate"]
    baseline = _aggregate_iterations("baseline", baseline_checkpoint, baseline_iterations)
    candidate = _aggregate_iterations("candidate", candidate_checkpoint, candidate_iterations)
    delta_metrics = _delta_metrics(baseline, candidate)
    better_checkpoint_label = _pick_better_checkpoint(baseline, candidate)
    summary_payload = {
        "base_url": base_url,
        "comparison_dir": str(comparison_dir),
        "baseline_checkpoint_path": str(baseline_checkpoint),
        "candidate_checkpoint_path": str(candidate_checkpoint),
        "repeat_count": repeats,
        "prepare_target": prepare_target,
        "better_checkpoint_label": better_checkpoint_label,
        "delta_metrics": delta_metrics,
        "baseline": baseline,
        "candidate": candidate,
        "summary_path": str(summary_path),
        "iterations_path": str(iterations_path),
        "diagnostics_path": str(diagnostics_path),
        "log_path": str(log_path),
        "paired_diagnostics": [],
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return CombatCheckpointComparisonReport(
        base_url=base_url,
        comparison_dir=comparison_dir,
        summary_path=summary_path,
        iterations_path=iterations_path,
        log_path=log_path,
        baseline_checkpoint_path=baseline_checkpoint,
        candidate_checkpoint_path=candidate_checkpoint,
        repeat_count=repeats,
        prepare_target=prepare_target,
        better_checkpoint_label=better_checkpoint_label,
        delta_metrics=delta_metrics,
        baseline=baseline,
        candidate=candidate,
        iterations=iterations,
        diagnostics_path=diagnostics_path,
        paired_diagnostics=[],
    )


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
    }


def _run_strategic_checkpoint_evaluation(
    *,
    base_url: str,
    checkpoint_path: str | Path,
    checkpoint_metadata: dict[str, Any],
    output_root: str | Path,
    session_name: str | None,
    max_env_steps: int | None,
    max_runs: int | None,
    max_combats: int | None,
    poll_interval_seconds: float,
    max_idle_polls: int,
    request_timeout_seconds: float,
    policy_profile: str,
    predictor_config: Any | None,
    community_prior_config: CommunityPriorRuntimeConfig | None,
    game_run_contract: GameRunContract | None,
    env_factory: Callable[[str, float], Any],
) -> CombatEvaluationReport:
    resolved_checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    strategic_model_config = StrategicRuntimeConfig(
        checkpoint_path=resolved_checkpoint_path,
        mode="dominant",
    )
    report = run_policy_pack_evaluation(
        base_url=base_url,
        output_root=output_root,
        session_name=session_name,
        policy_profile=policy_profile,
        max_env_steps=max_env_steps,
        max_runs=max_runs,
        max_combats=max_combats,
        poll_interval_seconds=poll_interval_seconds,
        max_idle_polls=max_idle_polls,
        request_timeout_seconds=request_timeout_seconds,
        predictor_config=predictor_config,
        community_prior_config=community_prior_config,
        strategic_model_config=strategic_model_config,
        game_run_contract=game_run_contract,
        env_factory=env_factory,
    )
    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "algorithm": checkpoint_metadata.get("algorithm"),
            "checkpoint_path": str(resolved_checkpoint_path),
            "checkpoint_metadata": checkpoint_metadata,
            "policy_profile": policy_profile,
            "strategic_model": {
                **strategic_model_config.as_dict(),
                "checkpoint_algorithm": checkpoint_metadata.get("algorithm"),
            },
        }
    )
    report.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return CombatEvaluationReport(
        base_url=report.base_url,
        env_steps=report.env_steps,
        combat_steps=report.combat_steps,
        heuristic_steps=report.heuristic_steps,
        total_reward=report.total_reward,
        final_screen=report.final_screen,
        final_run_id=report.final_run_id,
        log_path=report.log_path,
        summary_path=report.summary_path,
        combat_outcomes_path=report.combat_outcomes_path,
        checkpoint_path=resolved_checkpoint_path,
        combat_performance=report.combat_performance,
        stop_reason=report.stop_reason,
        completed_run_count=report.completed_run_count,
        completed_combat_count=report.completed_combat_count,
        observed_run_seeds=list(report.observed_run_seeds),
        observed_run_seed_histogram=dict(report.observed_run_seed_histogram),
        runs_without_observed_seed=report.runs_without_observed_seed,
        last_observed_seed=report.last_observed_seed,
    )
