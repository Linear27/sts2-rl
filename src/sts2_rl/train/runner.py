from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Callable, Protocol

from sts2_rl.collect import CommunityPriorRuntimeConfig, StrategicRuntimeConfig
from sts2_rl.collect.policy import build_policy_pack
from sts2_rl.data.trajectory import TrajectorySessionMetadata, TrajectorySessionRecorder
from sts2_rl.env.wrapper import Sts2Env
from sts2_rl.game_run_contract import GameRunContract, merge_game_run_contract_config
from sts2_rl.lifecycle import ObservationHeartbeat, SessionBudgets, normalize_optional_limit
from sts2_rl.predict import PredictorRuntimeConfig
from sts2_rl.runtime.custom_run import contract_requires_custom_run_prepare, prepare_custom_run_from_contract
from sts2_rl.runtime.normalize import normalize_runtime_state

from .combat_encoder import CombatStateEncoder
from .combat_reward import compute_combat_reward
from .combat_space import CombatActionSpace
from .dqn import DqnAgent, DqnConfig


@dataclass(frozen=True)
class CombatTrainingReport:
    base_url: str
    env_steps: int
    rl_steps: int
    heuristic_steps: int
    update_steps: int
    total_reward: float
    final_screen: str
    final_run_id: str
    log_path: Path
    summary_path: Path
    combat_outcomes_path: Path
    checkpoint_path: Path
    best_checkpoint_path: Path | None
    periodic_checkpoint_count: int
    learning_metrics: dict[str, object] = field(default_factory=dict)
    replay_metrics: dict[str, object] = field(default_factory=dict)
    checkpoint_comparison: dict[str, object] = field(default_factory=dict)
    stop_reason: str = ""
    completed_run_count: int = 0
    completed_combat_count: int = 0
    observed_run_seeds: list[str] = field(default_factory=list)
    observed_run_seed_histogram: dict[str, int] = field(default_factory=dict)
    runs_without_observed_seed: int = 0
    last_observed_seed: str | None = None


@dataclass(frozen=True)
class CombatEvaluationReport:
    base_url: str
    env_steps: int
    combat_steps: int
    heuristic_steps: int
    total_reward: float
    final_screen: str
    final_run_id: str
    log_path: Path
    summary_path: Path
    combat_outcomes_path: Path
    checkpoint_path: Path
    combat_performance: dict[str, object] = field(default_factory=dict)
    stop_reason: str = ""
    completed_run_count: int = 0
    completed_combat_count: int = 0
    observed_run_seeds: list[str] = field(default_factory=list)
    observed_run_seed_histogram: dict[str, int] = field(default_factory=dict)
    runs_without_observed_seed: int = 0
    last_observed_seed: str | None = None


@dataclass
class _LearningMetricAccumulator:
    update_call_count: int = 0
    update_batch_count: int = 0
    sample_count: int = 0
    target_sync_count: int = 0
    loss_sum: float = 0.0
    abs_td_error_sum: float = 0.0
    predicted_q_sum: float = 0.0
    target_q_sum: float = 0.0
    importance_weight_sum: float = 0.0
    sample_priority_sum: float = 0.0
    transition_steps_sum: float = 0.0
    max_abs_td_error: float = 0.0
    last_loss: float | None = None

    def add(self, loss_stats) -> None:
        if not loss_stats.performed or loss_stats.sample_count <= 0:
            return
        self.update_call_count += 1
        self.update_batch_count += loss_stats.update_batches
        self.sample_count += loss_stats.sample_count
        self.target_sync_count += loss_stats.target_sync_count
        self.loss_sum += (loss_stats.loss or 0.0) * loss_stats.sample_count
        self.abs_td_error_sum += (loss_stats.mean_abs_td_error or 0.0) * loss_stats.sample_count
        self.predicted_q_sum += (loss_stats.mean_predicted_q or 0.0) * loss_stats.sample_count
        self.target_q_sum += (loss_stats.mean_target_q or 0.0) * loss_stats.sample_count
        self.importance_weight_sum += (loss_stats.mean_importance_weight or 0.0) * loss_stats.sample_count
        self.sample_priority_sum += (loss_stats.mean_sample_priority or 0.0) * loss_stats.sample_count
        self.transition_steps_sum += (loss_stats.mean_transition_steps or 0.0) * loss_stats.sample_count
        self.max_abs_td_error = max(self.max_abs_td_error, loss_stats.max_abs_td_error or 0.0)
        self.last_loss = loss_stats.loss

    def as_dict(self, *, agent: DqnAgent) -> dict[str, object]:
        sample_count = self.sample_count
        return {
            "update_call_count": self.update_call_count,
            "update_batch_count": self.update_batch_count,
            "target_sync_count": self.target_sync_count,
            "sample_count": sample_count,
            "mean_loss": (self.loss_sum / sample_count) if sample_count else None,
            "last_loss": self.last_loss,
            "mean_abs_td_error": (self.abs_td_error_sum / sample_count) if sample_count else None,
            "max_abs_td_error": self.max_abs_td_error if sample_count else None,
            "mean_predicted_q": (self.predicted_q_sum / sample_count) if sample_count else None,
            "mean_target_q": (self.target_q_sum / sample_count) if sample_count else None,
            "mean_importance_weight": (self.importance_weight_sum / sample_count) if sample_count else None,
            "mean_sample_priority": (self.sample_priority_sum / sample_count) if sample_count else None,
            "mean_transition_steps": (self.transition_steps_sum / sample_count) if sample_count else None,
            "final_epsilon": agent.current_epsilon(),
            "final_priority_beta": agent.current_priority_beta(),
            "double_dqn": agent.config.double_dqn,
            "n_step": agent.config.n_step,
            "prioritized_replay": agent.config.prioritized_replay,
        }


class SupportsEnv(Protocol):
    def observe(self): ...

    def step(self, action): ...

    def close(self) -> None: ...


def _default_env_factory(base_url: str, timeout: float) -> SupportsEnv:
    return Sts2Env.from_base_url(base_url, timeout=timeout)


def _dqn_config_payload(agent: DqnAgent) -> dict[str, object]:
    return {
        "learning_rate": agent.config.learning_rate,
        "gamma": agent.config.gamma,
        "epsilon_start": agent.config.epsilon_start,
        "epsilon_end": agent.config.epsilon_end,
        "epsilon_decay_steps": agent.config.epsilon_decay_steps,
        "replay_capacity": agent.config.replay_capacity,
        "batch_size": agent.config.batch_size,
        "min_replay_size": agent.config.min_replay_size,
        "target_sync_interval": agent.config.target_sync_interval,
        "updates_per_env_step": agent.config.updates_per_env_step,
        "huber_delta": agent.config.huber_delta,
        "hidden_sizes": list(agent.config.hidden_sizes),
        "seed": agent.config.seed,
        "double_dqn": agent.config.double_dqn,
        "n_step": agent.config.n_step,
        "prioritized_replay": agent.config.prioritized_replay,
        "priority_alpha": agent.config.priority_alpha,
        "priority_beta_start": agent.config.priority_beta_start,
        "priority_beta_end": agent.config.priority_beta_end,
        "priority_beta_decay_steps": agent.config.priority_beta_decay_steps,
        "priority_epsilon": agent.config.priority_epsilon,
    }


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _checkpoint_snapshot(
    *,
    label: str,
    path: Path,
    agent: DqnAgent,
    rl_steps: int,
    env_steps: int,
    total_reward: float,
    completed_run_count: int,
    won_runs: int,
    completed_combat_count: int,
    won_combats: int,
    learning_metrics: dict[str, object],
) -> dict[str, object]:
    replay_metrics = agent.replay_stats().as_dict()
    return {
        "label": label,
        "path": str(path),
        "env_steps": env_steps,
        "rl_steps": rl_steps,
        "update_steps": agent.update_steps,
        "total_reward": total_reward,
        "average_reward_per_rl_step": _safe_rate(total_reward, rl_steps),
        "completed_run_count": completed_run_count,
        "won_runs": won_runs,
        "run_win_rate": _safe_rate(won_runs, completed_run_count),
        "completed_combat_count": completed_combat_count,
        "won_combats": won_combats,
        "combat_win_rate": _safe_rate(won_combats, completed_combat_count),
        "mean_loss": learning_metrics.get("mean_loss"),
        "mean_abs_td_error": learning_metrics.get("mean_abs_td_error"),
        "mean_target_q": learning_metrics.get("mean_target_q"),
        "final_epsilon": learning_metrics.get("final_epsilon"),
        "final_priority_beta": learning_metrics.get("final_priority_beta"),
        "replay_metrics": replay_metrics,
    }


def _checkpoint_comparison_payload(
    *,
    latest: dict[str, object] | None,
    best: dict[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "comparison_basis": "session_total_reward",
        "latest": latest,
        "best": best,
        "deltas": {},
    }
    if latest is None or best is None:
        return payload

    deltas: dict[str, object] = {}
    for key in (
        "env_steps",
        "rl_steps",
        "update_steps",
        "total_reward",
        "average_reward_per_rl_step",
        "completed_run_count",
        "won_runs",
        "run_win_rate",
        "completed_combat_count",
        "won_combats",
        "combat_win_rate",
        "mean_loss",
        "mean_abs_td_error",
        "mean_target_q",
        "final_epsilon",
        "final_priority_beta",
    ):
        latest_value = latest.get(key)
        best_value = best.get(key)
        if latest_value is None or best_value is None:
            continue
        if isinstance(latest_value, (int, float)) and isinstance(best_value, (int, float)):
            deltas[key] = float(latest_value) - float(best_value)
        elif latest_value != best_value:
            deltas[key] = {
                "latest": latest_value,
                "best": best_value,
            }

    latest_replay = latest.get("replay_metrics")
    best_replay = best.get("replay_metrics")
    if isinstance(latest_replay, dict) and isinstance(best_replay, dict):
        replay_deltas: dict[str, object] = {}
        for key in ("size", "utilization", "priority_beta", "priority_min", "priority_mean", "priority_max"):
            latest_value = latest_replay.get(key)
            best_value = best_replay.get(key)
            if latest_value is None or best_value is None:
                continue
            if isinstance(latest_value, (int, float)) and isinstance(best_value, (int, float)):
                replay_deltas[key] = float(latest_value) - float(best_value)
        if replay_deltas:
            deltas["replay_metrics"] = replay_deltas

    payload["deltas"] = deltas
    return payload


def _combat_performance_payload(
    *,
    summary,
    combat_steps: int,
    total_reward: float,
) -> dict[str, object]:
    return {
        "combat_steps": combat_steps,
        "completed_combat_count": summary.completed_combat_count,
        "won_combats": summary.won_combats,
        "lost_combats": summary.lost_combats,
        "interrupted_combats": summary.interrupted_combats,
        "combat_win_rate": _safe_rate(summary.won_combats, summary.completed_combat_count),
        "reward_per_combat": _safe_rate(total_reward, summary.completed_combat_count),
        "reward_per_combat_step": _safe_rate(total_reward, combat_steps),
        "completed_run_count": summary.completed_run_count,
        "won_runs": summary.won_runs,
        "run_win_rate": _safe_rate(summary.won_runs, summary.completed_run_count),
    }


def _checkpoint_metadata_payload(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "schema_version": payload.get("schema_version"),
        "algorithm": payload.get("algorithm"),
        "config": payload.get("config", {}),
        "training_state": payload.get("training_state", {}),
        "metadata": payload.get("metadata", {}),
    }


def run_combat_dqn_training(
    *,
    base_url: str,
    output_root: str | Path,
    session_name: str | None = None,
    max_env_steps: int | None = 64,
    max_runs: int | None = 1,
    max_combats: int | None = None,
    poll_interval_seconds: float = 0.25,
    max_idle_polls: int = 40,
    dqn_config: DqnConfig | None = None,
    resume_from: str | Path | None = None,
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
    checkpoint_every_rl_steps: int = 25,
    request_timeout_seconds: float = 30.0,
    policy_profile: str = "baseline",
    predictor_config: PredictorRuntimeConfig | None = None,
    community_prior_config: CommunityPriorRuntimeConfig | None = None,
    game_run_contract: GameRunContract | None = None,
    env_factory: Callable[[str, float], SupportsEnv] = _default_env_factory,
) -> CombatTrainingReport:
    max_env_steps = normalize_optional_limit(max_env_steps)
    max_runs = normalize_optional_limit(max_runs)
    max_combats = normalize_optional_limit(max_combats)
    budgets = SessionBudgets(max_env_steps=max_env_steps, max_runs=max_runs, max_combats=max_combats)
    session_dir = Path(output_root) / (session_name or default_training_session_name())
    session_dir.mkdir(parents=True, exist_ok=True)
    instance_id = instance_label_from_base_url(base_url)

    encoder = CombatStateEncoder()
    action_space = CombatActionSpace()
    if resume_from is not None:
        agent = DqnAgent.load(
            resume_from,
            expected_action_count=action_space.slot_count,
            expected_feature_count=encoder.feature_count,
        )
        agent.reconfigure(
            learning_rate=learning_rate_override,
            gamma=gamma_override,
            epsilon_start=epsilon_start_override,
            epsilon_end=epsilon_end_override,
            epsilon_decay_steps=epsilon_decay_steps_override,
            replay_capacity=replay_capacity_override,
            batch_size=batch_size_override,
            min_replay_size=min_replay_size_override,
            target_sync_interval=target_sync_interval_override,
            updates_per_env_step=updates_per_env_step_override,
            huber_delta=huber_delta_override,
            seed=seed_override,
            double_dqn=double_dqn_override,
            n_step=n_step_override,
            prioritized_replay=prioritized_replay_override,
            priority_alpha=priority_alpha_override,
            priority_beta_start=priority_beta_start_override,
            priority_beta_end=priority_beta_end_override,
            priority_beta_decay_steps=priority_beta_decay_steps_override,
            priority_epsilon=priority_epsilon_override,
        )
    else:
        agent = DqnAgent(
            action_count=action_space.slot_count,
            feature_count=encoder.feature_count,
            config=dqn_config or DqnConfig(),
        )
    heuristic_policy = build_policy_pack(
        policy_profile,
        predictor_config=predictor_config,
        community_prior_config=community_prior_config,
    )
    env = env_factory(base_url, request_timeout_seconds)

    log_path = session_dir / "combat-train.jsonl"
    summary_path = session_dir / "summary.json"
    combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
    checkpoint_path = session_dir / "combat-dqn-checkpoint.json"
    best_checkpoint_path = session_dir / "combat-dqn-best.json"
    periodic_checkpoint_dir = session_dir / "checkpoints"
    periodic_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    recorder = TrajectorySessionRecorder(
        log_path=log_path,
        summary_path=summary_path,
        combat_outcomes_path=combat_outcomes_path,
        metadata=TrajectorySessionMetadata(
            session_name=session_dir.name,
            session_kind="train",
            base_url=base_url,
            policy_name=heuristic_policy.name,
            algorithm="dqn",
            config=merge_game_run_contract_config(
                {
                    **budgets.as_dict(),
                    "poll_interval_seconds": poll_interval_seconds,
                    "max_idle_polls": max_idle_polls,
                    "resume_from": str(resume_from) if resume_from is not None else None,
                    "checkpoint_every_rl_steps": checkpoint_every_rl_steps,
                    "dqn_config": _dqn_config_payload(agent),
                    "predictor": predictor_config.as_dict() if predictor_config is not None else None,
                    "community_prior": (
                        community_prior_config.as_dict() if community_prior_config is not None else None
                    ),
                },
                game_run_contract,
            ),
            game_run_contract=game_run_contract,
        ),
    )

    env_steps = 0
    rl_steps = 0
    heuristic_steps = 0
    total_reward = 0.0
    learning_accumulator = _LearningMetricAccumulator()
    best_total_reward = float("-inf")
    best_path_written: Path | None = None
    best_snapshot: dict[str, object] | None = None
    latest_snapshot: dict[str, object] | None = None
    periodic_checkpoint_count = 0
    heartbeat = ObservationHeartbeat()
    final_screen = "UNKNOWN"
    final_run_id = "run_unknown"
    final_observation = None
    stop_reason = "session_completed"

    recorder.logger.log_event(
        "trainer_started",
        {
            "base_url": base_url,
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path),
        },
    )
    if contract_requires_custom_run_prepare(game_run_contract):
        preparation = prepare_custom_run_from_contract(
            base_url=base_url,
            contract=game_run_contract,
            request_timeout_seconds=request_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=max_idle_polls,
            max_prepare_steps=None,
            env_factory=env_factory,
        )
        recorder.metadata.config["custom_run_prepare"] = preparation.as_dict()
        recorder.logger.log_event(
            "custom_run_prepared",
            {
                "base_url": base_url,
                "custom_run_prepare": preparation.as_dict(),
            },
        )

    try:
        while True:
            observation = env.observe()
            final_observation = observation
            final_screen = observation.screen_type
            final_run_id = observation.run_id
            heartbeat.observe(observation)
            recorder.sync_observation(observation, instance_id=instance_id, step_index=env_steps)
            contract_stop_reason = recorder.enforce_game_run_contract(
                instance_id=instance_id,
                step_index=env_steps,
            )
            if contract_stop_reason is not None:
                stop_reason = contract_stop_reason
                break
            budget_reason = _budget_stop_reason(budgets=budgets, env_steps=env_steps, recorder=recorder)
            if budget_reason is not None:
                stop_reason = budget_reason
                break

            if not observation.legal_actions:
                heartbeat.note_wait()
                if heartbeat.reached_idle_poll_limit(max_idle_polls):
                    stop_reason = "max_idle_polls_reached"
                    break
                sleep(poll_interval_seconds)
                continue

            if observation.screen_type == "COMBAT":
                binding = action_space.bind(observation)
                if binding.available_slots:
                    features = encoder.encode(observation)
                    selection = agent.select_action(features, binding.mask)
                    action = binding.candidates[selection.action_index]
                    if action is None:
                        raise RuntimeError(f"Combat action slot {selection.action_index} was masked but had no candidate.")

                    result = env.step(action)
                    reward = compute_combat_reward(observation, result.observation)
                    result = result.model_copy(update={"reward": reward})
                    total_reward += reward
                    rl_steps += 1
                    env_steps += 1
                    final_observation = result.observation
                    final_screen = result.observation.screen_type
                    final_run_id = result.observation.run_id
                    heartbeat.mark_progress(result.observation)

                    next_features: list[float] | None = None
                    next_mask: list[bool] | None = None
                    done = result.terminated or result.observation.screen_type != "COMBAT"
                    if not done and result.observation.screen_type == "COMBAT":
                        next_binding = action_space.bind(result.observation)
                        if next_binding.available_slots:
                            next_features = encoder.encode(result.observation)
                            next_mask = next_binding.mask

                    agent.add_transition(
                        features=features,
                        action_index=selection.action_index,
                        reward=reward,
                        next_features=next_features,
                        next_mask=next_mask,
                        done=done,
                    )
                    update_stats = agent.maybe_update()
                    learning_accumulator.add(update_stats)
                    replay_metrics = agent.replay_stats()

                    recorder.log_step(
                        instance_id=instance_id,
                        step_index=env_steps,
                        previous_observation=observation,
                        result=result,
                        chosen_action=action,
                        policy_name="combat-dqn",
                        algorithm="dqn",
                        decision_source="dqn",
                        decision_stage="combat",
                        decision_reason="epsilon_greedy" if selection.exploratory else "greedy_q",
                        decision_score=selection.q_values[selection.action_index],
                        reward_source="combat_reward_v2",
                        model_metrics={
                            "action_index": selection.action_index,
                            "epsilon": selection.epsilon,
                            "exploratory": selection.exploratory,
                            "q_value": selection.q_values[selection.action_index],
                            "update_performed": update_stats.performed,
                            "loss": update_stats.loss,
                            "mean_abs_td_error": update_stats.mean_abs_td_error,
                            "max_abs_td_error": update_stats.max_abs_td_error,
                            "mean_predicted_q": update_stats.mean_predicted_q,
                            "mean_target_q": update_stats.mean_target_q,
                            "mean_importance_weight": update_stats.mean_importance_weight,
                            "mean_sample_priority": update_stats.mean_sample_priority,
                            "mean_transition_steps": update_stats.mean_transition_steps,
                            "replay_size": update_stats.replay_size,
                            "batch_size": update_stats.batch_size,
                            "sample_count": update_stats.sample_count,
                            "update_batches": update_stats.update_batches,
                            "update_step": update_stats.update_step,
                            "target_synced": update_stats.target_synced,
                            "target_sync_count": update_stats.target_sync_count,
                            "priority_beta": update_stats.priority_beta,
                            "replay_utilization": replay_metrics.utilization,
                            "replay_priority_min": replay_metrics.priority_min,
                            "replay_priority_mean": replay_metrics.priority_mean,
                            "replay_priority_max": replay_metrics.priority_max,
                            "pending_n_step": replay_metrics.pending_n_step,
                            "double_dqn": agent.config.double_dqn,
                            "n_step": agent.config.n_step,
                            "prioritized_replay": replay_metrics.prioritized_replay,
                        },
                    )
                    recorder.sync_observation(result.observation, instance_id=instance_id, step_index=env_steps)
                    contract_stop_reason = recorder.enforce_game_run_contract(
                        instance_id=instance_id,
                        step_index=env_steps,
                    )
                    if contract_stop_reason is not None:
                        stop_reason = contract_stop_reason
                        break

                    if total_reward > best_total_reward:
                        best_total_reward = total_reward
                        progress = recorder.progress()
                        learning_metrics = learning_accumulator.as_dict(agent=agent)
                        best_snapshot = _checkpoint_snapshot(
                            label="best",
                            path=best_checkpoint_path,
                            agent=agent,
                            rl_steps=rl_steps,
                            env_steps=env_steps,
                            total_reward=total_reward,
                            completed_run_count=progress.completed_run_count,
                            won_runs=progress.won_runs,
                            completed_combat_count=progress.completed_combat_count,
                            won_combats=progress.won_combats,
                            learning_metrics=learning_metrics,
                        )
                        best_path_written = agent.save(
                            best_checkpoint_path,
                            metadata={"kind": "best", "snapshot": best_snapshot},
                        )
                        recorder.logger.log_event(
                            "checkpoint_saved",
                            {
                                "kind": "best",
                                "rl_step": rl_steps,
                                "total_reward": total_reward,
                                "path": str(best_path_written),
                                "snapshot": best_snapshot,
                            },
                        )

                    if checkpoint_every_rl_steps > 0 and rl_steps % checkpoint_every_rl_steps == 0:
                        progress = recorder.progress()
                        learning_metrics = learning_accumulator.as_dict(agent=agent)
                        latest_snapshot = _checkpoint_snapshot(
                            label="latest",
                            path=checkpoint_path,
                            agent=agent,
                            rl_steps=rl_steps,
                            env_steps=env_steps,
                            total_reward=total_reward,
                            completed_run_count=progress.completed_run_count,
                            won_runs=progress.won_runs,
                            completed_combat_count=progress.completed_combat_count,
                            won_combats=progress.won_combats,
                            learning_metrics=learning_metrics,
                        )
                        agent.save(
                            checkpoint_path,
                            metadata={"kind": "latest", "snapshot": latest_snapshot},
                        )
                        snapshot_path = periodic_checkpoint_dir / f"combat-dqn-step-{rl_steps:06d}.json"
                        periodic_snapshot = dict(latest_snapshot)
                        periodic_snapshot["label"] = "periodic"
                        periodic_snapshot["path"] = str(snapshot_path)
                        agent.save(
                            snapshot_path,
                            metadata={"kind": "periodic", "snapshot": periodic_snapshot},
                        )
                        periodic_checkpoint_count += 1
                        recorder.logger.log_event(
                            "checkpoint_saved",
                            {
                                "kind": "periodic",
                                "rl_step": rl_steps,
                                "total_reward": total_reward,
                                "path": str(snapshot_path),
                                "snapshot": periodic_snapshot,
                            },
                        )

                    budget_reason = _budget_stop_reason(budgets=budgets, env_steps=env_steps, recorder=recorder)
                    if budget_reason is not None:
                        stop_reason = budget_reason
                        break
                    continue

            decision = heuristic_policy.choose(observation)
            if decision.action is None:
                heartbeat.note_wait()
                if heartbeat.reached_idle_poll_limit(max_idle_polls):
                    stop_reason = f"policy_no_action_timeout:{observation.screen_type.lower()}:{decision.reason}"
                    recorder.record_capability_diagnostics(
                        instance_id=instance_id,
                        step_index=env_steps,
                        observation=observation,
                        decision_reason=decision.reason,
                        stop_reason=stop_reason,
                        decision_metadata=decision.trace_metadata,
                    )
                    break
                sleep(poll_interval_seconds)
                continue

            result = env.step(decision.action)
            heuristic_steps += 1
            env_steps += 1
            final_observation = result.observation
            final_screen = result.observation.screen_type
            final_run_id = result.observation.run_id
            heartbeat.mark_progress(result.observation)
            recorder.log_step(
                instance_id=instance_id,
                step_index=env_steps,
                previous_observation=observation,
                result=result,
                        chosen_action=decision.action,
                        policy_name=heuristic_policy.name,
                        policy_pack=decision.policy_pack,
                        policy_handler=decision.policy_handler,
                        decision_source="heuristic",
                        decision_stage=decision.stage,
                        decision_reason=decision.reason,
                        decision_score=decision.score,
                        planner_name=decision.planner_name,
                        planner_strategy=decision.planner_strategy,
                        ranked_actions=[_ranked_action_payload(item) for item in decision.ranked_actions],
                        decision_metadata=decision.trace_metadata,
                        reward_source="non_combat_transition",
                    )
            recorder.sync_observation(result.observation, instance_id=instance_id, step_index=env_steps)
            contract_stop_reason = recorder.enforce_game_run_contract(
                instance_id=instance_id,
                step_index=env_steps,
            )
            if contract_stop_reason is not None:
                stop_reason = contract_stop_reason
                break

            budget_reason = _budget_stop_reason(budgets=budgets, env_steps=env_steps, recorder=recorder)
            if budget_reason is not None:
                stop_reason = budget_reason
                break
    finally:
        env.close()

    summary = recorder.finalize(
        instance_id=instance_id,
        stop_reason=stop_reason,
        step_index=env_steps,
        final_observation=final_observation,
    )
    learning_metrics = learning_accumulator.as_dict(agent=agent)
    latest_snapshot = _checkpoint_snapshot(
        label="latest",
        path=checkpoint_path,
        agent=agent,
        rl_steps=rl_steps,
        env_steps=env_steps,
        total_reward=total_reward,
        completed_run_count=summary.completed_run_count,
        won_runs=summary.won_runs,
        completed_combat_count=summary.completed_combat_count,
        won_combats=summary.won_combats,
        learning_metrics=learning_metrics,
    )
    agent.save(
        checkpoint_path,
        metadata={"kind": "latest", "snapshot": latest_snapshot},
    )
    replay_metrics = agent.replay_stats().as_dict()
    checkpoint_comparison = _checkpoint_comparison_payload(latest=latest_snapshot, best=best_snapshot)
    training_summary = {
        **summary.as_dict(),
        "rl_steps": rl_steps,
        "heuristic_steps": heuristic_steps,
        "update_steps": agent.update_steps,
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_path_written) if best_path_written is not None else None,
        "periodic_checkpoint_count": periodic_checkpoint_count,
        "combat_outcomes_path": str(combat_outcomes_path),
        "log_path": str(log_path),
        "learning_metrics": learning_metrics,
        "replay_metrics": replay_metrics,
        "checkpoint_comparison": checkpoint_comparison,
        "checkpoint_metadata": _checkpoint_metadata_payload(checkpoint_path),
    }
    summary_path.write_text(json.dumps(training_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    recorder.logger.log_event(
        "trainer_stopped",
        {
            "env_steps": env_steps,
            "rl_steps": rl_steps,
            "heuristic_steps": heuristic_steps,
            "update_steps": agent.update_steps,
            "total_reward": total_reward,
            "final_screen": final_screen,
            "final_run_id": final_run_id,
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_path_written) if best_path_written is not None else None,
            "summary_path": str(summary_path),
            "learning_metrics": learning_metrics,
            "replay_metrics": replay_metrics,
        },
    )

    return CombatTrainingReport(
        base_url=base_url,
        env_steps=env_steps,
        rl_steps=rl_steps,
        heuristic_steps=heuristic_steps,
        update_steps=agent.update_steps,
        total_reward=total_reward,
        final_screen=final_screen,
        final_run_id=final_run_id,
        log_path=log_path,
        summary_path=summary_path,
        combat_outcomes_path=combat_outcomes_path,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_path_written,
        periodic_checkpoint_count=periodic_checkpoint_count,
        learning_metrics=learning_metrics,
        replay_metrics=replay_metrics,
        checkpoint_comparison=checkpoint_comparison,
        stop_reason=summary.stop_reason,
        completed_run_count=summary.completed_run_count,
        completed_combat_count=summary.completed_combat_count,
        observed_run_seeds=list(summary.observed_run_seeds),
        observed_run_seed_histogram=dict(summary.observed_run_seed_histogram),
        runs_without_observed_seed=summary.runs_without_observed_seed,
        last_observed_seed=summary.last_observed_seed,
    )


def run_combat_dqn_evaluation(
    *,
    base_url: str,
    checkpoint_path: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    max_env_steps: int | None = 64,
    max_runs: int | None = 1,
    max_combats: int | None = None,
    poll_interval_seconds: float = 0.25,
    max_idle_polls: int = 40,
    request_timeout_seconds: float = 30.0,
    policy_profile: str = "baseline",
    predictor_config: PredictorRuntimeConfig | None = None,
    community_prior_config: CommunityPriorRuntimeConfig | None = None,
    game_run_contract: GameRunContract | None = None,
    env_factory: Callable[[str, float], SupportsEnv] = _default_env_factory,
) -> CombatEvaluationReport:
    max_env_steps = normalize_optional_limit(max_env_steps)
    max_runs = normalize_optional_limit(max_runs)
    max_combats = normalize_optional_limit(max_combats)
    budgets = SessionBudgets(max_env_steps=max_env_steps, max_runs=max_runs, max_combats=max_combats)
    session_dir = Path(output_root) / (session_name or default_evaluation_session_name())
    session_dir.mkdir(parents=True, exist_ok=True)
    instance_id = instance_label_from_base_url(base_url)

    encoder = CombatStateEncoder()
    action_space = CombatActionSpace()
    agent = DqnAgent.load(
        checkpoint_path,
        expected_action_count=action_space.slot_count,
        expected_feature_count=encoder.feature_count,
    )
    checkpoint_metadata = _checkpoint_metadata_payload(checkpoint_path)
    heuristic_policy = build_policy_pack(
        policy_profile,
        predictor_config=predictor_config,
        community_prior_config=community_prior_config,
    )
    env = env_factory(base_url, request_timeout_seconds)

    log_path = session_dir / "combat-eval.jsonl"
    summary_path = session_dir / "summary.json"
    combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
    recorder = TrajectorySessionRecorder(
        log_path=log_path,
        summary_path=summary_path,
        combat_outcomes_path=combat_outcomes_path,
        metadata=TrajectorySessionMetadata(
            session_name=session_dir.name,
            session_kind="eval",
            base_url=base_url,
            policy_name=heuristic_policy.name,
            algorithm="dqn",
            config=merge_game_run_contract_config(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_metadata": checkpoint_metadata,
                    **budgets.as_dict(),
                    "poll_interval_seconds": poll_interval_seconds,
                    "max_idle_polls": max_idle_polls,
                    "predictor": predictor_config.as_dict() if predictor_config is not None else None,
                    "community_prior": (
                        community_prior_config.as_dict() if community_prior_config is not None else None
                    ),
                },
                game_run_contract,
            ),
            game_run_contract=game_run_contract,
        ),
    )

    env_steps = 0
    combat_steps = 0
    heuristic_steps = 0
    total_reward = 0.0
    heartbeat = ObservationHeartbeat()
    final_screen = "UNKNOWN"
    final_run_id = "run_unknown"
    final_observation = None
    stop_reason = "session_completed"
    if contract_requires_custom_run_prepare(game_run_contract):
        preparation = prepare_custom_run_from_contract(
            base_url=base_url,
            contract=game_run_contract,
            request_timeout_seconds=request_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=max_idle_polls,
            max_prepare_steps=None,
            env_factory=env_factory,
        )
        recorder.metadata.config["custom_run_prepare"] = preparation.as_dict()
        recorder.logger.log_event(
            "custom_run_prepared",
            {
                "base_url": base_url,
                "custom_run_prepare": preparation.as_dict(),
            },
        )

    try:
        while True:
            observation = env.observe()
            final_observation = observation
            final_screen = observation.screen_type
            final_run_id = observation.run_id
            heartbeat.observe(observation)
            recorder.sync_observation(observation, instance_id=instance_id, step_index=env_steps)
            contract_stop_reason = recorder.enforce_game_run_contract(
                instance_id=instance_id,
                step_index=env_steps,
            )
            if contract_stop_reason is not None:
                stop_reason = contract_stop_reason
                break
            budget_reason = _budget_stop_reason(budgets=budgets, env_steps=env_steps, recorder=recorder)
            if budget_reason is not None:
                stop_reason = budget_reason
                break

            if not observation.legal_actions:
                heartbeat.note_wait()
                if heartbeat.reached_idle_poll_limit(max_idle_polls):
                    stop_reason = "max_idle_polls_reached"
                    break
                sleep(poll_interval_seconds)
                continue

            if observation.screen_type == "COMBAT":
                binding = action_space.bind(observation)
                if binding.available_slots:
                    features = encoder.encode(observation)
                    action_index, q_values = agent.select_greedy_action(features, binding.mask)
                    action = binding.candidates[action_index]
                    if action is None:
                        raise RuntimeError(f"Combat action slot {action_index} was masked but had no candidate.")

                    result = env.step(action)
                    reward = compute_combat_reward(observation, result.observation)
                    result = result.model_copy(update={"reward": reward})
                    total_reward += reward
                    combat_steps += 1
                    env_steps += 1
                    final_observation = result.observation
                    final_screen = result.observation.screen_type
                    final_run_id = result.observation.run_id
                    heartbeat.mark_progress(result.observation)

                    recorder.log_step(
                        instance_id=instance_id,
                        step_index=env_steps,
                        previous_observation=observation,
                        result=result,
                        chosen_action=action,
                        policy_name="combat-dqn",
                        algorithm="dqn",
                        decision_source="dqn",
                        decision_stage="combat",
                        decision_reason="greedy_q",
                        decision_score=q_values[action_index],
                        reward_source="combat_reward_v2",
                        model_metrics={
                            "action_index": action_index,
                            "q_value": q_values[action_index],
                        },
                    )
                    recorder.sync_observation(result.observation, instance_id=instance_id, step_index=env_steps)
                    contract_stop_reason = recorder.enforce_game_run_contract(
                        instance_id=instance_id,
                        step_index=env_steps,
                    )
                    if contract_stop_reason is not None:
                        stop_reason = contract_stop_reason
                        break

                    budget_reason = _budget_stop_reason(budgets=budgets, env_steps=env_steps, recorder=recorder)
                    if budget_reason is not None:
                        stop_reason = budget_reason
                        break
                    continue

            decision = heuristic_policy.choose(observation)
            if decision.action is None:
                heartbeat.note_wait()
                if heartbeat.reached_idle_poll_limit(max_idle_polls):
                    stop_reason = f"policy_no_action_timeout:{observation.screen_type.lower()}:{decision.reason}"
                    recorder.record_capability_diagnostics(
                        instance_id=instance_id,
                        step_index=env_steps,
                        observation=observation,
                        decision_reason=decision.reason,
                        stop_reason=stop_reason,
                        decision_metadata=decision.trace_metadata,
                    )
                    break
                sleep(poll_interval_seconds)
                continue

            result = env.step(decision.action)
            heuristic_steps += 1
            env_steps += 1
            final_observation = result.observation
            final_screen = result.observation.screen_type
            final_run_id = result.observation.run_id
            heartbeat.mark_progress(result.observation)
            recorder.log_step(
                instance_id=instance_id,
                step_index=env_steps,
                previous_observation=observation,
                result=result,
                chosen_action=decision.action,
                policy_name=heuristic_policy.name,
                policy_pack=decision.policy_pack,
                policy_handler=decision.policy_handler,
                decision_source="heuristic",
                decision_stage=decision.stage,
                decision_reason=decision.reason,
                decision_score=decision.score,
                planner_name=decision.planner_name,
                planner_strategy=decision.planner_strategy,
                ranked_actions=[_ranked_action_payload(item) for item in decision.ranked_actions],
                decision_metadata=decision.trace_metadata,
                reward_source="non_combat_transition",
            )
            recorder.sync_observation(result.observation, instance_id=instance_id, step_index=env_steps)
            contract_stop_reason = recorder.enforce_game_run_contract(
                instance_id=instance_id,
                step_index=env_steps,
            )
            if contract_stop_reason is not None:
                stop_reason = contract_stop_reason
                break

            budget_reason = _budget_stop_reason(budgets=budgets, env_steps=env_steps, recorder=recorder)
            if budget_reason is not None:
                stop_reason = budget_reason
                break
    finally:
        env.close()

    summary = recorder.finalize(
        instance_id=instance_id,
        stop_reason=stop_reason,
        step_index=env_steps,
        final_observation=final_observation,
    )
    combat_performance = _combat_performance_payload(
        summary=summary,
        combat_steps=combat_steps,
        total_reward=total_reward,
    )
    evaluation_summary = {
        **summary.as_dict(),
        "combat_steps": combat_steps,
        "heuristic_steps": heuristic_steps,
        "checkpoint_path": str(checkpoint_path),
        "combat_outcomes_path": str(combat_outcomes_path),
        "log_path": str(log_path),
        "combat_performance": combat_performance,
        "checkpoint_metadata": checkpoint_metadata,
    }
    summary_path.write_text(json.dumps(evaluation_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    recorder.logger.log_event(
        "eval_stopped",
        {
            "env_steps": env_steps,
            "combat_steps": combat_steps,
            "heuristic_steps": heuristic_steps,
            "total_reward": total_reward,
            "final_screen": final_screen,
            "final_run_id": final_run_id,
            "checkpoint_path": str(checkpoint_path),
            "summary_path": str(summary_path),
            "combat_performance": combat_performance,
        },
    )

    return CombatEvaluationReport(
        base_url=base_url,
        env_steps=env_steps,
        combat_steps=combat_steps,
        heuristic_steps=heuristic_steps,
        total_reward=total_reward,
        final_screen=final_screen,
        final_run_id=final_run_id,
        log_path=log_path,
        summary_path=summary_path,
        combat_outcomes_path=combat_outcomes_path,
        checkpoint_path=Path(checkpoint_path),
        combat_performance=combat_performance,
        stop_reason=summary.stop_reason,
        completed_run_count=summary.completed_run_count,
        completed_combat_count=summary.completed_combat_count,
        observed_run_seeds=list(summary.observed_run_seeds),
        observed_run_seed_histogram=dict(summary.observed_run_seed_histogram),
        runs_without_observed_seed=summary.runs_without_observed_seed,
        last_observed_seed=summary.last_observed_seed,
    )


def run_policy_pack_evaluation(
    *,
    base_url: str,
    output_root: str | Path,
    session_name: str | None = None,
    policy_profile: str = "baseline",
    max_env_steps: int | None = 64,
    max_runs: int | None = 1,
    max_combats: int | None = None,
    poll_interval_seconds: float = 0.25,
    max_idle_polls: int = 40,
    request_timeout_seconds: float = 30.0,
    prepare_target: str | None = None,
    prepare_max_steps: int = 0,
    prepare_max_idle_polls: int = 40,
    predictor_config: PredictorRuntimeConfig | None = None,
    community_prior_config: CommunityPriorRuntimeConfig | None = None,
    strategic_model_config: StrategicRuntimeConfig | None = None,
    game_run_contract: GameRunContract | None = None,
    env_factory: Callable[[str, float], SupportsEnv] = _default_env_factory,
) -> CombatEvaluationReport:
    max_env_steps = normalize_optional_limit(max_env_steps)
    max_runs = normalize_optional_limit(max_runs)
    max_combats = normalize_optional_limit(max_combats)
    budgets = SessionBudgets(max_env_steps=max_env_steps, max_runs=max_runs, max_combats=max_combats)
    session_dir = Path(output_root) / (session_name or default_evaluation_session_name())
    session_dir.mkdir(parents=True, exist_ok=True)
    instance_id = instance_label_from_base_url(base_url)
    policy = build_policy_pack(
        policy_profile,
        predictor_config=predictor_config,
        community_prior_config=community_prior_config,
        strategic_model_config=strategic_model_config,
    )
    resolved_prepare_target = None if prepare_target is None else prepare_target.strip().lower()
    if resolved_prepare_target == "none":
        resolved_prepare_target = None
    if resolved_prepare_target not in {None, "main_menu", "character_select"}:
        raise ValueError("prepare_target must be one of: none, main_menu, character_select.")
    normalization_report: dict[str, object] | None = None
    if contract_requires_custom_run_prepare(game_run_contract):
        preparation = prepare_custom_run_from_contract(
            base_url=base_url,
            contract=game_run_contract,
            request_timeout_seconds=request_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=prepare_max_idle_polls,
            max_prepare_steps=None,
            env_factory=env_factory,
        )
        normalization_report = {
            "mode": "custom_run_prepare",
            **preparation.as_dict(),
        }
        resolved_prepare_target = "custom_run"
    elif resolved_prepare_target is not None:
        normalization = normalize_runtime_state(
            base_url=base_url,
            target=resolved_prepare_target,
            poll_interval_seconds=poll_interval_seconds,
            max_idle_polls=prepare_max_idle_polls,
            max_steps=prepare_max_steps,
            request_timeout_seconds=request_timeout_seconds,
            env_factory=env_factory,
        )
        normalization_report = normalization.as_dict()
        if not normalization.reached_target or normalization.final_observation is None:
            raise RuntimeError(
                "Failed to prepare policy-pack evaluation start. "
                f"target={resolved_prepare_target} initial={normalization.initial_screen} "
                f"final={normalization.final_screen} reason={normalization.stop_reason}"
            )
    env = env_factory(base_url, request_timeout_seconds)

    log_path = session_dir / "combat-eval.jsonl"
    summary_path = session_dir / "summary.json"
    combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
    recorder = TrajectorySessionRecorder(
        log_path=log_path,
        summary_path=summary_path,
        combat_outcomes_path=combat_outcomes_path,
        metadata=TrajectorySessionMetadata(
            session_name=session_dir.name,
            session_kind="eval",
            base_url=base_url,
            policy_name=policy.name,
            algorithm="heuristic",
            config=merge_game_run_contract_config(
                {
                    "policy_profile": policy_profile,
                    **budgets.as_dict(),
                    "poll_interval_seconds": poll_interval_seconds,
                    "max_idle_polls": max_idle_polls,
                    "prepare_target": resolved_prepare_target or "none",
                    "prepare_max_steps": prepare_max_steps if resolved_prepare_target is not None else None,
                    "prepare_max_idle_polls": prepare_max_idle_polls if resolved_prepare_target is not None else None,
                    "predictor": predictor_config.as_dict() if predictor_config is not None else None,
                    "community_prior": (
                        community_prior_config.as_dict() if community_prior_config is not None else None
                    ),
                    "strategic_model": (
                        strategic_model_config.as_dict() if strategic_model_config is not None else None
                    ),
                },
                game_run_contract,
            ),
            game_run_contract=game_run_contract,
        ),
    )
    if normalization_report is not None and normalization_report.get("mode") == "custom_run_prepare":
        recorder.metadata.config["custom_run_prepare"] = dict(normalization_report)
        recorder.logger.log_event(
            "custom_run_prepared",
            {
                "base_url": base_url,
                "custom_run_prepare": dict(normalization_report),
            },
        )

    env_steps = 0
    combat_steps = 0
    heuristic_steps = 0
    total_reward = 0.0
    heartbeat = ObservationHeartbeat()
    final_screen = "UNKNOWN"
    final_run_id = "run_unknown"
    final_observation = None
    stop_reason = "session_completed"

    try:
        while True:
            observation = env.observe()
            final_observation = observation
            final_screen = observation.screen_type
            final_run_id = observation.run_id
            heartbeat.observe(observation)
            recorder.sync_observation(observation, instance_id=instance_id, step_index=env_steps)
            contract_stop_reason = recorder.enforce_game_run_contract(
                instance_id=instance_id,
                step_index=env_steps,
            )
            if contract_stop_reason is not None:
                stop_reason = contract_stop_reason
                break
            budget_reason = _budget_stop_reason(budgets=budgets, env_steps=env_steps, recorder=recorder)
            if budget_reason is not None:
                stop_reason = budget_reason
                break

            if not observation.legal_actions:
                heartbeat.note_wait()
                if heartbeat.reached_idle_poll_limit(max_idle_polls):
                    stop_reason = "max_idle_polls_reached"
                    break
                sleep(poll_interval_seconds)
                continue

            decision = policy.choose(observation)
            if decision.action is None:
                heartbeat.note_wait()
                if heartbeat.reached_idle_poll_limit(max_idle_polls):
                    stop_reason = f"policy_no_action_timeout:{observation.screen_type.lower()}:{decision.reason}"
                    recorder.record_capability_diagnostics(
                        instance_id=instance_id,
                        step_index=env_steps,
                        observation=observation,
                        decision_reason=decision.reason,
                        stop_reason=stop_reason,
                        decision_metadata=decision.trace_metadata,
                    )
                    break
                sleep(poll_interval_seconds)
                continue

            result = env.step(decision.action)
            reward = 0.0
            if observation.screen_type == "COMBAT":
                reward = compute_combat_reward(observation, result.observation)
                combat_steps += 1
            result = result.model_copy(update={"reward": reward})
            total_reward += reward
            heuristic_steps += 1
            env_steps += 1
            final_observation = result.observation
            final_screen = result.observation.screen_type
            final_run_id = result.observation.run_id
            heartbeat.mark_progress(result.observation)

            recorder.log_step(
                instance_id=instance_id,
                step_index=env_steps,
                previous_observation=observation,
                result=result,
                chosen_action=decision.action,
                policy_name=policy.name,
                policy_pack=decision.policy_pack,
                policy_handler=decision.policy_handler,
                algorithm="heuristic",
                decision_source="heuristic",
                decision_stage=decision.stage,
                decision_reason=decision.reason,
                decision_score=decision.score,
                planner_name=decision.planner_name,
                planner_strategy=decision.planner_strategy,
                ranked_actions=[_ranked_action_payload(item) for item in decision.ranked_actions],
                decision_metadata=decision.trace_metadata,
                reward_source="combat_reward_v2" if observation.screen_type == "COMBAT" else "non_combat_transition",
            )
            recorder.sync_observation(result.observation, instance_id=instance_id, step_index=env_steps)
            contract_stop_reason = recorder.enforce_game_run_contract(
                instance_id=instance_id,
                step_index=env_steps,
            )
            if contract_stop_reason is not None:
                stop_reason = contract_stop_reason
                break

            budget_reason = _budget_stop_reason(budgets=budgets, env_steps=env_steps, recorder=recorder)
            if budget_reason is not None:
                stop_reason = budget_reason
                break
    finally:
        env.close()

    summary = recorder.finalize(
        instance_id=instance_id,
        stop_reason=stop_reason,
        step_index=env_steps,
        final_observation=final_observation,
    )
    combat_performance = _combat_performance_payload(
        summary=summary,
        combat_steps=combat_steps,
        total_reward=total_reward,
    )
    evaluation_summary = {
        **summary.as_dict(),
        "combat_steps": combat_steps,
        "heuristic_steps": heuristic_steps,
        "policy_profile": policy_profile,
        "prepare_target": resolved_prepare_target or "none",
        "normalization_report": normalization_report,
        "combat_outcomes_path": str(combat_outcomes_path),
        "log_path": str(log_path),
        "combat_performance": combat_performance,
    }
    summary_path.write_text(json.dumps(evaluation_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    recorder.logger.log_event(
        "policy_eval_stopped",
        {
            "env_steps": env_steps,
            "combat_steps": combat_steps,
            "heuristic_steps": heuristic_steps,
            "total_reward": total_reward,
            "final_screen": final_screen,
            "final_run_id": final_run_id,
            "policy_profile": policy_profile,
            "prepare_target": resolved_prepare_target or "none",
            "normalization_report": normalization_report,
            "summary_path": str(summary_path),
            "combat_performance": combat_performance,
        },
    )

    return CombatEvaluationReport(
        base_url=base_url,
        env_steps=env_steps,
        combat_steps=combat_steps,
        heuristic_steps=heuristic_steps,
        total_reward=total_reward,
        final_screen=final_screen,
        final_run_id=final_run_id,
        log_path=log_path,
        summary_path=summary_path,
        combat_outcomes_path=combat_outcomes_path,
        checkpoint_path=session_dir / "policy-pack-eval.json",
        combat_performance=combat_performance,
        stop_reason=summary.stop_reason,
        completed_run_count=summary.completed_run_count,
        completed_combat_count=summary.completed_combat_count,
        observed_run_seeds=list(summary.observed_run_seeds),
        observed_run_seed_histogram=dict(summary.observed_run_seed_histogram),
        runs_without_observed_seed=summary.runs_without_observed_seed,
        last_observed_seed=summary.last_observed_seed,
    )


def _ranked_action_payload(item) -> dict[str, object]:
    return {
        "action_id": item.action_id,
        "action": item.action,
        "score": item.score,
        "reason": item.reason,
        "metadata": dict(item.metadata),
    }


def default_training_session_name() -> str:
    return datetime.now(UTC).strftime("combat-dqn-%Y%m%d-%H%M%S")


def default_evaluation_session_name() -> str:
    return datetime.now(UTC).strftime("combat-dqn-eval-%Y%m%d-%H%M%S")


def instance_label_from_base_url(base_url: str) -> str:
    port = base_url.rsplit(":", maxsplit=1)[-1]
    if port.isdigit():
        return f"inst-{int(port):04d}"
    return "inst-local"


def _budget_stop_reason(
    *,
    budgets: SessionBudgets,
    env_steps: int,
    recorder: TrajectorySessionRecorder,
) -> str | None:
    progress = recorder.progress()
    return budgets.stop_reason(
        env_steps=env_steps,
        completed_run_count=progress.completed_run_count,
        completed_combat_count=progress.completed_combat_count,
    )
