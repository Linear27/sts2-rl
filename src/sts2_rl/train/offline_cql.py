from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any, Callable, Sequence

from sts2_rl.collect import CommunityPriorRuntimeConfig
from sts2_rl.collect.policy import build_policy_pack
from sts2_rl.data import (
    OFFLINE_RL_TRANSITIONS_FILENAME,
    TrajectorySessionMetadata,
    TrajectorySessionRecorder,
    load_dataset_summary,
    load_offline_rl_transition_records,
    resolve_dataset_split_paths,
)
from sts2_rl.data.offline_rl import OfflineRlTransitionRecord
from sts2_rl.game_run_contract import GameRunContract, merge_game_run_contract_config
from sts2_rl.lifecycle import ObservationHeartbeat, SessionBudgets, normalize_optional_limit
from sts2_rl.predict import PredictorRuntimeConfig
from sts2_rl.runtime.custom_run import contract_requires_custom_run_prepare, prepare_custom_run_from_contract

from .combat_encoder import CombatStateEncoder
from .combat_reward import compute_combat_reward
from .combat_space import CombatActionSpace
from .dqn import DqnAgent, DqnConfig
from .runner import CombatEvaluationReport, SupportsEnv, _budget_stop_reason, _combat_performance_payload, _default_env_factory

OFFLINE_CQL_CHECKPOINT_SCHEMA_VERSION = 1
OFFLINE_CQL_TRAINING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OfflineCqlTrainConfig:
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.97
    huber_delta: float = 1.0
    hidden_sizes: tuple[int, ...] = (64, 64)
    l2: float = 0.0001
    conservative_alpha: float = 1.0
    conservative_temperature: float = 1.0
    target_sync_interval: int = 50
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0
    early_stopping_patience: int = 8
    include_action_space_names: tuple[str, ...] = ("combat_v1",)
    min_floor: int | None = None
    max_floor: int | None = None
    min_reward: float | None = None
    max_reward: float | None = None
    live_base_url: str | None = None
    live_eval_max_env_steps: int | None = 0
    live_eval_max_runs: int | None = 1
    live_eval_max_combats: int | None = 0
    live_eval_poll_interval_seconds: float = 0.25
    live_eval_max_idle_polls: int = 40
    live_eval_request_timeout_seconds: float = 30.0
    benchmark_manifest_path: Path | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if self.huber_delta <= 0.0 or self.conservative_temperature <= 0.0:
            raise ValueError("huber_delta and conservative_temperature must be positive.")
        if self.l2 < 0.0 or self.conservative_alpha < 0.0:
            raise ValueError("l2 and conservative_alpha must be non-negative.")
        if not self.hidden_sizes or any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive values.")
        if self.validation_fraction < 0.0 or self.test_fraction < 0.0 or self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("validation_fraction + test_fraction must be in [0, 1).")
        if self.target_sync_interval < 1 or self.early_stopping_patience < 1:
            raise ValueError("target_sync_interval and early_stopping_patience must be positive.")
        if not self.include_action_space_names:
            raise ValueError("include_action_space_names must not be empty.")
        if self.min_floor is not None and self.max_floor is not None and self.min_floor > self.max_floor:
            raise ValueError("min_floor must be <= max_floor.")
        if self.min_reward is not None and self.max_reward is not None and self.min_reward > self.max_reward:
            raise ValueError("min_reward must be <= max_reward.")


@dataclass(frozen=True)
class OfflineCqlTrainingReport:
    output_dir: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    warmstart_checkpoint_path: Path
    metrics_path: Path
    summary_path: Path
    dataset_path: Path
    example_count: int
    train_example_count: int
    validation_example_count: int
    test_example_count: int
    feature_count: int
    action_count: int
    best_epoch: int
    split_strategy: str
    live_eval_summary_path: Path | None
    benchmark_summary_path: Path | None


@dataclass(frozen=True)
class _ResolvedOfflineDataset:
    dataset_path: Path
    records_path: Path | None
    train_path: Path | None
    validation_path: Path | None
    test_path: Path | None
    records: list[OfflineRlTransitionRecord]
    train_records: list[OfflineRlTransitionRecord]
    validation_records: list[OfflineRlTransitionRecord]
    test_records: list[OfflineRlTransitionRecord]
    split_strategy: str
    dropped_records: int
    dataset_summary: dict[str, Any] | None


@dataclass
class _OfflineMetricAccumulator:
    loss_sum: float = 0.0
    td_loss_sum: float = 0.0
    cql_penalty_sum: float = 0.0
    predicted_q_sum: float = 0.0
    target_q_sum: float = 0.0
    support_count: int = 0
    legal_action_count_sum: float = 0.0
    chosen_action_histogram: dict[int, int] = field(default_factory=dict)

    def add(self, metrics: dict[str, float | int | None], record: OfflineRlTransitionRecord) -> None:
        self.support_count += 1
        self.loss_sum += float(metrics.get("loss", 0.0) or 0.0)
        self.td_loss_sum += float(metrics.get("td_loss", 0.0) or 0.0)
        self.cql_penalty_sum += float(metrics.get("cql_penalty", 0.0) or 0.0)
        self.predicted_q_sum += float(metrics.get("predicted_q", 0.0) or 0.0)
        self.target_q_sum += float(metrics.get("target_q", 0.0) or 0.0)
        self.legal_action_count_sum += float(record.legal_action_count)
        if record.action_index is not None:
            self.chosen_action_histogram[record.action_index] = self.chosen_action_histogram.get(record.action_index, 0) + 1

    def as_dict(self, *, records: Sequence[OfflineRlTransitionRecord]) -> dict[str, Any]:
        count = self.support_count
        return {
            "example_count": len(records),
            "supported_example_count": count,
            "support_coverage": (count / len(records)) if records else None,
            "mean_loss": (self.loss_sum / count) if count else None,
            "mean_td_loss": (self.td_loss_sum / count) if count else None,
            "mean_cql_penalty": (self.cql_penalty_sum / count) if count else None,
            "mean_predicted_q": (self.predicted_q_sum / count) if count else None,
            "mean_target_q": (self.target_q_sum / count) if count else None,
            "mean_legal_action_count": (self.legal_action_count_sum / count) if count else None,
            "chosen_action_histogram": {str(key): value for key, value in sorted(self.chosen_action_histogram.items())},
            "return_proxy_stats": _episode_return_stats(records),
        }


def default_offline_cql_training_session_name() -> str:
    return datetime.now(UTC).strftime("offline-cql-%Y%m%d-%H%M%S")


def default_offline_cql_evaluation_session_name() -> str:
    return datetime.now(UTC).strftime("offline-cql-eval-%Y%m%d-%H%M%S")


def train_offline_cql_policy(
    *,
    dataset_source: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    config: OfflineCqlTrainConfig | None = None,
    benchmark_suite_name: str | None = None,
) -> OfflineCqlTrainingReport:
    train_config = config or OfflineCqlTrainConfig()
    dataset_path = Path(dataset_source).expanduser().resolve()
    resolved = _resolve_offline_dataset(dataset_path, config=train_config)
    if not resolved.train_records:
        raise ValueError("Offline CQL training requires at least one supported train transition.")

    output_dir = Path(output_root).expanduser().resolve() / (session_name or default_offline_cql_training_session_name())
    if output_dir.exists():
        raise FileExistsError(f"Offline CQL output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    action_space = CombatActionSpace()
    encoder = CombatStateEncoder()
    agent = DqnAgent(
        action_count=action_space.slot_count,
        feature_count=encoder.feature_count,
        config=DqnConfig(
            learning_rate=train_config.learning_rate,
            gamma=train_config.gamma,
            huber_delta=train_config.huber_delta,
            hidden_sizes=train_config.hidden_sizes,
            target_sync_interval=train_config.target_sync_interval,
            seed=train_config.seed,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay_steps=1,
            replay_capacity=max(64, len(resolved.train_records)),
            batch_size=max(1, min(train_config.batch_size, len(resolved.train_records))),
            min_replay_size=1,
            updates_per_env_step=1,
            prioritized_replay=False,
            n_step=1,
        ),
    )

    metrics_path = output_dir / "training-metrics.jsonl"
    checkpoint_path = output_dir / "offline-cql-checkpoint.json"
    best_checkpoint_path = output_dir / "offline-cql-best.json"
    warmstart_checkpoint_path = output_dir / "offline-cql-dqn-seed.json"
    best_epoch = 0
    best_validation_loss: float | None = None
    best_online_state = agent.online.state_dict()
    best_target_state = agent.target.state_dict()
    best_train_metrics: dict[str, Any] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    best_test_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    rng = random.Random(train_config.seed)

    with metrics_path.open("w", encoding="utf-8", newline="\n") as handle:
        for epoch in range(1, train_config.epochs + 1):
            train_records = list(resolved.train_records)
            rng.shuffle(train_records)
            train_metrics = _train_epoch(agent=agent, records=train_records, config=train_config)
            validation_metrics = evaluate_offline_cql_records(
                agent=agent,
                records=resolved.validation_records,
                gamma=train_config.gamma,
                conservative_alpha=train_config.conservative_alpha,
                conservative_temperature=train_config.conservative_temperature,
            )
            test_metrics = evaluate_offline_cql_records(
                agent=agent,
                records=resolved.test_records,
                gamma=train_config.gamma,
                conservative_alpha=train_config.conservative_alpha,
                conservative_temperature=train_config.conservative_temperature,
            )
            handle.write(json.dumps({"epoch": epoch, "train_metrics": train_metrics, "validation_metrics": validation_metrics, "test_metrics": test_metrics, "update_steps": agent.update_steps}, ensure_ascii=False))
            handle.write("\n")

            _save_offline_cql_checkpoint(
                agent=agent,
                path=checkpoint_path,
                metadata=_checkpoint_metadata_payload(
                    train_config=train_config,
                    dataset_path=resolved.dataset_path,
                    dataset_summary=resolved.dataset_summary,
                    split_strategy=resolved.split_strategy,
                    train_metrics=train_metrics,
                    validation_metrics=validation_metrics,
                    test_metrics=test_metrics,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    action_count=action_space.slot_count,
                    feature_count=encoder.feature_count,
                    warm_start_compatible_algorithms=["dqn", "offline_cql"],
                ),
            )

            validation_loss = validation_metrics.get("mean_loss")
            if validation_loss is None:
                validation_loss = train_metrics.get("mean_loss")
            if best_validation_loss is None or (validation_loss is not None and float(validation_loss) < best_validation_loss):
                best_validation_loss = None if validation_loss is None else float(validation_loss)
                best_epoch = epoch
                best_online_state = agent.online.state_dict()
                best_target_state = agent.target.state_dict()
                best_train_metrics = train_metrics
                best_validation_metrics = validation_metrics
                best_test_metrics = test_metrics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= train_config.early_stopping_patience:
                    break

    agent.online.load_state_dict(best_online_state)
    agent.target.load_state_dict(best_target_state)
    _save_offline_cql_checkpoint(
        agent=agent,
        path=best_checkpoint_path,
        metadata=_checkpoint_metadata_payload(
            train_config=train_config,
            dataset_path=resolved.dataset_path,
            dataset_summary=resolved.dataset_summary,
            split_strategy=resolved.split_strategy,
            train_metrics=best_train_metrics or {},
            validation_metrics=best_validation_metrics or {},
            test_metrics=best_test_metrics or {},
            epoch=best_epoch,
            best_epoch=best_epoch,
            action_count=action_space.slot_count,
            feature_count=encoder.feature_count,
            warm_start_compatible_algorithms=["dqn", "offline_cql"],
        ),
    )
    agent.save(
        warmstart_checkpoint_path,
        metadata={
            "exported_from": "offline_cql",
            "source_checkpoint_path": str(best_checkpoint_path),
            "best_epoch": best_epoch,
        },
    )

    live_eval_summary_path: Path | None = None
    if train_config.live_base_url is not None:
        live_report = run_offline_cql_evaluation(
            base_url=train_config.live_base_url,
            checkpoint_path=best_checkpoint_path,
            output_root=output_dir / "live-eval",
            session_name="best-live",
            max_env_steps=train_config.live_eval_max_env_steps,
            max_runs=train_config.live_eval_max_runs,
            max_combats=train_config.live_eval_max_combats,
            poll_interval_seconds=train_config.live_eval_poll_interval_seconds,
            max_idle_polls=train_config.live_eval_max_idle_polls,
            request_timeout_seconds=train_config.live_eval_request_timeout_seconds,
        )
        live_eval_summary_path = live_report.summary_path

    benchmark_summary_path: Path | None = None
    if train_config.benchmark_manifest_path is not None:
        from .benchmark_suite import run_benchmark_suite

        benchmark_report = run_benchmark_suite(
            train_config.benchmark_manifest_path,
            output_root=output_dir / "benchmarks",
            suite_name=benchmark_suite_name or "offline-cql-benchmark",
        )
        benchmark_summary_path = benchmark_report.summary_path

    summary_payload = {
        "schema_version": OFFLINE_CQL_TRAINING_SCHEMA_VERSION,
        "algorithm": "offline_cql",
        "dataset_path": str(resolved.dataset_path),
        "records_path": str(resolved.records_path) if resolved.records_path is not None else None,
        "train_path": str(resolved.train_path) if resolved.train_path is not None else None,
        "validation_path": str(resolved.validation_path) if resolved.validation_path is not None else None,
        "test_path": str(resolved.test_path) if resolved.test_path is not None else None,
        "split_strategy": resolved.split_strategy,
        "example_count": len(resolved.records),
        "train_example_count": len(resolved.train_records),
        "validation_example_count": len(resolved.validation_records),
        "test_example_count": len(resolved.test_records),
        "feature_count": encoder.feature_count,
        "action_count": action_space.slot_count,
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "warmstart_checkpoint_path": str(warmstart_checkpoint_path),
        "metrics_path": str(metrics_path),
        "live_eval_summary_path": str(live_eval_summary_path) if live_eval_summary_path is not None else None,
        "benchmark_summary_path": str(benchmark_summary_path) if benchmark_summary_path is not None else None,
        "train_metrics": best_train_metrics or {},
        "validation_metrics": best_validation_metrics or {},
        "test_metrics": best_test_metrics or {},
        "config": {
            "epochs": train_config.epochs,
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
            "gamma": train_config.gamma,
            "huber_delta": train_config.huber_delta,
            "hidden_sizes": list(train_config.hidden_sizes),
            "l2": train_config.l2,
            "conservative_alpha": train_config.conservative_alpha,
            "conservative_temperature": train_config.conservative_temperature,
            "target_sync_interval": train_config.target_sync_interval,
            "seed": train_config.seed,
            "early_stopping_patience": train_config.early_stopping_patience,
            "include_action_space_names": list(train_config.include_action_space_names),
            "min_floor": train_config.min_floor,
            "max_floor": train_config.max_floor,
            "min_reward": train_config.min_reward,
            "max_reward": train_config.max_reward,
            "benchmark_manifest_path": str(train_config.benchmark_manifest_path) if train_config.benchmark_manifest_path is not None else None,
        },
        "dataset_summary": resolved.dataset_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return OfflineCqlTrainingReport(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        warmstart_checkpoint_path=warmstart_checkpoint_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        dataset_path=resolved.dataset_path,
        example_count=len(resolved.records),
        train_example_count=len(resolved.train_records),
        validation_example_count=len(resolved.validation_records),
        test_example_count=len(resolved.test_records),
        feature_count=encoder.feature_count,
        action_count=action_space.slot_count,
        best_epoch=best_epoch,
        split_strategy=resolved.split_strategy,
        live_eval_summary_path=live_eval_summary_path,
        benchmark_summary_path=benchmark_summary_path,
    )


def evaluate_offline_cql_records(
    *,
    agent: DqnAgent,
    records: Sequence[OfflineRlTransitionRecord],
    gamma: float,
    conservative_alpha: float,
    conservative_temperature: float,
) -> dict[str, Any]:
    accumulator = _OfflineMetricAccumulator()
    for record in records:
        if not _is_supported_transition(record):
            continue
        metrics = _offline_cql_sample_metrics(
            agent=agent,
            record=record,
            gamma=gamma,
            huber_delta=agent.config.huber_delta,
            conservative_alpha=conservative_alpha,
            conservative_temperature=conservative_temperature,
        )
        accumulator.add(metrics, record)
    return accumulator.as_dict(records=records)


def run_offline_cql_evaluation(
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
    session_dir = Path(output_root) / (session_name or default_offline_cql_evaluation_session_name())
    session_dir.mkdir(parents=True, exist_ok=True)
    port = base_url.rsplit(":", maxsplit=1)[-1]
    instance_id = f"inst-{int(port):04d}" if port.isdigit() else "inst-local"

    encoder = CombatStateEncoder()
    action_space = CombatActionSpace()
    agent = DqnAgent.load(checkpoint_path, expected_action_count=action_space.slot_count, expected_feature_count=encoder.feature_count)
    checkpoint_metadata = _load_checkpoint_metadata(checkpoint_path)
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
            policy_name="offline-cql",
            algorithm="offline_cql",
            config=merge_game_run_contract_config(
                {
                    "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
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
                        policy_name="offline-cql",
                        algorithm="offline_cql",
                        decision_source="offline_cql",
                        decision_stage="combat",
                        decision_reason="greedy_q",
                        decision_score=q_values[action_index],
                        reward_source="combat_reward_v2",
                        model_metrics={"action_index": action_index, "q_value": q_values[action_index]},
                    )
                    recorder.sync_observation(result.observation, instance_id=instance_id, step_index=env_steps)
                    contract_stop_reason = recorder.enforce_game_run_contract(
                        instance_id=instance_id,
                        step_index=env_steps,
                    )
                    if contract_stop_reason is not None:
                        stop_reason = contract_stop_reason
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
                ranked_actions=[{"action_id": item.action_id, "action": item.action, "score": item.score, "reason": item.reason, "metadata": dict(item.metadata)} for item in decision.ranked_actions],
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
    finally:
        env.close()

    summary = recorder.finalize(instance_id=instance_id, stop_reason=stop_reason, step_index=env_steps, final_observation=final_observation)
    combat_performance = _combat_performance_payload(summary=summary, combat_steps=combat_steps, total_reward=total_reward)
    evaluation_summary = {
        **summary.as_dict(),
        "combat_steps": combat_steps,
        "heuristic_steps": heuristic_steps,
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "combat_outcomes_path": str(combat_outcomes_path),
        "log_path": str(log_path),
        "combat_performance": combat_performance,
        "checkpoint_metadata": checkpoint_metadata,
    }
    summary_path.write_text(json.dumps(evaluation_summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
        checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
        combat_performance=combat_performance,
        stop_reason=summary.stop_reason,
        completed_run_count=summary.completed_run_count,
        completed_combat_count=summary.completed_combat_count,
    )


def _resolve_offline_dataset(dataset_path: Path, *, config: OfflineCqlTrainConfig) -> _ResolvedOfflineDataset:
    records_path: Path | None = None
    train_path: Path | None = None
    validation_path: Path | None = None
    test_path: Path | None = None
    dataset_summary: dict[str, Any] | None = None
    if dataset_path.is_dir():
        try:
            dataset_summary = load_dataset_summary(dataset_path)
        except FileNotFoundError:
            dataset_summary = None
        if dataset_summary is not None and dataset_summary.get("dataset_kind") == "offline_rl_transitions":
            split_paths = resolve_dataset_split_paths(dataset_path)
            train_path = split_paths.get("train")
            validation_path = split_paths.get("validation")
            test_path = split_paths.get("test")
            records_path = dataset_path / OFFLINE_RL_TRANSITIONS_FILENAME
            all_records = _load_offline_records([path for path in (train_path, validation_path, test_path) if path is not None and path.exists()], config=config)
            train_records = _load_offline_records([train_path] if train_path is not None else [], config=config)
            validation_records = _load_offline_records([validation_path] if validation_path is not None else [], config=config)
            test_records = _load_offline_records([test_path] if test_path is not None else [], config=config)
            return _ResolvedOfflineDataset(
                dataset_path=dataset_path,
                records_path=records_path if records_path.exists() else None,
                train_path=train_path,
                validation_path=validation_path,
                test_path=test_path,
                records=all_records,
                train_records=train_records,
                validation_records=validation_records,
                test_records=test_records,
                split_strategy="manifest_split",
                dropped_records=max(0, int(dataset_summary.get("record_count", 0)) - len(all_records)),
                dataset_summary=dataset_summary,
            )
    records = _load_offline_records([dataset_path], config=config)
    train_records, validation_records, test_records = _split_offline_records(records, validation_fraction=config.validation_fraction, test_fraction=config.test_fraction, seed=config.seed)
    return _ResolvedOfflineDataset(
        dataset_path=dataset_path,
        records_path=dataset_path,
        train_path=None,
        validation_path=None,
        test_path=None,
        records=records,
        train_records=train_records,
        validation_records=validation_records,
        test_records=test_records,
        split_strategy="random_episode_fraction",
        dropped_records=0,
        dataset_summary=dataset_summary,
    )


def _load_offline_records(paths: Sequence[Path | None], *, config: OfflineCqlTrainConfig) -> list[OfflineRlTransitionRecord]:
    records: list[OfflineRlTransitionRecord] = []
    for path in paths:
        if path is None or not path.exists():
            continue
        for record in load_offline_rl_transition_records(path):
            if _record_matches_training_filters(record, config):
                records.append(record)
    return records


def _record_matches_training_filters(record: OfflineRlTransitionRecord, config: OfflineCqlTrainConfig) -> bool:
    if record.action_space_name is None or record.action_space_name not in set(config.include_action_space_names):
        return False
    if not _is_supported_transition(record):
        return False
    if config.min_floor is not None and (record.floor is None or record.floor < config.min_floor):
        return False
    if config.max_floor is not None and (record.floor is None or record.floor > config.max_floor):
        return False
    if config.min_reward is not None and record.reward < config.min_reward:
        return False
    if config.max_reward is not None and record.reward > config.max_reward:
        return False
    return True


def _split_offline_records(
    records: Sequence[OfflineRlTransitionRecord],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[OfflineRlTransitionRecord], list[OfflineRlTransitionRecord], list[OfflineRlTransitionRecord]]:
    grouped: dict[str, list[OfflineRlTransitionRecord]] = defaultdict(list)
    for record in records:
        grouped[record.episode_id].append(record)
    episode_ids = sorted(grouped)
    random.Random(seed).shuffle(episode_ids)
    total = len(episode_ids)
    validation_count = int(round(total * validation_fraction))
    test_count = int(round(total * test_fraction))
    train_count = total - validation_count - test_count
    if train_count <= 0:
        train_count = 1
        if validation_count > 0:
            validation_count -= 1
        elif test_count > 0:
            test_count -= 1
    train_ids = set(episode_ids[:train_count])
    validation_ids = set(episode_ids[train_count : train_count + validation_count])
    test_ids = set(episode_ids[train_count + validation_count :])
    return (
        [record for record in records if record.episode_id in train_ids],
        [record for record in records if record.episode_id in validation_ids],
        [record for record in records if record.episode_id in test_ids],
    )


def _train_epoch(*, agent: DqnAgent, records: Sequence[OfflineRlTransitionRecord], config: OfflineCqlTrainConfig) -> dict[str, Any]:
    accumulator = _OfflineMetricAccumulator()
    for start in range(0, len(records), config.batch_size):
        batch = records[start : start + config.batch_size]
        for record in batch:
            metrics = _apply_offline_cql_update(agent=agent, record=record, config=config)
            accumulator.add(metrics, record)
        agent.update_steps += 1
        if agent.update_steps % config.target_sync_interval == 0:
            agent.target.copy_from(agent.online)
    return accumulator.as_dict(records=records)


def _apply_offline_cql_update(*, agent: DqnAgent, record: OfflineRlTransitionRecord, config: OfflineCqlTrainConfig) -> dict[str, float | int | None]:
    metrics = _offline_cql_sample_metrics(
        agent=agent,
        record=record,
        gamma=config.gamma,
        huber_delta=config.huber_delta,
        conservative_alpha=config.conservative_alpha,
        conservative_temperature=config.conservative_temperature,
    )
    agent.online.train_on_output_deltas(
        features=list(record.feature_vector),
        output_deltas=list(metrics["output_deltas"]),
        learning_rate=config.learning_rate,
        l2=config.l2,
    )
    return metrics


def _offline_cql_sample_metrics(
    *,
    agent: DqnAgent,
    record: OfflineRlTransitionRecord,
    gamma: float,
    huber_delta: float,
    conservative_alpha: float,
    conservative_temperature: float,
) -> dict[str, Any]:
    q_values = agent.online.forward(list(record.feature_vector))
    predicted_q = q_values[record.action_index]
    target_q = record.reward
    if not record.done and record.next_feature_vector is not None and record.next_action_mask is not None:
        available = [index for index, enabled in enumerate(record.next_action_mask) if enabled]
        if available:
            next_target_q_values = agent.target.forward(list(record.next_feature_vector))
            target_q += gamma * max(next_target_q_values[index] for index in available)
    td_diff = predicted_q - target_q
    abs_diff = abs(td_diff)
    if abs_diff <= huber_delta:
        td_loss = 0.5 * td_diff * td_diff
        td_grad = td_diff
    else:
        td_loss = huber_delta * (abs_diff - (0.5 * huber_delta))
        td_grad = huber_delta if td_diff > 0 else -huber_delta

    available_actions = [index for index, enabled in enumerate(record.action_mask) if enabled]
    logits = [q_values[index] / conservative_temperature for index in available_actions]
    max_logit = max(logits) if logits else 0.0
    exp_values = [math.exp(value - max_logit) for value in logits]
    exp_sum = sum(exp_values)
    logsumexp = max_logit + math.log(exp_sum) if exp_sum > 0.0 else 0.0
    cql_penalty = (conservative_temperature * logsumexp) - predicted_q if available_actions else -predicted_q
    probabilities = [
        (value / exp_sum) if exp_sum > 0.0 else (1.0 / len(available_actions) if available_actions else 0.0)
        for value in exp_values
    ]
    output_deltas = [0.0] * len(q_values)
    output_deltas[record.action_index] += td_grad
    if available_actions:
        for slot_index, probability in zip(available_actions, probabilities, strict=True):
            output_deltas[slot_index] += conservative_alpha * probability
        output_deltas[record.action_index] -= conservative_alpha
    else:
        output_deltas[record.action_index] -= conservative_alpha
    return {
        "loss": td_loss + (conservative_alpha * cql_penalty),
        "td_loss": td_loss,
        "cql_penalty": cql_penalty,
        "predicted_q": predicted_q,
        "target_q": target_q,
        "output_deltas": output_deltas,
    }


def _save_offline_cql_checkpoint(*, agent: DqnAgent, path: Path, metadata: dict[str, Any]) -> Path:
    payload = {
        "schema_version": OFFLINE_CQL_CHECKPOINT_SCHEMA_VERSION,
        "algorithm": "offline_cql",
        "action_count": agent.action_count,
        "feature_count": agent.feature_count,
        "config": {
            "learning_rate": agent.config.learning_rate,
            "gamma": agent.config.gamma,
            "huber_delta": agent.config.huber_delta,
            "hidden_sizes": list(agent.config.hidden_sizes),
            "seed": agent.config.seed,
            "target_sync_interval": agent.config.target_sync_interval,
        },
        "env_steps": 0,
        "update_steps": agent.update_steps,
        "training_state": {"update_steps": agent.update_steps},
        "metadata": metadata,
        "online": agent.online.state_dict(),
        "target": agent.target.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _checkpoint_metadata_payload(
    *,
    train_config: OfflineCqlTrainConfig,
    dataset_path: Path,
    dataset_summary: dict[str, Any] | None,
    split_strategy: str,
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    epoch: int,
    best_epoch: int,
    action_count: int,
    feature_count: int,
    warm_start_compatible_algorithms: list[str],
) -> dict[str, Any]:
    return {
        "training_schema_version": OFFLINE_CQL_TRAINING_SCHEMA_VERSION,
        "epoch": epoch,
        "best_epoch": best_epoch,
        "dataset_path": str(dataset_path),
        "dataset_summary": dataset_summary,
        "split_strategy": split_strategy,
        "offline_metrics": {"train": train_metrics, "validation": validation_metrics, "test": test_metrics},
        "warm_start_compatible_algorithms": warm_start_compatible_algorithms,
        "model_shape": {"action_count": action_count, "feature_count": feature_count},
        "train_config": {
            "epochs": train_config.epochs,
            "batch_size": train_config.batch_size,
            "learning_rate": train_config.learning_rate,
            "gamma": train_config.gamma,
            "huber_delta": train_config.huber_delta,
            "hidden_sizes": list(train_config.hidden_sizes),
            "l2": train_config.l2,
            "conservative_alpha": train_config.conservative_alpha,
            "conservative_temperature": train_config.conservative_temperature,
            "target_sync_interval": train_config.target_sync_interval,
            "seed": train_config.seed,
            "early_stopping_patience": train_config.early_stopping_patience,
            "include_action_space_names": list(train_config.include_action_space_names),
            "min_floor": train_config.min_floor,
            "max_floor": train_config.max_floor,
            "min_reward": train_config.min_reward,
            "max_reward": train_config.max_reward,
        },
    }


def _load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return {
        "schema_version": payload.get("schema_version"),
        "algorithm": payload.get("algorithm"),
        "config": payload.get("config", {}),
        "training_state": payload.get("training_state", {}),
        "metadata": payload.get("metadata", {}),
    }


def _is_supported_transition(record: OfflineRlTransitionRecord) -> bool:
    return record.action_supported and record.action_index is not None and bool(record.feature_vector) and bool(record.action_mask)


def _episode_return_stats(records: Sequence[OfflineRlTransitionRecord]) -> dict[str, Any]:
    if not records:
        return {"count": 0, "min": None, "mean": None, "max": None}
    grouped: dict[str, float] = defaultdict(float)
    for record in records:
        grouped[record.episode_id] += record.reward
    values = [float(value) for value in grouped.values()]
    return {"count": len(values), "min": min(values), "mean": sum(values) / len(values), "max": max(values)}
