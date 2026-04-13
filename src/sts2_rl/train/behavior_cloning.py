from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from sts2_rl.collect.policy import PolicyDecision, RankedAction, card_text_score, estimate_block, estimate_damage, text_score
from sts2_rl.data import TrajectorySessionMetadata, TrajectorySessionRecorder, TrajectoryStepRecord, build_state_summary
from sts2_rl.data import load_dataset_summary, resolve_dataset_split_paths
from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import ActionRequest, AvailableActionsPayload, GameStatePayload
from sts2_rl.env.types import CandidateAction, StepObservation
from sts2_rl.env.wrapper import Sts2Env
from sts2_rl.game_run_contract import GameRunContract, merge_game_run_contract_config
from sts2_rl.lifecycle import ObservationHeartbeat, SessionBudgets, normalize_optional_limit
from sts2_rl.runtime.custom_run import contract_requires_custom_run_prepare, prepare_custom_run_from_contract

from .combat_reward import compute_combat_reward
from .runner import instance_label_from_base_url

if TYPE_CHECKING:
    from .runner import CombatEvaluationReport

BC_CHECKPOINT_SCHEMA_VERSION = 1
BC_TRAINING_SCHEMA_VERSION = 1
BC_FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BehaviorCloningFloorBandWeight:
    min_floor: int | None = None
    max_floor: int | None = None
    weight: float = 1.0

    def matches(self, floor: int | None) -> bool:
        if floor is None:
            return False
        if self.min_floor is not None and floor < self.min_floor:
            return False
        if self.max_floor is not None and floor > self.max_floor:
            return False
        return True


@dataclass(frozen=True)
class BehaviorCloningTrainConfig:
    epochs: int = 40
    learning_rate: float = 0.035
    l2: float = 0.0001
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0
    include_stages: tuple[str, ...] = ()
    include_decision_sources: tuple[str, ...] = ()
    include_policy_packs: tuple[str, ...] = ()
    include_policy_names: tuple[str, ...] = ()
    min_floor: int | None = None
    max_floor: int | None = None
    min_legal_actions: int = 2
    top_k: tuple[int, ...] = (1, 3)
    stage_weights: dict[str, float] = field(default_factory=dict)
    decision_source_weights: dict[str, float] = field(default_factory=dict)
    policy_pack_weights: dict[str, float] = field(default_factory=dict)
    policy_name_weights: dict[str, float] = field(default_factory=dict)
    run_outcome_weights: dict[str, float] = field(default_factory=dict)
    floor_band_weights: tuple[BehaviorCloningFloorBandWeight, ...] = ()
    benchmark_manifest_path: Path | None = None
    live_base_url: str | None = None
    live_eval_max_env_steps: int | None = 0
    live_eval_max_runs: int | None = 1
    live_eval_max_combats: int | None = 0
    live_eval_poll_interval_seconds: float = 0.25
    live_eval_max_idle_polls: int = 40
    live_eval_request_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.l2 < 0.0:
            raise ValueError("l2 must be non-negative.")
        if self.validation_fraction < 0.0 or self.test_fraction < 0.0:
            raise ValueError("validation_fraction and test_fraction must be non-negative.")
        if self.validation_fraction + self.test_fraction >= 1.0:
            raise ValueError("validation_fraction + test_fraction must be < 1.0.")
        if self.min_floor is not None and self.max_floor is not None and self.min_floor > self.max_floor:
            raise ValueError("min_floor must be <= max_floor.")
        if self.min_legal_actions < 1:
            raise ValueError("min_legal_actions must be positive.")
        if not self.top_k or any(value < 1 for value in self.top_k):
            raise ValueError("top_k must contain positive entries.")
        for mapping in (
            self.stage_weights,
            self.decision_source_weights,
            self.policy_pack_weights,
            self.policy_name_weights,
            self.run_outcome_weights,
        ):
            for value in mapping.values():
                if value <= 0.0:
                    raise ValueError("weight values must be positive.")
        for band in self.floor_band_weights:
            if band.weight <= 0.0:
                raise ValueError("floor_band_weights values must be positive.")


@dataclass(frozen=True)
class BehaviorCloningTrainingReport:
    output_dir: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    metrics_path: Path
    summary_path: Path
    dataset_path: Path
    example_count: int
    train_example_count: int
    validation_example_count: int
    test_example_count: int
    feature_count: int
    stage_count: int
    best_epoch: int
    split_strategy: str
    live_eval_summary_path: Path | None
    benchmark_summary_path: Path | None


@dataclass(frozen=True)
class _ResolvedBehaviorCloningDataset:
    dataset_path: Path
    examples_path: Path | None
    train_examples_path: Path | None
    validation_examples_path: Path | None
    test_examples_path: Path | None
    examples: list["BehaviorCloningExample"]
    train_examples: list["BehaviorCloningExample"]
    validation_examples: list["BehaviorCloningExample"]
    test_examples: list["BehaviorCloningExample"]
    split_strategy: str
    skipped_records: int
    dropped_records: int
    dataset_summary: dict[str, Any] | None


@dataclass(frozen=True)
class BehaviorCloningExample:
    example_id: str
    stage: str
    screen_type: str
    floor: int | None
    run_outcome: str | None
    decision_source: str | None
    policy_pack: str | None
    policy_name: str | None
    run_id: str
    chosen_action_id: str
    chosen_index: int
    candidate_ids: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    candidate_feature_maps: tuple[dict[str, float], ...]
    sample_weight: float
    source_path: Path


@dataclass(frozen=True)
class _ResolvedRecord:
    record: TrajectoryStepRecord
    observation: StepObservation | None
    legal_actions: tuple[CandidateAction, ...]
    chosen_index: int


@dataclass
class _SparseScoringHead:
    name: str
    weights: dict[str, float] = field(default_factory=dict)
    example_count: int = 0

    def score(self, feature_map: dict[str, float]) -> float:
        return sum(self.weights.get(name, 0.0) * value for name, value in feature_map.items())

    def clone(self) -> "_SparseScoringHead":
        return _SparseScoringHead(name=self.name, weights=dict(self.weights), example_count=self.example_count)


@dataclass(frozen=True)
class BehaviorCloningModel:
    global_head: _SparseScoringHead
    stage_heads: dict[str, _SparseScoringHead]
    metadata: dict[str, Any] = field(default_factory=dict)

    def choose(self, observation: StepObservation) -> PolicyDecision:
        if not observation.legal_actions:
            return PolicyDecision(
                action=None,
                stage=_stage_from_screen(observation.screen_type),
                reason="no_legal_actions",
                policy_pack="behavior_cloning",
                policy_handler="behavior-cloning:no-actions",
            )

        stage = _stage_from_screen(observation.screen_type)
        summary = build_state_summary(observation)
        candidate_features = [
            _candidate_feature_map(summary=summary, state=observation.state, candidate=candidate, stage=stage)
            for candidate in observation.legal_actions
        ]
        scores = [self.score_feature_map(stage=stage, feature_map=feature_map) for feature_map in candidate_features]
        probabilities = _softmax(scores)
        ranked_indices = sorted(
            range(len(observation.legal_actions)),
            key=lambda index: (-scores[index], observation.legal_actions[index].action_id),
        )
        best_index = ranked_indices[0]
        head_name = stage if stage in self.stage_heads else "global"
        return PolicyDecision(
            action=observation.legal_actions[best_index],
            stage=stage,
            reason="behavior_cloning_ranker",
            score=scores[best_index],
            policy_pack="behavior_cloning",
            policy_handler=f"behavior-cloning:{head_name}",
            ranked_actions=tuple(
                RankedAction(
                    action_id=observation.legal_actions[index].action_id,
                    action=observation.legal_actions[index].action,
                    score=scores[index],
                    reason="behavior_cloning_rank",
                    metadata={"probability": probabilities[index]},
                )
                for index in ranked_indices
            ),
            trace_metadata={
                "model_family": "behavior_cloning",
                "head_name": head_name,
                "candidate_count": len(observation.legal_actions),
                "selected_probability": probabilities[best_index],
                "feature_schema_version": BC_FEATURE_SCHEMA_VERSION,
            },
        )

    def score_feature_map(self, *, stage: str, feature_map: dict[str, float]) -> float:
        score = self.global_head.score(feature_map)
        stage_head = self.stage_heads.get(stage)
        if stage_head is not None:
            score += stage_head.score(feature_map)
        return score

    def save(self, path: str | Path) -> Path:
        checkpoint_path = Path(path).expanduser().resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": BC_CHECKPOINT_SCHEMA_VERSION,
            "algorithm": "behavior_cloning",
            "feature_schema_version": BC_FEATURE_SCHEMA_VERSION,
            "global_head": _head_payload(self.global_head),
            "stage_heads": {name: _head_payload(head) for name, head in sorted(self.stage_heads.items())},
            "metadata": self.metadata,
        }
        checkpoint_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return checkpoint_path

    @classmethod
    def load(cls, path: str | Path) -> "BehaviorCloningModel":
        checkpoint_path = Path(path).expanduser().resolve()
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != BC_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported behavior cloning checkpoint schema_version.")
        if payload.get("algorithm") != "behavior_cloning":
            raise ValueError(f"Unsupported checkpoint algorithm: {payload.get('algorithm')}")
        return cls(
            global_head=_head_from_payload(payload["global_head"]),
            stage_heads={
                str(name): _head_from_payload(head_payload)
                for name, head_payload in dict(payload.get("stage_heads", {})).items()
            },
            metadata=dict(payload.get("metadata", {})),
        )


def default_behavior_cloning_training_session_name() -> str:
    return datetime.now(UTC).strftime("behavior-cloning-%Y%m%d-%H%M%S")


def default_behavior_cloning_evaluation_session_name() -> str:
    return datetime.now(UTC).strftime("behavior-cloning-eval-%Y%m%d-%H%M%S")


def load_policy_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return {
        "schema_version": payload.get("schema_version"),
        "algorithm": payload.get("algorithm"),
        "feature_schema_version": payload.get("feature_schema_version"),
        "metadata": payload.get("metadata", {}),
    }


def train_behavior_cloning_policy(
    *,
    dataset_source: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    config: BehaviorCloningTrainConfig | None = None,
    benchmark_suite_name: str | None = None,
) -> BehaviorCloningTrainingReport:
    from .benchmark_suite import run_benchmark_suite

    train_config = config or BehaviorCloningTrainConfig()
    dataset_path = Path(dataset_source).expanduser().resolve()
    resolved = _resolve_behavior_cloning_dataset(dataset_path, config=train_config)
    if not resolved.train_examples:
        raise ValueError("Behavior cloning training requires at least one train example.")

    output_dir = Path(output_root).expanduser().resolve() / (
        session_name or default_behavior_cloning_training_session_name()
    )
    if output_dir.exists():
        raise FileExistsError(f"Behavior cloning output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    feature_inventory = sorted(
        {
            feature_name
            for example in resolved.examples
            for candidate_features in example.candidate_feature_maps
            for feature_name in candidate_features
        }
    )
    feature_hash = hashlib.sha256("\n".join(feature_inventory).encode("utf-8")).hexdigest()
    stage_names = sorted({example.stage for example in resolved.train_examples})

    global_head = _SparseScoringHead(name="global")
    stage_heads = {stage: _SparseScoringHead(name=stage) for stage in stage_names}
    for example in resolved.train_examples:
        global_head.example_count += 1
        stage_heads[example.stage].example_count += 1

    metrics_path = output_dir / "training-metrics.jsonl"
    best_epoch = 0
    best_objective: float | None = None
    best_global_head = global_head.clone()
    best_stage_heads = {name: head.clone() for name, head in stage_heads.items()}
    best_train_metrics: dict[str, Any] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    best_test_metrics: dict[str, Any] | None = None

    rng = random.Random(train_config.seed)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as metrics_handle:
        for epoch in range(1, train_config.epochs + 1):
            shuffled_examples = list(resolved.train_examples)
            rng.shuffle(shuffled_examples)
            for example in shuffled_examples:
                _train_on_example(
                    example=example,
                    global_head=global_head,
                    stage_head=stage_heads[example.stage],
                    learning_rate=train_config.learning_rate,
                    l2=train_config.l2,
                )

            epoch_model = BehaviorCloningModel(
                global_head=global_head.clone(),
                stage_heads={name: head.clone() for name, head in stage_heads.items()},
                metadata={},
            )
            train_metrics = evaluate_behavior_cloning_examples(epoch_model, resolved.train_examples, top_k=train_config.top_k)
            validation_metrics = evaluate_behavior_cloning_examples(
                epoch_model,
                resolved.validation_examples,
                top_k=train_config.top_k,
            )
            test_metrics = evaluate_behavior_cloning_examples(epoch_model, resolved.test_examples, top_k=train_config.top_k)
            objective = _select_behavior_cloning_objective(train_metrics, validation_metrics, test_metrics)
            is_best = best_objective is None or (objective is not None and objective < best_objective)
            if is_best:
                best_epoch = epoch
                best_objective = objective
                best_global_head = global_head.clone()
                best_stage_heads = {name: head.clone() for name, head in stage_heads.items()}
                best_train_metrics = train_metrics
                best_validation_metrics = validation_metrics
                best_test_metrics = test_metrics

            metrics_handle.write(
                json.dumps(
                    {
                        "schema_version": BC_TRAINING_SCHEMA_VERSION,
                        "epoch": epoch,
                        "objective": objective,
                        "selected_as_best": is_best,
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "test": test_metrics,
                    },
                    ensure_ascii=False,
                )
            )
            metrics_handle.write("\n")

    model_metadata = {
        "dataset_path": str(dataset_path),
        "examples_path": str(resolved.examples_path) if resolved.examples_path is not None else None,
        "train_examples_path": str(resolved.train_examples_path) if resolved.train_examples_path is not None else None,
        "validation_examples_path": (
            str(resolved.validation_examples_path) if resolved.validation_examples_path is not None else None
        ),
        "test_examples_path": str(resolved.test_examples_path) if resolved.test_examples_path is not None else None,
        "split_strategy": resolved.split_strategy,
        "feature_count": len(feature_inventory),
        "feature_hash": feature_hash,
        "feature_inventory_sample": feature_inventory[:50],
        "stage_names": stage_names,
        "dataset_summary": resolved.dataset_summary,
        "training_config": _behavior_cloning_config_payload(train_config),
        "best_epoch": best_epoch,
        "best_objective": best_objective,
        "train_metrics": best_train_metrics,
        "validation_metrics": best_validation_metrics,
        "test_metrics": best_test_metrics,
    }
    checkpoint_path = BehaviorCloningModel(
        global_head=global_head,
        stage_heads=stage_heads,
        metadata=model_metadata,
    ).save(output_dir / "behavior-cloning-checkpoint.json")
    best_checkpoint_path = BehaviorCloningModel(
        global_head=best_global_head,
        stage_heads=best_stage_heads,
        metadata=model_metadata,
    ).save(output_dir / "behavior-cloning-best.json")

    live_eval_summary_path: Path | None = None
    if train_config.live_base_url:
        live_eval_report = run_behavior_cloning_evaluation(
            base_url=train_config.live_base_url,
            checkpoint_path=best_checkpoint_path,
            output_root=output_dir / "live-eval",
            session_name="best-live-eval",
            max_env_steps=train_config.live_eval_max_env_steps,
            max_runs=train_config.live_eval_max_runs,
            max_combats=train_config.live_eval_max_combats,
            poll_interval_seconds=train_config.live_eval_poll_interval_seconds,
            max_idle_polls=train_config.live_eval_max_idle_polls,
            request_timeout_seconds=train_config.live_eval_request_timeout_seconds,
        )
        live_eval_summary_path = live_eval_report.summary_path

    benchmark_summary_path: Path | None = None
    if train_config.benchmark_manifest_path is not None:
        benchmark_report = run_benchmark_suite(
            train_config.benchmark_manifest_path,
            output_root=output_dir / "benchmarks",
            suite_name=benchmark_suite_name or "behavior-cloning-benchmark",
            replace_existing=True,
        )
        benchmark_summary_path = benchmark_report.summary_path

    summary_payload = {
        "schema_version": BC_TRAINING_SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "examples_path": str(resolved.examples_path) if resolved.examples_path is not None else None,
        "train_examples_path": str(resolved.train_examples_path) if resolved.train_examples_path is not None else None,
        "validation_examples_path": (
            str(resolved.validation_examples_path) if resolved.validation_examples_path is not None else None
        ),
        "test_examples_path": str(resolved.test_examples_path) if resolved.test_examples_path is not None else None,
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "metrics_path": str(metrics_path),
        "example_count": len(resolved.examples),
        "train_example_count": len(resolved.train_examples),
        "validation_example_count": len(resolved.validation_examples),
        "test_example_count": len(resolved.test_examples),
        "split_strategy": resolved.split_strategy,
        "feature_count": len(feature_inventory),
        "feature_hash": feature_hash,
        "stage_count": len(stage_names),
        "stage_names": stage_names,
        "best_epoch": best_epoch,
        "best_objective": best_objective,
        "skipped_records": resolved.skipped_records,
        "dropped_records": resolved.dropped_records,
        "config": _behavior_cloning_config_payload(train_config),
        "train": best_train_metrics,
        "validation": best_validation_metrics,
        "test": best_test_metrics,
        "dataset_lineage": None if resolved.dataset_summary is None else resolved.dataset_summary.get("lineage"),
        "live_eval_summary_path": str(live_eval_summary_path) if live_eval_summary_path is not None else None,
        "benchmark_summary_path": str(benchmark_summary_path) if benchmark_summary_path is not None else None,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return BehaviorCloningTrainingReport(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        dataset_path=dataset_path,
        example_count=len(resolved.examples),
        train_example_count=len(resolved.train_examples),
        validation_example_count=len(resolved.validation_examples),
        test_example_count=len(resolved.test_examples),
        feature_count=len(feature_inventory),
        stage_count=len(stage_names),
        best_epoch=best_epoch,
        split_strategy=resolved.split_strategy,
        live_eval_summary_path=live_eval_summary_path,
        benchmark_summary_path=benchmark_summary_path,
    )


def run_behavior_cloning_evaluation(
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
    game_run_contract: GameRunContract | None = None,
    env_factory: Callable[[str, float], Any] | None = None,
) -> CombatEvaluationReport:
    from .runner import CombatEvaluationReport

    max_env_steps = normalize_optional_limit(max_env_steps)
    max_runs = normalize_optional_limit(max_runs)
    max_combats = normalize_optional_limit(max_combats)
    budgets = SessionBudgets(max_env_steps=max_env_steps, max_runs=max_runs, max_combats=max_combats)
    session_dir = Path(output_root).expanduser().resolve() / (
        session_name or default_behavior_cloning_evaluation_session_name()
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    instance_id = instance_label_from_base_url(base_url)
    model = BehaviorCloningModel.load(checkpoint_path)
    checkpoint_metadata = load_policy_checkpoint_metadata(checkpoint_path)
    env_builder = env_factory or _default_env_factory
    env = env_builder(base_url, request_timeout_seconds)

    log_path = session_dir / "behavior-cloning-eval.jsonl"
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
            policy_name="behavior-cloning",
            algorithm="behavior_cloning",
            config=merge_game_run_contract_config(
                {
                    "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
                    "checkpoint_metadata": checkpoint_metadata,
                    **budgets.as_dict(),
                    "poll_interval_seconds": poll_interval_seconds,
                    "max_idle_polls": max_idle_polls,
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
            env_factory=env_builder,
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
            progress = recorder.progress()
            budget_reason = budgets.stop_reason(
                env_steps=env_steps,
                completed_run_count=progress.completed_run_count,
                completed_combat_count=progress.completed_combat_count,
            )
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

            decision = model.choose(observation)
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
                policy_name="behavior-cloning",
                policy_pack=decision.policy_pack,
                policy_handler=decision.policy_handler,
                algorithm="behavior_cloning",
                decision_source="behavior_cloning",
                decision_stage=decision.stage,
                decision_reason=decision.reason,
                decision_score=decision.score,
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
    finally:
        env.close()

    summary = recorder.finalize(
        instance_id=instance_id,
        stop_reason=stop_reason,
        step_index=env_steps,
        final_observation=final_observation,
    )
    combat_performance = _combat_performance_payload(
        completed_combat_count=summary.completed_combat_count,
        won_combats=summary.won_combats,
        lost_combats=summary.lost_combats,
        completed_run_count=summary.completed_run_count,
        won_runs=summary.won_runs,
        total_reward=total_reward,
        combat_steps=combat_steps,
    )
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


def evaluate_behavior_cloning_examples(
    model: BehaviorCloningModel,
    examples: Sequence[BehaviorCloningExample],
    *,
    top_k: Sequence[int] = (1, 3),
) -> dict[str, Any]:
    if not examples:
        return {
            "example_count": 0,
            "weighted_example_count": 0.0,
            "loss": None,
            "top_k_accuracy": {str(value): None for value in top_k},
            "mean_candidate_count": None,
            "mean_chosen_rank": None,
            "mean_chosen_probability": None,
            "mean_entropy": None,
            "stage_metrics": {},
        }

    stage_examples: dict[str, list[BehaviorCloningExample]] = defaultdict(list)
    metrics = _MetricAccumulator(top_k=top_k)
    for example in examples:
        stage_examples[example.stage].append(example)
        _accumulate_example_metrics(metrics, model, example, top_k=top_k)
    return metrics.as_dict(
        stage_metrics={
            stage: _evaluate_stage(model, stage_items, top_k=top_k)
            for stage, stage_items in sorted(stage_examples.items())
        }
    )


def _resolve_behavior_cloning_dataset(
    dataset_path: Path,
    *,
    config: BehaviorCloningTrainConfig,
) -> _ResolvedBehaviorCloningDataset:
    examples_path: Path | None = None
    train_examples_path: Path | None = None
    validation_examples_path: Path | None = None
    test_examples_path: Path | None = None
    dataset_summary: dict[str, Any] | None = None

    if dataset_path.is_dir():
        try:
            dataset_summary = load_dataset_summary(dataset_path)
        except FileNotFoundError:
            dataset_summary = None
        if dataset_summary is not None and dataset_summary.get("dataset_kind") == "trajectory_steps":
            split_paths = resolve_dataset_split_paths(dataset_path)
            train_examples_path = split_paths.get("train")
            validation_examples_path = split_paths.get("validation")
            test_examples_path = split_paths.get("test")
            if train_examples_path is not None:
                records_path_raw = dataset_summary.get("records_path")
                if records_path_raw:
                    records_path = Path(str(records_path_raw)).expanduser().resolve()
                    if records_path.exists():
                        examples_path = records_path
                all_examples, all_skipped, all_dropped = _load_behavior_cloning_examples_from_paths(
                    [
                        path
                        for path in (train_examples_path, validation_examples_path, test_examples_path)
                        if path is not None and path.exists()
                    ],
                    config=config,
                )
                train_examples, _, _ = _load_behavior_cloning_examples_from_paths(
                    [train_examples_path],
                    config=config,
                )
                validation_examples, _, _ = _load_behavior_cloning_examples_from_paths(
                    [validation_examples_path] if validation_examples_path is not None else [],
                    config=config,
                )
                test_examples, _, _ = _load_behavior_cloning_examples_from_paths(
                    [test_examples_path] if test_examples_path is not None else [],
                    config=config,
                )
                return _ResolvedBehaviorCloningDataset(
                    dataset_path=dataset_path,
                    examples_path=examples_path,
                    train_examples_path=train_examples_path,
                    validation_examples_path=validation_examples_path,
                    test_examples_path=test_examples_path,
                    examples=all_examples,
                    train_examples=train_examples,
                    validation_examples=validation_examples,
                    test_examples=test_examples,
                    split_strategy="manifest_split",
                    skipped_records=all_skipped,
                    dropped_records=all_dropped,
                    dataset_summary=dataset_summary,
                )

    examples, skipped_records, dropped_records = _load_behavior_cloning_examples_from_paths([dataset_path], config=config)
    train_examples, validation_examples, test_examples = _random_split_behavior_cloning_examples(
        examples,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    return _ResolvedBehaviorCloningDataset(
        dataset_path=dataset_path,
        examples_path=examples_path if examples_path is not None else dataset_path,
        train_examples_path=None,
        validation_examples_path=None,
        test_examples_path=None,
        examples=examples,
        train_examples=train_examples,
        validation_examples=validation_examples,
        test_examples=test_examples,
        split_strategy="random_fraction",
        skipped_records=skipped_records,
        dropped_records=dropped_records,
        dataset_summary=dataset_summary,
    )


def _load_behavior_cloning_examples_from_paths(
    paths: Sequence[Path | None],
    *,
    config: BehaviorCloningTrainConfig,
) -> tuple[list[BehaviorCloningExample], int, int]:
    examples: list[BehaviorCloningExample] = []
    skipped_records = 0
    dropped_records = 0
    for path in paths:
        if path is None or not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("record_type") != "step":
                    skipped_records += 1
                    continue
                record = TrajectoryStepRecord.model_validate(payload)
                example = _example_from_step_record(record, source_path=path, config=config)
                if example is None:
                    dropped_records += 1
                    continue
                examples.append(example)
    return examples, skipped_records, dropped_records


def _example_from_step_record(
    record: TrajectoryStepRecord,
    *,
    source_path: Path,
    config: BehaviorCloningTrainConfig,
) -> BehaviorCloningExample | None:
    stage = _stage_from_record(record)
    decision_source = _normalized_optional(record.decision_source)
    policy_pack = _normalized_optional(record.policy_pack)
    policy_name = _normalized_optional(record.policy_name)
    run_outcome = _normalized_optional((record.info or {}).get("run_outcome"))
    if isinstance(record.state, dict):
        run_outcome = run_outcome or _normalized_optional(record.state.get("run_outcome"))
    if config.include_stages and stage not in {item.lower() for item in config.include_stages}:
        return None
    if config.include_decision_sources and decision_source not in {item.lower() for item in config.include_decision_sources}:
        return None
    if config.include_policy_packs and policy_pack not in {item.lower() for item in config.include_policy_packs}:
        return None
    if config.include_policy_names and policy_name not in {item.lower() for item in config.include_policy_names}:
        return None
    if config.min_floor is not None and (record.floor is None or record.floor < config.min_floor):
        return None
    if config.max_floor is not None and (record.floor is None or record.floor > config.max_floor):
        return None
    if record.chosen_action_id is None:
        return None

    resolved = _resolve_record_candidates(record)
    if resolved is None or len(resolved.legal_actions) < config.min_legal_actions:
        return None

    state = resolved.observation.state if resolved.observation is not None else _safe_game_state(record.state)
    summary = record.state_summary if record.state_summary else (
        {} if resolved.observation is None else build_state_summary(resolved.observation)
    )
    sample_weight = _sample_weight_for_record(
        record=record,
        config=config,
        stage=stage,
        decision_source=decision_source,
        policy_pack=policy_pack,
        policy_name=policy_name,
        run_outcome=run_outcome,
    )
    if sample_weight <= 0.0:
        return None

    return BehaviorCloningExample(
        example_id=f"{record.session_name}:{record.instance_id}:{record.run_id}:{record.step_index}",
        stage=stage,
        screen_type=record.screen_type,
        floor=record.floor,
        run_outcome=run_outcome,
        decision_source=decision_source,
        policy_pack=policy_pack,
        policy_name=policy_name,
        run_id=record.run_id,
        chosen_action_id=record.chosen_action_id,
        chosen_index=resolved.chosen_index,
        candidate_ids=tuple(candidate.action_id for candidate in resolved.legal_actions),
        candidate_labels=tuple(candidate.label for candidate in resolved.legal_actions),
        candidate_feature_maps=tuple(
            _candidate_feature_map(summary=summary, state=state, candidate=candidate, stage=stage)
            for candidate in resolved.legal_actions
        ),
        sample_weight=sample_weight,
        source_path=source_path,
    )


def _resolve_record_candidates(record: TrajectoryStepRecord) -> _ResolvedRecord | None:
    state = _safe_game_state(record.state)
    descriptors = _safe_action_descriptors(record.action_descriptors)
    observation: StepObservation | None = None
    legal_actions: tuple[CandidateAction, ...] = ()

    if state is not None and descriptors is not None:
        build_result = build_candidate_actions(state, descriptors)
        candidates = list(build_result.candidates)
        candidate_by_id = {candidate.action_id: candidate for candidate in candidates}
        if record.legal_action_ids and all(action_id in candidate_by_id for action_id in record.legal_action_ids):
            ordered_candidates = [candidate_by_id[action_id] for action_id in record.legal_action_ids]
        else:
            ordered_candidates = candidates
        observation = StepObservation(
            screen_type=state.screen,
            run_id=state.run_id,
            state=state,
            action_descriptors=descriptors,
            legal_actions=ordered_candidates,
            build_warnings=build_result.unsupported_actions,
        )
        legal_actions = tuple(ordered_candidates)

    if not legal_actions:
        legal_actions = tuple(_fallback_record_candidates(record))
    if not legal_actions:
        return None

    chosen_index = next(
        (index for index, candidate in enumerate(legal_actions) if candidate.action_id == record.chosen_action_id),
        -1,
    )
    if chosen_index < 0:
        return None
    return _ResolvedRecord(record=record, observation=observation, legal_actions=legal_actions, chosen_index=chosen_index)


def _fallback_record_candidates(record: TrajectoryStepRecord) -> list[CandidateAction]:
    candidates: list[CandidateAction] = []
    if record.chosen_action_id and record.chosen_action:
        request = ActionRequest.model_validate(record.chosen_action)
        candidates.append(
            CandidateAction(
                action_id=record.chosen_action_id,
                action=request.action,
                label=record.chosen_action_label or record.chosen_action_id,
                request=request,
                source=record.chosen_action_source or "recorded",
                metadata={"fallback_reconstructed": True},
            )
        )
    for action_id in record.legal_action_ids:
        if any(candidate.action_id == action_id for candidate in candidates):
            continue
        candidates.append(
            _candidate_from_action_id(
                action_id,
                label=record.chosen_action_label if action_id == record.chosen_action_id else None,
                source="recorded",
            )
        )
    return candidates


def _candidate_feature_map(
    *,
    summary: dict[str, Any],
    state: GameStatePayload | None,
    candidate: CandidateAction,
    stage: str,
) -> dict[str, float]:
    features: dict[str, float] = {"bias": 1.0}
    _add_one_hot(features, "stage", stage)
    _add_one_hot(features, "screen", str(summary.get("screen_type") or stage).lower())
    _add_one_hot(features, "action", candidate.action)
    _add_one_hot(features, "action_source", candidate.source)
    _add_numeric(features, "legal_action_count", float(summary.get("available_action_count") or 0) / 10.0)

    run_summary = summary.get("run") if isinstance(summary.get("run"), dict) else {}
    if run_summary:
        floor = run_summary.get("floor")
        current_hp = float(run_summary.get("current_hp") or 0.0)
        max_hp = max(float(run_summary.get("max_hp") or 0.0), 1.0)
        _add_one_hot(features, "character", str(run_summary.get("character_id") or "unknown").lower())
        _add_numeric(features, "floor", (float(floor) if floor is not None else 0.0) / 60.0)
        _add_numeric(features, "hp_ratio", current_hp / max_hp)
        _add_numeric(features, "gold", float(run_summary.get("gold") or 0.0) / 300.0)
        _add_numeric(features, "max_energy", float(run_summary.get("max_energy") or 0.0) / 6.0)
        _add_numeric(features, "deck_size", float(run_summary.get("deck_size") or 0.0) / 40.0)
        _bucketize(features, "floor_band", float(floor) if floor is not None else 0.0, (4, 8, 16, 32, 64))

    combat_summary = summary.get("combat") if isinstance(summary.get("combat"), dict) else {}
    if combat_summary:
        enemy_hp = [float(value) for value in combat_summary.get("enemy_hp", [])]
        _add_numeric(features, "combat_player_hp", float(combat_summary.get("player_hp") or 0.0) / 100.0)
        _add_numeric(features, "combat_player_block", float(combat_summary.get("player_block") or 0.0) / 50.0)
        _add_numeric(features, "combat_energy", float(combat_summary.get("energy") or 0.0) / 5.0)
        _add_numeric(features, "combat_enemy_count", float(len(enemy_hp)) / 5.0)
        _add_numeric(features, "combat_enemy_hp_sum", sum(enemy_hp) / 200.0)
        _add_numeric(features, "combat_playable_hand_count", float(combat_summary.get("playable_hand_count") or 0.0) / 10.0)

    _add_flag(features, "candidate_has_target", candidate.request.target_index is not None)
    _add_flag(features, "candidate_has_card_index", candidate.request.card_index is not None)
    _add_flag(features, "candidate_has_option_index", candidate.request.option_index is not None)

    if candidate.action == "play_card":
        _add_play_card_features(features, state, candidate)
    elif candidate.action in {"choose_reward_card", "select_deck_card", "buy_card"}:
        _add_card_option_features(features, state, candidate)
    elif candidate.action == "choose_map_node":
        _add_map_node_features(features, state, candidate)
    elif candidate.action == "claim_reward":
        _add_reward_claim_features(features, state, candidate)
    elif candidate.action == "choose_event_option":
        _add_event_option_features(features, state, candidate)
    elif candidate.action == "choose_rest_option":
        _add_rest_option_features(features, state, candidate)
    elif candidate.action in {"buy_relic", "choose_treasure_relic"}:
        _add_relic_features(features, state, candidate)
    elif candidate.action in {"buy_potion", "use_potion", "discard_potion"}:
        _add_potion_features(features, state, candidate)
    elif candidate.action == "select_character":
        _add_character_features(features, state, candidate)
    elif candidate.action == "skip_reward_cards":
        _add_flag(features, "candidate_skip_reward_cards")
    elif candidate.action == "end_turn":
        _add_flag(features, "candidate_end_turn")
    return features


def _add_play_card_features(features: dict[str, float], state: GameStatePayload | None, candidate: CandidateAction) -> None:
    if state is None or state.combat is None or candidate.request.card_index is None:
        return
    card = next((item for item in state.combat.hand if item.index == candidate.request.card_index), None)
    if card is None:
        return
    rules_text = card.resolved_rules_text or card.rules_text
    _add_one_hot(features, "card_id", card.card_id.lower())
    _add_numeric(features, "card_energy_cost", float(card.energy_cost) / 4.0)
    _add_numeric(features, "card_text_score", card_text_score(card.card_id, card.name, rules_text, upgraded=card.upgraded) / 10.0)
    _add_numeric(features, "card_estimated_damage", float(estimate_damage(rules_text)) / 50.0)
    _add_numeric(features, "card_estimated_block", float(estimate_block(rules_text)) / 50.0)
    _add_flag(features, "card_upgraded", card.upgraded)
    if candidate.request.target_index is not None:
        enemy = next((item for item in state.combat.enemies if item.index == candidate.request.target_index), None)
        if enemy is not None:
            _add_one_hot(features, "target_enemy_id", enemy.enemy_id.lower())
            _add_numeric(features, "target_enemy_hp", float(enemy.current_hp) / 100.0)
            _add_flag(features, "target_lethal", estimate_damage(rules_text) >= enemy.current_hp > 0)


def _add_card_option_features(features: dict[str, float], state: GameStatePayload | None, candidate: CandidateAction) -> None:
    option_index = candidate.request.option_index
    if option_index is None or state is None:
        return
    card = None
    rules_text = ""
    if candidate.action == "choose_reward_card" and state.reward is not None:
        card = next((item for item in state.reward.card_options if item.index == option_index), None)
        if card is not None:
            rules_text = card.resolved_rules_text or card.rules_text
    elif candidate.action == "select_deck_card" and state.selection is not None:
        card = next((item for item in state.selection.cards if item.index == option_index), None)
    elif candidate.action == "buy_card" and state.shop is not None:
        card = next((item for item in state.shop.cards if item.index == option_index), None)
        if card is not None:
            _add_numeric(features, "shop_card_price", float(card.price) / 250.0)
    if card is None:
        return
    _add_one_hot(features, "card_id", card.card_id.lower())
    _add_numeric(features, "card_text_score", card_text_score(card.card_id, card.name, rules_text, upgraded=card.upgraded) / 10.0)
    _add_flag(features, "card_upgraded", card.upgraded)


def _add_map_node_features(features: dict[str, float], state: GameStatePayload | None, candidate: CandidateAction) -> None:
    option_index = candidate.request.option_index
    if option_index is None or state is None or state.map is None:
        return
    node = next((item for item in state.map.available_nodes if item.index == option_index), None)
    if node is None:
        return
    node_kind = _map_node_kind(node.node_type)
    _add_one_hot(features, "map_node_type", node_kind)
    _add_numeric(features, "map_row", float(node.row) / 20.0)
    if state.run is not None and state.run.max_hp > 0:
        hp_ratio = state.run.current_hp / state.run.max_hp
        hp_band = "low" if hp_ratio <= 0.45 else "high"
        _add_one_hot(features, "map_node_type_hp_band", f"{node_kind}:{hp_band}")


def _add_reward_claim_features(features: dict[str, float], state: GameStatePayload | None, candidate: CandidateAction) -> None:
    option_index = candidate.request.option_index
    if option_index is None or state is None or state.reward is None:
        return
    reward = next((item for item in state.reward.rewards if item.index == option_index), None)
    if reward is None:
        return
    _add_one_hot(features, "reward_type", reward.reward_type.lower())
    _add_numeric(features, "reward_text_score", text_score(reward.description or "") / 10.0)


def _add_event_option_features(features: dict[str, float], state: GameStatePayload | None, candidate: CandidateAction) -> None:
    option_index = candidate.request.option_index
    if option_index is None or state is None or state.event is None:
        return
    option = next((item for item in state.event.options if item.index == option_index), None)
    if option is None:
        return
    _add_flag(features, "event_option_is_proceed", option.is_proceed)
    _add_flag(features, "event_option_has_relic_preview", option.has_relic_preview)
    _add_numeric(features, "event_option_text_score", text_score(f"{option.title} {option.description}") / 10.0)


def _add_rest_option_features(features: dict[str, float], state: GameStatePayload | None, candidate: CandidateAction) -> None:
    option_index = candidate.request.option_index
    if option_index is None or state is None or state.rest is None:
        return
    option = next((item for item in state.rest.options if item.index == option_index), None)
    if option is None:
        return
    _add_one_hot(features, "rest_option_id", option.option_id.lower())
    _add_numeric(features, "rest_text_score", text_score(f"{option.title} {option.description}") / 10.0)


def _add_relic_features(features: dict[str, float], state: GameStatePayload | None, candidate: CandidateAction) -> None:
    option_index = candidate.request.option_index
    if option_index is None or state is None:
        return
    relic = None
    if candidate.action == "buy_relic" and state.shop is not None:
        relic = next((item for item in state.shop.relics if item.index == option_index), None)
        if relic is not None:
            _add_numeric(features, "relic_price", float(relic.price) / 350.0)
    elif candidate.action == "choose_treasure_relic" and state.chest is not None:
        relic = next((item for item in state.chest.relic_options if item.index == option_index), None)
    if relic is None:
        return
    _add_one_hot(features, "relic_id", relic.relic_id.lower())


def _add_potion_features(features: dict[str, float], state: GameStatePayload | None, candidate: CandidateAction) -> None:
    option_index = candidate.request.option_index
    if option_index is None or state is None:
        return
    potion = None
    if candidate.action in {"use_potion", "discard_potion"} and state.run is not None:
        potion = next((item for item in state.run.potions if item.index == option_index), None)
    elif candidate.action == "buy_potion" and state.shop is not None:
        potion = next((item for item in state.shop.potions if item.index == option_index), None)
    if potion is None:
        return
    _add_one_hot(features, "potion_id", str(potion.potion_id or "unknown").lower())


def _add_character_features(features: dict[str, float], state: GameStatePayload | None, candidate: CandidateAction) -> None:
    option_index = candidate.request.option_index
    if option_index is None or state is None or state.character_select is None:
        return
    character = next((item for item in state.character_select.characters if item.index == option_index), None)
    if character is not None:
        _add_one_hot(features, "select_character_id", character.character_id.lower())


def _sample_weight_for_record(
    *,
    record: TrajectoryStepRecord,
    config: BehaviorCloningTrainConfig,
    stage: str,
    decision_source: str | None,
    policy_pack: str | None,
    policy_name: str | None,
    run_outcome: str | None,
) -> float:
    weight = 1.0
    weight *= config.stage_weights.get(stage, 1.0)
    if decision_source is not None:
        weight *= config.decision_source_weights.get(decision_source, 1.0)
    if policy_pack is not None:
        weight *= config.policy_pack_weights.get(policy_pack, 1.0)
    if policy_name is not None:
        weight *= config.policy_name_weights.get(policy_name, 1.0)
    if run_outcome is not None:
        weight *= config.run_outcome_weights.get(run_outcome, 1.0)
    for band in config.floor_band_weights:
        if band.matches(record.floor):
            weight *= band.weight
    return weight


def _random_split_behavior_cloning_examples(
    examples: Sequence[BehaviorCloningExample],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[BehaviorCloningExample], list[BehaviorCloningExample], list[BehaviorCloningExample]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    if len(indices) <= 1:
        return [examples[index] for index in indices], [], []
    validation_count = int(round(len(indices) * validation_fraction))
    test_count = int(round(len(indices) * test_fraction))
    if validation_count + test_count >= len(indices):
        overflow = (validation_count + test_count) - (len(indices) - 1)
        if overflow > 0:
            if test_count >= overflow:
                test_count -= overflow
            else:
                validation_count = max(0, validation_count - (overflow - test_count))
                test_count = 0
    validation_indices = set(indices[:validation_count])
    test_indices = set(indices[validation_count : validation_count + test_count])
    train_examples = [examples[index] for index in indices if index not in validation_indices and index not in test_indices]
    validation_examples = [examples[index] for index in indices if index in validation_indices]
    test_examples = [examples[index] for index in indices if index in test_indices]
    return train_examples, validation_examples, test_examples


def _train_on_example(
    *,
    example: BehaviorCloningExample,
    global_head: _SparseScoringHead,
    stage_head: _SparseScoringHead,
    learning_rate: float,
    l2: float,
) -> None:
    scores = [global_head.score(feature_map) + stage_head.score(feature_map) for feature_map in example.candidate_feature_maps]
    probabilities = _softmax(scores)
    for candidate_index, feature_map in enumerate(example.candidate_feature_maps):
        target = 1.0 if candidate_index == example.chosen_index else 0.0
        gradient_scale = (probabilities[candidate_index] - target) * example.sample_weight
        _apply_sparse_gradient(global_head.weights, feature_map, gradient_scale, learning_rate, l2)
        _apply_sparse_gradient(stage_head.weights, feature_map, gradient_scale, learning_rate, l2)


def _apply_sparse_gradient(
    weights: dict[str, float],
    feature_map: dict[str, float],
    gradient_scale: float,
    learning_rate: float,
    l2: float,
) -> None:
    for feature_name, feature_value in feature_map.items():
        current = weights.get(feature_name, 0.0)
        gradient = (gradient_scale * feature_value) + (l2 * current)
        updated = current - (learning_rate * gradient)
        if abs(updated) < 1e-12:
            weights.pop(feature_name, None)
        else:
            weights[feature_name] = updated


@dataclass
class _MetricAccumulator:
    top_k: Sequence[int]
    example_count: int = 0
    weighted_example_count: float = 0.0
    loss_sum: float = 0.0
    chosen_rank_sum: float = 0.0
    chosen_probability_sum: float = 0.0
    entropy_sum: float = 0.0
    candidate_count_sum: float = 0.0
    top_k_hits: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value in self.top_k:
            self.top_k_hits.setdefault(int(value), 0.0)

    def as_dict(self, *, stage_metrics: dict[str, Any]) -> dict[str, Any]:
        if self.weighted_example_count <= 0.0:
            return {
                "example_count": self.example_count,
                "weighted_example_count": 0.0,
                "loss": None,
                "top_k_accuracy": {str(value): None for value in self.top_k},
                "mean_candidate_count": None,
                "mean_chosen_rank": None,
                "mean_chosen_probability": None,
                "mean_entropy": None,
                "stage_metrics": stage_metrics,
            }
        return {
            "example_count": self.example_count,
            "weighted_example_count": self.weighted_example_count,
            "loss": self.loss_sum / self.weighted_example_count,
            "top_k_accuracy": {
                str(value): self.top_k_hits[int(value)] / self.weighted_example_count for value in self.top_k
            },
            "mean_candidate_count": self.candidate_count_sum / self.weighted_example_count,
            "mean_chosen_rank": self.chosen_rank_sum / self.weighted_example_count,
            "mean_chosen_probability": self.chosen_probability_sum / self.weighted_example_count,
            "mean_entropy": self.entropy_sum / self.weighted_example_count,
            "stage_metrics": stage_metrics,
        }


def _accumulate_example_metrics(
    metrics: _MetricAccumulator,
    model: BehaviorCloningModel,
    example: BehaviorCloningExample,
    *,
    top_k: Sequence[int],
) -> None:
    metrics.example_count += 1
    metrics.weighted_example_count += example.sample_weight
    scores = [model.score_feature_map(stage=example.stage, feature_map=feature_map) for feature_map in example.candidate_feature_maps]
    probabilities = _softmax(scores)
    chosen_probability = max(probabilities[example.chosen_index], 1e-12)
    metrics.loss_sum += (-math.log(chosen_probability)) * example.sample_weight
    ranked_indices = sorted(
        range(len(example.candidate_feature_maps)),
        key=lambda index: (-scores[index], example.candidate_ids[index]),
    )
    chosen_rank = ranked_indices.index(example.chosen_index) + 1
    metrics.chosen_rank_sum += chosen_rank * example.sample_weight
    metrics.chosen_probability_sum += probabilities[example.chosen_index] * example.sample_weight
    metrics.entropy_sum += _distribution_entropy(probabilities) * example.sample_weight
    metrics.candidate_count_sum += len(example.candidate_feature_maps) * example.sample_weight
    for value in top_k:
        if chosen_rank <= value:
            metrics.top_k_hits[int(value)] += example.sample_weight


def _evaluate_stage(
    model: BehaviorCloningModel,
    examples: Sequence[BehaviorCloningExample],
    *,
    top_k: Sequence[int],
) -> dict[str, Any]:
    metrics = _MetricAccumulator(top_k=top_k)
    for example in examples:
        _accumulate_example_metrics(metrics, model, example, top_k=top_k)
    return metrics.as_dict(stage_metrics={})


def _select_behavior_cloning_objective(
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> float | None:
    for payload in (validation_metrics, test_metrics, train_metrics):
        loss = payload.get("loss")
        if loss is not None:
            return float(loss)
    return None


def _head_payload(head: _SparseScoringHead) -> dict[str, Any]:
    return {"name": head.name, "weights": dict(sorted(head.weights.items())), "example_count": head.example_count}


def _head_from_payload(payload: dict[str, Any]) -> _SparseScoringHead:
    return _SparseScoringHead(
        name=str(payload.get("name") or "unknown"),
        weights={str(name): float(value) for name, value in dict(payload.get("weights", {})).items()},
        example_count=int(payload.get("example_count", 0) or 0),
    )


def _behavior_cloning_config_payload(config: BehaviorCloningTrainConfig) -> dict[str, Any]:
    return {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "l2": config.l2,
        "validation_fraction": config.validation_fraction,
        "test_fraction": config.test_fraction,
        "seed": config.seed,
        "include_stages": list(config.include_stages),
        "include_decision_sources": list(config.include_decision_sources),
        "include_policy_packs": list(config.include_policy_packs),
        "include_policy_names": list(config.include_policy_names),
        "min_floor": config.min_floor,
        "max_floor": config.max_floor,
        "min_legal_actions": config.min_legal_actions,
        "top_k": list(config.top_k),
        "stage_weights": dict(config.stage_weights),
        "decision_source_weights": dict(config.decision_source_weights),
        "policy_pack_weights": dict(config.policy_pack_weights),
        "policy_name_weights": dict(config.policy_name_weights),
        "run_outcome_weights": dict(config.run_outcome_weights),
        "floor_band_weights": [
            {"min_floor": band.min_floor, "max_floor": band.max_floor, "weight": band.weight}
            for band in config.floor_band_weights
        ],
        "benchmark_manifest_path": (
            str(config.benchmark_manifest_path.expanduser().resolve())
            if config.benchmark_manifest_path is not None
            else None
        ),
        "live_base_url": config.live_base_url,
        "live_eval_max_env_steps": config.live_eval_max_env_steps,
        "live_eval_max_runs": config.live_eval_max_runs,
        "live_eval_max_combats": config.live_eval_max_combats,
    }


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    shifted = [math.exp(score - max_score) for score in scores]
    total = sum(shifted)
    return [value / total for value in shifted] if total > 0.0 else [1.0 / len(scores) for _ in scores]


def _distribution_entropy(probabilities: Sequence[float]) -> float:
    return sum((0.0 if value <= 0.0 else (-value * math.log(value))) for value in probabilities)


def _safe_game_state(payload: dict[str, Any]) -> GameStatePayload | None:
    try:
        return GameStatePayload.model_validate(payload)
    except Exception:
        return None


def _safe_action_descriptors(payload: dict[str, Any]) -> AvailableActionsPayload | None:
    try:
        return AvailableActionsPayload.model_validate(payload)
    except Exception:
        return None


def _candidate_from_action_id(action_id: str, *, label: str | None, source: str) -> CandidateAction:
    action, card_index, option_index, target_index = _parse_action_id(action_id)
    resolved_label = label or action_id
    return CandidateAction(
        action_id=action_id,
        action=action,
        label=resolved_label,
        request=ActionRequest(
            action=action,
            card_index=card_index,
            option_index=option_index,
            target_index=target_index,
            client_context={"candidate_id": action_id, "label": resolved_label},
        ),
        source=source,
        metadata={"fallback_reconstructed": True},
    )


def _parse_action_id(action_id: str) -> tuple[str, int | None, int | None, int | None]:
    action = action_id
    card_index = None
    option_index = None
    target_index = None
    if "|" in action_id:
        parts = action_id.split("|")
        action = parts[0]
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, raw_value = part.split("=", maxsplit=1)
            try:
                parsed = int(raw_value)
            except ValueError:
                continue
            if key == "card":
                card_index = parsed
            elif key == "option":
                option_index = parsed
            elif key == "target":
                target_index = parsed
    return action, card_index, option_index, target_index


def _stage_from_record(record: TrajectoryStepRecord) -> str:
    return record.decision_stage.strip().lower() if record.decision_stage else _stage_from_screen(record.screen_type)


def _stage_from_screen(screen_type: str) -> str:
    normalized = screen_type.strip().lower()
    return normalized if normalized else "unknown"


def _normalized_optional(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _add_flag(features: dict[str, float], name: str, enabled: bool = True) -> None:
    if enabled:
        features[name] = 1.0


def _add_one_hot(features: dict[str, float], prefix: str, value: str) -> None:
    normalized = value.strip().lower()
    if normalized:
        features[f"{prefix}={normalized}"] = 1.0


def _add_numeric(features: dict[str, float], name: str, value: float) -> None:
    features[name] = float(value)


def _bucketize(features: dict[str, float], prefix: str, value: float, buckets: Iterable[float]) -> None:
    thresholds = list(buckets)
    for threshold in thresholds:
        if value <= threshold:
            features[f"{prefix}<={threshold:g}"] = 1.0
            return
    features[f"{prefix}>{thresholds[-1]:g}"] = 1.0


def _map_node_kind(node_type: str) -> str:
    normalized = (node_type or "").lower()
    if "rest" in normalized or "camp" in normalized:
        return "rest"
    if "elite" in normalized:
        return "elite"
    if "shop" in normalized:
        return "shop"
    if "treasure" in normalized or "chest" in normalized:
        return "treasure"
    if "event" in normalized or "?" in normalized:
        return "event"
    if "monster" in normalized or "fight" in normalized:
        return "monster"
    return "unknown"


def _ranked_action_payload(item: RankedAction) -> dict[str, object]:
    return {
        "action_id": item.action_id,
        "action": item.action,
        "score": item.score,
        "reason": item.reason,
        "metadata": dict(item.metadata),
    }


def _combat_performance_payload(
    *,
    completed_combat_count: int,
    won_combats: int,
    lost_combats: int,
    completed_run_count: int,
    won_runs: int,
    total_reward: float,
    combat_steps: int,
) -> dict[str, object]:
    return {
        "combat_steps": combat_steps,
        "completed_combat_count": completed_combat_count,
        "won_combats": won_combats,
        "lost_combats": lost_combats,
        "combat_win_rate": (won_combats / completed_combat_count) if completed_combat_count else None,
        "reward_per_combat": (total_reward / completed_combat_count) if completed_combat_count else None,
        "reward_per_combat_step": (total_reward / combat_steps) if combat_steps else None,
        "completed_run_count": completed_run_count,
        "won_runs": won_runs,
        "run_win_rate": (won_runs / completed_run_count) if completed_run_count else None,
    }


def _default_env_factory(base_url: str, timeout: float) -> Any:
    return Sts2Env.from_base_url(base_url, timeout=timeout)
