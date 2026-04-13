from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Sequence

from sts2_rl.data import (
    TrajectoryStepRecord,
    build_state_summary,
    load_dataset_summary,
    load_public_strategic_decision_records,
    resolve_dataset_split_paths,
)
from sts2_rl.collect.strategic_runtime import selection_decision_type_for_mode
from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import AvailableActionsPayload, GameStatePayload
from sts2_rl.env.types import CandidateAction, StepObservation

from .strategic_pretrain import (
    STRATEGIC_PRETRAIN_FEATURE_SCHEMA_VERSION,
    StrategicPretrainModel,
    StrategicPretrainTrainConfig,
    _SparseScoringHead,
    _apply_sparse_gradient,
    _decision_stage_from_type,
    _head_from_payload,
    _head_payload,
    _normalized_optional,
    _select_strategic_pretrain_objective,
    _strategic_candidate_feature_map,
    _strategic_context_feature_map,
    _strategic_sample_weight,
    evaluate_strategic_pretrain_examples,
    load_strategic_pretrain_checkpoint_metadata,
)

STRATEGIC_FINETUNE_CHECKPOINT_SCHEMA_VERSION = 1
STRATEGIC_FINETUNE_TRAINING_SCHEMA_VERSION = 1

StrategicFinetuneSchedule = Literal["weighted_shuffle", "round_robin"]
StrategicFinetuneSourceKind = Literal["runtime", "public"]

_RUNTIME_ALLOWED_DECISION_TYPES = {
    "reward_card_pick",
    "shop_buy",
    "selection_pick",
    "selection_remove",
    "selection_upgrade",
    "selection_transform",
    "event_choice",
    "rest_site_action",
}
@dataclass(frozen=True)
class StrategicFinetuneTrainConfig:
    epochs: int = 60
    learning_rate: float = 0.04
    l2: float = 0.0001
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0
    include_decision_types: tuple[str, ...] = ()
    include_runtime_decision_sources: tuple[str, ...] = ()
    include_runtime_policy_packs: tuple[str, ...] = ()
    include_runtime_policy_names: tuple[str, ...] = ()
    include_public_support_qualities: tuple[str, ...] = ()
    include_public_source_names: tuple[str, ...] = ()
    include_public_build_ids: tuple[str, ...] = ()
    runtime_min_floor: int | None = None
    runtime_max_floor: int | None = None
    public_min_floor: int | None = None
    public_max_floor: int | None = None
    public_min_confidence: float = 0.0
    top_k: tuple[int, ...] = (1, 3)
    decision_type_weights: dict[str, float] = field(default_factory=dict)
    source_name_weights: dict[str, float] = field(default_factory=dict)
    build_id_weights: dict[str, float] = field(default_factory=dict)
    run_outcome_weights: dict[str, float] = field(default_factory=dict)
    runtime_example_weight: float = 1.0
    public_example_weight: float = 1.0
    confidence_power: float = 1.0
    chosen_only_positive_weight: float = 0.35
    auxiliary_value_weight: float = 0.75
    schedule: StrategicFinetuneSchedule = "weighted_shuffle"
    runtime_replay_passes: int = 1
    public_replay_passes: int = 1
    warmstart_checkpoint_path: Path | None = None
    freeze_transferred_ranking_epochs: int = 0
    freeze_transferred_value_epochs: int = 0
    enforce_public_build_match: bool = True
    enforce_runtime_public_build_match: bool = True
    runtime_source_name: str = "local_runtime"
    runtime_game_mode: str = "standard"
    runtime_platform_type: str = "local"
    runtime_build_id: str | None = None

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
        if self.runtime_min_floor is not None and self.runtime_max_floor is not None and self.runtime_min_floor > self.runtime_max_floor:
            raise ValueError("runtime_min_floor must be <= runtime_max_floor.")
        if self.public_min_floor is not None and self.public_max_floor is not None and self.public_min_floor > self.public_max_floor:
            raise ValueError("public_min_floor must be <= public_max_floor.")
        if not (0.0 <= self.public_min_confidence <= 1.0):
            raise ValueError("public_min_confidence must be between 0 and 1.")
        if not self.top_k or any(value < 1 for value in self.top_k):
            raise ValueError("top_k must contain positive entries.")
        for mapping in (self.decision_type_weights, self.source_name_weights, self.build_id_weights, self.run_outcome_weights):
            for value in mapping.values():
                if value <= 0.0:
                    raise ValueError("weight values must be positive.")
        if self.runtime_example_weight <= 0.0:
            raise ValueError("runtime_example_weight must be positive.")
        if self.public_example_weight <= 0.0:
            raise ValueError("public_example_weight must be positive.")
        if self.confidence_power < 0.0:
            raise ValueError("confidence_power must be non-negative.")
        if self.chosen_only_positive_weight <= 0.0:
            raise ValueError("chosen_only_positive_weight must be positive.")
        if self.auxiliary_value_weight < 0.0:
            raise ValueError("auxiliary_value_weight must be non-negative.")
        if self.schedule not in {"weighted_shuffle", "round_robin"}:
            raise ValueError("schedule must be 'weighted_shuffle' or 'round_robin'.")
        if self.runtime_replay_passes < 1:
            raise ValueError("runtime_replay_passes must be positive.")
        if self.public_replay_passes < 1:
            raise ValueError("public_replay_passes must be positive.")
        if self.freeze_transferred_ranking_epochs < 0:
            raise ValueError("freeze_transferred_ranking_epochs must be non-negative.")
        if self.freeze_transferred_value_epochs < 0:
            raise ValueError("freeze_transferred_value_epochs must be non-negative.")


@dataclass(frozen=True)
class StrategicFinetuneTrainingReport:
    output_dir: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    metrics_path: Path
    summary_path: Path
    runtime_dataset_path: Path
    public_dataset_path: Path | None
    example_count: int
    runtime_example_count: int
    public_example_count: int
    train_example_count: int
    validation_example_count: int
    test_example_count: int
    feature_count: int
    decision_type_count: int
    best_epoch: int
    warmstart_checkpoint_path: Path | None


@dataclass(frozen=True)
class StrategicFinetuneExample:
    example_id: str
    decision_type: str
    decision_stage: str
    support_quality: str
    source_name: str
    build_id: str | None
    game_version: str | None
    branch: str | None
    content_channel: str | None
    floor: int | None
    run_outcome: str | None
    chosen_action: str
    chosen_index: int
    candidate_ids: tuple[str, ...]
    candidate_feature_maps: tuple[dict[str, float], ...]
    context_feature_map: dict[str, float]
    sample_weight: float
    confidence_weight: float
    source_path: Path
    source_kind: StrategicFinetuneSourceKind


@dataclass(frozen=True)
class _ResolvedSourceDataset:
    dataset_path: Path
    examples_path: Path | None
    train_examples_path: Path | None
    validation_examples_path: Path | None
    test_examples_path: Path | None
    examples: list[StrategicFinetuneExample]
    train_examples: list[StrategicFinetuneExample]
    validation_examples: list[StrategicFinetuneExample]
    test_examples: list[StrategicFinetuneExample]
    split_strategy: str
    skipped_records: int
    dropped_records: int
    dataset_summary: dict[str, Any] | None


@dataclass(frozen=True)
class _TransferState:
    checkpoint_path: Path
    metadata: dict[str, Any]
    global_ranking_head: _SparseScoringHead
    decision_ranking_heads: dict[str, _SparseScoringHead]
    global_value_head: _SparseScoringHead
    decision_value_heads: dict[str, _SparseScoringHead]


@dataclass(frozen=True)
class StrategicFinetuneModel:
    global_ranking_head: _SparseScoringHead
    decision_ranking_heads: dict[str, _SparseScoringHead]
    global_value_head: _SparseScoringHead
    decision_value_heads: dict[str, _SparseScoringHead]
    metadata: dict[str, Any] = field(default_factory=dict)

    def score_candidate(self, *, decision_type: str, feature_map: dict[str, float]) -> float:
        score = self.global_ranking_head.score(feature_map)
        decision_head = self.decision_ranking_heads.get(decision_type)
        if decision_head is not None:
            score += decision_head.score(feature_map)
        return score

    def predict_value(self, *, decision_type: str, context_feature_map: dict[str, float]) -> float:
        score = self.global_value_head.score(context_feature_map)
        decision_head = self.decision_value_heads.get(decision_type)
        if decision_head is not None:
            score += decision_head.score(context_feature_map)
        return StrategicPretrainModel(
            global_ranking_head=self.global_ranking_head,
            decision_ranking_heads=self.decision_ranking_heads,
            global_value_head=self.global_value_head,
            decision_value_heads=self.decision_value_heads,
            metadata=self.metadata,
        ).predict_value(decision_type=decision_type, context_feature_map=context_feature_map)

    def rank_actions(
        self,
        *,
        decision_type: str,
        context: dict[str, Any],
        candidate_actions: Sequence[str],
        support_quality: str = "full_candidates",
    ) -> list[dict[str, Any]]:
        return StrategicPretrainModel(
            global_ranking_head=self.global_ranking_head,
            decision_ranking_heads=self.decision_ranking_heads,
            global_value_head=self.global_value_head,
            decision_value_heads=self.decision_value_heads,
            metadata=self.metadata,
        ).rank_actions(
            decision_type=decision_type,
            context=context,
            candidate_actions=candidate_actions,
            support_quality=support_quality,
        )

    def save(self, path: str | Path) -> Path:
        checkpoint_path = Path(path).expanduser().resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": STRATEGIC_FINETUNE_CHECKPOINT_SCHEMA_VERSION,
            "algorithm": "strategic_finetune",
            "feature_schema_version": STRATEGIC_PRETRAIN_FEATURE_SCHEMA_VERSION,
            "global_ranking_head": _head_payload(self.global_ranking_head),
            "decision_ranking_heads": {
                name: _head_payload(head) for name, head in sorted(self.decision_ranking_heads.items())
            },
            "global_value_head": _head_payload(self.global_value_head),
            "decision_value_heads": {
                name: _head_payload(head) for name, head in sorted(self.decision_value_heads.items())
            },
            "metadata": self.metadata,
        }
        checkpoint_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return checkpoint_path

    @classmethod
    def load(cls, path: str | Path) -> "StrategicFinetuneModel":
        payload = _load_transfer_payload(path)
        if payload.get("algorithm") != "strategic_finetune":
            raise ValueError(f"Unsupported strategic finetune checkpoint algorithm: {payload.get('algorithm')}")
        return _strategic_finetune_model_from_payload(payload)


def default_strategic_finetune_session_name() -> str:
    return datetime.now(UTC).strftime("strategic-finetune-%Y%m%d-%H%M%S")


def load_strategic_finetune_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    payload = _load_transfer_payload(path)
    return {
        "schema_version": payload.get("schema_version"),
        "algorithm": payload.get("algorithm"),
        "feature_schema_version": payload.get("feature_schema_version"),
        "metadata": payload.get("metadata", {}),
    }


def train_strategic_finetune_policy(
    *,
    runtime_dataset_source: str | Path,
    output_root: str | Path,
    public_dataset_source: str | Path | None = None,
    session_name: str | None = None,
    config: StrategicFinetuneTrainConfig | None = None,
) -> StrategicFinetuneTrainingReport:
    train_config = config or StrategicFinetuneTrainConfig()
    runtime_dataset_path = Path(runtime_dataset_source).expanduser().resolve()
    public_dataset_path = None if public_dataset_source is None else Path(public_dataset_source).expanduser().resolve()

    runtime_resolved = _resolve_runtime_dataset(runtime_dataset_path, config=train_config)
    if not runtime_resolved.train_examples:
        raise ValueError("Strategic finetuning requires at least one runtime train example.")
    public_resolved = None if public_dataset_path is None else _resolve_public_dataset(public_dataset_path, config=train_config)

    transfer = _resolve_transfer_state(train_config.warmstart_checkpoint_path)
    _validate_build_lineage_guardrails(runtime_resolved, public_resolved, transfer, config=train_config)

    output_dir = Path(output_root).expanduser().resolve() / (session_name or default_strategic_finetune_session_name())
    if output_dir.exists():
        raise FileExistsError(f"Strategic finetuning output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    all_examples = list(runtime_resolved.examples)
    if public_resolved is not None:
        all_examples.extend(public_resolved.examples)
    train_examples = list(runtime_resolved.train_examples)
    validation_examples = list(runtime_resolved.validation_examples)
    test_examples = list(runtime_resolved.test_examples)
    if public_resolved is not None:
        train_examples.extend(public_resolved.train_examples)
        validation_examples.extend(public_resolved.validation_examples)
        test_examples.extend(public_resolved.test_examples)

    feature_inventory = sorted(
        {
            feature_name
            for example in all_examples
            for feature_name in example.context_feature_map
        }
        | {
            feature_name
            for example in all_examples
            for candidate_features in example.candidate_feature_maps
            for feature_name in candidate_features
        }
    )
    feature_hash = hashlib.sha256("\n".join(feature_inventory).encode("utf-8")).hexdigest()
    decision_types = sorted({example.decision_type for example in train_examples})

    global_ranking_head, decision_ranking_heads, global_value_head, decision_value_heads, transferred_modules = _initialize_heads(
        decision_types=decision_types,
        transfer=transfer,
    )
    for example in train_examples:
        global_ranking_head.example_count += 1
        global_value_head.example_count += 1
        decision_ranking_heads[example.decision_type].example_count += 1
        decision_value_heads[example.decision_type].example_count += 1

    metrics_path = output_dir / "training-metrics.jsonl"
    best_epoch = 0
    best_objective: float | None = None
    best_global_ranking_head = global_ranking_head.clone()
    best_decision_ranking_heads = {name: head.clone() for name, head in decision_ranking_heads.items()}
    best_global_value_head = global_value_head.clone()
    best_decision_value_heads = {name: head.clone() for name, head in decision_value_heads.items()}
    best_train_metrics: dict[str, Any] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    best_test_metrics: dict[str, Any] | None = None

    rng = random.Random(train_config.seed)
    with metrics_path.open("w", encoding="utf-8", newline="\n") as metrics_handle:
        for epoch in range(1, train_config.epochs + 1):
            epoch_examples, schedule_counts = _build_epoch_training_plan(
                runtime_examples=runtime_resolved.train_examples,
                public_examples=[] if public_resolved is None else public_resolved.train_examples,
                config=train_config,
                rng=rng,
            )
            ranking_frozen = transfer is not None and epoch <= train_config.freeze_transferred_ranking_epochs
            value_frozen = transfer is not None and epoch <= train_config.freeze_transferred_value_epochs
            for example in epoch_examples:
                _train_on_finetune_example(
                    example=example,
                    global_ranking_head=global_ranking_head,
                    decision_ranking_head=decision_ranking_heads[example.decision_type],
                    global_value_head=global_value_head,
                    decision_value_head=decision_value_heads[example.decision_type],
                    learning_rate=train_config.learning_rate,
                    l2=train_config.l2,
                    chosen_only_positive_weight=train_config.chosen_only_positive_weight,
                    auxiliary_value_weight=train_config.auxiliary_value_weight,
                    freeze_ranking=ranking_frozen and _ranking_transfer_applies(example.decision_type, transferred_modules),
                    freeze_value=value_frozen and _value_transfer_applies(example.decision_type, transferred_modules),
                )

            epoch_model = StrategicFinetuneModel(
                global_ranking_head=global_ranking_head.clone(),
                decision_ranking_heads={name: head.clone() for name, head in decision_ranking_heads.items()},
                global_value_head=global_value_head.clone(),
                decision_value_heads={name: head.clone() for name, head in decision_value_heads.items()},
                metadata={},
            )
            train_metrics = _evaluate_finetune_examples(epoch_model, train_examples, top_k=train_config.top_k)
            validation_metrics = _evaluate_finetune_examples(epoch_model, validation_examples, top_k=train_config.top_k)
            test_metrics = _evaluate_finetune_examples(epoch_model, test_examples, top_k=train_config.top_k)
            objective = _select_strategic_pretrain_objective(train_metrics, validation_metrics, test_metrics)
            is_best = best_objective is None or (objective is not None and objective < best_objective)
            if is_best:
                best_epoch = epoch
                best_objective = objective
                best_global_ranking_head = global_ranking_head.clone()
                best_decision_ranking_heads = {name: head.clone() for name, head in decision_ranking_heads.items()}
                best_global_value_head = global_value_head.clone()
                best_decision_value_heads = {name: head.clone() for name, head in decision_value_heads.items()}
                best_train_metrics = train_metrics
                best_validation_metrics = validation_metrics
                best_test_metrics = test_metrics

            metrics_handle.write(
                json.dumps(
                    {
                        "schema_version": STRATEGIC_FINETUNE_TRAINING_SCHEMA_VERSION,
                        "epoch": epoch,
                        "objective": objective,
                        "selected_as_best": is_best,
                        "schedule_mode": train_config.schedule,
                        "schedule_counts": schedule_counts,
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "test": test_metrics,
                    },
                    ensure_ascii=False,
                )
            )
            metrics_handle.write("\n")

    warmstart_metadata = None if transfer is None else load_strategic_pretrain_checkpoint_metadata(transfer.checkpoint_path)
    model_metadata = {
        "runtime_dataset_path": str(runtime_dataset_path),
        "public_dataset_path": str(public_dataset_path) if public_dataset_path is not None else None,
        "feature_count": len(feature_inventory),
        "feature_hash": feature_hash,
        "feature_inventory_sample": feature_inventory[:50],
        "decision_types": decision_types,
        "best_epoch": best_epoch,
        "best_objective": best_objective,
        "train_metrics": best_train_metrics,
        "validation_metrics": best_validation_metrics,
        "test_metrics": best_test_metrics,
        "training_config": _strategic_finetune_config_payload(train_config),
        "artifact_kind": "strategic_finetune",
        "transferred_modules": transferred_modules,
        "warmstart_checkpoint_metadata": warmstart_metadata,
        "runtime_dataset_summary": _dataset_summary_payload(runtime_resolved),
        "public_dataset_summary": None if public_resolved is None else _dataset_summary_payload(public_resolved),
    }

    checkpoint_path = StrategicFinetuneModel(
        global_ranking_head=global_ranking_head,
        decision_ranking_heads=decision_ranking_heads,
        global_value_head=global_value_head,
        decision_value_heads=decision_value_heads,
        metadata=model_metadata,
    ).save(output_dir / "strategic-finetune-checkpoint.json")
    best_checkpoint_path = StrategicFinetuneModel(
        global_ranking_head=best_global_ranking_head,
        decision_ranking_heads=best_decision_ranking_heads,
        global_value_head=best_global_value_head,
        decision_value_heads=best_decision_value_heads,
        metadata=model_metadata,
    ).save(output_dir / "strategic-finetune-best.json")

    summary_payload = {
        "schema_version": STRATEGIC_FINETUNE_TRAINING_SCHEMA_VERSION,
        "algorithm": "strategic_finetune",
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "metrics_path": str(metrics_path),
        "runtime_dataset_path": str(runtime_dataset_path),
        "public_dataset_path": str(public_dataset_path) if public_dataset_path is not None else None,
        "example_count": len(all_examples),
        "runtime_example_count": len(runtime_resolved.examples),
        "public_example_count": 0 if public_resolved is None else len(public_resolved.examples),
        "train_example_count": len(train_examples),
        "validation_example_count": len(validation_examples),
        "test_example_count": len(test_examples),
        "feature_count": len(feature_inventory),
        "feature_hash": feature_hash,
        "decision_type_count": len(decision_types),
        "decision_types": decision_types,
        "best_epoch": best_epoch,
        "best_objective": best_objective,
        "config": _strategic_finetune_config_payload(train_config),
        "schedule": {
            "mode": train_config.schedule,
            "runtime_replay_passes": train_config.runtime_replay_passes,
            "public_replay_passes": train_config.public_replay_passes,
        },
        "warmstart_checkpoint_path": (
            str(transfer.checkpoint_path) if transfer is not None else None
        ),
        "warmstart_checkpoint_metadata": warmstart_metadata,
        "transferred_modules": transferred_modules,
        "runtime_dataset": _dataset_summary_payload(runtime_resolved),
        "public_dataset": None if public_resolved is None else _dataset_summary_payload(public_resolved),
        "train": best_train_metrics,
        "validation": best_validation_metrics,
        "test": best_test_metrics,
        "checkpoint_metadata": load_strategic_finetune_checkpoint_metadata(best_checkpoint_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return StrategicFinetuneTrainingReport(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        runtime_dataset_path=runtime_dataset_path,
        public_dataset_path=public_dataset_path,
        example_count=len(all_examples),
        runtime_example_count=len(runtime_resolved.examples),
        public_example_count=0 if public_resolved is None else len(public_resolved.examples),
        train_example_count=len(train_examples),
        validation_example_count=len(validation_examples),
        test_example_count=len(test_examples),
        feature_count=len(feature_inventory),
        decision_type_count=len(decision_types),
        best_epoch=best_epoch,
        warmstart_checkpoint_path=None if transfer is None else transfer.checkpoint_path,
    )


def _resolve_runtime_dataset(
    dataset_path: Path,
    *,
    config: StrategicFinetuneTrainConfig,
) -> _ResolvedSourceDataset:
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
                all_examples, all_skipped, all_dropped = _load_runtime_examples_from_paths(
                    [path for path in (train_examples_path, validation_examples_path, test_examples_path) if path is not None and path.exists()],
                    config=config,
                )
                train_examples, _, _ = _load_runtime_examples_from_paths([train_examples_path], config=config)
                validation_examples, _, _ = _load_runtime_examples_from_paths(
                    [validation_examples_path] if validation_examples_path is not None else [],
                    config=config,
                )
                test_examples, _, _ = _load_runtime_examples_from_paths(
                    [test_examples_path] if test_examples_path is not None else [],
                    config=config,
                )
                return _ResolvedSourceDataset(
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

    examples, skipped_records, dropped_records = _load_runtime_examples_from_paths([dataset_path], config=config)
    train_examples, validation_examples, test_examples = _random_split_examples(
        examples,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    return _ResolvedSourceDataset(
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


def _resolve_public_dataset(
    dataset_path: Path,
    *,
    config: StrategicFinetuneTrainConfig,
) -> _ResolvedSourceDataset:
    public_config = StrategicPretrainTrainConfig(
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
        include_decision_types=config.include_decision_types,
        include_support_qualities=config.include_public_support_qualities,
        include_source_names=config.include_public_source_names,
        include_build_ids=config.include_public_build_ids,
        min_floor=config.public_min_floor,
        max_floor=config.public_max_floor,
        min_confidence=config.public_min_confidence,
        top_k=config.top_k,
        decision_type_weights=config.decision_type_weights,
        source_name_weights=config.source_name_weights,
        build_id_weights=config.build_id_weights,
        run_outcome_weights=config.run_outcome_weights,
        confidence_power=config.confidence_power,
        chosen_only_positive_weight=config.chosen_only_positive_weight,
        auxiliary_value_weight=config.auxiliary_value_weight,
    )
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
        if dataset_summary is not None and dataset_summary.get("dataset_kind") == "public_strategic_decisions":
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
                all_examples, all_skipped, all_dropped = _load_public_examples_from_paths(
                    [path for path in (train_examples_path, validation_examples_path, test_examples_path) if path is not None and path.exists()],
                    config=config,
                    public_config=public_config,
                )
                train_examples, _, _ = _load_public_examples_from_paths([train_examples_path], config=config, public_config=public_config)
                validation_examples, _, _ = _load_public_examples_from_paths(
                    [validation_examples_path] if validation_examples_path is not None else [],
                    config=config,
                    public_config=public_config,
                )
                test_examples, _, _ = _load_public_examples_from_paths(
                    [test_examples_path] if test_examples_path is not None else [],
                    config=config,
                    public_config=public_config,
                )
                return _ResolvedSourceDataset(
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

    examples, skipped_records, dropped_records = _load_public_examples_from_paths(
        [dataset_path],
        config=config,
        public_config=public_config,
    )
    train_examples, validation_examples, test_examples = _random_split_examples(
        examples,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    return _ResolvedSourceDataset(
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


def _load_runtime_examples_from_paths(
    paths: Sequence[Path | None],
    *,
    config: StrategicFinetuneTrainConfig,
) -> tuple[list[StrategicFinetuneExample], int, int]:
    examples: list[StrategicFinetuneExample] = []
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
                record = _trajectory_step_record_from_payload(payload)
                if record is None:
                    skipped_records += 1
                    continue
                example = _runtime_example_from_step_record(
                    record=record,
                    raw_payload=payload,
                    source_path=path,
                    config=config,
                )
                if example is None:
                    dropped_records += 1
                else:
                    examples.append(example)
    return examples, skipped_records, dropped_records


def _load_public_examples_from_paths(
    paths: Sequence[Path | None],
    *,
    config: StrategicFinetuneTrainConfig,
    public_config: StrategicPretrainTrainConfig,
) -> tuple[list[StrategicFinetuneExample], int, int]:
    examples: list[StrategicFinetuneExample] = []
    skipped_records = 0
    dropped_records = 0
    for path in paths:
        if path is None or not path.exists():
            continue
        try:
            records = load_public_strategic_decision_records(path)
        except (ValueError, KeyError):
            skipped_records += 1
            continue
        for record in records:
            example = _public_example_from_record(
                record=record,
                source_path=path,
                config=config,
                public_config=public_config,
            )
            if example is None:
                dropped_records += 1
                continue
            examples.append(example)
    return examples, skipped_records, dropped_records


def _public_example_from_record(
    *,
    record: Any,
    source_path: Path,
    config: StrategicFinetuneTrainConfig,
    public_config: StrategicPretrainTrainConfig,
) -> StrategicFinetuneExample | None:
    decision_type = str(record.decision_type).strip().lower()
    support_quality = str(record.support_quality).strip().lower()
    source_name = str(record.source_name).strip().lower()
    build_id = None if record.build_id is None else str(record.build_id).strip()
    game_version = None if getattr(record, "game_version", None) is None else str(record.game_version).strip()
    branch = None if getattr(record, "branch", None) is None else str(record.branch).strip()
    content_channel = None if getattr(record, "content_channel", None) is None else str(record.content_channel).strip()
    run_outcome = None if record.run_outcome is None else str(record.run_outcome).strip().lower()

    if config.include_decision_types and decision_type not in {item.lower() for item in config.include_decision_types}:
        return None
    if config.include_public_support_qualities and support_quality not in {item.lower() for item in config.include_public_support_qualities}:
        return None
    if config.include_public_source_names and source_name not in {item.lower() for item in config.include_public_source_names}:
        return None
    if config.include_public_build_ids:
        allowed_builds = {item for item in config.include_public_build_ids}
        if build_id is None or build_id not in allowed_builds:
            return None
    if config.public_min_floor is not None and (record.floor is None or record.floor < config.public_min_floor):
        return None
    if config.public_max_floor is not None and (record.floor is None or record.floor > config.public_max_floor):
        return None
    if float(record.reconstruction_confidence) < config.public_min_confidence:
        return None

    candidate_ids = tuple(record.candidate_actions) if record.candidate_actions else (record.chosen_action,)
    chosen_index = candidate_ids.index(record.chosen_action) if record.chosen_action in candidate_ids else 0
    context_payload = {
        "source_name": source_name,
        "character_id": record.character_id,
        "ascension": record.ascension,
        "build_id": build_id,
        "game_version": game_version,
        "branch": branch,
        "content_channel": content_channel,
        "game_mode": record.game_mode,
        "platform_type": record.platform_type,
        "acts_reached": record.acts_reached,
        "act_index": record.act_index,
        "act_id": record.act_id,
        "floor": record.floor,
        "floor_within_act": record.floor_within_act,
        "room_type": record.room_type,
        "map_point_type": record.map_point_type,
        "source_type": record.source_type,
        "support_quality": support_quality,
        "candidate_count": len(candidate_ids),
        "metadata": dict(record.metadata),
    }
    context_feature_map = _strategic_context_feature_map(
        context=context_payload,
        decision_type=decision_type,
        support_quality=support_quality,
    )
    candidate_feature_maps = tuple(
        _strategic_candidate_feature_map(
            context=context_payload,
            decision_type=decision_type,
            support_quality=support_quality,
            candidate_id=candidate_id,
            candidate_count=len(candidate_ids),
        )
        for candidate_id in candidate_ids
    )
    sample_weight = _strategic_sample_weight(
        config=public_config,
        decision_type=decision_type,
        support_quality=support_quality,
        source_name=source_name,
        build_id=build_id,
        run_outcome=run_outcome,
        reconstruction_confidence=float(record.reconstruction_confidence),
    ) * config.public_example_weight
    if sample_weight <= 0.0:
        return None
    confidence_weight = (
        float(record.reconstruction_confidence) ** config.confidence_power if config.confidence_power > 0.0 else 1.0
    )
    return StrategicFinetuneExample(
        example_id=record.decision_id,
        decision_type=decision_type,
        decision_stage=_decision_stage_from_type(decision_type),
        support_quality=support_quality,
        source_name=source_name,
        build_id=build_id,
        game_version=game_version,
        branch=branch,
        content_channel=content_channel,
        floor=record.floor,
        run_outcome=run_outcome,
        chosen_action=record.chosen_action,
        chosen_index=chosen_index,
        candidate_ids=candidate_ids,
        candidate_feature_maps=candidate_feature_maps,
        context_feature_map=context_feature_map,
        sample_weight=sample_weight,
        confidence_weight=confidence_weight,
        source_path=source_path,
        source_kind="public",
    )


def _runtime_example_from_step_record(
    *,
    record: TrajectoryStepRecord,
    raw_payload: dict[str, Any],
    source_path: Path,
    config: StrategicFinetuneTrainConfig,
) -> StrategicFinetuneExample | None:
    if record.chosen_action_id is None:
        return None
    if config.runtime_min_floor is not None and (record.floor is None or record.floor < config.runtime_min_floor):
        return None
    if config.runtime_max_floor is not None and (record.floor is None or record.floor > config.runtime_max_floor):
        return None

    state = _safe_game_state(record.state)
    descriptors = _safe_action_descriptors(record.action_descriptors)
    if state is None or descriptors is None:
        return None
    build_result = build_candidate_actions(state, descriptors)
    candidate_by_id = {candidate.action_id: candidate for candidate in build_result.candidates}
    if record.legal_action_ids and all(action_id in candidate_by_id for action_id in record.legal_action_ids):
        legal_actions = [candidate_by_id[action_id] for action_id in record.legal_action_ids]
    else:
        legal_actions = list(build_result.candidates)
    if not legal_actions:
        return None
    chosen_candidate = next((candidate for candidate in legal_actions if candidate.action_id == record.chosen_action_id), None)
    if chosen_candidate is None:
        return None

    decision_source = _normalized_optional(record.decision_source)
    policy_pack = _normalized_optional(record.policy_pack)
    policy_name = _normalized_optional(record.policy_name)
    if config.include_runtime_decision_sources and decision_source not in {item.lower() for item in config.include_runtime_decision_sources}:
        return None
    if config.include_runtime_policy_packs and policy_pack not in {item.lower() for item in config.include_runtime_policy_packs}:
        return None
    if config.include_runtime_policy_names and policy_name not in {item.lower() for item in config.include_runtime_policy_names}:
        return None

    semantic_binding = _resolve_runtime_semantic_binding(
        record=record,
        raw_payload=raw_payload,
        state=state,
        legal_actions=legal_actions,
        chosen_candidate=chosen_candidate,
    )
    if semantic_binding is None:
        return None
    if config.include_decision_types and semantic_binding["decision_type"] not in {item.lower() for item in config.include_decision_types}:
        return None

    runtime_lineage = dict(raw_payload.get("strategic_context", {})) if isinstance(raw_payload.get("strategic_context"), dict) else {}
    summary = record.state_summary if record.state_summary else build_state_summary(
        StepObservation(
            screen_type=state.screen,
            run_id=state.run_id,
            state=state,
            action_descriptors=descriptors,
            legal_actions=legal_actions,
            build_warnings=build_result.unsupported_actions,
        )
    )
    run_summary = dict(summary.get("run", {}))
    runtime_build = state.build
    runtime_build_id = config.runtime_build_id or (runtime_build.build_id if runtime_build is not None else None)
    runtime_game_version = runtime_build.game_version if runtime_build is not None else None
    runtime_branch = runtime_build.branch if runtime_build is not None else None
    runtime_content_channel = runtime_build.content_channel if runtime_build is not None else None
    floor_within_act = runtime_lineage.get("floor_within_act")
    if floor_within_act is None and record.floor is not None:
        floor_within_act = record.floor
    act_index = run_summary.get("act_index")
    acts_reached = runtime_lineage.get("acts_reached")
    if acts_reached is None and act_index is not None:
        acts_reached = act_index
    run_outcome = _runtime_run_outcome(record)
    context_payload = {
        "source_name": config.runtime_source_name,
        "character_id": run_summary.get("character_id"),
        "ascension": run_summary.get("ascension"),
        "build_id": runtime_build_id,
        "game_version": runtime_game_version,
        "branch": runtime_branch,
        "content_channel": runtime_content_channel,
        "game_mode": config.runtime_game_mode,
        "platform_type": config.runtime_platform_type,
        "acts_reached": acts_reached,
        "act_index": act_index,
        "act_id": run_summary.get("act_id"),
        "floor": record.floor,
        "floor_within_act": floor_within_act,
        "room_type": semantic_binding["room_type"],
        "map_point_type": semantic_binding["map_point_type"],
        "source_type": semantic_binding["source_type"],
        "support_quality": semantic_binding["support_quality"],
        "candidate_count": len(semantic_binding["candidate_ids"]),
        "metadata": {
            "artifact_family": "trajectory_steps",
            "decision_source": decision_source,
            "policy_pack": policy_pack,
            "policy_name": policy_name,
            "has_detail_payload": bool(runtime_lineage.get("has_route_planner_trace") or record.decision_metadata),
            "has_room_history": bool(runtime_lineage),
            **dict(semantic_binding.get("metadata", {})),
        },
    }
    context_feature_map = _strategic_context_feature_map(
        context=context_payload,
        decision_type=semantic_binding["decision_type"],
        support_quality=semantic_binding["support_quality"],
    )
    candidate_feature_maps = tuple(
        _strategic_candidate_feature_map(
            context=context_payload,
            decision_type=semantic_binding["decision_type"],
            support_quality=semantic_binding["support_quality"],
            candidate_id=candidate_id,
            candidate_count=len(semantic_binding["candidate_ids"]),
        )
        for candidate_id in semantic_binding["candidate_ids"]
    )
    sample_weight = _strategic_sample_weight(
        config=StrategicPretrainTrainConfig(
            decision_type_weights=config.decision_type_weights,
            source_name_weights=config.source_name_weights,
            build_id_weights=config.build_id_weights,
            run_outcome_weights=config.run_outcome_weights,
            confidence_power=config.confidence_power,
            chosen_only_positive_weight=config.chosen_only_positive_weight,
            auxiliary_value_weight=config.auxiliary_value_weight,
        ),
        decision_type=semantic_binding["decision_type"],
        support_quality=semantic_binding["support_quality"],
        source_name=config.runtime_source_name,
        build_id=runtime_build_id,
        run_outcome=run_outcome,
        reconstruction_confidence=1.0,
    ) * config.runtime_example_weight
    if sample_weight <= 0.0:
        return None

    return StrategicFinetuneExample(
        example_id=f"{record.session_name}:{record.instance_id}:{record.run_id}:{record.step_index}",
        decision_type=semantic_binding["decision_type"],
        decision_stage=_decision_stage_from_type(semantic_binding["decision_type"]),
        support_quality=semantic_binding["support_quality"],
        source_name=config.runtime_source_name,
        build_id=runtime_build_id,
        game_version=runtime_game_version,
        branch=runtime_branch,
        content_channel=runtime_content_channel,
        floor=record.floor,
        run_outcome=run_outcome,
        chosen_action=semantic_binding["chosen_action"],
        chosen_index=semantic_binding["chosen_index"],
        candidate_ids=semantic_binding["candidate_ids"],
        candidate_feature_maps=candidate_feature_maps,
        context_feature_map=context_feature_map,
        sample_weight=sample_weight,
        confidence_weight=1.0,
        source_path=source_path,
        source_kind="runtime",
    )


def _resolve_runtime_semantic_binding(
    *,
    record: TrajectoryStepRecord,
    raw_payload: dict[str, Any],
    state: GameStatePayload,
    legal_actions: Sequence[CandidateAction],
    chosen_candidate: CandidateAction,
) -> dict[str, Any] | None:
    action = chosen_candidate.action
    if action == "choose_reward_card" or action == "skip_reward_cards":
        candidate_ids = [_runtime_reward_candidate_id(state, candidate) for candidate in legal_actions if candidate.action in {"choose_reward_card", "skip_reward_cards"}]
        candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id is not None]
        chosen_action = _runtime_reward_candidate_id(state, chosen_candidate)
        if chosen_action is None or not candidate_ids:
            return None
        chosen_index = _semantic_chosen_index(tuple(candidate_ids), chosen_action)
        reward_room_type, reward_map_point_type = _runtime_reward_domain(raw_payload=raw_payload, state=state)
        return {
            "decision_type": "reward_card_pick",
            "support_quality": "full_candidates",
            "chosen_action": chosen_action,
            "chosen_index": chosen_index,
            "candidate_ids": tuple(candidate_ids),
            "room_type": reward_room_type,
            "map_point_type": reward_map_point_type,
            "source_type": "reward",
        }
    if action == "buy_card":
        candidate_ids = [_runtime_shop_buy_candidate_id(state, candidate) for candidate in legal_actions if candidate.action == "buy_card"]
        candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id is not None]
        chosen_action = _runtime_shop_buy_candidate_id(state, chosen_candidate)
        if chosen_action is None or not candidate_ids:
            return None
        return {
            "decision_type": "shop_buy",
            "support_quality": "full_candidates",
            "chosen_action": chosen_action,
            "chosen_index": _semantic_chosen_index(tuple(candidate_ids), chosen_action),
            "candidate_ids": tuple(candidate_ids),
            "room_type": "shop",
            "map_point_type": "shop",
            "source_type": "shop",
        }
    if action == "choose_event_option":
        candidate_ids = [_runtime_event_candidate_id(state, candidate) for candidate in legal_actions if candidate.action == "choose_event_option"]
        candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id is not None]
        chosen_action = _runtime_event_candidate_id(state, chosen_candidate)
        if chosen_action is None or not candidate_ids:
            return None
        return {
            "decision_type": "event_choice",
            "support_quality": "full_candidates",
            "chosen_action": chosen_action,
            "chosen_index": _semantic_chosen_index(tuple(candidate_ids), chosen_action),
            "candidate_ids": tuple(candidate_ids),
            "room_type": "event",
            "map_point_type": _runtime_map_point_type(raw_payload, fallback="event"),
            "source_type": "event",
        }
    if action == "choose_rest_option":
        candidate_ids = [_runtime_rest_candidate_id(state, candidate) for candidate in legal_actions if candidate.action == "choose_rest_option"]
        candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id is not None]
        chosen_action = _runtime_rest_candidate_id(state, chosen_candidate)
        if chosen_action is None or not candidate_ids:
            return None
        return {
            "decision_type": "rest_site_action",
            "support_quality": "full_candidates",
            "chosen_action": chosen_action,
            "chosen_index": _semantic_chosen_index(tuple(candidate_ids), chosen_action),
            "candidate_ids": tuple(candidate_ids),
            "room_type": "rest",
            "map_point_type": "rest",
            "source_type": "rest",
        }
    if action == "select_deck_card" and state.selection is not None:
        decision_type = _runtime_selection_decision_type(state)
        if decision_type is None:
            return None
        candidate_ids = [_runtime_selection_candidate_id(state, candidate) for candidate in legal_actions if candidate.action == "select_deck_card"]
        candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id is not None]
        chosen_action = _runtime_selection_candidate_id(state, chosen_candidate)
        if chosen_action is None or not candidate_ids:
            return None
        selection_metadata = _runtime_selection_metadata(raw_payload=raw_payload, state=state)
        return {
            "decision_type": decision_type,
            "support_quality": "full_candidates",
            "chosen_action": chosen_action,
            "chosen_index": _semantic_chosen_index(tuple(candidate_ids), chosen_action),
            "candidate_ids": tuple(candidate_ids),
            "room_type": _runtime_selection_room_type(raw_payload, state=state),
            "map_point_type": _runtime_selection_map_point_type(raw_payload, state=state),
            "source_type": selection_metadata.get("selection_source_type", "selection"),
            "metadata": selection_metadata,
        }
    return None


def _resolve_transfer_state(path: Path | None) -> _TransferState | None:
    if path is None:
        return None
    checkpoint_path = path.expanduser().resolve()
    payload = _load_transfer_payload(checkpoint_path)
    algorithm = str(payload.get("algorithm"))
    if algorithm not in {"strategic_pretrain", "strategic_finetune"}:
        raise ValueError(f"Unsupported warm-start checkpoint algorithm: {algorithm}")
    if int(payload.get("feature_schema_version", 0) or 0) != STRATEGIC_PRETRAIN_FEATURE_SCHEMA_VERSION:
        raise ValueError("Warm-start checkpoint feature schema is incompatible with strategic_finetune.")
    return _TransferState(
        checkpoint_path=checkpoint_path,
        metadata=dict(payload.get("metadata", {})),
        global_ranking_head=_head_from_payload(payload.get("global_ranking_head", {})),
        decision_ranking_heads={
            str(name): _head_from_payload(head_payload)
            for name, head_payload in dict(payload.get("decision_ranking_heads", {})).items()
        },
        global_value_head=_head_from_payload(payload.get("global_value_head", {})),
        decision_value_heads={
            str(name): _head_from_payload(head_payload)
            for name, head_payload in dict(payload.get("decision_value_heads", {})).items()
        },
    )


def _validate_build_lineage_guardrails(
    runtime_resolved: _ResolvedSourceDataset,
    public_resolved: _ResolvedSourceDataset | None,
    transfer: _TransferState | None,
    *,
    config: StrategicFinetuneTrainConfig,
) -> None:
    runtime_summary = _dataset_summary_payload(runtime_resolved)
    if public_resolved is not None and config.enforce_runtime_public_build_match:
        _validate_build_lineage_overlap(
            left_summary=runtime_summary,
            right_summary=_dataset_summary_payload(public_resolved),
            left_label="runtime dataset",
            right_label="public dataset",
        )
    if transfer is None:
        return
    if config.enforce_public_build_match and public_resolved is not None:
        transfer_public_summary = _transfer_public_dataset_summary(transfer)
        if transfer_public_summary is not None:
            _validate_build_lineage_overlap(
                left_summary=transfer_public_summary,
                right_summary=_dataset_summary_payload(public_resolved),
                left_label="warm-start public lineage",
                right_label="public dataset",
            )
    transfer_runtime_summary = _transfer_runtime_dataset_summary(transfer)
    if transfer_runtime_summary is not None:
        _validate_build_lineage_overlap(
            left_summary=transfer_runtime_summary,
            right_summary=runtime_summary,
            left_label="warm-start runtime lineage",
            right_label="runtime dataset",
        )


def _transfer_public_dataset_summary(transfer: _TransferState) -> dict[str, Any] | None:
    payload = transfer.metadata.get("public_dataset_summary")
    if isinstance(payload, dict):
        return payload
    payload = transfer.metadata.get("dataset_summary")
    if isinstance(payload, dict):
        return payload
    return None


def _transfer_runtime_dataset_summary(transfer: _TransferState) -> dict[str, Any] | None:
    payload = transfer.metadata.get("runtime_dataset_summary")
    return payload if isinstance(payload, dict) else None


def _validate_build_lineage_overlap(
    *,
    left_summary: dict[str, Any],
    right_summary: dict[str, Any],
    left_label: str,
    right_label: str,
) -> None:
    lineage_fields = (
        ("build_id_histogram", "build ids"),
        ("game_version_histogram", "game versions"),
        ("branch_histogram", "branches"),
        ("content_channel_histogram", "content channels"),
    )
    for field_name, label in lineage_fields:
        left_values = _histogram_value_set(left_summary.get(field_name))
        right_values = _histogram_value_set(right_summary.get(field_name))
        if left_values and right_values and left_values.isdisjoint(right_values):
            raise ValueError(
                f"{left_label} and {right_label} do not overlap on {label}. "
                f"{left_label}={sorted(left_values)} {right_label}={sorted(right_values)}"
            )


def _histogram_value_set(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        return set()
    return {str(key).strip() for key, value in payload.items() if value and str(key).strip()}


def _initialize_heads(
    *,
    decision_types: Sequence[str],
    transfer: _TransferState | None,
) -> tuple[_SparseScoringHead, dict[str, _SparseScoringHead], _SparseScoringHead, dict[str, _SparseScoringHead], dict[str, Any]]:
    if transfer is None:
        global_ranking_head = _SparseScoringHead(name="global_ranking")
        global_value_head = _SparseScoringHead(name="global_value")
        decision_ranking_heads = {decision_type: _SparseScoringHead(name=decision_type) for decision_type in decision_types}
        decision_value_heads = {decision_type: _SparseScoringHead(name=decision_type) for decision_type in decision_types}
        transferred_modules = {
            "global_ranking": False,
            "global_value": False,
            "decision_ranking_heads": [],
            "decision_value_heads": [],
        }
        return global_ranking_head, decision_ranking_heads, global_value_head, decision_value_heads, transferred_modules

    global_ranking_head = transfer.global_ranking_head.clone()
    global_value_head = transfer.global_value_head.clone()
    decision_ranking_heads: dict[str, _SparseScoringHead] = {}
    decision_value_heads: dict[str, _SparseScoringHead] = {}
    transferred_ranking_heads: list[str] = []
    transferred_value_heads: list[str] = []
    for decision_type in decision_types:
        ranking_head = transfer.decision_ranking_heads.get(decision_type)
        if ranking_head is None:
            decision_ranking_heads[decision_type] = _SparseScoringHead(name=decision_type)
        else:
            decision_ranking_heads[decision_type] = ranking_head.clone()
            transferred_ranking_heads.append(decision_type)
        value_head = transfer.decision_value_heads.get(decision_type)
        if value_head is None:
            decision_value_heads[decision_type] = _SparseScoringHead(name=decision_type)
        else:
            decision_value_heads[decision_type] = value_head.clone()
            transferred_value_heads.append(decision_type)
    transferred_modules = {
        "global_ranking": True,
        "global_value": True,
        "decision_ranking_heads": sorted(transferred_ranking_heads),
        "decision_value_heads": sorted(transferred_value_heads),
        "warmstart_algorithm": transfer.metadata.get("artifact_kind") or transfer.metadata.get("algorithm"),
    }
    return global_ranking_head, decision_ranking_heads, global_value_head, decision_value_heads, transferred_modules


def _ranking_transfer_applies(decision_type: str, transferred_modules: dict[str, Any]) -> bool:
    return bool(transferred_modules.get("global_ranking")) or decision_type in set(transferred_modules.get("decision_ranking_heads", []))


def _value_transfer_applies(decision_type: str, transferred_modules: dict[str, Any]) -> bool:
    return bool(transferred_modules.get("global_value")) or decision_type in set(transferred_modules.get("decision_value_heads", []))


def _build_epoch_training_plan(
    *,
    runtime_examples: Sequence[StrategicFinetuneExample],
    public_examples: Sequence[StrategicFinetuneExample],
    config: StrategicFinetuneTrainConfig,
    rng: random.Random,
) -> tuple[list[StrategicFinetuneExample], dict[str, int]]:
    runtime_plan = list(runtime_examples) * config.runtime_replay_passes
    public_plan = list(public_examples) * config.public_replay_passes
    if not public_plan:
        rng.shuffle(runtime_plan)
        return runtime_plan, {"runtime": len(runtime_plan), "public": 0}
    if config.schedule == "weighted_shuffle":
        epoch_examples = [*runtime_plan, *public_plan]
        rng.shuffle(epoch_examples)
        return epoch_examples, {"runtime": len(runtime_plan), "public": len(public_plan)}
    runtime_plan = _shuffled_copy(runtime_plan, rng)
    public_plan = _shuffled_copy(public_plan, rng)
    epoch_examples: list[StrategicFinetuneExample] = []
    while runtime_plan or public_plan:
        if runtime_plan:
            epoch_examples.append(runtime_plan.pop())
        if public_plan:
            epoch_examples.append(public_plan.pop())
    return epoch_examples, {"runtime": config.runtime_replay_passes * len(runtime_examples), "public": config.public_replay_passes * len(public_examples)}


def _train_on_finetune_example(
    *,
    example: StrategicFinetuneExample,
    global_ranking_head: _SparseScoringHead,
    decision_ranking_head: _SparseScoringHead,
    global_value_head: _SparseScoringHead,
    decision_value_head: _SparseScoringHead,
    learning_rate: float,
    l2: float,
    chosen_only_positive_weight: float,
    auxiliary_value_weight: float,
    freeze_ranking: bool,
    freeze_value: bool,
) -> None:
    if not freeze_ranking:
        ranking_weight = example.sample_weight
        if example.support_quality == "full_candidates":
            max_score = max(scores) if (scores := [
                global_ranking_head.score(feature_map) + decision_ranking_head.score(feature_map)
                for feature_map in example.candidate_feature_maps
            ]) else 0.0
            total = sum(_safe_exp(score - max_score) for score in scores) if scores else 0.0
            probabilities = [
                (_safe_exp(score - max_score) / total) if total > 0.0 else (1.0 / len(scores))
                for score in scores
            ]
            for candidate_index, feature_map in enumerate(example.candidate_feature_maps):
                target = 1.0 if candidate_index == example.chosen_index else 0.0
                gradient_scale = (probabilities[candidate_index] - target) * ranking_weight
                _apply_sparse_gradient(global_ranking_head.weights, feature_map, gradient_scale, learning_rate, l2)
                _apply_sparse_gradient(decision_ranking_head.weights, feature_map, gradient_scale, learning_rate, l2)
        else:
            chosen_feature_map = example.candidate_feature_maps[example.chosen_index]
            chosen_score = global_ranking_head.score(chosen_feature_map) + decision_ranking_head.score(chosen_feature_map)
            chosen_probability = _sigmoid(chosen_score)
            gradient_scale = (chosen_probability - 1.0) * example.sample_weight * chosen_only_positive_weight
            _apply_sparse_gradient(global_ranking_head.weights, chosen_feature_map, gradient_scale, learning_rate, l2)
            _apply_sparse_gradient(decision_ranking_head.weights, chosen_feature_map, gradient_scale, learning_rate, l2)

    outcome_label = 1.0 if example.run_outcome == "win" else 0.0 if example.run_outcome == "loss" else None
    if outcome_label is not None and auxiliary_value_weight > 0.0 and not freeze_value:
        context_score = global_value_head.score(example.context_feature_map) + decision_value_head.score(example.context_feature_map)
        predicted_probability = _sigmoid(context_score)
        gradient_scale = (predicted_probability - outcome_label) * example.sample_weight * auxiliary_value_weight
        _apply_sparse_gradient(global_value_head.weights, example.context_feature_map, gradient_scale, learning_rate, l2)
        _apply_sparse_gradient(decision_value_head.weights, example.context_feature_map, gradient_scale, learning_rate, l2)


def _evaluate_finetune_examples(
    model: StrategicFinetuneModel,
    examples: Sequence[StrategicFinetuneExample],
    *,
    top_k: Sequence[int],
) -> dict[str, Any]:
    payload = evaluate_strategic_pretrain_examples(model, examples, top_k=top_k)
    source_groups: dict[str, list[StrategicFinetuneExample]] = defaultdict(list)
    for example in examples:
        source_groups[example.source_kind].append(example)
    payload["source_metrics"] = {
        source_kind: evaluate_strategic_pretrain_examples(model, grouped_examples, top_k=top_k)
        for source_kind, grouped_examples in sorted(source_groups.items())
    }
    return payload


def _dataset_summary_payload(resolved: _ResolvedSourceDataset) -> dict[str, Any]:
    decision_type_histogram = dict(Counter(example.decision_type for example in resolved.examples))
    support_quality_histogram = dict(Counter(example.support_quality for example in resolved.examples))
    build_id_histogram = dict(Counter(example.build_id for example in resolved.examples if example.build_id))
    game_version_histogram = dict(Counter(example.game_version for example in resolved.examples if example.game_version))
    branch_histogram = dict(Counter(example.branch for example in resolved.examples if example.branch))
    content_channel_histogram = dict(Counter(example.content_channel for example in resolved.examples if example.content_channel))
    return {
        "dataset_path": str(resolved.dataset_path),
        "examples_path": str(resolved.examples_path) if resolved.examples_path is not None else None,
        "train_examples_path": str(resolved.train_examples_path) if resolved.train_examples_path is not None else None,
        "validation_examples_path": str(resolved.validation_examples_path) if resolved.validation_examples_path is not None else None,
        "test_examples_path": str(resolved.test_examples_path) if resolved.test_examples_path is not None else None,
        "example_count": len(resolved.examples),
        "train_example_count": len(resolved.train_examples),
        "validation_example_count": len(resolved.validation_examples),
        "test_example_count": len(resolved.test_examples),
        "split_strategy": resolved.split_strategy,
        "skipped_records": resolved.skipped_records,
        "dropped_records": resolved.dropped_records,
        "dataset_lineage": None if resolved.dataset_summary is None else resolved.dataset_summary.get("lineage"),
        "decision_type_histogram": decision_type_histogram,
        "support_quality_histogram": support_quality_histogram,
        "build_id_histogram": build_id_histogram,
        "game_version_histogram": game_version_histogram,
        "branch_histogram": branch_histogram,
        "content_channel_histogram": content_channel_histogram,
    }


def _trajectory_step_record_from_payload(payload: dict[str, Any]) -> TrajectoryStepRecord | None:
    try:
        sanitized = {name: payload[name] for name in TrajectoryStepRecord.model_fields if name in payload}
        return TrajectoryStepRecord.model_validate(sanitized)
    except Exception:
        return None


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


def _runtime_reward_candidate_id(state: GameStatePayload, candidate: CandidateAction) -> str | None:
    if candidate.action == "skip_reward_cards":
        return "skip"
    if state.reward is None or candidate.request.option_index is None:
        return None
    option = next((item for item in state.reward.card_options if item.index == candidate.request.option_index), None)
    return None if option is None else option.card_id


def _runtime_shop_buy_candidate_id(state: GameStatePayload, candidate: CandidateAction) -> str | None:
    if state.shop is None or candidate.request.option_index is None:
        return None
    option = next((item for item in state.shop.cards if item.index == candidate.request.option_index), None)
    return None if option is None else option.card_id


def _runtime_event_candidate_id(state: GameStatePayload, candidate: CandidateAction) -> str | None:
    if state.event is None or candidate.request.option_index is None:
        return None
    option = next((item for item in state.event.options if item.index == candidate.request.option_index), None)
    if option is None:
        return None
    return option.text_key or _slug_token(option.title)


def _runtime_rest_candidate_id(state: GameStatePayload, candidate: CandidateAction) -> str | None:
    if state.rest is None or candidate.request.option_index is None:
        return None
    option = next((item for item in state.rest.options if item.index == candidate.request.option_index), None)
    return None if option is None else option.option_id


def _runtime_selection_candidate_id(state: GameStatePayload, candidate: CandidateAction) -> str | None:
    if state.selection is None or candidate.request.option_index is None:
        return None
    option = next((item for item in state.selection.cards if item.index == candidate.request.option_index), None)
    return None if option is None else option.card_id


def _selection_mode_from_state(state: GameStatePayload) -> str:
    selection = state.selection
    if selection is None:
        return "pick"
    mode = str(selection.semantic_mode or "").strip().lower()
    return mode or "pick"


def _runtime_selection_decision_type(state: GameStatePayload) -> str | None:
    return selection_decision_type_for_mode(_selection_mode_from_state(state))


def _runtime_selection_metadata(*, raw_payload: dict[str, Any], state: GameStatePayload) -> dict[str, Any]:
    selection = state.selection
    if selection is None:
        return {}
    transaction = {}
    decision_metadata = raw_payload.get("decision_metadata")
    if isinstance(decision_metadata, dict):
        payload = decision_metadata.get("selection_transaction")
        if isinstance(payload, dict):
            transaction = payload
    return {
        "selection_semantic_mode": _selection_mode_from_state(state),
        "selection_source_type": str(selection.source_type or "").strip().lower() or "selection",
        "selection_source_action": _normalized_optional(selection.source_action),
        "selection_source_event_id": _normalized_optional(selection.source_event_id),
        "selection_source_event_option_text_key": _normalized_optional(selection.source_event_option_text_key),
        "selection_source_rest_option_id": _normalized_optional(selection.source_rest_option_id),
        "selection_required_count": int(selection.required_count or 0),
        "selection_selected_count": int(selection.selected_count or 0),
        "selection_remaining_count": int(selection.remaining_count or 0),
        "selection_supports_multi_select": bool(selection.supports_multi_select),
        "selection_phase": _normalized_optional(transaction.get("phase")),
        "selection_recovery_count": int(transaction.get("recovery_count", 0) or 0),
    }


def _runtime_reward_domain(*, raw_payload: dict[str, Any], state: GameStatePayload) -> tuple[str, str]:
    reward = state.reward
    fallback_source = _normalized_optional(None if reward is None else reward.source_type)
    if fallback_source == "combat":
        fallback_source = "monster"
    fallback_room_type = _normalized_optional(None if reward is None else reward.source_room_type)
    room_type = _runtime_room_type(raw_payload, fallback=fallback_room_type or fallback_source or "monster")
    map_point_type = _runtime_map_point_type(raw_payload, fallback=fallback_source or room_type)
    return room_type, map_point_type


def _runtime_selection_room_type(raw_payload: dict[str, Any], *, state: GameStatePayload) -> str:
    selection = state.selection
    fallback = str(selection.source_room_type or selection.source_type or "selection").strip().lower() if selection is not None else "selection"
    return _runtime_room_type(raw_payload, fallback=fallback or "selection")


def _runtime_selection_map_point_type(raw_payload: dict[str, Any], *, state: GameStatePayload) -> str:
    selection = state.selection
    fallback = str(selection.source_type or "selection").strip().lower() if selection is not None else "selection"
    return _runtime_map_point_type(raw_payload, fallback=fallback or "selection")


def _runtime_room_type(raw_payload: dict[str, Any], *, fallback: str) -> str:
    strategic_context = raw_payload.get("strategic_context")
    if isinstance(strategic_context, dict):
        value = _normalized_optional(strategic_context.get("room_type"))
        if value is not None:
            return value
    return fallback


def _runtime_map_point_type(raw_payload: dict[str, Any], *, fallback: str) -> str:
    strategic_context = raw_payload.get("strategic_context")
    if isinstance(strategic_context, dict):
        value = _normalized_optional(strategic_context.get("map_point_type"))
        if value is not None:
            return value
    return fallback


def _runtime_run_outcome(record: TrajectoryStepRecord) -> str | None:
    if isinstance(record.info, dict):
        outcome = _normalized_optional(record.info.get("run_outcome"))
        if outcome is not None:
            return outcome
    if isinstance(record.state, dict):
        return _normalized_optional(record.state.get("run_outcome"))
    return None


def _semantic_chosen_index(candidate_ids: tuple[str, ...], chosen_action: str) -> int:
    for index, candidate_id in enumerate(candidate_ids):
        if candidate_id == chosen_action:
            return index
    return 0


def _random_split_examples(
    examples: Sequence[StrategicFinetuneExample],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[StrategicFinetuneExample], list[StrategicFinetuneExample], list[StrategicFinetuneExample]]:
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


def _load_transfer_payload(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", 0) or 0)
    if schema_version != STRATEGIC_FINETUNE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported strategic checkpoint schema_version.")
    return payload


def _strategic_finetune_model_from_payload(payload: dict[str, Any]) -> StrategicFinetuneModel:
    return StrategicFinetuneModel(
        global_ranking_head=_head_from_payload(payload.get("global_ranking_head", {})),
        decision_ranking_heads={
            str(name): _head_from_payload(head_payload)
            for name, head_payload in dict(payload.get("decision_ranking_heads", {})).items()
        },
        global_value_head=_head_from_payload(payload.get("global_value_head", {})),
        decision_value_heads={
            str(name): _head_from_payload(head_payload)
            for name, head_payload in dict(payload.get("decision_value_heads", {})).items()
        },
        metadata=dict(payload.get("metadata", {})),
    )


def _shuffled_copy(
    examples: Sequence[StrategicFinetuneExample],
    rng: random.Random,
) -> list[StrategicFinetuneExample]:
    copied = list(examples)
    rng.shuffle(copied)
    return copied
def _safe_exp(value: float) -> float:
    import math

    return math.exp(value)


def _sigmoid(value: float) -> float:
    import math

    if value >= 0.0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _slug_token(value: Any) -> str:
    normalized = _normalized_optional(value)
    if normalized is None:
        return ""
    return normalized.replace(" ", "_")


def _strategic_finetune_config_payload(config: StrategicFinetuneTrainConfig) -> dict[str, Any]:
    return {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "l2": config.l2,
        "validation_fraction": config.validation_fraction,
        "test_fraction": config.test_fraction,
        "seed": config.seed,
        "include_decision_types": list(config.include_decision_types),
        "include_runtime_decision_sources": list(config.include_runtime_decision_sources),
        "include_runtime_policy_packs": list(config.include_runtime_policy_packs),
        "include_runtime_policy_names": list(config.include_runtime_policy_names),
        "include_public_support_qualities": list(config.include_public_support_qualities),
        "include_public_source_names": list(config.include_public_source_names),
        "include_public_build_ids": list(config.include_public_build_ids),
        "runtime_min_floor": config.runtime_min_floor,
        "runtime_max_floor": config.runtime_max_floor,
        "public_min_floor": config.public_min_floor,
        "public_max_floor": config.public_max_floor,
        "public_min_confidence": config.public_min_confidence,
        "top_k": list(config.top_k),
        "decision_type_weights": dict(config.decision_type_weights),
        "source_name_weights": dict(config.source_name_weights),
        "build_id_weights": dict(config.build_id_weights),
        "run_outcome_weights": dict(config.run_outcome_weights),
        "runtime_example_weight": config.runtime_example_weight,
        "public_example_weight": config.public_example_weight,
        "confidence_power": config.confidence_power,
        "chosen_only_positive_weight": config.chosen_only_positive_weight,
        "auxiliary_value_weight": config.auxiliary_value_weight,
        "schedule": config.schedule,
        "runtime_replay_passes": config.runtime_replay_passes,
        "public_replay_passes": config.public_replay_passes,
        "warmstart_checkpoint_path": (
            str(config.warmstart_checkpoint_path.expanduser().resolve())
            if config.warmstart_checkpoint_path is not None
            else None
        ),
        "freeze_transferred_ranking_epochs": config.freeze_transferred_ranking_epochs,
        "freeze_transferred_value_epochs": config.freeze_transferred_value_epochs,
        "enforce_public_build_match": config.enforce_public_build_match,
        "enforce_runtime_public_build_match": config.enforce_runtime_public_build_match,
        "runtime_source_name": config.runtime_source_name,
        "runtime_game_mode": config.runtime_game_mode,
        "runtime_platform_type": config.runtime_platform_type,
        "runtime_build_id": config.runtime_build_id,
    }
