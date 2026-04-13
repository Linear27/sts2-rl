from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from sts2_rl.data import (
    load_dataset_summary,
    load_public_strategic_decision_records,
    resolve_dataset_split_paths,
)

STRATEGIC_PRETRAIN_CHECKPOINT_SCHEMA_VERSION = 1
STRATEGIC_PRETRAIN_TRAINING_SCHEMA_VERSION = 1
STRATEGIC_PRETRAIN_FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StrategicPretrainTrainConfig:
    epochs: int = 60
    learning_rate: float = 0.05
    l2: float = 0.0001
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 0
    include_decision_types: tuple[str, ...] = ()
    include_support_qualities: tuple[str, ...] = ()
    include_source_names: tuple[str, ...] = ()
    include_build_ids: tuple[str, ...] = ()
    min_floor: int | None = None
    max_floor: int | None = None
    min_confidence: float = 0.0
    top_k: tuple[int, ...] = (1, 3)
    decision_type_weights: dict[str, float] = field(default_factory=dict)
    support_quality_weights: dict[str, float] = field(default_factory=dict)
    source_name_weights: dict[str, float] = field(default_factory=dict)
    build_id_weights: dict[str, float] = field(default_factory=dict)
    run_outcome_weights: dict[str, float] = field(default_factory=dict)
    confidence_power: float = 1.0
    chosen_only_positive_weight: float = 0.35
    auxiliary_value_weight: float = 0.75

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
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError("min_confidence must be between 0 and 1.")
        if not self.top_k or any(value < 1 for value in self.top_k):
            raise ValueError("top_k must contain positive entries.")
        for mapping in (
            self.decision_type_weights,
            self.support_quality_weights,
            self.source_name_weights,
            self.build_id_weights,
            self.run_outcome_weights,
        ):
            for value in mapping.values():
                if value <= 0.0:
                    raise ValueError("weight values must be positive.")
        if self.confidence_power < 0.0:
            raise ValueError("confidence_power must be non-negative.")
        if self.chosen_only_positive_weight <= 0.0:
            raise ValueError("chosen_only_positive_weight must be positive.")
        if self.auxiliary_value_weight < 0.0:
            raise ValueError("auxiliary_value_weight must be non-negative.")


@dataclass(frozen=True)
class StrategicPretrainTrainingReport:
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
    decision_type_count: int
    best_epoch: int
    split_strategy: str


@dataclass(frozen=True)
class StrategicPretrainExample:
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


@dataclass(frozen=True)
class _ResolvedStrategicDataset:
    dataset_path: Path
    examples_path: Path | None
    train_examples_path: Path | None
    validation_examples_path: Path | None
    test_examples_path: Path | None
    examples: list[StrategicPretrainExample]
    train_examples: list[StrategicPretrainExample]
    validation_examples: list[StrategicPretrainExample]
    test_examples: list[StrategicPretrainExample]
    split_strategy: str
    skipped_records: int
    dropped_records: int
    dataset_summary: dict[str, Any] | None


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
class StrategicPretrainModel:
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
        return _sigmoid(score)

    def rank_actions(
        self,
        *,
        decision_type: str,
        context: dict[str, Any],
        candidate_actions: Sequence[str],
        support_quality: str = "full_candidates",
    ) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for candidate_id in candidate_actions:
            feature_map = _strategic_candidate_feature_map(
                context=context,
                decision_type=decision_type,
                support_quality=support_quality,
                candidate_id=candidate_id,
                candidate_count=len(candidate_actions),
            )
            ranked.append(
                {
                    "candidate_id": candidate_id,
                    "score": self.score_candidate(decision_type=decision_type, feature_map=feature_map),
                }
            )
        return sorted(ranked, key=lambda item: (-float(item["score"]), str(item["candidate_id"])))

    def save(self, path: str | Path) -> Path:
        checkpoint_path = Path(path).expanduser().resolve()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": STRATEGIC_PRETRAIN_CHECKPOINT_SCHEMA_VERSION,
            "algorithm": "strategic_pretrain",
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
    def load(cls, path: str | Path) -> "StrategicPretrainModel":
        checkpoint_path = Path(path).expanduser().resolve()
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != STRATEGIC_PRETRAIN_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("Unsupported strategic pretrain checkpoint schema_version.")
        if payload.get("algorithm") != "strategic_pretrain":
            raise ValueError(f"Unsupported checkpoint algorithm: {payload.get('algorithm')}")
        return cls(
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


def default_strategic_pretrain_session_name() -> str:
    return datetime.now(UTC).strftime("strategic-pretrain-%Y%m%d-%H%M%S")


def load_strategic_pretrain_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return {
        "schema_version": payload.get("schema_version"),
        "algorithm": payload.get("algorithm"),
        "feature_schema_version": payload.get("feature_schema_version"),
        "metadata": payload.get("metadata", {}),
    }


def train_strategic_pretrain_policy(
    *,
    dataset_source: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    config: StrategicPretrainTrainConfig | None = None,
) -> StrategicPretrainTrainingReport:
    train_config = config or StrategicPretrainTrainConfig()
    dataset_path = Path(dataset_source).expanduser().resolve()
    resolved = _resolve_strategic_dataset(dataset_path, config=train_config)
    if not resolved.train_examples:
        raise ValueError("Strategic pretraining requires at least one train example.")

    output_dir = Path(output_root).expanduser().resolve() / (
        session_name or default_strategic_pretrain_session_name()
    )
    if output_dir.exists():
        raise FileExistsError(f"Strategic pretraining output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    feature_inventory = sorted(
        {
            feature_name
            for example in resolved.examples
            for feature_name in example.context_feature_map
        }
        | {
            feature_name
            for example in resolved.examples
            for candidate_features in example.candidate_feature_maps
            for feature_name in candidate_features
        }
    )
    feature_hash = hashlib.sha256("\n".join(feature_inventory).encode("utf-8")).hexdigest()
    decision_types = sorted({example.decision_type for example in resolved.train_examples})

    global_ranking_head = _SparseScoringHead(name="global_ranking")
    global_value_head = _SparseScoringHead(name="global_value")
    decision_ranking_heads = {decision_type: _SparseScoringHead(name=decision_type) for decision_type in decision_types}
    decision_value_heads = {decision_type: _SparseScoringHead(name=decision_type) for decision_type in decision_types}
    for example in resolved.train_examples:
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
            shuffled_examples = list(resolved.train_examples)
            rng.shuffle(shuffled_examples)
            for example in shuffled_examples:
                _train_on_strategic_example(
                    example=example,
                    global_ranking_head=global_ranking_head,
                    decision_ranking_head=decision_ranking_heads[example.decision_type],
                    global_value_head=global_value_head,
                    decision_value_head=decision_value_heads[example.decision_type],
                    learning_rate=train_config.learning_rate,
                    l2=train_config.l2,
                    chosen_only_positive_weight=train_config.chosen_only_positive_weight,
                    auxiliary_value_weight=train_config.auxiliary_value_weight,
                )

            epoch_model = StrategicPretrainModel(
                global_ranking_head=global_ranking_head.clone(),
                decision_ranking_heads={name: head.clone() for name, head in decision_ranking_heads.items()},
                global_value_head=global_value_head.clone(),
                decision_value_heads={name: head.clone() for name, head in decision_value_heads.items()},
                metadata={},
            )
            train_metrics = evaluate_strategic_pretrain_examples(epoch_model, resolved.train_examples, top_k=train_config.top_k)
            validation_metrics = evaluate_strategic_pretrain_examples(
                epoch_model,
                resolved.validation_examples,
                top_k=train_config.top_k,
            )
            test_metrics = evaluate_strategic_pretrain_examples(epoch_model, resolved.test_examples, top_k=train_config.top_k)
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
                        "schema_version": STRATEGIC_PRETRAIN_TRAINING_SCHEMA_VERSION,
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
        "decision_types": decision_types,
        "dataset_summary": resolved.dataset_summary,
        "training_config": _strategic_pretrain_config_payload(train_config),
        "best_epoch": best_epoch,
        "best_objective": best_objective,
        "train_metrics": best_train_metrics,
        "validation_metrics": best_validation_metrics,
        "test_metrics": best_test_metrics,
        "artifact_kind": "strategic_pretrain",
    }

    checkpoint_path = StrategicPretrainModel(
        global_ranking_head=global_ranking_head,
        decision_ranking_heads=decision_ranking_heads,
        global_value_head=global_value_head,
        decision_value_heads=decision_value_heads,
        metadata=model_metadata,
    ).save(output_dir / "strategic-pretrain-checkpoint.json")
    best_checkpoint_path = StrategicPretrainModel(
        global_ranking_head=best_global_ranking_head,
        decision_ranking_heads=best_decision_ranking_heads,
        global_value_head=best_global_value_head,
        decision_value_heads=best_decision_value_heads,
        metadata=model_metadata,
    ).save(output_dir / "strategic-pretrain-best.json")

    summary_payload = {
        "schema_version": STRATEGIC_PRETRAIN_TRAINING_SCHEMA_VERSION,
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
        "decision_type_count": len(decision_types),
        "decision_types": decision_types,
        "best_epoch": best_epoch,
        "best_objective": best_objective,
        "skipped_records": resolved.skipped_records,
        "dropped_records": resolved.dropped_records,
        "config": _strategic_pretrain_config_payload(train_config),
        "train": best_train_metrics,
        "validation": best_validation_metrics,
        "test": best_test_metrics,
        "dataset_lineage": None if resolved.dataset_summary is None else resolved.dataset_summary.get("lineage"),
        "checkpoint_metadata": load_strategic_pretrain_checkpoint_metadata(best_checkpoint_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return StrategicPretrainTrainingReport(
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
        decision_type_count=len(decision_types),
        best_epoch=best_epoch,
        split_strategy=resolved.split_strategy,
    )


def _resolve_strategic_dataset(
    dataset_path: Path,
    *,
    config: StrategicPretrainTrainConfig,
) -> _ResolvedStrategicDataset:
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
                all_examples, all_skipped, all_dropped = _load_strategic_examples_from_paths(
                    [path for path in (train_examples_path, validation_examples_path, test_examples_path) if path is not None and path.exists()],
                    config=config,
                )
                train_examples, _, _ = _load_strategic_examples_from_paths([train_examples_path], config=config)
                validation_examples, _, _ = _load_strategic_examples_from_paths(
                    [validation_examples_path] if validation_examples_path is not None else [],
                    config=config,
                )
                test_examples, _, _ = _load_strategic_examples_from_paths(
                    [test_examples_path] if test_examples_path is not None else [],
                    config=config,
                )
                return _ResolvedStrategicDataset(
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

    examples, skipped_records, dropped_records = _load_strategic_examples_from_paths([dataset_path], config=config)
    train_examples, validation_examples, test_examples = _random_split_strategic_examples(
        examples,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    return _ResolvedStrategicDataset(
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


def _load_strategic_examples_from_paths(
    paths: Sequence[Path | None],
    *,
    config: StrategicPretrainTrainConfig,
) -> tuple[list[StrategicPretrainExample], int, int]:
    examples: list[StrategicPretrainExample] = []
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
            example = _example_from_public_record(record=record, source_path=path, config=config)
            if example is None:
                dropped_records += 1
                continue
            examples.append(example)
    return examples, skipped_records, dropped_records


def _example_from_public_record(
    *,
    record: Any,
    source_path: Path,
    config: StrategicPretrainTrainConfig,
) -> StrategicPretrainExample | None:
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
    if config.include_support_qualities and support_quality not in {item.lower() for item in config.include_support_qualities}:
        return None
    if config.include_source_names and source_name not in {item.lower() for item in config.include_source_names}:
        return None
    if config.include_build_ids:
        allowed_builds = {item for item in config.include_build_ids}
        if build_id is None or build_id not in allowed_builds:
            return None
    if config.min_floor is not None and (record.floor is None or record.floor < config.min_floor):
        return None
    if config.max_floor is not None and (record.floor is None or record.floor > config.max_floor):
        return None
    if float(record.reconstruction_confidence) < config.min_confidence:
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
        "model_id": record.model_id,
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
        config=config,
        decision_type=decision_type,
        support_quality=support_quality,
        source_name=source_name,
        build_id=build_id,
        run_outcome=run_outcome,
        reconstruction_confidence=float(record.reconstruction_confidence),
    )
    if sample_weight <= 0.0:
        return None
    confidence_weight = float(record.reconstruction_confidence) ** config.confidence_power if config.confidence_power > 0.0 else 1.0
    return StrategicPretrainExample(
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
    )


def evaluate_strategic_pretrain_examples(
    model: StrategicPretrainModel,
    examples: Sequence[StrategicPretrainExample],
    *,
    top_k: Sequence[int] = (1, 3),
) -> dict[str, Any]:
    payload = _evaluate_strategic_pretrain_examples_without_submetrics(model, examples, top_k=top_k)
    if not examples:
        return payload
    decision_type_examples: dict[str, list[StrategicPretrainExample]] = defaultdict(list)
    for example in examples:
        decision_type_examples[example.decision_type].append(example)
    payload["decision_type_metrics"] = {
        decision_type: _evaluate_strategic_pretrain_examples_without_submetrics(model, stage_examples, top_k=top_k)
        for decision_type, stage_examples in sorted(decision_type_examples.items())
    }
    return payload


def _evaluate_strategic_pretrain_examples_without_submetrics(
    model: StrategicPretrainModel,
    examples: Sequence[StrategicPretrainExample],
    *,
    top_k: Sequence[int] = (1, 3),
) -> dict[str, Any]:
    if not examples:
        return {
            "example_count": 0,
            "weighted_example_count": 0.0,
            "candidate_choice": _empty_ranking_metrics(top_k),
            "positive_only": _empty_positive_only_metrics(),
            "value": _empty_value_metrics(),
            "decision_type_metrics": {},
        }

    ranking = _RankingMetrics(top_k=top_k)
    chosen_only = _PositiveOnlyMetrics()
    value = _ValueMetrics()
    for example in examples:
        _accumulate_strategic_example_metrics(
            model=model,
            example=example,
            top_k=top_k,
            ranking=ranking,
            chosen_only=chosen_only,
            value=value,
        )

    return {
        "example_count": len(examples),
        "weighted_example_count": sum(example.sample_weight for example in examples),
        "candidate_choice": ranking.as_dict(),
        "positive_only": chosen_only.as_dict(),
        "value": value.as_dict(),
        "decision_type_metrics": {},
    }


def _strategic_context_feature_map(
    *,
    context: dict[str, Any],
    decision_type: str,
    support_quality: str,
) -> dict[str, float]:
    features: dict[str, float] = {"bias": 1.0}
    _add_one_hot(features, "decision_type", decision_type)
    _add_one_hot(features, "decision_stage", _decision_stage_from_type(decision_type))
    _add_one_hot(features, "support_quality", support_quality)
    _add_one_hot(features, "source_name", _normalized_optional(context.get("source_name")) or "unknown")
    _add_one_hot(features, "character", _normalized_optional(context.get("character_id")) or "unknown")
    _add_one_hot(features, "game_mode", _normalized_optional(context.get("game_mode")) or "unknown")
    _add_one_hot(features, "platform_type", _normalized_optional(context.get("platform_type")) or "unknown")
    _add_one_hot(features, "room_type", _normalized_optional(context.get("room_type")) or "unknown")
    _add_one_hot(features, "map_point_type", _normalized_optional(context.get("map_point_type")) or "unknown")
    _add_one_hot(features, "source_type", _normalized_optional(context.get("source_type")) or "unknown")
    _add_one_hot(features, "build_id", _normalized_optional(context.get("build_id")) or "unknown")
    _add_one_hot(features, "game_version", _normalized_optional(context.get("game_version")) or "unknown")
    _add_one_hot(features, "build_branch", _normalized_optional(context.get("branch")) or "unknown")
    _add_one_hot(features, "content_channel", _normalized_optional(context.get("content_channel")) or "unknown")
    _add_one_hot(features, "act_id", _normalized_optional(context.get("act_id")) or "unknown")

    floor = context.get("floor")
    if floor is not None:
        _add_numeric(features, "floor", float(floor) / 60.0)
        _bucketize(features, "floor_band", float(floor), (4, 8, 16, 32, 64))
    floor_within_act = context.get("floor_within_act")
    if floor_within_act is not None:
        _add_numeric(features, "floor_within_act", float(floor_within_act) / 20.0)
    act_index = context.get("act_index")
    if act_index is not None:
        _add_numeric(features, "act_index", float(act_index) / 5.0)
    acts_reached = context.get("acts_reached")
    if acts_reached is not None:
        _add_numeric(features, "acts_reached", float(acts_reached) / 5.0)
    ascension = context.get("ascension")
    if ascension is not None:
        _add_numeric(features, "ascension", float(ascension) / 20.0)
        _bucketize(features, "ascension_band", float(ascension), (1, 5, 10, 15, 20))

    candidate_count = int(context.get("candidate_count") or 0)
    _add_numeric(features, "candidate_count", float(candidate_count) / 8.0)
    metadata = dict(context.get("metadata") or {})
    _add_flag(features, "has_detail_payload", bool(metadata.get("has_detail_payload")))
    _add_flag(features, "has_room_history", bool(metadata.get("has_room_history")))
    selection_mode = _normalized_optional(metadata.get("selection_semantic_mode"))
    if selection_mode is not None:
        _add_one_hot(features, "selection_semantic_mode", selection_mode)
    selection_phase = _normalized_optional(metadata.get("selection_phase"))
    if selection_phase is not None:
        _add_one_hot(features, "selection_phase", selection_phase)
    selection_source_type = _normalized_optional(metadata.get("selection_source_type"))
    if selection_source_type is not None:
        _add_one_hot(features, "selection_source_type", selection_source_type)
    if metadata.get("selection_supports_multi_select") is not None:
        _add_flag(features, "selection_supports_multi_select", bool(metadata.get("selection_supports_multi_select")))
    for field_name, scale in (
        ("selection_required_count", 4.0),
        ("selection_selected_count", 4.0),
        ("selection_remaining_count", 4.0),
        ("selection_recovery_count", 4.0),
    ):
        value = metadata.get(field_name)
        if value is not None:
            _add_numeric(features, field_name, float(value) / scale)
    return features


def _strategic_candidate_feature_map(
    *,
    context: dict[str, Any],
    decision_type: str,
    support_quality: str,
    candidate_id: str,
    candidate_count: int,
) -> dict[str, float]:
    features = _strategic_context_feature_map(context=context, decision_type=decision_type, support_quality=support_quality)
    normalized_candidate = str(candidate_id).strip().lower()
    _add_one_hot(features, "candidate", normalized_candidate)
    _add_one_hot(features, "candidate_prefix", _candidate_prefix(normalized_candidate))
    _add_flag(features, "candidate_is_skip", normalized_candidate.startswith("skip"))
    _add_flag(features, "candidate_is_rest", "rest" in normalized_candidate)
    _add_flag(features, "candidate_is_remove", "remove" in normalized_candidate)
    _add_flag(features, "candidate_is_upgrade", "upgrade" in normalized_candidate)
    _add_numeric(features, "candidate_count", float(candidate_count) / 8.0)
    _add_one_hot(features, "decision_candidate", f"{decision_type}:{normalized_candidate}")
    floor = context.get("floor")
    if floor is not None:
        _add_one_hot(features, "candidate_floor_band", f"{normalized_candidate}:{_floor_bucket_label(float(floor))}")
    room_type = _normalized_optional(context.get("room_type"))
    if room_type is not None:
        _add_one_hot(features, "candidate_room_type", f"{normalized_candidate}:{room_type}")
    source_type = _normalized_optional(context.get("source_type"))
    if source_type is not None:
        _add_one_hot(features, "candidate_source_type", f"{normalized_candidate}:{source_type}")
    return features


def _strategic_sample_weight(
    *,
    config: StrategicPretrainTrainConfig,
    decision_type: str,
    support_quality: str,
    source_name: str,
    build_id: str | None,
    run_outcome: str | None,
    reconstruction_confidence: float,
) -> float:
    weight = 1.0
    weight *= config.decision_type_weights.get(decision_type, 1.0)
    weight *= config.support_quality_weights.get(support_quality, 1.0)
    weight *= config.source_name_weights.get(source_name, 1.0)
    if build_id is not None:
        weight *= config.build_id_weights.get(build_id, 1.0)
    if run_outcome is not None:
        weight *= config.run_outcome_weights.get(run_outcome, 1.0)
    if config.confidence_power > 0.0:
        weight *= reconstruction_confidence ** config.confidence_power
    return weight


def _random_split_strategic_examples(
    examples: Sequence[StrategicPretrainExample],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[StrategicPretrainExample], list[StrategicPretrainExample], list[StrategicPretrainExample]]:
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


def _train_on_strategic_example(
    *,
    example: StrategicPretrainExample,
    global_ranking_head: _SparseScoringHead,
    decision_ranking_head: _SparseScoringHead,
    global_value_head: _SparseScoringHead,
    decision_value_head: _SparseScoringHead,
    learning_rate: float,
    l2: float,
    chosen_only_positive_weight: float,
    auxiliary_value_weight: float,
) -> None:
    ranking_weight = example.sample_weight
    if example.support_quality == "full_candidates":
        scores = [
            global_ranking_head.score(feature_map) + decision_ranking_head.score(feature_map)
            for feature_map in example.candidate_feature_maps
        ]
        probabilities = _softmax(scores)
        for candidate_index, feature_map in enumerate(example.candidate_feature_maps):
            target = 1.0 if candidate_index == example.chosen_index else 0.0
            gradient_scale = (probabilities[candidate_index] - target) * ranking_weight
            _apply_sparse_gradient(global_ranking_head.weights, feature_map, gradient_scale, learning_rate, l2)
            _apply_sparse_gradient(decision_ranking_head.weights, feature_map, gradient_scale, learning_rate, l2)
    else:
        chosen_feature_map = example.candidate_feature_maps[example.chosen_index]
        chosen_score = global_ranking_head.score(chosen_feature_map) + decision_ranking_head.score(chosen_feature_map)
        chosen_probability = _sigmoid(chosen_score)
        gradient_scale = (chosen_probability - 1.0) * ranking_weight * chosen_only_positive_weight
        _apply_sparse_gradient(global_ranking_head.weights, chosen_feature_map, gradient_scale, learning_rate, l2)
        _apply_sparse_gradient(decision_ranking_head.weights, chosen_feature_map, gradient_scale, learning_rate, l2)

    outcome_label = 1.0 if example.run_outcome == "win" else 0.0 if example.run_outcome == "loss" else None
    if outcome_label is not None and auxiliary_value_weight > 0.0:
        context_score = global_value_head.score(example.context_feature_map) + decision_value_head.score(example.context_feature_map)
        predicted_probability = _sigmoid(context_score)
        gradient_scale = (predicted_probability - outcome_label) * ranking_weight * auxiliary_value_weight
        _apply_sparse_gradient(global_value_head.weights, example.context_feature_map, gradient_scale, learning_rate, l2)
        _apply_sparse_gradient(decision_value_head.weights, example.context_feature_map, gradient_scale, learning_rate, l2)


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
class _RankingMetrics:
    top_k: Sequence[int]
    example_count: int = 0
    weighted_example_count: float = 0.0
    loss_sum: float = 0.0
    chosen_rank_sum: float = 0.0
    reciprocal_rank_sum: float = 0.0
    chosen_probability_sum: float = 0.0
    entropy_sum: float = 0.0
    candidate_count_sum: float = 0.0
    top_k_hits: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value in self.top_k:
            self.top_k_hits.setdefault(int(value), 0.0)

    def as_dict(self) -> dict[str, Any]:
        if self.weighted_example_count <= 0.0:
            return _empty_ranking_metrics(self.top_k)
        return {
            "example_count": self.example_count,
            "weighted_example_count": self.weighted_example_count,
            "loss": self.loss_sum / self.weighted_example_count,
            "top_k_accuracy": {
                str(value): self.top_k_hits[int(value)] / self.weighted_example_count for value in self.top_k
            },
            "mean_candidate_count": self.candidate_count_sum / self.weighted_example_count,
            "mean_chosen_rank": self.chosen_rank_sum / self.weighted_example_count,
            "mean_reciprocal_rank": self.reciprocal_rank_sum / self.weighted_example_count,
            "mean_chosen_probability": self.chosen_probability_sum / self.weighted_example_count,
            "mean_entropy": self.entropy_sum / self.weighted_example_count,
        }


@dataclass
class _PositiveOnlyMetrics:
    example_count: int = 0
    weighted_example_count: float = 0.0
    log_loss_sum: float = 0.0
    score_sum: float = 0.0
    probability_sum: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        if self.weighted_example_count <= 0.0:
            return _empty_positive_only_metrics()
        return {
            "example_count": self.example_count,
            "weighted_example_count": self.weighted_example_count,
            "loss": self.log_loss_sum / self.weighted_example_count,
            "mean_score": self.score_sum / self.weighted_example_count,
            "mean_probability": self.probability_sum / self.weighted_example_count,
        }


@dataclass
class _ValueMetrics:
    example_count: int = 0
    weighted_example_count: float = 0.0
    log_loss_sum: float = 0.0
    brier_sum: float = 0.0
    accuracy_hits: float = 0.0
    probability_sum: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        if self.weighted_example_count <= 0.0:
            return _empty_value_metrics()
        return {
            "example_count": self.example_count,
            "weighted_example_count": self.weighted_example_count,
            "log_loss": self.log_loss_sum / self.weighted_example_count,
            "brier": self.brier_sum / self.weighted_example_count,
            "accuracy": self.accuracy_hits / self.weighted_example_count,
            "mean_predicted_win_probability": self.probability_sum / self.weighted_example_count,
        }


def _accumulate_strategic_example_metrics(
    *,
    model: StrategicPretrainModel,
    example: StrategicPretrainExample,
    top_k: Sequence[int],
    ranking: _RankingMetrics,
    chosen_only: _PositiveOnlyMetrics,
    value: _ValueMetrics,
) -> None:
    if example.support_quality == "full_candidates":
        ranking.example_count += 1
        ranking.weighted_example_count += example.sample_weight
        scores = [
            model.score_candidate(decision_type=example.decision_type, feature_map=feature_map)
            for feature_map in example.candidate_feature_maps
        ]
        probabilities = _softmax(scores)
        chosen_probability = max(probabilities[example.chosen_index], 1e-12)
        ranking.loss_sum += (-math.log(chosen_probability)) * example.sample_weight
        ranked_indices = sorted(
            range(len(example.candidate_feature_maps)),
            key=lambda index: (-scores[index], example.candidate_ids[index]),
        )
        chosen_rank = ranked_indices.index(example.chosen_index) + 1
        ranking.chosen_rank_sum += chosen_rank * example.sample_weight
        ranking.reciprocal_rank_sum += (1.0 / chosen_rank) * example.sample_weight
        ranking.chosen_probability_sum += probabilities[example.chosen_index] * example.sample_weight
        ranking.entropy_sum += _distribution_entropy(probabilities) * example.sample_weight
        ranking.candidate_count_sum += len(example.candidate_feature_maps) * example.sample_weight
        for value_at_k in top_k:
            if chosen_rank <= value_at_k:
                ranking.top_k_hits[int(value_at_k)] += example.sample_weight
    else:
        chosen_only.example_count += 1
        chosen_only.weighted_example_count += example.sample_weight
        chosen_feature_map = example.candidate_feature_maps[example.chosen_index]
        score = model.score_candidate(decision_type=example.decision_type, feature_map=chosen_feature_map)
        probability = max(_sigmoid(score), 1e-12)
        chosen_only.log_loss_sum += (-math.log(probability)) * example.sample_weight
        chosen_only.score_sum += score * example.sample_weight
        chosen_only.probability_sum += probability * example.sample_weight

    outcome_label = 1.0 if example.run_outcome == "win" else 0.0 if example.run_outcome == "loss" else None
    if outcome_label is not None:
        predicted_probability = min(
            max(model.predict_value(decision_type=example.decision_type, context_feature_map=example.context_feature_map), 1e-12),
            1.0 - 1e-12,
        )
        value.example_count += 1
        value.weighted_example_count += example.sample_weight
        value.log_loss_sum += (
            -(outcome_label * math.log(predicted_probability) + ((1.0 - outcome_label) * math.log(1.0 - predicted_probability)))
        ) * example.sample_weight
        value.brier_sum += ((predicted_probability - outcome_label) ** 2) * example.sample_weight
        value.accuracy_hits += (1.0 if ((predicted_probability >= 0.5) == (outcome_label >= 0.5)) else 0.0) * example.sample_weight
        value.probability_sum += predicted_probability * example.sample_weight


def _select_strategic_pretrain_objective(
    train_metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> float | None:
    for payload in (validation_metrics, test_metrics, train_metrics):
        candidate_loss = (payload.get("candidate_choice") or {}).get("loss")
        value_loss = (payload.get("value") or {}).get("log_loss")
        positive_only_loss = (payload.get("positive_only") or {}).get("loss")
        active_losses = [float(loss) for loss in (candidate_loss, value_loss, positive_only_loss) if loss is not None]
        if active_losses:
            return sum(active_losses) / len(active_losses)
    return None


def _head_payload(head: _SparseScoringHead) -> dict[str, Any]:
    return {"name": head.name, "weights": dict(sorted(head.weights.items())), "example_count": head.example_count}


def _head_from_payload(payload: dict[str, Any]) -> _SparseScoringHead:
    return _SparseScoringHead(
        name=str(payload.get("name") or "unknown"),
        weights={str(name): float(value) for name, value in dict(payload.get("weights", {})).items()},
        example_count=int(payload.get("example_count", 0) or 0),
    )


def _decision_stage_from_type(decision_type: str) -> str:
    if decision_type == "reward_card_pick":
        return "reward"
    if decision_type == "shop_buy":
        return "shop"
    if decision_type.startswith("selection_"):
        return "selection"
    if decision_type == "event_choice":
        return "event"
    return "rest"


def _candidate_prefix(candidate_id: str) -> str:
    normalized = candidate_id.strip().lower()
    if "." in normalized:
        return normalized.split(".", maxsplit=1)[0]
    if "_" in normalized:
        return normalized.split("_", maxsplit=1)[0]
    return normalized


def _normalized_optional(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _add_one_hot(features: dict[str, float], namespace: str, value: str) -> None:
    features[f"{namespace}:{value}"] = 1.0


def _add_numeric(features: dict[str, float], name: str, value: float) -> None:
    features[f"num:{name}"] = float(value)


def _add_flag(features: dict[str, float], name: str, value: bool = True) -> None:
    if value:
        features[f"flag:{name}"] = 1.0


def _bucketize(features: dict[str, float], namespace: str, value: float, cutoffs: Sequence[float]) -> None:
    for cutoff in cutoffs:
        if value <= cutoff:
            _add_one_hot(features, namespace, f"<= {cutoff:g}")
            return
    _add_one_hot(features, namespace, f"> {cutoffs[-1]:g}")


def _floor_bucket_label(value: float) -> str:
    if value <= 6:
        return "very_early"
    if value <= 12:
        return "early"
    if value <= 24:
        return "mid"
    return "late"


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    total = sum(exp_scores)
    if total <= 0.0:
        return [1.0 / len(scores)] * len(scores)
    return [score / total for score in exp_scores]


def _distribution_entropy(probabilities: Sequence[float]) -> float:
    entropy = 0.0
    for value in probabilities:
        if value > 1e-12:
            entropy -= value * math.log(value)
    return entropy


def _empty_ranking_metrics(top_k: Sequence[int]) -> dict[str, Any]:
    return {
        "example_count": 0,
        "weighted_example_count": 0.0,
        "loss": None,
        "top_k_accuracy": {str(value): None for value in top_k},
        "mean_candidate_count": None,
        "mean_chosen_rank": None,
        "mean_reciprocal_rank": None,
        "mean_chosen_probability": None,
        "mean_entropy": None,
    }


def _empty_positive_only_metrics() -> dict[str, Any]:
    return {
        "example_count": 0,
        "weighted_example_count": 0.0,
        "loss": None,
        "mean_score": None,
        "mean_probability": None,
    }


def _empty_value_metrics() -> dict[str, Any]:
    return {
        "example_count": 0,
        "weighted_example_count": 0.0,
        "log_loss": None,
        "brier": None,
        "accuracy": None,
        "mean_predicted_win_probability": None,
    }


def _strategic_pretrain_config_payload(config: StrategicPretrainTrainConfig) -> dict[str, Any]:
    return {
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "l2": config.l2,
        "validation_fraction": config.validation_fraction,
        "test_fraction": config.test_fraction,
        "seed": config.seed,
        "include_decision_types": list(config.include_decision_types),
        "include_support_qualities": list(config.include_support_qualities),
        "include_source_names": list(config.include_source_names),
        "include_build_ids": list(config.include_build_ids),
        "min_floor": config.min_floor,
        "max_floor": config.max_floor,
        "min_confidence": config.min_confidence,
        "top_k": list(config.top_k),
        "decision_type_weights": dict(config.decision_type_weights),
        "support_quality_weights": dict(config.support_quality_weights),
        "source_name_weights": dict(config.source_name_weights),
        "build_id_weights": dict(config.build_id_weights),
        "run_outcome_weights": dict(config.run_outcome_weights),
        "confidence_power": config.confidence_power,
        "chosen_only_positive_weight": config.chosen_only_positive_weight,
        "auxiliary_value_weight": config.auxiliary_value_weight,
    }
