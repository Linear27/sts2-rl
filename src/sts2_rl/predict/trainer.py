from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from sts2_rl.data import load_dataset_summary, resolve_dataset_split_paths

from .dataset import load_predictor_examples, resolve_predictor_examples_path
from .model import CombatOutcomePredictor, PredictorHead
from .schema import PREDICTOR_TRAINING_SCHEMA_VERSION, PredictorExample


@dataclass(frozen=True)
class CombatOutcomePredictorTrainConfig:
    epochs: int = 250
    learning_rate: float = 0.05
    l2: float = 0.0005
    validation_fraction: float = 0.2
    seed: int = 0


@dataclass(frozen=True)
class CombatOutcomePredictorTrainingReport:
    output_dir: Path
    model_path: Path
    metrics_path: Path
    summary_path: Path
    dataset_path: Path
    examples_path: Path | None
    train_examples_path: Path | None
    validation_examples_path: Path | None
    example_count: int
    train_example_count: int
    validation_example_count: int
    feature_count: int
    best_epoch: int
    split_strategy: str


@dataclass
class _PreparedExample:
    example: PredictorExample
    vector: list[float]


@dataclass(frozen=True)
class _ResolvedTrainingExamples:
    dataset_path: Path
    examples_path: Path | None
    train_examples_path: Path | None
    validation_examples_path: Path | None
    examples: list[PredictorExample]
    train_examples: list[PredictorExample]
    validation_examples: list[PredictorExample]
    split_strategy: str


def default_predictor_training_session_name() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def train_combat_outcome_predictor(
    *,
    dataset_source: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    config: CombatOutcomePredictorTrainConfig | None = None,
) -> CombatOutcomePredictorTrainingReport:
    train_config = config or CombatOutcomePredictorTrainConfig()
    dataset_path = Path(dataset_source).expanduser().resolve()
    resolved = _resolve_training_examples(
        dataset_path,
        validation_fraction=train_config.validation_fraction,
        seed=train_config.seed,
    )
    examples_path = resolved.examples_path
    examples = resolved.examples
    if not examples:
        if examples_path is not None:
            raise ValueError(f"Predictor dataset is empty: {examples_path}")
        raise ValueError(f"Predictor dataset is empty: {dataset_path}")

    output_dir = Path(output_root).expanduser().resolve() / (session_name or default_predictor_training_session_name())
    if output_dir.exists():
        raise FileExistsError(f"Predictor training output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    train_examples = resolved.train_examples
    validation_examples = resolved.validation_examples
    feature_names = sorted({name for example in train_examples for name in example.feature_map})
    if not feature_names:
        raise ValueError("Predictor training requires at least one extracted feature.")

    feature_means, feature_stds = _compute_feature_stats(train_examples, feature_names)
    prepared_train = _prepare_examples(train_examples, feature_names, feature_means, feature_stds)
    prepared_validation = _prepare_examples(validation_examples, feature_names, feature_means, feature_stds)

    reward_mean, reward_std = _weighted_target_stats(
        [example.reward_label for example in train_examples],
        [example.reward_weight for example in train_examples],
    )
    damage_mean, damage_std = _weighted_target_stats(
        [example.damage_delta_label for example in train_examples],
        [example.damage_weight for example in train_examples],
    )
    outcome_prior = _weighted_label_mean(
        [example.outcome_win_label for example in train_examples],
        [example.outcome_weight for example in train_examples],
    )

    outcome_head = PredictorHead(
        name="outcome_win",
        kind="logistic",
        weights=[0.0] * len(feature_names),
        bias=_logit(outcome_prior) if outcome_prior is not None else 0.0,
    )
    reward_head = PredictorHead(
        name="reward",
        kind="linear",
        weights=[0.0] * len(feature_names),
        bias=0.0,
        target_mean=reward_mean,
        target_std=reward_std,
    )
    damage_head = PredictorHead(
        name="damage_delta",
        kind="linear",
        weights=[0.0] * len(feature_names),
        bias=0.0,
        target_mean=damage_mean,
        target_std=damage_std,
    )

    metrics_path = output_dir / "training-metrics.jsonl"
    best_epoch = 0
    best_objective: float | None = None
    best_heads = _clone_heads((outcome_head, reward_head, damage_head))
    best_train_metrics: dict[str, Any] | None = None
    best_validation_metrics: dict[str, Any] | None = None

    with metrics_path.open("w", encoding="utf-8", newline="\n") as metrics_handle:
        for epoch in range(1, train_config.epochs + 1):
            _update_logistic_head(
                head=outcome_head,
                prepared_examples=prepared_train,
                label_getter=lambda item: item.example.outcome_win_label,
                weight_getter=lambda item: item.example.outcome_weight,
                learning_rate=train_config.learning_rate,
                l2=train_config.l2,
            )
            _update_linear_head(
                head=reward_head,
                prepared_examples=prepared_train,
                label_getter=lambda item: item.example.reward_label,
                weight_getter=lambda item: item.example.reward_weight,
                learning_rate=train_config.learning_rate,
                l2=train_config.l2,
                target_mean=reward_mean,
                target_std=reward_std,
            )
            _update_linear_head(
                head=damage_head,
                prepared_examples=prepared_train,
                label_getter=lambda item: item.example.damage_delta_label,
                weight_getter=lambda item: item.example.damage_weight,
                learning_rate=train_config.learning_rate,
                l2=train_config.l2,
                target_mean=damage_mean,
                target_std=damage_std,
            )

            train_metrics = _evaluate_heads(
                prepared_examples=prepared_train,
                outcome_head=outcome_head,
                reward_head=reward_head,
                damage_head=damage_head,
            )
            validation_metrics = _evaluate_heads(
                prepared_examples=prepared_validation,
                outcome_head=outcome_head,
                reward_head=reward_head,
                damage_head=damage_head,
            )

            objective = _select_objective(train_metrics, validation_metrics)
            is_best = best_objective is None or (objective is not None and objective < best_objective)
            if is_best:
                best_epoch = epoch
                best_objective = objective
                best_heads = _clone_heads((outcome_head, reward_head, damage_head))
                best_train_metrics = train_metrics
                best_validation_metrics = validation_metrics

            metrics_handle.write(
                json.dumps(
                    {
                        "schema_version": PREDICTOR_TRAINING_SCHEMA_VERSION,
                        "epoch": epoch,
                        "objective": objective,
                        "selected_as_best": is_best,
                        "train": train_metrics,
                        "validation": validation_metrics,
                    },
                    ensure_ascii=False,
                )
            )
            metrics_handle.write("\n")

    predictor = CombatOutcomePredictor(
        feature_names=feature_names,
        feature_means=feature_means,
        feature_stds=feature_stds,
        outcome_head=best_heads[0],
        reward_head=best_heads[1],
        damage_head=best_heads[2],
        metadata={
            "dataset_path": str(dataset_path),
            "examples_path": str(examples_path) if examples_path is not None else None,
            "train_examples_path": (
                str(resolved.train_examples_path) if resolved.train_examples_path is not None else None
            ),
            "validation_examples_path": (
                str(resolved.validation_examples_path) if resolved.validation_examples_path is not None else None
            ),
            "split_strategy": resolved.split_strategy,
            "training_config": {
                "epochs": train_config.epochs,
                "learning_rate": train_config.learning_rate,
                "l2": train_config.l2,
                "validation_fraction": train_config.validation_fraction,
                "seed": train_config.seed,
            },
            "best_epoch": best_epoch,
            "train_metrics": best_train_metrics,
            "validation_metrics": best_validation_metrics,
            "calibration": {
                "best_epoch": best_epoch,
                "train": best_train_metrics,
                "validation": best_validation_metrics,
            },
        },
    )

    model_path = predictor.save(output_dir / "combat-outcome-predictor.json")
    summary_payload = {
        "schema_version": PREDICTOR_TRAINING_SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "examples_path": str(examples_path) if examples_path is not None else None,
        "train_examples_path": str(resolved.train_examples_path) if resolved.train_examples_path is not None else None,
        "validation_examples_path": (
            str(resolved.validation_examples_path) if resolved.validation_examples_path is not None else None
        ),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "example_count": len(examples),
        "train_example_count": len(train_examples),
        "validation_example_count": len(validation_examples),
        "split_strategy": resolved.split_strategy,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "best_epoch": best_epoch,
        "best_objective": best_objective,
        "effective_label_counts": {
            "train_outcome": sum(1 for example in train_examples if example.outcome_weight > 0),
            "train_reward": sum(1 for example in train_examples if example.reward_weight > 0),
            "train_damage_delta": sum(1 for example in train_examples if example.damage_weight > 0),
            "validation_outcome": sum(1 for example in validation_examples if example.outcome_weight > 0),
            "validation_reward": sum(1 for example in validation_examples if example.reward_weight > 0),
            "validation_damage_delta": sum(1 for example in validation_examples if example.damage_weight > 0),
        },
        "target_stats": {
            "reward": {"mean": reward_mean, "std": reward_std},
            "damage_delta": {"mean": damage_mean, "std": damage_std},
            "outcome_prior": outcome_prior,
        },
        "config": {
            "epochs": train_config.epochs,
            "learning_rate": train_config.learning_rate,
            "l2": train_config.l2,
            "validation_fraction": train_config.validation_fraction,
            "seed": train_config.seed,
        },
        "train": best_train_metrics,
        "validation": best_validation_metrics,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return CombatOutcomePredictorTrainingReport(
        output_dir=output_dir,
        model_path=model_path,
        metrics_path=metrics_path,
        summary_path=summary_path,
        dataset_path=dataset_path,
        examples_path=examples_path,
        train_examples_path=resolved.train_examples_path,
        validation_examples_path=resolved.validation_examples_path,
        example_count=len(examples),
        train_example_count=len(train_examples),
        validation_example_count=len(validation_examples),
        feature_count=len(feature_names),
        best_epoch=best_epoch,
        split_strategy=resolved.split_strategy,
    )


def _resolve_training_examples(
    dataset_path: Path,
    *,
    validation_fraction: float,
    seed: int,
) -> _ResolvedTrainingExamples:
    if dataset_path.is_dir():
        try:
            dataset_summary = load_dataset_summary(dataset_path)
        except FileNotFoundError:
            dataset_summary = None
        if dataset_summary is not None and dataset_summary.get("dataset_kind") == "predictor_combat_outcomes":
            split_paths = resolve_dataset_split_paths(dataset_path)
            train_examples_path = split_paths.get("train")
            validation_examples_path = split_paths.get("validation")
            if train_examples_path is not None and validation_examples_path is not None:
                train_examples = load_predictor_examples(train_examples_path)
                validation_examples = load_predictor_examples(validation_examples_path)
                examples_path = None
                examples: list[PredictorExample]
                records_path_raw = dataset_summary.get("records_path")
                if records_path_raw:
                    records_path = Path(str(records_path_raw)).expanduser().resolve()
                    if records_path.exists():
                        examples_path = records_path
                        examples = load_predictor_examples(records_path)
                    else:
                        examples = _merge_split_examples(split_paths)
                else:
                    examples = _merge_split_examples(split_paths)
                return _ResolvedTrainingExamples(
                    dataset_path=dataset_path,
                    examples_path=examples_path,
                    train_examples_path=train_examples_path,
                    validation_examples_path=validation_examples_path,
                    examples=examples,
                    train_examples=train_examples,
                    validation_examples=validation_examples,
                    split_strategy="manifest_split",
                )

    examples_path = resolve_predictor_examples_path(dataset_path)
    examples = load_predictor_examples(dataset_path)
    train_examples, validation_examples = _split_examples(
        examples,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    return _ResolvedTrainingExamples(
        dataset_path=dataset_path,
        examples_path=examples_path,
        train_examples_path=None,
        validation_examples_path=None,
        examples=examples,
        train_examples=train_examples,
        validation_examples=validation_examples,
        split_strategy="random_fraction",
    )


def _merge_split_examples(split_paths: dict[str, Path]) -> list[PredictorExample]:
    merged: list[PredictorExample] = []
    for split_name in ("train", "validation", "test"):
        path = split_paths.get(split_name)
        if path is None or not path.exists():
            continue
        merged.extend(load_predictor_examples(path))
    return merged


def _split_examples(
    examples: Sequence[PredictorExample],
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[list[PredictorExample], list[PredictorExample]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)

    if len(indices) <= 1 or validation_fraction <= 0:
        return [examples[index] for index in indices], []

    validation_count = max(1, int(round(len(indices) * validation_fraction)))
    validation_count = min(validation_count, len(indices) - 1)
    validation_indices = set(indices[:validation_count])

    train_examples = [examples[index] for index in indices if index not in validation_indices]
    validation_examples = [examples[index] for index in indices if index in validation_indices]
    return train_examples, validation_examples


def _prepare_examples(
    examples: Sequence[PredictorExample],
    feature_names: Sequence[str],
    feature_means: Sequence[float],
    feature_stds: Sequence[float],
) -> list[_PreparedExample]:
    feature_index = {name: index for index, name in enumerate(feature_names)}
    prepared: list[_PreparedExample] = []
    for example in examples:
        vector = [0.0] * len(feature_names)
        for name, value in example.feature_map.items():
            index = feature_index.get(name)
            if index is None:
                continue
            vector[index] = float(value)
        for index, raw_value in enumerate(vector):
            vector[index] = (raw_value - feature_means[index]) / feature_stds[index]
        prepared.append(_PreparedExample(example=example, vector=vector))
    return prepared


def _compute_feature_stats(
    examples: Sequence[PredictorExample],
    feature_names: Sequence[str],
) -> tuple[list[float], list[float]]:
    means: list[float] = []
    stds: list[float] = []
    for feature_name in feature_names:
        values = [float(example.feature_map.get(feature_name, 0.0)) for example in examples]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        means.append(mean)
        stds.append(std if std > 1e-8 else 1.0)
    return means, stds


def _weighted_target_stats(values: Sequence[float], weights: Sequence[float]) -> tuple[float, float]:
    total_weight = sum(weight for weight in weights if weight > 0)
    if total_weight <= 0:
        return 0.0, 1.0
    mean = sum(value * weight for value, weight in zip(values, weights, strict=True) if weight > 0) / total_weight
    variance = (
        sum(weight * ((value - mean) ** 2) for value, weight in zip(values, weights, strict=True) if weight > 0)
        / total_weight
    )
    std = math.sqrt(variance)
    return mean, std if std > 1e-8 else 1.0


def _weighted_label_mean(values: Sequence[float | None], weights: Sequence[float]) -> float | None:
    total_weight = sum(weight for weight in weights if weight > 0)
    if total_weight <= 0:
        return None
    weighted_sum = 0.0
    for value, weight in zip(values, weights, strict=True):
        if weight <= 0 or value is None:
            continue
        weighted_sum += value * weight
    return weighted_sum / total_weight


def _update_logistic_head(
    *,
    head: PredictorHead,
    prepared_examples: Sequence[_PreparedExample],
    label_getter,
    weight_getter,
    learning_rate: float,
    l2: float,
) -> None:
    total_weight = 0.0
    grad_bias = 0.0
    grad_weights = [0.0] * len(head.weights)

    for item in prepared_examples:
        weight = float(weight_getter(item))
        label = label_getter(item)
        if weight <= 0 or label is None:
            continue
        total_weight += weight
        prediction = _sigmoid(head.bias + _dot(head.weights, item.vector))
        error = (prediction - float(label)) * weight
        grad_bias += error
        for index, value in enumerate(item.vector):
            grad_weights[index] += error * value

    if total_weight <= 0:
        return

    scale = 1.0 / total_weight
    head.bias -= learning_rate * (grad_bias * scale)
    for index, gradient in enumerate(grad_weights):
        head.weights[index] -= learning_rate * ((gradient * scale) + (l2 * head.weights[index]))


def _update_linear_head(
    *,
    head: PredictorHead,
    prepared_examples: Sequence[_PreparedExample],
    label_getter,
    weight_getter,
    learning_rate: float,
    l2: float,
    target_mean: float,
    target_std: float,
) -> None:
    total_weight = 0.0
    grad_bias = 0.0
    grad_weights = [0.0] * len(head.weights)

    for item in prepared_examples:
        weight = float(weight_getter(item))
        if weight <= 0:
            continue
        label = float(label_getter(item))
        normalized_target = (label - target_mean) / target_std
        prediction = head.bias + _dot(head.weights, item.vector)
        error = (prediction - normalized_target) * weight
        total_weight += weight
        grad_bias += error
        for index, value in enumerate(item.vector):
            grad_weights[index] += error * value

    if total_weight <= 0:
        return

    scale = 2.0 / total_weight
    head.bias -= learning_rate * (grad_bias * scale)
    for index, gradient in enumerate(grad_weights):
        head.weights[index] -= learning_rate * ((gradient * scale) + (l2 * head.weights[index]))


def _evaluate_heads(
    *,
    prepared_examples: Sequence[_PreparedExample],
    outcome_head: PredictorHead,
    reward_head: PredictorHead,
    damage_head: PredictorHead,
) -> dict[str, Any]:
    outcome_metrics = _evaluate_logistic_head(
        head=outcome_head,
        prepared_examples=prepared_examples,
        label_getter=lambda item: item.example.outcome_win_label,
        weight_getter=lambda item: item.example.outcome_weight,
    )
    reward_metrics = _evaluate_linear_head(
        head=reward_head,
        prepared_examples=prepared_examples,
        label_getter=lambda item: item.example.reward_label,
        weight_getter=lambda item: item.example.reward_weight,
        target_mean=reward_head.target_mean,
        target_std=reward_head.target_std,
    )
    damage_metrics = _evaluate_linear_head(
        head=damage_head,
        prepared_examples=prepared_examples,
        label_getter=lambda item: item.example.damage_delta_label,
        weight_getter=lambda item: item.example.damage_weight,
        target_mean=damage_head.target_mean,
        target_std=damage_head.target_std,
    )
    objective = _composite_objective(outcome_metrics, reward_metrics, damage_metrics)
    return {
        "example_count": len(prepared_examples),
        "objective": objective,
        "outcome": outcome_metrics,
        "reward": reward_metrics,
        "damage_delta": damage_metrics,
    }


def _evaluate_logistic_head(
    *,
    head: PredictorHead,
    prepared_examples: Sequence[_PreparedExample],
    label_getter,
    weight_getter,
) -> dict[str, Any]:
    total_weight = 0.0
    loss_sum = 0.0
    accuracy_sum = 0.0
    prediction_sum = 0.0
    label_sum = 0.0
    positive_count = 0
    count = 0

    for item in prepared_examples:
        weight = float(weight_getter(item))
        label = label_getter(item)
        if weight <= 0 or label is None:
            continue
        count += 1
        total_weight += weight
        prediction = _sigmoid(head.bias + _dot(head.weights, item.vector))
        label_value = float(label)
        clipped = min(max(prediction, 1e-6), 1.0 - 1e-6)
        loss_sum += weight * (-(label_value * math.log(clipped) + ((1.0 - label_value) * math.log(1.0 - clipped))))
        accuracy_sum += weight * (1.0 if (prediction >= 0.5) == (label_value >= 0.5) else 0.0)
        prediction_sum += prediction * weight
        label_sum += label_value * weight
        if label_value >= 0.5:
            positive_count += 1

    if total_weight <= 0:
        return {
            "count": 0,
            "total_weight": 0.0,
            "loss": None,
            "accuracy": None,
            "label_mean": None,
            "prediction_mean": None,
            "positive_count": 0,
        }

    return {
        "count": count,
        "total_weight": total_weight,
        "loss": loss_sum / total_weight,
        "accuracy": accuracy_sum / total_weight,
        "label_mean": label_sum / total_weight,
        "prediction_mean": prediction_sum / total_weight,
        "positive_count": positive_count,
    }


def _evaluate_linear_head(
    *,
    head: PredictorHead,
    prepared_examples: Sequence[_PreparedExample],
    label_getter,
    weight_getter,
    target_mean: float,
    target_std: float,
) -> dict[str, Any]:
    total_weight = 0.0
    count = 0
    normalized_mse_sum = 0.0
    mse_sum = 0.0
    mae_sum = 0.0
    prediction_sum = 0.0
    label_sum = 0.0

    for item in prepared_examples:
        weight = float(weight_getter(item))
        if weight <= 0:
            continue
        count += 1
        total_weight += weight
        label = float(label_getter(item))
        normalized_target = (label - target_mean) / target_std
        prediction_norm = head.bias + _dot(head.weights, item.vector)
        prediction = target_mean + (target_std * prediction_norm)
        normalized_error = prediction_norm - normalized_target
        raw_error = prediction - label

        normalized_mse_sum += weight * (normalized_error**2)
        mse_sum += weight * (raw_error**2)
        mae_sum += weight * abs(raw_error)
        prediction_sum += prediction * weight
        label_sum += label * weight

    if total_weight <= 0:
        return {
            "count": 0,
            "total_weight": 0.0,
            "normalized_mse": None,
            "mse": None,
            "rmse": None,
            "mae": None,
            "label_mean": None,
            "prediction_mean": None,
        }

    mse = mse_sum / total_weight
    return {
        "count": count,
        "total_weight": total_weight,
        "normalized_mse": normalized_mse_sum / total_weight,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae_sum / total_weight,
        "label_mean": label_sum / total_weight,
        "prediction_mean": prediction_sum / total_weight,
    }


def _composite_objective(
    outcome_metrics: dict[str, Any],
    reward_metrics: dict[str, Any],
    damage_metrics: dict[str, Any],
) -> float | None:
    values: list[float] = []
    if outcome_metrics.get("loss") is not None:
        values.append(float(outcome_metrics["loss"]))
    if reward_metrics.get("normalized_mse") is not None:
        values.append(float(reward_metrics["normalized_mse"]))
    if damage_metrics.get("normalized_mse") is not None:
        values.append(float(damage_metrics["normalized_mse"]))
    if not values:
        return None
    return sum(values)


def _select_objective(train_metrics: dict[str, Any], validation_metrics: dict[str, Any]) -> float | None:
    validation_objective = validation_metrics.get("objective")
    if validation_objective is not None:
        return float(validation_objective)
    train_objective = train_metrics.get("objective")
    if train_objective is not None:
        return float(train_objective)
    return None


def _clone_heads(heads: Sequence[PredictorHead]) -> tuple[PredictorHead, ...]:
    return tuple(
        PredictorHead(
            name=head.name,
            kind=head.kind,
            weights=list(head.weights),
            bias=head.bias,
            target_mean=head.target_mean,
            target_std=head.target_std,
        )
        for head in heads
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _logit(probability: float | None) -> float:
    if probability is None:
        return 0.0
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))
