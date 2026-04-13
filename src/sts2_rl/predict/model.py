from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_rl.data.trajectory import build_state_summary
from sts2_rl.env.types import StepObservation

from .features import extract_feature_map_from_summary
from .schema import PREDICTOR_MODEL_SCHEMA_VERSION


@dataclass
class PredictorHead:
    name: str
    kind: str
    weights: list[float]
    bias: float = 0.0
    target_mean: float = 0.0
    target_std: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "weights": self.weights,
            "bias": self.bias,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictorHead:
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            weights=[float(value) for value in payload.get("weights", [])],
            bias=float(payload.get("bias", 0.0)),
            target_mean=float(payload.get("target_mean", 0.0)),
            target_std=float(payload.get("target_std", 1.0)),
        )


@dataclass(frozen=True)
class PredictorScores:
    outcome_win_probability: float | None
    expected_reward: float | None
    expected_damage_delta: float | None
    feature_count: int
    feature_map: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome_win_probability": self.outcome_win_probability,
            "expected_reward": self.expected_reward,
            "expected_damage_delta": self.expected_damage_delta,
            "feature_count": self.feature_count,
            "feature_map": self.feature_map,
        }


class CombatOutcomePredictor:
    def __init__(
        self,
        *,
        feature_names: list[str],
        feature_means: list[float],
        feature_stds: list[float],
        outcome_head: PredictorHead,
        reward_head: PredictorHead,
        damage_head: PredictorHead,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not (len(feature_names) == len(feature_means) == len(feature_stds)):
            raise ValueError("Feature schema lengths must match.")
        head_size = len(feature_names)
        for head in (outcome_head, reward_head, damage_head):
            if len(head.weights) != head_size:
                raise ValueError(f"Head {head.name} weight count does not match feature count.")

        self.feature_names = feature_names
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.outcome_head = outcome_head
        self.reward_head = reward_head
        self.damage_head = damage_head
        self.metadata = metadata or {}
        self._feature_index = {name: index for index, name in enumerate(feature_names)}

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    def standardized_vector(self, feature_map: dict[str, float]) -> list[float]:
        vector = [0.0] * len(self.feature_names)
        for feature_name, feature_value in feature_map.items():
            index = self._feature_index.get(feature_name)
            if index is None:
                continue
            vector[index] = float(feature_value)
        for index, raw_value in enumerate(vector):
            std = self.feature_stds[index]
            mean = self.feature_means[index]
            vector[index] = 0.0 if std <= 0 else (raw_value - mean) / std
        return vector

    def score_feature_map(self, feature_map: dict[str, float]) -> PredictorScores:
        vector = self.standardized_vector(feature_map)
        outcome_logit = _forward(self.outcome_head, vector)
        reward_norm = _forward(self.reward_head, vector)
        damage_norm = _forward(self.damage_head, vector)

        outcome_probability = _sigmoid(outcome_logit)
        expected_reward = self.reward_head.target_mean + (self.reward_head.target_std * reward_norm)
        expected_damage_delta = self.damage_head.target_mean + (self.damage_head.target_std * damage_norm)

        return PredictorScores(
            outcome_win_probability=outcome_probability,
            expected_reward=expected_reward,
            expected_damage_delta=expected_damage_delta,
            feature_count=self.feature_count,
            feature_map=feature_map,
        )

    def score_summary(self, summary: dict[str, Any]) -> PredictorScores:
        return self.score_feature_map(extract_feature_map_from_summary(summary))

    def score_observation(self, observation: StepObservation) -> PredictorScores:
        return self.score_summary(build_state_summary(observation))

    def save(self, path: str | Path) -> Path:
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": PREDICTOR_MODEL_SCHEMA_VERSION,
            "feature_names": self.feature_names,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "outcome_head": self.outcome_head.as_dict(),
            "reward_head": self.reward_head.as_dict(),
            "damage_head": self.damage_head.as_dict(),
            "metadata": self.metadata,
        }
        model_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return model_path

    @classmethod
    def load(cls, path: str | Path) -> CombatOutcomePredictor:
        model_path = Path(path)
        payload = json.loads(model_path.read_text(encoding="utf-8"))
        schema_version = int(payload.get("schema_version", PREDICTOR_MODEL_SCHEMA_VERSION))
        if schema_version != PREDICTOR_MODEL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported predictor model schema_version={schema_version}; "
                f"expected {PREDICTOR_MODEL_SCHEMA_VERSION}."
            )

        return cls(
            feature_names=[str(name) for name in payload.get("feature_names", [])],
            feature_means=[float(value) for value in payload.get("feature_means", [])],
            feature_stds=[float(value) for value in payload.get("feature_stds", [])],
            outcome_head=PredictorHead.from_dict(payload["outcome_head"]),
            reward_head=PredictorHead.from_dict(payload["reward_head"]),
            damage_head=PredictorHead.from_dict(payload["damage_head"]),
            metadata=dict(payload.get("metadata", {})),
        )


def _forward(head: PredictorHead, vector: list[float]) -> float:
    return head.bias + sum(weight * value for weight, value in zip(head.weights, vector, strict=True))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)
