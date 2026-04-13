from __future__ import annotations

import json
import random
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LinearQConfig:
    alpha: float = 0.05
    gamma: float = 0.95
    epsilon: float = 0.10
    seed: int = 0


class LinearQAgent:
    def __init__(
        self,
        *,
        action_count: int,
        feature_count: int,
        config: LinearQConfig | None = None,
    ) -> None:
        self.config = config or LinearQConfig()
        self.action_count = action_count
        self.feature_count = feature_count
        self.weights = [[0.0 for _ in range(feature_count)] for _ in range(action_count)]
        self._random = random.Random(self.config.seed)

    def q_values(self, features: list[float]) -> list[float]:
        return [_dot(row, features) for row in self.weights]

    def select_action(self, features: list[float], mask: list[bool]) -> tuple[int, bool, list[float]]:
        available = [index for index, enabled in enumerate(mask) if enabled]
        if not available:
            raise ValueError("Cannot select an action from an empty mask.")

        q_values = self.q_values(features)
        exploratory = self._random.random() < self.config.epsilon
        if exploratory:
            return self._random.choice(available), True, q_values

        best_score = max(q_values[index] for index in available)
        best_actions = [index for index in available if q_values[index] == best_score]
        return self._random.choice(best_actions), False, q_values

    def select_greedy_action(self, features: list[float], mask: list[bool]) -> tuple[int, list[float]]:
        available = [index for index, enabled in enumerate(mask) if enabled]
        if not available:
            raise ValueError("Cannot select an action from an empty mask.")

        q_values = self.q_values(features)
        best_score = max(q_values[index] for index in available)
        for index in available:
            if q_values[index] == best_score:
                return index, q_values
        raise RuntimeError("Failed to select a greedy action from a non-empty mask.")

    def update(
        self,
        *,
        features: list[float],
        action_index: int,
        reward: float,
        next_features: list[float] | None,
        next_mask: list[bool] | None,
        done: bool,
    ) -> float:
        current_q = _dot(self.weights[action_index], features)
        target = reward
        if not done and next_features is not None and next_mask is not None:
            future_values = self.q_values(next_features)
            available = [index for index, enabled in enumerate(next_mask) if enabled]
            if available:
                target += self.config.gamma * max(future_values[index] for index in available)

        td_error = target - current_q
        row = self.weights[action_index]
        for feature_index, value in enumerate(features):
            row[feature_index] += self.config.alpha * td_error * value
        return td_error

    def reconfigure(
        self,
        *,
        alpha: float | None = None,
        gamma: float | None = None,
        epsilon: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = replace(
            self.config,
            alpha=self.config.alpha if alpha is None else alpha,
            gamma=self.config.gamma if gamma is None else gamma,
            epsilon=self.config.epsilon if epsilon is None else epsilon,
            seed=self.config.seed if seed is None else seed,
        )
        self._random = random.Random(self.config.seed)

    def save(self, path: str | Path) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "action_count": self.action_count,
            "feature_count": self.feature_count,
            "config": {
                "alpha": self.config.alpha,
                "gamma": self.config.gamma,
                "epsilon": self.config.epsilon,
                "seed": self.config.seed,
            },
            "weights": self.weights,
        }
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
        return checkpoint_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_action_count: int | None = None,
        expected_feature_count: int | None = None,
    ) -> LinearQAgent:
        checkpoint_path = Path(path)
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        schema_version = payload.get("schema_version", CHECKPOINT_SCHEMA_VERSION)
        if schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported checkpoint schema_version={schema_version}; expected {CHECKPOINT_SCHEMA_VERSION}."
            )

        action_count = int(payload["action_count"])
        feature_count = int(payload["feature_count"])

        if expected_action_count is not None and action_count != expected_action_count:
            raise ValueError(
                f"Checkpoint action_count={action_count} does not match expected {expected_action_count}."
            )
        if expected_feature_count is not None and feature_count != expected_feature_count:
            raise ValueError(
                f"Checkpoint feature_count={feature_count} does not match expected {expected_feature_count}."
            )

        config_payload = payload.get("config", {})
        agent = cls(
            action_count=action_count,
            feature_count=feature_count,
            config=LinearQConfig(
                alpha=float(config_payload.get("alpha", 0.05)),
                gamma=float(config_payload.get("gamma", 0.95)),
                epsilon=float(config_payload.get("epsilon", 0.10)),
                seed=int(config_payload.get("seed", 0)),
            ),
        )
        agent.weights = [[float(value) for value in row] for row in payload["weights"]]
        if len(agent.weights) != action_count:
            raise ValueError(
                f"Checkpoint weights row count={len(agent.weights)} does not match action_count={action_count}."
            )
        for row in agent.weights:
            if len(row) != feature_count:
                raise ValueError(
                    f"Checkpoint weight row length={len(row)} does not match feature_count={feature_count}."
                )
        return agent


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
