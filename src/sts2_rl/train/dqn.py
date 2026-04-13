from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

DQN_CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class DqnConfig:
    learning_rate: float = 0.001
    gamma: float = 0.97
    epsilon_start: float = 0.20
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 2_000
    replay_capacity: int = 4_096
    batch_size: int = 32
    min_replay_size: int = 64
    target_sync_interval: int = 50
    updates_per_env_step: int = 1
    huber_delta: float = 1.0
    hidden_sizes: tuple[int, ...] = (64, 64)
    seed: int = 0
    double_dqn: bool = True
    n_step: int = 3
    prioritized_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    priority_beta_decay_steps: int = 10_000
    priority_epsilon: float = 0.0001

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")
        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon_start must be in [0, 1].")
        if not 0.0 <= self.epsilon_end <= 1.0:
            raise ValueError("epsilon_end must be in [0, 1].")
        if self.epsilon_decay_steps < 0:
            raise ValueError("epsilon_decay_steps must be non-negative.")
        if self.replay_capacity < 1:
            raise ValueError("replay_capacity must be positive.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.min_replay_size < 1:
            raise ValueError("min_replay_size must be positive.")
        if self.target_sync_interval < 1:
            raise ValueError("target_sync_interval must be positive.")
        if self.updates_per_env_step < 1:
            raise ValueError("updates_per_env_step must be positive.")
        if self.huber_delta <= 0:
            raise ValueError("huber_delta must be positive.")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must not be empty.")
        if any(size < 1 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must be positive.")
        if self.n_step < 1:
            raise ValueError("n_step must be positive.")
        if self.priority_alpha < 0.0:
            raise ValueError("priority_alpha must be non-negative.")
        if self.priority_beta_start < 0.0 or self.priority_beta_end < 0.0:
            raise ValueError("priority_beta values must be non-negative.")
        if self.priority_beta_end < self.priority_beta_start:
            raise ValueError("priority_beta_end must be >= priority_beta_start.")
        if self.priority_beta_decay_steps < 1:
            raise ValueError("priority_beta_decay_steps must be positive.")
        if self.priority_epsilon <= 0.0:
            raise ValueError("priority_epsilon must be positive.")


@dataclass(frozen=True)
class ReplayTransition:
    features: list[float]
    action_index: int
    reward: float
    next_features: list[float] | None
    next_mask: list[bool] | None
    done: bool
    bootstrap_discount: float = 0.0
    transition_steps: int = 1


@dataclass(frozen=True)
class ReplaySample:
    index: int
    transition: ReplayTransition
    probability: float
    weight: float
    priority: float


@dataclass(frozen=True)
class DqnSelection:
    action_index: int
    exploratory: bool
    epsilon: float
    q_values: list[float]


@dataclass(frozen=True)
class DqnReplayStats:
    size: int
    capacity: int
    utilization: float
    prioritized_replay: bool
    priority_beta: float
    priority_min: float | None
    priority_mean: float | None
    priority_max: float | None
    pending_n_step: int

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.utilization,
            "prioritized_replay": self.prioritized_replay,
            "priority_beta": self.priority_beta,
            "priority_min": self.priority_min,
            "priority_mean": self.priority_mean,
            "priority_max": self.priority_max,
            "pending_n_step": self.pending_n_step,
        }


@dataclass(frozen=True)
class DqnUpdateStats:
    performed: bool
    loss: float | None = None
    mean_abs_td_error: float | None = None
    max_abs_td_error: float | None = None
    mean_predicted_q: float | None = None
    mean_target_q: float | None = None
    mean_importance_weight: float | None = None
    mean_sample_priority: float | None = None
    mean_transition_steps: float | None = None
    replay_size: int = 0
    batch_size: int = 0
    sample_count: int = 0
    update_batches: int = 0
    update_step: int = 0
    target_synced: bool = False
    target_sync_count: int = 0
    priority_beta: float | None = None

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "performed": self.performed,
            "loss": self.loss,
            "mean_abs_td_error": self.mean_abs_td_error,
            "max_abs_td_error": self.max_abs_td_error,
            "mean_predicted_q": self.mean_predicted_q,
            "mean_target_q": self.mean_target_q,
            "mean_importance_weight": self.mean_importance_weight,
            "mean_sample_priority": self.mean_sample_priority,
            "mean_transition_steps": self.mean_transition_steps,
            "replay_size": self.replay_size,
            "batch_size": self.batch_size,
            "sample_count": self.sample_count,
            "update_batches": self.update_batches,
            "update_step": self.update_step,
            "target_synced": self.target_synced,
            "target_sync_count": self.target_sync_count,
            "priority_beta": self.priority_beta,
        }


@dataclass(frozen=True)
class _PendingTransition:
    features: list[float]
    action_index: int
    reward: float
    next_features: list[float] | None
    next_mask: list[bool] | None
    done: bool


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        rng: random.Random,
        *,
        prioritized_replay: bool,
        priority_alpha: float,
        priority_beta_start: float,
        priority_beta_end: float,
        priority_beta_decay_steps: int,
        priority_epsilon: float,
    ) -> None:
        self._capacity = capacity
        self._rng = rng
        self._prioritized_replay = prioritized_replay and priority_alpha > 0.0
        self._priority_alpha = priority_alpha
        self._priority_beta_start = priority_beta_start
        self._priority_beta_end = priority_beta_end
        self._priority_beta_decay_steps = priority_beta_decay_steps
        self._priority_epsilon = priority_epsilon
        self._items: list[ReplayTransition] = []
        self._priorities: list[float] = []
        self._cursor = 0
        self._max_priority = 1.0

    def add(self, transition: ReplayTransition) -> None:
        priority = self._max_priority if self._prioritized_replay else 1.0
        if len(self._items) < self._capacity:
            self._items.append(transition)
            self._priorities.append(priority)
            self._max_priority = max(self._max_priority, priority)
            return

        self._items[self._cursor] = transition
        self._priorities[self._cursor] = priority
        self._cursor = (self._cursor + 1) % self._capacity
        self._max_priority = max(self._priorities, default=1.0)

    def sample(self, size: int, *, step: int) -> list[ReplaySample]:
        if not self._items:
            raise ValueError("Cannot sample an empty replay buffer.")

        size = min(size, len(self._items))
        beta = self.current_beta(step)
        if not self._prioritized_replay:
            indices = self._rng.sample(range(len(self._items)), k=size)
            probability = 1.0 / float(len(self._items))
            return [
                ReplaySample(
                    index=index,
                    transition=self._items[index],
                    probability=probability,
                    weight=1.0,
                    priority=self._priorities[index],
                )
                for index in indices
            ]

        scaled_priorities = [priority**self._priority_alpha for priority in self._priorities]
        total_priority = sum(scaled_priorities)
        if total_priority <= 0.0:
            uniform_probability = 1.0 / float(len(self._items))
            indices = self._rng.sample(range(len(self._items)), k=size)
            return [
                ReplaySample(
                    index=index,
                    transition=self._items[index],
                    probability=uniform_probability,
                    weight=1.0,
                    priority=self._priorities[index],
                )
                for index in indices
            ]

        probabilities = [value / total_priority for value in scaled_priorities]
        indices = self._rng.choices(range(len(self._items)), weights=probabilities, k=size)
        raw_weights = [((len(self._items) * probabilities[index]) ** (-beta)) for index in indices]
        max_weight = max(raw_weights, default=1.0)
        return [
            ReplaySample(
                index=index,
                transition=self._items[index],
                probability=probabilities[index],
                weight=raw_weight / max_weight if max_weight > 0.0 else 1.0,
                priority=self._priorities[index],
            )
            for index, raw_weight in zip(indices, raw_weights, strict=True)
        ]

    def update_priorities(self, priority_updates: dict[int, float]) -> None:
        if not self._prioritized_replay:
            return
        for index, priority in priority_updates.items():
            if index < 0 or index >= len(self._priorities):
                continue
            self._priorities[index] = max(self._priority_epsilon, float(priority))
        self._max_priority = max(self._priorities, default=1.0)

    def current_beta(self, step: int) -> float:
        progress = min(1.0, max(0, step) / float(self._priority_beta_decay_steps))
        return self._priority_beta_start + (self._priority_beta_end - self._priority_beta_start) * progress

    def stats(self, *, step: int, pending_n_step: int) -> DqnReplayStats:
        if not self._priorities:
            priority_min = None
            priority_mean = None
            priority_max = None
        else:
            priority_min = min(self._priorities)
            priority_mean = sum(self._priorities) / len(self._priorities)
            priority_max = max(self._priorities)

        return DqnReplayStats(
            size=len(self._items),
            capacity=self._capacity,
            utilization=(len(self._items) / float(self._capacity)) if self._capacity > 0 else 0.0,
            prioritized_replay=self._prioritized_replay,
            priority_beta=self.current_beta(step),
            priority_min=priority_min,
            priority_mean=priority_mean,
            priority_max=priority_max,
            pending_n_step=pending_n_step,
        )

    def transitions(self) -> list[ReplayTransition]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


class DenseNetwork:
    def __init__(self, layer_sizes: list[int], *, seed: int) -> None:
        self.layer_sizes = list(layer_sizes)
        rng = random.Random(seed)
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []

        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:], strict=True):
            limit = (6.0 / (input_size + output_size)) ** 0.5
            self.weights.append(
                [
                    [rng.uniform(-limit, limit) for _ in range(input_size)]
                    for _ in range(output_size)
                ]
            )
            self.biases.append([0.0 for _ in range(output_size)])

    def clone(self) -> DenseNetwork:
        cloned = DenseNetwork(self.layer_sizes, seed=0)
        cloned.copy_from(self)
        return cloned

    def copy_from(self, other: DenseNetwork) -> None:
        self.layer_sizes = list(other.layer_sizes)
        self.weights = [
            [[float(value) for value in row] for row in layer]
            for layer in other.weights
        ]
        self.biases = [[float(value) for value in layer] for layer in other.biases]

    def forward(self, inputs: list[float]) -> list[float]:
        activations = list(inputs)
        for layer_index, (weights, biases) in enumerate(zip(self.weights, self.biases, strict=True)):
            outputs = []
            for row, bias in zip(weights, biases, strict=True):
                outputs.append(bias + sum(weight * value for weight, value in zip(row, activations, strict=True)))
            if layer_index < len(self.weights) - 1:
                activations = [max(0.0, value) for value in outputs]
            else:
                activations = outputs
        return activations

    def train_on_sample(
        self,
        *,
        features: list[float],
        action_index: int,
        target_value: float,
        learning_rate: float,
        huber_delta: float,
        sample_weight: float = 1.0,
    ) -> dict[str, float]:
        activations, pre_activations = self._forward_with_cache(features)
        output = activations[-1]
        predicted = output[action_index]
        td_error = target_value - predicted
        diff = predicted - target_value
        abs_diff = abs(diff)
        if abs_diff <= huber_delta:
            unweighted_loss = 0.5 * diff * diff
            grad_output_value = diff
        else:
            unweighted_loss = huber_delta * (abs_diff - (0.5 * huber_delta))
            grad_output_value = huber_delta if diff > 0 else -huber_delta

        weighted_loss = sample_weight * unweighted_loss
        grad_output_value *= sample_weight

        delta = [0.0 for _ in output]
        delta[action_index] = grad_output_value
        self.train_on_output_deltas(
            features=features,
            output_deltas=delta,
            learning_rate=learning_rate,
        )

        return {
            "loss": weighted_loss,
            "unweighted_loss": unweighted_loss,
            "td_error": td_error,
            "abs_td_error": abs(td_error),
            "predicted_q": predicted,
            "target_q": target_value,
        }

    def train_on_output_deltas(
        self,
        *,
        features: list[float],
        output_deltas: list[float],
        learning_rate: float,
        l2: float = 0.0,
    ) -> None:
        activations, pre_activations = self._forward_with_cache(features)
        delta = list(output_deltas)

        if len(delta) != len(activations[-1]):
            raise ValueError("Output delta length does not match network output size.")

        for layer_index in reversed(range(len(self.weights))):
            layer_weights = self.weights[layer_index]
            layer_biases = self.biases[layer_index]
            input_activation = activations[layer_index]
            previous_delta = [0.0 for _ in input_activation]

            for output_index, row in enumerate(layer_weights):
                node_delta = delta[output_index]
                for input_index, weight in enumerate(row):
                    previous_delta[input_index] += weight * node_delta

            for output_index, row in enumerate(layer_weights):
                node_delta = delta[output_index]
                for input_index, input_value in enumerate(input_activation):
                    grad = (node_delta * input_value) + (l2 * row[input_index] if l2 > 0.0 else 0.0)
                    row[input_index] -= learning_rate * grad
                layer_biases[output_index] -= learning_rate * node_delta

            if layer_index > 0:
                delta = [
                    previous_delta[input_index] if pre_activations[layer_index - 1][input_index] > 0 else 0.0
                    for input_index in range(len(previous_delta))
                ]

    def state_dict(self) -> dict[str, object]:
        return {
            "layer_sizes": self.layer_sizes,
            "weights": self.weights,
            "biases": self.biases,
        }

    def load_state_dict(self, payload: dict[str, object]) -> None:
        self.layer_sizes = [int(value) for value in payload["layer_sizes"]]
        self.weights = [
            [[float(value) for value in row] for row in layer]
            for layer in payload["weights"]
        ]
        self.biases = [
            [float(value) for value in layer]
            for layer in payload["biases"]
        ]

    def _forward_with_cache(self, inputs: list[float]) -> tuple[list[list[float]], list[list[float]]]:
        activations = [list(inputs)]
        pre_activations: list[list[float]] = []
        current = list(inputs)

        for layer_index, (weights, biases) in enumerate(zip(self.weights, self.biases, strict=True)):
            pre_activation = []
            for row, bias in zip(weights, biases, strict=True):
                pre_activation.append(bias + sum(weight * value for weight, value in zip(row, current, strict=True)))
            pre_activations.append(pre_activation)
            if layer_index < len(self.weights) - 1:
                current = [max(0.0, value) for value in pre_activation]
            else:
                current = pre_activation
            activations.append(list(current))

        return activations, pre_activations


class DqnAgent:
    def __init__(
        self,
        *,
        action_count: int,
        feature_count: int,
        config: DqnConfig | None = None,
    ) -> None:
        self.config = config or DqnConfig()
        self.action_count = action_count
        self.feature_count = feature_count
        self.online = DenseNetwork(
            [feature_count, *self.config.hidden_sizes, action_count],
            seed=self.config.seed,
        )
        self.target = self.online.clone()
        self._rng = random.Random(self.config.seed)
        self._replay = self._build_replay()
        self._n_step_pending: deque[_PendingTransition] = deque()
        self.env_steps = 0
        self.update_steps = 0

    def current_epsilon(self) -> float:
        if self.config.epsilon_decay_steps <= 0:
            return self.config.epsilon_end
        progress = min(1.0, self.env_steps / float(self.config.epsilon_decay_steps))
        return self.config.epsilon_start + (self.config.epsilon_end - self.config.epsilon_start) * progress

    def current_priority_beta(self) -> float:
        return self._replay.current_beta(self.update_steps)

    def replay_stats(self) -> DqnReplayStats:
        return self._replay.stats(step=self.update_steps, pending_n_step=len(self._n_step_pending))

    def replay_transitions(self) -> list[ReplayTransition]:
        return self._replay.transitions()

    def select_action(self, features: list[float], mask: list[bool]) -> DqnSelection:
        available = [index for index, enabled in enumerate(mask) if enabled]
        if not available:
            raise ValueError("Cannot select an action from an empty mask.")

        q_values = self.online.forward(features)
        epsilon = self.current_epsilon()
        exploratory = self._rng.random() < epsilon
        if exploratory:
            action_index = self._rng.choice(available)
        else:
            action_index = _argmax_masked(q_values, available)
        self.env_steps += 1
        return DqnSelection(
            action_index=action_index,
            exploratory=exploratory,
            epsilon=epsilon,
            q_values=q_values,
        )

    def select_greedy_action(self, features: list[float], mask: list[bool]) -> tuple[int, list[float]]:
        available = [index for index, enabled in enumerate(mask) if enabled]
        if not available:
            raise ValueError("Cannot select an action from an empty mask.")
        q_values = self.online.forward(features)
        return _argmax_masked(q_values, available), q_values

    def add_transition(
        self,
        *,
        features: list[float],
        action_index: int,
        reward: float,
        next_features: list[float] | None,
        next_mask: list[bool] | None,
        done: bool,
    ) -> None:
        self._n_step_pending.append(
            _PendingTransition(
                features=list(features),
                action_index=action_index,
                reward=reward,
                next_features=list(next_features) if next_features is not None else None,
                next_mask=list(next_mask) if next_mask is not None else None,
                done=done,
            )
        )
        self._flush_pending_transitions(force_all=done)

    def maybe_update(self) -> DqnUpdateStats:
        replay_size = len(self._replay)
        if replay_size < self.config.min_replay_size:
            return DqnUpdateStats(
                performed=False,
                replay_size=replay_size,
                update_step=self.update_steps,
                priority_beta=self.current_priority_beta(),
            )

        target_sync_count = 0
        losses: list[float] = []
        td_errors: list[float] = []
        predicted_q_values: list[float] = []
        target_q_values: list[float] = []
        importance_weights: list[float] = []
        sample_priorities: list[float] = []
        transition_steps: list[int] = []
        effective_batch_size = min(self.config.batch_size, replay_size)

        for _ in range(self.config.updates_per_env_step):
            batch = self._replay.sample(effective_batch_size, step=self.update_steps)
            priority_updates: dict[int, float] = {}
            for sample in batch:
                target_value = self._compute_target_value(sample.transition)
                metrics = self.online.train_on_sample(
                    features=sample.transition.features,
                    action_index=sample.transition.action_index,
                    target_value=target_value,
                    learning_rate=self.config.learning_rate,
                    huber_delta=self.config.huber_delta,
                    sample_weight=sample.weight,
                )
                losses.append(metrics["loss"])
                td_errors.append(metrics["abs_td_error"])
                predicted_q_values.append(metrics["predicted_q"])
                target_q_values.append(metrics["target_q"])
                importance_weights.append(sample.weight)
                sample_priorities.append(sample.priority)
                transition_steps.append(sample.transition.transition_steps)
                priority_updates[sample.index] = max(
                    priority_updates.get(sample.index, 0.0),
                    metrics["abs_td_error"] + self.config.priority_epsilon,
                )

            self._replay.update_priorities(priority_updates)
            self.update_steps += 1
            if self.config.target_sync_interval > 0 and self.update_steps % self.config.target_sync_interval == 0:
                self.target.copy_from(self.online)
                target_sync_count += 1

        sample_count = len(losses)
        return DqnUpdateStats(
            performed=True,
            loss=(sum(losses) / sample_count) if sample_count else 0.0,
            mean_abs_td_error=(sum(td_errors) / sample_count) if sample_count else 0.0,
            max_abs_td_error=max(td_errors) if td_errors else 0.0,
            mean_predicted_q=(sum(predicted_q_values) / sample_count) if sample_count else 0.0,
            mean_target_q=(sum(target_q_values) / sample_count) if sample_count else 0.0,
            mean_importance_weight=(sum(importance_weights) / sample_count) if sample_count else 0.0,
            mean_sample_priority=(sum(sample_priorities) / sample_count) if sample_count else 0.0,
            mean_transition_steps=(sum(transition_steps) / sample_count) if sample_count else 0.0,
            replay_size=replay_size,
            batch_size=effective_batch_size,
            sample_count=sample_count,
            update_batches=self.config.updates_per_env_step,
            update_step=self.update_steps,
            target_synced=target_sync_count > 0,
            target_sync_count=target_sync_count,
            priority_beta=self.current_priority_beta(),
        )

    def reconfigure(
        self,
        *,
        learning_rate: float | None = None,
        gamma: float | None = None,
        epsilon_start: float | None = None,
        epsilon_end: float | None = None,
        epsilon_decay_steps: int | None = None,
        replay_capacity: int | None = None,
        batch_size: int | None = None,
        min_replay_size: int | None = None,
        target_sync_interval: int | None = None,
        updates_per_env_step: int | None = None,
        huber_delta: float | None = None,
        seed: int | None = None,
        double_dqn: bool | None = None,
        n_step: int | None = None,
        prioritized_replay: bool | None = None,
        priority_alpha: float | None = None,
        priority_beta_start: float | None = None,
        priority_beta_end: float | None = None,
        priority_beta_decay_steps: int | None = None,
        priority_epsilon: float | None = None,
    ) -> None:
        self.config = replace(
            self.config,
            learning_rate=self.config.learning_rate if learning_rate is None else learning_rate,
            gamma=self.config.gamma if gamma is None else gamma,
            epsilon_start=self.config.epsilon_start if epsilon_start is None else epsilon_start,
            epsilon_end=self.config.epsilon_end if epsilon_end is None else epsilon_end,
            epsilon_decay_steps=self.config.epsilon_decay_steps if epsilon_decay_steps is None else epsilon_decay_steps,
            replay_capacity=self.config.replay_capacity if replay_capacity is None else replay_capacity,
            batch_size=self.config.batch_size if batch_size is None else batch_size,
            min_replay_size=self.config.min_replay_size if min_replay_size is None else min_replay_size,
            target_sync_interval=(
                self.config.target_sync_interval if target_sync_interval is None else target_sync_interval
            ),
            updates_per_env_step=(
                self.config.updates_per_env_step if updates_per_env_step is None else updates_per_env_step
            ),
            huber_delta=self.config.huber_delta if huber_delta is None else huber_delta,
            seed=self.config.seed if seed is None else seed,
            double_dqn=self.config.double_dqn if double_dqn is None else double_dqn,
            n_step=self.config.n_step if n_step is None else n_step,
            prioritized_replay=self.config.prioritized_replay if prioritized_replay is None else prioritized_replay,
            priority_alpha=self.config.priority_alpha if priority_alpha is None else priority_alpha,
            priority_beta_start=(
                self.config.priority_beta_start if priority_beta_start is None else priority_beta_start
            ),
            priority_beta_end=self.config.priority_beta_end if priority_beta_end is None else priority_beta_end,
            priority_beta_decay_steps=(
                self.config.priority_beta_decay_steps
                if priority_beta_decay_steps is None
                else priority_beta_decay_steps
            ),
            priority_epsilon=self.config.priority_epsilon if priority_epsilon is None else priority_epsilon,
        )
        self._rng = random.Random(self.config.seed)
        self._replay = self._build_replay()
        self._n_step_pending.clear()

    def save(self, path: str | Path, *, metadata: dict[str, object] | None = None) -> Path:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        replay_stats = self.replay_stats()
        checkpoint = {
            "schema_version": DQN_CHECKPOINT_SCHEMA_VERSION,
            "algorithm": "dqn",
            "action_count": self.action_count,
            "feature_count": self.feature_count,
            "config": {
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
                "epsilon_start": self.config.epsilon_start,
                "epsilon_end": self.config.epsilon_end,
                "epsilon_decay_steps": self.config.epsilon_decay_steps,
                "replay_capacity": self.config.replay_capacity,
                "batch_size": self.config.batch_size,
                "min_replay_size": self.config.min_replay_size,
                "target_sync_interval": self.config.target_sync_interval,
                "updates_per_env_step": self.config.updates_per_env_step,
                "huber_delta": self.config.huber_delta,
                "hidden_sizes": list(self.config.hidden_sizes),
                "seed": self.config.seed,
                "double_dqn": self.config.double_dqn,
                "n_step": self.config.n_step,
                "prioritized_replay": self.config.prioritized_replay,
                "priority_alpha": self.config.priority_alpha,
                "priority_beta_start": self.config.priority_beta_start,
                "priority_beta_end": self.config.priority_beta_end,
                "priority_beta_decay_steps": self.config.priority_beta_decay_steps,
                "priority_epsilon": self.config.priority_epsilon,
            },
            "env_steps": self.env_steps,
            "update_steps": self.update_steps,
            "training_state": {
                "env_steps": self.env_steps,
                "update_steps": self.update_steps,
                "current_epsilon": self.current_epsilon(),
                "current_priority_beta": replay_stats.priority_beta,
                "replay": replay_stats.as_dict(),
            },
            "metadata": metadata or {},
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
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
    ) -> DqnAgent:
        checkpoint_path = Path(path)
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        schema_version = payload.get("schema_version", 1)
        if schema_version not in {1, DQN_CHECKPOINT_SCHEMA_VERSION}:
            raise ValueError(
                f"Unsupported checkpoint schema_version={schema_version}; expected 1 or {DQN_CHECKPOINT_SCHEMA_VERSION}."
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
        defaults = DqnConfig()
        agent = cls(
            action_count=action_count,
            feature_count=feature_count,
            config=DqnConfig(
                learning_rate=float(config_payload.get("learning_rate", defaults.learning_rate)),
                gamma=float(config_payload.get("gamma", defaults.gamma)),
                epsilon_start=float(config_payload.get("epsilon_start", defaults.epsilon_start)),
                epsilon_end=float(config_payload.get("epsilon_end", defaults.epsilon_end)),
                epsilon_decay_steps=int(config_payload.get("epsilon_decay_steps", defaults.epsilon_decay_steps)),
                replay_capacity=int(config_payload.get("replay_capacity", defaults.replay_capacity)),
                batch_size=int(config_payload.get("batch_size", defaults.batch_size)),
                min_replay_size=int(config_payload.get("min_replay_size", defaults.min_replay_size)),
                target_sync_interval=int(
                    config_payload.get("target_sync_interval", defaults.target_sync_interval)
                ),
                updates_per_env_step=int(
                    config_payload.get("updates_per_env_step", defaults.updates_per_env_step)
                ),
                huber_delta=float(config_payload.get("huber_delta", defaults.huber_delta)),
                hidden_sizes=tuple(int(value) for value in config_payload.get("hidden_sizes", defaults.hidden_sizes)),
                seed=int(config_payload.get("seed", defaults.seed)),
                double_dqn=bool(config_payload.get("double_dqn", defaults.double_dqn)),
                n_step=int(config_payload.get("n_step", defaults.n_step)),
                prioritized_replay=bool(
                    config_payload.get("prioritized_replay", defaults.prioritized_replay)
                ),
                priority_alpha=float(config_payload.get("priority_alpha", defaults.priority_alpha)),
                priority_beta_start=float(
                    config_payload.get("priority_beta_start", defaults.priority_beta_start)
                ),
                priority_beta_end=float(config_payload.get("priority_beta_end", defaults.priority_beta_end)),
                priority_beta_decay_steps=int(
                    config_payload.get("priority_beta_decay_steps", defaults.priority_beta_decay_steps)
                ),
                priority_epsilon=float(config_payload.get("priority_epsilon", defaults.priority_epsilon)),
            ),
        )
        training_state = payload.get("training_state", {})
        agent.env_steps = int(payload.get("env_steps", training_state.get("env_steps", 0)))
        agent.update_steps = int(payload.get("update_steps", training_state.get("update_steps", 0)))
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        return agent

    def _build_replay(self) -> ReplayBuffer:
        return ReplayBuffer(
            self.config.replay_capacity,
            self._rng,
            prioritized_replay=self.config.prioritized_replay,
            priority_alpha=self.config.priority_alpha,
            priority_beta_start=self.config.priority_beta_start,
            priority_beta_end=self.config.priority_beta_end,
            priority_beta_decay_steps=self.config.priority_beta_decay_steps,
            priority_epsilon=self.config.priority_epsilon,
        )

    def _flush_pending_transitions(self, *, force_all: bool) -> None:
        while self._n_step_pending:
            if not force_all and len(self._n_step_pending) < self.config.n_step:
                break
            self._replay.add(self._build_n_step_transition())
            self._n_step_pending.popleft()

    def _build_n_step_transition(self) -> ReplayTransition:
        if not self._n_step_pending:
            raise ValueError("Cannot build an n-step transition from an empty queue.")

        first = self._n_step_pending[0]
        total_reward = 0.0
        next_features: list[float] | None = None
        next_mask: list[bool] | None = None
        done = False
        transition_steps = 0

        for transition_steps, pending in enumerate(self._n_step_pending, start=1):
            if transition_steps > self.config.n_step:
                break
            total_reward += (self.config.gamma ** (transition_steps - 1)) * pending.reward
            done = pending.done
            next_features = pending.next_features
            next_mask = pending.next_mask
            if pending.done or transition_steps >= self.config.n_step:
                break

        return ReplayTransition(
            features=list(first.features),
            action_index=first.action_index,
            reward=total_reward,
            next_features=list(next_features) if next_features is not None and not done else None,
            next_mask=list(next_mask) if next_mask is not None and not done else None,
            done=done,
            bootstrap_discount=self.config.gamma**transition_steps,
            transition_steps=transition_steps,
        )

    def _compute_target_value(self, transition: ReplayTransition) -> float:
        target_value = transition.reward
        if transition.done or transition.next_features is None or transition.next_mask is None:
            return target_value

        available = [index for index, enabled in enumerate(transition.next_mask) if enabled]
        if not available:
            return target_value

        if self.config.double_dqn:
            online_next_q_values = self.online.forward(transition.next_features)
            next_action_index = _argmax_masked(online_next_q_values, available)
            target_next_q_values = self.target.forward(transition.next_features)
            bootstrap_value = target_next_q_values[next_action_index]
        else:
            target_next_q_values = self.target.forward(transition.next_features)
            bootstrap_value = max(target_next_q_values[index] for index in available)
        return target_value + transition.bootstrap_discount * bootstrap_value


def _argmax_masked(values: list[float], available: list[int]) -> int:
    best_value = max(values[index] for index in available)
    best_indices = [index for index in available if values[index] == best_value]
    return min(best_indices)
