import random
from collections import Counter

from sts2_rl.train import DqnAgent, DqnConfig, ReplayBuffer, ReplayTransition


def test_dqn_update_changes_q_values() -> None:
    agent = DqnAgent(
        action_count=3,
        feature_count=4,
        config=DqnConfig(
            learning_rate=0.05,
            gamma=0.0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            min_replay_size=1,
            batch_size=1,
            replay_capacity=8,
            target_sync_interval=1,
            hidden_sizes=(8,),
            seed=1,
        ),
    )
    features = [1.0, 0.5, 0.0, 1.0]
    before = agent.online.forward(features)

    agent.add_transition(
        features=features,
        action_index=1,
        reward=2.0,
        next_features=None,
        next_mask=None,
        done=True,
    )
    stats = agent.maybe_update()
    after = agent.online.forward(features)

    assert stats.performed is True
    assert stats.loss is not None
    assert after[1] != before[1]


def test_dqn_save_load_roundtrip(tmp_path) -> None:
    agent = DqnAgent(
        action_count=2,
        feature_count=3,
        config=DqnConfig(
            hidden_sizes=(4,),
            seed=9,
            double_dqn=False,
            n_step=4,
            prioritized_replay=False,
            priority_alpha=0.2,
            priority_beta_start=0.1,
            priority_beta_end=0.8,
            priority_beta_decay_steps=123,
            priority_epsilon=0.0002,
        ),
    )
    checkpoint_path = tmp_path / "checkpoint.json"

    agent.save(checkpoint_path)
    restored = DqnAgent.load(checkpoint_path, expected_action_count=2, expected_feature_count=3)

    assert restored.config == agent.config
    assert restored.online.state_dict() == agent.online.state_dict()
    assert restored.target.state_dict() == agent.target.state_dict()


def test_prioritized_replay_prefers_high_priority_items() -> None:
    buffer = ReplayBuffer(
        8,
        random.Random(7),
        prioritized_replay=True,
        priority_alpha=1.0,
        priority_beta_start=0.4,
        priority_beta_end=1.0,
        priority_beta_decay_steps=100,
        priority_epsilon=0.0001,
    )
    buffer.add(
        ReplayTransition(
            features=[0.0],
            action_index=0,
            reward=0.0,
            next_features=None,
            next_mask=None,
            done=True,
        )
    )
    buffer.add(
        ReplayTransition(
            features=[1.0],
            action_index=0,
            reward=0.0,
            next_features=None,
            next_mask=None,
            done=True,
        )
    )
    buffer.update_priorities({0: 100.0, 1: 1.0})

    counts = Counter(buffer.sample(1, step=50)[0].index for _ in range(300))

    assert counts[0] > counts[1] * 10


def test_n_step_terminal_flush_builds_discounted_transitions() -> None:
    agent = DqnAgent(
        action_count=2,
        feature_count=1,
        config=DqnConfig(
            gamma=0.5,
            n_step=3,
            min_replay_size=10,
            replay_capacity=8,
            batch_size=1,
            hidden_sizes=(4,),
            seed=0,
        ),
    )

    agent.add_transition(
        features=[0.0],
        action_index=0,
        reward=1.0,
        next_features=[1.0],
        next_mask=[True, False],
        done=False,
    )
    assert agent.replay_transitions() == []

    agent.add_transition(
        features=[1.0],
        action_index=1,
        reward=2.0,
        next_features=None,
        next_mask=None,
        done=True,
    )

    transitions = agent.replay_transitions()
    assert len(transitions) == 2
    assert transitions[0].reward == 2.0
    assert transitions[0].transition_steps == 2
    assert transitions[0].bootstrap_discount == 0.25
    assert transitions[0].done is True
    assert transitions[1].reward == 2.0
    assert transitions[1].transition_steps == 1


def test_double_dqn_uses_online_argmax_for_bootstrap_target() -> None:
    agent = DqnAgent(
        action_count=2,
        feature_count=1,
        config=DqnConfig(
            gamma=0.5,
            double_dqn=True,
            prioritized_replay=False,
            n_step=1,
            min_replay_size=1,
            batch_size=1,
            replay_capacity=8,
            hidden_sizes=(4,),
            seed=0,
        ),
    )
    captured_targets: list[float] = []
    next_features = [9.0]

    agent.online.forward = lambda features: [1.0, 5.0] if features == next_features else [0.0, 0.0]  # type: ignore[method-assign]
    agent.target.forward = lambda features: [10.0, 3.0]  # type: ignore[method-assign]

    def capture_train_on_sample(**kwargs):
        captured_targets.append(kwargs["target_value"])
        return {
            "loss": 0.0,
            "unweighted_loss": 0.0,
            "td_error": kwargs["target_value"],
            "abs_td_error": abs(kwargs["target_value"]),
            "predicted_q": 0.0,
            "target_q": kwargs["target_value"],
        }

    agent.online.train_on_sample = capture_train_on_sample  # type: ignore[method-assign]
    agent.add_transition(
        features=[0.0],
        action_index=0,
        reward=1.0,
        next_features=next_features,
        next_mask=[True, True],
        done=False,
    )

    agent.maybe_update()

    assert captured_targets == [2.5]


def test_classic_dqn_uses_target_max_for_bootstrap_target() -> None:
    agent = DqnAgent(
        action_count=2,
        feature_count=1,
        config=DqnConfig(
            gamma=0.5,
            double_dqn=False,
            prioritized_replay=False,
            n_step=1,
            min_replay_size=1,
            batch_size=1,
            replay_capacity=8,
            hidden_sizes=(4,),
            seed=0,
        ),
    )
    captured_targets: list[float] = []
    next_features = [9.0]

    agent.online.forward = lambda features: [1.0, 5.0] if features == next_features else [0.0, 0.0]  # type: ignore[method-assign]
    agent.target.forward = lambda features: [10.0, 3.0]  # type: ignore[method-assign]

    def capture_train_on_sample(**kwargs):
        captured_targets.append(kwargs["target_value"])
        return {
            "loss": 0.0,
            "unweighted_loss": 0.0,
            "td_error": kwargs["target_value"],
            "abs_td_error": abs(kwargs["target_value"]),
            "predicted_q": 0.0,
            "target_q": kwargs["target_value"],
        }

    agent.online.train_on_sample = capture_train_on_sample  # type: ignore[method-assign]
    agent.add_transition(
        features=[0.0],
        action_index=0,
        reward=1.0,
        next_features=next_features,
        next_mask=[True, True],
        done=False,
    )

    agent.maybe_update()

    assert captured_targets == [6.0]
