import json
from pathlib import Path

from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    CombatEnemyPayload,
    CombatHandCardPayload,
    CombatPayload,
    CombatPlayerPayload,
    CharacterSelectOptionPayload,
    CharacterSelectPayload,
    GameOverPayload,
    GameStatePayload,
    RunPayload,
    SelectionCardPayload,
    SelectionPayload,
)
from sts2_rl.env.types import StepObservation, StepResult
from sts2_rl.train import DqnConfig, run_combat_dqn_evaluation, run_combat_dqn_training, run_policy_pack_evaluation


class FakeEnv:
    def __init__(self, initial_observation: StepObservation, steps: dict[str, list[StepResult]]) -> None:
        self.current = initial_observation
        self.steps = {key: list(value) for key, value in steps.items()}
        self.closed = False

    def observe(self) -> StepObservation:
        return self.current

    def step(self, action):
        queue = self.steps[action.action_id]
        result = queue.pop(0)
        self.current = result.observation
        return result

    def close(self) -> None:
        self.closed = True


def test_run_combat_dqn_training_writes_checkpoints_summary_and_predictor_data(tmp_path: Path) -> None:
    observation_1 = _combat_observation(run_id="run-1", run_seed="TRAIN-SEED-001", enemy_hp=12, hand_card_index=0)
    observation_2 = _combat_observation(run_id="run-1", run_seed="TRAIN-SEED-001", enemy_hp=6, hand_card_index=0)
    map_observation = _map_observation(run_id="run-1", run_seed="TRAIN-SEED-001")

    step_1 = StepResult(observation=observation_2, terminated=False, info={}, response=None)
    step_2 = StepResult(observation=map_observation, terminated=False, info={}, response=None)

    fake_env = FakeEnv(
        observation_1,
        {
            "play_card|card=0|target=0": [step_1, step_2],
        },
    )

    report = run_combat_dqn_training(
        base_url="http://127.0.0.1:8080",
        output_root=tmp_path,
        session_name="unit-train",
        max_env_steps=2,
        dqn_config=DqnConfig(
            learning_rate=0.05,
            gamma=0.0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            min_replay_size=1,
            batch_size=1,
            replay_capacity=8,
            target_sync_interval=1,
            hidden_sizes=(8,),
            seed=0,
        ),
        checkpoint_every_rl_steps=1,
        env_factory=lambda _base_url, _timeout: fake_env,
    )

    session_dir = tmp_path / "unit-train"
    latest_checkpoint = session_dir / "combat-dqn-checkpoint.json"
    best_checkpoint = session_dir / "combat-dqn-best.json"
    periodic_checkpoint_1 = session_dir / "checkpoints" / "combat-dqn-step-000001.json"
    periodic_checkpoint_2 = session_dir / "checkpoints" / "combat-dqn-step-000002.json"

    assert report.rl_steps == 2
    assert report.heuristic_steps == 0
    assert report.update_steps >= 1
    assert report.periodic_checkpoint_count == 2
    assert report.best_checkpoint_path == best_checkpoint
    assert latest_checkpoint.exists()
    assert best_checkpoint.exists()
    assert periodic_checkpoint_1.exists()
    assert periodic_checkpoint_2.exists()
    assert report.summary_path.exists()
    assert report.combat_outcomes_path.exists()
    assert fake_env.closed is True

    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["rl_steps"] == 2
    assert summary["combat_count"] == 1
    assert summary["won_combats"] == 1
    assert summary["stop_reason"] == "max_env_steps_reached"
    assert summary["observed_run_seeds"] == ["TRAIN-SEED-001"]
    assert summary["observed_run_seed_histogram"] == {"TRAIN-SEED-001": 1}
    assert summary["runs_without_observed_seed"] == 0
    assert report.observed_run_seeds == ["TRAIN-SEED-001"]
    assert report.last_observed_seed == "TRAIN-SEED-001"
    assert summary["learning_metrics"]["update_call_count"] >= 1
    assert summary["learning_metrics"]["final_epsilon"] == 0.0
    assert summary["replay_metrics"]["size"] >= 1
    assert summary["checkpoint_comparison"]["latest"]["path"] == str(latest_checkpoint)
    assert summary["checkpoint_metadata"]["config"]["n_step"] == 3

    combat_outcomes = [json.loads(line) for line in report.combat_outcomes_path.read_text(encoding="utf-8").splitlines()]
    assert combat_outcomes[0]["outcome"] == "won"


def test_run_combat_dqn_training_recovers_from_game_over_and_stops_on_combat_budget(tmp_path: Path) -> None:
    initial = _game_over_observation(run_id="run-old")
    main_menu = _menu_observation()
    character_select = _character_select_observation(seed="TRAIN-SEED-RECOVER")
    combat = _combat_observation(run_id="run-new", run_seed="TRAIN-SEED-RECOVER", enemy_hp=6, hand_card_index=0)
    map_observation = _map_observation(run_id="run-new", run_seed="TRAIN-SEED-RECOVER")

    fake_env = FakeEnv(
        initial,
        {
            "return_to_main_menu": [StepResult(observation=main_menu, terminated=False, info={}, response=None)],
            "open_character_select": [StepResult(observation=character_select, terminated=False, info={}, response=None)],
            "embark": [StepResult(observation=combat, terminated=False, info={}, response=None)],
            "play_card|card=0|target=0": [StepResult(observation=map_observation, terminated=False, info={}, response=None)],
        },
    )

    report = run_combat_dqn_training(
        base_url="http://127.0.0.1:8080",
        output_root=tmp_path,
        session_name="recover-train",
        max_env_steps=0,
        max_runs=0,
        max_combats=1,
        dqn_config=DqnConfig(
            learning_rate=0.05,
            gamma=0.0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            min_replay_size=1,
            batch_size=1,
            replay_capacity=8,
            target_sync_interval=1,
            hidden_sizes=(8,),
            seed=0,
        ),
        checkpoint_every_rl_steps=1,
        env_factory=lambda _base_url, _timeout: fake_env,
    )

    assert report.rl_steps == 1
    assert report.heuristic_steps == 3
    assert report.completed_combat_count == 1
    assert report.completed_run_count == 1
    assert report.stop_reason == "max_combats_reached"
    assert report.final_screen == "MAP"

    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["completed_combat_count"] == 1
    assert summary["stop_reason"] == "max_combats_reached"
    assert summary["completed_run_count"] == 1
    assert summary["interrupted_runs"] == 1
    assert summary["observed_run_seed_histogram"] == {"TRAIN-SEED-RECOVER": 1}
    assert report.observed_run_seed_histogram == {"TRAIN-SEED-RECOVER": 1}


def test_run_combat_dqn_evaluation_stops_on_run_budget(tmp_path: Path) -> None:
    seed_report = run_combat_dqn_training(
        base_url="http://127.0.0.1:8080",
        output_root=tmp_path,
        session_name="seed-train",
        max_env_steps=1,
        max_runs=0,
        max_combats=0,
        dqn_config=DqnConfig(
            learning_rate=0.05,
            gamma=0.0,
            epsilon_start=0.0,
            epsilon_end=0.0,
            min_replay_size=1,
            batch_size=1,
            replay_capacity=8,
            target_sync_interval=1,
            hidden_sizes=(8,),
            seed=0,
        ),
        checkpoint_every_rl_steps=0,
        env_factory=lambda _base_url, _timeout: FakeEnv(
            _combat_observation(run_id="run-seed", run_seed="EVAL-SEED-TRAIN", enemy_hp=12, hand_card_index=0),
            {
                "play_card|card=0|target=0": [
                    StepResult(
                        observation=_combat_observation(
                            run_id="run-seed",
                            run_seed="EVAL-SEED-TRAIN",
                            enemy_hp=6,
                            hand_card_index=0,
                        ),
                        terminated=False,
                        info={},
                        response=None,
                    )
                ]
            },
        ),
    )
    checkpoint = seed_report.checkpoint_path

    game_over = _victory_game_over_observation(run_id="run-eval", run_seed="EVAL-SEED-001")
    fake_env = FakeEnv(
        _combat_observation(run_id="run-eval", run_seed="EVAL-SEED-001", enemy_hp=6, hand_card_index=0),
        {
            "play_card|card=0|target=0": [StepResult(observation=game_over, terminated=True, info={}, response=None)],
        },
    )

    report = run_combat_dqn_evaluation(
        base_url="http://127.0.0.1:8080",
        checkpoint_path=checkpoint,
        output_root=tmp_path,
        session_name="eval-run-budget",
        max_env_steps=0,
        max_runs=1,
        max_combats=0,
        env_factory=lambda _base_url, _timeout: fake_env,
    )

    assert report.combat_steps == 1
    assert report.completed_run_count == 1
    assert report.stop_reason == "max_runs_reached"

    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["completed_run_count"] == 1
    assert summary["won_runs"] == 1
    assert summary["stop_reason"] == "max_runs_reached"
    assert summary["observed_run_seed_histogram"] == {"EVAL-SEED-001": 1}
    assert summary["combat_performance"]["combat_win_rate"] == 1.0
    assert summary["checkpoint_metadata"]["algorithm"] == "dqn"
    assert report.observed_run_seeds == ["EVAL-SEED-001"]


def test_run_policy_pack_evaluation_prepares_to_main_menu_before_running(tmp_path: Path) -> None:
    normalize_env = FakeEnv(
        _game_over_observation(run_id="old-run"),
        {
            "return_to_main_menu": [StepResult(observation=_menu_observation(), terminated=False, info={}, response=None)],
        },
    )
    eval_env = FakeEnv(
        _menu_observation(),
        {
            "open_character_select": [StepResult(observation=_character_select_observation(), terminated=False, info={}, response=None)],
            "embark": [
                StepResult(
                    observation=_combat_observation(
                        run_id="run-eval",
                        run_seed="POLICY-SEED-001",
                        enemy_hp=6,
                        hand_card_index=0,
                    ),
                    terminated=False,
                    info={},
                    response=None,
                )
            ],
            "play_card|card=0|target=0": [
                StepResult(
                    observation=_victory_game_over_observation(run_id="run-eval", run_seed="POLICY-SEED-001"),
                    terminated=True,
                    info={},
                    response=None,
                )
            ],
        },
    )
    envs = [normalize_env, eval_env]

    report = run_policy_pack_evaluation(
        base_url="http://127.0.0.1:8080",
        output_root=tmp_path,
        session_name="policy-pack-prepare",
        policy_profile="baseline",
        max_env_steps=0,
        max_runs=1,
        max_combats=0,
        prepare_target="main_menu",
        prepare_max_steps=4,
        prepare_max_idle_polls=4,
        env_factory=lambda _base_url, _timeout: envs.pop(0),
    )

    assert report.completed_run_count == 1
    assert report.stop_reason == "max_runs_reached"
    assert report.final_screen == "GAME_OVER"
    assert normalize_env.closed is True
    assert eval_env.closed is True

    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["prepare_target"] == "main_menu"
    assert summary["normalization_report"]["reached_target"] is True
    assert summary["normalization_report"]["action_sequence"] == ["return_to_main_menu"]
    assert summary["observed_run_seed_histogram"] == {"POLICY-SEED-001": 1}
    assert report.observed_run_seeds == ["POLICY-SEED-001"]


def test_run_policy_pack_evaluation_records_non_combat_capability_timeout(tmp_path: Path) -> None:
    fake_env = FakeEnv(_selection_missing_semantic_observation(), {})

    report = run_policy_pack_evaluation(
        base_url="http://127.0.0.1:8080",
        output_root=tmp_path,
        session_name="policy-pack-capability-gap",
        policy_profile="baseline",
        max_env_steps=0,
        max_runs=0,
        max_combats=0,
        max_idle_polls=1,
        poll_interval_seconds=0.0,
        prepare_target="none",
        env_factory=lambda _base_url, _timeout: fake_env,
    )

    assert report.stop_reason == "policy_no_action_timeout:selection:missing_selection_semantic_mode"
    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    capability = summary["non_combat_capability"]
    assert capability["diagnostic_count"] == 1
    assert capability["no_action_timeout_count"] == 1
    assert capability["ambiguous_semantic_block_count"] == 1
    assert capability["bucket_histogram"]["runtime_contract_gap"] == 1
    assert capability["owner_histogram"]["STS2-Agent"] == 1
    assert capability["reason_histogram"]["missing_selection_semantic_mode"] == 1


def _combat_observation(*, run_id: str, run_seed: str | None = None, enemy_hp: int, hand_card_index: int) -> StepObservation:
    state = GameStatePayload(
        screen="COMBAT",
        run_id=run_id,
        turn=1,
        in_combat=True,
        combat=CombatPayload(
            player=CombatPlayerPayload(current_hp=40, max_hp=80, block=0, energy=3),
            hand=[
                CombatHandCardPayload(
                    index=hand_card_index,
                    card_id="strike",
                    name="Strike",
                    playable=True,
                    requires_target=True,
                    valid_target_indices=[0],
                    energy_cost=1,
                    rules_text="Deal 6 damage.",
                )
            ],
            enemies=[
                CombatEnemyPayload(
                    index=0,
                    enemy_id="slime",
                    name="Slime",
                    current_hp=enemy_hp,
                    max_hp=12,
                    block=0,
                    is_alive=enemy_hp > 0,
                    intent="ATTACK",
                )
            ],
        ),
        run=RunPayload(
            character_id="IRONCLAD",
            seed=run_seed,
            current_hp=40,
            max_hp=80,
            floor=2,
            gold=99,
            max_energy=3,
        ),
    )
    descriptors = AvailableActionsPayload(
        screen="COMBAT",
        actions=[ActionDescriptor(name="play_card", requires_index=True, requires_target=True)],
    )
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type="COMBAT",
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )


def _map_observation(*, run_id: str, run_seed: str | None = None) -> StepObservation:
    state = GameStatePayload(
        screen="MAP",
        run_id=run_id,
        run=RunPayload(
            character_id="IRONCLAD",
            seed=run_seed,
            current_hp=40,
            max_hp=80,
            floor=3,
            gold=99,
            max_energy=3,
        ),
    )
    descriptors = AvailableActionsPayload(screen="MAP", actions=[])
    return StepObservation(
        screen_type="MAP",
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=[],
        build_warnings=[],
    )


def _menu_observation() -> StepObservation:
    state = GameStatePayload(screen="MAIN_MENU", run_id="run_unknown")
    descriptors = AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")])
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type="MAIN_MENU",
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )


def _selection_missing_semantic_observation() -> StepObservation:
    state = GameStatePayload(
        screen="SELECTION",
        run_id="run-selection-gap",
        run=RunPayload(
            character_id="IRONCLAD",
            seed="SELECTION-GAP-001",
            current_hp=40,
            max_hp=80,
            floor=7,
            gold=99,
            max_energy=3,
        ),
        selection=SelectionPayload(
            kind="deck_card_select",
            selection_family="deck",
            prompt="Remove a card from your deck",
            required_count=1,
            cards=[
                SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                SelectionCardPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
            ],
        ),
    )
    descriptors = AvailableActionsPayload(
        screen="SELECTION",
        actions=[ActionDescriptor(name="select_deck_card", requires_index=True)],
    )
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type="SELECTION",
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )


def _character_select_observation(*, seed: str | None = None) -> StepObservation:
    state = GameStatePayload(
        screen="CHARACTER_SELECT",
        run_id="run_unknown",
        character_select=CharacterSelectPayload(
            selected_character_id="IRONCLAD",
            seed=seed,
            characters=[
                CharacterSelectOptionPayload(
                    index=0,
                    character_id="IRONCLAD",
                    name="Ironclad",
                    is_selected=True,
                )
            ],
        ),
    )
    descriptors = AvailableActionsPayload(screen="CHARACTER_SELECT", actions=[ActionDescriptor(name="embark")])
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type="CHARACTER_SELECT",
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )


def _game_over_observation(*, run_id: str) -> StepObservation:
    state = GameStatePayload(
        screen="GAME_OVER",
        run_id=run_id,
        game_over=GameOverPayload(
            is_victory=False,
            floor=1,
            character_id="IRONCLAD",
            can_return_to_main_menu=True,
            showing_summary=True,
        ),
    )
    descriptors = AvailableActionsPayload(screen="GAME_OVER", actions=[ActionDescriptor(name="return_to_main_menu")])
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type="GAME_OVER",
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )


def _victory_game_over_observation(*, run_id: str, run_seed: str | None = None) -> StepObservation:
    state = GameStatePayload(
        screen="GAME_OVER",
        run_id=run_id,
        run=RunPayload(
            character_id="IRONCLAD",
            seed=run_seed,
            current_hp=40,
            max_hp=80,
            floor=2,
            gold=99,
            max_energy=3,
        ),
        game_over=GameOverPayload(
            is_victory=True,
            floor=2,
            character_id="IRONCLAD",
            can_return_to_main_menu=True,
            showing_summary=True,
        ),
    )
    descriptors = AvailableActionsPayload(screen="GAME_OVER", actions=[ActionDescriptor(name="return_to_main_menu")])
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type="GAME_OVER",
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )
