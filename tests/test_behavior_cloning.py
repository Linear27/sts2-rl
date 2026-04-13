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
    GameOverPayload,
    GameStatePayload,
    MapNodePayload,
    MapPayload,
    RunPayload,
)
from sts2_rl.env.types import StepObservation, StepResult
from sts2_rl.train import (
    BehaviorCloningModel,
    BehaviorCloningTrainConfig,
    run_behavior_cloning_evaluation,
    run_benchmark_suite,
    train_behavior_cloning_policy,
)


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


def test_train_behavior_cloning_policy_writes_artifacts_and_scores_actions(tmp_path: Path) -> None:
    dataset_dir = _write_bc_dataset(tmp_path)
    report = train_behavior_cloning_policy(
        dataset_source=dataset_dir,
        output_root=tmp_path / "artifacts" / "bc",
        session_name="bc-unit",
        config=BehaviorCloningTrainConfig(
            epochs=45,
            learning_rate=0.08,
            l2=0.0001,
            validation_fraction=0.1,
            test_fraction=0.1,
            seed=7,
            run_outcome_weights={"won": 1.15, "lost": 0.9},
        ),
    )

    assert report.checkpoint_path.exists()
    assert report.best_checkpoint_path.exists()
    assert report.metrics_path.exists()
    assert report.summary_path.exists()
    assert report.train_example_count == 4
    assert report.validation_example_count == 2
    assert report.test_example_count == 2
    assert report.feature_count > 10
    assert report.stage_count == 2
    assert report.best_epoch >= 1
    assert report.split_strategy == "manifest_split"

    model = BehaviorCloningModel.load(report.best_checkpoint_path)
    low_hp_map = _map_observation(run_id="RUN-EVAL-MAP", floor=9, hp=18, node_types=("REST", "ELITE"))
    high_hp_map = _map_observation(run_id="RUN-EVAL-MAP-2", floor=9, hp=72, node_types=("REST", "ELITE"))
    lethal_combat = _combat_observation(run_id="RUN-EVAL-COMBAT", floor=10, enemy_hp=5)
    stall_combat = _combat_observation(run_id="RUN-EVAL-COMBAT-2", floor=10, enemy_hp=18)

    assert model.choose(low_hp_map).action.action_id == "choose_map_node|option=0"
    assert model.choose(high_hp_map).action.action_id == "choose_map_node|option=1"
    assert model.choose(lethal_combat).action.action_id == "play_card|card=0|target=0"
    assert model.choose(stall_combat).action.action_id == "end_turn"


def test_run_behavior_cloning_evaluation_and_benchmark_suite_support_bc_checkpoints(tmp_path: Path) -> None:
    dataset_dir = _write_bc_dataset(tmp_path)
    report = train_behavior_cloning_policy(
        dataset_source=dataset_dir,
        output_root=tmp_path / "artifacts" / "bc",
        session_name="bc-eval",
        config=BehaviorCloningTrainConfig(
            epochs=35,
            learning_rate=0.08,
            seed=5,
        ),
    )

    map_observation = _map_observation(run_id="RUN-LIVE", floor=9, hp=18, node_types=("REST", "ELITE"))
    combat_observation = _combat_observation(run_id="RUN-LIVE", floor=10, enemy_hp=5)
    game_over = _game_over_observation(run_id="RUN-LIVE")

    def env_factory(_base_url: str, _timeout: float) -> FakeEnv:
        return FakeEnv(
            map_observation,
            {
                "choose_map_node|option=0": [StepResult(observation=combat_observation, terminated=False, info={}, response=None)],
                "play_card|card=0|target=0": [StepResult(observation=game_over, terminated=True, info={}, response=None)],
            },
        )

    eval_report = run_behavior_cloning_evaluation(
        base_url="http://127.0.0.1:8080",
        checkpoint_path=report.best_checkpoint_path,
        output_root=tmp_path / "artifacts" / "eval",
        session_name="live-bc",
        max_env_steps=0,
        max_runs=1,
        max_combats=0,
        env_factory=env_factory,
    )
    assert eval_report.summary_path.exists()
    eval_summary = json.loads(eval_report.summary_path.read_text(encoding="utf-8"))
    assert eval_summary["stop_reason"] == "max_runs_reached"
    assert eval_summary["combat_performance"]["combat_win_rate"] == 1.0
    assert eval_summary["checkpoint_metadata"]["algorithm"] == "behavior_cloning"

    manifest_path = tmp_path / "suite.toml"
    manifest_path.write_text(
        f"""
schema_version = 1
suite_name = "bc-suite"
base_url = "http://127.0.0.1:8080"

[stats]
bootstrap_resamples = 200
confidence_level = 0.95
seed = 13

[[cases]]
case_id = "bc-eval"
mode = "eval"
checkpoint_path = "{report.best_checkpoint_path.as_posix()}"
repeats = 2
prepare_target = "none"
""".strip(),
        encoding="utf-8",
    )

    benchmark_report = run_benchmark_suite(
        manifest_path,
        output_root=tmp_path / "artifacts" / "benchmarks",
        env_factory=env_factory,
    )
    summary = json.loads(benchmark_report.summary_path.read_text(encoding="utf-8"))
    assert summary["case_count"] == 1
    assert summary["cases"][0]["checkpoint_paths"]["checkpoint_path"] == str(report.best_checkpoint_path)
    assert summary["cases"][0]["metrics"]["combat_win_rate"]["mean"] == 1.0


def _write_bc_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "data" / "bc"
    dataset_dir.mkdir(parents=True)

    train_records = [
        _trajectory_step_payload(_map_observation(run_id="RUN-MAP-LOW-1", floor=7, hp=18, node_types=("REST", "ELITE")), "choose_map_node|option=0", 1, "won"),
        _trajectory_step_payload(_map_observation(run_id="RUN-MAP-HIGH-1", floor=8, hp=72, node_types=("REST", "ELITE")), "choose_map_node|option=1", 2, "won"),
        _trajectory_step_payload(_combat_observation(run_id="RUN-COMBAT-LETHAL-1", floor=10, enemy_hp=5), "play_card|card=0|target=0", 3, "won"),
        _trajectory_step_payload(_combat_observation(run_id="RUN-COMBAT-STALL-1", floor=10, enemy_hp=18), "end_turn", 4, "lost"),
    ]
    validation_records = [
        _trajectory_step_payload(_map_observation(run_id="RUN-MAP-LOW-2", floor=9, hp=20, node_types=("REST", "ELITE")), "choose_map_node|option=0", 5, "won"),
        _trajectory_step_payload(_combat_observation(run_id="RUN-COMBAT-LETHAL-2", floor=11, enemy_hp=4), "play_card|card=0|target=0", 6, "won"),
    ]
    test_records = [
        _trajectory_step_payload(_map_observation(run_id="RUN-MAP-HIGH-2", floor=9, hp=70, node_types=("REST", "ELITE")), "choose_map_node|option=1", 7, "won"),
        _trajectory_step_payload(_combat_observation(run_id="RUN-COMBAT-STALL-2", floor=11, enemy_hp=16), "end_turn", 8, "lost"),
    ]

    _write_jsonl(dataset_dir / "train.steps.jsonl", train_records)
    _write_jsonl(dataset_dir / "validation.steps.jsonl", validation_records)
    _write_jsonl(dataset_dir / "test.steps.jsonl", test_records)
    _write_jsonl(dataset_dir / "steps.jsonl", [*train_records, *validation_records, *test_records])

    summary_payload = {
        "schema_version": 1,
        "dataset_name": "bc-synthetic",
        "dataset_kind": "trajectory_steps",
        "records_path": str((dataset_dir / "steps.jsonl").resolve()),
        "split": {
            "split_paths": {
                "train": str((dataset_dir / "train.steps.jsonl").resolve()),
                "validation": str((dataset_dir / "validation.steps.jsonl").resolve()),
                "test": str((dataset_dir / "test.steps.jsonl").resolve()),
            }
        },
        "lineage": {"source_paths": ["synthetic"]},
    }
    (dataset_dir / "dataset-summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dataset_dir


def _trajectory_step_payload(
    observation: StepObservation,
    chosen_action_id: str,
    step_index: int,
    run_outcome: str,
) -> dict:
    chosen_action = next(candidate for candidate in observation.legal_actions if candidate.action_id == chosen_action_id)
    return {
        "schema_version": 2,
        "record_type": "step",
        "timestamp_utc": "2026-04-12T00:00:00+00:00",
        "session_name": "bc-synthetic",
        "session_kind": "train",
        "instance_id": "inst-01",
        "step_index": step_index,
        "run_id": observation.run_id,
        "screen_type": observation.screen_type,
        "floor": observation.state.run.floor if observation.state.run is not None else None,
        "legal_action_count": len(observation.legal_actions),
        "legal_action_ids": [candidate.action_id for candidate in observation.legal_actions],
        "build_warnings": list(observation.build_warnings),
        "chosen_action_id": chosen_action.action_id,
        "chosen_action_label": chosen_action.label,
        "chosen_action_source": chosen_action.source,
        "chosen_action": chosen_action.request.model_dump(mode="json"),
        "policy_name": "policy-pack:planner",
        "policy_pack": "planner",
        "policy_handler": "synthetic",
        "algorithm": "heuristic",
        "decision_source": "heuristic",
        "decision_stage": observation.screen_type.lower(),
        "decision_reason": "synthetic",
        "decision_score": 1.0,
        "reward": 0.0,
        "reward_source": "synthetic",
        "terminated": False,
        "truncated": False,
        "info": {"run_outcome": run_outcome},
        "model_metrics": {},
        "state_summary": {
            **(json.loads(json.dumps(observation.state.model_dump(mode="json"))) if False else {}),
            **(
                {
                    "screen_type": observation.screen_type,
                    "run_id": observation.run_id,
                    "available_action_count": len(observation.legal_actions),
                    "run": _run_summary(observation),
                    "combat": _combat_summary(observation),
                }
            ),
        },
        "action_descriptors": observation.action_descriptors.model_dump(mode="json"),
        "state": observation.state.model_dump(mode="json"),
        "response": None,
    }


def _run_summary(observation: StepObservation) -> dict:
    run = observation.state.run
    if run is None:
        return {}
    return {
        "character_id": run.character_id,
        "floor": run.floor,
        "current_hp": run.current_hp,
        "max_hp": run.max_hp,
        "gold": run.gold,
        "max_energy": run.max_energy,
        "deck_size": 12,
    }


def _combat_summary(observation: StepObservation) -> dict:
    combat = observation.state.combat
    if combat is None:
        return {}
    return {
        "player_hp": combat.player.current_hp,
        "player_block": combat.player.block,
        "energy": combat.player.energy,
        "enemy_hp": [enemy.current_hp for enemy in combat.enemies if enemy.is_alive],
        "playable_hand_count": len([card for card in combat.hand if card.playable]),
    }


def _map_observation(*, run_id: str, floor: int, hp: int, node_types: tuple[str, str]) -> StepObservation:
    state = GameStatePayload(
        screen="MAP",
        run_id=run_id,
        run=RunPayload(character_id="IRONCLAD", current_hp=hp, max_hp=80, floor=floor, gold=120, max_energy=3),
        map=MapPayload(
            available_nodes=[
                MapNodePayload(index=0, row=floor, col=1, node_type=node_types[0]),
                MapNodePayload(index=1, row=floor, col=2, node_type=node_types[1]),
            ],
            is_travel_enabled=True,
            is_traveling=False,
        ),
    )
    descriptors = AvailableActionsPayload(
        screen="MAP",
        actions=[ActionDescriptor(name="choose_map_node", requires_index=True)],
    )
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type="MAP",
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )


def _combat_observation(*, run_id: str, floor: int, enemy_hp: int) -> StepObservation:
    state = GameStatePayload(
        screen="COMBAT",
        run_id=run_id,
        in_combat=True,
        turn=1,
        run=RunPayload(character_id="IRONCLAD", current_hp=48, max_hp=80, floor=floor, gold=140, max_energy=3),
        combat=CombatPayload(
            player=CombatPlayerPayload(current_hp=48, max_hp=80, block=0, energy=1),
            hand=[
                CombatHandCardPayload(
                    index=0,
                    card_id="STRIKE_IRONCLAD",
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
                    enemy_id="SLIME_SMALL",
                    name="Slime",
                    current_hp=enemy_hp,
                    max_hp=20,
                    block=0,
                    is_alive=True,
                )
            ],
        ),
    )
    descriptors = AvailableActionsPayload(
        screen="COMBAT",
        actions=[
            ActionDescriptor(name="play_card", requires_index=True, requires_target=True),
            ActionDescriptor(name="end_turn"),
        ],
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


def _game_over_observation(*, run_id: str) -> StepObservation:
    state = GameStatePayload(
        screen="GAME_OVER",
        run_id=run_id,
        run=RunPayload(character_id="IRONCLAD", current_hp=30, max_hp=80, floor=11, gold=150, max_energy=3),
        game_over=GameOverPayload(
            is_victory=True,
            floor=11,
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


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
