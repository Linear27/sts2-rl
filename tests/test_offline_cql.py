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
    RunPayload,
)
from sts2_rl.env.types import StepObservation, StepResult
from sts2_rl.train import (
    OfflineCqlTrainConfig,
    run_benchmark_suite,
    run_offline_cql_evaluation,
    train_offline_cql_policy,
)
from sts2_rl.train.combat_encoder import CombatStateEncoder
from sts2_rl.train.combat_space import CombatActionSpace


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


def test_train_offline_cql_policy_writes_artifacts_and_warmstart_export(tmp_path: Path) -> None:
    dataset_dir = _write_offline_dataset(tmp_path)
    report = train_offline_cql_policy(
        dataset_source=dataset_dir,
        output_root=tmp_path / "artifacts" / "offline",
        session_name="offline-unit",
        config=OfflineCqlTrainConfig(
            epochs=12,
            batch_size=2,
            learning_rate=0.01,
            hidden_sizes=(8,),
            target_sync_interval=2,
            seed=3,
        ),
    )

    assert report.checkpoint_path.exists()
    assert report.best_checkpoint_path.exists()
    assert report.warmstart_checkpoint_path.exists()
    assert report.summary_path.exists()
    assert report.train_example_count == 4
    assert report.validation_example_count == 2
    assert report.test_example_count == 2
    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["algorithm"] == "offline_cql"
    assert summary["best_epoch"] >= 1
    best_payload = json.loads(report.best_checkpoint_path.read_text(encoding="utf-8"))
    assert best_payload["algorithm"] == "offline_cql"
    assert "dqn" in best_payload["metadata"]["warm_start_compatible_algorithms"]


def test_run_offline_cql_evaluation_and_benchmark_suite_support_offline_checkpoints(tmp_path: Path) -> None:
    dataset_dir = _write_offline_dataset(tmp_path)
    report = train_offline_cql_policy(
        dataset_source=dataset_dir,
        output_root=tmp_path / "artifacts" / "offline",
        session_name="offline-eval",
        config=OfflineCqlTrainConfig(
            epochs=8,
            batch_size=2,
            learning_rate=0.01,
            hidden_sizes=(8,),
            target_sync_interval=2,
            seed=5,
        ),
    )

    combat_observation = _combat_observation(run_id="RUN-LIVE", floor=9, enemy_hp=6)
    game_over = _game_over_observation(run_id="RUN-LIVE")

    def env_factory(_base_url: str, _timeout: float) -> FakeEnv:
        return FakeEnv(
            combat_observation,
            {
                "play_card|card=0|target=0": [StepResult(observation=game_over, terminated=True, info={}, response=None)],
                "end_turn": [StepResult(observation=game_over, terminated=True, info={}, response=None)],
            },
        )

    eval_report = run_offline_cql_evaluation(
        base_url="http://127.0.0.1:8080",
        checkpoint_path=report.best_checkpoint_path,
        output_root=tmp_path / "artifacts" / "eval",
        session_name="offline-live",
        max_env_steps=0,
        max_runs=1,
        max_combats=0,
        env_factory=env_factory,
    )
    eval_summary = json.loads(eval_report.summary_path.read_text(encoding="utf-8"))
    assert eval_summary["checkpoint_metadata"]["algorithm"] == "offline_cql"

    manifest_path = tmp_path / "suite.toml"
    manifest_path.write_text(
        f"""
schema_version = 1
suite_name = "offline-suite"
base_url = "http://127.0.0.1:8080"

[stats]
bootstrap_resamples = 200
confidence_level = 0.95
seed = 13

[[cases]]
case_id = "offline-eval"
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


def _write_offline_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "data" / "offline"
    dataset_dir.mkdir(parents=True)
    train_records = [
        _offline_transition_payload("RUN-TRAIN-1", 1, floor=6, enemy_hp=12, chosen_action_id="play_card|card=0|target=0", reward=0.8, next_enemy_hp=6),
        _offline_transition_payload("RUN-TRAIN-2", 2, floor=6, enemy_hp=18, chosen_action_id="end_turn", reward=0.2, next_enemy_hp=None),
        _offline_transition_payload("RUN-TRAIN-3", 3, floor=7, enemy_hp=10, chosen_action_id="play_card|card=0|target=0", reward=0.7, next_enemy_hp=4),
        _offline_transition_payload("RUN-TRAIN-4", 4, floor=7, enemy_hp=16, chosen_action_id="end_turn", reward=0.1, next_enemy_hp=None),
    ]
    validation_records = [
        _offline_transition_payload("RUN-VAL-1", 5, floor=8, enemy_hp=11, chosen_action_id="play_card|card=0|target=0", reward=0.75, next_enemy_hp=5),
        _offline_transition_payload("RUN-VAL-2", 6, floor=8, enemy_hp=17, chosen_action_id="end_turn", reward=0.15, next_enemy_hp=None),
    ]
    test_records = [
        _offline_transition_payload("RUN-TEST-1", 7, floor=9, enemy_hp=9, chosen_action_id="play_card|card=0|target=0", reward=0.7, next_enemy_hp=3),
        _offline_transition_payload("RUN-TEST-2", 8, floor=9, enemy_hp=15, chosen_action_id="end_turn", reward=0.1, next_enemy_hp=None),
    ]
    _write_jsonl(dataset_dir / "train.transitions.jsonl", train_records)
    _write_jsonl(dataset_dir / "validation.transitions.jsonl", validation_records)
    _write_jsonl(dataset_dir / "test.transitions.jsonl", test_records)
    _write_jsonl(dataset_dir / "transitions.jsonl", [*train_records, *validation_records, *test_records])
    (dataset_dir / "dataset-summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "offline-synthetic",
                "dataset_kind": "offline_rl_transitions",
                "records_path": str((dataset_dir / "transitions.jsonl").resolve()),
                "record_count": 8,
                "split": {
                    "split_paths": {
                        "train": str((dataset_dir / "train.transitions.jsonl").resolve()),
                        "validation": str((dataset_dir / "validation.transitions.jsonl").resolve()),
                        "test": str((dataset_dir / "test.transitions.jsonl").resolve()),
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_dir


def _offline_transition_payload(run_id: str, transition_index: int, *, floor: int, enemy_hp: int, chosen_action_id: str, reward: float, next_enemy_hp: int | None) -> dict:
    encoder = CombatStateEncoder()
    action_space = CombatActionSpace()
    observation = _combat_observation(run_id=run_id, floor=floor, enemy_hp=enemy_hp)
    next_observation = _combat_observation(run_id=run_id, floor=floor, enemy_hp=next_enemy_hp) if next_enemy_hp is not None else None
    binding = action_space.bind(observation)
    action_index = next(index for index, candidate in enumerate(binding.candidates) if candidate is not None and candidate.action_id == chosen_action_id)
    next_binding = action_space.bind(next_observation) if next_observation is not None else None
    return {
        "schema_version": 1,
        "record_type": "transition",
        "transition_id": f"offline:{run_id}:{transition_index}",
        "episode_id": f"offline:{run_id}",
        "session_name": "offline-synthetic",
        "session_kind": "train",
        "instance_id": "inst-01",
        "run_id": run_id,
        "character_id": "IRONCLAD",
        "floor": floor,
        "step_index": transition_index,
        "transition_index": transition_index,
        "screen_type": "COMBAT",
        "decision_stage": "combat",
        "decision_source": "heuristic",
        "policy_name": "policy-pack:planner",
        "policy_pack": "planner",
        "algorithm": "heuristic",
        "run_outcome": "won",
        "run_finish_reason": "game_over",
        "action_space_name": action_space.action_space_name,
        "action_schema_version": action_space.action_schema_version,
        "feature_space_name": encoder.feature_space_name,
        "feature_schema_version": encoder.feature_schema_version,
        "action_supported": True,
        "action_index": action_index,
        "chosen_action_id": chosen_action_id,
        "chosen_action_label": chosen_action_id,
        "chosen_action_source": "combat.hand",
        "legal_action_count": len(observation.legal_actions),
        "legal_action_ids": [candidate.action_id for candidate in observation.legal_actions],
        "action_mask": list(binding.mask),
        "reward": reward,
        "done": next_observation is None,
        "truncated": False,
        "environment_terminated": next_observation is None,
        "environment_truncated": False,
        "next_transition_id": None if next_observation is None else f"offline:{run_id}:{transition_index + 1}",
        "next_screen_type": None if next_observation is None else "COMBAT",
        "next_floor": None if next_observation is None else floor,
        "next_legal_action_count": None if next_observation is None else len(next_observation.legal_actions),
        "next_legal_action_ids": [] if next_observation is None else [candidate.action_id for candidate in next_observation.legal_actions],
        "next_action_mask": None if next_binding is None else list(next_binding.mask),
        "feature_vector": encoder.encode(observation),
        "next_feature_vector": None if next_observation is None else encoder.encode(next_observation),
        "state_summary": {"screen_type": "COMBAT", "run_id": run_id},
        "next_state_summary": None if next_observation is None else {"screen_type": "COMBAT", "run_id": run_id},
    }


def _combat_observation(*, run_id: str, floor: int, enemy_hp: int) -> StepObservation:
    state = GameStatePayload(
        screen="COMBAT",
        run_id=run_id,
        in_combat=True,
        turn=1,
        run=RunPayload(character_id="IRONCLAD", current_hp=48, max_hp=80, floor=floor, gold=120, max_energy=3),
        combat=CombatPayload(
            player=CombatPlayerPayload(current_hp=48, max_hp=80, block=0, energy=1),
            hand=[CombatHandCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike", playable=True, requires_target=True, valid_target_indices=[0], energy_cost=1)],
            enemies=[CombatEnemyPayload(index=0, enemy_id="SLIME_SMALL", name="Slime", current_hp=enemy_hp, max_hp=20, block=0, is_alive=True, intent="ATTACK")],
        ),
    )
    descriptors = AvailableActionsPayload(screen="COMBAT", actions=[ActionDescriptor(name="play_card", requires_index=True, requires_target=True), ActionDescriptor(name="end_turn")])
    build = build_candidate_actions(state, descriptors)
    return StepObservation(screen_type="COMBAT", run_id=state.run_id, state=state, action_descriptors=descriptors, legal_actions=build.candidates, build_warnings=build.unsupported_actions)


def _game_over_observation(*, run_id: str) -> StepObservation:
    state = GameStatePayload(
        screen="GAME_OVER",
        run_id=run_id,
        run=RunPayload(character_id="IRONCLAD", current_hp=30, max_hp=80, floor=9, gold=120, max_energy=3),
        game_over=GameOverPayload(is_victory=True, floor=9, character_id="IRONCLAD", can_return_to_main_menu=True, showing_summary=True),
    )
    descriptors = AvailableActionsPayload(screen="GAME_OVER", actions=[ActionDescriptor(name="return_to_main_menu")])
    build = build_candidate_actions(state, descriptors)
    return StepObservation(screen_type="GAME_OVER", run_id=state.run_id, state=state, action_descriptors=descriptors, legal_actions=build.candidates, build_warnings=build.unsupported_actions)


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
