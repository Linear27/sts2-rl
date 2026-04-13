import json
from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app
from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    CombatEnemyPayload,
    CombatHandCardPayload,
    CombatPayload,
    CombatPlayerPayload,
    GameStatePayload,
    RunPayload,
)
from sts2_rl.env.types import StepObservation
from sts2_rl.train.combat_encoder import CombatStateEncoder
from sts2_rl.train.combat_space import CombatActionSpace

runner = CliRunner()


def test_offline_cql_cli_train_command(tmp_path: Path) -> None:
    dataset_dir = _write_offline_dataset(tmp_path)
    result = runner.invoke(
        app,
        [
            "train",
            "offline-cql",
            "--dataset",
            str(dataset_dir),
            "--output-root",
            str(tmp_path / "artifacts" / "offline"),
            "--session-name",
            "cli-offline",
            "--epochs",
            "6",
            "--batch-size",
            "2",
            "--learning-rate",
            "0.01",
            "--hidden-size",
            "8",
        ],
    )

    assert result.exit_code == 0, result.stdout
    session_dir = tmp_path / "artifacts" / "offline" / "cli-offline"
    assert (session_dir / "offline-cql-checkpoint.json").exists()
    assert (session_dir / "offline-cql-best.json").exists()
    assert (session_dir / "offline-cql-dqn-seed.json").exists()
    assert (session_dir / "summary.json").exists()


def _write_offline_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "data" / "offline-cli"
    dataset_dir.mkdir(parents=True)
    train_records = [
        _offline_transition_payload("RUN-TRAIN-1", 1, floor=6, enemy_hp=12, chosen_action_id="play_card|card=0|target=0", reward=0.8, next_enemy_hp=6),
        _offline_transition_payload("RUN-TRAIN-2", 2, floor=6, enemy_hp=18, chosen_action_id="end_turn", reward=0.2, next_enemy_hp=None),
    ]
    _write_jsonl(dataset_dir / "train.transitions.jsonl", train_records)
    _write_jsonl(dataset_dir / "validation.transitions.jsonl", train_records[:1])
    _write_jsonl(dataset_dir / "test.transitions.jsonl", train_records[1:])
    _write_jsonl(dataset_dir / "transitions.jsonl", train_records)
    (dataset_dir / "dataset-summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "offline-cli",
                "dataset_kind": "offline_rl_transitions",
                "records_path": str((dataset_dir / "transitions.jsonl").resolve()),
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
        "session_name": "offline-cli",
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


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
