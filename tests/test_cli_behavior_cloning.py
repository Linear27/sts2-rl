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
    MapNodePayload,
    MapPayload,
    RunPayload,
)
from sts2_rl.env.types import StepObservation


def test_behavior_cloning_cli_train_command(tmp_path: Path) -> None:
    dataset_dir = _write_bc_dataset(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "train",
            "behavior-cloning",
            "--dataset",
            str(dataset_dir),
            "--output-root",
            str(tmp_path / "artifacts" / "bc"),
            "--session-name",
            "cli-bc",
            "--epochs",
            "20",
            "--learning-rate",
            "0.08",
            "--run-outcome-weight",
            "won=1.1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "artifacts" / "bc" / "cli-bc" / "behavior-cloning-checkpoint.json").exists()
    assert (tmp_path / "artifacts" / "bc" / "cli-bc" / "behavior-cloning-best.json").exists()
    assert (tmp_path / "artifacts" / "bc" / "cli-bc" / "summary.json").exists()


def _write_bc_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "data" / "bc-cli"
    dataset_dir.mkdir(parents=True)

    train_records = [
        _trajectory_step_payload(_map_observation(run_id="RUN-MAP-LOW-1", floor=7, hp=18), "choose_map_node|option=0", 1, "won"),
        _trajectory_step_payload(_map_observation(run_id="RUN-MAP-HIGH-1", floor=8, hp=74), "choose_map_node|option=1", 2, "won"),
        _trajectory_step_payload(_combat_observation(run_id="RUN-COMBAT-LETHAL-1", floor=10, enemy_hp=5), "play_card|card=0|target=0", 3, "won"),
        _trajectory_step_payload(_combat_observation(run_id="RUN-COMBAT-STALL-1", floor=10, enemy_hp=18), "end_turn", 4, "lost"),
    ]
    _write_jsonl(dataset_dir / "train.steps.jsonl", train_records)
    _write_jsonl(dataset_dir / "validation.steps.jsonl", train_records[:2])
    _write_jsonl(dataset_dir / "test.steps.jsonl", train_records[2:])
    _write_jsonl(dataset_dir / "steps.jsonl", train_records)
    (dataset_dir / "dataset-summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "bc-cli",
                "dataset_kind": "trajectory_steps",
                "records_path": str((dataset_dir / "steps.jsonl").resolve()),
                "split": {
                    "split_paths": {
                        "train": str((dataset_dir / "train.steps.jsonl").resolve()),
                        "validation": str((dataset_dir / "validation.steps.jsonl").resolve()),
                        "test": str((dataset_dir / "test.steps.jsonl").resolve()),
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
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
        "session_name": "bc-cli",
        "session_kind": "train",
        "instance_id": "inst-01",
        "step_index": step_index,
        "run_id": observation.run_id,
        "screen_type": observation.screen_type,
        "floor": observation.state.run.floor if observation.state.run is not None else None,
        "legal_action_count": len(observation.legal_actions),
        "legal_action_ids": [candidate.action_id for candidate in observation.legal_actions],
        "build_warnings": [],
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
            "screen_type": observation.screen_type,
            "run_id": observation.run_id,
            "available_action_count": len(observation.legal_actions),
            "run": {
                "character_id": observation.state.run.character_id,
                "floor": observation.state.run.floor,
                "current_hp": observation.state.run.current_hp,
                "max_hp": observation.state.run.max_hp,
                "gold": observation.state.run.gold,
                "max_energy": observation.state.run.max_energy,
                "deck_size": 12,
            },
            "combat": (
                {
                    "player_hp": observation.state.combat.player.current_hp,
                    "player_block": observation.state.combat.player.block,
                    "energy": observation.state.combat.player.energy,
                    "enemy_hp": [enemy.current_hp for enemy in observation.state.combat.enemies if enemy.is_alive],
                    "playable_hand_count": len([card for card in observation.state.combat.hand if card.playable]),
                }
                if observation.state.combat is not None
                else {}
            ),
        },
        "action_descriptors": observation.action_descriptors.model_dump(mode="json"),
        "state": observation.state.model_dump(mode="json"),
        "response": None,
    }


def _map_observation(*, run_id: str, floor: int, hp: int) -> StepObservation:
    state = GameStatePayload(
        screen="MAP",
        run_id=run_id,
        run=RunPayload(character_id="IRONCLAD", current_hp=hp, max_hp=80, floor=floor, gold=120, max_energy=3),
        map=MapPayload(
            available_nodes=[
                MapNodePayload(index=0, row=floor, col=1, node_type="REST"),
                MapNodePayload(index=1, row=floor, col=2, node_type="ELITE"),
            ],
            is_travel_enabled=True,
            is_traveling=False,
        ),
    )
    descriptors = AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="choose_map_node", requires_index=True)])
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
            enemies=[CombatEnemyPayload(index=0, enemy_id="SLIME_SMALL", name="Slime", current_hp=enemy_hp, max_hp=20, block=0, is_alive=True)],
        ),
    )
    descriptors = AvailableActionsPayload(
        screen="COMBAT",
        actions=[ActionDescriptor(name="play_card", requires_index=True, requires_target=True), ActionDescriptor(name="end_turn")],
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


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
