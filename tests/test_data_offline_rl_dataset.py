import json
from pathlib import Path

from sts2_rl.data import build_dataset_from_manifest, load_dataset_summary
from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    CombatEnemyPayload,
    CombatHandCardPayload,
    CombatPayload,
    CombatPlayerPayload,
    GameStatePayload,
    MapPayload,
    RunPayload,
)
from sts2_rl.env.types import StepObservation


def test_build_offline_rl_dataset_writes_transition_episode_and_feature_stats(tmp_path: Path) -> None:
    trajectory_root = tmp_path / "data" / "trajectories" / "session-offline"
    trajectory_root.mkdir(parents=True)
    trajectory_path = trajectory_root / "inst-01.jsonl"
    combat_a = _combat_observation(run_id="RUN-001", floor=3, enemy_hp=12)
    combat_b = _combat_observation(run_id="RUN-001", floor=3, enemy_hp=6)
    map_observation = _map_observation(run_id="RUN-001", floor=4)
    records = [
        _trajectory_step_payload(combat_a, "play_card|card=0|target=0", 1, reward=0.75),
        _trajectory_step_payload(combat_b, "end_turn", 2, reward=0.25),
        _trajectory_step_payload(map_observation, "proceed", 3, reward=0.0),
        {
            "schema_version": 2,
            "record_type": "run_finished",
            "timestamp_utc": "2026-04-12T00:00:00+00:00",
            "session_name": "session-offline",
            "session_kind": "collect",
            "instance_id": "inst-01",
            "run_id": "RUN-001",
            "started_step_index": 1,
            "finished_step_index": 3,
            "reason": "game_over",
            "victory": True,
            "outcome": "won",
            "state_summary": {"screen_type": "GAME_OVER", "run_id": "RUN-001"},
        },
    ]
    with trajectory_path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in records:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    manifest_path = tmp_path / "offline.toml"
    manifest_path.write_text(
        f"""
schema_version = 1
dataset_name = "offline-repro"
dataset_kind = "offline_rl_transitions"

[[sources]]
path = "{(tmp_path / "data" / "trajectories").as_posix()}"
source_kind = "trajectory_log"
recursive = true

[filters]
screen_types = ["COMBAT"]
session_kinds = ["collect"]

[split]
train_fraction = 1.0
validation_fraction = 0.0
test_fraction = 0.0
seed = 11
group_by = "run_id"

[output]
export_csv = true
include_top_level_records = true
write_split_files = true
""".strip(),
        encoding="utf-8",
    )

    report = build_dataset_from_manifest(manifest_path, output_dir=tmp_path / "data" / "offline-out")
    summary = load_dataset_summary(report.output_dir)

    assert report.dataset_kind == "offline_rl_transitions"
    assert report.record_count == 2
    assert (report.output_dir / "transitions.jsonl").exists()
    assert (report.output_dir / "episodes.jsonl").exists()
    assert (report.output_dir / "feature-stats.json").exists()
    assert (report.output_dir / "transitions.csv").exists()
    assert (report.output_dir / "train.transitions.jsonl").exists()
    assert (report.output_dir / "train.episodes.jsonl").exists()
    assert summary["episode_count"] == 1
    assert summary["supported_transition_count"] == 2
    assert summary["run_outcome_histogram"] == {"won": 2}
    assert summary["boss_histogram"] == {"THE_CHAMP": 2}
    assert summary["route_reason_tag_histogram"] == {"search_aoe_tools": 2}
    assert summary["route_profile_histogram"] == {"search_aoe_tools": 2}
    assert summary["normalization"]["feature_schema"]["feature_count"] > 0
    assert summary["action_support_histogram"]

    transitions = (report.output_dir / "transitions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    first_transition = json.loads(transitions[0])
    assert first_transition["strategic_context"]["boss_id"] == "THE_CHAMP"
    assert first_transition["strategic_context"]["route_profile"] == "search_aoe_tools"


def _trajectory_step_payload(observation: StepObservation, chosen_action_id: str, step_index: int, *, reward: float) -> dict:
    chosen_action = next(candidate for candidate in observation.legal_actions if candidate.action_id == chosen_action_id)
    return {
        "schema_version": 2,
        "record_type": "step",
        "timestamp_utc": "2026-04-12T00:00:00+00:00",
        "session_name": "session-offline",
        "session_kind": "collect",
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
        "planner_name": "boss-conditioned-route-planner-v1",
        "planner_strategy": "boss_pathing",
        "decision_metadata": {
            "route_planner": {
                "planner_name": "boss-conditioned-route-planner-v1",
                "planner_strategy": "boss_pathing",
                "boss_encounter_id": "THE_CHAMP",
                "selected": {
                    "score": 2.1,
                    "path": [{"row": 4, "col": 0, "node_type": "Monster"}],
                    "path_node_types": ["Monster", "Shop", "Boss"],
                    "shop_count": 1,
                    "elite_count": 0,
                    "reason_tags": ["search_aoe_tools"],
                },
            }
        },
        "reward": reward,
        "reward_source": "synthetic",
        "terminated": False,
        "truncated": False,
        "info": {"run_outcome": "won"},
        "model_metrics": {},
        "state_summary": {
            "screen_type": observation.screen_type,
            "run_id": observation.run_id,
            "run": {
                "character_id": observation.state.run.character_id if observation.state.run is not None else "",
                "floor": observation.state.run.floor if observation.state.run is not None else None,
                "act_id": "THE_CITY",
                "act_index": 1,
                "boss_encounter_id": "THE_CHAMP",
            },
            "map": {
                "planned_node_types": ["Monster", "Shop", "Boss"],
                "planned_shop_count": 1,
                "route_plan": {
                    "path_node_types": ["Monster", "Shop", "Boss"],
                    "shop_count": 1,
                    "reason_tags": ["search_aoe_tools"],
                },
            },
        },
        "action_descriptors": observation.action_descriptors.model_dump(mode="json"),
        "state": observation.state.model_dump(mode="json"),
        "response": None,
    }


def _combat_observation(*, run_id: str, floor: int, enemy_hp: int) -> StepObservation:
    state = GameStatePayload(
        screen="COMBAT",
        run_id=run_id,
        in_combat=True,
        turn=1,
        run=RunPayload(character_id="IRONCLAD", current_hp=40, max_hp=80, floor=floor, gold=90, max_energy=3),
        combat=CombatPayload(
            player=CombatPlayerPayload(current_hp=40, max_hp=80, block=0, energy=1),
            hand=[
                CombatHandCardPayload(
                    index=0,
                    card_id="strike",
                    name="Strike",
                    playable=True,
                    requires_target=True,
                    valid_target_indices=[0],
                    energy_cost=1,
                )
            ],
            enemies=[CombatEnemyPayload(index=0, enemy_id="slime", name="Slime", current_hp=enemy_hp, max_hp=12, block=0, is_alive=True, intent="ATTACK")],
        ),
    )
    descriptors = AvailableActionsPayload(screen="COMBAT", actions=[ActionDescriptor(name="play_card", requires_index=True, requires_target=True), ActionDescriptor(name="end_turn")])
    build = build_candidate_actions(state, descriptors)
    return StepObservation(screen_type="COMBAT", run_id=state.run_id, state=state, action_descriptors=descriptors, legal_actions=build.candidates, build_warnings=build.unsupported_actions)


def _map_observation(*, run_id: str, floor: int) -> StepObservation:
    state = GameStatePayload(
        screen="MAP",
        run_id=run_id,
        run=RunPayload(character_id="IRONCLAD", current_hp=40, max_hp=80, floor=floor, gold=90, max_energy=3),
        map=MapPayload(is_travel_enabled=True, is_traveling=False, available_nodes=[]),
    )
    descriptors = AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="proceed")])
    build = build_candidate_actions(state, descriptors)
    return StepObservation(screen_type="MAP", run_id=state.run_id, state=state, action_descriptors=descriptors, legal_actions=build.candidates, build_warnings=build.unsupported_actions)
