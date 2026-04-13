import json
from pathlib import Path

from sts2_rl.data import TrajectorySessionMetadata, TrajectorySessionRecorder
from sts2_rl.game_run_contract import build_game_run_contract
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


def test_session_recorder_writes_boundaries_summary_and_combat_outcomes(tmp_path: Path) -> None:
    game_run_contract = build_game_run_contract(
        run_mode="custom",
        game_seed="SEED-001",
        seed_source="custom_mode_manual",
        character_id="ironclad",
        ascension=0,
        strict=True,
    )
    combat_observation = _build_observation(
        GameStatePayload(
            screen="COMBAT",
            run_id="run-1",
            turn=1,
            in_combat=True,
            run=RunPayload(
                character_id="IRONCLAD",
                seed="SEED-001",
                ascension=0,
                floor=3,
                current_hp=40,
                max_hp=80,
                gold=99,
                max_energy=3,
            ),
            combat=CombatPayload(
                player=CombatPlayerPayload(current_hp=40, max_hp=80, block=0, energy=3),
                hand=[
                    CombatHandCardPayload(
                        index=0,
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
                        current_hp=12,
                        max_hp=12,
                        block=0,
                        is_alive=True,
                        intent="ATTACK",
                    )
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="COMBAT",
            actions=[ActionDescriptor(name="play_card", requires_index=True, requires_target=True)],
        ),
    )
    game_over_observation = _build_observation(
        GameStatePayload(
            screen="GAME_OVER",
            run_id="run-1",
            run=RunPayload(
                character_id="IRONCLAD",
                seed="SEED-001",
                ascension=0,
                floor=3,
                current_hp=40,
                max_hp=80,
                gold=99,
                max_energy=3,
            ),
            game_over=GameOverPayload(is_victory=True, floor=3, character_id="IRONCLAD"),
        ),
        AvailableActionsPayload(screen="GAME_OVER", actions=[]),
    )
    chosen_action = combat_observation.legal_actions[0]
    result = StepResult(observation=game_over_observation, reward=1.5, terminated=True, truncated=False, response=None, info={})

    recorder = TrajectorySessionRecorder(
        log_path=tmp_path / "trajectory.jsonl",
        summary_path=tmp_path / "summary.json",
        combat_outcomes_path=tmp_path / "combat-outcomes.jsonl",
        metadata=TrajectorySessionMetadata(
            session_name="unit-session",
            session_kind="train",
            base_url="http://127.0.0.1:8080",
            policy_name="simple-policy-v2",
            algorithm="dqn",
            config={"max_env_steps": 8},
            game_run_contract=game_run_contract,
        ),
    )

    recorder.sync_observation(combat_observation, instance_id="inst-01", step_index=0)
    recorder.log_step(
        instance_id="inst-01",
        step_index=1,
        previous_observation=combat_observation,
        result=result,
        chosen_action=chosen_action,
        policy_name="combat-dqn",
        policy_pack="planner",
        policy_handler="combat-hand-planner",
        algorithm="dqn",
        decision_source="dqn",
        decision_stage="combat",
        decision_reason="greedy_q",
        decision_score=2.5,
        planner_name="combat-hand-planner-v1",
        planner_strategy="planner",
        ranked_actions=[
            {
                "action_id": "play_card|card=0|target=0",
                "action": "play_card",
                "score": 2.5,
                "reason": "planner_selected_sequence",
                "metadata": {"sequence": ["play_card|card=0|target=0"]},
            }
        ],
        decision_metadata={
            "sequence": ["play_card|card=0|target=0"],
            "predictor": {
                "mode": "assist",
                "domain": "combat",
                "model_label": "combat-outcome-predictor.json",
                "selected": {
                    "value_estimate": 1.25,
                    "outcome_win_probability": 0.7,
                    "expected_reward": 1.8,
                    "expected_damage_delta": 9.0,
                },
            },
        },
        reward_source="combat_reward_v2",
        model_metrics={"loss": 0.25},
    )
    summary = recorder.finalize(
        instance_id="inst-01",
        stop_reason="game_over",
        step_index=1,
        final_observation=game_over_observation,
    )

    summary_payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary.session_kind == "train"
    assert summary_payload["combat_count"] == 1
    assert summary_payload["won_combats"] == 1
    assert summary_payload["completed_run_count"] == 1
    assert summary_payload["won_runs"] == 1
    assert summary_payload["run_outcome_histogram"]["won"] == 1
    assert summary_payload["run_finish_reason_histogram"]["game_over"] == 1
    assert summary_payload["decision_stage_histogram"]["combat"] == 1
    assert summary_payload["decision_reason_histogram"]["greedy_q"] == 1
    assert summary_payload["decision_source_histogram"]["dqn"] == 1
    assert summary_payload["policy_pack_histogram"]["planner"] == 1
    assert summary_payload["policy_handler_histogram"]["combat-hand-planner"] == 1
    assert summary_payload["planner_histogram"]["combat-hand-planner-v1"] == 1
    assert summary_payload["observed_run_seeds"] == ["SEED-001"]
    assert summary_payload["observed_run_seed_histogram"] == {"SEED-001": 1}
    assert summary_payload["game_run_contract"]["run_mode"] == "custom"
    assert summary_payload["game_run_contract"]["character_id"] == "IRONCLAD"
    assert summary_payload["game_run_contract_validation"]["status"] == "matched"
    assert summary_payload["game_run_contract_validation"]["observation_mismatch_count"] == 0
    assert summary_payload["game_run_contract_validation"]["seed_matches_expected"] is True
    assert summary_payload["game_run_contract_validation"]["character_matches_expected"] is True
    assert summary_payload["game_run_contract_validation"]["ascension_matches_expected"] is True
    assert summary_payload["runs_without_observed_seed"] == 0
    assert summary_payload["last_observed_seed"] == "SEED-001"
    assert summary_payload["planner_candidate_count_stats"]["count"] == 1
    assert summary_payload["predictor_mode_histogram"]["assist"] == 1
    assert summary_payload["predictor_domain_histogram"]["combat"] == 1
    assert summary_payload["predictor_model_histogram"]["combat-outcome-predictor.json"] == 1
    assert summary_payload["predictor_value_estimate_stats"]["count"] == 1
    assert summary_payload["config"]["max_env_steps"] == 8
    assert summary_payload["non_combat_capability"]["diagnostic_count"] == 0

    trajectory_records = [json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text(encoding="utf-8").splitlines()]
    record_types = [record["record_type"] for record in trajectory_records]
    assert record_types == [
        "session_started",
        "run_started",
        "floor_started",
        "combat_started",
        "step",
        "combat_finished",
        "floor_finished",
        "run_finished",
        "session_finished",
    ]
    step_record = trajectory_records[4]
    assert step_record["chosen_action_id"] == "play_card|card=0|target=0"
    assert step_record["screen_type"] == "COMBAT"
    assert step_record["next_screen_type"] == "GAME_OVER"
    assert step_record["observed_seed"] == "SEED-001"
    assert step_record["next_observed_seed"] == "SEED-001"
    assert step_record["legal_action_ids"] == ["play_card|card=0|target=0"]
    assert step_record["next_legal_action_ids"] == []
    assert step_record["decision_source"] == "dqn"
    assert step_record["reward"] == 1.5
    assert step_record["policy_pack"] == "planner"
    assert step_record["planner_name"] == "combat-hand-planner-v1"
    assert step_record["ranked_action_count"] == 1
    assert step_record["capability_diagnostics"] == []
    assert step_record["decision_metadata"]["predictor"]["selected"]["value_estimate"] == 1.25
    assert step_record["state"]["screen"] == "COMBAT"
    assert step_record["next_state"]["screen"] == "GAME_OVER"

    combat_records = [json.loads(line) for line in (tmp_path / "combat-outcomes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert combat_records[0]["observed_seed"] == "SEED-001"
    assert combat_records[0]["outcome"] == "won"
    assert combat_records[0]["damage_dealt"] == 12


def test_session_recorder_summarizes_non_combat_capability_gaps(tmp_path: Path) -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="MAP",
            run_id="run-map-gap",
            run=RunPayload(
                character_id="IRONCLAD",
                seed="MAP-SEED-001",
                ascension=0,
                floor=5,
                current_hp=50,
                max_hp=80,
                gold=120,
                max_energy=3,
            ),
        ),
        AvailableActionsPayload(
            screen="MAP",
            actions=[
                ActionDescriptor(name="choose_map_node", requires_index=True),
                ActionDescriptor(name="mystery_map_action"),
            ],
        ),
    )
    recorder = TrajectorySessionRecorder(
        log_path=tmp_path / "trajectory.jsonl",
        summary_path=tmp_path / "summary.json",
        metadata=TrajectorySessionMetadata(
            session_name="capability-session",
            session_kind="eval",
            base_url="http://127.0.0.1:8080",
            policy_name="baseline",
        ),
    )

    recorder.sync_observation(observation, instance_id="inst-01", step_index=0)
    recorder.record_capability_diagnostics(
        instance_id="inst-01",
        step_index=0,
        observation=observation,
    )
    recorder.finalize(
        instance_id="inst-01",
        stop_reason="manual_stop",
        step_index=0,
        final_observation=observation,
    )

    summary_payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    capability = summary_payload["non_combat_capability"]
    assert capability["diagnostic_count"] == 1
    assert capability["unsupported_descriptor_count"] == 1
    assert capability["bucket_histogram"]["repo_action_space_gap"] == 1
    assert capability["descriptor_histogram"]["mystery_map_action"] == 1


def _build_observation(state: GameStatePayload, descriptors: AvailableActionsPayload) -> StepObservation:
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type=state.screen,
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )
