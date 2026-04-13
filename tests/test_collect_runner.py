import json
from pathlib import Path

from sts2_rl.collect.runner import collect_round_robin
from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    CharacterSelectOptionPayload,
    CharacterSelectPayload,
    GameOverPayload,
    GameStatePayload,
    MapNodePayload,
    MapPayload,
    RunPayload,
)
from sts2_rl.env.types import StepObservation, StepResult
from sts2_rl.game_run_contract import build_game_run_contract
from sts2_rl.runtime.manifest import InstanceSpec


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


def test_collect_round_robin_recovers_runs_and_stops_on_run_budget(tmp_path: Path) -> None:
    initial = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
    )
    character_select_1 = _observation(
        GameStatePayload(
            screen="CHARACTER_SELECT",
            run_id="run_unknown",
            character_select=CharacterSelectPayload(
                selected_character_id="IRONCLAD",
                characters=[
                    CharacterSelectOptionPayload(
                        index=0,
                        character_id="IRONCLAD",
                        name="Ironclad",
                        is_selected=True,
                    )
                ],
            ),
        ),
        AvailableActionsPayload(screen="CHARACTER_SELECT", actions=[ActionDescriptor(name="embark")]),
    )
    game_over_1 = _observation(
        GameStatePayload(
            screen="GAME_OVER",
            run_id="run-1",
            game_over=GameOverPayload(
                is_victory=False,
                floor=1,
                character_id="IRONCLAD",
                can_return_to_main_menu=True,
                showing_summary=True,
            ),
        ),
        AvailableActionsPayload(screen="GAME_OVER", actions=[ActionDescriptor(name="return_to_main_menu")]),
    )
    map_1 = _observation(
        GameStatePayload(
            screen="MAP",
            run_id="run-1",
            run=RunPayload(character_id="IRONCLAD", floor=1, current_hp=80, max_hp=80, gold=99, max_energy=3),
            map=MapPayload(
                available_nodes=[
                    MapNodePayload(index=0, row=1, col=0, node_type="monster"),
                ],
                is_travel_enabled=True,
            ),
        ),
        AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="choose_map_node", requires_index=True)]),
    )
    main_menu_2 = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
    )
    character_select_2 = _observation(
        GameStatePayload(
            screen="CHARACTER_SELECT",
            run_id="run_unknown",
            character_select=CharacterSelectPayload(
                selected_character_id="IRONCLAD",
                characters=[
                    CharacterSelectOptionPayload(
                        index=0,
                        character_id="IRONCLAD",
                        name="Ironclad",
                        is_selected=True,
                    )
                ],
            ),
        ),
        AvailableActionsPayload(screen="CHARACTER_SELECT", actions=[ActionDescriptor(name="embark")]),
    )
    game_over_2 = _observation(
        GameStatePayload(
            screen="GAME_OVER",
            run_id="run-2",
            game_over=GameOverPayload(
                is_victory=True,
                floor=1,
                character_id="IRONCLAD",
                can_return_to_main_menu=True,
                showing_summary=True,
            ),
        ),
        AvailableActionsPayload(screen="GAME_OVER", actions=[ActionDescriptor(name="return_to_main_menu")]),
    )
    map_2 = _observation(
        GameStatePayload(
            screen="MAP",
            run_id="run-2",
            run=RunPayload(character_id="IRONCLAD", floor=1, current_hp=80, max_hp=80, gold=99, max_energy=3),
            map=MapPayload(
                available_nodes=[
                    MapNodePayload(index=0, row=1, col=0, node_type="monster"),
                ],
                is_travel_enabled=True,
            ),
        ),
        AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="choose_map_node", requires_index=True)]),
    )

    fake_env = FakeEnv(
        initial,
        {
            "open_character_select": [
                StepResult(observation=character_select_1, terminated=False, response=None, info={}),
                StepResult(observation=character_select_2, terminated=False, response=None, info={}),
            ],
            "embark": [
                StepResult(observation=map_1, terminated=False, response=None, info={}),
                StepResult(observation=map_2, terminated=False, response=None, info={}),
            ],
            "choose_map_node|option=0": [
                StepResult(observation=game_over_1, terminated=True, response=None, info={}),
                StepResult(observation=game_over_2, terminated=True, response=None, info={}),
            ],
            "return_to_main_menu": [
                StepResult(observation=main_menu_2, terminated=False, response=None, info={}),
            ],
        },
    )

    spec = InstanceSpec(
        instance_id="inst-01",
        instance_root=tmp_path / "inst-01",
        logs_root=tmp_path / "logs",
        api_port=8080,
        base_url="http://127.0.0.1:8080",
    )

    reports = collect_round_robin(
        [spec],
        output_root=tmp_path / "out",
        max_steps_per_instance=10,
        max_runs_per_instance=2,
        poll_interval_seconds=0.01,
        idle_timeout_seconds=0.1,
        env_factory=lambda _: fake_env,
    )

    assert reports[0].step_count == 7
    assert reports[0].stop_reason == "max_runs_reached"
    assert reports[0].completed_run_count == 2
    assert fake_env.closed is True
    assert reports[0].summary_path.exists()

    lines = (tmp_path / "out" / "inst-01.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert records[0]["record_type"] == "session_started"
    step_records = [record for record in records if record["record_type"] == "step"]
    assert len(step_records) == 7
    assert step_records[0]["chosen_action_id"] == "open_character_select"
    assert step_records[0]["decision_source"] == "heuristic"
    assert step_records[0]["policy_pack"] == "baseline"
    assert step_records[0]["policy_handler"] == "main_menu-handler"
    assert step_records[1]["chosen_action_id"] == "embark"
    assert step_records[2]["chosen_action_id"] == "choose_map_node|option=0"
    assert step_records[3]["chosen_action_id"] == "return_to_main_menu"
    assert records[-1]["record_type"] == "session_finished"

    summary = json.loads(reports[0].summary_path.read_text(encoding="utf-8"))
    assert summary["env_steps"] == 7
    assert summary["completed_run_count"] == 2
    assert summary["won_runs"] == 1
    assert summary["lost_runs"] == 1
    assert summary["policy_pack_histogram"]["baseline"] >= 1
    assert summary["policy_handler_histogram"]["map-handler"] >= 1
    assert summary["decision_stage_histogram"]["default"] >= 1
    assert summary["decision_stage_histogram"]["character_select"] >= 1
    assert summary["decision_stage_histogram"]["map"] >= 1
    assert summary["decision_reason_histogram"]["route_to_monster_progression"] >= 1


def test_collect_round_robin_stops_on_game_run_contract_mismatch(tmp_path: Path) -> None:
    observation = _observation(
        GameStatePayload(
            screen="MAP",
            run_id="run-1",
            run=RunPayload(
                character_id="IRONCLAD",
                seed="WRONG-SEED",
                ascension=0,
                floor=1,
                current_hp=80,
                max_hp=80,
                gold=99,
                max_energy=3,
            ),
            map=MapPayload(
                available_nodes=[MapNodePayload(index=0, row=1, col=0, node_type="monster")],
                is_travel_enabled=True,
            ),
        ),
        AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="choose_map_node", requires_index=True)]),
    )
    fake_env = FakeEnv(observation, {"choose_map_node|option=0": []})
    spec = InstanceSpec(
        instance_id="inst-01",
        instance_root=tmp_path / "inst-01",
        logs_root=tmp_path / "logs",
        api_port=8080,
        base_url="http://127.0.0.1:8080",
    )

    reports = collect_round_robin(
        [spec],
        output_root=tmp_path / "contract-out",
        max_steps_per_instance=10,
        max_runs_per_instance=1,
        poll_interval_seconds=0.01,
        idle_timeout_seconds=0.1,
        game_run_contract=build_game_run_contract(
            run_mode="custom",
            game_seed="EXPECTED-SEED",
            character_id="IRONCLAD",
            ascension=0,
            strict=True,
        ),
        env_factory=lambda _: fake_env,
    )

    assert reports[0].stop_reason == "game_run_contract_mismatch"
    summary = json.loads(reports[0].summary_path.read_text(encoding="utf-8"))
    assert summary["game_run_contract_validation"]["status"] == "mismatch"
    assert summary["game_run_contract_validation"]["mismatch_histogram"]["seed_mismatch"] >= 1
    records = [
        json.loads(line)
        for line in (tmp_path / "contract-out" / "inst-01.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["record_type"] == "game_run_contract_mismatch" for record in records)


def _observation(state: GameStatePayload, descriptors: AvailableActionsPayload) -> StepObservation:
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type=state.screen,
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )
