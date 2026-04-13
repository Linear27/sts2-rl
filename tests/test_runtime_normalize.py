import json
from pathlib import Path

from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    CharacterSelectPayload,
    GameStatePayload,
    ModalPayload,
)
from sts2_rl.env.types import StepObservation, StepResult
from sts2_rl.runtime import normalize_runtime_state, write_runtime_normalization_report


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


def test_normalize_runtime_state_abandons_run_to_main_menu(tmp_path: Path) -> None:
    reward = _observation(
        GameStatePayload(screen="REWARD", run_id="run-1"),
        AvailableActionsPayload(screen="REWARD", actions=[ActionDescriptor(name="abandon_run")]),
    )
    confirm_modal = _observation(
        GameStatePayload(
            screen="MODAL",
            run_id="run-1",
            modal=ModalPayload(type_name="ConfirmAbandon", can_confirm=True, can_dismiss=False),
        ),
        AvailableActionsPayload(screen="MODAL", actions=[ActionDescriptor(name="confirm_modal")]),
    )
    main_menu = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
    )

    envs: list[FakeEnv] = []

    def fake_env_factory(_base_url: str, _timeout: float) -> FakeEnv:
        env = FakeEnv(
            reward,
            {
                "abandon_run": [
                    StepResult(
                        observation=confirm_modal,
                        terminated=False,
                        truncated=False,
                        reward=0.0,
                        response=None,
                        info={},
                    )
                ],
                "confirm_modal": [
                    StepResult(
                        observation=main_menu,
                        terminated=False,
                        truncated=False,
                        reward=0.0,
                        response=None,
                        info={},
                    )
                ],
            },
        )
        envs.append(env)
        return env

    report = normalize_runtime_state(
        base_url="http://127.0.0.1:8080",
        target="main_menu",
        env_factory=fake_env_factory,
    )

    assert report.reached_target is True
    assert report.stop_reason == "target_reached"
    assert report.initial_screen == "REWARD"
    assert report.final_screen == "MAIN_MENU"
    assert report.action_sequence == ["abandon_run", "confirm_modal"]
    assert report.strategy_histogram == {
        "abandon_run": 1,
        "dismiss_modal": 1,
    } or report.strategy_histogram == {
        "abandon_run": 1,
        "confirm_modal": 1,
    }
    assert report.step_count == 2
    assert report.final_observation is not None
    assert report.final_observation.screen_type == "MAIN_MENU"
    assert all(env.closed for env in envs)

    output_path = write_runtime_normalization_report(report, tmp_path / "normalize-report.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["target"] == "main_menu"
    assert payload["reached_target"] is True
    assert payload["action_sequence"] == ["abandon_run", "confirm_modal"]


def test_normalize_runtime_state_reaches_character_select_from_main_menu() -> None:
    main_menu = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
    )
    character_select = _observation(
        GameStatePayload(
            screen="CHARACTER_SELECT",
            run_id="run_unknown",
            character_select=CharacterSelectPayload(
                selected_character_id="IRONCLAD",
                can_embark=True,
                ascension=0,
            ),
        ),
        AvailableActionsPayload(screen="CHARACTER_SELECT", actions=[ActionDescriptor(name="embark")]),
    )

    envs: list[FakeEnv] = []

    def fake_env_factory(_base_url: str, _timeout: float) -> FakeEnv:
        env = FakeEnv(
            main_menu,
            {
                "open_character_select": [
                    StepResult(
                        observation=character_select,
                        terminated=False,
                        truncated=False,
                        reward=0.0,
                        response=None,
                        info={},
                    )
                ]
            },
        )
        envs.append(env)
        return env

    report = normalize_runtime_state(
        base_url="http://127.0.0.1:8080",
        target="character_select",
        env_factory=fake_env_factory,
    )

    assert report.reached_target is True
    assert report.stop_reason == "target_reached"
    assert report.initial_screen == "MAIN_MENU"
    assert report.final_screen == "CHARACTER_SELECT"
    assert report.action_sequence == ["open_character_select"]
    assert report.strategy_histogram == {"menu_to_character_select": 1}
    assert report.step_count == 1
    assert all(env.closed for env in envs)


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
