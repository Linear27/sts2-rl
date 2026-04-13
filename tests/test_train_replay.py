import json
from pathlib import Path

from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    GameOverPayload,
    GameStatePayload,
)
from sts2_rl.env.types import StepObservation, StepResult
from sts2_rl.train import CombatEvaluationReport, run_combat_dqn_replay_suite


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


def test_run_combat_dqn_replay_suite_writes_comparison_artifacts(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "seed.json"
    checkpoint_path.write_text("seed", encoding="utf-8")
    main_menu = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
    )

    def fake_env_factory(_base_url: str, _timeout: float) -> FakeEnv:
        return FakeEnv(main_menu, {})

    def fake_evaluation_fn(**kwargs) -> CombatEvaluationReport:
        session_name = kwargs["session_name"]
        iteration_index = int(session_name.rsplit("-", maxsplit=1)[-1])
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        combat_outcomes_path.write_text("", encoding="utf-8")

        if iteration_index == 1:
            summary_payload = {
                "action_histogram": {"play_card": 2, "end_turn": 1},
                "run_outcome_histogram": {"won": 1},
            }
            action_ids = ["play_card|card=0|target=0", "end_turn", "play_card|card=1|target=0"]
            env_steps = 3
            total_reward = 1.5
        elif iteration_index == 2:
            summary_payload = {
                "action_histogram": {"play_card": 2, "end_turn": 1},
                "run_outcome_histogram": {"won": 1},
            }
            action_ids = ["play_card|card=0|target=0", "end_turn", "play_card|card=1|target=0"]
            env_steps = 3
            total_reward = 1.5
        else:
            summary_payload = {
                "action_histogram": {"play_card": 1, "end_turn": 2},
                "run_outcome_histogram": {"lost": 1},
            }
            action_ids = ["play_card|card=0|target=0", "end_turn", "end_turn"]
            env_steps = 3
            total_reward = -0.5

        summary_payload.update(
            {
                "stop_reason": "max_runs_reached",
                "final_screen": "GAME_OVER",
                "completed_run_count": 1,
                "completed_combat_count": 1,
                "observed_run_seeds": [f"REPLAY-SEED-{iteration_index:03d}"],
                "observed_run_seed_histogram": {f"REPLAY-SEED-{iteration_index:03d}": 1},
                "runs_without_observed_seed": 0,
                "last_observed_seed": f"REPLAY-SEED-{iteration_index:03d}",
            }
        )
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            for index, action_id in enumerate(action_ids, start=1):
                handle.write(
                    json.dumps(
                        {
                            "record_type": "step",
                            "step_index": index,
                            "screen_type": "COMBAT",
                            "chosen_action_id": action_id,
                        },
                        ensure_ascii=False,
                    )
                )
                handle.write("\n")

        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=env_steps,
            combat_steps=3,
            heuristic_steps=0,
            total_reward=total_reward,
            final_screen="GAME_OVER",
            final_run_id=f"run-{iteration_index}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=Path(kwargs["checkpoint_path"]),
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
        )

    report = run_combat_dqn_replay_suite(
        base_url="http://127.0.0.1:8080",
        checkpoint_path=checkpoint_path,
        output_root=tmp_path,
        suite_name="replay-suite",
        repeats=3,
        prepare_main_menu=False,
        env_factory=fake_env_factory,
        evaluation_fn=fake_evaluation_fn,
    )

    assert report.repeat_count == 3
    assert report.comparison_count == 2
    assert report.exact_match_count == 1
    assert report.divergent_iteration_count == 1
    assert report.prepare_target == "none"
    assert report.summary_path.exists()
    assert report.comparisons_path.exists()

    summary_payload = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary_payload["prepare_target"] == "none"
    assert summary_payload["status_histogram"]["exact_match"] == 1
    assert summary_payload["status_histogram"]["policy_choice_diverged"] == 1
    assert summary_payload["observed_run_seed_histogram"] == {
        "REPLAY-SEED-001": 1,
        "REPLAY-SEED-002": 1,
        "REPLAY-SEED-003": 1,
    }
    assert summary_payload["divergence_family_histogram"]["policy_choice"] == 1
    assert summary_payload["divergence_category_histogram"]["chosen_action_mismatch"] == 1
    assert summary_payload["all_normalization_targets_reached"] is True
    assert summary_payload["all_exact_match"] is False
    assert report.iterations[0].prepare_target == "none"
    assert report.iterations[0].normalization_report["target"] == "none"
    assert report.iterations[0].normalization_report["initial_screen"] == "MAIN_MENU"

    comparison_payloads = [
        json.loads(line) for line in report.comparisons_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert comparison_payloads[0]["candidate_iteration"] == 2
    assert comparison_payloads[0]["status"] == "exact_match"
    assert comparison_payloads[1]["candidate_iteration"] == 3
    assert comparison_payloads[1]["status"] == "policy_choice_diverged"
    assert comparison_payloads[1]["first_action_divergence_index"] == 3
    assert comparison_payloads[1]["action_histogram_delta"]["end_turn"]["delta"] == 1
    assert comparison_payloads[1]["diagnostic"]["family"] == "policy_choice"
    assert comparison_payloads[1]["diagnostic"]["category"] == "chosen_action_mismatch"


def test_run_combat_dqn_replay_suite_prepares_main_menu(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "seed.json"
    checkpoint_path.write_text("seed", encoding="utf-8")

    game_over = _observation(
        GameStatePayload(
            screen="GAME_OVER",
            run_id="run-old",
            game_over=GameOverPayload(
                is_victory=False,
                can_return_to_main_menu=True,
                showing_summary=True,
            ),
        ),
        AvailableActionsPayload(screen="GAME_OVER", actions=[ActionDescriptor(name="return_to_main_menu")]),
    )
    main_menu = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
    )

    prepared_envs: list[FakeEnv] = []

    def fake_env_factory(_base_url: str, _timeout: float) -> FakeEnv:
        env = FakeEnv(
            game_over,
            {
                "return_to_main_menu": [
                    StepResult(observation=main_menu, terminated=False, truncated=False, reward=0.0, response=None, info={})
                ]
            },
        )
        prepared_envs.append(env)
        return env

    def fake_evaluation_fn(**kwargs) -> CombatEvaluationReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        summary_path.write_text(
            json.dumps(
                {
                    "action_histogram": {"open_character_select": 1},
                    "run_outcome_histogram": {"interrupted": 1},
                    "observed_run_seeds": ["PREP-SEED-001"],
                    "observed_run_seed_histogram": {"PREP-SEED-001": 1},
                    "runs_without_observed_seed": 0,
                    "last_observed_seed": "PREP-SEED-001",
                    "stop_reason": "max_env_steps_reached",
                    "final_screen": "MAIN_MENU",
                    "completed_run_count": 0,
                    "completed_combat_count": 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log_path.write_text(
            json.dumps({"record_type": "step", "step_index": 1, "screen_type": "MAIN_MENU", "chosen_action_id": "open_character_select"})
            + "\n",
            encoding="utf-8",
        )
        combat_outcomes_path.write_text("", encoding="utf-8")
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=1,
            combat_steps=0,
            heuristic_steps=1,
            total_reward=0.0,
            final_screen="MAIN_MENU",
            final_run_id="run_unknown",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=Path(kwargs["checkpoint_path"]),
            stop_reason="max_env_steps_reached",
            completed_run_count=0,
            completed_combat_count=0,
        )

    report = run_combat_dqn_replay_suite(
        base_url="http://127.0.0.1:8080",
        checkpoint_path=checkpoint_path,
        output_root=tmp_path,
        suite_name="prepared-suite",
        repeats=2,
        prepare_main_menu=True,
        env_factory=fake_env_factory,
        evaluation_fn=fake_evaluation_fn,
    )

    assert all(env.closed for env in prepared_envs)
    assert report.prepare_target == "main_menu"
    assert report.iterations[0].start_screen == "MAIN_MENU"
    assert report.iterations[0].prepare_target == "main_menu"
    assert report.iterations[0].normalization_report["target"] == "main_menu"
    assert report.iterations[0].normalization_report["reached_target"] is True
    assert report.iterations[0].prepare_action_ids == ["return_to_main_menu"]
    assert report.comparisons[0].status == "exact_match"


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
