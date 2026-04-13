import json
from pathlib import Path

from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import ActionDescriptor, AvailableActionsPayload, GameStatePayload
from sts2_rl.env.types import StepObservation
from sts2_rl.train import CombatEvaluationReport, run_combat_dqn_checkpoint_comparison


class FakeEnv:
    def __init__(self, observation: StepObservation) -> None:
        self._observation = observation
        self.closed = False

    def observe(self) -> StepObservation:
        return self._observation

    def step(self, action):  # pragma: no cover - not exercised in this test
        raise AssertionError(f"Unexpected step call: {action}")

    def close(self) -> None:
        self.closed = True


def test_run_combat_dqn_checkpoint_comparison_writes_aggregate_summary(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text("baseline", encoding="utf-8")
    candidate.write_text("candidate", encoding="utf-8")
    main_menu = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
    )

    def fake_env_factory(_base_url: str, _timeout: float) -> FakeEnv:
        return FakeEnv(main_menu)

    def fake_evaluation_fn(**kwargs) -> CombatEvaluationReport:
        checkpoint_name = Path(kwargs["checkpoint_path"]).stem
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")

        if checkpoint_name == "candidate":
            total_reward = 2.0
            combat_performance = {
                "combat_steps": 4,
                "completed_combat_count": 1,
                "won_combats": 1,
                "lost_combats": 0,
                "combat_win_rate": 1.0,
                "reward_per_combat": 2.0,
                "reward_per_combat_step": 0.5,
            }
        else:
            total_reward = 0.5
            combat_performance = {
                "combat_steps": 4,
                "completed_combat_count": 1,
                "won_combats": 0,
                "lost_combats": 1,
                "combat_win_rate": 0.0,
                "reward_per_combat": 0.5,
                "reward_per_combat_step": 0.125,
            }

        summary_path.write_text(
            json.dumps(
                {
                    "stop_reason": "max_runs_reached",
                    "completed_run_count": 1,
                    "completed_combat_count": 1,
                    "observed_run_seeds": [f"{checkpoint_name.upper()}-SEED"],
                    "observed_run_seed_histogram": {f"{checkpoint_name.upper()}-SEED": 1},
                    "runs_without_observed_seed": 0,
                    "last_observed_seed": f"{checkpoint_name.upper()}-SEED",
                    "combat_performance": combat_performance,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=5,
            combat_steps=4,
            heuristic_steps=1,
            total_reward=total_reward,
            final_screen="GAME_OVER",
            final_run_id=f"run-{checkpoint_name}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=Path(kwargs["checkpoint_path"]),
            combat_performance=combat_performance,
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
        )

    report = run_combat_dqn_checkpoint_comparison(
        base_url="http://127.0.0.1:8080",
        baseline_checkpoint_path=baseline,
        candidate_checkpoint_path=candidate,
        output_root=tmp_path,
        comparison_name="compare-smoke",
        repeats=2,
        prepare_main_menu=False,
        env_factory=fake_env_factory,
        evaluation_fn=fake_evaluation_fn,
    )

    assert report.summary_path.exists()
    assert report.iterations_path.exists()
    assert report.better_checkpoint_label == "candidate"
    assert report.prepare_target == "none"
    assert report.delta_metrics["mean_total_reward"] == 1.5
    assert report.delta_metrics["combat_win_rate"] == 1.0
    assert report.diagnostics_path.exists()
    assert all(item.prepare_target == "none" for item in report.iterations)
    assert all(item.normalization_report["target"] == "none" for item in report.iterations)

    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["prepare_target"] == "none"
    assert summary["better_checkpoint_label"] == "candidate"
    assert summary["candidate"]["combat_win_rate"] == 1.0
    assert summary["baseline"]["combat_win_rate"] == 0.0
    assert summary["baseline"]["observed_run_seed_histogram"] == {"BASELINE-SEED": 2}
    assert summary["candidate"]["observed_run_seed_histogram"] == {"CANDIDATE-SEED": 2}
    assert summary["baseline"]["all_normalization_targets_reached"] is True
    assert summary["diagnostics_path"].endswith("comparison-diagnostics.jsonl")
    assert "divergence_family_histogram" in summary
    assert len(summary["paired_diagnostics"]) == 2


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
