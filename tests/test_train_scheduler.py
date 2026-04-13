import json
from pathlib import Path

from sts2_rl.train import (
    CombatCheckpointComparisonReport,
    CombatTrainingReport,
    DqnConfig,
    run_combat_dqn_schedule,
)


def test_schedule_uses_latest_checkpoint_between_sessions(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_training_fn(**kwargs) -> CombatTrainingReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        latest = session_dir / "combat-dqn-checkpoint.json"
        best = session_dir / "combat-dqn-best.json"
        latest.write_text("latest", encoding="utf-8")
        best.write_text("best", encoding="utf-8")
        summary = session_dir / "summary.json"
        summary.write_text("{}", encoding="utf-8")
        combat_outcomes = session_dir / "combat-outcomes.jsonl"
        combat_outcomes.write_text("", encoding="utf-8")
        calls.append(
            {
                "session_name": session_name,
                "resume_from": kwargs.get("resume_from"),
            }
        )
        return CombatTrainingReport(
            base_url=kwargs["base_url"],
            env_steps=5,
            rl_steps=3,
            heuristic_steps=2,
            update_steps=4,
            total_reward=1.5,
            final_screen="COMBAT",
            final_run_id="run-1",
            log_path=session_dir / "combat-train.jsonl",
            summary_path=summary,
            combat_outcomes_path=combat_outcomes,
            checkpoint_path=latest,
            best_checkpoint_path=best,
            periodic_checkpoint_count=1,
        )

    report = run_combat_dqn_schedule(
        base_url="http://127.0.0.1:8080",
        output_root=tmp_path,
        schedule_name="schedule-latest",
        max_sessions=2,
        checkpoint_source="latest",
        dqn_config=DqnConfig(),
        training_fn=fake_training_fn,
    )

    assert report.session_count == 2
    assert report.checkpoint_source == "latest"
    assert calls[0]["resume_from"] is None
    assert calls[1]["resume_from"] == tmp_path / "schedule-latest" / "session-001" / "combat-dqn-checkpoint.json"
    assert report.sessions[0].selected_checkpoint_label == "latest"
    assert report.sessions[0].checkpoint_selection.selection_mode == "direct"
    summary = json.loads((tmp_path / "schedule-latest" / "schedule-summary.json").read_text(encoding="utf-8"))
    assert summary["session_count"] == 2
    assert summary["final_checkpoint_path"].endswith("session-002\\combat-dqn-checkpoint.json")


def test_schedule_uses_best_checkpoint_when_requested(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_training_fn(**kwargs) -> CombatTrainingReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        latest = session_dir / "combat-dqn-checkpoint.json"
        best = session_dir / "combat-dqn-best.json"
        latest.write_text("latest", encoding="utf-8")
        best.write_text("best", encoding="utf-8")
        summary = session_dir / "summary.json"
        summary.write_text("{}", encoding="utf-8")
        combat_outcomes = session_dir / "combat-outcomes.jsonl"
        combat_outcomes.write_text("", encoding="utf-8")
        calls.append(
            {
                "session_name": session_name,
                "resume_from": kwargs.get("resume_from"),
            }
        )
        return CombatTrainingReport(
            base_url=kwargs["base_url"],
            env_steps=4,
            rl_steps=2,
            heuristic_steps=2,
            update_steps=3,
            total_reward=0.5,
            final_screen="MAP",
            final_run_id="run-2",
            log_path=session_dir / "combat-train.jsonl",
            summary_path=summary,
            combat_outcomes_path=combat_outcomes,
            checkpoint_path=latest,
            best_checkpoint_path=best,
            periodic_checkpoint_count=1,
        )

    report = run_combat_dqn_schedule(
        base_url="http://127.0.0.1:8080",
        output_root=tmp_path,
        schedule_name="schedule-best",
        max_sessions=2,
        checkpoint_source="best",
        initial_resume_from=tmp_path / "seed.json",
        training_fn=fake_training_fn,
    )

    assert report.session_count == 2
    assert calls[0]["resume_from"] == tmp_path / "seed.json"
    assert calls[1]["resume_from"] == tmp_path / "schedule-best" / "session-001" / "combat-dqn-best.json"
    assert report.sessions[0].selected_checkpoint_label == "best"
    assert report.sessions[0].checkpoint_selection.selection_mode == "direct"


def test_schedule_uses_best_eval_selection_when_requested(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    comparison_calls: list[dict[str, object]] = []

    def fake_training_fn(**kwargs) -> CombatTrainingReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        latest = session_dir / "combat-dqn-checkpoint.json"
        best = session_dir / "combat-dqn-best.json"
        latest.write_text("latest", encoding="utf-8")
        best.write_text("best", encoding="utf-8")
        summary = session_dir / "summary.json"
        summary.write_text("{}", encoding="utf-8")
        combat_outcomes = session_dir / "combat-outcomes.jsonl"
        combat_outcomes.write_text("", encoding="utf-8")
        calls.append({"session_name": session_name, "resume_from": kwargs.get("resume_from")})
        return CombatTrainingReport(
            base_url=kwargs["base_url"],
            env_steps=6,
            rl_steps=4,
            heuristic_steps=2,
            update_steps=5,
            total_reward=2.0,
            final_screen="MAP",
            final_run_id="run-best-eval",
            log_path=session_dir / "combat-train.jsonl",
            summary_path=summary,
            combat_outcomes_path=combat_outcomes,
            checkpoint_path=latest,
            best_checkpoint_path=best,
            periodic_checkpoint_count=1,
        )

    def fake_comparison_fn(**kwargs) -> CombatCheckpointComparisonReport:
        comparison_dir = Path(kwargs["output_root"]) / kwargs["comparison_name"]
        comparison_dir.mkdir(parents=True, exist_ok=True)
        summary_path = comparison_dir / "comparison-summary.json"
        iterations_path = comparison_dir / "comparison-iterations.jsonl"
        log_path = comparison_dir / "comparison-log.jsonl"
        summary_path.write_text("{}", encoding="utf-8")
        iterations_path.write_text("", encoding="utf-8")
        log_path.write_text("", encoding="utf-8")
        comparison_calls.append(kwargs)
        return CombatCheckpointComparisonReport(
            base_url=kwargs["base_url"],
            comparison_dir=comparison_dir,
            summary_path=summary_path,
            iterations_path=iterations_path,
            log_path=log_path,
            baseline_checkpoint_path=Path(kwargs["baseline_checkpoint_path"]),
            candidate_checkpoint_path=Path(kwargs["candidate_checkpoint_path"]),
            repeat_count=kwargs["repeats"],
            prepare_target=str(kwargs["prepare_target"]),
            better_checkpoint_label="candidate",
            delta_metrics={"combat_win_rate": 0.5},
            baseline={"combat_win_rate": 0.5},
            candidate={"combat_win_rate": 1.0},
            iterations=[],
            diagnostics_path=comparison_dir / "comparison-diagnostics.jsonl",
            paired_diagnostics=[],
        )

    report = run_combat_dqn_schedule(
        base_url="http://127.0.0.1:8080",
        output_root=tmp_path,
        schedule_name="schedule-best-eval",
        max_sessions=2,
        checkpoint_source="best_eval",
        best_eval_repeats=2,
        best_eval_prepare_target="character_select",
        dqn_config=DqnConfig(),
        training_fn=fake_training_fn,
        comparison_fn=fake_comparison_fn,
    )

    assert report.session_count == 2
    assert report.promotion_artifacts_root == tmp_path / "schedule-best-eval" / "promotions"
    assert calls[0]["resume_from"] is None
    assert calls[1]["resume_from"] == tmp_path / "schedule-best-eval" / "session-001" / "combat-dqn-best.json"
    assert len(comparison_calls) == 2
    assert comparison_calls[0]["baseline_checkpoint_path"] == tmp_path / "schedule-best-eval" / "session-001" / "combat-dqn-checkpoint.json"
    assert comparison_calls[0]["candidate_checkpoint_path"] == tmp_path / "schedule-best-eval" / "session-001" / "combat-dqn-best.json"
    assert comparison_calls[0]["prepare_target"] == "character_select"
    assert report.sessions[0].selected_checkpoint_label == "best"
    assert report.sessions[0].checkpoint_selection.selection_mode == "best_eval"
    assert report.sessions[0].checkpoint_selection.comparison_summary_path is not None

    summary = json.loads((tmp_path / "schedule-best-eval" / "schedule-summary.json").read_text(encoding="utf-8"))
    assert summary["checkpoint_source"] == "best_eval"
    assert summary["best_eval"]["prepare_target"] == "character_select"
    assert summary["sessions"][0]["checkpoint_selection"]["selected_checkpoint_label"] == "best"
    assert summary["sessions"][0]["checkpoint_selection"]["comparison_delta_metrics"]["combat_win_rate"] == 0.5


def test_schedule_best_eval_falls_back_to_latest_on_comparison_error(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_training_fn(**kwargs) -> CombatTrainingReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        latest = session_dir / "combat-dqn-checkpoint.json"
        best = session_dir / "combat-dqn-best.json"
        latest.write_text("latest", encoding="utf-8")
        best.write_text("best", encoding="utf-8")
        summary = session_dir / "summary.json"
        summary.write_text("{}", encoding="utf-8")
        combat_outcomes = session_dir / "combat-outcomes.jsonl"
        combat_outcomes.write_text("", encoding="utf-8")
        calls.append({"session_name": session_name, "resume_from": kwargs.get("resume_from")})
        return CombatTrainingReport(
            base_url=kwargs["base_url"],
            env_steps=3,
            rl_steps=2,
            heuristic_steps=1,
            update_steps=2,
            total_reward=1.0,
            final_screen="COMBAT",
            final_run_id="run-fallback",
            log_path=session_dir / "combat-train.jsonl",
            summary_path=summary,
            combat_outcomes_path=combat_outcomes,
            checkpoint_path=latest,
            best_checkpoint_path=best,
            periodic_checkpoint_count=1,
        )

    def failing_comparison_fn(**kwargs) -> CombatCheckpointComparisonReport:
        raise RuntimeError("boom")

    report = run_combat_dqn_schedule(
        base_url="http://127.0.0.1:8080",
        output_root=tmp_path,
        schedule_name="schedule-best-eval-fallback",
        max_sessions=2,
        checkpoint_source="best_eval",
        best_eval_fallback="latest",
        dqn_config=DqnConfig(),
        training_fn=fake_training_fn,
        comparison_fn=failing_comparison_fn,
    )

    assert report.session_count == 2
    assert calls[1]["resume_from"] == tmp_path / "schedule-best-eval-fallback" / "session-001" / "combat-dqn-checkpoint.json"
    assert report.sessions[0].selected_checkpoint_label == "latest"
    assert report.sessions[0].checkpoint_selection.selection_mode == "best_eval_fallback"
    assert report.sessions[0].checkpoint_selection.fallback_reason == "comparison_failed"
    assert report.sessions[0].checkpoint_selection.comparison_error == "RuntimeError: boom"
