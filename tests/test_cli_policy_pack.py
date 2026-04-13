from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app
from sts2_rl.train import CombatEvaluationReport

runner = CliRunner()


def test_cli_eval_policy_pack_passes_prepare_target_options(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_policy_pack_evaluation(**kwargs):
        captured.update(kwargs)
        session_dir = tmp_path / "policy-pack-cli"
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        summary_path.write_text("{}", encoding="utf-8")
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")
        return CombatEvaluationReport(
            base_url="http://127.0.0.1:8081",
            env_steps=12,
            combat_steps=7,
            heuristic_steps=12,
            total_reward=1.5,
            final_screen="GAME_OVER",
            final_run_id="run-1",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={"combat_win_rate": 0.5, "reward_per_combat": 0.75},
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=3,
        )

    monkeypatch.setattr("sts2_rl.cli.run_policy_pack_evaluation", _fake_run_policy_pack_evaluation)

    result = runner.invoke(
        app,
        [
            "eval",
            "policy-pack",
            "--base-url",
            "http://127.0.0.1:8081",
            "--policy-pack",
            "planner",
            "--output-root",
            str(tmp_path),
            "--prepare-target",
            "character_select",
            "--prepare-max-steps",
            "6",
            "--prepare-max-idle-polls",
            "9",
            "--no-prepare-main-menu",
        ],
    )

    assert result.exit_code == 0
    assert captured["prepare_target"] == "character_select"
    assert captured["prepare_max_steps"] == 6
    assert captured["prepare_max_idle_polls"] == 9
    assert "Prepare Target" in result.stdout


def test_cli_eval_policy_pack_passes_strategic_options(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_policy_pack_evaluation(**kwargs):
        captured.update(kwargs)
        session_dir = tmp_path / "policy-pack-cli-strategic"
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        summary_path.write_text("{}", encoding="utf-8")
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")
        return CombatEvaluationReport(
            base_url="http://127.0.0.1:8081",
            env_steps=4,
            combat_steps=2,
            heuristic_steps=4,
            total_reward=0.5,
            final_screen="MAP",
            final_run_id="run-2",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={"combat_win_rate": 1.0, "reward_per_combat": 0.5},
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
        )

    monkeypatch.setattr("sts2_rl.cli.run_policy_pack_evaluation", _fake_run_policy_pack_evaluation)

    strategic_path = tmp_path / "strategic-best.json"
    strategic_path.write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "eval",
            "policy-pack",
            "--base-url",
            "http://127.0.0.1:8081",
            "--output-root",
            str(tmp_path),
            "--strategic-checkpoint-path",
            str(strategic_path),
            "--strategic-mode",
            "dominant",
            "--strategic-hook",
            "reward",
        ],
    )

    assert result.exit_code == 0
    strategic_config = captured["strategic_model_config"]
    assert strategic_config is not None
    assert strategic_config.checkpoint_path == strategic_path.resolve()
    assert strategic_config.mode == "dominant"
    assert strategic_config.hooks == ("reward",)
