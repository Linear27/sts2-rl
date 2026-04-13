import json
from pathlib import Path

from sts2_rl.train import CombatEvaluationReport, run_policy_checkpoint_evaluation


def test_run_policy_checkpoint_evaluation_routes_strategic_checkpoints(monkeypatch, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "strategic-pretrain-best.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "strategic_pretrain",
                "feature_schema_version": 1,
                "metadata": {"decision_types": ["reward_card_pick"]},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def _fake_run_policy_pack_evaluation(**kwargs):
        captured.update(kwargs)
        session_dir = Path(kwargs["output_root"]) / str(kwargs["session_name"])
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        summary_path.write_text(json.dumps({"stop_reason": "max_runs_reached"}, ensure_ascii=False), encoding="utf-8")
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=3,
            combat_steps=1,
            heuristic_steps=3,
            total_reward=1.25,
            final_screen="MAP",
            final_run_id="run-strategic",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={"combat_win_rate": 1.0, "reward_per_combat": 1.25},
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
        )

    monkeypatch.setattr("sts2_rl.train.policy_checkpoint.run_policy_pack_evaluation", _fake_run_policy_pack_evaluation)

    report = run_policy_checkpoint_evaluation(
        base_url="http://127.0.0.1:8080",
        checkpoint_path=checkpoint_path,
        output_root=tmp_path / "artifacts",
        session_name="strategic-eval",
        max_runs=1,
    )

    strategic_config = captured["strategic_model_config"]
    assert strategic_config is not None
    assert strategic_config.checkpoint_path == checkpoint_path.resolve()
    assert strategic_config.mode == "dominant"
    assert report.checkpoint_path == checkpoint_path.resolve()

    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["algorithm"] == "strategic_pretrain"
    assert summary["checkpoint_path"] == str(checkpoint_path.resolve())
    assert summary["checkpoint_metadata"]["algorithm"] == "strategic_pretrain"
    assert summary["strategic_model"]["checkpoint_algorithm"] == "strategic_pretrain"
    assert summary["strategic_model"]["mode"] == "dominant"
