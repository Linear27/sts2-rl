from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app
from sts2_rl.collect.runner import CollectionReport
from sts2_rl.runtime.manifest import InstanceSpec
from sts2_rl.train import CombatEvaluationReport, CombatTrainingReport

runner = CliRunner()


def test_cli_collect_rollouts_passes_game_run_contract(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    spec = InstanceSpec(
        instance_id="inst-01",
        instance_root=tmp_path / "inst-01",
        logs_root=tmp_path / "logs",
        api_port=8080,
        base_url="http://127.0.0.1:8080",
    )

    def _fake_collect_round_robin(instance_specs, **kwargs):
        captured["instance_specs"] = instance_specs
        captured.update(kwargs)
        output_root = Path(kwargs["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "inst-01-summary.json"
        output_path = output_root / "inst-01.jsonl"
        combat_outcomes_path = output_root / "inst-01-combat-outcomes.jsonl"
        summary_path.write_text("{}", encoding="utf-8")
        output_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")
        return [
            CollectionReport(
                instance_id="inst-01",
                base_url=spec.base_url,
                output_path=output_path,
                summary_path=summary_path,
                combat_outcomes_path=combat_outcomes_path,
                step_count=0,
                last_screen="MAP",
                last_run_id="run-1",
                stop_reason="game_run_contract_mismatch",
            )
        ]

    monkeypatch.setattr("sts2_rl.cli.load_instance_config", lambda _config: object())
    monkeypatch.setattr("sts2_rl.cli.build_instance_specs", lambda _resolved: [spec])
    monkeypatch.setattr("sts2_rl.cli.collect_round_robin", _fake_collect_round_robin)

    result = runner.invoke(
        app,
        [
            "collect",
            "rollouts",
            "--output-root",
            str(tmp_path / "out"),
            "--session-name",
            "cli-contract",
            "--run-mode",
            "custom",
            "--game-seed",
            "SEED-001",
            "--game-character-id",
            "IRONCLAD",
            "--game-ascension",
            "0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    contract = captured["game_run_contract"]
    assert contract is not None
    assert contract.game_seed == "SEED-001"
    assert contract.character_id == "IRONCLAD"
    assert contract.ascension == 0


def test_cli_train_combat_dqn_passes_game_run_contract(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run_combat_dqn_training(**kwargs):
        captured.update(kwargs)
        session_dir = tmp_path / "combat-dqn-cli"
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-train.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        checkpoint_path = session_dir / "combat-dqn-checkpoint.json"
        best_checkpoint_path = session_dir / "combat-dqn-best.json"
        for path in (summary_path, log_path, combat_outcomes_path, checkpoint_path, best_checkpoint_path):
            path.write_text("{}" if path.suffix == ".json" else "", encoding="utf-8")
        return CombatTrainingReport(
            base_url="http://127.0.0.1:8080",
            env_steps=0,
            rl_steps=0,
            heuristic_steps=0,
            update_steps=0,
            total_reward=0.0,
            final_screen="MAP",
            final_run_id="run-1",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            periodic_checkpoint_count=0,
            stop_reason="game_run_contract_mismatch",
        )

    monkeypatch.setattr("sts2_rl.cli.run_combat_dqn_training", _fake_run_combat_dqn_training)

    result = runner.invoke(
        app,
        [
            "train",
            "combat-dqn",
            "--output-root",
            str(tmp_path),
            "--session-name",
            "cli-dqn",
            "--run-mode",
            "custom",
            "--game-seed",
            "SEED-TRAIN",
            "--game-character-id",
            "IRONCLAD",
            "--game-ascension",
            "1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    contract = captured["game_run_contract"]
    assert contract is not None
    assert contract.run_mode == "custom"
    assert contract.game_seed == "SEED-TRAIN"
    assert contract.ascension == 1


def test_cli_eval_policy_pack_passes_game_run_contract(monkeypatch, tmp_path: Path) -> None:
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
            env_steps=0,
            combat_steps=0,
            heuristic_steps=0,
            total_reward=0.0,
            final_screen="GAME_OVER",
            final_run_id="run-1",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={},
            stop_reason="game_run_contract_mismatch",
            completed_run_count=0,
            completed_combat_count=0,
        )

    monkeypatch.setattr("sts2_rl.cli.run_policy_pack_evaluation", _fake_run_policy_pack_evaluation)

    result = runner.invoke(
        app,
        [
            "eval",
            "policy-pack",
            "--base-url",
            "http://127.0.0.1:8081",
            "--output-root",
            str(tmp_path),
            "--run-mode",
            "custom",
            "--game-seed",
            "SEED-EVAL",
            "--game-character-id",
            "IRONCLAD",
            "--game-ascension",
            "2",
        ],
    )

    assert result.exit_code == 0, result.stdout
    contract = captured["game_run_contract"]
    assert contract is not None
    assert contract.game_seed == "SEED-EVAL"
    assert contract.character_id == "IRONCLAD"
    assert contract.ascension == 2
