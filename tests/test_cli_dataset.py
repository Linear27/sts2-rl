import json
from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app

runner = CliRunner()


def test_dataset_cli_validate_build_and_summary(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "session-a"
    artifacts_root.mkdir(parents=True)
    combat_path = artifacts_root / "combat-outcomes.jsonl"
    combat_path.write_text(
        json.dumps(_combat_outcome_payload(run_id="RUN-001"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "predictor.toml"
    manifest_path.write_text(
        f"""
schema_version = 1
dataset_name = "cli-predictor"
dataset_kind = "predictor_combat_outcomes"

[[sources]]
path = "{(tmp_path / "artifacts").as_posix()}"
source_kind = "combat_outcomes"
recursive = true
""".strip(),
        encoding="utf-8",
    )
    output_dir = tmp_path / "data" / "cli-predictor"

    validate_result = runner.invoke(app, ["dataset", "validate", "--manifest", str(manifest_path)])
    build_result = runner.invoke(
        app,
        ["dataset", "build", "--manifest", str(manifest_path), "--output-dir", str(output_dir)],
    )
    summary_result = runner.invoke(app, ["dataset", "summary", "--source", str(output_dir)])

    assert validate_result.exit_code == 0
    assert "cli-predictor" in validate_result.stdout
    assert build_result.exit_code == 0
    assert "Dataset Build" in build_result.stdout
    assert output_dir.joinpath("dataset-summary.json").exists()
    assert summary_result.exit_code == 0
    assert "predictor_combat_outcomes" in summary_result.stdout


def _combat_outcome_payload(*, run_id: str) -> dict:
    return {
        "schema_version": 2,
        "record_type": "combat_finished",
        "timestamp_utc": "2026-04-12T00:00:00+00:00",
        "session_name": "session-a",
        "session_kind": "train",
        "instance_id": "inst-01",
        "run_id": run_id,
        "floor": 3,
        "combat_index": 3,
        "started_step_index": 0,
        "finished_step_index": 5,
        "outcome": "won",
        "cumulative_reward": 1.0,
        "step_count": 5,
        "enemy_ids": ["SLIME_SMALL"],
        "damage_dealt": 30,
        "damage_taken": 6,
        "start_summary": {
            "screen_type": "COMBAT",
            "run_id": run_id,
            "state_version": 8,
            "turn": 1,
            "in_combat": True,
            "available_action_count": 4,
            "build_warning_count": 0,
            "session_phase": "run",
            "control_scope": "local_player",
            "run": {
                "character_id": "IRONCLAD",
                "character_name": "Ironclad",
                "ascension": 0,
                "floor": 3,
                "current_hp": 70,
                "max_hp": 80,
                "gold": 120,
                "max_energy": 3,
                "occupied_potions": 0,
            },
            "combat": {
                "player_hp": 70,
                "player_block": 0,
                "energy": 3,
                "stars": 0,
                "focus": 0,
                "enemy_ids": ["SLIME_SMALL"],
                "enemy_hp": [18],
                "hand_card_ids": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
                "playable_hand_count": 3,
            },
        },
        "end_summary": {"screen_type": "MAP", "run_id": run_id},
        "reason": "combat_exited",
    }
