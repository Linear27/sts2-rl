import json
from pathlib import Path

from typer.testing import CliRunner

from sts2_rl.cli import app

runner = CliRunner()


def test_strategic_pretrain_cli_train_command(tmp_path: Path) -> None:
    dataset_dir = _write_dataset(tmp_path)

    result = runner.invoke(
        app,
        [
            "train",
            "strategic-pretrain",
            "--dataset",
            str(dataset_dir),
            "--output-root",
            str(tmp_path / "artifacts" / "strategic"),
            "--session-name",
            "cli-strategic",
            "--epochs",
            "50",
            "--learning-rate",
            "0.1",
            "--run-outcome-weight",
            "win=1.1",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "artifacts" / "strategic" / "cli-strategic" / "strategic-pretrain-checkpoint.json").exists()
    assert (tmp_path / "artifacts" / "strategic" / "cli-strategic" / "strategic-pretrain-best.json").exists()
    assert (tmp_path / "artifacts" / "strategic" / "cli-strategic" / "summary.json").exists()


def _write_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "data" / "strategic-cli"
    dataset_dir.mkdir(parents=True)
    records = [
        _record("reward-low", "reward_card_pick", "full_candidates", 4, "CARD.BLOCK", ["CARD.BLOCK", "CARD.DAMAGE"], "loss"),
        _record("reward-high", "reward_card_pick", "full_candidates", 18, "CARD.DAMAGE", ["CARD.BLOCK", "CARD.DAMAGE"], "win"),
        _record("rest-low", "rest_site_action", "chosen_only", 6, "rest", [], "loss"),
        _record("rest-high", "rest_site_action", "chosen_only", 17, "upgrade", [], "win"),
    ]
    _write_jsonl(dataset_dir / "train.strategic-decisions.jsonl", records)
    _write_jsonl(dataset_dir / "validation.strategic-decisions.jsonl", records[:2])
    _write_jsonl(dataset_dir / "test.strategic-decisions.jsonl", records[2:])
    _write_jsonl(dataset_dir / "strategic-decisions.jsonl", records)
    (dataset_dir / "dataset-summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "strategic-cli",
                "dataset_kind": "public_strategic_decisions",
                "records_path": str((dataset_dir / "strategic-decisions.jsonl").resolve()),
                "split": {
                    "split_paths": {
                        "train": str((dataset_dir / "train.strategic-decisions.jsonl").resolve()),
                        "validation": str((dataset_dir / "validation.strategic-decisions.jsonl").resolve()),
                        "test": str((dataset_dir / "test.strategic-decisions.jsonl").resolve()),
                    }
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_dir


def _record(
    decision_id: str,
    decision_type: str,
    support_quality: str,
    floor: int,
    chosen_action: str,
    candidate_actions: list[str],
    run_outcome: str,
) -> dict:
    return {
        "schema_version": 1,
        "record_type": "public_strategic_decision",
        "decision_id": decision_id,
        "source_name": "sts2runs",
        "snapshot_date": "2026-04-14",
        "source_run_id": floor,
        "run_id": f"RUN-{decision_id}",
        "character_id": "REGENT",
        "ascension": 10,
        "build_id": "v0.103.0",
        "game_mode": "standard",
        "platform_type": "steam",
        "run_outcome": run_outcome,
        "acts_reached": 1,
        "act_index": 1,
        "act_id": "ACT_1",
        "floor": floor,
        "floor_within_act": floor,
        "room_type": "rest" if decision_type == "rest_site_action" else "monster",
        "map_point_type": "rest" if decision_type == "rest_site_action" else "monster",
        "model_id": f"MODEL-{decision_type}",
        "decision_type": decision_type,
        "support_quality": support_quality,
        "reconstruction_confidence": 1.0 if support_quality == "full_candidates" else 0.5,
        "source_type": "rest" if decision_type == "rest_site_action" else "reward",
        "candidate_actions": candidate_actions,
        "chosen_action": chosen_action,
        "alternate_actions": [item for item in candidate_actions if item != chosen_action],
        "chosen_present_in_candidates": True if candidate_actions else None,
        "source_record_path": "synthetic",
        "source_record_index": floor,
        "provenance": {"source_url": "synthetic"},
        "metadata": {"artifact_family": "public_strategic_decisions", "has_detail_payload": True, "has_room_history": True},
    }


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
