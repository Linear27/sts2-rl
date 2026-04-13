import json
from pathlib import Path

from sts2_rl.train import (
    StrategicPretrainModel,
    StrategicPretrainTrainConfig,
    train_strategic_pretrain_policy,
)


def test_train_strategic_pretrain_policy_writes_artifacts_and_ranks_actions(tmp_path: Path) -> None:
    dataset_dir = _write_strategic_dataset(tmp_path)
    report = train_strategic_pretrain_policy(
        dataset_source=dataset_dir,
        output_root=tmp_path / "artifacts" / "strategic",
        session_name="strategic-unit",
        config=StrategicPretrainTrainConfig(
            epochs=70,
            learning_rate=0.12,
            l2=0.0001,
            seed=9,
            run_outcome_weights={"win": 1.1, "loss": 0.9},
        ),
    )

    assert report.checkpoint_path.exists()
    assert report.best_checkpoint_path.exists()
    assert report.metrics_path.exists()
    assert report.summary_path.exists()
    assert report.train_example_count == 6
    assert report.validation_example_count == 3
    assert report.test_example_count == 3
    assert report.feature_count > 10
    assert report.decision_type_count == 3
    assert report.best_epoch >= 1
    assert report.split_strategy == "manifest_split"

    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["checkpoint_metadata"]["algorithm"] == "strategic_pretrain"
    assert summary["train"]["candidate_choice"]["loss"] is not None
    assert summary["train"]["positive_only"]["loss"] is not None
    assert summary["train"]["value"]["log_loss"] is not None

    model = StrategicPretrainModel.load(report.best_checkpoint_path)
    low_floor_reward = model.rank_actions(
        decision_type="reward_card_pick",
        context=_context_payload(floor=4, room_type="monster", map_point_type="monster", source_type="reward"),
        candidate_actions=["CARD.BLOCK", "CARD.DAMAGE"],
    )
    high_floor_reward = model.rank_actions(
        decision_type="reward_card_pick",
        context=_context_payload(floor=18, room_type="monster", map_point_type="monster", source_type="reward"),
        candidate_actions=["CARD.BLOCK", "CARD.DAMAGE"],
    )
    low_floor_shop = model.rank_actions(
        decision_type="shop_buy",
        context=_context_payload(floor=5, room_type="shop", map_point_type="shop", source_type="shop"),
        candidate_actions=["CARD.CHEAP", "CARD.POWER"],
    )
    high_floor_shop = model.rank_actions(
        decision_type="shop_buy",
        context=_context_payload(floor=17, room_type="shop", map_point_type="shop", source_type="shop"),
        candidate_actions=["CARD.CHEAP", "CARD.POWER"],
    )

    assert low_floor_reward[0]["candidate_id"] == "CARD.BLOCK"
    assert high_floor_reward[0]["candidate_id"] == "CARD.DAMAGE"
    assert low_floor_shop[0]["candidate_id"] == "CARD.CHEAP"
    assert high_floor_shop[0]["candidate_id"] == "CARD.POWER"
    assert model.predict_value(
        decision_type="rest_site_action",
        context_feature_map={
            "bias": 1.0,
            "decision_type:rest_site_action": 1.0,
            "decision_stage:rest": 1.0,
            "support_quality:chosen_only": 1.0,
            "source_name:sts2runs": 1.0,
            "character:regent": 1.0,
            "game_mode:standard": 1.0,
            "platform_type:steam": 1.0,
            "room_type:rest": 1.0,
            "map_point_type:rest": 1.0,
            "source_type:rest": 1.0,
            "build_id:v0.103.0": 1.0,
            "act_id:act_1": 1.0,
            "num:floor": 18 / 60.0,
            "floor_band:<= 32": 1.0,
            "num:floor_within_act": 18 / 20.0,
            "num:act_index": 1 / 5.0,
            "num:acts_reached": 1 / 5.0,
            "num:ascension": 10 / 20.0,
            "ascension_band:<= 10": 1.0,
            "num:candidate_count": 1 / 8.0,
            "flag:has_detail_payload": 1.0,
            "flag:has_room_history": 1.0,
        },
    ) > 0.5


def _write_strategic_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "data" / "strategic"
    dataset_dir.mkdir(parents=True)

    train_records = [
        _decision_record("reward-low-1", "reward_card_pick", "full_candidates", 4, "CARD.BLOCK", ["CARD.BLOCK", "CARD.DAMAGE"], "loss", "monster", "monster", "reward"),
        _decision_record("reward-high-1", "reward_card_pick", "full_candidates", 18, "CARD.DAMAGE", ["CARD.BLOCK", "CARD.DAMAGE"], "win", "monster", "monster", "reward"),
        _decision_record("shop-low-1", "shop_buy", "full_candidates", 5, "CARD.CHEAP", ["CARD.CHEAP", "CARD.POWER"], "loss", "shop", "shop", "shop"),
        _decision_record("shop-high-1", "shop_buy", "full_candidates", 17, "CARD.POWER", ["CARD.CHEAP", "CARD.POWER"], "win", "shop", "shop", "shop"),
        _decision_record("rest-low-1", "rest_site_action", "chosen_only", 6, "rest", [], "loss", "rest", "rest", "rest"),
        _decision_record("rest-high-1", "rest_site_action", "chosen_only", 18, "upgrade", [], "win", "rest", "rest", "rest"),
    ]
    validation_records = [
        _decision_record("reward-low-2", "reward_card_pick", "full_candidates", 3, "CARD.BLOCK", ["CARD.BLOCK", "CARD.DAMAGE"], "loss", "monster", "monster", "reward"),
        _decision_record("shop-high-2", "shop_buy", "full_candidates", 19, "CARD.POWER", ["CARD.CHEAP", "CARD.POWER"], "win", "shop", "shop", "shop"),
        _decision_record("rest-low-2", "rest_site_action", "chosen_only", 5, "rest", [], "loss", "rest", "rest", "rest"),
    ]
    test_records = [
        _decision_record("reward-high-2", "reward_card_pick", "full_candidates", 20, "CARD.DAMAGE", ["CARD.BLOCK", "CARD.DAMAGE"], "win", "monster", "monster", "reward"),
        _decision_record("shop-low-2", "shop_buy", "full_candidates", 4, "CARD.CHEAP", ["CARD.CHEAP", "CARD.POWER"], "loss", "shop", "shop", "shop"),
        _decision_record("rest-high-2", "rest_site_action", "chosen_only", 17, "upgrade", [], "win", "rest", "rest", "rest"),
    ]

    _write_jsonl(dataset_dir / "train.strategic-decisions.jsonl", train_records)
    _write_jsonl(dataset_dir / "validation.strategic-decisions.jsonl", validation_records)
    _write_jsonl(dataset_dir / "test.strategic-decisions.jsonl", test_records)
    _write_jsonl(dataset_dir / "strategic-decisions.jsonl", [*train_records, *validation_records, *test_records])
    (dataset_dir / "dataset-summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "strategic-synthetic",
                "dataset_kind": "public_strategic_decisions",
                "records_path": str((dataset_dir / "strategic-decisions.jsonl").resolve()),
                "split": {
                    "split_paths": {
                        "train": str((dataset_dir / "train.strategic-decisions.jsonl").resolve()),
                        "validation": str((dataset_dir / "validation.strategic-decisions.jsonl").resolve()),
                        "test": str((dataset_dir / "test.strategic-decisions.jsonl").resolve()),
                    }
                },
                "lineage": {"source_paths": ["synthetic"], "source_kinds": ["public_run_normalized"]},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_dir


def _decision_record(
    decision_id: str,
    decision_type: str,
    support_quality: str,
    floor: int,
    chosen_action: str,
    candidate_actions: list[str],
    run_outcome: str,
    room_type: str,
    map_point_type: str,
    source_type: str,
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
        "room_type": room_type,
        "map_point_type": map_point_type,
        "model_id": f"MODEL-{decision_type}",
        "decision_type": decision_type,
        "support_quality": support_quality,
        "reconstruction_confidence": 1.0 if support_quality == "full_candidates" else 0.5,
        "source_type": source_type,
        "candidate_actions": candidate_actions,
        "chosen_action": chosen_action,
        "alternate_actions": [item for item in candidate_actions if item != chosen_action],
        "chosen_present_in_candidates": True if candidate_actions else None,
        "source_record_path": "synthetic",
        "source_record_index": floor,
        "provenance": {"source_url": "synthetic"},
        "metadata": {"artifact_family": "public_strategic_decisions", "has_detail_payload": True, "has_room_history": True},
    }


def _context_payload(*, floor: int, room_type: str, map_point_type: str, source_type: str) -> dict[str, object]:
    return {
        "source_name": "sts2runs",
        "character_id": "regent",
        "ascension": 10,
        "build_id": "v0.103.0",
        "game_mode": "standard",
        "platform_type": "steam",
        "acts_reached": 1,
        "act_index": 1,
        "act_id": "act_1",
        "floor": floor,
        "floor_within_act": floor,
        "room_type": room_type,
        "map_point_type": map_point_type,
        "source_type": source_type,
        "candidate_count": 2,
        "metadata": {"has_detail_payload": True, "has_room_history": True},
    }


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
