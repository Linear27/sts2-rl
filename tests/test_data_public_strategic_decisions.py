import json
from pathlib import Path

from sts2_rl.data import (
    PublicRunArchiveDetailRecord,
    PublicRunArchiveIndexRecord,
    build_dataset_from_manifest,
    load_dataset_summary,
    load_public_strategic_decision_records,
    normalize_public_run_archive,
    validate_dataset_manifest,
)


def test_build_public_strategic_decision_dataset_from_normalized_runs(tmp_path: Path) -> None:
    archive_root = _build_public_archive_fixture(tmp_path)
    normalized = normalize_public_run_archive(
        source=archive_root,
        output_root=tmp_path / "normalized",
        session_name="norm-strategic",
    )

    manifest_path = tmp_path / "public-strategic.toml"
    manifest_path.write_text(
        f"""
schema_version = 1
dataset_name = "public-strategic"
dataset_kind = "public_strategic_decisions"

[[sources]]
path = "{normalized.output_dir.as_posix()}"
source_kind = "public_run_normalized"
recursive = true

[split]
train_fraction = 0.5
validation_fraction = 0.0
test_fraction = 0.5
seed = 11
group_by = "run_id"

[output]
export_csv = true
include_top_level_records = true
write_split_files = true
""".strip(),
        encoding="utf-8",
    )

    validation = validate_dataset_manifest(manifest_path)
    report = build_dataset_from_manifest(manifest_path, output_dir=tmp_path / "datasets" / "public-strategic")
    summary = load_dataset_summary(report.output_dir)
    records = load_public_strategic_decision_records(report.output_dir)

    assert validation.source_files == (normalized.output_dir / "normalized-public-runs.jsonl",)
    assert report.dataset_kind == "public_strategic_decisions"
    assert report.record_count == 7
    assert report.filtered_out_count == 1
    assert sorted(report.split_counts.values()) == [0, 1, 6]
    assert (report.output_dir / "strategic-decisions.csv").exists()
    assert (report.output_dir / "train.strategic-decisions.jsonl").exists()
    assert (report.output_dir / "test.strategic-decisions.jsonl").exists()

    assert summary["dataset_kind"] == "public_strategic_decisions"
    assert summary["decision_type_histogram"] == {
        "reward_card_pick": 3,
        "shop_buy": 1,
        "selection_remove": 1,
        "event_choice": 1,
        "rest_site_action": 1,
    }
    assert summary["support_quality_histogram"] == {"full_candidates": 4, "chosen_only": 3}
    assert summary["build_id_histogram"] == {"v0.103.0": 6, "v0.104.0": 1}
    assert summary["game_version_histogram"] == {"v0.103.0": 6, "v0.104.0": 1}
    assert summary["source_name_histogram"] == {"sts2runs": 7}
    assert summary["candidate_coverage"]["full_candidate_count"] == 4
    assert summary["candidate_coverage"]["chosen_only_count"] == 3
    assert summary["screen_histogram"] == {"PUBLIC_STRATEGIC": 7}
    assert summary["decision_source_histogram"] == {"public_run": 7}
    assert summary["decision_stage_histogram"] == {"reward": 3, "shop": 1, "selection": 1, "event": 1, "rest": 1}
    assert summary["exports"]["strategic_decisions_table_csv"] == str((report.output_dir / "strategic-decisions.csv").resolve())

    reward_pick = next(record for record in records if record.decision_type == "reward_card_pick" and record.chosen_action == "CARD.ALPHA")
    assert reward_pick.support_quality == "full_candidates"
    assert reward_pick.candidate_actions == ["CARD.ALPHA", "CARD.GAMMA"]
    assert reward_pick.alternate_actions == ["CARD.GAMMA"]
    assert reward_pick.chosen_present_in_candidates is True
    assert reward_pick.snapshot_date is not None

    selection_remove = next(record for record in records if record.decision_type == "selection_remove")
    assert selection_remove.support_quality == "chosen_only"
    assert selection_remove.candidate_actions == []
    assert selection_remove.reconstruction_confidence == 0.4
    assert selection_remove.source_type == "shop"

    rest_action = next(record for record in records if record.decision_type == "rest_site_action")
    assert rest_action.chosen_action == "rest"
    assert rest_action.floor == 3


def test_public_strategic_dataset_filters_build_and_confidence(tmp_path: Path) -> None:
    archive_root = _build_public_archive_fixture(tmp_path)
    normalized = normalize_public_run_archive(
        source=archive_root,
        output_root=tmp_path / "normalized",
        session_name="norm-filtered",
    )

    manifest_path = tmp_path / "public-strategic-filtered.toml"
    manifest_path.write_text(
        f"""
schema_version = 1
dataset_name = "public-strategic-filtered"
dataset_kind = "public_strategic_decisions"

[[sources]]
path = "{normalized.output_dir.as_posix()}"
source_kind = "public_run_normalized"
recursive = true

[filters]
build_ids = ["v0.103.0"]
min_confidence = 0.9
decision_types = ["reward_card_pick", "shop_buy", "selection_remove"]

[split]
train_fraction = 1.0
validation_fraction = 0.0
test_fraction = 0.0
seed = 3
group_by = "run_id"

[output]
export_csv = true
include_top_level_records = true
write_split_files = false
""".strip(),
        encoding="utf-8",
    )

    report = build_dataset_from_manifest(manifest_path, output_dir=tmp_path / "datasets" / "public-strategic-filtered")
    summary = load_dataset_summary(report.output_dir)
    records = load_public_strategic_decision_records(report.output_dir)

    assert report.record_count == 3
    assert all(record.build_id == "v0.103.0" for record in records)
    assert all(record.reconstruction_confidence >= 0.9 for record in records)
    assert {record.decision_type for record in records} == {"reward_card_pick", "shop_buy"}
    assert summary["decision_type_histogram"] == {"reward_card_pick": 2, "shop_buy": 1}
    assert summary["support_quality_histogram"] == {"full_candidates": 3}


def _build_public_archive_fixture(tmp_path: Path) -> Path:
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True)
    raw_dir = archive_root / "raw" / "run-details" / "sync-a"
    raw_dir.mkdir(parents=True)

    detail_path_101 = raw_dir / "run-0000101.json"
    detail_path_102 = raw_dir / "run-0000102.json"
    detail_path_101.write_text(json.dumps(_detail_payload_101(), ensure_ascii=False, indent=2), encoding="utf-8")
    detail_path_102.write_text(json.dumps(_detail_payload_102(), ensure_ascii=False, indent=2), encoding="utf-8")

    index_records = [
        PublicRunArchiveIndexRecord(
            source_run_id=101,
            user_id=1,
            seed="SEED101",
            start_time_unix=1700000101,
            character_id="Regent",
            ascension=10,
            win=True,
            was_abandoned=False,
            killed_by=None,
            floors_reached=15,
            run_time_seconds=600,
            deck_size=2,
            relic_count=2,
            build_id="v0.103.0",
            source_url="https://sts2runs.com/api/runs/101",
            list_page=0,
            uploaded_at_unix=1700000201,
            fetched_at_utc="2026-04-14T00:00:00+00:00",
            raw_payload_path=str(archive_root / "raw" / "list-pages" / "sync-a" / "page-000000.json"),
            dedupe_key="sts2runs:101",
            identity_fingerprint="fingerprint-101",
        ),
        PublicRunArchiveIndexRecord(
            source_run_id=102,
            user_id=2,
            seed="SEED102",
            start_time_unix=1700000102,
            character_id="Regent",
            ascension=8,
            win=False,
            was_abandoned=False,
            killed_by="Boss",
            floors_reached=8,
            run_time_seconds=420,
            deck_size=0,
            relic_count=0,
            build_id="v0.104.0",
            source_url="https://sts2runs.com/api/runs/102",
            list_page=0,
            uploaded_at_unix=1700000202,
            fetched_at_utc="2026-04-14T00:00:00+00:00",
            raw_payload_path=str(archive_root / "raw" / "list-pages" / "sync-a" / "page-000000.json"),
            dedupe_key="sts2runs:102",
            identity_fingerprint="fingerprint-102",
        ),
        PublicRunArchiveIndexRecord(
            source_run_id=103,
            user_id=3,
            seed="SEED103",
            start_time_unix=1700000103,
            character_id="Regent",
            ascension=5,
            win=False,
            was_abandoned=False,
            killed_by="Elite",
            floors_reached=5,
            run_time_seconds=300,
            deck_size=0,
            relic_count=0,
            build_id="v0.103.0",
            source_url="https://sts2runs.com/api/runs/103",
            list_page=0,
            uploaded_at_unix=1700000203,
            fetched_at_utc="2026-04-14T00:00:00+00:00",
            raw_payload_path=str(archive_root / "raw" / "list-pages" / "sync-a" / "page-000000.json"),
            dedupe_key="sts2runs:103",
            identity_fingerprint="fingerprint-103",
        ),
    ]
    detail_records = [
        PublicRunArchiveDetailRecord(
            source_run_id=101,
            user_id=1,
            source_url="https://sts2runs.com/api/runs/101",
            fetched_at_utc="2026-04-14T00:00:00+00:00",
            raw_payload_path=str(detail_path_101),
            raw_payload_sha256="sha101",
            detail_root_keys=["run", "userId"],
            run_root_keys=["acts", "ascension", "build_id", "game_mode", "map_point_history", "players", "platform_type", "run_time", "seed", "start_time", "was_abandoned", "win"],
        ),
        PublicRunArchiveDetailRecord(
            source_run_id=102,
            user_id=2,
            source_url="https://sts2runs.com/api/runs/102",
            fetched_at_utc="2026-04-14T00:00:00+00:00",
            raw_payload_path=str(detail_path_102),
            raw_payload_sha256="sha102",
            detail_root_keys=["run", "userId"],
            run_root_keys=["acts", "ascension", "build_id", "game_mode", "map_point_history", "players", "platform_type", "run_time", "seed", "start_time", "was_abandoned", "win"],
        ),
    ]

    _write_jsonl(archive_root / "public-run-index.jsonl", [record.as_dict() for record in index_records])
    _write_jsonl(archive_root / "public-run-details.jsonl", [record.as_dict() for record in detail_records])
    (archive_root / "summary.json").write_text(
        json.dumps(
            {
                "archive_root": str(archive_root),
                "summary_path": str(archive_root / "summary.json"),
                "known_run_count": 3,
                "detailed_run_count": 2,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return archive_root


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _detail_payload_101() -> dict:
    return {
        "userId": 1,
        "run": {
            "acts": ["ACT.OVERGROWTH"],
            "ascension": 10,
            "build_id": "v0.103.0",
            "game_mode": "standard",
            "platform_type": "steam",
            "run_time": 600,
            "seed": "SEED101",
            "start_time": 1700000101,
            "was_abandoned": False,
            "win": True,
            "players": [
                {
                    "character": "CHARACTER.REGENT",
                    "deck": [{"id": "CARD.STRIKE_REGENT"}, {"id": "CARD.ALPHA"}],
                    "relics": [{"id": "RELIC.STARTER"}, {"id": "RELIC.CROWN"}],
                }
            ],
            "map_point_history": [
                [
                    {
                        "map_point_type": "ancient",
                        "rooms": [{"room_type": "event", "model_id": "EVENT.NEOW", "turns_taken": 0}],
                        "player_stats": [
                            {
                                "damage_taken": 0,
                                "gold_gained": 0,
                                "gold_spent": 0,
                                "hp_healed": 0,
                                "card_choices": [
                                    {
                                        "source_type": "reward",
                                        "offered_cards": [{"id": "CARD.ALPHA"}, {"id": "CARD.GAMMA"}],
                                        "picked_card": {"id": "CARD.ALPHA"},
                                    }
                                ],
                                "cards_gained": [{"id": "CARD.ALPHA"}],
                                "relic_choices": [{"choice": "RELIC.CROWN", "was_picked": True}],
                                "event_choices": [{"title": {"key": "NEOW.choice", "table": "events"}}],
                            }
                        ],
                    },
                    {
                        "map_point_type": "shop",
                        "rooms": [{"room_type": "shop", "turns_taken": 0, "cards": [{"id": "CARD.BETA"}, {"id": "CARD.DELTA"}]}],
                        "player_stats": [
                            {
                                "damage_taken": 0,
                                "gold_gained": 0,
                                "gold_spent": 75,
                                "hp_healed": 0,
                                "cards_purchased": [{"id": "CARD.BETA"}],
                                "cards_removed": [{"id": "CARD.DEFEND_REGENT"}],
                            }
                        ],
                    },
                    {
                        "map_point_type": "rest",
                        "rooms": [{"room_type": "rest", "model_id": "REST.CAMPFIRE", "turns_taken": 0}],
                        "player_stats": [
                            {
                                "damage_taken": 0,
                                "gold_gained": 0,
                                "gold_spent": 0,
                                "hp_healed": 18,
                                "rest_site_choices": ["rest"],
                            }
                        ],
                    },
                    {
                        "map_point_type": "monster",
                        "rooms": [{"room_type": "monster", "model_id": "ENCOUNTER.SLIME", "turns_taken": 4}],
                        "player_stats": [
                            {
                                "damage_taken": 3,
                                "gold_gained": 18,
                                "gold_spent": 0,
                                "hp_healed": 0,
                                "card_choices": [
                                    {
                                        "source_type": "reward",
                                        "offered_cards": [{"id": "CARD.EPSILON"}, {"id": "CARD.ZETA"}],
                                        "picked_card": {"id": "CARD.EPSILON"},
                                    }
                                ],
                            }
                        ],
                    },
                ]
            ],
        },
    }


def _detail_payload_102() -> dict:
    return {
        "userId": 2,
        "run": {
            "acts": ["ACT.OVERGROWTH"],
            "ascension": 8,
            "build_id": "v0.104.0",
            "game_mode": "standard",
            "platform_type": "steam",
            "run_time": 420,
            "seed": "SEED102",
            "start_time": 1700000102,
            "was_abandoned": False,
            "win": False,
            "players": [{"character": "CHARACTER.REGENT", "deck": [{"id": "CARD.STRIKE_REGENT"}], "relics": [{"id": "RELIC.STARTER"}]}],
            "map_point_history": [
                [
                    {
                        "map_point_type": "monster",
                        "rooms": [{"room_type": "monster", "model_id": "ENCOUNTER.SNAKE", "turns_taken": 5}],
                        "player_stats": [
                            {
                                "damage_taken": 6,
                                "gold_gained": 12,
                                "gold_spent": 0,
                                "hp_healed": 0,
                                "card_choices": [
                                    {
                                        "source_type": "reward",
                                        "offered_cards": [{"id": "CARD.THETA"}, {"id": "CARD.IOTA"}],
                                        "picked_card": {"id": "CARD.THETA"},
                                    }
                                ],
                            }
                        ],
                    }
                ]
            ],
        },
    }
