import json
from pathlib import Path

from sts2_rl.data import (
    PublicRunArchiveDetailRecord,
    PublicRunArchiveIndexRecord,
    load_public_run_normalized_records,
    load_public_run_normalized_summary,
    load_public_run_strategic_stat_records,
    normalize_public_run_archive,
)


def test_normalize_public_run_archive_writes_normalized_and_strategic_artifacts(tmp_path: Path) -> None:
    archive_root = _build_archive_fixture(tmp_path)

    report = normalize_public_run_archive(
        source=archive_root,
        output_root=tmp_path / "normalized",
        session_name="norm-a",
    )
    records = load_public_run_normalized_records(report.output_dir)
    summary = load_public_run_normalized_summary(report.output_dir)

    assert len(records) == 2
    assert records[0].source_run_id == 101
    assert records[0].character_id == "REGENT"
    assert records[0].coverage_flags["has_detail_payload"] is True
    assert records[0].final_deck == ["CARD.STRIKE_REGENT", "CARD.ALPHA"]
    assert records[0].final_relics == ["RELIC.STARTER", "RELIC.CROWN"]
    assert records[0].rooms[0].room_type == "event"
    assert records[0].rooms[0].source_type == "reward"
    assert records[0].rooms[1].room_type == "shop"
    assert records[0].rooms[1].shop_purchased_cards == ["CARD.BETA"]
    assert records[0].rooms[2].room_type == "monster"
    assert records[1].coverage_flags["has_detail_payload"] is False
    assert summary["record_count"] == 2
    assert summary["detail_coverage_count"] == 1
    assert summary["character_histogram"] == {"REGENT": 2}
    assert report.strategic_card_stats_path.exists()
    assert report.strategic_shop_stats_path.exists()
    assert report.strategic_event_stats_path.exists()
    assert report.strategic_relic_stats_path.exists()
    assert report.strategic_encounter_stats_path.exists()
    assert report.strategic_route_stats_path.exists()

    strategic_cards = load_public_run_strategic_stat_records(report.output_dir, stat_family="card")
    beta_card = next(
        record
        for record in strategic_cards
        if record.subject_id == "CARD.BETA" and record.source_type == "shop" and record.act_id == "ACT_1"
    )
    alpha_card = next(
        record
        for record in strategic_cards
        if record.subject_id == "CARD.ALPHA" and record.source_type == "reward" and record.act_id == "ACT_1"
    )
    assert beta_card.shop_offer_count == 1
    assert beta_card.buy_count == 1
    assert beta_card.buy_rate == 1.0
    assert alpha_card.offer_count == 2
    assert alpha_card.pick_count == 1
    assert alpha_card.pick_rate == 0.5

    strategic_routes = load_public_run_strategic_stat_records(report.output_dir, stat_family="route")
    shop_route = next(record for record in strategic_routes if record.subject_id == "shop" and record.act_id == "ACT_1")
    assert shop_route.win_rate == 1.0
    assert shop_route.win_delta == 0.5


def _build_archive_fixture(tmp_path: Path) -> Path:
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True)
    raw_dir = archive_root / "raw" / "run-details" / "sync-a"
    raw_dir.mkdir(parents=True)

    detail_payload_path = raw_dir / "run-0000101.json"
    detail_payload_path.write_text(json.dumps(_detail_payload(), ensure_ascii=False, indent=2), encoding="utf-8")

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
            ascension=5,
            win=False,
            was_abandoned=False,
            killed_by="Boss",
            floors_reached=10,
            run_time_seconds=420,
            deck_size=0,
            relic_count=0,
            build_id="v0.103.0",
            source_url="https://sts2runs.com/api/runs/102",
            list_page=0,
            uploaded_at_unix=1700000202,
            fetched_at_utc="2026-04-14T00:00:00+00:00",
            raw_payload_path=str(archive_root / "raw" / "list-pages" / "sync-a" / "page-000000.json"),
            dedupe_key="sts2runs:102",
            identity_fingerprint="fingerprint-102",
        ),
    ]
    detail_records = [
        PublicRunArchiveDetailRecord(
            source_run_id=101,
            user_id=1,
            source_url="https://sts2runs.com/api/runs/101",
            fetched_at_utc="2026-04-14T00:00:00+00:00",
            raw_payload_path=str(detail_payload_path),
            raw_payload_sha256="sha101",
            detail_root_keys=["run", "userId"],
            run_root_keys=["acts", "ascension", "build_id", "game_mode", "map_point_history", "players", "platform_type", "run_time", "seed", "start_time", "was_abandoned", "win"],
        )
    ]

    _write_jsonl(archive_root / "public-run-index.jsonl", [record.as_dict() for record in index_records])
    _write_jsonl(archive_root / "public-run-details.jsonl", [record.as_dict() for record in detail_records])
    (archive_root / "summary.json").write_text(
        json.dumps(
            {
                "archive_root": str(archive_root),
                "summary_path": str(archive_root / "summary.json"),
                "known_run_count": 2,
                "detailed_run_count": 1,
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


def _detail_payload() -> dict:
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
                                        "offered_cards": [{"id": "CARD.ALPHA"}, {"id": "CARD.EPSILON"}],
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
