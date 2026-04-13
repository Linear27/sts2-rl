import json
from pathlib import Path

import httpx

from sts2_rl.data import (
    import_community_card_stats,
    import_spiremeta_community_card_stats,
    load_community_card_stat_records,
    load_community_card_stats_source_manifest,
    load_community_card_stats_summary,
)


def test_import_community_card_stats_from_csv_and_compute_rates(tmp_path: Path) -> None:
    source_path = tmp_path / "community.csv"
    source_path.write_text(
        "\n".join(
            [
                "character_id,card_id,card_name,source_type,offer_count,pick_count,shop_offer_count,buy_count,run_count,win_rate_with_card,win_delta",
                "SILENT,ACROBATICS,Acrobatics,reward,100,42,0,0,600,0.58,0.03",
                "IRONCLAD,SHRUG_IT_OFF,Shrug It Off,shop,0,0,50,12,550,0.61,0.04",
            ]
        ),
        encoding="utf-8",
    )

    report = import_community_card_stats(
        source=source_path,
        output_root=tmp_path / "artifacts",
        session_name="community-smoke",
        source_name="ststracker",
        source_url="https://ststracker.app/",
        snapshot_date="2026-04-13",
        snapshot_label="community-smoke",
        game_version="0.3.7",
    )
    records = load_community_card_stat_records(report.output_dir)
    summary = load_community_card_stats_summary(report.output_dir)

    assert report.record_count == 2
    assert report.card_count == 2
    assert records[0].pick_rate == 0.42
    assert records[1].buy_rate == 0.24
    assert summary["record_count"] == 2
    assert summary["card_count"] == 2
    assert summary["source_histogram"] == {"ststracker": 2}
    assert summary["character_histogram"] == {"SILENT": 1, "IRONCLAD": 1}
    assert summary["source_type_histogram"] == {"reward": 1, "shop": 1}
    assert summary["game_version_histogram"] == {"0.3.7": 2}
    assert summary["top_pick_rate_cards"][0]["card_id"] == "ACROBATICS"
    assert summary["top_buy_rate_cards"][0]["card_id"] == "SHRUG_IT_OFF"
    assert summary["top_win_delta_cards"][0]["card_id"] == "SHRUG_IT_OFF"
    assert (report.output_dir / "community-card-stats.csv").exists()
    assert (report.output_dir / "community-card-stats.jsonl").exists()


def test_import_community_card_stats_from_json_with_embedded_metadata(tmp_path: Path) -> None:
    source_path = tmp_path / "community.json"
    source_path.write_text(
        json.dumps(
            {
                "source_name": "slaythestats",
                "source_url": "https://www.nexusmods.com/slaythespire2/mods/349",
                "snapshot_date": "2026-04-13",
                "snapshot_label": "personal-profile-a20",
                "game_version": "0.3.7",
                "records": [
                    {
                        "character_id": "SILENT",
                        "ascension_min": 20,
                        "ascension_max": 20,
                        "floor_band": "1-16",
                        "card_id": "BACKFLIP",
                        "card_name": "Backflip",
                        "source_type": "reward",
                        "offer_count": 80,
                        "pick_count": 36,
                        "win_rate_with_card": 0.55,
                        "win_delta": 0.01,
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = import_community_card_stats(
        source=source_path,
        output_root=tmp_path / "artifacts",
        session_name="community-json",
    )
    records = load_community_card_stat_records(report.records_path)

    assert records[0].source_name == "slaythestats"
    assert records[0].snapshot_label == "personal-profile-a20"
    assert records[0].pick_rate == 0.45
    assert records[0].ascension_min == 20
    assert records[0].ascension_max == 20


def test_import_spiremeta_community_card_stats_writes_lineage_and_normalizes_rates(tmp_path: Path) -> None:
    page_payload = {
        "items": [
            {
                "card_id": 1421,
                "name": "Seeking Edge",
                "slug": "seeking_edge",
                "card_type": "power",
                "rarity": "rare",
                "cost": 1,
                "image_url": "https://assets.spiremeta.gg/cards/regent/seeking_edge.png",
                "pick_rate": 0.80645,
                "win_rate": 47.5,
                "times_offered": 33,
                "times_picked": 21,
                "runs_with_card": 22,
                "avg_floor_obtained": 20.6,
            },
            {
                "card_id": 7,
                "name": "Strike",
                "slug": "strike",
                "card_type": "attack",
                "rarity": "starter",
                "cost": 1,
                "image_url": "https://assets.spiremeta.gg/cards/regent/strike.png",
                "pick_rate": 0.5,
                "win_rate": 55.0,
                "times_offered": 10,
                "times_picked": 5,
                "runs_with_card": 40,
                "avg_floor_obtained": 1.0,
            },
        ],
        "total": 2,
        "page": 1,
        "per_page": 100,
        "total_pages": 1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sm_pub_test_key"
        assert request.url.path == "/api/v1/stats/global/cards"
        assert request.url.params["character"] == "regent"
        return httpx.Response(
            200,
            json=page_payload,
            headers={
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "99",
                "X-RateLimit-Reset": "1713000000",
            },
            request=request,
        )

    report = import_spiremeta_community_card_stats(
        output_root=tmp_path / "artifacts",
        characters=["regent"],
        api_key="sm_pub_test_key",
        session_name="spiremeta-smoke",
        snapshot_date="2026-04-13",
        snapshot_label="spiremeta-smoke",
        game_version="0.3.7",
        transport=httpx.MockTransport(handler),
    )
    records = load_community_card_stat_records(report.output_dir)
    summary = load_community_card_stats_summary(report.output_dir)
    manifest = load_community_card_stats_source_manifest(report.output_dir)

    assert report.record_count == 2
    assert report.source_manifest_path is not None
    assert report.raw_payload_root is not None
    assert records[0].source_name == "spiremeta"
    assert records[0].card_id == "SEEKING_EDGE"
    assert records[0].pick_rate == 0.80645
    assert records[0].win_rate_with_card == 0.475
    assert records[0].metadata["aliases"] == ["SEEKING_EDGE"]
    assert records[1].card_id == "STRIKE_REGENT"
    assert "STRIKE" in records[1].metadata["aliases"]
    assert summary["source_kind"] == "spiremeta_api"
    assert summary["source_request_count"] == 1
    assert summary["raw_payload_file_count"] == 1
    assert summary["snapshot_label_histogram"] == {"spiremeta-smoke": 2}
    assert manifest.source_kind == "spiremeta_api"
    assert manifest.request_parameters["characters"] == ["REGENT"]
    assert manifest.source_files[0].response_headers["X-RateLimit-Limit"] == "100"
    assert (report.output_dir / "raw" / "spiremeta" / "regent" / "page-0001.json").exists()
