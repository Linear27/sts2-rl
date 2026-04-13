import httpx

from sts2_rl.data import (
    load_public_run_archive_detail_records,
    load_public_run_archive_index_records,
    load_public_run_archive_state,
    load_public_run_archive_summary,
    sync_sts2runs_public_run_archive,
)


def test_sync_sts2runs_public_run_archive_collects_pages_and_details(tmp_path) -> None:
    transport = httpx.MockTransport(_sts2runs_handler())

    report = sync_sts2runs_public_run_archive(
        archive_root=tmp_path / "archive",
        session_name="sync-a",
        limit=2,
        max_list_pages=2,
        max_detail_fetches=2,
        retry_backoff_seconds=0.0,
        transport=transport,
    )

    index_records = load_public_run_archive_index_records(report.archive_root)
    detail_records = load_public_run_archive_detail_records(report.archive_root)
    state = load_public_run_archive_state(report.archive_root)
    summary = load_public_run_archive_summary(report.archive_root)

    assert report.new_run_count == 4
    assert report.detail_fetched_count == 2
    assert len(index_records) == 4
    assert len(detail_records) == 2
    assert state.pending_detail_run_ids == [10, 9]
    assert summary["known_run_count"] == 4
    assert summary["detailed_run_count"] == 2
    assert summary["pending_detail_run_count"] == 2
    assert summary["character_histogram"] == {"Regent": 2, "Defect": 2}
    assert summary["build_id_histogram"] == {"v0.103.0": 4}
    assert (report.archive_root / "raw" / "list-pages" / "sync-a" / "page-000000.json").exists()
    assert (report.archive_root / "raw" / "run-details" / "sync-a" / "run-0000012.json").exists()
    assert report.session_summary_path.exists()


def test_sync_sts2runs_public_run_archive_resumes_detail_backfill_without_duplicate_index_records(tmp_path) -> None:
    transport = httpx.MockTransport(_sts2runs_handler())

    first = sync_sts2runs_public_run_archive(
        archive_root=tmp_path / "archive",
        session_name="sync-a",
        limit=2,
        max_list_pages=2,
        max_detail_fetches=1,
        retry_backoff_seconds=0.0,
        transport=transport,
    )
    second = sync_sts2runs_public_run_archive(
        archive_root=tmp_path / "archive",
        session_name="sync-b",
        limit=2,
        max_list_pages=2,
        retry_backoff_seconds=0.0,
        transport=transport,
    )

    index_records = load_public_run_archive_index_records(first.archive_root)
    detail_records = load_public_run_archive_detail_records(first.archive_root)
    state = load_public_run_archive_state(first.archive_root)
    summary = load_public_run_archive_summary(first.archive_root)

    assert first.new_run_count == 4
    assert first.detail_fetched_count == 1
    assert second.new_run_count == 0
    assert second.duplicate_run_count == 4
    assert len(index_records) == 4
    assert len(detail_records) == 4
    assert state.pending_detail_run_ids == []
    assert summary["known_run_count"] == 4
    assert summary["detailed_run_count"] == 4
    assert summary["pending_detail_run_count"] == 0
    assert summary["session_count"] == 2


def _sts2runs_handler():
    run_rows = {
        12: {
            "id": 12,
            "user_id": 1,
            "sha256": "sha-12",
            "seed": "SEED12",
            "start_time": 1700000012,
            "character": "Regent",
            "ascension": 10,
            "win": 1,
            "was_abandoned": 0,
            "killed_by": None,
            "floors_reached": 49,
            "run_time": 2200,
            "deck_size": 31,
            "relic_count": 11,
            "build_id": "v0.103.0",
            "profile": None,
            "uploaded_at": 1700000112,
        },
        11: {
            "id": 11,
            "user_id": 1,
            "sha256": "sha-11",
            "seed": "SEED11",
            "start_time": 1700000011,
            "character": "Defect",
            "ascension": 10,
            "win": 0,
            "was_abandoned": 0,
            "killed_by": "Boss",
            "floors_reached": 42,
            "run_time": 1800,
            "deck_size": 28,
            "relic_count": 9,
            "build_id": "v0.103.0",
            "profile": None,
            "uploaded_at": 1700000111,
        },
        10: {
            "id": 10,
            "user_id": 2,
            "sha256": "sha-10",
            "seed": "SEED10",
            "start_time": 1700000010,
            "character": "Regent",
            "ascension": 8,
            "win": 1,
            "was_abandoned": 0,
            "killed_by": None,
            "floors_reached": 49,
            "run_time": 2100,
            "deck_size": 29,
            "relic_count": 10,
            "build_id": "v0.103.0",
            "profile": None,
            "uploaded_at": 1700000110,
        },
        9: {
            "id": 9,
            "user_id": 2,
            "sha256": "sha-09",
            "seed": "SEED09",
            "start_time": 1700000009,
            "character": "Defect",
            "ascension": 8,
            "win": 0,
            "was_abandoned": 1,
            "killed_by": "Elite",
            "floors_reached": 21,
            "run_time": 900,
            "deck_size": 20,
            "relic_count": 6,
            "build_id": "v0.103.0",
            "profile": None,
            "uploaded_at": 1700000109,
        },
    }
    page_map = {
        0: [run_rows[12], run_rows[11]],
        1: [run_rows[10], run_rows[9]],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/runs":
            page = int(request.url.params["page"])
            limit = int(request.url.params["limit"])
            rows = page_map.get(page, [])
            return httpx.Response(
                200,
                json={"runs": rows[:limit], "total": 4},
                request=request,
            )
        if request.url.path.startswith("/api/runs/"):
            run_id = int(request.url.path.rsplit("/", maxsplit=1)[-1])
            return httpx.Response(
                200,
                json={
                    "userId": run_rows[run_id]["user_id"],
                    "run": {
                        "acts": ["ACT.OVERGROWTH", "ACT.HIVE"],
                        "build_id": "v0.103.0",
                        "game_mode": "standard",
                        "map_point_history": [[], []],
                    },
                },
                request=request,
            )
        raise AssertionError(f"Unexpected request: {request.url}")

    return handler
