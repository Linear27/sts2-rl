from pathlib import Path

from typer.testing import CliRunner

import sts2_rl.cli as cli_module
from sts2_rl.cli import app
from sts2_rl.data import PublicRunArchiveSyncReport

runner = CliRunner()


def test_public_runs_cli_sync_and_summary(monkeypatch, tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True)
    summary_path = archive_root / "summary.json"
    session_root = archive_root / "syncs" / "sync-a"
    session_root.mkdir(parents=True)
    session_summary_path = session_root / "sync-summary.json"
    session_summary_path.write_text("{}", encoding="utf-8")
    summary_path.write_text(
        (
            '{"known_run_count": 4, "detailed_run_count": 3, "pending_detail_run_count": 1, '
            '"failed_detail_run_count": 0, "detail_coverage": 0.75, '
            '"total_list_requests": 2, "total_detail_requests": 3, '
            '"last_sync_session": "sync-a", "highest_source_run_id": 12, '
            '"character_histogram": {"Regent": 2}, "build_id_histogram": {"v0.103.0": 4}, '
            '"ascension_histogram": {"10": 2}, "win_histogram": {"True": 2}}'
        ),
        encoding="utf-8",
    )

    def fake_sync(**kwargs):
        return PublicRunArchiveSyncReport(
            archive_root=archive_root,
            index_path=archive_root / "public-run-index.jsonl",
            details_path=archive_root / "public-run-details.jsonl",
            summary_path=summary_path,
            state_path=archive_root / "sync-state.json",
            source_manifest_path=archive_root / "source-manifest.json",
            session_root=session_root,
            session_summary_path=session_summary_path,
            new_run_count=4,
            duplicate_run_count=0,
            duplicate_sha256_count=0,
            detail_fetched_count=3,
            pending_detail_run_count=1,
            failed_detail_run_count=0,
            total_run_count=4,
            detailed_run_count=3,
            list_page_count=2,
            detail_request_count=3,
        )

    monkeypatch.setattr(cli_module, "sync_sts2runs_public_run_archive", fake_sync)

    sync_result = runner.invoke(
        app,
        [
            "public-runs",
            "sync",
            "--archive-root",
            str(archive_root),
            "--session-name",
            "sync-a",
        ],
    )
    summary_result = runner.invoke(
        app,
        [
            "public-runs",
            "summary",
            "--source",
            str(archive_root),
        ],
    )

    assert sync_result.exit_code == 0
    assert "Public Run Archive Sync" in sync_result.stdout
    assert summary_result.exit_code == 0
    assert "Public Run Archive Summary" in summary_result.stdout
