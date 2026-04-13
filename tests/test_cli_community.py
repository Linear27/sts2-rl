from pathlib import Path

from typer.testing import CliRunner

import sts2_rl.cli as cli_module
from sts2_rl.cli import app
from sts2_rl.data import CommunityCardStatsImportReport

runner = CliRunner()


def test_community_cli_import_and_summary(tmp_path: Path) -> None:
    source_path = tmp_path / "community.csv"
    source_path.write_text(
        "\n".join(
            [
                "character_id,card_id,card_name,source_type,offer_count,pick_count,run_count,win_rate_with_card,win_delta",
                "SILENT,ACROBATICS,Acrobatics,reward,100,42,600,0.58,0.03",
            ]
        ),
        encoding="utf-8",
    )

    import_result = runner.invoke(
        app,
        [
            "community",
            "import",
            "--source",
            str(source_path),
            "--output-root",
            str(tmp_path / "artifacts"),
            "--session-name",
            "cli-community",
            "--source-name",
            "ststracker",
            "--snapshot-date",
            "2026-04-13",
            "--game-version",
            "0.3.7",
        ],
    )
    summary_result = runner.invoke(
        app,
        [
            "community",
            "summary",
            "--source",
            str(tmp_path / "artifacts" / "cli-community"),
        ],
    )

    assert import_result.exit_code == 0
    assert "Community Card Stats Import" in import_result.stdout
    assert summary_result.exit_code == 0
    assert "Community Card Stats Summary" in summary_result.stdout
    assert (tmp_path / "artifacts" / "cli-community" / "summary.json").exists()


def test_community_cli_import_spiremeta(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_import_spiremeta_community_card_stats(**kwargs):
        captured.update(kwargs)
        output_dir = tmp_path / "artifacts" / "spiremeta-cli"
        output_dir.mkdir(parents=True)
        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            (
                '{"record_count": 2, "card_count": 2, "source_kind": "spiremeta_api", '
                '"source_request_count": 1, "character_histogram": {"REGENT": 2}, '
                '"snapshot_label_histogram": {"spiremeta-cli": 2}, '
                '"pick_rate_stats": {"count": 2, "min": 0.5, "mean": 0.65, "max": 0.8}, '
                '"win_rate_with_card_stats": {"count": 2, "min": 0.47, "mean": 0.51, "max": 0.55}, '
                '"summary_path": "'
                + str(summary_path).replace("\\", "\\\\")
                + '"}'
            ),
            encoding="utf-8",
        )
        manifest_path = output_dir / "source-manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")
        raw_root = output_dir / "raw" / "spiremeta"
        raw_root.mkdir(parents=True)
        return CommunityCardStatsImportReport(
            output_dir=output_dir,
            records_path=output_dir / "community-card-stats.jsonl",
            table_path=output_dir / "community-card-stats.csv",
            summary_path=summary_path,
            record_count=2,
            card_count=2,
            source_manifest_path=manifest_path,
            raw_payload_root=raw_root,
        )

    monkeypatch.setattr(cli_module, "import_spiremeta_community_card_stats", fake_import_spiremeta_community_card_stats)

    result = runner.invoke(
        app,
        [
            "community",
            "import-spiremeta",
            "--character",
            "regent",
            "--output-root",
            str(tmp_path / "artifacts"),
            "--api-key",
            "sm_pub_test_key",
            "--session-name",
            "spiremeta-cli",
            "--snapshot-date",
            "2026-04-13",
            "--snapshot-label",
            "spiremeta-cli",
        ],
    )

    assert result.exit_code == 0
    assert "SpireMeta Community Import" in result.stdout
    assert captured["characters"] == ["regent"]
    assert captured["api_key"] == "sm_pub_test_key"
