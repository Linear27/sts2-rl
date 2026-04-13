from pathlib import Path

from typer.testing import CliRunner

import sts2_rl.cli as cli_module
from sts2_rl.cli import app
from sts2_rl.data import PublicRunNormalizationReport

runner = CliRunner()


def test_public_runs_normalize_and_normalized_summary_cli(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "normalized" / "norm-a"
    output_dir.mkdir(parents=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        (
            '{"record_count": 2, "detail_coverage_count": 1, '
            '"character_histogram": {"REGENT": 2}, "build_id_histogram": {"v0.103.0": 2}, '
            '"ascension_histogram": {"10": 1, "5": 1}, "outcome_histogram": {"win": 1, "loss": 1}, '
            '"room_type_histogram": {"event": 1, "shop": 1}, "acts_reached_histogram": {"1": 2}, '
            '"strategic_card_stats_path": "cards.jsonl", "strategic_shop_stats_path": "shops.jsonl", '
            '"strategic_event_stats_path": "events.jsonl", "strategic_relic_stats_path": "relics.jsonl", '
            '"strategic_encounter_stats_path": "encounters.jsonl", "strategic_route_stats_path": "routes.jsonl"}'
        ),
        encoding="utf-8",
    )

    def fake_normalize(**kwargs):
        return PublicRunNormalizationReport(
            output_dir=output_dir,
            normalized_runs_path=output_dir / "normalized-public-runs.jsonl",
            normalized_runs_table_path=output_dir / "normalized-public-runs.csv",
            strategic_card_stats_path=output_dir / "strategic-card-stats.jsonl",
            strategic_shop_stats_path=output_dir / "strategic-shop-stats.jsonl",
            strategic_event_stats_path=output_dir / "strategic-event-stats.jsonl",
            strategic_relic_stats_path=output_dir / "strategic-relic-stats.jsonl",
            strategic_encounter_stats_path=output_dir / "strategic-encounter-stats.jsonl",
            strategic_route_stats_path=output_dir / "strategic-route-stats.jsonl",
            summary_path=summary_path,
            source_manifest_path=output_dir / "source-manifest.json",
        )

    monkeypatch.setattr(cli_module, "normalize_public_run_archive", fake_normalize)

    source_dir = tmp_path / "archive"
    source_dir.mkdir()

    normalize_result = runner.invoke(
        app,
        [
            "public-runs",
            "normalize",
            "--source",
            str(source_dir),
            "--output-root",
            str(tmp_path / "normalized"),
            "--session-name",
            "norm-a",
        ],
    )
    summary_result = runner.invoke(
        app,
        [
            "public-runs",
            "normalized-summary",
            "--source",
            str(output_dir),
        ],
    )

    assert normalize_result.exit_code == 0
    assert "Public Run Normalize" in normalize_result.stdout
    assert summary_result.exit_code == 0
    assert "Public Run Normalized Summary" in summary_result.stdout
