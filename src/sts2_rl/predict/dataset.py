from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sts2_rl.data import (
    DATASET_SUMMARY_FILENAME as BUILT_DATASET_SUMMARY_FILENAME,
    PREDICTOR_EXAMPLES_FILENAME as BUILT_PREDICTOR_EXAMPLES_FILENAME,
    BuiltDatasetReport,
    DatasetManifest,
    DatasetOutputSpec,
    DatasetSourceSpec,
    DatasetSplitSpec,
    build_dataset_from_manifest,
    discover_combat_outcome_paths as discover_predictor_combat_outcome_paths,
)

from .schema import PredictorExample

COMBAT_OUTCOMES_FILENAME = "combat-outcomes.jsonl"
DATASET_SUMMARY_FILENAME = BUILT_DATASET_SUMMARY_FILENAME
PREDICTOR_EXAMPLES_FILENAME = BUILT_PREDICTOR_EXAMPLES_FILENAME


@dataclass(frozen=True)
class DatasetExtractionReport:
    output_dir: Path
    examples_path: Path
    summary_path: Path
    manifest_path: Path
    source_paths: tuple[Path, ...]
    combat_outcome_paths: tuple[Path, ...]
    example_count: int
    feature_count: int
    outcome_histogram: dict[str, int]
    character_histogram: dict[str, int]
    act_histogram: dict[str, int]
    boss_histogram: dict[str, int]
    planner_strategy_histogram: dict[str, int]
    route_profile_histogram: dict[str, int]
    route_reason_tag_histogram: dict[str, int]
    strategic_coverage: dict[str, int | float | None]
    split_counts: dict[str, int]


def discover_combat_outcome_paths(sources: Sequence[str | Path]) -> list[Path]:
    return discover_predictor_combat_outcome_paths(sources)


def resolve_predictor_examples_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    if source_path.is_dir():
        examples_path = source_path / PREDICTOR_EXAMPLES_FILENAME
        if not examples_path.exists():
            raise FileNotFoundError(f"Predictor examples file not found: {examples_path}")
        return examples_path
    if not source_path.exists():
        raise FileNotFoundError(f"Predictor examples source does not exist: {source_path}")
    return source_path


def load_predictor_examples(source: str | Path) -> list[PredictorExample]:
    examples_path = resolve_predictor_examples_path(source)
    examples: list[PredictorExample] = []
    with examples_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                examples.append(PredictorExample.from_dict(payload))
            except Exception as exc:  # pragma: no cover - error path
                raise ValueError(f"Failed to parse predictor example at {examples_path}:{line_number}") from exc
    return examples


def extract_predictor_dataset(
    sources: Sequence[str | Path],
    *,
    output_dir: str | Path,
    replace_existing: bool = False,
    split_seed: int = 0,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    split_group_by: str = "session_run",
) -> DatasetExtractionReport:
    output_path = Path(output_dir).expanduser().resolve()
    report = build_dataset_from_manifest(
        DatasetManifest(
            dataset_name=output_path.name,
            dataset_kind="predictor_combat_outcomes",
            sources=[
                DatasetSourceSpec(
                    path=str(Path(source).expanduser().resolve()),
                    source_kind="combat_outcomes",
                )
                for source in sources
            ],
            split=DatasetSplitSpec(
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                seed=split_seed,
                group_by=split_group_by,
            ),
            output=DatasetOutputSpec(export_csv=True, include_top_level_records=True, write_split_files=True),
            metadata={"builder": "predict.extract_predictor_dataset"},
        ),
        output_dir=output_path,
        replace_existing=replace_existing,
    )
    return _predictor_extraction_report(report, sources=sources)


def _predictor_extraction_report(
    report: BuiltDatasetReport,
    *,
    sources: Sequence[str | Path],
) -> DatasetExtractionReport:
    summary_payload = json.loads(report.summary_path.read_text(encoding="utf-8"))
    source_paths = tuple(Path(source).expanduser().resolve() for source in sources)
    combat_outcome_paths = tuple(Path(path) for path in summary_payload["lineage"]["resolved_source_files"])
    return DatasetExtractionReport(
        output_dir=report.output_dir,
        examples_path=report.records_path,
        summary_path=report.summary_path,
        manifest_path=report.manifest_path,
        source_paths=source_paths,
        combat_outcome_paths=combat_outcome_paths,
        example_count=report.record_count,
        feature_count=report.feature_count,
        outcome_histogram=dict(summary_payload.get("outcome_histogram", {})),
        character_histogram=dict(summary_payload.get("character_histogram", {})),
        act_histogram=dict(summary_payload.get("act_histogram", {})),
        boss_histogram=dict(summary_payload.get("boss_histogram", {})),
        planner_strategy_histogram=dict(summary_payload.get("planner_strategy_histogram", {})),
        route_profile_histogram=dict(summary_payload.get("route_profile_histogram", {})),
        route_reason_tag_histogram=dict(summary_payload.get("route_reason_tag_histogram", {})),
        strategic_coverage=dict(summary_payload.get("strategic_coverage", {})),
        split_counts=dict(summary_payload.get("split", {}).get("split_counts", {})),
    )
