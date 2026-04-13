from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

REGISTRY_SCHEMA_VERSION = 1
REGISTRY_MANIFEST_FILENAME = "registry-manifest.json"
REGISTRY_ALIASES_FILENAME = "aliases.json"
REGISTRY_ALIAS_HISTORY_FILENAME = "alias-history.jsonl"
REGISTRY_EXPERIMENTS_DIRNAME = "experiments"
REGISTRY_REPORTS_DIRNAME = "reports"
REGISTRY_LEADERBOARD_SUMMARY_FILENAME = "leaderboard-summary.json"
REGISTRY_COMPARE_SUMMARY_FILENAME = "compare-summary.json"


@dataclass(frozen=True)
class RegistryInitReport:
    root_dir: Path
    manifest_path: Path
    experiments_dir: Path
    reports_dir: Path
    aliases_path: Path
    alias_history_path: Path


@dataclass(frozen=True)
class RegistryRegistrationReport:
    root_dir: Path
    experiment_id: str
    family: str
    artifact_kind: str
    display_name: str
    entry_path: Path
    source_summary_path: Path
    primary_metric_name: str | None
    primary_metric_value: float | None


@dataclass(frozen=True)
class RegistryAliasReport:
    root_dir: Path
    alias_name: str
    experiment_id: str
    artifact_path: Path | None
    artifact_path_key: str | None
    aliases_path: Path
    alias_history_path: Path


@dataclass(frozen=True)
class RegistryLeaderboardReport:
    root_dir: Path
    output_dir: Path
    summary_path: Path
    markdown_path: Path
    row_count: int


@dataclass(frozen=True)
class RegistryCompareReport:
    root_dir: Path
    output_dir: Path
    summary_path: Path
    markdown_path: Path
    compared_count: int


def initialize_registry(
    root: str | Path,
    *,
    registry_name: str = "sts2-rl-local",
    replace_existing: bool = False,
) -> RegistryInitReport:
    root_dir = Path(root).expanduser().resolve()
    if root_dir.exists():
        if not replace_existing:
            raise FileExistsError(f"Registry root already exists: {root_dir}")
        shutil.rmtree(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = root_dir / REGISTRY_EXPERIMENTS_DIRNAME
    reports_dir = root_dir / REGISTRY_REPORTS_DIRNAME
    experiments_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root_dir / REGISTRY_MANIFEST_FILENAME
    aliases_path = root_dir / REGISTRY_ALIASES_FILENAME
    alias_history_path = root_dir / REGISTRY_ALIAS_HISTORY_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": REGISTRY_SCHEMA_VERSION,
                "registry_name": registry_name,
                "created_at_utc": _now_utc(),
                "updated_at_utc": _now_utc(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    aliases_path.write_text(json.dumps({"schema_version": REGISTRY_SCHEMA_VERSION, "aliases": {}}, indent=2), encoding="utf-8")
    alias_history_path.write_text("", encoding="utf-8")
    return RegistryInitReport(
        root_dir=root_dir,
        manifest_path=manifest_path,
        experiments_dir=experiments_dir,
        reports_dir=reports_dir,
        aliases_path=aliases_path,
        alias_history_path=alias_history_path,
    )


def register_experiment(
    root: str | Path,
    *,
    source: str | Path,
    experiment_id: str | None = None,
    tags: Sequence[str] | None = None,
    notes: str | None = None,
    aliases: Sequence[str] | None = None,
    replace_existing: bool = False,
) -> RegistryRegistrationReport:
    paths = _ensure_registry(root)
    inspection = inspect_artifact_source(source)
    resolved_experiment_id = experiment_id or _default_experiment_id(
        family=inspection["family"],
        display_name=inspection["display_name"],
        source_summary_path=Path(inspection["source_summary_path"]),
    )
    normalized_id = _normalize_experiment_id(resolved_experiment_id)
    entry_path = paths["experiments_dir"] / f"{normalized_id}.json"
    if entry_path.exists() and not replace_existing:
        raise FileExistsError(f"Registry entry already exists: {entry_path}")
    if entry_path.exists() and replace_existing:
        existing_aliases = [alias for alias, payload in load_registry_aliases(root).items() if payload["experiment_id"] == normalized_id]
        if existing_aliases:
            raise ValueError(
                "Cannot replace a registered experiment that is referenced by aliases: "
                + ", ".join(sorted(existing_aliases))
            )

    entry_payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "experiment_id": normalized_id,
        "family": inspection["family"],
        "artifact_kind": inspection["artifact_kind"],
        "display_name": inspection["display_name"],
        "registered_at_utc": _now_utc(),
        "source_path": str(Path(source).expanduser().resolve()),
        "source_summary_path": inspection["source_summary_path"],
        "output_dir": inspection["output_dir"],
        "tags": sorted(set([*inspection.get("tags", []), *(_normalize_tags(tags) if tags is not None else [])])),
        "notes": notes,
        "config": inspection["config"],
        "config_fingerprint": _stable_hash(inspection["config"]),
        "lineage": inspection["lineage"],
        "references": inspection["references"],
        "artifact_paths": inspection["artifact_paths"],
        "metrics": inspection["metrics"],
        "metadata": inspection["metadata"],
    }
    entry_path.write_text(json.dumps(entry_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _touch_manifest(paths["manifest_path"])
    if aliases:
        for alias_name in aliases:
            set_registry_alias(
                root,
                alias_name=alias_name,
                experiment_id=normalized_id,
                artifact_path_key=_default_alias_artifact_key(entry_payload),
                reason="register",
            )
    primary = dict(entry_payload["metrics"].get("primary") or {})
    return RegistryRegistrationReport(
        root_dir=paths["root_dir"],
        experiment_id=normalized_id,
        family=str(entry_payload["family"]),
        artifact_kind=str(entry_payload["artifact_kind"]),
        display_name=str(entry_payload["display_name"]),
        entry_path=entry_path,
        source_summary_path=Path(str(entry_payload["source_summary_path"])),
        primary_metric_name=str(primary["name"]) if primary.get("name") is not None else None,
        primary_metric_value=_float_or_none(primary.get("value")),
    )


def list_registry_experiments(
    root: str | Path,
    *,
    family: str | None = None,
    tag: str | None = None,
    alias: str | None = None,
) -> list[dict[str, Any]]:
    entries = _load_all_entries(root)
    aliases = load_registry_aliases(root)
    alias_to_experiment = {name: payload["experiment_id"] for name, payload in aliases.items()}
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if family is not None and str(entry["family"]) != family:
            continue
        if tag is not None and tag not in entry.get("tags", []):
            continue
        entry_aliases = sorted(name for name, experiment_id in alias_to_experiment.items() if experiment_id == entry["experiment_id"])
        if alias is not None and alias not in entry_aliases:
            continue
        rows.append({**entry, "aliases": entry_aliases})
    return sorted(rows, key=lambda item: (str(item["family"]), str(item["display_name"]), str(item["experiment_id"])))


def get_registry_experiment(root: str | Path, experiment_id_or_alias: str) -> dict[str, Any]:
    experiment_id = resolve_registry_experiment_id(root, experiment_id_or_alias)
    entry_path = _ensure_registry(root)["experiments_dir"] / f"{experiment_id}.json"
    if not entry_path.exists():
        raise FileNotFoundError(f"Registry experiment does not exist: {experiment_id}")
    entry = json.loads(entry_path.read_text(encoding="utf-8"))
    aliases = load_registry_aliases(root)
    entry["aliases"] = sorted(name for name, payload in aliases.items() if payload["experiment_id"] == experiment_id)
    return entry


def resolve_registry_experiment_id(root: str | Path, experiment_id_or_alias: str) -> str:
    candidate = _normalize_experiment_id(experiment_id_or_alias)
    entry_path = _ensure_registry(root)["experiments_dir"] / f"{candidate}.json"
    if entry_path.exists():
        return candidate
    aliases = load_registry_aliases(root)
    alias_payload = aliases.get(experiment_id_or_alias)
    if alias_payload is None:
        raise FileNotFoundError(f"Unknown experiment id or alias: {experiment_id_or_alias}")
    return str(alias_payload["experiment_id"])


def load_registry_aliases(root: str | Path) -> dict[str, dict[str, Any]]:
    aliases_path = _ensure_registry(root)["aliases_path"]
    payload = json.loads(aliases_path.read_text(encoding="utf-8"))
    return {str(name): dict(value) for name, value in payload.get("aliases", {}).items()}


def set_registry_alias(
    root: str | Path,
    *,
    alias_name: str,
    experiment_id: str,
    artifact_path_key: str | None = None,
    reason: str | None = None,
    updated_by: str = "manual",
) -> RegistryAliasReport:
    paths = _ensure_registry(root)
    entry = get_registry_experiment(root, experiment_id)
    normalized_alias = alias_name.strip()
    if not normalized_alias:
        raise ValueError("Alias name must not be empty.")
    artifact_paths = dict(entry.get("artifact_paths", {}))
    selected_key = artifact_path_key
    if selected_key is not None and selected_key not in artifact_paths:
        raise ValueError(f"Unknown artifact_path_key '{selected_key}' for experiment {entry['experiment_id']}.")
    selected_artifact_path = None if selected_key is None else Path(str(artifact_paths[selected_key])).expanduser().resolve()

    aliases_payload = json.loads(paths["aliases_path"].read_text(encoding="utf-8"))
    aliases = dict(aliases_payload.get("aliases", {}))
    alias_record = {
        "alias_name": normalized_alias,
        "experiment_id": str(entry["experiment_id"]),
        "artifact_path_key": selected_key,
        "artifact_path": None if selected_artifact_path is None else str(selected_artifact_path),
        "family": entry["family"],
        "artifact_kind": entry["artifact_kind"],
        "updated_at_utc": _now_utc(),
        "updated_by": updated_by,
        "reason": reason,
        "primary_metric": entry.get("metrics", {}).get("primary"),
    }
    aliases[normalized_alias] = alias_record
    aliases_payload["schema_version"] = REGISTRY_SCHEMA_VERSION
    aliases_payload["aliases"] = aliases
    paths["aliases_path"].write_text(json.dumps(aliases_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with paths["alias_history_path"].open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(alias_record, ensure_ascii=False))
        handle.write("\n")
    _touch_manifest(paths["manifest_path"])
    return RegistryAliasReport(
        root_dir=paths["root_dir"],
        alias_name=normalized_alias,
        experiment_id=str(entry["experiment_id"]),
        artifact_path=selected_artifact_path,
        artifact_path_key=selected_key,
        aliases_path=paths["aliases_path"],
        alias_history_path=paths["alias_history_path"],
    )


def build_registry_leaderboard(
    root: str | Path,
    *,
    output_root: str | Path | None = None,
    session_name: str | None = None,
    family: str | None = None,
    tag: str | None = None,
    benchmark_suite_name: str | None = None,
) -> RegistryLeaderboardReport:
    paths = _ensure_registry(root)
    output_dir = _prepare_report_dir(
        root=paths["reports_dir"] if output_root is None else Path(output_root).expanduser().resolve(),
        session_name=session_name or _default_report_session_name("leaderboard"),
    )
    rows = list_registry_experiments(root, family=family, tag=tag)
    leaderboard_rows = [_leaderboard_row(entry) for entry in rows]
    if benchmark_suite_name is not None:
        leaderboard_rows = [
            row
            for row in leaderboard_rows
            if row.get("benchmark_suite_name") == benchmark_suite_name
            or row.get("primary_metric", {}).get("source_benchmark_suite_name") == benchmark_suite_name
        ]
    leaderboard_rows = _sorted_leaderboard_rows(leaderboard_rows)
    for index, row in enumerate(leaderboard_rows, start=1):
        row["rank"] = index
    summary_payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "report_kind": "registry_leaderboard",
        "generated_at_utc": _now_utc(),
        "registry_root": str(paths["root_dir"]),
        "filters": {
            "family": family,
            "tag": tag,
            "benchmark_suite_name": benchmark_suite_name,
        },
        "row_count": len(leaderboard_rows),
        "rows": leaderboard_rows,
    }
    summary_path = output_dir / REGISTRY_LEADERBOARD_SUMMARY_FILENAME
    markdown_path = output_dir / "leaderboard-summary.md"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_leaderboard_markdown(summary_payload), encoding="utf-8")
    return RegistryLeaderboardReport(
        root_dir=paths["root_dir"],
        output_dir=output_dir,
        summary_path=summary_path,
        markdown_path=markdown_path,
        row_count=len(leaderboard_rows),
    )


def compare_registry_experiments(
    root: str | Path,
    *,
    experiment_ids: Sequence[str],
    output_root: str | Path | None = None,
    session_name: str | None = None,
) -> RegistryCompareReport:
    if len(experiment_ids) < 2:
        raise ValueError("Compare reports require at least two experiments.")
    paths = _ensure_registry(root)
    entries = [get_registry_experiment(root, experiment_id) for experiment_id in experiment_ids]
    output_dir = _prepare_report_dir(
        root=paths["reports_dir"] if output_root is None else Path(output_root).expanduser().resolve(),
        session_name=session_name or _default_report_session_name("compare"),
    )
    flattened = {entry["experiment_id"]: _flatten_metrics(entry.get("metrics", {}).get("snapshot", {})) for entry in entries}
    metric_names = sorted({name for payload in flattened.values() for name in payload})
    metric_rows: list[dict[str, Any]] = []
    for metric_name in metric_names:
        values = {experiment_id: payload.get(metric_name) for experiment_id, payload in flattened.items()}
        metric_rows.append({"metric_name": metric_name, "values": values})
    summary_payload = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "report_kind": "registry_compare",
        "generated_at_utc": _now_utc(),
        "registry_root": str(paths["root_dir"]),
        "compared_count": len(entries),
        "experiments": [
            {
                "experiment_id": entry["experiment_id"],
                "family": entry["family"],
                "artifact_kind": entry["artifact_kind"],
                "display_name": entry["display_name"],
                "aliases": entry.get("aliases", []),
                "primary_metric": entry.get("metrics", {}).get("primary"),
                "lineage": entry.get("lineage", {}),
                "references": entry.get("references", {}),
                "tags": entry.get("tags", []),
            }
            for entry in entries
        ],
        "metric_rows": metric_rows,
    }
    summary_path = output_dir / REGISTRY_COMPARE_SUMMARY_FILENAME
    markdown_path = output_dir / "compare-summary.md"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_compare_markdown(summary_payload), encoding="utf-8")
    return RegistryCompareReport(
        root_dir=paths["root_dir"],
        output_dir=output_dir,
        summary_path=summary_path,
        markdown_path=markdown_path,
        compared_count=len(entries),
    )


def inspect_artifact_source(source: str | Path) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    summary_path = _resolve_source_summary_path(source_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    source_dir = summary_path.parent
    if payload.get("dataset_kind") is not None:
        return _inspect_dataset(summary_path, payload)
    if str(payload.get("report_kind", "")).startswith("predictor_"):
        return _inspect_predictor_report(summary_path, payload)
    if payload.get("suite_name") is not None and isinstance(payload.get("cases"), list):
        return _inspect_benchmark_suite(summary_path, payload)
    if payload.get("case_id") is not None and payload.get("mode") is not None and payload.get("primary_metric") is not None:
        return _inspect_benchmark_case(summary_path, payload)
    if payload.get("algorithm") == "offline_cql" and payload.get("best_checkpoint_path") is not None:
        return _inspect_offline_cql_training(summary_path, payload)
    if payload.get("algorithm") == "offline_cql" and payload.get("checkpoint_metadata") is not None:
        return _inspect_policy_evaluation(summary_path, payload, family="offline_cql")
    if payload.get("stage_count") is not None and payload.get("best_checkpoint_path") is not None:
        return _inspect_behavior_cloning_training(summary_path, payload)
    if payload.get("algorithm") == "strategic_pretrain" and payload.get("best_checkpoint_path") is not None:
        return _inspect_strategic_pretrain_training(summary_path, payload)
    if payload.get("algorithm") == "strategic_finetune" and payload.get("best_checkpoint_path") is not None:
        return _inspect_strategic_finetune_training(summary_path, payload)
    if payload.get("checkpoint_metadata") is not None and payload.get("checkpoint_path") is not None:
        algorithm = str(dict(payload.get("checkpoint_metadata", {})).get("algorithm", "evaluation"))
        family = "combat_dqn" if algorithm == "dqn" else algorithm
        return _inspect_policy_evaluation(summary_path, payload, family=family)
    if payload.get("learning_metrics") is not None and payload.get("checkpoint_path") is not None:
        return _inspect_combat_dqn_training(summary_path, payload)
    if payload.get("model_path") is not None and payload.get("feature_names") is not None:
        return _inspect_predictor_model(summary_path, payload)
    if payload.get("model_path") is not None and payload.get("target_stats") is not None:
        return _inspect_predictor_training(summary_path, payload)
    raise ValueError(f"Unsupported artifact source: {source_dir}")


def _inspect_dataset(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    dataset_kind = str(payload.get("dataset_kind", "dataset"))
    record_count = _float_or_none(payload.get("record_count"))
    return _entry_payload(
        family="dataset",
        artifact_kind="dataset",
        display_name=str(payload.get("dataset_name", summary_path.parent.name)),
        source_summary_path=summary_path,
        output_dir=summary_path.parent,
        tags=["dataset", dataset_kind],
        config={
            "dataset_kind": dataset_kind,
            "split": payload.get("split"),
            "exports": payload.get("exports"),
        },
        lineage={
            "source_paths": _collect_strings(
                [
                    *dict(payload.get("lineage", {})).get("source_paths", []),
                    *dict(payload.get("lineage", {})).get("resolved_source_files", []),
                ]
            ),
            "dataset_paths": [str(summary_path.parent)],
            "dataset_lineage": payload.get("lineage"),
            "benchmark_summary_paths": [],
        },
        references={
            "dataset_path": str(summary_path.parent),
            "records_path": payload.get("records_path"),
            "summary_path": str(summary_path),
        },
        artifact_paths={
            "dataset_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
            **({"records_path": str(Path(str(payload["records_path"])).expanduser().resolve())} if payload.get("records_path") else {}),
        },
        metrics=_build_metrics(
            primary_name="record_count",
            primary_value=record_count,
            higher_is_better=True,
            snapshot={
                "record_count": record_count,
                "feature_count": _float_or_none(payload.get("feature_count")),
                "source_file_count": _float_or_none(payload.get("source_file_count")),
                "filtered_out_count": _float_or_none(payload.get("filtered_out_count")),
                "supported_transition_count": _float_or_none(payload.get("supported_transition_count")),
            },
        ),
        metadata={
            "dataset_kind": dataset_kind,
            "character_histogram": payload.get("character_histogram"),
            "outcome_histogram": payload.get("outcome_histogram"),
        },
    )


def _inspect_behavior_cloning_training(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    benchmark_payload = _load_optional_json(payload.get("benchmark_summary_path"))
    live_eval_payload = _load_optional_json(payload.get("live_eval_summary_path"))
    benchmark_primary = _benchmark_primary_metric(benchmark_payload)
    validation = dict(payload.get("validation", {}))
    top1 = _float_or_none(dict(validation.get("top_k_accuracy", {})).get("1"))
    loss = _float_or_none(validation.get("loss"))
    if benchmark_primary is not None:
        primary_name = benchmark_primary["name"]
        primary_value = benchmark_primary["value"]
        higher_is_better = True
    elif _combat_win_rate_from_eval(live_eval_payload) is not None:
        primary_name = "live_eval_combat_win_rate"
        primary_value = _combat_win_rate_from_eval(live_eval_payload)
        higher_is_better = True
    elif top1 is not None:
        primary_name = "validation_top_1_accuracy"
        primary_value = top1
        higher_is_better = True
    else:
        primary_name = "validation_loss"
        primary_value = loss
        higher_is_better = False
    return _entry_payload(
        family="behavior_cloning",
        artifact_kind="training_run",
        display_name=summary_path.parent.name,
        source_summary_path=summary_path,
        output_dir=summary_path.parent,
        tags=["policy", "behavior_cloning"],
        config=payload.get("config", {}),
        lineage={
            "source_paths": _collect_strings(_dataset_lineage_source_paths(payload.get("dataset_lineage"))),
            "dataset_paths": _collect_strings([payload.get("dataset_path")]),
            "dataset_lineage": payload.get("dataset_lineage"),
            "benchmark_summary_paths": _collect_strings([payload.get("benchmark_summary_path")]),
            "live_eval_summary_paths": _collect_strings([payload.get("live_eval_summary_path")]),
        },
        references={
            "dataset_path": payload.get("dataset_path"),
            "benchmark_summary_path": payload.get("benchmark_summary_path"),
            "live_eval_summary_path": payload.get("live_eval_summary_path"),
            "summary_path": str(summary_path),
        },
        artifact_paths={
            "output_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
            **_optional_path_map(payload, "checkpoint_path", "best_checkpoint_path", "metrics_path"),
        },
        metrics=_build_metrics(
            primary_name=primary_name,
            primary_value=primary_value,
            higher_is_better=higher_is_better,
            snapshot={
                "best_epoch": _float_or_none(payload.get("best_epoch")),
                "example_count": _float_or_none(payload.get("example_count")),
                "train_loss": _float_or_none(dict(payload.get("train", {})).get("loss")),
                "validation_loss": loss,
                "validation_top_1_accuracy": top1,
                "test_top_1_accuracy": _float_or_none(dict(dict(payload.get("test", {})).get("top_k_accuracy", {})).get("1")),
                "benchmark_primary_metric": None if benchmark_primary is None else benchmark_primary["value"],
                "live_eval_combat_win_rate": _combat_win_rate_from_eval(live_eval_payload),
            },
            benchmark_payload=benchmark_payload,
        ),
        metadata={
            "split_strategy": payload.get("split_strategy"),
            "feature_count": payload.get("feature_count"),
            "stage_count": payload.get("stage_count"),
        },
    )


def _inspect_strategic_pretrain_training(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    validation = dict(payload.get("validation", {}))
    ranking = dict(validation.get("candidate_choice", {}))
    value_metrics = dict(validation.get("value", {}))
    top1 = _float_or_none(dict(ranking.get("top_k_accuracy", {})).get("1"))
    loss = _float_or_none(ranking.get("loss"))
    value_accuracy = _float_or_none(value_metrics.get("accuracy"))
    if top1 is not None:
        primary_name = "validation_top_1_accuracy"
        primary_value = top1
        higher_is_better = True
    elif value_accuracy is not None:
        primary_name = "validation_value_accuracy"
        primary_value = value_accuracy
        higher_is_better = True
    else:
        primary_name = "best_objective"
        primary_value = _float_or_none(payload.get("best_objective")) or loss
        higher_is_better = False
    return _entry_payload(
        family="strategic_pretrain",
        artifact_kind="training_run",
        display_name=summary_path.parent.name,
        source_summary_path=summary_path,
        output_dir=summary_path.parent,
        tags=["policy", "strategic_pretrain"],
        config=payload.get("config", {}),
        lineage={
            "source_paths": _collect_strings(_dataset_lineage_source_paths(payload.get("dataset_lineage"))),
            "dataset_paths": _collect_strings([payload.get("dataset_path")]),
            "dataset_lineage": payload.get("dataset_lineage"),
            "benchmark_summary_paths": [],
        },
        references={
            "dataset_path": payload.get("dataset_path"),
            "summary_path": str(summary_path),
        },
        artifact_paths={
            "output_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
            **_optional_path_map(payload, "checkpoint_path", "best_checkpoint_path", "metrics_path"),
        },
        metrics=_build_metrics(
            primary_name=primary_name,
            primary_value=primary_value,
            higher_is_better=higher_is_better,
            snapshot={
                "best_epoch": _float_or_none(payload.get("best_epoch")),
                "example_count": _float_or_none(payload.get("example_count")),
                "validation_candidate_loss": loss,
                "validation_top_1_accuracy": top1,
                "validation_value_accuracy": value_accuracy,
            },
        ),
        metadata={
            "decision_type_count": payload.get("decision_type_count"),
            "feature_count": payload.get("feature_count"),
        },
    )


def _inspect_strategic_finetune_training(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    validation = dict(payload.get("validation", {}))
    source_metrics = dict(validation.get("source_metrics", {}))
    runtime_validation = dict(source_metrics.get("runtime", {}))
    runtime_ranking = dict(runtime_validation.get("candidate_choice", {}))
    runtime_value = dict(runtime_validation.get("value", {}))
    runtime_top1 = _float_or_none(dict(runtime_ranking.get("top_k_accuracy", {})).get("1"))
    runtime_value_accuracy = _float_or_none(runtime_value.get("accuracy"))
    runtime_loss = _float_or_none(runtime_ranking.get("loss"))
    if runtime_top1 is not None:
        primary_name = "validation_runtime_top_1_accuracy"
        primary_value = runtime_top1
        higher_is_better = True
    elif runtime_value_accuracy is not None:
        primary_name = "validation_runtime_value_accuracy"
        primary_value = runtime_value_accuracy
        higher_is_better = True
    else:
        primary_name = "best_objective"
        primary_value = _float_or_none(payload.get("best_objective")) or runtime_loss
        higher_is_better = False
    runtime_dataset = dict(payload.get("runtime_dataset", {}))
    public_dataset = payload.get("public_dataset")
    return _entry_payload(
        family="strategic_finetune",
        artifact_kind="training_run",
        display_name=summary_path.parent.name,
        source_summary_path=summary_path,
        output_dir=summary_path.parent,
        tags=["policy", "strategic_finetune"],
        config=payload.get("config", {}),
        lineage={
            "source_paths": _collect_strings(
                [
                    *_dataset_lineage_source_paths(runtime_dataset.get("dataset_lineage")),
                    *_dataset_lineage_source_paths(None if not isinstance(public_dataset, dict) else public_dataset.get("dataset_lineage")),
                ]
            ),
            "dataset_paths": _collect_strings([payload.get("runtime_dataset_path"), payload.get("public_dataset_path")]),
            "dataset_lineage": {
                "runtime": runtime_dataset.get("dataset_lineage"),
                "public": None if not isinstance(public_dataset, dict) else public_dataset.get("dataset_lineage"),
            },
            "benchmark_summary_paths": [],
        },
        references={
            "runtime_dataset_path": payload.get("runtime_dataset_path"),
            "public_dataset_path": payload.get("public_dataset_path"),
            "warmstart_checkpoint_path": payload.get("warmstart_checkpoint_path"),
            "summary_path": str(summary_path),
        },
        artifact_paths={
            "output_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
            **_optional_path_map(payload, "checkpoint_path", "best_checkpoint_path", "warmstart_checkpoint_path", "metrics_path"),
        },
        metrics=_build_metrics(
            primary_name=primary_name,
            primary_value=primary_value,
            higher_is_better=higher_is_better,
            snapshot={
                "best_epoch": _float_or_none(payload.get("best_epoch")),
                "example_count": _float_or_none(payload.get("example_count")),
                "runtime_example_count": _float_or_none(payload.get("runtime_example_count")),
                "public_example_count": _float_or_none(payload.get("public_example_count")),
                "validation_runtime_loss": runtime_loss,
                "validation_runtime_top_1_accuracy": runtime_top1,
                "validation_runtime_value_accuracy": runtime_value_accuracy,
            },
        ),
        metadata={
            "decision_type_count": payload.get("decision_type_count"),
            "feature_count": payload.get("feature_count"),
            "schedule": payload.get("schedule"),
            "transferred_modules": payload.get("transferred_modules"),
        },
    )


def _inspect_offline_cql_training(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    benchmark_payload = _load_optional_json(payload.get("benchmark_summary_path"))
    live_eval_payload = _load_optional_json(payload.get("live_eval_summary_path"))
    validation_metrics = dict(payload.get("validation_metrics", {}))
    validation_loss = _float_or_none(validation_metrics.get("mean_loss"))
    benchmark_primary = _benchmark_primary_metric(benchmark_payload)
    if benchmark_primary is not None:
        primary_name = benchmark_primary["name"]
        primary_value = benchmark_primary["value"]
        higher_is_better = True
    elif _combat_win_rate_from_eval(live_eval_payload) is not None:
        primary_name = "live_eval_combat_win_rate"
        primary_value = _combat_win_rate_from_eval(live_eval_payload)
        higher_is_better = True
    else:
        primary_name = "validation_mean_loss"
        primary_value = validation_loss
        higher_is_better = False
    dataset_summary = payload.get("dataset_summary")
    return _entry_payload(
        family="offline_cql",
        artifact_kind="training_run",
        display_name=summary_path.parent.name,
        source_summary_path=summary_path,
        output_dir=summary_path.parent,
        tags=["policy", "offline_cql"],
        config=payload.get("config", {}),
        lineage={
            "source_paths": _collect_strings(_dataset_lineage_source_paths(dataset_summary.get("lineage") if isinstance(dataset_summary, dict) else None)),
            "dataset_paths": _collect_strings([payload.get("dataset_path")]),
            "dataset_lineage": None if not isinstance(dataset_summary, dict) else dataset_summary.get("lineage"),
            "benchmark_summary_paths": _collect_strings([payload.get("benchmark_summary_path")]),
            "live_eval_summary_paths": _collect_strings([payload.get("live_eval_summary_path")]),
        },
        references={
            "dataset_path": payload.get("dataset_path"),
            "benchmark_summary_path": payload.get("benchmark_summary_path"),
            "live_eval_summary_path": payload.get("live_eval_summary_path"),
            "summary_path": str(summary_path),
        },
        artifact_paths={
            "output_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
            **_optional_path_map(payload, "checkpoint_path", "best_checkpoint_path", "warmstart_checkpoint_path", "metrics_path"),
        },
        metrics=_build_metrics(
            primary_name=primary_name,
            primary_value=primary_value,
            higher_is_better=higher_is_better,
            snapshot={
                "best_epoch": _float_or_none(payload.get("best_epoch")),
                "train_mean_loss": _float_or_none(dict(payload.get("train_metrics", {})).get("mean_loss")),
                "validation_mean_loss": validation_loss,
                "test_mean_loss": _float_or_none(dict(payload.get("test_metrics", {})).get("mean_loss")),
                "validation_support_coverage": _float_or_none(validation_metrics.get("support_coverage")),
                "benchmark_primary_metric": None if benchmark_primary is None else benchmark_primary["value"],
                "live_eval_combat_win_rate": _combat_win_rate_from_eval(live_eval_payload),
            },
            benchmark_payload=benchmark_payload,
        ),
        metadata={
            "algorithm": payload.get("algorithm"),
            "split_strategy": payload.get("split_strategy"),
            "dropped_records": payload.get("dropped_records"),
        },
    )


def _inspect_combat_dqn_training(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    checkpoint_metadata = dict(payload.get("checkpoint_metadata", {}))
    checkpoint_algorithm = str(checkpoint_metadata.get("algorithm", "dqn"))
    snapshot = {
        "total_reward": _float_or_none(payload.get("total_reward")),
        "completed_combat_count": _float_or_none(payload.get("completed_combat_count")),
        "combat_win_rate": _float_or_none(dict(payload.get("combat_performance", {})).get("combat_win_rate")),
        "learning_mean_loss": _float_or_none(dict(payload.get("learning_metrics", {})).get("mean_loss")),
        "replay_utilization": _float_or_none(dict(payload.get("replay_metrics", {})).get("utilization")),
    }
    primary_value = snapshot["combat_win_rate"] if snapshot["combat_win_rate"] is not None else snapshot["total_reward"]
    return _entry_payload(
        family="combat_dqn",
        artifact_kind="training_run",
        display_name=summary_path.parent.name,
        source_summary_path=summary_path,
        output_dir=summary_path.parent,
        tags=["policy", checkpoint_algorithm],
        config=checkpoint_metadata.get("config", {}),
        lineage={
            "source_paths": [],
            "dataset_paths": [],
            "dataset_lineage": None,
            "benchmark_summary_paths": [],
            "live_eval_summary_paths": [],
        },
        references={"summary_path": str(summary_path)},
        artifact_paths={
            "output_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
            **_optional_path_map(payload, "checkpoint_path", "best_checkpoint_path", "combat_outcomes_path", "log_path"),
        },
        metrics=_build_metrics(
            primary_name="combat_win_rate" if snapshot["combat_win_rate"] is not None else "total_reward",
            primary_value=primary_value,
            higher_is_better=True,
            snapshot=snapshot,
        ),
        metadata={"checkpoint_algorithm": checkpoint_algorithm},
    )


def _inspect_policy_evaluation(summary_path: Path, payload: dict[str, Any], *, family: str) -> dict[str, Any]:
    checkpoint_metadata = dict(payload.get("checkpoint_metadata", {}))
    combat_performance = dict(payload.get("combat_performance", {}))
    snapshot = {
        "total_reward": _float_or_none(payload.get("total_reward")),
        "completed_run_count": _float_or_none(payload.get("completed_run_count")),
        "completed_combat_count": _float_or_none(payload.get("completed_combat_count")),
        "combat_win_rate": _float_or_none(combat_performance.get("combat_win_rate")),
        "reward_per_combat": _float_or_none(combat_performance.get("reward_per_combat")),
    }
    primary_value = snapshot["combat_win_rate"] if snapshot["combat_win_rate"] is not None else snapshot["total_reward"]
    return _entry_payload(
        family=family,
        artifact_kind="evaluation",
        display_name=summary_path.parent.name,
        source_summary_path=summary_path,
        output_dir=summary_path.parent,
        tags=["evaluation", family],
        config=checkpoint_metadata.get("config", {}),
        lineage={
            "source_paths": [],
            "dataset_paths": [],
            "dataset_lineage": None,
            "benchmark_summary_paths": [],
            "live_eval_summary_paths": [],
        },
        references={"summary_path": str(summary_path), "checkpoint_path": payload.get("checkpoint_path")},
        artifact_paths={
            "output_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
            **_optional_path_map(payload, "checkpoint_path", "log_path", "combat_outcomes_path"),
        },
        metrics=_build_metrics(
            primary_name="combat_win_rate" if snapshot["combat_win_rate"] is not None else "total_reward",
            primary_value=primary_value,
            higher_is_better=True,
            snapshot=snapshot,
        ),
        metadata={"checkpoint_metadata": checkpoint_metadata},
    )


def _inspect_predictor_training(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    validation = dict(payload.get("validation", {}))
    objective = _float_or_none(validation.get("objective"))
    reward_rmse = _float_or_none(dict(validation.get("reward", {})).get("rmse"))
    return _entry_payload(
        family="predictor",
        artifact_kind="predictor_model",
        display_name=summary_path.parent.name,
        source_summary_path=summary_path,
        output_dir=summary_path.parent,
        tags=["predictor", "training"],
        config=payload.get("config", {}),
        lineage={
            "source_paths": [],
            "dataset_paths": _collect_strings([payload.get("dataset_path")]),
            "dataset_lineage": None,
            "benchmark_summary_paths": [],
            "live_eval_summary_paths": [],
        },
        references={"dataset_path": payload.get("dataset_path"), "summary_path": str(summary_path)},
        artifact_paths={
            "output_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
            **_optional_path_map(payload, "model_path", "metrics_path"),
        },
        metrics=_build_metrics(
            primary_name="validation_objective",
            primary_value=objective,
            higher_is_better=False,
            snapshot={
                "example_count": _float_or_none(payload.get("example_count")),
                "best_epoch": _float_or_none(payload.get("best_epoch")),
                "validation_objective": objective,
                "validation_outcome_loss": _float_or_none(dict(validation.get("outcome", {})).get("loss")),
                "validation_reward_rmse": reward_rmse,
                "validation_damage_rmse": _float_or_none(dict(validation.get("damage_delta", {})).get("rmse")),
            },
        ),
        metadata={"split_strategy": payload.get("split_strategy")},
    )


def _inspect_predictor_model(model_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(payload.get("metadata", {}))
    validation_metrics = dict(metadata.get("validation_metrics", {}))
    objective = _float_or_none(validation_metrics.get("objective"))
    return _entry_payload(
        family="predictor",
        artifact_kind="predictor_model",
        display_name=model_path.parent.name,
        source_summary_path=model_path,
        output_dir=model_path.parent,
        tags=["predictor", "model"],
        config={k: payload.get(k) for k in ("schema_version", "feature_names", "feature_means", "feature_stds")},
        lineage={
            "source_paths": [],
            "dataset_paths": _collect_strings([metadata.get("dataset_path")]),
            "dataset_lineage": None,
            "benchmark_summary_paths": [],
            "live_eval_summary_paths": [],
        },
        references={"dataset_path": metadata.get("dataset_path"), "summary_path": str(model_path)},
        artifact_paths={"model_path": str(model_path)},
        metrics=_build_metrics(
            primary_name="validation_objective",
            primary_value=objective,
            higher_is_better=False,
            snapshot={
                "feature_count": _float_or_none(len(payload.get("feature_names", []))),
                "validation_objective": objective,
                "validation_outcome_loss": _float_or_none(dict(validation_metrics.get("outcome", {})).get("loss")),
                "validation_reward_rmse": _float_or_none(dict(validation_metrics.get("reward", {})).get("rmse")),
                "validation_damage_rmse": _float_or_none(dict(validation_metrics.get("damage_delta", {})).get("rmse")),
            },
        ),
        metadata={"model_metadata": metadata},
    )


def _inspect_predictor_report(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    report_kind = str(payload.get("report_kind", "predictor_report"))
    promotion = dict(payload.get("promotion", {}))
    if report_kind == "predictor_calibration":
        primary_name = "outcome_ece"
        primary_value = _float_or_none(dict(dict(payload.get("overall", {})).get("outcome_win_probability", {})).get("ece"))
        higher_is_better = False
    elif report_kind == "predictor_ranking":
        primary_name = "reward_pairwise_accuracy"
        primary_value = _float_or_none(dict(dict(payload.get("overall", {})).get("expected_reward", {})).get("pairwise_accuracy"))
        higher_is_better = True
    else:
        primary_name = "promotion_candidate_count"
        primary_value = _float_or_none(promotion.get("promotion_candidate_count"))
        higher_is_better = True
    return _entry_payload(
        family="predictor_report",
        artifact_kind="predictor_report",
        display_name=summary_path.parent.name,
        source_summary_path=summary_path,
        output_dir=summary_path.parent,
        tags=["predictor", "report", report_kind],
        config={
            "report_kind": report_kind,
            "split": payload.get("split"),
            "group_by": payload.get("group_by"),
            "thresholds": payload.get("thresholds"),
        },
        lineage={
            "source_paths": _collect_strings(payload.get("source_paths", [])),
            "dataset_paths": _collect_strings([payload.get("dataset_path")]),
            "dataset_lineage": None,
            "benchmark_summary_paths": _collect_strings(payload.get("source_paths", [])) if report_kind == "predictor_benchmark_compare" else [],
            "live_eval_summary_paths": [],
        },
        references={
            "dataset_path": payload.get("dataset_path"),
            "summary_path": str(summary_path),
            "model_path": payload.get("model_path"),
        },
        artifact_paths={
            "output_dir": str(summary_path.parent),
            "summary_path": str(summary_path),
            **({"model_path": str(Path(str(payload["model_path"])).expanduser().resolve())} if payload.get("model_path") else {}),
        },
        metrics=_build_metrics(
            primary_name=primary_name,
            primary_value=primary_value,
            higher_is_better=higher_is_better,
            snapshot={
                "promotion_passed": 1.0 if promotion.get("passed") else 0.0,
                "promotion_check_count": _float_or_none(promotion.get("check_count")),
                "example_count": _float_or_none(payload.get("example_count")),
                "group_count": _float_or_none(payload.get("group_count")),
                "compare_case_count": _float_or_none(payload.get("compare_case_count")),
                "promotion_candidate_count": _float_or_none(promotion.get("promotion_candidate_count")),
                "rollback_signal_count": _float_or_none(promotion.get("rollback_signal_count")),
                "outcome_ece": _float_or_none(dict(dict(payload.get("overall", {})).get("outcome_win_probability", {})).get("ece")),
                "reward_pairwise_accuracy": _float_or_none(dict(dict(payload.get("overall", {})).get("expected_reward", {})).get("pairwise_accuracy")),
                "reward_ndcg_at_k": _float_or_none(dict(dict(payload.get("overall", {})).get("expected_reward", {})).get("ndcg_at_k")),
            },
        ),
        metadata={"report_kind": report_kind},
    )


def _inspect_benchmark_suite(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    primary = _benchmark_primary_metric(payload)
    promotion = dict(payload.get("promotion", {}))
    shadow = dict(payload.get("shadow", {}))
    return _entry_payload(
        family="benchmark_suite",
        artifact_kind="benchmark_suite",
        display_name=str(payload.get("suite_name", summary_path.parent.name)),
        source_summary_path=summary_path,
        output_dir=summary_path.parent,
        tags=["benchmark", "suite"],
        config={"stats": payload.get("stats"), "base_url": payload.get("base_url")},
        lineage={
            "source_paths": [],
            "dataset_paths": [],
            "dataset_lineage": None,
            "benchmark_summary_paths": [str(summary_path)],
            "live_eval_summary_paths": [],
        },
        references={"summary_path": str(summary_path), "manifest_path": payload.get("manifest_path")},
        artifact_paths={"suite_dir": str(summary_path.parent), "summary_path": str(summary_path)},
        metrics=_build_metrics(
            primary_name=None if primary is None else primary["name"],
            primary_value=None if primary is None else primary["value"],
            higher_is_better=True,
            snapshot={
                "case_count": _float_or_none(payload.get("case_count")),
                "case_mode_count": _float_or_none(len(payload.get("case_mode_histogram", {}))),
                "primary_metric_mean": None if primary is None else primary["value"],
                "promotion_configured_case_count": _float_or_none(promotion.get("configured_case_count")),
                "promotion_passed_case_count": _float_or_none(promotion.get("passed_case_count")),
                "promotion_failed_case_count": _float_or_none(promotion.get("failed_case_count")),
                "promotion_candidate_count": _float_or_none(promotion.get("promotion_candidate_count")),
                "rollback_signal_count": _float_or_none(promotion.get("rollback_signal_count")),
                "shadow_configured_case_count": _float_or_none(shadow.get("configured_case_count")),
                "shadow_comparable_encounter_count": _float_or_none(shadow.get("comparable_encounter_count")),
                "shadow_candidate_advantage_rate": _float_or_none(
                    dict(shadow.get("candidate_advantage_rate_stats", {})).get("mean")
                ),
            },
            benchmark_payload=payload,
        ),
        metadata={
            "suite_name": payload.get("suite_name"),
            "case_mode_histogram": payload.get("case_mode_histogram"),
            "promotion": promotion,
            "shadow": shadow,
        },
    )


def _inspect_benchmark_case(summary_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    primary_metric = dict(payload.get("primary_metric", {}))
    promotion = dict(payload.get("promotion", {}))
    metrics = dict(payload.get("metrics", {}))
    route_comparison = dict(dict(payload.get("route_diagnostics", {})).get("comparison", {}))
    shadow = dict(payload.get("shadow", {}))
    shadow_delta_metrics = dict(shadow.get("delta_metrics", {}))
    return _entry_payload(
        family="benchmark_case",
        artifact_kind="benchmark_case",
        display_name=str(payload.get("case_id", summary_path.parent.name)),
        source_summary_path=summary_path,
        output_dir=summary_path.parent,
        tags=["benchmark", "case", str(payload.get("mode", "unknown"))],
        config={"mode": payload.get("mode"), "scenario": payload.get("scenario"), "config": payload.get("config")},
        lineage={
            "source_paths": [],
            "dataset_paths": [],
            "dataset_lineage": None,
            "benchmark_summary_paths": [str(summary_path)],
            "live_eval_summary_paths": [],
        },
        references={"summary_path": str(summary_path)},
        artifact_paths={"case_dir": str(summary_path.parent), "summary_path": str(summary_path)},
        metrics=_build_metrics(
            primary_name=primary_metric.get("name"),
            primary_value=_float_or_none(primary_metric.get("estimate")),
            higher_is_better=True,
            snapshot={
                "paired_iteration_count": _float_or_none(payload.get("paired_iteration_count")),
                "promotion_passed": 1.0 if promotion.get("passed") else 0.0 if promotion.get("enabled") else None,
                "promotion_check_count": _float_or_none(promotion.get("check_count")),
                "promotion_candidate_count": _float_or_none(promotion.get("promotion_candidate_count")),
                "rollback_signal_count": _float_or_none(promotion.get("rollback_signal_count")),
                "route_decision_pair_count": _float_or_none(route_comparison.get("route_decision_pair_count")),
                "route_decision_overlap_rate": _float_or_none(dict(metrics.get("route_decision_overlap_rate", {})).get("mean")),
                "first_node_agreement_rate": _float_or_none(dict(metrics.get("first_node_agreement_rate", {})).get("mean")),
                "delta_route_quality_score": _float_or_none(dict(metrics.get("delta_route_quality_score", {})).get("mean")),
                "delta_pre_boss_readiness": _float_or_none(dict(metrics.get("delta_pre_boss_readiness", {})).get("mean")),
                "delta_route_risk_score": _float_or_none(dict(metrics.get("delta_route_risk_score", {})).get("mean")),
                "shadow_comparable_encounter_count": _float_or_none(shadow.get("comparable_encounter_count")),
                "shadow_candidate_advantage_rate": _float_or_none(shadow.get("candidate_advantage_rate")),
                "shadow_agreement_rate": _float_or_none(shadow.get("agreement_rate")),
                "shadow_delta_first_action_match_rate": _float_or_none(
                    shadow_delta_metrics.get("delta_first_action_match_rate")
                ),
                "shadow_delta_trace_hit_rate": _float_or_none(shadow_delta_metrics.get("delta_trace_hit_rate")),
            },
        ),
        metadata={"case_id": payload.get("case_id"), "mode": payload.get("mode"), "promotion": promotion, "shadow": shadow},
    )


def _entry_payload(
    *,
    family: str,
    artifact_kind: str,
    display_name: str,
    source_summary_path: Path,
    output_dir: Path,
    tags: Sequence[str],
    config: Any,
    lineage: dict[str, Any],
    references: dict[str, Any],
    artifact_paths: dict[str, Any],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "family": family,
        "artifact_kind": artifact_kind,
        "display_name": display_name,
        "source_summary_path": str(source_summary_path),
        "output_dir": str(output_dir),
        "tags": sorted(set(_normalize_tags(tags))),
        "config": config,
        "lineage": lineage,
        "references": references,
        "artifact_paths": artifact_paths,
        "metrics": metrics,
        "metadata": metadata,
    }


def _build_metrics(
    *,
    primary_name: str | None,
    primary_value: float | None,
    higher_is_better: bool,
    snapshot: dict[str, Any],
    benchmark_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark_summary = _benchmark_summary_snapshot(benchmark_payload)
    primary = None
    if primary_name is not None:
        primary = {
            "name": primary_name,
            "value": primary_value,
            "higher_is_better": higher_is_better,
            "source_benchmark_suite_name": None if benchmark_summary is None else benchmark_summary.get("suite_name"),
        }
    return {"primary": primary, "snapshot": snapshot, "benchmark": benchmark_summary}


def _benchmark_summary_snapshot(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None or payload.get("suite_name") is None:
        return None
    cases = list(payload.get("cases", []))
    primary_values = [
        _float_or_none(dict(case.get("primary_metric", {})).get("estimate"))
        for case in cases
        if dict(case.get("primary_metric", {})).get("estimate") is not None
    ]
    return {
        "suite_name": payload.get("suite_name"),
        "case_count": len(cases),
        "case_ids": [case.get("case_id") for case in cases],
        "primary_metric_mean": None if not primary_values else (sum(primary_values) / len(primary_values)),
    }


def _benchmark_primary_metric(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    if payload.get("suite_name") is not None and isinstance(payload.get("cases"), list):
        cases = list(payload.get("cases", []))
        estimates: list[float] = []
        names: list[str] = []
        for case in cases:
            primary_metric = dict(case.get("primary_metric", {}))
            estimate = _float_or_none(primary_metric.get("estimate"))
            if estimate is None:
                continue
            estimates.append(estimate)
            if primary_metric.get("name") is not None:
                names.append(str(primary_metric["name"]))
        if not estimates:
            return None
        metric_name = names[0] if len(set(names)) == 1 else "benchmark_primary_mean"
        return {
            "name": metric_name,
            "value": sum(estimates) / len(estimates),
        }
    primary_metric = dict(payload.get("primary_metric", {}))
    estimate = _float_or_none(primary_metric.get("estimate"))
    if primary_metric.get("name") is None or estimate is None:
        return None
    return {"name": str(primary_metric["name"]), "value": estimate}


def _combat_win_rate_from_eval(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    return _float_or_none(dict(payload.get("combat_performance", {})).get("combat_win_rate"))


def _dataset_lineage_source_paths(lineage: Any) -> list[str]:
    lineage_payload = {} if not isinstance(lineage, dict) else lineage
    return _collect_strings(
        [
            *lineage_payload.get("source_paths", []),
            *lineage_payload.get("resolved_source_files", []),
        ]
    )


def _optional_path_map(payload: dict[str, Any], *keys: str) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        mapped[key] = str(Path(str(value)).expanduser().resolve())
    return mapped


def _resolve_source_summary_path(source_path: Path) -> Path:
    if source_path.is_file():
        return source_path
    for filename in (
        "dataset-summary.json",
        "benchmark-suite-summary.json",
        "case-summary.json",
        "summary.json",
        "combat-outcome-predictor.json",
    ):
        candidate = source_path / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve a supported summary file from: {source_path}")


def _ensure_registry(root: str | Path) -> dict[str, Path]:
    root_dir = Path(root).expanduser().resolve()
    manifest_path = root_dir / REGISTRY_MANIFEST_FILENAME
    aliases_path = root_dir / REGISTRY_ALIASES_FILENAME
    alias_history_path = root_dir / REGISTRY_ALIAS_HISTORY_FILENAME
    experiments_dir = root_dir / REGISTRY_EXPERIMENTS_DIRNAME
    reports_dir = root_dir / REGISTRY_REPORTS_DIRNAME
    if not manifest_path.exists():
        initialize_registry(root_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    if not aliases_path.exists():
        aliases_path.write_text(json.dumps({"schema_version": REGISTRY_SCHEMA_VERSION, "aliases": {}}, indent=2), encoding="utf-8")
    if not alias_history_path.exists():
        alias_history_path.write_text("", encoding="utf-8")
    return {
        "root_dir": root_dir,
        "manifest_path": manifest_path,
        "aliases_path": aliases_path,
        "alias_history_path": alias_history_path,
        "experiments_dir": experiments_dir,
        "reports_dir": reports_dir,
    }


def _load_all_entries(root: str | Path) -> list[dict[str, Any]]:
    experiments_dir = _ensure_registry(root)["experiments_dir"]
    entries: list[dict[str, Any]] = []
    for path in sorted(experiments_dir.glob("*.json")):
        entries.append(json.loads(path.read_text(encoding="utf-8")))
    return entries


def _prepare_report_dir(*, root: Path, session_name: str) -> Path:
    output_dir = root / session_name
    if output_dir.exists():
        raise FileExistsError(f"Registry report output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _leaderboard_row(entry: dict[str, Any]) -> dict[str, Any]:
    primary_metric = dict(entry.get("metrics", {}).get("primary") or {})
    benchmark = dict(entry.get("metrics", {}).get("benchmark") or {})
    return {
        "experiment_id": entry["experiment_id"],
        "family": entry["family"],
        "artifact_kind": entry["artifact_kind"],
        "display_name": entry["display_name"],
        "aliases": entry.get("aliases", []),
        "tags": entry.get("tags", []),
        "primary_metric": primary_metric,
        "primary_metric_value": _float_or_none(primary_metric.get("value")),
        "primary_metric_higher_is_better": bool(primary_metric.get("higher_is_better", True)),
        "benchmark_suite_name": benchmark.get("suite_name"),
        "dataset_paths": list(dict(entry.get("lineage", {})).get("dataset_paths", [])),
        "source_summary_path": entry.get("source_summary_path"),
    }


def _sorted_leaderboard_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[float, str, str]:
        value = _float_or_none(row.get("primary_metric_value"))
        if value is None:
            value = float("-inf")
        if not bool(row.get("primary_metric_higher_is_better", True)):
            value = -value
        return (-value, str(row["family"]), str(row["experiment_id"]))

    return sorted(rows, key=sort_key)


def _flatten_metrics(payload: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        dotted = key if not prefix else f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_metrics(value, prefix=dotted))
        else:
            flattened[dotted] = value
    return flattened


def _render_leaderboard_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Registry Leaderboard",
        "",
        f"- Rows: `{payload['row_count']}`",
    ]
    for row in payload["rows"][:10]:
        metric = dict(row.get("primary_metric") or {})
        lines.append(
            f"- `{row['rank']}` `{row['experiment_id']}` `{row['family']}` "
            f"`{metric.get('name', '-')}`=`{_fmt(metric.get('value'))}`"
        )
    return "\n".join(lines)


def _render_compare_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Registry Compare",
        "",
        f"- Compared: `{payload['compared_count']}`",
    ]
    for experiment in payload["experiments"]:
        metric = dict(experiment.get("primary_metric") or {})
        lines.append(
            f"- `{experiment['experiment_id']}` `{experiment['family']}` "
            f"`{metric.get('name', '-')}`=`{_fmt(metric.get('value'))}`"
        )
    return "\n".join(lines)


def _load_optional_json(path_value: Any) -> dict[str, Any] | None:
    if path_value is None:
        return None
    path = Path(str(path_value)).expanduser().resolve()
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _default_experiment_id(*, family: str, display_name: str, source_summary_path: Path) -> str:
    stem = _normalize_experiment_id(f"{family}-{display_name}")[:48].strip("-")
    digest = hashlib.sha256(str(source_summary_path).encode("utf-8")).hexdigest()[:8]
    return f"{stem}-{digest}" if stem else digest


def _default_alias_artifact_key(entry: dict[str, Any]) -> str | None:
    artifact_paths = dict(entry.get("artifact_paths", {}))
    for key in ("best_checkpoint_path", "model_path", "checkpoint_path", "dataset_dir", "summary_path"):
        if key in artifact_paths:
            return key
    return None


def _touch_manifest(manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["updated_at_utc"] = _now_utc()
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_experiment_id(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    if not normalized:
        raise ValueError("Experiment id must contain at least one alphanumeric character.")
    return normalized


def _normalize_tags(values: Sequence[str] | None) -> list[str]:
    if values is None:
        return []
    return [tag.strip().lower() for tag in values if str(tag).strip()]


def _collect_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text:
            continue
        if text not in result:
            result.append(text)
    return result


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _default_report_session_name(prefix: str) -> str:
    return datetime.now(UTC).strftime(f"registry-{prefix}-%Y%m%d-%H%M%S")


def _now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"
