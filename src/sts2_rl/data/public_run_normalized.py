from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .public_runs import (
    STS2RUNS_SOURCE_NAME,
    load_public_run_archive_detail_records,
    load_public_run_archive_index_records,
    load_public_run_archive_summary,
    resolve_public_run_archive_details_path,
)

PUBLIC_RUN_NORMALIZED_SCHEMA_VERSION = 1
PUBLIC_RUN_NORMALIZED_FILENAME = "normalized-public-runs.jsonl"
PUBLIC_RUN_NORMALIZED_TABLE_FILENAME = "normalized-public-runs.csv"
PUBLIC_RUN_NORMALIZED_SUMMARY_FILENAME = "summary.json"
PUBLIC_RUN_NORMALIZED_SOURCE_MANIFEST_FILENAME = "source-manifest.json"

PUBLIC_RUN_STRATEGIC_CARD_STATS_FILENAME = "strategic-card-stats.jsonl"
PUBLIC_RUN_STRATEGIC_CARD_STATS_TABLE_FILENAME = "strategic-card-stats.csv"
PUBLIC_RUN_STRATEGIC_SHOP_STATS_FILENAME = "strategic-shop-stats.jsonl"
PUBLIC_RUN_STRATEGIC_SHOP_STATS_TABLE_FILENAME = "strategic-shop-stats.csv"
PUBLIC_RUN_STRATEGIC_EVENT_STATS_FILENAME = "strategic-event-stats.jsonl"
PUBLIC_RUN_STRATEGIC_EVENT_STATS_TABLE_FILENAME = "strategic-event-stats.csv"
PUBLIC_RUN_STRATEGIC_RELIC_STATS_FILENAME = "strategic-relic-stats.jsonl"
PUBLIC_RUN_STRATEGIC_RELIC_STATS_TABLE_FILENAME = "strategic-relic-stats.csv"
PUBLIC_RUN_STRATEGIC_ENCOUNTER_STATS_FILENAME = "strategic-encounter-stats.jsonl"
PUBLIC_RUN_STRATEGIC_ENCOUNTER_STATS_TABLE_FILENAME = "strategic-encounter-stats.csv"
PUBLIC_RUN_STRATEGIC_ROUTE_STATS_FILENAME = "strategic-route-stats.jsonl"
PUBLIC_RUN_STRATEGIC_ROUTE_STATS_TABLE_FILENAME = "strategic-route-stats.csv"

_STRATEGIC_STAT_FILENAMES: dict[str, str] = {
    "card": PUBLIC_RUN_STRATEGIC_CARD_STATS_FILENAME,
    "shop": PUBLIC_RUN_STRATEGIC_SHOP_STATS_FILENAME,
    "event": PUBLIC_RUN_STRATEGIC_EVENT_STATS_FILENAME,
    "relic": PUBLIC_RUN_STRATEGIC_RELIC_STATS_FILENAME,
    "encounter": PUBLIC_RUN_STRATEGIC_ENCOUNTER_STATS_FILENAME,
    "route": PUBLIC_RUN_STRATEGIC_ROUTE_STATS_FILENAME,
}


@dataclass(frozen=True)
class PublicRunNormalizationReport:
    output_dir: Path
    normalized_runs_path: Path
    normalized_runs_table_path: Path
    strategic_card_stats_path: Path
    strategic_shop_stats_path: Path
    strategic_event_stats_path: Path
    strategic_relic_stats_path: Path
    strategic_encounter_stats_path: Path
    strategic_route_stats_path: Path
    summary_path: Path
    source_manifest_path: Path


class PublicRunNormalizedRoomSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    act_index: int
    floor_within_act: int
    room_type: str
    map_point_type: str | None = None
    model_id: str | None = None
    turns_taken: int | None = None
    damage_taken: int | None = None
    gold_delta: int | None = None
    hp_healed: int | None = None
    source_type: str | None = None
    offered_cards: list[str] = Field(default_factory=list)
    picked_cards: list[str] = Field(default_factory=list)
    shop_offered_cards: list[str] = Field(default_factory=list)
    shop_purchased_cards: list[str] = Field(default_factory=list)
    cards_gained: list[str] = Field(default_factory=list)
    cards_removed: list[str] = Field(default_factory=list)
    relics_gained: list[str] = Field(default_factory=list)
    event_choice_keys: list[str] = Field(default_factory=list)
    rest_site_actions: list[str] = Field(default_factory=list)


class PublicRunNormalizedRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PUBLIC_RUN_NORMALIZED_SCHEMA_VERSION
    record_type: Literal["public_run_normalized"] = "public_run_normalized"
    source_name: str
    source_run_id: int
    character_id: str | None = None
    ascension: int | None = None
    build_id: str | None = None
    game_mode: str | None = None
    platform_type: str | None = None
    seed: str | None = None
    start_time_unix: int | None = None
    run_time_seconds: int | None = None
    win: bool | None = None
    was_abandoned: bool | None = None
    killed_by_encounter: str | None = None
    killed_by_event: str | None = None
    acts_reached: int = 0
    rooms: list[PublicRunNormalizedRoomSummary] = Field(default_factory=list)
    final_deck: list[str] = Field(default_factory=list)
    final_relics: list[str] = Field(default_factory=list)
    coverage_flags: dict[str, bool] = Field(default_factory=dict)
    benchmark_slice: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record(self) -> PublicRunNormalizedRunRecord:
        self.source_name = str(self.source_name).strip()
        if not self.source_name:
            raise ValueError("source_name is required.")
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PublicRunNormalizedSourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PUBLIC_RUN_NORMALIZED_SCHEMA_VERSION
    artifact_kind: Literal["public_run_normalized_source_manifest"] = "public_run_normalized_source_manifest"
    source_name: str
    generated_at_utc: str
    archive_root: str
    output_dir: str
    normalized_runs_path: str
    normalized_runs_table_path: str
    summary_path: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PublicRunStrategicStatRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PUBLIC_RUN_NORMALIZED_SCHEMA_VERSION
    record_type: Literal["public_run_strategic_stat"] = "public_run_strategic_stat"
    stat_family: Literal["card", "shop", "event", "relic", "encounter", "route"]
    source_name: str | None = None
    snapshot_date: str | None = None
    subject_id: str
    subject_name: str | None = None
    character_id: str | None = None
    act_id: str | None = None
    room_type: str | None = None
    source_type: str | None = None
    run_count: int = 0
    offer_count: int | None = None
    shop_offer_count: int | None = None
    pick_count: int | None = None
    pick_rate: float | None = None
    buy_count: int | None = None
    buy_rate: float | None = None
    seen_count: int | None = None
    use_count: int | None = None
    win_count: int = 0
    win_rate: float | None = None
    win_rate_with_card: float | None = None
    win_delta: float | None = None
    deck_presence_runs: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record(self) -> PublicRunStrategicStatRecord:
        self.subject_id = str(self.subject_id).strip()
        if not self.subject_id:
            raise ValueError("subject_id is required.")
        if self.run_count < 0 or self.win_count < 0:
            raise ValueError("run_count and win_count must be non-negative.")
        for field_name in (
            "offer_count",
            "shop_offer_count",
            "pick_count",
            "buy_count",
            "seen_count",
            "use_count",
            "deck_presence_runs",
        ):
            value = getattr(self, field_name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        for field_name in ("pick_rate", "buy_rate", "win_rate", "win_rate_with_card"):
            value = getattr(self, field_name)
            if value is not None and not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{field_name} must be between 0 and 1.")
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def default_public_run_normalization_session_name(prefix: str = "public-run-normalized") -> str:
    return datetime.now(UTC).strftime(f"{prefix}-%Y%m%d-%H%M%S")


def resolve_public_run_normalized_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    path = source_path / PUBLIC_RUN_NORMALIZED_FILENAME if source_path.is_dir() else source_path
    if not path.exists():
        raise FileNotFoundError(f"Normalized public runs do not exist: {path}")
    return path


def resolve_public_run_normalized_summary_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    path = source_path / PUBLIC_RUN_NORMALIZED_SUMMARY_FILENAME if source_path.is_dir() else source_path
    if not path.exists():
        raise FileNotFoundError(f"Normalized public run summary does not exist: {path}")
    return path


def load_public_run_normalized_records(source: str | Path) -> list[PublicRunNormalizedRunRecord]:
    path = resolve_public_run_normalized_path(source)
    records: list[PublicRunNormalizedRunRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PublicRunNormalizedRunRecord.model_validate(json.loads(line)))
    return records


def load_public_run_normalized_summary(source: str | Path) -> dict[str, Any]:
    path = resolve_public_run_normalized_summary_path(source)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_public_run_strategic_stats_path(source: str | Path, *, stat_family: str) -> Path:
    source_path = Path(source).expanduser().resolve()
    try:
        filename = _STRATEGIC_STAT_FILENAMES[stat_family]
    except KeyError as exc:
        raise ValueError(f"Unsupported strategic stat family: {stat_family}") from exc
    path = source_path / filename if source_path.is_dir() else source_path
    if not path.exists():
        raise FileNotFoundError(f"Public run strategic stats do not exist: {path}")
    return path


def load_public_run_strategic_stat_records(
    source: str | Path,
    *,
    stat_family: str | None = None,
) -> list[PublicRunStrategicStatRecord]:
    path = (
        resolve_public_run_strategic_stats_path(source, stat_family=stat_family)
        if stat_family is not None
        else Path(source).expanduser().resolve()
    )
    if not path.exists():
        raise FileNotFoundError(f"Public run strategic stats do not exist: {path}")
    records: list[PublicRunStrategicStatRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = PublicRunStrategicStatRecord.model_validate(json.loads(line))
        if stat_family is None or record.stat_family == stat_family:
            records.append(record)
    return records


def normalize_public_run_archive(
    *,
    source: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    replace_existing: bool = False,
) -> PublicRunNormalizationReport:
    archive_summary = load_public_run_archive_summary(source)
    index_records = load_public_run_archive_index_records(source)
    detail_records = load_public_run_archive_detail_records(source)
    detail_by_id = {record.source_run_id: record for record in detail_records}

    output_dir = _prepare_output_dir(output_root=output_root, session_name=session_name, replace_existing=replace_existing)
    normalized_runs_path = output_dir / PUBLIC_RUN_NORMALIZED_FILENAME
    normalized_runs_table_path = output_dir / PUBLIC_RUN_NORMALIZED_TABLE_FILENAME
    strategic_card_stats_path = output_dir / PUBLIC_RUN_STRATEGIC_CARD_STATS_FILENAME
    strategic_shop_stats_path = output_dir / PUBLIC_RUN_STRATEGIC_SHOP_STATS_FILENAME
    strategic_event_stats_path = output_dir / PUBLIC_RUN_STRATEGIC_EVENT_STATS_FILENAME
    strategic_relic_stats_path = output_dir / PUBLIC_RUN_STRATEGIC_RELIC_STATS_FILENAME
    strategic_encounter_stats_path = output_dir / PUBLIC_RUN_STRATEGIC_ENCOUNTER_STATS_FILENAME
    strategic_route_stats_path = output_dir / PUBLIC_RUN_STRATEGIC_ROUTE_STATS_FILENAME
    summary_path = output_dir / PUBLIC_RUN_NORMALIZED_SUMMARY_FILENAME
    source_manifest_path = output_dir / PUBLIC_RUN_NORMALIZED_SOURCE_MANIFEST_FILENAME

    records = [
        _normalize_archive_run(
            index_record=index_record,
            detail_record=detail_by_id.get(index_record.source_run_id),
        )
        for index_record in index_records
    ]
    _write_normalized_records(normalized_runs_path, records)
    _write_normalized_table(normalized_runs_table_path, records)
    strategic_stats = _build_strategic_stats(records)
    _write_strategic_records(strategic_card_stats_path, strategic_stats["card"])
    _write_strategic_records(strategic_shop_stats_path, strategic_stats["shop"])
    _write_strategic_records(strategic_event_stats_path, strategic_stats["event"])
    _write_strategic_records(strategic_relic_stats_path, strategic_stats["relic"])
    _write_strategic_records(strategic_encounter_stats_path, strategic_stats["encounter"])
    _write_strategic_records(strategic_route_stats_path, strategic_stats["route"])

    manifest = PublicRunNormalizedSourceManifest(
        source_name=STS2RUNS_SOURCE_NAME,
        generated_at_utc=datetime.now(UTC).isoformat(),
        archive_root=str(Path(archive_summary["archive_root"]).resolve()),
        output_dir=str(output_dir),
        normalized_runs_path=str(normalized_runs_path),
        normalized_runs_table_path=str(normalized_runs_table_path),
        summary_path=str(summary_path),
        metadata={
            "source_summary_path": archive_summary.get("summary_path"),
            "source_detail_path": str(resolve_public_run_archive_details_path(source)),
            "known_run_count": archive_summary.get("known_run_count"),
            "detailed_run_count": archive_summary.get("detailed_run_count"),
            "strategic_card_stats_path": str(strategic_card_stats_path),
            "strategic_shop_stats_path": str(strategic_shop_stats_path),
            "strategic_event_stats_path": str(strategic_event_stats_path),
            "strategic_relic_stats_path": str(strategic_relic_stats_path),
            "strategic_encounter_stats_path": str(strategic_encounter_stats_path),
            "strategic_route_stats_path": str(strategic_route_stats_path),
        },
    )
    source_manifest_path.write_text(json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(_build_normalized_summary(records=records, output_dir=output_dir, manifest=manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return PublicRunNormalizationReport(
        output_dir=output_dir,
        normalized_runs_path=normalized_runs_path,
        normalized_runs_table_path=normalized_runs_table_path,
        strategic_card_stats_path=strategic_card_stats_path,
        strategic_shop_stats_path=strategic_shop_stats_path,
        strategic_event_stats_path=strategic_event_stats_path,
        strategic_relic_stats_path=strategic_relic_stats_path,
        strategic_encounter_stats_path=strategic_encounter_stats_path,
        strategic_route_stats_path=strategic_route_stats_path,
        summary_path=summary_path,
        source_manifest_path=source_manifest_path,
    )


def _normalize_archive_run(
    *,
    index_record: Any,
    detail_record: Any | None,
) -> PublicRunNormalizedRunRecord:
    raw_payload = None if detail_record is None else json.loads(Path(detail_record.raw_payload_path).read_text(encoding="utf-8"))
    run_payload = {} if raw_payload is None else dict(raw_payload.get("run") or {})
    final_player = _first_player(run_payload)
    rooms = _normalize_room_summaries(run_payload)
    character_id = _normalize_character_id(_lookup_character(index_record=index_record, run_payload=run_payload, final_player=final_player))
    return PublicRunNormalizedRunRecord.model_validate(
        {
            "source_name": index_record.source_name,
            "source_run_id": index_record.source_run_id,
            "character_id": character_id,
            "ascension": index_record.ascension if run_payload == {} else _optional_int(run_payload.get("ascension")),
            "build_id": _normalize_optional_string(run_payload.get("build_id")) or index_record.build_id,
            "game_mode": _normalize_optional_string(run_payload.get("game_mode")),
            "platform_type": _normalize_optional_string(run_payload.get("platform_type")),
            "seed": _normalize_optional_string(run_payload.get("seed")) or index_record.seed,
            "start_time_unix": _optional_int(run_payload.get("start_time")) or index_record.start_time_unix,
            "run_time_seconds": _optional_int(run_payload.get("run_time")) or index_record.run_time_seconds,
            "win": index_record.win if raw_payload is None else _optional_bool(run_payload.get("win")),
            "was_abandoned": index_record.was_abandoned if raw_payload is None else _optional_bool(run_payload.get("was_abandoned")),
            "killed_by_encounter": _normalize_optional_string(run_payload.get("killed_by_encounter")) or index_record.killed_by,
            "killed_by_event": _normalize_optional_string(run_payload.get("killed_by_event")),
            "acts_reached": len(run_payload.get("map_point_history") or run_payload.get("acts") or []),
            "rooms": rooms,
            "final_deck": _extract_final_card_ids(final_player.get("deck") if isinstance(final_player, dict) else []),
            "final_relics": _extract_final_card_ids(final_player.get("relics") if isinstance(final_player, dict) else []),
            "coverage_flags": {
                "has_detail_payload": raw_payload is not None,
                "has_room_history": bool(run_payload.get("map_point_history")),
                "has_final_player_state": bool(final_player),
            },
            "benchmark_slice": {
                "character_id": character_id,
                "ascension_band": _ascension_band(index_record.ascension if raw_payload is None else _optional_int(run_payload.get("ascension"))),
                "outcome": "win" if (index_record.win if raw_payload is None else _optional_bool(run_payload.get("win"))) else "loss",
                "acts_reached": len(run_payload.get("map_point_history") or run_payload.get("acts") or []),
                "build_id": _normalize_optional_string(run_payload.get("build_id")) or index_record.build_id,
            },
            "provenance": {
                "source_url": None if detail_record is None else detail_record.source_url,
                "index_raw_payload_path": index_record.raw_payload_path,
                "detail_raw_payload_path": None if detail_record is None else detail_record.raw_payload_path,
                "identity_fingerprint": index_record.identity_fingerprint,
                "dedupe_key": index_record.dedupe_key,
            },
            "metadata": {
                "source_user_id": index_record.user_id,
                "detail_root_keys": [] if detail_record is None else detail_record.detail_root_keys,
                "run_root_keys": [] if detail_record is None else detail_record.run_root_keys,
                "room_count": len(rooms),
            },
        }
    )


def _normalize_room_summaries(run_payload: dict[str, Any]) -> list[PublicRunNormalizedRoomSummary]:
    rooms: list[PublicRunNormalizedRoomSummary] = []
    for act_index, act_points in enumerate(run_payload.get("map_point_history") or [], start=1):
        for floor_within_act, point_payload in enumerate(act_points or [], start=1):
            player_stats = _first_player_stats(point_payload)
            room_payload = _first_room(point_payload)
            if room_payload is None:
                continue
            room_type = _normalize_optional_string(room_payload.get("room_type")) or "unknown"
            source_type, offered_cards, picked_cards = _extract_room_card_choices(
                room_payload=room_payload,
                player_stats=player_stats,
                room_type=room_type,
            )
            shop_offered_cards, shop_purchased_cards = _extract_room_shop_cards(
                room_payload=room_payload,
                player_stats=player_stats,
            )
            rooms.append(
                PublicRunNormalizedRoomSummary.model_validate(
                    {
                        "act_index": act_index,
                        "floor_within_act": floor_within_act,
                        "room_type": room_type,
                        "map_point_type": _normalize_optional_string(point_payload.get("map_point_type")),
                        "model_id": _normalize_optional_string(room_payload.get("model_id")),
                        "turns_taken": _optional_int(room_payload.get("turns_taken")),
                        "damage_taken": _optional_int(player_stats.get("damage_taken")),
                        "gold_delta": (_optional_int(player_stats.get("gold_gained")) or 0) - (_optional_int(player_stats.get("gold_spent")) or 0),
                        "hp_healed": _optional_int(player_stats.get("hp_healed")),
                        "source_type": source_type,
                        "offered_cards": offered_cards,
                        "picked_cards": picked_cards,
                        "shop_offered_cards": shop_offered_cards,
                        "shop_purchased_cards": shop_purchased_cards,
                        "cards_gained": _extract_ids(player_stats.get("cards_gained")),
                        "cards_removed": _extract_ids(player_stats.get("cards_removed")),
                        "relics_gained": _extract_relic_ids(player_stats),
                        "event_choice_keys": _extract_event_choice_keys(player_stats.get("event_choices")),
                        "rest_site_actions": _extract_scalar_strings(player_stats.get("rest_site_choices")),
                    }
                )
            )
    return rooms


def _build_normalized_summary(
    *,
    records: list[PublicRunNormalizedRunRecord],
    output_dir: Path,
    manifest: PublicRunNormalizedSourceManifest,
) -> dict[str, Any]:
    strategic_paths = manifest.metadata
    return {
        "schema_version": PUBLIC_RUN_NORMALIZED_SCHEMA_VERSION,
        "artifact_kind": "public_run_normalized",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
        "records_path": manifest.normalized_runs_path,
        "table_path": manifest.normalized_runs_table_path,
        "summary_path": manifest.summary_path,
        "source_manifest_path": str(output_dir / PUBLIC_RUN_NORMALIZED_SOURCE_MANIFEST_FILENAME),
        "record_count": len(records),
        "detail_coverage_count": sum(1 for record in records if record.coverage_flags.get("has_detail_payload")),
        "character_histogram": dict(Counter(record.character_id for record in records if record.character_id)),
        "build_id_histogram": dict(Counter(record.build_id for record in records if record.build_id)),
        "ascension_histogram": dict(Counter(str(record.ascension) for record in records if record.ascension is not None)),
        "outcome_histogram": dict(Counter(record.benchmark_slice.get("outcome") for record in records if record.benchmark_slice.get("outcome"))),
        "room_type_histogram": dict(Counter(room.room_type for record in records for room in record.rooms)),
        "acts_reached_histogram": dict(Counter(str(record.acts_reached) for record in records)),
        "final_deck_coverage_count": sum(1 for record in records if record.final_deck),
        "final_relic_coverage_count": sum(1 for record in records if record.final_relics),
        "strategic_card_stats_path": strategic_paths.get("strategic_card_stats_path"),
        "strategic_shop_stats_path": strategic_paths.get("strategic_shop_stats_path"),
        "strategic_event_stats_path": strategic_paths.get("strategic_event_stats_path"),
        "strategic_relic_stats_path": strategic_paths.get("strategic_relic_stats_path"),
        "strategic_encounter_stats_path": strategic_paths.get("strategic_encounter_stats_path"),
        "strategic_route_stats_path": strategic_paths.get("strategic_route_stats_path"),
    }


def _write_normalized_records(path: Path, records: list[PublicRunNormalizedRunRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False))
            handle.write("\n")


def _write_normalized_table(path: Path, records: list[PublicRunNormalizedRunRecord]) -> None:
    fieldnames = [
        "source_run_id",
        "character_id",
        "ascension",
        "build_id",
        "game_mode",
        "platform_type",
        "seed",
        "run_time_seconds",
        "win",
        "was_abandoned",
        "killed_by_encounter",
        "killed_by_event",
        "acts_reached",
        "room_count",
        "final_deck_count",
        "final_relic_count",
        "has_detail_payload",
        "has_room_history",
        "has_final_player_state",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_run_id": record.source_run_id,
                    "character_id": record.character_id,
                    "ascension": record.ascension,
                    "build_id": record.build_id,
                    "game_mode": record.game_mode,
                    "platform_type": record.platform_type,
                    "seed": record.seed,
                    "run_time_seconds": record.run_time_seconds,
                    "win": record.win,
                    "was_abandoned": record.was_abandoned,
                    "killed_by_encounter": record.killed_by_encounter,
                    "killed_by_event": record.killed_by_event,
                    "acts_reached": record.acts_reached,
                    "room_count": len(record.rooms),
                    "final_deck_count": len(record.final_deck),
                    "final_relic_count": len(record.final_relics),
                    "has_detail_payload": record.coverage_flags.get("has_detail_payload"),
                    "has_room_history": record.coverage_flags.get("has_room_history"),
                    "has_final_player_state": record.coverage_flags.get("has_final_player_state"),
                }
            )


def _write_strategic_records(path: Path, records: list[PublicRunStrategicStatRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False))
            handle.write("\n")


def _build_strategic_stats(records: list[PublicRunNormalizedRunRecord]) -> dict[str, list[PublicRunStrategicStatRecord]]:
    source_name = records[0].source_name if records else None
    snapshot_date = datetime.now(UTC).date().isoformat()
    return {
        "card": _build_card_stats(records, source_name=source_name, snapshot_date=snapshot_date),
        "shop": _build_shop_stats(records, source_name=source_name, snapshot_date=snapshot_date),
        "event": _build_event_stats(records, source_name=source_name, snapshot_date=snapshot_date),
        "relic": _build_relic_stats(records, source_name=source_name, snapshot_date=snapshot_date),
        "encounter": _build_encounter_stats(records, source_name=source_name, snapshot_date=snapshot_date),
        "route": _build_route_stats(records, source_name=source_name, snapshot_date=snapshot_date),
    }


def _build_card_stats(
    records: list[PublicRunNormalizedRunRecord],
    *,
    source_name: str | None,
    snapshot_date: str | None,
) -> list[PublicRunStrategicStatRecord]:
    buckets: dict[tuple[str, str | None, str | None, str], dict[str, Any]] = {}
    baseline_by_character = _win_rate_baseline_by_character(records)
    for record in records:
        win_flag = 1 if record.win else 0
        deck_presence_cards = set(record.final_deck)
        picked_keys: set[tuple[str, str | None, str | None, str]] = set()
        offered_keys: set[tuple[str, str | None, str | None, str]] = set()
        shop_offered_keys: set[tuple[str, str | None, str | None, str]] = set()
        for room in record.rooms:
            act_id = f"ACT_{room.act_index}"
            source_type = room.source_type or ("shop" if room.room_type == "shop" else "reward")
            for card_id in set(room.offered_cards):
                key = (card_id, record.character_id, act_id, source_type)
                bucket = buckets.setdefault(
                    key,
                    _card_stat_bucket(card_id=card_id, character_id=record.character_id, act_id=act_id, source_type=source_type),
                )
                bucket["offer_count"] += 1
                offered_keys.add(key)
            for card_id in set(room.picked_cards):
                key = (card_id, record.character_id, act_id, source_type)
                bucket = buckets.setdefault(
                    key,
                    _card_stat_bucket(card_id=card_id, character_id=record.character_id, act_id=act_id, source_type=source_type),
                )
                bucket["pick_count"] += 1
                bucket["selected_run_count"] += 1
                bucket["selected_win_count"] += win_flag
                picked_keys.add(key)
            for card_id in set(room.shop_offered_cards):
                key = (card_id, record.character_id, act_id, "shop")
                bucket = buckets.setdefault(
                    key,
                    _card_stat_bucket(card_id=card_id, character_id=record.character_id, act_id=act_id, source_type="shop"),
                )
                bucket["shop_offer_count"] += 1
                shop_offered_keys.add(key)
            for card_id in set(room.shop_purchased_cards):
                key = (card_id, record.character_id, act_id, "shop")
                bucket = buckets.setdefault(
                    key,
                    _card_stat_bucket(card_id=card_id, character_id=record.character_id, act_id=act_id, source_type="shop"),
                )
                bucket["buy_count"] += 1
                bucket["selected_run_count"] += 1
                bucket["selected_win_count"] += win_flag
                picked_keys.add(key)
        for key in offered_keys | shop_offered_keys:
            buckets[key]["run_count"] += 1
            buckets[key]["win_count"] += win_flag
        for card_id in deck_presence_cards:
            for key, bucket in buckets.items():
                if key[0] != card_id or key[1] != record.character_id:
                    continue
                bucket["deck_presence_runs"] += 1
    return _finalize_card_stat_buckets(
        buckets.values(),
        baseline_by_character=baseline_by_character,
        source_name=source_name,
        snapshot_date=snapshot_date,
    )


def _build_shop_stats(
    records: list[PublicRunNormalizedRunRecord],
    *,
    source_name: str | None,
    snapshot_date: str | None,
) -> list[PublicRunStrategicStatRecord]:
    buckets: dict[tuple[str, str | None], dict[str, int | str | None]] = {}
    for record in records:
        for room in record.rooms:
            if room.room_type != "shop":
                continue
            key = ("shop_visit", record.character_id)
            bucket = buckets.setdefault(
                key,
                {"subject_id": "shop_visit", "character_id": record.character_id, "run_count": 0, "win_count": 0, "seen_count": 0},
            )
            bucket["run_count"] = int(bucket["run_count"]) + 1
            bucket["seen_count"] = int(bucket["seen_count"]) + 1
            if record.win:
                bucket["win_count"] = int(bucket["win_count"]) + 1
    return _finalize_stat_buckets("shop", buckets.values(), source_name=source_name, snapshot_date=snapshot_date)


def _build_event_stats(
    records: list[PublicRunNormalizedRunRecord],
    *,
    source_name: str | None,
    snapshot_date: str | None,
) -> list[PublicRunStrategicStatRecord]:
    buckets: dict[tuple[str, str | None], dict[str, int | str | None]] = {}
    for record in records:
        for room in record.rooms:
            for choice_key in room.event_choice_keys:
                key = (choice_key, record.character_id)
                bucket = buckets.setdefault(
                    key,
                    {"subject_id": choice_key, "character_id": record.character_id, "run_count": 0, "win_count": 0, "seen_count": 0},
                )
                bucket["run_count"] = int(bucket["run_count"]) + 1
                bucket["seen_count"] = int(bucket["seen_count"]) + 1
                if record.win:
                    bucket["win_count"] = int(bucket["win_count"]) + 1
    return _finalize_stat_buckets("event", buckets.values(), source_name=source_name, snapshot_date=snapshot_date)


def _build_relic_stats(
    records: list[PublicRunNormalizedRunRecord],
    *,
    source_name: str | None,
    snapshot_date: str | None,
) -> list[PublicRunStrategicStatRecord]:
    buckets: dict[tuple[str, str | None], dict[str, int | str | None]] = {}
    for record in records:
        for relic_id in set(record.final_relics):
            key = (relic_id, record.character_id)
            bucket = buckets.setdefault(
                key,
                {"subject_id": relic_id, "character_id": record.character_id, "run_count": 0, "win_count": 0},
            )
            bucket["run_count"] = int(bucket["run_count"]) + 1
            if record.win:
                bucket["win_count"] = int(bucket["win_count"]) + 1
    return _finalize_stat_buckets("relic", buckets.values(), source_name=source_name, snapshot_date=snapshot_date)


def _build_encounter_stats(
    records: list[PublicRunNormalizedRunRecord],
    *,
    source_name: str | None,
    snapshot_date: str | None,
) -> list[PublicRunStrategicStatRecord]:
    buckets: dict[tuple[str, str | None], dict[str, int | str | None]] = {}
    for record in records:
        for room in record.rooms:
            if room.room_type not in {"monster", "elite", "boss"}:
                continue
            subject_id = room.model_id or room.room_type
            key = (subject_id, record.character_id)
            bucket = buckets.setdefault(
                key,
                {
                    "subject_id": subject_id,
                    "character_id": record.character_id,
                    "room_type": room.room_type,
                    "run_count": 0,
                    "win_count": 0,
                    "seen_count": 0,
                },
            )
            bucket["run_count"] = int(bucket["run_count"]) + 1
            bucket["seen_count"] = int(bucket["seen_count"]) + 1
            if record.win:
                bucket["win_count"] = int(bucket["win_count"]) + 1
    return _finalize_stat_buckets("encounter", buckets.values(), source_name=source_name, snapshot_date=snapshot_date)


def _build_route_stats(
    records: list[PublicRunNormalizedRunRecord],
    *,
    source_name: str | None,
    snapshot_date: str | None,
) -> list[PublicRunStrategicStatRecord]:
    buckets: dict[tuple[str, str | None, str], dict[str, int | str | None]] = {}
    baseline_by_character = _win_rate_baseline_by_character(records)
    for record in records:
        for room in record.rooms:
            act_id = f"ACT_{room.act_index}"
            key = (room.room_type, record.character_id, act_id)
            bucket = buckets.setdefault(
                key,
                {
                    "subject_id": room.room_type,
                    "character_id": record.character_id,
                    "act_id": act_id,
                    "room_type": room.room_type,
                    "run_count": 0,
                    "win_count": 0,
                    "seen_count": 0,
                },
            )
            bucket["run_count"] = int(bucket["run_count"]) + 1
            bucket["seen_count"] = int(bucket["seen_count"]) + 1
            if record.win:
                bucket["win_count"] = int(bucket["win_count"]) + 1
    route_records = _finalize_stat_buckets(
        "route",
        buckets.values(),
        source_name=source_name,
        snapshot_date=snapshot_date,
    )
    for record in route_records:
        baseline = baseline_by_character.get(record.character_id, baseline_by_character.get(None))
        if baseline is not None and record.win_rate is not None:
            record.win_delta = record.win_rate - baseline
    return route_records


def _card_stat_bucket(
    *,
    card_id: str,
    character_id: str | None,
    act_id: str | None,
    source_type: str,
) -> dict[str, Any]:
    return {
        "subject_id": card_id,
        "character_id": character_id,
        "act_id": act_id,
        "source_type": source_type,
        "run_count": 0,
        "win_count": 0,
        "offer_count": 0,
        "shop_offer_count": 0,
        "pick_count": 0,
        "buy_count": 0,
        "selected_run_count": 0,
        "selected_win_count": 0,
        "deck_presence_runs": 0,
    }


def _finalize_card_stat_buckets(
    buckets: Any,
    *,
    baseline_by_character: dict[str | None, float],
    source_name: str | None,
    snapshot_date: str | None,
) -> list[PublicRunStrategicStatRecord]:
    records: list[PublicRunStrategicStatRecord] = []
    for bucket in buckets:
        run_count = int(bucket.get("run_count") or 0)
        win_count = int(bucket.get("win_count") or 0)
        selected_run_count = int(bucket.get("selected_run_count") or 0)
        selected_win_count = int(bucket.get("selected_win_count") or 0)
        offer_count = int(bucket.get("offer_count") or 0)
        shop_offer_count = int(bucket.get("shop_offer_count") or 0)
        pick_count = int(bucket.get("pick_count") or 0)
        buy_count = int(bucket.get("buy_count") or 0)
        baseline = baseline_by_character.get(bucket.get("character_id"), baseline_by_character.get(None))
        win_rate_with_card = None if selected_run_count == 0 else selected_win_count / selected_run_count
        records.append(
            PublicRunStrategicStatRecord.model_validate(
                {
                    "stat_family": "card",
                    "source_name": source_name,
                    "snapshot_date": snapshot_date,
                    "subject_id": bucket.get("subject_id"),
                    "character_id": bucket.get("character_id"),
                    "act_id": bucket.get("act_id"),
                    "source_type": bucket.get("source_type"),
                    "run_count": run_count,
                    "offer_count": offer_count or None,
                    "shop_offer_count": shop_offer_count or None,
                    "pick_count": pick_count or None,
                    "pick_rate": None if offer_count == 0 else pick_count / offer_count,
                    "buy_count": buy_count or None,
                    "buy_rate": None if shop_offer_count == 0 else buy_count / shop_offer_count,
                    "win_count": win_count,
                    "win_rate": None if run_count == 0 else win_count / run_count,
                    "win_rate_with_card": win_rate_with_card,
                    "win_delta": (
                        None
                        if win_rate_with_card is None or baseline is None
                        else win_rate_with_card - baseline
                    ),
                    "deck_presence_runs": int(bucket.get("deck_presence_runs") or 0) or None,
                    "metadata": {
                        "selected_run_count": selected_run_count,
                        "selected_win_count": selected_win_count,
                        "artifact_family": "public_run_strategic_card_stats",
                    },
                }
            )
        )
    return sorted(
        records,
        key=lambda record: (
            record.subject_id,
            record.character_id or "",
            record.act_id or "",
            record.source_type or "",
        ),
    )


def _win_rate_baseline_by_character(records: list[PublicRunNormalizedRunRecord]) -> dict[str | None, float]:
    wins_by_character: dict[str | None, int] = {}
    runs_by_character: dict[str | None, int] = {}
    for record in records:
        key = record.character_id
        runs_by_character[key] = runs_by_character.get(key, 0) + 1
        runs_by_character[None] = runs_by_character.get(None, 0) + 1
        if record.win:
            wins_by_character[key] = wins_by_character.get(key, 0) + 1
            wins_by_character[None] = wins_by_character.get(None, 0) + 1
    return {
        key: wins_by_character.get(key, 0) / run_count
        for key, run_count in runs_by_character.items()
        if run_count > 0
    }


def _finalize_stat_buckets(
    family: Literal["card", "shop", "event", "relic", "encounter", "route"],
    buckets: Any,
    *,
    source_name: str | None,
    snapshot_date: str | None,
) -> list[PublicRunStrategicStatRecord]:
    records: list[PublicRunStrategicStatRecord] = []
    for bucket in buckets:
        run_count = int(bucket.get("run_count") or 0)
        win_count = int(bucket.get("win_count") or 0)
        records.append(
            PublicRunStrategicStatRecord.model_validate(
                {
                    "stat_family": family,
                    "source_name": source_name,
                    "snapshot_date": snapshot_date,
                    "subject_id": bucket.get("subject_id"),
                    "character_id": bucket.get("character_id"),
                    "act_id": bucket.get("act_id"),
                    "room_type": bucket.get("room_type"),
                    "run_count": run_count,
                    "seen_count": bucket.get("seen_count"),
                    "win_count": win_count,
                    "win_rate": None if run_count == 0 else win_count / run_count,
                }
            )
        )
    return sorted(records, key=lambda record: (record.subject_id, record.character_id or "", record.act_id or ""))


def _prepare_output_dir(*, output_root: str | Path, session_name: str | None, replace_existing: bool) -> Path:
    output_root_path = Path(output_root).expanduser().resolve()
    output_dir = output_root_path / (session_name or default_public_run_normalization_session_name())
    if replace_existing and output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _lookup_character(*, index_record: Any, run_payload: dict[str, Any], final_player: dict[str, Any]) -> Any:
    return final_player.get("character") or run_payload.get("character") or index_record.character_id


def _first_player(run_payload: dict[str, Any]) -> dict[str, Any]:
    players = run_payload.get("players")
    if isinstance(players, list) and players:
        first = players[0]
        if isinstance(first, dict):
            return first
    return {}


def _first_player_stats(point_payload: dict[str, Any]) -> dict[str, Any]:
    players = point_payload.get("player_stats")
    if isinstance(players, list) and players:
        first = players[0]
        if isinstance(first, dict):
            return first
    return {}


def _first_room(point_payload: dict[str, Any]) -> dict[str, Any] | None:
    rooms = point_payload.get("rooms")
    if isinstance(rooms, list) and rooms:
        first = rooms[0]
        if isinstance(first, dict):
            return first
    return None


def _extract_final_card_ids(items: Any) -> list[str]:
    return _extract_ids(items)


def _extract_ids(items: Any) -> list[str]:
    values: list[str] = []
    if not isinstance(items, list):
        return values
    for item in items:
        card_id = _extract_card_id(item)
        if card_id is not None:
            values.append(card_id)
    return values


def _extract_relic_ids(player_stats: dict[str, Any]) -> list[str]:
    return _extract_choice_ids(player_stats.get("relic_choices")) + _extract_ancient_choice_ids(player_stats.get("ancient_choice"))


def _extract_choice_ids(items: Any) -> list[str]:
    values: list[str] = []
    if not isinstance(items, list):
        return values
    for item in items:
        if isinstance(item, dict):
            choice = _normalize_optional_string(item.get("choice"))
            if choice and item.get("was_picked") is not False:
                values.append(choice)
    return values


def _extract_ancient_choice_ids(items: Any) -> list[str]:
    values: list[str] = []
    if not isinstance(items, list):
        return values
    for item in items:
        if isinstance(item, dict) and item.get("was_chosen"):
            choice = _normalize_optional_string(item.get("TextKey"))
            if choice:
                values.append(choice)
    return values


def _extract_event_choice_keys(items: Any) -> list[str]:
    values: list[str] = []
    if not isinstance(items, list):
        return values
    for item in items:
        if isinstance(item, dict):
            title = item.get("title")
            if isinstance(title, dict):
                key = _normalize_optional_string(title.get("key"))
                if key:
                    values.append(key)
    return values


def _extract_scalar_strings(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return [text for text in (_normalize_optional_string(item) for item in items) if text is not None]


def _extract_room_card_choices(
    *,
    room_payload: dict[str, Any],
    player_stats: dict[str, Any],
    room_type: str,
) -> tuple[str | None, list[str], list[str]]:
    choice_items = player_stats.get("card_choices")
    if not isinstance(choice_items, list):
        return _source_type_from_room(room_type), [], []

    offered_cards: list[str] = []
    picked_cards: list[str] = []
    source_types: list[str] = []
    for item in choice_items:
        if not isinstance(item, dict):
            continue
        source_type = _canonical_card_source_type(item.get("source_type") or item.get("source") or item.get("choice_type"))
        if source_type is not None:
            source_types.append(source_type)
        offered_cards.extend(_collect_candidate_card_ids(item.get("offered_cards")))
        offered_cards.extend(_collect_candidate_card_ids(item.get("offers")))
        offered_cards.extend(_collect_candidate_card_ids(item.get("available_cards")))
        offered_cards.extend(_collect_candidate_card_ids(item.get("cards")))
        offered_cards.extend(_collect_candidate_card_ids(item.get("not_picked")))
        offered_cards.extend(_collect_candidate_card_ids(item.get("choices")))
        picked_cards.extend(_collect_candidate_card_ids(item.get("picked_cards")))
        picked_cards.extend(_collect_candidate_card_ids(item.get("picked_card")))
        picked_cards.extend(_collect_candidate_card_ids(item.get("picked")))
        picked_cards.extend(_collect_candidate_card_ids(item.get("choice")))
    return (
        source_types[0] if source_types else _source_type_from_room(room_type),
        _unique_preserve_order(offered_cards),
        _unique_preserve_order(picked_cards),
    )


def _extract_room_shop_cards(
    *,
    room_payload: dict[str, Any],
    player_stats: dict[str, Any],
) -> tuple[list[str], list[str]]:
    offered_cards = _collect_candidate_card_ids(room_payload.get("cards"))
    offered_cards.extend(_collect_candidate_card_ids(room_payload.get("card_offers")))
    offered_cards.extend(_collect_candidate_card_ids(room_payload.get("shop_cards")))
    offered_cards.extend(_collect_candidate_card_ids(player_stats.get("shop_cards")))
    offered_cards.extend(_collect_candidate_card_ids(player_stats.get("shop_card_offers")))
    purchased_cards = _collect_candidate_card_ids(player_stats.get("cards_purchased"))
    purchased_cards.extend(_collect_candidate_card_ids(player_stats.get("shop_cards_purchased")))
    purchased_cards.extend(_collect_candidate_card_ids(player_stats.get("purchased_cards")))
    return _unique_preserve_order(offered_cards), _unique_preserve_order(purchased_cards)


def _collect_candidate_card_ids(items: Any) -> list[str]:
    if items is None:
        return []
    if isinstance(items, list):
        values: list[str] = []
        for item in items:
            values.extend(_collect_candidate_card_ids(item))
        return values
    if isinstance(items, dict):
        direct = _extract_card_id(items)
        if direct is not None:
            return [direct]
        values: list[str] = []
        for key in ("card", "final_card", "picked_card", "choice", "reward"):
            if key in items:
                values.extend(_collect_candidate_card_ids(items.get(key)))
        return values
    direct = _extract_card_id(items)
    return [] if direct is None else [direct]


def _source_type_from_room(room_type: str) -> str | None:
    lowered = room_type.strip().lower()
    if lowered == "shop":
        return "shop"
    if lowered == "event":
        return "event"
    if lowered in {"neow", "start"}:
        return "starter"
    if lowered:
        return "reward"
    return None


def _canonical_card_source_type(value: Any) -> str | None:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return None
    lowered = normalized.lower()
    if lowered in {"reward", "shop", "event", "colorless", "starter", "unknown"}:
        return lowered
    if "shop" in lowered or "merchant" in lowered:
        return "shop"
    if "event" in lowered:
        return "event"
    if "colorless" in lowered:
        return "colorless"
    if "starter" in lowered or "neow" in lowered:
        return "starter"
    if "reward" in lowered or "combat" in lowered:
        return "reward"
    return "unknown"


def _unique_preserve_order(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _extract_card_id(item: Any) -> str | None:
    if isinstance(item, str):
        return _normalize_optional_string(item)
    if isinstance(item, dict):
        direct = _normalize_optional_string(item.get("id"))
        if direct:
            return direct
        nested_card = item.get("card")
        if isinstance(nested_card, dict):
            nested = _normalize_optional_string(nested_card.get("id"))
            if nested:
                return nested
        final_card = item.get("final_card")
        if isinstance(final_card, dict):
            final_id = _normalize_optional_string(final_card.get("id"))
            if final_id:
                return final_id
    return None


def _normalize_character_id(value: Any) -> str | None:
    text = _normalize_optional_string(value)
    if text is None:
        return None
    return text.removeprefix("CHARACTER.").upper()


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    text = _normalize_optional_string(value)
    if text is None:
        return None
    return int(float(text))


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(int(value))


def _ascension_band(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value <= 5:
        return "0-5"
    if value <= 10:
        return "6-10"
    if value <= 15:
        return "11-15"
    return "16+"
