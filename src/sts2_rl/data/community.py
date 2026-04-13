from __future__ import annotations

import csv
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

COMMUNITY_CARD_STATS_SCHEMA_VERSION = 1
COMMUNITY_CARD_STATS_FILENAME = "community-card-stats.jsonl"
COMMUNITY_CARD_STATS_TABLE_FILENAME = "community-card-stats.csv"
COMMUNITY_CARD_STATS_SUMMARY_FILENAME = "summary.json"
COMMUNITY_CARD_STATS_SOURCE_MANIFEST_SCHEMA_VERSION = 1
COMMUNITY_CARD_STATS_SOURCE_MANIFEST_FILENAME = "source-manifest.json"
COMMUNITY_CARD_STATS_RAW_DIRNAME = "raw"

SPIREMETA_DEFAULT_SOURCE_NAME = "spiremeta"
SPIREMETA_DEFAULT_API_BASE_URL = "https://api.spiremeta.gg"
SPIREMETA_GLOBAL_CARDS_PATH = "/api/v1/stats/global/cards"
SPIREMETA_API_KEY_ENV_VAR = "SPIREMETA_API_KEY"

_TOKEN_PATTERN = re.compile(r"[^A-Za-z0-9]+")

CommunityCardSourceType = Literal["reward", "shop", "event", "colorless", "starter", "unknown"]
CommunityCardStatsSourceKind = Literal["local_file", "spiremeta_api"]


class CommunityCardStatRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = COMMUNITY_CARD_STATS_SCHEMA_VERSION
    record_type: Literal["community_card_stat"] = "community_card_stat"
    source_name: str
    source_url: str | None = None
    snapshot_date: str
    snapshot_label: str | None = None
    game_version: str | None = None
    character_id: str | None = None
    ascension_min: int | None = None
    ascension_max: int | None = None
    act_id: str | None = None
    floor_band: str | None = None
    source_type: CommunityCardSourceType = "unknown"
    card_id: str
    card_name: str | None = None
    offer_count: int | None = None
    pick_count: int | None = None
    pick_rate: float | None = None
    shop_offer_count: int | None = None
    buy_count: int | None = None
    buy_rate: float | None = None
    deck_presence_runs: int | None = None
    run_count: int | None = None
    win_rate_with_card: float | None = None
    win_delta: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_rates(self) -> CommunityCardStatRecord:
        self.source_name = str(self.source_name).strip()
        if not self.source_name:
            raise ValueError("source_name is required.")
        self.snapshot_date = _normalize_snapshot_date(self.snapshot_date)
        self.source_url = _normalize_optional_string(self.source_url)
        self.snapshot_label = _normalize_optional_string(self.snapshot_label)
        self.game_version = _normalize_optional_string(self.game_version)
        self.character_id = _normalize_optional_string(self.character_id)
        self.act_id = _normalize_optional_string(self.act_id)
        self.floor_band = _normalize_optional_string(self.floor_band)
        self.card_id = str(self.card_id).strip()
        if not self.card_id:
            raise ValueError("card_id is required.")
        self.card_name = _normalize_optional_string(self.card_name)

        if self.pick_rate is None and self.offer_count not in {None, 0} and self.pick_count is not None:
            self.pick_rate = float(self.pick_count) / float(self.offer_count)
        if self.buy_rate is None and self.shop_offer_count not in {None, 0} and self.buy_count is not None:
            self.buy_rate = float(self.buy_count) / float(self.shop_offer_count)

        for field_name in ("pick_rate", "buy_rate", "win_rate_with_card"):
            value = getattr(self, field_name)
            if value is not None and not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{field_name} must be between 0 and 1.")
        for field_name in ("offer_count", "pick_count", "shop_offer_count", "buy_count", "deck_presence_runs", "run_count"):
            value = getattr(self, field_name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{field_name} must be non-negative.")
        if self.ascension_min is not None and self.ascension_min < 0:
            raise ValueError("ascension_min must be non-negative.")
        if self.ascension_max is not None and self.ascension_max < 0:
            raise ValueError("ascension_max must be non-negative.")
        if (
            self.ascension_min is not None
            and self.ascension_max is not None
            and self.ascension_min > self.ascension_max
        ):
            raise ValueError("ascension_min must be <= ascension_max.")
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CommunityCardStatsSourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["local_input_copy", "spiremeta_page"]
    path: str
    source_url: str | None = None
    original_path: str | None = None
    character_id: str | None = None
    page: int | None = None
    item_count: int | None = None
    total_pages: int | None = None
    sha256: str | None = None
    request_params: dict[str, Any] = Field(default_factory=dict)
    response_headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CommunityCardStatsSourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = COMMUNITY_CARD_STATS_SOURCE_MANIFEST_SCHEMA_VERSION
    artifact_kind: Literal["community_card_stats_source_manifest"] = "community_card_stats_source_manifest"
    source_name: str
    source_kind: CommunityCardStatsSourceKind
    source_url: str | None = None
    snapshot_date: str
    snapshot_label: str | None = None
    game_version: str | None = None
    generated_at_utc: str
    fetch_started_at_utc: str | None = None
    fetch_completed_at_utc: str | None = None
    output_dir: str
    records_path: str
    table_path: str
    summary_path: str
    raw_payload_root: str | None = None
    request_count: int = 0
    request_parameters: dict[str, Any] = Field(default_factory=dict)
    source_files: list[CommunityCardStatsSourceFile] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> CommunityCardStatsSourceManifest:
        self.source_name = str(self.source_name).strip()
        if not self.source_name:
            raise ValueError("source_name is required.")
        self.snapshot_date = _normalize_snapshot_date(self.snapshot_date)
        self.source_url = _normalize_optional_string(self.source_url)
        self.snapshot_label = _normalize_optional_string(self.snapshot_label)
        self.game_version = _normalize_optional_string(self.game_version)
        self.fetch_started_at_utc = _normalize_optional_timestamp(self.fetch_started_at_utc)
        self.fetch_completed_at_utc = _normalize_optional_timestamp(self.fetch_completed_at_utc)
        self.generated_at_utc = _normalize_required_timestamp(self.generated_at_utc)
        if self.request_count < 0:
            raise ValueError("request_count must be non-negative.")
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True)
class CommunityCardStatsImportReport:
    output_dir: Path
    records_path: Path
    table_path: Path
    summary_path: Path
    record_count: int
    card_count: int
    source_manifest_path: Path | None = None
    raw_payload_root: Path | None = None


def default_community_card_stats_session_name(prefix: str = "community-card-stats") -> str:
    return datetime.now(UTC).strftime(f"{prefix}-%Y%m%d-%H%M%S")


def load_community_card_stat_records(source: str | Path) -> list[CommunityCardStatRecord]:
    records_path = resolve_community_card_stats_path(source)
    records: list[CommunityCardStatRecord] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(CommunityCardStatRecord.model_validate(json.loads(line)))
    return records


def resolve_community_card_stats_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    records_path = source_path / COMMUNITY_CARD_STATS_FILENAME if source_path.is_dir() else source_path
    if not records_path.exists():
        raise FileNotFoundError(f"Community card stats do not exist: {records_path}")
    return records_path


def resolve_community_card_stats_source_manifest_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    manifest_path = source_path / COMMUNITY_CARD_STATS_SOURCE_MANIFEST_FILENAME if source_path.is_dir() else source_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"Community card stats source manifest does not exist: {manifest_path}")
    return manifest_path


def load_community_card_stats_summary(source: str | Path) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    summary_path = source_path / COMMUNITY_CARD_STATS_SUMMARY_FILENAME if source_path.is_dir() else source_path
    if not summary_path.exists():
        raise FileNotFoundError(f"Community card stats summary does not exist: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_community_card_stats_source_manifest(source: str | Path) -> CommunityCardStatsSourceManifest:
    manifest_path = resolve_community_card_stats_source_manifest_path(source)
    return CommunityCardStatsSourceManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))


def import_community_card_stats(
    *,
    source: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    source_name: str | None = None,
    source_url: str | None = None,
    snapshot_date: str | None = None,
    snapshot_label: str | None = None,
    game_version: str | None = None,
    replace_existing: bool = False,
) -> CommunityCardStatsImportReport:
    input_path = Path(source).expanduser().resolve()
    output_dir = _prepare_output_dir(
        output_root=output_root,
        session_name=session_name,
        replace_existing=replace_existing,
    )
    raw_payload_root = output_dir / COMMUNITY_CARD_STATS_RAW_DIRNAME
    raw_payload_root.mkdir(parents=True, exist_ok=True)

    raw_payloads, embedded_metadata = _load_import_payload(input_path)
    records = [
        _record_from_raw_payload(
            payload,
            source_name=source_name or embedded_metadata.get("source_name"),
            source_url=source_url or embedded_metadata.get("source_url"),
            snapshot_date=snapshot_date or embedded_metadata.get("snapshot_date"),
            snapshot_label=snapshot_label or embedded_metadata.get("snapshot_label"),
            game_version=game_version or embedded_metadata.get("game_version"),
        )
        for payload in raw_payloads
    ]
    copied_source_path = raw_payload_root / f"local-source{input_path.suffix.lower() or '.txt'}"
    shutil.copy2(input_path, copied_source_path)
    imported_at_utc = datetime.now(UTC).isoformat()
    manifest = _build_source_manifest(
        source_name=source_name or embedded_metadata.get("source_name") or "local-import",
        source_kind="local_file",
        source_url=source_url or embedded_metadata.get("source_url"),
        snapshot_date=snapshot_date or embedded_metadata.get("snapshot_date"),
        snapshot_label=snapshot_label or embedded_metadata.get("snapshot_label"),
        game_version=game_version or embedded_metadata.get("game_version"),
        output_dir=output_dir,
        raw_payload_root=raw_payload_root,
        fetch_started_at_utc=imported_at_utc,
        fetch_completed_at_utc=imported_at_utc,
        request_count=1,
        request_parameters={
            "source_path": str(input_path),
            "source_format": input_path.suffix.lower(),
        },
        source_files=[
            CommunityCardStatsSourceFile(
                kind="local_input_copy",
                path=str(copied_source_path),
                original_path=str(input_path),
                item_count=len(raw_payloads),
                sha256=_sha256_hex(copied_source_path),
                metadata={"embedded_metadata": embedded_metadata},
            )
        ],
        metadata={
            "embedded_metadata": embedded_metadata,
            "import_mode": "manual",
        },
    )
    return _persist_community_card_stats_artifacts(
        output_dir=output_dir,
        records=records,
        source_path=input_path,
        source_manifest=manifest,
        raw_payload_root=raw_payload_root,
    )


def import_spiremeta_community_card_stats(
    *,
    output_root: str | Path,
    characters: list[str],
    api_key: str | None = None,
    session_name: str | None = None,
    snapshot_date: str | None = None,
    snapshot_label: str | None = None,
    game_version: str | None = None,
    per_page: int = 100,
    max_pages: int | None = None,
    source_type: CommunityCardSourceType = "reward",
    replace_existing: bool = False,
    api_base_url: str = SPIREMETA_DEFAULT_API_BASE_URL,
    request_timeout_seconds: float = 20.0,
    client: httpx.Client | None = None,
    transport: httpx.BaseTransport | None = None,
) -> CommunityCardStatsImportReport:
    if client is not None and transport is not None:
        raise ValueError("Provide either client or transport, not both.")
    if per_page < 1:
        raise ValueError("per_page must be >= 1.")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be >= 1 when provided.")

    normalized_characters = _normalize_spiremeta_character_list(characters)
    if not normalized_characters:
        raise ValueError("At least one SpireMeta character must be provided.")
    resolved_api_key = _resolve_spiremeta_api_key(api_key)

    output_dir = _prepare_output_dir(
        output_root=output_root,
        session_name=session_name,
        replace_existing=replace_existing,
    )
    raw_payload_root = output_dir / COMMUNITY_CARD_STATS_RAW_DIRNAME / SPIREMETA_DEFAULT_SOURCE_NAME
    raw_payload_root.mkdir(parents=True, exist_ok=True)

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Accept": "application/json",
        "User-Agent": "sts2-rl community importer",
    }
    owns_client = client is None
    http_client = client or httpx.Client(
        base_url=api_base_url.rstrip("/"),
        timeout=request_timeout_seconds,
        transport=transport,
        headers=headers,
    )

    fetch_started_at_utc = datetime.now(UTC).isoformat()
    request_count = 0
    records: list[CommunityCardStatRecord] = []
    source_files: list[CommunityCardStatsSourceFile] = []
    try:
        for character_id in normalized_characters:
            page = 1
            while True:
                params = {
                    "character": character_id.lower(),
                    "page": page,
                    "per_page": per_page,
                }
                response = http_client.get(SPIREMETA_GLOBAL_CARDS_PATH, params=params)
                request_count += 1
                _raise_for_spiremeta_error(response)
                payload = response.json()
                items = payload.get("items") or []
                total_pages = int(payload.get("total_pages") or 1)
                payload_path = _write_spiremeta_raw_payload(
                    raw_payload_root=raw_payload_root,
                    character_id=character_id,
                    page=page,
                    payload=payload,
                )
                source_files.append(
                    CommunityCardStatsSourceFile(
                        kind="spiremeta_page",
                        path=str(payload_path),
                        source_url=str(response.request.url),
                        character_id=character_id,
                        page=page,
                        item_count=len(items),
                        total_pages=total_pages,
                        sha256=_sha256_hex(payload_path),
                        request_params=params,
                        response_headers=_extract_rate_limit_headers(response.headers),
                        metadata={
                            "total": payload.get("total"),
                            "page": payload.get("page"),
                            "per_page": payload.get("per_page"),
                        },
                    )
                )
                for item in items:
                    records.append(
                        _record_from_spiremeta_payload(
                            item,
                            character_id=character_id,
                            source_url=str(response.request.url),
                            snapshot_date=snapshot_date,
                            snapshot_label=snapshot_label,
                            game_version=game_version,
                            source_type=source_type,
                        )
                    )
                if max_pages is not None and page >= max_pages:
                    break
                if page >= total_pages or not items:
                    break
                page += 1
    finally:
        if owns_client:
            http_client.close()

    manifest = _build_source_manifest(
        source_name=SPIREMETA_DEFAULT_SOURCE_NAME,
        source_kind="spiremeta_api",
        source_url=f"{api_base_url.rstrip('/')}{SPIREMETA_GLOBAL_CARDS_PATH}",
        snapshot_date=snapshot_date,
        snapshot_label=snapshot_label,
        game_version=game_version,
        output_dir=output_dir,
        raw_payload_root=raw_payload_root,
        fetch_started_at_utc=fetch_started_at_utc,
        fetch_completed_at_utc=datetime.now(UTC).isoformat(),
        request_count=request_count,
        request_parameters={
            "characters": normalized_characters,
            "per_page": per_page,
            "max_pages": max_pages,
            "source_type": source_type,
            "api_base_url": api_base_url.rstrip("/"),
        },
        source_files=source_files,
        metadata={
            "api_key_env_var": SPIREMETA_API_KEY_ENV_VAR,
            "auth_mode": "bearer",
        },
    )
    return _persist_community_card_stats_artifacts(
        output_dir=output_dir,
        records=records,
        source_path=Path(f"{api_base_url.rstrip('/')}{SPIREMETA_GLOBAL_CARDS_PATH}"),
        source_manifest=manifest,
        raw_payload_root=raw_payload_root,
    )


def _load_import_payload(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader], {}
    if suffix == ".jsonl":
        payloads = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return payloads, {}
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload], {}
        if isinstance(payload, dict):
            records = payload.get("records", [])
            metadata = {key: value for key, value in payload.items() if key != "records"}
            return [dict(item) for item in records], metadata
    raise ValueError(f"Unsupported community card stats format: {path.suffix}")


def _record_from_raw_payload(
    payload: dict[str, Any],
    *,
    source_name: str | None,
    source_url: str | None,
    snapshot_date: str | None,
    snapshot_label: str | None,
    game_version: str | None,
) -> CommunityCardStatRecord:
    normalized_payload = {key: value for key, value in payload.items()}
    normalized_payload["source_name"] = source_name or normalized_payload.get("source_name")
    normalized_payload["source_url"] = source_url or normalized_payload.get("source_url")
    normalized_payload["snapshot_date"] = snapshot_date or normalized_payload.get("snapshot_date") or str(date.today())
    normalized_payload["snapshot_label"] = snapshot_label or normalized_payload.get("snapshot_label")
    normalized_payload["game_version"] = game_version or normalized_payload.get("game_version")
    normalized_payload["source_type"] = _normalize_source_type(normalized_payload.get("source_type"))
    for field_name in (
        "offer_count",
        "pick_count",
        "shop_offer_count",
        "buy_count",
        "deck_presence_runs",
        "run_count",
        "ascension_min",
        "ascension_max",
    ):
        normalized_payload[field_name] = _optional_int(normalized_payload.get(field_name))
    for field_name in ("pick_rate", "buy_rate", "win_rate_with_card", "win_delta"):
        normalized_payload[field_name] = _optional_float(normalized_payload.get(field_name))
    metadata = normalized_payload.get("metadata")
    if not isinstance(metadata, dict):
        normalized_payload["metadata"] = {}
    return CommunityCardStatRecord.model_validate(normalized_payload)


def _record_from_spiremeta_payload(
    payload: dict[str, Any],
    *,
    character_id: str,
    source_url: str,
    snapshot_date: str | None,
    snapshot_label: str | None,
    game_version: str | None,
    source_type: CommunityCardSourceType,
) -> CommunityCardStatRecord:
    slug = _normalize_optional_string(payload.get("slug"))
    card_name = _normalize_optional_string(payload.get("name"))
    canonical_card_id = _canonicalize_spiremeta_card_id(
        slug=slug,
        card_name=card_name,
        character_id=character_id,
    )
    aliases = _build_spiremeta_card_aliases(
        canonical_card_id=canonical_card_id,
        slug=slug,
        card_name=card_name,
        character_id=character_id,
    )
    metadata = {
        "aliases": aliases,
        "spiremeta_numeric_card_id": _optional_int(payload.get("card_id")),
        "spiremeta_slug": slug,
        "spiremeta_card_type": _normalize_optional_string(payload.get("card_type")),
        "spiremeta_rarity": _normalize_optional_string(payload.get("rarity")),
        "spiremeta_cost": _optional_int(payload.get("cost")),
        "spiremeta_image_url": _normalize_optional_string(payload.get("image_url")),
        "spiremeta_avg_floor_obtained": _optional_float(payload.get("avg_floor_obtained")),
        "spiremeta_runs_with_card": _optional_int(payload.get("runs_with_card")),
        "spiremeta_pick_rate_raw": _optional_float(payload.get("pick_rate")),
        "spiremeta_win_rate_raw": _optional_float(payload.get("win_rate")),
    }
    return CommunityCardStatRecord.model_validate(
        {
            "source_name": SPIREMETA_DEFAULT_SOURCE_NAME,
            "source_url": source_url,
            "snapshot_date": snapshot_date or str(date.today()),
            "snapshot_label": snapshot_label,
            "game_version": game_version,
            "character_id": character_id,
            "source_type": source_type,
            "card_id": canonical_card_id,
            "card_name": card_name,
            "offer_count": _optional_int(payload.get("times_offered")),
            "pick_count": _optional_int(payload.get("times_picked")),
            "pick_rate": _coerce_rate_fraction(payload.get("pick_rate")),
            "deck_presence_runs": _optional_int(payload.get("runs_with_card")),
            "run_count": None,
            "win_rate_with_card": _coerce_rate_fraction(payload.get("win_rate")),
            "win_delta": None,
            "metadata": metadata,
        }
    )


def _build_source_manifest(
    *,
    source_name: str,
    source_kind: CommunityCardStatsSourceKind,
    source_url: str | None,
    snapshot_date: str | None,
    snapshot_label: str | None,
    game_version: str | None,
    output_dir: Path,
    raw_payload_root: Path | None,
    fetch_started_at_utc: str | None,
    fetch_completed_at_utc: str | None,
    request_count: int,
    request_parameters: dict[str, Any],
    source_files: list[CommunityCardStatsSourceFile],
    metadata: dict[str, Any] | None = None,
) -> CommunityCardStatsSourceManifest:
    placeholder = output_dir / "_placeholder"
    return CommunityCardStatsSourceManifest(
        source_name=source_name,
        source_kind=source_kind,
        source_url=source_url,
        snapshot_date=snapshot_date or str(date.today()),
        snapshot_label=snapshot_label,
        game_version=game_version,
        generated_at_utc=datetime.now(UTC).isoformat(),
        fetch_started_at_utc=fetch_started_at_utc,
        fetch_completed_at_utc=fetch_completed_at_utc,
        output_dir=str(output_dir),
        records_path=str(placeholder),
        table_path=str(placeholder),
        summary_path=str(placeholder),
        raw_payload_root=None if raw_payload_root is None else str(raw_payload_root),
        request_count=request_count,
        request_parameters=request_parameters,
        source_files=source_files,
        metadata=metadata or {},
    )


def _normalize_source_type(value: Any) -> CommunityCardSourceType:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return "unknown"
    lowered = normalized.lower()
    if lowered in {"reward", "shop", "event", "colorless", "starter", "unknown"}:
        return lowered  # type: ignore[return-value]
    return "unknown"


def _normalize_snapshot_date(value: Any) -> str:
    text = _normalize_optional_string(value)
    if text is None:
        return str(date.today())
    if "T" in text:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    return date.fromisoformat(text).isoformat()


def _normalize_required_timestamp(value: Any) -> str:
    text = _normalize_optional_string(value)
    if text is None:
        raise ValueError("timestamp is required.")
    return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()


def _normalize_optional_timestamp(value: Any) -> str | None:
    text = _normalize_optional_string(value)
    if text is None:
        return None
    return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()


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


def _optional_float(value: Any) -> float | None:
    text = _normalize_optional_string(value)
    if text is None:
        return None
    return float(text)


def _write_records(path: Path, records: list[CommunityCardStatRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False))
            handle.write("\n")


def _write_table(path: Path, records: list[CommunityCardStatRecord]) -> None:
    fieldnames = [
        "source_name",
        "snapshot_date",
        "snapshot_label",
        "game_version",
        "character_id",
        "ascension_min",
        "ascension_max",
        "act_id",
        "floor_band",
        "source_type",
        "card_id",
        "card_name",
        "offer_count",
        "pick_count",
        "pick_rate",
        "shop_offer_count",
        "buy_count",
        "buy_rate",
        "deck_presence_runs",
        "run_count",
        "win_rate_with_card",
        "win_delta",
        "source_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            payload = record.as_dict()
            writer.writerow({key: payload.get(key) for key in fieldnames})


def _build_summary(
    *,
    records: list[CommunityCardStatRecord],
    source_path: Path,
    output_dir: Path,
    records_path: Path,
    table_path: Path,
    summary_path: Path,
    source_manifest_path: Path | None = None,
    source_manifest: CommunityCardStatsSourceManifest | None = None,
) -> dict[str, Any]:
    source_file_kind_histogram = (
        dict(Counter(source_file.kind for source_file in source_manifest.source_files))
        if source_manifest is not None
        else {}
    )
    return {
        "schema_version": COMMUNITY_CARD_STATS_SCHEMA_VERSION,
        "artifact_kind": "community_card_stats",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "records_path": str(records_path),
        "table_path": str(table_path),
        "summary_path": str(summary_path),
        "source_manifest_path": None if source_manifest_path is None else str(source_manifest_path),
        "source_kind": None if source_manifest is None else source_manifest.source_kind,
        "source_request_count": 0 if source_manifest is None else source_manifest.request_count,
        "source_request_parameters": {} if source_manifest is None else source_manifest.request_parameters,
        "raw_payload_root": None if source_manifest is None else source_manifest.raw_payload_root,
        "raw_payload_file_count": 0 if source_manifest is None else len(source_manifest.source_files),
        "source_file_kind_histogram": source_file_kind_histogram,
        "fetch_started_at_utc": None if source_manifest is None else source_manifest.fetch_started_at_utc,
        "fetch_completed_at_utc": None if source_manifest is None else source_manifest.fetch_completed_at_utc,
        "record_count": len(records),
        "card_count": len({record.card_id for record in records}),
        "source_histogram": dict(Counter(record.source_name for record in records)),
        "snapshot_date_histogram": dict(Counter(record.snapshot_date for record in records)),
        "snapshot_label_histogram": dict(Counter(record.snapshot_label for record in records if record.snapshot_label)),
        "character_histogram": dict(Counter(record.character_id for record in records if record.character_id)),
        "source_type_histogram": dict(Counter(record.source_type for record in records)),
        "act_histogram": dict(Counter(record.act_id for record in records if record.act_id)),
        "floor_band_histogram": dict(Counter(record.floor_band for record in records if record.floor_band)),
        "game_version_histogram": dict(Counter(record.game_version for record in records if record.game_version)),
        "pick_rate_stats": _basic_stats([record.pick_rate for record in records if record.pick_rate is not None]),
        "buy_rate_stats": _basic_stats([record.buy_rate for record in records if record.buy_rate is not None]),
        "win_rate_with_card_stats": _basic_stats(
            [record.win_rate_with_card for record in records if record.win_rate_with_card is not None]
        ),
        "win_delta_stats": _basic_stats([record.win_delta for record in records if record.win_delta is not None]),
        "offer_count_stats": _basic_stats([float(record.offer_count) for record in records if record.offer_count is not None]),
        "run_count_stats": _basic_stats([float(record.run_count) for record in records if record.run_count is not None]),
        "top_pick_rate_cards": _top_records(records, key_name="pick_rate"),
        "top_buy_rate_cards": _top_records(records, key_name="buy_rate"),
        "top_win_delta_cards": _top_records(records, key_name="win_delta"),
    }


def _top_records(records: list[CommunityCardStatRecord], *, key_name: str, limit: int = 10) -> list[dict[str, Any]]:
    eligible = [record for record in records if getattr(record, key_name) is not None]
    eligible.sort(key=lambda item: float(getattr(item, key_name) or 0.0), reverse=True)
    return [
        {
            "card_id": record.card_id,
            "card_name": record.card_name,
            "character_id": record.character_id,
            "source_type": record.source_type,
            key_name: getattr(record, key_name),
            "source_name": record.source_name,
            "snapshot_date": record.snapshot_date,
        }
        for record in eligible[:limit]
    ]


def _basic_stats(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _prepare_output_dir(*, output_root: str | Path, session_name: str | None, replace_existing: bool) -> Path:
    output_dir = Path(output_root).expanduser().resolve() / (
        session_name or default_community_card_stats_session_name()
    )
    if output_dir.exists():
        if not replace_existing:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _persist_community_card_stats_artifacts(
    *,
    output_dir: Path,
    records: list[CommunityCardStatRecord],
    source_path: Path,
    source_manifest: CommunityCardStatsSourceManifest,
    raw_payload_root: Path | None,
) -> CommunityCardStatsImportReport:
    records_path = output_dir / COMMUNITY_CARD_STATS_FILENAME
    table_path = output_dir / COMMUNITY_CARD_STATS_TABLE_FILENAME
    summary_path = output_dir / COMMUNITY_CARD_STATS_SUMMARY_FILENAME
    manifest_path = output_dir / COMMUNITY_CARD_STATS_SOURCE_MANIFEST_FILENAME
    manifest = source_manifest.model_copy(
        update={
            "output_dir": str(output_dir),
            "records_path": str(records_path),
            "table_path": str(table_path),
            "summary_path": str(summary_path),
            "raw_payload_root": None if raw_payload_root is None else str(raw_payload_root),
        }
    )
    _write_records(records_path, records)
    _write_table(table_path, records)
    summary_payload = _build_summary(
        records=records,
        source_path=source_path,
        output_dir=output_dir,
        records_path=records_path,
        table_path=table_path,
        summary_path=summary_path,
        source_manifest_path=manifest_path,
        source_manifest=manifest,
    )
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return CommunityCardStatsImportReport(
        output_dir=output_dir,
        records_path=records_path,
        table_path=table_path,
        summary_path=summary_path,
        record_count=len(records),
        card_count=len({record.card_id for record in records}),
        source_manifest_path=manifest_path,
        raw_payload_root=raw_payload_root,
    )


def _resolve_spiremeta_api_key(api_key: str | None) -> str:
    resolved = _normalize_optional_string(api_key) or _normalize_optional_string(os.getenv(SPIREMETA_API_KEY_ENV_VAR))
    if resolved is None:
        raise ValueError(
            f"Provide a SpireMeta API key with --api-key or set {SPIREMETA_API_KEY_ENV_VAR} in the environment."
        )
    return resolved


def _normalize_spiremeta_character_list(characters: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in characters:
        character_id = _slug_token(item)
        if not character_id:
            continue
        if character_id in seen:
            continue
        seen.add(character_id)
        normalized.append(character_id)
    return normalized


def _canonicalize_spiremeta_card_id(*, slug: str | None, card_name: str | None, character_id: str) -> str:
    candidate = _slug_token(slug) or _slug_token(card_name)
    if candidate is None:
        raise ValueError("SpireMeta card payload is missing both slug and name.")
    if candidate in {"STRIKE", "DEFEND"}:
        return f"{candidate}_{character_id}"
    return candidate


def _build_spiremeta_card_aliases(
    *,
    canonical_card_id: str,
    slug: str | None,
    card_name: str | None,
    character_id: str,
) -> list[str]:
    aliases: list[str] = []
    for candidate in (canonical_card_id, _slug_token(slug), _slug_token(card_name)):
        if candidate and candidate not in aliases:
            aliases.append(candidate)
    if canonical_card_id.startswith("STRIKE_") and "STRIKE" not in aliases:
        aliases.append("STRIKE")
    if canonical_card_id.startswith("DEFEND_") and "DEFEND" not in aliases:
        aliases.append("DEFEND")
    if "STRIKE" in aliases and f"STRIKE_{character_id}" not in aliases:
        aliases.append(f"STRIKE_{character_id}")
    if "DEFEND" in aliases and f"DEFEND_{character_id}" not in aliases:
        aliases.append(f"DEFEND_{character_id}")
    return aliases


def _slug_token(value: Any) -> str | None:
    text = _normalize_optional_string(value)
    if text is None:
        return None
    token = _TOKEN_PATTERN.sub("_", text).strip("_").upper()
    return token or None


def _coerce_rate_fraction(value: Any) -> float | None:
    numeric = _optional_float(value)
    if numeric is None:
        return None
    if 1.0 < numeric <= 100.0:
        return numeric / 100.0
    return numeric


def _write_spiremeta_raw_payload(
    *,
    raw_payload_root: Path,
    character_id: str,
    page: int,
    payload: dict[str, Any],
) -> Path:
    character_root = raw_payload_root / character_id.lower()
    character_root.mkdir(parents=True, exist_ok=True)
    payload_path = character_root / f"page-{page:04d}.json"
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload_path


def _extract_rate_limit_headers(headers: httpx.Headers) -> dict[str, str]:
    payload: dict[str, str] = {}
    for header_name in ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"):
        value = headers.get(header_name)
        if value is not None:
            payload[header_name] = value
    return payload


def _raise_for_spiremeta_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = json.dumps(response.json(), ensure_ascii=False)
    except ValueError:
        detail = response.text[:500]
    raise RuntimeError(
        "SpireMeta API request failed "
        f"({response.status_code}) for {response.request.url}: {detail}"
    )


def _sha256_hex(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
