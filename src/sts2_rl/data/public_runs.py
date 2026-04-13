from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

PUBLIC_RUN_ARCHIVE_SCHEMA_VERSION = 1
PUBLIC_RUN_ARCHIVE_INDEX_FILENAME = "public-run-index.jsonl"
PUBLIC_RUN_ARCHIVE_DETAILS_FILENAME = "public-run-details.jsonl"
PUBLIC_RUN_ARCHIVE_SUMMARY_FILENAME = "summary.json"
PUBLIC_RUN_ARCHIVE_STATE_FILENAME = "sync-state.json"
PUBLIC_RUN_ARCHIVE_SOURCE_MANIFEST_FILENAME = "source-manifest.json"
PUBLIC_RUN_ARCHIVE_RAW_DIRNAME = "raw"
PUBLIC_RUN_ARCHIVE_SYNC_DIRNAME = "syncs"

STS2RUNS_SOURCE_NAME = "sts2runs"
STS2RUNS_SOURCE_KIND = "sts2runs_api"
STS2RUNS_DEFAULT_BASE_URL = "https://sts2runs.com"
STS2RUNS_LIST_PATH = "/api/runs"
STS2RUNS_DETAIL_PATH_TEMPLATE = "/api/runs/{run_id}"


class PublicRunArchiveIndexRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PUBLIC_RUN_ARCHIVE_SCHEMA_VERSION
    record_type: Literal["public_run_index"] = "public_run_index"
    source_name: str = STS2RUNS_SOURCE_NAME
    source_run_id: int
    user_id: int | None = None
    sha256_hex: str | None = None
    seed: str | None = None
    start_time_unix: int | None = None
    character_id: str | None = None
    ascension: int | None = None
    win: bool | None = None
    was_abandoned: bool | None = None
    killed_by: str | None = None
    floors_reached: int | None = None
    run_time_seconds: int | None = None
    deck_size: int | None = None
    relic_count: int | None = None
    build_id: str | None = None
    profile: str | None = None
    source_url: str | None = None
    list_page: int
    uploaded_at_unix: int | None = None
    fetched_at_utc: str
    raw_payload_path: str
    dedupe_key: str
    identity_fingerprint: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record(self) -> PublicRunArchiveIndexRecord:
        self.fetched_at_utc = _normalize_required_timestamp(self.fetched_at_utc)
        self.seed = _normalize_optional_string(self.seed)
        self.character_id = _normalize_optional_string(self.character_id)
        self.killed_by = _normalize_optional_string(self.killed_by)
        self.build_id = _normalize_optional_string(self.build_id)
        self.profile = _normalize_optional_string(self.profile)
        self.source_url = _normalize_optional_string(self.source_url)
        self.sha256_hex = _normalize_optional_string(self.sha256_hex)
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PublicRunArchiveDetailRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PUBLIC_RUN_ARCHIVE_SCHEMA_VERSION
    record_type: Literal["public_run_detail"] = "public_run_detail"
    source_name: str = STS2RUNS_SOURCE_NAME
    source_run_id: int
    user_id: int | None = None
    source_url: str
    fetched_at_utc: str
    raw_payload_path: str
    raw_payload_sha256: str
    detail_root_keys: list[str] = Field(default_factory=list)
    run_root_keys: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record(self) -> PublicRunArchiveDetailRecord:
        self.fetched_at_utc = _normalize_required_timestamp(self.fetched_at_utc)
        self.source_url = str(self.source_url).strip()
        self.raw_payload_sha256 = str(self.raw_payload_sha256).strip()
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PublicRunArchiveFailureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_run_id: int
    attempt_count: int = 0
    last_error: str
    last_attempt_at_utc: str

    @model_validator(mode="after")
    def validate_record(self) -> PublicRunArchiveFailureRecord:
        self.last_error = str(self.last_error).strip()
        self.last_attempt_at_utc = _normalize_required_timestamp(self.last_attempt_at_utc)
        return self


class PublicRunArchiveState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PUBLIC_RUN_ARCHIVE_SCHEMA_VERSION
    artifact_kind: Literal["public_run_archive_state"] = "public_run_archive_state"
    source_name: str = STS2RUNS_SOURCE_NAME
    source_kind: Literal["sts2runs_api"] = STS2RUNS_SOURCE_KIND
    archive_root: str
    source_base_url: str
    list_path: str
    detail_path_template: str
    created_at_utc: str
    updated_at_utc: str
    session_count: int = 0
    total_list_requests: int = 0
    total_detail_requests: int = 0
    known_run_count: int = 0
    detailed_run_count: int = 0
    highest_source_run_id: int | None = None
    highest_uploaded_at_unix: int | None = None
    pending_detail_run_ids: list[int] = Field(default_factory=list)
    detail_failures: list[PublicRunArchiveFailureRecord] = Field(default_factory=list)
    last_sync_session: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> PublicRunArchiveState:
        self.created_at_utc = _normalize_required_timestamp(self.created_at_utc)
        self.updated_at_utc = _normalize_required_timestamp(self.updated_at_utc)
        self.pending_detail_run_ids = sorted({int(run_id) for run_id in self.pending_detail_run_ids}, reverse=True)
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class PublicRunArchiveSourceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = PUBLIC_RUN_ARCHIVE_SCHEMA_VERSION
    artifact_kind: Literal["public_run_archive_source_manifest"] = "public_run_archive_source_manifest"
    source_name: str = STS2RUNS_SOURCE_NAME
    source_kind: Literal["sts2runs_api"] = STS2RUNS_SOURCE_KIND
    source_base_url: str
    list_path: str
    detail_path_template: str
    archive_root: str
    index_path: str
    details_path: str
    summary_path: str
    state_path: str
    raw_payload_root: str
    sync_root: str
    generated_at_utc: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_manifest(self) -> PublicRunArchiveSourceManifest:
        self.generated_at_utc = _normalize_required_timestamp(self.generated_at_utc)
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


@dataclass(frozen=True)
class PublicRunArchiveSyncReport:
    archive_root: Path
    index_path: Path
    details_path: Path
    summary_path: Path
    state_path: Path
    source_manifest_path: Path
    session_root: Path
    session_summary_path: Path
    new_run_count: int
    duplicate_run_count: int
    duplicate_sha256_count: int
    detail_fetched_count: int
    pending_detail_run_count: int
    failed_detail_run_count: int
    total_run_count: int
    detailed_run_count: int
    list_page_count: int
    detail_request_count: int


def default_public_run_archive_sync_session_name(prefix: str = "sts2runs-sync") -> str:
    return datetime.now(UTC).strftime(f"{prefix}-%Y%m%d-%H%M%S")


def sync_sts2runs_public_run_archive(
    *,
    archive_root: str | Path,
    session_name: str | None = None,
    limit: int = 100,
    max_list_pages: int | None = None,
    max_detail_fetches: int | None = None,
    stop_after_consecutive_known_pages: int = 2,
    initial_page: int = 0,
    source_base_url: str = STS2RUNS_DEFAULT_BASE_URL,
    request_timeout_seconds: float = 30.0,
    max_retries: int = 3,
    retry_backoff_seconds: float = 0.5,
    replace_existing_archive: bool = False,
    client: httpx.Client | None = None,
    transport: httpx.BaseTransport | None = None,
) -> PublicRunArchiveSyncReport:
    if client is not None and transport is not None:
        raise ValueError("Provide either client or transport, not both.")
    if limit < 1:
        raise ValueError("limit must be >= 1.")
    if max_list_pages is not None and max_list_pages < 1:
        raise ValueError("max_list_pages must be >= 1.")
    if max_detail_fetches is not None and max_detail_fetches < 1:
        raise ValueError("max_detail_fetches must be >= 1.")
    if stop_after_consecutive_known_pages < 1:
        raise ValueError("stop_after_consecutive_known_pages must be >= 1.")
    if initial_page < 0:
        raise ValueError("initial_page must be >= 0.")

    resolved_root = Path(archive_root).expanduser().resolve()
    if replace_existing_archive and resolved_root.exists():
        shutil.rmtree(resolved_root)
    resolved_root.mkdir(parents=True, exist_ok=True)
    session_name = session_name or default_public_run_archive_sync_session_name()

    index_path = resolved_root / PUBLIC_RUN_ARCHIVE_INDEX_FILENAME
    details_path = resolved_root / PUBLIC_RUN_ARCHIVE_DETAILS_FILENAME
    summary_path = resolved_root / PUBLIC_RUN_ARCHIVE_SUMMARY_FILENAME
    state_path = resolved_root / PUBLIC_RUN_ARCHIVE_STATE_FILENAME
    source_manifest_path = resolved_root / PUBLIC_RUN_ARCHIVE_SOURCE_MANIFEST_FILENAME
    raw_root = resolved_root / PUBLIC_RUN_ARCHIVE_RAW_DIRNAME
    list_raw_root = raw_root / "list-pages" / session_name
    detail_raw_root = raw_root / "run-details" / session_name
    sync_root = resolved_root / PUBLIC_RUN_ARCHIVE_SYNC_DIRNAME
    session_root = sync_root / session_name
    list_raw_root.mkdir(parents=True, exist_ok=True)
    detail_raw_root.mkdir(parents=True, exist_ok=True)
    session_root.mkdir(parents=True, exist_ok=True)

    state = load_public_run_archive_state(resolved_root) if state_path.exists() else PublicRunArchiveState(
        archive_root=str(resolved_root),
        source_base_url=source_base_url.rstrip("/"),
        list_path=STS2RUNS_LIST_PATH,
        detail_path_template=STS2RUNS_DETAIL_PATH_TEMPLATE,
        created_at_utc=datetime.now(UTC).isoformat(),
        updated_at_utc=datetime.now(UTC).isoformat(),
    )
    index_records = load_public_run_archive_index_records(resolved_root) if index_path.exists() else []
    detail_records = load_public_run_archive_detail_records(resolved_root) if details_path.exists() else []
    known_by_id = {record.source_run_id: record for record in index_records}
    detail_by_id = {record.source_run_id: record for record in detail_records}
    known_sha256 = Counter(record.sha256_hex for record in known_by_id.values() if record.sha256_hex)

    owns_client = client is None
    http_client = client or httpx.Client(
        base_url=source_base_url.rstrip("/"),
        timeout=request_timeout_seconds,
        transport=transport,
    )
    new_run_count = 0
    duplicate_run_count = 0
    duplicate_sha256_count = 0
    detail_fetched_count = 0
    list_page_count = 0
    detail_request_count = 0
    consecutive_known_pages = 0
    stopped_reason = "completed"
    started_at_utc = datetime.now(UTC).isoformat()
    try:
        page = initial_page
        while True:
            if max_list_pages is not None and list_page_count >= max_list_pages:
                stopped_reason = "max_list_pages"
                break
            response = _request_with_retry(
                http_client,
                "GET",
                STS2RUNS_LIST_PATH,
                params={"page": page, "limit": limit},
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            state.total_list_requests += 1
            list_page_count += 1
            payload = response.json()
            page_path = list_raw_root / f"page-{page:06d}.json"
            page_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            runs = payload.get("runs") or []
            if not runs:
                stopped_reason = "empty_page"
                break

            page_new_run_count = 0
            fetched_at_utc = datetime.now(UTC).isoformat()
            for item in runs:
                record = _build_index_record(
                    payload=item,
                    page=page,
                    fetched_at_utc=fetched_at_utc,
                    raw_payload_path=page_path,
                    source_url=str(response.request.url),
                )
                if record.source_run_id in known_by_id:
                    duplicate_run_count += 1
                    continue
                if record.sha256_hex is not None and known_sha256[record.sha256_hex] > 0:
                    duplicate_sha256_count += 1
                known_by_id[record.source_run_id] = record
                index_records.append(record)
                page_new_run_count += 1
                new_run_count += 1
                if record.source_run_id not in detail_by_id:
                    state.pending_detail_run_ids.append(record.source_run_id)
                if state.highest_source_run_id is None or record.source_run_id > state.highest_source_run_id:
                    state.highest_source_run_id = record.source_run_id
                if record.uploaded_at_unix is not None and (
                    state.highest_uploaded_at_unix is None or record.uploaded_at_unix > state.highest_uploaded_at_unix
                ):
                    state.highest_uploaded_at_unix = record.uploaded_at_unix
                if record.sha256_hex is not None:
                    known_sha256[record.sha256_hex] += 1
            consecutive_known_pages = consecutive_known_pages + 1 if page_new_run_count == 0 else 0
            state.pending_detail_run_ids = sorted({*state.pending_detail_run_ids, *[run_id for run_id in known_by_id if run_id not in detail_by_id]}, reverse=True)
            _persist_public_run_archive(
                index_path=index_path,
                detail_path=details_path,
                summary_path=summary_path,
                state_path=state_path,
                source_manifest_path=source_manifest_path,
                resolved_root=resolved_root,
                raw_root=raw_root,
                sync_root=sync_root,
                state=state,
                index_records=index_records,
                detail_records=list(detail_by_id.values()),
                source_base_url=source_base_url.rstrip("/"),
                session_name=session_name,
            )
            if len(runs) < limit:
                stopped_reason = "short_page"
                break
            if consecutive_known_pages >= stop_after_consecutive_known_pages:
                stopped_reason = "consecutive_known_pages"
                break
            page += 1

        failures_by_id = {failure.source_run_id: failure for failure in state.detail_failures}
        pending_ids = list(state.pending_detail_run_ids)
        for run_id in pending_ids:
            if max_detail_fetches is not None and detail_request_count >= max_detail_fetches:
                break
            try:
                response = _request_with_retry(
                    http_client,
                    "GET",
                    STS2RUNS_DETAIL_PATH_TEMPLATE.format(run_id=run_id),
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                detail_request_count += 1
                state.total_detail_requests += 1
                payload = response.json()
                detail_path_raw = detail_raw_root / f"run-{run_id:07d}.json"
                detail_path_raw.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                detail_by_id[run_id] = _build_detail_record(
                    source_run_id=run_id,
                    payload=payload,
                    fetched_at_utc=datetime.now(UTC).isoformat(),
                    raw_payload_path=detail_path_raw,
                    source_url=str(response.request.url),
                )
                failures_by_id.pop(run_id, None)
                detail_fetched_count += 1
            except Exception as exc:
                detail_request_count += 1
                state.total_detail_requests += 1
                prior = failures_by_id.get(run_id)
                failures_by_id[run_id] = PublicRunArchiveFailureRecord(
                    source_run_id=run_id,
                    attempt_count=1 if prior is None else prior.attempt_count + 1,
                    last_error=str(exc),
                    last_attempt_at_utc=datetime.now(UTC).isoformat(),
                )
            state.pending_detail_run_ids = sorted(run_id for run_id in known_by_id if run_id not in detail_by_id)
            state.detail_failures = sorted(failures_by_id.values(), key=lambda record: record.source_run_id, reverse=True)
            _persist_public_run_archive(
                index_path=index_path,
                detail_path=details_path,
                summary_path=summary_path,
                state_path=state_path,
                source_manifest_path=source_manifest_path,
                resolved_root=resolved_root,
                raw_root=raw_root,
                sync_root=sync_root,
                state=state,
                index_records=list(known_by_id.values()),
                detail_records=list(detail_by_id.values()),
                source_base_url=source_base_url.rstrip("/"),
                session_name=session_name,
            )
    finally:
        if owns_client:
            http_client.close()

    state.pending_detail_run_ids = sorted(run_id for run_id in known_by_id if run_id not in detail_by_id)
    state.detail_failures = sorted(state.detail_failures, key=lambda record: record.source_run_id, reverse=True)
    state.session_count += 1
    state.known_run_count = len(known_by_id)
    state.detailed_run_count = len(detail_by_id)
    state.last_sync_session = session_name
    state.updated_at_utc = datetime.now(UTC).isoformat()
    _persist_public_run_archive(
        index_path=index_path,
        detail_path=details_path,
        summary_path=summary_path,
        state_path=state_path,
        source_manifest_path=source_manifest_path,
        resolved_root=resolved_root,
        raw_root=raw_root,
        sync_root=sync_root,
        state=state,
        index_records=list(known_by_id.values()),
        detail_records=list(detail_by_id.values()),
        source_base_url=source_base_url.rstrip("/"),
        session_name=session_name,
    )
    session_summary_path = session_root / "sync-summary.json"
    session_summary_path.write_text(
        json.dumps(
            {
                "schema_version": PUBLIC_RUN_ARCHIVE_SCHEMA_VERSION,
                "artifact_kind": "public_run_archive_sync_session",
                "session_name": session_name,
                "archive_root": str(resolved_root),
                "started_at_utc": started_at_utc,
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "stopped_reason": stopped_reason,
                "list_page_count": list_page_count,
                "detail_request_count": detail_request_count,
                "new_run_count": new_run_count,
                "duplicate_run_count": duplicate_run_count,
                "duplicate_sha256_count": duplicate_sha256_count,
                "detail_fetched_count": detail_fetched_count,
                "pending_detail_run_count_after": len(state.pending_detail_run_ids),
                "failed_detail_run_count_after": len(state.detail_failures),
                "highest_source_run_id": state.highest_source_run_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return PublicRunArchiveSyncReport(
        archive_root=resolved_root,
        index_path=index_path,
        details_path=details_path,
        summary_path=summary_path,
        state_path=state_path,
        source_manifest_path=source_manifest_path,
        session_root=session_root,
        session_summary_path=session_summary_path,
        new_run_count=new_run_count,
        duplicate_run_count=duplicate_run_count,
        duplicate_sha256_count=duplicate_sha256_count,
        detail_fetched_count=detail_fetched_count,
        pending_detail_run_count=len(state.pending_detail_run_ids),
        failed_detail_run_count=len(state.detail_failures),
        total_run_count=len(known_by_id),
        detailed_run_count=len(detail_by_id),
        list_page_count=list_page_count,
        detail_request_count=detail_request_count,
    )


def load_public_run_archive_index_records(source: str | Path) -> list[PublicRunArchiveIndexRecord]:
    path = resolve_public_run_archive_index_path(source)
    records: list[PublicRunArchiveIndexRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PublicRunArchiveIndexRecord.model_validate(json.loads(line)))
    return records


def load_public_run_archive_detail_records(source: str | Path) -> list[PublicRunArchiveDetailRecord]:
    path = resolve_public_run_archive_details_path(source)
    records: list[PublicRunArchiveDetailRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PublicRunArchiveDetailRecord.model_validate(json.loads(line)))
    return records


def load_public_run_archive_state(source: str | Path) -> PublicRunArchiveState:
    path = resolve_public_run_archive_state_path(source)
    return PublicRunArchiveState.model_validate(json.loads(path.read_text(encoding="utf-8")))


def load_public_run_archive_summary(source: str | Path) -> dict[str, Any]:
    path = resolve_public_run_archive_summary_path(source)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_public_run_archive_index_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    path = source_path / PUBLIC_RUN_ARCHIVE_INDEX_FILENAME if source_path.is_dir() else source_path
    if not path.exists():
        raise FileNotFoundError(f"Public run archive index does not exist: {path}")
    return path


def resolve_public_run_archive_details_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    path = source_path / PUBLIC_RUN_ARCHIVE_DETAILS_FILENAME if source_path.is_dir() else source_path
    if not path.exists():
        raise FileNotFoundError(f"Public run archive details do not exist: {path}")
    return path


def resolve_public_run_archive_state_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    path = source_path / PUBLIC_RUN_ARCHIVE_STATE_FILENAME if source_path.is_dir() else source_path
    if not path.exists():
        raise FileNotFoundError(f"Public run archive state does not exist: {path}")
    return path


def resolve_public_run_archive_summary_path(source: str | Path) -> Path:
    source_path = Path(source).expanduser().resolve()
    path = source_path / PUBLIC_RUN_ARCHIVE_SUMMARY_FILENAME if source_path.is_dir() else source_path
    if not path.exists():
        raise FileNotFoundError(f"Public run archive summary does not exist: {path}")
    return path


def _build_index_record(
    *,
    payload: dict[str, Any],
    page: int,
    fetched_at_utc: str,
    raw_payload_path: Path,
    source_url: str,
) -> PublicRunArchiveIndexRecord:
    source_run_id = int(payload["id"])
    sha_hex = _normalize_optional_string(payload.get("sha256"))
    seed = _normalize_optional_string(payload.get("seed"))
    character_id = _normalize_optional_string(payload.get("character"))
    start_time_unix = _optional_int(payload.get("start_time"))
    return PublicRunArchiveIndexRecord(
        source_run_id=source_run_id,
        user_id=_optional_int(payload.get("user_id")),
        sha256_hex=sha_hex,
        seed=seed,
        start_time_unix=start_time_unix,
        character_id=character_id,
        ascension=_optional_int(payload.get("ascension")),
        win=_optional_bool(payload.get("win")),
        was_abandoned=_optional_bool(payload.get("was_abandoned")),
        killed_by=_normalize_optional_string(payload.get("killed_by")),
        floors_reached=_optional_int(payload.get("floors_reached")),
        run_time_seconds=_optional_int(payload.get("run_time")),
        deck_size=_optional_int(payload.get("deck_size")),
        relic_count=_optional_int(payload.get("relic_count")),
        build_id=_normalize_optional_string(payload.get("build_id")),
        profile=_normalize_optional_string(payload.get("profile")),
        source_url=source_url,
        list_page=page,
        uploaded_at_unix=_optional_int(payload.get("uploaded_at")),
        fetched_at_utc=fetched_at_utc,
        raw_payload_path=str(raw_payload_path),
        dedupe_key=f"sts2runs:{source_run_id}",
        identity_fingerprint=_identity_fingerprint(
            source_run_id=source_run_id,
            sha256_hex=sha_hex,
            seed=seed,
            start_time_unix=start_time_unix,
            character_id=character_id,
        ),
    )


def _build_detail_record(
    *,
    source_run_id: int,
    payload: dict[str, Any],
    fetched_at_utc: str,
    raw_payload_path: Path,
    source_url: str,
) -> PublicRunArchiveDetailRecord:
    run_payload = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    return PublicRunArchiveDetailRecord(
        source_run_id=source_run_id,
        user_id=_optional_int(payload.get("userId")),
        source_url=source_url,
        fetched_at_utc=fetched_at_utc,
        raw_payload_path=str(raw_payload_path),
        raw_payload_sha256=_sha256_hex(raw_payload_path),
        detail_root_keys=sorted(payload.keys()),
        run_root_keys=sorted(run_payload.keys()),
        metadata={
            "acts_count": len(run_payload.get("acts") or []),
            "map_history_act_count": len(run_payload.get("map_point_history") or []),
            "game_mode": _normalize_optional_string(run_payload.get("game_mode")),
            "build_id": _normalize_optional_string(run_payload.get("build_id")),
        },
    )


def _persist_public_run_archive(
    *,
    index_path: Path,
    detail_path: Path,
    summary_path: Path,
    state_path: Path,
    source_manifest_path: Path,
    resolved_root: Path,
    raw_root: Path,
    sync_root: Path,
    state: PublicRunArchiveState,
    index_records: list[PublicRunArchiveIndexRecord],
    detail_records: list[PublicRunArchiveDetailRecord],
    source_base_url: str,
    session_name: str,
) -> None:
    sorted_index = sorted(index_records, key=lambda record: record.source_run_id, reverse=True)
    sorted_details = sorted(detail_records, key=lambda record: record.source_run_id, reverse=True)
    state.known_run_count = len(sorted_index)
    state.detailed_run_count = len(sorted_details)
    state.updated_at_utc = datetime.now(UTC).isoformat()
    with index_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted_index:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False))
            handle.write("\n")
    with detail_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in sorted_details:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False))
            handle.write("\n")
    manifest = PublicRunArchiveSourceManifest(
        source_base_url=source_base_url,
        list_path=STS2RUNS_LIST_PATH,
        detail_path_template=STS2RUNS_DETAIL_PATH_TEMPLATE,
        archive_root=str(resolved_root),
        index_path=str(index_path),
        details_path=str(detail_path),
        summary_path=str(summary_path),
        state_path=str(state_path),
        raw_payload_root=str(raw_root),
        sync_root=str(sync_root),
        generated_at_utc=datetime.now(UTC).isoformat(),
        metadata={"last_sync_session": session_name},
    )
    source_manifest_path.write_text(json.dumps(manifest.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    state_path.write_text(json.dumps(state.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            _build_summary(
                resolved_root=resolved_root,
                index_path=index_path,
                detail_path=detail_path,
                state_path=state_path,
                source_manifest_path=source_manifest_path,
                index_records=sorted_index,
                detail_records=sorted_details,
                state=state,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_summary(
    *,
    resolved_root: Path,
    index_path: Path,
    detail_path: Path,
    state_path: Path,
    source_manifest_path: Path,
    index_records: list[PublicRunArchiveIndexRecord],
    detail_records: list[PublicRunArchiveDetailRecord],
    state: PublicRunArchiveState,
) -> dict[str, Any]:
    detail_ids = {record.source_run_id for record in detail_records}
    duplicate_sha256_count = sum(
        count - 1
        for count in Counter(record.sha256_hex for record in index_records if record.sha256_hex).values()
        if count > 1
    )
    return {
        "schema_version": PUBLIC_RUN_ARCHIVE_SCHEMA_VERSION,
        "artifact_kind": "public_run_archive",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "archive_root": str(resolved_root),
        "index_path": str(index_path),
        "details_path": str(detail_path),
        "state_path": str(state_path),
        "source_manifest_path": str(source_manifest_path),
        "source_name": STS2RUNS_SOURCE_NAME,
        "source_kind": STS2RUNS_SOURCE_KIND,
        "known_run_count": len(index_records),
        "detailed_run_count": len(detail_records),
        "pending_detail_run_count": len(state.pending_detail_run_ids),
        "failed_detail_run_count": len(state.detail_failures),
        "detail_coverage": 0.0 if not index_records else len(detail_ids) / len(index_records),
        "total_list_requests": state.total_list_requests,
        "total_detail_requests": state.total_detail_requests,
        "session_count": state.session_count,
        "last_sync_session": state.last_sync_session,
        "highest_source_run_id": state.highest_source_run_id,
        "highest_uploaded_at_unix": state.highest_uploaded_at_unix,
        "duplicate_sha256_count": duplicate_sha256_count,
        "character_histogram": dict(Counter(record.character_id for record in index_records if record.character_id)),
        "build_id_histogram": dict(Counter(record.build_id for record in index_records if record.build_id)),
        "ascension_histogram": dict(Counter(str(record.ascension) for record in index_records if record.ascension is not None)),
        "win_histogram": dict(Counter(str(record.win) for record in index_records if record.win is not None)),
        "abandoned_histogram": dict(Counter(str(record.was_abandoned) for record in index_records if record.was_abandoned is not None)),
        "top_floor_runs": [
            {
                "source_run_id": record.source_run_id,
                "character_id": record.character_id,
                "floors_reached": record.floors_reached,
                "ascension": record.ascension,
                "win": record.win,
                "build_id": record.build_id,
            }
            for record in sorted(
                [record for record in index_records if record.floors_reached is not None],
                key=lambda item: int(item.floors_reached or 0),
                reverse=True,
            )[:10]
        ],
    }


def _request_with_retry(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    max_retries: int,
    retry_backoff_seconds: float,
) -> httpx.Response:
    attempt = 0
    while True:
        try:
            response = client.request(method, path, params=params)
            response.raise_for_status()
            return response
        except httpx.HTTPError:
            attempt += 1
            if attempt > max_retries:
                raise
            if retry_backoff_seconds > 0.0:
                time.sleep(retry_backoff_seconds * attempt)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return bool(int(value))


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_required_timestamp(value: Any) -> str:
    text = _normalize_optional_string(value)
    if text is None:
        raise ValueError("timestamp is required.")
    return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat()


def _identity_fingerprint(
    *,
    source_run_id: int,
    sha256_hex: str | None,
    seed: str | None,
    start_time_unix: int | None,
    character_id: str | None,
) -> str:
    components = [
        f"id={source_run_id}",
        f"sha256={sha256_hex or ''}",
        f"seed={seed or ''}",
        f"start={'' if start_time_unix is None else start_time_unix}",
        f"character={character_id or ''}",
    ]
    return sha256("|".join(components).encode("utf-8")).hexdigest()


def _sha256_hex(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
