from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math
import re
from pathlib import Path
from typing import Any, Literal

from sts2_rl.data import (
    CommunityCardStatRecord,
    PublicRunStrategicStatRecord,
    load_community_card_stat_records,
    load_public_run_normalized_summary,
    load_public_run_strategic_stat_records,
)

CommunityPriorDomain = Literal[
    "reward_pick",
    "selection_pick",
    "selection_upgrade",
    "selection_remove",
    "shop_buy",
    "map_node",
]

_REWARD_SOURCE_TYPES = {"reward", "colorless", "event", "starter", "unknown"}
_SHOP_SOURCE_TYPES = {"shop", "unknown"}
_ROUTE_NODE_TYPES = {"rest", "elite", "shop", "treasure", "event", "monster", "boss", "unknown"}
_ACT_PATTERN = re.compile(r"^(?:act|a)[-_ ]?(\d+)$")
_RANGE_PATTERN = re.compile(r"^(\d+)\s*[-:]\s*(\d+)$")
_LE_PATTERN = re.compile(r"^(?:<=?|up\s*to)\s*(\d+)$")
_GE_PATTERN = re.compile(r"^(?:>=?|from)\s*(\d+)$")


@dataclass(frozen=True)
class CommunityPriorRuntimeConfig:
    source_path: Path
    route_source_path: Path | None = None
    reward_pick_weight: float = 1.15
    selection_pick_weight: float = 1.05
    selection_upgrade_weight: float = 0.55
    selection_remove_weight: float = 0.95
    shop_buy_weight: float = 1.00
    route_weight: float = 0.90
    reward_pick_neutral_rate: float = 0.33
    shop_buy_neutral_rate: float = 0.10
    route_neutral_win_rate: float = 0.50
    pick_rate_scale: float = 3.0
    buy_rate_scale: float = 5.0
    win_delta_scale: float = 12.0
    route_win_rate_scale: float = 8.0
    min_sample_size: int = 40
    route_min_sample_size: int = 30
    max_confidence_sample_size: int = 1200
    max_source_age_days: int | None = None

    def __post_init__(self) -> None:
        if self.min_sample_size < 1:
            raise ValueError("community prior min_sample_size must be >= 1.")
        if self.route_min_sample_size < 1:
            raise ValueError("community prior route_min_sample_size must be >= 1.")
        if self.max_confidence_sample_size < self.min_sample_size:
            raise ValueError("community prior max_confidence_sample_size must be >= min_sample_size.")
        for field_name in (
            "reward_pick_weight",
            "selection_pick_weight",
            "selection_upgrade_weight",
            "selection_remove_weight",
            "shop_buy_weight",
            "route_weight",
            "pick_rate_scale",
            "buy_rate_scale",
            "win_delta_scale",
            "route_win_rate_scale",
        ):
            if float(getattr(self, field_name)) < 0.0:
                raise ValueError(f"community prior {field_name} must be >= 0.")
        for field_name in ("reward_pick_neutral_rate", "shop_buy_neutral_rate", "route_neutral_win_rate"):
            value = float(getattr(self, field_name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"community prior {field_name} must be between 0 and 1.")
        if self.max_source_age_days is not None and self.max_source_age_days < 0:
            raise ValueError("community prior max_source_age_days must be >= 0.")

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": str(self.source_path),
            "route_source_path": None if self.route_source_path is None else str(self.route_source_path),
            "reward_pick_weight": self.reward_pick_weight,
            "selection_pick_weight": self.selection_pick_weight,
            "selection_upgrade_weight": self.selection_upgrade_weight,
            "selection_remove_weight": self.selection_remove_weight,
            "shop_buy_weight": self.shop_buy_weight,
            "route_weight": self.route_weight,
            "reward_pick_neutral_rate": self.reward_pick_neutral_rate,
            "shop_buy_neutral_rate": self.shop_buy_neutral_rate,
            "route_neutral_win_rate": self.route_neutral_win_rate,
            "pick_rate_scale": self.pick_rate_scale,
            "buy_rate_scale": self.buy_rate_scale,
            "win_delta_scale": self.win_delta_scale,
            "route_win_rate_scale": self.route_win_rate_scale,
            "min_sample_size": self.min_sample_size,
            "route_min_sample_size": self.route_min_sample_size,
            "max_confidence_sample_size": self.max_confidence_sample_size,
            "max_source_age_days": self.max_source_age_days,
        }


@dataclass(frozen=True)
class CommunityCardPrior:
    domain: CommunityPriorDomain
    card_id: str
    source_type: str
    matched_record_count: int
    sample_count: int
    confidence: float
    pick_rate: float | None
    buy_rate: float | None
    win_delta: float | None
    win_rate_with_card: float | None
    score_bonus: float
    rate_baseline: float | None
    rate_edge: float | None
    specificity_level: int
    matched_character_id: str | None
    matched_act_id: str | None
    matched_floor_band: str | None
    source_name: str | None
    snapshot_date: str | None
    artifact_family: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "card_id": self.card_id,
            "source_type": self.source_type,
            "matched_record_count": self.matched_record_count,
            "sample_count": self.sample_count,
            "confidence": self.confidence,
            "pick_rate": self.pick_rate,
            "buy_rate": self.buy_rate,
            "win_delta": self.win_delta,
            "win_rate_with_card": self.win_rate_with_card,
            "score_bonus": self.score_bonus,
            "rate_baseline": self.rate_baseline,
            "rate_edge": self.rate_edge,
            "specificity_level": self.specificity_level,
            "matched_character_id": self.matched_character_id,
            "matched_act_id": self.matched_act_id,
            "matched_floor_band": self.matched_floor_band,
            "source_name": self.source_name,
            "snapshot_date": self.snapshot_date,
            "artifact_family": self.artifact_family,
        }


@dataclass(frozen=True)
class CommunityRoutePrior:
    domain: Literal["map_node"]
    subject_id: str
    matched_record_count: int
    sample_count: int
    confidence: float
    win_rate: float | None
    win_delta: float | None
    score_bonus: float
    specificity_level: int
    matched_character_id: str | None
    matched_act_id: str | None
    room_type: str | None
    source_name: str | None
    snapshot_date: str | None
    artifact_family: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "subject_id": self.subject_id,
            "matched_record_count": self.matched_record_count,
            "sample_count": self.sample_count,
            "confidence": self.confidence,
            "win_rate": self.win_rate,
            "win_delta": self.win_delta,
            "score_bonus": self.score_bonus,
            "specificity_level": self.specificity_level,
            "matched_character_id": self.matched_character_id,
            "matched_act_id": self.matched_act_id,
            "room_type": self.room_type,
            "source_name": self.source_name,
            "snapshot_date": self.snapshot_date,
            "artifact_family": self.artifact_family,
        }


@dataclass(frozen=True)
class CommunityPriorSourceDiagnostics:
    domain: str
    source_path: Path
    artifact_family: str
    stat_family: str | None
    source_name: str | None
    snapshot_date: str | None
    generated_at_utc: str | None
    age_days: int | None
    record_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "source_path": str(self.source_path),
            "artifact_family": self.artifact_family,
            "stat_family": self.stat_family,
            "source_name": self.source_name,
            "snapshot_date": self.snapshot_date,
            "generated_at_utc": self.generated_at_utc,
            "age_days": self.age_days,
            "record_count": self.record_count,
        }


class CommunityCardPriorSource:
    def __init__(
        self,
        *,
        config: CommunityPriorRuntimeConfig,
        card_records: list[CommunityCardStatRecord],
        route_records: list[PublicRunStrategicStatRecord] | None = None,
        diagnostics: dict[str, CommunityPriorSourceDiagnostics] | None = None,
    ) -> None:
        self._config = config
        self._records_by_card: dict[str, list[CommunityCardStatRecord]] = {}
        self._route_records_by_subject: dict[str, list[PublicRunStrategicStatRecord]] = {}
        self._diagnostics = diagnostics or {}
        for record in card_records:
            for lookup_key in _record_lookup_keys(record):
                self._records_by_card.setdefault(lookup_key, []).append(record)
        for record in route_records or []:
            lookup_key = (record.subject_id or "").strip().lower()
            if lookup_key:
                self._route_records_by_subject.setdefault(lookup_key, []).append(record)

    @classmethod
    def from_config(cls, config: CommunityPriorRuntimeConfig | None) -> CommunityCardPriorSource | None:
        if config is None:
            return None
        card_records, card_diagnostics = _load_card_prior_records(config.source_path)
        route_records: list[PublicRunStrategicStatRecord] = []
        diagnostics = {"card": card_diagnostics}
        if config.route_source_path is not None:
            route_records, route_diagnostics = _load_route_prior_records(config.route_source_path)
            diagnostics["route"] = route_diagnostics
        _validate_source_freshness(config, diagnostics.values())
        return cls(
            config=config,
            card_records=card_records,
            route_records=route_records,
            diagnostics=diagnostics,
        )

    @property
    def config(self) -> CommunityPriorRuntimeConfig:
        return self._config

    def diagnostics(self) -> dict[str, dict[str, object]]:
        return {key: value.as_dict() for key, value in self._diagnostics.items()}

    def score_card(
        self,
        *,
        domain: CommunityPriorDomain,
        card_id: str,
        character_id: str | None,
        ascension: int | None,
        act_id: str | None,
        floor: int,
    ) -> CommunityCardPrior | None:
        records = self._records_by_card.get(card_id.strip().lower())
        if not records:
            return None

        source_types = _domain_source_types(domain)
        candidates: list[tuple[int, CommunityCardStatRecord]] = []
        for record in records:
            if record.source_type not in source_types:
                continue
            if record.character_id is not None and (character_id is None or record.character_id.upper() != character_id.upper()):
                continue
            if ascension is not None and not _matches_ascension(record, ascension):
                continue
            if record.act_id is not None and not _matches_act(record.act_id, act_id):
                continue
            if record.floor_band is not None and not _matches_floor_band(record.floor_band, floor=floor, act_id=act_id):
                continue
            candidates.append((_specificity_level(record), record))
        if not candidates:
            return None

        best_specificity = max(level for level, _record in candidates)
        matched = [record for level, record in candidates if level == best_specificity]
        sample_count = sum(_sample_size(record, domain=domain) for record in matched)
        if sample_count <= 0:
            return None

        pick_rate = _weighted_average(matched, domain=domain, field_name="pick_rate")
        buy_rate = _weighted_average(matched, domain=domain, field_name="buy_rate")
        win_delta = _weighted_average(matched, domain=domain, field_name="win_delta")
        win_rate_with_card = _weighted_average(matched, domain=domain, field_name="win_rate_with_card")
        rate_baseline = _rate_baseline(self._config, domain)
        rate_value = buy_rate if domain == "shop_buy" else pick_rate
        rate_edge = None if rate_value is None else rate_value - rate_baseline
        confidence = min(1.0, math.sqrt(sample_count / float(self._config.max_confidence_sample_size)))
        if sample_count < self._config.min_sample_size:
            confidence *= sample_count / float(self._config.min_sample_size)

        score_bonus = _score_bonus(
            config=self._config,
            domain=domain,
            rate_edge=rate_edge,
            win_delta=win_delta,
            confidence=confidence,
        )
        exemplar = max(matched, key=lambda record: (_sample_size(record, domain=domain), record.snapshot_date))
        return CommunityCardPrior(
            domain=domain,
            card_id=card_id,
            source_type=exemplar.source_type,
            matched_record_count=len(matched),
            sample_count=sample_count,
            confidence=confidence,
            pick_rate=pick_rate,
            buy_rate=buy_rate,
            win_delta=win_delta,
            win_rate_with_card=win_rate_with_card,
            score_bonus=score_bonus,
            rate_baseline=rate_baseline,
            rate_edge=rate_edge,
            specificity_level=best_specificity,
            matched_character_id=exemplar.character_id,
            matched_act_id=exemplar.act_id,
            matched_floor_band=exemplar.floor_band,
            source_name=exemplar.source_name,
            snapshot_date=exemplar.snapshot_date,
            artifact_family=(
                exemplar.metadata.get("artifact_family") if isinstance(exemplar.metadata, dict) else None
            ),
        )

    def score_route(
        self,
        *,
        subject_id: str,
        character_id: str | None,
        act_id: str | None,
    ) -> CommunityRoutePrior | None:
        records = self._route_records_by_subject.get(subject_id.strip().lower())
        if not records:
            return None

        candidates: list[tuple[int, PublicRunStrategicStatRecord]] = []
        for record in records:
            if record.subject_id.strip().lower() not in _ROUTE_NODE_TYPES:
                continue
            if record.character_id is not None and (character_id is None or record.character_id.upper() != character_id.upper()):
                continue
            if record.act_id is not None and not _matches_act(record.act_id, act_id):
                continue
            candidates.append((_route_specificity_level(record), record))
        if not candidates:
            return None

        best_specificity = max(level for level, _record in candidates)
        matched = [record for level, record in candidates if level == best_specificity]
        sample_count = sum(int(record.seen_count or record.run_count or 0) for record in matched)
        if sample_count <= 0:
            return None
        win_rate = _weighted_strategic_average(matched, field_name="win_rate")
        win_delta = _weighted_strategic_average(matched, field_name="win_delta")
        if win_delta is None and win_rate is not None:
            win_delta = win_rate - self._config.route_neutral_win_rate
        confidence = min(1.0, math.sqrt(sample_count / float(self._config.max_confidence_sample_size)))
        if sample_count < self._config.route_min_sample_size:
            confidence *= sample_count / float(self._config.route_min_sample_size)
        score_bonus = self._config.route_weight * confidence * (
            (0.0 if win_delta is None else win_delta * self._config.route_win_rate_scale)
        )
        exemplar = max(matched, key=lambda record: (int(record.seen_count or record.run_count or 0), record.snapshot_date or ""))
        return CommunityRoutePrior(
            domain="map_node",
            subject_id=subject_id,
            matched_record_count=len(matched),
            sample_count=sample_count,
            confidence=confidence,
            win_rate=win_rate,
            win_delta=win_delta,
            score_bonus=score_bonus,
            specificity_level=best_specificity,
            matched_character_id=exemplar.character_id,
            matched_act_id=exemplar.act_id,
            room_type=exemplar.room_type,
            source_name=exemplar.source_name,
            snapshot_date=exemplar.snapshot_date,
            artifact_family=exemplar.metadata.get("artifact_family") if isinstance(exemplar.metadata, dict) else None,
        )


def _load_card_prior_records(source_path: Path) -> tuple[list[CommunityCardStatRecord], CommunityPriorSourceDiagnostics]:
    resolved = source_path.expanduser().resolve()
    if resolved.is_dir():
        community_path = resolved / "community-card-stats.jsonl"
        strategic_path = resolved / "strategic-card-stats.jsonl"
        if community_path.exists():
            return _community_card_records_with_diagnostics(community_path)
        if strategic_path.exists():
            return _strategic_card_records_with_diagnostics(strategic_path)
        raise FileNotFoundError(f"Unsupported community prior source directory: {resolved}")
    if resolved.name == "strategic-card-stats.jsonl":
        return _strategic_card_records_with_diagnostics(resolved)
    try:
        return _community_card_records_with_diagnostics(resolved)
    except Exception:
        if resolved.exists():
            return _strategic_card_records_with_diagnostics(resolved)
        raise


def _load_route_prior_records(source_path: Path) -> tuple[list[PublicRunStrategicStatRecord], CommunityPriorSourceDiagnostics]:
    resolved = source_path.expanduser().resolve()
    route_path = resolved / "strategic-route-stats.jsonl" if resolved.is_dir() else resolved
    records = load_public_run_strategic_stat_records(route_path, stat_family="route")
    diagnostics = _strategic_diagnostics(route_path, domain="route", stat_family="route")
    return records, diagnostics


def _community_card_records_with_diagnostics(
    path: Path,
) -> tuple[list[CommunityCardStatRecord], CommunityPriorSourceDiagnostics]:
    records = load_community_card_stat_records(path)
    for record in records:
        if isinstance(record.metadata, dict):
            record.metadata.setdefault("artifact_family", "community_card_stats")
    exemplar = records[0] if records else None
    return records, CommunityPriorSourceDiagnostics(
        domain="card",
        source_path=path,
        artifact_family="community_card_stats",
        stat_family="card",
        source_name=None if exemplar is None else exemplar.source_name,
        snapshot_date=None if exemplar is None else exemplar.snapshot_date,
        generated_at_utc=None,
        age_days=_age_days(None if exemplar is None else exemplar.snapshot_date),
        record_count=len(records),
    )


def _strategic_card_records_with_diagnostics(
    path: Path,
) -> tuple[list[CommunityCardStatRecord], CommunityPriorSourceDiagnostics]:
    strategic_records = load_public_run_strategic_stat_records(path, stat_family="card")
    diagnostics = _strategic_diagnostics(path, domain="card", stat_family="card")
    return [_community_record_from_strategic(record) for record in strategic_records], diagnostics


def _strategic_diagnostics(path: Path, *, domain: str, stat_family: str) -> CommunityPriorSourceDiagnostics:
    records = load_public_run_strategic_stat_records(path, stat_family=stat_family)
    exemplar = records[0] if records else None
    generated_at_utc = None
    if path.parent.exists():
        try:
            summary = load_public_run_normalized_summary(path.parent)
        except FileNotFoundError:
            summary = None
        if isinstance(summary, dict):
            generated_at_utc = _as_optional_str(summary.get("generated_at_utc"))
    snapshot_date = None if exemplar is None else exemplar.snapshot_date
    return CommunityPriorSourceDiagnostics(
        domain=domain,
        source_path=path,
        artifact_family="public_run_strategic_stats",
        stat_family=stat_family,
        source_name=None if exemplar is None else exemplar.source_name,
        snapshot_date=snapshot_date,
        generated_at_utc=generated_at_utc,
        age_days=_age_days(snapshot_date or generated_at_utc),
        record_count=len(records),
    )


def _community_record_from_strategic(record: PublicRunStrategicStatRecord) -> CommunityCardStatRecord:
    return CommunityCardStatRecord.model_validate(
        {
            "source_name": record.source_name or "public-run-strategic",
            "snapshot_date": record.snapshot_date or datetime.now(UTC).date().isoformat(),
            "character_id": record.character_id,
            "act_id": record.act_id,
            "source_type": record.source_type or "unknown",
            "card_id": record.subject_id,
            "offer_count": record.offer_count,
            "pick_count": record.pick_count,
            "pick_rate": record.pick_rate,
            "shop_offer_count": record.shop_offer_count,
            "buy_count": record.buy_count,
            "buy_rate": record.buy_rate,
            "deck_presence_runs": record.deck_presence_runs,
            "run_count": record.run_count,
            "win_rate_with_card": record.win_rate_with_card,
            "win_delta": record.win_delta,
            "metadata": {
                **(record.metadata if isinstance(record.metadata, dict) else {}),
                "artifact_family": "public_run_strategic_stats",
            },
        }
    )


def _validate_source_freshness(
    config: CommunityPriorRuntimeConfig,
    diagnostics: Any,
) -> None:
    if config.max_source_age_days is None:
        return
    stale_sources = [
        item
        for item in diagnostics
        if item.age_days is not None and item.age_days > config.max_source_age_days
    ]
    if stale_sources:
        labels = ", ".join(f"{item.domain}:{item.source_path}" for item in stale_sources)
        raise ValueError(f"Community prior source is older than max_source_age_days={config.max_source_age_days}: {labels}")


def _score_bonus(
    *,
    config: CommunityPriorRuntimeConfig,
    domain: CommunityPriorDomain,
    rate_edge: float | None,
    win_delta: float | None,
    confidence: float,
) -> float:
    if domain == "shop_buy":
        weight = config.shop_buy_weight
        rate_scale = config.buy_rate_scale
        direction = 1.0
    elif domain == "selection_remove":
        weight = config.selection_remove_weight
        rate_scale = config.pick_rate_scale
        direction = -1.0
    elif domain == "selection_upgrade":
        weight = config.selection_upgrade_weight
        rate_scale = config.pick_rate_scale
        direction = 1.0
    elif domain == "selection_pick":
        weight = config.selection_pick_weight
        rate_scale = config.pick_rate_scale
        direction = 1.0
    else:
        weight = config.reward_pick_weight
        rate_scale = config.pick_rate_scale
        direction = 1.0
    metric = 0.0
    if rate_edge is not None:
        metric += rate_edge * rate_scale
    if win_delta is not None:
        metric += win_delta * config.win_delta_scale
    return direction * weight * confidence * metric


def _weighted_average(
    records: list[CommunityCardStatRecord],
    *,
    domain: CommunityPriorDomain,
    field_name: str,
) -> float | None:
    weighted_sum = 0.0
    weight_total = 0.0
    for record in records:
        value = getattr(record, field_name)
        if value is None:
            continue
        weight = float(_sample_size(record, domain=domain))
        if weight <= 0.0:
            continue
        weighted_sum += float(value) * weight
        weight_total += weight
    if weight_total <= 0.0:
        return None
    return weighted_sum / weight_total


def _weighted_strategic_average(
    records: list[PublicRunStrategicStatRecord],
    *,
    field_name: str,
) -> float | None:
    weighted_sum = 0.0
    weight_total = 0.0
    for record in records:
        value = getattr(record, field_name)
        if value is None:
            continue
        weight = float(record.seen_count or record.run_count or 0)
        if weight <= 0.0:
            continue
        weighted_sum += float(value) * weight
        weight_total += weight
    if weight_total <= 0.0:
        return None
    return weighted_sum / weight_total


def _sample_size(record: CommunityCardStatRecord, *, domain: CommunityPriorDomain) -> int:
    if domain == "shop_buy":
        return int(record.shop_offer_count or record.buy_count or record.run_count or 0)
    if domain == "map_node":
        return int(record.run_count or 0)
    return int(record.offer_count or record.pick_count or record.run_count or record.deck_presence_runs or 0)


def _record_lookup_keys(record: CommunityCardStatRecord) -> list[str]:
    keys: set[str] = set()
    card_id = record.card_id.strip().lower()
    if card_id:
        keys.add(card_id)
    aliases = record.metadata.get("aliases")
    if isinstance(aliases, list):
        for alias in aliases:
            alias_text = str(alias).strip().lower()
            if alias_text:
                keys.add(alias_text)
    return sorted(keys)


def _rate_baseline(config: CommunityPriorRuntimeConfig, domain: CommunityPriorDomain) -> float:
    return config.shop_buy_neutral_rate if domain == "shop_buy" else config.reward_pick_neutral_rate


def _domain_source_types(domain: CommunityPriorDomain) -> set[str]:
    if domain == "map_node":
        return _ROUTE_NODE_TYPES
    return _SHOP_SOURCE_TYPES if domain == "shop_buy" else _REWARD_SOURCE_TYPES


def _specificity_level(record: CommunityCardStatRecord) -> int:
    return sum(
        1
        for value in (record.character_id, record.act_id, record.floor_band, record.ascension_min, record.ascension_max)
        if value is not None
    )


def _route_specificity_level(record: PublicRunStrategicStatRecord) -> int:
    return sum(1 for value in (record.character_id, record.act_id) if value is not None)


def _matches_ascension(record: CommunityCardStatRecord, ascension: int) -> bool:
    if record.ascension_min is not None and ascension < record.ascension_min:
        return False
    if record.ascension_max is not None and ascension > record.ascension_max:
        return False
    return True


def _matches_act(record_act_id: str, act_id: str | None) -> bool:
    normalized_record = record_act_id.strip().lower()
    if act_id is None:
        return False
    normalized_act = act_id.strip().lower()
    if normalized_record == normalized_act:
        return True
    record_match = _ACT_PATTERN.fullmatch(normalized_record)
    act_match = _ACT_PATTERN.fullmatch(normalized_act)
    if record_match and act_match:
        return record_match.group(1) == act_match.group(1)
    return False


def _matches_floor_band(floor_band: str, *, floor: int, act_id: str | None) -> bool:
    normalized = floor_band.strip().lower().replace(" ", "")
    if not normalized:
        return True
    if _matches_act(normalized, act_id):
        return True
    if match := _ACT_PATTERN.fullmatch(normalized):
        if act_id is None:
            return _derived_act_number(floor) == int(match.group(1))
        return _matches_act(normalized, act_id)
    if match := _RANGE_PATTERN.fullmatch(normalized):
        lower = int(match.group(1))
        upper = int(match.group(2))
        return lower <= floor <= upper
    if match := _LE_PATTERN.fullmatch(normalized):
        return floor <= int(match.group(1))
    if match := _GE_PATTERN.fullmatch(normalized):
        return floor >= int(match.group(1))
    if normalized.isdigit():
        return floor == int(normalized)
    return False


def _derived_act_number(floor: int) -> int:
    if floor <= 17:
        return 1
    if floor <= 34:
        return 2
    return 3


def _age_days(value: str | None) -> int | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        if "T" in normalized:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00")).date()
        else:
            parsed = datetime.fromisoformat(normalized).date()
    except ValueError:
        return None
    return (datetime.now(UTC).date() - parsed).days


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
