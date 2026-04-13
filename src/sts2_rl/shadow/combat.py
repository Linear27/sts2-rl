from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from sts2_rl.collect.policy import build_policy_pack
from sts2_rl.data import ShadowCombatEncounterRecord, load_shadow_combat_encounter_records
from sts2_rl.env import build_candidate_actions
from sts2_rl.env.models import AvailableActionsPayload, GameStatePayload
from sts2_rl.env.types import StepObservation
from sts2_rl.predict import PredictorRuntimeConfig

SHADOW_COMBAT_REPORT_SCHEMA_VERSION = 1
SHADOW_COMBAT_RESULTS_FILENAME = "results.jsonl"
SHADOW_COMBAT_COMPARE_RESULTS_FILENAME = "comparisons.jsonl"
SHADOW_COMBAT_SUMMARY_FILENAME = "summary.json"


@dataclass(frozen=True)
class ShadowCombatEvaluationReport:
    output_dir: Path
    summary_path: Path
    results_path: Path
    encounter_count: int
    usable_encounter_count: int
    skipped_encounter_count: int


@dataclass(frozen=True)
class ShadowCombatComparisonReport:
    output_dir: Path
    summary_path: Path
    comparisons_path: Path
    encounter_count: int
    comparable_encounter_count: int


def default_shadow_combat_session_name(prefix: str = "shadow-combat") -> str:
    return datetime.now(UTC).strftime(f"{prefix}-%Y%m%d-%H%M%S")


def load_shadow_combat_report(source: str | Path) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    summary_path = source_path / SHADOW_COMBAT_SUMMARY_FILENAME if source_path.is_dir() else source_path
    if not summary_path.exists():
        raise FileNotFoundError(f"Shadow combat report does not exist: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_shadow_combat_evaluation(
    *,
    source: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    policy_profile: str = "planner",
    predictor_config: PredictorRuntimeConfig | None = None,
    replace_existing: bool = False,
) -> ShadowCombatEvaluationReport:
    output_dir = Path(output_root).expanduser().resolve() / (session_name or default_shadow_combat_session_name("shadow-eval"))
    _prepare_output_dir(output_dir, replace_existing=replace_existing)
    summary_path = output_dir / SHADOW_COMBAT_SUMMARY_FILENAME
    results_path = output_dir / SHADOW_COMBAT_RESULTS_FILENAME

    records = load_shadow_combat_encounter_records(source)
    policy = build_policy_pack(policy_profile, predictor_config=predictor_config)
    results: list[dict[str, Any]] = []
    skip_reason_histogram: Counter[str] = Counter()
    for record in records:
        result = _evaluate_encounter(record=record, policy_profile=policy_profile, policy=policy)
        if result.get("status") != "ok":
            skip_reason_histogram[str(result.get("skip_reason") or "unknown")] += 1
        results.append(result)

    _write_jsonl(results_path, results)
    usable = [result for result in results if result.get("status") == "ok"]
    summary_payload = {
        "schema_version": SHADOW_COMBAT_REPORT_SCHEMA_VERSION,
        "report_kind": "shadow_combat_eval",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_path": str(Path(source).expanduser().resolve()),
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "results_path": str(results_path),
        "policy_profile": policy_profile,
        "predictor": None if predictor_config is None else predictor_config.as_dict(),
        "encounter_count": len(records),
        "usable_encounter_count": len(usable),
        "skipped_encounter_count": len(records) - len(usable),
        "skip_reason_histogram": dict(skip_reason_histogram),
        "metrics": _shadow_eval_metrics(usable),
        "chosen_action_id_histogram": dict(Counter(result["chosen_action_id"] for result in usable if result.get("chosen_action_id"))),
        "logged_first_action_id_histogram": dict(
            Counter(result["logged_first_action_id"] for result in usable if result.get("logged_first_action_id"))
        ),
        "decision_reason_histogram": dict(Counter(result["decision_reason"] for result in usable if result.get("decision_reason"))),
        "encounter_family_histogram": dict(
            Counter(result["encounter_family"] for result in usable if result.get("encounter_family"))
        ),
        "boss_histogram": dict(Counter(result["boss_id"] for result in usable if result.get("boss_id"))),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ShadowCombatEvaluationReport(
        output_dir=output_dir,
        summary_path=summary_path,
        results_path=results_path,
        encounter_count=len(records),
        usable_encounter_count=len(usable),
        skipped_encounter_count=len(records) - len(usable),
    )


def run_shadow_combat_comparison(
    *,
    source: str | Path,
    output_root: str | Path,
    session_name: str | None = None,
    baseline_policy_profile: str = "baseline",
    candidate_policy_profile: str = "planner",
    baseline_predictor_config: PredictorRuntimeConfig | None = None,
    candidate_predictor_config: PredictorRuntimeConfig | None = None,
    replace_existing: bool = False,
) -> ShadowCombatComparisonReport:
    output_dir = Path(output_root).expanduser().resolve() / (session_name or default_shadow_combat_session_name("shadow-compare"))
    _prepare_output_dir(output_dir, replace_existing=replace_existing)
    summary_path = output_dir / SHADOW_COMBAT_SUMMARY_FILENAME
    comparisons_path = output_dir / SHADOW_COMBAT_COMPARE_RESULTS_FILENAME

    records = load_shadow_combat_encounter_records(source)
    baseline_policy = build_policy_pack(baseline_policy_profile, predictor_config=baseline_predictor_config)
    candidate_policy = build_policy_pack(candidate_policy_profile, predictor_config=candidate_predictor_config)
    baseline_results: list[dict[str, Any]] = []
    candidate_results: list[dict[str, Any]] = []
    for record in records:
        baseline_results.append(_evaluate_encounter(record=record, policy_profile=baseline_policy_profile, policy=baseline_policy))
        candidate_results.append(_evaluate_encounter(record=record, policy_profile=candidate_policy_profile, policy=candidate_policy))

    paired_results = _pair_shadow_results(baseline_results, candidate_results)
    _write_jsonl(comparisons_path, paired_results)
    comparable = [result for result in paired_results if result.get("status") == "ok"]
    comparison_skip_reason_histogram = Counter(
        str(result.get("skip_reason") or "unknown")
        for result in paired_results
        if result.get("status") != "ok"
    )
    baseline_usable = [result for result in baseline_results if result.get("status") == "ok"]
    candidate_usable = [result for result in candidate_results if result.get("status") == "ok"]
    summary_payload = {
        "schema_version": SHADOW_COMBAT_REPORT_SCHEMA_VERSION,
        "report_kind": "shadow_combat_compare",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_path": str(Path(source).expanduser().resolve()),
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "comparisons_path": str(comparisons_path),
        "baseline_policy_profile": baseline_policy_profile,
        "candidate_policy_profile": candidate_policy_profile,
        "baseline_predictor": None if baseline_predictor_config is None else baseline_predictor_config.as_dict(),
        "candidate_predictor": None if candidate_predictor_config is None else candidate_predictor_config.as_dict(),
        "encounter_count": len(records),
        "baseline_usable_count": len(baseline_usable),
        "candidate_usable_count": len(candidate_usable),
        "comparable_encounter_count": len(comparable),
        "comparison_skip_reason_histogram": dict(comparison_skip_reason_histogram),
        "baseline": _shadow_eval_metrics(baseline_usable),
        "candidate": _shadow_eval_metrics(candidate_usable),
        "delta_metrics": _shadow_compare_metrics(comparable),
        "agreement_rate": _mean([1.0 if result.get("same_action") else 0.0 for result in comparable]),
        "candidate_advantage_rate": _mean([float(result.get("candidate_advantage", 0.0)) for result in comparable]),
        "encounter_family_histogram": dict(
            Counter(result["encounter_family"] for result in comparable if result.get("encounter_family"))
        ),
        "boss_histogram": dict(Counter(result["boss_id"] for result in comparable if result.get("boss_id"))),
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return ShadowCombatComparisonReport(
        output_dir=output_dir,
        summary_path=summary_path,
        comparisons_path=comparisons_path,
        encounter_count=len(records),
        comparable_encounter_count=len(comparable),
    )


def _evaluate_encounter(
    *,
    record: ShadowCombatEncounterRecord,
    policy_profile: str,
    policy,
) -> dict[str, Any]:
    if not record.has_full_snapshot:
        return _shadow_skip_result(record, policy_profile=policy_profile, skip_reason="missing_full_snapshot")
    if not record.state or not record.action_descriptors:
        return _shadow_skip_result(record, policy_profile=policy_profile, skip_reason="missing_state_payload")
    try:
        state = GameStatePayload.model_validate(record.state)
        descriptors = AvailableActionsPayload.model_validate(record.action_descriptors)
    except Exception as exc:  # pragma: no cover - defensive
        return _shadow_skip_result(record, policy_profile=policy_profile, skip_reason=f"payload_validation_failed:{exc.__class__.__name__}")
    build_result = build_candidate_actions(state, descriptors)
    if not build_result.candidates:
        return _shadow_skip_result(record, policy_profile=policy_profile, skip_reason="empty_candidate_actions")
    observation = StepObservation(
        screen_type=state.screen,
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build_result.candidates,
        build_warnings=build_result.unsupported_actions,
    )
    decision = policy.choose(observation)
    logged_first_action_id = record.action_trace_ids[0] if record.action_trace_ids else None
    chosen_action_id = None if decision.action is None else decision.action.action_id
    return {
        "encounter_id": record.encounter_id,
        "policy_profile": policy_profile,
        "status": "ok",
        "skip_reason": None,
        "character_id": record.character_id,
        "floor": record.floor,
        "combat_index": record.combat_index,
        "outcome": record.outcome,
        "encounter_family": record.encounter_family,
        "boss_id": record.strategic_context.get("boss_id"),
        "logged_first_action_id": logged_first_action_id,
        "logged_action_trace_ids": list(record.action_trace_ids),
        "chosen_action_id": chosen_action_id,
        "first_action_match": chosen_action_id == logged_first_action_id if chosen_action_id is not None and logged_first_action_id is not None else False,
        "trace_hit": chosen_action_id in record.action_trace_ids if chosen_action_id is not None else False,
        "same_legal_action_count": len(build_result.candidates) == (record.legal_action_count or len(build_result.candidates)),
        "decision_stage": decision.stage,
        "decision_reason": decision.reason,
        "decision_score": decision.score,
        "ranked_action_count": len(decision.ranked_actions),
        "ranked_action_ids": [item.action_id for item in decision.ranked_actions],
        "legal_action_count": len(build_result.candidates),
        "legal_action_ids": [candidate.action_id for candidate in build_result.candidates],
        "build_warnings": list(build_result.unsupported_actions),
        "state_fingerprint": record.state_fingerprint,
        "action_space_fingerprint": record.action_space_fingerprint,
        "trace_metadata": dict(decision.trace_metadata),
    }


def _shadow_skip_result(record: ShadowCombatEncounterRecord, *, policy_profile: str, skip_reason: str) -> dict[str, Any]:
    return {
        "encounter_id": record.encounter_id,
        "policy_profile": policy_profile,
        "status": "skipped",
        "skip_reason": skip_reason,
        "character_id": record.character_id,
        "floor": record.floor,
        "combat_index": record.combat_index,
        "outcome": record.outcome,
        "encounter_family": record.encounter_family,
        "boss_id": record.strategic_context.get("boss_id"),
    }


def _pair_shadow_results(
    baseline_results: list[dict[str, Any]],
    candidate_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate_by_id = {str(result.get("encounter_id")): result for result in candidate_results}
    paired: list[dict[str, Any]] = []
    for baseline in baseline_results:
        encounter_id = str(baseline.get("encounter_id"))
        candidate = candidate_by_id.get(encounter_id)
        if candidate is None:
            paired.append(
                {
                    "encounter_id": encounter_id,
                    "status": "skipped",
                    "skip_reason": "missing_candidate_result",
                    "baseline": baseline,
                    "candidate": None,
                }
            )
            continue
        if baseline.get("status") != "ok" or candidate.get("status") != "ok":
            paired.append(
                {
                    "encounter_id": encounter_id,
                    "status": "skipped",
                    "skip_reason": baseline.get("skip_reason") or candidate.get("skip_reason") or "non_comparable",
                    "baseline": baseline,
                    "candidate": candidate,
                }
            )
            continue
        baseline_first = bool(baseline.get("first_action_match"))
        candidate_first = bool(candidate.get("first_action_match"))
        baseline_trace = bool(baseline.get("trace_hit"))
        candidate_trace = bool(candidate.get("trace_hit"))
        candidate_advantage = 0.0
        if candidate_first and not baseline_first:
            candidate_advantage += 1.0
        elif baseline_first and not candidate_first:
            candidate_advantage -= 1.0
        if candidate_trace and not baseline_trace:
            candidate_advantage += 0.5
        elif baseline_trace and not candidate_trace:
            candidate_advantage -= 0.5
        paired.append(
            {
                "encounter_id": encounter_id,
                "status": "ok",
                "skip_reason": None,
                "floor": baseline.get("floor"),
                "outcome": baseline.get("outcome"),
                "encounter_family": baseline.get("encounter_family"),
                "boss_id": baseline.get("boss_id"),
                "same_action": baseline.get("chosen_action_id") == candidate.get("chosen_action_id"),
                "baseline_first_action_match": baseline_first,
                "candidate_first_action_match": candidate_first,
                "baseline_trace_hit": baseline_trace,
                "candidate_trace_hit": candidate_trace,
                "candidate_advantage": candidate_advantage,
                "baseline": baseline,
                "candidate": candidate,
            }
        )
    return paired


def _shadow_eval_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "first_action_match_rate": _mean([1.0 if result.get("first_action_match") else 0.0 for result in results]),
        "trace_hit_rate": _mean([1.0 if result.get("trace_hit") else 0.0 for result in results]),
        "decision_score_stats": _basic_stats([result["decision_score"] for result in results if result.get("decision_score") is not None]),
        "ranked_action_count_stats": _basic_stats([float(result.get("ranked_action_count", 0) or 0) for result in results]),
        "legal_action_count_stats": _basic_stats([float(result.get("legal_action_count", 0) or 0) for result in results]),
    }


def _shadow_compare_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "delta_first_action_match_rate": _mean(
            [
                float(result.get("candidate_first_action_match", False)) - float(result.get("baseline_first_action_match", False))
                for result in results
            ]
        ),
        "delta_trace_hit_rate": _mean(
            [float(result.get("candidate_trace_hit", False)) - float(result.get("baseline_trace_hit", False)) for result in results]
        ),
    }


def _basic_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _prepare_output_dir(path: Path, *, replace_existing: bool) -> None:
    if path.exists():
        if not replace_existing:
            raise FileExistsError(f"Output directory already exists: {path}")
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
