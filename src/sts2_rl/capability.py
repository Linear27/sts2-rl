from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from sts2_rl.env.types import StepObservation

NO_ACTION_TIMEOUT_PREFIX = "policy_no_action_timeout:"
UNSUPPORTED_DESCRIPTOR_PREFIX = "Unsupported action descriptor:"
AMBIGUOUS_SEMANTIC_CATEGORIES = {
    "missing_selection_semantic_mode",
    "missing_selection_source_type",
    "unsupported_selection_family",
    "unsupported_selection_mode",
}


@dataclass(frozen=True)
class CapabilityDiagnosticReport:
    status: str
    bucket: str
    owner: str
    category: str
    screen_type: str
    step_index: int | None
    descriptor: str | None
    decision_reason: str | None
    stop_reason: str | None
    explanation: str
    regression_key: str
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "bucket": self.bucket,
            "owner": self.owner,
            "category": self.category,
            "screen_type": self.screen_type,
            "step_index": self.step_index,
            "descriptor": self.descriptor,
            "decision_reason": self.decision_reason,
            "stop_reason": self.stop_reason,
            "explanation": self.explanation,
            "regression_key": self.regression_key,
            "details": self.details,
        }


def empty_capability_summary() -> dict[str, Any]:
    return {
        "diagnostic_count": 0,
        "owner_histogram": {},
        "bucket_histogram": {},
        "category_histogram": {},
        "screen_histogram": {},
        "descriptor_histogram": {},
        "reason_histogram": {},
        "regression_key_histogram": {},
        "unsupported_descriptor_count": 0,
        "no_action_timeout_count": 0,
        "ambiguous_semantic_block_count": 0,
        "unexpected_runtime_divergence_count": 0,
        "diagnostics": [],
    }


def diagnostics_from_observation(
    *,
    observation: StepObservation | None,
    step_index: int | None,
) -> list[CapabilityDiagnosticReport]:
    if observation is None:
        return []
    screen_type = _normalized_screen_type(observation.screen_type)
    if not _is_non_combat_screen(screen_type):
        return []
    reports: list[CapabilityDiagnosticReport] = []
    for warning in observation.build_warnings:
        descriptor = _unsupported_descriptor_from_warning(warning)
        if descriptor is None:
            continue
        reports.append(
            CapabilityDiagnosticReport(
                status="issue",
                bucket="repo_action_space_gap",
                owner="sts2-rl",
                category="unsupported_action_descriptor",
                screen_type=screen_type,
                step_index=step_index,
                descriptor=descriptor,
                decision_reason=None,
                stop_reason=None,
                explanation="Runtime exposed a non-combat action descriptor that this repo does not map into the action space.",
                regression_key=_regression_key(
                    bucket="repo_action_space_gap",
                    category="unsupported_action_descriptor",
                    screen_type=screen_type,
                    detail=descriptor,
                ),
                details={
                    "warning": str(warning),
                    "descriptor": descriptor,
                },
            )
        )
    return reports


def diagnostics_from_no_action(
    *,
    observation: StepObservation | None,
    step_index: int | None,
    decision_reason: str | None,
    stop_reason: str | None,
    decision_metadata: Mapping[str, Any] | None = None,
) -> list[CapabilityDiagnosticReport]:
    screen_type = _screen_type_from_inputs(observation=observation, stop_reason=stop_reason)
    if not _is_non_combat_screen(screen_type):
        return []
    normalized_reason = _normalized_optional_str(decision_reason)
    normalized_stop_reason = _normalized_optional_str(stop_reason)
    if normalized_reason is None and (
        normalized_stop_reason is None or not normalized_stop_reason.startswith(NO_ACTION_TIMEOUT_PREFIX)
    ):
        return []
    if normalized_reason is None:
        normalized_reason = _decision_reason_from_stop_reason(stop_reason)

    category, descriptor, bucket, owner, explanation = _classify_no_action_issue(
        decision_reason=normalized_reason,
        stop_reason=normalized_stop_reason,
        decision_metadata=decision_metadata,
        screen_type=screen_type,
    )
    return [
        CapabilityDiagnosticReport(
            status="issue",
            bucket=bucket,
            owner=owner,
            category=category,
            screen_type=screen_type,
            step_index=step_index,
            descriptor=descriptor,
            decision_reason=normalized_reason,
            stop_reason=normalized_stop_reason,
            explanation=explanation,
            regression_key=_regression_key(
                bucket=bucket,
                category=category,
                screen_type=screen_type,
                detail=descriptor or normalized_reason or normalized_stop_reason,
            ),
            details=_no_action_details(
                decision_reason=normalized_reason,
                stop_reason=normalized_stop_reason,
                decision_metadata=decision_metadata,
            ),
        )
    ]


def summarize_capability_diagnostics(
    diagnostics: Iterable[CapabilityDiagnosticReport | Mapping[str, Any] | dict[str, Any]],
) -> dict[str, Any]:
    items = [_coerce_diagnostic_dict(item) for item in diagnostics]
    if not items:
        return empty_capability_summary()

    owner_histogram = Counter(str(item.get("owner", "unknown")) for item in items)
    bucket_histogram = Counter(str(item.get("bucket", "unknown")) for item in items)
    category_histogram = Counter(str(item.get("category", "unknown")) for item in items)
    screen_histogram = Counter(str(item.get("screen_type", "UNKNOWN")) for item in items)
    descriptor_histogram = Counter(
        str(item.get("descriptor"))
        for item in items
        if _normalized_optional_str(item.get("descriptor")) is not None
    )
    reason_histogram = Counter(
        str(reason)
        for item in items
        for reason in [_primary_reason(item)]
        if reason is not None
    )
    regression_key_histogram = Counter(str(item.get("regression_key", "unknown")) for item in items)
    return {
        "diagnostic_count": len(items),
        "owner_histogram": dict(owner_histogram),
        "bucket_histogram": dict(bucket_histogram),
        "category_histogram": dict(category_histogram),
        "screen_histogram": dict(screen_histogram),
        "descriptor_histogram": dict(descriptor_histogram),
        "reason_histogram": dict(reason_histogram),
        "regression_key_histogram": dict(regression_key_histogram),
        "unsupported_descriptor_count": sum(1 for item in items if item.get("category") == "unsupported_action_descriptor"),
        "no_action_timeout_count": sum(
            1 for item in items if str(item.get("stop_reason") or "").startswith(NO_ACTION_TIMEOUT_PREFIX)
        ),
        "ambiguous_semantic_block_count": sum(
            1 for item in items if str(item.get("category") or "") in AMBIGUOUS_SEMANTIC_CATEGORIES
        ),
        "unexpected_runtime_divergence_count": sum(
            1 for item in items if item.get("bucket") == "unexpected_runtime_divergence"
        ),
        "diagnostics": items,
    }


def merge_capability_summaries(summaries: Iterable[Mapping[str, Any] | dict[str, Any] | None]) -> dict[str, Any]:
    diagnostics: list[dict[str, Any]] = []
    for summary in summaries:
        if not isinstance(summary, Mapping):
            continue
        raw_diagnostics = summary.get("diagnostics")
        if not isinstance(raw_diagnostics, list):
            continue
        diagnostics.extend(_coerce_diagnostic_dict(item) for item in raw_diagnostics if isinstance(item, Mapping))
    return summarize_capability_diagnostics(diagnostics)


def compare_capability_summaries(
    *,
    baseline: Mapping[str, Any] | dict[str, Any] | None,
    candidate: Mapping[str, Any] | dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_summary = _normalized_summary(baseline)
    candidate_summary = _normalized_summary(candidate)
    baseline_histogram = Counter(dict(baseline_summary.get("regression_key_histogram", {})))
    candidate_histogram = Counter(dict(candidate_summary.get("regression_key_histogram", {})))

    new_regression_histogram: dict[str, int] = {}
    resolved_regression_histogram: dict[str, int] = {}
    for key in sorted(set(baseline_histogram) | set(candidate_histogram)):
        delta = int(candidate_histogram.get(key, 0)) - int(baseline_histogram.get(key, 0))
        if delta > 0:
            new_regression_histogram[key] = delta
        elif delta < 0:
            resolved_regression_histogram[key] = abs(delta)

    baseline_issue_count = int(baseline_summary.get("diagnostic_count", 0) or 0)
    candidate_issue_count = int(candidate_summary.get("diagnostic_count", 0) or 0)
    return {
        "baseline_issue_count": baseline_issue_count,
        "candidate_issue_count": candidate_issue_count,
        "delta_issue_count": candidate_issue_count - baseline_issue_count,
        "new_regression_count": sum(new_regression_histogram.values()),
        "new_regression_keys": sorted(new_regression_histogram),
        "new_regression_histogram": new_regression_histogram,
        "resolved_regression_count": sum(resolved_regression_histogram.values()),
        "resolved_regression_keys": sorted(resolved_regression_histogram),
        "resolved_regression_histogram": resolved_regression_histogram,
    }


def load_capability_summary(source: str | Path) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    if source_path.is_dir():
        for candidate_name in ("case-summary.json", "summary.json", "benchmark-suite-summary.json"):
            candidate_path = source_path / candidate_name
            if candidate_path.exists():
                source_path = candidate_path
                break
        else:
            raise FileNotFoundError(f"No case-summary.json, summary.json, or benchmark-suite-summary.json under {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    summary = _load_summary_payload(payload)
    summary["source_path"] = str(source_path)
    return summary


def _coerce_diagnostic_dict(item: CapabilityDiagnosticReport | Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, CapabilityDiagnosticReport):
        return item.as_dict()
    if not isinstance(item, Mapping):
        raise TypeError(f"Unsupported capability diagnostic payload: {type(item)!r}")
    details = dict(item.get("details", {})) if isinstance(item.get("details"), Mapping) else {}
    screen_type = _normalized_screen_type(item.get("screen_type"))
    category = str(item.get("category", "unknown"))
    bucket = str(item.get("bucket", "unknown"))
    descriptor = _normalized_optional_str(item.get("descriptor"))
    decision_reason = _normalized_optional_str(item.get("decision_reason"))
    stop_reason = _normalized_optional_str(item.get("stop_reason"))
    regression_key = _normalized_optional_str(item.get("regression_key")) or _regression_key(
        bucket=bucket,
        category=category,
        screen_type=screen_type,
        detail=descriptor or decision_reason or stop_reason,
    )
    return {
        "status": str(item.get("status", "issue")),
        "bucket": bucket,
        "owner": str(item.get("owner", "unknown")),
        "category": category,
        "screen_type": screen_type,
        "step_index": item.get("step_index"),
        "descriptor": descriptor,
        "decision_reason": decision_reason,
        "stop_reason": stop_reason,
        "explanation": str(item.get("explanation", "")),
        "regression_key": regression_key,
        "details": details,
    }


def _normalized_summary(summary: Mapping[str, Any] | dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        return empty_capability_summary()
    if isinstance(summary.get("diagnostics"), list):
        return summarize_capability_diagnostics(
            item for item in summary["diagnostics"] if isinstance(item, Mapping) or isinstance(item, CapabilityDiagnosticReport)
        )
    return empty_capability_summary()


def _load_summary_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("mode") == "compare":
        non_combat_capability = dict(payload.get("non_combat_capability", {}))
        return {
            "artifact_kind": "benchmark_compare_case",
            "case_id": payload.get("case_id"),
            "baseline": _normalized_summary(non_combat_capability.get("baseline")),
            "candidate": _normalized_summary(non_combat_capability.get("candidate")),
            "comparison": dict(non_combat_capability.get("comparison", {})),
        }
    if payload.get("mode") == "eval":
        return {
            "artifact_kind": "benchmark_eval_case",
            "case_id": payload.get("case_id"),
            "summary": _normalized_summary(payload.get("non_combat_capability")),
        }
    if "suite_name" in payload and "cases" in payload:
        non_combat_capability = dict(payload.get("non_combat_capability", {}))
        compare_payload = dict(non_combat_capability.get("compare", {}))
        return {
            "artifact_kind": "benchmark_suite",
            "suite_name": payload.get("suite_name"),
            "eval": _normalized_summary(non_combat_capability.get("eval")),
            "compare": {
                "baseline": _normalized_summary(compare_payload.get("baseline")),
                "candidate": _normalized_summary(compare_payload.get("candidate")),
                "comparison": dict(compare_payload.get("comparison", {})),
            },
        }
    return {
        "artifact_kind": "session_summary",
        "session_name": payload.get("session_name"),
        "summary": _normalized_summary(payload.get("non_combat_capability")),
    }


def _classify_no_action_issue(
    *,
    decision_reason: str | None,
    stop_reason: str | None,
    decision_metadata: Mapping[str, Any] | None,
    screen_type: str,
) -> tuple[str, str | None, str, str, str]:
    if decision_reason == "missing_selection_semantic_mode":
        return (
            "missing_selection_semantic_mode",
            None,
            "runtime_contract_gap",
            "STS2-Agent",
            "Selection semantics were missing the runtime semantic mode, so this repo could not plan a supported non-combat action.",
        )
    if decision_reason == "missing_selection_source_type":
        return (
            "missing_selection_source_type",
            None,
            "runtime_contract_gap",
            "STS2-Agent",
            "Selection semantics were missing the runtime source type, so this repo could not plan a supported non-combat action.",
        )
    if decision_reason and decision_reason.startswith("unsupported_selection_family:"):
        family = decision_reason.split(":", maxsplit=1)[1].strip() or "unknown"
        return (
            "unsupported_selection_family",
            family,
            "repo_policy_gap",
            "sts2-rl",
            "Runtime exposed a non-combat selection family that this repo does not yet support end-to-end.",
        )
    if decision_reason and decision_reason.startswith("unsupported_selection_mode:"):
        mode = decision_reason.split(":", maxsplit=1)[1].strip() or "unknown"
        return (
            "unsupported_selection_mode",
            mode,
            "repo_policy_gap",
            "sts2-rl",
            "Runtime exposed a non-combat selection mode that this repo does not yet support end-to-end.",
        )
    if decision_reason in {
        "selection_transaction_diverged",
        "selection_transaction_missing_confirm_action",
        "selection_transaction_completed",
    }:
        divergence_reason = _selection_divergence_reason(decision_metadata)
        return (
            decision_reason,
            divergence_reason,
            "unexpected_runtime_divergence",
            "shared",
            "A supported non-combat control path diverged at runtime after semantic planning had already succeeded.",
        )
    if decision_reason == "policy_no_action_timeout":
        return (
            "policy_no_action_timeout",
            None,
            "repo_policy_gap",
            "sts2-rl",
            "The repo timed out on a non-combat screen without selecting an action.",
        )
    if stop_reason and stop_reason.startswith(NO_ACTION_TIMEOUT_PREFIX):
        return (
            "policy_no_action_timeout",
            None,
            "repo_policy_gap",
            "sts2-rl",
            "The repo timed out on a non-combat screen without selecting an action.",
        )
    return (
        "policy_no_action_timeout",
        decision_reason,
        "repo_policy_gap",
        "sts2-rl",
        f"The repo stalled on the {screen_type.lower()} screen without resolving a supported non-combat decision.",
    )


def _selection_divergence_reason(decision_metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(decision_metadata, Mapping):
        return None
    transaction = decision_metadata.get("selection_transaction")
    if not isinstance(transaction, Mapping):
        return None
    return _normalized_optional_str(transaction.get("divergence_reason"))


def _no_action_details(
    *,
    decision_reason: str | None,
    stop_reason: str | None,
    decision_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if decision_reason is not None:
        details["decision_reason"] = decision_reason
    if stop_reason is not None:
        details["stop_reason"] = stop_reason
    divergence_reason = _selection_divergence_reason(decision_metadata)
    if divergence_reason is not None:
        details["divergence_reason"] = divergence_reason
    if isinstance(decision_metadata, Mapping) and decision_metadata:
        details["decision_metadata"] = dict(decision_metadata)
    return details


def _primary_reason(item: Mapping[str, Any]) -> str | None:
    decision_reason = _normalized_optional_str(item.get("decision_reason"))
    if decision_reason is not None:
        return decision_reason
    return _normalized_optional_str(item.get("stop_reason"))


def _unsupported_descriptor_from_warning(warning: Any) -> str | None:
    text = _normalized_optional_str(warning)
    if text is None:
        return None
    if not text.startswith(UNSUPPORTED_DESCRIPTOR_PREFIX):
        return None
    descriptor = text.split(":", maxsplit=1)[1].strip()
    return descriptor or None


def _screen_type_from_inputs(*, observation: StepObservation | None, stop_reason: str | None) -> str:
    if observation is not None:
        return _normalized_screen_type(observation.screen_type)
    if stop_reason and stop_reason.startswith(NO_ACTION_TIMEOUT_PREFIX):
        parts = stop_reason.split(":")
        if len(parts) >= 2:
            return _normalized_screen_type(parts[1])
    return "UNKNOWN"


def _decision_reason_from_stop_reason(stop_reason: str | None) -> str | None:
    normalized = _normalized_optional_str(stop_reason)
    if normalized is None or not normalized.startswith(NO_ACTION_TIMEOUT_PREFIX):
        return None
    parts = normalized.split(":", maxsplit=2)
    if len(parts) < 3:
        return "policy_no_action_timeout"
    reason = parts[2].strip()
    return reason or "policy_no_action_timeout"


def _is_non_combat_screen(screen_type: str | None) -> bool:
    normalized = _normalized_screen_type(screen_type)
    return normalized != "COMBAT"


def _regression_key(*, bucket: str, category: str, screen_type: str, detail: str | None) -> str:
    normalized_detail = _normalized_optional_str(detail) or "-"
    return "|".join([bucket, category, _normalized_screen_type(screen_type), normalized_detail])


def _normalized_screen_type(value: Any) -> str:
    normalized = _normalized_optional_str(value)
    return "UNKNOWN" if normalized is None else normalized.upper()


def _normalized_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
