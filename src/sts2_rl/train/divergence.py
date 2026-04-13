from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sts2_rl.runtime import build_runtime_metadata_snapshot, fingerprint_payload


@dataclass(frozen=True)
class DivergenceDiagnosticReport:
    status: str
    family: str
    category: str
    explanation: str
    step_index: int | None
    runtime_fingerprint_match: bool
    start_state_fingerprint_match: bool
    start_action_space_fingerprint_match: bool
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "family": self.family,
            "category": self.category,
            "explanation": self.explanation,
            "step_index": self.step_index,
            "runtime_fingerprint_match": self.runtime_fingerprint_match,
            "start_state_fingerprint_match": self.start_state_fingerprint_match,
            "start_action_space_fingerprint_match": self.start_action_space_fingerprint_match,
            "details": self.details,
        }


def load_step_trace(log_path: str | Path) -> dict[str, Any]:
    path = Path(log_path)
    steps: list[dict[str, Any]] = []
    action_sequence: list[str] = []
    screen_sequence: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("record_type") != "step":
                continue
            action_id = str(payload.get("chosen_action_id") or "none")
            screen_type = str(payload.get("screen_type") or "UNKNOWN")
            legal_action_ids = [str(item) for item in payload.get("legal_action_ids", [])]
            build_warnings = [str(item) for item in payload.get("build_warnings", [])]
            state_summary = payload.get("state_summary", {})
            action_space_snapshot = {
                "legal_action_count": int(payload.get("legal_action_count", len(legal_action_ids)) or 0),
                "legal_action_ids": legal_action_ids,
                "build_warnings": build_warnings,
            }
            state_fingerprint = fingerprint_payload(state_summary if isinstance(state_summary, dict) else {})
            action_space_fingerprint = fingerprint_payload(action_space_snapshot)
            step = {
                "step_index": int(payload.get("step_index", len(steps) + 1)),
                "screen_type": screen_type,
                "chosen_action_id": action_id,
                "chosen_action_label": payload.get("chosen_action_label"),
                "policy_name": payload.get("policy_name"),
                "policy_pack": payload.get("policy_pack"),
                "policy_handler": payload.get("policy_handler"),
                "decision_source": payload.get("decision_source"),
                "decision_stage": payload.get("decision_stage"),
                "decision_reason": payload.get("decision_reason"),
                "legal_action_ids": legal_action_ids,
                "legal_action_count": action_space_snapshot["legal_action_count"],
                "build_warnings": build_warnings,
                "state_summary": state_summary if isinstance(state_summary, dict) else {},
                "state_fingerprint": state_fingerprint,
                "action_space_fingerprint": action_space_fingerprint,
                "transition_fingerprint": fingerprint_payload(
                    {
                        "screen_type": screen_type,
                        "state_fingerprint": state_fingerprint,
                        "action_space_fingerprint": action_space_fingerprint,
                    }
                ),
            }
            steps.append(step)
            action_sequence.append(action_id)
            screen_sequence.append(screen_type)
    return {
        "steps": steps,
        "action_sequence": action_sequence,
        "screen_sequence": screen_sequence,
        "trace_fingerprint": fingerprint_payload(
            [
                {
                    "chosen_action_id": step["chosen_action_id"],
                    "state_fingerprint": step["state_fingerprint"],
                    "action_space_fingerprint": step["action_space_fingerprint"],
                }
                for step in steps
            ]
        ),
    }


def build_iteration_runtime_metadata(
    *,
    base_url: str,
    prepare_target: str,
    summary_payload: dict[str, Any],
    checkpoint_path: str | Path | None = None,
    checkpoint_label: str | None = None,
    policy_profile: str | None = None,
) -> tuple[dict[str, Any], str]:
    snapshot = build_runtime_metadata_snapshot(
        base_url=base_url,
        prepare_target=prepare_target,
        summary_payload=summary_payload,
        checkpoint_path=checkpoint_path,
        checkpoint_label=checkpoint_label,
        policy_profile=policy_profile,
    )
    return snapshot, fingerprint_payload(snapshot)


def diagnose_iteration_divergence(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> DivergenceDiagnosticReport:
    runtime_fingerprint_match = baseline.get("runtime_fingerprint") == candidate.get("runtime_fingerprint")
    start_state_fingerprint_match = (
        _start_payload_fingerprint(baseline, "state_fingerprint")
        == _start_payload_fingerprint(candidate, "state_fingerprint")
    )
    start_action_space_fingerprint_match = (
        _start_payload_fingerprint(baseline, "action_space_fingerprint")
        == _start_payload_fingerprint(candidate, "action_space_fingerprint")
    )

    if not _normalization_equivalent(baseline.get("normalization_report"), candidate.get("normalization_report")):
        return DivergenceDiagnosticReport(
            status="preparation_diverged",
            family="preparation",
            category=_normalization_divergence_category(
                baseline.get("normalization_report"),
                candidate.get("normalization_report"),
            ),
            explanation="Preparation drift detected before the compared runs started.",
            step_index=None,
            runtime_fingerprint_match=runtime_fingerprint_match,
            start_state_fingerprint_match=start_state_fingerprint_match,
            start_action_space_fingerprint_match=start_action_space_fingerprint_match,
            details={
                "baseline_normalization_report": baseline.get("normalization_report"),
                "candidate_normalization_report": candidate.get("normalization_report"),
                "baseline_prepare_action_ids": baseline.get("prepare_action_ids"),
                "candidate_prepare_action_ids": candidate.get("prepare_action_ids"),
            },
        )

    if not start_state_fingerprint_match:
        return DivergenceDiagnosticReport(
            status="start_state_diverged",
            family="start_state",
            category="start_state_fingerprint_mismatch",
            explanation="The prepared start state differs before policy execution begins.",
            step_index=None,
            runtime_fingerprint_match=runtime_fingerprint_match,
            start_state_fingerprint_match=False,
            start_action_space_fingerprint_match=start_action_space_fingerprint_match,
            details={
                "baseline_start_payload": baseline.get("start_payload"),
                "candidate_start_payload": candidate.get("start_payload"),
            },
        )

    if not start_action_space_fingerprint_match:
        return DivergenceDiagnosticReport(
            status="action_space_diverged",
            family="action_space",
            category="start_action_space_mismatch",
            explanation="The prepared start state exposes a different legal action set.",
            step_index=1,
            runtime_fingerprint_match=runtime_fingerprint_match,
            start_state_fingerprint_match=True,
            start_action_space_fingerprint_match=False,
            details={
                "baseline_action_space": _start_action_space_snapshot(baseline),
                "candidate_action_space": _start_action_space_snapshot(candidate),
            },
        )

    baseline_steps = list((baseline.get("step_trace") or {}).get("steps", []))
    candidate_steps = list((candidate.get("step_trace") or {}).get("steps", []))
    shared_count = min(len(baseline_steps), len(candidate_steps))

    for index in range(shared_count):
        baseline_step = baseline_steps[index]
        candidate_step = candidate_steps[index]
        step_index = index + 1
        if baseline_step.get("chosen_action_id") != candidate_step.get("chosen_action_id"):
            return DivergenceDiagnosticReport(
                status="policy_choice_diverged",
                family="policy_choice",
                category="chosen_action_mismatch",
                explanation="Both runs reached the same decision boundary but selected different actions.",
                step_index=step_index,
                runtime_fingerprint_match=runtime_fingerprint_match,
                start_state_fingerprint_match=True,
                start_action_space_fingerprint_match=True,
                details={
                    "baseline_step": baseline_step,
                    "candidate_step": candidate_step,
                },
            )
        if baseline_step.get("state_fingerprint") != candidate_step.get("state_fingerprint"):
            return DivergenceDiagnosticReport(
                status="runtime_transition_diverged",
                family="runtime_transition",
                category="post_action_state_mismatch",
                explanation="The same action sequence produced different resulting states.",
                step_index=step_index,
                runtime_fingerprint_match=runtime_fingerprint_match,
                start_state_fingerprint_match=True,
                start_action_space_fingerprint_match=True,
                details={
                    "baseline_step": baseline_step,
                    "candidate_step": candidate_step,
                },
            )
        if baseline_step.get("action_space_fingerprint") != candidate_step.get("action_space_fingerprint"):
            return DivergenceDiagnosticReport(
                status="action_space_diverged",
                family="action_space",
                category="next_action_space_mismatch",
                explanation="The next decision boundary exposed a different legal action set.",
                step_index=step_index,
                runtime_fingerprint_match=runtime_fingerprint_match,
                start_state_fingerprint_match=True,
                start_action_space_fingerprint_match=True,
                details={
                    "baseline_step": baseline_step,
                    "candidate_step": candidate_step,
                },
            )

    if len(baseline_steps) != len(candidate_steps):
        return DivergenceDiagnosticReport(
            status="runtime_transition_diverged",
            family="runtime_transition",
            category="trace_length_mismatch",
            explanation="The compared runs produced different numbers of logged steps.",
            step_index=shared_count + 1,
            runtime_fingerprint_match=runtime_fingerprint_match,
            start_state_fingerprint_match=True,
            start_action_space_fingerprint_match=True,
            details={
                "baseline_step_count": len(baseline_steps),
                "candidate_step_count": len(candidate_steps),
            },
        )

    if (
        baseline.get("stop_reason") != candidate.get("stop_reason")
        or baseline.get("final_screen") != candidate.get("final_screen")
        or baseline.get("run_outcome_histogram") != candidate.get("run_outcome_histogram")
    ):
        return DivergenceDiagnosticReport(
            status="runtime_transition_diverged",
            family="runtime_transition",
            category="terminal_outcome_mismatch",
            explanation="The runs matched through the trace but ended with different terminal outcomes.",
            step_index=None,
            runtime_fingerprint_match=runtime_fingerprint_match,
            start_state_fingerprint_match=True,
            start_action_space_fingerprint_match=True,
            details={
                "baseline_stop_reason": baseline.get("stop_reason"),
                "candidate_stop_reason": candidate.get("stop_reason"),
                "baseline_final_screen": baseline.get("final_screen"),
                "candidate_final_screen": candidate.get("final_screen"),
                "baseline_run_outcome_histogram": baseline.get("run_outcome_histogram"),
                "candidate_run_outcome_histogram": candidate.get("run_outcome_histogram"),
            },
        )

    return DivergenceDiagnosticReport(
        status="exact_match",
        family="exact_match",
        category="exact_match",
        explanation="The compared runs matched on preparation, start state, action choices, and resulting states.",
        step_index=None,
        runtime_fingerprint_match=runtime_fingerprint_match,
        start_state_fingerprint_match=True,
        start_action_space_fingerprint_match=True,
        details={},
    )


def build_replay_diagnostic_summary(summary_payload: dict[str, Any]) -> dict[str, Any]:
    comparisons = list(summary_payload.get("comparisons", []))
    return _diagnostic_summary_payload(
        artifact_kind="replay_suite",
        source_payload=summary_payload,
        diagnostics=[item.get("diagnostic", {}) for item in comparisons],
    )


def build_checkpoint_comparison_diagnostic_summary(summary_payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = list(summary_payload.get("paired_diagnostics", []))
    return _diagnostic_summary_payload(
        artifact_kind="checkpoint_comparison",
        source_payload=summary_payload,
        diagnostics=diagnostics,
    )


def load_divergence_summary(source: str | Path) -> dict[str, Any]:
    source_path = Path(source).expanduser().resolve()
    if source_path.is_dir():
        replay_path = source_path / "replay-summary.json"
        compare_path = source_path / "comparison-summary.json"
        if replay_path.exists():
            source_path = replay_path
        elif compare_path.exists():
            source_path = compare_path
        else:
            raise FileNotFoundError(f"No replay-summary.json or comparison-summary.json under {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if "paired_diagnostics" in payload:
        summary = build_checkpoint_comparison_diagnostic_summary(payload)
    elif "comparisons" in payload:
        summary = build_replay_diagnostic_summary(payload)
    else:
        raise ValueError(f"Unsupported divergence summary source: {source_path}")
    summary["source_path"] = str(source_path)
    return summary


def _diagnostic_summary_payload(
    *,
    artifact_kind: str,
    source_payload: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    family_histogram = Counter(
        str(item.get("family", "unknown"))
        for item in diagnostics
        if isinstance(item, dict)
    )
    category_histogram = Counter(
        str(item.get("category", "unknown"))
        for item in diagnostics
        if isinstance(item, dict)
    )
    status_histogram = Counter(
        str(item.get("status", "unknown"))
        for item in diagnostics
        if isinstance(item, dict)
    )
    return {
        "artifact_kind": artifact_kind,
        "diagnostic_count": len(diagnostics),
        "family_histogram": dict(family_histogram),
        "category_histogram": dict(category_histogram),
        "status_histogram": dict(status_histogram),
        "diagnostics": diagnostics,
        "summary_path": source_payload.get("summary_path"),
        "log_path": source_payload.get("log_path"),
    }


def _normalization_equivalent(lhs: Any, rhs: Any) -> bool:
    if not isinstance(lhs, dict) or not isinstance(rhs, dict):
        return lhs == rhs
    return fingerprint_payload(
        {
            "target": lhs.get("target"),
            "reached_target": lhs.get("reached_target"),
            "stop_reason": lhs.get("stop_reason"),
            "initial_screen": lhs.get("initial_screen"),
            "final_screen": lhs.get("final_screen"),
            "action_sequence": lhs.get("action_sequence"),
            "strategy_histogram": lhs.get("strategy_histogram"),
        }
    ) == fingerprint_payload(
        {
            "target": rhs.get("target"),
            "reached_target": rhs.get("reached_target"),
            "stop_reason": rhs.get("stop_reason"),
            "initial_screen": rhs.get("initial_screen"),
            "final_screen": rhs.get("final_screen"),
            "action_sequence": rhs.get("action_sequence"),
            "strategy_histogram": rhs.get("strategy_histogram"),
        }
    )


def _normalization_divergence_category(lhs: Any, rhs: Any) -> str:
    if not isinstance(lhs, dict) or not isinstance(rhs, dict):
        return "normalization_payload_mismatch"
    for key in ("target", "reached_target", "stop_reason", "initial_screen", "final_screen"):
        if lhs.get(key) != rhs.get(key):
            return f"normalization_{key}_mismatch"
    if lhs.get("action_sequence") != rhs.get("action_sequence"):
        return "normalization_action_sequence_mismatch"
    return "normalization_strategy_mismatch"


def _start_payload_fingerprint(payload: dict[str, Any], key: str) -> Any:
    start_payload = payload.get("start_payload") or {}
    if isinstance(start_payload, dict) and key in start_payload:
        return start_payload.get(key)
    return None


def _start_action_space_snapshot(payload: dict[str, Any]) -> dict[str, Any] | None:
    start_payload = payload.get("start_payload") or {}
    if not isinstance(start_payload, dict):
        return None
    snapshot = start_payload.get("action_space_snapshot")
    return snapshot if isinstance(snapshot, dict) else None
