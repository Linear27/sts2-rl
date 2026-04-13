import json
from pathlib import Path

from sts2_rl.train.divergence import diagnose_iteration_divergence, load_divergence_summary


def test_diagnose_iteration_divergence_classifies_preparation_policy_and_runtime() -> None:
    baseline = _iteration_payload()
    candidate = _iteration_payload()

    preparation = diagnose_iteration_divergence(
        baseline=baseline,
        candidate={
            **candidate,
            "normalization_report": {
                **candidate["normalization_report"],
                "action_sequence": ["return_to_main_menu"],
            },
        },
    )
    assert preparation.family == "preparation"
    assert preparation.category == "normalization_action_sequence_mismatch"

    policy = diagnose_iteration_divergence(
        baseline=baseline,
        candidate={
            **candidate,
            "step_trace": {
                **candidate["step_trace"],
                "steps": [
                    candidate["step_trace"]["steps"][0],
                    {
                        **candidate["step_trace"]["steps"][1],
                        "chosen_action_id": "alt-end-turn",
                    },
                ],
            },
        },
    )
    assert policy.family == "policy_choice"
    assert policy.category == "chosen_action_mismatch"
    assert policy.step_index == 2

    runtime = diagnose_iteration_divergence(
        baseline=baseline,
        candidate={
            **candidate,
            "step_trace": {
                **candidate["step_trace"],
                "steps": [
                    candidate["step_trace"]["steps"][0],
                    {
                        **candidate["step_trace"]["steps"][1],
                        "state_fingerprint": "different-state",
                    },
                ],
            },
        },
    )
    assert runtime.family == "runtime_transition"
    assert runtime.category == "post_action_state_mismatch"


def test_load_divergence_summary_reads_replay_and_compare_summaries(tmp_path: Path) -> None:
    replay_summary = tmp_path / "replay-summary.json"
    replay_summary.write_text(
        json.dumps(
            {
                "summary_path": str(replay_summary),
                "log_path": str(tmp_path / "replay-suite.jsonl"),
                "comparisons": [
                    {
                        "diagnostic": {
                            "status": "policy_choice_diverged",
                            "family": "policy_choice",
                            "category": "chosen_action_mismatch",
                            "explanation": "action mismatch",
                            "step_index": 3,
                        }
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    compare_summary = tmp_path / "comparison-summary.json"
    compare_summary.write_text(
        json.dumps(
            {
                "summary_path": str(compare_summary),
                "log_path": str(tmp_path / "comparison-log.jsonl"),
                "paired_diagnostics": [
                    {
                        "status": "exact_match",
                        "family": "exact_match",
                        "category": "exact_match",
                        "explanation": "match",
                        "step_index": None,
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    replay = load_divergence_summary(replay_summary)
    compare = load_divergence_summary(compare_summary)

    assert replay["artifact_kind"] == "replay_suite"
    assert replay["family_histogram"] == {"policy_choice": 1}
    assert compare["artifact_kind"] == "checkpoint_comparison"
    assert compare["status_histogram"] == {"exact_match": 1}


def _iteration_payload() -> dict[str, object]:
    return {
        "normalization_report": {
            "target": "main_menu",
            "reached_target": True,
            "stop_reason": "target_reached",
            "initial_screen": "GAME_OVER",
            "final_screen": "MAIN_MENU",
            "action_sequence": [],
            "strategy_histogram": {},
        },
        "prepare_action_ids": [],
        "start_payload": {
            "state_fingerprint": "same-state",
            "action_space_fingerprint": "same-actions",
            "action_space_snapshot": {"legal_action_ids": ["play", "end_turn"]},
        },
        "runtime_fingerprint": "runtime",
        "stop_reason": "max_runs_reached",
        "final_screen": "GAME_OVER",
        "run_outcome_histogram": {"won": 1},
        "step_trace": {
            "steps": [
                {
                    "step_index": 1,
                    "chosen_action_id": "play",
                    "state_fingerprint": "state-1",
                    "action_space_fingerprint": "space-1",
                },
                {
                    "step_index": 2,
                    "chosen_action_id": "end-turn",
                    "state_fingerprint": "state-2",
                    "action_space_fingerprint": "space-2",
                },
            ]
        },
    }
