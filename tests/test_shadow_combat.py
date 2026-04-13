import json
from pathlib import Path

from sts2_rl.shadow import load_shadow_combat_report, run_shadow_combat_comparison, run_shadow_combat_evaluation
from tests.shadow_fixtures import build_shadow_encounter_fixture


def test_run_shadow_combat_evaluation_and_comparison(tmp_path: Path) -> None:
    source_dir = build_shadow_encounter_fixture(tmp_path)

    eval_report = run_shadow_combat_evaluation(
        source=source_dir,
        output_root=tmp_path / "artifacts" / "shadow",
        session_name="shadow-eval",
        policy_profile="planner",
    )
    eval_summary = load_shadow_combat_report(eval_report.output_dir)
    eval_results = [
        json.loads(line)
        for line in eval_report.results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert eval_report.encounter_count == 2
    assert eval_report.usable_encounter_count == 1
    assert eval_report.skipped_encounter_count == 1
    assert eval_summary["report_kind"] == "shadow_combat_eval"
    assert eval_summary["policy_profile"] == "planner"
    assert eval_summary["metrics"]["first_action_match_rate"] == 1.0
    assert eval_summary["skip_reason_histogram"] == {"missing_full_snapshot": 1}
    assert eval_summary["encounter_family_histogram"] == {"SLIME_SMALL": 1}
    assert eval_results[0]["status"] == "ok"
    assert eval_results[0]["chosen_action_id"] == "play_card|card=0|target=0"
    assert eval_results[0]["trace_hit"] is True
    assert eval_results[1]["status"] == "skipped"
    assert eval_results[1]["skip_reason"] == "missing_full_snapshot"

    compare_report = run_shadow_combat_comparison(
        source=source_dir,
        output_root=tmp_path / "artifacts" / "shadow",
        session_name="shadow-compare",
        baseline_policy_profile="baseline",
        candidate_policy_profile="planner",
    )
    compare_summary = load_shadow_combat_report(compare_report.summary_path)
    comparisons = [
        json.loads(line)
        for line in compare_report.comparisons_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert compare_report.encounter_count == 2
    assert compare_report.comparable_encounter_count == 1
    assert compare_summary["report_kind"] == "shadow_combat_compare"
    assert compare_summary["agreement_rate"] == 1.0
    assert compare_summary["candidate_advantage_rate"] == 0.0
    assert compare_summary["comparison_skip_reason_histogram"] == {"missing_full_snapshot": 1}
    assert compare_summary["delta_metrics"]["delta_first_action_match_rate"] == 0.0
    assert comparisons[0]["status"] == "ok"
    assert comparisons[0]["same_action"] is True
    assert comparisons[1]["status"] == "skipped"
    assert comparisons[1]["skip_reason"] == "missing_full_snapshot"
