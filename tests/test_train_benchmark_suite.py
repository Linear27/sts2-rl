import json
from pathlib import Path

import pytest

from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import ActionDescriptor, AvailableActionsPayload, GameStatePayload
from sts2_rl.env.types import StepObservation
from sts2_rl.train import (
    BenchmarkSuiteManifest,
    CombatCheckpointComparisonReport,
    CombatCheckpointEvalIterationReport,
    CombatEvaluationReport,
    CombatReplayComparisonReport,
    CombatReplayIterationReport,
    CombatReplaySuiteReport,
    DivergenceDiagnosticReport,
    load_benchmark_suite_manifest,
    load_benchmark_suite_summary,
    run_benchmark_suite,
)
from tests.shadow_fixtures import build_shadow_encounter_fixture


class FakeEnv:
    def __init__(self, observation: StepObservation) -> None:
        self._observation = observation

    def observe(self) -> StepObservation:
        return self._observation

    def step(self, action):  # pragma: no cover - suite tests do not step the env
        raise AssertionError(f"Unexpected step call: {action}")

    def close(self) -> None:
        return None


def test_load_benchmark_suite_manifest_resolves_relative_checkpoint_paths(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "latest.json").write_text("latest", encoding="utf-8")
    (checkpoints_dir / "best.json").write_text("best", encoding="utf-8")

    manifest_path = tmp_path / "suite.toml"
    manifest_path.write_text(
        """
schema_version = 1
suite_name = "bench"

[[cases]]
case_id = "eval"
mode = "eval"
checkpoint_path = "./checkpoints/latest.json"
prepare_target = "none"

[[cases]]
case_id = "compare"
mode = "compare"
baseline_checkpoint_path = "./checkpoints/latest.json"
candidate_checkpoint_path = "./checkpoints/best.json"
prepare_target = "none"
""".strip(),
        encoding="utf-8",
    )

    manifest = load_benchmark_suite_manifest(manifest_path)

    assert isinstance(manifest, BenchmarkSuiteManifest)
    assert manifest.cases[0].checkpoint_path == str((checkpoints_dir / "latest.json").resolve())
    assert manifest.cases[1].baseline_checkpoint_path == str((checkpoints_dir / "latest.json").resolve())
    assert manifest.cases[1].candidate_checkpoint_path == str((checkpoints_dir / "best.json").resolve())


def test_load_benchmark_suite_manifest_resolves_relative_shadow_source_paths(tmp_path: Path) -> None:
    shadow_dir = build_shadow_encounter_fixture(tmp_path)
    manifest_path = tmp_path / "suite-shadow.toml"
    manifest_path.write_text(
        f"""
schema_version = 1
suite_name = "bench-shadow"

[[cases]]
case_id = "compare-shadow"
mode = "compare"
baseline_policy_profile = "baseline"
candidate_policy_profile = "planner"
prepare_target = "none"

[cases.shadow]
source = "{shadow_dir.relative_to(tmp_path).as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    manifest = load_benchmark_suite_manifest(manifest_path)

    assert isinstance(manifest, BenchmarkSuiteManifest)
    assert manifest.cases[0].shadow is not None
    assert manifest.cases[0].shadow.source == str(shadow_dir.resolve())


def test_benchmark_manifest_injects_contract_seed_into_scenario_seed_set() -> None:
    manifest = BenchmarkSuiteManifest.model_validate(
        {
            "schema_version": 1,
            "suite_name": "fixed-seed-bench",
            "cases": [
                {
                    "case_id": "fixed-seed-policy",
                    "mode": "eval",
                    "policy_profile": "baseline",
                    "prepare_target": "none",
                    "game_run_contract": {
                        "game_seed": "FIXED-SEED-001",
                        "character_id": "IRONCLAD",
                        "ascension": 0,
                        "progress_profile": "debug-unlocked",
                    },
                }
            ],
        }
    )

    case = manifest.cases[0]
    assert case.scenario.seed_set == ["FIXED-SEED-001"]


def test_run_benchmark_suite_writes_eval_compare_and_replay_reports(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    latest = checkpoints_dir / "latest.json"
    best = checkpoints_dir / "best.json"
    latest.write_text("latest", encoding="utf-8")
    best.write_text("best", encoding="utf-8")

    manifest_path = tmp_path / "suite.toml"
    manifest_path.write_text(
        """
schema_version = 1
suite_name = "bench"
base_url = "http://127.0.0.1:8080"

[stats]
bootstrap_resamples = 200
confidence_level = 0.95
seed = 11

[[cases]]
case_id = "eval-latest"
mode = "eval"
checkpoint_path = "./checkpoints/latest.json"
repeats = 3
prepare_target = "none"

[[cases]]
case_id = "compare-best"
mode = "compare"
baseline_checkpoint_path = "./checkpoints/latest.json"
candidate_checkpoint_path = "./checkpoints/best.json"
repeats = 2
prepare_target = "none"

[[cases]]
case_id = "replay-best"
mode = "replay"
checkpoint_path = "./checkpoints/best.json"
repeats = 3
prepare_target = "none"
""".strip(),
        encoding="utf-8",
    )

    main_menu = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
    )

    def fake_env_factory(_base_url: str, _timeout: float) -> FakeEnv:
        return FakeEnv(main_menu)

    def fake_eval_fn(**kwargs) -> CombatEvaluationReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        iteration_index = int(session_name.rsplit("-", maxsplit=1)[-1])
        checkpoint_name = Path(kwargs["checkpoint_path"]).stem
        log_path = session_dir / "combat-eval.jsonl"
        summary_path = session_dir / "summary.json"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")

        reward_base = 1.0 if checkpoint_name == "latest" else 3.0
        total_reward = reward_base + iteration_index
        combat_win_rate = 0.25 * iteration_index if checkpoint_name == "latest" else 0.5 + (0.25 * iteration_index)
        combat_performance = {
            "combat_steps": 4,
            "completed_combat_count": 1,
            "won_combats": 1 if combat_win_rate >= 0.5 else 0,
            "lost_combats": 0 if combat_win_rate >= 0.5 else 1,
            "combat_win_rate": combat_win_rate,
            "reward_per_combat": total_reward,
            "reward_per_combat_step": total_reward / 4.0,
        }
        summary_path.write_text(
            json.dumps(
                {
                    "combat_performance": combat_performance,
                    "stop_reason": "max_runs_reached",
                    "completed_run_count": 1,
                    "completed_combat_count": 1,
                    "observed_run_seeds": [f"{checkpoint_name.upper()}-SEED-{iteration_index:03d}"],
                    "observed_run_seed_histogram": {f"{checkpoint_name.upper()}-SEED-{iteration_index:03d}": 1},
                    "runs_without_observed_seed": 0,
                    "last_observed_seed": f"{checkpoint_name.upper()}-SEED-{iteration_index:03d}",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=5,
            combat_steps=4,
            heuristic_steps=1,
            total_reward=total_reward,
            final_screen="GAME_OVER",
            final_run_id=f"run-{checkpoint_name}-{iteration_index}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=Path(kwargs["checkpoint_path"]),
            combat_performance=combat_performance,
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
        )

    def fake_compare_fn(**kwargs) -> CombatCheckpointComparisonReport:
        comparison_dir = Path(kwargs["output_root"]) / kwargs["comparison_name"]
        comparison_dir.mkdir(parents=True, exist_ok=True)
        summary_path = comparison_dir / "comparison-summary.json"
        iterations_path = comparison_dir / "comparison-iterations.jsonl"
        log_path = comparison_dir / "comparison-log.jsonl"
        iterations_path.write_text("", encoding="utf-8")
        log_path.write_text("", encoding="utf-8")

        iterations: list[CombatCheckpointEvalIterationReport] = []
        for checkpoint_label, reward_base, win_rate in (
            ("baseline", 1.0, 0.0),
            ("candidate", 3.0, 1.0),
        ):
            for iteration_index in range(1, 3):
                session_name = f"{checkpoint_label}-iteration-{iteration_index:03d}"
                session_dir = comparison_dir / session_name
                session_dir.mkdir(parents=True, exist_ok=True)
                session_summary_path = session_dir / "summary.json"
                session_log_path = session_dir / "combat-eval.jsonl"
                session_outcomes_path = session_dir / "combat-outcomes.jsonl"
                session_summary_path.write_text(
                    json.dumps(
                        {
                            "observed_run_seeds": [f"{checkpoint_label.upper()}-SEED-{iteration_index:03d}"],
                            "observed_run_seed_histogram": {f"{checkpoint_label.upper()}-SEED-{iteration_index:03d}": 1},
                            "runs_without_observed_seed": 0,
                            "last_observed_seed": f"{checkpoint_label.upper()}-SEED-{iteration_index:03d}",
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                session_log_path.write_text("", encoding="utf-8")
                session_outcomes_path.write_text("", encoding="utf-8")
                iterations.append(
                    CombatCheckpointEvalIterationReport(
                        checkpoint_label=checkpoint_label,
                        checkpoint_path=Path(kwargs[f"{checkpoint_label}_checkpoint_path"]),
                        iteration_index=iteration_index,
                        session_name=session_name,
                        session_dir=session_dir,
                        summary_path=session_summary_path,
                        log_path=session_log_path,
                        combat_outcomes_path=session_outcomes_path,
                        prepare_target="none",
                        normalization_report={"target": "none", "reached_target": True, "stop_reason": "target_reached"},
                        start_screen="MAIN_MENU",
                        start_signature="same-start",
                        start_payload={"start_signature": "same-start"},
                        runtime_metadata={},
                        runtime_fingerprint=f"{checkpoint_label}-{iteration_index}",
                        step_trace_fingerprint=f"{checkpoint_label}-{iteration_index}",
                        prepare_action_ids=[],
                        env_steps=5,
                        combat_steps=4,
                        heuristic_steps=1,
                        total_reward=reward_base + iteration_index,
                        stop_reason="max_runs_reached",
                        final_screen="GAME_OVER",
                        completed_run_count=1,
                        completed_combat_count=1,
                        combat_performance={
                            "combat_win_rate": win_rate,
                            "reward_per_combat": reward_base + iteration_index,
                            "reward_per_combat_step": (reward_base + iteration_index) / 4.0,
                            "won_combats": int(win_rate > 0),
                            "lost_combats": int(win_rate == 0),
                        },
                    )
                )

        summary_path.write_text(
            json.dumps(
                {
                    "baseline": {
                        "combat_win_rate": 0.0,
                        "mean_total_reward": 2.5,
                        "observed_run_seed_histogram": {"BASELINE-SEED-001": 1, "BASELINE-SEED-002": 1},
                    },
                    "candidate": {
                        "combat_win_rate": 1.0,
                        "mean_total_reward": 4.5,
                        "observed_run_seed_histogram": {"CANDIDATE-SEED-001": 1, "CANDIDATE-SEED-002": 1},
                    },
                    "delta_metrics": {"combat_win_rate": 1.0, "mean_total_reward": 2.0},
                    "better_checkpoint_label": "candidate",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return CombatCheckpointComparisonReport(
            base_url=kwargs["base_url"],
            comparison_dir=comparison_dir,
            summary_path=summary_path,
            iterations_path=iterations_path,
            log_path=log_path,
            baseline_checkpoint_path=Path(kwargs["baseline_checkpoint_path"]),
            candidate_checkpoint_path=Path(kwargs["candidate_checkpoint_path"]),
            repeat_count=2,
            prepare_target="none",
            better_checkpoint_label="candidate",
            delta_metrics={"combat_win_rate": 1.0, "mean_total_reward": 2.0},
            baseline={
                "combat_win_rate": 0.0,
                "mean_total_reward": 2.5,
                "observed_run_seed_histogram": {"BASELINE-SEED-001": 1, "BASELINE-SEED-002": 1},
            },
            candidate={
                "combat_win_rate": 1.0,
                "mean_total_reward": 4.5,
                "observed_run_seed_histogram": {"CANDIDATE-SEED-001": 1, "CANDIDATE-SEED-002": 1},
            },
            iterations=iterations,
            diagnostics_path=comparison_dir / "comparison-diagnostics.jsonl",
            paired_diagnostics=[],
        )

    def fake_replay_fn(**kwargs) -> CombatReplaySuiteReport:
        suite_dir = Path(kwargs["output_root"]) / kwargs["suite_name"]
        suite_dir.mkdir(parents=True, exist_ok=True)
        summary_path = suite_dir / "replay-summary.json"
        comparisons_path = suite_dir / "replay-comparisons.jsonl"
        log_path = suite_dir / "replay-suite.jsonl"
        comparisons_path.write_text("", encoding="utf-8")
        log_path.write_text("", encoding="utf-8")

        iterations: list[CombatReplayIterationReport] = []
        for iteration_index in range(1, 4):
            session_dir = suite_dir / f"iteration-{iteration_index:03d}"
            session_dir.mkdir(parents=True, exist_ok=True)
            session_summary_path = session_dir / "summary.json"
            session_log_path = session_dir / "combat-eval.jsonl"
            session_outcomes_path = session_dir / "combat-outcomes.jsonl"
            session_summary_path.write_text("{}", encoding="utf-8")
            session_log_path.write_text("", encoding="utf-8")
            session_outcomes_path.write_text("", encoding="utf-8")
            iterations.append(
                CombatReplayIterationReport(
                    iteration_index=iteration_index,
                    session_name=f"iteration-{iteration_index:03d}",
                    session_dir=session_dir,
                    summary_path=session_summary_path,
                    log_path=session_log_path,
                    combat_outcomes_path=session_outcomes_path,
                    prepare_target="none",
                    normalization_report={"target": "none", "reached_target": True, "stop_reason": "target_reached"},
                    start_screen="MAIN_MENU",
                    start_signature="same-start",
                    start_payload={"start_signature": "same-start"},
                    runtime_metadata={},
                    runtime_fingerprint=f"runtime-{iteration_index}",
                    step_trace_fingerprint=f"trace-{iteration_index}",
                    prepare_action_ids=[],
                    env_steps=3,
                    combat_steps=2,
                    heuristic_steps=1,
                    total_reward=float(iteration_index),
                    stop_reason="max_runs_reached",
                    final_screen="GAME_OVER",
                    completed_run_count=1,
                    completed_combat_count=1,
                    action_count=3,
                    action_sequence=["play", "end", "play"],
                        action_id_histogram={"play": 2, "end": 1},
                        action_histogram={"play_card": 2, "end_turn": 1},
                        run_outcome_histogram={"won": 1},
                        observed_run_seeds=[f"REPLAY-SEED-{iteration_index:03d}"],
                        observed_run_seed_histogram={f"REPLAY-SEED-{iteration_index:03d}": 1},
                        runs_without_observed_seed=0,
                        last_observed_seed=f"REPLAY-SEED-{iteration_index:03d}",
                )
            )

        comparisons = [
            CombatReplayComparisonReport(
                baseline_iteration=1,
                candidate_iteration=2,
                status="exact_match",
                start_signature_match=True,
                stop_reason_match=True,
                final_screen_match=True,
                action_sequence_match=True,
                action_histogram_match=True,
                action_id_histogram_match=True,
                run_outcome_histogram_match=True,
                common_action_prefix_length=3,
                first_action_divergence_index=None,
                baseline_action_count=3,
                candidate_action_count=3,
                metric_differences={},
                action_histogram_delta={},
                action_id_histogram_delta={},
                diagnostic=DivergenceDiagnosticReport(
                    status="exact_match",
                    family="exact_match",
                    category="exact_match",
                    explanation="match",
                    step_index=None,
                    runtime_fingerprint_match=True,
                    start_state_fingerprint_match=True,
                    start_action_space_fingerprint_match=True,
                    details={},
                ),
            ),
            CombatReplayComparisonReport(
                baseline_iteration=1,
                candidate_iteration=3,
                status="policy_choice_diverged",
                start_signature_match=True,
                stop_reason_match=True,
                final_screen_match=True,
                action_sequence_match=False,
                action_histogram_match=False,
                action_id_histogram_match=False,
                run_outcome_histogram_match=True,
                common_action_prefix_length=2,
                first_action_divergence_index=3,
                baseline_action_count=3,
                candidate_action_count=3,
                metric_differences={"total_reward": {"baseline": 1.0, "candidate": 3.0}},
                action_histogram_delta={"end_turn": {"baseline": 1, "candidate": 2, "delta": 1}},
                action_id_histogram_delta={"end": {"baseline": 1, "candidate": 2, "delta": 1}},
                diagnostic=DivergenceDiagnosticReport(
                    status="policy_choice_diverged",
                    family="policy_choice",
                    category="chosen_action_mismatch",
                    explanation="action mismatch",
                    step_index=3,
                    runtime_fingerprint_match=True,
                    start_state_fingerprint_match=True,
                    start_action_space_fingerprint_match=True,
                    details={},
                ),
            ),
        ]
        summary_path.write_text(
            json.dumps(
                {
                    "status_histogram": {"exact_match": 1, "policy_choice_diverged": 1},
                    "comparison_count": 2,
                    "exact_match_count": 1,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return CombatReplaySuiteReport(
            base_url=kwargs["base_url"],
            checkpoint_path=Path(kwargs["checkpoint_path"]),
            suite_dir=suite_dir,
            summary_path=summary_path,
            comparisons_path=comparisons_path,
            log_path=log_path,
            repeat_count=3,
            comparison_count=2,
            exact_match_count=1,
            divergent_iteration_count=1,
            status_histogram={"exact_match": 1, "trajectory_diverged": 1},
            prepare_target="none",
            iterations=iterations,
            comparisons=comparisons,
        )

    report = run_benchmark_suite(
        manifest_path,
        output_root=tmp_path / "artifacts",
        env_factory=fake_env_factory,
        evaluation_fn=fake_eval_fn,
        comparison_fn=fake_compare_fn,
        replay_fn=fake_replay_fn,
    )

    assert report.summary_path.exists()
    summary = load_benchmark_suite_summary(report.summary_path)
    assert summary["case_count"] == 3


def test_run_benchmark_suite_fails_promotion_on_new_non_combat_capability_regression(tmp_path: Path) -> None:
    manifest = BenchmarkSuiteManifest.model_validate(
        {
            "schema_version": 1,
            "suite_name": "capability-gate",
            "cases": [
                {
                    "case_id": "compare-capability",
                    "mode": "compare",
                    "baseline_policy_profile": "baseline",
                    "candidate_policy_profile": "candidate",
                    "repeats": 1,
                    "prepare_target": "none",
                    "promotion": {
                        "min_seed_set_coverage": 0.0,
                        "min_route_decision_count": 0,
                        "min_route_decision_overlap_rate": 0.0,
                        "max_new_non_combat_capability_regressions": 0,
                    },
                }
            ],
        }
    )

    main_menu = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
    )

    def fake_env_factory(_base_url: str, _timeout: float) -> FakeEnv:
        return FakeEnv(main_menu)

    def fake_policy_evaluation_fn(**kwargs) -> CombatEvaluationReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "policy-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        profile = str(kwargs["policy_profile"])
        capability_summary = (
            {
                "diagnostic_count": 1,
                "owner_histogram": {"sts2-rl": 1},
                "bucket_histogram": {"repo_action_space_gap": 1},
                "category_histogram": {"unsupported_action_descriptor": 1},
                "screen_histogram": {"MAP": 1},
                "descriptor_histogram": {"mystery_map_action": 1},
                "reason_histogram": {},
                "regression_key_histogram": {
                    "repo_action_space_gap|unsupported_action_descriptor|MAP|mystery_map_action": 1
                },
                "unsupported_descriptor_count": 1,
                "no_action_timeout_count": 0,
                "ambiguous_semantic_block_count": 0,
                "unexpected_runtime_divergence_count": 0,
                "diagnostics": [
                    {
                        "status": "issue",
                        "bucket": "repo_action_space_gap",
                        "owner": "sts2-rl",
                        "category": "unsupported_action_descriptor",
                        "screen_type": "MAP",
                        "step_index": 0,
                        "descriptor": "mystery_map_action",
                        "decision_reason": None,
                        "stop_reason": None,
                        "explanation": "unsupported",
                        "regression_key": "repo_action_space_gap|unsupported_action_descriptor|MAP|mystery_map_action",
                        "details": {},
                    }
                ],
            }
            if profile == "candidate"
            else {
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
        )
        summary_path.write_text(
            json.dumps(
                {
                    "observed_run_seeds": [f"{profile.upper()}-SEED-001"],
                    "observed_run_seed_histogram": {f"{profile.upper()}-SEED-001": 1},
                    "runs_without_observed_seed": 0,
                    "last_observed_seed": f"{profile.upper()}-SEED-001",
                    "non_combat_capability": capability_summary,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=1,
            combat_steps=0,
            heuristic_steps=1,
            total_reward=0.0,
            final_screen="MAP",
            final_run_id=f"run-{profile}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=Path(profile),
            combat_performance={"combat_win_rate": 0.0, "reward_per_combat": 0.0, "reward_per_combat_step": 0.0},
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=0,
        )

    report = run_benchmark_suite(
        manifest,
        output_root=tmp_path / "artifacts",
        env_factory=fake_env_factory,
        policy_evaluation_fn=fake_policy_evaluation_fn,
    )

    case_summary = load_benchmark_suite_summary(report.summary_path)["cases"][0]
    promotion = case_summary["promotion"]
    capability = case_summary["non_combat_capability"]
    assert promotion["passed"] is False
    assert capability["comparison"]["new_regression_count"] == 1
    assert capability["comparison"]["new_regression_keys"] == [
        "repo_action_space_gap|unsupported_action_descriptor|MAP|mystery_map_action"
    ]
    failed_names = {check["name"] for check in promotion["failed_checks"]}
    assert "new_non_combat_capability_regressions" in failed_names


def test_run_benchmark_suite_supports_policy_profile_eval_and_compare(tmp_path: Path) -> None:
    manifest_path = tmp_path / "suite-policy.toml"
    manifest_path.write_text(
        """
schema_version = 1
suite_name = "bench-policy"

[stats]
bootstrap_resamples = 200
confidence_level = 0.95
seed = 17

[[cases]]
case_id = "eval-planner"
mode = "eval"
policy_profile = "planner"
repeats = 2
prepare_target = "none"

[[cases]]
case_id = "compare-policy"
mode = "compare"
baseline_policy_profile = "baseline"
candidate_policy_profile = "planner"
repeats = 2
prepare_target = "none"
""".strip(),
        encoding="utf-8",
    )

    main_menu = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
    )

    def fake_env_factory(_base_url: str, _timeout: float) -> FakeEnv:
        return FakeEnv(main_menu)

    def fake_policy_eval_fn(**kwargs) -> CombatEvaluationReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")

        policy_profile = kwargs["policy_profile"]
        reward = 1.0 if policy_profile == "baseline" else 3.0
        win_rate = 0.0 if policy_profile == "baseline" else 1.0
        summary_path.write_text(
            json.dumps(
                {
                    "policy_pack_histogram": {policy_profile: 3},
                    "policy_handler_histogram": {"combat-hand-planner" if policy_profile == "planner" else "combat-greedy-ranker": 2},
                    "planner_histogram": {"combat-hand-planner-v1": 2} if policy_profile == "planner" else {},
                    "stop_reason": "max_runs_reached",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=4,
            combat_steps=2,
            heuristic_steps=4,
            total_reward=reward,
            final_screen="GAME_OVER",
            final_run_id=f"run-{policy_profile}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={
                "combat_win_rate": win_rate,
                "reward_per_combat": reward,
                "reward_per_combat_step": reward / 2.0,
                "won_combats": int(win_rate > 0),
                "lost_combats": int(win_rate == 0),
            },
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
        )

    report = run_benchmark_suite(
        manifest_path,
        output_root=tmp_path / "artifacts",
        env_factory=fake_env_factory,
        policy_evaluation_fn=fake_policy_eval_fn,
    )

    summary = load_benchmark_suite_summary(report.summary_path)
    cases = {case["case_id"]: case for case in summary["cases"]}
    assert cases["eval-planner"]["checkpoint_paths"]["policy_profile"] == "planner"
    assert cases["eval-planner"]["planner_histogram"] == {"combat-hand-planner-v1": 4}
    assert cases["compare-policy"]["checkpoint_paths"]["baseline_policy_profile"] == "baseline"
    assert cases["compare-policy"]["better_checkpoint_label"] == "candidate"
    assert cases["compare-policy"]["planner_histogram"]["candidate"] == {"combat-hand-planner-v1": 4}


def test_run_benchmark_suite_passes_predictor_configs_into_eval_and_compare(tmp_path: Path) -> None:
    predictor_dir = tmp_path / "predictors"
    predictor_dir.mkdir()
    model_path = predictor_dir / "combat-outcome-predictor.json"
    model_path.write_text("{}", encoding="utf-8")

    manifest_path = tmp_path / "suite-predictor.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_name": "bench-predictor",
                "stats": {"bootstrap_resamples": 200, "confidence_level": 0.95, "seed": 19},
                "cases": [
                    {
                        "case_id": "eval-policy-predictor",
                        "mode": "eval",
                        "policy_profile": "baseline",
                        "repeats": 2,
                        "prepare_target": "none",
                        "predictor": {
                            "model_path": "./predictors/combat-outcome-predictor.json",
                            "mode": "assist",
                            "hooks": ["map", "combat"],
                        },
                    },
                    {
                        "case_id": "compare-policy-predictor",
                        "mode": "compare",
                        "baseline_policy_profile": "baseline",
                        "candidate_policy_profile": "baseline",
                        "repeats": 2,
                        "prepare_target": "none",
                        "baseline_predictor": {
                            "model_path": "./predictors/combat-outcome-predictor.json",
                            "mode": "heuristic_only",
                            "hooks": ["combat"],
                        },
                        "candidate_predictor": {
                            "model_path": "./predictors/combat-outcome-predictor.json",
                            "mode": "dominant",
                            "hooks": ["combat"],
                        },
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    policy_eval_calls: list[dict[str, object]] = []

    def fake_policy_eval_fn(**kwargs) -> CombatEvaluationReport:
        policy_eval_calls.append(kwargs)
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")
        predictor_config = kwargs.get("predictor_config")
        predictor_mode = None if predictor_config is None else predictor_config.mode
        summary_path.write_text(
            json.dumps(
                {
                    "policy_pack_histogram": {"baseline": 2},
                    "predictor_mode_histogram": {} if predictor_mode is None else {predictor_mode: 2},
                    "predictor_domain_histogram": {"combat": 2} if predictor_mode else {},
                    "predictor_model_histogram": (
                        {"combat-outcome-predictor.json": 2}
                        if predictor_config is not None and predictor_config.model_path is not None
                        else {}
                    ),
                    "predictor_value_estimate_stats": {
                        "count": 2 if predictor_mode else 0,
                        "min": 0.5 if predictor_mode else None,
                        "mean": 0.75 if predictor_mode else None,
                        "max": 1.0 if predictor_mode else None,
                    },
                    "stop_reason": "max_runs_reached",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        reward = 1.0 if predictor_mode in {None, "heuristic_only"} else 3.0
        win_rate = 0.0 if predictor_mode in {None, "heuristic_only"} else 1.0
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=4,
            combat_steps=2,
            heuristic_steps=4,
            total_reward=reward,
            final_screen="GAME_OVER",
            final_run_id=f"run-{predictor_mode or 'none'}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={
                "combat_win_rate": win_rate,
                "reward_per_combat": reward,
                "reward_per_combat_step": reward / 2.0,
                "won_combats": int(win_rate > 0),
                "lost_combats": int(win_rate == 0),
            },
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
        )

    report = run_benchmark_suite(
        manifest_path,
        output_root=tmp_path / "artifacts",
        env_factory=lambda _base_url, _timeout: FakeEnv(
            _observation(
                GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
                AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
            )
        ),
        policy_evaluation_fn=fake_policy_eval_fn,
    )

    summary = load_benchmark_suite_summary(report.summary_path)
    cases = {case["case_id"]: case for case in summary["cases"]}
    assert any(call["predictor_config"] is not None for call in policy_eval_calls)
    assert cases["eval-policy-predictor"]["predictor"]["mode"] == "assist"
    assert cases["eval-policy-predictor"]["predictor_mode_histogram"] == {"assist": 4}
    assert cases["compare-policy-predictor"]["predictor"]["candidate"]["mode"] == "dominant"
    assert cases["compare-policy-predictor"]["predictor_mode_histogram"]["candidate"] == {"dominant": 4}


def test_run_benchmark_suite_passes_community_prior_configs_into_policy_eval(tmp_path: Path) -> None:
    community_dir = tmp_path / "community"
    route_dir = tmp_path / "public-runs"
    community_dir.mkdir()
    route_dir.mkdir()
    source_path = community_dir / "community-card-stats.jsonl"
    route_source_path = route_dir / "strategic-route-stats.jsonl"
    source_path.write_text("", encoding="utf-8")
    route_source_path.write_text("", encoding="utf-8")

    manifest_path = tmp_path / "suite-community.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_name": "bench-community",
                "stats": {"bootstrap_resamples": 200, "confidence_level": 0.95, "seed": 31},
                "cases": [
                    {
                        "case_id": "eval-policy-community",
                        "mode": "eval",
                        "policy_profile": "baseline",
                        "repeats": 1,
                        "prepare_target": "none",
                        "community_prior": {
                            "source_path": "./community/community-card-stats.jsonl",
                            "route_source_path": "./public-runs/strategic-route-stats.jsonl",
                            "shop_buy_weight": 1.4,
                        },
                    },
                    {
                        "case_id": "compare-policy-community",
                        "mode": "compare",
                        "baseline_policy_profile": "baseline",
                        "candidate_policy_profile": "planner",
                        "repeats": 1,
                        "prepare_target": "none",
                        "baseline_community_prior": {
                            "source_path": "./community/community-card-stats.jsonl",
                            "route_source_path": "./public-runs/strategic-route-stats.jsonl",
                            "reward_pick_weight": 0.8,
                        },
                        "candidate_community_prior": {
                            "source_path": "./community/community-card-stats.jsonl",
                            "route_source_path": "./public-runs/strategic-route-stats.jsonl",
                            "reward_pick_weight": 1.6,
                        },
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    policy_eval_calls: list[dict[str, object]] = []

    def fake_policy_eval_fn(**kwargs) -> CombatEvaluationReport:
        policy_eval_calls.append(kwargs)
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")
        summary_path.write_text(
            json.dumps(
                {
                    "policy_pack_histogram": {str(kwargs["policy_profile"]): 1},
                    "stop_reason": "max_runs_reached",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=2,
            combat_steps=1,
            heuristic_steps=2,
            total_reward=1.0,
            final_screen="GAME_OVER",
            final_run_id=f"run-{kwargs['session_name']}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={"combat_win_rate": 1.0, "reward_per_combat": 1.0, "reward_per_combat_step": 1.0},
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
        )

    report = run_benchmark_suite(
        manifest_path,
        output_root=tmp_path / "artifacts",
        env_factory=lambda _base_url, _timeout: FakeEnv(
            _observation(
                GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
                AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
            )
        ),
        policy_evaluation_fn=fake_policy_eval_fn,
    )

    summary = load_benchmark_suite_summary(report.summary_path)
    cases = {case["case_id"]: case for case in summary["cases"]}
    assert any(call.get("community_prior_config") is not None for call in policy_eval_calls)
    eval_case = next(call for call in policy_eval_calls if call["session_name"] == "iteration-001")
    assert eval_case["community_prior_config"].source_path == source_path.resolve()
    assert eval_case["community_prior_config"].route_source_path == route_source_path.resolve()
    compare_calls = [call for call in policy_eval_calls if call["session_name"].startswith(("baseline-", "candidate-"))]
    assert compare_calls[0]["community_prior_config"] is not None
    assert compare_calls[1]["community_prior_config"] is not None
    assert cases["eval-policy-community"]["community_prior"]["shop_buy_weight"] == 1.4
    assert cases["eval-policy-community"]["public_sources"]["community_prior"]["diagnostics"]["card"]["artifact_family"] == "community_card_stats"
    assert cases["eval-policy-community"]["public_sources"]["community_prior"]["diagnostics"]["route"]["stat_family"] == "route"
    assert cases["compare-policy-community"]["community_prior"]["candidate"]["reward_pick_weight"] == 1.6
    assert cases["compare-policy-community"]["public_sources"]["candidate"]["community_prior"]["diagnostics"]["route"]["stat_family"] == "route"


def test_run_benchmark_suite_reports_seed_set_diagnostics(tmp_path: Path) -> None:
    manifest_path = tmp_path / "suite-seeds.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_name": "bench-seeds",
                "stats": {"bootstrap_resamples": 200, "confidence_level": 0.95, "seed": 23},
                "cases": [
                    {
                        "case_id": "eval-seeds",
                        "mode": "eval",
                        "policy_profile": "baseline",
                        "repeats": 2,
                        "prepare_target": "none",
                        "scenario": {"seed_set": ["TARGET-001", "TARGET-002"]},
                    },
                    {
                        "case_id": "compare-seeds",
                        "mode": "compare",
                        "baseline_policy_profile": "baseline",
                        "candidate_policy_profile": "planner",
                        "repeats": 2,
                        "prepare_target": "none",
                        "scenario": {"seed_set": ["COMMON-001", "COMMON-002"]},
                    },
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_policy_eval_fn(**kwargs) -> CombatEvaluationReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")

        policy_profile = kwargs["policy_profile"]
        iteration_index = int(session_name.rsplit("-", maxsplit=1)[-1])
        if session_name.startswith("iteration-"):
            observed_seed = "TARGET-001" if iteration_index == 1 else "UNEXPECTED-999"
        elif policy_profile == "baseline":
            observed_seed = "COMMON-001" if iteration_index == 1 else "BASELINE-EXTRA"
        else:
            observed_seed = "COMMON-001" if iteration_index == 1 else "COMMON-002"

        summary_path.write_text(
            json.dumps(
                {
                    "policy_pack_histogram": {policy_profile: 1},
                    "stop_reason": "max_runs_reached",
                    "observed_run_seeds": [observed_seed],
                    "observed_run_seed_histogram": {observed_seed: 1},
                    "runs_without_observed_seed": 0,
                    "last_observed_seed": observed_seed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=2,
            combat_steps=1,
            heuristic_steps=2,
            total_reward=1.0 if policy_profile == "baseline" else 2.0,
            final_screen="GAME_OVER",
            final_run_id=f"run-{policy_profile}-{iteration_index}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={
                "combat_win_rate": 1.0 if policy_profile == "planner" else 0.5,
                "reward_per_combat": 1.0 if policy_profile == "baseline" else 2.0,
                "reward_per_combat_step": 1.0,
                "won_combats": 1,
                "lost_combats": 0,
            },
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
            observed_run_seeds=[observed_seed],
            observed_run_seed_histogram={observed_seed: 1},
            runs_without_observed_seed=0,
            last_observed_seed=observed_seed,
        )

    report = run_benchmark_suite(
        manifest_path,
        output_root=tmp_path / "artifacts",
        env_factory=lambda _base_url, _timeout: FakeEnv(
            _observation(
                GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
                AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
            )
        ),
        policy_evaluation_fn=fake_policy_eval_fn,
    )

    summary = load_benchmark_suite_summary(report.summary_path)
    cases = {case["case_id"]: case for case in summary["cases"]}

    eval_diag = cases["eval-seeds"]["seed_set_diagnostics"]
    assert eval_diag["requested_seed_set"] == ["TARGET-001", "TARGET-002"]
    assert eval_diag["matched_seed_set"] == ["TARGET-001"]
    assert eval_diag["missing_seed_set"] == ["TARGET-002"]
    assert eval_diag["unexpected_seed_set"] == ["UNEXPECTED-999"]
    assert eval_diag["seed_set_coverage"] == 0.5

    compare_diag = cases["compare-seeds"]["seed_set_diagnostics"]
    assert compare_diag["baseline"]["missing_seed_set"] == ["COMMON-002"]
    assert compare_diag["baseline"]["unexpected_seed_set"] == ["BASELINE-EXTRA"]
    assert compare_diag["candidate"]["matched_seed_set"] == ["COMMON-001", "COMMON-002"]
    assert compare_diag["candidate"]["seed_set_coverage"] == 1.0


def test_run_benchmark_suite_emits_scenario_and_strategic_histograms(tmp_path: Path) -> None:
    manifest_path = tmp_path / "suite-strategic.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_name": "bench-strategic",
                "stats": {"bootstrap_resamples": 200, "confidence_level": 0.95, "seed": 29},
                "cases": [
                    {
                        "case_id": "eval-city-champ",
                        "mode": "eval",
                        "policy_profile": "baseline",
                        "repeats": 2,
                        "prepare_target": "none",
                        "scenario": {
                            "floor_min": 17,
                            "floor_max": 33,
                            "act_ids": ["THE_CITY"],
                            "boss_ids": ["THE_CHAMP"],
                            "planner_strategies": ["boss_pathing"],
                            "route_reason_tags": ["search_aoe_tools"],
                            "route_profiles": ["search_aoe_tools"],
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_policy_eval_fn(**kwargs) -> CombatEvaluationReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")
        summary_path.write_text(
            json.dumps(
                {
                    "policy_pack_histogram": {"baseline": 1},
                    "planner_histogram": {"boss-conditioned-route-planner-v1": 1},
                    "route_planner_step_count": 1,
                    "route_planner_boss_histogram": {"THE_CHAMP": 1},
                    "route_planner_reason_tag_histogram": {"search_aoe_tools": 1},
                    "route_planner_path_length_stats": {"count": 1, "min": 3.0, "mean": 3.0, "max": 3.0},
                    "route_planner_selected_score_stats": {"count": 1, "min": 2.4, "mean": 2.4, "max": 2.4},
                    "stop_reason": "max_runs_reached",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=2,
            combat_steps=1,
            heuristic_steps=1,
            total_reward=2.0,
            final_screen="GAME_OVER",
            final_run_id="run-strategic",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={
                "combat_win_rate": 1.0,
                "reward_per_combat": 2.0,
                "reward_per_combat_step": 2.0,
                "won_combats": 1,
                "lost_combats": 0,
            },
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
        )

    report = run_benchmark_suite(
        manifest_path,
        output_root=tmp_path / "artifacts",
        env_factory=lambda _base_url, _timeout: FakeEnv(
            _observation(
                GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
                AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
            )
        ),
        policy_evaluation_fn=fake_policy_eval_fn,
    )

    summary = load_benchmark_suite_summary(report.summary_path)
    assert summary["scenario_histograms"]["boss_ids"] == {"THE_CHAMP": 1}
    assert summary["scenario_histograms"]["planner_strategies"] == {"boss_pathing": 1}
    assert summary["strategic"]["eval"]["boss_histogram"] == {"THE_CHAMP": 2}
    case = summary["cases"][0]
    assert case["strategic"]["route_planner_step_count"] == 2
    assert case["route_reason_tag_histogram"] == {"search_aoe_tools": 2}


def test_run_benchmark_suite_emits_route_diagnostics_and_promotion_for_fixed_seed_compare(tmp_path: Path) -> None:
    manifest_path = tmp_path / "suite-fixed-seed.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_name": "bench-fixed-seed-strategic",
                "stats": {"bootstrap_resamples": 200, "confidence_level": 0.95, "seed": 31},
                "cases": [
                    {
                        "case_id": "strategic-compare",
                        "mode": "compare",
                        "baseline_policy_profile": "baseline",
                        "candidate_policy_profile": "planner",
                        "repeats": 2,
                        "prepare_target": "none",
                        "game_run_contract": {
                            "game_seed": "FIXED-SEED-001",
                            "character_id": "IRONCLAD",
                            "ascension": 0,
                            "progress_profile": "debug-unlocked",
                        },
                        "promotion": {
                            "min_seed_set_coverage": 1.0,
                            "min_route_decision_count": 4,
                            "min_route_decision_overlap_rate": 0.5,
                            "min_delta_total_reward": 1.0,
                            "min_delta_combat_win_rate": 0.4,
                            "min_delta_route_quality_score": 0.5,
                            "min_delta_pre_boss_readiness": 0.5,
                            "max_delta_route_risk_score": 0.0,
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    policy_eval_calls: list[dict[str, object]] = []

    def _write_route_log(log_path: Path, *, variant: str) -> None:
        shared_step = {
            "record_type": "step",
            "step_index": 1,
            "floor": 4,
            "decision_metadata": {
                "route_planner": {
                    "planner_name": "boss-conditioned-route-planner-v1",
                    "planner_strategy": "boss_pathing",
                    "boss_encounter_id": "THE_CHAMP",
                    "selected": {
                        "action_id": "path-1",
                        "score": 2.0,
                        "path": ["M", "E", "R"],
                        "path_node_types": ["monster", "elite", "rest"],
                        "reason_tags": ["search_aoe_tools"],
                        "first_rest_distance": 1,
                        "first_elite_distance": 1,
                        "rest_count": 1,
                        "shop_count": 0,
                        "elite_count": 1,
                        "event_count": 0,
                        "treasure_count": 0,
                        "monster_count": 1,
                        "elites_before_rest": 1,
                        "remaining_distance_to_boss": 9,
                    },
                }
            },
        }
        baseline_second = {
            "record_type": "step",
            "step_index": 2,
            "floor": 8,
            "decision_metadata": {
                "route_planner": {
                    "planner_name": "boss-conditioned-route-planner-v1",
                    "planner_strategy": "boss_pathing",
                    "boss_encounter_id": "THE_CHAMP",
                    "selected": {
                        "action_id": "path-2-baseline",
                        "score": 1.5,
                        "path": ["M", "E", "?"],
                        "path_node_types": ["monster", "elite", "event"],
                        "reason_tags": ["search_scaling"],
                        "first_rest_distance": 5,
                        "first_elite_distance": 1,
                        "rest_count": 0,
                        "shop_count": 0,
                        "elite_count": 2,
                        "event_count": 1,
                        "treasure_count": 0,
                        "monster_count": 2,
                        "elites_before_rest": 2,
                        "remaining_distance_to_boss": 6,
                    },
                }
            },
        }
        candidate_second = {
            "record_type": "step",
            "step_index": 2,
            "floor": 8,
            "decision_metadata": {
                "route_planner": {
                    "planner_name": "boss-conditioned-route-planner-v1",
                    "planner_strategy": "boss_pathing",
                    "boss_encounter_id": "THE_CHAMP",
                    "selected": {
                        "action_id": "path-2-candidate",
                        "score": 2.4,
                        "path": ["M", "$", "R"],
                        "path_node_types": ["monster", "shop", "rest"],
                        "reason_tags": ["search_scaling", "stabilize_before_boss"],
                        "first_rest_distance": 2,
                        "first_elite_distance": 3,
                        "rest_count": 1,
                        "shop_count": 1,
                        "elite_count": 1,
                        "event_count": 0,
                        "treasure_count": 0,
                        "monster_count": 1,
                        "elites_before_rest": 1,
                        "remaining_distance_to_boss": 6,
                    },
                }
            },
        }
        steps = [shared_step, baseline_second if variant == "baseline" else candidate_second]
        log_path.write_text(
            "\n".join(json.dumps(step, ensure_ascii=False) for step in steps) + "\n",
            encoding="utf-8",
        )

    def fake_policy_eval_fn(**kwargs) -> CombatEvaluationReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        combat_outcomes_path.write_text("", encoding="utf-8")

        policy_profile = str(kwargs["policy_profile"])
        variant = "baseline" if session_name.startswith("baseline-") else "candidate"
        reward = 2.0 if variant == "baseline" else 4.5
        win_rate = 0.5 if variant == "baseline" else 1.0
        game_run_contract = kwargs["game_run_contract"]
        assert game_run_contract is not None
        policy_eval_calls.append(
            {
                "policy_profile": policy_profile,
                "game_seed": game_run_contract.game_seed,
                "character_id": game_run_contract.character_id,
            }
        )

        _write_route_log(log_path, variant=variant)
        summary_path.write_text(
            json.dumps(
                {
                    "policy_pack_histogram": {policy_profile: 1},
                    "planner_histogram": {"boss-conditioned-route-planner-v1": 1},
                    "route_planner_step_count": 2,
                    "route_planner_boss_histogram": {"THE_CHAMP": 2},
                    "route_planner_reason_tag_histogram": {
                        "search_aoe_tools": 1,
                        **({"search_scaling": 1} if variant == "baseline" else {"search_scaling": 1, "stabilize_before_boss": 1}),
                    },
                    "route_planner_path_length_stats": {"count": 2, "min": 3.0, "mean": 3.0, "max": 3.0},
                    "route_planner_selected_score_stats": {
                        "count": 2,
                        "min": 1.5 if variant == "baseline" else 2.0,
                        "mean": 1.75 if variant == "baseline" else 2.2,
                        "max": 2.0 if variant == "baseline" else 2.4,
                    },
                    "stop_reason": "max_runs_reached",
                    "observed_run_seeds": [game_run_contract.game_seed],
                    "observed_run_seed_histogram": {game_run_contract.game_seed: 1},
                    "runs_without_observed_seed": 0,
                    "last_observed_seed": game_run_contract.game_seed,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=4,
            combat_steps=2,
            heuristic_steps=4,
            total_reward=reward,
            final_screen="GAME_OVER",
            final_run_id=f"run-{variant}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={
                "combat_win_rate": win_rate,
                "reward_per_combat": reward,
                "reward_per_combat_step": reward / 2.0,
                "won_combats": int(win_rate > 0.5),
                "lost_combats": int(win_rate <= 0.5),
            },
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
            observed_run_seeds=[game_run_contract.game_seed],
            observed_run_seed_histogram={game_run_contract.game_seed: 1},
            runs_without_observed_seed=0,
            last_observed_seed=game_run_contract.game_seed,
        )

    report = run_benchmark_suite(
        manifest_path,
        output_root=tmp_path / "artifacts",
        env_factory=lambda _base_url, _timeout: FakeEnv(
            _observation(
                GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
                AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
            )
        ),
        policy_evaluation_fn=fake_policy_eval_fn,
    )

    summary = load_benchmark_suite_summary(report.summary_path)
    case = summary["cases"][0]
    comparison = case["route_diagnostics"]["comparison"]
    assert all(call["game_seed"] == "FIXED-SEED-001" for call in policy_eval_calls)
    assert case["config"]["game_run_contract"]["game_seed"] == "FIXED-SEED-001"
    assert case["seed_set_diagnostics"]["baseline"]["seed_set_coverage"] == 1.0
    assert case["seed_set_diagnostics"]["candidate"]["seed_set_coverage"] == 1.0
    assert comparison["route_decision_pair_count"] == 4
    assert comparison["route_decision_overlap_rate"] == 0.5
    assert case["metrics"]["delta_route_quality_score"]["mean"] > 0.5
    assert case["metrics"]["delta_pre_boss_readiness"]["mean"] > 0.5
    assert case["metrics"]["delta_route_risk_score"]["mean"] < 0.0
    assert case["promotion"]["passed"] is True
    assert case["promotion"]["promotion_candidate_count"] == 1
    assert summary["promotion"]["passed_case_count"] == 1


def test_run_benchmark_suite_integrates_shadow_compare_into_case_suite_and_promotion(tmp_path: Path) -> None:
    shadow_dir = build_shadow_encounter_fixture(tmp_path)
    manifest_path = tmp_path / "suite-shadow-integrated.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_name": "bench-shadow-integrated",
                "stats": {"bootstrap_resamples": 200, "confidence_level": 0.95, "seed": 37},
                "cases": [
                    {
                        "case_id": "compare-shadow",
                        "mode": "compare",
                        "baseline_policy_profile": "baseline",
                        "candidate_policy_profile": "planner",
                        "repeats": 2,
                        "prepare_target": "none",
                        "shadow": {"source": str(shadow_dir.resolve())},
                        "promotion": {
                            "min_seed_set_coverage": 0.0,
                            "min_route_decision_count": 0,
                            "min_route_decision_overlap_rate": 0.0,
                            "min_delta_total_reward": 0.0,
                            "min_delta_combat_win_rate": 0.0,
                            "min_delta_route_quality_score": 0.0,
                            "min_delta_pre_boss_readiness": 0.0,
                            "max_delta_route_risk_score": 1.0,
                            "min_shadow_comparable_encounter_count": 1,
                            "min_shadow_candidate_advantage_rate": 0.0,
                            "min_shadow_delta_first_action_match_rate": 0.0,
                            "min_shadow_delta_trace_hit_rate": 0.0,
                        },
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_policy_eval_fn(**kwargs) -> CombatEvaluationReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        log_path.write_text("", encoding="utf-8")
        combat_outcomes_path.write_text("", encoding="utf-8")
        summary_path.write_text(
            json.dumps(
                {
                    "policy_pack_histogram": {str(kwargs["policy_profile"]): 1},
                    "planner_histogram": (
                        {"combat-hand-planner-v1": 1} if kwargs["policy_profile"] == "planner" else {}
                    ),
                    "route_planner_step_count": 0,
                    "stop_reason": "max_runs_reached",
                    "observed_run_seeds": [],
                    "observed_run_seed_histogram": {},
                    "runs_without_observed_seed": 0,
                    "last_observed_seed": None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        reward = 1.0 if kwargs["policy_profile"] == "baseline" else 1.5
        win_rate = 0.5 if kwargs["policy_profile"] == "baseline" else 1.0
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=2,
            combat_steps=1,
            heuristic_steps=1,
            total_reward=reward,
            final_screen="GAME_OVER",
            final_run_id=f"run-{kwargs['policy_profile']}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={
                "combat_win_rate": win_rate,
                "reward_per_combat": reward,
                "reward_per_combat_step": reward,
                "won_combats": int(win_rate > 0.5),
                "lost_combats": int(win_rate <= 0.5),
            },
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
        )

    report = run_benchmark_suite(
        manifest_path,
        output_root=tmp_path / "artifacts",
        env_factory=lambda _base_url, _timeout: FakeEnv(
            _observation(
                GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
                AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
            )
        ),
        policy_evaluation_fn=fake_policy_eval_fn,
    )

    summary = load_benchmark_suite_summary(report.summary_path)
    case = summary["cases"][0]
    assert case["shadow"]["enabled"] is True
    assert case["shadow"]["comparable_encounter_count"] == 1
    assert case["shadow"]["delta_metrics"]["delta_first_action_match_rate"] == 0.0
    assert case["artifacts"]["shadow_summary_path"] is not None
    assert case["promotion"]["passed"] is True
    assert any(check["name"] == "shadow_comparable_encounter_count" for check in case["promotion"]["checks"])
    assert summary["shadow"]["configured_case_count"] == 1
    assert summary["shadow"]["comparable_encounter_count"] == 1
    assert summary["shadow"]["candidate_advantage_rate_stats"]["mean"] == 0.0


def test_run_benchmark_suite_emits_eval_community_alignment_metrics(tmp_path: Path) -> None:
    manifest_path = tmp_path / "suite-community-eval.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_name": "bench-community-eval",
                "stats": {"bootstrap_resamples": 200, "confidence_level": 0.95, "seed": 41},
                "cases": [
                    {
                        "case_id": "community-eval",
                        "mode": "eval",
                        "policy_profile": "baseline",
                        "repeats": 2,
                        "prepare_target": "none",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_policy_eval_fn(**kwargs) -> CombatEvaluationReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        combat_outcomes_path.write_text("", encoding="utf-8")
        iteration_index = int(session_name.rsplit("-", maxsplit=1)[-1])
        _write_community_alignment_log(
            log_path,
            steps=_community_eval_steps(iteration_index),
        )
        summary_path.write_text(
            json.dumps(
                {
                    "policy_pack_histogram": {"baseline": 1},
                    "planner_histogram": {},
                    "stop_reason": "max_runs_reached",
                    "observed_run_seeds": [f"COMMUNITY-EVAL-{iteration_index:03d}"],
                    "observed_run_seed_histogram": {f"COMMUNITY-EVAL-{iteration_index:03d}": 1},
                    "runs_without_observed_seed": 0,
                    "last_observed_seed": f"COMMUNITY-EVAL-{iteration_index:03d}",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=4,
            combat_steps=1,
            heuristic_steps=3,
            total_reward=2.0 + iteration_index,
            final_screen="GAME_OVER",
            final_run_id=f"community-eval-{iteration_index}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={
                "combat_win_rate": 1.0,
                "reward_per_combat": 2.0 + iteration_index,
                "reward_per_combat_step": 2.0 + iteration_index,
                "won_combats": 1,
                "lost_combats": 0,
            },
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
            observed_run_seeds=[f"COMMUNITY-EVAL-{iteration_index:03d}"],
            observed_run_seed_histogram={f"COMMUNITY-EVAL-{iteration_index:03d}": 1},
            runs_without_observed_seed=0,
            last_observed_seed=f"COMMUNITY-EVAL-{iteration_index:03d}",
        )

    report = run_benchmark_suite(
        manifest_path,
        output_root=tmp_path / "artifacts",
        env_factory=lambda _base_url, _timeout: FakeEnv(
            _observation(
                GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
                AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
            )
        ),
        policy_evaluation_fn=fake_policy_eval_fn,
    )

    summary = load_benchmark_suite_summary(report.summary_path)
    case = summary["cases"][0]
    alignment = case["community_alignment"]["summary"]
    assert alignment["decision_step_count"] == 5
    assert alignment["eligible_decision_count"] == 3
    assert alignment["aligned_decision_count"] == 2
    assert alignment["top_choice_match_rate"] == 2 / 3
    assert alignment["weighted_top_choice_match_rate"] == pytest.approx(1.5 / 2.1)
    assert alignment["opportunity_coverage"] == 3 / 5
    assert alignment["domain_histogram"] == {"reward_pick": 1, "selection_remove": 1, "shop_buy": 1}
    assert alignment["domains"]["shop_buy"]["aligned_decision_count"] == 0
    assert alignment["alignment_regret_stats"]["mean"] == pytest.approx(1 / 6)
    assert case["metrics"]["community_top_choice_match_rate"]["mean"] == 0.75
    assert case["metrics"]["community_opportunity_coverage"]["mean"] == (2 / 3 + 1 / 2) / 2
    assert summary["community_alignment"]["eval"]["eligible_decision_count"] == 3
    assert summary["community_alignment"]["eval"]["source_name_histogram"] == {"fixture": 3}


def test_run_benchmark_suite_emits_compare_community_alignment_metrics(tmp_path: Path) -> None:
    manifest_path = tmp_path / "suite-community-compare.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite_name": "bench-community-compare",
                "stats": {"bootstrap_resamples": 200, "confidence_level": 0.95, "seed": 43},
                "cases": [
                    {
                        "case_id": "community-compare",
                        "mode": "compare",
                        "baseline_policy_profile": "baseline",
                        "candidate_policy_profile": "planner",
                        "repeats": 2,
                        "prepare_target": "none",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_policy_eval_fn(**kwargs) -> CombatEvaluationReport:
        session_name = kwargs["session_name"]
        session_dir = Path(kwargs["output_root"]) / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        summary_path = session_dir / "summary.json"
        log_path = session_dir / "combat-eval.jsonl"
        combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
        combat_outcomes_path.write_text("", encoding="utf-8")
        variant = "baseline" if session_name.startswith("baseline-") else "candidate"
        iteration_index = int(session_name.rsplit("-", maxsplit=1)[-1])
        _write_community_alignment_log(
            log_path,
            steps=_community_compare_steps(variant=variant, iteration_index=iteration_index),
        )
        reward = 2.0 if variant == "baseline" else 3.0
        win_rate = 0.5 if variant == "baseline" else 1.0
        summary_path.write_text(
            json.dumps(
                {
                    "policy_pack_histogram": {variant: 1},
                    "planner_histogram": {"planner": 1} if variant == "candidate" else {},
                    "stop_reason": "max_runs_reached",
                    "observed_run_seeds": [f"{variant.upper()}-COMMUNITY-{iteration_index:03d}"],
                    "observed_run_seed_histogram": {f"{variant.upper()}-COMMUNITY-{iteration_index:03d}": 1},
                    "runs_without_observed_seed": 0,
                    "last_observed_seed": f"{variant.upper()}-COMMUNITY-{iteration_index:03d}",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return CombatEvaluationReport(
            base_url=kwargs["base_url"],
            env_steps=4,
            combat_steps=1,
            heuristic_steps=3,
            total_reward=reward + iteration_index,
            final_screen="GAME_OVER",
            final_run_id=f"{variant}-community-{iteration_index}",
            log_path=log_path,
            summary_path=summary_path,
            combat_outcomes_path=combat_outcomes_path,
            checkpoint_path=session_dir / "policy-pack-eval.json",
            combat_performance={
                "combat_win_rate": win_rate,
                "reward_per_combat": reward + iteration_index,
                "reward_per_combat_step": reward + iteration_index,
                "won_combats": int(win_rate > 0.5),
                "lost_combats": int(win_rate <= 0.5),
            },
            stop_reason="max_runs_reached",
            completed_run_count=1,
            completed_combat_count=1,
            observed_run_seeds=[f"{variant.upper()}-COMMUNITY-{iteration_index:03d}"],
            observed_run_seed_histogram={f"{variant.upper()}-COMMUNITY-{iteration_index:03d}": 1},
            runs_without_observed_seed=0,
            last_observed_seed=f"{variant.upper()}-COMMUNITY-{iteration_index:03d}",
        )

    report = run_benchmark_suite(
        manifest_path,
        output_root=tmp_path / "artifacts",
        env_factory=lambda _base_url, _timeout: FakeEnv(
            _observation(
                GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
                AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_character_select")]),
            )
        ),
        policy_evaluation_fn=fake_policy_eval_fn,
    )

    summary = load_benchmark_suite_summary(report.summary_path)
    case = summary["cases"][0]
    comparison = case["community_alignment"]["comparison"]
    assert comparison["paired_iteration_count"] == 2
    assert comparison["comparable_iteration_count"] == 2
    assert comparison["delta_top_choice_match_rate_stats"]["mean"] == 0.75
    assert comparison["delta_weighted_top_choice_match_rate_stats"]["mean"] == pytest.approx(0.5 + (3 / 14))
    assert comparison["delta_opportunity_coverage_stats"]["mean"] == 0.0
    assert comparison["delta_alignment_regret_stats"]["mean"] == -0.375
    assert comparison["delta_selected_score_bonus_stats"]["mean"] == 0.375
    assert comparison["delta_best_score_bonus_stats"]["mean"] == 0.0
    assert case["metrics"]["delta_community_top_choice_match_rate"]["mean"] == 0.75
    assert case["metrics"]["delta_community_alignment_regret"]["mean"] == -0.375
    suite_compare = summary["community_alignment"]["compare"]
    assert suite_compare["baseline"]["eligible_decision_count"] == 3
    assert suite_compare["candidate"]["eligible_decision_count"] == 3
    assert suite_compare["comparison"]["configured_case_count"] == 1
    assert suite_compare["comparison"]["delta_top_choice_match_rate_stats"]["mean"] == 0.75


def _write_community_alignment_log(log_path: Path, *, steps: list[dict[str, object]]) -> None:
    log_path.write_text(
        "\n".join(json.dumps(step, ensure_ascii=False) for step in steps) + "\n",
        encoding="utf-8",
    )


def _community_eval_steps(iteration_index: int) -> list[dict[str, object]]:
    if iteration_index == 1:
        return [
            _community_step(
                step_index=1,
                floor=3,
                decision_stage="reward",
                selected=_community_candidate(
                    action_id="choose_reward_card|option=1",
                    action="choose_reward_card",
                    reason="take_reward_card",
                    heuristic_score=1.3,
                    final_score=2.5,
                    domain="reward_pick",
                    card_id="CARD_B",
                    score_bonus=1.2,
                    confidence=0.8,
                ),
                ranked_candidates=[
                    _community_candidate(
                        action_id="choose_reward_card|option=1",
                        action="choose_reward_card",
                        reason="take_reward_card",
                        heuristic_score=1.3,
                        final_score=2.5,
                        domain="reward_pick",
                        card_id="CARD_B",
                        score_bonus=1.2,
                        confidence=0.8,
                    ),
                    _community_candidate(
                        action_id="choose_reward_card|option=0",
                        action="choose_reward_card",
                        reason="take_reward_card",
                        heuristic_score=0.8,
                        final_score=1.1,
                        domain="reward_pick",
                        card_id="CARD_A",
                        score_bonus=0.4,
                        confidence=0.8,
                    ),
                ],
            ),
            _community_step(
                step_index=2,
                floor=4,
                decision_stage="shop",
                selected=_community_candidate(
                    action_id="buy_card|slot=0",
                    action="buy_card",
                    reason="shop_buy",
                    heuristic_score=1.0,
                    final_score=1.2,
                    domain="shop_buy",
                    card_id="CARD_A",
                    score_bonus=0.2,
                    confidence=0.6,
                ),
                ranked_candidates=[
                    _community_candidate(
                        action_id="buy_card|slot=1",
                        action="buy_card",
                        reason="shop_buy",
                        heuristic_score=1.1,
                        final_score=1.8,
                        domain="shop_buy",
                        card_id="CARD_B",
                        score_bonus=0.7,
                        confidence=0.6,
                    ),
                    _community_candidate(
                        action_id="buy_card|slot=0",
                        action="buy_card",
                        reason="shop_buy",
                        heuristic_score=1.0,
                        final_score=1.2,
                        domain="shop_buy",
                        card_id="CARD_A",
                        score_bonus=0.2,
                        confidence=0.6,
                    ),
                ],
            ),
            _plain_step(step_index=3, floor=5, decision_stage="combat"),
        ]
    return [
        _community_step(
            step_index=1,
            floor=7,
            decision_stage="selection",
            selected=_community_candidate(
                action_id="remove_card|index=0",
                action="remove_card",
                reason="selection_remove",
                heuristic_score=1.6,
                final_score=2.0,
                domain="selection_remove",
                card_id="Strike_R",
                score_bonus=0.5,
                confidence=0.7,
            ),
            ranked_candidates=[
                _community_candidate(
                    action_id="remove_card|index=0",
                    action="remove_card",
                    reason="selection_remove",
                    heuristic_score=1.6,
                    final_score=2.0,
                    domain="selection_remove",
                    card_id="Strike_R",
                    score_bonus=0.5,
                    confidence=0.7,
                ),
                _community_candidate(
                    action_id="remove_card|index=1",
                    action="remove_card",
                    reason="selection_remove",
                    heuristic_score=0.5,
                    final_score=0.9,
                    domain="selection_remove",
                    card_id="Defend_R",
                    score_bonus=0.1,
                    confidence=0.7,
                ),
            ],
        ),
        _plain_step(step_index=2, floor=8, decision_stage="map"),
    ]


def _community_compare_steps(*, variant: str, iteration_index: int) -> list[dict[str, object]]:
    if iteration_index == 1:
        shop_selected = (
            _community_candidate(
                action_id="buy_card|slot=0",
                action="buy_card",
                reason="shop_buy",
                heuristic_score=1.0,
                final_score=1.2,
                domain="shop_buy",
                card_id="CARD_A",
                score_bonus=0.2,
                confidence=0.6,
            )
            if variant == "baseline"
            else _community_candidate(
                action_id="buy_card|slot=1",
                action="buy_card",
                reason="shop_buy",
                heuristic_score=1.1,
                final_score=1.8,
                domain="shop_buy",
                card_id="CARD_B",
                score_bonus=0.7,
                confidence=0.6,
            )
        )
        return [
            _community_step(
                step_index=1,
                floor=3,
                decision_stage="reward",
                selected=_community_candidate(
                    action_id="choose_reward_card|option=1",
                    action="choose_reward_card",
                    reason="take_reward_card",
                    heuristic_score=1.3,
                    final_score=2.5,
                    domain="reward_pick",
                    card_id="CARD_B",
                    score_bonus=1.2,
                    confidence=0.8,
                ),
                ranked_candidates=[
                    _community_candidate(
                        action_id="choose_reward_card|option=1",
                        action="choose_reward_card",
                        reason="take_reward_card",
                        heuristic_score=1.3,
                        final_score=2.5,
                        domain="reward_pick",
                        card_id="CARD_B",
                        score_bonus=1.2,
                        confidence=0.8,
                    ),
                    _community_candidate(
                        action_id="choose_reward_card|option=0",
                        action="choose_reward_card",
                        reason="take_reward_card",
                        heuristic_score=0.8,
                        final_score=1.1,
                        domain="reward_pick",
                        card_id="CARD_A",
                        score_bonus=0.4,
                        confidence=0.8,
                    ),
                ],
            ),
            _community_step(
                step_index=2,
                floor=4,
                decision_stage="shop",
                selected=shop_selected,
                ranked_candidates=[
                    _community_candidate(
                        action_id="buy_card|slot=1",
                        action="buy_card",
                        reason="shop_buy",
                        heuristic_score=1.1,
                        final_score=1.8,
                        domain="shop_buy",
                        card_id="CARD_B",
                        score_bonus=0.7,
                        confidence=0.6,
                    ),
                    _community_candidate(
                        action_id="buy_card|slot=0",
                        action="buy_card",
                        reason="shop_buy",
                        heuristic_score=1.0,
                        final_score=1.2,
                        domain="shop_buy",
                        card_id="CARD_A",
                        score_bonus=0.2,
                        confidence=0.6,
                    ),
                ],
            ),
            _plain_step(step_index=3, floor=5, decision_stage="combat"),
        ]
    remove_selected = (
        _community_candidate(
            action_id="remove_card|index=1",
            action="remove_card",
            reason="selection_remove",
            heuristic_score=0.5,
            final_score=0.7,
            domain="selection_remove",
            card_id="Defend_R",
            score_bonus=0.1,
            confidence=0.7,
        )
        if variant == "baseline"
        else _community_candidate(
            action_id="remove_card|index=0",
            action="remove_card",
            reason="selection_remove",
            heuristic_score=1.4,
            final_score=1.9,
            domain="selection_remove",
            card_id="Strike_R",
            score_bonus=0.6,
            confidence=0.7,
        )
    )
    return [
        _community_step(
            step_index=1,
            floor=8,
            decision_stage="selection",
            selected=remove_selected,
            ranked_candidates=[
                _community_candidate(
                    action_id="remove_card|index=0",
                    action="remove_card",
                    reason="selection_remove",
                    heuristic_score=1.4,
                    final_score=1.9,
                    domain="selection_remove",
                    card_id="Strike_R",
                    score_bonus=0.6,
                    confidence=0.7,
                ),
                _community_candidate(
                    action_id="remove_card|index=1",
                    action="remove_card",
                    reason="selection_remove",
                    heuristic_score=0.5,
                    final_score=0.7,
                    domain="selection_remove",
                    card_id="Defend_R",
                    score_bonus=0.1,
                    confidence=0.7,
                ),
            ],
        ),
        _plain_step(step_index=2, floor=9, decision_stage="map"),
    ]


def _community_step(
    *,
    step_index: int,
    floor: int,
    decision_stage: str,
    selected: dict[str, object],
    ranked_candidates: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "record_type": "step",
        "step_index": step_index,
        "floor": floor,
        "decision_stage": decision_stage,
        "decision_metadata": {
            "community_prior": {
                "config": {"source_path": "fixture"},
                "selected": selected,
                "ranked_candidates": ranked_candidates,
            }
        },
    }


def _plain_step(*, step_index: int, floor: int, decision_stage: str) -> dict[str, object]:
    return {
        "record_type": "step",
        "step_index": step_index,
        "floor": floor,
        "decision_stage": decision_stage,
        "decision_metadata": {},
    }


def _community_candidate(
    *,
    action_id: str,
    action: str,
    reason: str,
    heuristic_score: float,
    final_score: float,
    domain: str,
    card_id: str,
    score_bonus: float,
    confidence: float,
) -> dict[str, object]:
    return {
        "action_id": action_id,
        "action": action,
        "reason": reason,
        "heuristic_score": heuristic_score,
        "final_score": final_score,
        "prior": {
            "domain": domain,
            "card_id": card_id,
            "score_bonus": score_bonus,
            "confidence": confidence,
            "source_name": "fixture",
        },
    }


def _observation(state: GameStatePayload, descriptors: AvailableActionsPayload) -> StepObservation:
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type=state.screen,
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )
