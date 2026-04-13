import json
from pathlib import Path

from sts2_rl.data import (
    build_dataset_from_manifest,
    load_dataset_manifest,
    load_dataset_summary,
    load_shadow_combat_encounter_records,
    validate_dataset_manifest,
)


def test_validate_dataset_manifest_supports_json_and_toml(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "session-a"
    artifacts_root.mkdir(parents=True)
    combat_path = artifacts_root / "combat-outcomes.jsonl"
    combat_path.write_text(
        json.dumps(_combat_outcome_payload(run_id="RUN-001", outcome="won", floor=3), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    json_manifest_path = tmp_path / "predictor.json"
    json_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "predictor-json",
                "dataset_kind": "predictor_combat_outcomes",
                "sources": [{"path": str(tmp_path / "artifacts"), "source_kind": "combat_outcomes"}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    toml_manifest_path = tmp_path / "predictor.toml"
    toml_manifest_path.write_text(
        """
schema_version = 1
dataset_name = "predictor-toml"
dataset_kind = "predictor_combat_outcomes"

[[sources]]
path = "artifacts"
source_kind = "combat_outcomes"
recursive = true
""".strip(),
        encoding="utf-8",
    )

    json_manifest = load_dataset_manifest(json_manifest_path)
    toml_manifest = load_dataset_manifest(toml_manifest_path)
    json_report = validate_dataset_manifest(json_manifest_path)
    toml_report = validate_dataset_manifest(toml_manifest_path)

    assert json_manifest.dataset_kind == "predictor_combat_outcomes"
    assert toml_manifest.dataset_name == "predictor-toml"
    assert json_report.source_files == toml_report.source_files
    assert len(json_report.source_files) == 1
    assert json_report.source_files[0].name == "combat-outcomes.jsonl"


def test_build_predictor_dataset_from_manifest_is_deterministic(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts" / "session-b"
    artifacts_root.mkdir(parents=True)
    combat_path = artifacts_root / "combat-outcomes.jsonl"
    payloads = [
        _combat_outcome_payload(run_id=f"RUN-{index:03d}", outcome="won" if index % 2 else "lost", floor=index + 1)
        for index in range(1, 7)
    ]
    with combat_path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    manifest_path = tmp_path / "predictor.toml"
    manifest_path.write_text(
        f"""
schema_version = 1
dataset_name = "predictor-repro"
dataset_kind = "predictor_combat_outcomes"
description = "Synthetic predictor dataset"

[[sources]]
path = "{(tmp_path / "artifacts").as_posix()}"
source_kind = "combat_outcomes"
recursive = true

[filters]
outcomes = ["won", "lost"]

[split]
train_fraction = 0.5
validation_fraction = 0.25
test_fraction = 0.25
seed = 17
group_by = "run_id"

[output]
export_csv = true
include_top_level_records = true
write_split_files = true
""".strip(),
        encoding="utf-8",
    )

    report_a = build_dataset_from_manifest(manifest_path, output_dir=tmp_path / "data" / "predictor-a")
    report_b = build_dataset_from_manifest(manifest_path, output_dir=tmp_path / "data" / "predictor-b")
    summary_a = load_dataset_summary(report_a.output_dir)
    summary_b = load_dataset_summary(report_b.output_dir)

    assert report_a.record_count == 6
    assert report_a.split_counts == {"train": 3, "validation": 2, "test": 1}
    assert report_a.split_counts == report_b.split_counts
    assert (report_a.output_dir / "feature-table.csv").exists()
    assert summary_a["lineage"]["resolved_source_files"] == [str(combat_path.resolve())]
    assert summary_a["split"]["split_group_counts"] == {"train": 3, "validation": 2, "test": 1}
    assert summary_a["boss_histogram"] == {"THE_CHAMP": 6}
    assert summary_a["route_reason_tag_histogram"] == {"search_aoe_tools": 6, "shop_access_before_boss": 6}
    assert summary_a["strategic_coverage"]["route_plan_count"] == 6
    assert summary_a["feature_names"] == summary_b["feature_names"]
    assert (report_a.output_dir / "train.examples.jsonl").read_text(encoding="utf-8") == (
        report_b.output_dir / "train.examples.jsonl"
    ).read_text(encoding="utf-8")
    assert (report_a.output_dir / "validation.examples.jsonl").read_text(encoding="utf-8") == (
        report_b.output_dir / "validation.examples.jsonl"
    ).read_text(encoding="utf-8")


def test_build_trajectory_dataset_writes_summary_and_exports(tmp_path: Path) -> None:
    trajectory_root = tmp_path / "data" / "trajectories" / "session-c"
    trajectory_root.mkdir(parents=True)
    trajectory_path = trajectory_root / "inst-01.jsonl"
    records = [
        {
            "schema_version": 2,
            "record_type": "session_started",
            "timestamp_utc": "2026-04-12T00:00:00+00:00",
            "session_name": "session-c",
        },
        _trajectory_step_payload(
            step_index=1,
            run_id="RUN-001",
            floor=2,
            screen_type="MAP",
            decision_source="heuristic",
        ),
        _trajectory_step_payload(
            step_index=2,
            run_id="RUN-001",
            floor=2,
            screen_type="COMBAT",
            decision_source="policy",
        ),
        _trajectory_step_payload(
            step_index=3,
            run_id="RUN-002",
            floor=4,
            screen_type="EVENT",
            decision_source="heuristic",
        ),
        {
            "schema_version": 2,
            "record_type": "run_finished",
            "timestamp_utc": "2026-04-12T00:00:00+00:00",
            "session_name": "session-c",
            "session_kind": "train",
            "instance_id": "inst-01",
            "run_id": "RUN-001",
            "started_step_index": 1,
            "finished_step_index": 2,
            "reason": "game_over",
            "victory": True,
            "outcome": "won",
            "state_summary": {"screen_type": "GAME_OVER", "run_id": "RUN-001"},
        },
        {
            "schema_version": 2,
            "record_type": "run_finished",
            "timestamp_utc": "2026-04-12T00:00:00+00:00",
            "session_name": "session-c",
            "session_kind": "train",
            "instance_id": "inst-01",
            "run_id": "RUN-002",
            "started_step_index": 3,
            "finished_step_index": 3,
            "reason": "game_over",
            "victory": False,
            "outcome": "lost",
            "state_summary": {"screen_type": "GAME_OVER", "run_id": "RUN-002"},
        },
    ]
    with trajectory_path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in records:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    manifest_path = tmp_path / "trajectory.toml"
    manifest_path.write_text(
        f"""
schema_version = 1
dataset_name = "trajectory-repro"
dataset_kind = "trajectory_steps"

[[sources]]
path = "{(tmp_path / "data" / "trajectories").as_posix()}"
source_kind = "trajectory_log"
recursive = true

[filters]
session_kinds = ["train"]
min_floor = 2

[split]
train_fraction = 0.67
validation_fraction = 0.0
test_fraction = 0.33
seed = 9
group_by = "run_id"

[output]
export_csv = true
include_top_level_records = true
write_split_files = true
""".strip(),
        encoding="utf-8",
    )

    report = build_dataset_from_manifest(manifest_path, output_dir=tmp_path / "data" / "trajectory-out")
    summary = load_dataset_summary(report.output_dir)

    assert report.record_count == 3
    assert report.filtered_out_count == 3
    assert (report.output_dir / "steps.csv").exists()
    assert (report.output_dir / "train.steps.jsonl").exists()
    assert (report.output_dir / "test.steps.jsonl").exists()
    assert summary["screen_histogram"] == {"MAP": 1, "COMBAT": 1, "EVENT": 1}
    assert summary["decision_source_histogram"] == {"heuristic": 2, "policy": 1}
    assert summary["run_outcome_histogram"] == {"won": 2, "lost": 1}
    assert summary["boss_histogram"] == {"THE_CHAMP": 2, "HEXAGHOST": 1}
    assert summary["planner_strategy_histogram"] == {"boss_pathing": 3}
    assert summary["route_reason_tag_histogram"] == {"search_aoe_tools": 2, "early_rest_for_survival": 1}
    assert summary["route_profile_histogram"] == {
        "search_aoe_tools": 2,
        "early_rest_for_survival": 1,
    }
    assert summary["split"]["split_counts"] == report.split_counts
    assert summary["lineage"]["resolved_source_files"] == [str(trajectory_path.resolve())]


def test_build_shadow_combat_dataset_writes_encounter_snapshots_and_summary(tmp_path: Path) -> None:
    trajectory_root = tmp_path / "data" / "trajectories" / "shadow-session"
    trajectory_root.mkdir(parents=True)
    trajectory_path = trajectory_root / "inst-01.jsonl"
    records = [
        {
            "schema_version": 2,
            "record_type": "combat_started",
            "timestamp_utc": "2026-04-12T00:00:00+00:00",
            "session_name": "shadow-session",
            "session_kind": "eval",
            "instance_id": "inst-01",
            "step_index": 0,
            "run_id": "RUN-001",
            "floor": 6,
            "combat_index": 1,
            "enemy_ids": ["SLIME_SMALL"],
            "state_summary": {
                "screen_type": "COMBAT",
                "run_id": "RUN-001",
                "observed_seed": "SEED-001",
                "run": {
                    "character_id": "IRONCLAD",
                    "seed": "SEED-001",
                    "floor": 6,
                    "act_index": 1,
                    "act_id": "THE_CITY",
                    "boss_encounter_id": "THE_CHAMP",
                },
                "combat": {"enemy_ids": ["SLIME_SMALL"], "player_hp": 52, "enemy_hp": [18]},
                "map": {
                    "planned_node_types": ["Monster", "Rest", "Boss"],
                    "planned_rest_count": 1,
                    "route_plan": {
                        "path_node_types": ["Monster", "Rest", "Boss"],
                        "rest_count": 1,
                        "reason_tags": ["stabilize_before_boss"],
                    },
                },
            },
        },
        _trajectory_step_payload(
            step_index=1,
            run_id="RUN-001",
            floor=6,
            screen_type="COMBAT",
            decision_source="policy",
            session_name="shadow-session",
            session_kind="eval",
            state_override={
                "run_id": "RUN-001",
                "screen": "COMBAT",
                "turn": 1,
                "in_combat": True,
            },
            action_descriptors_override={
                "screen": "COMBAT",
                "actions": [{"name": "play_card", "requires_index": True}],
            },
        ),
        _trajectory_step_payload(
            step_index=2,
            run_id="RUN-001",
            floor=6,
            screen_type="COMBAT",
            decision_source="policy",
            session_name="shadow-session",
            session_kind="eval",
            chosen_action_id="play_card|card=0|target=0",
        ),
        _combat_finished_payload(
            session_name="shadow-session",
            session_kind="eval",
            run_id="RUN-001",
            floor=6,
            combat_index=1,
            outcome="won",
            enemy_ids=["SLIME_SMALL"],
            damage_dealt=18,
            damage_taken=4,
            cumulative_reward=1.5,
            start_player_hp=52,
            end_player_hp=48,
            start_enemy_hp=18,
            end_enemy_hp=0,
        ),
        {
            "schema_version": 2,
            "record_type": "combat_started",
            "timestamp_utc": "2026-04-12T00:00:00+00:00",
            "session_name": "shadow-session",
            "session_kind": "eval",
            "instance_id": "inst-01",
            "step_index": 3,
            "run_id": "RUN-001",
            "floor": 7,
            "combat_index": 2,
            "enemy_ids": ["FUNGI_BEAST"],
            "state_summary": {
                "screen_type": "COMBAT",
                "run_id": "RUN-001",
                "observed_seed": "SEED-001",
                "run": {
                    "character_id": "IRONCLAD",
                    "seed": "SEED-001",
                    "floor": 7,
                    "act_index": 1,
                    "act_id": "THE_CITY",
                    "boss_encounter_id": "THE_CHAMP",
                },
                "combat": {"enemy_ids": ["FUNGI_BEAST"], "player_hp": 48, "enemy_hp": [26]},
            },
        },
        _trajectory_step_payload(
            step_index=4,
            run_id="RUN-001",
            floor=7,
            screen_type="COMBAT",
            decision_source="policy",
            session_name="shadow-session",
            session_kind="eval",
            chosen_action_id="end_turn",
        ),
        _combat_finished_payload(
            session_name="shadow-session",
            session_kind="eval",
            run_id="RUN-001",
            floor=7,
            combat_index=2,
            outcome="lost",
            enemy_ids=["FUNGI_BEAST"],
            damage_dealt=10,
            damage_taken=48,
            cumulative_reward=-1.0,
            start_player_hp=48,
            end_player_hp=0,
            start_enemy_hp=26,
            end_enemy_hp=16,
        ),
    ]
    with trajectory_path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in records:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    manifest_path = tmp_path / "shadow.toml"
    manifest_path.write_text(
        f"""
schema_version = 1
dataset_name = "shadow-encounters"
dataset_kind = "shadow_combat_encounters"

[[sources]]
path = "{(tmp_path / "data" / "trajectories").as_posix()}"
source_kind = "trajectory_log"
recursive = true

[split]
train_fraction = 0.5
validation_fraction = 0.0
test_fraction = 0.5
seed = 5
group_by = "combat_id"

[output]
export_csv = true
include_top_level_records = true
write_split_files = true
""".strip(),
        encoding="utf-8",
    )

    report = build_dataset_from_manifest(manifest_path, output_dir=tmp_path / "data" / "shadow-out")
    summary = load_dataset_summary(report.output_dir)
    records = load_shadow_combat_encounter_records(report.output_dir)

    assert report.record_count == 2
    assert report.split_counts == {"train": 1, "validation": 0, "test": 1}
    assert (report.output_dir / "encounters.csv").exists()
    assert summary["dataset_kind"] == "shadow_combat_encounters"
    assert summary["outcome_histogram"] == {"won": 1, "lost": 1}
    assert summary["encounter_family_histogram"] == {"SLIME_SMALL": 1, "FUNGI_BEAST": 1}
    assert summary["snapshot_coverage"]["full_snapshot_count"] == 2
    assert summary["action_id_histogram"]["end_turn"] == 2
    assert summary["action_id_histogram"]["play_card|card=0|target=0"] == 1
    assert summary["exports"]["encounters_table_csv"] == str((report.output_dir / "encounters.csv").resolve())
    assert summary["split"]["group_by"] == "combat_id"
    assert records[0].record_type == "shadow_combat_encounter"
    assert records[0].has_full_snapshot is True
    assert records[0].legal_action_count == 2
    assert records[0].state["screen"] == "COMBAT"
    assert records[0].strategic_context["boss_id"] == "THE_CHAMP"
    assert records[0].action_trace_count >= 1


def _combat_outcome_payload(*, run_id: str, outcome: str, floor: int) -> dict:
    player_hp = 62 if outcome == "won" else 14
    enemy_hp = 18 if outcome == "won" else 52
    return {
        "schema_version": 2,
        "record_type": "combat_finished",
        "timestamp_utc": "2026-04-12T00:00:00+00:00",
        "session_name": "session-a",
        "session_kind": "train",
        "instance_id": "inst-01",
        "run_id": run_id,
        "floor": floor,
        "combat_index": floor,
        "started_step_index": 0,
        "finished_step_index": 5,
        "outcome": outcome,
        "cumulative_reward": 1.0 if outcome == "won" else -0.5,
        "step_count": 5,
        "enemy_ids": ["SLIME_SMALL" if outcome == "won" else "FUNGI_BEAST"],
        "damage_dealt": 30 if outcome == "won" else 12,
        "damage_taken": 6 if outcome == "won" else 32,
        "start_summary": {
            "screen_type": "COMBAT",
            "run_id": run_id,
            "state_version": 8,
            "turn": 1,
            "in_combat": True,
            "available_action_count": 4,
            "build_warning_count": 0,
            "session_phase": "run",
            "control_scope": "local_player",
            "run": {
                "character_id": "IRONCLAD",
                "character_name": "Ironclad",
                "ascension": 0,
                "floor": floor,
                "act_index": 1,
                "act_id": "THE_CITY",
                "boss_encounter_id": "THE_CHAMP",
                "current_hp": player_hp,
                "max_hp": 80,
                "gold": 110 + floor,
                "max_energy": 3,
                "occupied_potions": 0,
            },
            "map": {
                "planned_node_types": ["Monster", "Shop", "Boss"],
                "planned_shop_count": 1,
                "planned_first_shop_distance": 1,
                "route_plan": {
                    "path_node_types": ["Monster", "Shop", "Boss"],
                    "shop_count": 1,
                    "first_shop_distance": 1,
                    "reason_tags": ["search_aoe_tools", "shop_access_before_boss"],
                },
            },
            "combat": {
                "player_hp": player_hp,
                "player_block": 0,
                "energy": 3,
                "stars": 0,
                "focus": 0,
                "enemy_ids": ["SLIME_SMALL" if outcome == "won" else "FUNGI_BEAST"],
                "enemy_hp": [enemy_hp],
                "hand_card_ids": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
                "playable_hand_count": 3,
            },
        },
        "end_summary": {
            "screen_type": "MAP" if outcome == "won" else "GAME_OVER",
            "run_id": run_id,
            "run": {
                "character_id": "IRONCLAD",
                "floor": floor,
                "current_hp": player_hp if outcome == "won" else 0,
                "max_hp": 80,
                "gold": 110 + floor,
                "max_energy": 3,
                "occupied_potions": 0,
            },
        },
        "reason": "combat_exited",
    }


def _trajectory_step_payload(
    *,
    step_index: int,
    run_id: str,
    floor: int,
    screen_type: str,
    decision_source: str,
    session_name: str = "session-c",
    session_kind: str = "train",
    chosen_action_id: str = "end_turn",
    state_override: dict | None = None,
    action_descriptors_override: dict | None = None,
) -> dict:
    payload = {
        "schema_version": 2,
        "record_type": "step",
        "timestamp_utc": "2026-04-12T00:00:00+00:00",
        "session_name": session_name,
        "session_kind": session_kind,
        "instance_id": "inst-01",
        "step_index": step_index,
        "run_id": run_id,
        "screen_type": screen_type,
        "floor": floor,
        "legal_action_count": 2,
        "legal_action_ids": ["end_turn", "play_card|card=0|target=0"],
        "build_warnings": [],
        "chosen_action_id": chosen_action_id,
        "chosen_action_label": "End Turn",
        "chosen_action_source": decision_source,
        "chosen_action": {"action_id": chosen_action_id},
        "policy_name": "test-policy",
        "algorithm": "heuristic",
        "decision_source": decision_source,
        "decision_stage": "combat" if screen_type == "COMBAT" else "meta",
        "decision_reason": "test",
        "decision_score": 0.0,
        "planner_name": "boss-conditioned-route-planner-v1",
        "planner_strategy": "boss_pathing",
        "decision_metadata": {
            "route_planner": {
                "planner_name": "boss-conditioned-route-planner-v1",
                "planner_strategy": "boss_pathing",
                "boss_encounter_id": "THE_CHAMP" if run_id == "RUN-001" else "HEXAGHOST",
                "selected": {
                    "score": 2.5 if run_id == "RUN-001" else 1.2,
                    "path": [
                        {"row": 4, "col": 0, "node_type": "Monster"},
                        {"row": 5, "col": 1, "node_type": "Shop" if run_id == "RUN-001" else "Rest"},
                    ],
                    "path_node_types": ["Monster", "Shop" if run_id == "RUN-001" else "Rest"],
                    "rest_count": 0 if run_id == "RUN-001" else 1,
                    "shop_count": 1 if run_id == "RUN-001" else 0,
                    "elite_count": 0,
                    "reason_tags": ["search_aoe_tools"] if run_id == "RUN-001" else ["early_rest_for_survival"],
                },
            }
        },
        "reward": 0.5,
        "reward_source": "synthetic",
        "terminated": False,
        "truncated": False,
        "info": {},
        "model_metrics": {},
        "state_summary": {
            "screen_type": screen_type,
            "run_id": run_id,
            "run": {
                "character_id": "IRONCLAD",
                "floor": floor,
                "act_index": 1,
                "act_id": "THE_CITY",
                "boss_encounter_id": "THE_CHAMP" if run_id == "RUN-001" else "HEXAGHOST",
            },
            "map": {
                "planned_node_types": ["Monster", "Shop" if run_id == "RUN-001" else "Rest"],
                "planned_shop_count": 1 if run_id == "RUN-001" else 0,
                "planned_rest_count": 0 if run_id == "RUN-001" else 1,
                "current_to_boss_distance": 3 if run_id == "RUN-001" else 2,
                "route_plan": {
                    "path_node_types": ["Monster", "Shop" if run_id == "RUN-001" else "Rest"],
                    "shop_count": 1 if run_id == "RUN-001" else 0,
                    "rest_count": 0 if run_id == "RUN-001" else 1,
                    "reason_tags": ["search_aoe_tools"] if run_id == "RUN-001" else ["early_rest_for_survival"],
                },
            },
        },
        "action_descriptors": {"screen": screen_type, "actions": []},
        "state": {"run_id": run_id, "screen": screen_type},
        "response": None,
    }
    if state_override is not None:
        payload["state"] = state_override
    if action_descriptors_override is not None:
        payload["action_descriptors"] = action_descriptors_override
    return payload


def _combat_finished_payload(
    *,
    session_name: str,
    session_kind: str,
    run_id: str,
    floor: int,
    combat_index: int,
    outcome: str,
    enemy_ids: list[str],
    damage_dealt: int,
    damage_taken: int,
    cumulative_reward: float,
    start_player_hp: int,
    end_player_hp: int,
    start_enemy_hp: int,
    end_enemy_hp: int,
) -> dict:
    return {
        "schema_version": 2,
        "record_type": "combat_finished",
        "timestamp_utc": "2026-04-12T00:00:00+00:00",
        "session_name": session_name,
        "session_kind": session_kind,
        "instance_id": "inst-01",
        "run_id": run_id,
        "observed_seed": "SEED-001",
        "floor": floor,
        "combat_index": combat_index,
        "started_step_index": combat_index - 1,
        "finished_step_index": combat_index + 1,
        "outcome": outcome,
        "cumulative_reward": cumulative_reward,
        "step_count": 2,
        "enemy_ids": enemy_ids,
        "damage_dealt": damage_dealt,
        "damage_taken": damage_taken,
        "start_summary": {
            "screen_type": "COMBAT",
            "run_id": run_id,
            "run": {"character_id": "IRONCLAD", "seed": "SEED-001", "floor": floor},
            "combat": {"player_hp": start_player_hp, "enemy_hp": [start_enemy_hp], "enemy_ids": enemy_ids},
        },
        "end_summary": {
            "screen_type": "MAP" if outcome == "won" else "GAME_OVER",
            "run_id": run_id,
            "run": {"character_id": "IRONCLAD", "floor": floor, "current_hp": end_player_hp},
            "combat": {"player_hp": end_player_hp, "enemy_hp": [end_enemy_hp], "enemy_ids": enemy_ids},
        },
        "reason": "combat_exited",
    }
