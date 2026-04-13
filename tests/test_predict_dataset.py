import json
from pathlib import Path

from sts2_rl.predict import extract_predictor_dataset, load_predictor_examples


def test_extract_predictor_dataset_from_combat_outcomes(tmp_path: Path) -> None:
    session_dir = tmp_path / "artifacts" / "session-a"
    session_dir.mkdir(parents=True)
    combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
    records = [
        _combat_outcome_payload(outcome="won", floor=3, player_hp=62, enemy_hp=18, enemy_ids=["SLIME_SMALL"]),
        _combat_outcome_payload(outcome="lost", floor=6, player_hp=14, enemy_hp=52, enemy_ids=["FUNGI_BEAST"]),
    ]
    with combat_outcomes_path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in records:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    report = extract_predictor_dataset(
        [tmp_path / "artifacts"],
        output_dir=tmp_path / "data" / "predict" / "smoke",
    )

    assert report.example_count == 2
    assert report.outcome_histogram == {"won": 1, "lost": 1}
    assert report.character_histogram == {"IRONCLAD": 2}
    assert report.act_histogram == {"THE_CITY": 2}
    assert report.boss_histogram == {"THE_CHAMP": 2}
    assert report.route_reason_tag_histogram == {"search_aoe_tools": 2, "shop_access_before_boss": 2}
    assert report.route_profile_histogram == {"search_aoe_tools+shop_access_before_boss": 2}
    assert report.strategic_coverage["route_plan_count"] == 2
    assert report.examples_path.exists()
    assert report.summary_path.exists()
    assert report.manifest_path.exists()
    assert report.split_counts == {"train": 2, "validation": 0, "test": 0}

    examples = load_predictor_examples(report.output_dir)
    assert len(examples) == 2
    assert examples[0].feature_map["run:hp_ratio"] > examples[1].feature_map["run:hp_ratio"]
    assert "enemy_id:SLIME_SMALL" in examples[0].feature_map
    assert "hand_card_id:STRIKE_IRONCLAD" in examples[0].feature_map
    assert examples[0].reward_weight == 1.0
    assert (report.output_dir / "train.examples.jsonl").exists()


def _combat_outcome_payload(
    *,
    outcome: str,
    floor: int,
    player_hp: int,
    enemy_hp: int,
    enemy_ids: list[str],
) -> dict:
    start_summary = {
        "screen_type": "COMBAT",
        "run_id": "RUN-001",
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
            "gold": 110,
            "max_energy": 3,
            "occupied_potions": 0,
        },
        "map": {
            "planned_node_types": ["Monster", "Shop", "Boss"],
            "planned_shop_count": 1,
            "planned_elite_count": 0,
            "planned_first_shop_distance": 1,
            "route_plan": {
                "path_node_types": ["Monster", "Shop", "Boss"],
                "shop_count": 1,
                "elite_count": 0,
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
            "enemy_ids": enemy_ids,
            "enemy_hp": [enemy_hp],
            "hand_card_ids": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
            "playable_hand_count": 3,
        },
    }
    end_summary = {
        "screen_type": "MAP" if outcome == "won" else "GAME_OVER",
        "run_id": "RUN-001",
        "run": {
            "character_id": "IRONCLAD",
            "floor": floor,
            "current_hp": player_hp if outcome == "won" else 0,
            "max_hp": 80,
            "gold": 110,
            "max_energy": 3,
            "occupied_potions": 0,
        },
    }
    if outcome == "lost":
        end_summary["game_over"] = {"is_victory": False, "floor": floor, "character_id": "IRONCLAD"}

    return {
        "schema_version": 2,
        "record_type": "combat_finished",
        "timestamp_utc": "2026-04-12T00:00:00+00:00",
        "session_name": "session-a",
        "session_kind": "train",
        "instance_id": "inst-01",
        "run_id": "RUN-001",
        "floor": floor,
        "combat_index": floor,
        "started_step_index": 0,
        "finished_step_index": 5,
        "outcome": outcome,
        "cumulative_reward": 1.0 if outcome == "won" else -0.5,
        "step_count": 5,
        "enemy_ids": enemy_ids,
        "damage_dealt": 30 if outcome == "won" else 12,
        "damage_taken": 6 if outcome == "won" else 32,
        "start_summary": start_summary,
        "end_summary": end_summary,
        "reason": "combat_exited",
    }
