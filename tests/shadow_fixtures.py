import json
from pathlib import Path

from sts2_rl.data import SHADOW_COMBAT_ENCOUNTERS_FILENAME, ShadowCombatEncounterRecord


def build_shadow_encounter_fixture(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "shadow-dataset"
    dataset_dir.mkdir(parents=True)
    records = [
        ShadowCombatEncounterRecord(
            encounter_id="RUN-001:6:1",
            session_name="shadow-session",
            session_kind="eval",
            instance_id="inst-02",
            run_id="RUN-001",
            observed_seed="SEED-001",
            character_id="IRONCLAD",
            floor=6,
            combat_index=1,
            start_step_index=0,
            finished_step_index=2,
            outcome="won",
            outcome_reason="combat_exited",
            enemy_ids=["SLIME_SMALL"],
            encounter_family="SLIME_SMALL",
            action_trace_ids=["play_card|card=0|target=0", "end_turn"],
            action_trace_count=2,
            unique_action_id_count=2,
            action_id_histogram={"play_card|card=0|target=0": 1, "end_turn": 1},
            cumulative_reward=1.5,
            step_count=2,
            damage_dealt=14,
            damage_taken=0,
            start_player_hp=52,
            end_player_hp=52,
            start_enemy_hp=10,
            end_enemy_hp=0,
            legal_action_count=2,
            legal_action_ids=["play_card|card=0|target=0", "end_turn"],
            state_summary={
                "screen_type": "COMBAT",
                "run_id": "RUN-001",
                "run": {
                    "character_id": "IRONCLAD",
                    "seed": "SEED-001",
                    "floor": 6,
                    "act_index": 1,
                    "act_id": "THE_CITY",
                    "boss_encounter_id": "THE_CHAMP",
                    "current_hp": 52,
                    "max_hp": 80,
                    "gold": 124,
                    "max_energy": 3,
                },
                "combat": {
                    "player_hp": 52,
                    "player_block": 0,
                    "energy": 2,
                    "enemy_ids": ["SLIME_SMALL"],
                    "enemy_hp": [10],
                    "hand_card_ids": ["BASH"],
                    "playable_hand_count": 1,
                },
            },
            end_state_summary={
                "screen_type": "MAP",
                "run_id": "RUN-001",
                "run": {"character_id": "IRONCLAD", "floor": 6, "current_hp": 52},
            },
            action_descriptors={
                "screen": "COMBAT",
                "actions": [
                    {"name": "play_card", "requires_index": True, "requires_target": True},
                    {"name": "end_turn", "requires_index": False, "requires_target": False},
                ],
            },
            state={
                "run_id": "RUN-001",
                "screen": "COMBAT",
                "in_combat": True,
                "turn": 1,
                "available_actions": ["play_card", "end_turn"],
                "run": {
                    "character_id": "IRONCLAD",
                    "character_name": "Ironclad",
                    "seed": "SEED-001",
                    "floor": 6,
                    "act_index": 1,
                    "act_id": "THE_CITY",
                    "current_hp": 52,
                    "max_hp": 80,
                    "gold": 124,
                    "max_energy": 3,
                },
                "combat": {
                    "player": {"current_hp": 52, "max_hp": 80, "block": 0, "energy": 2},
                    "hand": [
                        {
                            "index": 0,
                            "card_id": "BASH",
                            "name": "Bash",
                            "playable": True,
                            "requires_target": True,
                            "target_type": "Enemy",
                            "valid_target_indices": [0],
                            "energy_cost": 2,
                            "rules_text": "Deal 14 damage. Apply 2 Vulnerable.",
                            "resolved_rules_text": "Deal 14 damage. Apply 2 Vulnerable.",
                        }
                    ],
                    "enemies": [
                        {
                            "index": 0,
                            "enemy_id": "SLIME_SMALL",
                            "name": "Small Slime",
                            "current_hp": 10,
                            "max_hp": 10,
                            "block": 0,
                            "is_alive": True,
                            "is_hittable": True,
                            "intent": "ATTACK",
                        }
                    ],
                },
            },
            strategic_context={
                "boss_id": "THE_CHAMP",
                "boss_name": "The Champ",
                "route_reason_tags": ["search_aoe_tools"],
            },
            state_fingerprint="state-fp-001",
            action_space_fingerprint="action-fp-001",
            has_full_snapshot=True,
            has_terminal_outcome=True,
        ),
        ShadowCombatEncounterRecord(
            encounter_id="RUN-001:7:2",
            session_name="shadow-session",
            session_kind="eval",
            instance_id="inst-02",
            run_id="RUN-001",
            observed_seed="SEED-001",
            character_id="IRONCLAD",
            floor=7,
            combat_index=2,
            outcome="lost",
            outcome_reason="combat_exited",
            enemy_ids=["FUNGI_BEAST"],
            encounter_family="FUNGI_BEAST",
            action_trace_ids=["end_turn"],
            action_trace_count=1,
            unique_action_id_count=1,
            action_id_histogram={"end_turn": 1},
            legal_action_count=1,
            legal_action_ids=["end_turn"],
            state_summary={"screen_type": "COMBAT", "run_id": "RUN-001"},
            end_state_summary={"screen_type": "GAME_OVER", "run_id": "RUN-001"},
            action_descriptors={},
            state={},
            strategic_context={"boss_id": "THE_CHAMP"},
            has_full_snapshot=False,
            has_terminal_outcome=True,
        ),
    ]
    records_path = dataset_dir / SHADOW_COMBAT_ENCOUNTERS_FILENAME
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False))
            handle.write("\n")
    return dataset_dir
