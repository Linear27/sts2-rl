from __future__ import annotations

from sts2_rl.predict.features import extract_feature_map_from_summary


def test_extract_feature_map_from_summary_includes_strategic_run_and_map_features() -> None:
    feature_map = extract_feature_map_from_summary(
        {
            "run": {
                "character_id": "SILENT",
                "gold": 143,
                "current_hp": 49,
                "max_hp": 70,
                "max_energy": 3,
                "ascension": 5,
                "act_index": 1,
                "act_number": 2,
                "act_id": "THE_CITY",
                "has_second_boss": True,
                "boss_encounter_id": "THE_CHAMP",
                "second_boss_encounter_id": "THE_COLLECTOR",
                "occupied_potions": 1,
                "deck_size": 18,
                "strike_count": 4,
                "defend_count": 4,
                "curse_count": 0,
                "status_count": 0,
                "relic_count": 5,
            },
            "map": {
                "available_node_types": ["Monster", "Shop"],
                "available_node_count": 2,
                "travel_enabled": True,
                "traveling": False,
                "map_generation_count": 2,
                "rows": 5,
                "cols": 3,
                "graph_node_count": 8,
                "graph_edge_count": 7,
                "visited_node_count": 2,
                "available_graph_node_count": 2,
                "current_to_boss_distance": 3,
                "current_to_second_boss_distance": 3,
                "frontier_to_boss_min_distance": 2,
                "frontier_to_second_boss_min_distance": 2,
                "node_type_counts": {
                    "Start": 1,
                    "Monster": 2,
                    "Shop": 1,
                    "Elite": 1,
                    "Event": 1,
                    "Boss": 2,
                },
                "planned_node_types": ["Monster", "Shop", "Boss"],
                "planned_first_rest_distance": 2,
                "planned_first_shop_distance": 1,
                "planned_first_elite_distance": 3,
                "planned_rest_count": 1,
                "planned_shop_count": 1,
                "planned_elite_count": 0,
                "route_plan": {
                    "path": [
                        {"row": 4, "col": 0, "node_type": "Monster"},
                        {"row": 5, "col": 1, "node_type": "Shop"},
                        {"row": 6, "col": 1, "node_type": "Boss"},
                    ],
                    "path_node_types": ["Monster", "Shop", "Boss"],
                    "first_rest_distance": 2,
                    "first_shop_distance": 1,
                    "first_elite_distance": 3,
                    "rest_count": 1,
                    "shop_count": 1,
                    "elite_count": 0,
                    "event_count": 0,
                    "treasure_count": 0,
                    "monster_count": 1,
                    "elites_before_rest": 0,
                    "remaining_distance_to_boss": 1,
                    "score": 2.75,
                    "reason_tags": ["shop_access_before_boss", "search_aoe_tools"],
                },
            },
        }
    )

    assert feature_map["run:act_index"] == 1.0
    assert feature_map["run:act_number"] == 2.0
    assert feature_map["run:has_second_boss"] == 1.0
    assert feature_map["act_id:THE_CITY"] == 1.0
    assert feature_map["boss_encounter_id:THE_CHAMP"] == 1.0
    assert feature_map["second_boss_encounter_id:THE_COLLECTOR"] == 1.0
    assert feature_map["map:graph_node_count"] == 8.0
    assert feature_map["map:graph_edge_count"] == 7.0
    assert feature_map["map:current_to_boss_distance"] == 3.0
    assert feature_map["map:frontier_to_boss_min_distance"] == 2.0
    assert feature_map["map:route_plan_present"] == 1.0
    assert feature_map["map:planned_path_length"] == 3.0
    assert feature_map["map:planned_shop_count"] == 1.0
    assert feature_map["map:selected_route_score"] == 2.75
    assert feature_map["map_node_type:Monster"] == 1.0
    assert feature_map["map_graph_node_type:Monster"] == 2.0
    assert feature_map["map_graph_node_type:Boss"] == 2.0
    assert feature_map["map_planned_node_type:Shop"] == 1.0
    assert feature_map["route_reason_tag:search_aoe_tools"] == 1.0
