from __future__ import annotations

from collections import Counter
from typing import Any


def extract_feature_map_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    feature_map: dict[str, float] = {}

    feature_map["root:available_action_count"] = float(summary.get("available_action_count", 0) or 0)
    feature_map["root:build_warning_count"] = float(summary.get("build_warning_count", 0) or 0)
    feature_map["root:state_version"] = float(summary.get("state_version", 0) or 0)
    feature_map["root:turn"] = float(summary.get("turn", 0) or 0)
    feature_map["root:in_combat"] = 1.0 if summary.get("in_combat") else 0.0

    run = summary.get("run") or {}
    current_hp = float(run.get("current_hp", 0) or 0)
    max_hp = float(run.get("max_hp", 0) or 0)
    feature_map["run:floor"] = float(run.get("floor", 0) or 0)
    feature_map["run:current_hp"] = current_hp
    feature_map["run:max_hp"] = max_hp
    feature_map["run:hp_ratio"] = current_hp / max_hp if max_hp > 0 else 0.0
    feature_map["run:gold"] = float(run.get("gold", 0) or 0)
    feature_map["run:max_energy"] = float(run.get("max_energy", 0) or 0)
    feature_map["run:ascension"] = float(run.get("ascension", 0) or 0)
    feature_map["run:act_index"] = float(run.get("act_index", 0) or 0)
    feature_map["run:act_number"] = float(run.get("act_number", 0) or 0)
    feature_map["run:has_second_boss"] = 1.0 if run.get("has_second_boss") else 0.0
    feature_map["run:occupied_potions"] = float(run.get("occupied_potions", 0) or 0)
    feature_map["run:deck_size"] = float(run.get("deck_size", 0) or 0)
    feature_map["run:strike_count"] = float(run.get("strike_count", 0) or 0)
    feature_map["run:defend_count"] = float(run.get("defend_count", 0) or 0)
    feature_map["run:curse_count"] = float(run.get("curse_count", 0) or 0)
    feature_map["run:status_count"] = float(run.get("status_count", 0) or 0)
    feature_map["run:relic_count"] = float(run.get("relic_count", 0) or 0)

    character_id = run.get("character_id")
    if character_id:
        feature_map[f"character_id:{character_id}"] = 1.0
    if act_id := run.get("act_id"):
        feature_map[f"act_id:{act_id}"] = 1.0
    if boss_encounter_id := run.get("boss_encounter_id"):
        feature_map[f"boss_encounter_id:{boss_encounter_id}"] = 1.0
    if second_boss_encounter_id := run.get("second_boss_encounter_id"):
        feature_map[f"second_boss_encounter_id:{second_boss_encounter_id}"] = 1.0

    combat = summary.get("combat") or {}
    enemy_ids = [str(enemy_id) for enemy_id in combat.get("enemy_ids", [])]
    enemy_hp_values = [float(value) for value in combat.get("enemy_hp", [])]
    hand_card_ids = [str(card_id) for card_id in combat.get("hand_card_ids", [])]

    feature_map["combat:player_hp"] = float(combat.get("player_hp", 0) or 0)
    feature_map["combat:player_block"] = float(combat.get("player_block", 0) or 0)
    feature_map["combat:energy"] = float(combat.get("energy", 0) or 0)
    feature_map["combat:stars"] = float(combat.get("stars", 0) or 0)
    feature_map["combat:focus"] = float(combat.get("focus", 0) or 0)
    feature_map["combat:enemy_count"] = float(len(enemy_ids))
    feature_map["combat:total_enemy_hp"] = float(sum(enemy_hp_values))
    feature_map["combat:max_enemy_hp"] = float(max(enemy_hp_values)) if enemy_hp_values else 0.0
    feature_map["combat:min_enemy_hp"] = float(min(enemy_hp_values)) if enemy_hp_values else 0.0
    feature_map["combat:mean_enemy_hp"] = (
        float(sum(enemy_hp_values) / len(enemy_hp_values)) if enemy_hp_values else 0.0
    )
    feature_map["combat:hand_size"] = float(len(hand_card_ids))
    feature_map["combat:unique_hand_cards"] = float(len(set(hand_card_ids)))
    feature_map["combat:playable_hand_count"] = float(combat.get("playable_hand_count", 0) or 0)

    for enemy_id, count in Counter(enemy_ids).items():
        feature_map[f"enemy_id:{enemy_id}"] = float(count)
    for card_id, count in Counter(hand_card_ids).items():
        feature_map[f"hand_card_id:{card_id}"] = float(count)

    map_summary = summary.get("map") or {}
    node_types = [str(node_type) for node_type in map_summary.get("available_node_types", [])]
    planned_node_types = [str(node_type) for node_type in map_summary.get("planned_node_types", [])]
    route_plan = map_summary.get("route_plan") or {}
    route_reason_tags = [str(tag) for tag in route_plan.get("reason_tags", [])]
    feature_map["map:available_node_count"] = float(map_summary.get("available_node_count", len(node_types)) or 0)
    feature_map["map:travel_enabled"] = 1.0 if map_summary.get("travel_enabled") else 0.0
    feature_map["map:traveling"] = 1.0 if map_summary.get("traveling") else 0.0
    feature_map["map:map_generation_count"] = float(map_summary.get("map_generation_count", 0) or 0)
    feature_map["map:rows"] = float(map_summary.get("rows", 0) or 0)
    feature_map["map:cols"] = float(map_summary.get("cols", 0) or 0)
    feature_map["map:graph_node_count"] = float(map_summary.get("graph_node_count", 0) or 0)
    feature_map["map:graph_edge_count"] = float(map_summary.get("graph_edge_count", 0) or 0)
    feature_map["map:visited_node_count"] = float(map_summary.get("visited_node_count", 0) or 0)
    feature_map["map:available_graph_node_count"] = float(map_summary.get("available_graph_node_count", 0) or 0)
    feature_map["map:current_to_boss_distance"] = float(map_summary.get("current_to_boss_distance", 0) or 0)
    feature_map["map:current_to_second_boss_distance"] = float(
        map_summary.get("current_to_second_boss_distance", 0) or 0
    )
    feature_map["map:frontier_to_boss_min_distance"] = float(
        map_summary.get("frontier_to_boss_min_distance", 0) or 0
    )
    feature_map["map:frontier_to_second_boss_min_distance"] = float(
        map_summary.get("frontier_to_second_boss_min_distance", 0) or 0
    )
    feature_map["map:route_plan_present"] = 1.0 if route_plan else 0.0
    feature_map["map:planned_path_length"] = float(
        len(route_plan.get("path", [])) or len(planned_node_types) or 0
    )
    feature_map["map:planned_first_rest_distance"] = float(
        map_summary.get("planned_first_rest_distance", route_plan.get("first_rest_distance", 0)) or 0
    )
    feature_map["map:planned_first_shop_distance"] = float(
        map_summary.get("planned_first_shop_distance", route_plan.get("first_shop_distance", 0)) or 0
    )
    feature_map["map:planned_first_elite_distance"] = float(
        map_summary.get("planned_first_elite_distance", route_plan.get("first_elite_distance", 0)) or 0
    )
    feature_map["map:planned_rest_count"] = float(
        map_summary.get("planned_rest_count", route_plan.get("rest_count", 0)) or 0
    )
    feature_map["map:planned_shop_count"] = float(
        map_summary.get("planned_shop_count", route_plan.get("shop_count", 0)) or 0
    )
    feature_map["map:planned_elite_count"] = float(
        map_summary.get("planned_elite_count", route_plan.get("elite_count", 0)) or 0
    )
    feature_map["map:planned_event_count"] = float(route_plan.get("event_count", 0) or 0)
    feature_map["map:planned_treasure_count"] = float(route_plan.get("treasure_count", 0) or 0)
    feature_map["map:planned_monster_count"] = float(route_plan.get("monster_count", 0) or 0)
    feature_map["map:planned_elites_before_rest"] = float(route_plan.get("elites_before_rest", 0) or 0)
    feature_map["map:remaining_distance_to_boss"] = float(route_plan.get("remaining_distance_to_boss", 0) or 0)
    feature_map["map:selected_route_score"] = float(route_plan.get("score", 0) or 0)
    for node_type, count in Counter(node_types).items():
        feature_map[f"map_node_type:{node_type}"] = float(count)
    for node_type, count in (map_summary.get("node_type_counts") or {}).items():
        feature_map[f"map_graph_node_type:{node_type}"] = float(count)
    for node_type, count in Counter(planned_node_types).items():
        feature_map[f"map_planned_node_type:{node_type}"] = float(count)
    for tag, count in Counter(route_reason_tags).items():
        feature_map[f"route_reason_tag:{tag}"] = float(count)

    reward = summary.get("reward") or {}
    reward_types = [str(reward_type) for reward_type in reward.get("reward_types", [])]
    reward_card_ids = [str(card_id) for card_id in reward.get("card_option_ids", [])]
    feature_map["reward:pending_card_choice"] = 1.0 if reward.get("pending_card_choice") else 0.0
    feature_map["reward:reward_count"] = float(reward.get("reward_count", len(reward_types)) or 0)
    feature_map["reward:card_option_count"] = float(reward.get("card_option_count", len(reward_card_ids)) or 0)
    for reward_type, count in Counter(reward_types).items():
        feature_map[f"reward_type:{reward_type}"] = float(count)
    for card_id, count in Counter(reward_card_ids).items():
        feature_map[f"reward_card_id:{card_id}"] = float(count)

    selection = summary.get("selection") or {}
    selection_card_ids = [str(card_id) for card_id in selection.get("card_ids", [])]
    feature_map["selection:min_select"] = float(selection.get("min_select", 0) or 0)
    feature_map["selection:max_select"] = float(selection.get("max_select", 0) or 0)
    feature_map["selection:selected_count"] = float(selection.get("selected_count", 0) or 0)
    feature_map["selection:card_count"] = float(selection.get("card_count", len(selection_card_ids)) or 0)
    if selection_kind := selection.get("kind"):
        feature_map[f"selection_kind:{selection_kind}"] = 1.0
    for card_id, count in Counter(selection_card_ids).items():
        feature_map[f"selection_card_id:{card_id}"] = float(count)

    shop = summary.get("shop") or {}
    feature_map["shop:is_open"] = 1.0 if shop.get("is_open") else 0.0
    feature_map["shop:card_count"] = float(shop.get("card_count", 0) or 0)
    feature_map["shop:relic_count"] = float(shop.get("relic_count", 0) or 0)
    feature_map["shop:potion_count"] = float(shop.get("potion_count", 0) or 0)
    feature_map["shop:card_removal_available"] = 1.0 if shop.get("card_removal_available") else 0.0
    feature_map["shop:card_removal_price"] = float(shop.get("card_removal_price", 0) or 0)

    event = summary.get("event") or {}
    feature_map["event:option_count"] = float(event.get("option_count", 0) or 0)
    if event_id := event.get("event_id"):
        feature_map[f"event_id:{event_id}"] = 1.0

    rest = summary.get("rest") or {}
    rest_option_ids = [str(option_id) for option_id in rest.get("option_ids", [])]
    feature_map["rest:option_count"] = float(rest.get("option_count", len(rest_option_ids)) or 0)
    for option_id, count in Counter(rest_option_ids).items():
        feature_map[f"rest_option_id:{option_id}"] = float(count)

    return feature_map
