from pathlib import Path

from sts2_rl.collect.strategic_runtime import StrategicRuntimeConfig, reward_runtime_domain
from sts2_rl.collect.policy import SimplePolicy, build_policy_pack
from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    CharacterSelectOptionPayload,
    CharacterSelectPayload,
    ChestPayload,
    ChestRelicOptionPayload,
    CombatEnemyPayload,
    CombatHandCardPayload,
    CombatPayload,
    CombatPlayerPayload,
    EventOptionPayload,
    EventPayload,
    GameOverPayload,
    GameStatePayload,
    EncounterSummaryPayload,
    MapCoordPayload,
    MapGraphNodePayload,
    MapNodePayload,
    MapPayload,
    RestOptionPayload,
    RestPayload,
    RewardCardOptionPayload,
    RewardPayload,
    SelectionCardPayload,
    SelectionPayload,
    ShopCardPayload,
    ShopCardRemovalPayload,
    ShopPayload,
    ShopRelicPayload,
    RunPayload,
)
from sts2_rl.env.types import StepObservation
from sts2_rl.predict import CombatOutcomePredictor, PredictorHead, PredictorRuntimeConfig
from sts2_rl.train.strategic_pretrain import StrategicPretrainModel, _SparseScoringHead


def test_policy_prefers_open_character_select_from_main_menu() -> None:
    observation = _build_observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(
            screen="MAIN_MENU",
            actions=[ActionDescriptor(name="open_timeline"), ActionDescriptor(name="open_character_select")],
        ),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "open_character_select"
    assert decision.reason == "fallback_priority_order"


def test_policy_selects_preferred_character_then_embarks() -> None:
    state = GameStatePayload(
        screen="CHARACTER_SELECT",
        run_id="run_unknown",
        character_select=CharacterSelectPayload(
            selected_character_id=None,
            characters=[
                CharacterSelectOptionPayload(index=2, character_id="DEFECT", name="Defect"),
                CharacterSelectOptionPayload(index=0, character_id="IRONCLAD", name="Ironclad"),
            ],
        ),
    )
    observation = _build_observation(
        state,
        AvailableActionsPayload(
            screen="CHARACTER_SELECT",
            actions=[
                ActionDescriptor(name="select_character", requires_index=True),
                ActionDescriptor(name="embark"),
            ],
        ),
    )

    select_decision = SimplePolicy().choose(observation)

    assert select_decision.action is not None
    assert select_decision.action.action == "select_character"
    assert select_decision.action.request.option_index == 0

    selected_state = state.model_copy(
        update={
            "character_select": state.character_select.model_copy(
                update={
                    "selected_character_id": "IRONCLAD",
                    "characters": [
                        state.character_select.characters[0],
                        state.character_select.characters[1].model_copy(update={"is_selected": True}),
                    ],
                }
            )
        }
    )
    selected_observation = _build_observation(
        selected_state,
        AvailableActionsPayload(screen="CHARACTER_SELECT", actions=[ActionDescriptor(name="embark")]),
    )

    embark_decision = SimplePolicy().choose(selected_observation)

    assert embark_decision.action is not None
    assert embark_decision.action.action == "embark"
    assert embark_decision.reason == "embark_selected_character"


def test_policy_routes_to_rest_when_hp_is_low() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="MAP",
            run_id="run-1",
            run=RunPayload(character_id="IRONCLAD", floor=6, current_hp=20, max_hp=80, gold=110, max_energy=3),
            map=MapPayload(
                available_nodes=[
                    MapNodePayload(index=0, row=6, col=0, node_type="elite"),
                    MapNodePayload(index=1, row=6, col=1, node_type="rest"),
                    MapNodePayload(index=2, row=6, col=2, node_type="monster"),
                ],
                is_travel_enabled=True,
            ),
        ),
        AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="choose_map_node", requires_index=True)]),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "choose_map_node"
    assert decision.action.request.option_index == 1
    assert decision.reason == "route_to_rest_for_survival"


def test_policy_routes_to_elite_when_hp_is_safe() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="MAP",
            run_id="run-2",
            run=RunPayload(character_id="IRONCLAD", floor=6, current_hp=78, max_hp=80, gold=90, max_energy=3),
            map=MapPayload(
                available_nodes=[
                    MapNodePayload(index=0, row=6, col=0, node_type="elite"),
                    MapNodePayload(index=1, row=6, col=1, node_type="rest"),
                    MapNodePayload(index=2, row=6, col=2, node_type="monster"),
                ],
                is_travel_enabled=True,
            ),
        ),
        AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="choose_map_node", requires_index=True)]),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.request.option_index == 0
    assert decision.reason == "route_to_elite_with_safe_hp"


def test_strategic_route_planner_beats_legacy_single_node_heuristic() -> None:
    map_payload = _build_boss_route_map()

    strategic = _build_observation(
        GameStatePayload(
            screen="MAP",
            run_id="run-route-collector",
            run=_run_payload(
                current_hp=72,
                max_hp=80,
                floor=10,
                gold=190,
                boss_encounter_id="THE_COLLECTOR",
                boss_encounter_name="The Collector",
            ),
            map=map_payload,
            agent_view=_agent_view_with_deck(
                [
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Bash [2]: Deal 8 damage. Apply 2 Vulnerable.",
                ]
            ),
        ),
        AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="choose_map_node", requires_index=True)]),
    )
    legacy = strategic.model_copy(deep=True)
    strategic_decision = build_policy_pack("baseline").choose(strategic)
    legacy_decision = build_policy_pack("legacy").choose(legacy)

    assert strategic_decision.action is not None
    assert strategic_decision.action.request.option_index == 1
    assert strategic_decision.reason == "route_plan_shop_for_boss_tools"
    assert strategic_decision.planner_name == "boss-conditioned-route-planner-v1"
    assert strategic_decision.trace_metadata["route_planner"]["selected"]["path_node_types"] == ["Monster", "Shop", "Boss"]

    assert legacy_decision.action is not None
    assert legacy_decision.action.request.option_index == 0
    assert legacy_decision.reason == "route_to_elite_with_safe_hp"


def test_strategic_route_planner_changes_choice_for_different_bosses() -> None:
    map_payload = _build_boss_route_map()
    collector_observation = _build_observation(
        GameStatePayload(
            screen="MAP",
            run_id="run-route-collector-2",
            run=_run_payload(
                current_hp=72,
                max_hp=80,
                floor=10,
                gold=190,
                boss_encounter_id="THE_COLLECTOR",
                boss_encounter_name="The Collector",
            ),
            map=map_payload,
            agent_view=_agent_view_with_deck(
                [
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Bash [2]: Deal 8 damage. Apply 2 Vulnerable.",
                ]
            ),
        ),
        AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="choose_map_node", requires_index=True)]),
    )
    champ_observation = collector_observation.model_copy(
        update={
            "run_id": "run-route-champ",
            "state": collector_observation.state.model_copy(
                update={
                    "run": collector_observation.state.run.model_copy(
                        update={
                            "boss_encounter": EncounterSummaryPayload(
                                encounter_id="THE_CHAMP",
                                name="The Champ",
                                room_type="Boss",
                            )
                        }
                    )
                }
            ),
        },
        deep=True,
    )

    collector_decision = build_policy_pack("baseline").choose(collector_observation)
    champ_decision = build_policy_pack("baseline").choose(champ_observation)

    assert collector_decision.action is not None
    assert collector_decision.action.request.option_index == 1
    assert champ_decision.action is not None
    assert champ_decision.action.request.option_index == 0


def test_reward_and_shop_scores_shift_with_boss_context() -> None:
    collector_policy = build_policy_pack("baseline")
    champ_policy = build_policy_pack("baseline")

    reward_observation = _build_observation(
        GameStatePayload(
            screen="REWARD",
            run_id="run-reward-collector",
            run=_run_payload(
                current_hp=60,
                max_hp=80,
                floor=12,
                gold=140,
                boss_encounter_id="THE_COLLECTOR",
                boss_encounter_name="The Collector",
            ),
            reward=RewardPayload(
                pending_card_choice=True,
                card_options=[
                    RewardCardOptionPayload(
                        index=0,
                        card_id="CLEAVE",
                        name="Cleave",
                        resolved_rules_text="Deal 5 damage to all enemies.",
                    ),
                    RewardCardOptionPayload(
                        index=1,
                        card_id="INFLAME",
                        name="Inflame",
                        upgraded=True,
                        resolved_rules_text="Gain 3 Strength.",
                    ),
                ],
            ),
        ),
        AvailableActionsPayload(screen="REWARD", actions=[ActionDescriptor(name="choose_reward_card", requires_index=True)]),
    )
    reward_champ = reward_observation.model_copy(
        update={
            "run_id": "run-reward-champ",
            "state": reward_observation.state.model_copy(
                update={
                    "run": reward_observation.state.run.model_copy(
                        update={
                            "boss_encounter": EncounterSummaryPayload(
                                encounter_id="THE_CHAMP",
                                name="The Champ",
                                room_type="Boss",
                            )
                        }
                    )
                }
            ),
        },
        deep=True,
    )

    collector_reward_decision = collector_policy.choose(reward_observation)
    champ_reward_decision = champ_policy.choose(reward_champ)

    assert collector_reward_decision.action is not None
    assert collector_reward_decision.action.request.option_index == 0
    assert champ_reward_decision.action is not None
    assert champ_reward_decision.action.request.option_index == 1

    shop_observation = _build_observation(
        GameStatePayload(
            screen="SHOP",
            run_id="run-shop-collector",
            run=_run_payload(
                current_hp=62,
                max_hp=80,
                floor=11,
                gold=210,
                boss_encounter_id="THE_COLLECTOR",
                boss_encounter_name="The Collector",
            ),
            shop=ShopPayload(
                is_open=True,
                cards=[
                    ShopCardPayload(
                        index=0,
                        card_id="AOE_5_DAMAGE_CLEAVE",
                        name="AOE Cleave 5 Damage",
                        price=95,
                        is_stocked=True,
                        enough_gold=True,
                    ),
                    ShopCardPayload(
                        index=1,
                        card_id="INFLAME_3_STRENGTH",
                        name="Inflame 3 Strength",
                        price=95,
                        is_stocked=True,
                        enough_gold=True,
                    )
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="SHOP",
            actions=[ActionDescriptor(name="buy_card", requires_index=True)],
        ),
    )
    shop_champ = shop_observation.model_copy(
        update={
            "run_id": "run-shop-champ",
            "state": shop_observation.state.model_copy(
                update={
                    "run": shop_observation.state.run.model_copy(
                        update={
                            "boss_encounter": EncounterSummaryPayload(
                                encounter_id="THE_CHAMP",
                                name="The Champ",
                                room_type="Boss",
                            )
                        }
                    )
                }
            ),
        },
        deep=True,
    )

    collector_shop_decision = collector_policy.choose(shop_observation)
    champ_shop_decision = champ_policy.choose(shop_champ)

    assert collector_shop_decision.action is not None
    assert collector_shop_decision.action.action == "buy_card"
    assert collector_shop_decision.action.request.option_index == 0
    assert champ_shop_decision.action is not None
    assert champ_shop_decision.action.action == "buy_card"
    assert champ_shop_decision.action.request.option_index == 1


def test_rest_decision_can_shift_for_sustain_boss() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="REST",
            run_id="run-rest-hex",
            run=_run_payload(
                current_hp=44,
                max_hp=80,
                floor=12,
                gold=150,
                boss_encounter_id="HEXAGHOST",
                boss_encounter_name="Hexaghost",
            ),
            rest=RestPayload(
                options=[
                    RestOptionPayload(index=0, option_id="rest", title="Rest", is_enabled=True),
                    RestOptionPayload(index=1, option_id="smith", title="Smith", is_enabled=True),
                ]
            ),
        ),
        AvailableActionsPayload(screen="REST", actions=[ActionDescriptor(name="choose_rest_option", requires_index=True)]),
    )

    strategic_decision = build_policy_pack("baseline").choose(observation)
    legacy_decision = build_policy_pack("legacy").choose(observation)

    assert strategic_decision.action is not None
    assert strategic_decision.action.request.option_index == 0
    assert strategic_decision.reason == "rest_for_boss_sustain"
    assert legacy_decision.action is not None
    assert legacy_decision.action.request.option_index == 1


def test_policy_prefers_reward_card_on_card_selection_screen() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="CARD_SELECTION",
            run_id="run-3",
            run=_run_payload(current_hp=70, max_hp=80, floor=2, gold=105),
            selection=SelectionPayload(
                kind="deck_card_select",
                selection_family="deck",
                semantic_mode="pick",
                source_type="reward",
                prompt="Choose a card",
                required_count=1,
                cards=[
                    SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                    SelectionCardPayload(index=1, card_id="BASH", name="Bash"),
                ],
            ),
            reward=RewardPayload(
                pending_card_choice=True,
                card_options=[
                    RewardCardOptionPayload(
                        index=0,
                        card_id="STRIKE_IRONCLAD",
                        name="Strike",
                        resolved_rules_text="Deal 6 damage.",
                    ),
                    RewardCardOptionPayload(
                        index=1,
                        card_id="BASH",
                        name="Bash",
                        resolved_rules_text="Deal 8 damage. Apply 2 Vulnerable.",
                    ),
                ],
            ),
            agent_view=_agent_view_with_deck(
                [
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Bash [2]: Deal 8 damage. Apply 2 Vulnerable.",
                ]
            ),
        ),
        AvailableActionsPayload(
            screen="CARD_SELECTION",
            actions=[
                ActionDescriptor(name="choose_reward_card", requires_index=True),
                ActionDescriptor(name="select_deck_card", requires_index=True),
                ActionDescriptor(name="skip_reward_cards"),
            ],
        ),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "choose_reward_card"
    assert decision.action.request.option_index == 1
    assert decision.reason == "take_reward_card"


def test_policy_skips_low_value_reward_cards_for_large_deck() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="REWARD",
            run_id="run-4",
            run=_run_payload(current_hp=55, max_hp=80, floor=12, gold=160),
            reward=RewardPayload(
                pending_card_choice=True,
                card_options=[
                    RewardCardOptionPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                    RewardCardOptionPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
                ],
            ),
            agent_view=_agent_view_with_deck(
                [
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Bash [2]: Deal 8 damage. Apply 2 Vulnerable.",
                    "Shrug It Off [1]: Gain 8 Block. Draw 1.",
                    "Pommel Strike [1]: Deal 9 damage. Draw 1.",
                    "Battle Trance [0]: Draw 3 cards.",
                    "Shockwave [2]: Apply Weak and Vulnerable to all enemies.",
                    "Flame Barrier [2]: Gain Block.",
                    "Armaments [1]: Gain Block. Upgrade a card.",
                    "Bludgeon [3]: Deal big damage.",
                    "Second Wind [1]: Exhaust non-Attacks. Gain Block.",
                    "Inflame [1]: Gain Strength.",
                ]
            ),
        ),
        AvailableActionsPayload(
            screen="REWARD",
            actions=[ActionDescriptor(name="choose_reward_card", requires_index=True), ActionDescriptor(name="skip_reward_cards")],
        ),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "skip_reward_cards"
    assert decision.reason == "skip_low_value_reward_card"


def test_policy_remove_selection_prefers_strike_cleanup() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="SELECTION",
            run_id="run-5",
            run=_run_payload(current_hp=66, max_hp=80, floor=8, gold=120),
            selection=SelectionPayload(
                kind="deck_card_select",
                selection_family="deck",
                semantic_mode="remove",
                source_type="event",
                prompt="Remove a card from your deck",
                required_count=1,
                cards=[
                    SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                    SelectionCardPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
                    SelectionCardPayload(index=2, card_id="BASH", name="Bash"),
                ],
            ),
            agent_view=_agent_view_with_deck(
                [
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Bash [2]: Deal 8 damage. Apply 2 Vulnerable.",
                ]
            ),
        ),
        AvailableActionsPayload(screen="SELECTION", actions=[ActionDescriptor(name="select_deck_card", requires_index=True)]),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.request.option_index == 0
    assert decision.reason == "remove_worst_card"


def test_policy_selection_requires_explicit_semantic_mode() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="SELECTION",
            run_id="run-selection-missing-semantic",
            run=_run_payload(current_hp=66, max_hp=80, floor=8, gold=120),
            selection=SelectionPayload(
                kind="deck_card_select",
                selection_family="deck",
                prompt="Remove a card from your deck",
                required_count=1,
                cards=[
                    SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                    SelectionCardPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
                ],
            ),
        ),
        AvailableActionsPayload(screen="SELECTION", actions=[ActionDescriptor(name="select_deck_card", requires_index=True)]),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is None
    assert decision.reason == "missing_selection_semantic_mode"


def test_policy_selection_uses_required_count_before_confirming() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="CARD_SELECTION",
            run_id="run-5b",
            run=_run_payload(current_hp=66, max_hp=80, floor=1, gold=99),
            selection=SelectionPayload(
                kind="deck_card_select",
                selection_family="deck",
                semantic_mode="remove",
                source_type="event",
                prompt="选择[blue]2[/blue]张牌来[gold]移除[/gold]。",
                min_select=0,
                max_select=0,
                required_count=2,
                remaining_count=2,
                selected_count=0,
                can_confirm=True,
                cards=[
                    SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                    SelectionCardPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
                ],
            ),
            agent_view=_agent_view_with_deck(
                [
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Bash [2]: Deal 8 damage. Apply 2 Vulnerable.",
                ]
            ),
        ),
        AvailableActionsPayload(
            screen="CARD_SELECTION",
            actions=[ActionDescriptor(name="select_deck_card", requires_index=True), ActionDescriptor(name="confirm_selection")],
        ),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "select_deck_card"
    assert decision.action.request.option_index == 0
    assert decision.reason == "remove_worst_card"


def test_policy_selection_confirms_after_required_count_is_satisfied() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="CARD_SELECTION",
            run_id="run-5c",
            run=_run_payload(current_hp=66, max_hp=80, floor=1, gold=99),
            selection=SelectionPayload(
                kind="deck_card_select",
                selection_family="deck",
                semantic_mode="remove",
                source_type="event",
                prompt="选择[blue]2[/blue]张牌来[gold]移除[/gold]。",
                min_select=0,
                max_select=0,
                required_count=2,
                remaining_count=0,
                selected_count=2,
                can_confirm=True,
                cards=[
                    SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                    SelectionCardPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
                ],
            ),
            agent_view=_agent_view_with_deck(
                [
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Bash [2]: Deal 8 damage. Apply 2 Vulnerable.",
                ]
            ),
        ),
        AvailableActionsPayload(
            screen="CARD_SELECTION",
            actions=[ActionDescriptor(name="select_deck_card", requires_index=True), ActionDescriptor(name="confirm_selection")],
        ),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "confirm_selection"
    assert decision.reason == "confirm_required_selection"


def test_policy_selection_transaction_advances_multi_remove_bundle_across_steps() -> None:
    policy = SimplePolicy()

    first_observation = _selection_observation(
        run_id="run-selection-transaction",
        mode="remove",
        required_count=2,
        remaining_count=2,
        selected_count=0,
        cards=[
            SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
            SelectionCardPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
            SelectionCardPayload(index=2, card_id="BASH", name="Bash"),
        ],
    )

    first_decision = policy.choose(first_observation)

    assert first_decision.action is not None
    assert first_decision.action.action == "select_deck_card"
    assert first_decision.action.request.option_index == 0
    assert first_decision.trace_metadata["selection_transaction"]["phase"] == "selecting"
    assert first_decision.trace_metadata["selection_transaction"]["planned_option_indices"] == [0, 1]
    assert first_decision.trace_metadata["selection_transaction"]["next_option_index"] == 0

    second_observation = _selection_observation(
        run_id="run-selection-transaction",
        mode="remove",
        required_count=2,
        remaining_count=1,
        selected_count=1,
        cards=[
            SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
            SelectionCardPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
            SelectionCardPayload(index=2, card_id="BASH", name="Bash"),
        ],
    )

    second_decision = policy.choose(second_observation)

    assert second_decision.action is not None
    assert second_decision.action.action == "select_deck_card"
    assert second_decision.action.request.option_index == 1
    assert second_decision.trace_metadata["selection_transaction"]["completed_option_indices"] == [0]
    assert second_decision.trace_metadata["selection_transaction"]["next_option_index"] == 1

    third_observation = _selection_observation(
        run_id="run-selection-transaction",
        mode="remove",
        required_count=2,
        remaining_count=0,
        selected_count=2,
        cards=[
            SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
            SelectionCardPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
            SelectionCardPayload(index=2, card_id="BASH", name="Bash"),
        ],
    )

    third_decision = policy.choose(third_observation)

    assert third_decision.action is not None
    assert third_decision.action.action == "confirm_selection"
    assert third_decision.trace_metadata["selection_transaction"]["phase"] == "confirming"
    assert third_decision.trace_metadata["selection_transaction"]["completed_option_indices"] == [0, 1]


def test_policy_selection_transaction_replans_when_remaining_bundle_diverges() -> None:
    policy = SimplePolicy()

    first_observation = _selection_observation(
        run_id="run-selection-replan",
        mode="remove",
        required_count=2,
        remaining_count=2,
        selected_count=0,
        cards=[
            SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
            SelectionCardPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
            SelectionCardPayload(index=2, card_id="BASH", name="Bash"),
        ],
    )

    first_decision = policy.choose(first_observation)

    assert first_decision.action is not None
    assert first_decision.action.request.option_index == 0

    diverged_observation = _selection_observation(
        run_id="run-selection-replan",
        mode="remove",
        required_count=2,
        remaining_count=1,
        selected_count=1,
        cards=[
            SelectionCardPayload(index=2, card_id="BASH", name="Bash"),
            SelectionCardPayload(index=3, card_id="CURSE_DOUBT", name="Doubt"),
        ],
    )

    second_decision = policy.choose(diverged_observation)

    assert second_decision.action is not None
    assert second_decision.action.action == "select_deck_card"
    assert second_decision.action.request.option_index == 3
    assert second_decision.trace_metadata["selection_transaction"]["planned_option_indices"] == [0, 3]
    assert second_decision.trace_metadata["selection_transaction"]["completed_option_indices"] == [0]
    assert second_decision.trace_metadata["selection_transaction"]["recovery_count"] >= 1


def test_policy_selection_treats_transform_as_distinct_transaction_mode() -> None:
    observation = _selection_observation(
        run_id="run-selection-transform",
        mode="transform",
        required_count=1,
        remaining_count=1,
        selected_count=0,
        cards=[
            SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
            SelectionCardPayload(index=1, card_id="BASH", name="Bash"),
        ],
        can_confirm=False,
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "select_deck_card"
    assert decision.action.request.option_index == 0
    assert decision.reason == "transform_worst_card"
    assert decision.trace_metadata["selection_transaction"]["semantic_mode"] == "transform"


def test_policy_shop_prefers_card_removal_for_cluttered_deck() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="SHOP",
            run_id="run-6",
            run=_run_payload(current_hp=62, max_hp=80, floor=7, gold=180),
            shop=ShopPayload(
                is_open=True,
                cards=[
                    ShopCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike", price=45, is_stocked=True, enough_gold=True),
                ],
                relics=[
                    ShopRelicPayload(index=1, relic_id="ORNAMENT", name="Ornament", rarity="common", price=150, is_stocked=True, enough_gold=True),
                ],
                card_removal=ShopCardRemovalPayload(price=75, available=True, enough_gold=True),
            ),
            agent_view=_agent_view_with_deck(
                [
                    "Curse: Bad thing.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Bash [2]: Deal 8 damage. Apply 2 Vulnerable.",
                ]
            ),
        ),
        AvailableActionsPayload(
            screen="SHOP",
            actions=[
                ActionDescriptor(name="remove_card_at_shop"),
                ActionDescriptor(name="buy_card", requires_index=True),
                ActionDescriptor(name="buy_relic", requires_index=True),
                ActionDescriptor(name="close_shop_inventory"),
            ],
        ),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "remove_card_at_shop"
    assert decision.reason == "buy_card_removal"


def test_policy_shop_closed_proceeds_when_no_worthy_purchase_exists() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="SHOP",
            run_id="run-6b",
            run=_run_payload(current_hp=62, max_hp=80, floor=7, gold=63),
            shop=ShopPayload(
                is_open=False,
                can_open=True,
                can_close=False,
                cards=[
                    ShopCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike", price=45, is_stocked=True, enough_gold=True),
                ],
                relics=[],
                potions=[],
                card_removal=ShopCardRemovalPayload(price=0, available=False, enough_gold=False),
            ),
            agent_view=_agent_view_with_deck(
                [
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Bash [2]: Deal 8 damage. Apply 2 Vulnerable.",
                ]
            ),
        ),
        AvailableActionsPayload(
            screen="SHOP",
            actions=[ActionDescriptor(name="proceed"), ActionDescriptor(name="open_shop_inventory")],
        ),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "proceed"
    assert decision.reason == "leave_shop_preserve_gold"


def test_policy_shop_closed_opens_inventory_when_card_removal_is_worthy() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="SHOP",
            run_id="run-6c",
            run=_run_payload(current_hp=62, max_hp=80, floor=7, gold=180),
            shop=ShopPayload(
                is_open=False,
                can_open=True,
                can_close=False,
                cards=[],
                relics=[],
                potions=[],
                card_removal=ShopCardRemovalPayload(price=75, available=True, enough_gold=True),
            ),
            agent_view=_agent_view_with_deck(
                [
                    "Curse: Bad thing.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Bash [2]: Deal 8 damage. Apply 2 Vulnerable.",
                ]
            ),
        ),
        AvailableActionsPayload(
            screen="SHOP",
            actions=[ActionDescriptor(name="proceed"), ActionDescriptor(name="open_shop_inventory")],
        ),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "open_shop_inventory"
    assert decision.reason == "open_shop_inventory"


def test_policy_rest_chooses_heal_then_upgrade() -> None:
    low_hp_observation = _build_observation(
        GameStatePayload(
            screen="REST",
            run_id="run-7",
            run=_run_payload(current_hp=18, max_hp=80, floor=10, gold=150),
            rest=RestPayload(
                options=[
                    RestOptionPayload(index=0, option_id="rest", title="Rest", is_enabled=True),
                    RestOptionPayload(index=1, option_id="smith", title="Smith", is_enabled=True),
                ]
            ),
        ),
        AvailableActionsPayload(screen="REST", actions=[ActionDescriptor(name="choose_rest_option", requires_index=True)]),
    )
    high_hp_observation = _build_observation(
        GameStatePayload(
            screen="REST",
            run_id="run-8",
            run=_run_payload(current_hp=72, max_hp=80, floor=10, gold=150),
            rest=RestPayload(
                options=[
                    RestOptionPayload(index=0, option_id="rest", title="Rest", is_enabled=True),
                    RestOptionPayload(index=1, option_id="smith", title="Smith", is_enabled=True),
                ]
            ),
        ),
        AvailableActionsPayload(screen="REST", actions=[ActionDescriptor(name="choose_rest_option", requires_index=True)]),
    )

    low_hp_decision = SimplePolicy().choose(low_hp_observation)
    high_hp_decision = SimplePolicy().choose(high_hp_observation)

    assert low_hp_decision.action is not None
    assert low_hp_decision.action.request.option_index == 0
    assert low_hp_decision.reason == "rest_for_survival"

    assert high_hp_decision.action is not None
    assert high_hp_decision.action.request.option_index == 1
    assert high_hp_decision.reason == "upgrade_at_rest"


def test_policy_rest_uses_canonical_option_id_for_special_rest_semantics() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="REST",
            run_id="run-rest-canonical-upgrade",
            run=_run_payload(current_hp=72, max_hp=80, floor=10, gold=150),
            rest=RestPayload(
                options=[
                    RestOptionPayload(index=0, option_id="rest", title="Recover Resolve", is_enabled=True),
                    RestOptionPayload(index=1, option_id="upgrade", title="Forge Memory", is_enabled=True),
                ]
            ),
        ),
        AvailableActionsPayload(screen="REST", actions=[ActionDescriptor(name="choose_rest_option", requires_index=True)]),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.request.option_index == 1
    assert decision.reason == "upgrade_at_rest"


def test_reward_runtime_domain_uses_explicit_follow_up_source_type() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="REWARD",
            run_id="run-event-followup-reward",
            run=_run_payload(current_hp=70, max_hp=80, floor=8, gold=120),
            reward=RewardPayload(
                pending_card_choice=True,
                source_type="event",
                source_room_type="Event",
                source_action="choose_event_option",
                source_event_id="EVENT.EPIC_QUEST",
                source_event_option_index=1,
                source_event_option_text_key="event.quest.solo",
                source_event_option_title="Solo Quest",
                card_options=[
                    RewardCardOptionPayload(index=0, card_id="BASH", name="Bash"),
                    RewardCardOptionPayload(index=1, card_id="POMMEL_STRIKE", name="Pommel Strike"),
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="REWARD",
            actions=[ActionDescriptor(name="choose_reward_card", requires_index=True), ActionDescriptor(name="skip_reward_cards")],
        ),
    )

    assert reward_runtime_domain(observation) == ("event", "event")


def test_policy_advances_game_over_then_returns_to_menu() -> None:
    continue_observation = _build_observation(
        GameStatePayload(
            screen="GAME_OVER",
            run_id="run-9",
            game_over=GameOverPayload(
                is_victory=False,
                floor=3,
                character_id="IRONCLAD",
                can_continue=True,
                can_return_to_main_menu=False,
                showing_summary=False,
            ),
        ),
        AvailableActionsPayload(screen="GAME_OVER", actions=[ActionDescriptor(name="continue_run"), ActionDescriptor(name="return_to_main_menu")]),
    )
    summary_observation = _build_observation(
        GameStatePayload(
            screen="GAME_OVER",
            run_id="run-9",
            game_over=GameOverPayload(
                is_victory=False,
                floor=3,
                character_id="IRONCLAD",
                can_continue=False,
                can_return_to_main_menu=True,
                showing_summary=True,
            ),
        ),
        AvailableActionsPayload(screen="GAME_OVER", actions=[ActionDescriptor(name="continue_run"), ActionDescriptor(name="return_to_main_menu")]),
    )

    continue_decision = SimplePolicy().choose(continue_observation)
    return_decision = SimplePolicy().choose(summary_observation)

    assert continue_decision.action is not None
    assert continue_decision.action.action == "continue_run"
    assert continue_decision.reason == "continue_game_over_flow"

    assert return_decision.action is not None
    assert return_decision.action.action == "return_to_main_menu"
    assert return_decision.reason == "return_to_main_menu_after_game_over"


def test_policy_prefers_event_relic_over_proceed() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="EVENT",
            run_id="run-10",
            run=_run_payload(current_hp=70, max_hp=80, floor=5, gold=90),
            event=EventPayload(
                event_id="GoldenIdol",
                title="Golden Idol",
                options=[
                    EventOptionPayload(index=0, title="Take relic", description="Gain a Relic.", has_relic_preview=True),
                    EventOptionPayload(index=1, title="Leave", description="Proceed onward.", is_proceed=True),
                ],
            ),
        ),
        AvailableActionsPayload(screen="EVENT", actions=[ActionDescriptor(name="choose_event_option", requires_index=True)]),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.request.option_index == 0
    assert decision.reason == "take_event_relic_or_power"
    assert decision.policy_handler == "event-handler"


def test_policy_avoids_multi_remove_event_flow_when_safe_alternative_exists() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="EVENT",
            run_id="run-10b",
            run=_run_payload(current_hp=80, max_hp=80, floor=1, gold=99),
            event=EventPayload(
                event_id="NEOW",
                title="Neow",
                options=[
                    EventOptionPayload(
                        index=0,
                        title="Rare Card",
                        description="Obtain a random rare card.",
                        has_relic_preview=True,
                    ),
                    EventOptionPayload(
                        index=1,
                        title="Loose Shears",
                        description="Remove 2 cards from your deck, then lose 16 HP.",
                        has_relic_preview=True,
                    ),
                ],
            ),
        ),
        AvailableActionsPayload(screen="EVENT", actions=[ActionDescriptor(name="choose_event_option", requires_index=True)]),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.request.option_index == 0
    assert decision.reason == "take_event_relic_or_power"


def test_policy_chest_claims_relic_before_proceeding() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="CHEST",
            run_id="run-10c",
            chest=ChestPayload(
                is_opened=True,
                has_relic_been_claimed=False,
                relic_options=[ChestRelicOptionPayload(index=0, relic_id="VAJRA", name="Vajra", rarity="common")],
            ),
        ),
        AvailableActionsPayload(
            screen="CHEST",
            actions=[ActionDescriptor(name="choose_treasure_relic", requires_index=True), ActionDescriptor(name="proceed")],
        ),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "choose_treasure_relic"
    assert decision.reason == "take_best_treasure_relic"


def test_policy_chest_proceeds_after_relic_has_been_claimed() -> None:
    observation = _build_observation(
        GameStatePayload(
            screen="CHEST",
            run_id="run-10d",
            chest=ChestPayload(
                is_opened=True,
                has_relic_been_claimed=True,
                relic_options=[ChestRelicOptionPayload(index=0, relic_id="VAJRA", name="Vajra", rarity="common")],
            ),
        ),
        AvailableActionsPayload(
            screen="CHEST",
            actions=[ActionDescriptor(name="choose_treasure_relic", requires_index=True), ActionDescriptor(name="proceed")],
        ),
    )

    decision = SimplePolicy().choose(observation)

    assert decision.action is not None
    assert decision.action.action == "proceed"
    assert decision.reason == "leave_opened_chest"


def test_planner_policy_emits_ranked_combat_actions() -> None:
    planner_policy = build_policy_pack("planner")
    observation = _build_observation(
        GameStatePayload(
            screen="COMBAT",
            run_id="run-11",
            turn=1,
            in_combat=True,
            run=_run_payload(current_hp=70, max_hp=80, floor=3, gold=90),
            combat=CombatPayload(
                player=CombatPlayerPayload(current_hp=70, max_hp=80, block=0, energy=3),
                hand=[
                    CombatHandCardPayload(
                        index=0,
                        card_id="STRIKE_IRONCLAD",
                        name="Strike",
                        playable=True,
                        requires_target=True,
                        valid_target_indices=[0],
                        energy_cost=1,
                        rules_text="Deal 6 damage.",
                    ),
                    CombatHandCardPayload(
                        index=1,
                        card_id="BASH",
                        name="Bash",
                        playable=True,
                        requires_target=True,
                        valid_target_indices=[0],
                        energy_cost=2,
                        rules_text="Deal 8 damage. Apply 2 Vulnerable.",
                    ),
                    CombatHandCardPayload(
                        index=2,
                        card_id="DEFEND_IRONCLAD",
                        name="Defend",
                        playable=True,
                        requires_target=False,
                        energy_cost=1,
                        rules_text="Gain 5 Block.",
                    ),
                ],
                enemies=[
                    CombatEnemyPayload(
                        index=0,
                        enemy_id="SLIME_SMALL",
                        name="Slime",
                        current_hp=10,
                        max_hp=10,
                        block=0,
                        is_alive=True,
                        intent="ATTACK",
                    )
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="COMBAT",
            actions=[
                ActionDescriptor(name="play_card", requires_index=True, requires_target=True),
                ActionDescriptor(name="end_turn"),
            ],
        ),
    )

    decision = planner_policy.choose(observation)

    assert decision.action is not None
    assert decision.action.action == "play_card"
    assert decision.planner_name == "combat-hand-planner-v1"
    assert decision.policy_pack == "planner"
    assert decision.policy_handler == "combat-hand-planner"
    assert decision.ranked_actions
    assert decision.ranked_actions[0].metadata["sequence"][0] == decision.action.action_id


def test_predictor_dominant_mode_can_override_map_route(tmp_path: Path) -> None:
    predictor_path = _write_predictor_model(
        tmp_path / "map-predictor.json",
        feature_names=["run:gold"],
        reward_weights=[0.25],
    )
    policy = build_policy_pack(
        "baseline",
        predictor_config=PredictorRuntimeConfig(
            model_path=predictor_path,
            mode="dominant",
            hooks=("map",),
        ),
    )
    observation = _build_observation(
        GameStatePayload(
            screen="MAP",
            run_id="run-predict-map",
            run=RunPayload(character_id="IRONCLAD", floor=6, current_hp=20, max_hp=80, gold=110, max_energy=3),
            map=MapPayload(
                available_nodes=[
                    MapNodePayload(index=0, row=6, col=0, node_type="elite"),
                    MapNodePayload(index=1, row=6, col=1, node_type="rest"),
                ],
                is_travel_enabled=True,
            ),
        ),
        AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="choose_map_node", requires_index=True)]),
    )

    decision = policy.choose(observation)

    assert decision.action is not None
    assert decision.action.request.option_index == 0
    assert decision.trace_metadata["predictor"]["mode"] == "dominant"
    assert decision.trace_metadata["predictor"]["domain"] == "map"


def test_predictor_dominant_mode_can_change_combat_pick(tmp_path: Path) -> None:
    predictor_path = _write_predictor_model(
        tmp_path / "combat-predictor.json",
        feature_names=["combat:player_block"],
        reward_weights=[1.0],
    )
    policy = build_policy_pack(
        "baseline",
        predictor_config=PredictorRuntimeConfig(
            model_path=predictor_path,
            mode="dominant",
            hooks=("combat",),
        ),
    )
    observation = _build_observation(
        GameStatePayload(
            screen="COMBAT",
            run_id="run-predict-combat",
            turn=1,
            in_combat=True,
            run=_run_payload(current_hp=24, max_hp=80, floor=3, gold=90),
            combat=CombatPayload(
                player=CombatPlayerPayload(current_hp=24, max_hp=80, block=0, energy=3),
                hand=[
                    CombatHandCardPayload(
                        index=0,
                        card_id="STRIKE_IRONCLAD",
                        name="Strike",
                        playable=True,
                        requires_target=True,
                        valid_target_indices=[0],
                        energy_cost=1,
                        rules_text="Deal 6 damage.",
                    ),
                    CombatHandCardPayload(
                        index=1,
                        card_id="DEFEND_IRONCLAD",
                        name="Defend",
                        playable=True,
                        requires_target=False,
                        energy_cost=1,
                        rules_text="Gain 5 Block.",
                    ),
                ],
                enemies=[
                    CombatEnemyPayload(
                        index=0,
                        enemy_id="SLIME_SMALL",
                        name="Slime",
                        current_hp=14,
                        max_hp=14,
                        block=0,
                        is_alive=True,
                        intent="ATTACK",
                    )
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="COMBAT",
            actions=[
                ActionDescriptor(name="play_card", requires_index=True, requires_target=True),
                ActionDescriptor(name="end_turn"),
            ],
        ),
    )

    decision = policy.choose(observation)

    assert decision.action is not None
    assert decision.action.action == "play_card"
    assert decision.action.request.card_index == 1
    assert decision.trace_metadata["predictor"]["selected"]["action_id"] == decision.action.action_id
    assert decision.trace_metadata["predictor"]["domain"] == "combat"


def test_strategic_runtime_can_choose_reward_skip(tmp_path: Path) -> None:
    checkpoint_path = _write_strategic_checkpoint(
        tmp_path / "strategic-reward.json",
        decision_types=("reward_card_pick",),
        decision_weights={"reward_card_pick": {"candidate:skip": 3.0}},
    )
    policy = build_policy_pack(
        "baseline",
        strategic_model_config=StrategicRuntimeConfig(
            checkpoint_path=checkpoint_path,
            mode="dominant",
            hooks=("reward",),
        ),
    )
    observation = _build_observation(
        GameStatePayload(
            screen="REWARD",
            run_id="run-strategic-skip",
            run=_run_payload(current_hp=52, max_hp=80, floor=12, gold=140),
            reward=RewardPayload(
                pending_card_choice=True,
                card_options=[
                    RewardCardOptionPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                    RewardCardOptionPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="REWARD",
            actions=[
                ActionDescriptor(name="choose_reward_card", requires_index=True),
                ActionDescriptor(name="skip_reward_cards"),
            ],
        ),
    )

    decision = policy.choose(observation)

    assert decision.action is not None
    assert decision.action.action == "skip_reward_cards"
    assert decision.reason == "skip_low_value_reward_card"
    assert decision.trace_metadata["strategic"]["selected"]["decision_type"] == "reward_card_pick"
    assert decision.trace_metadata["strategic"]["selected"]["candidate_id"] == "skip"


def test_strategic_selection_remove_guidance_is_not_shop_only(tmp_path: Path) -> None:
    checkpoint_path = _write_strategic_checkpoint(
        tmp_path / "strategic-remove.json",
        decision_types=("selection_remove",),
        decision_weights={"selection_remove": {"candidate:strike_ironclad": 3.0}},
    )
    strategic_config = StrategicRuntimeConfig(
        checkpoint_path=checkpoint_path,
        mode="dominant",
        hooks=("selection",),
    )
    event_selection_observation = _build_observation(
        GameStatePayload(
            screen="SELECTION",
            run_id="run-shop-remove",
            run=_run_payload(current_hp=68, max_hp=80, floor=7, gold=160),
            selection=SelectionPayload(
                kind="deck_card_select",
                selection_family="deck",
                semantic_mode="remove",
                source_type="event",
                prompt="Remove a card from your deck",
                required_count=1,
                cards=[
                    SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                    SelectionCardPayload(index=1, card_id="CURSE_DOUBT", name="Doubt"),
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="SELECTION",
            actions=[ActionDescriptor(name="select_deck_card", requires_index=True)],
        ),
    )
    event_policy = build_policy_pack("baseline", strategic_model_config=strategic_config)

    event_decision = event_policy.choose(event_selection_observation)

    assert event_decision.action is not None
    assert event_decision.action.request.option_index == 0
    assert event_decision.trace_metadata["strategic"]["selected"]["decision_type"] == "selection_remove"
    assert event_decision.trace_metadata["strategic"]["domain"] == "selection"

    shop_selection_observation = _build_observation(
        GameStatePayload(
            screen="SELECTION",
            run_id="run-shop-remove",
            run=_run_payload(current_hp=68, max_hp=80, floor=7, gold=160),
            selection=SelectionPayload(
                kind="deck_card_select",
                selection_family="deck",
                semantic_mode="remove",
                source_type="shop",
                source_room_type="Shop",
                prompt="Remove a card from your deck",
                required_count=1,
                cards=[
                    SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                    SelectionCardPayload(index=1, card_id="CURSE_DOUBT", name="Doubt"),
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="SELECTION",
            actions=[ActionDescriptor(name="select_deck_card", requires_index=True)],
        ),
    )
    policy = build_policy_pack("baseline", strategic_model_config=strategic_config)

    selection_decision = policy.choose(shop_selection_observation)
    assert selection_decision.action is not None
    assert selection_decision.action.request.option_index == 0
    assert selection_decision.trace_metadata["strategic"]["selected"]["decision_type"] == "selection_remove"
    assert selection_decision.trace_metadata["strategic"]["domain"] == "selection"


def test_strategic_selection_upgrade_guidance_uses_upgrade_taxonomy(tmp_path: Path) -> None:
    checkpoint_path = _write_strategic_checkpoint(
        tmp_path / "strategic-upgrade.json",
        decision_types=("selection_upgrade",),
        decision_weights={"selection_upgrade": {"candidate:bash": 3.0}},
    )
    policy = build_policy_pack(
        "baseline",
        strategic_model_config=StrategicRuntimeConfig(
            checkpoint_path=checkpoint_path,
            mode="dominant",
            hooks=("selection",),
        ),
    )
    observation = _build_observation(
        GameStatePayload(
            screen="SELECTION",
            run_id="run-selection-upgrade",
            run=_run_payload(current_hp=68, max_hp=80, floor=7, gold=160),
            selection=SelectionPayload(
                kind="deck_upgrade_select",
                selection_family="deck",
                semantic_mode="upgrade",
                source_type="rest",
                source_room_type="Rest",
                prompt="Upgrade a card",
                required_count=1,
                cards=[
                    SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                    SelectionCardPayload(index=1, card_id="BASH", name="Bash"),
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="SELECTION",
            actions=[ActionDescriptor(name="select_deck_card", requires_index=True)],
        ),
    )

    decision = policy.choose(observation)

    assert decision.action is not None
    assert decision.action.request.option_index == 1
    assert decision.trace_metadata["strategic"]["selected"]["decision_type"] == "selection_upgrade"


def _build_observation(state: GameStatePayload, descriptors: AvailableActionsPayload) -> StepObservation:
    result = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type=state.screen,
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=result.candidates,
        build_warnings=result.unsupported_actions,
    )


def _selection_observation(
    *,
    run_id: str,
    mode: str,
    required_count: int,
    remaining_count: int,
    selected_count: int,
    cards: list[SelectionCardPayload],
    can_confirm: bool = True,
) -> StepObservation:
    return _build_observation(
        GameStatePayload(
            screen="CARD_SELECTION",
            run_id=run_id,
            run=_run_payload(current_hp=66, max_hp=80, floor=6, gold=120),
            selection=SelectionPayload(
                kind="deck_transform_select" if mode == "transform" else "deck_card_select",
                selection_family="deck",
                semantic_mode=mode,
                source_type="event",
                source_room_type="Event",
                prompt="Select cards",
                required_count=required_count,
                remaining_count=remaining_count,
                selected_count=selected_count,
                requires_confirmation=can_confirm,
                can_confirm=can_confirm,
                supports_multi_select=required_count > 1,
                cards=cards,
            ),
            agent_view=_agent_view_with_deck(
                [
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Strike [1]: Deal 6 damage.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Defend [1]: Gain 5 Block.",
                    "Bash [2]: Deal 8 damage. Apply 2 Vulnerable.",
                    "Doubt: Unplayable.",
                ]
            ),
        ),
        AvailableActionsPayload(
            screen="CARD_SELECTION",
            actions=[
                ActionDescriptor(name="select_deck_card", requires_index=True),
                ActionDescriptor(name="confirm_selection"),
            ],
        ),
    )


def _run_payload(
    *,
    current_hp: int,
    max_hp: int,
    floor: int,
    gold: int,
    boss_encounter_id: str | None = None,
    boss_encounter_name: str | None = None,
) -> RunPayload:
    return RunPayload(
        character_id="IRONCLAD",
        character_name="Ironclad",
        act_index=0,
        act_number=1,
        act_id="ACT_1",
        act_name="Act 1",
        boss_encounter=(
            None
            if boss_encounter_id is None and boss_encounter_name is None
            else EncounterSummaryPayload(
                encounter_id=boss_encounter_id or "UNKNOWN_BOSS",
                name=boss_encounter_name or "Unknown Boss",
                room_type="Boss",
            )
        ),
        floor=floor,
        current_hp=current_hp,
        max_hp=max_hp,
        gold=gold,
        max_energy=3,
    )


def _build_boss_route_map() -> MapPayload:
    return MapPayload(
        current_node=MapCoordPayload(row=1, col=1),
        is_travel_enabled=True,
        rows=5,
        cols=3,
        boss_node=MapCoordPayload(row=4, col=1),
        available_nodes=[
            MapNodePayload(index=0, row=2, col=0, node_type="Elite", state="Travelable"),
            MapNodePayload(index=1, row=2, col=2, node_type="Monster", state="Travelable"),
        ],
        nodes=[
            MapGraphNodePayload(
                row=1,
                col=1,
                node_type="Monster",
                state="Current",
                is_current=True,
                children=[MapCoordPayload(row=2, col=0), MapCoordPayload(row=2, col=2)],
            ),
            MapGraphNodePayload(
                row=2,
                col=0,
                node_type="Elite",
                state="Travelable",
                is_available=True,
                parents=[MapCoordPayload(row=1, col=1)],
                children=[MapCoordPayload(row=3, col=0)],
            ),
            MapGraphNodePayload(
                row=3,
                col=0,
                node_type="Monster",
                state="Visible",
                parents=[MapCoordPayload(row=2, col=0)],
                children=[MapCoordPayload(row=4, col=1)],
            ),
            MapGraphNodePayload(
                row=2,
                col=2,
                node_type="Monster",
                state="Travelable",
                is_available=True,
                parents=[MapCoordPayload(row=1, col=1)],
                children=[MapCoordPayload(row=3, col=2)],
            ),
            MapGraphNodePayload(
                row=3,
                col=2,
                node_type="Shop",
                state="Visible",
                parents=[MapCoordPayload(row=2, col=2)],
                children=[MapCoordPayload(row=4, col=1)],
            ),
            MapGraphNodePayload(
                row=4,
                col=1,
                node_type="Boss",
                state="Visible",
                is_boss=True,
                parents=[MapCoordPayload(row=3, col=0), MapCoordPayload(row=3, col=2)],
            ),
        ],
    )


def _agent_view_with_deck(deck_lines: list[str]) -> dict:
    return {"run": {"deck": [{"line": line} for line in deck_lines], "relics": ["Burning Blood"]}}


def _write_predictor_model(path: Path, *, feature_names: list[str], reward_weights: list[float]) -> Path:
    predictor = CombatOutcomePredictor(
        feature_names=feature_names,
        feature_means=[0.0 for _ in feature_names],
        feature_stds=[1.0 for _ in feature_names],
        outcome_head=PredictorHead(name="outcome_win", kind="logistic", weights=[0.0 for _ in feature_names], bias=0.0),
        reward_head=PredictorHead(
            name="reward",
            kind="linear",
            weights=reward_weights,
            bias=0.0,
            target_mean=0.0,
            target_std=1.0,
        ),
        damage_head=PredictorHead(
            name="damage_delta",
            kind="linear",
            weights=[0.0 for _ in feature_names],
            bias=0.0,
            target_mean=0.0,
            target_std=1.0,
        ),
        metadata={"calibration": {"validation": {"objective": 0.123}}},
    )
    predictor.save(path)
    return path


def _write_strategic_checkpoint(
    path: Path,
    *,
    decision_types: tuple[str, ...],
    global_weights: dict[str, float] | None = None,
    decision_weights: dict[str, dict[str, float]] | None = None,
) -> Path:
    model = StrategicPretrainModel(
        global_ranking_head=_SparseScoringHead(name="global_ranking", weights=dict(global_weights or {})),
        decision_ranking_heads={
            decision_type: _SparseScoringHead(
                name=decision_type,
                weights=dict((decision_weights or {}).get(decision_type, {})),
            )
            for decision_type in decision_types
        },
        global_value_head=_SparseScoringHead(name="global_value"),
        decision_value_heads={
            decision_type: _SparseScoringHead(name=decision_type)
            for decision_type in decision_types
        },
        metadata={"decision_types": list(decision_types)},
    )
    return model.save(path)
