from __future__ import annotations

from collections.abc import Callable, Iterable

from .models import ActionDescriptor, ActionRequest, AvailableActionsPayload, GameStatePayload
from .types import CandidateAction, CandidateBuildResult

_SIMPLE_ACTIONS = {
    "abandon_run",
    "close_cards_view",
    "close_main_menu_submenu",
    "close_shop_inventory",
    "collect_rewards_and_proceed",
    "confirm_modal",
    "confirm_selection",
    "confirm_timeline_overlay",
    "continue_run",
    "decrease_ascension",
    "dismiss_modal",
    "disconnect_multiplayer_lobby",
    "embark",
    "end_turn",
    "host_multiplayer_lobby",
    "increase_ascension",
    "join_multiplayer_lobby",
    "open_character_select",
    "open_custom_run",
    "open_chest",
    "open_shop_inventory",
    "open_timeline",
    "proceed",
    "ready_multiplayer_lobby",
    "remove_card_at_shop",
    "return_to_main_menu",
    "skip_reward_cards",
    "unready",
}


def build_candidate_actions(
    state: GameStatePayload,
    descriptors: AvailableActionsPayload,
) -> CandidateBuildResult:
    candidates: list[CandidateAction] = []
    warnings: list[str] = []

    for descriptor in descriptors.actions:
        builder = _BUILDERS.get(descriptor.name)
        if builder is not None:
            candidates.extend(builder(state, descriptor))
            continue

        if descriptor.name in _SIMPLE_ACTIONS and not descriptor.requires_index and not descriptor.requires_target:
            candidates.append(_simple_candidate(descriptor.name))
            continue

        warnings.append(f"Unsupported action descriptor: {descriptor.name}")

    return CandidateBuildResult(
        candidates=candidates,
        unsupported_actions=warnings,
    )


def _simple_candidate(action: str) -> CandidateAction:
    action_id = _action_id(action)
    label = _titleize(action)
    return CandidateAction(
        action_id=action_id,
        action=action,
        label=label,
        request=ActionRequest(
            action=action,
            client_context={"candidate_id": action_id, "label": label},
        ),
        source="simple",
    )


def _play_card_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    cards = state.combat.hand if state.combat is not None else []
    candidates: list[CandidateAction] = []
    for card in cards:
        if not card.playable:
            continue

        if card.requires_target:
            for target_index in card.valid_target_indices:
                action_id = _action_id("play_card", card_index=card.index, target_index=target_index)
                label = f"Play {card.name} [{card.index}] -> target {target_index}"
                candidates.append(
                    CandidateAction(
                        action_id=action_id,
                        action="play_card",
                        label=label,
                        request=ActionRequest(
                            action="play_card",
                            card_index=card.index,
                            target_index=target_index,
                            client_context={"candidate_id": action_id, "label": label},
                        ),
                        source="combat.hand",
                    )
                )
            continue

        action_id = _action_id("play_card", card_index=card.index)
        label = f"Play {card.name} [{card.index}]"
        candidates.append(
            CandidateAction(
                action_id=action_id,
                action="play_card",
                label=label,
                request=ActionRequest(
                    action="play_card",
                    card_index=card.index,
                    client_context={"candidate_id": action_id, "label": label},
                ),
                source="combat.hand",
            )
        )
    return candidates


def _map_node_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    nodes = state.map.available_nodes if state.map is not None else []
    return _option_candidates(
        "choose_map_node",
        nodes,
        label_fn=lambda node: f"Choose map node ({node.row}, {node.col}) {node.node_type}",
        source="map.available_nodes",
    )


def _claim_reward_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    rewards = state.reward.rewards if state.reward is not None else []
    claimable = [reward for reward in rewards if reward.claimable]
    return _option_candidates(
        "claim_reward",
        claimable,
        label_fn=lambda reward: f"Claim reward [{reward.index}] {reward.reward_type}",
        source="reward.rewards",
    )


def _reward_card_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    cards = state.reward.card_options if state.reward is not None else []
    return _option_candidates(
        "choose_reward_card",
        cards,
        label_fn=lambda card: f"Choose reward card {card.name} [{card.index}]",
        source="reward.card_options",
    )


def _selection_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    cards = state.selection.cards if state.selection is not None else []
    return _option_candidates(
        "select_deck_card",
        cards,
        label_fn=lambda card: f"Select deck card {card.name} [{card.index}]",
        source="selection.cards",
    )


def _timeline_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    slots = state.timeline.slots if state.timeline is not None else []
    actionable = [slot for slot in slots if slot.is_actionable]
    return _option_candidates(
        "choose_timeline_epoch",
        actionable,
        label_fn=lambda slot: f"Choose epoch {slot.title} [{slot.index}]",
        source="timeline.slots",
    )


def _chest_relic_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    relics = state.chest.relic_options if state.chest is not None else []
    return _option_candidates(
        "choose_treasure_relic",
        relics,
        label_fn=lambda relic: f"Choose relic {relic.name} [{relic.index}]",
        source="chest.relic_options",
    )


def _event_option_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    options = state.event.options if state.event is not None else []
    unlocked = [option for option in options if not option.is_locked]
    return _option_candidates(
        "choose_event_option",
        unlocked,
        label_fn=lambda option: f"Choose event option {option.title} [{option.index}]",
        source="event.options",
    )


def _rest_option_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    options = state.rest.options if state.rest is not None else []
    enabled = [option for option in options if option.is_enabled]
    return _option_candidates(
        "choose_rest_option",
        enabled,
        label_fn=lambda option: f"Choose rest option {option.title} [{option.index}]",
        source="rest.options",
    )


def _shop_card_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    cards = state.shop.cards if state.shop is not None else []
    stocked = [card for card in cards if card.is_stocked and card.enough_gold]
    return _option_candidates(
        "buy_card",
        stocked,
        label_fn=lambda card: f"Buy card {card.name} [{card.index}] for {card.price}",
        source="shop.cards",
    )


def _shop_relic_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    relics = state.shop.relics if state.shop is not None else []
    stocked = [relic for relic in relics if relic.is_stocked and relic.enough_gold]
    return _option_candidates(
        "buy_relic",
        stocked,
        label_fn=lambda relic: f"Buy relic {relic.name} [{relic.index}] for {relic.price}",
        source="shop.relics",
    )


def _shop_potion_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    potions = state.shop.potions if state.shop is not None else []
    stocked = [potion for potion in potions if potion.is_stocked and potion.enough_gold]
    return _option_candidates(
        "buy_potion",
        stocked,
        label_fn=lambda potion: f"Buy potion {potion.name} [{potion.index}] for {potion.price}",
        source="shop.potions",
    )


def _character_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    if state.character_select is not None:
        characters = state.character_select.characters
        source = "character_select.characters"
    else:
        characters = state.custom_run.characters if state.custom_run is not None else []
        source = "custom_run.characters"
    unlocked = [character for character in characters if not character.is_locked]
    return _option_candidates(
        "select_character",
        unlocked,
        label_fn=lambda character: f"Select character {character.name} [{character.index}]",
        source=source,
    )


def _custom_modifier_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    modifiers = state.custom_run.modifiers if state.custom_run is not None else []
    candidates: list[CandidateAction] = []
    for modifier in modifiers:
        action_id = _action_id("toggle_custom_modifier", modifier_index=modifier.index)
        label = f"Toggle custom modifier {modifier.name} [{modifier.index}]"
        candidates.append(
            CandidateAction(
                action_id=action_id,
                action="toggle_custom_modifier",
                label=label,
                request=ActionRequest(
                    action="toggle_custom_modifier",
                    modifier_index=modifier.index,
                    client_context={"candidate_id": action_id, "label": label},
                ),
                source="custom_run.modifiers",
                metadata={
                    "modifier_id": modifier.modifier_id,
                    "modifier_name": modifier.name,
                    "is_selected": modifier.is_selected,
                },
            )
        )
    return candidates


def _custom_seed_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    custom = state.custom_run
    if custom is None:
        return []

    seed = (custom.seed or "").strip()
    if not seed:
        return []

    action_id = _action_id("set_custom_seed", seed=seed)
    label = f"Set custom seed {seed}"
    return [
        CandidateAction(
            action_id=action_id,
            action="set_custom_seed",
            label=label,
            request=ActionRequest(
                action="set_custom_seed",
                seed=seed,
                client_context={"candidate_id": action_id, "label": label},
            ),
            source="custom_run.seed",
            metadata={"seed": seed, "is_current_value": True},
        )
    ]


def _custom_ascension_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    custom = state.custom_run
    if custom is None:
        return []

    max_ascension = max(int(custom.max_ascension), int(custom.ascension), 0)
    candidates: list[CandidateAction] = []
    for ascension in range(max_ascension + 1):
        action_id = _action_id("set_custom_ascension", ascension=ascension)
        label = f"Set custom ascension {ascension}"
        candidates.append(
            CandidateAction(
                action_id=action_id,
                action="set_custom_ascension",
                label=label,
                request=ActionRequest(
                    action="set_custom_ascension",
                    ascension=ascension,
                    client_context={"candidate_id": action_id, "label": label},
                ),
                source="custom_run.ascension",
                metadata={"ascension": ascension, "is_current_value": ascension == custom.ascension},
            )
        )
    return candidates


def _set_custom_modifiers_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    custom = state.custom_run
    if custom is None:
        return []

    modifier_ids = [modifier.modifier_id for modifier in custom.modifiers if modifier.is_selected]
    action_id = _action_id("set_custom_modifiers", modifier_ids=modifier_ids)
    modifier_label = ", ".join(modifier_ids) if modifier_ids else "none"
    label = f"Set custom modifiers {modifier_label}"
    return [
        CandidateAction(
            action_id=action_id,
            action="set_custom_modifiers",
            label=label,
            request=ActionRequest(
                action="set_custom_modifiers",
                modifier_ids=list(modifier_ids),
                client_context={"candidate_id": action_id, "label": label},
            ),
            source="custom_run.modifiers",
            metadata={"modifier_ids": list(modifier_ids), "is_current_value": True},
        )
    ]


def _use_potion_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    potions = state.run.potions if state.run is not None else []
    usable = [potion for potion in potions if potion.can_use]
    return _potion_candidates("use_potion", usable)


def _discard_potion_candidates(state: GameStatePayload, _: ActionDescriptor) -> list[CandidateAction]:
    potions = state.run.potions if state.run is not None else []
    discardable = [potion for potion in potions if potion.can_discard]
    return _potion_candidates("discard_potion", discardable)


def _potion_candidates(action: str, potions: Iterable[object]) -> list[CandidateAction]:
    candidates: list[CandidateAction] = []
    for potion in potions:
        if getattr(potion, "requires_target", False):
            for target_index in getattr(potion, "valid_target_indices", []):
                action_id = _action_id(action, option_index=potion.index, target_index=target_index)
                label = f"{_titleize(action)} {potion.name} [{potion.index}] -> target {target_index}"
                candidates.append(
                    CandidateAction(
                        action_id=action_id,
                        action=action,
                        label=label,
                        request=ActionRequest(
                            action=action,
                            option_index=potion.index,
                            target_index=target_index,
                            client_context={"candidate_id": action_id, "label": label},
                        ),
                        source="run.potions",
                    )
                )
            continue

        action_id = _action_id(action, option_index=potion.index)
        label = f"{_titleize(action)} {potion.name} [{potion.index}]"
        candidates.append(
            CandidateAction(
                action_id=action_id,
                action=action,
                label=label,
                request=ActionRequest(
                    action=action,
                    option_index=potion.index,
                    client_context={"candidate_id": action_id, "label": label},
                ),
                source="run.potions",
            )
        )
    return candidates


def _option_candidates(
    action: str,
    items: Iterable[object],
    *,
    label_fn: Callable[[object], str],
    source: str,
) -> list[CandidateAction]:
    candidates: list[CandidateAction] = []
    for item in items:
        label = label_fn(item)
        action_id = _action_id(action, option_index=item.index)
        candidates.append(
            CandidateAction(
                action_id=action_id,
                action=action,
                label=label,
                request=ActionRequest(
                    action=action,
                    option_index=item.index,
                    client_context={"candidate_id": action_id, "label": label},
                ),
                source=source,
            )
        )
    return candidates


def _action_id(
    action: str,
    *,
    card_index: int | None = None,
    option_index: int | None = None,
    modifier_index: int | None = None,
    ascension: int | None = None,
    seed: str | None = None,
    modifier_ids: Iterable[str] | None = None,
    target_index: int | None = None,
) -> str:
    parts = [action]
    if card_index is not None:
        parts.append(f"card={card_index}")
    if option_index is not None:
        parts.append(f"option={option_index}")
    if modifier_index is not None:
        parts.append(f"modifier={modifier_index}")
    if ascension is not None:
        parts.append(f"ascension={ascension}")
    if seed is not None:
        parts.append(f"seed={seed}")
    if modifier_ids is not None:
        normalized_modifier_ids = [str(modifier_id) for modifier_id in modifier_ids if str(modifier_id)]
        encoded_modifier_ids = ",".join(normalized_modifier_ids) if normalized_modifier_ids else "none"
        parts.append(f"modifiers={encoded_modifier_ids}")
    if target_index is not None:
        parts.append(f"target={target_index}")
    return "|".join(parts)


def _titleize(action: str) -> str:
    return action.replace("_", " ").title()


_BUILDERS = {
    "play_card": _play_card_candidates,
    "choose_map_node": _map_node_candidates,
    "claim_reward": _claim_reward_candidates,
    "choose_reward_card": _reward_card_candidates,
    "select_deck_card": _selection_candidates,
    "choose_timeline_epoch": _timeline_candidates,
    "choose_treasure_relic": _chest_relic_candidates,
    "choose_event_option": _event_option_candidates,
    "choose_rest_option": _rest_option_candidates,
    "buy_card": _shop_card_candidates,
    "buy_relic": _shop_relic_candidates,
    "buy_potion": _shop_potion_candidates,
    "select_character": _character_candidates,
    "set_custom_seed": _custom_seed_candidates,
    "set_custom_ascension": _custom_ascension_candidates,
    "toggle_custom_modifier": _custom_modifier_candidates,
    "set_custom_modifiers": _set_custom_modifiers_candidates,
    "use_potion": _use_potion_candidates,
    "discard_potion": _discard_potion_candidates,
}
