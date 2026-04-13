from __future__ import annotations

from collections import deque
from copy import deepcopy
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal

from sts2_rl.collect.community_prior import (
    CommunityCardPriorSource,
    CommunityPriorDomain,
    CommunityPriorRuntimeConfig,
)
from sts2_rl.collect.strategic_runtime import (
    StrategicRuntimeAdapter,
    StrategicRuntimeConfig,
    StrategicRuntimeTrace,
    event_candidate_id,
    rest_candidate_id,
    reward_candidate_id,
    selection_decision_type_for_mode,
    reward_runtime_domain,
    selection_candidate_id,
    shop_buy_candidate_id,
)
from sts2_rl.data.trajectory import build_state_summary
from sts2_rl.env.models import (
    EventOptionPayload,
    MapGraphNodePayload,
    MapNodePayload,
    RestOptionPayload,
    RewardOptionPayload,
    RewardCardOptionPayload,
    SelectionCardPayload,
    SelectionPayload,
    ShopCardPayload,
    ShopPotionPayload,
    ShopRelicPayload,
)
from sts2_rl.env.types import CandidateAction, StepObservation
from sts2_rl.predict import PredictorRuntimeAdapter, PredictorRuntimeConfig

_NUMBERED_PATTERNS = {
    "damage": (r"(\d+)\s*damage", r"造成(\d+)点伤害", r"(\d+)点伤害"),
    "block": (r"(\d+)\s*block", r"获得(\d+)点格挡", r"(\d+)点格挡"),
    "draw": (r"draw\s*(\d+)", r"抽(\d+)张牌", r"抽取(\d+)张牌"),
    "energy": (r"(\d+)\s*energy", r"获得(\d+)点能量", r"(\d+)点能量"),
    "vulnerable": (r"(\d+)\s*vulnerable", r"(\d+)层易伤"),
    "weak": (r"(\d+)\s*weak", r"(\d+)层虚弱"),
    "strength": (r"(\d+)\s*strength", r"(\d+)点力量"),
    "dexterity": (r"(\d+)\s*dexterity", r"(\d+)点敏捷"),
    "focus": (r"(\d+)\s*focus", r"(\d+)点集中"),
}

_NUMBERED_KEYWORDS = {
    "damage": 0.45,
    "block": 0.38,
    "draw": 1.00,
    "energy": 1.35,
    "vulnerable": 0.80,
    "weak": 0.65,
    "strength": 1.10,
    "dexterity": 1.10,
    "focus": 1.15,
}

_TEXT_BONUSES = {
    ("all enemies", "所有敌人", "全体敌人"): 1.50,
    ("aoe",): 1.20,
    ("retain", "保留"): 0.30,
    ("exhaust", "消耗"): -0.20,
    ("ethereal", "虚无"): -0.30,
    ("poison", "中毒"): 0.75,
    ("shiv", "小刀"): 0.60,
    ("scry", "占卜"): 0.35,
    ("channel", "充能"): 0.90,
    ("gain", "获得"): 0.10,
}

_AOE_FRAGMENTS = ("all enemies", "所有敌人", "全体敌人", "aoe")
_SCALING_FRAGMENTS = ("strength", "dexterity", "focus", "ritual", "power", "每回合", "永久")
_SUSTAIN_FRAGMENTS = ("block", "格挡", "heal", "recover", "rest", "weak", "虚弱")
_FRONTLOAD_FRAGMENTS = ("damage", "伤害", "vulnerable", "易伤")
_SHOP_QUALITY_FRAGMENTS = ("shop", "merchant", "remove", "purge", "relic", "card", "remove a card")

_BOSS_AOE_FRAGMENTS = ("collector", "slime", "swarm", "brood", "queen", "horde", "nest")
_BOSS_FRONTLOAD_FRAGMENTS = ("slime", "collector", "guardian", "champ", "automaton", "golem")
_BOSS_SUSTAIN_FRAGMENTS = ("hex", "ghost", "ember", "flame", "burn", "ash", "pyre")
_BOSS_SCALING_FRAGMENTS = ("champ", "guardian", "automaton", "time", "awakened", "heart", "deca", "donu")

_RARITY_SCORES = {
    "common": 1.0,
    "uncommon": 2.5,
    "rare": 4.0,
    "boss": 5.0,
}

_CARD_ID_BONUSES = {
    "bash": 1.80,
    "pommel": 1.70,
    "shrug": 1.75,
    "armaments": 1.60,
    "battle_trance": 1.60,
    "inflame": 1.50,
    "shockwave": 2.10,
    "whirlwind": 1.40,
    "anger": 1.20,
    "bludgeon": 1.30,
    "molten_fist": 1.50,
    "blood_wall": 0.90,
    "perfected_strike": 0.80,
    "second_wind": 1.25,
    "cinder": 0.90,
    "ritual": 0.60,
}

_CARD_ID_PENALTIES = {
    "strike": -0.90,
    "defend": -0.70,
    "curse": -3.00,
    "wound": -3.00,
    "burn": -2.60,
    "dazed": -2.80,
    "slime": -2.40,
}

_RELIC_BONUSES = {
    "energy": 1.40,
    "strength": 1.10,
    "focus": 1.10,
    "draw": 0.90,
    "card": 0.60,
    "block": 0.50,
    "hp": 0.40,
}

_POTION_BONUSES = {
    "damage": 0.90,
    "block": 0.70,
    "heal": 1.00,
    "energy": 1.10,
    "strength": 0.90,
    "draw": 0.70,
    "vulnerable": 0.65,
    "weak": 0.55,
}

_REST_FRAGMENTS = ("rest", "heal", "recover", "休息", "回复", "恢复", "治疗")
_UPGRADE_FRAGMENTS = ("upgrade", "smith", "锻造", "升级")
_DIG_FRAGMENTS = ("dig", "挖掘")
_REMOVE_FRAGMENTS = ("remove", "purge", "delete", "净化", "移除", "删除")
_TRANSFORM_FRAGMENTS = ("transform", "变形", "转换")
_RELIC_FRAGMENTS = ("relic", "遗物")
_GOLD_FRAGMENTS = ("gold", "金币", "[gold]")
_POTION_FRAGMENTS = ("potion", "药水")
_DRAW_FRAGMENTS = ("draw", "抽")
_ENERGY_FRAGMENTS = ("energy", "能量")
_DAMAGE_FRAGMENTS = ("damage", "伤害", "造成")
_BLOCK_FRAGMENTS = ("block", "格挡")
_CURSE_FRAGMENTS = ("curse", "诅咒")
_STATUS_FRAGMENTS = ("wound", "burn", "dazed", "slime", "伤口", "灼伤", "眩晕", "黏液")
_SELF_DAMAGE_FRAGMENTS = ("lose hp", "lose health", "失去", "流失")
_STRIKE_TITLE_PATTERNS = (r"^strike(\b|[\+\* ])", r"^打击([\+\* ]|$)")
_DEFEND_TITLE_PATTERNS = (r"^defend(\b|[\+\* ])", r"^防御([\+\* ]|$)")
_RICH_TEXT_TAG_PATTERN = re.compile(r"\[[^\]]+\]")
_CARD_SELECTION_COUNT_PATTERNS = (
    r"(?:choose|select)\s+(\d+)\s+cards?",
    r"(?:remove|transform|upgrade)\s+(\d+)\s+cards?",
    r"选择\s*(\d+)\s*张牌",
    r"移除\s*(\d+)\s*张牌",
    r"升级\s*(\d+)\s*张牌",
    r"变形\s*(\d+)\s*张牌",
)


@dataclass(frozen=True)
class RankedAction:
    action_id: str
    action: str
    score: float
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    action: CandidateAction | None
    stage: str
    reason: str
    score: float | None = None
    policy_pack: str | None = None
    policy_handler: str | None = None
    planner_name: str | None = None
    planner_strategy: str | None = None
    ranked_actions: tuple[RankedAction, ...] = ()
    trace_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _GuidedCandidate:
    action: CandidateAction
    reason: str
    heuristic_score: float
    final_score: float
    predictor_trace: dict[str, Any] | None = None
    strategic_trace: dict[str, Any] | None = None
    trace_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SelectionPlanMember:
    option_index: int
    card_id: str
    name: str
    heuristic_score: float
    final_score: float


@dataclass
class _SelectionTransactionState:
    transaction_id: str
    run_id: str
    screen_type: str
    selection_kind: str
    selection_family: str
    semantic_mode: str
    source_type: str
    source_room_type: str
    prompt_loc_table: str | None
    prompt_loc_key: str | None
    required_count: int
    requires_confirmation: bool
    supports_multi_select: bool
    planned_members: list[_SelectionPlanMember] = field(default_factory=list)
    completed_option_indices: list[int] = field(default_factory=list)
    pending_option_index: int | None = None
    phase: str = "planning"
    observed_selected_count: int = 0
    observed_remaining_count: int = 0
    recovery_count: int = 0
    recovery_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SimplePolicyConfig:
    profile_name: str = "baseline"
    preferred_character_ids: tuple[str, ...] = ("IRONCLAD", "THE_SILENT", "DEFECT", "WATCHER")
    skip_reward_card_below_score: float = 0.95
    preferred_gold_for_shop: int = 140
    reserve_gold_after_shop: int = 35
    low_hp_ratio: float = 0.45
    critical_hp_ratio: float = 0.28
    elite_hp_ratio: float = 0.72
    upgrade_hp_ratio: float = 0.62
    buy_shop_card_above_score: float = 1.35
    buy_shop_potion_above_score: float = 1.25
    auto_recover_game_over: bool = True
    strategic_bias: bool = True
    route_strategy: Literal["legacy_node", "boss_path_planner"] = "boss_path_planner"
    route_planner_name: str = "boss-conditioned-route-planner-v1"
    route_planner_depth: int = 4
    route_planner_width: int = 5
    route_planner_discount: float = 0.86
    combat_strategy: Literal["heuristic", "planner", "planner_assist"] = "heuristic"
    planner_name: str = "combat-hand-planner-v1"
    planner_depth: int = 2
    planner_width: int = 4
    planner_node_budget: int = 24
    planner_discount: float = 0.92
    planner_min_score_to_commit: float = 0.15
    planner_end_turn_penalty: float = 0.45
    predictor_config: PredictorRuntimeConfig | None = None
    community_prior_config: CommunityPriorRuntimeConfig | None = None
    strategic_model_config: StrategicRuntimeConfig | None = None


@dataclass(frozen=True)
class DeckProfile:
    deck_lines: tuple[str, ...]
    relic_names: tuple[str, ...]
    deck_size: int
    strike_count: int
    defend_count: int
    curse_count: int
    status_count: int
    frontload_score: float
    aoe_score: float
    sustain_score: float
    scaling_score: float


@dataclass(frozen=True)
class BossStrategicProfile:
    encounter_id: str | None
    encounter_name: str | None
    frontload_demand: float
    aoe_demand: float
    sustain_demand: float
    scaling_demand: float
    upgrade_urgency: float
    shop_urgency: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunContext:
    floor: int
    gold: int
    current_hp: int
    max_hp: int
    hp_ratio: float
    max_energy: int
    occupied_potions: int
    empty_potion_slots: int
    character_id: str | None
    ascension: int | None
    deck: DeckProfile
    act_index: int | None
    act_number: int | None
    act_id: str | None
    act_name: str | None
    boss_encounter_id: str | None
    boss_encounter_name: str | None
    boss_profile: BossStrategicProfile
    current_to_boss_distance: int | None
    next_rest_distance: int | None
    next_shop_distance: int | None
    next_elite_distance: int | None
    next_event_distance: int | None
    next_treasure_distance: int | None
    available_map_branch_count: int


@dataclass(frozen=True)
class RoutePlan:
    action: CandidateAction
    first_node: MapNodePayload
    prefix: tuple[MapGraphNodePayload, ...]
    total_score: float
    first_rest_distance: int | None
    first_shop_distance: int | None
    first_elite_distance: int | None
    first_event_distance: int | None
    first_treasure_distance: int | None
    rest_count: int
    elite_count: int
    shop_count: int
    event_count: int
    treasure_count: int
    monster_count: int
    elites_before_rest: int
    shops_before_boss: int
    rests_before_boss: int
    remaining_distance_to_boss: int | None
    reason_tags: tuple[str, ...] = ()
    community_prior_score_bonus: float = 0.0
    community_prior_contributions: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class _PlannerState:
    energy: int
    remaining_hand_indices: tuple[int, ...]
    used_potion_indices: tuple[int, ...]
    enemy_hp_by_index: dict[int, int]
    depth_remaining: int
    node_budget: int


def build_policy_config(profile_name: str) -> SimplePolicyConfig:
    normalized = profile_name.strip().lower()
    if normalized == "baseline":
        return SimplePolicyConfig(profile_name="baseline", combat_strategy="heuristic")
    if normalized == "legacy":
        return SimplePolicyConfig(
            profile_name="legacy",
            combat_strategy="heuristic",
            strategic_bias=False,
            route_strategy="legacy_node",
        )
    if normalized == "planner":
        return SimplePolicyConfig(
            profile_name="planner",
            combat_strategy="planner",
            planner_depth=3,
            planner_width=5,
            planner_node_budget=32,
        )
    if normalized == "planner_assist":
        return SimplePolicyConfig(
            profile_name="planner_assist",
            combat_strategy="planner_assist",
            planner_depth=2,
            planner_width=4,
            planner_node_budget=24,
            planner_min_score_to_commit=0.35,
        )
    if normalized == "conservative":
        return SimplePolicyConfig(
            profile_name="conservative",
            combat_strategy="heuristic",
            skip_reward_card_below_score=1.10,
            low_hp_ratio=0.52,
            critical_hp_ratio=0.35,
            elite_hp_ratio=0.82,
            upgrade_hp_ratio=0.70,
            buy_shop_card_above_score=1.60,
            buy_shop_potion_above_score=1.45,
            route_planner_discount=0.84,
        )
    raise ValueError(f"Unknown policy profile: {profile_name}")


def build_policy_pack(
    profile_name: str = "baseline",
    *,
    predictor_config: PredictorRuntimeConfig | None = None,
    community_prior_config: CommunityPriorRuntimeConfig | None = None,
    strategic_model_config: StrategicRuntimeConfig | None = None,
) -> SimplePolicy:
    config = build_policy_config(profile_name)
    if predictor_config is not None:
        config = replace(config, predictor_config=predictor_config)
    if community_prior_config is not None:
        config = replace(config, community_prior_config=community_prior_config)
    if strategic_model_config is not None:
        config = replace(config, strategic_model_config=strategic_model_config)
    return SimplePolicy(config=config)


class SimplePolicy:
    def __init__(self, config: SimplePolicyConfig | None = None) -> None:
        self._config = config or SimplePolicyConfig()
        self._predictor = PredictorRuntimeAdapter.from_config(self._config.predictor_config)
        self._community_prior = CommunityCardPriorSource.from_config(self._config.community_prior_config)
        self._strategic = StrategicRuntimeAdapter.from_config(self._config.strategic_model_config)
        self._previous_screen_type: str | None = None
        self._previous_action_name: str | None = None
        self._selection_transaction: _SelectionTransactionState | None = None
        self._selection_transaction_serial = 0
        self.name = f"policy-pack:{self._config.profile_name}"
        if self._strategic is not None:
            self.name = f"{self.name}+{self._strategic.checkpoint_algorithm}"
        self.handler_name = f"{self._config.profile_name}-policy-pack"

    def choose(self, observation: StepObservation) -> PolicyDecision:
        if not observation.legal_actions:
            return self._finalize_decision(
                PolicyDecision(action=None, stage=observation.screen_type.lower(), reason="no_legal_actions"),
                observation=observation,
            )

        if self._has_pending_reward_card_choice(observation):
            return self._finalize_decision(self._choose_reward_action(observation), observation=observation)

        screen = observation.screen_type
        if screen == "CHARACTER_SELECT":
            return self._finalize_decision(self._choose_character_select_action(observation), observation=observation)
        if screen == "CUSTOM_RUN":
            return self._finalize_decision(self._choose_custom_run_action(observation), observation=observation)
        if screen == "MODAL":
            return self._finalize_decision(self._choose_modal_action(observation), observation=observation)
        if screen == "EVENT":
            return self._finalize_decision(self._choose_event_action(observation), observation=observation)
        if screen == "REWARD":
            return self._finalize_decision(self._choose_reward_action(observation), observation=observation)
        if screen in {"SELECTION", "CARD_SELECTION"}:
            return self._finalize_decision(self._choose_selection_action(observation), observation=observation)
        if screen == "SHOP":
            return self._finalize_decision(self._choose_shop_action(observation), observation=observation)
        if screen == "CHEST":
            return self._finalize_decision(self._choose_chest_action(observation), observation=observation)
        if screen == "COMBAT":
            return self._finalize_decision(self._choose_combat_action(observation), observation=observation)
        if screen == "MAP":
            return self._finalize_decision(self._choose_map_action(observation), observation=observation)
        if screen == "REST":
            return self._finalize_decision(self._choose_rest_action(observation), observation=observation)
        if screen == "GAME_OVER":
            if self._config.auto_recover_game_over:
                return self._finalize_decision(self._choose_game_over_action(observation), observation=observation)
            return self._finalize_decision(
                PolicyDecision(action=None, stage="game_over", reason="session_should_stop"),
                observation=observation,
            )

        return self._finalize_decision(self._choose_default_action(observation), observation=observation)

    def choose_action(self, observation: StepObservation) -> CandidateAction | None:
        return self.choose(observation).action

    def _choose_default_action(self, observation: StepObservation) -> PolicyDecision:
        action = (
            self._find_first(
                observation,
                "dismiss_modal",
                "confirm_modal",
                "open_character_select",
                "continue_run",
                "return_to_main_menu",
                "embark",
                "confirm_timeline_overlay",
                "choose_timeline_epoch",
                "choose_map_node",
                "choose_rest_option",
                "choose_event_option",
                "open_chest",
                "choose_treasure_relic",
                "claim_reward",
                "choose_reward_card",
                "skip_reward_cards",
                "collect_rewards_and_proceed",
                "select_deck_card",
                "confirm_selection",
                "buy_relic",
                "buy_card",
                "buy_potion",
                "remove_card_at_shop",
                "close_shop_inventory",
                "proceed",
                "play_card",
                "use_potion",
                "end_turn",
                "close_main_menu_submenu",
            )
            or observation.legal_actions[0]
        )
        return PolicyDecision(action=action, stage="default", reason="fallback_priority_order")

    def _choose_character_select_action(self, observation: StepObservation) -> PolicyDecision:
        state = observation.state.character_select
        if state is None:
            return self._choose_default_action(observation)

        selected_character_id = state.selected_character_id
        if selected_character_id:
            embark = self._find_first(observation, "embark")
            if embark is not None:
                return PolicyDecision(action=embark, stage="character_select", reason="embark_selected_character")

        for character_id in self._config.preferred_character_ids:
            for option in state.characters:
                if option.is_locked or option.character_id != character_id:
                    continue
                if option.is_selected:
                    embark = self._find_first(observation, "embark")
                    if embark is not None:
                        return PolicyDecision(
                            action=embark,
                            stage="character_select",
                            reason=f"embark_preferred_character:{character_id}",
                        )
                candidate = self._find_indexed_action(observation, "select_character", option.index)
                if candidate is not None:
                    return PolicyDecision(
                        action=candidate,
                        stage="character_select",
                        reason=f"select_preferred_character:{character_id}",
                    )

        return self._choose_default_action(observation)

    def _choose_custom_run_action(self, observation: StepObservation) -> PolicyDecision:
        state = observation.state.custom_run
        if state is None:
            return self._choose_default_action(observation)

        if state.selected_character_id:
            embark = self._find_first(observation, "embark")
            if embark is not None and state.can_embark:
                return PolicyDecision(action=embark, stage="custom_run", reason="embark_selected_character")

        for character_id in self._config.preferred_character_ids:
            for option in state.characters:
                if option.is_locked or option.character_id != character_id:
                    continue
                if option.is_selected:
                    embark = self._find_first(observation, "embark")
                    if embark is not None and state.can_embark:
                        return PolicyDecision(
                            action=embark,
                            stage="custom_run",
                            reason=f"embark_preferred_character:{character_id}",
                        )
                candidate = self._find_indexed_action(observation, "select_character", option.index)
                if candidate is not None:
                    return PolicyDecision(
                        action=candidate,
                        stage="custom_run",
                        reason=f"select_preferred_character:{character_id}",
                    )

        return self._choose_default_action(observation)

    def _choose_modal_action(self, observation: StepObservation) -> PolicyDecision:
        modal = observation.state.modal
        if modal is not None:
            if modal.type_name == "NAcceptTutorialsFtue":
                action = self._find_first(observation, "confirm_modal", "dismiss_modal")
                return PolicyDecision(action=action, stage="modal", reason="accept_ftue_modal")
            if modal.confirm_label == "YesButton" and modal.dismiss_label == "NoButton":
                action = self._find_first(observation, "confirm_modal", "dismiss_modal")
                return PolicyDecision(action=action, stage="modal", reason="confirm_yes_no_modal")

        action = self._find_first(observation, "dismiss_modal", "confirm_modal")
        return PolicyDecision(action=action, stage="modal", reason="dismiss_non_critical_modal")

    def _choose_event_action(self, observation: StepObservation) -> PolicyDecision:
        event = observation.state.event
        if event is None:
            return self._choose_default_action(observation)

        context = self._build_run_context(observation)
        indexed = {option.index: option for option in event.options if not option.is_locked}
        candidates = [
            candidate
            for candidate in observation.legal_actions
            if candidate.action == "choose_event_option" and candidate.request.option_index in indexed
        ]
        if not candidates:
            return self._choose_default_action(observation)

        scored: list[tuple[CandidateAction, float, str]] = []
        for candidate in candidates:
            option = indexed[candidate.request.option_index]
            score = self._event_option_score(option, context=context)
            reason = self._event_option_reason(option)
            scored.append((candidate, score, reason))

        guided = self._guide_candidates(
            observation,
            hook="event",
            scored_candidates=[
                {
                    "action": action,
                    "heuristic_score": score,
                    "reason": reason,
                    "payload": {"option": indexed[action.request.option_index]},
                    "strategic_candidate_id": event_candidate_id(observation, action),
                }
                for action, score, reason in scored
            ],
            strategic_context={
                "decision_type": "event_choice",
                "room_type": "event",
                "map_point_type": "event",
                "source_type": "event",
            },
        )
        return self._decision_from_guided(guided, stage="event")

    def _choose_reward_action(self, observation: StepObservation) -> PolicyDecision:
        reward = observation.state.reward
        if reward is None:
            return self._choose_default_action(observation)

        context = self._build_run_context(observation)
        if reward.pending_card_choice:
            card_candidates = [
                candidate
                for candidate in observation.legal_actions
                if candidate.action in {"choose_reward_card", "select_deck_card"}
            ]
            if card_candidates:
                scored_cards: list[tuple[CandidateAction, float, dict[str, Any], dict[str, Any]]] = []
                reward_by_index = {card.index: card for card in reward.card_options}
                selection_by_index = {
                    card.index: card for card in (observation.state.selection.cards if observation.state.selection else [])
                }
                for candidate in card_candidates:
                    option = reward_by_index.get(candidate.request.option_index)
                    if option is not None:
                        score = self._reward_card_score(option, context=context)
                        prior_bonus, prior_metadata = self._community_prior_bonus(
                            domain="reward_pick",
                            card_id=option.card_id,
                            context=context,
                        )
                        score += prior_bonus
                        payload = {"card": option, "selection_mode": "pick"}
                        metadata = {} if prior_metadata is None else {"community_prior": prior_metadata}
                    else:
                        selection_card = selection_by_index.get(candidate.request.option_index)
                        if selection_card is None:
                            continue
                        score = self._selection_pick_score(selection_card, context=context)
                        prior_bonus, prior_metadata = self._community_prior_bonus(
                            domain="selection_pick",
                            card_id=selection_card.card_id,
                            context=context,
                        )
                        score += prior_bonus
                        payload = {"card": selection_card, "selection_mode": "pick"}
                        metadata = {} if prior_metadata is None else {"community_prior": prior_metadata}
                    scored_cards.append((candidate, score, payload, metadata))
                skip = self._find_first(observation, "skip_reward_cards")
                threshold = self._reward_skip_threshold(context)
                if skip is not None:
                    scored_cards.append(
                        (
                            skip,
                            threshold - 1e-6,
                            {"skip_threshold": threshold},
                            {},
                        )
                    )

                if scored_cards:
                    reward_room_type, reward_map_point_type = reward_runtime_domain(observation)
                    guided = self._guide_candidates(
                        observation,
                        hook="reward",
                        scored_candidates=[
                            {
                                "action": action,
                                "heuristic_score": score,
                                "reason": "skip_low_value_reward_card" if action.action == "skip_reward_cards" else "take_reward_card",
                                "payload": payload,
                                "metadata": metadata,
                                "strategic_candidate_id": reward_candidate_id(observation, action),
                            }
                            for action, score, payload, metadata in scored_cards
                        ],
                        strategic_context={
                            "decision_type": "reward_card_pick",
                            "room_type": reward_room_type,
                            "map_point_type": reward_map_point_type,
                            "source_type": "reward",
                        },
                    )
                    return self._decision_from_guided(guided, stage="reward")

        claim_candidates = [candidate for candidate in observation.legal_actions if candidate.action == "claim_reward"]
        if claim_candidates and reward.rewards:
            reward_by_index = {item.index: item for item in reward.rewards}
            scored_claims: list[tuple[CandidateAction, float, str]] = []
            for candidate in claim_candidates:
                option = reward_by_index.get(candidate.request.option_index)
                if option is None:
                    continue
                reward_type = (option.reward_type or "").lower()
                score = self._reward_claim_score(option.reward_type, option.description, context=context)
                reason = {
                    "relic": "claim_reward_relic",
                    "gold": "claim_reward_gold",
                    "card": "claim_reward_card",
                    "potion": "claim_reward_potion",
                }.get(reward_type, "claim_reward_best_value")
                scored_claims.append((candidate, score, reason, {"reward": option}))

            if scored_claims:
                guided = self._guide_candidates(
                    observation,
                    hook="reward",
                    scored_candidates=[
                        {
                            "action": action,
                            "heuristic_score": score,
                            "reason": reason,
                            "payload": payload,
                        }
                        for action, score, reason, payload in scored_claims
                    ],
                )
                return self._decision_from_guided(guided, stage="reward")

        action = self._find_first(observation, "collect_rewards_and_proceed", "proceed", "skip_reward_cards")
        return PolicyDecision(action=action, stage="reward", reason="collect_claimed_rewards")

    def _choose_selection_action(self, observation: StepObservation) -> PolicyDecision:
        selection = observation.state.selection
        if selection is None:
            self._clear_selection_transaction()
            return self._choose_default_action(observation)

        reward = observation.state.reward
        if reward is not None and reward.pending_card_choice:
            self._clear_selection_transaction()
            return self._choose_reward_action(observation)

        if selection.selection_family != "deck":
            self._clear_selection_transaction()
            return PolicyDecision(action=None, stage="selection", reason=f"unsupported_selection_family:{selection.selection_family or 'unknown'}")

        mode = self._selection_mode(selection)
        if not mode:
            self._clear_selection_transaction()
            return PolicyDecision(action=None, stage="selection", reason="missing_selection_semantic_mode")
        if mode not in {"pick", "remove", "upgrade", "transform"}:
            self._clear_selection_transaction()
            return PolicyDecision(action=None, stage="selection", reason=f"unsupported_selection_mode:{mode}")
        if not str(selection.source_type or "").strip():
            self._clear_selection_transaction()
            return PolicyDecision(action=None, stage="selection", reason="missing_selection_source_type")

        required_count = self._selection_required_count(selection)
        guided = self._rank_selection_candidates(observation, selection=selection, mode=mode)
        runtime_completed_count = self._selection_runtime_completed_count(selection, required_count=required_count)

        transaction = self._prepare_selection_transaction(
            observation,
            selection=selection,
            mode=mode,
            required_count=required_count,
            guided=guided,
            runtime_completed_count=runtime_completed_count,
        )

        if transaction is None:
            metadata = {
                "selection_transaction": {
                    "phase": "diverged",
                    "semantic_mode": mode,
                    "source_type": selection.source_type,
                    "source_room_type": selection.source_room_type,
                    "required_count": required_count,
                    "selected_count": selection.selected_count,
                    "remaining_count": selection.remaining_count,
                    "runtime_completed_count": runtime_completed_count,
                }
            }
            return PolicyDecision(action=None, stage="selection", reason="selection_transaction_diverged", trace_metadata=metadata)

        if runtime_completed_count >= required_count:
            transaction.phase = "confirming" if selection.can_confirm else "completed"
            metadata = self._selection_transaction_trace_metadata(
                guided,
                transaction=transaction,
                selection=selection,
                phase=transaction.phase,
            )
            if selection.can_confirm:
                confirm = self._find_first(observation, "confirm_selection")
                if confirm is None:
                    return PolicyDecision(
                        action=None,
                        stage="selection",
                        reason="selection_transaction_missing_confirm_action",
                        trace_metadata=metadata,
                    )
                return PolicyDecision(
                    action=confirm,
                    stage="selection",
                    reason="confirm_required_selection",
                    trace_metadata=metadata,
                )
            return PolicyDecision(
                action=None,
                stage="selection",
                reason="selection_transaction_completed",
                trace_metadata=metadata,
            )

        next_member = self._next_selection_plan_member(transaction)
        if next_member is None:
            metadata = self._selection_transaction_trace_metadata(
                guided,
                transaction=transaction,
                selection=selection,
                phase="diverged",
                divergence_reason="missing_next_selection_plan_member",
            )
            return PolicyDecision(action=None, stage="selection", reason="selection_transaction_diverged", trace_metadata=metadata)

        candidate = self._find_indexed_action(observation, "select_deck_card", next_member.option_index)
        if candidate is None:
            self._replan_selection_transaction_remaining(transaction, guided=guided)
            next_member = self._next_selection_plan_member(transaction)
            if next_member is None:
                metadata = self._selection_transaction_trace_metadata(
                    guided,
                    transaction=transaction,
                    selection=selection,
                    phase="diverged",
                    divergence_reason="selection_candidate_missing_after_replan",
                )
                return PolicyDecision(action=None, stage="selection", reason="selection_transaction_diverged", trace_metadata=metadata)
            candidate = self._find_indexed_action(observation, "select_deck_card", next_member.option_index)
            if candidate is None:
                metadata = self._selection_transaction_trace_metadata(
                    guided,
                    transaction=transaction,
                    selection=selection,
                    phase="diverged",
                    divergence_reason="selection_candidate_not_legal",
                )
                return PolicyDecision(action=None, stage="selection", reason="selection_transaction_diverged", trace_metadata=metadata)

        transaction.pending_option_index = next_member.option_index
        transaction.phase = "selecting"
        return PolicyDecision(
            action=candidate,
            stage="selection",
            reason=self._selection_reason_for_mode(mode),
            score=next_member.final_score,
            trace_metadata=self._selection_transaction_trace_metadata(
                guided,
                transaction=transaction,
                selection=selection,
                phase="selecting",
                next_member=next_member,
            ),
        )

    def _rank_selection_candidates(
        self,
        observation: StepObservation,
        *,
        selection: SelectionPayload,
        mode: str,
    ) -> list[_GuidedCandidate]:
        card_actions = [candidate for candidate in observation.legal_actions if candidate.action == "select_deck_card"]
        if not card_actions:
            return []

        context = self._build_run_context(observation)
        card_by_index = {card.index: card for card in selection.cards}
        scored_candidates: list[dict[str, Any]] = []
        reason = self._selection_reason_for_mode(mode)
        strategic_context = None
        selection_decision_type = selection_decision_type_for_mode(mode)
        if selection_decision_type is not None and self._strategic is not None and self._strategic.enabled_for("selection"):
            strategic_context = {
                "decision_type": selection_decision_type,
                "room_type": self._normalized_selection_room_type(selection),
                "map_point_type": self._normalized_selection_map_point_type(selection),
                "source_type": self._normalized_selection_source_type(selection),
            }

        for candidate in card_actions:
            card = card_by_index.get(candidate.request.option_index)
            if card is None:
                continue
            score = self._selection_score_for_mode(mode, card=card, context=context)
            metadata: dict[str, Any] = {}
            prior_domain = self._selection_prior_domain(mode)
            if prior_domain is not None:
                prior_bonus, prior_metadata = self._community_prior_bonus(
                    domain=prior_domain,
                    card_id=card.card_id,
                    context=context,
                )
                score += prior_bonus
                if prior_metadata is not None:
                    metadata["community_prior"] = prior_metadata
            metadata["selection_card"] = card
            scored_candidates.append(
                {
                    "action": candidate,
                    "heuristic_score": score,
                    "reason": reason,
                    "payload": {"card": card, "selection_mode": mode},
                    "metadata": metadata,
                    "strategic_candidate_id": selection_candidate_id(observation, candidate),
                }
            )

        if not scored_candidates:
            return []
        return self._guide_candidates(
            observation,
            hook="selection",
            scored_candidates=scored_candidates,
            strategic_context=strategic_context,
        )

    def _prepare_selection_transaction(
        self,
        observation: StepObservation,
        *,
        selection: SelectionPayload,
        mode: str,
        required_count: int,
        guided: list[_GuidedCandidate],
        runtime_completed_count: int,
    ) -> _SelectionTransactionState | None:
        if not self._selection_transaction_matches(observation, selection=selection, mode=mode, required_count=required_count):
            self._selection_transaction = self._new_selection_transaction(
                observation,
                selection=selection,
                mode=mode,
                required_count=required_count,
                guided=guided,
            )
        transaction = self._selection_transaction
        if transaction is None:
            return None

        self._reconcile_selection_transaction(
            transaction,
            selection=selection,
            guided=guided,
            runtime_completed_count=runtime_completed_count,
        )
        if len(transaction.planned_members) < required_count:
            self._replan_selection_transaction_remaining(transaction, guided=guided)
        if len(transaction.planned_members) < required_count:
            return None

        transaction.observed_selected_count = selection.selected_count
        transaction.observed_remaining_count = selection.remaining_count
        return transaction

    def _selection_transaction_matches(
        self,
        observation: StepObservation,
        *,
        selection: SelectionPayload,
        mode: str,
        required_count: int,
    ) -> bool:
        transaction = self._selection_transaction
        if transaction is None:
            return False
        return (
            transaction.run_id == observation.run_id
            and transaction.screen_type == observation.screen_type
            and transaction.selection_kind == selection.kind
            and transaction.selection_family == selection.selection_family
            and transaction.semantic_mode == mode
            and transaction.source_type == selection.source_type
            and transaction.source_room_type == selection.source_room_type
            and transaction.prompt_loc_table == selection.prompt_loc_table
            and transaction.prompt_loc_key == selection.prompt_loc_key
            and transaction.required_count == required_count
        )

    def _new_selection_transaction(
        self,
        observation: StepObservation,
        *,
        selection: SelectionPayload,
        mode: str,
        required_count: int,
        guided: list[_GuidedCandidate],
    ) -> _SelectionTransactionState:
        self._selection_transaction_serial += 1
        transaction = _SelectionTransactionState(
            transaction_id=f"{observation.run_id}:{mode}:{selection.source_type}:{self._selection_transaction_serial}",
            run_id=observation.run_id,
            screen_type=observation.screen_type,
            selection_kind=selection.kind,
            selection_family=selection.selection_family,
            semantic_mode=mode,
            source_type=selection.source_type,
            source_room_type=selection.source_room_type,
            prompt_loc_table=selection.prompt_loc_table,
            prompt_loc_key=selection.prompt_loc_key,
            required_count=required_count,
            requires_confirmation=selection.requires_confirmation,
            supports_multi_select=selection.supports_multi_select,
        )
        self._selection_transaction = transaction
        self._replan_selection_transaction_remaining(transaction, guided=guided)
        return transaction

    def _reconcile_selection_transaction(
        self,
        transaction: _SelectionTransactionState,
        *,
        selection: SelectionPayload,
        guided: list[_GuidedCandidate],
        runtime_completed_count: int,
    ) -> None:
        if transaction.pending_option_index is not None and runtime_completed_count > len(transaction.completed_option_indices):
            if transaction.pending_option_index not in transaction.completed_option_indices:
                transaction.completed_option_indices.append(transaction.pending_option_index)
            transaction.pending_option_index = None

        if runtime_completed_count < len(transaction.completed_option_indices):
            transaction.completed_option_indices = transaction.completed_option_indices[:runtime_completed_count]
            transaction.phase = "recovered"
            transaction.recovery_count += 1
            transaction.recovery_reasons.append("runtime_selected_count_receded")
        elif runtime_completed_count > len(transaction.completed_option_indices):
            missing = runtime_completed_count - len(transaction.completed_option_indices)
            current_option_indices = {
                item.action.request.option_index
                for item in guided
                if item.action.request.option_index is not None
            }
            inferred = self._infer_selection_completed_option_indices(
                transaction,
                current_option_indices=current_option_indices,
                count=missing,
            )
            for option_index in inferred:
                if option_index not in transaction.completed_option_indices:
                    transaction.completed_option_indices.append(option_index)
            if inferred:
                transaction.phase = "recovered"
                transaction.recovery_count += 1
                transaction.recovery_reasons.append("runtime_selected_count_advanced")

        remaining_needed = max(0, transaction.required_count - len(transaction.completed_option_indices))
        available_remaining = [
            member for member in transaction.planned_members if member.option_index not in transaction.completed_option_indices
        ]
        if len(available_remaining) < remaining_needed:
            self._replan_selection_transaction_remaining(transaction, guided=guided)
            available_remaining = [
                member for member in transaction.planned_members if member.option_index not in transaction.completed_option_indices
            ]
            if len(available_remaining) < remaining_needed:
                transaction.phase = "diverged"

        transaction.observed_selected_count = selection.selected_count
        transaction.observed_remaining_count = selection.remaining_count

    def _infer_selection_completed_option_indices(
        self,
        transaction: _SelectionTransactionState,
        *,
        current_option_indices: set[int],
        count: int,
    ) -> list[int]:
        preferred: list[int] = []
        if (
            transaction.pending_option_index is not None
            and transaction.pending_option_index not in transaction.completed_option_indices
        ):
            preferred.append(transaction.pending_option_index)
        remaining = [
            member.option_index
            for member in transaction.planned_members
            if member.option_index not in transaction.completed_option_indices and member.option_index not in preferred
        ]
        missing_first = [option_index for option_index in remaining if option_index not in current_option_indices]
        still_visible = [option_index for option_index in remaining if option_index in current_option_indices]
        ordered = [*preferred, *missing_first, *still_visible]
        return ordered[:count]

    def _replan_selection_transaction_remaining(
        self,
        transaction: _SelectionTransactionState,
        *,
        guided: list[_GuidedCandidate],
    ) -> None:
        previous_plan = [member.option_index for member in transaction.planned_members]
        completed_lookup = set(transaction.completed_option_indices)
        existing_by_index = {member.option_index: member for member in transaction.planned_members}
        rebuilt: list[_SelectionPlanMember] = []
        for option_index in transaction.completed_option_indices:
            member = existing_by_index.get(option_index)
            if member is None:
                member = _SelectionPlanMember(
                    option_index=option_index,
                    card_id=f"option_{option_index}",
                    name=f"Selected option {option_index}",
                    heuristic_score=0.0,
                    final_score=0.0,
                )
            rebuilt.append(member)
        for item in guided:
            option_index = item.action.request.option_index
            if option_index is None or option_index in completed_lookup:
                continue
            card = item.trace_metadata.get("selection_card")
            payload_card = card if isinstance(card, SelectionCardPayload) else None
            member = _SelectionPlanMember(
                option_index=option_index,
                card_id=(payload_card.card_id if isinstance(payload_card, SelectionCardPayload) else f"option_{option_index}"),
                name=(payload_card.name if isinstance(payload_card, SelectionCardPayload) else item.action.label),
                heuristic_score=item.heuristic_score,
                final_score=item.final_score,
            )
            rebuilt.append(member)
            if len(rebuilt) >= transaction.required_count:
                break
        transaction.planned_members = rebuilt
        if previous_plan and previous_plan != [member.option_index for member in rebuilt]:
            transaction.recovery_count += 1
            transaction.recovery_reasons.append("planned_bundle_rebuilt")
            transaction.phase = "replanned"

    def _next_selection_plan_member(self, transaction: _SelectionTransactionState) -> _SelectionPlanMember | None:
        completed_lookup = set(transaction.completed_option_indices)
        for member in transaction.planned_members:
            if member.option_index not in completed_lookup:
                return member
        return None

    def _selection_transaction_trace_metadata(
        self,
        guided: list[_GuidedCandidate],
        *,
        transaction: _SelectionTransactionState,
        selection: SelectionPayload,
        phase: str,
        next_member: _SelectionPlanMember | None = None,
        divergence_reason: str | None = None,
    ) -> dict[str, Any]:
        metadata = self._decision_trace_metadata(guided) if guided else {}
        planned: list[dict[str, Any]] = []
        current_option_indices = {
            item.action.request.option_index
            for item in guided
            if item.action.request.option_index is not None
        }
        completed_lookup = set(transaction.completed_option_indices)
        for member in transaction.planned_members:
            planned.append(
                {
                    "option_index": member.option_index,
                    "card_id": member.card_id,
                    "name": member.name,
                    "heuristic_score": member.heuristic_score,
                    "final_score": member.final_score,
                    "completed": member.option_index in completed_lookup,
                    "pending": member.option_index == transaction.pending_option_index,
                    "legal": member.option_index in current_option_indices,
                }
            )
        payload: dict[str, Any] = {
            "transaction_id": transaction.transaction_id,
            "phase": phase,
            "selection_kind": transaction.selection_kind,
            "selection_family": transaction.selection_family,
            "semantic_mode": transaction.semantic_mode,
            "source_type": transaction.source_type,
            "source_room_type": transaction.source_room_type,
            "required_count": transaction.required_count,
            "selected_count": selection.selected_count,
            "remaining_count": selection.remaining_count,
            "runtime_completed_count": self._selection_runtime_completed_count(
                selection,
                required_count=transaction.required_count,
            ),
            "requires_confirmation": selection.requires_confirmation,
            "can_confirm": selection.can_confirm,
            "supports_multi_select": selection.supports_multi_select,
            "planned_members": planned,
            "planned_option_indices": [member.option_index for member in transaction.planned_members],
            "planned_card_ids": [member.card_id for member in transaction.planned_members],
            "completed_option_indices": list(transaction.completed_option_indices),
            "completed_card_ids": [
                member.card_id for member in transaction.planned_members if member.option_index in completed_lookup
            ],
            "recovery_count": transaction.recovery_count,
            "recovery_reasons": list(transaction.recovery_reasons),
        }
        if next_member is not None:
            payload["next_option_index"] = next_member.option_index
            payload["next_card_id"] = next_member.card_id
        if divergence_reason is not None:
            payload["divergence_reason"] = divergence_reason
        metadata["selection_transaction"] = payload
        return metadata

    def _clear_selection_transaction(self) -> None:
        self._selection_transaction = None

    def _choose_shop_action(self, observation: StepObservation) -> PolicyDecision:
        state = observation.state.shop
        run = observation.state.run
        if state is None or run is None:
            return self._choose_default_action(observation)

        context = self._build_run_context(observation)
        if not state.is_open:
            open_inventory = self._find_first(observation, "open_shop_inventory")
            if open_inventory is not None and self._shop_should_open_inventory(state, context=context):
                return PolicyDecision(action=open_inventory, stage="shop", reason="open_shop_inventory")
            proceed = self._find_first(observation, "proceed")
            if proceed is not None:
                return PolicyDecision(action=proceed, stage="shop", reason="leave_shop_preserve_gold")

        scored: list[tuple[CandidateAction, float, str, dict[str, Any], dict[str, Any]]] = []
        removal = self._find_first(observation, "remove_card_at_shop")
        if removal is not None and state.card_removal is not None and state.card_removal.available and state.card_removal.enough_gold:
            removal_score = self._shop_removal_score(state.card_removal.price, context=context)
            scored.append((removal, removal_score, "buy_card_removal", {"card_removal": state.card_removal}, {}))

        relic_by_index = {relic.index: relic for relic in state.relics}
        card_by_index = {card.index: card for card in state.cards}
        potion_by_index = {potion.index: potion for potion in state.potions}
        for candidate in observation.legal_actions:
            if candidate.action == "buy_relic":
                relic = relic_by_index.get(candidate.request.option_index)
                if relic is not None:
                    scored.append((candidate, self._shop_relic_score(relic, context=context), "buy_shop_relic", {"relic": relic}, {}))
            elif candidate.action == "buy_card":
                card = card_by_index.get(candidate.request.option_index)
                if card is not None:
                    score = self._shop_card_score(card, context=context)
                    prior_bonus, prior_metadata = self._community_prior_bonus(
                        domain="shop_buy",
                        card_id=card.card_id,
                        context=context,
                    )
                    score += prior_bonus
                    scored.append(
                        (
                            candidate,
                            score,
                            "buy_shop_card",
                            {"card": card},
                            {} if prior_metadata is None else {"community_prior": prior_metadata},
                        )
                    )
            elif candidate.action == "buy_potion":
                potion = potion_by_index.get(candidate.request.option_index)
                if potion is not None:
                    scored.append(
                        (
                            candidate,
                            self._shop_potion_score(potion, context=context),
                            "buy_shop_potion",
                            {"potion": potion},
                            {},
                        )
                    )

        if scored:
            guided = self._guide_candidates(
                observation,
                hook="shop",
                scored_candidates=[
                    {
                        "action": action,
                        "heuristic_score": score,
                        "reason": reason,
                        "payload": payload,
                        "metadata": metadata,
                        "strategic_candidate_id": (
                            shop_buy_candidate_id(observation, action) if action.action == "buy_card" else None
                        ),
                    }
                    for action, score, reason, payload, metadata in scored
                ],
                strategic_context={
                    "decision_type": "shop_buy",
                    "room_type": "shop",
                    "map_point_type": "shop",
                    "source_type": "shop",
                },
            )
            best = guided[0]
            if self._shop_purchase_allowed(best.action, best.final_score):
                return self._decision_from_guided(guided, stage="shop")

        action = self._find_first(observation, "close_shop_inventory", "proceed", "open_shop_inventory")
        return PolicyDecision(action=action, stage="shop", reason="leave_shop_preserve_gold")

    def _shop_should_open_inventory(self, state, *, context: RunContext) -> bool:
        if not state.can_open:
            return False
        if state.card_removal is not None and state.card_removal.available and state.card_removal.enough_gold:
            removal_score = self._shop_removal_score(state.card_removal.price, context=context)
            if removal_score >= 0.85:
                return True
        for relic in state.relics:
            if relic.is_stocked and relic.enough_gold and self._shop_purchase_allowed_by_action(
                "buy_relic",
                self._shop_relic_score(relic, context=context)
            ):
                return True
        for card in state.cards:
            if card.is_stocked and card.enough_gold and self._shop_purchase_allowed_by_action(
                "buy_card",
                self._shop_card_score(card, context=context),
            ):
                return True
        for potion in state.potions:
            if potion.is_stocked and potion.enough_gold and self._shop_purchase_allowed_by_action(
                "buy_potion",
                self._shop_potion_score(potion, context=context),
            ):
                return True
        return False

    def _shop_purchase_allowed_by_action(self, action: str, score: float) -> bool:
        if action == "buy_card":
            return score >= self._config.buy_shop_card_above_score
        if action == "buy_potion":
            return score >= self._config.buy_shop_potion_above_score
        return score >= 0.85

    def _choose_chest_action(self, observation: StepObservation) -> PolicyDecision:
        chest = observation.state.chest
        if chest is None:
            return self._choose_default_action(observation)

        if chest.has_relic_been_claimed:
            action = self._find_first(observation, "proceed", "open_chest")
            return PolicyDecision(action=action, stage="chest", reason="leave_opened_chest")

        if not chest.is_opened:
            action = self._find_first(observation, "open_chest", "proceed")
            return PolicyDecision(action=action, stage="chest", reason="open_chest_then_continue")

        choose_relic = [candidate for candidate in observation.legal_actions if candidate.action == "choose_treasure_relic"]
        if choose_relic and chest.relic_options:
            relic_by_index = {relic.index: relic for relic in chest.relic_options}
            action, best_score = self._pick_best(
                choose_relic,
                lambda candidate: _RARITY_SCORES.get(
                    (relic_by_index.get(candidate.request.option_index).rarity or "").lower(),
                    0.0,
                ),
            )
            return PolicyDecision(action=action, stage="chest", reason="take_best_treasure_relic", score=best_score)

        action = self._find_first(observation, "proceed", "open_chest")
        return PolicyDecision(action=action, stage="chest", reason="leave_opened_chest")

    def _choose_map_action(self, observation: StepObservation) -> PolicyDecision:
        state_map = observation.state.map
        if state_map is None:
            return self._choose_default_action(observation)

        context = self._build_run_context(observation)
        node_by_index = {node.index: node for node in state_map.available_nodes}
        candidates = [candidate for candidate in observation.legal_actions if candidate.action == "choose_map_node"]
        if not candidates:
            return self._choose_default_action(observation)

        if self._config.route_strategy == "legacy_node" or not state_map.nodes:
            return self._choose_legacy_map_action(observation, context=context, candidates=candidates, node_by_index=node_by_index)

        route_plans = self._plan_map_routes(observation, context=context, candidates=candidates, node_by_index=node_by_index)
        if not route_plans:
            return self._choose_legacy_map_action(observation, context=context, candidates=candidates, node_by_index=node_by_index)

        best_plan_by_action: dict[str, RoutePlan] = {}
        for plan in route_plans:
            existing = best_plan_by_action.get(plan.action.action_id)
            if existing is None or plan.total_score > existing.total_score:
                best_plan_by_action[plan.action.action_id] = plan

        ranked_plans = sorted(
            best_plan_by_action.values(),
            key=lambda item: (item.total_score, -self._candidate_sort_key(item.action)),
            reverse=True,
        )
        guided = self._guide_candidates(
            observation,
            hook="map",
            scored_candidates=[
                {
                    "action": plan.action,
                    "heuristic_score": plan.total_score,
                    "reason": self._route_plan_reason(plan, context=context),
                    "payload": {
                        "node": plan.first_node,
                        "route_plan": self._route_plan_payload(plan),
                    },
                    "metadata": {} if self._route_plan_prior_payload(plan) is None else {"community_prior": self._route_plan_prior_payload(plan)},
                }
                for plan in ranked_plans
            ],
        )
        selected = guided[0]
        selected_plan = best_plan_by_action[selected.action.action_id]
        ranked_actions = tuple(
            RankedAction(
                action_id=plan.action.action_id,
                action=plan.action.action,
                score=plan.total_score,
                reason=self._route_plan_reason(plan, context=context),
                metadata=self._route_plan_payload(plan),
            )
            for plan in ranked_plans[: self._config.route_planner_width]
        )
        trace_metadata = self._decision_trace_metadata(guided)
        trace_metadata["route_planner"] = self._route_planner_metadata(
            selected=selected_plan,
            ranked_plans=ranked_plans,
            context=context,
        )
        return PolicyDecision(
            action=selected.action,
            stage="map",
            reason=selected.reason,
            score=selected.final_score,
            planner_name=self._config.route_planner_name,
            planner_strategy=self._config.route_strategy,
            ranked_actions=ranked_actions,
            trace_metadata=trace_metadata,
        )

    def _choose_legacy_map_action(
        self,
        observation: StepObservation,
        *,
        context: RunContext,
        candidates: list[CandidateAction],
        node_by_index: dict[int, MapNodePayload],
    ) -> PolicyDecision:
        scored_nodes: list[tuple[CandidateAction, float, str]] = []
        for candidate in candidates:
            node = node_by_index.get(candidate.request.option_index)
            if node is None:
                continue
            score = self._map_node_score(node, context=context)
            route_prior_bonus, route_prior_metadata = self._community_route_prior_bonus(
                _map_node_kind(node.node_type),
                context=context,
            )
            score += route_prior_bonus
            reason = self._map_node_reason(node, context=context)
            scored_nodes.append(
                (
                    candidate,
                    score,
                    reason,
                    {"node": node},
                    {} if route_prior_metadata is None else {"community_prior": route_prior_metadata},
                )
            )

        guided = self._guide_candidates(
            observation,
            hook="map",
            scored_candidates=[
                {
                    "action": action,
                    "heuristic_score": score,
                    "reason": reason,
                    "payload": payload,
                    "metadata": metadata,
                }
                for action, score, reason, payload, metadata in scored_nodes
            ],
        )
        return self._decision_from_guided(guided, stage="map")

    def _plan_map_routes(
        self,
        observation: StepObservation,
        *,
        context: RunContext,
        candidates: list[CandidateAction],
        node_by_index: dict[int, MapNodePayload],
    ) -> list[RoutePlan]:
        state_map = observation.state.map
        if state_map is None:
            return []

        graph_by_coord = {(node.row, node.col): node for node in state_map.nodes}
        route_plans: list[RoutePlan] = []
        for candidate in candidates:
            first_node = node_by_index.get(candidate.request.option_index)
            if first_node is None:
                continue
            graph_node = graph_by_coord.get((first_node.row, first_node.col))
            if graph_node is None:
                continue
            prefixes = self._enumerate_route_prefixes(graph_node, graph_by_coord)
            for prefix in prefixes:
                route_plans.append(self._score_route_prefix(candidate, first_node, prefix, context=context))
        return route_plans

    def _enumerate_route_prefixes(
        self,
        start_node: MapGraphNodePayload,
        graph_by_coord: dict[tuple[int, int], MapGraphNodePayload],
    ) -> list[tuple[MapGraphNodePayload, ...]]:
        prefixes: list[tuple[MapGraphNodePayload, ...]] = []

        def dfs(node: MapGraphNodePayload, path: list[MapGraphNodePayload], depth_remaining: int) -> None:
            path.append(node)
            if depth_remaining <= 1 or not node.children or node.is_boss or node.is_second_boss:
                prefixes.append(tuple(path))
            else:
                extended = False
                for child in node.children:
                    child_node = graph_by_coord.get((child.row, child.col))
                    if child_node is None:
                        continue
                    dfs(child_node, path, depth_remaining - 1)
                    extended = True
                if not extended:
                    prefixes.append(tuple(path))
            path.pop()

        dfs(start_node, [], self._config.route_planner_depth)
        return prefixes

    def _score_route_prefix(
        self,
        action: CandidateAction,
        first_node: MapNodePayload,
        prefix: tuple[MapGraphNodePayload, ...],
        *,
        context: RunContext,
    ) -> RoutePlan:
        total_score = 0.0
        first_rest_distance: int | None = None
        first_shop_distance: int | None = None
        first_elite_distance: int | None = None
        first_event_distance: int | None = None
        first_treasure_distance: int | None = None
        rest_count = 0
        elite_count = 0
        shop_count = 0
        event_count = 0
        treasure_count = 0
        monster_count = 0
        elites_before_rest = 0
        shops_before_boss = 0
        rests_before_boss = 0
        seen_rest = False
        reason_tags: list[str] = []
        community_prior_score_bonus = 0.0
        community_prior_contributions: list[dict[str, Any]] = []

        for step_index, node in enumerate(prefix):
            kind = _map_node_kind(node.node_type)
            discount = self._config.route_planner_discount ** step_index
            total_score += discount * self._map_graph_node_score(
                node,
                context=context,
                step_index=step_index,
            )
            route_prior_bonus, route_prior_metadata = self._community_route_prior_bonus(kind, context=context)
            if route_prior_metadata is not None:
                discounted_bonus = discount * route_prior_bonus
                total_score += discounted_bonus
                community_prior_score_bonus += discounted_bonus
                community_prior_contributions.append(
                    {
                        **route_prior_metadata,
                        "step_index": step_index,
                        "node_type": node.node_type,
                        "discounted_score_bonus": discounted_bonus,
                    }
                )

            if kind == "rest":
                rest_count += 1
                first_rest_distance = step_index if first_rest_distance is None else first_rest_distance
                seen_rest = True
            elif kind == "elite":
                elite_count += 1
                first_elite_distance = step_index if first_elite_distance is None else first_elite_distance
                if not seen_rest:
                    elites_before_rest += 1
            elif kind == "shop":
                shop_count += 1
                first_shop_distance = step_index if first_shop_distance is None else first_shop_distance
                shops_before_boss += 1
            elif kind == "event":
                event_count += 1
                first_event_distance = step_index if first_event_distance is None else first_event_distance
            elif kind == "treasure":
                treasure_count += 1
                first_treasure_distance = step_index if first_treasure_distance is None else first_treasure_distance
            elif kind == "monster":
                monster_count += 1

            if kind == "rest":
                rests_before_boss += 1

        total_score += self._route_structural_bonus(
            context=context,
            first_rest_distance=first_rest_distance,
            first_shop_distance=first_shop_distance,
            first_elite_distance=first_elite_distance,
            elite_count=elite_count,
            shop_count=shop_count,
            event_count=event_count,
            treasure_count=treasure_count,
            monster_count=monster_count,
            elites_before_rest=elites_before_rest,
            reason_tags=reason_tags,
        )
        remaining_distance_to_boss = context.current_to_boss_distance - len(prefix) if context.current_to_boss_distance is not None else None

        return RoutePlan(
            action=action,
            first_node=first_node,
            prefix=prefix,
            total_score=total_score,
            first_rest_distance=first_rest_distance,
            first_shop_distance=first_shop_distance,
            first_elite_distance=first_elite_distance,
            first_event_distance=first_event_distance,
            first_treasure_distance=first_treasure_distance,
            rest_count=rest_count,
            elite_count=elite_count,
            shop_count=shop_count,
            event_count=event_count,
            treasure_count=treasure_count,
            monster_count=monster_count,
            elites_before_rest=elites_before_rest,
            shops_before_boss=shops_before_boss,
            rests_before_boss=rests_before_boss,
            remaining_distance_to_boss=remaining_distance_to_boss,
            reason_tags=tuple(reason_tags),
            community_prior_score_bonus=community_prior_score_bonus,
            community_prior_contributions=tuple(community_prior_contributions),
        )

    def _map_graph_node_score(self, node: MapGraphNodePayload, *, context: RunContext, step_index: int) -> float:
        base = self._map_node_score(
            MapNodePayload(index=step_index, row=node.row, col=node.col, node_type=node.node_type, state=node.state),
            context=context,
        )
        kind = _map_node_kind(node.node_type)
        if step_index > 0:
            base -= 0.10 * step_index
        if kind == "rest" and context.boss_profile.upgrade_urgency > 0.9 and context.hp_ratio >= 0.52:
            base += 0.65
        if kind == "shop" and context.gold >= self._config.preferred_gold_for_shop:
            base += 0.40 * context.boss_profile.shop_urgency
        if kind == "event" and self._boss_gap(context, "aoe") > 0.0:
            base += 0.35
        if kind == "elite" and self._boss_gap(context, "sustain") > 0.0 and context.hp_ratio <= 0.70:
            base -= 1.10 * self._boss_gap(context, "sustain")
        return base

    def _route_structural_bonus(
        self,
        *,
        context: RunContext,
        first_rest_distance: int | None,
        first_shop_distance: int | None,
        first_elite_distance: int | None,
        elite_count: int,
        shop_count: int,
        event_count: int,
        treasure_count: int,
        monster_count: int,
        elites_before_rest: int,
        reason_tags: list[str],
    ) -> float:
        bonus = 0.0
        aoe_gap = self._boss_gap(context, "aoe")
        scaling_gap = self._boss_gap(context, "scaling")
        frontload_gap = self._boss_gap(context, "frontload")

        if context.hp_ratio <= self._config.low_hp_ratio:
            if first_rest_distance is not None:
                bonus += max(0.0, 3.20 - (0.75 * first_rest_distance))
                reason_tags.append("early_rest_for_survival")
            bonus -= 1.45 * elites_before_rest
        elif first_rest_distance is not None and scaling_gap > 0.0 and context.hp_ratio >= 0.58:
            bonus += 0.80 * context.boss_profile.upgrade_urgency / max(1, first_rest_distance + 1)
            reason_tags.append("upgrade_window_before_boss")

        if first_shop_distance is not None and context.gold >= self._config.preferred_gold_for_shop:
            bonus += max(0.0, (1.00 + context.boss_profile.shop_urgency) - (0.35 * first_shop_distance))
            reason_tags.append("shop_access_before_boss")

        if aoe_gap > 0.0:
            bonus += (0.70 * shop_count) + (0.35 * event_count) + (0.20 * monster_count)
            if elite_count > 0 and first_shop_distance is None:
                bonus -= 0.85 * aoe_gap
            reason_tags.append("search_aoe_tools")

        if scaling_gap > 0.0:
            bonus += (0.45 * treasure_count) + (0.35 * event_count)
            if first_rest_distance is not None:
                bonus += 0.25 * context.boss_profile.upgrade_urgency
            reason_tags.append("prepare_scaling_boss")

        if frontload_gap > 0.0:
            bonus += 0.22 * monster_count
            if first_elite_distance == 0 and context.hp_ratio >= self._config.elite_hp_ratio:
                bonus += 0.40
            reason_tags.append("frontload_pressure")

        return bonus

    def _route_plan_reason(self, plan: RoutePlan, *, context: RunContext) -> str:
        if "early_rest_for_survival" in plan.reason_tags:
            return "route_plan_rest_before_risk"
        if "shop_access_before_boss" in plan.reason_tags and self._boss_gap(context, "aoe") > 0.0:
            return "route_plan_shop_for_boss_tools"
        if "prepare_scaling_boss" in plan.reason_tags and self._boss_gap(context, "scaling") > 0.0:
            return "route_plan_upgrade_for_scaling_boss"
        if "search_aoe_tools" in plan.reason_tags:
            return "route_plan_search_aoe"
        if plan.first_node is not None and _map_node_kind(plan.first_node.node_type) == "elite":
            return "route_plan_elite_progression"
        return "route_plan_best_prefix"

    def _route_plan_payload(self, plan: RoutePlan) -> dict[str, Any]:
        return {
            "first_node": {
                "row": plan.first_node.row,
                "col": plan.first_node.col,
                "node_type": plan.first_node.node_type,
            },
            "path": [{"row": node.row, "col": node.col, "node_type": node.node_type} for node in plan.prefix],
            "path_node_types": [node.node_type for node in plan.prefix],
            "first_rest_distance": plan.first_rest_distance,
            "first_shop_distance": plan.first_shop_distance,
            "first_elite_distance": plan.first_elite_distance,
            "first_event_distance": plan.first_event_distance,
            "first_treasure_distance": plan.first_treasure_distance,
            "rest_count": plan.rest_count,
            "elite_count": plan.elite_count,
            "shop_count": plan.shop_count,
            "event_count": plan.event_count,
            "treasure_count": plan.treasure_count,
            "monster_count": plan.monster_count,
            "elites_before_rest": plan.elites_before_rest,
            "shops_before_boss": plan.shops_before_boss,
            "rests_before_boss": plan.rests_before_boss,
            "remaining_distance_to_boss": plan.remaining_distance_to_boss,
            "reason_tags": list(plan.reason_tags),
            "community_prior_score_bonus": plan.community_prior_score_bonus,
            "community_prior_contributions": [dict(item) for item in plan.community_prior_contributions],
        }

    def _route_planner_metadata(
        self,
        *,
        selected: RoutePlan,
        ranked_plans: list[RoutePlan],
        context: RunContext,
    ) -> dict[str, Any]:
        return {
            "planner_name": self._config.route_planner_name,
            "planner_strategy": self._config.route_strategy,
            "boss_encounter_id": context.boss_encounter_id,
            "boss_encounter_name": context.boss_encounter_name,
            "boss_notes": list(context.boss_profile.notes),
            "selected": {
                "action_id": selected.action.action_id,
                "score": selected.total_score,
                **self._route_plan_payload(selected),
            },
            "ranked_paths": [
                {
                    "action_id": plan.action.action_id,
                    "score": plan.total_score,
                    **self._route_plan_payload(plan),
                }
                for plan in ranked_plans[: self._config.route_planner_width]
            ],
        }

    def _route_plan_prior_payload(self, plan: RoutePlan) -> dict[str, Any] | None:
        if not plan.community_prior_contributions:
            return None
        first = plan.community_prior_contributions[0]
        confidences = [
            float(item["confidence"])
            for item in plan.community_prior_contributions
            if item.get("confidence") is not None
        ]
        return {
            "domain": "map_node",
            "subject_id": first.get("subject_id") or _map_node_kind(plan.first_node.node_type),
            "room_type": first.get("room_type") or _map_node_kind(plan.first_node.node_type),
            "score_bonus": plan.community_prior_score_bonus,
            "confidence": (sum(confidences) / len(confidences)) if confidences else None,
            "source_name": first.get("source_name"),
            "snapshot_date": first.get("snapshot_date"),
            "artifact_family": first.get("artifact_family"),
            "metadata": {
                "contribution_count": len(plan.community_prior_contributions),
                "contributions": [dict(item) for item in plan.community_prior_contributions],
            },
        }

    def _choose_rest_action(self, observation: StepObservation) -> PolicyDecision:
        rest = observation.state.rest
        if rest is None:
            return self._choose_default_action(observation)

        context = self._build_run_context(observation)
        options = {option.index: option for option in rest.options}
        candidates = [candidate for candidate in observation.legal_actions if candidate.action == "choose_rest_option"]
        if not candidates:
            return self._choose_default_action(observation)

        scored: list[tuple[CandidateAction, float, str]] = []
        for candidate in candidates:
            option = options.get(candidate.request.option_index)
            if option is None:
                continue
            score = self._rest_option_score(option, context=context)
            reason = self._rest_option_reason(option, context=context)
            scored.append((candidate, score, reason, {"option": option}))

        guided = self._guide_candidates(
            observation,
            hook="rest",
            scored_candidates=[
                {
                    "action": action,
                    "heuristic_score": score,
                    "reason": reason,
                    "payload": payload,
                    "strategic_candidate_id": rest_candidate_id(observation, action),
                }
                for action, score, reason, payload in scored
            ],
            strategic_context={
                "decision_type": "rest_site_action",
                "room_type": "rest",
                "map_point_type": "rest",
                "source_type": "rest",
            },
        )
        return self._decision_from_guided(guided, stage="rest")

    def _choose_combat_action(self, observation: StepObservation) -> PolicyDecision:
        if self._config.combat_strategy in {"planner", "planner_assist"}:
            planner_decision = self._choose_planned_combat_action(observation)
            if planner_decision is not None:
                if self._config.combat_strategy == "planner":
                    return planner_decision
                if planner_decision.score is not None and planner_decision.score >= self._config.planner_min_score_to_commit:
                    return planner_decision

        playable = [candidate for candidate in observation.legal_actions if candidate.action == "play_card"]
        if playable:
            guided = self._guide_candidates(
                observation,
                hook="combat",
                scored_candidates=[
                    {
                        "action": candidate,
                        "heuristic_score": self._combat_card_score(observation, candidate),
                        "reason": "play_highest_value_card",
                        "payload": {},
                    }
                    for candidate in playable
                ],
            )
            return self._decision_from_guided(guided, stage="combat")

        potions = [candidate for candidate in observation.legal_actions if candidate.action == "use_potion"]
        if potions:
            guided = self._guide_candidates(
                observation,
                hook="combat",
                scored_candidates=[
                    {
                        "action": candidate,
                        "heuristic_score": self._combat_potion_score(observation, candidate),
                        "reason": "use_high_value_potion",
                        "payload": {},
                    }
                    for candidate in potions
                ],
            )
            if guided[0].final_score > 0.75:
                return self._decision_from_guided(guided, stage="combat")

        action = self._find_first(observation, "end_turn")
        return PolicyDecision(action=action, stage="combat", reason="end_turn_without_better_play")

    def _choose_game_over_action(self, observation: StepObservation) -> PolicyDecision:
        game_over = observation.state.game_over
        if game_over is not None:
            if game_over.can_return_to_main_menu or game_over.showing_summary:
                action = self._find_first(observation, "return_to_main_menu", "continue_run")
                return PolicyDecision(action=action, stage="game_over", reason="return_to_main_menu_after_game_over")
            if game_over.can_continue:
                action = self._find_first(observation, "continue_run", "return_to_main_menu")
                return PolicyDecision(action=action, stage="game_over", reason="continue_game_over_flow")

        action = self._find_first(observation, "continue_run", "return_to_main_menu")
        return PolicyDecision(action=action, stage="game_over", reason="advance_game_over_screen")

    def _decorate_decision(self, decision: PolicyDecision, *, observation: StepObservation) -> PolicyDecision:
        handler_name = decision.policy_handler or self._policy_handler_name(decision.stage, observation.screen_type)
        return PolicyDecision(
            action=decision.action,
            stage=decision.stage,
            reason=decision.reason,
            score=decision.score,
            policy_pack=decision.policy_pack or self._config.profile_name,
            policy_handler=handler_name,
            planner_name=decision.planner_name,
            planner_strategy=decision.planner_strategy,
            ranked_actions=decision.ranked_actions,
            trace_metadata=dict(decision.trace_metadata),
        )

    def _finalize_decision(self, decision: PolicyDecision, *, observation: StepObservation) -> PolicyDecision:
        decorated = self._decorate_decision(decision, observation=observation)
        if decorated.stage != "selection":
            self._clear_selection_transaction()
        elif decorated.reason in {"selection_transaction_completed", "selection_transaction_diverged"}:
            self._clear_selection_transaction()
        self._previous_screen_type = observation.screen_type
        self._previous_action_name = None if decorated.action is None else decorated.action.action
        return decorated

    def _policy_handler_name(self, stage: str, screen_type: str) -> str:
        normalized_screen = screen_type.strip().lower()
        if stage == "combat":
            return "combat-hand-planner" if self._config.combat_strategy != "heuristic" else "combat-greedy-ranker"
        if stage in {"selection", "reward", "shop", "map", "rest", "event", "character_select", "modal", "chest", "game_over"}:
            return f"{stage}-handler"
        if normalized_screen:
            return f"{normalized_screen}-handler"
        return "default-handler"

    def _find_first(self, observation: StepObservation, *actions: str) -> CandidateAction | None:
        for action in actions:
            for candidate in observation.legal_actions:
                if candidate.action == action:
                    return candidate
        return None

    def _find_indexed_action(
        self,
        observation: StepObservation,
        action_name: str,
        option_index: int,
    ) -> CandidateAction | None:
        for candidate in observation.legal_actions:
            if candidate.action != action_name:
                continue
            if candidate.request.option_index == option_index:
                return candidate
        return None

    def _decision_from_guided(self, guided: list[_GuidedCandidate], *, stage: str) -> PolicyDecision:
        best = guided[0]
        return PolicyDecision(
            action=best.action,
            stage=stage,
            reason=best.reason,
            score=best.final_score,
            trace_metadata=self._decision_trace_metadata(guided),
        )

    def _predictor_skip_metadata(
        self,
        guided: list[_GuidedCandidate],
        *,
        hook: str,
    ) -> dict[str, Any]:
        if not guided:
            return {}
        metadata = self._decision_trace_metadata(guided)
        metadata["predictor_skip"] = {
            "hook": hook,
            "selected_reason": guided[0].reason,
            "selected_final_score": guided[0].final_score,
        }
        return metadata

    def _normalized_selection_source_type(self, selection: SelectionPayload) -> str:
        source_type = str(selection.source_type or "").strip().lower()
        return source_type or "selection"

    def _normalized_selection_room_type(self, selection: SelectionPayload) -> str:
        room_type = str(selection.source_room_type or "").strip().lower()
        if room_type:
            return room_type
        return self._normalized_selection_source_type(selection)

    def _normalized_selection_map_point_type(self, selection: SelectionPayload) -> str:
        return self._normalized_selection_source_type(selection)

    def _guide_candidates(
        self,
        observation: StepObservation,
        *,
        hook: str,
        scored_candidates: list[dict[str, Any]],
        strategic_context: dict[str, Any] | None = None,
    ) -> list[_GuidedCandidate]:
        strategic_traces_by_action_id: dict[str, StrategicRuntimeTrace] = {}
        if self._strategic is not None and strategic_context is not None and self._strategic.enabled_for(hook):
            candidate_ids_by_action_id = {
                item["action"].action_id: str(item["strategic_candidate_id"])
                for item in scored_candidates
                if item.get("strategic_candidate_id") is not None
            }
            if candidate_ids_by_action_id:
                strategic_traces = self._strategic.score_candidates(
                    observation=observation,
                    hook=hook,
                    decision_type=str(strategic_context["decision_type"]),
                    candidate_ids=tuple(candidate_ids_by_action_id.values()),
                    room_type=str(strategic_context["room_type"]),
                    map_point_type=str(strategic_context["map_point_type"]),
                    source_type=str(strategic_context["source_type"]),
                )
                strategic_traces_by_action_id = {
                    action_id: strategic_traces[candidate_id]
                    for action_id, candidate_id in candidate_ids_by_action_id.items()
                    if candidate_id in strategic_traces
                }
        guided: list[_GuidedCandidate] = []
        for item in scored_candidates:
            action = item["action"]
            heuristic_score = float(item["heuristic_score"])
            reason = str(item["reason"])
            payload = dict(item.get("payload", {}))
            trace_metadata = dict(item.get("metadata", {}))
            predictor_trace: dict[str, Any] | None = None
            strategic_runtime_trace = strategic_traces_by_action_id.get(action.action_id)
            strategic_trace = None if strategic_runtime_trace is None else strategic_runtime_trace.as_dict()
            final_score = heuristic_score
            if self._predictor is not None and self._predictor.enabled_for(hook):
                projected_summary = self._project_summary_for_candidate(
                    observation,
                    hook=hook,
                    action=action,
                    payload=payload,
                )
                runtime_trace = self._predictor.score_summary(projected_summary, hook=hook)
                predictor_trace = runtime_trace.as_dict(include_feature_map=True)
                final_score = self._predictor.blend(final_score, runtime_trace)
            if strategic_runtime_trace is not None and self._strategic is not None:
                final_score = self._strategic.blend(final_score, strategic_runtime_trace)
            guided.append(
                _GuidedCandidate(
                    action=action,
                    reason=reason,
                    heuristic_score=heuristic_score,
                    final_score=final_score,
                    predictor_trace=predictor_trace,
                    strategic_trace=strategic_trace,
                    trace_metadata=trace_metadata,
                )
            )
        guided.sort(key=lambda item: (item.final_score, -self._candidate_sort_key(item.action)), reverse=True)
        return guided

    def _decision_trace_metadata(self, guided: list[_GuidedCandidate]) -> dict[str, Any]:
        metadata = self._predictor_decision_metadata(guided)
        strategic_metadata = self._strategic_decision_metadata(guided)
        if strategic_metadata:
            metadata["strategic"] = strategic_metadata
        community_metadata = self._community_prior_decision_metadata(guided)
        if community_metadata:
            metadata["community_prior"] = community_metadata
        return metadata

    def _predictor_decision_metadata(self, guided: list[_GuidedCandidate]) -> dict[str, Any]:
        if self._predictor is None or not guided or not guided[0].predictor_trace:
            return {}
        selected = guided[0]
        ranked: list[dict[str, Any]] = []
        for item in guided[:3]:
            candidate_payload = {
                "action_id": item.action.action_id,
                "action": item.action.action,
                "reason": item.reason,
                "heuristic_score": item.heuristic_score,
                "final_score": item.final_score,
            }
            if item.predictor_trace is not None:
                candidate_payload["predictor"] = {
                    key: value
                    for key, value in item.predictor_trace.items()
                    if key != "feature_map"
                }
            ranked.append(candidate_payload)
        return {
            "predictor": {
                **self._predictor.runtime_payload(),
                "domain": guided[0].predictor_trace.get("hook"),
                "selected": {
                    "action_id": selected.action.action_id,
                    "action": selected.action.action,
                    "reason": selected.reason,
                    "heuristic_score": selected.heuristic_score,
                    "final_score": selected.final_score,
                    **selected.predictor_trace,
                },
                "ranked_candidates": ranked,
            }
        }

    def _strategic_decision_metadata(self, guided: list[_GuidedCandidate]) -> dict[str, Any]:
        if self._strategic is None or not guided or not guided[0].strategic_trace:
            return {}
        selected = guided[0]
        ranked: list[dict[str, Any]] = []
        for item in guided[:3]:
            if item.strategic_trace is None:
                continue
            ranked.append(
                {
                    "action_id": item.action.action_id,
                    "action": item.action.action,
                    "reason": item.reason,
                    "heuristic_score": item.heuristic_score,
                    "final_score": item.final_score,
                    "strategic": dict(item.strategic_trace),
                }
            )
        return {
            **self._strategic.runtime_payload(),
            "domain": selected.strategic_trace.get("hook"),
            "selected": {
                "action_id": selected.action.action_id,
                "action": selected.action.action,
                "reason": selected.reason,
                "heuristic_score": selected.heuristic_score,
                "final_score": selected.final_score,
                **selected.strategic_trace,
            },
            "ranked_candidates": ranked,
        }

    def _community_prior_decision_metadata(self, guided: list[_GuidedCandidate]) -> dict[str, Any]:
        if self._community_prior is None or not guided:
            return {}
        selected_prior = guided[0].trace_metadata.get("community_prior")
        ranked_candidates: list[dict[str, Any]] = []
        for item in guided[:3]:
            prior_payload = item.trace_metadata.get("community_prior")
            if not isinstance(prior_payload, dict):
                continue
            ranked_candidates.append(
                {
                    "action_id": item.action.action_id,
                    "action": item.action.action,
                    "reason": item.reason,
                    "heuristic_score": item.heuristic_score,
                    "final_score": item.final_score,
                    "prior": prior_payload,
                }
            )
        if not isinstance(selected_prior, dict) and not ranked_candidates:
            return {}
        return {
            "config": self._community_prior.config.as_dict(),
            "selected": {
                "action_id": guided[0].action.action_id,
                "action": guided[0].action.action,
                "reason": guided[0].reason,
                "heuristic_score": guided[0].heuristic_score,
                "final_score": guided[0].final_score,
                "prior": selected_prior,
            },
            "ranked_candidates": ranked_candidates,
        }

    def _project_summary_for_candidate(
        self,
        observation: StepObservation,
        *,
        hook: str,
        action: CandidateAction,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        summary = deepcopy(build_state_summary(observation))
        run_summary = summary.setdefault("run", {})
        combat_summary = summary.get("combat")

        if hook == "map":
            node = payload.get("node")
            if isinstance(node, MapNodePayload):
                self._project_map_node_summary(summary, node=node, route_plan=payload.get("route_plan"))
        elif hook == "reward":
            card = payload.get("card")
            reward = payload.get("reward")
            if isinstance(card, RewardCardOptionPayload):
                self._add_card_to_run_summary(run_summary, card.card_id)
            elif isinstance(card, SelectionCardPayload):
                self._add_card_to_run_summary(run_summary, card.card_id)
            elif isinstance(reward, RewardOptionPayload):
                self._project_reward_claim_summary(run_summary, reward=reward)
        elif hook == "selection":
            card = payload.get("card")
            mode = str(payload.get("selection_mode", "pick"))
            if isinstance(card, SelectionCardPayload):
                self._project_selection_summary(run_summary, card=card, mode=mode)
        elif hook == "shop":
            if card := payload.get("card"):
                if isinstance(card, ShopCardPayload):
                    run_summary["gold"] = max(0, int(run_summary.get("gold", 0) or 0) - card.price)
                    self._add_card_to_run_summary(run_summary, card.card_id)
            elif relic := payload.get("relic"):
                if isinstance(relic, ShopRelicPayload):
                    run_summary["gold"] = max(0, int(run_summary.get("gold", 0) or 0) - relic.price)
                    run_summary["relic_count"] = int(run_summary.get("relic_count", 0) or 0) + 1
                    text = _normalized_text(relic.relic_id, relic.name, relic.rarity)
                    if "energy" in text:
                        run_summary["max_energy"] = int(run_summary.get("max_energy", 0) or 0) + 1
                    if "hp" in text:
                        run_summary["max_hp"] = int(run_summary.get("max_hp", 0) or 0) + 5
            elif potion := payload.get("potion"):
                if isinstance(potion, ShopPotionPayload):
                    run_summary["gold"] = max(0, int(run_summary.get("gold", 0) or 0) - potion.price)
                    occupied = int(run_summary.get("occupied_potions", 0) or 0)
                    run_summary["occupied_potions"] = min(3, occupied + 1)
            elif card_removal := payload.get("card_removal"):
                price = int(getattr(card_removal, "price", 0) or 0)
                run_summary["gold"] = max(0, int(run_summary.get("gold", 0) or 0) - price)
                self._remove_generic_card_from_run_summary(run_summary)
        elif hook == "rest":
            option = payload.get("option")
            if isinstance(option, RestOptionPayload):
                text = _normalized_text(option.option_id, option.title, option.description)
                if _contains_any(text, _REST_FRAGMENTS):
                    self._heal_run_summary(run_summary, fraction=0.30, minimum=12)
                elif _contains_any(text, _UPGRADE_FRAGMENTS):
                    run_summary["deck_size"] = int(run_summary.get("deck_size", 0) or 0)
                elif _contains_any(text, _REMOVE_FRAGMENTS):
                    self._remove_generic_card_from_run_summary(run_summary)
        elif hook == "event":
            option = payload.get("option")
            if isinstance(option, EventOptionPayload):
                text = _normalized_text(option.title, option.description)
                if _contains_any(text, _GOLD_FRAGMENTS):
                    run_summary["gold"] = int(run_summary.get("gold", 0) or 0) + 35
                if _contains_any(text, _RELIC_FRAGMENTS):
                    run_summary["relic_count"] = int(run_summary.get("relic_count", 0) or 0) + 1
                if _contains_any(text, _REMOVE_FRAGMENTS):
                    self._remove_generic_card_from_run_summary(run_summary)
                if _contains_any(text, _REST_FRAGMENTS):
                    self._heal_run_summary(run_summary, fraction=0.20, minimum=8)
                if _contains_any(text, _SELF_DAMAGE_FRAGMENTS):
                    self._damage_run_summary(run_summary, fraction=0.12, minimum=6)
        elif hook == "combat" and combat_summary is not None:
            self._project_combat_summary(observation, action=action, summary=summary)

        return summary

    def _project_map_node_summary(
        self,
        summary: dict[str, Any],
        *,
        node: MapNodePayload,
        route_plan: dict[str, Any] | None = None,
    ) -> None:
        run_summary = summary.setdefault("run", {})
        map_summary = summary.setdefault("map", {})
        kind = _map_node_kind(node.node_type)
        run_summary["floor"] = int(run_summary.get("floor", 0) or 0) + 1
        map_summary["available_node_types"] = [node.node_type]
        map_summary["available_node_count"] = 1
        map_summary["travel_enabled"] = True
        map_summary["traveling"] = False
        if isinstance(route_plan, dict):
            map_summary["route_plan"] = route_plan
            map_summary["current_node"] = route_plan.get("first_node")
            map_summary["planned_node_types"] = list(route_plan.get("path_node_types", []))
            map_summary["planned_first_rest_distance"] = route_plan.get("first_rest_distance")
            map_summary["planned_first_shop_distance"] = route_plan.get("first_shop_distance")
            map_summary["planned_first_elite_distance"] = route_plan.get("first_elite_distance")
            map_summary["planned_rest_count"] = route_plan.get("rest_count")
            map_summary["planned_shop_count"] = route_plan.get("shop_count")
            map_summary["planned_elite_count"] = route_plan.get("elite_count")
            map_summary["current_to_boss_distance"] = route_plan.get("remaining_distance_to_boss")
        if kind == "rest":
            self._heal_run_summary(run_summary, fraction=0.22, minimum=10)
        elif kind == "elite":
            self._damage_run_summary(run_summary, fraction=0.18, minimum=12)
            run_summary["gold"] = int(run_summary.get("gold", 0) or 0) + 35
        elif kind == "monster":
            self._damage_run_summary(run_summary, fraction=0.10, minimum=6)
            run_summary["gold"] = int(run_summary.get("gold", 0) or 0) + 18
        elif kind == "shop":
            run_summary["gold"] = int(run_summary.get("gold", 0) or 0)
        elif kind == "treasure":
            run_summary["relic_count"] = int(run_summary.get("relic_count", 0) or 0) + 1
        elif kind == "event":
            run_summary["gold"] = int(run_summary.get("gold", 0) or 0) + 10

    def _project_reward_claim_summary(self, run_summary: dict[str, Any], *, reward: RewardOptionPayload) -> None:
        reward_type = (reward.reward_type or "").lower()
        if reward_type == "gold":
            run_summary["gold"] = int(run_summary.get("gold", 0) or 0) + 30
        elif reward_type == "relic":
            run_summary["relic_count"] = int(run_summary.get("relic_count", 0) or 0) + 1
        elif reward_type == "potion":
            occupied = int(run_summary.get("occupied_potions", 0) or 0)
            run_summary["occupied_potions"] = min(3, occupied + 1)
        elif reward_type == "heal":
            self._heal_run_summary(run_summary, fraction=0.20, minimum=8)

    def _project_selection_summary(
        self,
        run_summary: dict[str, Any],
        *,
        card: SelectionCardPayload,
        mode: str,
    ) -> None:
        if mode == "remove":
            self._mutate_card_in_run_summary(run_summary, card.card_id, delta=-1)
        elif mode == "pick":
            self._mutate_card_in_run_summary(run_summary, card.card_id, delta=1)
        elif mode == "transform":
            self._mutate_card_in_run_summary(run_summary, card.card_id, delta=-1)
            run_summary["deck_size"] = int(run_summary.get("deck_size", 0) or 0) + 1

    def _project_combat_summary(
        self,
        observation: StepObservation,
        *,
        action: CandidateAction,
        summary: dict[str, Any],
    ) -> None:
        combat_summary = summary.get("combat")
        if not isinstance(combat_summary, dict):
            return

        if action.action == "end_turn":
            summary["turn"] = int(summary.get("turn", 0) or 0) + 1
            combat_summary["energy"] = 0
            return

        combat_state = observation.state.combat
        if combat_state is None:
            return

        enemy_position_by_index = {
            enemy.index: position for position, enemy in enumerate(enemy for enemy in combat_state.enemies if enemy.is_alive)
        }
        hand_card_ids = list(combat_summary.get("hand_card_ids", []))
        enemy_hp = [int(value) for value in combat_summary.get("enemy_hp", [])]
        player_block = int(combat_summary.get("player_block", 0) or 0)
        energy = int(combat_summary.get("energy", 0) or 0)

        if action.action == "play_card" and action.request.card_index is not None:
            card = next((item for item in combat_state.hand if item.index == action.request.card_index), None)
            if card is None:
                return
            rules_text = card.resolved_rules_text or card.rules_text
            if card.card_id in hand_card_ids:
                hand_card_ids.remove(card.card_id)
            card_cost = energy if card.costs_x else card.energy_cost
            combat_summary["energy"] = max(0, energy - card_cost)
            if action.request.target_index is not None:
                position = enemy_position_by_index.get(action.request.target_index)
                if position is not None and position < len(enemy_hp):
                    enemy_hp[position] = max(0, enemy_hp[position] - estimate_damage(rules_text))
            combat_summary["enemy_hp"] = enemy_hp
            combat_summary["player_block"] = player_block + estimate_block(rules_text)
            combat_summary["hand_card_ids"] = hand_card_ids
            combat_summary["playable_hand_count"] = max(0, int(combat_summary.get("playable_hand_count", 0) or 0) - 1)
            return

        if action.action == "use_potion":
            run_summary = summary.setdefault("run", {})
            occupied = int(run_summary.get("occupied_potions", 0) or 0)
            run_summary["occupied_potions"] = max(0, occupied - 1)

    def _heal_run_summary(self, run_summary: dict[str, Any], *, fraction: float, minimum: int) -> None:
        current_hp = int(run_summary.get("current_hp", 0) or 0)
        max_hp = max(1, int(run_summary.get("max_hp", 0) or 1))
        amount = max(minimum, int(round(max_hp * fraction)))
        run_summary["current_hp"] = min(max_hp, current_hp + amount)

    def _damage_run_summary(self, run_summary: dict[str, Any], *, fraction: float, minimum: int) -> None:
        current_hp = int(run_summary.get("current_hp", 0) or 0)
        max_hp = max(1, int(run_summary.get("max_hp", 0) or 1))
        amount = max(minimum, int(round(max_hp * fraction)))
        run_summary["current_hp"] = max(0, current_hp - amount)

    def _add_card_to_run_summary(self, run_summary: dict[str, Any], card_id: str) -> None:
        self._mutate_card_in_run_summary(run_summary, card_id, delta=1)

    def _remove_generic_card_from_run_summary(self, run_summary: dict[str, Any]) -> None:
        if int(run_summary.get("curse_count", 0) or 0) > 0:
            run_summary["curse_count"] = int(run_summary.get("curse_count", 0) or 0) - 1
        elif int(run_summary.get("strike_count", 0) or 0) > 0:
            run_summary["strike_count"] = int(run_summary.get("strike_count", 0) or 0) - 1
        elif int(run_summary.get("defend_count", 0) or 0) > 0:
            run_summary["defend_count"] = int(run_summary.get("defend_count", 0) or 0) - 1
        elif int(run_summary.get("status_count", 0) or 0) > 0:
            run_summary["status_count"] = int(run_summary.get("status_count", 0) or 0) - 1
        run_summary["deck_size"] = max(0, int(run_summary.get("deck_size", 0) or 0) - 1)

    def _mutate_card_in_run_summary(self, run_summary: dict[str, Any], card_id: str, *, delta: int) -> None:
        run_summary["deck_size"] = max(0, int(run_summary.get("deck_size", 0) or 0) + delta)
        normalized = card_id.lower()
        if _contains_any(normalized, _CURSE_FRAGMENTS):
            run_summary["curse_count"] = max(0, int(run_summary.get("curse_count", 0) or 0) + delta)
        elif _contains_any(normalized, _STATUS_FRAGMENTS):
            run_summary["status_count"] = max(0, int(run_summary.get("status_count", 0) or 0) + delta)
        elif "strike" in normalized:
            run_summary["strike_count"] = max(0, int(run_summary.get("strike_count", 0) or 0) + delta)
        elif "defend" in normalized:
            run_summary["defend_count"] = max(0, int(run_summary.get("defend_count", 0) or 0) + delta)

    def _pick_best(
        self,
        candidates: Iterable[CandidateAction],
        scorer,
    ) -> tuple[CandidateAction, float]:
        scored = [(candidate, float(scorer(candidate))) for candidate in candidates]
        best_candidate, best_score = max(scored, key=lambda item: (item[1], -self._candidate_sort_key(item[0])))
        return best_candidate, best_score

    def _candidate_sort_key(self, candidate: CandidateAction) -> int:
        request = candidate.request
        if request.option_index is not None:
            return request.option_index
        if request.card_index is not None:
            return request.card_index
        if request.target_index is not None:
            return request.target_index
        return 0

    def _choose_planned_combat_action(self, observation: StepObservation) -> PolicyDecision | None:
        ranked_actions = self._rank_planned_combat_actions(observation)
        if not ranked_actions:
            return None
        top_rank = ranked_actions[0]
        chosen_action = next((candidate for candidate in observation.legal_actions if candidate.action_id == top_rank.action_id), None)
        if chosen_action is None:
            return None
        sequence = top_rank.metadata.get("sequence", [])
        reason = "planner_selected_sequence"
        if top_rank.action == "end_turn":
            reason = "planner_end_turn"
        elif top_rank.action == "use_potion":
            reason = "planner_use_potion"
        return PolicyDecision(
            action=chosen_action,
            stage="combat",
            reason=reason,
            score=top_rank.score,
            policy_handler="combat-hand-planner",
            planner_name=self._config.planner_name,
            planner_strategy=self._config.combat_strategy,
            ranked_actions=tuple(ranked_actions),
            trace_metadata={
                "sequence": sequence,
                "planner_depth": self._config.planner_depth,
                "planner_width": self._config.planner_width,
                "planner_node_budget": self._config.planner_node_budget,
                **(
                    {"predictor": dict(top_rank.metadata["predictor"])}
                    if isinstance(top_rank.metadata.get("predictor"), dict)
                    else {}
                ),
            },
        )

    def _rank_planned_combat_actions(self, observation: StepObservation) -> list[RankedAction]:
        combat = observation.state.combat
        if combat is None:
            return []
        playable_or_end_turn = [
            candidate
            for candidate in observation.legal_actions
            if candidate.action in {"play_card", "use_potion", "end_turn"}
        ]
        if not playable_or_end_turn:
            return []

        enemy_hp_by_index = {enemy.index: enemy.current_hp for enemy in combat.enemies if enemy.is_alive}
        state = _PlannerState(
            energy=combat.player.energy,
            remaining_hand_indices=tuple(card.index for card in combat.hand),
            used_potion_indices=(),
            enemy_hp_by_index=enemy_hp_by_index,
            depth_remaining=max(1, self._config.planner_depth),
            node_budget=max(1, self._config.planner_node_budget),
        )
        evaluations: list[RankedAction] = []
        for candidate in playable_or_end_turn:
            total_score, sequence, immediate_score = self._planner_rollout(
                observation,
                candidate,
                state=state,
            )
            predictor_payload: dict[str, Any] | None = None
            if self._predictor is not None and self._predictor.enabled_for("combat"):
                projected_summary = self._project_summary_for_candidate(
                    observation,
                    hook="combat",
                    action=candidate,
                    payload={},
                )
                predictor_trace = self._predictor.score_summary(projected_summary, hook="combat")
                total_score = self._predictor.blend(total_score, predictor_trace)
                predictor_payload = predictor_trace.as_dict(include_feature_map=False)
            evaluations.append(
                RankedAction(
                    action_id=candidate.action_id,
                    action=candidate.action,
                    score=total_score,
                    reason=self._planner_rank_reason(candidate),
                    metadata={
                        "sequence": sequence,
                        "immediate_score": immediate_score,
                        "projected_score": total_score,
                        **({"predictor": predictor_payload} if predictor_payload is not None else {}),
                    },
                )
            )
        evaluations.sort(key=lambda item: (item.score, -self._planner_sort_key(item.action_id)), reverse=True)
        return evaluations[: self._config.planner_width]

    def _planner_rollout(
        self,
        observation: StepObservation,
        candidate: CandidateAction,
        *,
        state: _PlannerState,
    ) -> tuple[float, list[str], float]:
        immediate_score, next_state = self._planner_apply_action(observation, candidate, state=state)
        sequence = [candidate.action_id]
        if next_state is None or next_state.depth_remaining <= 0 or next_state.node_budget <= 0:
            return immediate_score, sequence, immediate_score

        follow_candidates = self._planner_follow_candidates(observation, state=next_state)
        if not follow_candidates:
            return immediate_score, sequence, immediate_score

        best_follow_score: float | None = None
        best_follow_sequence: list[str] = []
        for follow_candidate in follow_candidates[: self._config.planner_width]:
            follow_score, follow_sequence, _ = self._planner_rollout(
                observation,
                follow_candidate,
                state=next_state,
            )
            if best_follow_score is None or follow_score > best_follow_score:
                best_follow_score = follow_score
                best_follow_sequence = follow_sequence
        if best_follow_score is None:
            return immediate_score, sequence, immediate_score
        total_score = immediate_score + (self._config.planner_discount * best_follow_score)
        return total_score, sequence + best_follow_sequence, immediate_score

    def _planner_follow_candidates(self, observation: StepObservation, *, state: _PlannerState) -> list[CandidateAction]:
        candidates: list[CandidateAction] = []
        for candidate in observation.legal_actions:
            if candidate.action == "play_card":
                if candidate.request.card_index not in state.remaining_hand_indices:
                    continue
                card = next(
                    (item for item in (observation.state.combat.hand if observation.state.combat is not None else []) if item.index == candidate.request.card_index),
                    None,
                )
                if card is None:
                    continue
                effective_cost = state.energy if card.costs_x else card.energy_cost
                if effective_cost > state.energy:
                    continue
                target_index = candidate.request.target_index
                if target_index is not None and state.enemy_hp_by_index.get(target_index, 0) <= 0:
                    continue
                candidates.append(candidate)
            elif candidate.action == "use_potion":
                if candidate.request.option_index in state.used_potion_indices:
                    continue
                target_index = candidate.request.target_index
                if target_index is not None and state.enemy_hp_by_index.get(target_index, 0) <= 0:
                    continue
                candidates.append(candidate)
            elif candidate.action == "end_turn":
                candidates.append(candidate)
        candidates.sort(key=self._candidate_sort_key)
        return candidates

    def _planner_apply_action(
        self,
        observation: StepObservation,
        candidate: CandidateAction,
        *,
        state: _PlannerState,
    ) -> tuple[float, _PlannerState | None]:
        if candidate.action == "end_turn":
            end_turn_score = self._planner_end_turn_score(observation, state=state)
            return end_turn_score, None

        next_enemy_hp = dict(state.enemy_hp_by_index)
        next_hand_indices = tuple(index for index in state.remaining_hand_indices if index != candidate.request.card_index)
        next_used_potions = state.used_potion_indices
        next_energy = state.energy
        score = 0.0

        if candidate.action == "play_card":
            score = self._planner_card_score(observation, candidate, state=state)
            card = next(
                (item for item in (observation.state.combat.hand if observation.state.combat is not None else []) if item.index == candidate.request.card_index),
                None,
            )
            if card is not None:
                damage = estimate_damage(card.resolved_rules_text or card.rules_text)
                if card.costs_x:
                    next_energy = 0
                    damage = max(damage, state.energy * max(1, damage))
                else:
                    next_energy = max(0, state.energy - card.energy_cost)
                next_energy = min(next_energy + summed_keyword_values((card.resolved_rules_text or card.rules_text).lower(), "energy"), observation.state.run.max_energy if observation.state.run is not None else next_energy)
                if candidate.request.target_index is not None and damage > 0:
                    next_enemy_hp[candidate.request.target_index] = max(
                        0,
                        next_enemy_hp.get(candidate.request.target_index, 0) - damage,
                    )
                elif damage > 0 and _contains_any((card.resolved_rules_text or card.rules_text).lower(), ("all enemies", "所有敌人", "全体敌人")):
                    for enemy_index, hp in list(next_enemy_hp.items()):
                        next_enemy_hp[enemy_index] = max(0, hp - damage)
        elif candidate.action == "use_potion":
            score = self._planner_potion_score(observation, candidate, state=state)
            next_used_potions = tuple(sorted((*state.used_potion_indices, candidate.request.option_index or -1)))
            target_index = candidate.request.target_index
            if target_index is not None:
                next_enemy_hp[target_index] = max(0, next_enemy_hp.get(target_index, 0) - 20)

        next_state = _PlannerState(
            energy=next_energy,
            remaining_hand_indices=next_hand_indices,
            used_potion_indices=next_used_potions,
            enemy_hp_by_index=next_enemy_hp,
            depth_remaining=state.depth_remaining - 1,
            node_budget=state.node_budget - 1,
        )
        return score, next_state

    def _planner_card_score(self, observation: StepObservation, candidate: CandidateAction, *, state: _PlannerState) -> float:
        score = self._combat_card_score(observation, candidate)
        combat = observation.state.combat
        if combat is None or candidate.request.card_index is None:
            return score
        card = next((item for item in combat.hand if item.index == candidate.request.card_index), None)
        if card is None:
            return score
        estimated_damage = estimate_damage(card.resolved_rules_text or card.rules_text)
        if candidate.request.target_index is not None:
            target_hp = state.enemy_hp_by_index.get(candidate.request.target_index, 0)
            if estimated_damage >= target_hp > 0:
                score += 3.50
            elif target_hp > 0 and estimated_damage > 0:
                score += min(1.75, estimated_damage / target_hp)
        if estimated_damage == 0 and estimate_block(card.resolved_rules_text or card.rules_text) == 0:
            score -= 0.25
        if card.costs_x and state.energy > 1:
            score += 0.60 * state.energy
        return score

    def _planner_potion_score(self, observation: StepObservation, candidate: CandidateAction, *, state: _PlannerState) -> float:
        score = self._combat_potion_score(observation, candidate)
        if candidate.request.target_index is not None and state.enemy_hp_by_index.get(candidate.request.target_index, 0) <= 15:
            score += 1.25
        return score

    def _planner_end_turn_score(self, observation: StepObservation, *, state: _PlannerState) -> float:
        playable_cards = [
            candidate
            for candidate in self._planner_follow_candidates(observation, state=state)
            if candidate.action == "play_card"
        ]
        if not playable_cards:
            return 0.05
        return -self._config.planner_end_turn_penalty - (0.10 * min(len(playable_cards), 5))

    def _planner_rank_reason(self, candidate: CandidateAction) -> str:
        if candidate.action == "play_card":
            return "ranked_combat_card"
        if candidate.action == "use_potion":
            return "ranked_combat_potion"
        return "ranked_end_turn"

    def _planner_sort_key(self, action_id: str) -> int:
        total = 0
        for character in action_id:
            total = ((total * 131) + ord(character)) % 1_000_003
        return total

    def _has_pending_reward_card_choice(self, observation: StepObservation) -> bool:
        reward = observation.state.reward
        return bool(reward and reward.pending_card_choice)

    def _build_run_context(self, observation: StepObservation) -> RunContext:
        run = observation.state.run
        state_map = observation.state.map
        occupied_potions = len([potion for potion in (run.potions if run is not None else []) if potion.occupied])
        total_slots = len(run.potions) if run is not None and run.potions else 3
        current_hp = run.current_hp if run is not None else 0
        max_hp = run.max_hp if run is not None and run.max_hp > 0 else max(current_hp, 1)
        deck_profile = self._deck_profile(observation)
        boss_profile = self._build_boss_profile(run)
        return RunContext(
            floor=run.floor if run is not None else 0,
            gold=run.gold if run is not None else 0,
            current_hp=current_hp,
            max_hp=max_hp,
            hp_ratio=(current_hp / max_hp) if max_hp > 0 else 1.0,
            max_energy=run.max_energy if run is not None else 0,
            occupied_potions=occupied_potions,
            empty_potion_slots=max(0, total_slots - occupied_potions),
            character_id=run.character_id if run is not None and run.character_id else None,
            ascension=run.ascension if run is not None else None,
            deck=deck_profile,
            act_index=run.act_index if run is not None else None,
            act_number=run.act_number if run is not None else None,
            act_id=run.act_id if run is not None else None,
            act_name=run.act_name if run is not None else None,
            boss_encounter_id=run.boss_encounter.encounter_id if run is not None and run.boss_encounter is not None else None,
            boss_encounter_name=run.boss_encounter.name if run is not None and run.boss_encounter is not None else None,
            boss_profile=boss_profile,
            current_to_boss_distance=self._distance_from_current_to_target(state_map, lambda node: node.is_boss),
            next_rest_distance=self._distance_from_current_to_kind(state_map, "rest"),
            next_shop_distance=self._distance_from_current_to_kind(state_map, "shop"),
            next_elite_distance=self._distance_from_current_to_kind(state_map, "elite"),
            next_event_distance=self._distance_from_current_to_kind(state_map, "event"),
            next_treasure_distance=self._distance_from_current_to_kind(state_map, "treasure"),
            available_map_branch_count=len(state_map.available_nodes) if state_map is not None else 0,
        )

    def _deck_profile(self, observation: StepObservation) -> DeckProfile:
        agent_view = observation.state.agent_view if isinstance(observation.state.agent_view, dict) else {}
        run_view = agent_view.get("run", {}) if isinstance(agent_view.get("run", {}), dict) else {}
        raw_deck = run_view.get("deck", [])
        deck_lines = tuple(line for line in (_agent_view_line(item) for item in raw_deck) if line)
        relic_names = tuple(str(item) for item in run_view.get("relics", []) if item)
        titles = [_card_title_from_line(line) for line in deck_lines]
        strike_count = sum(1 for title in titles if _matches_any_pattern(title, _STRIKE_TITLE_PATTERNS))
        defend_count = sum(1 for title in titles if _matches_any_pattern(title, _DEFEND_TITLE_PATTERNS))
        curse_count = sum(1 for title in titles if _contains_any(title, _CURSE_FRAGMENTS))
        status_count = sum(1 for title in titles if _contains_any(title, _STATUS_FRAGMENTS))
        deck_size = len(deck_lines) if deck_lines else max(10, strike_count + defend_count)
        frontload_score = 0.0
        aoe_score = 0.0
        sustain_score = 0.0
        scaling_score = 0.0
        for line in deck_lines:
            normalized = _normalized_text(line)
            if _contains_any(normalized, _FRONTLOAD_FRAGMENTS):
                frontload_score += 0.80
            if _contains_any(normalized, _AOE_FRAGMENTS):
                aoe_score += 1.20
            if _contains_any(normalized, _SUSTAIN_FRAGMENTS):
                sustain_score += 0.75
            if _contains_any(normalized, _SCALING_FRAGMENTS):
                scaling_score += 0.85
        for relic in relic_names:
            normalized = _normalized_text(relic)
            if _contains_any(normalized, _SUSTAIN_FRAGMENTS):
                sustain_score += 0.50
            if _contains_any(normalized, _SCALING_FRAGMENTS + _ENERGY_FRAGMENTS):
                scaling_score += 0.45
        return DeckProfile(
            deck_lines=deck_lines,
            relic_names=relic_names,
            deck_size=deck_size,
            strike_count=strike_count,
            defend_count=defend_count,
            curse_count=curse_count,
            status_count=status_count,
            frontload_score=frontload_score,
            aoe_score=aoe_score,
            sustain_score=sustain_score,
            scaling_score=scaling_score,
        )

    def _build_boss_profile(self, run) -> BossStrategicProfile:
        if run is None:
            return BossStrategicProfile(
                encounter_id=None,
                encounter_name=None,
                frontload_demand=0.45,
                aoe_demand=0.0,
                sustain_demand=0.35,
                scaling_demand=0.45,
                upgrade_urgency=0.40,
                shop_urgency=0.35,
                notes=("boss_unknown",),
            )

        boss_id = run.boss_encounter.encounter_id if run.boss_encounter is not None else None
        boss_name = run.boss_encounter.name if run.boss_encounter is not None else None
        text = _normalized_text(boss_id, boss_name, run.act_id, run.act_name)
        frontload = 0.45
        aoe = 0.0
        sustain = 0.35
        scaling = 0.45
        upgrade = 0.40
        shop = 0.35
        notes: list[str] = []

        if _contains_any(text, _BOSS_AOE_FRAGMENTS):
            aoe += 1.35
            frontload += 0.45
            shop += 0.45
            notes.append("aoe_boss")
        if _contains_any(text, _BOSS_FRONTLOAD_FRAGMENTS):
            frontload += 0.95
            upgrade += 0.35
            notes.append("frontload_boss")
        if _contains_any(text, _BOSS_SUSTAIN_FRAGMENTS):
            sustain += 1.10
            upgrade += 0.20
            notes.append("sustain_boss")
        if _contains_any(text, _BOSS_SCALING_FRAGMENTS):
            scaling += 1.10
            upgrade += 0.70
            notes.append("scaling_boss")

        return BossStrategicProfile(
            encounter_id=boss_id,
            encounter_name=boss_name,
            frontload_demand=frontload,
            aoe_demand=aoe,
            sustain_demand=sustain,
            scaling_demand=scaling,
            upgrade_urgency=upgrade,
            shop_urgency=shop,
            notes=tuple(notes),
        )

    def _boss_gap(self, context: RunContext, category: Literal["frontload", "aoe", "sustain", "scaling"]) -> float:
        if not self._config.strategic_bias:
            return 0.0
        if category == "frontload":
            return max(0.0, context.boss_profile.frontload_demand - context.deck.frontload_score)
        if category == "aoe":
            return max(0.0, context.boss_profile.aoe_demand - context.deck.aoe_score)
        if category == "sustain":
            return max(0.0, context.boss_profile.sustain_demand - context.deck.sustain_score)
        return max(0.0, context.boss_profile.scaling_demand - context.deck.scaling_score)

    def _distance_from_current_to_kind(self, state_map, kind: str) -> int | None:
        return self._distance_from_current_to_target(state_map, lambda node: _map_node_kind(node.node_type) == kind)

    def _distance_from_current_to_target(self, state_map, predicate) -> int | None:
        if state_map is None or state_map.current_node is None or not state_map.nodes:
            return None
        graph_by_coord = {(node.row, node.col): node for node in state_map.nodes}
        start = (state_map.current_node.row, state_map.current_node.col)
        if start not in graph_by_coord:
            return None

        queue: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
        visited = {start}
        while queue:
            coord, distance = queue.popleft()
            node = graph_by_coord.get(coord)
            if node is None:
                continue
            if distance > 0 and predicate(node):
                return distance
            for child in node.children:
                child_coord = (child.row, child.col)
                if child_coord in visited:
                    continue
                visited.add(child_coord)
                queue.append((child_coord, distance + 1))
        return None

    def _community_prior_bonus(
        self,
        *,
        domain: CommunityPriorDomain,
        card_id: str,
        context: RunContext,
    ) -> tuple[float, dict[str, Any] | None]:
        if self._community_prior is None:
            return 0.0, None
        prior = self._community_prior.score_card(
            domain=domain,
            card_id=card_id,
            character_id=context.character_id,
            ascension=context.ascension,
            act_id=context.act_id,
            floor=context.floor,
        )
        if prior is None:
            return 0.0, None
        return prior.score_bonus, prior.as_dict()

    def _community_route_prior_bonus(
        self,
        subject_id: str,
        *,
        context: RunContext,
    ) -> tuple[float, dict[str, Any] | None]:
        if self._community_prior is None:
            return 0.0, None
        prior = self._community_prior.score_route(
            subject_id=subject_id,
            character_id=context.character_id,
            act_id=context.act_id,
        )
        if prior is None:
            return 0.0, None
        return prior.score_bonus, prior.as_dict()

    def _reward_card_score(self, card: RewardCardOptionPayload, *, context: RunContext) -> float:
        score = card_text_score(card.card_id, card.name, card.resolved_rules_text or card.rules_text, upgraded=card.upgraded)
        score += self._contextual_card_bonus(card.card_id, card.name, card.resolved_rules_text or card.rules_text, context)
        return score

    def _selection_pick_score(self, card: SelectionCardPayload, *, context: RunContext) -> float:
        score = card_text_score(card.card_id, card.name, "", upgraded=card.upgraded)
        score += self._contextual_card_bonus(card.card_id, card.name, "", context)
        return score

    def _selection_upgrade_score(self, card: SelectionCardPayload, *, context: RunContext) -> float:
        score = self._selection_pick_score(card, context=context)
        lower_id = card.card_id.lower()
        if "strike" in lower_id or _matches_any_pattern(lower_id, _STRIKE_TITLE_PATTERNS):
            score -= 0.35
        if "defend" in lower_id or _matches_any_pattern(lower_id, _DEFEND_TITLE_PATTERNS):
            score -= 0.25
        return score

    def _selection_remove_score(self, card: SelectionCardPayload, *, context: RunContext) -> float:
        score = -self._selection_pick_score(card, context=context)
        score += removable_card_bias(card.card_id, card.name)
        if _is_starter_strike(card.card_id, card.name):
            score += 0.45 * max(0, context.deck.strike_count - 4)
        if _is_starter_defend(card.card_id, card.name):
            score += 0.30 * max(0, context.deck.defend_count - 4)
        if _contains_any(_normalized_text(card.card_id, card.name), _CURSE_FRAGMENTS + _STATUS_FRAGMENTS):
            score += 2.50
        if "perfected_strike" in card.card_id.lower() and context.deck.strike_count >= 5:
            score -= 1.25
        return score

    def _selection_transform_score(self, card: SelectionCardPayload, *, context: RunContext) -> float:
        return self._selection_remove_score(card, context=context)

    def _selection_score_for_mode(
        self,
        mode: str,
        *,
        card: SelectionCardPayload,
        context: RunContext,
    ) -> float:
        if mode == "remove":
            return self._selection_remove_score(card, context=context)
        if mode == "upgrade":
            return self._selection_upgrade_score(card, context=context)
        if mode == "transform":
            return self._selection_transform_score(card, context=context)
        return self._selection_pick_score(card, context=context)

    def _selection_prior_domain(self, mode: str) -> CommunityPriorDomain | None:
        if mode == "remove":
            return "selection_remove"
        if mode == "upgrade":
            return "selection_upgrade"
        if mode == "pick":
            return "selection_pick"
        return None

    def _selection_reason_for_mode(self, mode: str) -> str:
        return {
            "remove": "remove_worst_card",
            "upgrade": "upgrade_best_card",
            "transform": "transform_worst_card",
        }.get(mode, "pick_best_selection_card")

    def _shop_removal_score(self, price: int, *, context: RunContext) -> float:
        score = 2.75
        score += 2.40 * context.deck.curse_count
        score += 1.25 * context.deck.status_count
        score += 0.45 * max(0, context.deck.strike_count - 4)
        score += 0.30 * max(0, context.deck.defend_count - 4)
        if context.floor <= 15:
            score += 0.90
        if context.gold - price < self._config.reserve_gold_after_shop:
            score -= 0.40
        return score

    def _shop_relic_score(self, relic: ShopRelicPayload, *, context: RunContext) -> float:
        rarity_score = _RARITY_SCORES.get((relic.rarity or "").lower(), 1.0)
        text = _normalized_text(relic.relic_id, relic.name, relic.rarity)
        score = rarity_score * 2.10
        score += text_score(text)
        for fragment, bonus in _RELIC_BONUSES.items():
            if fragment in text:
                score += bonus
        score += self._boss_conditioned_text_bonus(text, context=context, domain="shop")
        score -= relic.price / 120.0
        if context.gold - relic.price < self._config.reserve_gold_after_shop:
            score -= 0.25
        return score

    def _shop_card_score(self, card: ShopCardPayload, *, context: RunContext) -> float:
        score = card_text_score(card.card_id, card.name, "", upgraded=card.upgraded)
        score += self._contextual_card_bonus(card.card_id, card.name, "", context)
        if card.on_sale:
            score += 0.60
        score -= card.price / 125.0
        if context.gold - card.price < self._config.reserve_gold_after_shop:
            score -= 0.30
        return score

    def _shop_potion_score(self, potion: ShopPotionPayload, *, context: RunContext) -> float:
        text = _normalized_text(potion.potion_id, potion.name, potion.rarity, potion.usage)
        score = text_score(text) + 0.50
        for fragment, bonus in _POTION_BONUSES.items():
            if fragment in text:
                score += bonus
        score += self._boss_conditioned_text_bonus(text, context=context, domain="shop")
        if context.empty_potion_slots <= 0:
            score -= 1.40
        elif context.empty_potion_slots == 1:
            score -= 0.15
        score -= potion.price / 145.0
        return score

    def _shop_purchase_allowed(self, action: CandidateAction, score: float) -> bool:
        if action.action == "buy_potion":
            return score >= self._config.buy_shop_potion_above_score
        if action.action == "buy_card":
            return score >= self._config.buy_shop_card_above_score
        return score >= 0.85

    def _event_option_score(self, option, *, context: RunContext) -> float:
        text = _normalized_text(option.title, option.description)
        value = text_score(text)
        if option.will_kill_player:
            value -= 100.0
        if option.is_proceed:
            value -= 0.25
        if option.has_relic_preview:
            value += 1.80
        if _contains_any(text, _RELIC_FRAGMENTS):
            value += 1.20
        if _contains_any(text, _GOLD_FRAGMENTS):
            value += 0.80
        if _contains_any(text, _REMOVE_FRAGMENTS):
            value += 1.25
        if _contains_any(text, _CURSE_FRAGMENTS):
            value -= 1.60
        if _contains_any(text, _SELF_DAMAGE_FRAGMENTS) and context.hp_ratio <= self._config.low_hp_ratio:
            value -= 2.20
        if _contains_any(text, _REST_FRAGMENTS) and context.hp_ratio <= self._config.low_hp_ratio:
            value += 1.20
        value -= self._runtime_selection_capability_penalty(text)
        return value

    def _event_option_reason(self, option) -> str:
        text = _normalized_text(option.title, option.description)
        if option.will_kill_player:
            return "avoid_fatal_event_option"
        if option.has_relic_preview or _contains_any(text, _RELIC_FRAGMENTS):
            return "take_event_relic_or_power"
        if _contains_any(text, _GOLD_FRAGMENTS):
            return "take_event_gold"
        if _contains_any(text, _REMOVE_FRAGMENTS):
            return "take_event_card_cleanup"
        if option.is_proceed:
            return "take_event_proceed"
        return "take_best_event_option"

    def _reward_claim_score(self, reward_type: str, description: str, *, context: RunContext) -> float:
        reward_key = (reward_type or "").lower()
        base = {
            "relic": 7.50,
            "gold": 5.40,
            "card": 4.30,
            "potion": 2.70 if context.empty_potion_slots > 0 else 0.40,
            "heal": 4.80 if context.hp_ratio <= self._config.low_hp_ratio else 2.00,
        }.get(reward_key, 0.80)
        if reward_key == "card" and context.deck.deck_size >= 17:
            base -= 0.50
        return base + text_score(description) + self._boss_conditioned_text_bonus(
            _normalized_text(reward_type, description),
            context=context,
            domain="reward",
        )

    def _reward_skip_threshold(self, context: RunContext) -> float:
        threshold = self._config.skip_reward_card_below_score
        if context.floor <= 4:
            threshold -= 0.30
        if context.floor >= 10:
            threshold += 0.20
        if context.deck.deck_size >= 17:
            threshold += 0.15
        if context.hp_ratio <= self._config.critical_hp_ratio:
            threshold += 0.20
        return threshold

    def _map_node_score(self, node: MapNodePayload | None, *, context: RunContext) -> float:
        if node is None:
            return float("-inf")

        kind = _map_node_kind(node.node_type)
        if kind == "rest":
            base = 9.20 if context.hp_ratio <= self._config.low_hp_ratio else 4.60
            if context.hp_ratio >= self._config.upgrade_hp_ratio and context.floor <= 5:
                base -= 1.20
        elif kind == "elite":
            base = 8.80 if context.hp_ratio >= self._config.elite_hp_ratio else 1.80
            if context.occupied_potions > 0:
                base += 0.35
        elif kind == "shop":
            base = 7.50 if context.gold >= self._config.preferred_gold_for_shop else 3.10
            if context.deck.curse_count > 0:
                base += 1.20
        elif kind == "treasure":
            base = 7.00
        elif kind == "event":
            base = 5.90 if context.floor >= 4 else 4.80
        elif kind == "monster":
            base = 6.10 if context.floor <= 4 else 5.10
            if context.hp_ratio <= self._config.low_hp_ratio:
                base -= 1.25
        else:
            base = 3.00
        return base - (node.row * 0.01) - (node.col * 0.001)

    def _map_node_reason(self, node: MapNodePayload | None, *, context: RunContext) -> str:
        kind = _map_node_kind(node.node_type if node is not None else "")
        if kind == "rest" and context.hp_ratio <= self._config.low_hp_ratio:
            return "route_to_rest_for_survival"
        if kind == "elite" and context.hp_ratio >= self._config.elite_hp_ratio:
            return "route_to_elite_with_safe_hp"
        if kind == "shop" and context.gold >= self._config.preferred_gold_for_shop:
            return "route_to_shop_with_gold"
        if kind == "treasure":
            return "route_to_treasure_value"
        if kind == "event":
            return "route_to_event_flexibility"
        return "route_to_monster_progression"

    def _rest_option_score(self, option: RestOptionPayload | None, *, context: RunContext) -> float:
        if option is None:
            return float("-inf")
        text = _normalized_text(option.option_id, option.title, option.description)
        if context.hp_ratio <= self._config.critical_hp_ratio and _contains_any(text, _REST_FRAGMENTS):
            return 9.60 + self._boss_conditioned_text_bonus(text, context=context, domain="rest")
        if context.hp_ratio <= self._config.low_hp_ratio and _contains_any(text, _REST_FRAGMENTS):
            return 8.80 + self._boss_conditioned_text_bonus(text, context=context, domain="rest")
        if context.hp_ratio >= self._config.upgrade_hp_ratio and _contains_any(text, _UPGRADE_FRAGMENTS):
            return 8.20 + self._boss_conditioned_text_bonus(text, context=context, domain="rest")
        if context.hp_ratio >= self._config.upgrade_hp_ratio and _contains_any(text, _DIG_FRAGMENTS):
            return 7.10 + self._boss_conditioned_text_bonus(text, context=context, domain="rest")
        if _contains_any(text, _REST_FRAGMENTS):
            return 5.20 + self._boss_conditioned_text_bonus(text, context=context, domain="rest")
        if _contains_any(text, _UPGRADE_FRAGMENTS):
            return 6.40 + self._boss_conditioned_text_bonus(text, context=context, domain="rest")
        if _contains_any(text, _DIG_FRAGMENTS):
            return 5.90 + self._boss_conditioned_text_bonus(text, context=context, domain="rest")
        if _contains_any(text, _REMOVE_FRAGMENTS):
            return 6.80 + self._boss_conditioned_text_bonus(text, context=context, domain="rest")
        return 2.50 + text_score(text) + self._boss_conditioned_text_bonus(text, context=context, domain="rest")

    def _rest_option_reason(self, option: RestOptionPayload | None, *, context: RunContext) -> str:
        text = _normalized_text(option.option_id if option is not None else "", option.title if option is not None else "")
        if (
            _contains_any(text, _REST_FRAGMENTS)
            and "sustain_boss" in context.boss_profile.notes
            and self._boss_gap(context, "sustain") > 0.0
            and context.hp_ratio <= 0.62
        ):
            return "rest_for_boss_sustain"
        if (
            _contains_any(text, _UPGRADE_FRAGMENTS)
            and ("scaling_boss" in context.boss_profile.notes or context.boss_profile.upgrade_urgency >= 1.0)
            and context.hp_ratio >= 0.50
        ):
            return "upgrade_for_boss_plan"
        if _contains_any(text, _REST_FRAGMENTS) and context.hp_ratio <= self._config.low_hp_ratio:
            return "rest_for_survival"
        if _contains_any(text, _UPGRADE_FRAGMENTS):
            return "upgrade_at_rest"
        if _contains_any(text, _DIG_FRAGMENTS):
            return "dig_at_rest"
        return "take_special_rest_option"

    def _selection_mode(self, selection: SelectionPayload) -> str:
        return str(selection.semantic_mode or "").strip().lower()

    def _selection_required_count(self, selection: SelectionPayload) -> int:
        if selection.required_count > 0:
            return selection.required_count
        return 1

    def _selection_runtime_completed_count(self, selection: SelectionPayload, *, required_count: int) -> int:
        inferred_from_remaining = 0
        if (
            required_count > 1
            or selection.supports_multi_select
            or selection.selected_count > 0
            or selection.can_confirm
            or selection.requires_confirmation
            or selection.remaining_count > 0
        ):
            remaining = selection.remaining_count if selection.remaining_count >= 0 else required_count
            inferred_from_remaining = max(0, required_count - remaining)
        completed = max(selection.selected_count, inferred_from_remaining)
        return max(0, min(required_count, completed))

    def _runtime_selection_capability_penalty(self, text: str) -> float:
        required_count = _extract_card_selection_count(text)
        if required_count is None or required_count <= 1:
            return 0.0
        normalized = _normalized_text(text, "deck_card_select")
        if _contains_any(normalized, _REMOVE_FRAGMENTS) or _contains_any(normalized, _TRANSFORM_FRAGMENTS):
            mode = "remove"
        elif _contains_any(normalized, _UPGRADE_FRAGMENTS):
            mode = "upgrade"
        else:
            mode = "pick"
        if mode in {"remove", "upgrade"}:
            return 6.50
        return 0.0

    def _contextual_card_bonus(
        self,
        card_id: str,
        name: str,
        rules_text: str,
        context: RunContext,
    ) -> float:
        text = _normalized_text(card_id, name, rules_text)
        score = 0.0
        if "perfected_strike" in card_id.lower():
            score += max(0.0, (context.deck.strike_count - 3) * 0.45)
        if _contains_any(text, _SELF_DAMAGE_FRAGMENTS):
            score -= 1.40 if context.hp_ratio <= self._config.low_hp_ratio else 0.20
        if _contains_any(text, _BLOCK_FRAGMENTS) and context.hp_ratio <= self._config.low_hp_ratio:
            score += 0.90
        if _contains_any(text, _REST_FRAGMENTS) and context.hp_ratio <= self._config.low_hp_ratio:
            score += 0.90
        if _is_starter_strike(card_id, name):
            score -= 0.30 * max(0, context.deck.strike_count - 4)
        if _is_starter_defend(card_id, name):
            score -= 0.20 * max(0, context.deck.defend_count - 4)
        if _contains_any(text, _REMOVE_FRAGMENTS) and (context.deck.curse_count > 0 or context.deck.status_count > 0):
            score += 1.20
        score += self._boss_conditioned_text_bonus(text, context=context, domain="card")
        return score

    def _boss_conditioned_text_bonus(self, text: str, *, context: RunContext, domain: Literal["card", "shop", "reward", "rest"]) -> float:
        if not self._config.strategic_bias:
            return 0.0

        score = 0.0
        aoe_gap = self._boss_gap(context, "aoe")
        frontload_gap = self._boss_gap(context, "frontload")
        sustain_gap = self._boss_gap(context, "sustain")
        scaling_gap = self._boss_gap(context, "scaling")

        if aoe_gap > 0.0 and _contains_any(text, _AOE_FRAGMENTS):
            score += 1.35 * aoe_gap
        if frontload_gap > 0.0 and _contains_any(text, _FRONTLOAD_FRAGMENTS):
            score += 0.85 * frontload_gap
        if sustain_gap > 0.0 and _contains_any(text, _SUSTAIN_FRAGMENTS):
            score += 0.90 * sustain_gap
        if scaling_gap > 0.0 and _contains_any(text, _SCALING_FRAGMENTS + _DRAW_FRAGMENTS + _UPGRADE_FRAGMENTS):
            score += 0.80 * scaling_gap

        if domain == "rest":
            if _contains_any(text, _UPGRADE_FRAGMENTS) and context.hp_ratio >= 0.50:
                score += 0.75 * context.boss_profile.upgrade_urgency
            if _contains_any(text, _REST_FRAGMENTS) and context.hp_ratio <= 0.62:
                score += 0.80 * sustain_gap
        elif domain == "shop":
            if _contains_any(text, _SHOP_QUALITY_FRAGMENTS):
                score += 0.35 * context.boss_profile.shop_urgency
        return score

    def _combat_card_score(self, observation: StepObservation, candidate: CandidateAction) -> float:
        combat = observation.state.combat
        if combat is None or candidate.request.card_index is None:
            return float("-inf")
        card = next((item for item in combat.hand if item.index == candidate.request.card_index), None)
        if card is None:
            return float("-inf")

        rules_text = card.resolved_rules_text or card.rules_text
        context = self._build_run_context(observation)
        score = card_text_score(card.card_id, card.name, rules_text, upgraded=card.upgraded)
        score += self._contextual_card_bonus(card.card_id, card.name, rules_text, context)
        if card.energy_cost == 0:
            score += 0.35
        if candidate.request.target_index is not None:
            enemy = next((enemy for enemy in combat.enemies if enemy.index == candidate.request.target_index), None)
            if enemy is not None:
                estimated_damage = estimate_damage(rules_text)
                if estimated_damage >= enemy.current_hp > 0:
                    score += 6.0
                elif estimated_damage > 0:
                    score += min(3.0, estimated_damage / max(1, enemy.current_hp))
                if _contains_any(rules_text.lower(), ("vulnerable", "weak", "易伤", "虚弱")):
                    score += 0.6

        alive_enemies = len([enemy for enemy in combat.enemies if enemy.is_alive])
        if alive_enemies >= 2 and _contains_any(rules_text.lower(), ("all enemies", "所有敌人", "全体敌人")):
            score += 2.0

        if card.target_type == "Self":
            hp_ratio = combat.player.current_hp / combat.player.max_hp if combat.player.max_hp > 0 else 1.0
            score += (1.0 - hp_ratio) * estimate_block(rules_text) * 0.15

        return score

    def _combat_potion_score(self, observation: StepObservation, candidate: CandidateAction) -> float:
        run = observation.state.run
        combat = observation.state.combat
        if run is None or combat is None or candidate.request.option_index is None:
            return float("-inf")
        potion = next((item for item in run.potions if item.index == candidate.request.option_index), None)
        if potion is None:
            return float("-inf")
        text = _normalized_text(potion.potion_id, potion.name, potion.usage, potion.target_type)
        score = text_score(text) + 1.0
        if candidate.request.target_index is not None:
            enemy = next((item for item in combat.enemies if item.index == candidate.request.target_index), None)
            if enemy is not None and enemy.current_hp <= 15:
                score += 1.5
        return score


def card_text_score(card_id: str, name: str, rules_text: str, *, upgraded: bool) -> float:
    text = _normalized_text(card_id, name, rules_text)
    value = text_score(text)
    for keyword, weight in _NUMBERED_KEYWORDS.items():
        value += weight * summed_keyword_values(text, keyword)
    for fragments, bonus in _TEXT_BONUSES.items():
        if _contains_any(text, fragments):
            value += bonus
    for fragment, bonus in _CARD_ID_BONUSES.items():
        if fragment in card_id.lower():
            value += bonus
    for fragment, penalty in _CARD_ID_PENALTIES.items():
        if fragment in card_id.lower():
            value += penalty
    if upgraded:
        value += 0.20
    return value


def removable_card_bias(card_id: str, name: str) -> float:
    text = _normalized_text(card_id, name)
    if _contains_any(text, _CURSE_FRAGMENTS + _STATUS_FRAGMENTS):
        return 3.0
    if _is_starter_strike(card_id, name):
        return 1.4
    if _is_starter_defend(card_id, name):
        return 0.9
    return 0.0


def text_score(text: str) -> float:
    normalized = text.lower()
    score = 0.0
    if _contains_any(normalized, _RELIC_FRAGMENTS):
        score += 1.2
    if _contains_any(normalized, _GOLD_FRAGMENTS):
        score += 0.6
    if _contains_any(normalized, _DRAW_FRAGMENTS):
        score += 0.8
    if _contains_any(normalized, _ENERGY_FRAGMENTS):
        score += 1.0
    if _contains_any(normalized, _BLOCK_FRAGMENTS):
        score += 0.4
    if _contains_any(normalized, _DAMAGE_FRAGMENTS):
        score += 0.5
    if _contains_any(normalized, _REST_FRAGMENTS):
        score += 0.9
    if _contains_any(normalized, _UPGRADE_FRAGMENTS):
        score += 0.8
    if _contains_any(normalized, _REMOVE_FRAGMENTS):
        score += 1.0
    if _contains_any(normalized, _SELF_DAMAGE_FRAGMENTS) and ("hp" in normalized or "生命" in normalized):
        score -= 1.2
    if _contains_any(normalized, _CURSE_FRAGMENTS):
        score -= 1.5
    return score


def summed_keyword_values(text: str, keyword: str) -> int:
    patterns = _NUMBERED_PATTERNS[keyword]
    total = 0
    for pattern in patterns:
        total += sum(int(match.group(1)) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return total


def estimate_damage(rules_text: str) -> int:
    return summed_keyword_values(rules_text.lower(), "damage")


def estimate_block(rules_text: str) -> int:
    return summed_keyword_values(rules_text.lower(), "block")


def _normalized_text(*parts: object) -> str:
    normalized_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        text = _RICH_TEXT_TAG_PATTERN.sub("", str(part))
        normalized_parts.append(text)
    return " ".join(normalized_parts).lower()


def _extract_card_selection_count(*parts: object) -> int | None:
    text = _normalized_text(*parts)
    if not text:
        return None
    for pattern in _CARD_SELECTION_COUNT_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _contains_any(text: str, fragments: Iterable[str]) -> bool:
    return any(fragment in text for fragment in fragments)


def _agent_view_line(item: object) -> str:
    if isinstance(item, dict):
        value = item.get("line")
        return str(value) if value else ""
    return str(item) if item else ""


def _card_title_from_line(line: str) -> str:
    text = line.strip().lower()
    for separator in (" [", "[", "：", ":"):
        if separator in text:
            return text.split(separator, 1)[0].strip()
    return text


def _matches_any_pattern(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) is not None for pattern in patterns)


def _is_starter_strike(card_id: str, name: str) -> bool:
    text = _normalized_text(card_id, name)
    return text.startswith("strike") or text.startswith("打击") or "strike_" in text


def _is_starter_defend(card_id: str, name: str) -> bool:
    text = _normalized_text(card_id, name)
    return text.startswith("defend") or text.startswith("防御") or "defend_" in text


def _map_node_kind(node_type: str) -> str:
    normalized = (node_type or "").lower()
    if "rest" in normalized or "camp" in normalized:
        return "rest"
    if "elite" in normalized:
        return "elite"
    if "shop" in normalized:
        return "shop"
    if "treasure" in normalized or "chest" in normalized:
        return "treasure"
    if "event" in normalized or "?" in normalized:
        return "event"
    if "monster" in normalized or "fight" in normalized:
        return "monster"
    return "unknown"
