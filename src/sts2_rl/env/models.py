from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ApiErrorPayload(ApiModel):
    code: str
    message: str
    details: Any | None = None
    retryable: bool = False


class ApiEnvelope(ApiModel):
    ok: bool
    request_id: str | None = None
    data: Any | None = None
    error: ApiErrorPayload | None = None


class HealthPayload(ApiModel):
    service: str
    mod_version: str
    protocol_version: str
    game_version: str
    build_id: str | None = None
    branch: str | None = None
    content_channel: str | None = None
    commit: str | None = None
    build_date: str | None = None
    main_assembly_hash: str | None = None
    status: str


class RuntimeBuildPayload(ApiModel):
    build_id: str | None = None
    game_version: str | None = None
    branch: str | None = None
    content_channel: str | None = None
    commit: str | None = None
    build_date: str | None = None
    main_assembly_hash: str | None = None


class ActionDescriptor(ApiModel):
    name: str
    requires_target: bool = False
    requires_index: bool = False
    required_parameters: list[str] = Field(default_factory=list)
    optional_parameters: list[str] = Field(default_factory=list)


class AvailableActionsPayload(ApiModel):
    screen: str
    actions: list[ActionDescriptor] = Field(default_factory=list)


class SessionPayload(ApiModel):
    mode: str = "singleplayer"
    phase: str = "menu"
    control_scope: str = "local_player"


class CombatHandCardPayload(ApiModel):
    index: int
    card_id: str
    name: str
    playable: bool = False
    upgraded: bool = False
    target_type: str = ""
    requires_target: bool = False
    target_index_space: str | None = None
    valid_target_indices: list[int] = Field(default_factory=list)
    costs_x: bool = False
    star_costs_x: bool = False
    energy_cost: int = 0
    star_cost: int = 0
    rules_text: str = ""
    resolved_rules_text: str = ""
    unplayable_reason: str | None = None


class CombatEnemyPayload(ApiModel):
    index: int
    enemy_id: str
    name: str
    current_hp: int = 0
    max_hp: int = 0
    block: int = 0
    is_alive: bool = True
    is_hittable: bool = True
    intent: str | None = None


class CombatPlayerPayload(ApiModel):
    current_hp: int = 0
    max_hp: int = 0
    block: int = 0
    energy: int = 0
    stars: int = 0
    focus: int = 0


class CombatPayload(ApiModel):
    player: CombatPlayerPayload = Field(default_factory=CombatPlayerPayload)
    hand: list[CombatHandCardPayload] = Field(default_factory=list)
    enemies: list[CombatEnemyPayload] = Field(default_factory=list)


class RunPotionPayload(ApiModel):
    index: int
    potion_id: str | None = None
    name: str | None = None
    occupied: bool = False
    usage: str | None = None
    target_type: str | None = None
    requires_target: bool = False
    target_index_space: str | None = None
    valid_target_indices: list[int] = Field(default_factory=list)
    can_use: bool = False
    can_discard: bool = False


class EncounterSummaryPayload(ApiModel):
    encounter_id: str = ""
    name: str = ""
    room_type: str = ""


class RunPayload(ApiModel):
    character_id: str = ""
    character_name: str = ""
    seed: str | None = None
    ascension: int = 0
    floor: int = 0
    act_index: int | None = None
    act_number: int | None = None
    act_id: str | None = None
    act_name: str | None = None
    has_second_boss: bool = False
    boss_encounter: EncounterSummaryPayload | None = None
    second_boss_encounter: EncounterSummaryPayload | None = None
    current_hp: int = 0
    max_hp: int = 0
    gold: int = 0
    max_energy: int = 0
    potions: list[RunPotionPayload] = Field(default_factory=list)


class MapCoordPayload(ApiModel):
    row: int
    col: int


class MapNodePayload(ApiModel):
    index: int
    row: int
    col: int
    node_type: str = ""
    state: str = ""


class MapGraphNodePayload(ApiModel):
    row: int
    col: int
    node_type: str = ""
    state: str = ""
    visited: bool = False
    is_current: bool = False
    is_available: bool = False
    is_start: bool = False
    is_boss: bool = False
    is_second_boss: bool = False
    parents: list[MapCoordPayload] = Field(default_factory=list)
    children: list[MapCoordPayload] = Field(default_factory=list)


class MapPayload(ApiModel):
    current_node: MapCoordPayload | None = None
    available_nodes: list[MapNodePayload] = Field(default_factory=list)
    is_travel_enabled: bool = False
    is_traveling: bool = False
    map_generation_count: int = 0
    rows: int = 0
    cols: int = 0
    starting_node: MapCoordPayload | None = None
    boss_node: MapCoordPayload | None = None
    second_boss_node: MapCoordPayload | None = None
    nodes: list[MapGraphNodePayload] = Field(default_factory=list)


class SelectionCardPayload(ApiModel):
    index: int
    card_id: str
    name: str
    upgraded: bool = False


class SelectionPayload(ApiModel):
    kind: str = ""
    selection_family: str = ""
    semantic_mode: str = ""
    source_type: str = ""
    source_room_type: str = ""
    source_action: str | None = None
    source_event_id: str | None = None
    source_event_option_index: int | None = None
    source_event_option_text_key: str | None = None
    source_event_option_title: str | None = None
    source_rest_option_id: str | None = None
    source_rest_option_index: int | None = None
    source_rest_option_title: str | None = None
    prompt: str = ""
    prompt_loc_table: str | None = None
    prompt_loc_key: str | None = None
    min_select: int = 1
    max_select: int = 1
    required_count: int = 1
    remaining_count: int = 0
    selected_count: int = 0
    requires_confirmation: bool = False
    can_confirm: bool = False
    supports_multi_select: bool = False
    cards: list[SelectionCardPayload] = Field(default_factory=list)


class CharacterSelectOptionPayload(ApiModel):
    index: int
    character_id: str
    name: str
    is_locked: bool = False
    is_selected: bool = False
    is_random: bool = False


class CharacterSelectPayload(ApiModel):
    selected_character_id: str | None = None
    is_multiplayer: bool = False
    can_embark: bool = False
    can_unready: bool = False
    can_increase_ascension: bool = False
    can_decrease_ascension: bool = False
    local_ready: bool = False
    player_count: int = 1
    max_players: int = 1
    ascension: int = 0
    max_ascension: int = 0
    seed: str | None = None
    characters: list[CharacterSelectOptionPayload] = Field(default_factory=list)


class CustomRunPlayerPayload(ApiModel):
    player_id: str | None = None
    slot_index: int = 0
    is_local: bool = False
    character_id: str | None = None
    character_name: str | None = None
    is_ready: bool = False
    max_multiplayer_ascension_unlocked: int | None = None


class CustomRunModifierPayload(ApiModel):
    index: int
    modifier_id: str
    name: str
    description: str = ""
    is_selected: bool = False


class CustomRunPayload(ApiModel):
    selected_character_id: str | None = None
    is_multiplayer: bool = False
    net_game_type: str | None = None
    can_embark: bool = False
    can_unready: bool = False
    can_increase_ascension: bool = False
    can_decrease_ascension: bool = False
    can_set_seed: bool = False
    can_set_ascension: bool = False
    can_set_modifiers: bool = False
    local_ready: bool = False
    is_waiting_for_players: bool = False
    player_count: int = 1
    max_players: int = 1
    ascension: int = 0
    max_ascension: int = 0
    seed: str | None = None
    modifier_ids: list[str] = Field(default_factory=list)
    players: list[CustomRunPlayerPayload] = Field(default_factory=list)
    characters: list[CharacterSelectOptionPayload] = Field(default_factory=list)
    modifiers: list[CustomRunModifierPayload] = Field(default_factory=list)


class TimelineSlotPayload(ApiModel):
    index: int
    epoch_id: str
    title: str
    state: str = ""
    is_actionable: bool = False


class TimelinePayload(ApiModel):
    back_enabled: bool = False
    inspect_open: bool = False
    unlock_screen_open: bool = False
    can_choose_epoch: bool = False
    can_confirm_overlay: bool = False
    slots: list[TimelineSlotPayload] = Field(default_factory=list)


class ChestRelicOptionPayload(ApiModel):
    index: int
    relic_id: str
    name: str
    rarity: str = ""


class ChestPayload(ApiModel):
    is_opened: bool = False
    has_relic_been_claimed: bool = False
    relic_options: list[ChestRelicOptionPayload] = Field(default_factory=list)


class EventOptionPayload(ApiModel):
    index: int
    text_key: str = ""
    title: str
    description: str = ""
    is_locked: bool = False
    is_proceed: bool = False
    will_kill_player: bool = False
    has_relic_preview: bool = False


class EventPayload(ApiModel):
    event_id: str = ""
    title: str = ""
    description: str = ""
    is_finished: bool = False
    options: list[EventOptionPayload] = Field(default_factory=list)


class RestOptionPayload(ApiModel):
    index: int
    option_id: str
    title: str
    description: str = ""
    is_enabled: bool = False


class RestPayload(ApiModel):
    options: list[RestOptionPayload] = Field(default_factory=list)


class ShopCardPayload(ApiModel):
    index: int
    category: str = ""
    card_id: str
    name: str
    upgraded: bool = False
    price: int = 0
    on_sale: bool = False
    is_stocked: bool = False
    enough_gold: bool = False


class ShopRelicPayload(ApiModel):
    index: int
    relic_id: str
    name: str
    rarity: str = ""
    price: int = 0
    is_stocked: bool = False
    enough_gold: bool = False


class ShopPotionPayload(ApiModel):
    index: int
    potion_id: str | None = None
    name: str | None = None
    rarity: str | None = None
    usage: str | None = None
    price: int = 0
    is_stocked: bool = False
    enough_gold: bool = False


class ShopCardRemovalPayload(ApiModel):
    price: int = 0
    available: bool = False
    used: bool = False
    enough_gold: bool = False


class ShopPayload(ApiModel):
    is_open: bool = False
    can_open: bool = False
    can_close: bool = False
    cards: list[ShopCardPayload] = Field(default_factory=list)
    relics: list[ShopRelicPayload] = Field(default_factory=list)
    potions: list[ShopPotionPayload] = Field(default_factory=list)
    card_removal: ShopCardRemovalPayload | None = None


class RewardOptionPayload(ApiModel):
    index: int
    reward_type: str
    description: str = ""
    claimable: bool = False


class RewardCardOptionPayload(ApiModel):
    index: int
    card_id: str
    name: str
    upgraded: bool = False
    rules_text: str = ""
    resolved_rules_text: str = ""


class RewardAlternativePayload(ApiModel):
    index: int
    label: str


class RewardPayload(ApiModel):
    pending_card_choice: bool = False
    can_proceed: bool = False
    source_type: str = ""
    source_room_type: str = ""
    source_action: str | None = None
    source_event_id: str | None = None
    source_event_option_index: int | None = None
    source_event_option_text_key: str | None = None
    source_event_option_title: str | None = None
    source_rest_option_id: str | None = None
    source_rest_option_index: int | None = None
    source_rest_option_title: str | None = None
    rewards: list[RewardOptionPayload] = Field(default_factory=list)
    card_options: list[RewardCardOptionPayload] = Field(default_factory=list)
    alternatives: list[RewardAlternativePayload] = Field(default_factory=list)


class ModalPayload(ApiModel):
    type_name: str = ""
    underlying_screen: str | None = None
    can_confirm: bool = False
    can_dismiss: bool = False
    confirm_label: str | None = None
    dismiss_label: str | None = None


class GameOverPayload(ApiModel):
    is_victory: bool = False
    floor: int | None = None
    character_id: str | None = None
    can_continue: bool = False
    can_return_to_main_menu: bool = False
    showing_summary: bool = False


class GameStatePayload(ApiModel):
    state_version: int = 0
    run_id: str = "run_unknown"
    screen: str = "UNKNOWN"
    build: RuntimeBuildPayload | None = None
    session: SessionPayload = Field(default_factory=SessionPayload)
    in_combat: bool = False
    turn: int | None = None
    available_actions: list[str] = Field(default_factory=list)
    combat: CombatPayload | None = None
    run: RunPayload | None = None
    map: MapPayload | None = None
    selection: SelectionPayload | None = None
    character_select: CharacterSelectPayload | None = None
    custom_run: CustomRunPayload | None = None
    timeline: TimelinePayload | None = None
    chest: ChestPayload | None = None
    event: EventPayload | None = None
    shop: ShopPayload | None = None
    rest: RestPayload | None = None
    reward: RewardPayload | None = None
    modal: ModalPayload | None = None
    game_over: GameOverPayload | None = None
    agent_view: dict[str, Any] | None = None


class ActionRequest(ApiModel):
    action: str
    card_index: int | None = None
    target_index: int | None = None
    option_index: int | None = None
    seed: str | None = None
    ascension: int | None = None
    modifier_id: str | None = None
    modifier_ids: list[str] | None = None
    modifier_index: int | None = None
    modifier_indices: list[int] | None = None
    enabled: bool | None = None
    command: str | None = None
    client_context: dict[str, Any] | None = None


class ActionResponsePayload(ApiModel):
    action: str
    status: str
    stable: bool = False
    message: str = ""
    state: GameStatePayload
