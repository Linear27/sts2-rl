from __future__ import annotations

from sts2_rl.data.trajectory import build_state_summary, observed_seed_from_state
from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    CharacterSelectOptionPayload,
    CustomRunModifierPayload,
    CustomRunPayload,
    EncounterSummaryPayload,
    GameStatePayload,
    MapCoordPayload,
    MapGraphNodePayload,
    MapNodePayload,
    MapPayload,
    RewardPayload,
    RunPayload,
    SelectionCardPayload,
    SelectionPayload,
)
from sts2_rl.env.types import StepObservation, StepResult
from sts2_rl.game_run_contract import build_game_run_contract
from sts2_rl.runtime.custom_run import prepare_custom_run_from_contract


class FakeEnv:
    def __init__(self, initial_observation: StepObservation, steps: dict[tuple, list[StepResult]] | None = None) -> None:
        self.current = initial_observation
        self.steps = {key: list(value) for key, value in (steps or {}).items()}
        self.closed = False

    def observe(self) -> StepObservation:
        return self.current

    def step(self, action):
        request = action.request if hasattr(action, "request") else action
        key = (
            request.action,
            request.option_index,
            request.modifier_index,
            tuple(request.modifier_ids or []),
            request.enabled,
            request.ascension,
            request.seed,
        )
        queue = self.steps[key]
        result = queue.pop(0)
        self.current = result.observation
        return result

    def close(self) -> None:
        self.closed = True


def test_build_candidate_actions_supports_custom_run_character_and_modifier_actions() -> None:
    state = GameStatePayload(
        screen="CUSTOM_RUN",
        custom_run=CustomRunPayload(
            selected_character_id="IRONCLAD",
            can_embark=True,
            can_set_seed=True,
            can_set_ascension=True,
            can_set_modifiers=True,
            ascension=0,
            max_ascension=2,
            seed="SEED-000",
            characters=[
                CharacterSelectOptionPayload(index=0, character_id="IRONCLAD", name="Ironclad", is_selected=True),
                CharacterSelectOptionPayload(index=1, character_id="SILENT", name="Silent", is_selected=False),
            ],
            modifiers=[
                CustomRunModifierPayload(
                    index=0,
                    modifier_id="DRAFT",
                    name="Draft",
                    description="Draft starting deck.",
                    is_selected=True,
                )
            ],
        ),
    )
    descriptors = AvailableActionsPayload(
        screen="CUSTOM_RUN",
        actions=[
            ActionDescriptor(name="open_custom_run"),
            ActionDescriptor(name="select_character", requires_index=True),
            ActionDescriptor(name="set_custom_seed", required_parameters=["seed"]),
            ActionDescriptor(name="set_custom_ascension", required_parameters=["ascension"]),
            ActionDescriptor(name="set_custom_modifiers", required_parameters=["modifier_ids"]),
            ActionDescriptor(name="toggle_custom_modifier"),
        ],
    )

    result = build_candidate_actions(state, descriptors)

    action_ids = {candidate.action_id for candidate in result.candidates}
    assert "open_custom_run" in action_ids
    assert "select_character|option=0" in action_ids
    assert "select_character|option=1" in action_ids
    assert "set_custom_seed|seed=SEED-000" in action_ids
    assert "set_custom_ascension|ascension=0" in action_ids
    assert "set_custom_ascension|ascension=1" in action_ids
    assert "set_custom_ascension|ascension=2" in action_ids
    assert "set_custom_modifiers|modifiers=DRAFT" in action_ids
    assert "toggle_custom_modifier|modifier=0" in action_ids
    assert result.unsupported_actions == []


def test_prepare_custom_run_from_contract_applies_seed_character_ascension_and_modifiers() -> None:
    main_menu = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_custom_run")]),
    )
    custom_default = _custom_observation(
        selected_character_id="IRONCLAD",
        ascension=0,
        seed="DEFAULT-SEED",
        selected_modifier_indices={0},
    )
    custom_silent = _custom_observation(
        selected_character_id="SILENT",
        ascension=0,
        seed="DEFAULT-SEED",
        selected_modifier_indices={0},
    )
    custom_a2 = _custom_observation(
        selected_character_id="SILENT",
        ascension=2,
        seed="DEFAULT-SEED",
        selected_modifier_indices={0},
    )
    custom_seeded = _custom_observation(
        selected_character_id="SILENT",
        ascension=2,
        seed="SEED-001",
        selected_modifier_indices={0},
    )
    custom_cleared = _custom_observation(
        selected_character_id="SILENT",
        ascension=2,
        seed="SEED-001",
        selected_modifier_indices=set(),
    )
    started_map = _observation(
        GameStatePayload(
            screen="MAP",
            run_id="run-seed-001",
            run={
                "character_id": "SILENT",
                "character_name": "Silent",
                "seed": "SEED-001",
                "ascension": 2,
                "floor": 1,
                "current_hp": 70,
                "max_hp": 70,
                "gold": 99,
                "max_energy": 3,
                "potions": [],
            },
            map=MapPayload(),
        ),
        AvailableActionsPayload(screen="MAP", actions=[]),
    )

    normalization_env = FakeEnv(main_menu)
    prep_env = FakeEnv(
        main_menu,
        {
            ("open_custom_run", None, None, tuple(), None, None, None): [
                StepResult(observation=custom_default, terminated=False, response=None, info={})
            ],
            ("select_character", 1, None, tuple(), None, None, None): [
                StepResult(observation=custom_silent, terminated=False, response=None, info={})
            ],
            ("set_custom_ascension", None, None, tuple(), None, 2, None): [
                StepResult(observation=custom_a2, terminated=False, response=None, info={})
            ],
            ("set_custom_seed", None, None, tuple(), None, None, "SEED-001"): [
                StepResult(observation=custom_seeded, terminated=False, response=None, info={})
            ],
            ("set_custom_modifiers", None, None, tuple(), None, None, None): [
                StepResult(observation=custom_cleared, terminated=False, response=None, info={})
            ],
            ("embark", None, None, tuple(), None, None, None): [
                StepResult(observation=started_map, terminated=False, response=None, info={})
            ],
        },
    )
    envs = [normalization_env, prep_env]

    def env_factory(_base_url: str, _timeout: float):
        return envs.pop(0)

    contract = build_game_run_contract(
        run_mode="custom",
        game_seed="SEED-001",
        character_id="SILENT",
        ascension=2,
        custom_modifiers=[],
    )
    assert contract is not None

    report = prepare_custom_run_from_contract(
        base_url="http://127.0.0.1:8081",
        contract=contract,
        env_factory=env_factory,
    )

    assert report.initial_screen == "MAIN_MENU"
    assert report.final_screen == "MAP"
    assert report.final_run_id == "run-seed-001"
    assert report.final_custom_seed == "SEED-001"
    assert report.final_custom_character_id == "SILENT"
    assert report.final_custom_ascension == 2
    assert report.final_custom_modifier_ids == []
    assert report.action_sequence == [
        "open_custom_run",
        "select_character|option=1",
        "set_custom_ascension|ascension=2",
        "set_custom_seed|seed=SEED-001",
        "set_custom_modifiers|modifiers=none",
        "embark",
    ]
    assert normalization_env.closed is True
    assert prep_env.closed is True


def test_prepare_custom_run_sets_modifier_bundle_atomically() -> None:
    main_menu = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_custom_run")]),
    )
    custom_default = _custom_observation(
        selected_character_id="IRONCLAD",
        ascension=0,
        seed="SEED-777",
        selected_modifier_indices={0},
    )
    custom_configured = _custom_observation(
        selected_character_id="IRONCLAD",
        ascension=0,
        seed="SEED-777",
        selected_modifier_indices={0, 1},
    )
    started_map = _observation(
        GameStatePayload(screen="MAP", run_id="run-seed-777", map=MapPayload()),
        AvailableActionsPayload(screen="MAP", actions=[]),
    )

    normalization_env = FakeEnv(main_menu)
    prep_env = FakeEnv(
        main_menu,
        {
            ("open_custom_run", None, None, tuple(), None, None, None): [
                StepResult(observation=custom_default, terminated=False, response=None, info={})
            ],
            ("set_custom_modifiers", None, None, ("DRAFT", "SEALED_DECK"), None, None, None): [
                StepResult(observation=custom_configured, terminated=False, response=None, info={})
            ],
            ("embark", None, None, tuple(), None, None, None): [
                StepResult(observation=started_map, terminated=False, response=None, info={})
            ],
        },
    )
    envs = [normalization_env, prep_env]

    def env_factory(_base_url: str, _timeout: float):
        return envs.pop(0)

    contract = build_game_run_contract(
        run_mode="custom",
        game_seed="SEED-777",
        character_id="IRONCLAD",
        ascension=0,
        custom_modifiers=["DRAFT", "SEALED_DECK"],
    )
    assert contract is not None

    report = prepare_custom_run_from_contract(
        base_url="http://127.0.0.1:8081",
        contract=contract,
        env_factory=env_factory,
    )

    assert report.final_custom_modifier_ids == ["DRAFT", "SEALED_DECK"]
    assert report.action_sequence == [
        "open_custom_run",
        "set_custom_modifiers|modifiers=DRAFT,SEALED_DECK",
        "embark",
    ]


def test_observed_seed_falls_back_to_run_id_when_runtime_seed_field_is_missing() -> None:
    observation = _observation(
        GameStatePayload(
            screen="EVENT",
            run_id="FALLBACK-SEED-777",
            run={
                "character_id": "IRONCLAD",
                "character_name": "Ironclad",
                "seed": None,
                "ascension": 0,
                "floor": 1,
                "current_hp": 80,
                "max_hp": 80,
                "gold": 99,
                "max_energy": 3,
                "potions": [],
            },
        ),
        AvailableActionsPayload(screen="EVENT", actions=[]),
    )

    assert observed_seed_from_state(observation.state, run_id=observation.run_id) == "FALLBACK-SEED-777"
    assert build_state_summary(observation)["observed_seed"] == "FALLBACK-SEED-777"


def test_build_state_summary_includes_runtime_strategic_run_and_map_fields() -> None:
    observation = _observation(
        GameStatePayload(
            screen="MAP",
            run_id="STRATEGIC-SEED-001",
            agent_view={"run": {"deck": ["Strike", "Defend", "Neutralize"], "relics": ["Ring of the Snake"]}},
            run=RunPayload(
                character_id="SILENT",
                character_name="Silent",
                ascension=5,
                floor=21,
                act_index=1,
                act_number=2,
                act_id="THE_CITY",
                act_name="The City",
                has_second_boss=True,
                boss_encounter=EncounterSummaryPayload(
                    encounter_id="THE_CHAMP",
                    name="The Champ",
                    room_type="Boss",
                ),
                second_boss_encounter=EncounterSummaryPayload(
                    encounter_id="THE_COLLECTOR",
                    name="The Collector",
                    room_type="Boss",
                ),
                current_hp=49,
                max_hp=70,
                gold=143,
                max_energy=3,
                potions=[],
            ),
            map=MapPayload(
                current_node=MapCoordPayload(row=1, col=1),
                is_travel_enabled=True,
                is_traveling=False,
                map_generation_count=2,
                rows=5,
                cols=3,
                starting_node=MapCoordPayload(row=0, col=1),
                boss_node=MapCoordPayload(row=4, col=1),
                second_boss_node=MapCoordPayload(row=4, col=2),
                available_nodes=[
                    MapNodePayload(index=0, row=2, col=0, node_type="Monster", state="Travelable"),
                    MapNodePayload(index=1, row=2, col=2, node_type="Shop", state="Travelable"),
                ],
                nodes=[
                    MapGraphNodePayload(
                        row=0,
                        col=1,
                        node_type="Start",
                        state="Traveled",
                        visited=True,
                        is_start=True,
                        children=[MapCoordPayload(row=1, col=1)],
                    ),
                    MapGraphNodePayload(
                        row=1,
                        col=1,
                        node_type="Monster",
                        state="Traveled",
                        visited=True,
                        is_current=True,
                        children=[MapCoordPayload(row=2, col=0), MapCoordPayload(row=2, col=2)],
                        parents=[MapCoordPayload(row=0, col=1)],
                    ),
                    MapGraphNodePayload(
                        row=2,
                        col=0,
                        node_type="Monster",
                        state="Travelable",
                        is_available=True,
                        children=[MapCoordPayload(row=3, col=1)],
                        parents=[MapCoordPayload(row=1, col=1)],
                    ),
                    MapGraphNodePayload(
                        row=2,
                        col=2,
                        node_type="Shop",
                        state="Travelable",
                        is_available=True,
                        children=[MapCoordPayload(row=3, col=2)],
                        parents=[MapCoordPayload(row=1, col=1)],
                    ),
                    MapGraphNodePayload(
                        row=3,
                        col=1,
                        node_type="Elite",
                        state="Visible",
                        children=[MapCoordPayload(row=4, col=1)],
                        parents=[MapCoordPayload(row=2, col=0)],
                    ),
                    MapGraphNodePayload(
                        row=3,
                        col=2,
                        node_type="Event",
                        state="Visible",
                        children=[MapCoordPayload(row=4, col=2)],
                        parents=[MapCoordPayload(row=2, col=2)],
                    ),
                    MapGraphNodePayload(
                        row=4,
                        col=1,
                        node_type="Boss",
                        state="Visible",
                        is_boss=True,
                        parents=[MapCoordPayload(row=3, col=1)],
                    ),
                    MapGraphNodePayload(
                        row=4,
                        col=2,
                        node_type="Boss",
                        state="Visible",
                        is_second_boss=True,
                        parents=[MapCoordPayload(row=3, col=2)],
                    ),
                ],
            ),
        ),
        AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="choose_map_node", requires_index=True)]),
    )

    summary = build_state_summary(observation)

    assert summary["run"]["act_index"] == 1
    assert summary["run"]["act_number"] == 2
    assert summary["run"]["act_id"] == "THE_CITY"
    assert summary["run"]["act_name"] == "The City"
    assert summary["run"]["has_second_boss"] is True
    assert summary["run"]["boss_encounter_id"] == "THE_CHAMP"
    assert summary["run"]["second_boss_encounter_id"] == "THE_COLLECTOR"
    assert summary["map"]["current_node"] == {"row": 1, "col": 1}
    assert summary["map"]["boss_node"] == {"row": 4, "col": 1}
    assert summary["map"]["second_boss_node"] == {"row": 4, "col": 2}
    assert summary["map"]["graph_node_count"] == 8
    assert summary["map"]["graph_edge_count"] == 7
    assert summary["map"]["visited_node_count"] == 2
    assert summary["map"]["available_graph_node_count"] == 2
    assert summary["map"]["node_type_counts"] == {
        "Start": 1,
        "Monster": 2,
        "Shop": 1,
        "Elite": 1,
        "Event": 1,
        "Boss": 2,
    }
    assert summary["map"]["current_to_boss_distance"] == 3
    assert summary["map"]["current_to_second_boss_distance"] == 3
    assert summary["map"]["frontier_to_boss_min_distance"] == 2
    assert summary["map"]["frontier_to_second_boss_min_distance"] == 2


def test_build_state_summary_includes_selection_transaction_confirmation_fields() -> None:
    observation = _observation(
        GameStatePayload(
            screen="CARD_SELECTION",
            run_id="run-selection-summary",
            run=RunPayload(character_id="IRONCLAD", current_hp=65, max_hp=80, floor=4, gold=99, max_energy=3),
            selection=SelectionPayload(
                kind="deck_card_select",
                selection_family="deck",
                semantic_mode="remove",
                source_type="event",
                required_count=2,
                remaining_count=1,
                selected_count=1,
                requires_confirmation=True,
                can_confirm=True,
                supports_multi_select=True,
                cards=[
                    SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike"),
                    SelectionCardPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend"),
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="CARD_SELECTION",
            actions=[ActionDescriptor(name="select_deck_card", requires_index=True), ActionDescriptor(name="confirm_selection")],
        ),
    )

    summary = build_state_summary(observation)

    assert summary["selection"]["required_count"] == 2
    assert summary["selection"]["selected_count"] == 1
    assert summary["selection"]["requires_confirmation"] is True
    assert summary["selection"]["can_confirm"] is True


def test_build_state_summary_includes_non_combat_follow_up_lineage() -> None:
    observation = _observation(
        GameStatePayload(
            screen="REWARD",
            run_id="run-follow-up-summary",
            run=RunPayload(character_id="IRONCLAD", current_hp=65, max_hp=80, floor=4, gold=99, max_energy=3),
            reward=RewardPayload(
                pending_card_choice=True,
                source_type="event",
                source_room_type="Event",
                source_action="choose_event_option",
                source_event_id="EVENT.EPIC_QUEST",
                source_event_option_index=1,
                source_event_option_text_key="event.quest.solo",
                source_event_option_title="Solo Quest",
                card_options=[],
            ),
            selection=SelectionPayload(
                kind="deck_card_select",
                selection_family="deck",
                semantic_mode="remove",
                source_type="event",
                source_room_type="Event",
                source_action="choose_event_option",
                source_event_id="EVENT.EPIC_QUEST",
                source_event_option_index=1,
                source_event_option_text_key="event.quest.solo",
                source_event_option_title="Solo Quest",
                source_rest_option_id=None,
                source_rest_option_index=None,
                source_rest_option_title=None,
                required_count=1,
                cards=[SelectionCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike")],
            ),
        ),
        AvailableActionsPayload(
            screen="REWARD",
            actions=[ActionDescriptor(name="skip_reward_cards")],
        ),
    )

    summary = build_state_summary(observation)

    assert summary["reward"]["source_type"] == "event"
    assert summary["reward"]["source_event_id"] == "EVENT.EPIC_QUEST"
    assert summary["selection"]["source_action"] == "choose_event_option"
    assert summary["selection"]["source_event_option_text_key"] == "event.quest.solo"


def test_prepare_custom_run_clears_resume_menu_before_opening_custom_run() -> None:
    main_menu_with_resume = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(
            screen="MAIN_MENU",
            actions=[ActionDescriptor(name="continue_run"), ActionDescriptor(name="abandon_run")],
        ),
    )
    modal = _observation(
        GameStatePayload(screen="MODAL", run_id="run_unknown", modal={"can_confirm": True, "can_dismiss": True}),
        AvailableActionsPayload(screen="MODAL", actions=[ActionDescriptor(name="confirm_modal")]),
    )
    clean_main_menu = _observation(
        GameStatePayload(screen="MAIN_MENU", run_id="run_unknown"),
        AvailableActionsPayload(screen="MAIN_MENU", actions=[ActionDescriptor(name="open_custom_run")]),
    )
    custom_default = _custom_observation(
        selected_character_id="IRONCLAD",
        ascension=0,
        seed="SEED-001",
        selected_modifier_indices=set(),
    )
    started_map = _observation(
        GameStatePayload(screen="MAP", run_id="SEED-001", map=MapPayload()),
        AvailableActionsPayload(screen="MAP", actions=[]),
    )

    normalization_env = FakeEnv(clean_main_menu)
    prep_env = FakeEnv(
        main_menu_with_resume,
        {
            ("abandon_run", None, None, tuple(), None, None, None): [
                StepResult(observation=modal, terminated=False, response=None, info={})
            ],
            ("confirm_modal", None, None, tuple(), None, None, None): [
                StepResult(observation=clean_main_menu, terminated=False, response=None, info={})
            ],
            ("open_custom_run", None, None, tuple(), None, None, None): [
                StepResult(observation=custom_default, terminated=False, response=None, info={})
            ],
            ("embark", None, None, tuple(), None, None, None): [
                StepResult(observation=started_map, terminated=False, response=None, info={})
            ],
        },
    )
    envs = [normalization_env, prep_env]

    def env_factory(_base_url: str, _timeout: float):
        return envs.pop(0)

    contract = build_game_run_contract(run_mode="custom", game_seed="SEED-001", character_id="IRONCLAD", ascension=0)
    assert contract is not None

    report = prepare_custom_run_from_contract(
        base_url="http://127.0.0.1:8081",
        contract=contract,
        env_factory=env_factory,
    )

    assert report.action_sequence == [
        "abandon_run",
        "confirm_modal",
        "open_custom_run",
        "embark",
    ]


def _custom_observation(
    *,
    selected_character_id: str,
    ascension: int,
    seed: str,
    selected_modifier_indices: set[int],
) -> StepObservation:
    modifiers = [
        CustomRunModifierPayload(
            index=0,
            modifier_id="DRAFT",
            name="Draft",
            description="Draft starting deck.",
            is_selected=0 in selected_modifier_indices,
        ),
        CustomRunModifierPayload(
            index=1,
            modifier_id="SEALED_DECK",
            name="Sealed Deck",
            description="Sealed deck start.",
            is_selected=1 in selected_modifier_indices,
        ),
    ]
    state = GameStatePayload(
        screen="CUSTOM_RUN",
        run_id="run_unknown",
        custom_run=CustomRunPayload(
            selected_character_id=selected_character_id,
            can_embark=True,
            can_set_seed=True,
            can_set_ascension=True,
            can_set_modifiers=True,
            ascension=ascension,
            max_ascension=10,
            seed=seed,
            modifier_ids=[modifier.modifier_id for modifier in modifiers if modifier.is_selected],
            characters=[
                CharacterSelectOptionPayload(
                    index=0,
                    character_id="IRONCLAD",
                    name="Ironclad",
                    is_selected=selected_character_id == "IRONCLAD",
                ),
                CharacterSelectOptionPayload(
                    index=1,
                    character_id="SILENT",
                    name="Silent",
                    is_selected=selected_character_id == "SILENT",
                ),
            ],
            modifiers=modifiers,
        ),
    )
    descriptors = AvailableActionsPayload(
        screen="CUSTOM_RUN",
        actions=[
            ActionDescriptor(name="select_character", requires_index=True),
            ActionDescriptor(name="embark"),
            ActionDescriptor(name="set_custom_seed", required_parameters=["seed"]),
            ActionDescriptor(name="set_custom_ascension", required_parameters=["ascension"]),
            ActionDescriptor(name="toggle_custom_modifier", required_parameters=["modifier_id"]),
            ActionDescriptor(name="set_custom_modifiers", required_parameters=["modifier_ids"]),
        ],
    )
    return _observation(state, descriptors)


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
