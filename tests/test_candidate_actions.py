from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    CombatHandCardPayload,
    CombatPayload,
    GameStatePayload,
    MapNodePayload,
    MapPayload,
    RewardCardOptionPayload,
    RewardOptionPayload,
    RewardPayload,
    RunPayload,
    RunPotionPayload,
)


def test_build_candidate_actions_expands_play_card_targets() -> None:
    state = GameStatePayload(
        screen="COMBAT",
        run_id="run-1",
        combat=CombatPayload(
            hand=[
                CombatHandCardPayload(index=0, card_id="strike", name="Strike", playable=True),
                CombatHandCardPayload(
                    index=1,
                    card_id="bash",
                    name="Bash",
                    playable=True,
                    requires_target=True,
                    valid_target_indices=[0, 2],
                ),
                CombatHandCardPayload(index=2, card_id="wound", name="Wound", playable=False),
            ]
        ),
        run=RunPayload(),
    )
    descriptors = AvailableActionsPayload(
        screen="COMBAT",
        actions=[
            ActionDescriptor(name="play_card", requires_index=True),
            ActionDescriptor(name="end_turn"),
        ],
    )

    result = build_candidate_actions(state, descriptors)

    assert [candidate.action_id for candidate in result.candidates] == [
        "play_card|card=0",
        "play_card|card=1|target=0",
        "play_card|card=1|target=2",
        "end_turn",
    ]
    assert result.unsupported_actions == []


def test_build_candidate_actions_expands_indexed_candidates() -> None:
    state = GameStatePayload(
        screen="REWARD",
        run_id="run-2",
        map=MapPayload(
            available_nodes=[
                MapNodePayload(index=3, row=5, col=1, node_type="ELITE"),
                MapNodePayload(index=5, row=5, col=2, node_type="SHOP"),
            ]
        ),
        reward=RewardPayload(
            rewards=[
                RewardOptionPayload(index=0, reward_type="gold", claimable=True),
                RewardOptionPayload(index=1, reward_type="relic", claimable=False),
            ],
            card_options=[
                RewardCardOptionPayload(index=0, card_id="anger", name="Anger"),
                RewardCardOptionPayload(index=1, card_id="shrug", name="Shrug It Off"),
            ],
        ),
        run=RunPayload(
            potions=[
                RunPotionPayload(index=0, name="Dex Potion", can_use=True),
                RunPotionPayload(index=1, name="Fire Potion", can_use=True, requires_target=True, valid_target_indices=[4]),
                RunPotionPayload(index=2, name="Weak Potion", can_discard=True),
            ]
        ),
    )
    descriptors = AvailableActionsPayload(
        screen="REWARD",
        actions=[
            ActionDescriptor(name="choose_map_node", requires_index=True),
            ActionDescriptor(name="claim_reward", requires_index=True),
            ActionDescriptor(name="choose_reward_card", requires_index=True),
            ActionDescriptor(name="use_potion", requires_index=True),
            ActionDescriptor(name="discard_potion", requires_index=True),
        ],
    )

    result = build_candidate_actions(state, descriptors)

    assert [candidate.action_id for candidate in result.candidates] == [
        "choose_map_node|option=3",
        "choose_map_node|option=5",
        "claim_reward|option=0",
        "choose_reward_card|option=0",
        "choose_reward_card|option=1",
        "use_potion|option=0",
        "use_potion|option=1|target=4",
        "discard_potion|option=2",
    ]
