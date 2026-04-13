from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    CombatHandCardPayload,
    CombatPayload,
    GameStatePayload,
    RunPayload,
)
from sts2_rl.env.types import StepObservation
from sts2_rl.train import CombatActionSpace


def test_combat_action_space_binds_end_turn_and_targeted_cards() -> None:
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
            ]
        ),
        run=RunPayload(),
    )
    descriptors = AvailableActionsPayload(
        screen="COMBAT",
        actions=[ActionDescriptor(name="end_turn"), ActionDescriptor(name="play_card", requires_index=True)],
    )

    build = build_candidate_actions(state, descriptors)
    observation = StepObservation(
        screen_type="COMBAT",
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )

    binding = CombatActionSpace().bind(observation)
    bound_ids = [candidate.action_id for candidate in binding.candidates if candidate is not None]

    assert "end_turn" in bound_ids
    assert "play_card|card=0" in bound_ids
    assert "play_card|card=1|target=0" in bound_ids
    assert "play_card|card=1|target=2" in bound_ids
