from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    CombatEnemyPayload,
    CombatHandCardPayload,
    CombatPayload,
    CombatPlayerPayload,
    GameStatePayload,
    RunPayload,
)
from sts2_rl.env.types import StepObservation
from sts2_rl.train import CombatStateEncoder


def test_combat_encoder_returns_stable_feature_count() -> None:
    state = GameStatePayload(
        screen="COMBAT",
        run_id="run-1",
        turn=2,
        combat=CombatPayload(
            player=CombatPlayerPayload(current_hp=40, max_hp=80, block=7, energy=3),
            hand=[
                CombatHandCardPayload(
                    index=0,
                    card_id="strike",
                    name="Strike",
                    playable=True,
                    requires_target=True,
                    valid_target_indices=[0],
                    energy_cost=1,
                ),
                CombatHandCardPayload(
                    index=1,
                    card_id="defend",
                    name="Defend",
                    playable=True,
                    energy_cost=1,
                    target_type="Self",
                ),
            ],
            enemies=[
                CombatEnemyPayload(
                    index=0,
                    enemy_id="slime",
                    name="Slime",
                    current_hp=20,
                    max_hp=30,
                    block=3,
                    intent="ATTACK",
                )
            ],
        ),
        run=RunPayload(current_hp=40, max_hp=80, floor=12, gold=99, max_energy=3),
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

    encoder = CombatStateEncoder()
    features = encoder.encode(observation)

    assert len(features) == encoder.feature_count
    assert features[0] == 1.0
