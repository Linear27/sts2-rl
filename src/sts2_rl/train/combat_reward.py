from __future__ import annotations

from sts2_rl.env.types import StepObservation


def compute_combat_reward(previous: StepObservation, current: StepObservation) -> float:
    reward = 0.0

    previous_combat = previous.state.combat
    current_combat = current.state.combat
    previous_run = previous.state.run
    current_run = current.state.run

    if previous_combat is not None:
        previous_enemy_hp = sum(enemy.current_hp for enemy in previous_combat.enemies if enemy.is_alive)
        current_enemy_hp = 0
        if current_combat is not None:
            current_enemy_hp = sum(enemy.current_hp for enemy in current_combat.enemies if enemy.is_alive)
        reward += 0.03 * max(0, previous_enemy_hp - current_enemy_hp)

        previous_player_hp = previous_combat.player.current_hp
        current_player_hp = current_combat.player.current_hp if current_combat is not None else previous_player_hp
        reward -= 0.05 * max(0, previous_player_hp - current_player_hp)

        previous_block = previous_combat.player.block
        current_block = current_combat.player.block if current_combat is not None else 0
        reward += 0.01 * max(0, current_block - previous_block)

    if previous.screen_type == "COMBAT" and current.screen_type != "COMBAT":
        reward += 1.0

    if previous_run is not None and current_run is not None:
        reward += 0.5 * max(0, current_run.floor - previous_run.floor)

    if current.screen_type == "GAME_OVER" and current.state.game_over is not None:
        reward += 10.0 if current.state.game_over.is_victory else -10.0

    return reward
