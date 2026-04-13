from __future__ import annotations

import hashlib

from sts2_rl.env.types import StepObservation

MAX_ENEMIES = 4
MAX_HAND_CARDS = 10
HAND_HASH_BUCKETS = 16
ENEMY_HASH_BUCKETS = 8
INTENT_HASH_BUCKETS = 8
COMBAT_ENCODER_SCHEMA_VERSION = 1
COMBAT_ENCODER_NAME = "combat_v1"


class CombatStateEncoder:
    def __init__(self) -> None:
        self._feature_names = self._build_feature_names()
        self._feature_count = len(self._feature_names)

    @property
    def feature_count(self) -> int:
        return self._feature_count

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def feature_schema_version(self) -> int:
        return COMBAT_ENCODER_SCHEMA_VERSION

    @property
    def feature_space_name(self) -> str:
        return COMBAT_ENCODER_NAME

    def encode(self, observation: StepObservation) -> list[float]:
        state = observation.state
        combat = state.combat
        run = state.run

        if combat is None or run is None:
            raise ValueError("CombatStateEncoder requires a COMBAT observation with run data.")

        player = combat.player
        enemies = combat.enemies[:MAX_ENEMIES]
        hand = combat.hand[:MAX_HAND_CARDS]

        features: list[float] = [1.0]
        features.extend(
            [
                _ratio(player.current_hp, player.max_hp),
                _clip(player.block / 20.0),
                _clip(player.energy / 5.0),
                _clip(player.stars / 5.0),
                _clip(player.focus / 10.0),
                _clip(run.floor / 60.0),
                _clip(run.gold / 200.0),
                _ratio(run.current_hp, run.max_hp),
                _clip(run.max_energy / 5.0),
                _clip((state.turn or 0) / 10.0),
                _clip(len([enemy for enemy in enemies if enemy.is_alive]) / float(MAX_ENEMIES)),
            ]
        )

        for index in range(MAX_ENEMIES):
            if index < len(enemies):
                enemy = enemies[index]
                features.extend(
                    [
                        1.0 if enemy.is_alive else 0.0,
                        _clip(enemy.current_hp / 100.0),
                        _ratio(enemy.current_hp, enemy.max_hp),
                        _clip(enemy.block / 20.0),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

        for index in range(MAX_HAND_CARDS):
            if index < len(hand):
                card = hand[index]
                features.extend(
                    [
                        1.0,
                        1.0 if card.playable else 0.0,
                        _clip(card.energy_cost / 3.0),
                        1.0 if card.upgraded else 0.0,
                        1.0 if card.requires_target else 0.0,
                        _clip(len(card.valid_target_indices) / 4.0),
                        1.0 if card.target_type == "Self" else 0.0,
                    ]
                )
            else:
                features.extend([0.0] * 7)

        features.extend(self._hashed_counts([card.card_id for card in hand], HAND_HASH_BUCKETS))
        features.extend(self._hashed_counts([enemy.enemy_id for enemy in enemies], ENEMY_HASH_BUCKETS))
        features.extend(
            self._hashed_counts(
                [enemy.intent or "unknown" for enemy in enemies if enemy.intent is not None],
                INTENT_HASH_BUCKETS,
            )
        )

        if len(features) != self.feature_count:
            raise RuntimeError(f"Expected {self.feature_count} features but encoded {len(features)}.")

        return features

    def schema_payload(self) -> dict[str, object]:
        return {
            "feature_space_name": self.feature_space_name,
            "feature_schema_version": self.feature_schema_version,
            "feature_count": self.feature_count,
            "feature_names": list(self.feature_names),
        }

    def _hashed_counts(self, values: list[str], bucket_count: int) -> list[float]:
        buckets = [0.0] * bucket_count
        if not values:
            return buckets

        scale = 1.0 / len(values)
        for value in values:
            digest = hashlib.md5(value.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:2], byteorder="little") % bucket_count
            buckets[bucket] += scale
        return buckets

    def _build_feature_names(self) -> tuple[str, ...]:
        names = [
            "bias",
            "player_current_hp_ratio",
            "player_block_clipped",
            "player_energy_clipped",
            "player_stars_clipped",
            "player_focus_clipped",
            "run_floor_clipped",
            "run_gold_clipped",
            "run_current_hp_ratio",
            "run_max_energy_clipped",
            "turn_clipped",
            "alive_enemy_ratio",
        ]
        for index in range(MAX_ENEMIES):
            names.extend(
                [
                    f"enemy_{index}_alive",
                    f"enemy_{index}_current_hp_clipped",
                    f"enemy_{index}_current_hp_ratio",
                    f"enemy_{index}_block_clipped",
                ]
            )
        for index in range(MAX_HAND_CARDS):
            names.extend(
                [
                    f"hand_{index}_present",
                    f"hand_{index}_playable",
                    f"hand_{index}_energy_cost_clipped",
                    f"hand_{index}_upgraded",
                    f"hand_{index}_requires_target",
                    f"hand_{index}_target_count_clipped",
                    f"hand_{index}_target_self",
                ]
            )
        names.extend(f"hand_hash_bucket_{index}" for index in range(HAND_HASH_BUCKETS))
        names.extend(f"enemy_hash_bucket_{index}" for index in range(ENEMY_HASH_BUCKETS))
        names.extend(f"intent_hash_bucket_{index}" for index in range(INTENT_HASH_BUCKETS))
        return tuple(names)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _clip(numerator / denominator)


def _clip(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
