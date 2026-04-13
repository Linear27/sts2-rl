import json
from pathlib import Path

from sts2_rl.predict import CombatOutcomePredictor, PredictorHead, extract_predictor_dataset


def build_synthetic_predictor_fixture(tmp_path: Path) -> tuple[Path, Path]:
    artifacts_root = tmp_path / "artifacts" / "session-report"
    artifacts_root.mkdir(parents=True)
    combat_path = artifacts_root / "combat-outcomes.jsonl"

    records = [
        _combat_outcome_payload("RUN-001", 3, "won", 72, 115, ["SLIME_SMALL"], [18], 1.6, 34, 4),
        _combat_outcome_payload("RUN-002", 4, "lost", 14, 72, ["SLIME_SMALL"], [40], -0.8, 12, 24),
        _combat_outcome_payload("RUN-003", 6, "won", 68, 120, ["SLIME_SMALL"], [22], 1.4, 30, 5),
        _combat_outcome_payload("RUN-004", 19, "won", 60, 150, ["SLIME_SMALL"], [26], 1.1, 28, 8),
        _combat_outcome_payload("RUN-005", 21, "lost", 10, 76, ["SLIME_SMALL"], [46], -1.1, 8, 28),
        _combat_outcome_payload("RUN-006", 23, "won", 56, 160, ["SLIME_SMALL"], [24], 1.0, 26, 10),
        _combat_outcome_payload("RUN-007", 5, "won", 70, 130, ["SENTRY", "SENTRY"], [18, 18], 1.3, 38, 6),
        _combat_outcome_payload("RUN-008", 7, "lost", 16, 70, ["SENTRY", "SENTRY"], [28, 28], -0.9, 14, 30),
        _combat_outcome_payload("RUN-009", 8, "won", 66, 132, ["SENTRY", "SENTRY"], [20, 20], 1.2, 36, 7),
        _combat_outcome_payload("RUN-010", 18, "won", 58, 162, ["SENTRY", "SENTRY"], [24, 24], 1.0, 30, 9),
        _combat_outcome_payload("RUN-011", 20, "lost", 12, 74, ["SENTRY", "SENTRY"], [30, 30], -1.2, 10, 32),
        _combat_outcome_payload("RUN-012", 22, "won", 54, 170, ["SENTRY", "SENTRY"], [22, 22], 0.9, 28, 12),
    ]
    with combat_path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in records:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    dataset_report = extract_predictor_dataset(
        [tmp_path / "artifacts"],
        output_dir=tmp_path / "data" / "predict" / "synthetic",
        split_seed=3,
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        split_group_by="record",
    )
    predictor = CombatOutcomePredictor(
        feature_names=["run:current_hp", "combat:total_enemy_hp", "run:gold", "combat:enemy_count"],
        feature_means=[0.0, 0.0, 0.0, 0.0],
        feature_stds=[1.0, 1.0, 1.0, 1.0],
        outcome_head=PredictorHead(
            name="outcome_win",
            kind="logistic",
            weights=[0.08, -0.03, 0.01, -0.12],
            bias=-0.2,
        ),
        reward_head=PredictorHead(
            name="reward",
            kind="linear",
            weights=[0.04, -0.015, 0.008, -0.08],
            bias=-0.1,
            target_mean=0.0,
            target_std=1.0,
        ),
        damage_head=PredictorHead(
            name="damage_delta",
            kind="linear",
            weights=[0.05, -0.04, 0.006, -0.1],
            bias=-0.2,
            target_mean=0.0,
            target_std=1.0,
        ),
        metadata={"fixture": "synthetic-predictor-report"},
    )
    model_path = predictor.save(tmp_path / "artifacts" / "predictor" / "combat-outcome-predictor.json")
    return dataset_report.output_dir, model_path


def _combat_outcome_payload(
    run_id: str,
    floor: int,
    outcome: str,
    player_hp: int,
    gold: int,
    enemy_ids: list[str],
    enemy_hp: list[int],
    cumulative_reward: float,
    damage_dealt: int,
    damage_taken: int,
) -> dict:
    return {
        "schema_version": 2,
        "record_type": "combat_finished",
        "timestamp_utc": "2026-04-12T00:00:00+00:00",
        "session_name": "session-report",
        "session_kind": "train",
        "instance_id": "inst-01",
        "run_id": run_id,
        "floor": floor,
        "combat_index": floor,
        "started_step_index": 0,
        "finished_step_index": 6,
        "outcome": outcome,
        "cumulative_reward": cumulative_reward,
        "step_count": 6,
        "enemy_ids": enemy_ids,
        "damage_dealt": damage_dealt,
        "damage_taken": damage_taken,
        "start_summary": {
            "screen_type": "COMBAT",
            "run_id": run_id,
            "state_version": 8,
            "turn": 1,
            "in_combat": True,
            "available_action_count": 4,
            "build_warning_count": 0,
            "session_phase": "run",
            "control_scope": "local_player",
            "run": {
                "character_id": "IRONCLAD",
                "character_name": "Ironclad",
                "ascension": 0,
                "floor": floor,
                "current_hp": player_hp,
                "max_hp": 80,
                "gold": gold,
                "max_energy": 3,
                "occupied_potions": 0,
            },
            "combat": {
                "player_hp": player_hp,
                "player_block": 0,
                "energy": 3,
                "stars": 0,
                "focus": 0,
                "enemy_ids": enemy_ids,
                "enemy_hp": enemy_hp,
                "hand_card_ids": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
                "playable_hand_count": 3,
            },
        },
        "end_summary": {"screen_type": "MAP" if outcome == "won" else "GAME_OVER", "run_id": run_id},
        "reason": "combat_exited",
    }
