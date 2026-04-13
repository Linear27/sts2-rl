import json
from pathlib import Path

from sts2_rl.env.models import AvailableActionsPayload, CombatEnemyPayload, CombatHandCardPayload, CombatPayload, RunPayload
from sts2_rl.env.models import GameStatePayload
from sts2_rl.env.types import StepObservation
from sts2_rl.predict import CombatOutcomePredictor, CombatOutcomePredictorTrainConfig, extract_predictor_dataset
from sts2_rl.predict import train_combat_outcome_predictor


def test_train_combat_outcome_predictor_writes_artifacts_and_scores(tmp_path: Path) -> None:
    session_dir = tmp_path / "artifacts" / "session-b"
    session_dir.mkdir(parents=True)
    combat_outcomes_path = session_dir / "combat-outcomes.jsonl"
    records = [
        _combat_outcome_payload(outcome="won", floor=3, player_hp=70, enemy_hp=16, gold=120, enemy_id="SLIME_SMALL"),
        _combat_outcome_payload(outcome="won", floor=5, player_hp=64, enemy_hp=20, gold=128, enemy_id="SLAVER_RED"),
        _combat_outcome_payload(outcome="won", floor=7, player_hp=58, enemy_hp=24, gold=140, enemy_id="JAW_WORM"),
        _combat_outcome_payload(outcome="won", floor=9, player_hp=52, enemy_hp=28, gold=156, enemy_id="LOUSE_GREEN"),
        _combat_outcome_payload(outcome="lost", floor=3, player_hp=18, enemy_hp=46, gold=88, enemy_id="FUNGI_BEAST"),
        _combat_outcome_payload(outcome="lost", floor=5, player_hp=14, enemy_hp=54, gold=74, enemy_id="SENTRY"),
        _combat_outcome_payload(outcome="lost", floor=7, player_hp=10, enemy_hp=60, gold=62, enemy_id="GREMLIN_WIZARD"),
        _combat_outcome_payload(outcome="lost", floor=9, player_hp=8, enemy_hp=66, gold=58, enemy_id="ORB_WALKER"),
    ]
    with combat_outcomes_path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in records:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")

    dataset_report = extract_predictor_dataset(
        [tmp_path / "artifacts"],
        output_dir=tmp_path / "data" / "predict" / "bootstrap",
    )

    training_report = train_combat_outcome_predictor(
        dataset_source=dataset_report.output_dir,
        output_root=tmp_path / "artifacts" / "predict",
        session_name="bootstrap-run",
        config=CombatOutcomePredictorTrainConfig(
            epochs=120,
            learning_rate=0.08,
            l2=0.0001,
            validation_fraction=0.4,
            seed=7,
        ),
    )

    assert training_report.model_path.exists()
    assert training_report.metrics_path.exists()
    assert training_report.summary_path.exists()
    assert training_report.example_count == 8
    assert training_report.train_example_count == dataset_report.split_counts["train"]
    assert training_report.validation_example_count == dataset_report.split_counts["validation"]
    assert training_report.examples_path == dataset_report.examples_path
    assert training_report.train_examples_path == dataset_report.output_dir / "train.examples.jsonl"
    assert training_report.validation_examples_path == dataset_report.output_dir / "validation.examples.jsonl"
    assert training_report.split_strategy == "manifest_split"
    assert training_report.feature_count > 10
    assert training_report.best_epoch >= 1

    predictor = CombatOutcomePredictor.load(training_report.model_path)
    strong_summary = _start_summary(player_hp=68, floor=11, gold=170, enemy_hp=20, enemy_id="SLAVER_BLUE")
    weak_summary = _start_summary(player_hp=10, floor=11, gold=50, enemy_hp=64, enemy_id="BOOK_OF_STABBING")

    strong_scores = predictor.score_summary(strong_summary)
    weak_scores = predictor.score_summary(weak_summary)
    assert strong_scores.outcome_win_probability is not None
    assert weak_scores.outcome_win_probability is not None
    assert strong_scores.outcome_win_probability > weak_scores.outcome_win_probability
    assert strong_scores.expected_reward > weak_scores.expected_reward
    assert strong_scores.expected_damage_delta > weak_scores.expected_damage_delta

    observation = StepObservation(
        screen_type="COMBAT",
        run_id="RUN-OBS",
        state=GameStatePayload(
            state_version=8,
            run_id="RUN-OBS",
            screen="COMBAT",
            in_combat=True,
            turn=1,
            run=RunPayload(
                character_id="IRONCLAD",
                character_name="Ironclad",
                ascension=0,
                floor=11,
                current_hp=68,
                max_hp=80,
                gold=170,
                max_energy=3,
                potions=[],
            ),
            combat=CombatPayload(
                player={"current_hp": 68, "max_hp": 80, "block": 0, "energy": 3, "stars": 0, "focus": 0},
                hand=[
                    CombatHandCardPayload(index=0, card_id="STRIKE_IRONCLAD", name="Strike", playable=True),
                    CombatHandCardPayload(index=1, card_id="DEFEND_IRONCLAD", name="Defend", playable=True),
                    CombatHandCardPayload(index=2, card_id="BASH", name="Bash", playable=True),
                ],
                enemies=[CombatEnemyPayload(index=0, enemy_id="SLAVER_BLUE", name="Blue Slaver", current_hp=20)],
            ),
        ),
        action_descriptors=AvailableActionsPayload(screen="COMBAT", actions=[]),
        legal_actions=[],
        build_warnings=[],
    )
    observation_scores = predictor.score_observation(observation)
    assert 0.0 <= observation_scores.outcome_win_probability <= 1.0
    assert observation_scores.expected_reward is not None
    assert observation_scores.expected_damage_delta is not None


def _combat_outcome_payload(
    *,
    outcome: str,
    floor: int,
    player_hp: int,
    enemy_hp: int,
    gold: int,
    enemy_id: str,
) -> dict:
    return {
        "schema_version": 2,
        "record_type": "combat_finished",
        "timestamp_utc": "2026-04-12T00:00:00+00:00",
        "session_name": "session-b",
        "session_kind": "train",
        "instance_id": "inst-01",
        "run_id": f"RUN-{floor}",
        "floor": floor,
        "combat_index": floor,
        "started_step_index": 0,
        "finished_step_index": 7,
        "outcome": outcome,
        "cumulative_reward": 1.5 if outcome == "won" else -1.0,
        "step_count": 7,
        "enemy_ids": [enemy_id],
        "damage_dealt": 32 if outcome == "won" else 10,
        "damage_taken": 5 if outcome == "won" else 30,
        "start_summary": _start_summary(
            player_hp=player_hp,
            floor=floor,
            gold=gold,
            enemy_hp=enemy_hp,
            enemy_id=enemy_id,
        ),
        "end_summary": {
            "screen_type": "MAP" if outcome == "won" else "GAME_OVER",
            "run_id": f"RUN-{floor}",
        },
        "reason": "combat_exited",
    }


def _start_summary(
    *,
    player_hp: int,
    floor: int,
    gold: int,
    enemy_hp: int,
    enemy_id: str,
) -> dict:
    return {
        "screen_type": "COMBAT",
        "run_id": f"RUN-{floor}",
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
            "enemy_ids": [enemy_id],
            "enemy_hp": [enemy_hp],
            "hand_card_ids": ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH"],
            "playable_hand_count": 3,
        },
    }
