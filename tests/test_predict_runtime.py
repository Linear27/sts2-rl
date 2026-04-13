from pathlib import Path

from sts2_rl.predict import (
    CombatOutcomePredictor,
    PredictorHead,
    PredictorRuntimeAdapter,
    PredictorRuntimeConfig,
)


def test_predictor_runtime_adapter_scores_and_blends(tmp_path: Path) -> None:
    model_path = tmp_path / "combat-outcome-predictor.json"
    predictor = CombatOutcomePredictor(
        feature_names=["run:gold", "combat:player_block"],
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
        outcome_head=PredictorHead(name="outcome_win", kind="logistic", weights=[0.1, 0.0], bias=0.0),
        reward_head=PredictorHead(
            name="reward",
            kind="linear",
            weights=[0.2, 0.4],
            bias=0.0,
            target_mean=0.0,
            target_std=1.0,
        ),
        damage_head=PredictorHead(
            name="damage_delta",
            kind="linear",
            weights=[0.0, 0.2],
            bias=0.0,
            target_mean=0.0,
            target_std=1.0,
        ),
        metadata={"calibration": {"validation": {"objective": 0.25}}},
    )
    predictor.save(model_path)

    adapter = PredictorRuntimeAdapter.from_config(
        PredictorRuntimeConfig(model_path=model_path, mode="assist", hooks=("combat",))
    )

    assert adapter is not None
    trace = adapter.score_summary(
        {
            "run": {"gold": 120},
            "combat": {"player_block": 8},
        },
        hook="combat",
    )

    assert trace.calibration == {"validation": {"objective": 0.25}}
    assert trace.scores.expected_reward is not None
    assert trace.scores.expected_damage_delta is not None
    assert adapter.blend(heuristic_score=2.0, trace=trace) > 2.0
