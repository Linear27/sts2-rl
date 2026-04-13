from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from .model import CombatOutcomePredictor, PredictorScores

PredictorGuidanceMode = Literal["heuristic_only", "assist", "dominant"]
PredictorGuidanceHook = Literal["map", "reward", "selection", "shop", "rest", "event", "combat"]

PREDICTOR_GUIDANCE_MODES: tuple[PredictorGuidanceMode, ...] = ("heuristic_only", "assist", "dominant")
PREDICTOR_GUIDANCE_HOOKS: tuple[PredictorGuidanceHook, ...] = (
    "map",
    "reward",
    "selection",
    "shop",
    "rest",
    "event",
    "combat",
)


def normalize_predictor_mode(mode: str | PredictorGuidanceMode) -> PredictorGuidanceMode:
    normalized = str(mode).strip().lower()
    if normalized not in PREDICTOR_GUIDANCE_MODES:
        raise ValueError(
            "Unsupported predictor mode: "
            f"{mode}. Expected one of: {', '.join(PREDICTOR_GUIDANCE_MODES)}."
        )
    return normalized  # type: ignore[return-value]


def normalize_predictor_hooks(hooks: Sequence[str] | None) -> tuple[PredictorGuidanceHook, ...]:
    if hooks is None:
        return PREDICTOR_GUIDANCE_HOOKS
    normalized: list[PredictorGuidanceHook] = []
    seen: set[str] = set()
    for hook in hooks:
        raw = str(hook).strip().lower()
        if not raw:
            continue
        if raw == "all":
            return PREDICTOR_GUIDANCE_HOOKS
        if raw not in PREDICTOR_GUIDANCE_HOOKS:
            raise ValueError(
                "Unsupported predictor hook: "
                f"{hook}. Expected one of: {', '.join(PREDICTOR_GUIDANCE_HOOKS)} or 'all'."
            )
        if raw in seen:
            continue
        seen.add(raw)
        normalized.append(raw)  # type: ignore[arg-type]
    return tuple(normalized) if normalized else PREDICTOR_GUIDANCE_HOOKS


@dataclass(frozen=True)
class PredictorRuntimeConfig:
    model_path: Path | None = None
    mode: PredictorGuidanceMode = "heuristic_only"
    hooks: tuple[PredictorGuidanceHook, ...] = PREDICTOR_GUIDANCE_HOOKS
    win_probability_weight: float = 1.50
    reward_weight: float = 0.75
    damage_weight: float = 0.50
    assist_heuristic_weight: float = 1.00
    assist_predictor_weight: float = 0.85
    dominant_heuristic_weight: float = 0.35
    dominant_predictor_weight: float = 1.75

    def __post_init__(self) -> None:
        normalized_mode = normalize_predictor_mode(self.mode)
        normalized_hooks = normalize_predictor_hooks(self.hooks)
        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "hooks", normalized_hooks)
        if self.model_path is not None:
            object.__setattr__(self, "model_path", Path(self.model_path).expanduser().resolve())
        if self.mode != "heuristic_only" and self.model_path is None:
            raise ValueError("Predictor-guided modes require model_path to be set.")

    @property
    def enabled(self) -> bool:
        return self.mode != "heuristic_only" and self.model_path is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "mode": self.mode,
            "hooks": list(self.hooks),
            "win_probability_weight": self.win_probability_weight,
            "reward_weight": self.reward_weight,
            "damage_weight": self.damage_weight,
            "assist_heuristic_weight": self.assist_heuristic_weight,
            "assist_predictor_weight": self.assist_predictor_weight,
            "dominant_heuristic_weight": self.dominant_heuristic_weight,
            "dominant_predictor_weight": self.dominant_predictor_weight,
        }


@dataclass(frozen=True)
class PredictorRuntimeTrace:
    hook: PredictorGuidanceHook
    mode: PredictorGuidanceMode
    model_path: str
    model_label: str
    value_estimate: float
    win_probability_component: float | None
    reward_component: float | None
    damage_component: float | None
    scores: PredictorScores
    calibration: dict[str, Any]

    def as_dict(self, *, include_feature_map: bool) -> dict[str, Any]:
        payload = {
            "hook": self.hook,
            "mode": self.mode,
            "model_path": self.model_path,
            "model_label": self.model_label,
            "value_estimate": self.value_estimate,
            "outcome_win_probability": self.scores.outcome_win_probability,
            "expected_reward": self.scores.expected_reward,
            "expected_damage_delta": self.scores.expected_damage_delta,
            "feature_count": self.scores.feature_count,
            "win_probability_component": self.win_probability_component,
            "reward_component": self.reward_component,
            "damage_component": self.damage_component,
            "calibration": self.calibration,
        }
        if include_feature_map:
            payload["feature_map"] = self.scores.feature_map
        return payload


class PredictorRuntimeAdapter:
    def __init__(self, *, config: PredictorRuntimeConfig, predictor: CombatOutcomePredictor) -> None:
        self.config = config
        self.predictor = predictor
        self.model_path = config.model_path or Path("<memory>")
        self.model_label = self.model_path.name
        self.calibration = _calibration_summary_from_metadata(self.predictor.metadata)

    @classmethod
    def from_config(cls, config: PredictorRuntimeConfig | None) -> PredictorRuntimeAdapter | None:
        if config is None or not config.enabled:
            return None
        predictor = CombatOutcomePredictor.load(config.model_path)
        return cls(config=config, predictor=predictor)

    def enabled_for(self, hook: str) -> bool:
        return self.config.enabled and hook in self.config.hooks

    def score_summary(self, summary: dict[str, Any], *, hook: PredictorGuidanceHook) -> PredictorRuntimeTrace:
        scores = self.predictor.score_summary(summary)
        reward_component = _normalized_component(
            scores.expected_reward,
            self.predictor.reward_head.target_mean,
            self.predictor.reward_head.target_std,
        )
        damage_component = _normalized_component(
            scores.expected_damage_delta,
            self.predictor.damage_head.target_mean,
            self.predictor.damage_head.target_std,
        )
        win_probability_component = (
            None
            if scores.outcome_win_probability is None
            else (float(scores.outcome_win_probability) - 0.5) * 2.0
        )
        value_estimate = 0.0
        if win_probability_component is not None:
            value_estimate += self.config.win_probability_weight * win_probability_component
        if reward_component is not None:
            value_estimate += self.config.reward_weight * reward_component
        if damage_component is not None:
            value_estimate += self.config.damage_weight * damage_component
        return PredictorRuntimeTrace(
            hook=hook,
            mode=self.config.mode,
            model_path=str(self.model_path),
            model_label=self.model_label,
            value_estimate=value_estimate,
            win_probability_component=win_probability_component,
            reward_component=reward_component,
            damage_component=damage_component,
            scores=scores,
            calibration=self.calibration,
        )

    def blend(self, heuristic_score: float, trace: PredictorRuntimeTrace) -> float:
        if self.config.mode == "assist":
            return (
                self.config.assist_heuristic_weight * heuristic_score
                + self.config.assist_predictor_weight * trace.value_estimate
            )
        if self.config.mode == "dominant":
            return (
                self.config.dominant_heuristic_weight * heuristic_score
                + self.config.dominant_predictor_weight * trace.value_estimate
            )
        return heuristic_score

    def runtime_payload(self) -> dict[str, Any]:
        return {
            **self.config.as_dict(),
            "model_label": self.model_label,
            "calibration": self.calibration,
        }


def _normalized_component(value: float | None, mean: float, std: float) -> float | None:
    if value is None:
        return None
    if std <= 0:
        return 0.0
    return (float(value) - float(mean)) / float(std)


def _calibration_summary_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    calibration = metadata.get("calibration")
    if isinstance(calibration, dict):
        return dict(calibration)
    validation_metrics = metadata.get("validation_metrics")
    if isinstance(validation_metrics, dict):
        return dict(validation_metrics)
    return {}
