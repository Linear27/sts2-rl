from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

from sts2_rl.data.trajectory import build_state_summary
from sts2_rl.env.types import CandidateAction, StepObservation

if TYPE_CHECKING:
    from sts2_rl.train.strategic_finetune import StrategicFinetuneModel
    from sts2_rl.train.strategic_pretrain import StrategicPretrainModel

StrategicGuidanceMode = Literal["heuristic_only", "assist", "dominant"]
StrategicGuidanceHook = Literal["reward", "shop", "selection", "rest", "event"]

STRATEGIC_GUIDANCE_MODES: tuple[StrategicGuidanceMode, ...] = ("heuristic_only", "assist", "dominant")
STRATEGIC_GUIDANCE_HOOKS: tuple[StrategicGuidanceHook, ...] = ("reward", "shop", "selection", "rest", "event")
_SELECTION_DECISION_TYPES: dict[str, str] = {
    "pick": "selection_pick",
    "remove": "selection_remove",
    "upgrade": "selection_upgrade",
    "transform": "selection_transform",
}


def normalize_strategic_mode(mode: str | StrategicGuidanceMode) -> StrategicGuidanceMode:
    normalized = str(mode).strip().lower()
    if normalized not in STRATEGIC_GUIDANCE_MODES:
        raise ValueError(
            "Unsupported strategic mode: "
            f"{mode}. Expected one of: {', '.join(STRATEGIC_GUIDANCE_MODES)}."
        )
    return normalized  # type: ignore[return-value]


def normalize_strategic_hooks(hooks: Sequence[str] | None) -> tuple[StrategicGuidanceHook, ...]:
    if hooks is None:
        return STRATEGIC_GUIDANCE_HOOKS
    normalized: list[StrategicGuidanceHook] = []
    seen: set[str] = set()
    for hook in hooks:
        raw = str(hook).strip().lower()
        if not raw:
            continue
        if raw == "all":
            return STRATEGIC_GUIDANCE_HOOKS
        if raw not in STRATEGIC_GUIDANCE_HOOKS:
            raise ValueError(
                "Unsupported strategic hook: "
                f"{hook}. Expected one of: {', '.join(STRATEGIC_GUIDANCE_HOOKS)} or 'all'."
            )
        if raw in seen:
            continue
        seen.add(raw)
        normalized.append(raw)  # type: ignore[arg-type]
    return tuple(normalized) if normalized else STRATEGIC_GUIDANCE_HOOKS


@dataclass(frozen=True)
class StrategicRuntimeConfig:
    checkpoint_path: Path | None = None
    mode: StrategicGuidanceMode = "heuristic_only"
    hooks: tuple[StrategicGuidanceHook, ...] = STRATEGIC_GUIDANCE_HOOKS
    runtime_source_name: str = "live_policy_pack"
    runtime_game_mode: str = "standard"
    runtime_platform_type: str = "live"
    runtime_build_id: str | None = None
    support_quality: str = "full_candidates"
    assist_heuristic_weight: float = 1.00
    assist_strategic_weight: float = 0.85
    dominant_heuristic_weight: float = 0.35
    dominant_strategic_weight: float = 1.75

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_strategic_mode(self.mode))
        object.__setattr__(self, "hooks", normalize_strategic_hooks(self.hooks))
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path).expanduser().resolve())
        if self.mode != "heuristic_only" and self.checkpoint_path is None:
            raise ValueError("Strategic-guided modes require checkpoint_path to be set.")

    @property
    def enabled(self) -> bool:
        return self.mode != "heuristic_only" and self.checkpoint_path is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path is not None else None,
            "mode": self.mode,
            "hooks": list(self.hooks),
            "runtime_source_name": self.runtime_source_name,
            "runtime_game_mode": self.runtime_game_mode,
            "runtime_platform_type": self.runtime_platform_type,
            "runtime_build_id": self.runtime_build_id,
            "support_quality": self.support_quality,
            "assist_heuristic_weight": self.assist_heuristic_weight,
            "assist_strategic_weight": self.assist_strategic_weight,
            "dominant_heuristic_weight": self.dominant_heuristic_weight,
            "dominant_strategic_weight": self.dominant_strategic_weight,
        }


@dataclass(frozen=True)
class StrategicRuntimeTrace:
    hook: StrategicGuidanceHook
    mode: StrategicGuidanceMode
    checkpoint_path: str
    checkpoint_algorithm: str
    model_label: str
    feature_schema_version: int | None
    decision_type: str
    candidate_id: str
    raw_score: float
    centered_score: float
    standardized_score: float
    predicted_value: float
    candidate_count: int
    feature_count: int
    context_feature_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "hook": self.hook,
            "mode": self.mode,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_algorithm": self.checkpoint_algorithm,
            "model_label": self.model_label,
            "feature_schema_version": self.feature_schema_version,
            "decision_type": self.decision_type,
            "candidate_id": self.candidate_id,
            "raw_score": self.raw_score,
            "centered_score": self.centered_score,
            "standardized_score": self.standardized_score,
            "predicted_value": self.predicted_value,
            "candidate_count": self.candidate_count,
            "feature_count": self.feature_count,
            "context_feature_count": self.context_feature_count,
        }


class StrategicRuntimeAdapter:
    def __init__(
        self,
        *,
        config: StrategicRuntimeConfig,
        model: StrategicPretrainModel | StrategicFinetuneModel | Any,
        checkpoint_algorithm: str,
        feature_schema_version: int | None,
        metadata: dict[str, Any],
    ) -> None:
        self.config = config
        self.model = model
        self.checkpoint_path = config.checkpoint_path or Path("<memory>")
        self.checkpoint_algorithm = checkpoint_algorithm
        self.feature_schema_version = feature_schema_version
        self.metadata = metadata
        self.model_label = self.checkpoint_path.name
        decision_types = metadata.get("decision_types")
        if isinstance(decision_types, list):
            supported = {str(item).strip().lower() for item in decision_types if str(item).strip()}
        else:
            supported = set(getattr(model, "decision_ranking_heads", {}).keys())
        self.supported_decision_types = tuple(sorted(supported))

    @classmethod
    def from_config(cls, config: StrategicRuntimeConfig | None) -> StrategicRuntimeAdapter | None:
        if config is None or not config.enabled:
            return None
        from sts2_rl.train.strategic_finetune import StrategicFinetuneModel
        from sts2_rl.train.strategic_pretrain import (
            STRATEGIC_PRETRAIN_FEATURE_SCHEMA_VERSION,
            StrategicPretrainModel,
        )

        checkpoint_path = Path(config.checkpoint_path).expanduser().resolve()
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        algorithm = str(payload.get("algorithm") or "").strip().lower()
        feature_schema_version = payload.get("feature_schema_version")
        if int(feature_schema_version or 0) != STRATEGIC_PRETRAIN_FEATURE_SCHEMA_VERSION:
            raise ValueError("Unsupported strategic checkpoint feature schema version for runtime guidance.")
        if algorithm == "strategic_pretrain":
            model = StrategicPretrainModel.load(checkpoint_path)
        elif algorithm == "strategic_finetune":
            model = StrategicFinetuneModel.load(checkpoint_path)
        else:
            raise ValueError(
                "Unsupported strategic runtime checkpoint algorithm: "
                f"{payload.get('algorithm')}. Expected strategic_pretrain or strategic_finetune."
            )
        return cls(
            config=config,
            model=model,
            checkpoint_algorithm=algorithm,
            feature_schema_version=feature_schema_version if feature_schema_version is None else int(feature_schema_version),
            metadata=dict(payload.get("metadata", {})),
        )

    def enabled_for(self, hook: str) -> bool:
        return self.config.enabled and hook in self.config.hooks

    def supports_decision_type(self, decision_type: str) -> bool:
        normalized = str(decision_type).strip().lower()
        return normalized in set(self.supported_decision_types)

    def score_candidates(
        self,
        *,
        observation: StepObservation,
        hook: StrategicGuidanceHook,
        decision_type: str,
        candidate_ids: Sequence[str],
        room_type: str,
        map_point_type: str,
        source_type: str,
    ) -> dict[str, StrategicRuntimeTrace]:
        from sts2_rl.train.strategic_pretrain import (
            _strategic_candidate_feature_map,
            _strategic_context_feature_map,
        )

        normalized_decision_type = str(decision_type).strip().lower()
        ordered_candidate_ids = tuple(str(candidate_id) for candidate_id in candidate_ids if str(candidate_id).strip())
        if not ordered_candidate_ids or not self.enabled_for(hook) or not self.supports_decision_type(normalized_decision_type):
            return {}
        context = _runtime_context_payload(
            observation=observation,
            config=self.config,
            decision_type=normalized_decision_type,
            candidate_count=len(ordered_candidate_ids),
            room_type=room_type,
            map_point_type=map_point_type,
            source_type=source_type,
            checkpoint_algorithm=self.checkpoint_algorithm,
        )
        context_feature_map = _strategic_context_feature_map(
            context=context,
            decision_type=normalized_decision_type,
            support_quality=self.config.support_quality,
        )
        predicted_value = float(
            self.model.predict_value(
                decision_type=normalized_decision_type,
                context_feature_map=context_feature_map,
            )
        )
        raw_scores: dict[str, float] = {}
        feature_maps: dict[str, dict[str, float]] = {}
        for candidate_id in ordered_candidate_ids:
            feature_map = _strategic_candidate_feature_map(
                context=context,
                decision_type=normalized_decision_type,
                support_quality=self.config.support_quality,
                candidate_id=candidate_id,
                candidate_count=len(ordered_candidate_ids),
            )
            feature_maps[candidate_id] = feature_map
            raw_scores[candidate_id] = float(
                self.model.score_candidate(
                    decision_type=normalized_decision_type,
                    feature_map=feature_map,
                )
            )
        mean_score = sum(raw_scores.values()) / len(raw_scores)
        if len(raw_scores) <= 1:
            std_score = 0.0
        else:
            variance = sum((value - mean_score) ** 2 for value in raw_scores.values()) / len(raw_scores)
            std_score = math.sqrt(max(variance, 0.0))
        traces: dict[str, StrategicRuntimeTrace] = {}
        for candidate_id in ordered_candidate_ids:
            raw_score = raw_scores[candidate_id]
            centered_score = raw_score - mean_score
            standardized_score = 0.0 if std_score <= 1e-9 else centered_score / std_score
            traces[candidate_id] = StrategicRuntimeTrace(
                hook=hook,
                mode=self.config.mode,
                checkpoint_path=str(self.checkpoint_path),
                checkpoint_algorithm=self.checkpoint_algorithm,
                model_label=self.model_label,
                feature_schema_version=self.feature_schema_version,
                decision_type=normalized_decision_type,
                candidate_id=candidate_id,
                raw_score=raw_score,
                centered_score=centered_score,
                standardized_score=standardized_score,
                predicted_value=predicted_value,
                candidate_count=len(ordered_candidate_ids),
                feature_count=len(feature_maps[candidate_id]),
                context_feature_count=len(context_feature_map),
            )
        return traces

    def blend(self, current_score: float, trace: StrategicRuntimeTrace) -> float:
        if self.config.mode == "assist":
            return (
                self.config.assist_heuristic_weight * current_score
                + self.config.assist_strategic_weight * trace.standardized_score
            )
        if self.config.mode == "dominant":
            return (
                self.config.dominant_heuristic_weight * current_score
                + self.config.dominant_strategic_weight * trace.standardized_score
            )
        return current_score

    def runtime_payload(self) -> dict[str, Any]:
        return {
            **self.config.as_dict(),
            "model_label": self.model_label,
            "checkpoint_algorithm": self.checkpoint_algorithm,
            "feature_schema_version": self.feature_schema_version,
            "supported_decision_types": list(self.supported_decision_types),
        }


def reward_candidate_id(observation: StepObservation, candidate: CandidateAction) -> str | None:
    if candidate.action == "skip_reward_cards":
        return "skip"
    if candidate.request.option_index is None:
        return None
    reward = observation.state.reward
    if reward is not None:
        option = next((item for item in reward.card_options if item.index == candidate.request.option_index), None)
        if option is not None:
            return option.card_id
    selection = observation.state.selection
    if selection is None:
        return None
    option = next((item for item in selection.cards if item.index == candidate.request.option_index), None)
    return None if option is None else option.card_id


def shop_buy_candidate_id(observation: StepObservation, candidate: CandidateAction) -> str | None:
    shop = observation.state.shop
    if shop is None or candidate.request.option_index is None:
        return None
    option = next((item for item in shop.cards if item.index == candidate.request.option_index), None)
    return None if option is None else option.card_id


def event_candidate_id(observation: StepObservation, candidate: CandidateAction) -> str | None:
    event = observation.state.event
    if event is None or candidate.request.option_index is None:
        return None
    option = next((item for item in event.options if item.index == candidate.request.option_index), None)
    if option is None:
        return None
    return option.text_key or _slug_token(option.title)


def rest_candidate_id(observation: StepObservation, candidate: CandidateAction) -> str | None:
    rest = observation.state.rest
    if rest is None or candidate.request.option_index is None:
        return None
    option = next((item for item in rest.options if item.index == candidate.request.option_index), None)
    return None if option is None else option.option_id


def selection_candidate_id(observation: StepObservation, candidate: CandidateAction) -> str | None:
    selection = observation.state.selection
    if selection is None or candidate.request.option_index is None:
        return None
    option = next((item for item in selection.cards if item.index == candidate.request.option_index), None)
    return None if option is None else option.card_id


def selection_mode(observation: StepObservation) -> str:
    selection = observation.state.selection
    if selection is None:
        return "pick"
    mode = str(selection.semantic_mode or "").strip().lower()
    return mode or "pick"


def selection_decision_type(observation: StepObservation) -> str | None:
    return selection_decision_type_for_mode(selection_mode(observation))


def selection_decision_type_for_mode(mode: str) -> str | None:
    normalized = str(mode).strip().lower()
    return _SELECTION_DECISION_TYPES.get(normalized)


def reward_runtime_domain(observation: StepObservation) -> tuple[str, str]:
    reward = observation.state.reward
    normalized_source_type = str("" if reward is None else reward.source_type).strip().lower()
    if normalized_source_type:
        if normalized_source_type == "combat":
            return "monster", "monster"
        return normalized_source_type, normalized_source_type
    if observation.state.event is not None:
        return "event", "event"
    if observation.state.chest is not None:
        return "treasure", "treasure"
    return "monster", "monster"


def _runtime_context_payload(
    *,
    observation: StepObservation,
    config: StrategicRuntimeConfig,
    decision_type: str,
    candidate_count: int,
    room_type: str,
    map_point_type: str,
    source_type: str,
    checkpoint_algorithm: str,
) -> dict[str, Any]:
    summary = build_state_summary(observation)
    run_summary = dict(summary.get("run", {}))
    build_summary = dict(summary.get("build", {}))
    selection_summary = dict(summary.get("selection", {}))
    floor = run_summary.get("floor")
    act_index = run_summary.get("act_index")
    act_number = run_summary.get("act_number")
    floor_within_act = floor
    acts_reached: int | None = None
    if act_number is not None:
        acts_reached = int(act_number)
    elif act_index is not None:
        acts_reached = int(act_index) + 1
    return {
        "source_name": config.runtime_source_name,
        "character_id": run_summary.get("character_id"),
        "ascension": run_summary.get("ascension"),
        "build_id": config.runtime_build_id or ((observation.state.build.build_id if observation.state.build is not None else None)),
        "game_version": build_summary.get("game_version"),
        "branch": build_summary.get("branch"),
        "content_channel": build_summary.get("content_channel"),
        "game_mode": config.runtime_game_mode,
        "platform_type": config.runtime_platform_type,
        "acts_reached": acts_reached,
        "act_index": act_index,
        "act_id": run_summary.get("act_id"),
        "floor": floor,
        "floor_within_act": floor_within_act,
        "room_type": room_type,
        "map_point_type": map_point_type,
        "source_type": source_type,
        "support_quality": config.support_quality,
        "candidate_count": candidate_count,
        "metadata": {
            "artifact_family": "live_runtime_strategic_guidance",
            "checkpoint_algorithm": checkpoint_algorithm,
            "has_detail_payload": True,
            "has_room_history": False,
            "decision_type": decision_type,
            "selection_semantic_mode": selection_summary.get("semantic_mode"),
            "selection_source_type": selection_summary.get("source_type"),
            "selection_required_count": selection_summary.get("required_count"),
            "selection_selected_count": selection_summary.get("selected_count"),
            "selection_remaining_count": selection_summary.get("remaining_count"),
            "selection_supports_multi_select": selection_summary.get("supports_multi_select"),
        },
    }


def _normalized_optional(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _slug_token(value: Any) -> str:
    normalized = _normalized_optional(value)
    if normalized is None:
        return ""
    return normalized.replace(" ", "_")
