from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .models import ActionRequest, ActionResponsePayload, AvailableActionsPayload, GameStatePayload


class CandidateAction(BaseModel):
    action_id: str = Field(description="Stable action identifier for a fully bound action candidate")
    action: str
    label: str
    request: ActionRequest
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateBuildResult(BaseModel):
    candidates: list[CandidateAction] = Field(default_factory=list)
    unsupported_actions: list[str] = Field(default_factory=list)


class StepObservation(BaseModel):
    screen_type: str
    run_id: str = "run_unknown"
    state: GameStatePayload
    action_descriptors: AvailableActionsPayload
    legal_actions: list[CandidateAction] = Field(default_factory=list)
    build_warnings: list[str] = Field(default_factory=list)


class StepResult(BaseModel):
    observation: StepObservation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    response: ActionResponsePayload | None = None
    info: dict[str, Any] = Field(default_factory=dict)
