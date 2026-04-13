from __future__ import annotations

from collections.abc import Callable
from time import monotonic, sleep
from typing import Any

from sts2_rl.lifecycle import observation_signature

from .candidate_actions import build_candidate_actions
from .client import Sts2ApiError, Sts2Client
from .models import ActionRequest, ActionResponsePayload, AvailableActionsPayload, GameStatePayload
from .types import CandidateAction, StepObservation, StepResult

RewardFn = Callable[
    [StepObservation | None, CandidateAction | None, ActionResponsePayload | None, StepObservation],
    float,
]


class Sts2Env:
    def __init__(
        self,
        client: Sts2Client,
        *,
        reward_fn: RewardFn | None = None,
        transition_timeout: float = 5.0,
        transition_poll_interval: float = 0.1,
        action_retry_attempts: int = 1,
    ) -> None:
        self._client = client
        self._reward_fn = reward_fn
        self._last_observation: StepObservation | None = None
        self._transition_timeout = transition_timeout
        self._transition_poll_interval = transition_poll_interval
        self._action_retry_attempts = max(0, action_retry_attempts)

    @classmethod
    def from_base_url(
        cls,
        base_url: str,
        *,
        reward_fn: RewardFn | None = None,
        timeout: float = 20.0,
        transition_timeout: float = 5.0,
        transition_poll_interval: float = 0.1,
        action_retry_attempts: int = 1,
    ) -> Sts2Env:
        return cls(
            Sts2Client(base_url, timeout=timeout),
            reward_fn=reward_fn,
            transition_timeout=transition_timeout,
            transition_poll_interval=transition_poll_interval,
            action_retry_attempts=action_retry_attempts,
        )

    def close(self) -> None:
        self._client.close()

    def observe(self) -> StepObservation:
        state = self._client.get_state()
        descriptors = self._client.get_available_actions()
        observation = self._build_observation(state, descriptors)
        self._last_observation = observation
        return observation

    def reset(self) -> StepObservation:
        return self.observe()

    def step(self, action: CandidateAction | ActionRequest) -> StepResult:
        previous = self._last_observation
        candidate = action if isinstance(action, CandidateAction) else None
        request = action.request if isinstance(action, CandidateAction) else action

        response, recovery_info = self._post_action_with_recovery(request, previous)
        descriptors = self._client.get_available_actions()
        observation = self._build_observation(response.state, descriptors)
        transition_waited = False
        transition_settled = False
        if not response.stable:
            transition_waited = True
            observation, transition_settled = self._wait_for_transition(previous, observation)
        self._last_observation = observation

        reward = 0.0
        if self._reward_fn is not None:
            reward = self._reward_fn(previous, candidate, response, observation)

        action_status = response.status
        action_stable = response.stable
        action_message = response.message
        if transition_settled:
            action_status = "completed"
            action_stable = True
            action_message = "Action completed after transition polling."

        return StepResult(
            observation=observation,
            reward=reward,
            terminated=observation.screen_type == "GAME_OVER",
            truncated=False,
            response=response,
            info={
                "action_status": action_status,
                "action_stable": action_stable,
                "action_message": action_message,
                "response_action_status": response.status,
                "response_action_stable": response.stable,
                "response_action_message": response.message,
                "transition_waited": transition_waited,
                "transition_settled": transition_settled,
                **recovery_info,
            },
        )

    def _post_action_with_recovery(
        self,
        request: ActionRequest,
        previous: StepObservation | None,
    ) -> tuple[ActionResponsePayload, dict[str, Any]]:
        attempts_remaining = self._action_retry_attempts
        while True:
            try:
                return self._client.post_action(request), {"recovered_from_action_error": False}
            except Sts2ApiError as exc:
                if not self._is_retryable_action_error(exc):
                    raise
                recovered = self._recover_action_progress(previous, request=request, error=exc)
                if recovered is not None:
                    return recovered, {
                        "recovered_from_action_error": True,
                        "action_error_code": exc.code,
                        "action_error_message": exc.message,
                    }
                if attempts_remaining <= 0:
                    raise
                attempts_remaining -= 1

    def _recover_action_progress(
        self,
        previous: StepObservation | None,
        *,
        request: ActionRequest,
        error: Sts2ApiError,
    ) -> ActionResponsePayload | None:
        deadline = monotonic() + max(0.0, self._transition_timeout)
        baseline_signature = observation_signature(previous) if previous is not None else None

        while monotonic() < deadline:
            try:
                state = self._client.get_state()
                descriptors = self._client.get_available_actions()
            except Exception:
                sleep(self._transition_poll_interval)
                continue
            observation = self._build_observation(state, descriptors)
            if baseline_signature is None or observation_signature(observation) != baseline_signature:
                return ActionResponsePayload(
                    action=request.action,
                    status="completed",
                    stable=True,
                    message=f"Recovered after action API error: {error.message}",
                    state=observation.state,
                )
            sleep(self._transition_poll_interval)
        return None

    @staticmethod
    def _is_retryable_action_error(error: Sts2ApiError) -> bool:
        return error.retryable or error.code == "internal_error"

    def _wait_for_transition(
        self,
        previous: StepObservation | None,
        observation: StepObservation,
    ) -> tuple[StepObservation, bool]:
        deadline = monotonic() + max(0.0, self._transition_timeout)
        baseline_signature = observation_signature(previous) if previous is not None else observation_signature(observation)
        latest = observation

        while monotonic() < deadline:
            if observation_signature(latest) != baseline_signature:
                return latest, True
            sleep(self._transition_poll_interval)
            state = self._client.get_state()
            descriptors = self._client.get_available_actions()
            latest = self._build_observation(state, descriptors)

        return latest, False

    def _build_observation(
        self,
        state: GameStatePayload,
        descriptors: AvailableActionsPayload,
    ) -> StepObservation:
        build_result = build_candidate_actions(state, descriptors)
        return StepObservation(
            screen_type=state.screen,
            run_id=state.run_id,
            state=state,
            action_descriptors=descriptors,
            legal_actions=build_result.candidates,
            build_warnings=build_result.unsupported_actions,
        )
