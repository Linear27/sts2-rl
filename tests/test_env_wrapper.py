from sts2_rl.env.client import Sts2ApiError
from sts2_rl.env.models import (
    ActionDescriptor,
    ActionRequest,
    ActionResponsePayload,
    AvailableActionsPayload,
    ChestPayload,
    ChestRelicOptionPayload,
    GameStatePayload,
)
from sts2_rl.env.wrapper import Sts2Env


class _FakeClient:
    def __init__(
        self,
        *,
        action_response: ActionResponsePayload,
        polled_states: list[GameStatePayload],
        polled_actions: list[AvailableActionsPayload],
        initial_actions: AvailableActionsPayload,
        post_action_actions: AvailableActionsPayload | None = None,
        post_action_errors: list[Exception] | None = None,
    ) -> None:
        self._action_response = action_response
        self._polled_states = list(polled_states)
        self._polled_actions = list(polled_actions)
        self._initial_actions = initial_actions
        self._post_action_actions = post_action_actions or initial_actions
        self._post_action_errors = list(post_action_errors or [])
        self.posted_requests: list[ActionRequest] = []
        self.closed = False
        self._post_action_actions_consumed = False

    def get_state(self) -> GameStatePayload:
        if not self._polled_states:
            return self._action_response.state
        return self._polled_states.pop(0)

    def get_available_actions(self) -> AvailableActionsPayload:
        if not self.posted_requests:
            return self._initial_actions
        if not self._post_action_actions_consumed:
            self._post_action_actions_consumed = True
            return self._post_action_actions
        if not self._polled_actions:
            return self._post_action_actions
        return self._polled_actions.pop(0)

    def post_action(self, request: ActionRequest) -> ActionResponsePayload:
        self.posted_requests.append(request)
        if self._post_action_errors:
            raise self._post_action_errors.pop(0)
        return self._action_response

    def close(self) -> None:
        self.closed = True


def test_env_step_polls_until_pending_transition_changes_observation() -> None:
    initial_state = GameStatePayload(screen="CARD_SELECTION", run_id="run-1")
    settled_state = GameStatePayload(screen="MAP", run_id="run-1")
    initial_actions = AvailableActionsPayload(
        screen="CARD_SELECTION",
        actions=[ActionDescriptor(name="select_deck_card", requires_index=True)],
    )
    settled_actions = AvailableActionsPayload(
        screen="MAP",
        actions=[ActionDescriptor(name="choose_map_node", requires_index=True)],
    )
    client = _FakeClient(
        action_response=ActionResponsePayload(
            action="select_deck_card",
            status="pending",
            stable=False,
            message="Action queued but state is still transitioning.",
            state=initial_state,
        ),
        polled_states=[settled_state],
        polled_actions=[settled_actions],
        initial_actions=initial_actions,
        post_action_actions=initial_actions,
    )
    env = Sts2Env(client, transition_timeout=0.5, transition_poll_interval=0.0)
    env._last_observation = env._build_observation(initial_state, initial_actions)

    result = env.step(ActionRequest(action="select_deck_card", option_index=0))

    assert result.observation.screen_type == "MAP"
    assert result.observation.action_descriptors.screen == "MAP"
    assert result.info["action_status"] == "completed"
    assert result.info["action_stable"] is True
    assert result.info["response_action_status"] == "pending"
    assert result.info["response_action_stable"] is False
    assert result.info["transition_waited"] is True
    assert result.info["transition_settled"] is True
    assert client.posted_requests[0].option_index == 0


def test_env_step_returns_latest_observation_when_pending_transition_never_settles() -> None:
    initial_state = GameStatePayload(screen="CARD_SELECTION", run_id="run-2")
    initial_actions = AvailableActionsPayload(
        screen="CARD_SELECTION",
        actions=[ActionDescriptor(name="select_deck_card", requires_index=True)],
    )
    client = _FakeClient(
        action_response=ActionResponsePayload(
            action="select_deck_card",
            status="pending",
            stable=False,
            message="Action queued but state is still transitioning.",
            state=initial_state,
        ),
        polled_states=[initial_state, initial_state],
        polled_actions=[initial_actions, initial_actions],
        initial_actions=initial_actions,
        post_action_actions=initial_actions,
    )
    env = Sts2Env(client, transition_timeout=0.01, transition_poll_interval=0.0)
    env._last_observation = env._build_observation(initial_state, initial_actions)

    result = env.step(ActionRequest(action="select_deck_card", option_index=0))

    assert result.observation.screen_type == "CARD_SELECTION"
    assert result.info["action_status"] == "pending"
    assert result.info["action_stable"] is False
    assert result.info["transition_waited"] is True
    assert result.info["transition_settled"] is False


def test_env_step_recovers_from_retryable_action_error_after_state_progresses() -> None:
    initial_state = GameStatePayload(screen="CHEST", run_id="run-3")
    settled_state = GameStatePayload(
        screen="CHEST",
        run_id="run-3",
        chest=ChestPayload(
            is_opened=True,
            has_relic_been_claimed=False,
            relic_options=[ChestRelicOptionPayload(index=0, relic_id="KUSARIGAMA", name="Kusarigama", rarity="Uncommon")],
        ),
    )
    initial_actions = AvailableActionsPayload(screen="CHEST", actions=[ActionDescriptor(name="open_chest")])
    settled_actions = AvailableActionsPayload(
        screen="CHEST",
        actions=[ActionDescriptor(name="choose_treasure_relic", requires_index=True)],
    )
    client = _FakeClient(
        action_response=ActionResponsePayload(
            action="open_chest",
            status="completed",
            stable=True,
            message="Action completed.",
            state=settled_state,
        ),
        polled_states=[settled_state],
        polled_actions=[settled_actions],
        initial_actions=initial_actions,
        post_action_actions=settled_actions,
        post_action_errors=[
            Sts2ApiError(
                status_code=500,
                code="internal_error",
                message="Unhandled server error.",
                retryable=False,
            )
        ],
    )
    env = Sts2Env(client, transition_timeout=0.05, transition_poll_interval=0.0)
    env._last_observation = env._build_observation(initial_state, initial_actions)

    result = env.step(ActionRequest(action="open_chest"))

    assert result.observation.screen_type == "CHEST"
    assert result.observation.legal_actions[0].action_id == "choose_treasure_relic|option=0"
    assert result.info["action_status"] == "completed"
    assert result.info["recovered_from_action_error"] is True
    assert result.info["action_error_code"] == "internal_error"
