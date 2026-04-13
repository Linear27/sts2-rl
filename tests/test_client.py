import httpx

from sts2_rl.env.client import Sts2Client


def test_client_parses_health_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(
            200,
            json={
                "ok": True,
                "request_id": "req_1",
                "data": {
                    "service": "sts2-ai-agent",
                    "mod_version": "0.5.4",
                    "protocol_version": "2026-04-14-v2",
                    "game_version": "v0.103.0",
                    "build_id": "v0.103.0",
                    "branch": "public_beta",
                    "content_channel": "beta",
                    "commit": "abc1234",
                    "build_date": "2026-04-14T00:00:00.0000000Z",
                    "main_assembly_hash": "424242",
                    "status": "ready",
                },
            },
        )

    client = Sts2Client("http://127.0.0.1:8080/", transport=httpx.MockTransport(handler))
    try:
        health = client.get_health()
    finally:
        client.close()

    assert health.service == "sts2-ai-agent"
    assert health.status == "ready"
    assert health.build_id == "v0.103.0"
    assert health.branch == "public_beta"
    assert health.content_channel == "beta"


def test_client_parses_state_build_and_selection_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/state"
        return httpx.Response(
            200,
            json={
                "ok": True,
                "request_id": "req_2",
                "data": {
                    "state_version": 10,
                    "run_id": "run-1",
                    "screen": "SELECTION",
                    "build": {
                        "build_id": "v0.103.0",
                        "game_version": "v0.103.0",
                        "branch": "public_beta",
                        "content_channel": "beta",
                        "commit": "abc1234",
                        "build_date": "2026-04-14T00:00:00.0000000Z",
                        "main_assembly_hash": "424242",
                    },
                    "selection": {
                        "kind": "deck_card_select",
                        "selection_family": "deck",
                        "semantic_mode": "remove",
                        "source_type": "shop",
                        "source_room_type": "Shop",
                        "source_action": "choose_event_option",
                        "source_event_id": "EVENT.EPIC_QUEST",
                        "source_event_option_index": 1,
                        "source_event_option_text_key": "event.quest.solo",
                        "source_event_option_title": "Solo Quest",
                        "prompt": "Remove a card from your deck",
                        "prompt_loc_table": "card_selection",
                        "prompt_loc_key": "remove",
                        "min_select": 1,
                        "max_select": 1,
                        "required_count": 1,
                        "remaining_count": 1,
                        "selected_count": 0,
                        "requires_confirmation": False,
                        "can_confirm": False,
                        "supports_multi_select": False,
                        "cards": [
                            {"index": 0, "card_id": "STRIKE_IRONCLAD", "name": "Strike", "upgraded": False}
                        ],
                    },
                },
            },
        )

    client = Sts2Client("http://127.0.0.1:8080/", transport=httpx.MockTransport(handler))
    try:
        state = client.get_state()
    finally:
        client.close()

    assert state.build is not None
    assert state.build.build_id == "v0.103.0"
    assert state.build.branch == "public_beta"
    assert state.selection is not None
    assert state.selection.semantic_mode == "remove"
    assert state.selection.source_type == "shop"
    assert state.selection.source_action == "choose_event_option"
    assert state.selection.source_event_id == "EVENT.EPIC_QUEST"
    assert state.selection.required_count == 1
