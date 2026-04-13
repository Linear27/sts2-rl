import httpx

from sts2_rl.benchmark import benchmark_observe_with_client, summarize_measurements
from sts2_rl.env.client import Sts2Client


def test_summarize_measurements_computes_percentiles() -> None:
    summary = summarize_measurements([10.0, 20.0, 30.0, 40.0, 50.0])

    assert summary.count == 5
    assert summary.min_ms == 10.0
    assert summary.max_ms == 50.0
    assert summary.p50_ms == 30.0
    assert summary.p95_ms > 40.0


def test_benchmark_observe_with_mock_client() -> None:
    calls = {"state": 0, "actions": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/state":
            calls["state"] += 1
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "request_id": f"state_{calls['state']}",
                    "data": {
                        "state_version": 8,
                        "run_id": "run_x",
                        "screen": "COMBAT",
                        "session": {"mode": "singleplayer", "phase": "run", "control_scope": "local_player"},
                        "combat": {
                            "player": {"current_hp": 50, "max_hp": 80, "block": 0, "energy": 3, "stars": 0, "focus": 0},
                            "hand": [
                                {
                                    "index": 0,
                                    "card_id": "strike",
                                    "name": "Strike",
                                    "playable": True,
                                    "requires_target": False,
                                    "valid_target_indices": [],
                                }
                            ],
                            "enemies": [],
                        },
                        "run": {
                            "character_id": "ironclad",
                            "character_name": "Ironclad",
                            "ascension": 0,
                            "floor": 1,
                            "current_hp": 80,
                            "max_hp": 80,
                            "gold": 99,
                            "max_energy": 3,
                            "potions": [],
                        },
                    },
                },
            )
        if request.url.path == "/actions/available":
            calls["actions"] += 1
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "request_id": f"actions_{calls['actions']}",
                    "data": {
                        "screen": "COMBAT",
                        "actions": [
                            {"name": "play_card", "requires_index": True, "requires_target": False},
                            {"name": "end_turn", "requires_index": False, "requires_target": False},
                        ],
                    },
                },
            )
        raise AssertionError(f"Unexpected path: {request.url.path}")

    client = Sts2Client("http://127.0.0.1:8080", transport=httpx.MockTransport(handler))
    try:
        result = benchmark_observe_with_client(client, samples=3)
    finally:
        client.close()

    assert result.last_screen == "COMBAT"
    assert result.candidate_count.count == 3
    assert result.candidate_count.min_ms >= 2.0
