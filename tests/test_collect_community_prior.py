import json
from pathlib import Path

from sts2_rl.collect import CommunityCardPriorSource, CommunityPriorRuntimeConfig, build_policy_pack
from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    GameStatePayload,
    MapGraphNodePayload,
    MapNodePayload,
    MapPayload,
    RewardCardOptionPayload,
    RewardPayload,
    RunPayload,
    SelectionCardPayload,
    SelectionPayload,
    ShopCardPayload,
    ShopPayload,
)
from sts2_rl.env.types import StepObservation


def test_community_prior_source_scores_reward_shop_and_remove_domains(tmp_path: Path) -> None:
    source_path = _write_community_prior_fixture(tmp_path)
    source = CommunityCardPriorSource.from_config(
        CommunityPriorRuntimeConfig(
            source_path=source_path,
            reward_pick_weight=5.0,
            selection_remove_weight=5.0,
            shop_buy_weight=5.0,
        )
    )

    assert source is not None
    reward_prior = source.score_card(
        domain="reward_pick",
        card_id="CARD_B",
        character_id="IRONCLAD",
        ascension=0,
        act_id="act1",
        floor=4,
    )
    remove_prior = source.score_card(
        domain="selection_remove",
        card_id="CARD_KEEP",
        character_id="IRONCLAD",
        ascension=0,
        act_id="act1",
        floor=4,
    )
    shop_prior = source.score_card(
        domain="shop_buy",
        card_id="CARD_SHOP_A",
        character_id="IRONCLAD",
        ascension=0,
        act_id="act1",
        floor=4,
    )

    assert reward_prior is not None
    assert reward_prior.score_bonus > 0.0
    assert reward_prior.pick_rate == 0.75
    assert remove_prior is not None
    assert remove_prior.score_bonus < 0.0
    assert shop_prior is not None
    assert shop_prior.score_bonus > 0.0


def test_policy_community_prior_flips_reward_shop_and_remove_decisions(tmp_path: Path) -> None:
    source_path = _write_community_prior_fixture(tmp_path)
    policy = build_policy_pack(
        "baseline",
        community_prior_config=CommunityPriorRuntimeConfig(
            source_path=source_path,
            reward_pick_weight=5.0,
            selection_pick_weight=5.0,
            selection_upgrade_weight=2.0,
            selection_remove_weight=5.0,
            shop_buy_weight=5.0,
        ),
    )

    reward_observation = _build_observation(
        GameStatePayload(
            screen="REWARD",
            run_id="run-reward-prior",
            run=_run_payload(),
            reward=RewardPayload(
                pending_card_choice=True,
                card_options=[
                    RewardCardOptionPayload(index=0, card_id="CARD_A", name="Card A"),
                    RewardCardOptionPayload(index=1, card_id="CARD_B", name="Card B"),
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="REWARD",
            actions=[ActionDescriptor(name="choose_reward_card", requires_index=True)],
        ),
    )
    reward_decision = policy.choose(reward_observation)
    assert reward_decision.action is not None
    assert reward_decision.action.request.option_index == 1
    assert reward_decision.trace_metadata["community_prior"]["selected"]["prior"]["card_id"] == "CARD_B"

    selection_observation = _build_observation(
        GameStatePayload(
            screen="SELECTION",
            run_id="run-selection-prior",
            run=_run_payload(),
            selection=SelectionPayload(
                kind="remove",
                selection_family="deck",
                semantic_mode="remove",
                source_type="event",
                prompt="Remove a card from your deck",
                min_select=1,
                max_select=1,
                required_count=1,
                cards=[
                    SelectionCardPayload(index=0, card_id="CARD_KEEP", name="Card Keep"),
                    SelectionCardPayload(index=1, card_id="CARD_CUT", name="Card Cut"),
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="SELECTION",
            actions=[ActionDescriptor(name="select_deck_card", requires_index=True)],
        ),
    )
    selection_decision = policy.choose(selection_observation)
    assert selection_decision.action is not None
    assert selection_decision.action.request.option_index == 1
    assert selection_decision.trace_metadata["community_prior"]["selected"]["prior"]["card_id"] == "CARD_CUT"

    shop_observation = _build_observation(
        GameStatePayload(
            screen="SHOP",
            run_id="run-shop-prior",
            run=_run_payload(gold=200),
            shop=ShopPayload(
                is_open=True,
                can_open=False,
                cards=[
                    ShopCardPayload(index=0, card_id="CARD_SHOP_A", name="Shop A", price=60, is_stocked=True, enough_gold=True),
                    ShopCardPayload(index=1, card_id="CARD_SHOP_B", name="Shop B", price=60, is_stocked=True, enough_gold=True),
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="SHOP",
            actions=[ActionDescriptor(name="buy_card", requires_index=True), ActionDescriptor(name="close_shop_inventory")],
        ),
    )
    shop_decision = policy.choose(shop_observation)
    assert shop_decision.action is not None
    assert shop_decision.action.action == "buy_card"
    assert shop_decision.action.request.option_index == 0
    assert shop_decision.trace_metadata["community_prior"]["selected"]["prior"]["card_id"] == "CARD_SHOP_A"


def test_community_prior_source_matches_metadata_aliases(tmp_path: Path) -> None:
    source_path = tmp_path / "community-card-stats.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "record_type": "community_card_stat",
                "schema_version": 1,
                "source_name": "spiremeta",
                "snapshot_date": "2026-04-13",
                "character_id": "IRONCLAD",
                "source_type": "reward",
                "card_id": "STRIKE_IRONCLAD",
                "offer_count": 100,
                "pick_count": 50,
                "pick_rate": 0.5,
                "metadata": {"aliases": ["STRIKE", "strike_ironclad"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = CommunityCardPriorSource.from_config(
        CommunityPriorRuntimeConfig(
            source_path=source_path,
            reward_pick_weight=1.0,
        )
    )

    assert source is not None
    strike_prior = source.score_card(
        domain="reward_pick",
        card_id="STRIKE",
        character_id="IRONCLAD",
        ascension=0,
        act_id=None,
        floor=1,
    )
    starter_prior = source.score_card(
        domain="reward_pick",
        card_id="STRIKE_IRONCLAD",
        character_id="IRONCLAD",
        ascension=0,
        act_id=None,
        floor=1,
    )

    assert strike_prior is not None
    assert starter_prior is not None
    assert strike_prior.pick_rate == 0.5
    assert starter_prior.pick_rate == 0.5


def test_community_prior_source_loads_public_run_strategic_card_and_route_stats(tmp_path: Path) -> None:
    strategic_dir = _write_public_run_prior_fixture(tmp_path)
    source = CommunityCardPriorSource.from_config(
        CommunityPriorRuntimeConfig(
            source_path=strategic_dir,
            route_source_path=strategic_dir,
            reward_pick_weight=2.0,
            shop_buy_weight=2.0,
            route_weight=2.0,
            max_source_age_days=30,
        )
    )

    assert source is not None
    reward_prior = source.score_card(
        domain="reward_pick",
        card_id="CARD_REWARD",
        character_id="IRONCLAD",
        ascension=0,
        act_id="ACT_1",
        floor=4,
    )
    shop_prior = source.score_card(
        domain="shop_buy",
        card_id="CARD_SHOP",
        character_id="IRONCLAD",
        ascension=0,
        act_id="ACT_1",
        floor=4,
    )
    route_prior = source.score_route(subject_id="shop", character_id="IRONCLAD", act_id="ACT_1")

    assert reward_prior is not None
    assert reward_prior.source_name == "sts2runs"
    assert reward_prior.pick_rate == 0.75
    assert shop_prior is not None
    assert shop_prior.buy_rate == 0.6
    assert route_prior is not None
    assert route_prior.win_delta == 0.2
    assert route_prior.score_bonus > 0.0
    diagnostics = source.diagnostics()
    assert diagnostics["card"]["artifact_family"] == "public_run_strategic_stats"
    assert diagnostics["route"]["stat_family"] == "route"


def test_policy_community_route_prior_flips_map_choice(tmp_path: Path) -> None:
    strategic_dir = _write_public_run_prior_fixture(tmp_path)
    policy = build_policy_pack(
        "baseline",
        community_prior_config=CommunityPriorRuntimeConfig(
            source_path=strategic_dir,
            route_source_path=strategic_dir,
            route_weight=4.0,
            route_win_rate_scale=20.0,
            min_sample_size=1,
            route_min_sample_size=1,
            max_confidence_sample_size=10,
        ),
    )
    observation = _build_observation(
        GameStatePayload(
            screen="MAP",
            run_id="run-map-prior",
            run=RunPayload(
                character_id="IRONCLAD",
                ascension=0,
                floor=10,
                act_id="ACT_1",
                current_hp=70,
                max_hp=80,
                gold=50,
                max_energy=3,
                boss_encounter_id="THE_GUARDIAN",
                boss_encounter_name="The Guardian",
            ),
            map=MapPayload(
                available_nodes=[
                    MapNodePayload(index=0, row=2, col=0, node_type="Elite", state="Travelable"),
                    MapNodePayload(index=1, row=2, col=1, node_type="Shop", state="Travelable"),
                ],
                is_travel_enabled=True,
                nodes=[
                    MapGraphNodePayload(row=2, col=0, node_type="Elite", state="Travelable", children=[]),
                    MapGraphNodePayload(row=2, col=1, node_type="Shop", state="Travelable", children=[]),
                ],
            ),
        ),
        AvailableActionsPayload(screen="MAP", actions=[ActionDescriptor(name="choose_map_node", requires_index=True)]),
    )

    decision = policy.choose(observation)

    assert decision.action is not None
    assert decision.action.request.option_index == 1
    assert decision.trace_metadata["community_prior"]["selected"]["prior"]["domain"] == "map_node"
    assert decision.trace_metadata["community_prior"]["selected"]["prior"]["subject_id"] == "shop"


def _write_community_prior_fixture(tmp_path: Path) -> Path:
    source_path = tmp_path / "community-card-stats.jsonl"
    records = [
        {
            "record_type": "community_card_stat",
            "schema_version": 1,
            "source_name": "fixture",
            "snapshot_date": "2026-04-13",
            "character_id": "IRONCLAD",
            "ascension_min": 0,
            "ascension_max": 0,
            "act_id": "act1",
            "source_type": "reward",
            "card_id": "CARD_A",
            "offer_count": 200,
            "pick_count": 20,
            "pick_rate": 0.10,
            "run_count": 200,
            "win_delta": -0.03,
        },
        {
            "record_type": "community_card_stat",
            "schema_version": 1,
            "source_name": "fixture",
            "snapshot_date": "2026-04-13",
            "character_id": "IRONCLAD",
            "ascension_min": 0,
            "ascension_max": 0,
            "act_id": "act1",
            "source_type": "reward",
            "card_id": "CARD_B",
            "offer_count": 200,
            "pick_count": 150,
            "pick_rate": 0.75,
            "run_count": 200,
            "win_delta": 0.08,
        },
        {
            "record_type": "community_card_stat",
            "schema_version": 1,
            "source_name": "fixture",
            "snapshot_date": "2026-04-13",
            "character_id": "IRONCLAD",
            "ascension_min": 0,
            "ascension_max": 0,
            "act_id": "act1",
            "source_type": "reward",
            "card_id": "CARD_KEEP",
            "offer_count": 200,
            "pick_count": 140,
            "pick_rate": 0.70,
            "run_count": 200,
            "win_delta": 0.06,
        },
        {
            "record_type": "community_card_stat",
            "schema_version": 1,
            "source_name": "fixture",
            "snapshot_date": "2026-04-13",
            "character_id": "IRONCLAD",
            "ascension_min": 0,
            "ascension_max": 0,
            "act_id": "act1",
            "source_type": "reward",
            "card_id": "CARD_CUT",
            "offer_count": 200,
            "pick_count": 10,
            "pick_rate": 0.05,
            "run_count": 200,
            "win_delta": -0.07,
        },
        {
            "record_type": "community_card_stat",
            "schema_version": 1,
            "source_name": "fixture",
            "snapshot_date": "2026-04-13",
            "character_id": "IRONCLAD",
            "ascension_min": 0,
            "ascension_max": 0,
            "act_id": "act1",
            "source_type": "shop",
            "card_id": "CARD_SHOP_A",
            "shop_offer_count": 200,
            "buy_count": 80,
            "buy_rate": 0.40,
            "run_count": 200,
            "win_delta": 0.07,
        },
        {
            "record_type": "community_card_stat",
            "schema_version": 1,
            "source_name": "fixture",
            "snapshot_date": "2026-04-13",
            "character_id": "IRONCLAD",
            "ascension_min": 0,
            "ascension_max": 0,
            "act_id": "act1",
            "source_type": "shop",
            "card_id": "CARD_SHOP_B",
            "shop_offer_count": 200,
            "buy_count": 5,
            "buy_rate": 0.025,
            "run_count": 200,
            "win_delta": -0.04,
        },
    ]
    source_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return source_path


def _write_public_run_prior_fixture(tmp_path: Path) -> Path:
    output_dir = tmp_path / "public-run-strategic"
    output_dir.mkdir(parents=True, exist_ok=True)
    strategic_cards = [
        {
            "record_type": "public_run_strategic_stat",
            "schema_version": 1,
            "stat_family": "card",
            "source_name": "sts2runs",
            "snapshot_date": "2026-04-14",
            "subject_id": "CARD_REWARD",
            "character_id": "IRONCLAD",
            "act_id": "ACT_1",
            "source_type": "reward",
            "run_count": 80,
            "offer_count": 80,
            "pick_count": 60,
            "pick_rate": 0.75,
            "win_count": 44,
            "win_rate": 0.55,
            "win_rate_with_card": 0.65,
            "win_delta": 0.10,
            "metadata": {"artifact_family": "public_run_strategic_card_stats"},
        },
        {
            "record_type": "public_run_strategic_stat",
            "schema_version": 1,
            "stat_family": "card",
            "source_name": "sts2runs",
            "snapshot_date": "2026-04-14",
            "subject_id": "CARD_SHOP",
            "character_id": "IRONCLAD",
            "act_id": "ACT_1",
            "source_type": "shop",
            "run_count": 50,
            "shop_offer_count": 50,
            "buy_count": 30,
            "buy_rate": 0.6,
            "win_count": 30,
            "win_rate": 0.6,
            "win_rate_with_card": 0.7,
            "win_delta": 0.15,
            "metadata": {"artifact_family": "public_run_strategic_card_stats"},
        },
    ]
    strategic_routes = [
        {
            "record_type": "public_run_strategic_stat",
            "schema_version": 1,
            "stat_family": "route",
            "source_name": "sts2runs",
            "snapshot_date": "2026-04-14",
            "subject_id": "shop",
            "character_id": "IRONCLAD",
            "act_id": "ACT_1",
            "room_type": "shop",
            "run_count": 40,
            "seen_count": 40,
            "win_count": 28,
            "win_rate": 0.7,
            "win_delta": 0.2,
            "metadata": {"artifact_family": "public_run_strategic_route_stats"},
        },
        {
            "record_type": "public_run_strategic_stat",
            "schema_version": 1,
            "stat_family": "route",
            "source_name": "sts2runs",
            "snapshot_date": "2026-04-14",
            "subject_id": "elite",
            "character_id": "IRONCLAD",
            "act_id": "ACT_1",
            "room_type": "elite",
            "run_count": 40,
            "seen_count": 40,
            "win_count": 16,
            "win_rate": 0.4,
            "win_delta": -0.1,
            "metadata": {"artifact_family": "public_run_strategic_route_stats"},
        },
    ]
    (output_dir / "strategic-card-stats.jsonl").write_text(
        "\n".join(json.dumps(record) for record in strategic_cards),
        encoding="utf-8",
    )
    (output_dir / "strategic-route-stats.jsonl").write_text(
        "\n".join(json.dumps(record) for record in strategic_routes),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps({"generated_at_utc": "2026-04-14T00:00:00+00:00"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir


def _run_payload(*, gold: int = 120) -> RunPayload:
    return RunPayload(
        character_id="IRONCLAD",
        ascension=0,
        floor=4,
        act_id="act1",
        current_hp=60,
        max_hp=80,
        gold=gold,
        max_energy=3,
    )


def _build_observation(state: GameStatePayload, descriptors: AvailableActionsPayload) -> StepObservation:
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type=state.screen,
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )
