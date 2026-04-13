import json
from pathlib import Path

from sts2_rl.data import build_state_summary
from sts2_rl.data.trajectory import TrajectoryStepRecord
from sts2_rl.env.candidate_actions import build_candidate_actions
from sts2_rl.env.models import (
    ActionDescriptor,
    AvailableActionsPayload,
    EventOptionPayload,
    EventPayload,
    GameStatePayload,
    RestOptionPayload,
    RestPayload,
    RewardCardOptionPayload,
    RewardPayload,
    RunPayload,
    SelectionCardPayload,
    SelectionPayload,
    ShopCardPayload,
    ShopCardRemovalPayload,
    ShopPayload,
)
from sts2_rl.env.types import StepObservation
from sts2_rl.train import strategic_finetune as strategic_finetune_module
from sts2_rl.train import (
    StrategicFinetuneModel,
    StrategicFinetuneTrainConfig,
    StrategicPretrainTrainConfig,
    train_strategic_finetune_policy,
    train_strategic_pretrain_policy,
)


def test_train_strategic_finetune_policy_writes_lineage_and_warmstart(tmp_path: Path) -> None:
    runtime_dataset = _write_runtime_dataset(tmp_path)
    public_dataset = _write_public_dataset(tmp_path)
    pretrain_report = train_strategic_pretrain_policy(
        dataset_source=public_dataset,
        output_root=tmp_path / "artifacts" / "strategic-pretrain",
        session_name="warmstart-seed",
        config=StrategicPretrainTrainConfig(epochs=30, learning_rate=0.1, seed=7),
    )

    report = train_strategic_finetune_policy(
        runtime_dataset_source=runtime_dataset,
        public_dataset_source=public_dataset,
        output_root=tmp_path / "artifacts" / "strategic-finetune",
        session_name="strategic-ft-unit",
        config=StrategicFinetuneTrainConfig(
            epochs=35,
            learning_rate=0.09,
            seed=13,
            warmstart_checkpoint_path=pretrain_report.best_checkpoint_path,
            schedule="round_robin",
            runtime_replay_passes=2,
            public_replay_passes=1,
            freeze_transferred_ranking_epochs=1,
            runtime_build_id="v0.103.0",
        ),
    )

    assert report.checkpoint_path.exists()
    assert report.best_checkpoint_path.exists()
    assert report.metrics_path.exists()
    assert report.summary_path.exists()
    assert report.runtime_example_count == 8
    assert report.public_example_count == 6
    assert report.warmstart_checkpoint_path == pretrain_report.best_checkpoint_path

    summary = json.loads(report.summary_path.read_text(encoding="utf-8"))
    assert summary["algorithm"] == "strategic_finetune"
    assert summary["schedule"]["mode"] == "round_robin"
    assert summary["warmstart_checkpoint_path"] == str(pretrain_report.best_checkpoint_path)
    assert summary["runtime_dataset"]["dataset_lineage"]["source_paths"] == ["synthetic-runtime"]
    assert summary["public_dataset"]["dataset_lineage"]["source_paths"] == ["synthetic-public"]
    assert summary["runtime_dataset"]["decision_type_histogram"]["selection_remove"] == 1
    assert summary["public_dataset"]["decision_type_histogram"]["selection_remove"] == 1
    assert summary["runtime_dataset"]["build_id_histogram"] == {"v0.103.0": 8}
    assert "runtime" in summary["validation"]["source_metrics"]
    assert "public" in summary["validation"]["source_metrics"]
    assert summary["checkpoint_metadata"]["algorithm"] == "strategic_finetune"
    assert summary["transferred_modules"]["global_ranking"] is True

    model = StrategicFinetuneModel.load(report.best_checkpoint_path)
    ranking = model.rank_actions(
        decision_type="reward_card_pick",
        context=_context_payload(floor=7, room_type="monster", map_point_type="monster", source_type="reward"),
        candidate_actions=["CARD.RUNTIME", "CARD.BLOCK", "skip"],
    )
    assert ranking[0]["candidate_id"] in {"CARD.RUNTIME", "CARD.BLOCK", "skip"}


def test_train_strategic_finetune_rejects_runtime_public_build_mismatch(tmp_path: Path) -> None:
    runtime_dataset = _write_runtime_dataset(tmp_path)
    public_dataset = _write_public_dataset(tmp_path)

    try:
        train_strategic_finetune_policy(
            runtime_dataset_source=runtime_dataset,
            public_dataset_source=public_dataset,
            output_root=tmp_path / "artifacts" / "strategic-finetune-mismatch",
            session_name="strategic-ft-build-mismatch",
            config=StrategicFinetuneTrainConfig(
                epochs=5,
                learning_rate=0.05,
                seed=3,
                runtime_build_id="v0.999.0",
            ),
        )
    except ValueError as exc:
        assert "runtime dataset and public dataset do not overlap on build ids" in str(exc)
    else:
        raise AssertionError("Expected runtime/public build mismatch to raise ValueError.")


def test_runtime_reward_binding_uses_follow_up_source_type() -> None:
    observation = _observation_from_state(
        GameStatePayload(
            screen="REWARD",
            run_id="run-event-followup",
            run=_run_payload(floor=9),
            reward=RewardPayload(
                pending_card_choice=True,
                source_type="event",
                source_room_type="Event",
                source_action="choose_event_option",
                source_event_id="EVENT.EPIC_QUEST",
                source_event_option_index=1,
                source_event_option_text_key="event.quest.solo",
                source_event_option_title="Solo Quest",
                card_options=[
                    RewardCardOptionPayload(index=0, card_id="CARD.REWARD_A", name="Reward A"),
                    RewardCardOptionPayload(index=1, card_id="CARD.REWARD_B", name="Reward B"),
                ],
            ),
        ),
        AvailableActionsPayload(
            screen="REWARD",
            actions=[
                ActionDescriptor(name="choose_reward_card", requires_index=True),
                ActionDescriptor(name="skip_reward_cards"),
            ],
        ),
    )
    chosen_candidate = next(candidate for candidate in observation.legal_actions if candidate.action_id == "choose_reward_card|option=0")
    record = TrajectoryStepRecord(
        timestamp_utc="2026-04-14T00:00:00Z",
        session_name="unit",
        session_kind="collect",
        instance_id="inst-01",
        step_index=1,
        run_id=observation.run_id,
        screen_type=observation.screen_type,
        legal_action_count=len(observation.legal_actions),
        legal_action_ids=[candidate.action_id for candidate in observation.legal_actions],
        chosen_action_id=chosen_candidate.action_id,
        state_summary=build_state_summary(observation),
        action_descriptors=observation.action_descriptors.model_dump(mode="json"),
        state=observation.state.model_dump(mode="json"),
    )

    binding = strategic_finetune_module._resolve_runtime_semantic_binding(
        record=record,
        raw_payload={},
        state=observation.state,
        legal_actions=observation.legal_actions,
        chosen_candidate=chosen_candidate,
    )

    assert binding is not None
    assert binding["decision_type"] == "reward_card_pick"
    assert binding["room_type"] == "event"
    assert binding["map_point_type"] == "event"


def _write_runtime_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "data" / "runtime-strategic"
    dataset_dir.mkdir(parents=True)

    train_records = [
        _step_payload(
            _reward_observation(run_id="RUN-REWARD-1", floor=5, card_ids=("CARD.RUNTIME", "CARD.BLOCK")),
            chosen_action_id="choose_reward_card|option=0",
            step_index=1,
            run_outcome="win",
            decision_stage="reward",
            strategic_context={"source_type": "reward", "room_type": "monster", "map_point_type": "monster", "floor_within_act": 5, "acts_reached": 1},
        ),
        _step_payload(
            _reward_observation(run_id="RUN-REWARD-2", floor=6, card_ids=("CARD.RUNTIME", "CARD.BLOCK")),
            chosen_action_id="skip_reward_cards",
            step_index=2,
            run_outcome="loss",
            decision_stage="reward",
            strategic_context={"source_type": "reward", "room_type": "monster", "map_point_type": "monster", "floor_within_act": 6, "acts_reached": 1},
        ),
        _step_payload(
            _shop_observation(run_id="RUN-SHOP-1", floor=7, card_ids=("CARD.SHOP_A", "CARD.SHOP_B")),
            chosen_action_id="buy_card|option=1",
            step_index=3,
            run_outcome="win",
            decision_stage="shop",
            strategic_context={"source_type": "shop", "room_type": "shop", "map_point_type": "shop", "floor_within_act": 7, "acts_reached": 1},
        ),
        _step_payload(
            _event_observation(run_id="RUN-EVENT-1", floor=8),
            chosen_action_id="choose_event_option|option=1",
            step_index=4,
            run_outcome="win",
            decision_stage="event",
            strategic_context={"source_type": "event", "room_type": "event", "map_point_type": "event", "floor_within_act": 8, "acts_reached": 1},
        ),
        _step_payload(
            _rest_observation(run_id="RUN-REST-1", floor=9),
            chosen_action_id="choose_rest_option|option=1",
            step_index=5,
            run_outcome="win",
            decision_stage="rest",
            strategic_context={"source_type": "rest", "room_type": "rest", "map_point_type": "rest", "floor_within_act": 9, "acts_reached": 1},
        ),
        _step_payload(
            _shop_remove_trigger_observation(run_id="RUN-REMOVE-1", floor=10),
            chosen_action_id="remove_card_at_shop",
            step_index=6,
            run_outcome="loss",
            decision_stage="shop",
            strategic_context={"source_type": "shop", "room_type": "shop", "map_point_type": "shop", "floor_within_act": 10, "acts_reached": 1},
        ),
        _step_payload(
            _shop_remove_selection_observation(run_id="RUN-REMOVE-1", floor=10),
            chosen_action_id="select_deck_card|option=0",
            step_index=7,
            run_outcome="loss",
            decision_stage="selection",
            strategic_context={"source_type": "shop", "room_type": "shop", "map_point_type": "shop", "floor_within_act": 10, "acts_reached": 1},
        ),
    ]
    validation_records = [
        _step_payload(
            _reward_observation(run_id="RUN-REWARD-VAL", floor=11, card_ids=("CARD.RUNTIME", "CARD.BLOCK")),
            chosen_action_id="choose_reward_card|option=0",
            step_index=8,
            run_outcome="win",
            decision_stage="reward",
            strategic_context={"source_type": "reward", "room_type": "monster", "map_point_type": "monster", "floor_within_act": 11, "acts_reached": 1},
        ),
    ]
    test_records = [
        _step_payload(
            _rest_observation(run_id="RUN-REST-TEST", floor=12),
            chosen_action_id="choose_rest_option|option=0",
            step_index=9,
            run_outcome="loss",
            decision_stage="rest",
            strategic_context={"source_type": "rest", "room_type": "rest", "map_point_type": "rest", "floor_within_act": 12, "acts_reached": 1},
        ),
    ]

    _write_jsonl(dataset_dir / "train.steps.jsonl", train_records)
    _write_jsonl(dataset_dir / "validation.steps.jsonl", validation_records)
    _write_jsonl(dataset_dir / "test.steps.jsonl", test_records)
    _write_jsonl(dataset_dir / "steps.jsonl", [*train_records, *validation_records, *test_records])
    (dataset_dir / "dataset-summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "runtime-strategic",
                "dataset_kind": "trajectory_steps",
                "records_path": str((dataset_dir / "steps.jsonl").resolve()),
                "split": {
                    "split_paths": {
                        "train": str((dataset_dir / "train.steps.jsonl").resolve()),
                        "validation": str((dataset_dir / "validation.steps.jsonl").resolve()),
                        "test": str((dataset_dir / "test.steps.jsonl").resolve()),
                    }
                },
                "lineage": {"source_paths": ["synthetic-runtime"]},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_dir


def _write_public_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "data" / "public-strategic"
    dataset_dir.mkdir(parents=True)
    train_records = [
        _public_record("reward-1", "reward_card_pick", "full_candidates", 4, "CARD.BLOCK", ["CARD.BLOCK", "CARD.RUNTIME"], "loss", "monster", "monster", "reward"),
        _public_record("reward-2", "reward_card_pick", "full_candidates", 6, "CARD.RUNTIME", ["CARD.BLOCK", "CARD.RUNTIME"], "win", "monster", "monster", "reward"),
        _public_record("shop-1", "shop_buy", "full_candidates", 7, "CARD.SHOP_A", ["CARD.SHOP_A", "CARD.SHOP_B"], "loss", "shop", "shop", "shop"),
        _public_record("event-1", "event_choice", "chosen_only", 8, "event.take_gold", [], "win", "event", "event", "event"),
    ]
    validation_records = [
        _public_record("rest-1", "rest_site_action", "chosen_only", 9, "upgrade", [], "win", "rest", "rest", "rest"),
    ]
    test_records = [
        _public_record("remove-1", "selection_remove", "chosen_only", 10, "CARD.DEFEND_REGENT", [], "loss", "shop", "shop", "shop"),
    ]
    _write_jsonl(dataset_dir / "train.strategic-decisions.jsonl", train_records)
    _write_jsonl(dataset_dir / "validation.strategic-decisions.jsonl", validation_records)
    _write_jsonl(dataset_dir / "test.strategic-decisions.jsonl", test_records)
    _write_jsonl(dataset_dir / "strategic-decisions.jsonl", [*train_records, *validation_records, *test_records])
    (dataset_dir / "dataset-summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_name": "public-strategic",
                "dataset_kind": "public_strategic_decisions",
                "records_path": str((dataset_dir / "strategic-decisions.jsonl").resolve()),
                "split": {
                    "split_paths": {
                        "train": str((dataset_dir / "train.strategic-decisions.jsonl").resolve()),
                        "validation": str((dataset_dir / "validation.strategic-decisions.jsonl").resolve()),
                        "test": str((dataset_dir / "test.strategic-decisions.jsonl").resolve()),
                    }
                },
                "build_id_histogram": {"v0.103.0": 6},
                "lineage": {"source_paths": ["synthetic-public"]},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_dir


def _step_payload(
    observation: StepObservation,
    *,
    chosen_action_id: str,
    step_index: int,
    run_outcome: str,
    decision_stage: str,
    strategic_context: dict[str, object],
) -> dict:
    chosen_action = next(candidate for candidate in observation.legal_actions if candidate.action_id == chosen_action_id)
    return {
        "schema_version": 4,
        "record_type": "step",
        "timestamp_utc": "2026-04-14T00:00:00+00:00",
        "session_name": "strategic-ft-runtime",
        "session_kind": "train",
        "instance_id": "inst-01",
        "step_index": step_index,
        "run_id": observation.run_id,
        "screen_type": observation.screen_type,
        "floor": observation.state.run.floor if observation.state.run is not None else None,
        "legal_action_count": len(observation.legal_actions),
        "legal_action_ids": [candidate.action_id for candidate in observation.legal_actions],
        "build_warnings": list(observation.build_warnings),
        "chosen_action_id": chosen_action.action_id,
        "chosen_action_label": chosen_action.label,
        "chosen_action_source": chosen_action.source,
        "chosen_action": chosen_action.request.model_dump(mode="json"),
        "policy_name": "runtime-policy",
        "policy_pack": "planner",
        "policy_handler": "synthetic",
        "algorithm": "heuristic",
        "decision_source": "heuristic",
        "decision_stage": decision_stage,
        "decision_reason": "synthetic",
        "decision_score": 1.0,
        "reward": 0.0,
        "reward_source": "synthetic",
        "terminated": False,
        "truncated": False,
        "info": {"run_outcome": run_outcome},
        "model_metrics": {},
        "state_summary": build_state_summary(observation),
        "action_descriptors": observation.action_descriptors.model_dump(mode="json"),
        "state": observation.state.model_dump(mode="json"),
        "strategic_context": strategic_context,
    }


def _reward_observation(*, run_id: str, floor: int, card_ids: tuple[str, str]) -> StepObservation:
    state = GameStatePayload(
        screen="REWARD",
        run_id=run_id,
        run=_run_payload(floor=floor),
        reward=RewardPayload(
            pending_card_choice=True,
            card_options=[
                RewardCardOptionPayload(index=0, card_id=card_ids[0], name=card_ids[0]),
                RewardCardOptionPayload(index=1, card_id=card_ids[1], name=card_ids[1]),
            ],
        ),
    )
    descriptors = AvailableActionsPayload(
        screen="REWARD",
        actions=[ActionDescriptor(name="choose_reward_card", requires_index=True), ActionDescriptor(name="skip_reward_cards")],
    )
    return _observation_from_state(state, descriptors)


def _shop_observation(*, run_id: str, floor: int, card_ids: tuple[str, str]) -> StepObservation:
    state = GameStatePayload(
        screen="SHOP",
        run_id=run_id,
        run=_run_payload(floor=floor),
        shop=ShopPayload(
            is_open=True,
            cards=[
                ShopCardPayload(index=0, card_id=card_ids[0], name=card_ids[0], price=50, is_stocked=True, enough_gold=True),
                ShopCardPayload(index=1, card_id=card_ids[1], name=card_ids[1], price=55, is_stocked=True, enough_gold=True),
            ],
            card_removal=ShopCardRemovalPayload(price=75, available=True, enough_gold=True),
        ),
    )
    descriptors = AvailableActionsPayload(screen="SHOP", actions=[ActionDescriptor(name="buy_card", requires_index=True)])
    return _observation_from_state(state, descriptors)


def _event_observation(*, run_id: str, floor: int) -> StepObservation:
    state = GameStatePayload(
        screen="EVENT",
        run_id=run_id,
        run=_run_payload(floor=floor),
        event=EventPayload(
            event_id="EVENT.GOLD",
            options=[
                EventOptionPayload(index=0, text_key="event.leave", title="Leave"),
                EventOptionPayload(index=1, text_key="event.take_gold", title="Take Gold"),
            ],
        ),
    )
    descriptors = AvailableActionsPayload(screen="EVENT", actions=[ActionDescriptor(name="choose_event_option", requires_index=True)])
    return _observation_from_state(state, descriptors)


def _rest_observation(*, run_id: str, floor: int) -> StepObservation:
    state = GameStatePayload(
        screen="REST",
        run_id=run_id,
        run=_run_payload(floor=floor),
        rest=RestPayload(
            options=[
                RestOptionPayload(index=0, option_id="rest", title="Rest", is_enabled=True),
                RestOptionPayload(index=1, option_id="upgrade", title="Smith", is_enabled=True),
            ],
        ),
    )
    descriptors = AvailableActionsPayload(screen="REST", actions=[ActionDescriptor(name="choose_rest_option", requires_index=True)])
    return _observation_from_state(state, descriptors)


def _shop_remove_trigger_observation(*, run_id: str, floor: int) -> StepObservation:
    state = GameStatePayload(
        screen="SHOP",
        run_id=run_id,
        run=_run_payload(floor=floor),
        shop=ShopPayload(
            is_open=True,
            cards=[ShopCardPayload(index=0, card_id="CARD.SHOP_A", name="A", price=50, is_stocked=True, enough_gold=True)],
            card_removal=ShopCardRemovalPayload(price=75, available=True, enough_gold=True),
        ),
    )
    descriptors = AvailableActionsPayload(screen="SHOP", actions=[ActionDescriptor(name="remove_card_at_shop")])
    return _observation_from_state(state, descriptors)


def _shop_remove_selection_observation(*, run_id: str, floor: int) -> StepObservation:
    state = GameStatePayload(
        screen="SELECTION",
        run_id=run_id,
        run=_run_payload(floor=floor),
        selection=SelectionPayload(
            kind="deck_card_select",
            selection_family="deck",
            semantic_mode="remove",
            source_type="shop",
            source_room_type="Shop",
            prompt="Remove a card from your deck",
            min_select=1,
            max_select=1,
            required_count=1,
            cards=[
                SelectionCardPayload(index=0, card_id="CARD.DEFEND_REGENT", name="Defend"),
                SelectionCardPayload(index=1, card_id="CARD.STRIKE_REGENT", name="Strike"),
            ],
        ),
    )
    descriptors = AvailableActionsPayload(screen="SELECTION", actions=[ActionDescriptor(name="select_deck_card", requires_index=True)])
    return _observation_from_state(state, descriptors)


def _observation_from_state(state: GameStatePayload, descriptors: AvailableActionsPayload) -> StepObservation:
    build = build_candidate_actions(state, descriptors)
    return StepObservation(
        screen_type=state.screen,
        run_id=state.run_id,
        state=state,
        action_descriptors=descriptors,
        legal_actions=build.candidates,
        build_warnings=build.unsupported_actions,
    )


def _run_payload(*, floor: int) -> RunPayload:
    return RunPayload(
        character_id="REGENT",
        ascension=10,
        floor=floor,
        act_index=1,
        act_id="ACT_1",
        current_hp=62,
        max_hp=80,
        gold=150,
        max_energy=3,
    )


def _public_record(
    decision_id: str,
    decision_type: str,
    support_quality: str,
    floor: int,
    chosen_action: str,
    candidate_actions: list[str],
    run_outcome: str,
    room_type: str,
    map_point_type: str,
    source_type: str,
) -> dict:
    return {
        "schema_version": 1,
        "record_type": "public_strategic_decision",
        "decision_id": decision_id,
        "source_name": "sts2runs",
        "snapshot_date": "2026-04-14",
        "source_run_id": floor,
        "run_id": f"RUN-{decision_id}",
        "character_id": "REGENT",
        "ascension": 10,
        "build_id": "v0.103.0",
        "game_mode": "standard",
        "platform_type": "steam",
        "run_outcome": run_outcome,
        "acts_reached": 1,
        "act_index": 1,
        "act_id": "ACT_1",
        "floor": floor,
        "floor_within_act": floor,
        "room_type": room_type,
        "map_point_type": map_point_type,
        "model_id": f"MODEL-{decision_type}",
        "decision_type": decision_type,
        "support_quality": support_quality,
        "reconstruction_confidence": 1.0 if support_quality == "full_candidates" else 0.5,
        "source_type": source_type,
        "candidate_actions": candidate_actions,
        "chosen_action": chosen_action,
        "alternate_actions": [item for item in candidate_actions if item != chosen_action],
        "chosen_present_in_candidates": True if candidate_actions else None,
        "source_record_path": "synthetic",
        "source_record_index": floor,
        "provenance": {"source_url": "synthetic"},
        "metadata": {"artifact_family": "public_strategic_decisions", "has_detail_payload": True, "has_room_history": True},
    }


def _context_payload(*, floor: int, room_type: str, map_point_type: str, source_type: str) -> dict[str, object]:
    return {
        "source_name": "local_runtime",
        "character_id": "regent",
        "ascension": 10,
        "build_id": "v0.103.0",
        "game_mode": "standard",
        "platform_type": "local",
        "acts_reached": 1,
        "act_index": 1,
        "act_id": "act_1",
        "floor": floor,
        "floor_within_act": floor,
        "room_type": room_type,
        "map_point_type": map_point_type,
        "source_type": source_type,
        "candidate_count": 3,
        "metadata": {"has_detail_payload": True, "has_room_history": True},
    }


def _write_jsonl(path: Path, payloads: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
