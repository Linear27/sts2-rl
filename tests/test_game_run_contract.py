from sts2_rl.game_run_contract import (
    build_game_run_contract,
    build_game_run_contract_validation_payload,
    inspect_game_run_contract,
)


def test_build_game_run_contract_normalizes_fields() -> None:
    contract = build_game_run_contract(
        run_mode=" Custom ",
        game_seed=" SEED-001 ",
        seed_source=" custom_mode_manual ",
        character_id="ironclad",
        ascension=3,
        custom_modifiers=[" SealedDeck ", "sealeddeck", "Draft "],
        progress_profile=" unlocked_all ",
        benchmark_contract_id=" bench-a ",
        strict=True,
    )

    assert contract is not None
    assert contract.run_mode == "custom"
    assert contract.game_seed == "SEED-001"
    assert contract.seed_source == "custom_mode_manual"
    assert contract.character_id == "IRONCLAD"
    assert contract.ascension == 3
    assert contract.custom_modifiers == ("Draft", "SealedDeck", "sealeddeck")
    assert contract.progress_profile == "unlocked_all"
    assert contract.benchmark_contract_id == "bench-a"
    assert contract.strict is True


def test_inspect_game_run_contract_detects_seed_character_and_ascension_mismatch() -> None:
    contract = build_game_run_contract(
        run_mode="custom",
        game_seed="EXPECTED-SEED",
        character_id="IRONCLAD",
        ascension=5,
        strict=True,
    )

    observation = inspect_game_run_contract(
        contract=contract,
        screen_type="MAP",
        run_id="run-1",
        observed_seed="WRONG-SEED",
        observed_character_id="THE_SILENT",
        observed_ascension=3,
    )

    assert observation.checked is True
    assert observation.matches is False
    assert observation.mismatches == (
        "seed_mismatch",
        "character_id_mismatch",
        "ascension_mismatch",
    )


def test_build_game_run_contract_validation_payload_reports_matched_status() -> None:
    contract = build_game_run_contract(
        run_mode="custom",
        game_seed="SEED-001",
        character_id="IRONCLAD",
        ascension=0,
        strict=True,
    )

    payload = build_game_run_contract_validation_payload(
        contract=contract,
        observation_check_count=4,
        observation_match_count=4,
        observation_mismatch_count=0,
        mismatch_histogram={},
        last_mismatches=[],
        observed_seed_histogram={"SEED-001": 1},
        observed_character_histogram={"IRONCLAD": 1},
        observed_ascension_histogram={0: 1},
    )

    assert payload["enabled"] is True
    assert payload["status"] == "matched"
    assert payload["strict"] is True
    assert payload["enforced_fields"] == ["game_seed", "character_id", "ascension"]
    assert payload["unverified_fields"] == ["run_mode"]
    assert payload["seed_matches_expected"] is True
    assert payload["character_matches_expected"] is True
    assert payload["ascension_matches_expected"] is True
