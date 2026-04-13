from pathlib import Path

from sts2_rl.runtime import load_instance_config, run_preflight


def test_preflight_detects_missing_clean_baseline_and_dirty_reference_mods(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    (reference_root / "mods").mkdir(parents=True)
    (reference_root / "mods" / "SmartThoughtSpire.dll").write_text("", encoding="utf-8")
    (reference_root / "release_info.json").write_text(
        '{"version":"v0.103.0"}',
        encoding="utf-8",
    )
    (tmp_path / "runtime").mkdir()
    (tmp_path / "logs").mkdir()

    config_path = tmp_path / "local.toml"
    config_path.write_text(
        f"""
[reference]
game_root = "{reference_root.as_posix()}"
clean_baseline_root = "{(tmp_path / 'clean').as_posix()}"

[runtime]
instances_root = "{(tmp_path / 'runtime').as_posix()}"
first_api_port = 8080
instance_count = 1

[mods]
enable_sts2_agent = true
enable_animation_acceleration = true

[logging]
instance_logs_root = "{(tmp_path / 'logs').as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    config = load_instance_config(config_path)
    report = run_preflight(config)
    levels = {check.name: check.level for check in report.checks}

    assert report.game_version == "v0.103.0"
    assert levels["reference_game_root"] == "pass"
    assert levels["repo_windows_start_script"] == "pass"
    assert levels["clean_baseline_root"] == "fail"
    assert levels["reference_mods_cleanliness"] == "warn"


def test_preflight_warns_when_clean_baseline_lacks_sts2_agent_payload(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    clean_root = tmp_path / "clean"
    (reference_root / "mods").mkdir(parents=True)
    (reference_root / "release_info.json").write_text('{"version":"v0.103.0"}', encoding="utf-8")
    (clean_root / "mods").mkdir(parents=True)
    (tmp_path / "runtime").mkdir()
    (tmp_path / "logs").mkdir()

    config_path = tmp_path / "local.toml"
    config_path.write_text(
        f"""
[reference]
game_root = "{reference_root.as_posix()}"
clean_baseline_root = "{clean_root.as_posix()}"

[runtime]
instances_root = "{(tmp_path / 'runtime').as_posix()}"
first_api_port = 8080
instance_count = 1

[mods]
enable_sts2_agent = true
enable_animation_acceleration = false

[logging]
instance_logs_root = "{(tmp_path / 'logs').as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    config = load_instance_config(config_path)
    report = run_preflight(config)
    levels = {check.name: check.level for check in report.checks}

    assert levels["clean_baseline_root"] == "pass"
    assert levels["repo_windows_start_script"] == "pass"
    assert levels["clean_baseline_mods_dir"] == "pass"
    assert levels["clean_baseline_sts2_agent_payload"] == "warn"
