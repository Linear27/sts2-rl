from pathlib import Path

from sts2_rl.runtime import collect_instance_statuses, initialize_instances, load_instance_config


def test_collect_instance_statuses_handles_offline_instances(tmp_path: Path) -> None:
    config_path = tmp_path / "local.toml"
    config_path.write_text(
        f"""
[reference]
game_root = "{(tmp_path / 'reference').as_posix()}"
clean_baseline_root = "{(tmp_path / 'clean').as_posix()}"

[runtime]
instances_root = "{(tmp_path / 'runtime').as_posix()}"
first_api_port = 6550
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
    initialize_instances(config)
    statuses = collect_instance_statuses(config, timeout_seconds=0.1)

    assert len(statuses) == 1
    assert statuses[0].manifest_exists is True
    assert statuses[0].api_reachable is False
    assert statuses[0].manifest_status == "planned"
