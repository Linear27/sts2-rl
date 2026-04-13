from pathlib import Path

from sts2_rl.runtime import build_instance_specs, load_instance_config


def test_load_instance_config_resolves_relative_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "local.toml"
    config_path.write_text(
        """
[reference]
game_root = "../reference"
clean_baseline_root = "../clean"

[runtime]
instances_root = "./runtime"
first_api_port = 8080
instance_count = 2

[mods]
enable_sts2_agent = true
enable_animation_acceleration = false

[logging]
instance_logs_root = "./logs"
""".strip(),
        encoding="utf-8",
    )

    config = load_instance_config(config_path)
    specs = build_instance_specs(config)

    assert config.reference.game_root == (tmp_path / "../reference").resolve()
    assert config.runtime.instances_root == (tmp_path / "runtime").resolve()
    assert config.logging.instance_logs_root == (tmp_path / "logs").resolve()
    assert [spec.api_port for spec in specs] == [8080, 8081]
    assert [spec.instance_id for spec in specs] == ["inst-01", "inst-02"]
