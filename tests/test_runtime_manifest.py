from pathlib import Path

from sts2_rl.runtime import (
    build_instance_manifest,
    build_instance_specs,
    load_instance_config,
    read_instance_manifest,
    write_instance_manifest,
)


def test_instance_manifest_roundtrip(tmp_path: Path) -> None:
    config_path = tmp_path / "local.toml"
    config_path.write_text(
        """
[reference]
game_root = "D:\\\\ref"
clean_baseline_root = "D:\\\\clean"

[runtime]
instances_root = "D:\\\\runtime"
first_api_port = 8090
instance_count = 1

[mods]
enable_sts2_agent = true
enable_animation_acceleration = true

[logging]
instance_logs_root = "D:\\\\logs"
""".strip(),
        encoding="utf-8",
    )

    config = load_instance_config(config_path)
    spec = build_instance_specs(config)[0]
    manifest = build_instance_manifest(spec, config)
    manifest_path = write_instance_manifest(manifest, tmp_path / "instance-manifest.json")

    loaded = read_instance_manifest(manifest_path)

    assert loaded.instance_id == "inst-01"
    assert loaded.api_port == 8090
    assert loaded.base_url == "http://127.0.0.1:8090"
    assert loaded.reference_game_root == Path("D:/ref")
