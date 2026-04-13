from pathlib import Path

from sts2_rl.runtime import initialize_instances, load_instance_config, provision_instances


def test_initialize_instances_creates_runtime_layout_and_manifests(tmp_path: Path) -> None:
    config_path = tmp_path / "local.toml"
    config_path.write_text(
        f"""
[reference]
game_root = "{(tmp_path / 'reference').as_posix()}"
clean_baseline_root = "{(tmp_path / 'clean').as_posix()}"

[runtime]
instances_root = "{(tmp_path / 'runtime').as_posix()}"
first_api_port = 8100
instance_count = 2

[mods]
enable_sts2_agent = true
enable_animation_acceleration = true

[logging]
instance_logs_root = "{(tmp_path / 'logs').as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    config = load_instance_config(config_path)
    initialized = initialize_instances(config)

    assert [item.spec.instance_id for item in initialized] == ["inst-01", "inst-02"]
    assert (tmp_path / "runtime" / "inst-01").is_dir()
    assert (tmp_path / "logs" / "inst-02").is_dir()
    assert initialized[0].manifest_path.exists()


def test_provision_instances_copies_clean_baseline_and_marks_manifest(tmp_path: Path) -> None:
    reference_root = tmp_path / "reference"
    clean_root = tmp_path / "clean"
    runtime_root = tmp_path / "runtime"
    logs_root = tmp_path / "logs"
    (reference_root / "mods").mkdir(parents=True)
    (reference_root / "release_info.json").write_text('{"version":"v0.103.0"}', encoding="utf-8")
    (clean_root / "mods").mkdir(parents=True)
    (clean_root / "mods" / "STS2AIAgent.dll").write_text("dll", encoding="utf-8")
    (clean_root / "SlayTheSpire2.exe").write_text("exe", encoding="utf-8")

    config_path = tmp_path / "local.toml"
    config_path.write_text(
        f"""
[reference]
game_root = "{reference_root.as_posix()}"
clean_baseline_root = "{clean_root.as_posix()}"

[runtime]
instances_root = "{runtime_root.as_posix()}"
first_api_port = 8100
instance_count = 2

[mods]
enable_sts2_agent = true
enable_animation_acceleration = true

[logging]
instance_logs_root = "{logs_root.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    config = load_instance_config(config_path)
    provisioned = provision_instances(config)

    assert (runtime_root / "inst-01" / "SlayTheSpire2.exe").exists()
    assert (runtime_root / "inst-02" / "mods" / "STS2AIAgent.dll").exists()
    assert (runtime_root / "inst-01" / "override.cfg").exists()
    assert 'config/custom_user_dir_name="SlayTheSpire2-inst-01"' in (
        runtime_root / "inst-01" / "override.cfg"
    ).read_text(encoding="utf-8")
    assert provisioned[0].manifest.status == "provisioned"
