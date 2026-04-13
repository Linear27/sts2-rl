import json
from pathlib import Path

from sts2_rl.runtime import (
    bootstrap_instance_user_data,
    instance_user_data_dir,
    instance_user_data_dir_name,
    load_instance_config,
    prepare_user_data_template,
    seed_instance_user_data,
)


def test_instance_user_data_dir_helpers_use_instance_specific_names(tmp_path: Path) -> None:
    appdata_root = tmp_path / "appdata"
    path = instance_user_data_dir("inst-03", appdata_root=appdata_root)

    assert instance_user_data_dir_name("inst-03") == "SlayTheSpire2-inst-03"
    assert path == appdata_root / "SlayTheSpire2-inst-03"


def test_seed_instance_user_data_copies_seed_to_each_instance(tmp_path: Path) -> None:
    source_user_dir = tmp_path / "seed-user-dir"
    (source_user_dir / "steam" / "123").mkdir(parents=True)
    (source_user_dir / "steam" / "123" / "settings.save").write_text("seed", encoding="utf-8")

    config_path = tmp_path / "local.toml"
    config_path.write_text(
        f"""
[reference]
game_root = "{(tmp_path / 'reference').as_posix()}"
clean_baseline_root = "{(tmp_path / 'clean').as_posix()}"

[runtime]
instances_root = "{(tmp_path / 'runtime').as_posix()}"
first_api_port = 8080
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
    seeded = seed_instance_user_data(
        config,
        source_user_dir=source_user_dir,
        appdata_root=tmp_path / "appdata",
    )

    assert seeded[0] == tmp_path / "appdata" / "SlayTheSpire2-inst-01"
    assert (seeded[1] / "steam" / "123" / "settings.save").read_text(encoding="utf-8") == "seed"


def test_seed_instance_user_data_handles_source_equal_to_target(tmp_path: Path) -> None:
    appdata_root = tmp_path / "appdata"
    source_user_dir = appdata_root / "SlayTheSpire2-inst-01"
    (source_user_dir / "steam" / "123").mkdir(parents=True)
    (source_user_dir / "steam" / "123" / "settings.save").write_text("seed", encoding="utf-8")

    config_path = tmp_path / "local.toml"
    config_path.write_text(
        f"""
[reference]
game_root = "{(tmp_path / 'reference').as_posix()}"
clean_baseline_root = "{(tmp_path / 'clean').as_posix()}"

[runtime]
instances_root = "{(tmp_path / 'runtime').as_posix()}"
first_api_port = 8080
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
    seeded = seed_instance_user_data(
        config,
        source_user_dir=source_user_dir,
        appdata_root=appdata_root,
    )

    assert (seeded[0] / "steam" / "123" / "settings.save").read_text(encoding="utf-8") == "seed"
    assert (seeded[1] / "steam" / "123" / "settings.save").read_text(encoding="utf-8") == "seed"


def test_prepare_user_data_template_prunes_runtime_artifacts_and_patches_settings(tmp_path: Path) -> None:
    source_user_dir = tmp_path / "source-user-dir"
    settings_dir = source_user_dir / "steam" / "123"
    settings_dir.mkdir(parents=True)
    (source_user_dir / "logs").mkdir()
    (source_user_dir / "shader_cache").mkdir()
    (source_user_dir / "sentry").mkdir()
    (source_user_dir / "vulkan").mkdir()
    (source_user_dir / "sentry.dat").write_text("placeholder", encoding="utf-8")
    (settings_dir / "settings.save").write_text(
        json.dumps(
            {
                "fullscreen": True,
                "resize_windows": False,
                "skip_intro_logo": False,
                "mod_settings": {
                    "mods_enabled": False,
                    "mod_list": [
                        {"id": "STS2AIAgent", "is_enabled": False},
                        {"id": "STS2-QuickAnimationMode", "is_enabled": False},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    prepared = prepare_user_data_template(
        source_user_dir,
        template_dir=tmp_path / "golden",
        window_width=1440,
        window_height=900,
    )

    saved_payload = json.loads(prepared.settings_paths[0].read_text(encoding="utf-8"))
    assert prepared.template_root == tmp_path / "golden"
    assert saved_payload["fullscreen"] is False
    assert saved_payload["resize_windows"] is True
    assert saved_payload["window_size"] == {"X": 1440, "Y": 900}
    assert saved_payload["window_position"] == {"X": -1, "Y": -1}
    assert saved_payload["skip_intro_logo"] is True
    assert saved_payload["mod_settings"]["mods_enabled"] is True
    assert all(entry["is_enabled"] is True for entry in saved_payload["mod_settings"]["mod_list"])
    assert not (prepared.template_root / "logs").exists()
    assert not (prepared.template_root / "shader_cache").exists()
    assert not (prepared.template_root / "sentry").exists()
    assert not (prepared.template_root / "vulkan").exists()
    assert not (prepared.template_root / "sentry.dat").exists()
    assert len(prepared.pruned_paths) == 5


def test_bootstrap_instance_user_data_prepares_template_and_seeds_instances(tmp_path: Path) -> None:
    source_user_dir = tmp_path / "source-user-dir"
    settings_dir = source_user_dir / "steam" / "123"
    settings_dir.mkdir(parents=True)
    (settings_dir / "settings.save").write_text(
        json.dumps(
            {
                "fullscreen": True,
                "mod_settings": {
                    "mods_enabled": False,
                    "mod_list": [{"id": "STS2AIAgent", "is_enabled": False}],
                },
            }
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "local.toml"
    config_path.write_text(
        f"""
[reference]
game_root = "{(tmp_path / 'reference').as_posix()}"
clean_baseline_root = "{(tmp_path / 'clean').as_posix()}"

[runtime]
instances_root = "{(tmp_path / 'runtime').as_posix()}"
first_api_port = 8080
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
    bootstrapped = bootstrap_instance_user_data(
        config,
        source_user_dir=source_user_dir,
        template_dir=tmp_path / "golden",
        appdata_root=tmp_path / "appdata",
    )

    assert bootstrapped.template_root == tmp_path / "golden"
    assert len(bootstrapped.settings_paths) == 1
    assert len(bootstrapped.seeded_paths) == 2
    for seeded_root in bootstrapped.seeded_paths:
        seeded_payload = json.loads(
            (seeded_root / "steam" / "123" / "settings.save").read_text(encoding="utf-8")
        )
        assert seeded_payload["fullscreen"] is False
        assert seeded_payload["mod_settings"]["mods_enabled"] is True
