from pathlib import Path

from sts2_rl.runtime import build_windows_launch_plans, load_instance_config


def test_build_windows_launch_plans_use_repo_wrapper_and_keep_existing_after_first(tmp_path: Path) -> None:
    config_path = tmp_path / "local.toml"
    config_path.write_text(
        """
[reference]
game_root = "D:\\\\ref"
clean_baseline_root = "D:\\\\clean"

[runtime]
instances_root = "D:\\\\runtime"
first_api_port = 8080
instance_count = 2

[mods]
enable_sts2_agent = true
enable_animation_acceleration = true

[logging]
instance_logs_root = "D:\\\\logs"
""".strip(),
        encoding="utf-8",
    )

    config = load_instance_config(config_path)
    plans = build_windows_launch_plans(
        config,
        rendering_driver="vulkan",
        steam_app_id=2868840,
        launch_retries=1,
        enable_debug_actions=True,
        attempts=55,
        delay_seconds=3,
    )

    assert len(plans) == 2
    assert plans[0].api_port == 8080
    assert "start-sts2-instance.ps1" in plans[0].command
    assert 'D:\\runtime\\inst-01\\SlayTheSpire2.exe' in plans[0].command
    assert '-ApiPort 8080' in plans[0].command
    assert '-RenderingDriver vulkan' in plans[0].command
    assert '-SteamAppId 2868840' in plans[0].command
    assert '-LaunchRetries 1' in plans[0].command
    assert '-Attempts 55' in plans[0].command
    assert '-DelaySeconds 3' in plans[0].command
    assert '-EnableDebugActions' in plans[0].command
    assert '-KeepExistingProcesses' not in plans[0].command
    assert '-KeepExistingProcesses' in plans[1].command
