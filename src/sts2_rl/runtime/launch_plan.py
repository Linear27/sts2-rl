from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..paths import discover_repo_paths
from .config import LocalInstanceConfig
from .manager import plan_instances


@dataclass(frozen=True)
class LaunchPlan:
    instance_id: str
    api_port: int
    base_url: str
    command: str


DEFAULT_RENDERING_DRIVER = "opengl3"
DEFAULT_STEAM_APP_ID = 2868840


def build_windows_launch_plans(
    config: LocalInstanceConfig,
    *,
    rendering_driver: str = DEFAULT_RENDERING_DRIVER,
    steam_app_id: int = DEFAULT_STEAM_APP_ID,
    launch_retries: int = 1,
    enable_debug_actions: bool = False,
    attempts: int = 40,
    delay_seconds: int = 2,
) -> list[LaunchPlan]:
    start_script = discover_repo_paths(Path(__file__)).root / "scripts" / "start-sts2-instance.ps1"
    quoted_script = _quote_path(start_script)

    plans: list[LaunchPlan] = []
    for index, spec in enumerate(plan_instances(config)):
        exe_path = spec.instance_root / "SlayTheSpire2.exe"
        quoted_exe = _quote_path(exe_path)
        command_parts = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            quoted_script,
            "-ExePath",
            quoted_exe,
            "-ApiPort",
            str(spec.api_port),
            "-SteamAppId",
            str(steam_app_id),
            "-LaunchRetries",
            str(launch_retries),
            "-RenderingDriver",
            rendering_driver,
            "-Attempts",
            str(attempts),
            "-DelaySeconds",
            str(delay_seconds),
        ]
        if index > 0:
            command_parts.append("-KeepExistingProcesses")
        if enable_debug_actions:
            command_parts.append("-EnableDebugActions")

        plans.append(
            LaunchPlan(
                instance_id=spec.instance_id,
                api_port=spec.api_port,
                base_url=spec.base_url,
                command=" ".join(command_parts),
            )
        )

    return plans


def _quote_path(path: Path) -> str:
    return f'"{str(path)}"'
