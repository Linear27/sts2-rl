from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..paths import discover_repo_paths
from .config import LocalInstanceConfig


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    level: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    checks: list[PreflightCheck]
    game_version: str | None = None

    @property
    def has_failures(self) -> bool:
        return any(check.level == "fail" for check in self.checks)


def run_preflight(
    config: LocalInstanceConfig,
    *,
    sts2_agent_root: Path | None = None,
    exe_path: Path | None = None,
) -> PreflightReport:
    checks: list[PreflightCheck] = []
    game_version: str | None = None
    repo_root = discover_repo_paths(Path(__file__)).root

    checks.append(_exists_check("reference_game_root", config.reference.game_root))
    checks.append(
        _exists_check(
            "repo_windows_start_script",
            repo_root / "scripts" / "start-sts2-instance.ps1",
        )
    )

    release_info_path = config.reference.game_root / "release_info.json"
    if release_info_path.exists():
        try:
            payload = json.loads(release_info_path.read_text(encoding="utf-8"))
            game_version = payload.get("version")
            checks.append(
                PreflightCheck(
                    name="reference_release_info",
                    level="pass",
                    message=f"Found release_info.json with version {game_version or 'unknown'}.",
                )
            )
        except Exception as exc:
            checks.append(
                PreflightCheck(
                    name="reference_release_info",
                    level="fail",
                    message=f"Could not parse release_info.json: {exc}",
                )
            )
    else:
        checks.append(
            PreflightCheck(
                name="reference_release_info",
                level="fail",
                message="reference release_info.json is missing.",
            )
        )

    checks.append(_exists_check("clean_baseline_root", config.reference.clean_baseline_root))

    clean_mods = config.reference.clean_baseline_root / "mods"
    checks.append(_exists_check("clean_baseline_mods_dir", clean_mods))
    if clean_mods.exists():
        clean_mod_entries = {entry.name for entry in clean_mods.iterdir()}
        if config.mods.enable_sts2_agent:
            required_agent_files = {"STS2AIAgent.dll", "STS2AIAgent.pck", "mod_id.json"}
            missing_agent_files = sorted(required_agent_files - clean_mod_entries)
            checks.append(
                PreflightCheck(
                    name="clean_baseline_sts2_agent_payload",
                    level="pass" if not missing_agent_files else "warn",
                    message=(
                        "Clean baseline contains STS2-Agent payload files."
                        if not missing_agent_files
                        else "Clean baseline is missing STS2-Agent payload files: "
                        + ", ".join(missing_agent_files)
                    ),
                )
            )

        if config.mods.enable_animation_acceleration:
            has_ritsulib = any("ritsu" in entry.lower() for entry in clean_mod_entries)
            checks.append(
                PreflightCheck(
                    name="clean_baseline_animation_dependency_hint",
                    level="pass" if has_ritsulib else "warn",
                    message=(
                        "Detected a RitsuLib-like dependency in clean baseline mods."
                        if has_ritsulib
                        else "No RitsuLib-like dependency detected in clean baseline mods. "
                        "If you use Quick Animation Mode, verify its dependency payload is installed."
                    ),
                )
            )

    reference_mods = config.reference.game_root / "mods"
    if reference_mods.exists():
        mod_entries = [entry.name for entry in reference_mods.iterdir()]
        if mod_entries:
            checks.append(
                PreflightCheck(
                    name="reference_mods_cleanliness",
                    level="warn",
                    message=f"Reference mods directory is not clean: {', '.join(sorted(mod_entries))}",
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    name="reference_mods_cleanliness",
                    level="pass",
                    message="Reference mods directory is empty.",
                )
            )
    else:
        checks.append(
            PreflightCheck(
                name="reference_mods_cleanliness",
                level="warn",
                message="Reference mods directory does not exist yet.",
            )
        )

    checks.append(
        PreflightCheck(
            name="uv_available",
            level="pass" if shutil.which("uv") else "fail",
            message="uv is available on PATH." if shutil.which("uv") else "uv is not available on PATH.",
        )
    )

    checks.append(
        PreflightCheck(
            name="python_available",
            level="pass" if shutil.which("python") else "fail",
            message="python is available on PATH."
            if shutil.which("python")
            else "python is not available on PATH.",
        )
    )

    if sts2_agent_root is not None:
        checks.append(_exists_check("sts2_agent_root", sts2_agent_root))
        checks.append(
            _exists_check(
                "sts2_agent_start_script",
                sts2_agent_root / "scripts" / "start-game-session.ps1",
            )
        )

    if exe_path is not None:
        checks.append(_exists_check("instance_exe_path", exe_path))

    if config.mods.enable_animation_acceleration:
        checks.append(
            PreflightCheck(
                name="animation_acceleration_prereq",
                level="warn",
                message="Animation acceleration is enabled in config; Quick Animation Mode and its dependencies must be verified separately.",
            )
        )

    return PreflightReport(checks=checks, game_version=game_version)


def _exists_check(name: str, path: Path) -> PreflightCheck:
    if path.exists():
        return PreflightCheck(name=name, level="pass", message=f"Found {path}")
    return PreflightCheck(name=name, level="fail", message=f"Missing {path}")
