from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import LocalInstanceConfig
from .manager import plan_instances

DEFAULT_TEMPLATE_DIR = Path("data") / "user-data-templates" / "golden"
DEFAULT_WINDOW_WIDTH = 960
DEFAULT_WINDOW_HEIGHT = 540
PRUNABLE_USER_DATA_DIR_NAMES = ("logs", "shader_cache", "vulkan", "sentry")
PRUNABLE_USER_DATA_FILE_NAMES = ("sentry.dat",)


@dataclass(frozen=True)
class PreparedUserDataTemplate:
    template_root: Path
    settings_paths: tuple[Path, ...]
    pruned_paths: tuple[Path, ...]


@dataclass(frozen=True)
class BootstrappedUserData:
    template_root: Path
    settings_paths: tuple[Path, ...]
    pruned_paths: tuple[Path, ...]
    seeded_paths: tuple[Path, ...]


def instance_user_data_dir_name(instance_id: str) -> str:
    return f"SlayTheSpire2-{instance_id}"


def instance_user_data_dir(instance_id: str, *, appdata_root: Path | None = None) -> Path:
    base = appdata_root or _default_appdata_root()
    return base / instance_user_data_dir_name(instance_id)


def seed_instance_user_data(
    config: LocalInstanceConfig,
    *,
    source_user_dir: Path,
    replace_existing: bool = True,
    appdata_root: Path | None = None,
) -> list[Path]:
    source_root = source_user_dir.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source user-data directory does not exist: {source_root}")

    target_roots = [
        instance_user_data_dir(spec.instance_id, appdata_root=appdata_root)
        for spec in plan_instances(config)
    ]
    requires_snapshot = any(target.resolve() == source_root for target in target_roots if target.exists()) or (
        any(target == source_root for target in target_roots)
    )

    with tempfile.TemporaryDirectory(prefix="sts2-seed-user-data-") as temp_dir:
        snapshot_root = source_root
        if requires_snapshot:
            snapshot_root = Path(temp_dir) / "source"
            shutil.copytree(source_root, snapshot_root)

        seeded_paths: list[Path] = []
        for target_root in target_roots:
            if replace_existing and target_root.exists():
                shutil.rmtree(target_root)

            shutil.copytree(snapshot_root, target_root, dirs_exist_ok=not replace_existing)
            seeded_paths.append(target_root)

        return seeded_paths


def prepare_user_data_template(
    source_user_dir: Path,
    *,
    template_dir: Path = DEFAULT_TEMPLATE_DIR,
    replace_existing: bool = True,
    window_width: int = DEFAULT_WINDOW_WIDTH,
    window_height: int = DEFAULT_WINDOW_HEIGHT,
    window_pos_x: int = -1,
    window_pos_y: int = -1,
    skip_intro_logo: bool = True,
    mods_enabled: bool = True,
    enable_all_listed_mods: bool = True,
) -> PreparedUserDataTemplate:
    source_root = source_user_dir.resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source user-data directory does not exist: {source_root}")

    template_root = template_dir.resolve()
    if template_root != source_root:
        _copy_user_data_tree(source_root, template_root, replace_existing=replace_existing)

    pruned_paths = _prune_user_data_runtime_artifacts(template_root)
    settings_paths = _patch_user_data_settings(
        template_root,
        window_width=window_width,
        window_height=window_height,
        window_pos_x=window_pos_x,
        window_pos_y=window_pos_y,
        skip_intro_logo=skip_intro_logo,
        mods_enabled=mods_enabled,
        enable_all_listed_mods=enable_all_listed_mods,
    )
    return PreparedUserDataTemplate(
        template_root=template_root,
        settings_paths=tuple(settings_paths),
        pruned_paths=tuple(pruned_paths),
    )


def bootstrap_instance_user_data(
    config: LocalInstanceConfig,
    *,
    source_user_dir: Path,
    template_dir: Path = DEFAULT_TEMPLATE_DIR,
    replace_existing: bool = True,
    seed_instances: bool = True,
    appdata_root: Path | None = None,
    window_width: int = DEFAULT_WINDOW_WIDTH,
    window_height: int = DEFAULT_WINDOW_HEIGHT,
    window_pos_x: int = -1,
    window_pos_y: int = -1,
    skip_intro_logo: bool = True,
    mods_enabled: bool = True,
    enable_all_listed_mods: bool = True,
) -> BootstrappedUserData:
    prepared = prepare_user_data_template(
        source_user_dir,
        template_dir=template_dir,
        replace_existing=replace_existing,
        window_width=window_width,
        window_height=window_height,
        window_pos_x=window_pos_x,
        window_pos_y=window_pos_y,
        skip_intro_logo=skip_intro_logo,
        mods_enabled=mods_enabled,
        enable_all_listed_mods=enable_all_listed_mods,
    )

    seeded_paths: tuple[Path, ...] = ()
    if seed_instances:
        seeded_paths = tuple(
            seed_instance_user_data(
                config,
                source_user_dir=prepared.template_root,
                replace_existing=replace_existing,
                appdata_root=appdata_root,
            )
        )

    return BootstrappedUserData(
        template_root=prepared.template_root,
        settings_paths=prepared.settings_paths,
        pruned_paths=prepared.pruned_paths,
        seeded_paths=seeded_paths,
    )


def _copy_user_data_tree(source_root: Path, target_root: Path, *, replace_existing: bool) -> None:
    requires_snapshot = target_root.exists() and target_root.resolve() == source_root
    with tempfile.TemporaryDirectory(prefix="sts2-user-data-template-") as temp_dir:
        snapshot_root = source_root
        if requires_snapshot:
            snapshot_root = Path(temp_dir) / "source"
            shutil.copytree(source_root, snapshot_root)

        if replace_existing and target_root.exists():
            shutil.rmtree(target_root)

        shutil.copytree(snapshot_root, target_root, dirs_exist_ok=not replace_existing)


def _patch_user_data_settings(
    user_data_root: Path,
    *,
    window_width: int,
    window_height: int,
    window_pos_x: int,
    window_pos_y: int,
    skip_intro_logo: bool,
    mods_enabled: bool,
    enable_all_listed_mods: bool,
) -> list[Path]:
    settings_paths = sorted(
        path
        for path in user_data_root.rglob("settings.save")
        if "steam" in path.parts and path.is_file()
    )
    if not settings_paths:
        raise FileNotFoundError(f"No Steam settings.save files found under {user_data_root}")

    for settings_path in settings_paths:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
        payload["fullscreen"] = False
        payload["resize_windows"] = True
        payload["window_position"] = {"X": window_pos_x, "Y": window_pos_y}
        payload["window_size"] = {"X": window_width, "Y": window_height}
        payload["skip_intro_logo"] = skip_intro_logo

        mod_settings = payload.setdefault("mod_settings", {})
        mod_settings["mods_enabled"] = mods_enabled
        if enable_all_listed_mods:
            for mod_entry in mod_settings.get("mod_list", []):
                if isinstance(mod_entry, dict):
                    mod_entry["is_enabled"] = True

        settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return settings_paths


def _prune_user_data_runtime_artifacts(user_data_root: Path) -> list[Path]:
    pruned_paths: list[Path] = []
    for dir_name in PRUNABLE_USER_DATA_DIR_NAMES:
        path = user_data_root / dir_name
        if path.exists():
            shutil.rmtree(path)
            pruned_paths.append(path)

    for file_name in PRUNABLE_USER_DATA_FILE_NAMES:
        path = user_data_root / file_name
        if path.exists():
            path.unlink()
            pruned_paths.append(path)

    return pruned_paths


def _default_appdata_root() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA is not set in the current environment.")
    return Path(appdata)
