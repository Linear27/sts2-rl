from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import LocalInstanceConfig
from .manifest import (
    InstanceManifest,
    InstanceSpec,
    build_instance_manifest,
    build_instance_specs,
    write_instance_manifest,
)


@dataclass(frozen=True)
class InitializedInstance:
    spec: InstanceSpec
    manifest: InstanceManifest
    manifest_path: Path


def plan_instances(config: LocalInstanceConfig) -> list[InstanceSpec]:
    return build_instance_specs(config)


def initialize_instances(
    config: LocalInstanceConfig,
    *,
    create_roots: bool = True,
) -> list[InitializedInstance]:
    instances: list[InitializedInstance] = []

    if create_roots:
        config.runtime.instances_root.mkdir(parents=True, exist_ok=True)
        config.logging.instance_logs_root.mkdir(parents=True, exist_ok=True)

    for spec in build_instance_specs(config):
        if create_roots:
            spec.instance_root.mkdir(parents=True, exist_ok=True)
            spec.logs_root.mkdir(parents=True, exist_ok=True)

        manifest = build_instance_manifest(spec, config)
        manifest_path = write_instance_manifest(manifest)
        instances.append(
            InitializedInstance(
                spec=spec,
                manifest=manifest,
                manifest_path=manifest_path,
            )
        )

    return instances


def provision_instances(
    config: LocalInstanceConfig,
    *,
    replace_existing: bool = True,
) -> list[InitializedInstance]:
    instances: list[InitializedInstance] = []
    baseline_root = config.reference.clean_baseline_root
    if not baseline_root.exists():
        raise FileNotFoundError(f"Clean baseline root does not exist: {baseline_root}")

    config.runtime.instances_root.mkdir(parents=True, exist_ok=True)
    config.logging.instance_logs_root.mkdir(parents=True, exist_ok=True)

    for spec in build_instance_specs(config):
        _copy_clean_baseline(
            source_root=baseline_root,
            destination_root=spec.instance_root,
            replace_existing=replace_existing,
        )
        _write_instance_override_cfg(spec.instance_root, spec.instance_id)
        spec.logs_root.mkdir(parents=True, exist_ok=True)

        manifest = build_instance_manifest(spec, config).model_copy(update={"status": "provisioned"})
        manifest_path = write_instance_manifest(manifest)
        instances.append(
            InitializedInstance(
                spec=spec,
                manifest=manifest,
                manifest_path=manifest_path,
            )
        )

    return instances


def _copy_clean_baseline(
    *,
    source_root: Path,
    destination_root: Path,
    replace_existing: bool,
) -> None:
    if replace_existing and destination_root.exists():
        shutil.rmtree(destination_root)

    shutil.copytree(
        source_root,
        destination_root,
        dirs_exist_ok=not replace_existing,
    )


def _write_instance_override_cfg(instance_root: Path, instance_id: str) -> None:
    override_path = instance_root / "override.cfg"
    custom_user_dir_name = f"SlayTheSpire2-{instance_id}"
    override_path.write_text(
        "\n".join(
            [
                "[application]",
                "config/use_custom_user_dir=true",
                f'config/custom_user_dir_name="{custom_user_dir_name}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
