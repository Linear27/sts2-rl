from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .config import LocalInstanceConfig


class RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class InstanceSpec(RuntimeModel):
    instance_id: str
    instance_root: Path
    logs_root: Path
    api_port: int
    base_url: str


class InstanceManifest(RuntimeModel):
    instance_id: str
    instance_root: Path
    logs_root: Path
    api_port: int
    base_url: str
    reference_game_root: Path
    clean_baseline_root: Path
    sts2_agent_enabled: bool
    animation_acceleration_enabled: bool
    created_at_utc: str
    status: str = "planned"


def build_instance_specs(config: LocalInstanceConfig) -> list[InstanceSpec]:
    specs: list[InstanceSpec] = []
    for index, api_port in enumerate(config.allocated_ports, start=1):
        instance_id = f"inst-{index:02d}"
        instance_root = config.runtime.instances_root / instance_id
        logs_root = config.logging.instance_logs_root / instance_id
        specs.append(
            InstanceSpec(
                instance_id=instance_id,
                instance_root=instance_root,
                logs_root=logs_root,
                api_port=api_port,
                base_url=f"http://127.0.0.1:{api_port}",
            )
        )
    return specs


def build_instance_manifest(spec: InstanceSpec, config: LocalInstanceConfig) -> InstanceManifest:
    return InstanceManifest(
        instance_id=spec.instance_id,
        instance_root=spec.instance_root,
        logs_root=spec.logs_root,
        api_port=spec.api_port,
        base_url=spec.base_url,
        reference_game_root=config.reference.game_root,
        clean_baseline_root=config.reference.clean_baseline_root,
        sts2_agent_enabled=config.mods.enable_sts2_agent,
        animation_acceleration_enabled=config.mods.enable_animation_acceleration,
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def instance_manifest_path(instance_root: str | Path) -> Path:
    return Path(instance_root) / "instance-manifest.json"


def write_instance_manifest(manifest: InstanceManifest, path: str | Path | None = None) -> Path:
    manifest_path = Path(path) if path is not None else instance_manifest_path(manifest.instance_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def read_instance_manifest(path: str | Path) -> InstanceManifest:
    manifest_path = Path(path)
    return InstanceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
