from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import LocalInstanceConfig
from .manifest import instance_manifest_path, read_instance_manifest
from .manager import plan_instances


@dataclass(frozen=True)
class InstanceStatus:
    instance_id: str
    api_port: int
    base_url: str
    instance_root: Path
    manifest_exists: bool
    manifest_status: str | None
    api_reachable: bool
    api_status: str | None
    game_version: str | None
    mod_version: str | None
    error: str | None = None


def collect_instance_statuses(
    config: LocalInstanceConfig,
    *,
    timeout_seconds: float = 0.5,
) -> list[InstanceStatus]:
    statuses: list[InstanceStatus] = []

    for spec in plan_instances(config):
        manifest_path = instance_manifest_path(spec.instance_root)
        manifest_exists = manifest_path.exists()
        manifest_status: str | None = None
        if manifest_exists:
            manifest_status = read_instance_manifest(manifest_path).status

        try:
            with httpx.Client(base_url=spec.base_url, timeout=timeout_seconds) as client:
                response = client.get("/health")
                response.raise_for_status()
                payload = response.json()["data"]
            statuses.append(
                InstanceStatus(
                    instance_id=spec.instance_id,
                    api_port=spec.api_port,
                    base_url=spec.base_url,
                    instance_root=spec.instance_root,
                    manifest_exists=manifest_exists,
                    manifest_status=manifest_status,
                    api_reachable=True,
                    api_status=payload.get("status"),
                    game_version=payload.get("game_version"),
                    mod_version=payload.get("mod_version"),
                )
            )
        except Exception as exc:
            statuses.append(
                InstanceStatus(
                    instance_id=spec.instance_id,
                    api_port=spec.api_port,
                    base_url=spec.base_url,
                    instance_root=spec.instance_root,
                    manifest_exists=manifest_exists,
                    manifest_status=manifest_status,
                    api_reachable=False,
                    api_status=None,
                    game_version=None,
                    mod_version=None,
                    error=str(exc),
                )
            )

    return statuses
