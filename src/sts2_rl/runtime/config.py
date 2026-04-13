from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .ports import allocate_ports


class RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReferenceConfig(RuntimeModel):
    game_root: Path
    clean_baseline_root: Path


class RuntimeConfig(RuntimeModel):
    instances_root: Path
    first_api_port: int
    instance_count: int


class ModsConfig(RuntimeModel):
    enable_sts2_agent: bool = True
    enable_animation_acceleration: bool = True


class LoggingConfig(RuntimeModel):
    instance_logs_root: Path


class LocalInstanceConfig(RuntimeModel):
    reference: ReferenceConfig
    runtime: RuntimeConfig
    mods: ModsConfig
    logging: LoggingConfig

    @property
    def allocated_ports(self) -> list[int]:
        return allocate_ports(self.runtime.first_api_port, self.runtime.instance_count)


def load_instance_config(path: str | Path) -> LocalInstanceConfig:
    config_path = Path(path).resolve()
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config = LocalInstanceConfig.model_validate(raw)
    base_dir = config_path.parent

    return LocalInstanceConfig(
        reference=ReferenceConfig(
            game_root=_resolve_path(config.reference.game_root, base_dir),
            clean_baseline_root=_resolve_path(config.reference.clean_baseline_root, base_dir),
        ),
        runtime=RuntimeConfig(
            instances_root=_resolve_path(config.runtime.instances_root, base_dir),
            first_api_port=config.runtime.first_api_port,
            instance_count=config.runtime.instance_count,
        ),
        mods=config.mods,
        logging=LoggingConfig(
            instance_logs_root=_resolve_path(config.logging.instance_logs_root, base_dir),
        ),
    )


def _resolve_path(path: Path, base_dir: Path) -> Path:
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
