from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoPaths:
    root: Path
    configs: Path
    docs: Path
    runtime: Path
    data: Path
    artifacts: Path


def discover_repo_paths(start: Path | None = None) -> RepoPaths:
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return RepoPaths(
                root=candidate,
                configs=candidate / "configs",
                docs=candidate / "docs",
                runtime=candidate / "runtime",
                data=candidate / "data",
                artifacts=candidate / "artifacts",
            )
    raise FileNotFoundError("Could not locate repository root from pyproject.toml")
