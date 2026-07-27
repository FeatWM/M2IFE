from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install dependencies with pip install -r requirements.txt") from exc

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent)
    return config


def project_root(config: dict[str, Any]) -> Path:
    return Path(config.get("_project_root", ".")).resolve()


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (project_root(config) / path).resolve()


def with_device(config: dict[str, Any], device: str | None) -> dict[str, Any]:
    updated = deepcopy(config)
    if device:
        updated.setdefault("project", {})["device"] = device
    return updated

