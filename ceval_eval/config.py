from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving an optional relative ``extends`` parent."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    parent = raw.pop("extends", None)
    if parent:
        parent_path = Path(str(parent))
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        base = load_config(parent_path)
        raw = deep_merge(base, raw)

    raw.setdefault("_config_path", str(config_path))
    return raw


def get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested key such as ``generation.max_new_tokens``."""
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
