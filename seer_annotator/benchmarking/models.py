"""Model configuration loading for the offline formatting benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .store import ModelConfig


def _config_from_mapping(name: str, value: dict[str, Any]) -> ModelConfig:
    data = dict(value)
    data.setdefault("name", name)
    # ``params`` is the preferred spelling, while allowing model-specific
    # LiteLLM kwargs at the model table level keeps small TOML files pleasant.
    params = dict(data.pop("params", {}) or {})
    for key in list(data):
        if key not in {"name", "provider", "model", "structured_output", "temperature", "timeout", "drop_params"}:
            params[key] = data.pop(key)
    if "provider" not in data or "model" not in data:
        raise ValueError(f"model {name!r} must define provider and model")
    return ModelConfig(params=params, **data)


def load_model_configs(path: str | Path) -> dict[str, ModelConfig]:
    """Load ``[models.<name>]`` or ``[[models]]`` TOML configurations.

    The returned mapping is deterministic and contains no resolved provider
    credentials. Credentials belong in the normal Settings provider section.
    """
    import tomllib

    with Path(path).open("rb") as handle:
        document = tomllib.load(handle)
    raw = document.get("models", document.get("model", document))
    result: dict[str, ModelConfig] = {}
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict) or not item.get("name"):
                raise ValueError("each [[models]] entry requires name")
            config = _config_from_mapping(str(item["name"]), item)
            result[config.name] = config
    elif isinstance(raw, dict):
        for name, value in raw.items():
            if not isinstance(value, dict):
                continue
            config = _config_from_mapping(str(name), value)
            result[config.name] = config
    else:
        raise ValueError("TOML models must be a table or array of tables")
    if not result:
        raise ValueError("no model configurations found")
    return result


def select_model_configs(path: str | Path, names: list[str] | None = None) -> list[ModelConfig]:
    configs = load_model_configs(path)
    if not names:
        return list(configs.values())
    missing = [name for name in names if name not in configs]
    if missing:
        raise KeyError(f"unknown model config(s): {', '.join(missing)}")
    return [configs[name] for name in names]
