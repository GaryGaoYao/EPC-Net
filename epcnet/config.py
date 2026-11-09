"""Configuration handling: default YAML + optional user YAML + ``key=value`` overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Iterable

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.yaml")


def deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """Apply ``a.b.c=value`` strings; values are parsed as YAML (numbers, lists, bools)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def load_config(paths=None, overrides: Iterable[str] | None = None) -> dict:
    """Default config <- each YAML in ``paths`` (merged in order) <- ``key=value`` overrides."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    for path in paths or []:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    return apply_overrides(cfg, overrides)


def save_config(cfg: dict, path: str | os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)


def stage_config(cfg: dict, stage: int) -> dict:
    """Shared ``train`` settings merged with the stage-specific block."""
    return deep_merge(cfg["train"], cfg[f"stage{stage}"])


def raw_dir(cfg: dict) -> Path:
    if cfg["data"].get("raw_dir"):
        return Path(cfg["data"]["raw_dir"]).expanduser()
    env = os.environ.get("nnUNet_raw")
    if not env:
        raise RuntimeError(
            "Set data.raw_dir in the config (-o data.raw_dir=...) or the nnUNet_raw environment variable."
        )
    return Path(env) / cfg["data"]["dataset"]


def cache_dir(cfg: dict) -> Path:
    return Path(cfg["data"]["cache_dir"]).expanduser()


def run_dir(cfg: dict, stage: int) -> Path:
    return Path(cfg["runs_dir"]).expanduser() / cfg["experiment"] / f"stage{stage}"
