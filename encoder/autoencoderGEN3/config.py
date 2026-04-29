#!/usr/bin/env python3
"""Shared configuration helpers for GEN3 autoencoder scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

AUTOENCODER_DIR = Path(__file__).resolve().parent
ENCODER_DIR = AUTOENCODER_DIR.parent
PROJECT_ROOT = ENCODER_DIR.parent
DEFAULT_CONFIG_PATH = AUTOENCODER_DIR / "config.json"


def add_config_argument(parser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional JSON config path. Defaults to AUTOENCODER_GEN3_CONFIG, "
            "then autoencoderGEN3/config.json when present."
        ),
    )


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load JSON config if one is supplied or present, otherwise return an empty config."""
    path = config_path or os.environ.get("AUTOENCODER_GEN3_CONFIG")
    if path is None and DEFAULT_CONFIG_PATH.exists():
        path = str(DEFAULT_CONFIG_PATH)
    if path is None:
        return {}

    cfg_path = resolve_path(path, base_dir=Path.cwd())
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def config_get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def resolve_path(path: str | os.PathLike[str], base_dir: str | os.PathLike[str] = PROJECT_ROOT) -> str:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        expanded = Path(base_dir) / expanded
    return str(expanded.resolve())


def optional_path(
    value: str | os.PathLike[str] | None,
    base_dir: str | os.PathLike[str] = PROJECT_ROOT,
) -> str | None:
    if value is None or value == "":
        return None
    return resolve_path(value, base_dir=base_dir)


def configured_path(
    config: Mapping[str, Any],
    dotted_key: str,
    default: str | os.PathLike[str] | None = None,
    base_dir: str | os.PathLike[str] = PROJECT_ROOT,
) -> str | None:
    return optional_path(config_get(config, dotted_key, default), base_dir=base_dir)


def choose_path(
    cli_value: str | None,
    config: Mapping[str, Any],
    dotted_key: str,
    default: str | os.PathLike[str] | None = None,
    base_dir: str | os.PathLike[str] = PROJECT_ROOT,
) -> str | None:
    return optional_path(cli_value, base_dir=Path.cwd()) or configured_path(
        config,
        dotted_key,
        default=default,
        base_dir=base_dir,
    )


def default_checkpoint_path(filename: str, production: bool = True) -> str:
    folder = "saved_models_production" if production else "saved_models"
    return str(AUTOENCODER_DIR / folder / filename)
