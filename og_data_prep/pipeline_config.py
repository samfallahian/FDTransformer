"""Shared path configuration helpers for the OG data-preparation scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "pipeline_config.json"
MAPPING_FILENAME = "df_all_possible_combinations_with_neighbors.csv"


def add_config_argument(parser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path configuration file. Defaults to {DEFAULT_CONFIG_PATH.name}.",
    )


def load_config(config_path: str | None = None) -> tuple[dict[str, Any], Path]:
    """Load the local JSON path configuration."""
    path = Path(config_path).expanduser() if config_path else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")

    return raw, path.resolve()


def get_paths(config_path: str | None = None) -> dict[str, str | None]:
    config, path = load_config(config_path)
    base_dir = path.parent

    nested_paths = config.get("paths", {})
    if isinstance(nested_paths, dict):
        merged = {**config, **nested_paths}
    else:
        merged = dict(config)

    root_path = _resolve(merged.get("root_path") or merged.get("data_root"), base_dir)
    metadata_location = _resolve(merged.get("metadata_location"), base_dir)

    paths = {
        "root_path": root_path,
        "metadata_location": metadata_location,
        "raw_data_dir": _first_path(merged, base_dir, "raw_data_dir", "unmodified_data_dir", "raw_input"),
        "unmodified_data_dir": _first_path(merged, base_dir, "unmodified_data_dir", "raw_data_dir", "raw_input"),
        "corrected_data_dir": _first_path(merged, base_dir, "corrected_data_dir"),
        "scaled_data_dir": _first_path(merged, base_dir, "scaled_data_dir"),
        "cubed_data_dir": _first_path(merged, base_dir, "cubed_data_dir"),
        "final_cubed_data_dir": _first_path(merged, base_dir, "final_cubed_data_dir", "training_data_path"),
        "final_latent_data_dir": _first_path(merged, base_dir, "final_latent_data_dir"),
        "autoencoder_dataset_dir": _first_path(merged, base_dir, "autoencoder_dataset_dir", "dest_root"),
        "cube_mapping_csv": _first_path(merged, base_dir, "cube_mapping_csv", "mapping_csv", "filter_csv_path"),
        "autoencoder_model_path": _first_path(merged, base_dir, "autoencoder_model_path", "model_path"),
        "validation_input_file": _first_path(merged, base_dir, "validation_input_file"),
        "extremes_report_path": _first_path(merged, base_dir, "extremes_report_path"),
        "validation_results_csv": _first_path(merged, base_dir, "validation_results_csv"),
        "output_directory": _first_path(merged, base_dir, "output_directory"),
    }

    if root_path:
        derived = {
            "unmodified_data_dir": "Unmodified_OG_Data",
            "corrected_data_dir": "Corrected_OG_Data",
            "scaled_data_dir": "Scaled_OG_Data",
            "cubed_data_dir": "Cubed_OG_Data",
            "final_cubed_data_dir": "Final_Cubed_OG_Data",
            "final_latent_data_dir": "Final_Cubed_OG_Data_wLatent",
            "autoencoder_dataset_dir": "",
        }
        for key, suffix in derived.items():
            if not paths[key]:
                paths[key] = str(Path(root_path) / suffix) if suffix else root_path

    if not paths["cube_mapping_csv"] and metadata_location:
        paths["cube_mapping_csv"] = str(Path(metadata_location).parent.parent / "cube_centroid_mapping" / MAPPING_FILENAME)

    return paths


def resolve_path(
    config_path: str | None,
    key: str,
    override: str | None = None,
    *,
    required: bool = True,
) -> str | None:
    if override:
        return _resolve(override, Path.cwd())

    value = get_paths(config_path).get(key)
    if required and not value:
        raise ValueError(f"Missing `{key}`. Set it in the config file or pass the matching CLI argument.")
    return value


def _first_path(config: dict[str, Any], base_dir: Path, *keys: str) -> str | None:
    for key in keys:
        value = config.get(key)
        if value:
            return _resolve(value, base_dir)
    return None


def _resolve(value: Any, base_dir: Path) -> str | None:
    if value in (None, ""):
        return None

    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())
