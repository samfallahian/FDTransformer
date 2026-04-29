"""Compatibility path loader backed by the autoencoder JSON config."""

from __future__ import annotations

from pathlib import Path

try:
    from .config import (
        DEFAULT_CONFIG_PATH,
        config_get,
        configured_path,
        load_config,
    )
except ImportError:
    from config import (
        DEFAULT_CONFIG_PATH,
        config_get,
        configured_path,
        load_config,
    )


class ProjectPaths:
    """Expose the path attributes expected by older autoencoder scripts."""

    def __init__(self, filename: str | Path | None = None):
        self.filename = str(filename) if filename is not None else str(DEFAULT_CONFIG_PATH)
        config = load_config(str(filename) if filename is not None else None)

        data_root = configured_path(config, "data.data_root")
        self.root_path = data_root
        self.metadata_location = configured_path(config, "paths.metadata_location")
        self.logging_path = configured_path(config, "paths.logging_path")
        self.raw_input = configured_path(config, "data.validation_data_dir")
        self.output_directory = configured_path(config, "paths.results_dir")
        self.logging_level = config_get(config, "logging.level", "INFO")
        self.training_data_path = data_root


if __name__ == "__main__":
    project_paths = ProjectPaths()
    print("\nConfiguration loaded successfully:")
    print(f"Metadata Location: {project_paths.metadata_location}")
    print(f"Root Path: {project_paths.root_path}")
    print(f"Logging Path: {project_paths.logging_path}")
    print(f"Raw Input: {project_paths.raw_input}")
    print(f"Output Directory: {project_paths.output_directory}")
    print(f"Logging Level: {project_paths.logging_level}")
    print(f"Training Data Path: {project_paths.training_data_path}")
