"""Compatibility path loader backed by ``og_data_prep/pipeline_config.json``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from og_data_prep.pipeline_config import DEFAULT_CONFIG_PATH, get_paths  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ProjectPaths:
    """Expose common project paths for older helper scripts."""

    def __init__(self, filename: str | Path | None = None):
        self.filename = str(filename or DEFAULT_CONFIG_PATH)
        paths = get_paths(self.filename)

        self.root_path = paths.get("root_path")
        self.metadata_location = paths.get("metadata_location") or str(
            PROJECT_ROOT / "configs" / "Experiment_MetaData.json"
        )
        self.logging_path = paths.get("logging_path")
        self.raw_input = paths.get("raw_data_dir")
        self.output_directory = paths.get("output_directory") or paths.get("corrected_data_dir")
        self.logging_level = paths.get("logging_level") or "INFO"
        self.training_data_path = paths.get("final_cubed_data_dir")


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
