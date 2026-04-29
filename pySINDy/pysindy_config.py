import argparse
import copy
import json
import os
import sys
from pathlib import Path


CONFIG_FILE = Path(__file__).with_name("config.json")
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "project_root": "..",
    "data": {
        "evaluation_h5": ""
    },
    "models": {
        "transformer_checkpoint": "../transformer/best_ordered_transformer_v1.pt",
        "encoder_checkpoint": "../encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
    },
    "outputs": {
        "output_dir": ".",
        "documentation_dir": "../Documentation",
        "raw_grad": "raw_data_grad.npz",
        "encoded_grad": "encoded_data_grad.npz",
        "predicted_grad": "predicted_data_grad.npz",
        "raw_extended": "raw_extended.npz",
        "encoded_extended": "encoded_extended.npz",
        "predicted_extended": "predicted_extended.npz",
        "evaluation_results": "evaluation_results.csv",
        "evaluation_summary": "evaluation_summary.png",
        "cross_source_results": "cross_source_results.csv",
        "extended_physics_results": "extended_physics_results.csv",
        "noise_robustness": "noise_robustness.png",
        "all_params_recovery_results": "all_params_recovery_results.csv",
        "all_params_coefficient_trends": "all_params_coefficient_trends.png",
        "staircase_results": "staircase_physics_results.csv",
        "staircase_plot": "staircase_physics_trends.png",
        "staircase_plot_pdf": "staircase_physics_trends.pdf",
        "interpolation_results": "interpolation_physics_results.csv"
    },
    "runtime": {
        "device": "cpu",
        "n_search": 100000,
        "p_target": 5.2,
        "triplet_idx": 62,
        "batch_size": 64,
        "reynolds_numbers": [3.6, 4.6, 5.2, 6.4, 6.6, 7.2, 7.8, 8.4, 10.4, 11.4],
        "temporal_contexts": [7, 6, 5, 4, 3, 2, 1],
        "grid_sample_limit": 1000,
        "grid_point_limit": 30,
        "random_seed": 42
    }
}


def add_config_argument(parser):
    parser.add_argument(
        "--config",
        default=str(CONFIG_FILE),
        help="Path to config JSON. Defaults to pySINDy/config.json when it exists.",
    )
    return parser


def add_common_path_arguments(parser):
    parser.add_argument("--h5-path", help="Override data.evaluation_h5 from the config.")
    parser.add_argument("--output-dir", help="Override outputs.output_dir from the config.")
    parser.add_argument("--documentation-dir", help="Override outputs.documentation_dir from the config.")
    parser.add_argument("--project-root", help="Override project_root from the config.")
    parser.add_argument("--encoder-checkpoint", help="Override models.encoder_checkpoint from the config.")
    parser.add_argument("--transformer-checkpoint", help="Override models.transformer_checkpoint from the config.")
    return parser


def add_runtime_arguments(parser):
    parser.add_argument("--device", help="Override runtime.device from the config.")
    parser.add_argument("--n-search", type=int, help="Override runtime.n_search from the config.")
    parser.add_argument("--p-target", type=float, help="Override runtime.p_target from the config.")
    parser.add_argument("--triplet-idx", type=int, help="Override runtime.triplet_idx from the config.")
    parser.add_argument("--batch-size", type=int, help="Override runtime.batch_size from the config.")
    return parser


def load_config(config_path=None):
    path = Path(config_path).expanduser() if config_path else CONFIG_FILE
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["_config_path"] = str(path.resolve())
    config["_config_dir"] = str(path.resolve().parent)

    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            _deep_update(config, json.load(f))
    elif config_path and Path(config_path) != CONFIG_FILE:
        raise FileNotFoundError(f"Config file not found: {path}")

    return config


def apply_overrides(config, args):
    override_map = {
        "h5_path": ("data", "evaluation_h5"),
        "output_dir": ("outputs", "output_dir"),
        "documentation_dir": ("outputs", "documentation_dir"),
        "project_root": ("project_root",),
        "encoder_checkpoint": ("models", "encoder_checkpoint"),
        "transformer_checkpoint": ("models", "transformer_checkpoint"),
        "device": ("runtime", "device"),
        "n_search": ("runtime", "n_search"),
        "p_target": ("runtime", "p_target"),
        "triplet_idx": ("runtime", "triplet_idx"),
        "batch_size": ("runtime", "batch_size"),
    }
    for arg_name, keys in override_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            set_config_value(config, keys, value)
    return config


def load_config_from_args(args):
    config = load_config(getattr(args, "config", None))
    return apply_overrides(config, args)


def get_config_value(config, keys, default=None):
    if isinstance(keys, str):
        keys = tuple(keys.split("."))
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def set_config_value(config, keys, value):
    value_ref = config
    for key in keys[:-1]:
        value_ref = value_ref.setdefault(key, {})
    value_ref[keys[-1]] = value


def resolve_path(config, keys, base_dir=None, create_parent=False, required=False):
    raw_value = get_config_value(config, keys)
    label = ".".join(keys) if not isinstance(keys, str) else keys
    if raw_value in (None, ""):
        if required:
            raise ValueError(
                f"Missing required path '{label}'. Set it in config.json or pass a command-line override."
            )
        return None

    path = Path(os.path.expandvars(str(raw_value))).expanduser()
    if not path.is_absolute():
        root = Path(base_dir) if base_dir else Path(config.get("_config_dir", BASE_DIR))
        path = root / path
    path = path.resolve()

    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    if required and not path.exists():
        raise FileNotFoundError(f"Configured path for '{label}' does not exist: {path}")
    return path


def output_path(config, output_key, create_parent=True):
    output_dir = resolve_path(config, ("outputs", "output_dir")) or BASE_DIR
    filename = get_config_value(config, ("outputs", output_key), output_key)
    path = Path(str(filename)).expanduser()
    if not path.is_absolute():
        path = output_dir / path
    path = path.resolve()
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def documentation_path(config, output_key, create_parent=True):
    doc_dir = resolve_path(config, ("outputs", "documentation_dir")) or BASE_DIR
    filename = get_config_value(config, ("outputs", output_key), output_key)
    path = Path(str(filename)).expanduser()
    if not path.is_absolute():
        path = doc_dir / path
    path = path.resolve()
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def configure_project_imports(config):
    project_root = resolve_path(config, ("project_root",), required=True)
    transformer_dir = project_root / "transformer"
    for path in (project_root, transformer_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return project_root


def select_device(torch_module, configured_device=None):
    if configured_device and configured_device != "auto":
        return configured_device
    if torch_module.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _deep_update(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def make_parser(description, common_paths=True, runtime=True):
    parser = argparse.ArgumentParser(description=description)
    add_config_argument(parser)
    if common_paths:
        add_common_path_arguments(parser)
    if runtime:
        add_runtime_arguments(parser)
    return parser
