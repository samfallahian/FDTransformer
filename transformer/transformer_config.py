import copy
import json
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "transformer_config.json"

DEFAULT_CONFIG = {
    "paths": {
        "latent_input_root": "data/Final_Cubed_OG_Data_wLatent",
        "transformer_input_dir": "data/transformer_input",
        "training_h5": "data/transformer_input/training_data.h5",
        "validation_h5": "data/transformer_input/validation_data.h5",
        "evaluation_input_root": "data/new-version/orig-data",
        "evaluation_h5": "data/new-version/evaluation_data_new.h5",
        "transformer_checkpoint": "best_ordered_transformer_v1.pt",
        "encoder_checkpoint": "Model_GEN3_05_AttentionSE_absolute_best_scripted.pt",
        "checkpoint_dir": ".",
        "evaluation_results_json": "evaluation_results.json",
        "pred_gt_pickle": "evaluation_pred_gt.pkl",
        "plots_dir": "plots",
        "corruption_csv": "corruption_deterioration_metrics.csv",
        "corruption_plot": "corruption_deterioration_plot.png",
        "training_process_viz": "training_process_viz.png"
    },
    "data": {
        "num_samples": 250000,
        "test_num_samples": 1000,
        "seed": 42,
        "num_time": 80,
        "num_x": 26,
        "input_dim": 52,
        "latent_dim": 47,
        "workers": 8,
        "evaluation_start_t": 100
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 0.0001,
        "epochs": 100,
        "max_checkpoints": 5,
        "staircase_eval_freq": 0,
        "limit_samples": None,
        "num_workers": 2,
        "device": "auto",
        "checkpoint_base_name": "best_ordered_transformer_v1",
        "wandb_project": "transformer_OG_prepared_cubes",
        "wandb_mode": "online"
    },
    "model": {
        "embed_size": 256,
        "n_heads": 8,
        "n_layers": 6,
        "dropout": 0.1,
        "bias": True
    },
    "evaluation": {
        "batch_size": 8,
        "micro_batch_size": 4,
        "num_workers": 2,
        "prefetch_factor": 2,
        "limit_samples": None,
        "run_cpu_parallel": False,
        "enable_metrics": True,
        "enable_staircase": False,
        "enable_interleave": False,
        "pred_gt_export_time_steps": 0,
        "pred_gt_decode_chunk": 4096,
        "triplet_idx": 62,
        "rmse_limit": 0.05,
        "device": "auto"
    },
    "corruption": {
        "batch_size": 8,
        "limit_samples": 100,
        "levels": 101,
        "device": "auto",
        "triplet_idx": 62
    }
}


def deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    path = config_path or os.getenv("TRANSFORMER_CONFIG") or str(DEFAULT_CONFIG_PATH)
    if path and os.path.exists(os.path.expanduser(os.path.expandvars(path))):
        resolved = Path(os.path.expanduser(os.path.expandvars(path))).resolve()
        with resolved.open("r", encoding="utf-8") as f:
            deep_update(config, json.load(f))
        config["_config_path"] = str(resolved)
    else:
        config["_config_path"] = None
    return config


def add_config_arg(parser):
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to transformer_config.json. If omitted, TRANSFORMER_CONFIG is used, "
            "then ./transformer_config.json if it exists."
        ),
    )


def resolve_path(path_value, base_dir=PROJECT_DIR):
    if path_value is None:
        return None
    path = Path(os.path.expanduser(os.path.expandvars(str(path_value))))
    if not path.is_absolute():
        path = Path(base_dir) / path
    return str(path.resolve())


def ensure_parent_dir(path_value):
    if not path_value:
        return
    parent = Path(path_value).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def optional_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null", "all", "0"}:
        return None
    return int(text)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")
