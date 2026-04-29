#!/usr/bin/env python3
"""Project-level pipeline runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TRANSFORMER_DIR = PROJECT_ROOT / "transformer"


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    description: str


TRANSFORMER_STEPS = {
    "prepare-data": Step(
        "prepare-data",
        "Ordered_010_Prepare_Dataset.py",
        "Build training_data.h5 and validation_data.h5 from latent cube files.",
    ),
    "validate-data": Step(
        "validate-data",
        "Ordered_020_DataSet_Validations.py",
        "Validate generated transformer HDF5 files.",
    ),
    "train": Step(
        "train",
        "Ordered_100_TrainTransformer_v1.py",
        "Train OrderedTransformerV1.",
    ),
    "prepare-eval": Step(
        "prepare-eval",
        "Ordered_150_Prepare_Evaluation_Dataset.py",
        "Build the evaluation HDF5 with original velocity metadata.",
    ),
    "evaluate": Step(
        "evaluate",
        "Ordered_200_EvaluateTransformer_v1.py",
        "Evaluate the trained transformer.",
    ),
    "plots": Step(
        "plots",
        "Ordered_240_EvaluateTransformer_v1_drawplots.py",
        "Draw plots from evaluation_results.json.",
    ),
    "corruption": Step(
        "corruption",
        "Ordered_300_EvaluateTransformer_v1_with_datacorruption.py",
        "Run the optional latent-corruption robustness sweep.",
    ),
}

STEP_GROUPS = {
    "all": list(TRANSFORMER_STEPS),
    "data": ["prepare-data", "validate-data"],
    "eval": ["prepare-eval", "evaluate", "plots"],
}

STEP_ALIASES = {
    "prepare": "prepare-data",
    "dataset": "prepare-data",
    "validate": "validate-data",
    "validation": "validate-data",
    "prep-eval": "prepare-eval",
    "prepare-evaluation": "prepare-eval",
    "plot": "plots",
    "draw-plots": "plots",
    "robustness": "corruption",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FDTransformer project pipelines from the repository root.",
    )
    subparsers = parser.add_subparsers(dest="pipeline", required=True)

    transformer = subparsers.add_parser(
        "transformer",
        help="Run transformer data, training, evaluation, and plotting stages.",
    )
    transformer.add_argument(
        "steps",
        nargs="*",
        default=["all"],
        help=(
            "Steps or groups to run. Steps: "
            f"{', '.join(TRANSFORMER_STEPS)}. Groups: {', '.join(STEP_GROUPS)}."
        ),
    )
    transformer.add_argument(
        "--config",
        default=None,
        help="Path to transformer_config.json. Relative paths are resolved from the current directory.",
    )
    transformer.add_argument("--python", default=sys.executable, help="Python executable to use for stage scripts.")
    transformer.add_argument("--list", action="store_true", help="List available transformer steps and exit.")
    transformer.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    transformer.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later stages when one stage fails.",
    )
    transformer.add_argument(
        "--test-run",
        action="store_true",
        help="Pass --test-run to data-preparation stages for a small smoke-test dataset.",
    )
    transformer.add_argument(
        "--limit-samples",
        default=None,
        help="Pass a sample limit to train/evaluate/corruption stages. Use none/all/0 for full data.",
    )
    transformer.add_argument("--batch-size", type=int, default=None, help="Override batch size where supported.")
    transformer.add_argument("--num-time", type=int, default=None, help="Override sequence length in time steps.")
    transformer.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default=None,
        help="Override runtime device where supported.",
    )
    return parser


def resolve_config(config_path: str | None) -> str | None:
    if not config_path:
        return None
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def expand_steps(requested_steps: list[str]) -> list[str]:
    expanded: list[str] = []
    for raw_step in requested_steps:
        step = STEP_ALIASES.get(raw_step, raw_step)
        if step in STEP_GROUPS:
            expanded.extend(STEP_GROUPS[step])
        elif step in TRANSFORMER_STEPS:
            expanded.append(step)
        else:
            valid = sorted([*TRANSFORMER_STEPS, *STEP_GROUPS, *STEP_ALIASES])
            raise ValueError(f"Unknown transformer step '{raw_step}'. Valid choices: {', '.join(valid)}")

    deduped: list[str] = []
    for step in expanded:
        if step not in deduped:
            deduped.append(step)
    return deduped


def command_for_step(args: argparse.Namespace, step_name: str, config_path: str | None) -> list[str]:
    step = TRANSFORMER_STEPS[step_name]
    command = [args.python, str(TRANSFORMER_DIR / step.script)]

    if config_path:
        command.extend(["--config", config_path])

    if args.test_run and step_name in {"prepare-data", "prepare-eval"}:
        command.append("--test-run")
    if args.limit_samples is not None and step_name in {"train", "evaluate", "corruption"}:
        command.extend(["--limit-samples", str(args.limit_samples)])
    if args.batch_size is not None and step_name in {"train", "evaluate", "corruption"}:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.num_time is not None and step_name in {"prepare-data", "train", "prepare-eval", "evaluate", "corruption"}:
        command.extend(["--num-time", str(args.num_time)])
    if args.device is not None and step_name in {"train", "evaluate", "corruption"}:
        command.extend(["--device", args.device])

    return command


def print_transformer_steps() -> None:
    print("Transformer steps:")
    for step in TRANSFORMER_STEPS.values():
        print(f"  {step.name:13} {step.description}")
    print("\nGroups:")
    for group, steps in STEP_GROUPS.items():
        print(f"  {group:13} {', '.join(steps)}")


def run_transformer(args: argparse.Namespace) -> int:
    if args.list:
        print_transformer_steps()
        return 0

    config_path = resolve_config(args.config)
    steps = expand_steps(args.steps)

    for step_name in steps:
        command = command_for_step(args, step_name, config_path)
        display = " ".join(command)
        print(f"\n=== transformer:{step_name} ===")
        print(display)

        if args.dry_run:
            continue

        result = subprocess.run(command, cwd=str(TRANSFORMER_DIR), check=False)
        if result.returncode != 0:
            print(f"Step '{step_name}' failed with exit code {result.returncode}.", file=sys.stderr)
            if not args.continue_on_error:
                return result.returncode

    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.pipeline == "transformer":
        try:
            return run_transformer(args)
        except ValueError as exc:
            parser.error(str(exc))
    parser.error(f"Unknown pipeline: {args.pipeline}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
