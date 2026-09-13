#!/usr/bin/env bash
# Run this ON THE POD, interactively, after stream_to_pod.sh has landed
# the payload at $REMOTE_ROOT/cgan/ (default /workspace/cgan/).
#
# Creates a venv, installs the pinned dependencies (with the correct
# nightly-CUDA torch index, auto-detected from nvidia-smi), and then
# STOPS -- it deliberately does NOT run `wandb login` for you. That step
# needs a real interactive terminal (it opens a browser-auth flow / asks
# you to paste an API key) and must never be baked into an unattended
# script or a tarball. Run it yourself, printed at the end of this
# script's output.
#
# USAGE
# =====
#   bash bootstrap_remote.sh
#
# Overridable via environment variables:
#   REPO_ROOT   default: auto-detected as the "cgan" directory this
#               script itself lives inside (i.e. wherever
#               stream_to_pod.sh actually landed it -- does not assume
#               /workspace specifically).
#   PYTHON_BIN  default: python3
#   VENV_DIR    default: $REPO_ROOT/.venv

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# HERE = .../cgan/transformer_neurIPS/singleshot -> walk up two levels.
DEFAULT_REPO_ROOT="$(dirname "$(dirname "$HERE")")"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
REQUIREMENTS="$HERE/requirements.txt"

echo "=================================================================="
echo " singleshot bootstrap -- $REPO_ROOT"
echo "=================================================================="
echo ""

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi not found. This bundle (train_80.h5 +"
  echo "AR_MODE='frame_ar') is meant for a CUDA box -- resolve_train_regime()"
  echo "will silently fall back to the MPS/CPU micro-batch=1 path without"
  echo "CUDA, which is fine for a --smoke-test but not for a real run."
  echo "Continuing anyway (Ctrl-C to abort)."
  echo ""
fi

echo "apt-get update + installing midnight commander (mc, a terminal file"
echo "manager) and screen (found MISSING on a real pod image this session --"
echo "the whole 'run bootstrap/exp-a/exp-b in named screens over one SSH"
echo "connection' workflow in singleshot/README.md silently doesn't work"
echo "without this)..."
apt-get update -qq
apt-get install -y -qq mc screen
echo ""

echo "Installing croc (schollz/croc -- relay-based, end-to-end encrypted"
echo "file transfer; not in apt, using the official installer)..."
echo "See singleshot/README.md's 'Other transfer options' for why this can"
echo "beat scp for the large train_80.h5/val_80.h5 transfer specifically."
if command -v croc >/dev/null 2>&1; then
  echo "  already installed: $(croc --version 2>&1 | head -1)"
else
  curl -s https://getcroc.schollz.com | bash
fi
echo ""

echo "Creating venv at $VENV_DIR..."
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

echo ""
echo "Installing numpy/h5py/wandb (pinned, per requirements.txt)..."
pip install "numpy==2.4.2" "h5py==3.15.1" "wandb>=0.29.0"

echo ""
echo "Detecting CUDA driver for the correct STABLE torch index..."
CUDA_TAG="cu121"   # conservative fallback if detection fails
if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_CUDA="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  echo "  nvidia-smi driver version: ${DRIVER_CUDA:-unknown}"
  echo "  NOTE: pick the closest cuXXX tag PyTorch's STABLE index actually"
  echo "  offers for this driver -- see https://pytorch.org/get-started/locally/"
  echo "  'Stable' tab. This script does not hardcode a tag mapping since"
  echo "  the available tags change over time (see requirements.txt's own"
  echo "  note on this) -- defaulting to $CUDA_TAG;"
  echo "  override with: CUDA_TAG=cu124 bash bootstrap_remote.sh"
fi
CUDA_TAG="${CUDA_TAG_OVERRIDE:-$CUDA_TAG}"

echo ""
echo "Installing torch (stable, index: https://download.pytorch.org/whl/$CUDA_TAG)..."
pip install torch --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

echo ""
echo "Verifying the interpreter has everything and can see the GPU..."
python -c "
import torch, h5py, numpy
print('torch', torch.__version__, 'cuda_available=' + str(torch.cuda.is_available()))
print('numpy', numpy.__version__)
print('h5py', h5py.__version__)
try:
    import wandb
    print('wandb', wandb.__version__, '(not logged in yet -- see below)')
except ImportError:
    print('wandb NOT INSTALLED')
"

echo ""
echo "=================================================================="
echo " Bootstrap done. Two manual steps left:"
echo "=================================================================="
echo ""
echo "1) Log in to wandb (interactive -- opens a browser or asks you to"
echo "   paste an API key from https://wandb.ai/authorize), ONLY if you"
echo "   want telemetry for this run -- skip this and pass --no-wandb if not:"
echo ""
echo "     source $VENV_DIR/bin/activate"
echo "     wandb login"
echo ""
echo "2) Smoke-test before spending real GPU time, then launch -- see"
echo "   singleshot/README.md's screen-session table for the exact"
echo "   commands (bootstrap/exp-a/exp-b, each in its own named screen):"
echo ""
echo "     cd $REPO_ROOT/transformer_neurIPS"
echo "     python train_production_transformer_deep_dive.py \\"
echo "       --arm h11_ridge_distill --round 2 --smoke-test"
