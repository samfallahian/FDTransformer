#!/usr/bin/env bash
# Builds ONE tar.gz containing everything needed on the pod: code, data,
# the recovered checkpoint, the AE decoder, and this directory's own
# bootstrap files. Run this LOCALLY; hand the resulting file to
# scp_to_pod.sh (or scp it yourself).
#
# This used to stream straight over SSH with no local archive at all
# (see git history / OVERVIEW.md), back when this machine had very
# little free disk space. That constraint is gone now, and a real local
# .tar.gz is easier to inspect/confirm before sending -- `ls -la` it,
# `tar -tzf` it to list contents, re-use it for multiple pods, etc.
#
# Compression is `gzip -1` (fastest level), not the default -6: the
# dominant payload (train_80.h5, val_80.h5) is already internally
# gzip-compressed (`prepare_data.py`'s `compression='gzip'`), so a
# higher compression level would spend a lot more CPU time for close to
# zero extra size reduction on those files -- -1 gets the code files
# compressed reasonably and doesn't waste minutes re-squeezing data that
# won't shrink further.
#
# USAGE
# =====
#   bash singleshot/build_tarball.sh
#
# Overridable via environment variables:
#   OUT_FILE   default: singleshot/cgan_singleshot_payload.tar.gz

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMER_DIR="$(dirname "$HERE")"
REPO_ROOT="$(dirname "$TRANSFORMER_DIR")"
REPO_PARENT="$(dirname "$REPO_ROOT")"
REPO_NAME="$(basename "$REPO_ROOT")"

OUT_FILE="${OUT_FILE:-$HERE/cgan_singleshot_payload.tar.gz}"

# Paths relative to REPO_PARENT, so the tar naturally contains
# "$REPO_NAME/..." (== "cgan/...") -- extracting at $REMOTE_ROOT (see
# scp_to_pod.sh) then produces exactly $REMOTE_ROOT/cgan/... with zero
# renaming needed on either end.
PAYLOAD=(
  "$REPO_NAME/transformer_neurIPS/train_production_transformer_deep_dive.py"
  "$REPO_NAME/transformer_neurIPS/model_variants.py"
  "$REPO_NAME/transformer_neurIPS/sweep_deep_dive.py"
  "$REPO_NAME/transformer_neurIPS/run_sweep_h300_production_8h.sh"
  "$REPO_NAME/transformer_neurIPS/data/train_80.h5"
  "$REPO_NAME/transformer_neurIPS/data/val_80.h5"
  "$REPO_NAME/transformer_neurIPS/saved_models/r2_s7_h9_scaled_rollout_best.pt"
  "$REPO_NAME/transformer_neurIPS/saved_models/r2_s7_h9_scaled_rollout_best_scripted.pt"
  "$REPO_NAME/encoder/autoencoderGEN3/saved_models_production/Model_GEN3_05_AttentionSE_absolute_best_scripted.pt"
  "$REPO_NAME/transformer_neurIPS/singleshot/requirements.txt"
  "$REPO_NAME/transformer_neurIPS/singleshot/bootstrap_remote.sh"
)

echo "=================================================================="
echo " Building singleshot tarball"
echo "=================================================================="
echo ""
echo "Checking required local files..."
missing=0
total_bytes=0
for rel in "${PAYLOAD[@]}"; do
  f="$REPO_PARENT/$rel"
  if [[ -f "$f" ]]; then
    size="$(du -h "$f" 2>/dev/null | cut -f1)"
    bytes="$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)"
    total_bytes=$((total_bytes + bytes))
    printf "  [OK]      %-80s (%s)\n" "$rel" "$size"
  else
    printf "  [MISSING] %s\n" "$rel"
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo ""
  echo "One or more required files are missing -- fix the path above (or" >&2
  echo "regenerate it per OVERVIEW.md) before building the tarball." >&2
  exit 1
fi
echo ""
printf "Total uncompressed payload: ~%.1f GB\n" \
  "$(echo "$total_bytes / 1073741824" | bc -l 2>/dev/null || echo "?")"
echo ""

FREE_GB="$(df -g "$REPO_PARENT" 2>/dev/null | awk 'NR==2{print $4}')"
if [[ -n "$FREE_GB" ]]; then
  echo "Free disk space at destination: ${FREE_GB}GB"
fi
echo ""
echo "Building $OUT_FILE (gzip -1, fastest level -- see header comment"
echo "for why higher levels don't help here)..."
echo ""

PV_CMD="cat"
if command -v pv >/dev/null 2>&1; then
  PV_CMD="pv -s $total_bytes"
fi

tar -cf - -C "$REPO_PARENT" "${PAYLOAD[@]}" \
  | $PV_CMD \
  | gzip -1 \
  > "$OUT_FILE"

FINAL_SIZE="$(du -h "$OUT_FILE" | cut -f1)"
echo ""
echo "Done: $OUT_FILE ($FINAL_SIZE)"
echo ""
echo "Sanity-check the contents without extracting:"
echo "  tar -tzf $OUT_FILE"
echo ""
echo "Next: hand this file to scp_to_pod.sh, e.g."
echo "  POD_HOST=<ip> POD_PORT=<port> bash $HERE/scp_to_pod.sh"
