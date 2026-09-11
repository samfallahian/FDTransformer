#!/usr/bin/env bash
# Copies the tarball built by build_tarball.sh to a pod via scp, then
# extracts it there via one ssh command. Two clearly separate, easy-to-
# confirm steps -- you can watch the scp transfer complete, check the
# file landed with the right size, THEN extract, rather than one opaque
# pipeline.
#
# USAGE
# =====
#   bash singleshot/build_tarball.sh          # build it first (once)
#   POD_HOST=1.2.3.4 POD_PORT=22 bash singleshot/scp_to_pod.sh
#
# Overridable via environment variables:
#   POD_HOST      required -- the pod's SSH host/IP
#   POD_PORT      default: 22
#   POD_USER      default: root
#   SSH_KEY       default: ~/.ssh/id_ed25519
#   REMOTE_ROOT   default: /workspace  -- extraction lands at
#                 $REMOTE_ROOT/cgan/...
#   TARBALL       default: singleshot/cgan_singleshot_payload.tar.gz
#                 (the file build_tarball.sh produces)
#   KEEP_REMOTE_TARBALL   set to 1 to leave the .tar.gz on the pod after
#                 extracting (default: deleted, to not waste pod disk)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

POD_HOST="${POD_HOST:-}"
POD_PORT="${POD_PORT:-22}"
POD_USER="${POD_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace}"
TARBALL="${TARBALL:-$HERE/cgan_singleshot_payload.tar.gz}"
KEEP_REMOTE_TARBALL="${KEEP_REMOTE_TARBALL:-0}"

if [[ -z "$POD_HOST" ]]; then
  echo "POD_HOST is required, e.g.: POD_HOST=1.2.3.4 POD_PORT=22222 bash $0" >&2
  exit 1
fi

if [[ ! -f "$TARBALL" ]]; then
  echo "$TARBALL does not exist -- run build_tarball.sh first:" >&2
  echo "  bash $HERE/build_tarball.sh" >&2
  exit 1
fi

LOCAL_SIZE_BYTES="$(stat -f%z "$TARBALL" 2>/dev/null || stat -c%s "$TARBALL" 2>/dev/null)"
LOCAL_SIZE_HUMAN="$(du -h "$TARBALL" | cut -f1)"
REMOTE_TAR_PATH="${REMOTE_ROOT}/$(basename "$TARBALL")"

echo "=================================================================="
echo " scp -> ${POD_USER}@${POD_HOST}:${POD_PORT}${REMOTE_TAR_PATH}"
echo "=================================================================="
echo " Local file:  $TARBALL ($LOCAL_SIZE_HUMAN)"
echo ""

echo "Step 1/2: copying the tarball..."
ssh -p "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
  "$POD_USER@$POD_HOST" "mkdir -p '$REMOTE_ROOT'"
scp -P "$POD_PORT" -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new \
  "$TARBALL" "$POD_USER@$POD_HOST:$REMOTE_TAR_PATH"

echo ""
echo "Confirming the remote file size matches..."
REMOTE_SIZE_BYTES="$(ssh -p "$POD_PORT" -i "$SSH_KEY" "$POD_USER@$POD_HOST" \
  "stat -c%s '$REMOTE_TAR_PATH' 2>/dev/null || stat -f%z '$REMOTE_TAR_PATH'")"
if [[ "$LOCAL_SIZE_BYTES" != "$REMOTE_SIZE_BYTES" ]]; then
  echo "SIZE MISMATCH: local=$LOCAL_SIZE_BYTES remote=$REMOTE_SIZE_BYTES bytes." >&2
  echo "The transfer likely got truncated -- re-run this script before extracting." >&2
  exit 1
fi
echo "  OK -- $REMOTE_SIZE_BYTES bytes on both ends."
echo ""

echo "Step 2/2: extracting on the pod at $REMOTE_ROOT..."
CLEANUP_CMD="rm -f '$REMOTE_TAR_PATH'"
if [[ "$KEEP_REMOTE_TARBALL" == "1" ]]; then
  CLEANUP_CMD=":"
fi
ssh -p "$POD_PORT" -i "$SSH_KEY" "$POD_USER@$POD_HOST" \
  "tar -xzf '$REMOTE_TAR_PATH' -C '$REMOTE_ROOT' && $CLEANUP_CMD"

echo ""
echo "Done. Payload extracted at ${REMOTE_ROOT}/cgan/."
echo ""
echo "Next: SSH in and run the bootstrap script (it ends by printing the"
echo "wandb login step -- deliberately not run for you):"
echo ""
echo "  ssh -p $POD_PORT -i $SSH_KEY $POD_USER@$POD_HOST"
echo "  bash ${REMOTE_ROOT}/cgan/transformer_neurIPS/singleshot/bootstrap_remote.sh"
