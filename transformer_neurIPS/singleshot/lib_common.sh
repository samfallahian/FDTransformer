#!/usr/bin/env bash
# lib_common.sh -- tiny generic helpers with no dependency on any
# particular transfer method. Extracted out of lib_croc.sh so that
# scripts which no longer use croc at all (lib_scp.sh, scp_data_files.sh,
# bench_scp_variants.sh) don't need to source a croc-specific file just
# to get run_with_timeout.

# ---------------------------------------------------------------- #
# run_with_timeout SECS CMD...
#
# Portable per-command timeout -- macOS has no `timeout(1)` by
# default (confirmed: `timeout` -> "command not found" on this
# machine). Background watchdog: sleep SECS, then SIGKILL the target
# if it's still alive. Always `wait`s for the target's real exit
# code, whichever way it ends.
# ---------------------------------------------------------------- #
run_with_timeout() {
  local secs="$1"; shift
  "$@" &
  local target=$!
  ( sleep "$secs"; kill -9 "$target" 2>/dev/null ) &
  local watchdog=$!
  local rc=0
  wait "$target" 2>/dev/null || rc=$?
  kill "$watchdog" 2>/dev/null
  wait "$watchdog" 2>/dev/null
  return "$rc"
}
