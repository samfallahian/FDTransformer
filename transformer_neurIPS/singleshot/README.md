# `singleshot` -- one-shot deploy package for a rented GPU box

Everything needed to go from "empty rented pod" to "training running,
telemetry flowing to wandb": code, data, the recovered checkpoint, the
AE decoder, and the dependency/bootstrap scripts. No tarball -- files
transfer individually, in two phases, so bootstrapping the venv and
transferring the large data files can happen **at the same time**
instead of one blocking the other. On the pod side, everything runs
inside named `screen` sessions over a SINGLE SSH connection -- see
"Running everything on the pod via screen" below.

## What's in here

| File | Role |
|---|---|
| `provision_and_run.sh` | Run **locally**. The full lifecycle in one command: create a pod via `runpodctl`, wait for SSH, send env files, bootstrap the pod WHILE the data files transfer in the background, run a time-boxed training job, pull the new artifacts back (retried a few times, with progress printed for every attempt), terminate the pod -- but only if that pull-back actually succeeded; a genuine failure (as opposed to "no files matched") leaves the pod running rather than risk deleting undelivered results (`FORCE_TERMINATE_ON_PULL_FAILURE=1` overrides this). See "Fully automated lifecycle" below. |
| `scp_env_files.sh` | Run **locally**. Phase 1: sends code, the recovered checkpoint(s) (including the MPS baseline for Experiment A below), the ridge map, the AE decoder, and this directory's own `requirements.txt`/`bootstrap_remote.sh` -- everything except the large data files. Small, finishes in seconds. |
| `scp_data_files.sh` | Run **locally**. Phase 2: splits `train_80.h5` (~9.9G) and `val_80.h5` (~4.2G) into `LANES` pieces each (default 10) and sends them sequentially per file, `LANES` pieces concurrently within each file, via plain `scp`/ssh -- **confirmed** to hit 268.3 MB/s raw at 10 lanes (iperf3), close to this link's ~312 MB/s ceiling. Verifies size AND sha256 on reassembly. Independent of phase 1 -- run it any time before, during, or after phase 1. Both files are stored uncompressed (`decompress_h5.py`, see below) -- slightly larger on disk/wire, but no more single-threaded gzip decompression cost on every load. |
| `bootstrap_remote.sh` | Run **on the pod** (normally inside the `bootstrap` screen, see below). Creates a venv, installs pinned deps + `mc` (midnight commander) + `screen` + the correct CUDA **stable** (not nightly) torch build. Does NOT run `wandb login` for you if run standalone/by hand -- but `provision_and_run.sh --wandb=KEY` handles that non-interactively itself right before launching training, no manual step needed on that path. Only needs phase 1's files -- can run while phase 2 is still transferring. |
| `requirements.txt` | Copy of `/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/requirements_sweep.txt` (the verified-minimal dependency set), kept self-contained here. |
| `lib_common.sh` | Sourceable library of generic helpers (currently just `run_with_timeout`) with no dependency on any particular transfer method. |
| `lib_scp.sh` | Sourceable library of the parallel-scp split/send/reassemble/verify logic (`scp_send_split`) -- the actual production data-transfer mechanism, used by both `scp_data_files.sh` (the real thing) and `bench_scp_variants.sh` (the benchmark/smoke-test), so the two can't drift apart. |
| `bench_scp_variants.sh` | Run **locally**. Defaults to a LOCAL smoke test (no pod, no cost) against `localhost` via a real ssh connection, validating the split/concurrent-launch/reassemble/verify logic; point it at a real pod (`POD_HOST=...`) to benchmark for real. See its header for the full croc-vs-scp comparison story. |
| `lib_croc.sh` / `bench_croc_variants.sh` / `CROC_JOURNEY.md` | **Historical -- croc has been dropped from this pipeline entirely.** croc's own split-lanes phase measured far below plain scp at the same lane count (PAKE handshake + its own encryption layer on top of the already-encrypted SSH connection, for no security benefit here). Kept only as a record of that investigation and for `bench_croc_variants.sh` if you ever want to re-confirm the comparison; `bootstrap_remote.sh` no longer installs croc, and neither `scp_data_files.sh` nor `provision_and_run.sh` reference it. |

## Why not a tarball

An earlier version of this package built one `.tar.gz` (code + ~11.3GB
of data) and sent it as a single blob. Two problems with that: nothing
could start on the pod until the ENTIRE 11.3GB had landed, even though
bootstrapping the venv needs none of the data; and building the archive
first meant paying to write ~11GB locally, then read it back, before
any of it even started moving over the network. Splitting into
individual files, small-first, fixes both: phase 1 (a few small files)
finishes almost immediately, so you can start bootstrapping right away
while phase 2 (the actual bottleneck, `train_80.h5` + `val_80.h5`) is
still moving in the background or in a separate terminal.

## Sending the files

**Phase 1 (fast) -- code, checkpoint, ridge map, AE decoder:**

```bash
POD_HOST=<pod ip> POD_PORT=<ssh port> bash /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/scp_env_files.sh
```

Checks every required file exists locally first (same `[OK]`/`[MISSING]`
table every `run_sweep_h300_*.sh` launcher in this repo already uses),
then `scp`s each one individually and verifies the byte count matches on
both ends before moving to the next. Finishes in seconds. This includes
the MPS baseline checkpoint (`saved_models/r2_h11_ridge_distill_latest.pt`)
that Experiment A below resumes from -- if you retrain a new baseline,
re-run this phase before launching, or Experiment A will silently
cold-start instead (exactly the failure mode OVERVIEW.md v6.0 exists to
prevent -- always re-check the `[OK]` table above before assuming a
checkpoint made it to the pod).

**Phase 2 (slow) -- the training/validation data**, run any time before,
during, or after phase 1 -- it's independent, only needs the pod to have
somewhere to put the files:

```bash
POD_HOST=<pod ip> POD_PORT=<ssh port> bash /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/scp_data_files.sh
```

Sends `train_80.h5` and `val_80.h5` as two concurrent `scp` processes
(not sequential) and verifies both sizes match on completion.

Optional overrides on both scripts: `POD_USER` (default `root`),
`SSH_KEY` (default `/Users/kkreth/.ssh/id_ed25519`), `REMOTE_ROOT`
(default `/workspace` -- files land at `$REMOTE_ROOT/cgan/...`).

## Running everything on the pod via `screen` -- one SSH connection, not three

Older versions of this doc said "open a second terminal for bootstrap
while phase 2 transfers." You don't need that anymore: SSH in **once**,
then launch each job as its own detached `screen` session -- it keeps
running (and stays attachable) even if your SSH connection drops, which
is exactly the fragility that bit a run earlier this session (a dropped
foreground SSH session silently killed the training process with
nothing to reconnect to).

```bash
ssh -p <ssh port> -i ~/.ssh/id_ed25519 root@<pod ip>
```

Then, from that one shell, run each row below (any order -- each one
waits on its own prerequisite, so firing all three immediately is safe):

| Screen | Command | What it does |
|---|---|---|
| `bootstrap` | `screen -dmS bootstrap bash -c 'bash /workspace/cgan/transformer_neurIPS/singleshot/bootstrap_remote.sh; echo "--- bootstrap done, now tailing exp-a ---"; tail -F /workspace/cgan/transformer_neurIPS/exp_a.log'` | Builds the venv, installs deps + `mc` (see below). Once done, automatically switches to tailing Experiment A's log (`tail -F` waits for the file to exist, so this works regardless of launch order) -- this one screen becomes your live view of Experiment A without needing a 4th terminal. |
| `exp-a` | `screen -dmS exp-a bash -c 'until [ -f /workspace/cgan/.venv/bin/activate ]; do sleep 5; done; cd /workspace/cgan/transformer_neurIPS && source /workspace/cgan/.venv/bin/activate && python train_production_transformer_deep_dive.py --arm h11_ridge_distill --round 2 --max-steps 2500 --max-hours 0.5 --val-every 500 --no-wandb 2>&1 \| tee exp_a.log'` | **Resumes** from the MPS baseline checkpoint (`r2_h11_ridge_distill_latest.pt`, step 25 -- see phase 1). Waits for the venv to exist before starting, so it's safe to launch before/alongside `bootstrap`. Only start this after phase 2 (data) has also finished. |
| `exp-b` | `screen -dmS exp-b bash -c 'until [ -f /workspace/cgan/.venv/bin/activate ]; do sleep 5; done; cd /workspace/cgan/transformer_neurIPS && source /workspace/cgan/.venv/bin/activate && python train_production_transformer_deep_dive.py --arm h11_ridge_distill --round 3 --max-steps 2500 --max-hours 0.5 --val-every 500 --no-wandb --fresh --no-warm-start 2>&1 \| tee exp_b.log'` | Cold start, no baseline -- the control arm of the two-experiment comparison. `--round 3` (vs. `exp-a`'s `--round 2`) gives it distinct checkpoint/lock filenames, so both can run at once without colliding. |

**Reattach to any of them any time** (including after reconnecting from
a dropped SSH session):
```bash
screen -r bootstrap    # or exp-a / exp-b
```
Detach without killing it: `Ctrl-A` then `D`. List everything running:
`screen -ls`. If you want wandb telemetry on either experiment, attach
to a screen (or open a plain interactive shell) and run `wandb login`
yourself first, then drop `--no-wandb` from that experiment's command.

## Fully automated lifecycle: create pod -> run -> pull results -> terminate

`provision_and_run.sh` drives the whole thing via `runpodctl` -- no
manual pod creation, no copy-pasting an SSH address, no leaving a pod
running after you forget about it. Internally it gets the same
parallelism as the manual workflow above: it runs `scp_data_files.sh` in
the background, immediately opens a SEPARATE ssh connection to run
`bootstrap_remote.sh` while that transfer is still going, waits for the
data transfer to finish, then launches training over a third connection.

**One-time prerequisite** (do this yourself, in your own terminal --
never paste an API key into a chat message or a script argument):

1. Go to **`https://www.console.runpod.io/user/settings`** (RunPod
   moved this under the "console." subdomain -- the older
   `runpod.io/console/user/settings` link still works via redirect as
   of this writing, but the console URL above is the current one).
2. Expand the **API Keys** section and click **Create API Key**.
3. Give it a name, and for **Permissions** select **All (full access)**
   -- not **Restricted** or **Read Only**. `runpodctl` needs to create,
   list, and delete pods and fetch SSH connection info, all of which are
   write/management operations; RunPod's "Restricted" tier is scoped to
   per-Serverless-endpoint access control, not documented to cover Pod
   management, so "All" is the only tier confirmed to work for what
   `provision_and_run.sh` does.
4. Copy the key immediately -- RunPod shows it once.

Then, locally:

```bash
runpodctl doctor                    # interactive prompt, saves the key
# or:
export RUNPOD_API_KEY=your-key      # this shell session only
```

The script only ever calls `runpodctl user` to check that *some*
credential exists -- it never reads, prints, or stores the key itself.

Then, one command:

```bash
bash /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/provision_and_run.sh
```

Default behavior: finds an available H200 or H300 (matching every
prior run in this project), creates the pod, waits for SSH to actually
answer, sends the files (env files, then data files in the background
while bootstrap runs), runs `h11_ridge_distill` capped at **30 minutes
AND 2500 optimizer steps**, pulls back only the new checkpoint + log
files (not a second copy of the training data), then terminates the pod.

### Why `h11_ridge_distill` specifically

This is the ridge-map DISTILLATION loss (OVERVIEW.md v6.1) -- the safe
reformulation of the abandoned `h10_ridge_residual` idea: instead of
wiring the fitted ridge map into the model's own autoregressive feedback
loop as an anchor (which is what caused `h10`'s catastrophic blowup --
persistence is non-expansive under repeated feedback, the ridge map is
not), it pulls the network's one-step prediction toward the ridge map's
prediction as an extra loss term computed fresh from ground truth every
step, with no recursive chain for a bad eigendirection to compound over.

It was built and verified this session, but **only on MPS** -- and MPS
structurally can't test the thing that matters: `AR_MODE='frame_ar'`
(the autoregressive rollout loss this new term is meant to sit alongside)
is unconditionally disabled on MPS/CPU (`regime.disable_ar`), so the MPS
runs only proved the code doesn't crash and resumes correctly --
mechanical correctness, not a research result. Whether the distillation
term actually helps once the real AR loss is active too has never been
tested on real hardware. OVERVIEW.md v6.1 §31.5 says exactly this:
"needs a real CUDA shallow screen" was the explicit next step, left
undone until this run. `h11_ridge_distill` is `h9_ar_freq1`'s exact
config (the well-understood +43.78% baseline) plus the one new knob --
directly comparable to both `h9` (no distillation) and `h10`
(catastrophic failure), and deliberately cheap (30 min) so a bad result
costs little before ever considering scaling it.

**If you're watching the console output live**: right after bootstrap
finishes, you'll see `bootstrap_remote.sh` print a note about logging in
to wandb yourself -- that note is scoped to running `bootstrap_remote.sh`
standalone/by hand. In this automated path, passing `--wandb=KEY` (or
`WANDB_API_KEY=KEY`) to `provision_and_run.sh` already handles wandb
login non-interactively, right before training launches -- **ignore
that note and don't interrupt anything.**

### What you do vs. what the script does

You do exactly one thing before running it (the API key, above), and
run exactly one command. Everything else happens automatically:

1. create pod, wait for SSH
2. **in parallel**: send data files (phase 2, background, starts
   IMMEDIATELY -- it only needs SSH, no dependency on phase 1) WHILE
   sending env files (phase 1) THEN bootstrapping the pod (venv, deps,
   stable torch) over a second connection -- bootstrap itself DOES need
   phase 1 done first, since `bootstrap_remote.sh`/`requirements.txt`
   are among the files phase 1 ships
3. wait for the data transfer to finish, then run training, capped at
   30 min AND 2500 steps
4. **rsync the new checkpoint + logs back to this machine** -- automatic,
   not a step you do afterward
5. **terminate the pod** (`runpodctl pod delete`) -- also automatic,
   right after step 4, unless `KEEP_POD=1`

**This is not a background/fire-and-forget job.** Your machine has to
stay connected and the script has to keep running for the whole
duration. The parallelism in step 2 shortens the wall-clock cost
somewhat (env files, bootstrap, and the ~14GB data transfer all overlap
instead of stacking), but total wall time is still meaningfully longer than 30
minutes -- add however long the data transfer takes on your connection,
plus a few minutes of pod boot.

**Caveat on a dropped connection mid-run**: pod termination is wired to
a `trap ... EXIT`, so it fires even if something above it fails or the
script is killed -- you will not get billed forever by an orphaned pod.
But if the SSH connection to the pod drops *during* the 30-minute
training window specifically, the script aborts right there (`set -e`)
and never reaches the rsync pull-back step (it comes after, in the
script's linear order) -- the pod still gets deleted, but that run's
results would be lost with it. Set `KEEP_POD=1` if you'd rather recover
manually than risk that.

Most knobs are env vars; `--wandb=KEY`, `--round=N`, `--arm=NAME`, and
`--fresh` are also available as CLI flags (equivalent to `WANDB_API_KEY`,
`ROUND`, `ARM`, and `FRESH=1` respectively) -- different arm, different
round/budget, different GPU, reuse an existing pod, or keep the pod
alive after the run for debugging. e.g. to run `h11_ridge_distill`'s
cold-start control (exp-b) with wandb telemetry, all in one command:
```bash
bash singleshot/provision_and_run.sh --wandb=abc123yourkeyhere --round=3 --fresh
```
See the script's own header comment for the full list (`ARM`, `ROUND`,
`FRESH`, `MAX_HOURS`, `MAX_STEPS`, `GPU_ID`, `TEMPLATE_ID`, `IMAGE`,
`POD_ID`, `KEEP_POD`, `WANDB_API_KEY`, `LANES`, ...). Pod image defaults to the
`runpod-torch-v280` template (verified via `runpodctl template search
pytorch` to actually exist -- `bootstrap_remote.sh` installs its own
venv + torch on top regardless, so any real, working CUDA+Python3
image/template is fine). Set `IMAGE=<docker image>` instead to use a
specific image directly rather than a template -- verify it exists
first with `runpodctl template search <name>` or `runpodctl template
list --type official`.

**wandb**: unattended by design, so this can't do an interactive
`wandb login`. Without `WANDB_API_KEY`/`--wandb=KEY` set, the run passes
`--no-wandb`. Pass `--wandb=KEY` on the command line (unquoted, e.g.
`--wandb=abc123`) or set `WANDB_API_KEY=<key>` as an env var instead if
you'd rather it not sit in shell history -- either way, the key is
`export`ed into the remote training script's own environment, which the
wandb SDK reads directly at `wandb.init()` time. (An earlier version of
this only exported the key for a separate `wandb login --relogin`
command, which didn't persist far enough for the actual `python`
process to see it -- confirmed the hard way via a real failed run:
"wandb: ERROR ... No API key configured" despite the key being passed.
Fixed by exporting it for the whole remote script instead.)

**Honesty note on the RunPod JSON parsing**: `provision_and_run.sh`'s
parsing of `gpu list` / `pod create` / `ssh info` JSON output tries
several plausible field names defensively and ALWAYS prints the raw
output too. If a parse step comes up empty, the script stops and shows
you the raw JSON to fix the field name or fall back to the manual steps
below, rather than guessing further or silently doing the wrong thing.

## Manual step by step (what `provision_and_run.sh` automates)

Useful if you provision a pod yourself (RunPod's website, or once you've
completed the MCP sign-in below) and just want more control over timing,
or if `provision_and_run.sh`'s JSON parsing needs adjusting for your
account (see the honesty note above).

### 1. Provision a pod and get its SSH details

Via RunPod's website, `runpodctl pod create` directly, or (once signed
in) the MCP tools from "RunPod agent setup" below.

### 2. Send the files, then run everything via `screen`

Covered in full above -- "Sending the files" for the two `scp_*`
phases, then "Running everything on the pod via `screen`" for the
bootstrap/exp-a/exp-b table. `--smoke-test` is worth running once by
hand before committing to a real launch:
```bash
python /workspace/cgan/transformer_neurIPS/train_production_transformer_deep_dive.py --arm h11_ridge_distill --round 2 --smoke-test
```
Builds the model and runs the causality gate on synthetic data in
seconds, with no data/checkpoint/wandb I/O at all.

### 3. Why `h11_ridge_distill` bundles three acceleration ideas at once

**As of OVERVIEW.md v6.3**, `h11_ridge_distill` bundles three
acceleration ideas at once (deliberately, in one screen -- see v6.3 for
the honest "conflates three variables" caveat): a rollout-horizon
curriculum (`AR_FRAMES` ramps 2 -> 8 rather than starting at the full
horizon), ridge-map distillation extended into the AR rollout itself
(not just the one-step prediction), and `DELTA_ANCHOR='ridge'` engaged
safely for the first time (via `forward()`'s new
`force_persistence_anchor` split -- the ridge anchor only ever applies
to the non-recursive teacher-forced loss, never inside the AR/rollout
feedback loop that caused `h10_ridge_residual`'s blowup).

The `exp-a`/`exp-b` table above IS the fully-qualified manual launch
command for each -- if `provision_and_run.sh` (or a screen session)
ever dies/hangs again (as one did earlier this session -- see
OVERVIEW.md v6.0 §30.1 for the wandb-duplicate-run incident this whole
`singleshot/` package exists to guard against), those are exactly the
commands to re-run by hand.

To pull either experiment's result back yourself (in case the pod's
about to be interrupted/terminated and you can't wait for automation):

```bash
rsync -avP -e "ssh -p <ssh port> -i /Users/kkreth/.ssh/id_ed25519" \
  "root@<pod ip>:/workspace/cgan/transformer_neurIPS/saved_models/r2_h11_ridge_distill_"* \
  /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/saved_models/sweep_arms_local/
# exp-b used --round 3 -- substitute r3_ for r2_ above to pull that one instead.
```

## RunPod agent setup

This session installed the RunPod plugin for Claude Code:

```
claude plugin marketplace add runpod/runpod-plugins-official
claude plugin install runpod@runpod
```

Two steps only you can do (OAuth browser flow + an interactive slash
command):

1. `/reload-plugins`
2. `/mcp` -> `runpod` -> "Sign in with Runpod"

Once signed in, RunPod's MCP tools can list/provision pods directly from
a Claude Code session. `runpodctl` itself is already installed on this
machine (`/Users/kkreth/.local/bin/runpodctl`, on `PATH` via
`/Users/kkreth/.zshrc`) -- that's what `provision_and_run.sh`/
`scp_env_files.sh`/`scp_data_files.sh` actually use; the MCP sign-in
above is a separate, optional path for driving pods interactively from
a Claude Code session instead of via these scripts.

## Other transfer options

`scp` (what these scripts use) needs no setup beyond SSH access and
works with any provider, not just RunPod -- that's why it's the default
here, split into `LANES` (default 10) concurrent streams per file. This
is now the **confirmed-fastest** option measured against this
infrastructure: 268.3 MB/s raw at 10 independent concurrent connections
(iperf3), close to the ~312 MB/s real link ceiling -- see
`bench_scp_variants.sh`. A few alternatives, for context:

1. **`croc`, self-hosted relay, tunneled through SSH -- TRIED AND
   DROPPED.** Was briefly the production default (2.4x the public relay
   at real 1GB-file scale, 93.1 vs 39.4 MB/s), but its own split-lanes
   phase (real file, N independent relay+tunnel pairs) measured far
   below plain scp at the same lane count: it pays for a PAKE handshake
   and its own encryption layer ON TOP of the already-encrypted SSH
   connection, for no security benefit on a link you already control via
   SSH, plus compression/hashing overhead scp doesn't have. `bootstrap_
   remote.sh` no longer installs it, and neither `scp_data_files.sh` nor
   `provision_and_run.sh` reference it. Kept for the record: the full
   investigation (seven real pod rentals, six real bugs, the RunPod
   networking wall that forced the SSH-tunnel approach) is in
   [`CROC_JOURNEY.md`](CROC_JOURNEY.md); the implementation is
   [`lib_croc.sh`](lib_croc.sh)/`bench_croc_variants.sh`, both historical
   now.
2. **`runpodctl send` / `runpodctl receive`** -- RunPod's own tool
   (also built on `croc`, same relay network): NAT traversal + relay,
   no SSH key/port juggling. Already installed locally at
   `/Users/kkreth/.local/bin/runpodctl`:
   `runpodctl send /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/data/train_80.h5`
   prints a one-time code, run `runpodctl receive <code>` on the pod to
   pull it (one call per file -- it doesn't take a directory). Prefer
   plain `croc` (above) when you want to send multiple files/a glob in
   one call.
3. **Cloud-storage relay** -- upload the data files once to S3/R2/GCS
   from this machine, then `curl`/`aws s3 cp`/`gsutil cp` them down on
   the pod. Turns a slow home-upload into a one-time cost, and multiple
   pods can reuse the same uploaded objects without re-uploading from
   home each time. Best option if you expect to provision more than one
   pod against the same data.
4. **`rsync -avP`** in place of `scp` -- not faster for a first
   transfer, but resumable if the connection drops partway (the
   convention already used elsewhere in this repo for pulling results
   back, see `OVERVIEW.md` §0). `scp_data_files.sh` re-sends the whole
   file on a retry rather than resuming; switch to `rsync` here if a
   flaky connection makes that costly. Still the right tool for small,
   incremental result pulls (single checkpoint/status file) where
   `croc`'s one-time-code handshake is more ceremony than the transfer
   is worth.
