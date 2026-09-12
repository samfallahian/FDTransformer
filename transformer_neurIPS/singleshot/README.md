# `singleshot` -- one-shot deploy package for a rented GPU box

Everything needed to go from "empty rented pod" to "training running,
telemetry flowing to wandb": code, data, the recovered checkpoint, the
AE decoder, and the dependency/bootstrap scripts. No tarball -- files
transfer individually, in two phases, specifically so bootstrapping the
venv and transferring the large data files can happen **at the same
time** over two separate SSH connections, instead of one blocking the
other.

## What's in here

| File | Role |
|---|---|
| `provision_and_run.sh` | Run **locally**. The full lifecycle in one command: create a pod via `runpodctl`, wait for SSH, send env files, bootstrap the pod WHILE the data files transfer in the background, run a time-boxed training job, pull the new artifacts back, terminate the pod. See "Fully automated lifecycle" below. |
| `scp_env_files.sh` | Run **locally**. Phase 1: sends code, the recovered checkpoint, the ridge map, the AE decoder, and this directory's own `requirements.txt`/`bootstrap_remote.sh` -- everything except the large data files. Small, finishes in seconds. |
| `scp_data_files.sh` | Run **locally**. Phase 2: sends `train_80.h5` (7.9G) and `val_80.h5` (3.4G) as two concurrent `scp` processes. Independent of phase 1 -- run it any time after, or in a separate terminal at the same time as, phase 1. |
| `bootstrap_remote.sh` | Run **on the pod**. Creates a venv, installs pinned deps + the correct CUDA **stable** (not nightly) torch build, then stops and prints the `wandb login` step for you to run yourself -- never called non-interactively past that point unless `WANDB_API_KEY` is set. Only needs phase 1's files -- can run while phase 2 is still transferring. |
| `requirements.txt` | Copy of `/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/requirements_sweep.txt` (the verified-minimal dependency set), kept self-contained here. |

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
both ends before moving to the next. Finishes in seconds.

**As soon as phase 1 is done, open a second terminal and start
bootstrapping** (no need to wait for phase 2):

```bash
ssh -p <ssh port> -i ~/.ssh/id_ed25519 root@<pod ip>
bash /workspace/cgan/transformer_neurIPS/singleshot/bootstrap_remote.sh
```

**Phase 2 (slow) -- the training/validation data**, in your first
terminal (or a third one -- it doesn't depend on phase 1 or bootstrap at
all, only on the pod having somewhere to put the files):

```bash
POD_HOST=<pod ip> POD_PORT=<ssh port> bash /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/scp_data_files.sh
```

Sends `train_80.h5` and `val_80.h5` as two concurrent `scp` processes
(not sequential) and verifies both sizes match on completion. By the
time this finishes, bootstrap in your other terminal has likely already
finished too -- you're ready to smoke-test/launch as soon as both are done.

Optional overrides on both scripts: `POD_USER` (default `root`),
`SSH_KEY` (default `/Users/kkreth/.ssh/id_ed25519`), `REMOTE_ROOT`
(default `/workspace` -- files land at `$REMOTE_ROOT/cgan/...`).

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
finishes, you'll see `bootstrap_remote.sh`'s standalone-usage message
("Two manual steps left: 1) wandb login, 2) smoke-test/launch") --
**ignore it and don't interrupt anything.** That text always prints
(it's shared with the manual workflow below), but in this automated
path the pipeline does not actually stop there.

### What you do vs. what the script does

You do exactly one thing before running it (the API key, above), and
run exactly one command. Everything else happens automatically:

1. create pod, wait for SSH
2. send env files (phase 1)
3. **in parallel**: send data files (phase 2, background) WHILE
   bootstrapping the pod (venv, deps, stable torch) over a second
   connection
4. wait for the data transfer to finish, then run training, capped at
   30 min AND 2500 steps
5. **rsync the new checkpoint + logs back to this machine** -- automatic,
   not a step you do afterward
6. **terminate the pod** (`runpodctl pod delete`) -- also automatic,
   right after step 5, unless `KEEP_POD=1`

**This is not a background/fire-and-forget job.** Your machine has to
stay connected and the script has to keep running for the whole
duration. The parallelism in step 3 shortens the wall-clock cost
somewhat (bootstrap and the ~11GB data transfer overlap instead of
stacking), but total wall time is still meaningfully longer than 30
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

Every knob is an env var -- different arm, different budget, different
GPU, reuse an existing pod, or keep the pod alive after the run for
debugging. See the script's own header comment for the full list
(`ARM`, `MAX_HOURS`, `MAX_STEPS`, `GPU_ID`, `TEMPLATE_ID`, `IMAGE`,
`POD_ID`, `KEEP_POD`, `WANDB_API_KEY`, ...). Pod image defaults to the
`runpod-torch-v280` template (verified via `runpodctl template search
pytorch` to actually exist -- `bootstrap_remote.sh` installs its own
venv + torch on top regardless, so any real, working CUDA+Python3
image/template is fine). Set `IMAGE=<docker image>` instead to use a
specific image directly rather than a template -- verify it exists
first with `runpodctl template search <name>` or `runpodctl template
list --type official`.

**wandb**: unattended by design, so this can't do an interactive
`wandb login`. Without `WANDB_API_KEY` set, the run passes `--no-wandb`.
Set `WANDB_API_KEY=<key>` (same "never in chat" rule as the RunPod key)
to get telemetry for this run via wandb's own non-interactive login path.

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

### 2. Send the files (two phases, can overlap)

Covered above -- `scp_env_files.sh` (fast), then open a second terminal
to bootstrap (step 3) while `scp_data_files.sh` (slow) runs.

### 3. SSH in and bootstrap

```bash
ssh -p <ssh port> -i /Users/kkreth/.ssh/id_ed25519 root@<pod ip>
bash /workspace/cgan/transformer_neurIPS/singleshot/bootstrap_remote.sh
```

This creates a venv, installs `numpy`/`h5py`/`wandb` (pinned) and the
correct stable-CUDA `torch` build (auto-detected from `nvidia-smi`),
verifies the interpreter can see the GPU, and then **stops** -- it will
not log in to wandb for you. Only needs phase 1's files, so this can run
while phase 2 (the data) is still transferring.

### 4. Log in to wandb yourself

```bash
source /workspace/cgan/.venv/bin/activate
wandb login
```

This is interactive on purpose (browser auth / paste an API key from
https://wandb.ai/authorize) -- credentials never touch this package or
any script in it.

### 5. Smoke-test, then launch

Once both phases have finished (confirm phase 2's script printed its
size-verification "OK" lines):

```bash
cd /workspace/cgan/transformer_neurIPS
python /workspace/cgan/transformer_neurIPS/train_production_transformer_deep_dive.py --arm s8_h9_moreseqs_scaled --round 2 --smoke-test
bash /workspace/cgan/transformer_neurIPS/run_sweep_h300_production_8h.sh
```

`--smoke-test` builds the model and runs the causality gate on synthetic
data in seconds, with no data/checkpoint/wandb I/O at all -- confirms
the arm resolves and the pinned shapes are coherent before committing
to the real 8-hour run.

### 5a. Running `h11_ridge_distill` manually -- keep this at your fingertips

**If `provision_and_run.sh` ever dies/hangs again** (as it did earlier
this session -- see OVERVIEW.md v6.0 §30.1 for the wandb-duplicate-run
incident this whole `singleshot/` package exists to guard against),
this is the exact fully-qualified command to launch it by hand, once
you're SSH'd in and steps 3/4 above are done:

```bash
cd /workspace/cgan/transformer_neurIPS
source /workspace/cgan/.venv/bin/activate
python /workspace/cgan/transformer_neurIPS/train_production_transformer_deep_dive.py \
  --arm h11_ridge_distill --round 2 \
  --max-steps 2500 --max-hours 0.5 \
  --val-every 500 --no-wandb
```

(Drop `--no-wandb` and run `wandb login` first per step 4 if you want
telemetry for this run.) This is the literal command
`provision_and_run.sh` runs on your behalf -- same arm, same 30-minute/
2500-step cap, same everything -- just typed by hand instead of
generated inside a heredoc. `provision_and_run.sh` itself carries the
identical command in a comment right above where it builds that heredoc,
so the script and this README never drift apart on what "running h11
manually" actually means.

**As of OVERVIEW.md v6.3**, `h11_ridge_distill` bundles three
acceleration ideas at once (deliberately, in one screen -- see v6.3 for
the honest "conflates three variables" caveat): a rollout-horizon
curriculum (`AR_FRAMES` ramps 2 -> 8 rather than starting at the full
horizon), ridge-map distillation extended into the AR rollout itself
(not just the one-step prediction), and `DELTA_ANCHOR='ridge'` engaged
safely for the first time (via `forward()`'s new
`force_persistence_anchor` split -- the ridge anchor only ever applies
to the non-recursive teacher-forced loss, never inside the AR/rollout
feedback loop that caused `h10_ridge_residual`'s blowup). The command
above is unchanged; only what it now actually does changed.

To pull the result back yourself afterward (in case the pod's about to
be interrupted/terminated and you can't wait for automation):

```bash
rsync -avP -e "ssh -p <ssh port> -i /Users/kkreth/.ssh/id_ed25519" \
  "root@<pod ip>:/workspace/cgan/transformer_neurIPS/saved_models/r2_h11_ridge_distill_"* \
  /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/saved_models/sweep_arms_local/
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
here. A few alternatives, roughly fastest to slowest for the ~11.3GB
`train_80.h5`+`val_80.h5` pair specifically:

1. **`runpodctl send` / `runpodctl receive`** -- RunPod's own tool
   (built on `croc`): NAT traversal + relay, no SSH key/port juggling,
   often meaningfully faster than a raw `scp` stream for a large single
   transfer. Already installed at `/Users/kkreth/.local/bin/runpodctl`:
   `runpodctl send /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/data/train_80.h5`
   prints a one-time code, run `runpodctl receive <code>` on the pod to
   pull it (one call per file -- it doesn't take a directory).
2. **Cloud-storage relay** -- upload the data files once to S3/R2/GCS
   from this machine, then `curl`/`aws s3 cp`/`gsutil cp` them down on
   the pod. Turns a slow home-upload into a one-time cost, and multiple
   pods can reuse the same uploaded objects without re-uploading from
   home each time. Best option if you expect to provision more than one
   pod against the same data.
3. **`rsync -avP`** in place of `scp` -- not faster for a first
   transfer, but resumable if the connection drops partway (the
   convention already used elsewhere in this repo for pulling results
   back, see `OVERVIEW.md` §0). `scp_data_files.sh` re-sends the whole
   file on a retry rather than resuming; switch to `rsync` here if a
   flaky connection makes that costly.
