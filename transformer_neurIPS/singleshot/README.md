# `singleshot` -- one-shot deploy package for a rented GPU box

Everything needed to go from "empty rented pod" to "training running,
telemetry flowing to wandb" in one pass: code, data, the recovered
checkpoint, the AE decoder, and the dependency/bootstrap scripts. No
manual step-by-step rsync of individual files.

## What's in here

| File | Role |
|---|---|
| `provision_and_run.sh` | Run **locally**. The full lifecycle in one command: create a pod via `runpodctl`, wait for SSH, build + scp the tarball, run a time-boxed training job, pull the new artifacts back, terminate the pod. See "Fully automated lifecycle" below. |
| `build_tarball.sh` | Run **locally**. Builds the single `cgan_singleshot_payload.tar.gz` from the current repo state. |
| `scp_to_pod.sh` | Run **locally**. Copies that tarball to the pod via `scp`, confirms the byte count matches on both ends, then extracts it there via one `ssh` command. |
| `bootstrap_remote.sh` | Run **on the pod**. Creates a venv, installs pinned deps + the correct CUDA **stable** (not nightly) torch build, then stops and prints the `wandb login` step for you to run yourself -- never called non-interactively past that point unless `WANDB_API_KEY` is set. |
| `requirements.txt` | Copy of `/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/requirements_sweep.txt` (the verified-minimal dependency set), kept self-contained here. |
| `cgan_singleshot_payload.tar.gz` | The built payload (~11GB, gitignored) at `/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/cgan_singleshot_payload.tar.gz`. Not checked in; rebuild with `build_tarball.sh` whenever the code/data/checkpoint changes. |

## Building and sending the tarball

Two explicit, separately-confirmable steps -- build once, inspect it,
then send it (rebuilding is only needed again if the payload changes):

```bash
bash /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/build_tarball.sh
```

Checks every required file exists first (same `[OK]`/`[MISSING]` table
every `run_sweep_h300_*.sh` launcher in this repo already uses), then
builds `/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/cgan_singleshot_payload.tar.gz`.
Compression is `gzip -1` (fastest level) rather than the default -6: the
dominant payload (`train_80.h5` 7.9G + `val_80.h5` 3.4G) is already
internally gzip-compressed (`prepare_data.py`'s `compression='gzip'`),
so a higher compression level burns a lot more CPU for close to zero
extra size reduction -- the resulting file lands around 11GB either
way. You can inspect it without extracting anything:

```bash
tar -tzf /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/cgan_singleshot_payload.tar.gz
```

Then send it:

```bash
POD_HOST=<pod ip> POD_PORT=<ssh port> bash /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/scp_to_pod.sh
```

This does exactly two things, each printed and confirmable on its own:

1. `scp`s the tarball to `$REMOTE_ROOT/cgan_singleshot_payload.tar.gz`
   (default `REMOTE_ROOT=/workspace`), then **checks the byte count
   matches on both ends** before doing anything else -- if a transfer
   got truncated, this catches it instead of silently extracting a
   corrupt archive.
2. Runs one `ssh` command to `tar -xzf` it at `$REMOTE_ROOT`, landing at
   `$REMOTE_ROOT/cgan/...`, then deletes the remote copy of the tarball
   (set `KEEP_REMOTE_TARBALL=1` to leave it).

Optional overrides on either script: `POD_USER` (default `root`),
`SSH_KEY` (default `/Users/kkreth/.ssh/id_ed25519`), `REMOTE_ROOT`
(default `/workspace`), `TARBALL`/`OUT_FILE` (default
`/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/cgan_singleshot_payload.tar.gz`).

## Fully automated lifecycle: create pod -> run -> pull results -> terminate

`provision_and_run.sh` drives the whole thing via `runpodctl` -- no
manual pod creation, no copy-pasting an SSH address, no leaving a pod
running after you forget about it. Internally it just calls
`/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/build_tarball.sh`
(skipped if the tarball already exists) then
`/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/scp_to_pod.sh`
for the transfer step above.

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
answer, builds/scp's the payload, runs `h11_ridge_distill` (the one arm
from this session verified only on MPS so far -- see OVERVIEW.md v6.1
§31.5, "needs a real CUDA shallow screen") capped at **30 minutes AND
2500 optimizer steps** (both caps matter -- an oversized step budget
relative to the real 30-minute throughput breaks the LR warmup
schedule, the exact bug in OVERVIEW.md v5.0 §28.6), pulls back only the
new checkpoint + log files (not a second copy of the training data),
then terminates the pod.

### What you do vs. what the script does

You do exactly one thing before running it (the API key, above), and
run exactly one command. Everything else -- including the two things
that are easy to assume are separate manual steps -- happens
automatically, in this order, inside that single blocking run:

1. create pod, wait for SSH
2. build + scp the tarball, extract it remotely
3. bootstrap the pod (venv, deps, stable torch)
4. run training, capped at 30 min AND 2500 steps
5. **rsync the new checkpoint + logs back to this machine** -- automatic,
   not a step you do afterward
6. **terminate the pod** (`runpodctl pod delete`) -- also automatic,
   right after step 5, unless `KEEP_POD=1`

**This is not a background/fire-and-forget job.** Your machine has to
stay connected and the script has to keep running for the whole
duration -- pod boot + the ~11GB scp upload + remote bootstrap all
happen *before* the 30-minute training clock even starts, so total
wall time is meaningfully longer than 30 minutes (add however long an
11GB upload takes on your connection, plus a few minutes of pod
boot/dependency install).

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

**Honesty note on the RunPod JSON parsing**: this session has no RunPod
API key to test against, so `provision_and_run.sh`'s parsing of
`gpu list` / `pod create` / `ssh info` JSON output tries several
plausible field names defensively and ALWAYS prints the raw output too
-- but the exact schema hasn't been verified against a live authenticated
call. If a parse step comes up empty, the script stops and shows you the
raw JSON to fix the field name or fall back to the manual steps below,
rather than guessing further or silently doing the wrong thing.

## Manual step by step (what `provision_and_run.sh` automates)

Useful if you provision a pod yourself (RunPod's website, or once you've
completed the MCP sign-in below) and just want the deploy/run/pull-back
part automated, or if `provision_and_run.sh`'s JSON parsing needs
adjusting for your account (see the honesty note above).

### 1. Provision a pod and get its SSH details

Via RunPod's website, `runpodctl pod create` directly, or (once signed
in) the MCP tools from "RunPod agent setup" below.

### 2. Build and send the tarball

Covered above -- `/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/build_tarball.sh`
then `/Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/scp_to_pod.sh`.
`scp_to_pod.sh` also extracts it (one `ssh` command at the end), so by
the time you SSH in for step 3, the files are already there.

### 3. SSH in and bootstrap

```bash
ssh -p <ssh port> -i /Users/kkreth/.ssh/id_ed25519 root@<pod ip>
bash /workspace/cgan/transformer_neurIPS/singleshot/bootstrap_remote.sh
```

This creates a venv, installs `numpy`/`h5py`/`wandb` (pinned) and the
correct stable-CUDA `torch` build (auto-detected from `nvidia-smi`),
verifies the interpreter can see the GPU, and then **stops** -- it will
not log in to wandb for you.

### 4. Log in to wandb yourself

```bash
source /workspace/cgan/.venv/bin/activate
wandb login
```

This is interactive on purpose (browser auth / paste an API key from
https://wandb.ai/authorize) -- credentials never touch this package or
any script in it.

### 5. Smoke-test, then launch

```bash
cd /workspace/cgan/transformer_neurIPS
python /workspace/cgan/transformer_neurIPS/train_production_transformer_deep_dive.py --arm s8_h9_moreseqs_scaled --round 2 --smoke-test
bash /workspace/cgan/transformer_neurIPS/run_sweep_h300_production_8h.sh
```

`--smoke-test` builds the model and runs the causality gate on synthetic
data in seconds, with no data/checkpoint/wandb I/O at all -- confirms
the arm resolves and the pinned shapes are coherent before committing
to the real 8-hour run.

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
`scp_to_pod.sh` actually use; the MCP sign-in above is a separate,
optional path for driving pods interactively from a Claude Code session
instead of via these scripts.

## Faster than scp

`scp_to_pod.sh` is the default here (simple, resumable-by-rerun, easy to
confirm at each step) but it is not the fastest option for an ~11GB
transfer. Roughly fastest to slowest:

1. **`runpodctl send` / `runpodctl receive`** -- RunPod's own tool
   (built on `croc`): NAT traversal + relay, no SSH key/port juggling,
   often meaningfully faster than a raw `scp` stream for a large single
   transfer. Already installed at `/Users/kkreth/.local/bin/runpodctl`:
   `runpodctl send /Users/kkreth/PycharmProjects/cgan/transformer_neurIPS/singleshot/cgan_singleshot_payload.tar.gz`
   prints a one-time code, run `runpodctl receive <code>` on the pod to
   pull it.
2. **Cloud-storage relay** -- upload the tarball once to S3/R2/GCS from
   this machine, then `curl`/`aws s3 cp`/`gsutil cp` it down on the pod.
   Turns a slow home-upload into a one-time cost, and multiple pods can
   reuse the same uploaded object without re-uploading from home each
   time. Best option if you expect to provision more than one pod from
   this payload.
3. **Parallel/split transfer** -- `split` the tarball into a few chunks
   and run several `scp`/`rsync` streams concurrently (`xargs -P N`).
   Helps when a single stream is overhead/latency-bound rather than
   raw-bandwidth-bound; diminishing returns past a handful of streams on
   a typical home connection.
4. **`rsync -avP`** in place of `scp` -- not faster for a first transfer,
   but resumable if the connection drops partway (the convention already
   used elsewhere in this repo for pulling results back, see
   `OVERVIEW.md` §0).
5. **A raw `tar | ssh | tar` pipe with no local archive at all** -- what
   this package used before this machine had enough free disk space to
   comfortably hold an 11GB tarball; skips the local write+read entirely,
   at the cost of being harder to inspect/resume/reuse than a real file.
