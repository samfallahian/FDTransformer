# The croc self-hosted-relay journey

*How a "why is the upload slow" question turned into seven real pod
rentals, a RunPod networking rabbit hole, and a confirmed, production-
wired 2.4x speedup -- with a real ~312 MB/s target still open.*

> **TL;DR** — `croc`'s *default* public relay caps real transfers at
> ~17-41 MB/s regardless of local bandwidth, matching the exact
> bottleneck seen on `train_80.h5`/`val_80.h5` uploads. Self-hosting a
> relay ON the pod fixes it -- **confirmed 2.4x at real 1GB-file
> scale** (93.1 vs 39.4 MB/s), using croc's own default settings and
> nothing more -- but *only* by tunneling the relay's ports through
> the pod's SSH connection. RunPod's Secure Cloud pods do not expose
> arbitrary custom TCP ports the way this investigation initially (and
> repeatedly) assumed, and that took five separate real pod rentals to
> fully rule out. **A "scale to 100 ports" plan was also wrong** --
> croc's own source hard-caps real parallel connections at 8 per
> transfer, no matter how many ports are offered; confirmed by reading
> the source directly instead of spending a third pod rental finding
> out the hard way -- and going from croc's default of 4 real
> connections to its true max of 8 measured *slower*, not faster, so
> that's not the lever either. Now wired into production:
> `scp_data_files.sh`'s `TRANSFER_METHOD=croc` path and
> `provision_and_run.sh` both use the confirmed settings via
> [`lib_croc.sh`](lib_croc.sh). **The real target** (independently
> speed-tested by the operator) is ~312 MB/s, not the originally-
> stated 1GB/s, which exceeds their own connection and was never
> achievable. **Still open**: a genuinely-parallel-independent-tunnels
> approach is now built but not yet run against a real pod -- see
> "Built, NOT yet confirmed" below.

---

## Why this mattered

Real uploads of `train_80.h5`/`val_80.h5` (the actual training data,
~13GB combined once stored uncompressed) to a fresh RunPod pod were
stuck around 20-30 MB/s, regardless of the pod's advertised link speed
(some pods reported `maxUploadSpeedMbps` north of 2,000 Mbps --
~250+ MB/s theoretical). Something in the transfer path itself, not
the raw network, was the ceiling. `croc` looked like the fix (already
installed via `bootstrap_remote.sh`, parallelizes transfers over
multiple ports by design) -- but the *default* way this repo was using
it goes through `croc`'s public relay server, which turned out to be
exactly the bottleneck.

## Timeline

| # | What was tried | Real pod cost | Outcome |
|---|---|---|---|
| 1 | Benchmark `croc`'s default public relay vs. a self-hosted relay, single variant at a time | 1 pod | Public relay confirmed at ~18-21 MB/s. Self-hosted relay never completed -- timed out waiting for a sender that could never connect. |
| 2 | Root-cause the self-hosted-relay hang | 0 (local only) | Found: RunPod pods only expose ports listed at `pod create` time; the ports croc needed were never requested. |
| 3 | Add the missing ports to `pod create`, retry | 1 pod | **New** bug: `croc`'s local-discovery relay (ports 9009-9013) collided across 4 concurrently-launched `croc send` processes on the *local* machine -- unrelated to the pod at all. Also: `--transfers` passed in the wrong flag position (global vs. subcommand), crashing that variant outright. |
| 4 | Fix flag placement and local-port collisions | 1 pod | Self-hosted relay still unreachable from this machine -- `i/o timeout`, not "connection refused." |
| 5 | Add a debug dump of the pod's full `pod get` JSON, searching recursively for any port-mapping field | 1 pod | **Nothing found.** RunPod's CLI exposes a real external-port mapping for SSH (`ssh.port`) but nothing analogous for custom TCP ports -- confirmed empty via an exhaustive recursive search of the JSON. |
| 6 | Research RunPod's actual docs (after being asked to stop guessing empirically) | 0 (web research) | Found the mechanism: request `70000 + desired_port` to get a *symmetric* external==internal mapping, confirmable via a `$RUNPOD_TCP_PORT_<70000+port>` env var inside the pod. |
| 7 | Implement the 70000-offset request + a bounded retry loop checking the env var | 1 pod | Env var came back **completely empty**, every attempt, for the full 30s retry window. Not a propagation delay -- the mechanism itself didn't fire via `runpodctl`'s `--ports` flag. |
| 8 | Pivot: tunnel every relay port through the pod's SSH connection instead of asking RunPod to expose anything new | 1 pod | **Success.** `self-hosted-default`: 50.0 MB/s. Public-relay baseline in the same run: 25.0 MB/s. **2.0x measured, confirmed.** |
| 9 | Scale up: 100 relay ports, 1GB test file (vs. the original 16 ports / 100MB sanity check), same 4-variant CONCURRENT design | 1 pod | All 3 self-hosted variants **timed out** at ~31% (killed by the 90s cutoff), each crawling at ~1.7-1.8 MB/s -- combined ~5.4 MB/s, far worse than the single-variant 50 MB/s from attempt 8. Public-relay baseline unaffected (41.0 MB/s). |
| 10 | Root-cause attempt 9 | 0 (log analysis only) | All 3 self-hosted variants share ONE SSH tunnel -- one TCP connection to the pod. Fine at 100MB (finishes before contention matters); catastrophic at 1GB (sustained load exposes it). Fixed by running self-hosted variants SEQUENTIALLY against each other (each gets the tunnel to itself), while the public-relay baseline -- a fully separate path -- still runs concurrently alongside. |
| 11 | Before spending a *third* real pod rental chasing "100 workers," verify against croc's actual source whether `--transfers`/relay port count really controls real parallel connections per file | 0 (source read directly, no pod) | **The 100-port plan was wrong.** `croc` v11.5.2's own source (`activateRelayDataChannels`, confirmed against the exact tagged release, not just `main`) hard-caps real parallel data connections at **8 per transfer**, full stop -- `limit := min(len(c.Options.RelayPorts), 8)`, with a matching `index >= 8` guard. `RELAY_TRANSFERS=99`/`--transfers 100` were both silently capped to 8 real connections the whole time. Corrected to `RELAY_TRANSFERS=8` and a `self-hosted-transfers8` variant (was `self-hosted-transfers100`) to test the one axis of real, unmeasured upside: croc's send-side *default* is `--transfers 4` (which is what actually produced the confirmed 2.0x in attempt 8, not the relay's port count) -- going to the real max of 8 has never been directly measured. |
| 12 | Re-run the corrected scale test: `RELAY_TRANSFERS=8`, 1GB file, sequential self-hosted variants | 1 pod | **Success, confirmed at scale.** All 4 variants completed, every one size-verified byte-for-byte. `self-hosted-default` (croc's own defaults, no overrides) won at 93.1 MB/s -- **2.4x** the public-relay baseline (39.4 MB/s). `self-hosted-transfers8` came in *slower* (64.0 MB/s) than the default's 4 connections, confirming attempt 11's theory: more connections through one shared tunnel doesn't keep scaling. Wired into production (`scp_data_files.sh`, `provision_and_run.sh`) immediately after. |

Eight real attempts (six of them billed pod rentals), each one either
confirming a hypothesis or eliminating one. Every dead end is kept
here on purpose -- the next time something in this area looks broken,
this table (and the sections below) should mean nobody has to pay to
relearn any of it.

## The bugs, in the order they were actually found

These are worth reading even if you never touch RunPod again --
several are generic bash/SSH traps that will bite some *other* script
eventually.

### 1. `pkill -f` can kill the shell that invoked it

While trying to clean up a stray relay process between test runs,
`pkill -f 'croc relay' 2>/dev/null; <anything else>` in a single
non-interactive `ssh host "..."` command silently killed the **entire
SSH session** instead of just the target process. `ssh -v` showed
`exit-signal`, not a clean exit code -- the command just vanished with
no output.

Why: `pkill -f <pattern>` matches against every process's *full
command line*. When something follows the `pkill` invocation in the
same command string, the shell must fork a child to run `pkill` (it
can't `exec`-replace itself, since more comes after) -- and that
child's own pattern search sees the *parent shell's* command line too,
which literally contains the text of the `pkill` invocation itself.
`pkill`'s self-exclusion only covers the `pkill` process, never its
parent. Reproduced deterministically: **any** pattern, even a string
that had never existed anywhere else, killed the session, as long as
something followed the `pkill` call.

**Fix, used everywhere in `lib_croc.sh` now**: track the exact PID at
launch time (`echo $!` right after backgrounding), kill by that PID.
Numeric `kill <pid>` has no such hazard.

### 2. A backgrounded SSH command dies the instant its own shell exits

`ssh host "some_long_running_cmd &"` looked like it should leave
`some_long_running_cmd` running after the SSH command returns. It
doesn't -- the backgrounded job gets `SIGHUP`'d the moment that SSH
command's own remote shell exits. Confirmed directly: the process was
simply gone, with no log file even created, half a second later.

**Fix**: `setsid` inside a *nested* `bash -c '... &'`, which fully
detaches the child from the SSH session's controlling terminal.

### 3. `croc`'s global vs. subcommand flags are easy to mix up

`croc`'s usage is `[GLOBAL OPTIONS] [COMMAND] [COMMAND OPTIONS]
[file]`. `--relay`, `--yes`, `--no-compress` are global (must precede
`send`). `--no-local` and `--transfers` are `send`-subcommand-specific
(must follow it) -- passing them as global produces
`flag provided but not defined: -transfers`, which crashed a whole
benchmark variant the first time it happened. A comment written to
document this *also* made the identical mistake with `--no-local`
before being checked against the real `croc --help` / `croc send
--help` output directly. Moral: verify against the installed binary,
not against a remembered summary of it -- twice, in this case.

### 4. Concurrent local `croc send` processes collide on local-discovery ports

`croc send` -- unless given `--no-local` -- *also* tries to open a
local-network-discovery relay on ports 9009-9013, intended for
same-LAN transfers. Running 4 variants concurrently on one machine
meant 4 processes racing to bind those same 5 ports: `address already
in use`, and in one case a `panic:` after the transfer had actually
finished (harmless there, but not something to rely on).

**Fix**: `--no-local` on every send, always, since every variant here
talks to a real relay (public or self-hosted) and never needs local
discovery.

### 5. Send and receive need different relay addresses -- until they don't

The send side runs on the *local* machine and must reach the relay via
the pod's public IP. The receive side already runs *on* the pod (over
SSH) and must use `localhost`, not its own public IP. Collapsing these
into one shared `--relay` flag was a real bug: the receive side dialed
its own public address and got `i/o timeout` on every self-hosted
variant, every single time. Once the SSH tunnel (below) made *both*
sides reachable via `localhost`, this asymmetry disappeared entirely
-- one less thing to get wrong, not just a fixed version of the same
thing.

### 6. Concurrent self-hosted variants starve each other through one shared tunnel

Found only once the test file grew from 100MB to 1GB (attempt 9 in
the timeline above) -- at the smaller size every transfer finished in
a couple of seconds, long before sustained contention could show up.
All self-hosted variants necessarily share the *same* SSH tunnel (one
TCP connection to the pod, opened once). Running 3 of them
concurrently at 1GB meant 3 croc processes fighting over that one
connection: each crawled at ~1.7-1.8 MB/s, a combined ~5.4 MB/s --
worse than a tenth of the 50 MB/s a *single* self-hosted transfer
managed uncontended. The public-relay baseline, on a completely
separate network path, was unaffected (41.0 MB/s in the same run).

This isn't really a bug in the mechanism -- it's a bug in the
*benchmark's* comparison methodology once the file size made
contention visible. Production never sends 3 redundant copies of the
same file through 3 different croc settings at once; it sends at most
2 real files (`train_80.h5` + `val_80.h5`) concurrently. Fair
comparison of *settings* means giving each one the tunnel to itself.

**Fix**: self-hosted variants now run sequentially against each other
(one full transfer completes before the next starts), while the
public-relay baseline keeps running concurrently alongside them (it
never touches the tunnel, so it can't be starved by it). Not yet
re-confirmed against a real pod as of this writing.

## The wall: RunPod's custom TCP ports

This is the part that cost real money to fully rule out, so it gets
its own section.

RunPod's [Expose ports
docs](https://docs.runpod.io/pods/configuration/expose-ports) explain
that a requested port (e.g. `9019/tcp`) does **not** get exposed at
that literal external number -- it's remapped to something else
"to ensure security and allow multiple Pods to coexist on the same
infrastructure." SSH gets special treatment: `runpodctl`'s JSON
response includes a real `ssh.port` field showing the actual mapped
value (e.g. `22` -> `22158`). Nothing analogous exists for custom
ports -- confirmed by recursively searching a real pod's `pod get`
JSON for any key path containing `"port"`, at any depth. Nothing came
back except the literal echoed request and `ssh.port`.

The docs describe a workaround: request `70000 + desired_port` (e.g.
`79019` to ask for port `9019`) to get a **symmetric** mapping
(external == internal), confirmable via an environment variable
*inside* the pod named `$RUNPOD_TCP_PORT_<70000+port>`.

This was implemented, including a 30-second retry loop polling that
env var every 5 seconds (in case it was simply a propagation delay,
not a hard failure) -- and it came back **completely empty**, every
single attempt, on a real pod. Not "wrong value." Empty. The `--ports`
CLI flag on `runpodctl pod create` appears not to route through
whatever backend mechanism actually injects that env var -- possibly a
web-dashboard-only feature, possibly something else entirely. This
was not resolved; it was **ruled out** as a path worth pursuing
further, after research (docs + a targeted web search for the literal
env var name, to rule out a hallucinated detail from an earlier
summarized fetch) confirmed the implementation matched the documented
mechanism exactly.

## The fix: tunnel through SSH instead

The pod's SSH port has worked reliably across every single attempt in
this whole investigation. So: stop asking RunPod to expose anything
new, and tunnel the relay's ports through the connection that already
works.

```
 ┌─────────────────┐                                    ┌──────────────────┐
 │   local machine  │                                    │    RunPod pod    │
 │                  │   ssh -L 9019:localhost:9019  ...  │                  │
 │  croc send  ─────┼──────── existing SSH port ─────────┼──► croc relay     │
 │  (localhost:9019)│         (the ONLY port RunPod      │  (localhost:9019)│
 │                  │          has ever reliably mapped) │                  │
 │                  │                                    │  croc receive ───┤
 │                  │                                    │  (localhost:9019)│
 └─────────────────┘                                    └──────────────────┘
```

One `-L port:localhost:port` forward per relay port (base through
base+`RELAY_TRANSFERS`), all riding on the same `ssh` process,
`-o ExitOnForwardFailure=yes` so a single bad forward fails the whole
tunnel loudly instead of silently running short. Both the send side
(this machine, via the tunnel's local end) and the receive side (the
pod, talking to its own relay) now use the *identical*
`--relay localhost:<port>` flag -- bug #5 above simply can't recur,
because there's no longer a second address to get wrong.

**Result, first real run**: `self-hosted-default` at 50.0 MB/s vs.
`public-relay-baseline` at 25.0 MB/s in the *same* run, same pod, same
link. **2.0x**, measured, not estimated.

## The second wall: croc hard-caps parallelism at 8, not however many ports you offer

After the tunnel fix worked, the plan was to scale up: offer 100 relay
ports instead of 17, on the theory that "more ports = more parallel
workers = faster." Before spending a third real pod rental testing
that at 1GB (on top of the two already spent finding bug #6 above),
this was checked against `croc`'s actual source first --
specifically the exact tagged release matching the installed local
version (`v11.5.2`, confirmed via `croc --version`, not just
whatever's newest on `main`):

```go
// src/croc/croc.go
func (c *Client) activateRelayDataChannels(attempt *transferAttemptState) (err error) {
	limit := min(len(c.Options.RelayPorts), 8)
	...
	if index < 0 || index >= len(c.Options.RelayPorts) || index >= 8 {
```

`croc` will never open more than **8** real parallel data connections
for a single transfer, no matter how many ports the relay offers or
`--transfers` requests on the send side. Offering 100 ports and asking
for `--transfers 100` were both silently capped to 8 real connections
the whole time -- the "100 workers" premise was never true for this
tool, at this version, at all.

This also reframes the confirmed 2.0x result itself: `self-hosted-
default` never overrode `--transfers`, so it ran with croc's own
send-side *default* of **4** -- not even reaching the real cap of 8.
The actual, still-unmeasured question is whether going from 4 real
connections to croc's true maximum of 8 buys anything further --
`RELAY_TRANSFERS` is now `8` (9 ports total, matching the real limit
with a 1-port margin) and a `self-hosted-transfers8` variant (renamed
from the wrong `self-hosted-transfers100`) tests exactly that axis.

One AI-summarized web fetch along the way *also* claimed a different,
unrelated "cap at 8" detail that turned out to need this same
treatment -- checked by pulling the raw source with `curl` and
`grep`-ing it directly, not by trusting another summary of it. Given
this session's earlier `RUNPOD_TCP_PORT` env-var-name mixup (§ "The
wall" above) came from the exact same failure mode -- trusting a
plausible-sounding AI paraphrase of documentation instead of the
primary source -- this is now the second time that distinction
mattered enough to change what got built. Worth remembering the third
time this comes up too.

## Where this logic lives now

Every function above -- installing croc remotely, enabling its
"classic mode" for scripted receives, starting the relay, tunneling
its ports, sending, parsing the one-time code out of croc's output,
receiving with a size-verified result -- is in
[`lib_croc.sh`](lib_croc.sh), documented in the file's own header as
"the reusable class." Both `bench_croc_variants.sh` (the benchmark)
and `scp_data_files.sh` (the real `train_80.h5`/`val_80.h5` upload
path this was standing in for all along) source it and call its
functions instead of reimplementing any of this inline -- one tested
code path, not a fresh reimplementation with a fresh chance at bugs
1-6 above. `provision_and_run.sh`'s `pod create` call was also
simplified back to requesting only `22/tcp`, since the tunnel approach
needs nothing else.

## Confirmed at real scale: 2.4x, using croc's own defaults

The corrected scale test (`RELAY_TRANSFERS=8`, 1GB file, sequential
self-hosted variants) ran cleanly end to end -- all 4 variants
completed, every one size-verified byte-for-byte against the source:

  public-relay-baseline    39.4 MB/s   (1.0x)
  self-hosted-default      93.1 MB/s   (2.4x)  <- fastest
  self-hosted-transfers8   64.0 MB/s   (1.6x)
  self-hosted-no-compress  53.9 MB/s   (1.4x)

`self-hosted-default` -- self-hosted relay through the SSH tunnel,
with croc's own settings left alone (no `--transfers` override, no
`--no-compress`) -- won outright. `--transfers 8` was *slower* than
croc's own default of 4, confirming the "second wall" finding above
from a second angle: adding more concurrent streams through the one
shared tunnel has diminishing (here, negative) returns, it doesn't
keep scaling. Caveat worth naming honestly: these are single
sequential trials over a live network, not repeated/averaged, so some
of the exact gap between the three self-hosted numbers likely includes
real-world variance -- but the directional finding (don't add more
connections) is corroborated by the independent bug-#6 evidence, so it
can be trusted.

This result is now wired into production: `scp_data_files.sh`'s
`TRANSFER_METHOD=croc` path uses `lib_croc.sh`'s tunnel approach and
croc's own default settings, sending the two real files (`train_80.h5`
+ `val_80.h5`) SEQUENTIALLY rather than concurrently, for the same
tunnel-sharing reason. `provision_and_run.sh`'s `pod create` call no
longer requests any custom port range either -- SSH is the only port
this whole mechanism has ever actually needed.

## The real target: ~312 MB/s, not 1GB/s

1GB/s (1024 MB/s) was the first stated goal, but it exceeds the
operator's own confirmed real connection speed (2.5 Gb/s symmetric,
independently speed-tested, not just estimated) -- ~312 MB/s
theoretical, and real-world overhead always eats into that. No amount
of software-side optimization moves data out of this machine faster
than the physical link allows. The corrected, achievable target is
that ~312 MB/s ceiling. 93.1 MB/s is currently ~30% of it -- real
room to grow, just not to 1GB/s.

## Built, NOT yet confirmed: genuinely independent parallel tunnels

More connections through ONE shared tunnel has now measured *worse*,
twice (`self-hosted-transfers8` above, and bug #6's concurrent-
variants collapse). The next real lever is genuinely PARALLEL,
INDEPENDENT network paths: split the file into N pieces, send each
through its OWN dedicated relay+tunnel pair (own SSH process, own
local AND remote port range -- tunnels can't share a local port),
running all N truly concurrently, then reassemble and verify on the
pod.

This is now built -- `lib_croc.sh`'s `croc_relay_start_lane`/
`croc_tunnel_start_lane`/`croc_stop_lane`, orchestrated as a new
`SPLIT_LANES`-controlled phase at the end of `bench_croc_variants.sh`
(default 4 lanes, set `SPLIT_LANES=0` to skip it). Verified locally as
thoroughly as possible before ever touching a pod: split-then-`cat`
reassembly is byte-identical (tested directly), and the full
orchestration logic (lane startup, concurrent send/receive, success
path, a relay-startup-failure path, and a size-mismatch path) was
dry-run against mocked `ssh`/`croc` functions -- all three scenarios
behaved correctly, including a real bug caught before it ever touched
a pod (a `$!` captured after an `if`/`else` where one branch didn't
background anything, which would have grabbed a stale or unset pid
for that lane).

**First real-pod attempt found a 7th real bug: a "ready" tunnel that
wasn't.** 3 of
4 lanes succeeded; the 4th sat at "waiting for sender..." until it hit
`TRANSFER_TIMEOUT`. Telling asymmetry: that lane's *receiver* (on the
pod, talking to its own local relay) connected fine and was happily
waiting -- the *sender* (this machine, going through that lane's SSH
tunnel) never got through at all. Root cause: `croc_tunnel_start_lane`
only checked "is the ssh process still alive" (a 1s sleep then
`kill -0`), unlike the proven single-lane `croc_tunnel_start`, which
ALSO probes actual port reachability via `/dev/tcp`. With 4 tunnels'
SSH connections negotiating against the pod at close to the same
time, "alive" and "actually accepting connections" turned out not to
be the same moment for one of them. Fixed by giving
`croc_tunnel_start_lane` the same reachability probe, with a short
retry (3 attempts, 2s apart) rather than a single hard check, since
under real contention "not ready yet" and "will never be ready" need
to be told apart, not treated the same. Verified the retry logic
itself against a real (if synthetic) delayed-ready listener locally
before trusting it -- confirmed it correctly waits out a ~3s delay
instead of failing prematurely. Not yet re-confirmed against a real
pod.

Open questions this next real test needs to answer:
- Does aggregate throughput across N independent lanes actually beat
  a single tunnel, or does something else (the relay process itself,
  the pod's own network stack, this machine's per-process overhead)
  become the new bottleneck first?
- What's the right N? Started at 4 (matching croc's own confirmed-best
  per-transfer default) -- worth trying higher once 4 is confirmed to
  help at all.
- **Why the pod-mapping mechanism didn't fire via `runpodctl`** was
  ruled out as worth pursuing further for croc specifically (the
  tunnel sidesteps it entirely), not actually root-caused. If this
  matters again later for something that can't be tunneled through
  SSH, that's still an open question against RunPod's actual backend,
  not just its docs.
