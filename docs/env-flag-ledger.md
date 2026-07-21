# TORCHLETTE_* env-flag ledger

The canonical registry of every `TORCHLETTE_*` environment flag. **Every new
flag is born into this table** with a class and a sunset; a flag that outlives
its campaign is debt (CLAUDE.md "Complexity budget"). Audited 2026-07-21
("diamond hardening" flag-sunset sweep).

## Standing policy (restated)

- **Born with a sunset.** Opt-outs of a shipped default follow
  soak → default → *the opt-out dies*. The opt-out arm is deleted once its soak
  completes and nothing load-bearing still depends on it.
- **An opt-out arm that is a correctness differential is NOT expired**, whatever
  its age. `tools/parity-fullstack-tl.ts` needs `TORCHLETTE_COMPILED_PLAN=0` as
  its lowered control arm; that flag stays until the differential is retired.
- **A debug flag whose read hook was deleted is dead weight → delete.** All env
  reads in `src/` go through the `ENV` accessor in `src/core/env.ts` — a flag
  that appears in `src/` only in comments (never `ENV.TORCHLETTE_*`) is dead.
- **`envFlags` metric** (`tools/weight-norm.sh`): `grep -rhoE
  'TORCHLETTE_[A-Z_]+' src | sort -u | wc -l` — distinct names in `src/`,
  comments included. Scrubbing a dead flag's src comment drops the count.

## Audit result summary (2026-07-21)

| Class | Count | Verdict |
|---|---:|---|
| Opt-out of a shipped default | 17 | KEEP — none has a completed, unblocked soak (`PACK_OPTIM` born 2026-07-21, chain-packing P2) |
| Debug / observability | ~40 | KEEP — every one verified to have a live `ENV` read hook |
| Config / infra | ~13 | KEEP |
| Dead references DELETED this sweep | 7 | removed (see below) |
| Flagged, left for owner | 3 | UNROLLED_K, STREAM_GENERATE, LIVEDIFF |

**Headline finding: no live `src` opt-out is expired.** Every default opt-out is
still load-bearing (a differential control arm, a very young soak, or the
reference path a gate compares against). The sweep's deletions are all *dead
references to already-removed flags*. `envFlags` 66 → 64.

**The one flagged expiry — `TORCHLETTE_STRICT_LIFETIME=0` — is STOPPED, opt-out
KEPT.** See the diagnosis section at the bottom.

---

## Opt-outs of a shipped default (KEEP)

| Flag | Default / opt-out | src read | Why it stays |
|---|---|---|---|
| `TORCHLETTE_COMPILED_PLAN` | on; `=0` = lowered reference | `executor.ts:711,2051` | Lowered control arm of `parity-fullstack-tl.ts` + `compiled-plan-parity.spec.ts` gate. Explicitly NOT in sunset scope (docs/stage4). |
| `TORCHLETTE_STEP_TAPE` | off (opt-in); `record`/`1` | `step-tape.ts:68,70` | Campaign flag; soak NOT declared complete. Load-bearing ON/OFF differential (`t-uk-economics.ts`); shipped `=1` decode config. |
| `TORCHLETTE_WHOLE_STEP` | on since 2026-07-19; `=0` = eager | `step-tape.ts:103` | Opt-out IS the eager semantic reference the differential gates compare against. Soak "days old"; sunset deferred (re-ruled PERMANENT POLICY, docs/step-function-compiler-design). |
| `TORCHLETTE_STRICT_LIFETIME` | throw; `=0` = warn (soak) | `op-dispatch.ts:57`, `schedule/canonical.ts:579` | **STOP** — soak due ~2026-08 but blocked by a whole-step remat GC race (diagnosis below). Also coupled to the young whole-step soak. |
| `TORCHLETTE_ARENA_LIVENESS` | bounded arena; `=0` = legacy unbudgeted | `buffer-arena.ts:324` | `=0` used by 124M A/B tools; legacy-arena escape hatch still referenced. |
| `TORCHLETTE_MEMORY_PLANNER` | on; `=0` disables compiled replay | `buffer-arena.ts:348` | `=0` = lowered-wholesale reference (dynamic replay leaks). |
| `TORCHLETTE_COMPILED_PLANNED` | on; `=0` = lowered exec | `buffer-arena.ts:347` | Planned-buffer opt-out; used by `run-1peer-124M-ab.sh`. |
| `TORCHLETTE_LIVENESS_RELEASE` | on; `=0` disables early release | `executor.ts:2491,3690` | Buffer-release kill switch; used by `diloco-coordinator.ts`. |
| `TORCHLETTE_NO_ARENA` | off; `=1` disables arena | `executor.ts:3818` | Arena-free reference (witness census uses it). |
| `TORCHLETTE_FUSED_ADAM` | on (webgpu); `=0` | `adam.ts:292` | Fused-vs-graph Adam differential arms. |
| `TORCHLETTE_FOREACH_ADAM` | on (>1 param); `=0` = per-param | `adam.ts:294` | foreach-vs-per-param differential arms. |
| `TORCHLETTE_PACK_OPTIM` | on (Lion/SGD, >1 param); `=0` = per-param | `lion.ts`, `sgd.ts` | Chain-packing P2 (docs/chain-packing-design.md): routes Lion/SGD through `packOptimizerProgram`. Opt-out is the per-param control arm of `parity-packed-vs-unpacked.ts`. **Sunset:** soak → default → the opt-out dies once the packed Lion/SGD path is the proven default (P4-analogue) and nothing load-bearing still compares against the per-param arm. |
| `TORCHLETTE_PACKED_ADAM` | on; `=0` = per-item | `ops/fused.ts:276` | Packed-dispatch measurement kill switch. |
| `TORCHLETTE_DONATION` | on; `=0` disables buffer donation | `segment-executors.ts:269` | Donation kill switch. |
| `TORCHLETTE_BATCH_SUBMITS` | on; `=0` disables shared encoder | `gpu-context.ts:602` | Submit-batching kill switch. |
| `TORCHLETTE_GEMV` | on (M=1); `=0` = tiled | `matmul/variants.ts:180` | GEMV-vs-tiled differential (quant specs toggle it). |
| `TORCHLETTE_SCALAR_TABLE` | on; `=0` skips population | `scalar-table.ts:132` | Scalar-table kill switch (debug). |

## Debug / observability (KEEP — all have a live `ENV` read)

`TORCHLETTE_PROFILE`, `TORCHLETTE_PROFILE_JSON`, `TORCHLETTE_STRICT_GPU`,
`TORCHLETTE_RC_TRACE`, `TORCHLETTE_SCHEDULER_AUDIT`, `TORCHLETTE_TRACE_EPOCHS`,
`TORCHLETTE_MEASURE_D5`, `TORCHLETTE_AUTOTUNE`, `TORCHLETTE_LOG_REWRITES`,
`TORCHLETTE_LOG_DSL`, `TORCHLETTE_TAPE_PROFILE`, `TORCHLETTE_TAPE_VERIFY`,
`TORCHLETTE_STRICT_TAPE`, `TORCHLETTE_BUF_DEBUG`, and the `TORCHLETTE_DEBUG_*`
family: `DEBUG_COMPILED`, `DEBUG_FUSION`, `DEBUG_WRITES`, `DEBUG_SCALARS`,
`DEBUG_REBIND`, `DEBUG_SLOT`, `DEBUG_SLOTS`, `DEBUG_READSLOTS`, `DEBUG_HARVEST`,
`DEBUG_HARVEST_ALL`, `DEBUG_SHAPE`, `DEBUG_OPMATCH`, `DEBUG_SHARED_ENCODER`,
`DEBUG_POOLDUP`, `DEBUG_LIVENESS`, `DEBUG_BIGALLOC`, `DEBUG_AUTOTUNE`,
`DEBUG_ADAM_BUFS`, `DEBUG_OPTPLAN`, `DEBUG_DONATION`, `DEBUG_DESTROYED`,
`DEBUG_DESTROY_STACK_MB`, `DEBUG_STABLEBUF`, `DEBUG_BINDMISS`, `DEBUG_CENSUS`,
`DEBUG_PARAMS_CHANGED`, `DEBUG_OUTER_VERIFY`.

Each was cross-checked against the authoritative `ENV.TORCHLETTE_*` read set
(`grep -rhoE 'ENV\.(TORCHLETTE_[A-Z_]+)' src`) — none is an orphaned hook.
Note: `TAPE_PROFILE`/`TAPE_VERIFY`/`STRICT_TAPE` sunset rides `STEP_TAPE`.

## Config / infra (KEEP)

- Model/run config (tools): `TORCHLETTE_MODEL`, `TORCHLETTE_SEQ_LEN`,
  `TORCHLETTE_BATCH_SIZE`, `TORCHLETTE_PROF_AMP`, `TORCHLETTE_PROF_FUSION`.
- Runtime config (src): `TORCHLETTE_POOL_BUDGET_MB`,
  `TORCHLETTE_RECLAIM_INTERVAL`, `TORCHLETTE_ARENA_MAX_BUFFER_MB`,
  `TORCHLETTE_WEBGPU`, `TORCHLETTE_CPU_ONLY`, `TORCHLETTE_WEBGPU_OPTS`,
  `TORCHLETTE_WIRE_DTYPE`.
- GPU reservation infra (`tools/pick-gpu.sh`): `TORCHLETTE_PICKED_PHYS`,
  `TORCHLETTE_PRIMARY`, `TORCHLETTE_GPU_OWNER_PID`, `TORCHLETTE_GPU_LOCK_DIR`,
  `TORCHLETTE_GPU_LOCK_TTL`, `TORCHLETTE_GPU_FREE_MIB`,
  `TORCHLETTE_GPU_ERRORS_FATAL`.

Non-flag: `__TORCHLETTE_ENV__` (browser env-injection hook in `env.ts`) is
counted by the `envFlags` regex but is a mechanism, not a flag.

## DELETED this sweep (dead references to already-removed flags)

| Flag | Status | Action taken |
|---|---|---|
| `TORCHLETTE_GENERATED_PLAN` | src flag deleted 2026-07-08 (inc-3c) | Removed 5 dead `= "0"` no-op set-sites in `examples/qwen3/*.ts`. |
| `TORCHLETTE_NO_REWRITE` | no src read (hook gone) | Removed the set/delete in `tools/test-merged-plan-repro.ts` (Test G now runs the default path). |
| `TORCHLETTE_DISPATCH_REPLAY` | no src read (replay unconditional) | Fixed misleading "ON/OFF" log lines in `tools/profile-replay-cpu.ts`, `tools/test-replay-parity.ts`; scrubbed comment in `test/webgpu/dispatch-replay.spec.ts`. |
| `TORCHLETTE_TILE_MATMUL` | no src read (tests call the path directly) | Scrubbed stale comment in `test/webgpu/matmul-tile-ir.spec.ts`. |
| `TORCHLETTE_BLOCK_MATMUL` | no src read (tests realize kernels directly) | Removed the dead `beforeAll`/`afterAll` env-set in `test/webgpu/tile-ir-block.spec.ts`. |
| `TORCHLETTE_ARENA_POOL_ACQUIRE` | src flag deleted 2026-06-12 | Removed the flag name from the tombstone comment in `src/backend/webgpu/buffer-arena.ts` (`envFlags` −1). |
| `TORCHLETTE_TAPE_SLOTDIFF` | src flag removed in tape-1c | Removed the flag name from the historical comment in `src/core/tape-profile.ts` (`envFlags` −1). |

## Flagged — LEFT for the campaign owner (not expired)

- **`TORCHLETTE_UNROLLED_K`** — active decode campaign (design doc dated
  2026-07-19). No `ENV` read has landed in `src/` in this tree; only
  `tools/t-uk-cutover.ts` + `docs/unrolled-k-decode-design.md` reference it. Its
  `unrolledKFromEnv()` reader is not present here. NOT deleted — it is
  pre-landing/in-flight, not expired. Confirm with the campaign before touching.
- **`TORCHLETTE_STREAM_GENERATE`** — the flag's `ENV` read is gone, but its
  comments in `src/executor/stream-generate.ts` / `compiled-plan.ts` describe the
  still-live `generateStream` verify apparatus. Docs/stage4 marks deleting that
  apparatus explicitly out-of-scope. Left the comments (deleting them would
  misdescribe live code); it stays in the `envFlags` count until the apparatus
  is retired.
- **`TORCHLETTE_LIVEDIFF`** — tool-local opt-in debug switch
  (`tools/t-ledger-attack-probe.ts`, `=== "1"`). Read but never set; harmless,
  not a shippable src flag. Left in place.

---

## `TORCHLETTE_STRICT_LIFETIME=0` — STOP verdict + diagnosis (2026-07-21)

**Decision: opt-out KEPT. Do NOT remove the guard's warn-mode arm yet.**

The soak sunset (~2026-08) is due, but the standing blocker — a recurring
strict-lifetime GC-timing flake in `test/whole-step-checkpoint-refusal.spec.ts`
— is only PARTIALLY resolved. Removing the opt-out now would make a real
intermittent lifetime race a hard, ~10%-of-runs crash on a hot path.

### What flakes

`whole-step-checkpoint-refusal.spec.ts` throws `[lifetime] reading RECLAIMED
storage` intermittently (~3/8 runs on stock HEAD, VULKAN device reserved, serial
GPU). The throw fires during `markStep()`'s `forceAllPending`, on an `add` or
`gather` node whose *materialized* input storage was demoted at a prior step
boundary. Signatures observed vary run to run — an MLP bias param `[512]`, a
`[1]` scalar, an embedding gather `[32,128]` — across BOTH the eager-ckpt and
remat-ckpt (whole-step) cells. Only the **checkpoint** cells flake; the
non-checkpoint cell never does. The variety + intermittency is the GC-timing
signature: whether a lazy recompute node is force-completed while its input is
still live, or lingers until after the input was demoted, depends on when V8
finalizes RuntimeTensor wrappers.

### Root cause (as far as diagnosed)

Checkpoint eager recompute (and whole-step's single-merged-plan force) builds
forward nodes lazily during backward. Those nodes hold *materialized* refs to
pre-step (pre-Adam) parameter/activation storages. Adam then supersedes the
param (fused kernel writes a fresh buffer, wrapper moves via `_updateLazyRef`),
so the old storage loses its persistent owner and is reaped by the step-scoped
demotion sweep (`storageTracker.releaseStepTemps` / `destroyUnreachable`). A
lazy recompute node that has **not been force-completed within its own step**
then reads that reaped storage when `markStep` finally forces it. A materialized
input ref inside an unforced pending node is not classified as a kept holder by
`_derived` (unlike a live view base or a graph-retention clone), so nothing
protects the storage across the boundary.

### Partial fix found (evidence, not shipped)

Making the test loop faithful to a real training loop — `await loss.item()` each
step, exactly what the strict-clean census tools do
(`tools/t-witness-harvest-matrix.ts:302`) — forces the forward graph to
completion each step and **eliminates the eager-ckpt / eager-nockpt failures
entirely** (0 failures over 12 runs vs ~40% before). This confirms the "unforced
lazy node lingers past its input's demotion" mechanism. It was NOT committed
because a residual remains: the **remat-ckpt (whole-step) cell still flakes
~1/10** — under `wholeStep`, `loss.item()` is deferred (the whole point of the
merged-plan trace), so the forward is never forced mid-step and the recompute
node still races the demotion sweep.

### Named class + why STOP is correct

- **Class:** lingering-unforced-recompute-node reads a demoted parameter storage
  — a genuine intermittent lifetime race in the checkpoint / whole-step recompute
  path, exclusive to the very new whole-step default (graduated 2026-07-19,
  "young" soak).
- The proper fix lives in the whole-step / demotion-classification machinery
  (either force the checkpoint/whole-step recompute segment within its step, or
  teach `_derived` to treat an unforced pending node's materialized input as a
  kept holder). Both touch the highest-risk lifetime code in the repo — the area
  with a long ledger of tried-and-reverted fixes. It belongs to the whole-step
  campaign, not a flag sweep.
- Independently, the `STRICT_LIFETIME` opt-out's own sunset is coupled to the
  whole-step eager-path deletion, which is re-ruled PERMANENT POLICY / deferred
  (docs/step-function-compiler-design). Removing the opt-out now would be
  premature regardless of the flake.

**Follow-up owner:** whole-step campaign. Re-attempt the `STRICT_LIFETIME`
removal once the remat-ckpt recompute-lifetime race is closed and the spec
passes 20× consecutively under strict default.
