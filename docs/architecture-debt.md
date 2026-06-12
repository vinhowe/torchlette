# Architecture Debt: Why the Framework Keeps Producing Silent-Wrongness Bugs

*2026-06-10. Written after the frozen-step_size investigation. Every claim below is
either a count from the current tree or a bug with a reproduction in the session
tooling (`tools/parity-fullstack-tl.ts`, `tools/compiled-faithfulness.ts`).*

## Thesis

The framework's stated design is "the graph is the source of truth"
(`grad-scaler.ts` says it verbatim). The debt is that **every major performance
feature was implemented as a side channel *below* the graph** — a config payload,
a hand-fused kernel, a dispatch-level buffer trick, a positional cache — and every
one of those side channels created a seam that some *other* subsystem (the
compiled-plan replay, the plan fingerprint, the buffer pool, the batching
detector) cannot see. The bugs of the last two weeks are not independent
accidents; they are the same architectural event repeating: **a side channel
silently disagreeing with the graph.**

The optimizer and the training loop are the worst-affected because they are the
most stateful parts of the program, and the framework chose to make their state
invisible to its own IR.

## The experiment: write AdamW "the PyTorch way" today

The clean AdamW the project wants already exists in-tree: `Adam._stepElementwise`
(`src/optim/adam.ts:300-346`) is ~40 lines of pure `runtime.mul/add/div/sqrt/copy_`
— exactly the PyTorch formulation. It is currently demoted to "CPU fallback."
Forcing it on WebGPU (`TORCHLETTE_FUSED_ADAM=0`, added as a probe) gives:

| Configuration | Result |
|---|---|
| pure-graph AdamW, **sequential** (fusion off, compiled off) | **matches pure-JS PyTorch Adam digit-for-digit** (5e-8). The clean definition is correct — when every optimization is turned off. |
| pure-graph AdamW, **fusion on** (compiled off) | **silently wrong**: per-step updates inflate by exactly `bc1(t)/bc1(1)` (measured 1.9×, 2.71×, … — the bias-correction ratio, analytically). The plan-template cache bakes scalar values into cached **fusion recipes** as inlined constants; the fingerprint excludes scalar values, so steps 2+ template-hit step 1's recipe and run with `1−β^t` frozen at t=1. Reproduced by `test/optim/fused-vs-elementwise.spec.ts` (skipped known-bug test #2). **Corollary: any optimizer using JS-number hyperparams that change per step — e.g. SGD or Adam with an LR schedule through the non-fused path — is silently broken under fusion today.** |
| pure-graph AdamW, **compiled plan on** | **silently wrong, independently**: the same scalars are also baked into recorded params buffers and replay frozen (skipped known-bug test #1). Same root, second cache. |
| pure-graph AdamW, lowered + fused off | correct but **355 ms cleanup phase vs 13 ms fused** (DistilGPT-2/V100; 457 vs 98 ms total). GPU time is only ~20 ms — the cost is CPU per-op overhead + no batching. |
| fused AdamW (production) | fast, and correct **as of yesterday** — it had the identical frozen-value bug (uniform `step_size` baked at record time) for the compiled path's entire prior life, misdiagnosed twice as "benign fp32 noise". |
| bonus finding | the two AdamW implementations had **silently forked semantics**: `_stepElementwise` applied weight decay through the gradient (L2 Adam) while the fused kernel does decoupled AdamW. Fixed + pinned by the new differential test; before it, CPU CI exercised one implementation and GPU the other, and nothing compared them. |

So today the framework offers: **clean, fast, correct — pick at most one.**
That is the precise, measurable statement of the thorniness. And the frozen-
scalar disease is now confirmed in **three independent caches** (fused-kernel
uniforms, compiled-plan params slots, fusion recipes in the template cache),
all downstream of one contract violation: **the template fingerprint declares
scalar values irrelevant ("don't affect graph topology or lowered plan
structure" — true) while the things cached under that fingerprint bake the
values in (false).** Two sides of a seam, each locally reasonable, never made
to agree — the exact failure mode CLAUDE.md's "single source of truth at
seams" principle describes.

## Taxonomy

### 1. Op-granularity inversion: the optimizer is a mega-op with a config side channel

`adamStep` is a single backend op whose hyperparameters travel as a JS payload
(`AdamStepConfig`) → kernel uniform, not as graph values. Consequences, each
observed as a real defect:

- Per-step values (`stepSize` with bias correction, `invScale`, `lrTimesWd`)
  are invisible to the replay → **frozen-step_size bug** (wrong LR schedule for
  the compiled path's entire existence; killed the "4.78 vs 4.64" regression gap
  and the "0.73-nat clip divergence vs PyTorch", both previously ruled benign).
- The fix (TAG_UNIFORM volatile repack) is itself **more special-casing**: a
  per-op repack registry wired for exactly `adamStep` and `unscaleGrad`, plus a
  guard for everything else. It treats the symptom; the disease is hyperparams
  not being data.
- A second, independent implementation must exist for CPU — and it **drifted**
  (L2 vs decoupled WD, above).
- In-place semantics are bolted on at the call site: `adam.ts:268-278` literally
  patches `oldBt.destroy = () => {}` on a backend tensor from inside the
  *optimizer* to keep the buffer-pool from reclaiming a buffer the kernel will
  overwrite. The optimizer knows about buffer pools.

**Count:** 29 files in `src/` mention "adam"; PyTorch-equivalent is ~2
(optimizer + an optional fused kernel registration).

### 2. The compiled plan is a second interpreter fed by tracing the backend

Replay correctness depends on the recorder seeing **every** GPU side effect. The
backend has no obligation to route effects through recordable APIs, so each
omission is a silent-corruption bug discovered one at a time:

- unrecorded `copyBufferToBuffer` → embedding grads +1×/replay (22% wrong);
- unrecorded `clearBuffer` → stale accumulators (TAG_CLEAR);
- unrecorded uniform rewrites → frozen step_size (TAG_UNIFORM);
- `createParamsBuffer` data baked at record time → frozen scalars in the
  pure-graph optimizer probe (**still open**, guarded only in tile-dispatch).

The mitigation inventory is the evidence: **eight `record*` hooks**
(`recordDispatch/Alloc/Copy/Write/Clear/VolatileUniform/Barrier` +
`recordedCopyBufferToBuffer`), `endCounters` restoration, `setRecordingNodeIndex`
attribution, "bypass your cache while recording" conventions, and an
invalidation escape hatch. Every future backend optimization must remember to
feed the recorder, or it ships a silent training bug. This is structural: a
record-replay system can only be as correct as the discipline of the thing it
traces.

### 3. The engine owns the training loop

`beginStep`/`markStep`/`snapshotForStep`/`destroyStepScoped` make "one optimizer
step over one model with one scaler" a **framework concept**: persistence is
defined as "alive at the `beginStep` snapshot"; the GradScaler's inf readback is
deferred to *the next step* and internally calls `api.markStep()`
(`grad-scaler.ts:144`); the DiLoCo trainer must call `invalidateCompiledPlans()`
"after outer optimizer steps" because plan validity is coupled to param-buffer
identity. An exotic loop — two optimizers, K inner steps, alternating GAN
updates, meta-gradients, EMA shadows — has to reverse-engineer which of these
lifecycle hooks fire where, and the memory system (step-scoped vs persistent) is
inferred from *when a tensor was created* rather than declared.

### 4. Buffer lifetime has at least seven owners

pool (refcounts + `pendingRelease` + fences) · arena (positional persistence) ·
adopted/pinned plan buffers (new) · params-sequence cache · tile config caches ·
`f16WeightCache` · packed-optimizer buffer cache · persistent inf-flag buffer.
Each has its own reuse-safety rule (`canRecycle`, `arenaBufferSet` shield,
`sharedEncoderWriteSet`, "never destroy before `releaseParamsBuffer`"). The
worst historical bug class ("later step's data leaks into earlier step's
results") lives entirely on the seams between these owners, and the parked
planned-buffers experiment stalled on exactly such a seam (stale grads on the
second replay — three ownership regimes interacting).

### 5. Order-sensitive, string-matched plan transformations

`adam-batch` collects **consecutive** nodes named `"adamStep"`; epilogue fusion
claims **adjacent** nodes; batched reductions require **consecutive** same-dim
sums. When the lazy GradScaler changed DFS order (interleaving `unscaleGrad_i` /
clip's `stridedScatterCopy_i` between adams), batching silently degraded to
size-1 batches — 158 submits/step, packed dispatch never engaged, and *nothing
failed*. The fix (producer hoisting over `ADAM_HOISTABLE_OPS`) is again a
whitelist patch on a pattern-matcher. Dataflow-based rewrites (match on the
graph, not on emission order) would not have this failure mode. The executor's
action vocabulary — `adam-batch`, `batched-reduction`, `matmul-epilogue`,
`row-program`, `prologue-skip` — is a list of optimizations that each got a
bespoke executor extension instead of an IR-to-IR pass.

### 6. Mode combinatorics

27 `TORCHLETTE_*` env flags. The semantically heavy ones — compiled plan ×
arena-liveness × fusion × liveness-release × checkpoint × autocast × scaler ×
clip — multiply into a validation lattice in which each cell is a distinct
buffer-lifecycle semantics. Every recent bug occupied one cell
(autocast+clip+compiled; liveness+compiled; checkpoint+autocast). Tests cover
single cells; the differential gates added this week (`compiled-faithfulness`,
fullstack compiled-vs-lowered) are the right instrument, but the lattice itself
is the liability.

## Bug ledger → taxonomy

| Bug | Sins |
|---|---|
| frozen `step_size` (wrong LR schedule, misdiagnosed twice) | 1, 2 |
| frozen scalars: compiled-plan params slots (open, tracked by skipped test) | 1, 2 |
| frozen scalars: fusion recipes in template cache (open, tracked by skipped test; breaks LR schedules through any non-fused optimizer) | 2, 5 |
| embedding-grad +1×/replay (recordCopy dead) | 2 |
| adam batching silently lost (158 submits/step) | 5 |
| fused/elementwise AdamW semantic fork (L2 vs decoupled) — FIXED + pinned | 1 |
| `cached.data.set` crash (autocast+checkpoint) | 5 (sequence-positional cache), 6 |
| planned-buffers stale grads — ROOT-CAUSED & FIXED: destroyed buffers in replay submits. The "pin" added adopted buffers to `arenaBufferSet`, whose members the arena evicts/destroys at will; persistent-slot buffers (op-internal allocations unmapped at record time) were never pinned at all. Dawn rejects a submit binding ONE destroyed buffer IN ITS ENTIRETY → no compute executes → every plan output silently frozen at recorded values (all grads bit-identical to the recording step). FIX: first-class `pinnedBufferSet` consulted by every destroy/release path; planned-bind fallback now warns + invalidates (cross-plan frozen readers). Payoff: Medium@512 13.8GB at compiled speed — 2.6× over lowered liveness, <half default-arena memory | 2, 4 (ownership expressed by set-membership in a set someone else manages = no ownership at all) |
| planned-buffers follow-on at 124M + pool budget (caught by the REAL 4-peer DiLoCo soak, NOT by any gate — the budget env was the trigger no gate exercised): replay-harvest DOUBLE-OWNERSHIP. The harvest wrapped every result slot buffer in an OWNING storage; for in-place ops (fused adamStep updates param/m/v in their own buffers) that made a second owner of the live state buffer — the previous step's wrap died at markStep and ran the full release chain on it. Latent in unbudgeted modes (buffer pooled + ping-ponged back); with `TORCHLETTE_POOL_BUDGET_MB` the 167MB m/v buffers exceed the budget → deferred-destroyed → every later adamStep submit rejected. FIX: harvest mirrors `wrapResultAsStorage` — in-place results are NON-OWNING views chained via baseStorageId+rcRetain; pinned (plan-owned) buffers non-owning; only unpinned non-aliased buffers own | 4, 6 (the lowered path's ownership/aliasing semantics live in wrapResultAsStorage; the harvest was a second, divergent implementation of the same seam — ownership rules must have ONE implementation) |
| CSE outputIndex / SDPA grads (May) | 5 (structural-key drift) |
| arena 124M/medium OOM | 4 |
| multiple `Torchlette` instances in one process interfere (module-global template/params caches) — forced the differential test into subprocesses | 4, 6 |
| `cat` intra-plan copies unrecorded → compiled replays of any cat consumer use stale data — FIXED (stage 2) | 2 |
| `copy_` silently copy-on-write → persistent tensors migrate buffers every step, arena ping-pong, +640MB/step — FIXED via in-place fast path (stage 2) | 1 (name lies about semantics), 4 |
| per-param elementwise optimizer corrupts first param's m/v on WebGPU with >1 param — ROOT-CAUSED: persistence-contract UAF. `releaseStepTemps` demotes ANY tensor created mid-step (even strongly held by user code) at markStep — its buffer returns to the pool while the tensor is live, and a later plan's output lands in it. The per-param pattern `prevAvg.dispose(); this.expAvg[i] = avg` (replace-and-hold) is exactly this shape; 1-param survived by allocation luck. FIXED: (a) per-param state now `copy_`s in place into the snapshot-protected constructor tensors; (b) `runtime.persist(t)` / `storageTracker.adoptIntoSnapshot(t)` is THE supported way to create long-lived state mid-step (foreach lazy init uses it); (c) a `[lifetime]` read guard in `getInputStorage` makes the whole class LOUD (warn; `TORCHLETTE_STRICT_LIFETIME=1` throws). Pinned by the UAF regression test + `tools/persistence-probe.ts` | 4 (persistence inferred by snapshot membership, not liveness — the contract was implicit and violable with no diagnostic) |
| follow-on to the UAF fix: in-place `copy_(state, new)` as a DANGLING ROOT (its result unread by the param-update chain) defers to whatever plan next touches the state — one step late, after `zeroGrad()` zeroed/freed the grad buffer its pending source reads (NaN by step 4 in markStep-free `stepAsync` loops; Dawn read-write-usage errors when freed intermediates were reacquired as the copy's own dst). FIXED: the update chain reads m/v through the POST-copy_ refs, making the copies forced dependencies. RULE: an in-place write whose result the current chain doesn't read is deferred arbitrarily — sequence side-effecting nodes into the forced chain | 2 (effects vs values: the lazy graph has no notion of "must execute this step"), 4 |

## Target architecture

One sentence: **make the graph the only channel** — anything that affects
computed values must be a node, an edge, or tensor data; everything else
(fusion, batching, packing, replay, memory planning) must be a transformation
*of* the graph that cannot change its semantics.

1. **Optimizers are user-level tensor programs.** `_stepElementwise` *is* the
   definition. Hyperparameters that vary per step (`t`-derived coefficients,
   scheduled `lr`, `invScale`) enter as **data** — 0-d tensors written per step
   — never as op payloads or baked scalars. The replay then handles them with
   the machinery that already provably works (TAG_WRITE re-executes data
   sources; `CHANGING_INPUT` faithfulness is 1e-8). Per-group LR and new
   optimizers (Lion, Adafactor, DiLoCo's outer Nesterov) become user code.

2. **Keep the hand-optimized kernel — change what selects it.** A
   super-optimized AdamW in tile-IR is fine, even desirable. The problem was
   never the kernel; it is that the kernel is the *definition* instead of a
   *lowering*. The clean shape: the graph-level AdamW is the semantic source
   of truth; the engine (or an explicit registration) recognizes the pattern
   and swaps in the tile-IR kernel as an implementation, with (a) hyperparams
   still flowing as data, and (b) the differential test pinning kernel ==
   graph definition permanently (the new
   `test/optim/fused-vs-elementwise.spec.ts` is exactly this gate, kept for
   good). Whether the recognition is automatic (pattern match / codegen) or a
   one-line manual registration matters much less than the direction of
   authority. The engine already proved the generic pieces at small scale:
   scalar-transparent fusion formed single Adam chain groups in Feb ("no
   Adam-specific code needed"), tile-IR generates the current kernel, and
   Phase-2b batches same-shape singleton groups into multi-output dispatches.
   The two missing passes are generic:
   - **horizontal batching**: same-subgraph-different-tensors → one packed/looped
     dispatch (subsumes `adam-batch`, `dispatchPackedOptimizer`, and the
     bias-grad-sum target #3 in one mechanism, dataflow-matched so plan order
     is irrelevant);
   - **scalar uniformization**: per-step scalars lowered to a small per-step
     uniform block uploaded once per step (one writeBuffer), referenced by
     generated kernels — fast AND replay-visible.
   Success criterion: the generic path reaches ≤2× the hand-fused optimizer
   wall time, then the mega-op and its CPU twin are deleted.

3. **Compile from the IR, not from a trace of the backend.** The compiled plan
   should be *derived* (lower the graph → command list with planned buffer
   lifetimes computed from graph liveness) rather than *recorded*. Recording can
   remain as a verification oracle. This dissolves the eight record-hooks, makes
   the arena a memory-planning pass (fixing the medium/124M OOM properly — the
   parked planned-buffers experiment is this idea built on the wrong substrate),
   and turns "serializable compiled plans" (roadmap #10) from a feature into a
   property.

4. **The step becomes a library concept.** The engine keeps only: a memory
   fence (`markStep` as "flush + reclaim") and explicit pin/unpin for
   persistence. Snapshot-based persistence inference, scaler deferral hooks, and
   optimizer-aware cleanup move into `optim`/training-loop code. Exotic loops
   then compose instead of fight.

## Migration path (each stage independently shippable, gated by the existing differential harnesses)

| Stage | Work | Pays for itself by |
|---|---|---|
| 0 | ✅ DONE 2026-06-10: differential test fused vs elementwise (`test/optim/fused-vs-elementwise.spec.ts`, subprocess-isolated, vs pure-JS ground truth); `_stepElementwise` decoupled-WD fixed; two skipped tests pin the open frozen-scalar bugs and define stage 1's exit criteria | killed the semantic fork; the clean-optimizer correctness contract is now executable |
| 1 | ✅ DONE 2026-06-10: **scalars-as-data** via the per-template **scalar table** (`src/executor/scalar-table.ts`). (a) Scalar refs resolve to persistent 4-byte buffers the executor refreshes from the CURRENT step's values before EVERY execution — lowered or replay — replacing the value-baked `full([], v)` fill dispatch; compiled plans bind the table buffers and are value-independent by construction, no record/replay hook needed. (b) Fused recipes: scalar dedupe is positional (value-collapse could alias positions that diverge later), and inlined scalars whose values change get **adaptively demoted** to runtime 0-d inputs (recipe rebuilt, compiled plan re-recorded); truly-constant scalars stay inlined — zero cost for static graphs. (c) The replay gate checks inlined recipe constants against current values and drops stale compiled plans, so even a value that holds through recording and changes later (LR warmup plateau) is honored. RESULT: pure-graph AdamW under FULL default optimizations == pure-JS PyTorch digit-for-digit; late-LR-change probe bit-identical optimized-vs-sequential; both stage-0 known-bug tests unskipped + a late-LR regression test added; all production gates unchanged (fullstack 8.6e-6, faithfulness 1.3e-8, regression PASS, 1087+695 tests). Residuals: non-f32 scalar refs and `full`-payload fill values keep legacy paths; `createParamsBuffer` recording guard still TODO. | the frozen-value class is closed for scalar refs; LR schedulers safe-by-construction; "clean, fast, correct — pick one" is dead for the optimizer probe |
| 2 | ✅ DONE 2026-06-10 (foreach form): **`Adam._stepForeach`** — the per-param tensor program executed once per group over packed flat tensors (graph-level `reshape+cat` in, `narrow+copy_` out, m/v permanently packed and updated in place). Pure graph ops, so fusion, the compiled plan, and the scalar table all apply with zero optimizer-specific machinery. This is PyTorch's own answer (its AdamW lowers to `_foreach_*`); the inferred-from-loops compiler pass remains future work, as does generic reduction batching (bias-grad sums). RESULTS: pure-graph optimizer cleanup **355 → 10 ms** (hand-fused kernel: 11–13 ms), total step ≈ fused parity, memory dead flat; bit-exact vs ground truth incl. real packing (3-param ≡ 1-param equivalence test). Landed two framework-wide fixes it forced out: (a) `cat`'s intra-plan copies were UNRECORDED (any compiled plan containing a `cat` consumer replayed stale data — sin 2, now recorded); (b) `copy_` was silently **copy-on-write** — every full-tensor `copy_` migrated the destination into a fresh arena buffer, so persistent tensors (params! optimizer state) ping-ponged against their own plan positions (+640MB/step, perpetual conflicts); a true in-place fast path (one recorded DMA, adamStep's in-place discipline) fixes params/state stability framework-wide. NEW LEDGER ENTRY (since root-caused & fixed — see ledger): the per-param elementwise path corrupted the FIRST param's m/v state on WebGPU with >1 param (persistence-contract UAF). Also parked: arena-positions-acquire-from-pool (opt-in `TORCHLETTE_ARENA_POOL_ACQUIRE=1`; broke compiled replays via slot aliasing — superseded by the in-place fix). | the pure-graph optimizer is now clean, CORRECT, and FAST — all three; `adam-batch`/packed-dispatch deletion becomes possible in stage 3 |
| 3 | Optimizer-as-graph by default; fused kernel via codegen; delete `adamStep` mega-op, `unscaleGrad` op, ~~`setUnscaleConfig` threading~~ (dead `setUnscaleConfig`/`_pendingUnscale`/`FusedOptimizer` chain deleted 2026-06-10 — GradScaler never engaged the fused unscale+Adam variant; the kernel capability remains, unreferenced). UPDATE 2026-06-11 — **`ADAM_HOISTABLE_OPS` DELETED** (b791d72): the executor's last op-name whitelist is gone, replaced by a GENERIC same-op affinity tie-break in the unified scheduling pass (data edges + WAR anti-deps + checkpoint barriers + affinity, one Kahn); 9 submits/62ms preserved exactly. **`packed-dispatch.ts` RECLASSIFIED, kept**: measured A/B (TORCHLETTE_PACKED_ADAM=0 kill switch) shows packing is FREE on distil (60ms both ways — affinity+single-flush capture that win) but EARNS ITS PLACE on Medium (200ms/13.8GB packed vs 215-248ms/14.4GB unpacked). It is also already optimizer-generic by interface (PackedOptimizerItem + caller-supplied dispatch callback — a backend utility like the matmul tiler, not executor op-matching). The remaining Adam-shaped executor surface is the `adam-batch` action kind, which is multi-output result-protocol routing (node.op → backend op), the same dispatch every op gets. **Default flip MEASURED AND BLOCKED 2026-06-10**: foreach == fused to 1.5e-5/30 steps fp32 fullstack (AMP-profile loss deltas ~5e-3 are f16 quantization of the kernel's algebraically-equivalent `stepSize`/`epsAdjusted` reformulation), and foreach is within ~10% on step time — but the default arena gives each of foreach's ~30 full-model-size graph intermediates a persistent per-position slot: **20.3GB vs 9.1GB** (distilgpt2@512; 2.5GB current under ARENA_LIVENESS, which is the proof the program is fine and the memory policy is the blocker). Sequencing therefore inverts: stage 4 (graph-liveness memory planning) is the PREREQUISITE for stage 3's default flip, not its successor. UPDATE same day: planned compiled buffers landed (liveness mode now runs compiled at bounded memory — see ledger), and under it foreach reaches speed parity (~62-102ms vs fused 57-83ms, distilgpt2@512 A100) at 10.6GB peak vs fused 5.0GB — down from 20.3GB but still ~2× memory: the remaining premium is the foreach program's own packed G/P/pNew working set, not the allocator. Closing it needs graph-level buffer donation (write pNew into P's buffer, G packed in place), which is stage-4 work proper. | −"adam" from ~25 files; new optimizers in userland |
| 4 | Compile-from-IR with graph-liveness memory planning; recording demoted to cross-check | replaces arena + record-hooks; fixes medium/124M memory at compiled speed; absorbs parked planned-buffers |

Rules that should hold from now on regardless of pace (they are cheap and
they would have prevented every bug in the ledger):

- **No new op payloads carrying numeric values that change across steps.** If a
  value varies per step, it is tensor data.
- **No new `record*` hooks.** If an optimization needs one, it is implemented at
  the wrong layer.
- **No new op-name string matching in the executor.** Pattern-match on the
  graph in a compiler pass.
- **Any new optimized path lands with a differential gate that crosses its
  activation threshold** (the compiled plan needs ≥2 executions *with the
  optimizer running*; single-step parity certifies nothing about it).
