# Staged Execution, Phase 1: the step-tape (program-level replay)

**Status:** plan, approved direction (2026-07-06). Correctness bar: extraordinary —
this campaign edits the layer where every historical silently-wrong bug lived.
**Supersedes:** `dispatch-tape-design.md` (its Dawn-level tape shipped as the
generated-plan replay; this campaign sits one layer ABOVE it).
**Companions:** `stage4-compile-from-ir.md` (phases 0–4, the substrate this
builds on), `scoped-memory-design.md` (§1 epochs are the shared vocabulary),
`architecture-debt.md` (the sin taxonomy the gates encode).

## 0. What this is, in one paragraph

Today the engine re-derives the program every step: ~1,600 lazy IR nodes are
rebuilt in JS, fingerprinted, CSE'd, and rewritten — only to conclude "same as
last step" and hand off to the (already-shipped, already-proven) compiled-plan
replay. The step-tape makes "same as last step" a **declaration checked by
guards** instead of a conclusion re-derived by construction. A tape is NOT a
new dispatch recorder — it is a memoized *step program*: an ordered list of
`[slot writes] → [compiled-plan executions by template id] → [readbacks]`,
sitting entirely on top of plan replay. Replaying the tape skips graph build,
fingerprinting, CSE, rewrites, and plan lookup. Everything below the tape
(plans, buffers, kernels) is untouched.

Measured motivation (task #60 accounting, V100 Node, 1.7B decode): graph
rebuild ~5.5ms + fingerprint/CSE/rewrite ~5–6ms + replay-loop JS ~5ms +
sweeps ~4.5ms of the 47ms/token. Browser (worker, 1.7B-class): build 21 +
lower 15 + step 43 ms/token — the absolute win is larger exactly where the
demo lives. Honest ceiling: this does NOT touch `fence` (GPU time); on a
fence-dominated config (4B on an iGPU) the win is secondary.

## 1. Scope: decode only, at a library-owned seam, opt-in

Phase 1 captures ONE loop shape: the per-token decode body inside
`generateChat` (and the equivalent in the steering worker) — chosen because:
- The loop body is **library-controlled**: its structure cannot change between
  iterations by construction (no arbitrary user code inside the captured
  region). Capture-of-user-code (torch.compile's hard problem: guards on
  closures and control flow) is explicitly OUT of phase 1.
- Its dynamic surface is small and enumerable: the token id, the per-step
  data uploads (mask rows, scatter indices), the KV bucket length, and any
  steering-hook scalars.
- Its correctness oracle already exists and is strong: kv-differential
  (bit-exact greedy tokens), topk-equivalence, the steering smoke, and the
  phase-0 `stream-diff` harness (`src/executor/stream-diff.ts`) for
  byte-level command-stream comparison.

Training loops, arbitrary `api.scope()` bodies, and an implicit whole-program
cache are phases 2+. Multi-engine is orthogonal (the de-singleton foothill).

## 2. Architecture

### 2.1 The tape

```
StepTape = {
  bucketKey:  string           // model + seqLen bucket + steering-structure key
  entries:    TapeEntry[]      // ordered
  slots:      DynamicSlot[]    // enumerated per-step-varying inputs
  epoch:      number           // engine epoch at record (invalidation tie-in)
  structGen:  number           // structural generation (see guards)
}
TapeEntry =
  | { kind: "write";   slot: SlotId }                  // upload into a stable plan buffer (TAG_WRITE path)
  | { kind: "plan";    templateId: TemplateKey }       // execute an existing compiled plan
  | { kind: "readback"; which: "topk" | ... }          // the enumerated outputs
DynamicSlot = { name, shape, dtype, source: "tokenId" | "upload" | "payload" }
```

The tape references **template ids and slot ids, never raw buffers** — buffer
ownership stays with the planner registry (stage-4 phase 1.5). Tape lifetime is
tied to plan validity: any plan invalidation (staleness guard, eviction,
teardown) invalidates every tape referencing it. Nothing in the tape layer
owns GPU memory.

### 2.2 Recording

Recording is observation of the EXISTING path, not a parallel implementation:
during normal (un-taped) execution of a step, the engine appends what actually
happened — which slots were written (the TAG_WRITE stable-buffer path from the
CPU campaign already names them), which plan templates executed, which
readbacks ran. Two consecutive steps that produce structurally identical
records ⇒ the tape is eligible. This mirrors how the compiled plan itself
records on execution 1 and cuts over on execution 2+ — the pattern is proven.

### 2.3 Replay

`executeTape(tape, stepInputs)`: write the dynamic slots (queue.writeBuffer
into the stable plan-owned buffers), execute each referenced compiled plan via
its existing replay entry point, run the readbacks. No lazy nodes, no
fingerprint, no CSE, no plan lookup. The scope/step boundary still runs
(markStep semantics unchanged) — but with near-zero step-created temporaries,
the sweep cost collapses too.

### 2.4 Guards — the load-bearing correctness section

A tape replays IFF every guard passes; ANY miss falls back to the normal path
for that step (which re-records), increments a counter, and under
`TORCHLETTE_STRICT_TAPE=1` throws. **A silent stale replay is the failure mode
this design exists to prevent; fallback is always correct because the normal
path never went away.**

1. **Structural generation.** Every runtime-op creation bumps a per-engine
   op-sequence counter. The tape records the counter delta of its step; a step
   whose op count/sequence would differ (any code path change: hook toggled,
   collectHidden turned on, a new op interleaved by the app between steps)
   misses the guard. Cheap: one integer compare, no re-derivation. This is the
   guard that makes library-seam capture sound: within `generateChat`'s loop
   the structure is constant by construction, and anything that violates that
   assumption is DETECTED, not assumed away.
2. **Bucket key.** KV bucket length, model identity, steering-structure key
   (hook present? which layer? — structure, not values). Bucket transition ⇒
   different tape (record one per bucket; ≤16 tapes for 2048 ctx, tiny).
3. **Dynamic-slot completeness — the frozen-scalar defense.** The recorder
   diffs the FULL payload/config byte-image of consecutive steps (the
   `getConfigBuffer` staleness-guard technique, generalized): any byte that
   varied between the two recording steps must belong to a declared slot;
   any byte that varied and is NOT slot-covered ⇒ the tape is REFUSED (never
   recorded), with a diagnostic naming the op — the PAYLOAD THRASH message's
   sibling. This is what catches steering-α today: `api.mul(dir3d, alpha)`
   bakes α as a payload scalar; two recording steps at the same α would freeze
   it. Mitigation shipped WITH phase 1 (either is acceptable, first is
   simpler): (a) model-level — α becomes a persistent 1-element tensor written
   per-generation (position-as-data pattern, zero engine work); (b) engine —
   the #71 declared-dynamic-payload mechanism. NOTE: refusal-on-undeclared-
   variance is necessary but NOT sufficient (α constant across recording, user
   moves slider later). **[CORRECTED BY 1a — §8.4]** The original claim that
   guard 1 provides sufficiency (α-change → new ops → structural miss) is
   EMPIRICALLY FALSE: α is a graph scalar ref resolved via the scalar table;
   op sequence and fingerprint are byte-identical across an α change (w3
   measured: same fp 0x16e9a591, node[959] scalar 3 vs −3). α safety rests
   ENTIRELY on value-level coverage — α as a declared slot (4th DynamicSlot
   source: `scalar`, a scalar-table slot) or α-as-tensor, plus guard-3's
   byte-diff. Today's engine already catches the change via scalar-adapt
   demote→re-record; the tape must inherit exactly that path. G3 is
   specified against value-level coverage, not structural detection.
4. **Plan validity.** Tapes hold no validity of their own: each referenced
   plan's existing staleness/eviction machinery stands; any invalidation
   cascades to the tape.
5. **Epoch continuity.** Tape records the scoped-memory epoch discipline of
   its step; replay under a different boundary regime (stepScopedCleanup
   toggled, explicit beginStep interleaved) misses.
6. **Strict cross-check mode.** `TORCHLETTE_TAPE_VERIFY=N`: every Nth replay
   ALSO runs the full normal path and byte-compares command streams via
   `diffStreams` — in-suite paranoia, and the long-soak gate. N=1 = shadow
   mode (all verify, no speedup) used for the entire gate ladder before any
   default flips.

### 2.5 What phase 1 deliberately does NOT do

- No capture of user-authored loop bodies (no closure guards, no bytecode
  tricks). The seam is the library's.
- No implicit activation: opt-in flag (`TORCHLETTE_STEP_TAPE=1`), default off,
  even after gates pass. Default-flip is its own later decision with soak data
  (the liveness-default and cutover-default playbooks).
- No deletion of anything. The normal path is the permanent fallback AND the
  cross-check oracle. (Deletions are stage-4 phase-4 style dividends, earned
  later.)
- No new memory ownership; no touching kernels, plans, pools, or the sweep.

## 3. Gate ladder (each solo; ordered; ALL must pass before the flag ships even as opt-in)

- **G0 — measurement first**: extend `timeline-decode.ts` to report the exact
  skippable-JS total on current HEAD (Node + browser via the new worker
  breakdown), so the win is a prediction before it is a claim.
- **G1 — recorder is a pure observer**: with recording on and replay OFF,
  byte-identical behavior: full suite, kv-differential (cat+static),
  topk-equivalence, steering smoke, decode perf within noise. (The stage-0
  epoch-trace fingerprint technique; its coverage hole lesson applies — the
  gate set must include steered + unsteered + bucket-crossing generations.)
- **G2 — shadow equivalence**: `TAPE_VERIFY=1` (replay computes, normal path
  also runs, streams diffed) across: a 200-token generation crossing ≥2 KV
  buckets; steered and unsteered; α changed mid-session; hook toggled
  mid-session; model switched 0.6B↔1.7B. ZERO diffs, and every deliberate
  perturbation (α change, hook toggle) must appear as a COUNTED GUARD MISS,
  not a diff — proving the guards fire before divergence can.
- **G3 — the frozen-α differential**: record at α=3, replay, change slider to
  α=−3, generate: greedy tokens MUST match a never-taped α=−3 run bit-exactly.
  This gate exists because of the step_size history; it is the one that must
  never be waived.
- **G4 — lifetime**: STRICT_LIFETIME + STRICT_GPU clean over a taped
  generation; reachable-storage flat across 100 taped steps (tape must not
  leak wrappers or pin buffers).
- **G5 — inference oracles**: kv-differential grows a fourth arm (tape) —
  none/cat/static/taped all bit-identical; topk-equivalence under tape;
  steering smoke under tape.
- **G6 — training untouched**: test:gates 4/4, parity-fullstack both compiled
  settings, scope-training — all with the tape flag ON (tape must correctly
  decline non-decode workloads: guard misses, zero behavior change).
- **G7 — perf claim**: decode ms/token before/after on Node AND browser
  (1.7B), with the G0 prediction as the acceptance band; report guard-miss
  counters (steady-state misses must be 0).
- **G8 — soak**: `TAPE_VERIFY=16` over a long mixed session (many generations,
  slider changes, preset changes, model reload) — zero verify diffs, then the
  same in the browser e2e path.

## 4. Failure postures and stop conditions

- Guard miss ⇒ normal path + re-record. Never throw in default mode; strict
  modes exist to make CI loud.
- If recording finds >0 undeclared-variance refusals on the STOCK decode loop
  (i.e., the slot enumeration is incomplete in ways we didn't predict), STOP:
  that's a design gap in the slot model, not something to whitelist through.
- If G2 shows even one stream diff that isn't a guard-miss bug, STOP and
  root-cause; a tape that is "mostly identical" does not ship — this layer has
  no benign-divergence category (clip-divergence lesson: descending-faster IS
  a bug).
- If the win measured at G7 is <5ms/token on Node and <15ms in browser, the
  campaign pauses for re-evaluation before any further phases — the complexity
  must pay.

## 5. Sequencing and staffing

1a (G0 measurement + slot-model spec) → 1b (recorder, G1) → 1c (replay +
guards, G2–G8). Each lands as its own commit with its gates in the message.
One agent per sub-phase, briefed with this doc + the landmine ledger
(frozen-scalar, coverage-hole, ordering-luck, solo-GPU, no-stash); the
α-as-data model fix lands in 1c's commit. #71's engine mechanism is NOT a
prerequisite (α-as-tensor suffices for phase 1) but 1c must not foreclose it.
De-singleton (#74/engine-instance) is not a prerequisite: phase 1 is
single-engine by scope.

## 6. Tradeoffs this direction commits us to (discussed + accepted 2026-07-06)

The tape is a BET THAT THE WORKLOADS THAT MATTER ARE LOOPS. It shapes the
performance gradient, never the correctness surface (fallback is permanent and
correct). What that gradient does to workflow shapes:

- OPTIMIZED: fixed-program loops (decode, training steps, DiLoCo rounds);
  repeated-program-varying-values (steering α, schedules, positions) — IF
  values flow as data/declared slots (the discipline the guards enforce).
- NEUTRAL-WITH-HICCUPS: occasional structure changes (hook toggles, layer
  changes) — one guard-miss + re-record, then fast. UI CONSEQUENCE: warm
  (value-knob) vs cold (structure-toggle) interactions — the UI leaks the
  cache topology, same physics as shader-compile hitching. Mitigations owed:
  (a) guard-miss observability exposed to apps so UIs can show "re-tracing…"
  instead of mystery stutter (add to 1c's deliverables); (b) UI guidance:
  scrubbed controls should be value-knobs.
- UN-SERVED (not harmed): one-off computations (interp-notebook shape) — run
  at today's speed; NOTHING in this campaign reduces cold-start. If the
  thesis pivots to exploratory tooling, the right investment is the opposite
  axis (cheaper fingerprint/incremental planning) — revisit then.
- IMPLICITLY DISCOURAGED: structurally-dynamic programs (variable-width beam
  search, dynamic MoE routing, speculative decoding's variable accepts,
  per-token architecture variation, ARCHITECTURE MUTATION — the Menagerie/
  Picbreeder and model-surgery shapes). Permanent fallback-path citizens.
  Named risk: if mutation-playground becomes the primary thesis, this
  campaign optimized the wrong axis — REVISIT BEFORE PHASE 2.

Philosophical cost accepted: Fact 2's fine print gets heavier stakes (structure
-as-data vs structure-as-code = 20ms vs slow), though no third fact is added;
guards convert violations into loud-and-correct rather than silent-and-wrong.
Centralization pressure accepted for phase 1 only: the fast path lives behind
library seams; phase 2's explicit capture API is the owed counterweight —
do not quietly drop it.

## 7. The other pole is a mode, not a fork (2026-07-06 addendum)

Sharpening §6 after discussion. Decompose "the floor":
- The TAPE is ~Pareto: recording is observation, guards are integer compares;
  never-repeating workloads pay epsilon.
- The floor for non-loops was raised EARLIER, by the compiler-pole machinery
  itself: every step pays fingerprint/CSE/template-lookup (~5-6ms @1.7B) —
  pure reuse-discovery, worthless to a workload with no reuse — and novel
  SHAPES pay pipeline compilation because kernels are shape-specialized.
- A genuine interpreter pole exists (ggml-shaped): shape-polymorphic kernels
  (dims as uniforms — never compile on novel shapes), no reuse-discovery,
  streaming alloc. Flat low floor; price = 10-30% kernel peak, no cross-step
  memory packing, hot-loop ceiling ~2-5x above ours. Irreducible novelty floor
  exists (a new program must reach the GPU somehow; repetition-as-information
  always wins) but ours sits well above it today.
- KEY: because selection is data (#61) and kernels are IR, the interpreter
  pole decomposes into ADDABLE MODES, not a rival design: (a) shape-GENERIC
  kernel variants in the registry (uniform-driven dims; tile-IR grids half-do
  this); (b) THRASH-TRIGGERED DEMOTION — the PAYLOAD THRASH detector routes
  offenders to generic variants instead of just warning (same detect-and-
  degrade shape as every guard); (c) skip-fingerprint fast path for graphs
  learned never to match. Each lowers the non-loop floor without touching the
  loop ceiling. Scarce resource = attention, not architecture.
- Deepest frontier for change-a-little-constantly workloads (Menagerie
  mutation, notebook tweaks — rarely PURE novelty): SUB-PROGRAM caching —
  subgraph-level fingerprint/reuse instead of whole-template all-or-nothing,
  so a 95%-unchanged program reuses 95% of its plans. Incremental-compilation
  territory; the Menagerie-shaped optimization if that thesis leads.

## 8. 1a findings (G0 measurement + empirical slot model, 2026-07-07)

Measured on the V100 box (sivri, Node/Dawn), Qwen3-1.7B f32, static KV,
maxSeqLen 512, GPU solo. Instrumentation: `src/core/tape-profile.ts`
(`TORCHLETTE_TAPE_PROFILE=1` seam timers, `TORCHLETTE_TAPE_SLOTDIFF=1`
per-plan payload/scalar images) + `examples/qwen3/timeline-decode.ts` G0 mode
+ `examples/qwen3-steering/tape-slot-diff.ts` (w1/w2/w3 drivers). Both flags
and every guarded seam SUNSET when 1c lands.

### 8.1 G0 table — per-token seam attribution

`TORCHLETTE_TAPE_PROFILE=1 npx tsx examples/qwen3/timeline-decode.ts 18 f32`,
12 post-warmup tokens (steps 6–17; warmup 6 because pool/arena reuse settles
after plan cutover). The decode step is ONE 1,611-node plan (fp 0x2e67bb41;
1,613 steered — the §0 "~1,600 nodes" claim verified) + backend-direct
readTopK + markStep.

| seam | mean ms | p50 ms |
|---|---|---|
| wall (ms/token) | 47.78 | 45.04 |
| (a) lazy graph build (frontend → lazy IR) | 5.55 | 5.52 |
| — plan-collect (buildMergedPlan topo) | 1.86 | 1.55 |
| (b) template fingerprint | 2.12 | 2.03 |
| (c) CSE (pass:cse) | 1.63 | 1.55 |
| (d) rewrites (DSL rules + other passes + consumer maps) | 3.35 | 2.37 |
| (e) plan lookup / cache-hit perm | 0.03 | 0.03 |
| — scalar-table refresh | 0.20 | 0.20 |
| (f) replay-loop JS (excl. Dawn, slots, harvest) | 3.11 | 2.64 |
| — replay: slot population | 0.20 | 0.12 |
| — replay: result harvest | 6.15 | 5.96 |
| (g) Dawn encode+flush+submit inside replay | 9.41 | 7.93 |
| (h) markStep sweeps (+snapshot +runtime.markStep) | 7.80 | 7.49 |
| (i) readback (mapAsync waits) | 0.05 | 0.05 |
| — fence waits (markStep) | 0.13 | 0.11 |

(9 submits/token; createBindGroup and createBuffer are 0/token at steady
state. `replay-total` = 18.9 = (f) 3.11 + dawn 4.63 + barrier/flush 4.78 +
slots 0.20 + harvest 6.15. ~5 ms of `force` wall is unattributed glue:
tagPlanOutputs/audit/retain, getInputStorage resolution, materialize loop.)

Reconciliation vs the task-#60 coarse numbers: rebuild 5.5 → (a) 5.5 ✓;
fingerprint/CSE/rewrite 5–6 → (b)+(c)+(d)+(e) = 7.1 (slightly higher; the
coarse number didn't count the cache-hit consumer-map rebuild or plan-collect
at all); replay-JS ~5 → 3.1 loop JS **but the harvest (6.2 ms) was invisible
in the coarse accounting** — total non-Dawn replay JS is ~9.5; encode ~4 →
Dawn-in-loop 4.6 ✓; sweeps ~4.5 → **7.8 measured** (the earlier number
under-counted: it excluded the end-snapshot and runtime.markStep, and sweep
cost scales with the ~1,560 step-scoped storages/token the harvest creates —
see [sweep] destroy1=428 + release=1554 + destroy2=1131).

### 8.2 Tape-skippable subtotal → the G7 band

SKIPPABLE = (a) + plan-collect + (b) + (c) + (d) + (e)
= **14.55 ms mean / 14.26 ms p50** of a 47.8/45.0 ms token.

**f-partial = 0 in the subtotal**, and this is a spec correction to §2.3:
"No lazy nodes" is NOT achievable while the tape calls the EXISTING
`executeCompiledPlan` entry point — that function requires `planNodes` for
(i) external-slot resolution (`src.kind==="external"` reads
`planNodes[i].inputs`), (ii) TAG_WRITE payload extraction, (iii) TAG_UNIFORM
`pack(node)`, and (iv) the result harvest (6.15 ms/token) that hangs
StorageHandles on nodes for downstream consumers. 1b/1c must either keep a
per-tape skeleton graph (record the planNodes array once and re-dress its
per-step values — cheapest, keeps replay untouched) or add a slot-direct
replay variant. Until harvest is bypassed, (h) will NOT collapse either: the
step-scoped storages being swept ARE the harvest's handles + views. §0's
"sweeps ~4.5" win and §2.3's "sweep cost collapses too" are therefore
optimistic — count them as phase-1c stretch, not the G7 band.

**G7 acceptance band (Node, this box, 1.7B f32 static):** taped steady-state
decode must come in at **untaped − 11 to − 16 ms/token** (i.e. ~30–36 ms/token
against today's ~45–48), with steady-state guard misses = 0. The band's floor
(11 ms) allows ~3 ms of tape-side overhead + variance; anything under
−11 ms/token means the tape is not skipping what 1a measured and the campaign
re-evaluates per §4 (still comfortably above the §4 hard stop of 5 ms).
Observer effect of the instrumentation itself: ≈ +0.3 ms/token, within noise
(flag-ON G0 run mean 47.78 ms over 12 post-warmup tokens vs flag-OFF
profile-decode 47.5 ms avg over 8 post-warmup tokens, same box, same session).

### 8.3 Empirical slot model (Deliverable 2)

`TORCHLETTE_TAPE_SLOTDIFF=1 npx tsx examples/qwen3-steering/tape-slot-diff.ts
w1|w2|w3` — byte-diff of consecutive steady-state decode steps (writeBuffer
trace + per-plan-position payload/scalar images).

**Complete steady-state write set, per token (w1 stock; w2 steered is
byte-for-byte the same shape):** 33 writeBuffer calls —

| # | source | size | varies step→step? | §2.1 category | carried by TAG_WRITE stable path? |
|---|---|---|---|---|---|
| 0–27 | scalar-table refresh (28 slots, all = 1/√128 attn scale) | 4 B | **no** (bytes identical; see artifact below) | scalar (see 8.4) | n/a (scalar-table buffers) |
| 28 | token id `tensorFromArray [1,1]` | 4 B | yes (when token differs) | tokenId | **yes** |
| 29 | ropeIdx `[1,64]` (position rows) | 256 B | yes | upload | **yes** |
| 30 | scatterIdx `[1,8,1,128]` (KV write position) | 4 KB | yes | upload | **yes** |
| 31 | decode mask `[1,1,1,bucketLen=128]` | 512 B | yes (−1e9 boundary moves) | upload | **yes** |
| 32 | readTopK params (topk-kernel) | 8 B | no (k, vocab fixed) | readback config | no (backend-direct dispatch, outside plan replay) |

Plan-position payload diff agrees exactly: the ONLY per-step-varying payloads
are plan nodes 0–3, all `tensorFromArray` (token id, ropeIdx, scatterIdx,
mask). **No TAG_UNIFORM volatile configs exist in the decode plan** (those are
optimizer-era constructs), and nothing varies that fits no category → the §4
"undeclared variance" stop condition is NOT triggered for the stock loop.
DynamicSlot count for w1/w2: 4 slots (1 tokenId + 3 uploads), all already
carried by the TAG_WRITE stable-buffer path.

**w2 (steered, α fixed):** identical slot set. α appears ONLY as a scalar
value on plan node[959] (`mul`), constant within the generation — zero
per-step writes for it. The steered plan is the same template ± 2 nodes
(1,613 vs 1,611) under a different fingerprint than stock (hook present =
structure), exactly as guard 2 expects.

**w3 (α +3 → −3 across generations):**
- Steady-state A.step8 vs B.step8: same fingerprint (0x16e9a591 — scalar
  values are excluded from it), byte-diff shows `plan[0] node[959] op=mul
  SCALAR-DIFF [3] vs [-3]` → **hazard #1 empirically confirmed: guard-3-style
  byte/value diffing catches the α change across recordings.**
- What the engine does TODAY: the change trips the inlined-fusion-scalar
  staleness check → `[scalar-adapt] input 1: baked=3 current=-3` →
  `demoting 1 inlined scalar(s) on a 2-node group` → compiled plan dropped,
  2 lowered steps (796/795 writeBuffers — per-dispatch uniforms), re-record,
  back to 33-write replays by step 2, with α=-3 landing in a scalar-table
  buffer at B.step0 write[0]. Within one generation α is FIXED (w2/w3
  B.step8 vs B.step9: no α write, no scalar diff); across generations it is
  the one value-level variance.

### 8.4 Spec corrections 1b/1c MUST inherit

1. **α is a graph SCALAR REF, not an op payload** (§2.4 guard-3 text says
   "bakes α as a payload scalar"). `api.mul(dir3d, alpha)` produces an input
   of kind `scalar`; payload-hashing never sees it, the fingerprint excludes
   it, and the existing machinery (inline-then-demote + scalar table) already
   carries it as data after one adaptation. Consequence for §2.1: DynamicSlot
   needs a 4th source category — `scalar` (a scalar-table slot) — or the tape
   records scalar-table writes as slot writes. The mitigation ladder in §2.4
   stands, but option (b) is closer to shipped than the spec assumed.
2. **Guard-1 does NOT cover an α change — the sufficiency claim is false.**
   §2.4 asserts "α-change rebuilds the hook closure → new ops → structural
   miss." Measured: the op sequence, node count, and template fingerprint are
   IDENTICAL for α=+3 and α=−3 (0x16e9a591 both). A new closure produces the
   same two ops. α safety therefore rests ENTIRELY on value-level coverage
   (declared scalar slot / α-as-tensor / guard-3 byte-diff refusal) — G3/G6
   must be specified against that mechanism, not the structural counter.
3. **§2.3 "No lazy nodes" / "sweep cost collapses" are overstated** — see
   §8.2: the replay entry point consumes planNodes and its harvest generates
   the very step-scoped storages the sweep pays for. 1c needs the skeleton-
   graph (or slot-direct harvest) design decision made explicitly.
4. **Scalar-table change-detection artifact (fix in 1b, one line):** the CPU
   mirror is a `Float32Array` but the comparison is `Object.is(f32mirror, f64
   value)` — any scalar not exactly representable in f32 (e.g. 1/√128) fails
   the compare EVERY step → 28 redundant 4-byte writeBuffers/token, bytes
   identical. Harmless but noisy: a tape recorder diffing writes must treat
   byte-identical rewrites as non-slots (or fix the compare to
   `Object.is(mirror[i], Math.fround(v))` first).
5. **The readTopK dispatch lives OUTSIDE plan replay** (backend-direct, own
   params write + own submit). The tape's `readback` entry must record it as
   its own step (params buffer is constant per bucket/k), not assume every
   GPU command is inside a compiled plan.

Gates run for 1a (measurement-only): kv-differential PASS (all arms
bit-identical); full `npm run test` green with the seams in src/ (flags off);
decode perf unchanged with instrumentation off (profile-decode 12 f32 static
within the 47–57 ms/token band before and after); instrumentation-on observer
effect ≤ ~1 ms/token (reported above).

## 9. 2a findings (capture(), 2026-07-07) — rules phase 2b MUST inherit

1. **The probe-unsoundness rule: never execute a captured body past
   side-effecting ops for coverage.** 2a's first scalar-coverage design ran
   the fn deep enough to observe closure scalars (α at layer 14) inside an
   aborting reclamation scope. EMPIRICALLY UNSOUND: in-place ops commit
   version bookkeeping (`_inPlace` → `_debug_baseCommit`/`nextMutId`) eagerly
   at GRAPH BUILD, and a scope abort cannot roll them back — the steered
   decode's KV `copy_` chain desynced tensor versions ("[lifetime] reading
   RECLAIMED storage", token divergence). A follow-up in-place-suppressing
   probe mode was rejected: a probe must suppress ARBITRARY user JS effects
   (a cache-length advance, a counter), which is impossible — each fix
   surfaces the next effect class (the torch.compile closure-guard tarpit §1
   swore off). Consequence: a captured body runs EXACTLY ONCE per call,
   always for real; a hit short-circuits it AT its last upload (before any
   layer/in-place op), never after.
2. **The arg-boundary contract** (capture()'s value coverage): everything
   that varies crosses the ARGUMENT LIST — tensor args are warm dynamic
   slots (values re-dressed/read-live every call), plain-value args are
   hashed into the bucket key (cold: counted miss + re-record), and CLOSURE
   VALUES ARE FROZEN AT RECORD TIME (jax.jit semantics; documented + tested,
   TAPE_VERIFY as backstop). This supersedes both hand-rolled appKeys (the
   1c caveat) and derived-scalar probing.
3. **Frozen-upload disease found + fixed in the FUSION RECIPE cache** (the
   third frozen-scalar cache): `isInlinableScalar` inlined 1-element pending
   `tensorFromArray` payloads as WGSL constants; the recipe is cached by the
   payload-excluded fingerprint, and the staleness check silently skipped
   already-executed data-sources (`node.result` refusal) — so a
   mid-generation α-as-tensor flip DID NOTHING under default fusion, plain
   engine, no tape (silent wrongness, the worst class). Fixed: 1-element
   tensorFromArray is never inlined (it is per-step USER DATA; as a runtime
   input it rides TAG_WRITE), and the staleness read
   (`inlinedConstantValue`) sees payloads regardless of materialization.
   Without this, tensor-args could never be warm slots.
4. **Upload-arg donation**: a caller-built pending tensorFromArray arg whose
   values a replay consumed is DISPOSED by the hit (else the never-consumed
   node is force-executed as a wasted mini-plan at every markStep).
   Reclamation of the short-circuited partial body is by interceptor-tracked
   wrapper disposal, not a scope (a per-token scope open/abort cost
   ~1 ms/token; wrapper disposal is ~free).
