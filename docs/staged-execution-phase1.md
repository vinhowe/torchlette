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
   moves slider later). Sufficiency comes from guard 1 (α-change rebuilds the
   hook closure → new ops → structural miss) — covered by an explicit gate
   (§3, G6). Belt and suspenders, because this exact class trained wrong for
   weeks once.
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
