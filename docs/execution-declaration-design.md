# Execution Declaration: the command-stream stratum as data

**Status:** v1 design — 2026-07-17. Design-only (doc + staged campaign plan; NO
mechanism lands with this commit). Worktree off `main` @ `5cc50d3c`.
**Depends on:** `docs/schedule-state-design.md` (S1 three-tier object; the
intra-kernel stratum — this campaign is the *next altitude up* and consumes its
receipts), `docs/stage4-compile-from-ir.md` (build-from-IR is now the SOLE
compiler; the recorded build was deleted 2026-07-17 / D4 attempt #13), the
already-shipped mini-precedents `src/executor/contiguous-operands.ts`
(CONTIGUOUS_OPERANDS), `computeChunkGeometry`/`chunkedBinaryConfig`,
`computeBindingOrder`, and the `operand-layout-metadata-design.md` /
`islands-design.md` chunking indictment.
**Sibling charter:** schedule-state derives the *kernel body* (WGSL) from a
declaration; this derives the *command sequence around* the kernel from a
declaration. Together they retire the last two hand-written descriptions of op
execution.

---

## 0. Declaration (one sentence)

An op family's realization into GPU commands is a first-class **command-stream
declaration** — the ordered command skeleton (buffer roles, binding order,
intra-plan copies/clears, config packing) plus derived geometry (chunk splits,
operand prologues) computed by shared splitters — from which the live dispatch
is the INTERPRETER and the compiled `GpuCommand[]` is the SERIALIZER, so the
mirror between them stops existing.

**House rule form:** *there is exactly one description of how an op executes;
the WGSL body derives from its ScheduleState (schedule-state-design), the
command sequence derives from its ExecutionDeclaration (this doc), and every
consumer — live dispatch, compiled replay, the editor — reads the same object.*

---

## 1. Lineage and why

This is the **seventh application** of the house move "the latent decision
becomes an object," and the direct successor to the sixth (schedule-state):

1. scalars → tensors-as-data (`inc2a`, SGD/Adam payloads)
2. buffer assignment → planner-derived (`memory-planner.ts`)
3. compiled replay → built-from-IR (stage-4, recorded build DELETED)
4. variant selection → registry data (#61)
5. dispatch partition → islands (I0–I2)
6. **kernel body → ScheduleState** (schedule-state-design; kernel WGSL now
   LOWERS from the three-tier object — attention/Adam/matmul/reduction/
   elementwise bodies all derive; the mirror at *kernel* altitude is gone)
7. **command sequence → ExecutionDeclaration** (this doc; the mirror at
   *command-stream* altitude — dispatch vs stream-generate — is what remains)

**Why now.** The three hand-maintained descriptions of op execution were:
(a) the **kernel WGSL** — *already unified* by schedule-state (`realize*`
chokepoints lower the body from ScheduleState; the second body-owner factories
are deleted). (b) the **dispatch code** — hand-written, the live/lowered
interpreter (`src/backend/webgpu/index.ts`, `dispatch.ts`, `ops/*.ts`,
`matmul/dispatch.ts`, `fusion-dispatch.ts`). (c) the **stream generation** —
hand-mirrored serializer (`src/executor/stream-generate.ts`, 4307 SLOC), the
build-from-IR compiler's back end. With (a) closed, (b) and (c) are two dialects
of ONE fact — the command sequence — kept in sync only by a runtime differential
(`stream-diff.ts`) and 36 confessed "mirrors X" comments. Every stage-4 D4
sunset coverage errand (attempts #7–#13: all-dims sum, arange, batch>1,
chunked-elementwise, released strided/broadcast views, scalar-source scatter)
was a hand-synchronization of these two dialects, differential-guarded. The
proven mini-precedents (`CONTIGUOUS_OPERANDS`, `computeChunkGeometry`,
`chunkedBinaryConfig`, `computeBindingOrder`) already single-source *individual*
command-stream decisions; this campaign generalizes them into one declaration
per op family.

**The condition that makes this better, not worse (schedule-state's R22 rule,
inherited):** the campaign must EARN DELETIONS. A `GpuCommand[]` dump behind an
`opaqueGeneratorId` — replaying the old generator — would pass any
byte-differential while owning nothing. Acceptance is the null differential
PLUS the structural gate (schema-only serialization: the declaration schema has
no field able to hold a generator callback) PLUS §5's named deletions — not
"the stream still generates."

---

## 2. The triplication inventory (the empirical indictment)

Measured on `main` @ `5cc50d3c` (2026-07-17), post recorded-build deletion.
Pre-campaign size vector: **srcSLOC=65651, docLOC=22396, files=189, exports=26,
envFlags=65** (`docs/weight-norm.history`).

### 2.1 The three descriptions, per op family

Kernel WGSL is listed as ✓DERIVED where schedule-state already unified it
(the body lowers from ScheduleState); the surviving triplication is the
**Dispatch × Generation** pair.

| Op family | Kernel WGSL | Dispatch (interpreter) SLOC | Generation (serializer) SLOC | Sync mechanism |
|---|---|---|---|---|
| Elementwise (direct+chunked) | ✓ derived (`ops-tile-ir` specs) | `dispatch.ts` `dispatchBinary`/`dispatchUnary`/chunked ~470 | `generateSequential`+chunked+helpers **367** | `chunkedBinaryConfig` (single-source) + mirror |
| Reductions | ✓ derived (reduction-skeleton) | `ops/reductions.ts` **979** | 6 generators (`generateFullReduction`…`generateBatchedReduction`) **538** | hand-mirror + differential |
| Matmul / matmul-adjacent | ✓ derived (matmul-skeleton) | `matmul/dispatch.ts` **1510** | `generateBareMatmul`+`generateMatmulEpilogue` **205** | "mirrors dispatchTiledMatmul" (L3482) |
| Fused / attention (core) | ✓ derived (attention-skeleton) | `fusion-dispatch.ts` 393 + `ops/fused.ts` region | `generateAttention`+`generateFused`+decomp **446** | hand-mirror + differential |
| Fused special kernels (CE, LN, RoPE, narrowBwd, unscale) | ✓ derived | `ops/fused.ts` **1013** (region) | 7 generators **505** | hand-mirror |
| Optimizer / Adam | ✓ derived (adam-skeleton) | `op-dispatch.ts` `executeAdamStep` + `ops/fused.ts` `adamStepBatch` | `generateAdamBatch` **202** | "mirrors planPackedGroups" (L4020) |
| Views / copy / cat / arange / data-source | n/a (DMA/host) | `ops/views.ts` **934**, `ops/gather-scatter.ts` `cat` | `generateDataSource`+`generateCat`+`planContigCopy` **248** | "mirrors byte-for-byte" (L1170) |
| Scatter / gather DMA | ✓/DMA | `ops/gather-scatter.ts` **588**, `ops/strided-scatter.ts` **313** | `generateScatterAdd`+`generateGather`+`generateScatterCopyDMA` **169** | "mirror stridedScatterImpl exactly" (L3263) |
| Contiguity prologue (cross-cutting) | n/a | `asContiguous`/`ensureContiguous` at ~dozen sites | `resolveContiguousOperand` (+helpers) **181** | `CONTIGUOUS_OPERANDS` (single-source, 67) |

**stream-generate.ts totals:** 4307 SLOC, ~30 per-op generator functions, ONE
top-level dispatcher (`generateStream`, 444 SLOC). It emits a `GeneratedStream`
of `GpuCommand[]` under 7 tags (`TAG_ALLOC/DISPATCH/COPY/WRITE/BARRIER/CLEAR/
UNIFORM`) over slot kinds `external | arena | persistent | params`.

**The confession:** `grep -c "mirror|must match|byte-for-byte|exactly"`
`stream-generate.ts` = **36**; ~40 more across `executor.ts`. Every non-precedent
family carries a comment naming the dispatch function it hand-copies. The
enforcement is a runtime differential (`stream-diff.ts` `canonicalizeStream`,
309 SLOC) whose invariant is literally "generated and recorded streams for the
same plan must match" — a two-sources guard, not a single source.

### 2.2 The typed boundary already exists (47 bail codes)

`stream-generate.ts` refuses to serialize an uncovered op by RETURNING a bail
code string (never throws) → the plan stays lowered forever. There are **47
distinct bail codes** (`arity`, `scalar-no-table`, `chunked-noncontig`,
`undeclared-contiguous`, `config-missing`, `ksplit-temp-missing`,
`packed-buffers-missing`, `no-cached-plan`, …). This is the **incomplete
boundary** the campaign completes: today each bail is a hand-placed
"generation can't express this" guard; post-campaign each is either COVERED (a
declaration exists) or a named **authored command-stream** (the deliberately-
uncovered set §5.3), with no third state.

### 2.3 The two-sources sin's measured cost (cite these)

- **The stale-128 MB-constant incident (D4 attempts #9–#10).** Chunked
  elementwise silently mis-split because the generation side carried a HARDCODED
  128 MB threshold while dispatch read `maxStorageBufferBindingSize` from the
  device. The census mislabeled `fused[oversized]×8` from the stale constant.
  Fixed only by making `computeChunkGeometry` the SINGLE SOURCE both paths call
  — the derived-chunking seam. `stream-generate.ts:3760` now carries the scar:
  *"NOT a hardcoded 128 MB — on a device whose maxStorageBufferBindingSize…"*.
  This is the two-sources sin exactly: two descriptions of one split, agreeing
  in the test that ran, diverging on the device that didn't.
- **The D4 coverage errands (#7–#13).** Thirteen attempts to delete the recorded
  build. Each of #7–#13 was, at root, closing a place where the serializer had
  not yet been hand-taught a command sequence the interpreter already knew:
  all-dims sum (#7), arange (#7), batch>1 (#7), chunked-elementwise (#9–#10),
  released strided/broadcast views + scalar-source scatter (#13). The recorded
  build could not be deleted until each was hand-mirrored and differential-
  proven byte-identical. **A single declaration would have made all thirteen a
  no-op** — the serializer would derive what the interpreter derives, because
  they read the same object.
- **The route-decision distribution risk (schedule-state P1 note, task #95).**
  The matmul route (tiled vs GEMV, incl. the #95 inputCast axis) is selected at
  `selectMatmulChoice` and re-run INDEPENDENTLY by lowered dispatch, the
  build-from-IR capture, and `generateBareMatmul`/`generateMatmulEpilogue`. They
  agree today (`test/gemv-generated-route.spec.ts`) but this is the same
  distributed-decision shape that let the #93 int8 bypass ship. An
  ExecutionDeclaration makes the route a property of the declaration, selected
  once, consumed by all paths.

### 2.4 The precedents that prove the maneuver is achievable

| Precedent | File:line | One source → two consumers |
|---|---|---|
| `CONTIGUOUS_OPERANDS` | `contiguous-operands.ts:44` | table → generator prologue (`resolveContiguousOperand`) + dispatch `asContiguous` (asserted to agree) |
| `chunkedBinaryConfig` | `dispatch.ts:381` | one `{spec,uniforms,chunking}` → `binaryChunked` (exec) + `planChunkedBinary`→`generateChunkedBinary` (gen) |
| `computeChunkGeometry` | `tile-dispatch.ts:281` | one splitter → `dispatchChunked` (exec) + `planChunked` (gen) — GENUINE single-source (both in one object) |
| `computeBindingOrder` | `tile-dispatch.ts:243` | one binding-order → `plan()` + `computeChunkGeometry` |

These are four *point* single-sourcings, each bolted on to close one coverage
errand. The campaign is their generalization: instead of N points, ONE
declaration per family.

---

## 3. The declaration schema (and its relationship to ScheduleState)

### 3.1 What a command-stream declaration IS

An `ExecutionDeclaration` describes how one op-family NODE decomposes into an
ordered sequence of GPU commands. Its parts (each already exists, scattered):

```
ExecutionDeclaration = {
  family:     OpFamilyUid                 // elementwise | reduction | matmul | fused | adam | dma | data-source
  operands:   OperandRoleSpec[]           // per input: role (bind | scalar | offset-fold | contiguity-required)
  layout:     LayoutRequirement           // = CONTIGUOUS_OPERANDS generalized (which operands raw-bindable)
  decompose:  DecompositionRule           // = computeChunkGeometry / prologue synthesis — DERIVED, not stored
  skeleton:   CommandSkeleton             // the ordered command shape (below), over ROLES not buffers
  config:     ConfigPackingSpec           // uniform/params packing; volatile vs frozen (the TAG_UNIFORM packer)
  route?:     RouteSelector               // e.g. matmul tiled-vs-GEMV — selected ONCE (the #95 fix)
}

CommandSkeleton = ordered list of:
  | { alloc: SlotRole, bytesFrom: SizeExpr }                        // → TAG_ALLOC
  | { dispatch: KernelRef, bindings: SlotRole[], grid: GridFrom }   // → TAG_DISPATCH (KernelRef = a ScheduleState realization)
  | { copy: {src,dst}: SlotRole, bytesFrom }                        // → TAG_COPY
  | { clear: SlotRole }                                             // → TAG_CLEAR
  | { write: SlotRole, from: DataSourceNode }                       // → TAG_WRITE
  | { uniform: SlotRole, pack: ConfigRef }                          // → TAG_UNIFORM
  | { barrier }                                                     // → TAG_BARRIER
```

- **Roles, not buffers.** The skeleton names `SlotRole` (`input[i]`, `output`,
  `scratch:name`, `params`), never a `GPUBuffer`. Buffer binding is the
  realizer's job (§4): the interpreter binds live pooled buffers; the serializer
  emits slot indices (`external | arena | persistent | params`). This is exactly
  the abstraction `stream-diff.ts canonicalizeStream` already imposes to make two
  recordings serialize identically — the campaign PROMOTES that canonical form
  from a diff-time projection to the source of truth.
- **Derived geometry is DERIVED, not stored** (the loop-nest-VIEW analogue from
  schedule-state S2). `decompose` is a *rule* — the chunk split and the
  contiguity-prologue synthesis are computed by the shared splitter
  (`computeChunkGeometry`) and the shared synthesizer (`planContigCopy` gated by
  `LayoutRequirement`), never enumerated in the declaration. A device with a
  different `maxStorageBufferBindingSize` re-derives; nothing is baked (the
  stale-128 MB scar, structurally prevented).
- **Config packing is single-sourced** by the kernel's `volatilePack`
  (`tile-dispatch.ts`) — the declaration references it, so per-step-varying
  values (Adam `step_size`, GradScaler `inv_scale`, scheduled LR) flow as DATA,
  never frozen (the frozen-step_size class, structurally prevented).

### 3.2 Ruling: a NEW peer stratum, NOT a fourth ScheduleState tier

**The crux.** ScheduleState (schedule-state-design S1) is the **intra-kernel**
stratum: one dispatch's semantic schedule / backend requests / realization
receipts, producing ONE kernel's WGSL body + its dispatch geometry receipt.
ExecutionDeclaration is **one altitude up**: the **per-op-family command
sequence** that COMPOSES kernels (each `KernelRef` is a ScheduleState
realization) with the buffer/chunk/prologue/config decisions ScheduleState does
not carry.

Three candidate relationships were considered:

- **(a) A fourth ScheduleState tier** — REJECTED. It would put command-sequence
  facts (multi-dispatch decomposition, intra-plan copies, slot roles) inside an
  object whose identity is scoped to ONE kernel. That violates schedule-state's
  §2.6 leveling invariant (every fact at one level) and imposes a schema delta
  on S1 for facts that are not intra-kernel. **Zero-schema-delta on ScheduleState
  is a hard constraint** — S1 stays clean.
- **(b) Fully derivable FROM ScheduleState + operand metadata** — REJECTED as
  incomplete. Some command-stream facts genuinely are not intra-kernel and are
  not derivable from any single kernel's schedule: the >128 MB N-way split
  (a plan/device fact), scatterAdd's `a→out` intra-plan copy (TAG_COPY, a
  multi-node fact), `zeros()`→TAG_CLEAR, cat's per-input copies, the adam-batch
  packing across N params. These are *composition* facts.
- **(c) A NEW peer stratum that DERIVES per-dispatch facts from ScheduleState
  receipts and ADDS the composition facts** — RULED. This mirrors the S1
  three-tier shape one altitude up: the declaration's `skeleton` is its
  *semantic* part (the ordered command shape, backend-neutral), `decompose`/
  `layout` are DERIVED facts (the analogue of S2's derived loop-nest VIEW —
  computed by shared splitters, owning nothing), and the realized `GpuCommand[]`
  / live encode are the two *receipts* (the serialized vs interpreted forms).
  It **reads, never re-owns**, the per-dispatch facts ScheduleState already
  owns: `KernelRef` resolves a ScheduleState → WGSL + workgroup-geometry receipt;
  `grid: GridFrom` reads that receipt; binding order reads `computeBindingOrder`.

**Conclusion (schema ruling):** ExecutionDeclaration is a **new peer stratum at
the command-stream altitude**, not a tier of ScheduleState and not fully derived
from it. It composes ScheduleState receipts (no re-ownership) and adds the
composition facts (slot roles, decomposition rule, intra-plan copy/clear/write,
config packing). The four shipped mini-precedents are **seed fragments of this
schema already in-tree**: `CONTIGUOUS_OPERANDS` = `layout`; `computeChunkGeometry`/
`chunkedBinaryConfig` = `decompose`; `computeBindingOrder` = part of `skeleton`
binding order; the `GpuCommand`/`Slot` types = the serialized receipt. The
campaign does not invent the schema — it FACTORS the schema that already exists
in scattered, per-family-duplicated form.

### 3.3 The no-second-owner discipline (inherited)

Following schedule-state §12 and the ownership-derivation precedent: after a
family cuts over, the interpreter and serializer hold NO independent copy of a
command-stream fact — both read the declaration. The executable check
(§7 transitional mode) is `canonicalize(interpret(decl)) == canonicalize(
serialize(decl))` at the command level (`stream-diff.ts`), plus a schema gate
that the declaration cannot store a generator/callback/opaque-stream leaf.

---

## 4. The two consumers' contracts

### 4.1 Dispatch-as-INTERPRETER (the oldest code — migrate per-family)

`src/backend/webgpu/index.ts`, `dispatch.ts`, `ops/*.ts`, `matmul/dispatch.ts`,
`fusion-dispatch.ts` — the live/lowered path, the oldest code in the repo. The
contract: dispatch becomes a **walker of the declaration** — for each command in
the derived `CommandSkeleton`, it binds live pooled buffers to the slot roles,
resolves the derived geometry against the live device, and encodes the pass on
the shared encoder. It stops making command-sequence decisions locally.

**Migration story (the layout-campaign template, assert-agreement during
transition).** Per family, incrementally:
1. Author the family's `ExecutionDeclaration` (extract from the existing
   dispatch + generator, which already agree per the differential).
2. Route the SERIALIZER through it first (generation is younger, smaller,
   already funnels through `generateStream`) — replace the per-op generator with
   the one serializer walking the declaration. Gate: `stream-diff` null
   differential vs the retained generator (should-never-fire).
3. Route the INTERPRETER through it: replace the hand-written dispatch's
   command-sequence decisions with the same walker. Gate: assert-agreement —
   the walker's encode order must match the retained hand-dispatch's actual
   encode order (a transitional differential, deleted when the hand-path is).
4. Delete the hand-written second description.

The interpreter keeps the genuinely-dynamic decisions §8 enumerates (pool
acquisition, fence gating, autotune measurement) — those are NOT in the
declaration.

### 4.2 Generation-as-SERIALIZER (what replaces the per-op generators)

`generateStream` (444 SLOC) + ~30 per-op generators (`generateSequential`,
`generateBareMatmul`, `generateAttention`, `generateAdamBatch`, …) are REPLACED
by ONE serializer that walks any `ExecutionDeclaration` and emits `GpuCommand[]`
over the canonical slot model. The 47 bail codes collapse: a family either has a
declaration (covered) or is an authored command-stream (§5.3). `generateStream`
shrinks to the walker + the covered/uncovered census.

---

## 5. What retires

### 5.1 Deleted mechanism (named per house policy)

- **The ~30 per-op generator functions** in `stream-generate.ts` (~3400 of its
  4307 SLOC — everything except the top-level census/orchestration and shared
  helpers), replaced by ONE declaration-walker serializer (~300 SLOC).
- **The per-family hand-written command-sequence logic in dispatch** — the parts
  of `ops/reductions.ts`, `matmul/dispatch.ts`, `fusion-dispatch.ts`,
  `ops/fused.ts`, `ops/gather-scatter.ts` that decide command ORDER / slot roles
  / intra-plan copies (NOT the kernel-binding or pool logic). Estimated
  ~800–1400 SLOC of decision code reduced to declaration reads.
- **The 36 "mirrors X" comments** — the confession disappears with the second
  description.

### 5.2 Decayed to should-never-fire

- **`stream-diff.ts canonicalizeStream` (309 SLOC)** — its LOAD-BEARING role
  (catch serializer/interpreter drift) decays: with one source, drift is
  unconstructible. It survives ONLY as (a) the phase-transition assert-agreement
  gate and (b) the build-twice determinism gate (`test:gates`, already its
  post-#43 successor role). It becomes a should-never-fire structural check, not
  a safety net the campaign leans on.
- **The 47 bail codes** become the COMPLETE typed boundary: covered ⊕ authored,
  no third state.

### 5.3 The deliberately-uncovered set becomes NAMED authored command-streams

Today's "lowered forever, correct-and-slow" classes (released strided/broadcast
views, released-input scatterAdd/CE, non-f32 scalar scatter, batch>64/non-f32/
CoW — stage-4 D4 #13) stop being silent bail returns and become explicit
**authored `ExecutionDeclaration`s** with `skeleton.visibility:"opaque"` + a
refusal reason (the schedule-state F3 pattern). They still run lowered — but the
boundary is DECLARED and typed, not a scattered `return "chunked-noncontig"`.

### 5.4 Net SLOC target

Pre-campaign baseline: **srcSLOC 65651** (@`5cc50d3c`). Deletion pool ≈ 3400
(generators) + ~1000 (dispatch decision code) ≈ **−4400**; additions ≈ one
serializer walker (~300) + one interpreter walker (~300) + per-family
declaration tables (~600) + transitional assert-agreement (temporary, deleted
per-phase) ≈ **+1200**. **Net target: ≈ −2000 to −3000 srcSLOC**, landing the
framework at **≤ ~63000 srcSLOC** — below the pre-campaign size.

Per the schedule-state §8 gate-5 ruling, this number is DIRECTIONAL, not a
gameable hard threshold: the acceptance is (net-negative) AND (structural
gate: schema cannot hold a generator) AND (named deletions per phase). Growth
without deletions triggers a design re-review, not a paragraph. This is
positioned to be **the campaign that dips the framework below its pre-campaign
size** — the schedule-state + execution-declaration pair together delete both
remaining hand-written descriptions of op execution.

---

## 6. The realizer connection

Schedule-state (§4–§5) established: WGSL/Dawn = realizer #1 of a ScheduleState;
Triton = realizer #2 (kernel altitude; cross-backend differential 0.0); each
realizer has a capability profile + typed refusal; identity is
{semantic, compilation, artifact-cache} coordinates.

ExecutionDeclaration extends the same story one altitude up:

- **The declaration's realizers are the two forms of the SAME backend.** WGSL
  dispatch = the INTERPRETER realizer (encode live); the `GpuCommand[]`
  serializer = the SERIALIZED realizer (replay). They are not two backends —
  they are two realizations of one declaration, which is exactly why the mirror
  was pure waste. Their agreement is the artifact-digest story at command
  altitude: `canonicalizeStream` IS the command-stream artifact digest (abstract
  slot/pipeline identities), and it already exists.
- **A `KernelRef` in the skeleton carries a RealizerCoordinate.** When Triton
  becomes ScheduleState realizer #2, an ExecutionDeclaration whose `dispatch`
  commands reference Triton-realized kernels composes unchanged — the command
  sequence (allocs, copies, chunk splits, config packing) is realizer-neutral;
  only the `KernelRef` resolution differs. This is how command-stream
  declarations extend the RealizerCoordinate/artifact-digest story: they are the
  composition layer above the per-kernel realizer choice.
- **The editor consumes the same objects.** Sol's islands editor (schedule-state
  §9, P3) shows island → ScheduleState (intra-kernel). The ExecutionDeclaration
  is the NEXT-outward view: "what command sequence realizes this op" — the
  allocs, the chunk split, the prologue copies — rendered from the same
  declaration the interpreter runs. One object, three consumers (interpreter,
  serializer, editor), per the house spine.

---

## 7. Phased plan (each shippable behind differentials)

Per-family cutover, easiest-first (fewest command-stream facts → most). Each
phase: author the declaration, cut the serializer over (null differential vs
retained generator), cut the interpreter over (assert-agreement vs retained
dispatch), delete both hand descriptions. `npm run build` + full suite +
`test:gates` green at every boundary.

### P0 — the schema + the elementwise walking skeleton
- Deliverables: the `ExecutionDeclaration` schema (`src/executor/execution-
  declaration.ts`), the ONE declaration-walker serializer, the schema gate (no
  generator/callback/WGSL leaf serializable). Cut over ELEMENTWISE
  (direct + chunked) — it already has the most single-sourcing
  (`chunkedBinaryConfig`, `CONTIGUOUS_OPERANDS`), so it is the lowest-risk proof.
- Acceptance (pre-registered): `stream-diff` null differential over the
  elementwise corpus on BOTH plan paths; build-twice determinism green;
  `parity-fullstack-tl` compiled==lowered ≤ 1e-5 / 30 steps; distil@512 step
  time within noise of the pre-campaign 50 ms baseline; net SLOC for the phase
  reported (generators deleted − walker added).

#### P0 STATUS — LANDED (serializer cutover complete; interpreter boundary named)

Three commits off `main` @ `7b31fa6a` (branch `worktree-agent-a43a9fb9caae184b0`,
unpushed):
- **c1 — the schema** (`src/executor/execution-declaration.ts`): the
  `ExecutionDeclaration` types + the elementwise family authored as data
  (one entry per op, absorbing the per-op classification), + the STRUCTURAL
  schema gate `assertNoGeneratorLeaf` (no leaf may be a function/callback/buffer —
  the adapter-cheat is unconstructible). CPU gate `test/execution-declaration.spec.ts`
  7/7. Zero behaviour change.
- **c2 — serializer cutover**: `generateSequential`'s per-op inline emission
  (binary/unary/cast/where/contiguous/gelu direct + chunked) → one
  declaration-walker (`serializeElementwiseDirect`) + KernelRef resolver
  (`resolveElementwiseKernel`) + transport-windowing transform
  (`serializeElementwiseSizeSplit`). A transitional in-code shadow byte-diffed the
  walker against the retained legacy emitter on every generated plan (throw on
  divergence).
- **c3 — shadow + legacy deleted**: the wall proved green with the shadow live, so
  the legacy emitter + shadow are removed; the declaration is the sole source.

**Vin's chunking refinement (adopted, supersedes §3.1's "derived geometry carried
in the declaration").** Decomposition is NOT a per-declaration/per-family field.
It is TRANSPORT WINDOWING: when the oversized operand's split axis is PARALLEL
(elementwise = every axis), windowing is a universal WALKER transform
`windows = f(bytes, binding limit, 256-B align)`, applied identically by both
consumers; the declaration never mentions it. Parallelism is DERIVED
(`ELEMENTWISE_SPLIT_AXIS_PARALLEL` — the smallest honest P0 interim, sourced from
the schedule side, until schedule-state receipts classify parallel-vs-carried
axes). Oversized on a CARRIED axis (reductions) is a typed walker refusal —
schedule-state's territory, NOT this stratum's. The `decompose` schema field was
deleted. Full dissolution of `chunkedBinaryConfig`/`planChunkedBinary`/`planChunkedUnary`
into ONE index-map windowing transform (co-partitioning co-bound operands) is P1
work — it touches the interpreter's chunked dispatch (hot path), out of P0's
serializer scope.

**Deletions (named):** private `BINARY_OPS`/`UNARY_OPS` in `stream-generate.ts`
(→ `ELEMENTWISE_BINARY_WGSL`/`ELEMENTWISE_UNARY_OPS` in the declaration); the per-op
plan-build switch + inline ALLOC/UNIFORM/DISPATCH emission; the `decompose`
schema field; the transitional shadow + legacy emitter.

**Net SLOC (P0, directional per §5.4):** NET-POSITIVE, as designed — the
schema+walker land before P1–P4's family-generator deletions. `stream-generate.ts`
shrinks (classification tables + inline emission gone) but the new declaration file
(~330 SLOC) + walker/resolver/transform (~140 SLOC) exceed the elementwise-only
deletion. The campaign dips below the pre-campaign size only after the bigger
families (reductions 979+538, matmul 1510+205, fused 1013+505) cut over.

**Gates (measured, degraded host ~4×):** `test:gates` 5/5 with the shadow ACTIVE
(walker == legacy byte-identical), then 5/5 AGAIN after the shadow deletion (walker
stands alone) — compiled==lowered trajectory, build-twice determinism, chunked
full-reduction (>128 MB), view-offsets-as-data (the TAG_UNIFORM repack), cross-offset
replay. `parity-fullstack-tl` BOTH directions: max |Δloss| 1.1e-5 / 30 steps at the
run-to-run noise floor (two compiled runs of byte-identical streams differ by
5.7e-6; step-29 endpoints agree to all printed digits — the documented "~1e-5"
criterion). Full ledger: train-tape-matrix 4/4 zero refusals; step-object-null +
step-edit-null pass; witness matrix 5/5 cells `shadowEmpty:true`; ring-probe
bit-identical; ledger-attack default+48 pass; 124M DiLoCo regression EXACT
({0:9.81, 3:5.92, 6:5.15, 9:4.64}, peak flat 2087.6 MB); profile distil@512
late-step ≈54 ms (fwd 22/bwd 20/opt 1/cleanup 10; baseline 50), medium@512
late-step ≈200 ms (baseline 190) — the interpreter is UNTOUCHED by P0 (generation
is build-time-only), so steady-state ms/step is structurally unchanged; measured
values sit on baseline within degraded-host noise. Full suites green (all failures
were worktree-environmental — missing `.venv`/`models/` symlinks + 3 transient
`vkCreateDevice` — and pass on isolated rerun: cpu 44/44 oracle, webgpu 18/18
files 138/138 tests). `tsc` 134 (baseline 135, −1 dead code); build green; biome
net −1 error.

**Interpreter cutover — the named P0-continuation boundary.** The serializer now
derives from the declaration; the INTERPRETER (`dispatchBinary`/`dispatchUnary` →
`dispatchElementwise`) still hand-encodes, but it already realizes the SAME
skeleton (ALLOC via `resolveOutputBuffer` → params → DISPATCH bind=[…inputs, out,
params]) and is held in agreement by the record-vs-generate differential + the
compiled==lowered parity gate — NOT a stranded mirror. Its literal cutover is:
unify `dispatchElementwise`'s live-encode walk with `serializeElementwiseDirect`'s
command-emit walk under ONE skeleton walker + TWO emitters (interpreter binds live
pooled buffers; serializer pushes `GpuCommand[]`). It requires plumbing op identity
to the dispatch seam (today `dispatchBinary` receives the WGSL symbol, not
`node.op`, so the declaration is not reachable there) — deferred to keep the hot,
pool-entangled path untouched under the degraded verification window.

### P1 — reductions
- Cut over the 6 reduction generators (538 SLOC) + `ops/reductions.ts` decision
  code (979 SLOC). Reductions carry the all-dims / row-program / batched /
  chunked-full-reduction fan-out that ate D4 #7 and #9–#10 — the single
  declaration must reproduce every one from `decompose` alone.
- Acceptance: the chunked full-reduction (>128 MB) gate (`test:gates` #4) passes
  from the declaration's derived split, NOT a hand-mirror; all-dims sum + arange
  (the #7 errands) covered by construction.

### P2 — matmul-adjacent + the route-as-declaration fix
- Cut over `generateBareMatmul`/`generateMatmulEpilogue` (205) + the command-
  sequence parts of `matmul/dispatch.ts` (1510). Fold the tiled-vs-GEMV route
  (incl. #95 inputCast) into `RouteSelector`, selected ONCE, consumed by lowered
  / capture / serialize / tape-replay (the schedule-state P1 R9 single-selection
  ruling, at command altitude).
- Acceptance: `test/gemv-generated-route.spec.ts` passes with the route read from
  the declaration; K-split / epilogue chains reproduced from `decompose`;
  medium@512 step time within noise of the 190 ms baseline.

### P3 — fused / attention / adam
- Cut over `generateAttention`/`generateFused`/`generateAdamBatch` + the 7 fused
  special-kernel generators (505) + adam-batch packing. Attention and Adam are
  the schedule-state authored-set members; here their COMMAND STREAM (not the
  body) declares. Adam-batch packing across N params exercises the composition-
  fact half of the schema hardest.
- Acceptance: the volatile-uniform gate (Adam step_size flows as data from the
  declaration's `ConfigPackingSpec`); 124M DiLoCo regression loss baselines
  {0:9.81, 3:5.92, 6:5.15, 9:4.64} exact; profiler peaks byte-identical.

### P4 — views/dma/data-source + the boundary closure
- Cut over `generateDataSource`/`generateCat`/`planContigCopy` + scatter/gather.
  Convert the deliberately-uncovered set (§5.3) to named authored declarations.
  DELETE the 47 ad-hoc bail codes; the boundary is now covered ⊕ authored.
- Acceptance: `stream-diff` decays to should-never-fire (retained only as the
  determinism gate); the census reports covered/authored, no "uncovered";
  campaign-end weight-norm below pre-campaign 65651.

---

## 8. Risks (honest)

- **Oldest-code refactor risk.** `src/backend/webgpu/index.ts` + `ops/*.ts` +
  `dispatch.ts` are the repo's oldest, most load-bearing code, entangled with the
  buffer pool, shared encoder, and fence gating (every "used in submit while
  destroyed" / stale-buffer bug lived here). Mitigation: per-family increments;
  the interpreter cutover is assert-agreement-gated against the retained hand
  path, never a big-bang; the pool/encoder/fence logic is NOT in the declaration
  (§8 genuinely-dynamic) so it is untouched.
- **The transition carries FOUR descriptions temporarily** (kernel-WGSL ✓done;
  hand-dispatch; hand-generator; new declaration) — the D2b "mechanisms without
  consumers rot" lesson. Mitigation: phase discipline — a family's declaration
  lands WITH both cutovers in the same phase; no declaration ships without both
  consumers wired; no hand path survives its phase. No stranded halves.
- **The differential decays from load-bearing to should-never-fire — the moment
  it stops catching real drift is the moment a real bug could slip.** Mitigation:
  the schema gate (no serializable generator) makes the adapter-cheat
  unconstructible, so "should-never-fire" is structural, not hopeful; keep the
  build-twice determinism gate live permanently.
- **What CANNOT be declared (enumerate — the genuinely-dynamic dispatch
  decisions):**
  1. **Pool acquisition / buffer identity** — which physical `GPUBuffer` backs a
     slot is decided live by `bufferPool.acquire`/`canRecycle` under in-flight
     encoder claims. The declaration names ROLES; buffer identity stays dynamic.
  2. **Fence-gated destruction / deferred-destroy timing** — mid-step memory
     pressure, `deferredDestroy` after next fence. Not a command-sequence fact.
  3. **Autotune measurement** — `matmul/dispatch.ts` `autotuneIfNeeded` measures
     on-device and picks a tile config. The SELECTION becomes a `SelectionReceipt`
     (schedule-state P1); the measurement act is not declarable.
  4. **First-execution-lowered / witness-to-convergence** (stage-4 K_w=2) —
     WHICH execution cuts over is a runtime state-machine fact, not per-op.
  5. **The deliberately-uncovered classes** — released strided/broadcast views,
     CoW, batch>64 — become authored declarations (§5.3), explicitly NOT derived
     command streams. Their fate: declared-opaque, run lowered forever, correct-
     and-slow by construction.

---

## 9. Red-team (3 strongest objections + rulings)

**R1 — "This is just moving stream-generate.ts's per-op switch into a data table;
the same 4307 SLOC of family-specific knowledge has to live somewhere. You net
nothing but indirection."**
RULING: partially conceded, and it is the load-bearing risk. The win is NOT that
family knowledge vanishes — it is that it lives ONCE instead of twice (dispatch
AND generation). The measured duplication is the two columns of §2.1: reductions
are 979 (dispatch) + 538 (generation), matmul 1510 + 205, fused 1013 + 505. A
declaration collapses each PAIR to one description + two thin walkers. If the net
SLOC does not go negative (§5.4), the campaign has FAILED its own gate and is
reverted — the schema gate + weight-norm make that measurable, not arguable. The
indirection objection is answered by the walker being ~600 SLOC total for ALL
families vs ~4300 of per-family generators.

**R2 — "Schedule-state already claims to own the kernel; you are drawing an
altitude line between 'kernel body' and 'command sequence' that will leak. Chunk
splitting reads device limits AND affects the kernel's uniforms — which stratum
owns it?"**
RULING: the line is real and already drawn in-tree — `computeChunkGeometry`
(command-stream: how many dispatches, what byte windows) is SEPARATE from the
tile-IR spec (kernel: the per-element body). The chunk split owns the
DECOMPOSITION (a command-stream fact: N TAG_DISPATCH commands); the kernel owns
the per-invocation arithmetic (unchanged across chunks — same WGSL, different
sub-range binding). The `sizeUniform` patched per chunk is `config` (command-
stream, `ConfigPackingSpec`), packed by the kernel's `volatilePack` (kernel-
owned packer, command-stream-owned VALUES). No leak: the seam is exactly where
`computeChunkGeometry` already cuts. Zero-schema-delta on ScheduleState (§3.2) is
the enforcement — if a fact needs to be added to S1 to make this work, the line
leaked and the design is wrong.

**R3 — "The recorded build was just deleted (D4 #13, −822 SLOC, thirteen
attempts). You are proposing to re-plumb the exact machinery that took thirteen
attempts to stabilize. Why won't this destabilize the sole compiler?"**
RULING: the opposite — this is what the thirteen attempts were reaching for. Each
D4 errand (#7–#13) was a hand-mirror of a command sequence; the reason there were
THIRTEEN is that every uncovered family had to be hand-taught the serializer
separately, differential-by-differential. The declaration makes that class of
errand structurally impossible: a covered family's serializer and interpreter
CANNOT diverge because they read one object. The risk is the migration, not the
end state — and the migration is gated per-family by the SAME differential
(`stream-diff`) that guarded all thirteen attempts, now used to prove
equivalence during cutover rather than to chase drift forever. We are not
re-plumbing the recorded build (deleted); we are factoring the SOLE compiler's
back end (`stream-generate.ts`) against its front end (dispatch).

---

## Open questions for Vin (only where it materially forks)

1. **Altitude naming.** "ExecutionDeclaration" vs folding into the schedule-state
   charter as a named second stratum ("CommandSchedule"?). Forks the doc
   structure and the editor's zoom model (one more zoom level, or a sibling
   object). Does this get its own charter or live under schedule-state's spine?
2. **Phase ordering vs schedule-state P0.** Schedule-state P0 (semantic-IR
   boundary) and this P0 (command-stream schema) both touch the dispatch seam.
   Sequence them (schedule-state P0 first, so `KernelRef` resolves a real
   ScheduleState) or interleave (this campaign references today's kernel cache as
   an opaque `KernelRef`, cutting over independently)? The latter unblocks this
   campaign now; the former is cleaner. Recommendation: interleave — `KernelRef`
   is opaque to this stratum by design (§6), so it does not block on
   schedule-state P0.

---

## One-sentence test

If an op family's command sequence cannot be stated as one `ExecutionDeclaration`
that both the live dispatch interprets and the compiled plan serializes — with no
second hand-written description of that sequence anywhere — it is not yet
declared; reshape it (or mark it a named authored command-stream) before landing.
