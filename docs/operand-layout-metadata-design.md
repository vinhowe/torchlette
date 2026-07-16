# Operand Layout Requirements as Declared Metadata

**Campaign #99 fork (a): kill the `[non-contiguous]` uncovered class as a CLASS.**

## The sin (single-source-at-seams, layout axis)

Two consumers hand-maintain the same knowledge — "which operand of an op must be
laid out contiguously (raw-bindable: contiguous strides AND element offset 0)":

1. **The dispatch layer** (`src/backend/webgpu/`): every kernel wrapper prologues its
   contiguity-requiring operands through `ensureContiguous` / `asContiguous` /
   `contiguous()` before binding the buffer flat-from-element-0. This is the
   recorded path's IMPLICIT normalization — a real contiguous-copy dispatch that
   the recorder observes.
2. **The stream generator** (`src/executor/stream-generate.ts`): each per-op
   generator re-encodes the SAME per-operand requirement as a `[non-contiguous]`
   BAIL — "the real dispatch inserts a copy I don't reproduce, so give up (keep
   record/replay)".

When these diverge you get silent under-coverage: the generator bails on an op the
dispatcher would have normalized fine, and that op's whole plan stays on the recorded
compiled path — never reaching `registerResultEntries` / stage-3B liveness release.
That is exactly the #99 pin: the checkpoint config's `fusedCrossEntropyForward`
(`0xbd0dd584`) and `fusedCrossEntropyBackward` (`0x19e72088`) templates each have
ONE build-from-IR-uncovered op — the CE bail on a row-padded logits operand
(`shape=[512,50257] strides=[50304,1] offset=0`, a live buffer at the 128 MB binding
limit) — holding ~1915 MB of planner-registry `result` pinning hostage.

## The indictment table (dispatch side × generator side)

Every op below forces one or more INPUT operands raw-bindable at DISPATCH; the
generator side either BAILS `[non-contiguous]` (the class this campaign closes) or
already SYNTHESIZES the copy from a lowering capture (attention/reshape — the
precedent we generalize). Requirement is always **raw-bindable** unless noted.

| op | contiguous operand(s) | dispatch mechanism (file) | generator side (pre-campaign) |
|---|---|---|---|
| `fusedCrossEntropyForward` | logits(0) | `asContiguous` fused.ts:443 | BAIL `resolveContiguousInputSlot` |
| `fusedCrossEntropyBackward` | logits(0), gradOutput(2) | `asContiguous` fused.ts:464/467 | BAIL `resolveContiguousInputSlot` |
| `sum` (full reduction) | input(0) | `contiguous` reductions.ts:237 | BAIL L1313 |
| dim reduction | input(0) | `contiguous` reductions.ts:237 | BAIL L1470 |
| `mean` | input(0) | `contiguous` reductions.ts:237 | BAIL L1545 |
| `batchedReduction` | each input | `contiguous` reductions.ts:893 | BAIL L1689 (per node) |
| `unscaleGrad` | grad_in(0) | `asContiguous` fused.ts:401 | BAIL L2634 |
| fused elementwise kernel | each recipe input | `asContiguous` (fusion-dispatch) | BAIL L3089 |
| `fusedLayerNormForward` | x(0),weight(1),bias(2) | `asContiguous` fused.ts:505-507 | ASSERT via captured `inputContig` (attention-style) — none in covered traces |
| `fusedAttentionForward/Backward` | q/k/v/lse/dO/output | `asContiguous` fused.ts:670.. | SYNTHESIZE via `cachedInputContig` (lowering capture) |
| `reshape` (materializing) | input(0) | `contiguous` views.ts:281 | SYNTHESIZE via `cachedViewInput` (lowering capture) |
| `gather`/`scatterAdd` | table/index/src | `ensureContiguous` gather-scatter.ts:115.. | (kernel-op generators; operands contiguous by construction in covered traces) |
| `narrowBackward` | grad(0) | `ensureContiguous` views.ts:843 | SYNTHESIZE-free (grad contiguous by construction) |

Distinct co-located coercions (NOT contiguity, out of scope): CE targets→i32
(`ensureI32Targets`), stridedScatter/topk f32-guard.

## The declaration (smallest shape the inventory justifies)

The axis the generator needs is exactly: **for op `X`, which input positions require
raw-bindable layout** — so that a strided-but-LIVE operand there gets a synthesized
contiguous-copy prologue (byte-identical to the dispatcher's `asContiguous`) instead
of a bail. That is one bit per operand position. The declaration is therefore a
single table:

```ts
// src/executor/contiguous-operands.ts
CONTIGUOUS_OPERANDS: Map<opName, "all" | number[]>
```

`"all"` = every non-scalar input is contiguity-required (CE, unscale, fused kernels,
reductions — all their bindable inputs are raw-bound). A `number[]` would name
specific positions if an op mixed raw-bound and offset-folding operands; the
inventory shows the current generator-bail ops are all-inputs, so `"all"` covers
them and `number[]` is reserved for future mixed ops.

**Rejected shapes (zero-schema-delta pressure):**
- *Per-operand alignment / derivable-view-tolerance fields.* The inventory shows the
  offset-folding ops (fusedRoPE cos/sin, cat aligned, stridedScatter src, topk) live
  ENTIRELY on the dispatch side and never reach a generator bail — the generator
  already handles them (or they're not covered). No generator consumer needs an
  alignment sub-axis, so a single contiguity bit is the whole justified shape.
- *A new op registry.* `OP_REGISTRY` (src/ops/registry.ts) is the elementwise/unary/
  binary table; the bailing ops are backend KERNEL ops keyed by `LazyIRNode.op`, not
  in it. A `Map<opName,…>` keyed by the same string the generator switches on is the
  minimal home — no registry invented (follows where quantB/StorageFormat sit: at the
  altitude the consumer already keys on).

## Which consumer derives vs asserts

- **Generator (DERIVES — load-bearing):** `resolveContiguousOperand` reads the table.
  For a declared-contiguous operand that resolves to LIVE-strided storage, it
  synthesizes the contiguous-copy prologue via the existing `planContigCopy`
  (driven from the live layout: shape/strides/offset/dtype/bufferSize — the same
  fields `AttnInputContig` carries, read straight off the live `WebGPUTensor`).
  A RELEASED strided view with no lowering capture stays bailed (its layout isn't
  recoverable post-hoc — unchanged, and not the #99 case, whose operand is live).
- **Dispatch (ASSERTS — this pass):** the dispatch-site `asContiguous`/`ensureContiguous`
  calls are left in place (refactoring 40+ call sites to derive from the table is
  out of this pass's scope and risk), but the table is asserted to AGREE with them by
  the `t-stream-generate` differential itself: the generated prologue must reproduce
  the recorded copy command-for-command, so a table entry that disagrees with what
  the dispatcher actually did surfaces as a DIVERGE (loud build-time finding), never
  a third silent copy. The declaration is thus load-bearing on the generator and
  cross-checked against the dispatcher at the recording seam.

## Deletions named

- SIX near-identical `[non-contiguous]` BAIL blocks (the per-generator
  hand-maintained contiguity knowledge) collapse into ONE call to
  `resolveContiguousOperand` per operand: full-reduction, dim-reduction, mean,
  batched-reduction, unscaleGrad, and CE fwd/bwd (`resolveContiguousInputSlot`).
- `resolveContiguousInputSlot` (the CE-specific bail helper) is subsumed by the
  general `resolveContiguousOperand`.

## Scope boundary (honest, NOT the #99 case)

The fused-elementwise-kernel recipe-input loop keeps its `[non-contiguous]` bail.
Fusion codegen resolves strided inputs via per-input strided indexing OR an
`asContiguous` depending on the recipe (fusion-dispatch), so a blind copy
synthesized here would risk diverging from the fusion dispatcher's own layout
decision — a different (recipe-shaped) axis than the single-operand raw-bind the
kernel ops above declare. Left for a follow-on that models the fusion recipe's
per-input layout decision; it is NOT on the checkpoint config's CE critical path.

## The declaration is load-bearing (not a dead table)

`resolveContiguousOperand(op, operandIndex, …)` GATES synthesis on
`operandRequiresContiguous(op, operandIndex)`: a live strided operand at a
position the table does NOT declare bails `undeclared-contiguous` (a coverage
gap the differential surfaces) rather than getting a guessed copy. So the table —
not each generator's hardcoded knowledge — is the single source deciding which
operands are forced. The generator DERIVES from it; the dispatcher is ASSERTED
against it by t-stream-generate.

## The #99 re-test (headline — MEASURED, device 10, distil@512 + selective ckpt)

With the class closed, the two checkpoint templates go GENERATED (t-stream-generate:
both FULLY GENERATED, 0 diverged, uncovered set EMPTY for CE — the +4/+5 cmds are the
synthesized copy prologue) → `registerResultEntries` fires (entry count 602→699) →
stage-3B release engages.

| metric | pre-campaign (doc §R2) | post-campaign |
|---|---|---|
| planner-registry **result** | 1919.4 MB | **1260.9 MB** (−658.5, −34%) |
| planner-registry materialized | 2833.8 MB | 2355.7 MB (−478) |
| arena-ON current | 4584.7 MB | 4106.5 MB (−478) |
| arena-ON peak | 4756.6 MB | 4278.5 MB (−478) |
| arena-free current / peak | 1798.3 / 3933.5 MB | 1798.3 / 3933.5 MB (unchanged) |
| registry entries | 602 | 699 |
| witness pruned pairs (ckpt cell) | 22 | **211** |

- **`t-planner-pin-attribution --assert` FLIPS to PASS** ("planner-registry 2355.7MB
  collapsed" — below the 2500MB threshold; pre-campaign it FAILED at 2833.8MB). This
  is the #99 headline: the attribution oracle resolved by GENERALITY.
- **`t-checkpoint-ab-oracle` MEMORY side still FAILs** (arena-ON current 4106.5MB >
  arena-free+5% = 1888.2MB). This is EXPECTED and honest: per doc/arena-recompute-
  design.md phase R2, the residual arena-ON−arena-free delta is the registry's
  cross-STEP retention of the genuinely-LIVE working set (a retention-vs-return
  accounting difference), NOT the over-allocation the pin-attribution threshold
  targets. Closing the uncovered CLASS collapsed the dead-pin share (result −658MB);
  the live working-set retention the +5% budget measures is a separate (out-of-scope)
  mechanism. The WITNESS side PASSes both before and after (arena-ON engaged).
- **Peak re-measurement**: arena-ON peak 4278.5 vs arena-free 3933.5 MB (+8.8%). The
  task's phase-3 (+10.3%, 5397 vs 4893) and R2 (~parity, 4756 vs 4789) readings were
  the profiler config (which shows peak 5397MB, unchanged by this campaign — it is not
  the A/B arena comparison); the pin-attribution A/B above is the controlled arena vs
  arena-free peak. profile-training distil@512 checkpoint: flat 5233MB current steps
  9-12 (zero leak, LEAK STATUS OK).

**What remains pinned:** the 1260.9 MB registry result IS the genuinely-live saved
activations + gradients the compiled path retains cross-step for deterministic reuse
(doc §R2's "working set"). It is not a dead pin and this campaign does not target it —
the #99 uncovered-CLASS pin (the ~658MB the CE templates held on the recorded path) is
gone.
