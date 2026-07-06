> **SUPERSEDED (2026-07-06).** The Dawn-level tape described here shipped as the
> generated-plan replay (stage-4 phases 2-4). The successor campaign — program-level
> replay one layer above plans — is `staged-execution-phase1.md`.

# Dispatch Tape Replay System

Design document for the dispatch tape replay optimization. This eliminates redundant CPU dispatch overhead by recording the GPU dispatch sequence once and replaying it on subsequent steps.

## Motivation

After Phase 2 optimizations, DistilGPT-2 training runs at ~183ms/step with ~68% of wall time spent in CPU dispatch. The dispatch sequence is deterministic across steps — we re-dispatch ~560 ops from scratch every step. The architectural fix is to record the dispatch sequence once and replay it.

This approach is used by JAX's jit/XLA, torch.compile, and CUDA Graphs.

### Current CPU Overhead Breakdown

| Source | Cost/step | % Wall |
|--------|-----------|--------|
| `createBindGroup` | 20ms | 11% |
| `queue.submit` | 11ms | 6% |
| `createBuffer` | 0.7ms | <1% |
| `writeBuffer` | 1ms | <1% |
| **Total dispatch** | ~34ms | ~19% |

Beyond Dawn API calls, there is additional CPU time in plan building, fusion detection, cache lookups, and memory planning that would also be skipped during replay.

## Data Structures

### TapeEntry

One entry per dispatched GPU operation:

```typescript
type TapeEntry = {
  // Pipeline state
  pipelineId: number;          // Index into pipeline cache
  workgroupDims: [number, number, number];

  // Bind group layout
  bindGroupLayoutId: number;   // Index into layout cache

  // Buffer slots (resolved at replay time)
  bufferSlots: TapeBufferSlot[];

  // Uniforms (small constant data)
  uniformData?: ArrayBuffer;
};

type TapeBufferSlot = {
  role: "input" | "output" | "uniform";
  index: number;               // Which input/output of this op
  offset: number;              // Byte offset into buffer
  size: number;                // Byte size of binding

  // For replay resolution:
  tensorId: number;            // Logical tensor identity
  poolSlot?: number;           // Buffer pool slot (if pool-managed)
};
```

### TapeGuard

Guards determine when a recorded tape is valid for replay:

```typescript
type TapeGuard = {
  // Structural identity (must match exactly)
  structuralKey: string;       // Hash of IR graph + scalar values (§8.2.1)

  // Input shapes/dtypes (must match exactly)
  inputSignatures: InputSignature[];

  // Buffer stability check
  paramBufferIds: number[];    // GPU buffer IDs for model params
  optimizerBufferIds: number[]; // GPU buffer IDs for optimizer states

  // Pool stability
  poolGeneration: number;      // Buffer pool generation counter
};
```

### Tape

The full recorded sequence for one step:

```typescript
type Tape = {
  guard: TapeGuard;
  entries: TapeEntry[];
  submits: TapeSubmitBoundary[];  // Where queue.submit() calls go

  // Statistics
  recordedAtStep: number;
  totalDispatches: number;
  totalSubmits: number;
};

type TapeSubmitBoundary = {
  afterEntryIndex: number;     // Submit after this entry
};
```

## Recording Flow

Recording happens as a side-effect during normal dispatch:

1. **Entry point**: At the start of a step, if no valid tape exists, set `recording = true`
2. **Per-op instrumentation**: Each dispatch call (`dispatchCompute`, `createBindGroup`, etc.) appends a `TapeEntry`
3. **Submit tracking**: Each `queue.submit()` records a `TapeSubmitBoundary`
4. **Finalization**: At end of step, package entries + guard into a `Tape`
5. **Guard capture**: Snapshot buffer identities, pool generation, structural key

Recording adds minimal overhead (array pushes + snapshots). The first step after model load already pays full dispatch cost, so recording is effectively free.

## Replay Flow

When a valid tape exists and guards pass:

1. **Guard check**: Verify structural key, input signatures, buffer identities
2. **Buffer resolution**: Map each `TapeBufferSlot` to a concrete `GPUBuffer`
   - Param/optimizer buffers: Direct lookup by `tensorId` → buffer
   - Pool-managed buffers: Acquire from pool by slot assignment
3. **Encode**: For each `TapeEntry`:
   - Create bind group from resolved buffers + cached layout
   - Set pipeline, set bind group, dispatch with recorded workgroup dims
   - (Skip all: plan building, fusion detection, cache lookup, memory planning)
4. **Submit**: At each `TapeSubmitBoundary`, call `queue.submit()`
5. **Cleanup**: Release pool buffers, update pool generation

### What Replay Skips

- Plan building and topological sort
- Fusion group detection
- IR optimization (CSE, DCE)
- Compile cache lookup
- Memory planning and lifetime analysis
- Pipeline compilation (cached)
- Bind group layout creation (cached)
- Per-op dispatch logic (buffer allocation, stride calculation, etc.)

### What Replay Still Does

- Guard validation (cheap: hash comparison + buffer identity check)
- Buffer resolution (map tensor IDs to GPU buffers)
- Bind group creation (unavoidable WebGPU API cost)
- Queue submission (unavoidable)
- `writeBuffer` for dynamic data (learning rate, step count)

## Guard System

### Invalidation Triggers

| Trigger | Detection | Response |
|---------|-----------|----------|
| Different IR structure | `structuralKey` mismatch | Discard tape, re-record |
| Different input shapes | `inputSignatures` mismatch | Discard tape, re-record |
| Param buffer reallocated | `paramBufferIds` mismatch | Discard tape, re-record |
| Pool generation changed | `poolGeneration` mismatch | Re-resolve buffer slots |
| Dynamic scalar changed | Part of `structuralKey` (§8.2.1) | Discard tape, re-record |

### Guard Check Cost

Guard validation should be O(1) amortized:
- Structural key: single string comparison
- Input signatures: length + element comparison
- Buffer IDs: array comparison (param count is fixed)
- Pool generation: single integer comparison

Estimated: <0.1ms per guard check.

## Phase Checklist

### Phase 0: Foundations (this phase)
- [x] Design document
- [x] Scalar canonicalization in compile cache keys (§8.2.1)
- **Entry criteria**: None
- **Exit criteria**: Design doc reviewed, scalar cache keys pass tests

### Phase 1: Recording Infrastructure
- [x] Add `TapeDispatchEntry`, `TapeSubmitMarker`, `DispatchTape` types (`src/backend/webgpu/dispatch-tape.ts`)
- [x] Instrument dispatch path to emit `TapeDispatchEntry` records (index.ts, fusion-dispatch.ts, matmul/dispatch.ts)
- [x] Instrument submit path to emit `TapeSubmitMarker` (flushSharedEncoder, endSharedEncoder)
- [x] Always-on recording inside beginStep/endStep scope (no env var gating)
- [ ] Guard capture at end of step (deferred to Phase 2)
- **Entry criteria**: Phase 0 complete
- **Exit criteria**: Recording produces valid tape, no perf regression, all tests pass

### Phase 2: Replay (Single-Plan Steps)
- [ ] Guard validation logic
- [ ] Buffer resolution (tensor ID → GPUBuffer)
- [ ] Replay encoder loop
- [ ] Fallback: if replay fails, re-record
- **Entry criteria**: Phase 1 complete
- **Exit criteria**: Replay produces identical outputs to fresh dispatch, measurable speedup on simple models

### Phase 3: Replay (Multi-Plan Steps)
- [ ] Handle forward + backward + optimizer as separate tape segments
- [ ] Cross-segment buffer handoff
- [ ] Intra-step pool reclamation compatibility
- **Entry criteria**: Phase 2 complete
- **Exit criteria**: DistilGPT-2 training produces identical loss trajectory with tape replay

### Phase 4: Dynamic Updates
- [ ] Partial tape invalidation (e.g., only optimizer tape changes)
- [ ] `writeBuffer` patching for dynamic scalars (learning rate schedule)
- [ ] Warm restart after guard failure (re-record only affected segment)
- **Entry criteria**: Phase 3 complete
- **Exit criteria**: Training with LR schedule works correctly with tape replay

### Phase 5: Optimization
- [ ] Bind group caching within tape replay
- [ ] Submit batching (merge adjacent submits)
- [ ] Profile and measure actual ms/step improvement
- **Entry criteria**: Phase 4 complete
- **Exit criteria**: Measurable reduction in CPU dispatch overhead, documented in CLAUDE.md

## Decisions Log

Record design decisions as they are made. Format: `[date] Decision: <description>. Reason: <rationale>.`

- [2026-02-10] Decision: Use `scalarsByNode` map alongside IRGraph rather than adding scalar field to IRNode. Reason: Keeps IRNode unchanged for the trace-based path; scalars are an overlay concern only needed for cache keying. Update: Also added `scalarValues` directly to `IRNode` and `TraceEvent` so both the trace-based path (engine.ts) and the lazy-to-ir path have scalar access. This is the simplest approach — one field, propagated through both paths.
- [2026-02-10] Decision: Scalar canonicalization uses `canonicalizeF64Bits()` from `src/engine/scalar.ts`. Reason: Already implements §8.2.1 (canonical NaN, +0/-0 distinction). Reuse existing code.
- [2026-02-10] Decision: `hashIRGraph()` accepts optional `scalarsByNode` parameter for backward compatibility. Reason: Existing callers (tests, buildIRFromTrace path) should not break. When `scalarsByNode` is not provided, hash is identical to before.
- [2026-02-10] Decision: Always-on recording, no env var gating. Reason: Overhead is trivial (~560 array pushes per step, ~50KB memory). Recording active inside beginStep/endStep scope. The whole point is automatic optimization — gating behind an env var adds friction for no benefit.
