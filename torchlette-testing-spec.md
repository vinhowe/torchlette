# Piston / Sequence Toy Engine — Test-Driven Development Plan

This document describes a practical, layered testing strategy for the Piston engine and its ops, with an emphasis on:
- locking down **semantic correctness** (tokens, versions, planning determinism, retention),
- building **numerical correctness** incrementally (reference + PyTorch oracle),
- and running **performance benchmarks** (stable, repeatable microbenchmarks) without slowing the core test loop.

---

## 0) Goals

1. **Semantic ledger correctness comes first**
   Most engine bugs are in ordering/versioning/retention rules, not math. These tests must pass even with a mock backend.

2. **Numerical correctness is a separate track**
   Once ops exist, validate them with:
   - small-shape reference checks (fast, shrinkable),
   - and PyTorch oracle parity (broad coverage).

3. **Perf is measured with stable cases**
   Benchmarks are not unit tests. They run separately and produce JSON with GFLOPs/s and/or GB/s.

---

## 1) Test Harness Architecture (the prerequisite)

### 1.1 Deterministic trace recorder (always on in tests)
Implement internal tracing that records **semantic events** and stable identifiers:

- Node emission events:
  - `graphInstanceId`, `callInstanceId`, `mutId`, `opNonce`
  - op name, inputs/outputs, effects metadata (reads/writes, usesGlobal)
- Token events:
  - before/after `tokGlobal`, per-loc `tokLoc[loc]`
  - `after_all(...)` invocations and token edges
- Plan construction events:
  - forced roots (values/tokens)
  - expanded compiled-call subevents (from `SemanticSubeventSchedule`)
  - `PlanLinearOrder` with full `EventKey` list
  - planned increments: `locVersion`, `baseCommitVersion`
  - publish-save points / checkpoint boundary points
- Commit events:
  - which loc/base versions persisted
  - storageVersion bumps
- Cleanup-only events:
  - finalizeQueue enqueue
  - `drainFinalizeQueueCleanupOnly()` actions
  - refcount drops and allocator fencing bookkeeping

**Rule:** traces must be deterministic across runs given the same program and inputs.

### 1.2 Debug snapshot API (stable state only)
Expose an internal snapshot used by tests:

- Bases:
  - binding kind (`ssa` / `pending_loc` / `loc`)
  - `bindingVersion`, `baseLogicalVersion`, `baseCommitVersion`
  - `pinCount`, `mutatedSinceStep`
- Locs:
  - role, `hasValue`
  - `locLogicalVersion`, `locVersion`
  - tombstone kind
- Tokens:
  - current `tokGlobal`
  - current `tokLoc` map (loc → token id)

Avoid any backend-dependent details in snapshots.

### 1.3 Plan simulation API (critical for fast semantics tests)
Add internal-only entrypoints:

- `engine._debug_buildPlan(roots) -> Plan`
- `engine._debug_simulateCommit(plan) -> PredictedStateDelta`

This allows testing:
- deterministic linearization,
- version assignment,
- legality checks (tombstones, purity fence),
without requiring GPU or actual kernels.

---

## 2) Test Rings (layers)

### Ring 1 — Pure semantic tests (fast, mock backend)
**Purpose:** Lock down engine semantics.
**Backend:** mock/no-op kernels.
**Tools:** unit tests + property-based tests (`fast-check`).

### Ring 2 — Deterministic integration tests (small numerics)
**Purpose:** Validate lowering, view semantics, small numeric correctness.
**Backend:** CPU reference or simple JS reference; optional minimal GPU.
**Tools:** unit tests + property tests on tiny shapes.

### Ring 3 — Backend conformance + oracle parity (PyTorch)
**Purpose:** Compare Piston outputs to PyTorch for broad coverage; confirm WebGPU backend matches semantics + numerics.  
**Backend:** WebGPU (and/or CPU) + PyTorch oracle.

### Perf harness — Microbenchmarks (not gating correctness)
**Purpose:** Measure throughput and detect regressions; output stable JSON.
**Backend:** WebGPU + PyTorch (CPU/CUDA) optionally.

---

## 3) The “Spine” (implement + test first)

These tests should pass before any real kernel exists.

### 3.1 Token algebra + join rule
**Unit tests**
- `after_all` is associative/commutative/idempotent
- ordered access produces a **fresh** token
- join rule:
  - `tokIn = after_all(tokGlobal, tokLoc[loc] ?? tokGlobal)`
  - updates both `tokGlobal` and `tokLoc[loc]` to the fresh `tokOut`

**Behavioral tests**
- effectful op advances `tokGlobal` even if return value is dropped
- token-only replacements (`tok_after`) preserve well-formedness

### 3.2 Deterministic planning & `PlanLinearOrder`
**Tests**
- identical program + roots ⇒ identical `PlanLinearOrder`
- tie-breaking purely via `EventKey` lexicographic ordering
- stable under irrelevant object identity / JS iteration differences
- expanded compiled-call subevents appear as synthetic events with `planInstanceId = callInstanceId`

### 3.3 Version rules: `locLogicalVersion`, `locVersion`, `baseCommitVersion`
**Tests**
- `locLogicalVersion` increments at schedule-time and never rolls back
- `locVersion` increments only on committed stores and in `PlanLinearOrder`
- `baseCommitVersion` increments only via `base_commit(baseId, mutId)`
- **uniqueness:** exactly one `base_commit` per semantic in-place mutation

---

## 4) Feature Suites (semantic-focused)

### 4.1 Pending-loc + `initTok` + initializer subsumption
**Tests**
1. First access creates `initTok` via materialize store
2. Use-site rule: first ordered access is ordered after `initTok`
3. Subsumption: planner can elide materialize store if dominated by a semantic initializing store, but must insert token-only op to keep token graph well-formed
4. Checkpoint recompute ban: `ensure_initialized` must not materialize persistent-role locs during recompute; throw

### 4.2 Exec lock + finalizeQueue draining
**Tests**
- overlapping execution-affecting entrypoints throw `EngineBusyError`
- cleanup-only operations permitted while busy/poisoned:
  - `dispose`, `drainFinalizeQueueCleanupOnly`, tidy cleanup
- entry/exit drains happen in `finally` even on exceptions
- no microtask draining while `execLock.held === true`

### 4.3 Compile staging restrictions + trace epoch rules
**Tests**
- host read inside staging throws `HostReadInCompileError`
- async/await inside staging throws `AsyncInCompileError`
- stale trace tensor escape throws `InvalidTraceTensorEscapeError`
- compiled call emits lazy nodes only; no execution until forced
- compiled call’s `SemanticSubeventSchedule` expansion participates in global version assignment

### 4.4 Cache key stability (scalar canonicalization, etc.)
**Tests**
- scalar bit-pattern canonicalization distinguishes `+0.0` vs `-0.0`
- NaN behavior matches the chosen policy (payload-preserving or canonicalizing)
- little-endian encoding stable and deterministic

### 4.5 Autograd semantics (saved-for-backward, commit guards, lazy backward rooting)
**Tests**
- saved tensor then BaseId mutation → `SavedTensorModifiedError` on backward
- publish-save is a token-linear semantic event (appears in plan)
- lazy backward is rooted under tokGlobal so `markStep()` executes it even if handles dropped
- non-reentrant backward enforced (`NonReentrantBackwardError`)

### 4.6 Checkpointing: whole-body replay + purity fence + pending-loc ban
**Tests**
- recompute forbids persistent-role `loc_store`
- recompute forbids creating/populating `saved_state`
- recompute forbids mutating bases in `reachableBasesAtPack`
- pending-loc materialize during recompute throws (dedicated error or `CheckpointImpureRegionError`)

### 4.7 RNG: keyed RNG + checkpoint replay with no double-advance
**Tests**
- random ops are non-CSEable
- forward records `(opNonce, drawNonce)` list for checkpointed body
- recompute replays exact random outputs
- persistent RNG state does not double-advance (post-backward matches expected state)

### 4.8 Poisoning
**Tests**
- failed force poisons engine
- subsequent execution-affecting ops throw `PoisonedEngineError`
- cleanup-only operations continue to function
- optional: storages touched by failed plan quarantined from reuse

### 4.9 Fusion + memory planning (compile-only)
**Tests**
- fusion happens only inside compile (no graph rewrites outside compile)
- pointwise fusion: fused vs eager numeric parity (small shapes)
- broadcast-aware pointwise fusion: fused vs eager parity on broadcastable shapes
- scalar epilogues: fused vs eager parity (e.g. matmul + bias + relu when available)
- deterministic fusion recipes given identical compiled traces
- memory planning respects allocator fencing (no reuse before fence)
- in-flight plan retention pins buffers until execution completes
- cleanup-only disposal never schedules execution; GC/fencing occurs at step boundaries

#### 4.9.1 Multi-output fusion (§15.2)
**Tests**
- multi-output fusion: two outputs sharing common subexpression computed once
- multi-output fusion: outputs with independent subgraphs remain separate dispatches
- multi-output fusion: outputs of different shapes are not fused together
- multi-output fusion: cache key distinguishes different output topologies
- multi-output fusion vs eager: numeric parity for shared subexpression patterns

#### 4.9.2 Memory coalescing / vectorization (§15.3)
**Tests**
- vec4 coalescing: innermost dim divisible by 4 uses vec4 loads/stores
- vec2 coalescing: innermost dim divisible by 2 (not 4) uses vec2 loads/stores
- scalar fallback: non-divisible sizes use scalar path
- vectorized vs scalar: numeric parity for all vector widths
- broadcasting with vectors: scalar input correctly splatted to vec4/vec2
- alignment check: misaligned inputs fall back to scalar
- mixed vectorization: some inputs vectorized, others scalar (broadcast case)

---

## 5) Property-Based Testing Strategy (`fast-check`)

### 5.1 Two high-value property families

#### A) Determinism under equivalent programs (Ring 1)
Generate small random programs in a restricted DSL:
- ops: `alloc`, `view`, `loc_load/store` (synthetic), `inplace_mutate`, `save_for_backward`, `compiled_call_stub`, `markStep`, `read`
- constraints to avoid trivial invalid graphs

Properties:
- `PlanLinearOrder` stable for equivalent dependency graphs
- version increments match committed event counts
- token graph remains well-formed (no missing token edges)
- no unexpected increments on plan-only paths

#### B) Monotonicity & accounting invariants (Ring 1)
Across any run:
- `locLogicalVersion` is monotone
- `locVersion` increases by number of committed stores
- `baseCommitVersion` increases by number of committed `base_commit` events
- `storageVersion` bumps exactly once per committed in-place reuse store

---

## 6) Op Correctness & Oracle Parity (PyTorch)

This is separate from the engine semantic spine and starts once ops have real numerical behavior.

### 6.1 OpSpec Registry (single source of truth)
Define an `OpSpec` per op that drives both correctness fuzzing and benchmarks:

- property generator (`fast-check`) for shapes/dtypes/attrs
- tolerance policy (`atol`, `rtol`) dependent on dtype/shape/op
- Piston runner
- optional PyTorch oracle runner
- flop/byte count functions
- fixed benchmark cases
- optional expected codepath label (e.g., `gemv`, `gemm_tiled_128`)

### 6.2 Oracle bridge options
Preferred:
- Node test runner spawns Python and batches many cases per subprocess call:
  - serialize inputs (dtype/shape/bytes)
  - Python runs torch op (CPU or CUDA) and returns bytes

Optional:
- precomputed golden fixtures for CI speed (small shapes)

### 6.3 Correctness fuzzing workflow
For each op:
- small shapes (Ring 2): compare to a JS/CPU reference (fast shrinking)
- broader shapes (Ring 3): compare Piston to PyTorch oracle

Include:
- dtype coverage: f16, f32 (and i32/u32/bool where relevant)
- broadcasting/strides/view variants where supported
- edge cases: zeros, denorms (if relevant), NaNs/Inf (policy-defined)

### 6.4 MLP parity harness (future, after autograd+optimizer)
**Goal:** end-to-end parity of a tiny training step, not just single ops.

**Prerequisites (framework-level):**
- Autograd core (saved-for-backward, backward rooting, non-reentrancy).
- Linear/MatMul + bias add, ReLU, and a scalar loss (MSE or CE).
- Optimizer step (start with SGD; add Adam later).
- Deterministic RNG / seed control.

**What to compare (each step):**
- Forward activations + final logits.
- Loss scalar.
- Per-parameter grads.
- Parameter updates (and optimizer state for Adam).

**Recommended minimal model:**
- `Linear(4 -> 8) -> ReLU -> Linear(8 -> 3)` on a fixed batch.
- MSE loss first; CE later.

---

## 7) Performance Benchmarks (separate runner)

Benchmarks are not unit tests; they run on demand or nightly.

### 7.1 Metrics
For each benchmark case record:
- median time (with warmup)
- derived **GFLOPs/s** (compute-bound ops like GEMM/conv/attention)
- derived **GB/s** (bandwidth-bound ops like elementwise/reductions)
- selected kernel/codepath label (critical for matmul variants)
- device info, backend info, dtype, shape, flags (transpose, batch)

### 7.2 Benchmark methodology
- warmup iterations to stabilize compilation/caches
- measure enough iterations to dominate overhead
- WebGPU timing:
  - prefer timestamp queries; fallback to `performance.now()` around `markStep()` with enough work
- PyTorch timing:
  - CPU: `torch.utils.benchmark` or stable manual timing
  - CUDA: synchronize around timing

### 7.3 Canonical benchmark suites
#### Matmul (recommended fixed set)
Include cases designed to hit distinct kernel families:
- GEMV-ish:
  - `(1, k) @ (k, n)` and `(m, k) @ (k, 1)`
- square GEMM:
  - `(1024,1024)`, `(2048,2048)` (as device allows)
- tall-skinny:
  - `(4096,256) @ (256,4096)` and transpose variants
- batched:
  - `(b,m,k) @ (b,k,n)` for `b∈{8,32}`
- transpose flags:
  - NN, NT, TN, TT (if supported)
- dtypes:
  - f16 and f32 separately (and mixed if supported)

Add assertions in benchmark logs:
- “selected path == expected” for designated cases (e.g., gemv cases)

#### Other ops
- reductions/softmax/layernorm: focus on GB/s + stable shapes
- elementwise: a small representative set only (performance tends to scale similarly across elementwise ops)

### 7.4 Output
Bench runner outputs JSON:
- per-case results
- aggregate summaries
- optional regression comparison vs baseline JSON

---

## 8) Suggested Implementation Milestones (TDD order)

1. Trace recorder + snapshot + plan simulation
2. Token algebra + join rule tests (Ring 1)
3. Deterministic plan linearization + EventKey tests (Ring 1)
4. loc/base version rules + base_commit uniqueness tests (Ring 1)
5. Exec lock + finalizeQueue drain semantics tests (Ring 1)
6. Pending-loc `initTok` + subsumption tests (Ring 1)
7. markStep semantics tests (flush, retention, GC hooks) (Ring 1)
8. Compile staging + SemanticSubeventSchedule expansion tests (Ring 1/2)
9. Minimal numeric ops + small-shape reference tests (Ring 2)
10. PyTorch oracle bridge + property-based parity tests for first op (Ring 3)
11. Perf runner + first benchmark suite (matmul) (Perf harness)
12. Autograd + checkpoint + RNG replay suites (Ring 2/3)

---

## 9) CI / Running Strategy

- Default PR checks:
  - Ring 1 + Ring 2 (fast)
- Optional / nightly:
  - Ring 3 (PyTorch oracle parity, WebGPU backend conformance)
  - perf harness (bench JSON + regression tracking)

---

## 10) Deliverables / Repo Layout (suggested)

- `tests/semantics/` — Ring 1
- `tests/integration/` — Ring 2
- `tests/oracle/` — Ring 3 (PyTorch parity)
- `ops/specs/` — OpSpec registry definitions
- `tools/torch_oracle/` — Python oracle scripts + batching protocol
- `bench/` — perf runner + benchmark case definitions
- `bench/results/` — optional stored baselines (JSON)

---
