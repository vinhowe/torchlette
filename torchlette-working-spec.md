---

# Torchlette working spec (v1.22 — stateful-lazy + compiled regions + BaseId/Loc binding + non-reentrant checkpointing via saved-tensor hooks + abstract caching keys + loc roles + strong external-state retention + keyed RNG with planner-assigned draw nonces + committed mutation via unique base_commit(mutId) events + token-order commit versioning + saved_state lifetime + checkpoint ambient external-state policy + bind-time auto-externalize scheduling + deterministic draw ordering + poisoned-on-failure + state-slot alias-pattern caching + checkpoint RNG forking + engine execution lock + whole-body checkpoint replay + universal may-alias compiled variant + deterministic IR identity + deterministic plan linearization + RNG non-CSE + exec lock ownerId/depth + finalizer deferral queue + **lock-safe finalize draining** + **initTok pending-loc initialization** + stable null-state sentinels + pinned ABI state access order + pinned touched-locs token reconciliation + token-linear semantic effects + autograd frame liveness refcounts + **SemanticSubeventSchedule for compiled-call expansion** + **PyTorch-like lazy backward rooting** + scalar canonicalization by bit-pattern + **elementwise fusion** + **multi-output fusion** + **memory coalescing via vectorization**)

This spec defines a WebGPU-first tensor engine with **global laziness**, **explicit optimization boundaries**, **PyTorch-like autograd + mutation/view semantics**, and a **JAX/PT2-inspired compiled-region** model that remains lazy until forced.

v1.22 is **standalone**, defensible, and intended to fully specify runtime semantics and compiler boundaries.

---

## 0) Goals and invariants

### 0.1 Goals

1. **Pseudo-eager UX with global laziness**
   Ops return immediately; device/CPU work is delayed until explicitly required.

2. **Explicit optimization boundary**
   Fusion, CSE, algebraic rewrites/canonicalization, AMP transforms, memory planning, donation, and AOT autograd run **only inside compiled regions**.
   Outside compiled regions: only **ordering-correct semantic lowering/functionalization** is permitted; **no optimization rewrites or fusion**.

3. **TF.js-like lifetime management**
   `tidy()`, `keep()`, and `dispose()` define ownership. A `FinalizationRegistry` safety net exists but is explicitly **best-effort**, implemented via **deferred cleanup** plus safe draining (§1.4, §6.8).

4. **PyTorch-like mutation/view/autograd semantics**
   Aliasing identity, representable views, in-place ops, saved-for-backward, version checking, grad accumulation, and **lazy non-reentrant backward** by default.

5. **Two autograd modes, one differentiation core**

   * Outside compile: **lazy VJP construction** from lazy recipes (a “tape-at-realize” strategy may be used as an implementation optimization, not semantics).
   * Inside compile: **AOT autograd** (forward+backward+optional optimizer update in one compiled region).

6. **Non-reentrant checkpointing via saved-tensor hooks**
   Checkpointing is an **autograd saved-slot pack/unpack policy** (hooks), not a first-class tensor IR opcode. In compiled/AOT mode it is erased as a pure transform (subject to the semantic model in §10 and purity fence in §10.5).

7. **Keyed RNG without sequential head dependency**
   Randomness is keyed by a persistent RNG basis (algorithm id + seed) and a **planner-assigned draw nonce** per dynamic random execution, plus a stable per-op nonce. Random ops are **not value-CSEable** (§11, §15.4).

8. **Compilation caching**
   Executables cached by **normalized IR** + **policies** + **device caps** + **input signature** (shapes/dtypes/layout) + **alias/view pattern** + **scalar specialization schema** + **ordered, pinned abstract state access signature** (§8.4) + **bind-time state-slot alias pattern**, with a **universal may-alias fallback variant** (§8.5).

9. **Deterministic, serial semantics under async interleaving**
   The engine is single-queue and non-reentrant for execution-affecting public entrypoints; overlapping async callers are rejected with a defined error via an **ownerId/depth execution lock** (§6.8).

10. **Representation-independent committed mutation tracking**
    Any semantically in-place mutation that becomes externally observable must advance the BaseId’s **token-order committed mutation history** via a **unique** `base_commit(baseId, mutId)` event (§3.6, §4.3, §8.11, §9.3).

11. **Deterministic planning across compiled boundaries**
    Compiled calls are expanded at plan time into a deterministic list of semantic sub-events (a **SemanticSubeventSchedule**) so token-order commit assignment (`locVersion`, `baseCommitVersion`) is globally deterministic (§8.6.0).

---

### 0.2 Non-negotiable invariants

* **Lazy semantics are global:** `.compile()` and compiled calls never execute device work.
* **Explicit forcing exists:** effect-only programs are not guaranteed to run unless a dependent value is materialized or `await engine.markStep()` is called.
* **Effect ordering is token-linear:** every semantic effect returns a fresh token SSA value; program order is represented by token dependencies.
* **Ordered state reads/writes:** every ordered access is serialized by token chaining and the join rule.
* **Effectful ops advance engine tokens even if ignored:** emitting an effectful op must update the engine’s current token state even if the user drops returned values or a compiled function returns `void`.
* **No compile trace artifacts escape:** staging-created tensors not returned are invalid after staging ends.
* **Saved-for-backward is protected:** saved slots are guarded by **BaseId token-order committed mutation history** that respects program order under global laziness (§9.3).
* **Non-reentrant backward (default):** backward runs under a single backward engine; nested backward engines are forbidden.
* **Outside compiled regions there are no optimization rewrites/fusion.**
* **Checkpoint purity fence:** checkpoint recompute must not write persistent external state and must not rebind externally reachable bases (§10.5). Stateful-forward patterns like BatchNorm running-stats updates are **not supported** under checkpointing.
* **Host coercions are forbidden:** implicit Tensor→JS primitive conversions throw.
* **Donation never changes user-visible aliasing:** donation is internal reuse only.
* **BaseId coherence:** tensors that alias share a BaseId; when BaseId is loc-backed, all aliases observe mutations via ordered `loc_load/loc_store`.
* **Parallelism is out of scope:** engine serializes scheduling and execution; there are no correctness guarantees under concurrent multi-engine execution.
* **Engine execution lock (normative):** execution-affecting **public entrypoints** are non-reentrant across async interleavings; overlapping callers throw `EngineBusyError` (§6.8).
* **Failure policy:** if a forced execution fails, the engine becomes **poisoned**; token roots are dropped and further execution-affecting operations throw (§6.7). (Poisoning is “game over”; no transactional recovery.)
* **Missing state must not be represented by aliasing:** absent state slots must not bind by aliasing another slot or real state; use stable per-(compiledFn,stateSlot) sentinels (§8.8).
* **In-flight plan strong rooting:** any in-flight plan must hold strong references to every BaseId/Loc/Storage it might touch, independent of user pinCounts, tidy scope disposal, or finalizers (§14).

---

## 1) Runtime surface API and semantics

### 1.1 Tensor identity model: BaseId, pure values, views, and external state

Every user-visible `Tensor` handle has a **BaseId** identifying its alias class (PyTorch `Storage`-like identity). Representable views and explicit aliasing ops propagate BaseId; non-aliasing ops allocate fresh BaseIds.

A tensor’s current value is defined by its BaseId binding state:

* **SSA-backed:** value is the BaseId’s `baseValue` lazy recipe.
* **Loc-backed:** value is read via ordered `loc_load(loc)`.
* **Pending-loc (two-phase binding):** BaseId is bound to a LocId but not yet guaranteed initialized; initialization is represented by an `initTok` dependency (§2.2, §3.7).

---

### 1.2 Laziness and materialization

Materialization triggers:

* `await t.cpu()`
* `await t.read()` / `toArray()` / `await t.item()`
* any API requiring host-visible data

Materialization steps:

1. Resolve authoritative value for `t.baseId`:

   * SSA-backed: `baseValue`
   * pending-loc: ensure initialization dependency exists, then load
   * loc-backed: ordered `loc_load(loc)`
2. Build a forced execution plan for the required lazy subgraph (primitive kernels and/or compiled calls), including required semantic effects in the dependency closure (tokens) and required `base_commit` / publish-save events.
3. Submit work to CPU/GPU queue (serialized engine queue).
4. Await completion if host readback is needed; update caches.

**Minimal forcing rule (normative):**
Materializing a tensor value forces only the semantic effects that are in the **transitive dependency closure** of the values/tokens required to produce that tensor, not the entire global effect stream. (In contrast, `markStep()` forces the entire `tokGlobal` stream (§6.1).)

---

### 1.3 Host coercion traps

Tensor must not implicitly coerce to JS primitive. The following throw:

* `Tensor[Symbol.toPrimitive]`
* `Tensor.valueOf()`
* any implicit read during stringification or numeric conversion

Debug `toString()` may display metadata (shape/dtype/device/ids) but must not read device data.

---

### 1.4 Lifetime management: `tidy`, `keep`, `dispose` + GC safety net (deferred finalizers + safe draining)

#### `engine.tidy(fn)`

Creates a scope; tensors created inside are disposed at exit unless returned or `engine.keep(t)`’d.

#### `engine.keep(t)`

Marks `t` as escaping the current tidy scope.

#### Escape marking (normative)

Within a `tidy` scope, tensor lifetime is governed by **ownership**, not mere JS reachability.

* `engine.keep(t)` sets `t.escapes = true`. A kept tensor is treated as externally reachable even if `t.origin.kind === "tidy"`.
* Returning a tensor from `tidy(fn)` sets `t.escapes = true` for each returned tensor (recursively through arrays/tuples).
* Any tensor created inside a tidy scope that is neither returned nor `keep`’d is disposed at tidy exit **even if** user code still holds a JS reference to it. Accessing it thereafter throws `DisposedTensorError`.

#### `t.dispose()` / `engine.dispose(t)`

Logical disposal (idempotent):

* marks handle disposed
* releases per-handle cached storage refs
* releases tidy ownership if applicable
* decrements BaseId pin count
* updates autograd frame liveness refcounts where applicable (§9.8)

Disposal is a **cleanup-only** operation: it must never plan or submit execution. It may run even if the engine is busy or poisoned (§6.7, §6.8). Any reclamation of underlying buffers must respect allocator fencing and in-flight plan retention (§14).

#### FinalizationRegistry safety net (best-effort) — deferred cleanup queue (normative)

If a tensor handle is GC’d without explicit dispose, the finalizer must:

1. **Enqueue** a cleanup record into an engine-owned `finalizeQueue` (e.g., `{tensorHandleId, baseId, autogradFrameId?}`).
2. Perform no other mutations of engine state and must not throw.

The engine must be able to **drain `finalizeQueue` in cleanup-only mode** at the following safe points:

* the **start** of every execution-affecting public entrypoint,
* the **end** of every execution-affecting public entrypoint (in `finally`),
* as part of `markStep()` (before GC/fencing work),
* and optionally at explicit lock-holding safe points (§6.8.7).

Draining performs cleanup-only logical disposal actions (refcount drops, cache eviction, recording deferred releases subject to fencing) and must be permitted even if `engine.poisoned === true` and even if an execution lock is held (§6.7, §6.8). Draining must be idempotent and must not plan or submit execution.

**Microtask draining policy (normative):**
Implementations **must not** drain `finalizeQueue` from a microtask while `execLock.held === true`. (This prevents reentrancy and mid-plan state mutation.) Optional microtask draining is permitted only when the lock is not held (§6.8.7).

Finalizers are not prompt or guaranteed before process exit.

---

### 1.5 `engine.retainGrad(t)` and `.grad` semantics (leaf-only by default)

* By default, only **leaf** tensors have `.grad` populated/accumulated.
* Non-leaf/view tensors do not retain grads unless `engine.retainGrad(t)` is called.

Leaf definition is frontend-tagged: the engine supports a flag on Tensor creation/staging boundaries.

---

### 1.6 `.compile(fn, opts)` and compiled invocation model (lazy)

```ts
type CompiledFn = (...args: any[]) => Tensor | Tensor[] | void;
engine.compile(fn, opts?) => CompiledFn;
```

Key properties:

* `engine.compile(fn)` returns a callable.
* Calling it does **not** execute CPU/GPU work. It stages, builds region IR, and emits lazy nodes.
* Compile-only optimizations occur only inside compiled regions.

Staging is per call: each invocation stages in a restricted context, producing region IR + guards + abstract signatures and then emitting a lazy `CompiledCall` node.

Staging restrictions:

* host-visible reads in staging → `HostReadInCompileError`
* async/await in staging → `AsyncInCompileError`

Trace epoch enforcement:

* staging tensors are tagged `(regionId, traceEpochCreated)`;
* stale trace tensors throw `InvalidTraceTensorEscapeError`;
* non-returned trace tensors are disposed at epoch end;
* returned outputs are **re-wrapped** as normal tensors with origin `global` / `tidy` (not `compile_trace`).

Compiled invocations are execution-affecting public entrypoints and are subject to the engine execution lock (§6.8).

---

### 1.7 `await engine.markStep()` (explicit effect barrier + compaction + RNG + GC)

Purpose:

* force effect-only work to execute (optimizer updates, in-place writes with no readback, backward jobs)
* define a “step boundary” for GC and memory fencing
* perform compaction/promotion policy
* commit pending RNG counter updates

**Normative completion rule:**
`await engine.markStep()` returns only after all forced CPU work has completed and all forced GPU submissions required by the step’s forced roots have completed.

Full semantics in §6.

---

## 2) Core data types

### 2.1 DTypes, devices, storage

```ts
type DType = "f16" | "f32" | "i32" | "u32" | "bool";
type DeviceKind = "wgpu" | "cpu";
type StorageId = number;

type StorageHandle = {
  id: StorageId;
  device: DeviceKind;
  bytes: number;

  storageVersion: bigint;     // bump on committed in-place writes to this buffer (§5.3)
  refCount: number;           // engine-managed strong refs
  lastUsedSubmitId?: bigint;  // for allocator fencing
  quarantinedNeverReuse?: boolean; // optional, for poison failures
};
```

---

### 2.2 BaseId table (alias identity + bindingVersion + semantic intent + committed mutation history)

```ts
type BaseId = number;
type LocId = number;

type BaseBinding =
  | { kind: "ssa"; baseValue: LazyRef }
  | { kind: "pending_loc"; loc: LocId; seed: LazyRef; initTok?: ValueId /*effect_token*/ }
  | { kind: "loc"; loc: LocId };

type BaseEntry = {
  id: BaseId;

  binding: BaseBinding;

  // bumps ONLY on binding identity changes (ssa<->pending_loc<->loc, loc rebinding)
  bindingVersion: bigint;

  // bumps on each scheduled semantic value update while SSA-backed (and other SSA writebacks)
  baseLogicalVersion: bigint;

  // token-order committed mutation history:
  // increments exactly once per semantic in-place mutation via base_commit(baseId, mutId).
  baseCommitVersion: bigint;

  dtype: DType;
  device: DeviceKind;

  pinCount: number;            // # live JS tensor handles referencing this BaseId
  mutatedSinceStep: boolean;   // compaction bookkeeping

  lastRealizedStorage?: StorageHandle;
  lastRealizedGuards?: {
    bindingVersion: bigint;
    baseLogicalVersion: bigint;
    storageVersion?: bigint;
  };
};
```

**Normative separation:**

* `bindingVersion` is **identity of where the truth lives** (SSA vs which loc).
* `baseLogicalVersion` is **SSA-backed semantic intent freshness**.
* `baseCommitVersion` is **token-order committed in-place mutation history**, updated only by `base_commit`.

---

### 2.3 View metadata (representable)

```ts
type ViewMeta = {
  baseId: BaseId;
  offsetBytes: number;
  shape: number[];
  stridesBytes: number[];
  isContiguous: boolean;
};
```

Views never store base storage directly; they resolve through BaseId binding.

---

### 2.4 Loc table: commit version, logical version, strong storage, tombstones, roles

Locs represent **external state** or ordered scratch state. **Pure caches must not be represented as locs.**

```ts
type Tombstone =
  | { kind: "grad_invalidated" }
  | { kind: "user_freed" }
  | { kind: "internal" }
  | { kind: "null_state_slot"; compiledFnId: number; stateSlot: number };

type LocRole =
  | "tensor_state"
  | "grad_state"
  | "rng_state"
  | "engine_state"
  | "saved_state"
  | "internal_scratch";

type LocEntry = {
  id: LocId;
  dtype: DType;
  device: DeviceKind;

  locVersion: bigint;        // token-order committed store count
  hasValue: boolean;

  locLogicalVersion: bigint; // semantic-time store intent (never rolled back)

  role: LocRole;

  createdStepEpoch: bigint;
  createdAllocId: bigint;

  strongStorage?: StorageHandle;

  tombstone?: Tombstone;
};
```

**Retention policy (normative):**
For roles in `{"tensor_state","grad_state","rng_state","engine_state","saved_state"}` while reachable under reachability rules (§6.4, §9.8), authoritative storage **must** be strongly retained once `hasValue=true`.

**Logical version meaning (normative):**
`locLogicalVersion` and `baseLogicalVersion` are **semantic-time intent** only; they do not imply the value has been committed/executed. Committed existence is tracked by `hasValue` and committed store history (`locVersion`).

---

### 2.5 Tensor object

```ts
type Origin =
  | { kind: "global" }
  | { kind: "tidy"; scopeId: number }
  | { kind: "compile_trace"; regionId: number; traceEpoch: number };

type Tensor = {
  id: number;
  dtype: DType;
  device: DeviceKind;

  baseId: BaseId;
  view?: ViewMeta;

  storage?: StorageHandle;

  cachedLocId?: LocId;
  cachedLocVersion?: bigint;
  cachedStorageVersion?: bigint;

  cachedBindingVersion?: bigint;
  cachedBaseLogicalVersion?: bigint;

  cachedHost?: {
    value: any;
    bindingVersion: bigint;
    baseLogicalVersion?: bigint;
    locId?: LocId;
    locVersion?: bigint;
    storageVersion?: bigint;
  };

  origin: Origin;
  escapes: boolean;
  disposed: boolean;

  requiresGrad?: boolean;
  isLeaf?: boolean;
  grad?: Tensor | null;

  autogradFrameId?: number | null;
};
```

Definitions (normative):

* `LiveHandle(t) := !t.disposed && t.origin.kind !== "compile_trace"`
* `Escaped(t) := t.escapes === true`
* `ExternallyReachableTensor(t) := LiveHandle(t) && (t.origin.kind==="global" || Escaped(t))`
* `ExternallyReachableBases := { t.baseId | ExternallyReachableTensor(t) }`

---

### 2.6 Graph IR and metadata (key excerpts)

```ts
type ValueKind = "tensor" | "effect_token" | "rng_state" | "scalar";

type Effects = {
  reads?: LocId[];
  writes?: LocId[];
  usesGlobal?: boolean;
};

type MutationId = bigint; // runtime-only, unique per semantic in-place mutation

type Node = {
  id: number;
  op: OpCode;
  inputs: number[];
  attrs: Record<string, any>;   // must not contain varying scalar values
  outputs: number[];
  effects?: Effects;
  meta?: {
    checkpoint?: { frameId: number; phase: "forward" | "recompute"; preventCSE: boolean };
    rng?: { opNonce: bigint };

    baseCommit?: { baseId: BaseId; mutId: MutationId };
  };
};
```

---

## 3) Effects, tokens, external state, and forcing

### 3.1 Token algebra (normative)

The effect system uses SSA “tokens” for happens-before constraints.

* `after_all(a, b, ...) -> t` is associative, commutative, idempotent.
* `tok_after(tokIn) -> tokOut` produces a **fresh** token ordered after `tokIn`.

**Fresh token rule (normative):** any ordered access must produce a fresh token (directly or via `tok_after`) (§15.2).

---

### 3.2 Token state and tokGlobal-threading criterion (normative)

Engine maintains:

* `tokGlobal: EffectToken`
* `tokLoc: Map<LocId, EffectToken>`

If an operation (or synthetic semantic event) **can change externally observable behavior** or **can throw based on external state**, it must:

1. consume an effect token,
2. produce a fresh effect token,
3. advance `tokGlobal`.

This applies even if the user drops returned values or a compiled function returns `void`.

---

### 3.3 The join rule (program order across locs)

For any ordered access to `loc`:

* Let `tokR = tokLoc.get(loc) ?? tokGlobal`.
* Set `tokIn = after_all(tokGlobal, tokR)`.
* Perform ordered access with `tokIn`, producing fresh `tokOut`.
* Update both: `tokGlobal = tokOut` and `tokLoc[loc] = tokOut`.

---

### 3.4 Forcing model: execution planning with deterministic plan linearization

A **force boundary** is any operation that demands completion of some effects/values:

* host readback (`cpu()/read()/item()`),
* `markStep()`,
* explicit force primitives.

At a force boundary the engine:

1. Determines forced roots (values and/or tokens).
2. Builds a plan for the required lazy subgraph.
3. Submits CPU/GPU work respecting token and data dependencies.
4. Awaits completion as required.

During planning:

```ts
type PlannedLocState = {
  hasValue: boolean;
  storageRef?: { storage: StorageHandle | PlannedBufferRef };
  locVersion: bigint;
  storageVersion?: bigint;
};

type PlannedBaseState = { baseCommitVersion: bigint };
```

Initialize planned states from `LocEntry` and `BaseEntry`.

#### 3.4.1 Deterministic plan linearization (normative)

Define `PlanLinearOrder` as a deterministic total order over **live semantic events** reachable from the forced roots, including:

* ordered `loc_load`,
* ordered `loc_store` (semantic/materialize),
* `base_commit`,
* autograd “publish saved slot” events,
* checkpoint boundary/purity events.

It must respect all token and data edges and break ties deterministically via event identity (§3.4.2).

**Normative:** all plan-time “token-order” assignments—`locVersion` increments, `baseCommitVersion` increments, and saved-slot token points—are defined with respect to `PlanLinearOrder`.

#### 3.4.2 Deterministic Event Identity with plan-instance ids (normative)

Each graph emission instance and compiled-call emission instance carries a runtime-only id:

* `graphInstanceId: bigint` (monotonic, assigned at emission time under the execution lock),
* `callInstanceId: bigint` (monotonic, assigned per CompiledCall emission).

Event identity:

```ts
type EventKey = {
  planInstanceId: bigint;
  origin:
    | { kind: "ir_node"; normalizedGraphId: number; normalizedNodeId: number }
    | { kind: "synthetic"; normalizedGraphId: number; anchorNodeId: number; tag: string; index: number };
};
```

Tie-breaking compares `EventKey` lexicographically.

---

### 3.5 Token-order commit semantics (locVersion and baseCommitVersion)

* Each ordered committed `loc_store` in a forced plan is assigned the next `locVersion` in `PlanLinearOrder` (per-loc restricted).
* Each externally observable semantic in-place mutation increments `baseCommitVersion` **exactly once** via `base_commit(baseId, mutId)`.

On successful force completion: persist committed versions into `LocEntry.locVersion` and `BaseEntry.baseCommitVersion`.

---

### 3.6 `loc_load` / `loc_store` / `base_commit` semantics

#### `loc_load(loc, tokIn) -> (value, tokOut, storageRef)`

Checks missing/tombstone; resolves `storageRef` from planned state; returns fresh `tokOut`.

#### `loc_store(loc, tokIn, value, storeKind) -> (tokOut, storageRefOut)`

`storeKind ∈ {"semantic","materialize"}`

1. Tombstone checks.
2. If `storeKind==="semantic"`: increment `locLogicalVersion` at scheduling time (never rolled back).
3. During force planning: assign commit version `planned.locVersion += 1` at the event’s `PlanLinearOrder` position; set `planned.hasValue=true`; set `planned.storageRef=storageRefOut`.
4. **StorageVersion rule (normative):** if the committed store *may mutate an existing StorageHandle in-place* (rather than allocating fresh storage), it must bump that StorageHandle’s `storageVersion` **exactly once at commit**.
5. On success: set `locEntry.hasValue=true`; set `locEntry.strongStorage=committedStorage`; update `locEntry.locVersion`.

`loc_store` is semantic and must be tokGlobal-threaded via join rule.

**Elidable materialize store rule (normative):**
A `storeKind="materialize"` store may be elided during planning if its effect is subsumed by a dominating store that initializes the loc in token order; if elided, it must be replaced with a token-only operation (`tok_after`) so token dependencies remain well-formed.

#### `base_commit(baseId, mutId, tokIn) -> tokOut` (normative)

1. Consumes and produces an effect token; advances `tokGlobal`.
2. During planning at its `PlanLinearOrder` position: `PlannedBaseState[baseId].baseCommitVersion += 1`.
3. On success: update `BaseEntry.baseCommitVersion`.

**Uniqueness (normative):**

* Every semantic in-place mutation is assigned a fresh `mutId` at emission time.
* There must be **exactly one** `base_commit(baseId, mutId)` reachable for that mutation.
* No other operation increments `baseCommitVersion`.

---

### 3.7 BaseId load/store semantics (including pending-loc initTok)

#### `base_load(baseId) -> LazyRef`

* SSA-backed: return `baseValue`
* loc-backed: ordered `loc_load(loc)` via join rule
* pending-loc: `ensure_initialized(baseId)` then ordered `loc_load(loc)` such that the load is token-ordered after the initializer token.

#### `ensure_initialized(baseId) -> initTok?` (normative)

If binding is `pending_loc(loc, seed, initTok?)`:

* If `initTok` exists: return it.
* Else:

  1. Emit an ordered initializer store: `loc_store(loc, tokIn, seed, storeKind="materialize")` using join rule.
  2. Record the initializer store’s `tokOut` as `binding.initTok`.
  3. Return `initTok`.

**Use-site rule (normative):**
The first ordered access to the pending loc for this binding must be ordered after `initTok` by joining the access token input with `initTok`. (Implementations may do this by `after_all(tokIn, initTok)`.)

**Initializer subsumption rule (normative):**
If a semantic store to the same loc is scheduled such that it initializes the loc before any load in token order, the planner may elide the materialize store (as per §3.6) while preserving token well-formedness.

#### `base_store(baseId, newValue)` (outside compile)

Scheduling a semantic mutation:

* If SSA-backed: increment `baseLogicalVersion` immediately; set `mutatedSinceStep=true`.
* If loc-backed or pending-loc: set `mutatedSinceStep=true`.

Then:

* If loc-backed: schedule ordered `loc_store(..., storeKind="semantic")`, then schedule `base_commit(baseId, mutId)` at the commit token point.
* If SSA-backed: transition to pending-loc:

  * allocate `loc` with `role="tensor_state"`
  * set binding to `pending_loc(loc, seed=frozen_seed)` (no initTok yet)
  * bump `bindingVersion`
  * schedule semantic store to that loc
  * schedule `base_commit(baseId, mutId)` at commit token point

**Frozen seed capture (normative):**
The captured `seed` must be the SSA value as of before rebinding becomes visible and must not (directly or transitively) `loc_load` from the newly allocated loc.

---

## 4) Alias model, views, and mutation semantics

### 4.1 BaseId propagation

View outputs propagate BaseId and set ViewMeta; non-aliasing outputs allocate fresh BaseId. Donation is internal reuse only.

---

### 4.2 Representable views

Views resolve through BaseId binding.

---

### 4.3 In-place mutation and BaseId commit increments (normative)

Any semantic in-place mutation must cause **exactly one** `base_commit(baseId, mutId)` at the token point where the mutation becomes externally observable.

**Externalize-on-mutation rule:** the destination of any in-place write (including view writes) must be loc-backed before the mutation commits.

---

### 4.4 View mutation lowering (normative)

If mutation destination is a view:

1. `baseVal = base_load(baseId)`
2. `baseCopy = copy(baseVal)`
3. `baseNew = strided_scatter_<op>(baseCopy, viewMeta, ...)`
4. `base_store(baseId, baseNew)` (schedules semantic store + paired `base_commit`)
5. return view handle

---

## 5) Caching, freshness, and versioning

### 5.1 Loc-backed caching must be commit-guarded (normative)

* Loc-backed: cache guarded by `(locId, locVersion)` (and optionally `storageVersion`).
* SSA-backed: cache guarded by `(baseId, bindingVersion, baseLogicalVersion)`.

### 5.2 Semantic-intent versions remain authoritative for “meaning,” not caches

`locLogicalVersion` and `baseLogicalVersion` track semantic intent and may advance even if never forced.

### 5.3 StorageVersion rule (normative)

Any committed store that may reuse/mutate an existing `StorageHandle` in-place must bump that handle’s `storageVersion` exactly once at commit.

---

## 6) `markStep()` semantics: forcing, compaction, RNG, GC, token reset, failure policy, execution lock

Engine maintains `epoch`, `allocCounter`, `poisoned`, `execLock`, `finalizeQueue`, `mutCounter`.

### 6.1 Step 0: force all pending semantic effects

Force realization of `tokGlobal`. This flushes all tokGlobal-threaded semantic effects (including backward jobs, grad updates, `base_commit`, ordered loc stores).

### 6.2 Step 1: compaction/promotion policy (promotion-by-capture only)

As in v1.19, unchanged semantics: promote SSA bases only by capture of known-fresh realized storage, without forcing.

### 6.3 Step 2: finalize pending bindings opportunistically

If `pending_loc(loc)` and `loc.hasValue===true`, finalize to `loc(loc)` and bump `bindingVersion`.

### 6.4 Step 3: strong retention policy for reachable external state

Retain storages for reachable persistent locs, including autograd frame `saved_state` locs (§9.8).

### 6.5 Step 4: drain finalizers, GC + allocator fencing

Drain `finalizeQueue` cleanup-only; sweep unreachable graphs/frames; fence storage reuse (§14).

### 6.6 Step 5: await completion, reset tokens, reset flags, bump epoch

Await completion; then reset tokens and flags; `epoch++`.

### 6.7 Failure policy: poisoned engine is game over (normative)

As in v1.19, plus: ensure execLock is released/non-blocking for cleanup-only operations.

### 6.8 Engine execution lock and non-reentrancy (normative)

As in v1.19, with additional finalize-drain policy:

#### 6.8.7 Explicit safe-point finalize draining (normative)

The engine must expose an internal routine:

* `drainFinalizeQueueCleanupOnly()`

It may be invoked:

* at entry/exit of public entrypoints,
* during `markStep`,
* and **at explicit safe points inside a force boundary** (e.g., between submission and an awaited readback), but only by the lock-holding continuation.

**Microtask draining (optional):**
If implemented, microtask draining is permitted **only when `execLock.held === false`**.

---

## 7) Optimization boundary outside compile

Outside compiled regions, only semantic lowering is permitted; optimizations are forbidden.

---

## 8) Compiled regions: staging, normalized identity, caching, tokens, mutation, state-slot aliasing, writeback

### 8.1 Deterministic staging and normalized IR identity (normative)

Deterministic enumeration and normalization rules as before. Runtime-only ids (`callInstanceId`, `mutId`) are not part of normalized identity.

---

### 8.2 Abstract caching keys (includes scalar canonicalization)

Variants keyed by:

* normalized IR structural hash
* `inputSigAbstract` (shapes/dtypes/layout, alias groups, view transforms)
* scalar specialization schema + specialized scalar values canonicalized per §8.2.1
* policy bucket + device caps
* pinned `StateIfaceSigOrdered` (§8.4)
* bind-time `StateSlotAliasPattern` (§8.5), except universal may-alias

#### 8.2.1 Scalar specialization canonicalization (normative)

When scalar values are included in cache keys (because the scalar is specialized-by-value), they must be encoded as **typed bit patterns**:

* `f32`: IEEE-754 32-bit bit pattern (distinguish `+0.0` vs `-0.0`; preserve NaN payload bits as produced by f32 rounding).
* `i32`: 32-bit two’s complement bit pattern.
* `u32`: 32-bit unsigned bit pattern.
* `bool`: `0` or `1`.

Serialization for cache keys must be **little-endian**.

---

### 8.3 Canonical arg alias groups (normative)

As before.

---

### 8.4 Pinned ordered trace-local state access signature (ABI metadata) (normative)

As in v1.19. `touchedTargets` is pinned and enforced.

---

### 8.5 Bind-time state-slot alias pattern and variant selection (normative)

As in v1.19, with LRU variant explosion controls and universal may-alias.

---

### 8.6 CompiledCall token ABI and touched-locs reconciliation (normative)

As in v1.19: entry reconciliation, join rule per concrete loc, exit publish.

---

### 8.6.0 SemanticSubeventSchedule and compiled-call expansion for planning (normative)

Compiled calls are not semantically opaque for token-order commit versioning. Each executable variant must publish a deterministic schedule of its internal semantic sub-events:

```ts
type SubeventKind =
  | "loc_load"
  | "loc_store_semantic"
  | "loc_store_materialize"
  | "base_commit"
  | "publish_saved_slot"
  | "checkpoint_boundary"
  | "other_semantic";

type Subevent = {
  kind: SubeventKind;

  // Anchoring for deterministic identity:
  anchor: { normalizedGraphId: number; anchorNodeId: number; tag: string; index: number };

  // Abstract targets (preferred) or concrete locs:
  target?: AccessTarget;   // e.g., state(slot), arg_base(group), global
  loc?: LocId;             // allowed only after bind-time mapping

  baseId?: BaseId;         // for base_commit and mutation guards
};
type SemanticSubeventSchedule = Subevent[];
```

**Schedule derivation (normative):**

* The schedule must be derived deterministically from normalized IR identity and pinned ABI metadata, after compilation/AOT transforms that introduce semantic events (e.g., saved_state spills).
* The schedule order must be a deterministic linear extension of the call’s internal token/data dependency DAG (i.e., it must respect the call’s internal token ordering).

**Plan-time expansion rule (normative):**

At plan construction, a `CompiledCall` is expanded into its `SemanticSubeventSchedule` events. Each expanded event receives an `EventKey`:

* `planInstanceId = callInstanceId`
* `origin = synthetic(anchor.normalizedGraphId, anchor.anchorNodeId, anchor.tag, anchor.index)`

The planner includes these expanded events in `PlanLinearOrder` and uses them to assign:

* per-loc `locVersion` increments for committed stores,
* per-base `baseCommitVersion` increments for `base_commit`,
* and the token points for publish-save and checkpoint boundary events.

**Consequence (normative):**
Token-order commit assignment is globally deterministic even across compiled boundaries.

---

### 8.7 Bind-time mapping and auto-externalize (scheduling-only)

As before: auto-externalize SSA bases by binding to pending_loc and scheduling initializer (materialize) as needed; must not force.

---

### 8.8 Missing state must not be represented by aliasing (stable null-state sentinels) (normative)

As before.

---

### 8.9 State-slot aliasing: required unification of ordering chains

As before.

---

### 8.10 Mutation semantics inside compile: functionalization

As before; optional saved_state spills for AOT within one invocation allowed.

---

### 8.11 Region-exit persistence: commits + SSA writeback

As in v1.19; if `base_commit` scheduled, compiled call must be `usesGlobalToken=true`.

---

## 9) Autograd core: lazy backward, saved-for-backward, BaseId commit guards, grads, saved_state lifetime

### 9.1 Backward is lazy (PyTorch-like policy)

Calling `loss.backward()` schedules backward work lazily; it runs when forced, typically by `markStep()` or by forcing values that depend on backward outputs (e.g., reading grads, reading updated params).

**Normative rooting rule:**
Scheduling a backward job must produce a tokGlobal-threaded semantic effect such that the backward job is rooted by `tokGlobal`. This guarantees that `markStep()` forcing `tokGlobal` executes backward work even if user drops references.

### 9.2 Two autograd modes

As before.

### 9.3 Saved-for-backward and BaseId token-order committed mutation checks

As in v1.19, including publish-saved-slot as token-linear semantic event and use-time comparison against `baseCommitVersion`.

### 9.4 Leaf grads and accumulation

As before.

### 9.5 `zeroGrad({ setToNull=true })` tombstones

As before.

### 9.6 `retainGrad` for non-leaf/view tensors

As before.

### 9.7 Multiple backward calls (default)

As before.

### 9.8 `saved_state` locs: dedicated lifetime, liveness, and retention (normative)

As in v1.19; finalizer draining decrements `tensorRefCount` idempotently.

---

## 10) Non-reentrant checkpointing via saved-tensor hooks (whole-body replay)

### 10.1–10.4 Principle, hook API, wrapper behavior, recompute context

As before, with whole-body replay and recompute restrictions.

### 10.5 Purity fence (strict) + pending-loc initialization rule (normative)

During recompute:

* No semantic `loc_store` to persistent roles (`tensor_state`, `grad_state`, `rng_state`, `engine_state`).
* Only `internal_scratch` may be written, and it must not escape recompute.
* No creation or population of `saved_state`.
* Mutating any base in `reachableBasesAtPack` is forbidden.

**Pending-loc init rule (normative):**
While `engine.recomputeMode===true`, `ensure_initialized` must not emit (or cause execution of) any `storeKind="materialize"` store to a loc with a persistent role. If initialization would be required to satisfy a `base_load` of a pending-loc base, the engine must throw `CheckpointImpureRegionError` (or a dedicated `CheckpointPendingLocMaterializeError`).

### 10.6 Compiled calls during recompute: readonly-or-reject + no auto-externalize

As before.

### 10.7–10.9 Determinism, preventCSE, ambient snapshot

As before.

---

## 11) RNG: keyed RNG with planner-assigned draw nonces and checkpoint RNG forking

### 11.1 Keyed RNG semantics + non-CSE

`samples = RNG(algoId, seed, drawNonce, opNonce, opParams)`

Random ops are not value-CSEable.

### 11.2–11.6 Persistent rngLoc, opNonce, draw ordering, drawNonce reservation

Implementation-defined, must satisfy invariants and be token-threaded where state evolves.

### 11.7 Checkpoint RNG determinism (forked RNG basis; net effect)

Pack records RNG basis and a per-op replay list `(opNonce, drawNonce)` for the checkpointed body. Recompute replays that list exactly; persistent rngLoc counters are not consulted for those ops.

**Net effect rules (normative):**

1. Recompute must not cause the persistent RNG state to advance “again” relative to the original forward execution. (No double-advance.)
2. Any mechanism that reserves `drawNonce` from persistent RNG state must be defined so replay uses recorded nonces without consuming new ones.
3. After backward completes, the persistent RNG state must match the state that would have resulted from a non-checkpointed execution of the same program (modulo any intentional differences explicitly specified by policy).

---

## 12) AMP inside compile: select-gated commits

As before.

---

## 13) Transfers, CPU tensors, cross-device execution (lazy)

As before.

---

## 14) Memory planning, allocator fencing, in-flight plan retention

As before, including in-flight plan strong rooting.

### 14.1 Buffer pool recycling boundary (normative)

When the WebGPU backend uses a step-level shared encoder scope (`beginSharedEncoder` / `endSharedEncoder`), buffer pool recycling **must** align with encoder scope boundaries:

1. **Release timing:** When a tensor is destroyed, its backing buffer enters `pendingRelease` — a queue of buffers awaiting safe reuse.
2. **Flush timing:** Buffers in `pendingRelease` are moved to the acquirable pool **only** at `endSharedEncoder()` (end-of-step). They must **not** be flushed mid-step inside `flushSharedEncoder()`.
3. **Rationale:** Within a step, multiple plan executions (forward, backward, optimizer) share the encoder scope. `flushSharedEncoder()` submits the current encoder and creates a new one, but buffers from earlier submissions may still be in-flight on the GPU. Flushing them to the acquirable pool allows a subsequent op to write to a buffer the GPU is still reading, causing silent data corruption.
4. **Inter-segment reclamation:** Periodic reclamation *between* plan execution segments (where each segment's encoder is fully submitted) is safe because the flush and reacquisition occur in separate encoder scopes with an intervening submit boundary.

---

## 15) IR optimization inside compile: token/value CSE interaction + `tok_after` + RNG restrictions + fusion + memory coalescing

As before; redundant ordered-load replacement with `tok_after` only if throw behavior cannot change; random ops non-CSE.

### 15.1 Elementwise fusion (normative)

Chains of elementwise operations (unary and binary) are fused into single kernels when:

1. **Fusibility criterion:** All ops in the chain are elementwise (no reductions, matmuls, or shape-changing ops)
2. **Single output:** The chain produces exactly one output tensor (multi-output fusion in §15.2)
3. **Shape compatibility:** All intermediate shapes are broadcast-compatible with the output shape

**Fusion recipe construction:**
- Walk IR nodes in topological order from outputs backward
- Collect fusible nodes until reaching non-fusible producers or external inputs
- Build expression-based SSA representation mapping each node to a WGSL expression

**Expression inlining policy:**
- Single-use intermediate expressions are inlined directly
- Multi-use expressions are materialized to local variables to avoid recomputation

**Fusible operations (normative list):**
- Unary: `relu`, `gelu`, `silu`, `sigmoid`, `tanh`, `neg`, `abs`, `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `floor`, `ceil`, `round`, `cast_*`
- Binary: `add`, `sub`, `mul`, `div`, `pow`, `min`, `max`, `eq`, `ne`, `lt`, `le`, `gt`, `ge`

**Broadcasting in fused kernels:**
- Each input's index is computed via broadcast index calculation
- Scalar inputs (shape `[1]` or `[]`) use constant index `0`
- Non-broadcast inputs use the global thread index directly

### 15.2 Multi-output fusion (normative)

When multiple outputs share common subexpressions, they may be computed in a single kernel dispatch:

**Eligibility:**
1. All outputs have the same shape (no broadcasting at output level)
2. The DAG of operations feeding the outputs is entirely fusible (per §15.1)
3. No output is consumed by another output in the same fusion group

**Kernel structure:**
- Single dispatch over output shape
- Multiple output bindings (`out0`, `out1`, ..., `outN`)
- Common subexpressions computed once and stored in registers
- Each output writes its final expression

**Example:**
```
a = input0
b = input1
c = a + b        # common subexpression
out0 = relu(c)   # output 0
out1 = c * a     # output 1 (reuses c)
```
Fuses to single kernel with 2 outputs, computing `c` once.

**Cache key extension:**
- Multi-output recipes keyed by: output count + per-output expression hash + shared subexpression topology

### 15.3 Memory coalescing via vectorization (normative)

For bandwidth-bound elementwise kernels, vector loads/stores (`vec2<f32>`, `vec4<f32>`) improve memory throughput:

**Applicability:**
1. Innermost dimension is contiguous in memory (stride = 1)
2. Innermost dimension size is divisible by vector width (2 or 4)
3. All inputs and outputs have compatible alignment

**Vector width selection:**
- Prefer `vec4` when innermost dim divisible by 4
- Fall back to `vec2` when divisible by 2 but not 4
- Use scalar when neither applies or alignment is incompatible

**Implementation:**
```wgsl
// vec4 coalesced load
let v: vec4<f32> = vec4<f32>(
  in0[idx * 4], in0[idx * 4 + 1], in0[idx * 4 + 2], in0[idx * 4 + 3]
);
// Or using pointer cast when available:
// let v = *(&in0[idx * 4] as ptr<storage, vec4<f32>>);
```

**Dispatch adjustment:**
- Total elements divided by vector width
- Workgroup size remains unchanged (threads process more data)

**Broadcasting with vectors:**
- Scalar broadcasts replicate to all vector lanes: `vec4<f32>(scalar)`
- Non-aligned broadcasts fall back to scalar path for that input

### 15.4 Random ops are non-fusible (normative)

Random operations (`rand`, `randn`, `randint`, etc.) are excluded from fusion:
- Never merged via CSE (each draw must be unique)
- Act as fusion barriers (break chains at random op boundaries)
- Preserve draw order within fused regions preceding/following random ops

---

## 16) Error taxonomy

### 16.1 Staging/compile

* `HostReadInCompileError`
* `AsyncInCompileError`
* `InvalidTraceTensorEscapeError`
* `InputSignatureMismatchError`
* `StateSlotAliasMismatchError`
* `CompiledIllegalLocAccessError`

### 16.2 Coercion and API misuse

* `TensorHostCoercionError`
* `DisposedTensorError`

### 16.3 State/loc

* `MissingExternalStateError`
* `InvalidatedLocError`
* `InvalidatedGradError`
* `SavedTensorModifiedError`
* `NullStateSlotError`

### 16.4 Checkpoint

* `CheckpointImpureRegionError`
* `CheckpointDeterminismError`
* `CheckpointReentrancyError`
* `CheckpointCompiledNotPureError`
* `CheckpointCompiledAutoExternalizeError`
* `CheckpointPendingLocMaterializeError` (or folded into `CheckpointImpureRegionError`)

### 16.5 Autograd/backward

* `BackwardGraphFreedError`
* `NonReentrantBackwardError`

### 16.6 Execution / failure / serialization

* `KernelExecutionError`
* `CompiledExecutionError`
* `EngineBusyError`
* `PoisonedEngineError`

---

## 17) Minimal staged implementation plan (v1.22)

1. Tensor objects + `tidy/keep/dispose`
2. FinalizationRegistry safety net via deferred `finalizeQueue`; drain at entry/exit and `markStep`; forbid microtask drain while execLock held; add `drainFinalizeQueueCleanupOnly()`
3. BaseId allocation/propagation; representable views
4. Lazy graph representation + realize executor + engine queue
5. Token algebra + join rule; tokGlobal-threading criterion; effectful ops always advance tokens
6. Forcing planner with per-force planned state and deterministic plan linearization + deterministic event identity
7. Base commit tracking via unique `base_commit(baseId, mutId)`; validate mutId uniqueness
8. Loc table with roles/tombstones/retention; commit versions + schedule-time logical versions; storageVersion rule
9. Pending-loc init via `initTok` recorded in binding; ensure base_load orders after initTok; allow planner to elide materialize stores by replacing with token-only
10. Execution lock + forbid nested force boundaries; cleanup carve-outs; safe-point finalize draining
11. `markStep()` flushes `tokGlobal`, promotion-by-capture, finalize binds, retention, GC, token reset
12. `.compile()` staging-per-call; normalized identity; trace-epoch enforcement; output rewrap
13. Pinned `StateIfaceSigOrdered` + touchedTargets enforcement
14. Bind-time mapping + scheduling-only auto-externalize; null-state sentinels
15. Compiled-call ABI token reconciliation + **SemanticSubeventSchedule** export; plan-time expansion to subevents for global version assignment
16. Autograd core + saved-for-backward commit guards; publish-save is token-linear semantic event
17. Lazy backward: schedule backward jobs rooted under `tokGlobal`; grads/param reads force required backward dependencies; non-reentrant backward
18. Checkpointing: whole-body replay; strict purity fence; disable persistent writes; forbid pending-loc materialize during recompute; compiled calls readonly-or-reject; checkpoint RNG replay with no double-advance
19. Multi-output fusion (§15.2): detect shared subexpressions across multiple outputs; generate single kernel with multiple output bindings; cache key includes output topology
20. Memory coalescing (§15.3): detect vectorizable access patterns; generate vec4/vec2 load/store variants; dispatch adjustment for vector width; fallback to scalar for misaligned/non-divisible cases
21. Strided ViewMeta (§4.2-4.4): full ViewMeta with offsetBytes/stridesBytes/isContiguous; stride tracking in IR nodes; strided kernel access; slice operation; view mutation lowering via strided_scatter_* ops. See `torchlette/docs/strided-viewmeta-implementation-plan.md` for detailed implementation plan.

---
