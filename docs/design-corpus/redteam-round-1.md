# Red-team review: schedule state + model editor

## WGSL assumptions embedded in the v1 object

### R1 — `vec width`, workgroup geometry, and residency are not portable facts—and are not even one fact each on WGSL

> “`decorations: DecorationVector // tile sizes, vec widths, workgroup dims, unroll`”
> — `schedule-state-design.md:42`
>
> “memory spaces | `{global, workgroup-shared}` … thread hierarchy |
> `{invocation, subgroup(gated), workgroup}`”
> — `schedule-state-design.md:64-69`

**Attack.** Appendix A **A-R6–A-R11** cannot map these axes cleanly to Triton.
Triton exposes logical blocks and a CTA warp budget, while TTGIR owns vector load width,
register/lane/warp layouts, shared allocation, and element-to-thread mapping. The proposed
axes preserve WGSL's source vocabulary (`vec4`, x/y workgroup IDs, explicit
`var<workgroup>`) rather than the intent that another realizer could consume.

The current code also disproves the claim that “vec width” is one WGSL decoration.
`TileKernelSpec.vectorize` changes the global-ID formula, launch grid, binding view, and
whole body (`tile-ir.ts:3345-3349`); `ctx.loadVec4` is an operand-local aligned storage
load; shared vec4 arrays are another lowering choice. The compiler rejects combining the
first two (`tile-compiler.ts:472-477`). Likewise, `BlockOps.load` derives register/shared
placement from pointer kind (`tile-ops.ts:540-627`), while matmul workgroup x/y is derived
from tile/thread-tile ratios (`matmul/types.ts:244-253`). A single scalar axis cannot own
any of these without losing required preconditions and coupling.

**Severity: blocker.** This is the exact WGSL-ism Appendix A was meant to catch.

**Proposed fix.** Split the object before P0:

- semantic schedule: logical block shapes, loop/dispatch structure, no-materialization
  edges, program-grid map, and named value lifetimes;
- backend requests: warp budget, desired pipeline depth, cache policy, and placement
  preferences, each scoped to a loop/value-use interval and allowed to report the
  realized value;
- backend determinations: WGSL x/y/z workgroup geometry and exact vec-load forms, owned
  only by the WGSL realization receipt.

Do not call all three `DecorationVector`, and do not hash a requested value as if it were
the realized physical schedule.

### R2 — “pipeline depth = 1” invents a WGSL fact and is the wrong cardinality for Triton

> “pipeline depth | decoration, pinned 1 | Triton `num_stages` / CuTe stages”
> — `schedule-state-design.md:68`

**Attack.** No pipeline and a one-stage pipeline are not demonstrably the same state. A
WGSL kernel with a barriered load/compute loop has no abstract pipeline object merely
because the schema writes `1`. Triton makes this worse: kernel `num_stages` applies
primarily to loads feeding `dot`, while `tl.range(num_stages=...)` is loop-local and
tries to pipeline a broader load set. Multiple loops may need different settings, and
TTGIR still chooses the instruction schedule. Appendix A **A-R9/A-R14** therefore cannot
honor the proposed scalar.

**Severity: blocker.** The degenerate value creates false state rather than future-proof
state.

**Proposed fix.** Use `none | { loopUid, loadGroupUids, requestedStages,
realizedStages? }[]`. `none` must be legal. Keep the request out of semantic equivalence;
include it in compilation identity, and store the realized pipeline in a realizer receipt.

### R3 — the enumerated “sets/levels” erase the information synchronization needs

> “Represent dimensions as sets/levels even where WGSL is degenerate. Absences and
> booleans are forbidden representations…”
> — `schedule-state-design.md:61-62`

**Attack.** Memory spaces are not one level chain, synchronization scope is not a
barrier, and thread hierarchy is not a set of simultaneously selected labels. A correct
synchronization fact needs at least primitive, participants, memory order, affected
address spaces, and control-flow convergence. Triton exposes a block barrier, while its
atomics separately expose `cta`/`gpu`/`sys` visibility and acquire/release semantics.
The proposed `{workgroup}` cannot distinguish those. “Grid scope” is especially
misleading: a grid-wide barrier is not available inside an ordinary Triton or WGSL
kernel. See Appendix A **A-R10/A-R11**.

**Severity: design-round.** It will make legality either unsound or backend-specific.

**Proposed fix.** Replace the table with typed relations:
`MemoryEffect(space, valueUid, interval)`, `Barrier(participants, spaces, convergence)`,
and `Atomic(order, visibilityScope)`. Model hierarchy as a backend capability graph, not
as values stored in every schedule.

### R4 — concrete grammar counterexample: grouped program-grid ordering for L2 reuse

> “Mutators on ScheduleState: `{tile, stream, recolor, fuse, pack,
> role-partition, pipeline}`”
> — `schedule-state-design.md:73-75`
>
> “the closure of the grammar contains the state of the art”
> — `schedule-state-design.md:149-151`

**Attack.** Triton's published matmul maps a linear program ID into groups of output rows
before columns, specifically to improve L2 reuse; its tutorial reports more than 10% on
A100 and the public API includes `tl.swizzle2d`
([official tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html),
[official API](https://triton-lang.org/main/python-api/generated/triton.language.swizzle2d.html)).
This is WGSL-expressible with arithmetic on `workgroup_id`. It changes neither arithmetic,
tile size, streaming, fusion, packing, roles, nor pipeline. P2 defines `recolor` by using
it on the accumulator (`schedule-state-design.md:189`), so stretching `recolor` to mean
grid traversal would make it an untyped “change any mapping” escape hatch.

The live repo already contains the narrower same-class technique: `CodegenOptions.swapGrid`
“improves L2 cache reuse for wide shapes” (`matmul/types.ts:39-40`), and the matmul
generator implements it (`tile-matmul.ts:290-294`). Yet §2 has no grid-map field, §3 has
no program-remap move, and P0's proposed deletion would have nowhere to put this existing
fact. This is Appendix A **A-R15**.

**Severity: blocker** to the grammar-completeness claim and to P0 lossless reification.

**Proposed fix.** Add a semantic `program-map`/`traverse-grid` move and a canonical
bijection-valued `ProgramGridMap` (identity, axis swap, grouped, plus a checked affine or
index expression). Legality must prove in-bounds one-to-one coverage. Add current
`swapGrid` and Triton's grouped matmul as mandatory reification tests. If the seven-move
fence is non-negotiable, withdraw the completeness claim.

### R5 — the atom declarations encode one backend implementation and one undefined family

> “v1's atom set: `{atomicAddF32-CAS, subgroup ops (feature-gated)}`.”
> — `schedule-state-design.md:90-91`

**Attack.** `atomicAddF32-CAS` combines semantic operation with WebGPU's realization;
Triton has native float `tl.atomic_add` and also `tl.atomic_cas`. Requiring CAS would
force an inferior backend algorithm; allowing native add means the atom's identity lies.
It also omits memory order, visibility scope, NaN behavior, and progress assumptions.
“Subgroup ops” is not an atom: it is an open family with different signatures and
semantics, while ordinary Triton does not expose a stable generic subgroup value or lane
mapping. Appendix A **A-R12/A-R13** consequently has split/refused verdicts.

**Severity: blocker.** Atoms are the P4 escape-hatch boundary; vague/backend-shaped atoms
make that gate meaningless.

**Proposed fix.** Name the semantic atom `atomicAdd<f32>` with order/scope and numerical
contract; put `CASLoop` versus `NativeAtomic` in the realization receipt. Enumerate every
subgroup primitive separately (operation, width contract, convergence, fallback, and
feature requirement). Ban composite loop nests and whole authored kernels from atom
admission.

## Altitude and ownership violations

### R6 — ScheduleState cannot own structure while imperative tile-IR and its compiler still own it

> “Schedule state owns structure + decorations; tile-IR expressions own semantics…”
> — `schedule-state-design.md:55-57`
>
> “skeletons become derived from schedule state… Existing structural generators die”
> — `schedule-state-design.md:137-141`

**Attack.** The live tile-IR is explicitly an imperative scheduling language, not a
semantic expression DAG. It contains `forRange`, `forStride`, barriers, shared arrays,
thread IDs, workgroup size, placement-bearing pointer kinds, register blocks, and
operand-placement-dependent dot lowering (`tile-ir.ts:1-12, 678-715, 3333-3373`;
`tile-ops.ts:175-224, 540-685`). Then the compiler independently chooses vectorization,
subgroup enablement, threads-per-row, block layouts, barriers, LICM, and unrolling
(`tile-compiler.ts:381-467`). Every named ScheduleState fact is already owned once or
twice below it.

P0 says “derive ScheduleState from every existing tile-IR kernel,” but does not specify
an inverse from arbitrary TypeScript control flow to `SkeletonInstance`, nor a new
semantics-only IR from which structure has actually been removed. Observing the existing
structure and replaying it through the same generator would pass the byte differential
while leaving all ownership intact.

**Severity: blocker.** This violates the draft's own review-blocking leveling rule.

**Proposed fix.** Specify the new IR boundary before phases: exact semantic node types;
exact skeleton schema; a one-way `applySchedule(semanticIR, state) -> loweredTileIR`;
and a deletion/relocation table for every current structural field and compiler
auto-pass. P0 must assert that no source generator or lower IR contains a second value
for those facts. Kernels that cannot be inverted are authored atoms, not fake reified
states.

### R7 — `fuse` is owned simultaneously by ScheduleState and the islands partition

> “`fuse` at this altitude = absorb an adjacent island's algorithm, i.e. the islands
> merge seen from inside”
> — `schedule-state-design.md:73-75`
>
> “The island owns membership.”
> — `schedule-state-design.md:55-57`
>
> “`merge` and `split` as the ONLY mutators”
> — `islands-design.md:253-260`

**Attack.** Absorbing an adjacent island changes membership and dispatch count. It cannot
be a mutation of an intra-island ScheduleState while membership is owned exclusively by
`Partition`. It also creates two provenance entries and two cache coordinates for one
gesture: `boundaryHash` and `scheduleHash`. No transaction order or rollback rule says
which changes first or what happens if the merged interior fails realization. The editor
contract already says fuse-N→1 **is** `merge` (`islands-design.md:408-414`).

**Severity: blocker.** This is direct double ownership between sibling designs.

**Proposed fix.** Remove `fuse` from ScheduleState mutators. Define a composite editor
command at the next altitude: validate a proposed interior schedule, atomically
`merge(P,a,b)`, attach the new ScheduleState, and commit one provenance record containing
both hashes. Roll back all parts on realization failure. Only `Partition` changes
membership.

### R8 — `AlgorithmRef` duplicates island contents and has no defined boundary with semantic tile-IR

> “`algorithm: AlgorithmRef // the semantic computation (tile-IR expression DAG /
> island contents)`”
> — `schedule-state-design.md:39-41`

**Attack.** “tile-IR expression DAG” and “island contents” are not synonyms. The former
is a per-kernel program; the latter is membership in a step graph. Storing either inside
ScheduleState duplicates the owners asserted two lines later. It is also unclear whether
fusing changes `AlgorithmRef`, the island, both, or creates a new semantic DAG. The hash
then explicitly omits `AlgorithmRef` (`schedule-state-design.md:126-128`), so the object
both stores and declines to identify this supposed field.

**Severity: blocker.** The central object is not normalized.

**Proposed fix.** Make ScheduleState parameterized by an immutable `SemanticRegionUid`
owned by the semantic graph; it is a foreign key, not copied contents and not part of the
schedule's mutable payload. Define the exact semantic IR below that key. A composite
partition transaction creates a new region UID when membership changes.

### R9 — “registry entries become named schedule states” collapses algorithm family, applicability, choice, and instance

> “the variant registry's entries become named schedule states (registry = a catalog of
> ScheduleStates, selection stays data). No duplicate ownership.”
> — `schedule-state-design.md:46-49`

**Attack.** The actual matmul registry does more than store decoration vectors. Its union
selects algorithm families (`tiled` versus `gemv`), stores family-specific parameters,
and evaluates applicability from geometry, dtypes, transpose, epilogue, casts, explicit
pinning, and subgroup support (`matmul/variants.ts:49-104`). GEMV route code derives
split-K and dispatch geometry; tiled config has a different schema. A concrete
ScheduleState instance cannot simultaneously be a reusable named template, an
applicability predicate, and a measured per-shape selection. Calling every entry a state
either moves semantic algorithm choice into decorations or keeps a second registry
hidden beside it.

**Severity: design-round.** P1 cannot perform the claimed ownership move as written.

**Proposed fix.** Separate `{AlgorithmFamily, ScheduleTemplate, ApplicabilityPredicate,
RealizerRequirement}` from an instantiated ScheduleState and from a `SelectionReceipt`
(shape/device/realizer/version/measurement). Prove a field-by-field migration for GEMV,
tiled matmul, split-K, swap-grid, epilogues, and explicit configs before claiming the old
registry dies.

### R10 — authored kernels cannot expose generic decorations while their skeleton is opaque

> “an opaque ScheduleState whose DECORATIONS are tunable but whose internals take no
> macro moves”
> — `schedule-state-design.md:142-145`
>
> “the real fused flash-attention kernel with its real knobs… ships… BEFORE any
> macro-move work”
> — `schedule-state-design.md:207-210`

**Attack.** Decorations are legal only relative to the hidden skeleton. Current code has
exact examples: vec4 loads require alignment and a multiple-of-four binding length
(`tile-ir.ts:337-347`); whole-kernel vectorization cannot coexist with `loadVec4`;
matmul tile sizes must divide thread tiles and fit shared memory
(`matmul/types.ts:203-240`); GEMV epilogues are forbidden on split-K routes
(`matmul/variants.ts:110-125`). An opaque kernel cannot be validated by generic
DecorationVector legality because the facts needed for validation are precisely what is
opaque. A UI can therefore produce an accepted-looking state that violates the authored
kernel's own indexing, barrier, or storage assumptions.

**Severity: blocker** for the advertised early workbench.

**Proposed fix.** Every authored atom must publish a typed parameter schema, dependent
constraints, derived geometry, capability predicate, and canonical cache-key encoder.
Only those declared parameters are editable; generic decorations are refused. The atom
must return a checked realization receipt before the ledger commits the edit.

## Sibling-document composition failures

### R11 — the claimed shared identity spine is three incompatible schemes with no provenance bridge

> “Shared machinery: stable-identity + provenance-ledger idiom…”
> — `schedule-state-design.md:236-240`
>
> “Every node carries a UID; every edit appends a ledger entry.”
> — `model-editor-charter.md:53-57`

**Attack.** Model nodes have persistent UIDs; live islands use final-plan integer
positions (`fusion-detect.ts:1087-1095`); ScheduleState defines no UIDs for loops, staging
edges, values, roles, or decorations. `scheduleHash` is content identity, not an operand
identity for edits. On re-record, plan positions can change even if a model UID survives;
views may be outside islands; one model node may lower to many LazyIR nodes and one fused
node may carry many origins. There is no map on which “click model node → island →
schedule object,” diff, undo, or provenance can be sound.

**Severity: blocker.** The zoom continuum has an unmodeled seam at every zoom boundary.

**Proposed fix.** Define a revisioned many-to-many provenance map:
`ModelNodeUid -> SemanticOpUid -> (planFingerprint,nodePosition,outputIndex) -> IslandUid
-> ScheduleEntityUid`. Preserve origin sets through fusion and lowering. Ledger operands
use UIDs plus a base revision, never array positions or hashes. Specify invalidation and
rebase behavior when any mapping is regenerated.

### R12 — “one predicate language” is absent from the sibling schedule design

> “ONE predicate language… shared verbatim with optimizer param-groups and (later)
> structural schedules”
> — `model-editor-charter.md:46-52`

**Attack.** The schedule document never imports the predicate grammar, gives moves no
target-selection representation, and instead speaks of regions, operands, adjacent
islands, and pattern-matched lemmas. Those are exactly where a second selector language
will emerge. Conversely, optimizer predicates over parameter rank cannot automatically
name schedule loops, values, roles, or origin sets. “Verbatim” is an assertion, not a
composition design.

**Severity: design-round.** The charter declares its own likely review blocker.

**Proposed fix.** Define one typed predicate AST now with explicit domains and shared
combinators, not one untyped string grammar. State which leaves are valid for model
instances, parameters, semantic ops, islands, and schedule entities. Include type errors
and serialization in M0/P0 gates.

### R13 — the two “realizers” do not consume the same kind of object or share a capability contract

> “the model editor's PyTorch-emit is a realizer of the same shape”
> — `schedule-state-design.md:238-240`
>
> “PyTorch emit is a realizer with a capability profile — same registry shape as
> WGSL/Triton”
> — `model-editor-charter.md:145-148`

**Attack.** WGSL/Triton realize a ScheduleState into a kernel. PyTorch emit realizes only
model strata 1–2 and explicitly cannot represent schedules (`model-editor-charter.md:58-62`).
No shared input interface, version coordinate, refusal schema, verification harness, or
composition rule is defined. Reusing a four-field record name does not make the zoom
regimes compositional.

**Severity: design-round.** It invites a false generic abstraction that adds code without
shared semantics.

**Proposed fix.** Either drop the “same registry” claim or define a genuinely generic
`Realizer<Input,Artifact,Capability,Receipt>` protocol with versioned capabilities and
typed refusal. PyTorch emit and kernel emit should be separate instances; neither should
pretend to realize strata it treats as opaque.

### R14 — the model charter's own candidate acceptance set contradicts zero-change weight semantics

> “Every structural edit changes model behavior by ZERO at the moment of application”
> — `model-editor-charter.md:64-67`
>
> “candidate six: skip-connection reroute, QK-norm, ReLU², zero-init residual
> projections, untied embeddings, attention-window predicate — each
> function-preserving at apply”
> — `model-editor-charter.md:111-115`

**Attack.** QK normalization, replacing an activation with ReLU², restricting an
attention window, and rerouting a skip are not function-preserving for a general trained
model. Weight copying cannot make those operators identical. Only untie-by-copy and a
carefully zero-gated residual insertion have a generic identity point. The three-month
claim is therefore impossible as stated, or gameable by testing a degenerate input/model
where the changed path happens not to affect logits.

**Severity: blocker.** M3 and the headline claim demand mutually exclusive results.

**Proposed fix.** Split edits into `functionPreservingMorph` and `behaviorChangingEdit`.
Require a constructive morphism for the former (parameter mapping plus a proof obligation
over the supported operator contract). Permit the latter with an explicit preview/commit
and measured before/after behavior. Revise the candidate six accordingly; do not call
activation/window/reroute edits preserving.

### R15 — `duplicate = SHARE` and `insert = zero-init` do not define a function-preserving graph edit

> “duplicate | SHARE (weight tying; reversible)”
> — `model-editor-charter.md:69-74`

**Attack.** Sharing weights says nothing about how the duplicate is connected. Executing
the same block twice in series, adding a duplicate branch, or replacing one block with
two tied blocks generally changes the function. Similarly, zero-initializing an output
projection preserves only a specified residual topology; a generic inserted branch/block
may affect normalization, control flow, or side effects before that projection.
“Reversible” also fails once either shared use has accumulated distinct optimizer intent
or a later untie occurs.

**Severity: blocker** for weight correctness.

**Proposed fix.** Define graph rewrite schemas, not verbs: exact matched subgraph,
replacement graph, parameter/alias map, optimizer-state map, identity equation, and
inverse preconditions. Refuse “duplicate” or “insert” without a named schema.

### R16 — “true bijection with PyTorch” is broader than the fenced input language and is not tested by running emitted code

> “strata 1–2… PyTorch has the concepts → true bijection, cheap emit”
> — `model-editor-charter.md:58-62`

**Attack.** PyTorch modules can contain arbitrary Python control flow, hooks, mutation,
custom autograd, dynamic parameter creation, aliasing, buffers, and state-dict hooks.
The charter refuses arbitrary-source and `torch.export` ingest, so it has no inverse from
PyTorch and cannot claim a PyTorch bijection. At most it can claim a bijection with a
defined emitted subset. “Emitted PyTorch runs” proves neither inverse preservation nor
canonicality.

**Severity: design-round.** The per-stratum contract is unfalsifiable at its current
domain.

**Proposed fix.** Define `TorchlettePyTorchSubset` syntactically and semantically. Gate
`canonicalize(emit(G)) == G`, alias-group equality, state-dict round-trip, and independent
forward parity. Say “injective emit into this subset” unless a supported inverse parser
actually exists.

## Missing live-edit and persistence failure modes

### R17 — “generation boundary” is not a transaction or a concurrency protocol

> “edits apply at generation boundaries via the tape's invalidate→re-record path”
> — `model-editor-charter.md:84-93`

**Attack.** The boundary is undefined: next token, end of prompt, queue flush, or next
request. Two browser edits can race with each other and with a streaming generation.
There is no base revision, operation ID, compare-and-swap, cancellation rule, or commit
point. “Every cache misses naturally” is also false as a safety argument: this repo has
module-local pipeline, tuning, bind-group, config-buffer, and authored-kernel caches not
all keyed solely by plan fingerprint. Invalidation can occur while GPU work still owns
buffers; `CLAUDE.md:119-121` requires fence-gated destruction and `canRecycle`, but the
charter never quiesces or defers teardown.

**Severity: blocker.** This is a direct path to mixed-revision execution and use-after-
submit destruction.

**Proposed fix.** Requests carry `{schemaVersion, baseModelRevision, opId}` and commit by
CAS at a defined generation epoch. Freeze one immutable model snapshot per generation;
queue or cancel edits; quiesce/fence before state demotion; defer buffer destruction; and
publish the new revision only after graph validation, weight mapping, re-record, and
smoke execution all succeed. Reject stale edits with the current revision.

### R18 — checkpoint binding has no generator/schema version, UID manifest, or alias validation

> “checkpoints BIND to generators (safetensors ingest ≈ free)”
> — `model-editor-charter.md:34-41`

**Attack.** A checkpoint outlives generator code. Renamed/split parameters, changed
rotary convention, tied-versus-untied embeddings, added buffers, dtype/quantization
changes, or a new override expansion can all preserve plausible names/shapes while
changing semantics. The stated refusal of heuristic unknown ingest does not protect a
known family from version drift. No manifest binds checkpoint tensors to UIDs or alias
groups, and no migration/version policy fails closed.

**Severity: blocker.** Silent wrong-weight binding is the charter's own worst failure
class.

**Proposed fix.** Require a checkpoint manifest containing model-family ID, generator
schema/version digest, parameter UID, expected path, shape/dtype, alias group, semantic
role, and required migration chain. Validate all fields and known-logit fixtures before
opening. Refuse unknown versions; migrations produce a new auditable ledger entry.

### R19 — undo across re-record, training, generator edits, and overrides has no defined inverse state

> “The ledger triple-pays: undo/redo, lineage… and meaningful model-diff.”
> — `model-editor-charter.md:53-57`
>
> “Every state in the undo stack carries its measurements”
> — `schedule-state-design.md:206`

**Attack.** The islands contract says plan-change events invalidate the editor's undo
stack (`examples/schedule-editor/contract.md:95-99`), while the sibling docs sell one
ledger across zoom levels. After re-record, schedule operands identified by plan position
are stale. After training, undoing an insertion discards learned weights; undoing an
untie must decide which diverged copy survives; undoing a generator edit must reconcile
instance overrides created against the newer definition. No inverse payload, snapshot
retention budget, rebase rule, or cross-stratum transaction is specified.

**Severity: blocker** for the shared-ledger claim.

**Proposed fix.** Separate immutable semantic edit history from ephemeral realization
history. Store revisioned UID operands, inverse payloads and retained weight/optimizer
snapshots where an inverse loses information. Define override precedence and three-way
rebase. Invalidate only realization commands on re-record; semantic undo must regenerate
against UIDs or refuse with a stable conflict reason.

### R20 — measurements can be attached to the wrong state and “isolation” can reverse the ranking

> “Measured tier (~100ms on settle): island-in-isolation bench at real shapes”
> — `schedule-state-design.md:202-205`

**Attack.** An async timing result can return after another edit and be stored on the
wrong undo state. A schedule's performance depends on neighboring kernels, L2 warmth,
occupancy competition, dispatch overhead, fusion boundaries, power state, device,
compiler version, and input distribution; isolation deliberately removes several of
those factors. “Every state carries its measurements” without a measurement identity
turns stale numbers into authoritative-looking data.

**Severity: design-round.** It can drive the editor and autotuner toward regressions.

**Proposed fix.** Key every request/result by full state hash, partition hash, model
revision, shape/data case, device/driver, realizer/compiler version, and benchmark
protocol. Discard late results whose base revision is no longer current. Label isolated
and in-context measurements separately; require an in-context confirmation before a
winner enters the registry.

## Acceptance criteria that can pass while their claim is false

### R21 — P0's prerequisite is contradicted by the named #43 source of truth

> “#43 recorded-path deletions land FIRST (one codegen path for the null differential
> to hold invariant, not two)”
> — `schedule-state-design.md:165-166`

**Attack.** The final #43 map says the opposite: recording remains load-bearing for
uncovered plans including strided views, chunked buffers, typed-buffer ops, copy-on-write,
and contiguous-copy prologues; deleting it is unsafe until 4.4 coverage closes
(`stage4-compile-from-ir.md:2177-2197, 2219-2228`). The draft treats a stopped prerequisite
as landed and then claims P0 is independently shippable.

**Severity: blocker.** P0 has no valid entry path in the actual repository state.

**Proposed fix.** Make 4.4 full generator coverage an explicit predecessor with an
executable coverage census, or design the null differential across both surviving paths.
Do not claim #43 deletion or one codegen path until the final map's bail classes are zero.

### R22 — byte-identical P0 proves an adapter, not reification, single ownership, or grammar sufficiency

> “Derive ScheduleState from every existing tile-IR kernel; regenerate; byte-diff…”
> — `schedule-state-design.md:181-183`

**Attack.** An implementation can put `opaqueGeneratorId` in `SkeletonInstance`, retain
all old TypeScript structure, replay the old generator, and pass every byte/hash test.
It can also encode a lossless AST dump rather than the seven-move grammar. Neither result
earns the claimed deletions or makes ScheduleState authoritative. The acceptance list
does not inspect ownership until a vague “leveling review,” and the weight-norm gate lets
growth pass if the campaign “explains why not” (`schedule-state-design.md:230-231`).

**Severity: blocker.** The principal acceptance gate is gameable by the architecture the
document explicitly calls failure.

**Proposed fix.** Add structural gates: old generators deleted or reduced to semantic
builders; no schedule-bearing fields below the object; every corpus state serialized
using only the declared schema/moves/atoms/lemmas; no opaque body except a named authored
set; and a fixed net-deletion threshold or explicit design re-review rather than an
explanation waiver.

### R23 — per-move numerical differential does not establish legality and misses the repository's activation-threshold lesson

> “every move application numerically matched… on real inputs”
> — `schedule-state-design.md:225-227`

**Attack.** “Real inputs” has no shape/dtype/boundary/adversarial corpus, tolerance,
reference authority, repeated-run policy, or atomic-race treatment. Finite examples can
miss tail masks, odd dimensions, NaNs/infinities, aliasing, zero-size cases, subgroup
width changes, and nondeterministic atomics. It also does not require the multi-step
trajectory that `CLAUDE.md:108` says is necessary to cross compiled-plan activation.
Comparing each move only to the preceding optimized state lets an early common error
persist through the whole derivation.

**Severity: blocker** for a correctness proof.

**Proposed fix.** Compare every state independently to the semantic reference and across
both lowered/compiled paths. Pre-register randomized boundary-shape/dtype suites,
adversarial numeric values, device classes, tolerances, repeat counts, and trajectory
length. Add metamorphic coverage for bijective moves and race-specific atom tests.

### R24 — P2's performance factor and M5's “visible loss effect” can be chosen after the system is known

> “within a stated factor (set before measuring)”
> — `schedule-state-design.md:190-192`
>
> “the edit's effect on the loss curve is visible”
> — `model-editor-charter.md:117-119`

**Attack.** “Before measuring” still permits a 100× factor after implementation quality is
known. No device, shapes, sequence lengths, precision, warmup, statistical estimator, or
baseline version is fixed. “Visible” can be noise, and one training run has no seed or
effect threshold. Both gates can pass while performance or learning impact is null.

**Severity: design-round.** These are demonstrations, not falsifiable gates.

**Proposed fix.** Pre-register before implementation: exact hardware/software,
representative shape set, authored commit/hash, warmup/timing protocol, numeric tolerance,
maximum slowdown per case plus geometric mean, training seeds, confidence interval, and
minimum loss/effect size. A failed case cannot be averaged away silently.

### R25 — P4 can empty `authored` by relabeling kernels as atoms and cannot prove “state of the art” from two local kernels

> “Exit: the authored set is empty or atoms-only… executable form of the corpus claim
> ‘the closure contains the state of the art’”
> — `schedule-state-design.md:212-217`

**Attack.** The atom admission rule has no syntactic size or semantic primitiveness test.
Any resistant attention/Adam kernel can be wrapped as an atom, making the authored set
empty without increasing grammar closure. Even honestly re-deriving two in-repo kernels
proves only those two, not “state of the art.” R4 already supplies a published,
WGSL-expressible counterexample outside that gate.

**Severity: blocker** to the completeness claim.

**Proposed fix.** Define atom admissibility mechanically (single primitive effect or
hardware intrinsic; no composite loops/dispatches/whole algorithms), require reviewer
approval for additions, and maintain an external published-kernel conformance corpus.
Rename P4 to “local self-hosting” unless that corpus passes. Grammar completeness remains
false while R4 is unrepresented.

### R26 — model parity gates can share the same wrong binding and exact-zero on one probe is both too weak and too strict

> “M1… re-emits the SAME forward as the shipped implementation”
> — `model-editor-charter.md:130-132`
>
> “every edit function-preserving at apply (logit diff == 0 pre-training)”
> — `model-editor-charter.md:135-136`

**Attack.** If the generator, checkpoint binder, in-tab executor, and emitter all consume
the same mistaken UID/path map, same-path parity agrees on the wrong model. A single logit
probe can also miss a changed branch. Conversely, a legitimate algebraic identity may
change floating-point reduction order and fail exact bit equality while preserving the
declared real-valued function within tolerance. The gate neither independently verifies
weights nor defines the domain over which “function” is quantified.

**Severity: blocker** for silent-wrong-weight defense.

**Proposed fix.** Add independent reference checkpoints/logits from upstream PyTorch,
full parameter value and alias-group comparison by manifest, multiple seeded/adversarial
inputs, intermediate activation checkpoints, and edit-specific tolerances. For claimed
exact morphisms, separately assert the parameter mapping and graph identity equation;
do not infer them from one output.

### R27 — cache identity is collision-prone and omits realizer/compiler/device versioning

> “`scheduleHash` (FNV over SkeletonInstance + DecorationVector)… and a realizer
> coordinate — one field”
> — `schedule-state-design.md:126-128`
>
> “regenerate-and-compare on sampled hits under STRICT”
> — `schedule-state-design.md:132-133`

**Attack.** A single 32-bit-style FNV coordinate plus sampled comparison is not a
correctness guard: a collision on an unsampled production hit can execute the wrong
kernel. The live plan fingerprint uses two independent hashes and validates the secondary
(`fusion-detect.ts:1531-1534`), already stronger than the proposal. “Realizer” also does
not identify capability profile version, emitter/compiler version, target architecture,
device features, or driver—each can change realization and cached binaries while the
ScheduleState stays equal.

**Severity: blocker** for cache correctness and checkpoint/version drift.

**Proposed fix.** Key on a canonical serialization with a strong digest and verify full
canonical equality on every hit (sampling may audit regeneration, not content identity).
Add `{schemaVersion, capabilityProfileVersion, emitterVersion, compilerVersion,
targetArch, featureSet}`. Keep semantic schedule identity separate from realization and
artifact-cache identity.

## Missing evidence artifact

### R28 — the draft's defining schema evidence is not in the repository

> “Depends on… the ScheduleRecord spike (`docs/spike-schedule-record-findings.md`), the
> kernel-editor design corpus (containment analysis, 11-rung ladder).”
> — `schedule-state-design.md:3-4`

**Attack.** The named spike file and design corpus are absent from `docs/`. As a result,
`recolor`, `pack`, `role-partition`, the skeleton grammar, the rung boundary, the claimed
byte-identical lift, and the one-lemma sufficiency argument have no reviewable definitions
or evidence in this repo. Appendix A's split verdicts **A-R1–A-R5** are a direct result:
one cannot profile names whose state transition and postconditions are unstated.

**Severity: design-round.** Review cannot validate the central completeness premise from
the submitted sources.

**Proposed fix.** Check in the referenced artifacts or absorb their normative content
into this design: typed before/after schemas for every move, invariants, inverse data,
legality rules, and the complete corpus with expected derivations. Treat undefined moves
as refused until then.

## Surviving attack

The explicit refusal to infer unknown architectures from checkpoint names/shapes
(`model-editor-charter.md:34-41`) survives: it fails closed and does not claim evidence it
cannot obtain. It does not address R18's version drift for supposedly known generators.
