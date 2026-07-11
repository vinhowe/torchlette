# ScheduleState consumer-side findings

These findings came from instantiating `docs/schedule-state-design.md` §2 as a
consumer-facing JSON object and building the P3 static cost model. They are not
engine claims; they identify decisions the engine schema must settle before the
proposal becomes a wire contract.

## F1 — register residency contradicts the v1 memory-space set

The enumerated v1 memory-space axis is `{global, workgroup-shared}` and reserves
`register-explicit` for later. Section 6 simultaneously requires operand
residency to say register versus shared from P0. Both honest instances need
register residency: matmul accumulators and attention's Q/softmax/output state.
The proposal schema therefore admits `register` now. The design should either
add register to the v1 space axis or explicitly declare residency a different
logical-location vocabulary from addressable memory spaces.

## F2 — `AlgorithmRef` is insufficient for a cost model

An algorithm identity does not provide concrete M/N/K or B/H/N/D, tensor
dtypes, batch multiplicity, or output dtype. Bytes moved, FLOPs, dispatch grid,
and roofline position cannot be computed without them. The proposal adds an
`algorithm.workload` binding. The object design must decide whether workload is
part of ScheduleState identity, request context supplied beside it, or an
island/plan reference resolved by the engine. Putting shape in both an island
and ScheduleState would violate the leveling rule.

## F3 — authored state needs a discriminated opaque skeleton

The object requires a `SkeletonInstance`, while authored means the skeleton is
opaque and macro moves are unavailable. A boolean alone leaves consumers
guessing which skeleton fields may be absent. The schema uses
`skeleton.visibility: "derived" | "opaque"`; opaque requires a kernel reference
and refusal reason and forbids loop/staging/role data. This should be explicit
in the canonical object.

## F4 — the skeleton grammar has names but no serial grammar

“Explicit loop nest” does not specify bound-expression syntax, loop identity,
parallel versus sequential semantics, or how grid axes nest. The instance uses
human-readable symbolic expressions such as `ceil(M / tile.m)`. Those are good
for display but unsuitable for hashing, legality, or rewriting. The engine
needs a small typed expression AST or references into an existing expression
IR, with canonical ordering.

## F5 — role partition lacks a participant language

The design says which invocations do which part, but gives no representation
for invocation sets, subgroup gates, cooperative striding, or ownership of a
thread tile. The proposal's `participants` and `responsibility` are descriptive
strings and therefore display-only. Macro `role-partition` moves require a
typed participant/predicate grammar before this field can be executable.

## F6 — staging edges do not cover stores or allocation lifetime

The named direction `global→shared→register` describes input staging, but an
honest kernel also has register→global output flow. It is unclear whether stores
are staging edges, algorithm effects, or implicit. Separately, an edge does not
say whether allocations overlap. Attention loads K into shared and reuses that
same allocation for V; summing edges would double shared-memory use. The cost
model currently knows this kernel-family fact. The canonical skeleton needs
allocation identity plus lifetime/reuse, or the realizer cost model cannot be
derived only from state.

## F7 — decoration keys are not enumerated

The design enumerates axes conceptually but does not define canonical tile-key
names across skeleton families (`m/n/k`, `qRows/kvRows/headDimension`, thread
tiles), numeric domains, or cross-field divisibility. The schema permits a
positive-integer map, which is extensible but typo-prone and weak for UI
generation. Each skeleton grammar needs a decoration descriptor/capability
profile that owns names, domains, defaults, and legality relations.

## F8 — degenerate axes need value and capability metadata

`pipelineDepth: 1` faithfully preserves the v1-degenerate value, but the object
alone cannot explain whether 2 is core-illegal, unsupported by this realizer,
or merely absent from the current device profile. The UI locks it with the
reason “WGSL realizer pins this axis to 1.” More generally, selected values
belong in ScheduleState while allowed/refused values and reasons belong in the
realizer capability profile. That profile needs a machine-readable shape.

## F9 — subgroup appears in three different roles

Subgroup is a gated thread-hierarchy level, a feature-gated atom family, and an
existing matmul decoration (`useSubgroups`). The design does not state when an
enabled subgroup level implies an atom, which subgroup operations count as
atoms, or how subgroup size enters identity. The instances omit subgroup
because their real kernels do not request it; the schema can represent the
level but cannot validate the relationships.

## F10 — atoms need placement, not only membership

An atom with footprint/sync/cost is first-class, but a root `atoms[]` list does
not say where it occurs in the loop nest, which role invokes it, its operands,
or its multiplicity. Cost and legality require placement. A future schema
should make an atom invocation a skeleton node referencing a catalog entry;
the catalog entry itself should remain single-sourced.

## F11 — admitted lemmas need provenance in the state

The authored forward attention body already uses online-softmax rescaling.
Without a lemma reference, two semantically different derivations can have
identical structural decorations while carrying different proof obligations.
The proposal includes `admittedLemmas[]`, although the three-field object in §2
does not. The design should decide whether lemma application is part of the
algorithm DAG, skeleton provenance, or the ScheduleState ledger; it should not
be duplicated.

## F12 — schedule hashing is underspecified

Section 5 requires FNV over skeleton plus decorations, but JSON object order,
symbolic-expression normalization, numeric encoding, authored opaque fields,
atoms, lemmas, algorithm/workload, and the realizer coordinate are not given a
canonical byte encoding. The frontend uses sorted-key canonical JSON plus
FNV-1a only as a declared proposal hash for bench stubbing. It must not become a
cache identity accidentally. The engine contract needs a versioned canonical
encoding and test vectors, as the islands boundary hash already has.

## F13 — WebGPU limits cannot produce true occupancy

`navigator.gpu` exposes per-workgroup invocations and storage, not registers per
thread, shared memory per compute unit, maximum resident invocations, or
resident workgroup slots. The UI therefore labels occupancy a proxy and combines
adapter limits with documented configurable fallback architecture constants.
The realizer cost model should declare which inputs are measured adapter facts,
device-database facts, or heuristics; otherwise a precise-looking percentage is
misleading.

## F14 — decoration edits need a legality response before benchmarking

Numeric fields can create non-integral thread tilings, oversized workgroups, or
storage overflow continuously while editing. Static cost can show the pressure,
but a bench request needs a distinct server-authoritative realization/legality
result. The proposed measured contract therefore allows a refusal with a stable
code and reason rather than treating every state hash as benchmarkable.
