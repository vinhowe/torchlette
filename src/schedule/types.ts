/**
 * Schedule State — the intra-kernel stratum as data (campaign P0).
 *
 * THE TYPES ARE THE SPEC. No consumers are wired to these in P0 beyond the
 * elementwise walking skeleton (src/schedule/elementwise-skeleton.ts). This
 * file reifies the schema of docs/schedule-state-design.md (v2) §2–§3 and
 * docs/design-amendment-round-1.md (rulings S1/S2/S3, R3/R4/R5/R12/R27, F-series).
 *
 * ------------------------------------------------------------------------
 * THE ADVERSARY THESE TYPES MUST DEFEAT (redteam-round-1 R6 + R22)
 * ------------------------------------------------------------------------
 * R22's cheat: an implementation that puts `opaqueGeneratorId` on a skeleton,
 * replays the old TypeScript generator, and passes every byte/hash test while
 * owning nothing; OR encodes a lossless AST dump instead of the move grammar.
 * R6's cheat: "derive ScheduleState from every kernel" by observing the live
 * imperative tile-IR structure and replaying it through the same generator —
 * byte-identical, ownership untouched.
 *
 * The schema makes BOTH unrepresentable, structurally:
 *   - There is NO `opaqueGeneratorId`, NO `generatorFn`, NO callback field
 *     anywhere in `SemanticSchedule` / `BackendRequests` / `RealizationReceipts`.
 *     The ONLY opacity permitted is `OpaqueSkeletonRef` (§6 authored hatch),
 *     which REQUIRES a kernel reference + a refusal reason and FORBIDS carrying
 *     loop / staging / role data (F3 discriminated skeleton). An authored kernel
 *     cannot masquerade as a derived one.
 *   - There is NO `astDump: unknown`, NO `wgsl: string`, NO free-form string
 *     body. Bodies are `SemanticBodyNode` trees whose leaves reference a
 *     single-sourced op/atom catalog by NAME (`OpCatalogRef`), never emit code.
 *     A lossless AST dump has nowhere to live.
 *
 * See docs/schedule-state-design.md §7 P0 (d)/(e) and §8 gate 1 for the
 * executable enforcement of these structural properties (the schema-only
 * serialization gate + the no-second-owner assertion).
 */

// ============================================================================
// 0. Foreign keys and entity UIDs (§2, §9 provenance spine)
// ============================================================================

/**
 * A branded string UID. Nominal typing prevents accidental cross-assignment
 * (a LoopUid must never be usable where a ValueUid is expected). The brand
 * lives only in the type system; at runtime these are plain strings.
 */
type Brand<K, T extends string> = K & { readonly __brand: T };

/** Identity of one loop in the semantic loop nest (F4). */
export type LoopUid = Brand<string, "LoopUid">;
/** Identity of one named value / intermediate (F6). */
export type ValueUid = Brand<string, "ValueUid">;
/** Identity of one grid/iteration axis (§2.4). */
export type AxisUid = Brand<string, "AxisUid">;
/** Identity of one admitted lemma (§3.4). */
export type LemmaUid = Brand<string, "LemmaUid">;
/** Identity of one placed atom instance in the skeleton (§3.3, F10). */
export type AtomUid = Brand<string, "AtomUid">;
/**
 * FOREIGN KEY into the semantic graph (R8): the object STORES this key and
 * DECLINES to identify its contents into semantic identity. It is NOT the
 * region's contents copied inline. This replaces the deleted `AlgorithmRef`.
 */
export type SemanticRegionUid = Brand<string, "SemanticRegionUid">;
/**
 * Provenance-spine entity UID for a schedule entity (loop, value, edge, role) —
 * so a scheduleHash is CONTENT identity while UIDs are OPERAND identity for
 * edits (§9, R11).
 */
export type ScheduleEntityUid = Brand<string, "ScheduleEntityUid">;

/** A proof-obligation identity a lemma discharges (F28). Jam→lemma binding is
 *  by this ID, NEVER by matching human refusal text. */
export type ObligationId = Brand<string, "ObligationId">;

// ============================================================================
// 1. The typed predicate AST (R12 + F5) — ONE tree, shared with the editor
// ============================================================================

/**
 * ONE typed predicate AST, shared verbatim with the model editor's override
 * selection and optimizer param-groups (charter §2). A typed TREE with explicit
 * leaf domains — NOT an untyped string grammar (that would be an AST-dump
 * escape hatch; R12 forbids it). Loop-bound expressions (F4), role participant
 * sets (F5), and `checkedAffine` grid maps (§2.4) are all nodes of THIS one AST.
 *
 * P0 encodes the SCHEDULE-context leaves needed by the elementwise family plus
 * the shape of the full leaf/combinator set. The model-context leaves (model
 * instances, params, semantic ops, islands) attach as sibling `PredicateLeaf`
 * variants in the sibling charter; they are elided here, not forbidden.
 */
export type PredicateAstNode = PredicateLeaf | PredicateCombinator | AffineExpr;

/** Leaf domains valid in the SCHEDULE context (§2.5). Each leaf names WHICH
 *  schedule entity it references — never a free string. */
export type PredicateLeaf =
  | { kind: "loopRef"; loop: LoopUid }
  | { kind: "valueRef"; value: ValueUid }
  | { kind: "axisRef"; axis: AxisUid }
  | { kind: "roleParticipant"; role: RoleName }
  /** An integer literal (loop extents, factors). Explicit numeric encoding for
   *  canonical serialization (R27). */
  | { kind: "intLit"; value: number }
  /** A named uniform value delivered at dispatch (e.g. "total_elements"). This
   *  is a semantic reference to a dispatch scalar, NOT a baked constant. */
  | { kind: "uniformRef"; name: string };

/** Shared combinators (and/or/not, comparison, range, set-membership). */
export type PredicateCombinator =
  | { kind: "and"; terms: PredicateAstNode[] }
  | { kind: "or"; terms: PredicateAstNode[] }
  | { kind: "not"; term: PredicateAstNode }
  | {
      kind: "cmp";
      op: "<" | "<=" | "==" | ">=" | ">" | "!=";
      lhs: PredicateAstNode;
      rhs: PredicateAstNode;
    }
  | {
      kind: "range";
      value: PredicateAstNode;
      lo: PredicateAstNode;
      hiExclusive: PredicateAstNode;
    }
  | { kind: "member"; value: PredicateAstNode; set: PredicateAstNode[] };

/**
 * A checked affine / index expression node (§2.4 `checkedAffine`, F4 loop bounds
 * like `ceil(N/tile)`). `ceil(M/tile.m)` is a DISPLAY projection of this AST,
 * never the hashed form (§2.1).
 */
export type AffineExpr =
  | { kind: "affineAdd"; terms: AffineExpr[] }
  | { kind: "affineMul"; factors: AffineExpr[] }
  | { kind: "affineCeilDiv"; num: AffineExpr; den: AffineExpr }
  | { kind: "affineLeaf"; leaf: PredicateLeaf };

// ============================================================================
// 2. Typed sync relations (R3) and the address-space / order sets
// ============================================================================

/**
 * Explicit enumerated sets, never booleans/absences (R3, A-R10). WebGPU exposes
 * a subset; Triton exposes cta/gpu/sys visibility + acquire/release separately.
 * A single `{workgroup}` cannot distinguish them, so the sets are named.
 */
export type AddressSpace = "global" | "shared" | "register";
export type MemoryOrder =
  | "relaxed"
  | "acquire"
  | "release"
  | "acqRel"
  | "seqCst";
export type VisibilityScope =
  | "invocation"
  | "subgroup"
  | "workgroup"
  | "device"
  | "system";

/** A half-open interval over a value's uses, identified by loop position. */
export interface UseInterval {
  readonly fromLoop: LoopUid;
  readonly toLoop: LoopUid;
}

/**
 * Typed synchronization relations (R3). NOT one level chain, one barrier, or a
 * set of simultaneously-selected labels. Thread hierarchy is a backend
 * capability GRAPH, NOT values stored in every schedule — "grid barrier" is a
 * capability-graph ABSENCE, not a reserved degenerate value.
 */
export type SyncRelation =
  | {
      kind: "memoryEffect";
      space: AddressSpace;
      value: ValueUid;
      interval: UseInterval;
    }
  | {
      kind: "barrier";
      participants: ParticipantSet;
      spaces: AddressSpace[];
      convergence: ConvergenceFact;
    }
  | { kind: "atomic"; order: MemoryOrder; visibility: VisibilityScope };

/** Convergence guarantee for a barrier (uniform vs re-convergent control flow). */
export type ConvergenceFact = "uniform" | "reconvergent" | "unknown";

// ============================================================================
// 3. Roles and participant sets (§2.5 role-partition, F5)
// ============================================================================

export type RoleName = Brand<string, "RoleName">;

/**
 * A typed participant set (F5): invocation sets, subgroup gates, cooperative
 * striding, thread-tile ownership — a typed participant/predicate grammar, NOT
 * descriptive strings. The participant condition is a predicate-AST node so it
 * shares the ONE AST.
 */
export interface ParticipantSet {
  readonly role: RoleName;
  readonly condition: PredicateAstNode;
}

// ============================================================================
// 4. ProgramGridMap — the program-map move's payload (R4, §2.4)
// ============================================================================

/**
 * A canonical bijection-valued map from a linear program id to a work
 * assignment (R4). Legality = one-to-one in-bounds coverage over the launch
 * domain, checked at the move seam (not encoded in the type; a value that fails
 * bijection is REFUSED, never silently accepted). This is NOT `recolor` (R4:
 * stretching recolor to "change any mapping" makes it an untyped escape hatch).
 */
export type ProgramGridMap =
  | { kind: "identity" }
  | { kind: "swap"; axes: readonly [AxisUid, AxisUid] } // repo's CodegenOptions.swapGrid
  | { kind: "grouped"; groupAxis: AxisUid; groupSize: number } // Triton grouped-matmul L2 reuse
  | { kind: "checkedAffine"; expr: AffineExpr }; // a checked affine/index expression

// ============================================================================
// 5. Named value lifetimes and staging (§2.1, F6)
// ============================================================================

/**
 * The TIER-KIND a value stages through (§2.1 staging intent). This is a
 * SEMANTIC intent / no-materialization fact — the physical residency REQUEST a
 * realizer may or may not honor lives in `BackendRequests.placementPreferences`
 * (F1: register residency is intent HERE and a preference THERE, never a WGSL
 * determination).
 */
export type StagingTier = AddressSpace;

/**
 * A strided read descriptor: HOW a value is sampled out of its source tier. The
 * strides SPECIALIZE codegen (template identity); the base offset is a VOLATILE
 * per-replay scalar (task #71 — never a baked literal), so it is carried as a
 * `uniformRef` predicate leaf, not a number.
 */
export interface StridedAccess {
  /** The logical iteration (index) shape the flat program id addresses. */
  readonly indexShape: readonly number[];
  /** Per-axis strides into the source buffer (specialize codegen). */
  readonly strides: readonly number[];
  /** The element offset, delivered as a named volatile uniform (#71). */
  readonly offsetUniform: string;
}

/**
 * A named value in the schedule (F6): each has a ValueUid + a
 * ScheduleEntityUid, an allocation identity (which tier it lives in), and a
 * lifetime/reuse edge so the cost model can see shared allocations (attention's
 * K and V sharing one shared-memory allocation — summing edges would
 * double-count). Stores are EDGES, not implicit.
 */
export interface NamedValue {
  readonly uid: ValueUid;
  readonly entity: ScheduleEntityUid;
  /** The tier this value is allocated in (its residency INTENT). */
  readonly allocation: StagingTier;
  /** The dtype the value carries in this schedule (a semantic fact — a cast
   *  produces a value of a DIFFERENT dtype, expressed via a `cast` body op). */
  readonly dtype: ValueDtype;
  /**
   * If this value is loaded from an outer tier (e.g. `global→register`), the
   * strided access that produces it. Absent for values PRODUCED inside the loop
   * body (compute results, the store target).
   */
  readonly load?: StridedAccess;
  /**
   * Lifetime/reuse: the ValueUid whose allocation this value SHARES, if any
   * (F6 — a reuse edge, not a summed allocation). `null` = owns its allocation.
   */
  readonly aliasOf: ValueUid | null;
}

export type ValueDtype = "f32" | "f16" | "i32" | "u32";

/**
 * A no-materialization (fusion) edge (§2.1): the semantic no-store contract
 * (A-R2 `stream`'s determination half), SEPARATE from any residency request.
 * When present between a producer and consumer value, the intermediate is
 * produced/consumed without a global store.
 */
export interface NoMaterializationEdge {
  readonly producer: ValueUid;
  readonly consumer: ValueUid;
  readonly acrossLoop: LoopUid;
}

// ============================================================================
// 6. The semantic body — op catalog references, NEVER emitted code (§3.3)
// ============================================================================

/**
 * A reference to a SINGLE-SOURCED op catalog entry by NAME (§3.3 mechanical
 * admissibility). The catalog itself (the WGSL/BlockExpr builder) lives once in
 * the backend (`fusion-tile-ir.ts applyFusedOp`); the schedule stores only the
 * name + the input value list. This is what makes an AST dump unrepresentable:
 * there is no place to inline emitted code — a body node NAMES an op, it does
 * not carry its lowering.
 */
export interface OpCatalogRef {
  /** The catalog op name, e.g. "add", "mul", "select", "cast_f16". A name the
   *  single-source op catalog can resolve; an unknown name is REFUSED at apply,
   *  it does not fall through to an opaque body. */
  readonly op: string;
}

/**
 * A semantic body node for the ELEMENTWISE family: a tree over loaded values
 * and op-catalog applications, terminating in a store to an output value.
 * Leaves are `valueRef` (a loaded/produced NamedValue). This is the "algebraic
 * expression DAG" the redteam contrasts against the imperative tile-IR — it
 * carries NO thread ids, NO barriers, NO placement.
 */
export type SemanticBodyNode =
  | { kind: "value"; value: ValueUid }
  /** A literal scalar in the body (e.g. the `0` in `cond != 0`). Explicit
   *  numeric encoding for canonical serialization. */
  | { kind: "literal"; dtype: ValueDtype; value: number }
  /** An op-catalog application over child expressions (unary/binary/n-ary). */
  | { kind: "apply"; catalog: OpCatalogRef; args: SemanticBodyNode[] };

/**
 * A semantic atom placement (F10): a skeleton node with a LoopUid/role
 * placement, operands, and multiplicity, REFERENCING a single-sourced catalog
 * entry. A root `atoms[]` bag cannot support cost or legality, so an atom is
 * PLACED. The elementwise family uses no atoms; the type is present so the full
 * set has a home (scatter-add composes around `atomicAdd<f32>` here).
 */
export interface AtomPlacement {
  readonly uid: AtomUid;
  readonly catalog: OpCatalogRef;
  readonly atLoop: LoopUid;
  readonly role: RoleName | null;
  readonly operands: readonly ValueUid[];
  readonly multiplicity: number;
}

/**
 * An admitted-lemma application (§3.4, F11/F27/F28). Carries its LemmaUid + the
 * proof-obligation ID it discharges + its FIRST-CLASS carried state. Two
 * derivations with identical structural decorations but different proof
 * obligations are DISTINGUISHABLE by this reference. The elementwise family
 * uses none; present for schema completeness (online-softmax lives here).
 */
export interface LemmaApplication {
  readonly lemma: LemmaUid;
  readonly obligation: ObligationId;
  /** Opaque-to-the-schedule carried-state handle. The state SCHEMA lives in the
   *  lemma library, not here; this is the instance's carried-state identity. */
  readonly carriedStateRef: string;
}

// ============================================================================
// 7. The loop nest as typed identity (§2.1, F4)
// ============================================================================

export type LoopKind = "parallel" | "sequential";

/**
 * One loop in the semantic nest (F4). Each carries a LoopUid, a bound
 * expression in the typed predicate AST, and a parallel|sequential kind;
 * grid-axis nesting is explicit. The human-readable `ceil(M/tile.m)` is a
 * DISPLAY projection of `bound`, never the hashed form.
 *
 * NOTE (S2): the loop-nest VIEW derives from (region × schedule) by a canonical
 * ordering rule — but thread ROLES, BARRIERS, and LANES do NOT derive; they are
 * schedule facts (roles via `roles`, barriers via `sync`). This loop object
 * carries only the derivable structural spine.
 */
export interface SemanticLoop {
  readonly uid: LoopUid;
  readonly entity: ScheduleEntityUid;
  readonly axis: AxisUid;
  readonly kind: LoopKind;
  /** The iteration bound as an AST node (F4). */
  readonly bound: AffineExpr;
  /** Child loops nested inside this one (grid-axis nesting explicit). */
  readonly children: readonly SemanticLoop[];
}

/**
 * The canonical ORDERING RULE tag (S2 / NCD F12): matmul's m→n vs n→m nesting
 * realizes the same decorated term with different locality, so partition labels
 * alone do not derive a unique nest. The ordering rule is an EXPLICIT part of
 * `SemanticSchedule`, not left to decoration-array order. For the elementwise
 * family the domain is flat, so the rule is `flat`.
 */
export type NestOrderingRule =
  | { kind: "flat" } // single iteration domain (elementwise)
  | { kind: "rowMajor"; axes: readonly AxisUid[] }
  | { kind: "colMajor"; axes: readonly AxisUid[] };

// ============================================================================
// 8. SemanticSchedule (SEMANTIC identity) — §2.1
// ============================================================================

/**
 * The facts that are the same computation-shape regardless of who realizes
 * them. Hashes into SEMANTIC identity (§5). Contains NO warp budget, NO
 * pipeline stages, NO WGSL geometry — those are `requests`/`receipts`.
 */
export interface SemanticSchedule {
  /** Logical block shapes — logical tensor extents (BLOCK_*-class), NOT
   *  per-thread tiles, NOT layouts, NOT instruction shapes (A-R1). */
  readonly blockShapes: readonly (readonly number[])[];
  /** The loop nest as typed identity (F4). Roots of the nest forest. */
  readonly loopNest: readonly SemanticLoop[];
  /** The canonical ordering rule the loop-nest VIEW derives under (S2/NCD F12). */
  readonly ordering: NestOrderingRule;
  /** The program-id → work bijection (R4). Semantic (changes traversal only). */
  readonly programGridMap: ProgramGridMap;
  /** Named value lifetimes (F6) — inputs, intermediates, outputs. */
  readonly values: readonly NamedValue[];
  /** No-materialization (fusion) edges — the semantic no-store contract. */
  readonly noMaterialization: readonly NoMaterializationEdge[];
  /**
   * The output store edges: which value is written to which output binding.
   * Stores are EDGES, not implicit (F6). For the elementwise family this is the
   * single store of the body result.
   */
  readonly stores: readonly StoreEdge[];
  /** The semantic body (§3.3) — the op DAG producing each stored value. */
  readonly bodies: readonly SemanticBody[];
  /** Executor role partitions (S2/NCD F13 — do NOT derive). Empty for
   *  elementwise (homogeneous invocation set). */
  readonly roles: readonly ParticipantSet[];
  /** Typed sync relations (R3). Empty for elementwise (no barriers). */
  readonly sync: readonly SyncRelation[];
  /** Placed semantic atoms (F10). Empty for elementwise. */
  readonly atoms: readonly AtomPlacement[];
  /** Admitted-lemma applications (F27/F28). Empty for elementwise. */
  readonly lemmas: readonly LemmaApplication[];
}

/** The store of a produced value into a named output value (an EDGE, F6). */
export interface StoreEdge {
  readonly source: ValueUid; // the produced body-result value
  readonly target: ValueUid; // the output NamedValue (allocation "global")
  readonly atLoop: LoopUid;
}

/** A body rooted at a produced value: `result = <expr over loaded values>`. */
export interface SemanticBody {
  readonly result: ValueUid;
  readonly expr: SemanticBodyNode;
}

// ============================================================================
// 9. BackendRequests (COMPILATION identity, NOT semantic) — §2.2
// ============================================================================

/**
 * Requests a realizer receives but does not have to honor physically. These
 * change the compiled artifact but NOT the computation's semantic identity
 * (`num_warps=4` and `num_warps=8` are the SAME semantic schedule). Value
 * domains are explicit sets, never booleans/absences (R3 spirit).
 */
export interface BackendRequests {
  /** warp / thread BUDGET — a budget, not a geometry (A-R7). The WGSL x/y/z is
   *  a receipt. `null` = realizer default. */
  readonly warpBudget: number | null;
  /** pipeline requests: `none | [{loopUid, loadGroupUids, requestedStages}]`
   *  (R2, verbatim). `none` is LEGAL and carries no fake degenerate value —
   *  there is no "pipeline depth pinned 1" fact. The realized stage count is a
   *  receipt (§2.3), never this. */
  readonly pipeline: PipelineRequest;
  /** operand-residency PREFERENCES, scoped to a value-use interval (A-R8:
   *  `refused` as a WGSL determination → can only ever be a request). */
  readonly placementPreferences: readonly PlacementPreference[];
  /** load/eviction/volatile CACHE hints (Triton tl.load cache modifiers; A-R14).
   *  Explicit set members, never a boolean. */
  readonly cachePolicy: readonly CacheHint[];
}

export type PipelineRequest =
  | { kind: "none" }
  | { kind: "staged"; entries: readonly PipelineEntry[] };

export interface PipelineEntry {
  readonly loop: LoopUid;
  readonly loadGroups: readonly ValueUid[];
  /** ≥ 2 for an entry to exist (§3.1); else the request is `none`. */
  readonly requestedStages: number;
}

export interface PlacementPreference {
  readonly value: ValueUid;
  readonly preferTier: StagingTier;
  readonly interval: UseInterval;
}

export type CacheHint =
  | { kind: "loadCache"; mode: "cached" | "streaming" | "lastUse" }
  | { kind: "evict"; mode: "normal" | "first" | "last" }
  | { kind: "volatile" };

// ============================================================================
// 10. RealizationReceipts (hashed into NEITHER identity) — §2.3
// ============================================================================

/**
 * Physical facts a realizer CHOOSES and REPORTS. NEVER part of any schedule
 * identity — the realizer's answer, keyed by measurement identity (§5, R20)
 * when measured. The realizer FILLS this; the schedule author never sets it.
 */
export interface RealizationReceipts {
  /** the actual WGSL dispatch geometry (R1: WGSL-only; A-R7). */
  readonly workgroup?: readonly [number, number, number];
  /** exact vec-load forms the compiler picked (array<vec4<T>>, ctx.loadVec4,
   *  shared-vec4 arrays — three different lowering choices; A-R6 vec width is
   *  `refused` as a portable fact). */
  readonly vecLoadForms?: readonly VecLoadForm[];
  /** the pipeline stages the realizer actually emitted (R2). */
  readonly realizedStages?: number;
  /** CASLoop vs NativeAtomic for atomicAdd<f32> (R5/R25/A-R12). */
  readonly atomRealizations?: readonly AtomRealization[];
  /** timings / occupancy proxies, keyed by §5 measurement identity (R20). These
   *  NEVER back-propagate into semantic or compilation identity. */
  readonly measurements?: readonly MeasurementRecord[];
}

export interface VecLoadForm {
  readonly value: ValueUid;
  readonly form: "scalar" | "vec2" | "vec4";
}
export interface AtomRealization {
  readonly atom: AtomUid;
  readonly realization: "CASLoop" | "NativeAtomic";
}
export interface MeasurementRecord {
  /** the full measurement-identity key (R20/F29) as a canonical string. */
  readonly key: string;
  readonly milliseconds: number;
  readonly occupancyProxy?: number;
}

// ============================================================================
// 11. ScheduleState — the three-tier object (§2, S1)
// ============================================================================

/**
 * A kernel's schedule as a first-class three-tier data object (§0 declaration).
 * Each tier hashes into a DIFFERENT identity; `region` is a FOREIGN KEY, NOT
 * payload (R8). No tier stores a fact another tier owns (§2.6 no-second-owner).
 */
export interface ScheduleState {
  readonly semantic: SemanticSchedule; // hashes into SEMANTIC identity
  readonly requests: BackendRequests; // hashes into COMPILATION identity, NOT semantic
  readonly receipts: RealizationReceipts; // hashes into NEITHER
  readonly region: SemanticRegionUid; // FOREIGN KEY into the semantic graph (R8)
}

// ============================================================================
// 12. The authored hatch (§6) — the ONLY permitted opacity, discriminated (F3)
// ============================================================================

/**
 * A DISCRIMINATED opaque skeleton (F3). `visibility: "opaque"` REQUIRES a kernel
 * reference + a refusal reason and FORBIDS loop/staging/role data (a boolean
 * alone leaves consumers guessing which fields are absent). This is the ONLY
 * place an un-re-derived kernel may hide — and it announces itself as authored,
 * so it can never masquerade as a derived `ScheduleState` (R22 structural gate:
 * opaque bodies ONLY in the named authored set).
 *
 * There is deliberately NO `generatorFn` / `opaqueGeneratorId` field: an opaque
 * skeleton names a kernel and a reason; it does not carry a replayable generator.
 */
export type Skeleton =
  | { visibility: "derived"; schedule: ScheduleState }
  | {
      visibility: "opaque";
      kernelRef: string;
      refusalReason: string;
      params: TypedParamSchema;
    };

/**
 * A typed parameter schema for an authored skeleton (R10 + F7): dependent
 * constraints, derived geometry, a capability predicate, and a canonical
 * cache-key encoder. Generic decorations on an opaque skeleton are REFUSED —
 * only DECLARED parameters are editable. The elementwise skeleton is `derived`,
 * so it never uses this; the shape is present for the full set.
 */
export interface TypedParamSchema {
  readonly params: Record<
    string,
    { readonly domain: readonly number[]; readonly default: number }
  >;
  /** cross-field dependent constraints as predicate-AST nodes. */
  readonly constraints: readonly PredicateAstNode[];
  /** the capability predicate (device/realizer must satisfy). */
  readonly capabilityPredicate: PredicateAstNode;
}

// ============================================================================
// 13. Move grammar (§3) — the EIGHT moves as typed before/after schemas (R28)
// ============================================================================

/**
 * The FENCED move set (§3): seven intra-schedule mutators on `SemanticSchedule`
 * plus `program-map`. `fuse` is NOT here (S3 — it is a composite transaction at
 * the islands altitude). New move kinds require a design amendment. Each move
 * carries INVERSE DATA (S3 / ownership-derivation's inverse-payload discipline)
 * so a provenance entry can invert it. Refused-at-the-seam moves are not
 * silently dropped (the ncd-surface "jam" UX).
 *
 * P0 defines the move UNION as the schema; the executable move algebra (partial
 * functions refusing on invariant violation) is P2 work.
 */
export type ScheduleMove =
  | { move: "tile"; loop: LoopUid; axis: AxisUid; factor: number }
  | { move: "stream"; value: ValueUid; loop: LoopUid }
  | {
      move: "recolor";
      value: ValueUid;
      column: number;
      tier: StagingTier;
      transitionRole: "materialization-boundary" | "external-transfer";
    }
  | {
      move: "pack";
      loops: readonly LoopUid[];
      kind: "map" | "concatenate" | "chunked-binding";
    }
  | { move: "role-partition"; loop: LoopUid; roles: readonly RoleName[] }
  | {
      move: "pipeline";
      loop: LoopUid;
      loadGroups: readonly ValueUid[];
      requestedStages: number;
    }
  | { move: "program-map"; map: ProgramGridMap };

/** The provenance record for one applied move: the move + its inverse data. */
export interface MoveProvenance {
  readonly move: ScheduleMove;
  /** what the inverse move needs (S3). Serialized canonically. */
  readonly inverseData: unknown;
}

// ============================================================================
// 14. The semantic IR node set (P0 deliverable (a)) — ELEMENTWISE at minimum
// ============================================================================

/**
 * The SEMANTIC IR node set (P0 (a)): the algebraic expression DAG from which
 * structure has been removed. The ELEMENTWISE family is fully enumerated; the
 * full set's SHAPE is sketched (weave/rearrange included per F19 so faithful
 * routing has an owner and structure isn't hidden in persisted Bezier points).
 *
 * This is deliberately DISTINCT from the imperative tile-IR (`tile-ir.ts`
 * `IRNode`): a `SemanticIRNode` has NO thread ids, NO barriers, NO for-loops,
 * NO placement. `applySchedule` maps (SemanticIR × ScheduleState) → the
 * imperative tile-IR.
 */
export type SemanticIRNode =
  // ---- ELEMENTWISE family (fully enumerated in P0) ----
  | { kind: "load"; value: ValueUid } // a global read of a NamedValue
  | {
      kind: "elementwiseOp";
      catalog: OpCatalogRef;
      args: readonly SemanticIRRef[];
    }
  | {
      kind: "select";
      cond: SemanticIRRef;
      whenTrue: SemanticIRRef;
      whenFalse: SemanticIRRef;
    } // where
  | { kind: "castOp"; to: ValueDtype; arg: SemanticIRRef }
  | { kind: "store"; value: ValueUid; expr: SemanticIRRef }
  // ---- The SHAPE of the full set (sketched, not P0-enumerated) ----
  /** algebraic weave / faithful-routing node (F19). Owner of routing structure
   *  so it is not hidden in Bezier points. */
  | { kind: "weave"; spec: "TODO:weave-schema" }
  /** rearrange / transpose-like index remap (F19). */
  | { kind: "rearrange"; spec: "TODO:rearrange-schema" }
  /** reduction skeleton (per-row / full). Shape only in P0. */
  | { kind: "reduce"; spec: "TODO:reduce-schema" }
  /** contraction / matmul family. Shape only in P0. */
  | { kind: "contract"; spec: "TODO:contract-schema" };

/** A reference to another semantic IR node by index within a region's node list. */
export type SemanticIRRef = { readonly node: number };

/** A semantic region: an ordered list of semantic IR nodes behind one
 *  `SemanticRegionUid`. `applySchedule` consumes (region, state). */
export interface SemanticRegion {
  readonly uid: SemanticRegionUid;
  readonly nodes: readonly SemanticIRNode[];
}
