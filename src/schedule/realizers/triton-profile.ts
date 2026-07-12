/**
 * The paper Triton capability profile as CODE (schedule-state-design.md §4 —
 * the v2 realizer entry `{ capabilityProfile, emit, costModel, verificationHarness }`;
 * this file is the `capabilityProfile`).
 *
 * ------------------------------------------------------------------------
 * WHAT THIS IS
 * ------------------------------------------------------------------------
 * A machine-readable transcription of `docs/design-corpus/appendix-a-triton-profile.md`
 * (the paper profile, `main` inspected 2026-07-11). Per move / decoration / axis it
 * declares a VERDICT — `determination` | `request` | `refused` — plus the human
 * REASON and the Appendix-A finding tag. The registry (§4) consults this DATA to
 * answer "can the Triton realizer honor this schedule fact, and if not why"; the
 * refusals are TYPED, not exceptions thrown deep in an emitter (F8: allowed/refused
 * values + reasons live in the profile, not in the object — the object alone cannot
 * say whether a value is core-illegal, realizer-unsupported, or device-absent).
 *
 * A SPLIT verdict (a move name that is determination for one half and request/refused
 * for another) is itself a §2 representation bug — Appendix A flags five (A-R1..A-R5).
 * We encode those as `{ verdict: "split", ... }` so the split is DATA the registry can
 * see, never silently collapsed to one word.
 *
 * The S1 three-tier object already absorbed every found representation bug (§4: "any
 * inexpressible entry is a representation bug; all found ones are already folded into
 * S1"). So this profile ALSO records, per entry, WHERE in the S1 object the fact now
 * lives (`s1Home`) — the proof that the object expresses Triton's surface without
 * WGSL-isms. That is the load-bearing claim of Appendix A rendered checkable.
 */

/** The three verdicts of Appendix A, plus `split` (itself a §2 representation bug). */
export type CapabilityVerdict =
  | "determination"
  | "request"
  | "refused"
  | "split";

/** Which S1 tier / field now OWNS this fact (the "folded into S1" proof, §4). */
export type S1Home =
  | "semantic" // SemanticSchedule (computation-shape identity)
  | "requests" // BackendRequests (compilation identity)
  | "receipts" // RealizationReceipts (neither identity — the realizer reports it)
  | "islands" // above this realizer (island membership, S3 fuse)
  | "selection" // the registry / routing layer (R9), not the schedule
  | "none"; // no home — a genuine missing coordinate (an A-R14-class gap)

/** A single capability-profile row: one v1 move/decoration/axis, its Triton verdict. */
export interface CapabilityEntry {
  /** The v1 element name (a move, a decoration, or an enumerated axis/atom). */
  readonly element: string;
  /** The Appendix-A category the element belongs to. */
  readonly category: "move" | "decoration" | "axis" | "atom" | "mechanism";
  /** The verdict. `split` = a §2 representation bug (the name mixes two facts). */
  readonly verdict: CapabilityVerdict;
  /**
   * For a `split` verdict, the two halves (each with its own sub-verdict). For a
   * non-split verdict this is absent. This makes the split legible AS DATA — the
   * registry can refuse the physical half while honoring the semantic half.
   */
  readonly split?: {
    readonly determinationHalf: string;
    readonly refusedOrRequestHalf: string;
    readonly halfVerdict: Exclude<CapabilityVerdict, "split">;
  };
  /** The Triton surface fact (what `tl.*` actually controls, or why it cannot). */
  readonly reason: string;
  /** The Appendix-A finding tag(s) this row transcribes (e.g. "A-R1", "R4"). */
  readonly findings: readonly string[];
  /** Where the fact now lives in the S1 object (the "folded into S1" proof). */
  readonly s1Home: S1Home;
}

/**
 * The authority horizon (Appendix A "Authority horizon"): what the emitted SOURCE
 * fixes vs what TTGIR/backend passes independently choose. The emitter (deliverable
 * 2) MUST NOT emit anything on the `ttgirOwns` side — that is the receipt boundary
 * (`RealizationReceipts` owns nothing here; Triton's compiler does, and even the
 * receipt is only observable, not chosen by us).
 */
export interface AuthorityHorizon {
  /** Facts the emitted Triton SOURCE determines (before/at TTIR). */
  readonly sourceDetermines: readonly string[];
  /** Meta-parameters / attributes that are REQUESTS or bounds (num_warps, etc.). */
  readonly requestsOrBounds: readonly string[];
  /** Facts TTGIR / backend passes choose independently (the receipt boundary). */
  readonly ttgirOwns: readonly string[];
  /** The escape hatch we DECLINE to count (Config.ir_override → a second realizer). */
  readonly declinedEscapeHatch: string;
}

// ============================================================================
// The profile object (§4 WgslRealizer.capabilityProfile analogue, for Triton)
// ============================================================================

/** The versioned Triton capability profile (F8 + §5 `capabilityProfileVersion`). */
export interface TritonCapabilityProfile {
  readonly realizer: "triton";
  /** Versioned per §5/R27 — the profile is a compilation-identity input. */
  readonly capabilityProfileVersion: number;
  /**
   * The pinned surface (Appendix A: "a v2 profile must pin a Triton release/commit
   * and target architecture"). Public `main` is not a stable version contract, so we
   * pin what THIS harness actually runs against.
   */
  readonly pinnedSurface: {
    readonly tritonRelease: string;
    readonly targetArch: string;
    readonly inspected: string;
    readonly gluonCounted: false; // experimental Gluon is NOT the v2 surface
  };
  readonly entries: readonly CapabilityEntry[];
  readonly authorityHorizon: AuthorityHorizon;
}

/**
 * THE TRITON CAPABILITY PROFILE — Appendix A transcribed row-for-row.
 *
 * Pinned to triton 3.1.0 on sm_70 (V100, CC 7.0), the surface this campaign's
 * python harness executes against. Triton supports CC 7.0 for the fp16/fp32 FMA
 * paths (no wgmma / no Hopper warp-specialization); the profile's Blackwell-only
 * and SM90+-only entries are therefore `refused`/`request` on THIS pin regardless.
 */
export const TRITON_CAPABILITY_PROFILE: TritonCapabilityProfile = {
  realizer: "triton",
  capabilityProfileVersion: 1,
  pinnedSurface: {
    tritonRelease: "3.1.0",
    targetArch: "sm_70", // V100 SXM3, CC 7.0
    inspected: "2026-07-11",
    gluonCounted: false,
  },
  entries: [
    // ---- MOVES (Appendix A "Moves" table) ----
    {
      element: "tile",
      category: "move",
      verdict: "split",
      split: {
        determinationHalf: "logical block shape (tl.arange/masks/tl.constexpr)",
        refusedOrRequestHalf:
          "physical layout — lane/warp/register ownership + shared swizzles (TTGIR)",
        halfVerdict: "request",
      },
      reason:
        "tl.arange, masks, pointer arithmetic, and tl.constexpr fix the LOGICAL block; " +
        "lane/warp/register ownership and shared swizzles are TTGIR encodings chosen downstream.",
      findings: ["A-R1"],
      s1Home: "semantic", // blockShapes (logical extents) — receipts own the physical layout
    },
    {
      element: "stream",
      category: "move",
      verdict: "split",
      split: {
        determinationHalf:
          "no-materialization (eliminate the store, emit a source loop)",
        refusedOrRequestHalf:
          "physical residency (LICM, caching, staging, spills, load scheduling)",
        halfVerdict: "request",
      },
      reason:
        "The realizer can emit repeated loads in a loop, but LICM/caching/staging/spills " +
        "remain compiler decisions. Stream needs a semantic no-store contract SEPARATE from a residency request.",
      findings: ["A-R2"],
      s1Home: "semantic", // NoMaterializationEdge is the store-elimination half; residency → requests.placementPreferences
    },
    {
      element: "recolor",
      category: "move",
      verdict: "split",
      split: {
        determinationHalf: "rewrite logical indices only",
        refusedOrRequestHalf:
          "pin accumulator/operand to registers or remap lanes",
        halfVerdict: "refused",
      },
      reason:
        "Stable tl.* accepts no user tensor-layout or register-placement attributes. " +
        "If recolor pins a register/lane it is REFUSED; if it merely rewrites indices it is a determination.",
      findings: ["A-R3"],
      s1Home: "requests", // register residency is an INTENT (semantic) + a PREFERENCE (requests); never a determination
    },
    {
      element: "fuse",
      category: "move",
      verdict: "determination",
      reason:
        "Emitting one JIT kernel fixes one launch and removes global intermediates — within " +
        "its legality horizon. It CANNOT fuse a dependence needing inter-program synchronization " +
        "(Triton has no grid-wide barrier inside a kernel). Island membership changes ABOVE this realizer.",
      findings: [],
      s1Home: "islands", // S3: fuse is a composite transaction at the islands altitude, not an intra-schedule move
    },
    {
      element: "pack",
      category: "move",
      verdict: "split",
      split: {
        determinationHalf:
          "horizontal multi-tensor work in one emitted program",
        refusedOrRequestHalf: "a demanded SIMD/vec4 packing",
        halfVerdict: "refused",
      },
      reason:
        "Pointer/index code can pack independent work; physical vector load width is " +
        "compiler-selected (tl.* has no vec4 knob). The unqualified move name conflates the two.",
      findings: ["A-R4"],
      s1Home: "semantic", // horizontal pack is a schedule fact; vec width is a receipt
    },
    {
      element: "role-partition",
      category: "move",
      verdict: "split",
      split: {
        determinationHalf: "(none — no determination form)",
        refusedOrRequestHalf:
          "named producer/consumer roles (only tl.range(warp_specialize=True), Blackwell-only)",
        halfVerdict: "request",
      },
      reason:
        "tl.range(..., warp_specialize=True) ASKS the compiler to partition a simple matmul " +
        "loop (documented only for Blackwell). It does not describe named producer/consumer roles. " +
        "On sm_70 there is no warp-specialization surface at all → refused on this pin.",
      findings: ["A-R5"],
      s1Home: "semantic", // roles are STORED schedule facts (S2); the realizer may refuse to honor them
    },
    {
      element: "pipeline",
      category: "move",
      verdict: "request",
      reason:
        "Kernel num_stages requests dot-load pipelining; loop-local tl.range(num_stages=...) " +
        "requests broader load pipelining. TTGIR assigns latencies, schedules loops, inserts fences, " +
        "and may realize a different schedule.",
      findings: [],
      s1Home: "requests", // BackendRequests.pipeline (PipelineRequest); realizedStages is a receipt
    },

    // ---- DECORATIONS (Appendix A "Decorations" table) ----
    {
      element: "tile sizes",
      category: "decoration",
      verdict: "determination",
      reason:
        "BLOCK_*: tl.constexpr and tl.arange fix LOGICAL tensor extents only. They do NOT fix " +
        "per-thread tiles, layout, or instruction shape.",
      findings: ["A-R1"],
      s1Home: "semantic", // blockShapes
    },
    {
      element: "vec width",
      category: "decoration",
      verdict: "refused",
      reason:
        "No public exact vector-width parameter. Alignment/contiguity assertions inform " +
        "coalescing; TTGIR chooses load width. The repo's WGSL array<vec4<T>> reinterpretation is not portable.",
      findings: ["A-R6"],
      s1Home: "receipts", // vecLoadForms is a REPORTED receipt, never a determination
    },
    {
      element: "workgroup dimensions",
      category: "decoration",
      verdict: "refused",
      split: {
        determinationHalf: "(none)",
        refusedOrRequestHalf:
          "WGSL [x,y,z] geometry; nearest map is num_warps (a request)",
        halfVerdict: "request",
      },
      reason:
        "num_warps supplies a CTA warp BUDGET, not an x/y/z local-ID geometry or a user " +
        "mapping of elements to threads. The WGSL [x,y,z] form is refused; the budget is a request.",
      findings: ["A-R7"],
      s1Home: "requests", // warpBudget (a budget, not a geometry); the [x,y,z] is a WGSL-only receipt
    },
    {
      element: "operand residency",
      category: "decoration",
      verdict: "refused",
      reason:
        "Global pointer accesses are explicit, but register-vs-shared staging of block values is " +
        "encoded/allocated in TTGIR. There is no stable tl.* 'put operand B in shared' surface.",
      findings: ["A-R8"],
      s1Home: "requests", // placementPreferences (a request the realizer may ignore)
    },
    {
      element: "pipeline depth",
      category: "decoration",
      verdict: "request",
      reason:
        "num_stages is kernel- or loop-scoped and the two forms differ. One scalar on the whole " +
        "ScheduleState cannot represent multiple pipelined loops or 'no pipeline'.",
      findings: ["A-R9"],
      s1Home: "requests", // PipelineRequest carries per-loop entries, never one global scalar
    },
    {
      element: "unroll",
      category: "decoration",
      verdict: "request",
      reason:
        "tl.range(loop_unroll_factor=n) and tl.static_range guide IR unrolling. Loop-local, not " +
        "a kernel-global Boolean/scalar.",
      findings: [],
      s1Home: "requests",
    },

    // ---- ENUMERATED AXES / ATOMS / LEMMAS (Appendix A third table) ----
    {
      element: "memory:global",
      category: "axis",
      verdict: "determination",
      reason: "Pointer loads/stores are explicit global-memory effects.",
      findings: [],
      s1Home: "semantic", // NamedValue.allocation === "global"
    },
    {
      element: "memory:workgroup-shared",
      category: "axis",
      verdict: "refused",
      reason:
        "Shared allocation, layout, and staging are TTGIR/backend decisions. There is no portable " +
        "shared variable in stable tl.* (tensor descriptors may select TMA, not a shared var).",
      findings: ["A-R8"],
      s1Home: "semantic", // allocation "shared" is a STAGING INTENT (S2); Triton refuses to let us place it
    },
    {
      element: "reserved:register-explicit / distributed-shared / cluster",
      category: "axis",
      verdict: "refused",
      reason:
        "maxnreg is a ceiling, not placement. num_ctas requests a cluster size on SM90+ (absent " +
        "on sm_70), not a public distributed-shared allocation contract.",
      findings: [],
      s1Home: "none",
    },
    {
      element: "sync:workgroup",
      category: "axis",
      verdict: "determination",
      reason:
        "tl.debug_barrier() is block-wide. The axis still omits primitive, memory order, and " +
        "barrier-vs-atomic-scope — A-R10.",
      findings: ["A-R10"],
      s1Home: "semantic", // SyncRelation barrier (participants/spaces/convergence)
    },
    {
      element: "reserved:sync subgroup/cluster/grid",
      category: "axis",
      verdict: "refused",
      reason:
        "Atomics offer cta/gpu/sys visibility scopes; those are NOT subgroup/cluster/grid barriers. " +
        "Grid synchronization requires another launch / cooperative mechanism.",
      findings: [],
      s1Home: "none", // "grid barrier" is a capability-graph ABSENCE (R3), not a reserved value
    },
    {
      element: "hierarchy:workgroup",
      category: "axis",
      verdict: "determination",
      reason: "tl.program_id(axis) fixes the program instance and launch grid.",
      findings: [],
      s1Home: "semantic", // loop kind "parallel" + programGridMap
    },
    {
      element: "hierarchy:invocation / subgroup",
      category: "axis",
      verdict: "refused",
      reason:
        "Ordinary Triton intentionally hides lane/thread IDs and assigns block elements through " +
        "internal layouts.",
      findings: ["A-R11"],
      s1Home: "none",
    },
    {
      element: "reserved:hierarchy warp-role",
      category: "axis",
      verdict: "request",
      reason:
        "Limited automatic warp specialization is a compiler request; named roles require " +
        "experimental Gluon or lower IR.",
      findings: [],
      s1Home: "semantic",
    },
    {
      element: "reserved:hierarchy cluster",
      category: "axis",
      verdict: "request",
      reason:
        "num_ctas requests blocks per cluster on supported hardware (not sm_70); it does not " +
        "determine cluster layout or the shared-memory protocol.",
      findings: [],
      s1Home: "requests",
    },
    {
      element: "atom:atomicAddF32-CAS",
      category: "atom",
      verdict: "split",
      split: {
        determinationHalf:
          "native tl.atomic_add OR an explicit tl.atomic_cas retry loop",
        refusedOrRequestHalf:
          "the atom name bakes the WGSL CAS realization into the object",
        halfVerdict: "determination",
      },
      reason:
        "The semantic operation is portable (Triton has native float atomic_add + explicit sem/scope), " +
        "but the atom's NAME bakes the WGSL realization. The realization (CAS vs native) is a receipt.",
      findings: ["A-R12"],
      s1Home: "receipts", // AtomRealization: CASLoop vs NativeAtomic is the realizer's choice
    },
    {
      element: "atom:subgroup-ops (feature-gated family)",
      category: "atom",
      verdict: "refused",
      reason:
        "Block reductions may lower to warp instructions, but stable tl.* has no generic subgroup " +
        "value, lane ID, or exact shuffle contract. A family name without operation/signature/scope cannot be profiled.",
      findings: ["A-R13"],
      s1Home: "none",
    },
    {
      element: "mechanism:admitted-lemma",
      category: "mechanism",
      verdict: "determination",
      reason:
        "A lemma is applied at the host rewrite layer BEFORE emission; Triton compiles the " +
        "transformed algorithm. Capability is per the lemma's resulting operations, not a property of 'admission'.",
      findings: [],
      s1Home: "semantic", // LemmaApplication (e.g. the K-split fp-sum-reassociation license)
    },

    // ---- THE PUBLISHED COUNTEREXAMPLE (Appendix A A-R15 / review finding R4) ----
    {
      element: "program-map (grid traversal / remapping)",
      category: "move",
      verdict: "determination",
      reason:
        "Triton's published matmul remaps program IDs in row GROUPS to improve L2 reuse (>10% on A100); " +
        "tl.swizzle2d exposes the same class. This is LOGICAL index remapping — WGSL workgroup_id arithmetic " +
        "expresses it — not recolor/tile/stream/fuse/pack/role/pipeline. §2 had NO coordinate for it (the gap).",
      findings: ["A-R15", "R4"],
      s1Home: "semantic", // ProgramGridMap {identity|swap|grouped|checkedAffine} — the 8th move
    },
  ],
  authorityHorizon: {
    sourceDetermines: [
      "arithmetic",
      "pointer/index expressions",
      "masks",
      "explicit control flow",
      "launch grid",
      "dispatch count",
      "explicit atomics",
      "the block barrier (tl.debug_barrier)",
    ],
    requestsOrBounds: [
      "num_warps",
      "num_stages",
      "num_ctas",
      "maxnreg",
      "cache modifiers (tl.load eviction/volatile)",
      "loop attributes (unroll/flatten/LICM)",
    ],
    ttgirOwns: [
      "tensor encodings",
      "element-to-register/lane/warp layouts",
      "coalescing and vector width",
      "shared-memory allocation/swizzles",
      "tensor-core lowering",
      "thread locality",
      "loop fusion/LICM",
      "software-pipeline instruction order",
      "fence insertion",
      "register allocation and spills",
    ],
    declinedEscapeHatch:
      "Config.ir_override (TTGIR/LLVM/PTX) bypasses the horizon but would abandon §4's " +
      "'Triton pays the CUDA tax' premise and create a second low-level realizer — NOT counted.",
  },
};

// ============================================================================
// Registry consultation helpers (typed refusals — F8)
// ============================================================================

/** Look up one element's verdict by name (exact match on `element`). */
export function verdictOf(element: string): CapabilityEntry | undefined {
  return TRITON_CAPABILITY_PROFILE.entries.find((e) => e.element === element);
}

/** A typed refusal a realizer returns instead of throwing deep in an emitter (F8). */
export interface CapabilityRefusal {
  readonly element: string;
  readonly verdict: CapabilityVerdict;
  readonly reason: string;
  readonly findings: readonly string[];
}

/**
 * Count entries per verdict — the profile SUMMARY the report quotes. A `split`
 * counts as its own verdict (it is a §2 representation bug worth surfacing).
 */
export function verdictCounts(): Record<CapabilityVerdict, number> {
  const counts: Record<CapabilityVerdict, number> = {
    determination: 0,
    request: 0,
    refused: 0,
    split: 0,
  };
  for (const e of TRITON_CAPABILITY_PROFILE.entries) counts[e.verdict]++;
  return counts;
}
