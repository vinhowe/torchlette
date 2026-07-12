/**
 * Walking skeleton — ATTENTION family (campaign P0-FULL wave 3, the last census
 * block + the P2 prerequisite). Side-by-side with the live path; NO behavior
 * change; NO dispatch cutover. Exercised only by
 * test/schedule/attention-differential.spec.ts.
 *
 * ------------------------------------------------------------------------
 * TWO SHAPES, ONE FAMILY (why attention is different from matmul)
 * ------------------------------------------------------------------------
 * The FUSED FlashAttention kernels (fwd, D-precompute, dQ, dKV) are AUTHORED,
 * not derived (F3). Their computation-shape is online-softmax over KV tiles: an
 * m/l carried-state recurrence with an exp-rescale that is licensed by an
 * ADMITTED LEMMA (online-softmax-rescaling), register accumulators, cooperative
 * K/V shared tiles, and per-tile barriers whose lowering the tile-IR block ops
 * own. The move grammar (§3) cannot yet DERIVE this three-region composite from
 * the naive form — `merge`/`fuse` is an S3 composite transaction the engine has
 * not built. So per §6 the fused kernels wear an OPAQUE skeleton
 * (`visibility: "opaque"`) with a TYPED PARAMETER SCHEMA (R10): the block sizes
 * BR/BC/BQ_BW/BC_BW, the head-dim handling (incl. the 256-head-dim
 * workgroup-storage capability from the Gemma-2 work), the vec4 condition
 * (headDim % 4 == 0), and the score/mask modifier seam parameters (#64). The
 * opaque skeleton NAMES the live kernel and a refusal reason; it carries NO
 * loop/staging/role data (F3) — an authored kernel can never masquerade as a
 * derived one. `deriveAttentionState` READS the live kernel's config (headDim +
 * modifier), `applyAttentionSchedule` regenerates through the SAME live
 * single-source `make*Spec` factory, and the BYTE DIFFERENTIAL proves the
 * regeneration is byte-identical across the modifier corpus (causal /
 * sliding-window / softcap Gemma seams, fwd + D + dQ + dKV).
 *
 * The NAIVE (decomposed) attention is, by contrast, a DERIVABLE COMPOSITION:
 * QK^T (matmul NT) → softmax (row-program) → PV (matmul NN) — three
 * ScheduleStates over three semantic regions, expressed with the EXISTING
 * matmul + row-program family skeletons (no new derivation code). This is the
 * P2 STARTING POSITION: the fused kernel is the composite the move grammar must
 * learn to reach FROM this naive composition, and the online-softmax lemma is
 * the license the `merge` transaction will carry. `naiveAttentionComposition`
 * builds it; the differential verifies each region byte-identically via the
 * REUSED family differentials (matmul + row-program), never duplicating them.
 *
 * ------------------------------------------------------------------------
 * THE TIER MAPPING (authored form — §6 / R10)
 * ------------------------------------------------------------------------
 *   headDim, BR/BC/BQ_BW/BC_BW block sizes  → TypedParamSchema.params (the
 *                                             DECLARED editable parameters; a
 *                                             generic decoration is REFUSED).
 *   headDim % 4 == 0 (vec4), block          → TypedParamSchema.constraints
 *     divisibility                            (dependent constraints, predicate-AST).
 *   head_dim=256 ⇒ 32KB workgroup storage   → TypedParamSchema.capabilityPredicate
 *     (the Gemma-2 requirement)               (a device/realizer capability, F9).
 *   scoreMod / maskMod seam kinds (#64)     → the CACHE-KEY encoder's structural
 *                                             fragment (attnModifierKey — SINGLE
 *                                             SOURCE, template identity).
 *   online-softmax rescale                  → an AdmittedLemma (m,l carried state
 *                                             + proof obligation) on the skeleton.
 *   workgroup geometry, vec-load forms      → realizer RECEIPTS (never identity).
 *
 * The attention kernels REFUSE a spec they cannot express (headDim % 4 != 0
 * throws in the live factory; the seam surfaces it). They never wrap WGSL — the
 * opaque skeleton names a kernel + a reason, it does not carry a replayable
 * generator (there is no generatorFn field, structurally; R22).
 */

import type { AttnModifierSpec } from "../backend/types";
import {
  assertBackwardSupportsModifier,
  attnModifierKey,
  BC,
  BC_BW,
  BQ_BW,
  BR,
  makeBackwardDKVSpec,
  makeBackwardDQSpec,
  makeDPrecomputeSpec,
  makeForwardAttentionSpec,
} from "../backend/webgpu/attention-kernel";
import { DEFAULT_CONFIG } from "../backend/webgpu/matmul/types";
import type { TileKernelSpec } from "../backend/webgpu/tile-ir";
import type { RowProgram } from "../compiler/row-program-types";
import { reportNoSecondOwner } from "./canonical";
import {
  deriveTiledMatmulState,
  type TiledMatmulDescriptor,
} from "./matmul-skeleton";
import { deriveRowProgramState } from "./reduction-skeleton";
import type {
  LemmaApplication,
  LemmaUid,
  ObligationId,
  PredicateAstNode,
  ScheduleState,
  SemanticRegionUid,
  Skeleton,
  TypedParamSchema,
} from "./types";

const uid = <T>(s: string): T => s as unknown as T;

// ============================================================================
// The online-softmax admitted lemma (§3.4) — the fused kernel's license
// ============================================================================

/**
 * FlashAttention replaces the full-row softmax with an ONLINE recurrence over
 * KV tiles: it maintains a running max `m` and a running normalizer `l`, and
 * when a new tile raises the max it rescales the partial output accumulator by
 * `exp(m_old - m_new)`. This is NOT a free structural rewrite of the naive
 * three-region softmax — it is an admitted LEMMA (online-softmax-rescaling),
 * accepted within the training-tolerance envelope, whose CARRIED STATE is the
 * pair (m, l) and whose PROOF OBLIGATION is that the rescaled running sum equals
 * the batched softmax denominator at loop exit. Present as a first-class
 * `LemmaApplication` so the fused schedule is DISTINGUISHABLE from the naive
 * composition by its lemma set (F27/F28) — the two compute the same result but
 * the fused one carries a proof obligation the naive one does not.
 *
 * This is the ENGINE-side instantiation of the lemma the NCD spike carried
 * client-side (examples/schedule-editor authored-attention JSON's
 * `online-softmax-rescaling`): the schema (LemmaApplication) already existed in
 * types.ts; wave 3 fills the catalog entry with its carried state + obligation.
 */
export const ONLINE_SOFTMAX_LEMMA = uid<LemmaUid>(
  "lemma:online-softmax-rescaling",
);
export const ONLINE_SOFTMAX_OBLIGATION = uid<ObligationId>(
  "obl:online-softmax-normalizer-equals-batched-denominator",
);

/** The fused-attention online-softmax lemma application ((m,ℓ,o) carried state,
 *  §3.4 F27 — the state-machine from which inspection views derive). */
export function onlineSoftmaxLemma(): LemmaApplication {
  return {
    lemma: ONLINE_SOFTMAX_LEMMA,
    obligation: ONLINE_SOFTMAX_OBLIGATION,
    // Carried state is the (running-max m, running-normalizer ℓ, partial-output
    // accumulator o) state-machine + correction factor exp(m_old − m_new)
    // applied to o whenever the running max rises (§3.4 F27, verbatim).
    carriedStateRef:
      "carried=(m:running-max,l:running-normalizer,o:partial-output);correction=exp(m_old-m_new)",
  };
}

// ============================================================================
// The typed parameter schema (R10 + F7) — the authored attention constraint set
// ============================================================================

function paramLeaf(name: string): PredicateAstNode {
  return { kind: "uniformRef", name };
}

/** headDimension % 4 == 0 — the vec4 requirement (the live factory throws
 *  otherwise). Encoded as member(headDimension mod 4) == 0; `mod` is a display
 *  projection, the executable check is the live factory's throw (single owner). */
function headDimVec4Constraint(): PredicateAstNode {
  return {
    kind: "cmp",
    op: "==",
    lhs: {
      kind: "member",
      value: paramLeaf("headDimension"),
      set: [{ kind: "intLit", value: 4 }],
    },
    rhs: { kind: "intLit", value: 0 },
  };
}

/**
 * The 256-head-dim workgroup-storage requirement (Gemma-2). head_dim=256
 * attention tiles need 32KB of workgroup storage; the DEFAULT-16KB device
 * silently drops every attention submit. The capability predicate names it: a
 * head-dim ≥ 256 REQUIRES maxComputeWorkgroupStorageSize ≥ 32768. Harmless for
 * headDim ≤ 128 (fits 16KB). This is a device/realizer CAPABILITY (F9), owned
 * ONCE here — the fix (gpu-context.ts requesting the adapter max) is the
 * realizer honoring the request; the predicate is the request's typed shape.
 */
function workgroupStorageCapability(): PredicateAstNode {
  // headDimension < 256  OR  maxComputeWorkgroupStorageSize >= 32768
  return {
    kind: "or",
    terms: [
      {
        kind: "cmp",
        op: "<",
        lhs: paramLeaf("headDimension"),
        rhs: { kind: "intLit", value: 256 },
      },
      {
        kind: "cmp",
        op: ">=",
        lhs: paramLeaf("maxComputeWorkgroupStorageSize"),
        rhs: { kind: "intLit", value: 32768 },
      },
    ],
  };
}

/**
 * The attention family's declared parameter schema (§6, R10/F7). The block sizes
 * (BR/BC forward+dQ, BQ_BW/BC_BW backward-dKV) are DEFAULTED from the live
 * kernel's constants (single source — imported, not re-typed), headDim spans the
 * shipped head dims (64 Qwen3 / 128 / 256 Gemma-2), and the modifier seam is a
 * structural parameter (the scoreMod/maskMod kinds — #64 — key the template).
 * The dependent constraints are the vec4 divisibility; the capability predicate
 * is the 256-head-dim workgroup-storage requirement.
 */
export const ATTENTION_PARAM_SCHEMA: TypedParamSchema = {
  params: {
    // The F7 attention skeleton family keys (§6): qRows / kvRows / headDimension.
    // Head dim — the shipped set (Qwen3 64, GPT-2 64, Gemma-2 256).
    headDimension: { domain: [64, 128, 256], default: 64 },
    // Forward + dQ block sizes (BR Q-rows/workgroup, BC KV-rows/tile).
    qRows: { domain: [BR], default: BR },
    kvRows: { domain: [BC], default: BC },
    // Backward dKV block sizes (BQ_BW Q-rows/tile, BC_BW KV-rows/workgroup).
    bwdQRows: { domain: [BQ_BW], default: BQ_BW },
    bwdKvRows: { domain: [BC_BW], default: BC_BW },
  },
  constraints: [headDimVec4Constraint()],
  capabilityPredicate: workgroupStorageCapability(),
};

// ============================================================================
// The authored (opaque) attention skeletons (§6 / F3)
// ============================================================================

/** The four fused-attention kernel roles. Each is one authored kernel. */
export type AttentionKernelRole =
  | "forward"
  | "dPrecompute"
  | "backwardDQ"
  | "backwardDKV";

/**
 * A structured description of ONE authored attention kernel — the honest
 * reification of the live `make*Spec` factory's inputs (headDim + modifier).
 * The modifier's STRUCTURE (score/mask kinds) is template identity; the numeric
 * params (cap, window) are uniform DATA the realizer binds (not key material —
 * exactly as the live `attnModifierKey` single source declares).
 */
export interface AttentionDescriptor {
  readonly role: AttentionKernelRole;
  readonly headDim: number;
  /** The score/mask modifier (#64). Undefined = bare bounds-checked softmax. */
  readonly modifier?: AttnModifierSpec;
}

/**
 * The canonical cache-key encoder (R10) for an authored attention kernel: the
 * role + headDim + the modifier's STRUCTURAL key fragment (attnModifierKey — the
 * SINGLE SOURCE the live kernel already uses for its WGSL/pipeline/config
 * caches). This is the schema's cache-key encoder owned ONCE — it delegates to
 * the live single source rather than re-deriving the modifier key (no second
 * owner of the template-identity fact).
 */
export function attentionCacheKey(desc: AttentionDescriptor): string {
  const modKey = attnModifierKey(desc.modifier);
  const base = `attn:${desc.role}:D${desc.headDim}`;
  return modKey ? `${base}:${modKey}` : base;
}

/**
 * Derive the AUTHORED (opaque) skeleton for one attention kernel (§6 / F3). The
 * skeleton NAMES the live kernel (kernelRef) and a refusal reason ("authored —
 * the online-softmax composite is not yet derivable by the move grammar; S3
 * merge/fuse unbuilt"), and carries the TYPED PARAM SCHEMA. It FORBIDS loop /
 * staging / role data (F3) — the opaque variant of `Skeleton` has no place for
 * them, structurally.
 *
 * `deriveAttentionState` READS the live kernel's config (headDim + modifier) via
 * the descriptor; it does NOT observe WGSL. The differential proves the
 * regeneration (applyAttentionSchedule → the live make*Spec) is byte-identical.
 */
export function deriveAttentionSkeleton(desc: AttentionDescriptor): Skeleton {
  return {
    visibility: "opaque",
    kernelRef: kernelRefFor(desc.role),
    refusalReason:
      "authored — not yet re-derived. The fused online-softmax composite is " +
      "reachable only via the P2 FA-derivation (rungs 0–7: merge naive three " +
      "islands [S3] → tile → stream K/V → recolor accumulator → apply the " +
      "online-softmax admitted lemma; F17 sequence lemma→recolor→recolor→group→" +
      "stream), which the move grammar cannot yet run (S3 merge/fuse is an " +
      "unbuilt engine transaction). Exit: local self-hosting (attention backward " +
      "re-derived once the recomputation-identity + D-precompute lemmas are " +
      "admitted; §6/§7 P4).",
    params: ATTENTION_PARAM_SCHEMA,
  };
}

function kernelRefFor(role: AttentionKernelRole): string {
  switch (role) {
    case "forward":
      return "src/backend/webgpu/attention-kernel.ts::makeForwardAttentionSpec";
    case "dPrecompute":
      return "src/backend/webgpu/attention-kernel.ts::makeDPrecomputeSpec";
    case "backwardDQ":
      return "src/backend/webgpu/attention-kernel.ts::makeBackwardDQSpec";
    case "backwardDKV":
      return "src/backend/webgpu/attention-kernel.ts::makeBackwardDKVSpec";
  }
}

/**
 * `applyAttentionSchedule` for the authored family: assert the opaque skeleton
 * is well-formed at the seam (no loop/staging/role data leaked into it — F3),
 * then call the live single-source `make*Spec` factory the kernelRef names,
 * reconstructing its inputs from the descriptor. Returns the `TileKernelSpec`
 * the differential compiles (its WGSL equals the live make*Spec byte-for-byte).
 *
 * This is the R22-defeating shape: there is nowhere in the opaque skeleton to
 * store a generator; `applyAttentionSchedule` re-CALLS the named factory, it
 * does not replay a captured one. The skeleton owns the DECLARED params; the
 * kernel emission stays the live single source.
 */
export function applyAttentionSchedule(
  skeleton: Skeleton,
  desc: AttentionDescriptor,
): TileKernelSpec {
  assertAuthoredSeam(skeleton, desc);
  const mod = desc.modifier;
  switch (desc.role) {
    case "forward":
      return makeForwardAttentionSpec(desc.headDim, mod);
    case "dPrecompute":
      // D-precompute has no score/mask seam sites — one template for all mods.
      return makeDPrecomputeSpec(desc.headDim);
    case "backwardDQ":
      // Backward is inference-first for score modifiers (#64): the paired
      // derivative ("attn_dscore") is designed but unimplemented. The live
      // dispatch/plan path refuses a scoreMod via assertBackwardSupportsModifier
      // — re-call it here so the authored seam preserves the SAME refusal (one
      // owner of the backward-scoreMod legality fact).
      assertBackwardSupportsModifier(mod);
      return makeBackwardDQSpec(desc.headDim, mod);
    case "backwardDKV":
      assertBackwardSupportsModifier(mod);
      return makeBackwardDKVSpec(desc.headDim, mod);
  }
}

/**
 * Assert the authored attention skeleton carries no second owner and matches the
 * descriptor at the seam (§12 check 3, family-local). The opaque skeleton MUST
 * name the correct live kernel (kernelRef), MUST carry the typed param schema,
 * and MUST refuse a headDim the schema's vec4 constraint rejects (the live
 * factory throws — the seam surfaces the SAME refusal, one owner).
 */
export function assertAuthoredSeam(
  skeleton: Skeleton,
  desc: AttentionDescriptor,
): void {
  if (skeleton.visibility !== "opaque")
    reportNoSecondOwner(
      `no-second-owner[attention]: the fused attention kernel MUST wear an opaque ` +
        `skeleton (F3 authored hatch); got "${skeleton.visibility}". The online-softmax ` +
        `composite is not yet derivable (S3 merge/fuse unbuilt).`,
    );
  if (skeleton.visibility === "opaque") {
    if (skeleton.kernelRef !== kernelRefFor(desc.role))
      reportNoSecondOwner(
        `no-second-owner[attention]: skeleton kernelRef "${skeleton.kernelRef}" disagrees ` +
          `with descriptor role "${desc.role}" (${kernelRefFor(desc.role)}).`,
      );
    if (skeleton.params !== ATTENTION_PARAM_SCHEMA)
      reportNoSecondOwner(
        `no-second-owner[attention]: the opaque skeleton must carry the single ` +
          `ATTENTION_PARAM_SCHEMA — a second typed-param copy is a second owner of the ` +
          `validateConfig-class facts.`,
      );
  }
  // The vec4 constraint is owned by the schema AND enforced by the live factory
  // (headDim % 4 throws). The seam checks agreement: if the schema would reject
  // the headDim, so must the factory. (Divisibility check, single arithmetic.)
  if (desc.headDim % 4 !== 0)
    reportNoSecondOwner(
      `legality[attention]: headDim=${desc.headDim} violates the vec4 constraint ` +
        `(headDim % 4 == 0). The schema's dependent constraint and the live factory's ` +
        `throw are the SAME fact, owned once.`,
    );
}

// ============================================================================
// The NAIVE attention COMPOSITION (deliverable 2 — the P2 starting position)
// ============================================================================

/**
 * The naive (decomposed) attention as a THREE-REGION composition expressed with
 * the EXISTING family skeletons — matmul (QK^T, PV) + row-program (softmax) — as
 * three `ScheduleState`s over three semantic regions plus the island-level
 * structure connecting them (the region UIDs + the data-flow between them). This
 * is P2's starting position: the fused kernel is the composite `merge`/`fuse`
 * must reach FROM here, carrying the online-softmax lemma as its license.
 *
 * NO new derivation code: region 1 and 3 are `deriveTiledMatmulState`, region 2
 * is `deriveRowProgramState` over a softmax RowProgram. The differential
 * verifies each region byte-identically via the REUSED matmul + row-program
 * family apply seams (it does not duplicate them).
 */
export interface NaiveAttentionComposition {
  /** Region 1: scores = Q @ K^T (matmul NT, no epilogue — the scale/mask land
   *  in the softmax region's preamble to match the decomposed lowering). */
  readonly qkT: {
    region: SemanticRegionUid;
    state: ScheduleState;
    desc: TiledMatmulDescriptor;
  };
  /** Region 2: P = softmax(scores) along the KV (feature) axis (row-program:
   *  max-reduce, sum-of-exp-reduce, normalize-write). */
  readonly softmax: {
    region: SemanticRegionUid;
    state: ScheduleState;
    program: RowProgram;
  };
  /** Region 3: O = P @ V (matmul NN, no epilogue). */
  readonly pv: {
    region: SemanticRegionUid;
    state: ScheduleState;
    desc: TiledMatmulDescriptor;
  };
  /** The island-level structure: the ordered data-flow edges between regions
   *  (scores → softmax → O). This is the composite the move grammar's `merge`
   *  transaction consumes at P2 (S3 — engine side unbuilt). */
  readonly islandFlow: readonly {
    readonly from: SemanticRegionUid;
    readonly to: SemanticRegionUid;
    readonly via: string;
  }[];
  /** The head dim this composition is instantiated at (documents the geometry;
   *  the naive matmul/row-program regions are head-dim-agnostic templates). */
  readonly headDim: number;
}

/**
 * The softmax RowProgram (region 2): a numerically-stable row softmax over the
 * feature (KV) axis. Phase 1 reduces the row max; phase 2 reduces the sum of
 * exp(x - max); the write phase emits exp(x - max) / sum. This is the SAME
 * RowProgram shape the graph compiler detects for a real softmax, so it
 * round-trips byte-identically via the row-program family differential.
 */
export function softmaxRowProgram(): RowProgram {
  const input = (): RPExprInput => ({ kind: "input", bufferIndex: 0 });
  const maxR = (): RPExprRef => ({ kind: "reduceResult", phaseIndex: 0 });
  const sumR = (): RPExprRef => ({ kind: "reduceResult", phaseIndex: 1 });
  return {
    inputs: [{ dtype: "f32" }],
    output: { dtype: "f32" },
    dim: -1,
    phases: [
      // Phase 0: m = max_j scores[j]
      { kind: "reduce", reduceOp: "max", bodyExpr: input() },
      // Phase 1: l = sum_j exp(scores[j] - m)
      {
        kind: "reduce",
        reduceOp: "sum",
        bodyExpr: {
          op: "exp",
          inputs: [{ op: "sub", inputs: [input(), maxR()] }],
        },
      },
      // Write: P[j] = exp(scores[j] - m) / l
      {
        kind: "write",
        bodyExpr: {
          op: "div",
          inputs: [
            { op: "exp", inputs: [{ op: "sub", inputs: [input(), maxR()] }] },
            sumR(),
          ],
        },
      },
    ],
    cacheKey: "attn-naive-softmax-f32",
  };
}

// Narrow RPExpr helpers (the RowProgram RPValue leaves) — typed locally to keep
// the composition self-documenting without re-exporting the row-program types.
type RPExprInput = { kind: "input"; bufferIndex: number };
type RPExprRef = { kind: "reduceResult"; phaseIndex: number };

/**
 * Build the naive three-region composition. Regions 1/3 reuse the matmul family
 * (`deriveTiledMatmulState`); region 2 reuses the row-program family
 * (`deriveRowProgramState`) over `softmaxRowProgram()`. The island-flow edges
 * connect them (scores → softmax → O).
 */
export function naiveAttentionComposition(
  headDim: number,
): NaiveAttentionComposition {
  const qkTRegion = uid<SemanticRegionUid>("region:attn-naive-qkT");
  const softmaxRegion = uid<SemanticRegionUid>("region:attn-naive-softmax");
  const pvRegion = uid<SemanticRegionUid>("region:attn-naive-pv");

  // Region 1: Q @ K^T — NT matmul (K is transposed), f32, no epilogue.
  const qkTDesc: TiledMatmulDescriptor = {
    config: { ...DEFAULT_CONFIG },
    transposeMode: "NT",
    dtype: "f32",
  };
  const qkTState = deriveTiledMatmulState(qkTDesc, qkTRegion);

  // Region 2: softmax(scores) along the KV axis — the row-program family.
  const program = softmaxRowProgram();
  const softmaxState = deriveRowProgramState(program, softmaxRegion);

  // Region 3: P @ V — NN matmul, f32, no epilogue.
  const pvDesc: TiledMatmulDescriptor = {
    config: { ...DEFAULT_CONFIG },
    transposeMode: "NN",
    dtype: "f32",
  };
  const pvState = deriveTiledMatmulState(pvDesc, pvRegion);

  return {
    qkT: { region: qkTRegion, state: qkTState, desc: qkTDesc },
    softmax: { region: softmaxRegion, state: softmaxState, program },
    pv: { region: pvRegion, state: pvState, desc: pvDesc },
    islandFlow: [
      { from: qkTRegion, to: softmaxRegion, via: "scores" },
      { from: softmaxRegion, to: pvRegion, via: "P" },
    ],
    headDim,
  };
}
