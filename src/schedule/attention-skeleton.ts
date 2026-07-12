/**
 * ATTENTION family schedule module (P0-FULL wave 3 + §7 P4 CUTOVER-FLIP). The
 * live attention dispatch/plan path now ROUTES THROUGH this module: the four
 * fused-FlashAttention kernel bodies were ABSORBED here from attention-kernel.ts
 * (`lowerAttention*Body`), so the ScheduleState OWNS the kernel structure and
 * `realizeAttentionSpec` is the live dispatch chokepoint (the P1/matmul/reduction
 * `realize*` pattern). NO behavior change — the bodies are byte-identical to the
 * retired `make*Spec` factories (the 14-kernel modifier differential guards the
 * LIVE path). The skeleton STAYS opaque (F3): the online-softmax composite is
 * still authored (not yet reached by the move grammar's `merge` transaction — S3
 * unbuilt), but the body it names lives here, so there is one owner.
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
  hasCausalMask,
} from "../backend/webgpu/attention-kernel";
import { DEFAULT_CONFIG } from "../backend/webgpu/matmul/types";
import { F32_NEG_MAX, WORKGROUP_SIZE } from "../backend/webgpu/shape-utils";
import { compileTileKernel } from "../backend/webgpu/tile-compiler";
import type {
  BlockExpr,
  KernelContext,
  SeamFn,
  TileKernelSpec,
} from "../backend/webgpu/tile-ir";
import { tiledGrid } from "../backend/webgpu/tile-ir";
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
  ValueUid,
} from "./types";

const uid = <T>(s: string): T => s as unknown as T;

// ============================================================================
// Tiling parameters (the fused-attention block shapes — the ScheduleState's
// declared block-shape params). OWNED HERE now (the cutover-flip): the schedule
// module owns the kernel body, so the block sizes it lowers are its facts; the
// dispatch layer (attention-kernel.ts) imports them back for grid/plan math.
// Kept as leaf constants so attention-kernel↔attention-skeleton has no eval-time
// circular dependency (the bodies + ATTENTION_PARAM_SCHEMA read them at eval).
// ============================================================================

export const BR = 64; // Q rows per workgroup (forward, dQ)
export const BC = 32; // KV rows per tile (forward, dQ)
export const BQ_BW = 16; // Q rows per tile (backward dKV)
export const BC_BW = 64; // KV rows per workgroup (backward dKV)

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
// The RECOMPUTATION-identity admitted lemma (§7 P4) — attention backward's license
// ============================================================================

/**
 * FlashAttention BACKWARD does NOT materialize the [S,S] probability matrix P.
 * Instead it saves the per-row logsumexp statistic `L = m + log(ℓ)` (the running
 * max + log of the running normalizer at forward exit) and RECOMPUTES each
 * probability block on the fly:  `P[i,j] = exp((Q_i·K_j)·scale − L_i)`. The raw
 * score is rebuilt from Q·K inside the backward loop; the softmax normalization
 * comes from the saved L. This is the RECOMPUTE-FROM-SAVED-STATISTIC identity:
 * an admitted rewrite that trades a materialized O(S²) intermediate for a saved
 * O(S) statistic + recomputation, licensed by the algebra fact
 * `exp(s − (m + log ℓ)) == exp(s − m) / ℓ` (the forward softmax value).
 *
 * CARRIED STATE: the per-row logsumexp `L` (a saved forward statistic, not a
 * recurrence). PROOF OBLIGATION: `exp(s − L) == softmax_row(s)` for the L the
 * forward produced — i.e. the recomputed P equals the forward P. This is the
 * engine object the dQ/dKV backward kernels' `p = exp(sMod − L)` sites are the
 * consumer of (attention-kernel.ts makeBackwardDQSpec / makeBackwardDKVSpec).
 */
export const RECOMPUTE_P_LEMMA = uid<LemmaUid>("lemma:attention-P-recompute");
export const RECOMPUTE_P_OBLIGATION = uid<ObligationId>(
  "obl:attention-P-recompute-equals-forward-P-from-logsumexp",
);

/** The recomputation-identity lemma application (carried statistic: L = logsumexp). */
export function recomputePLemma(): LemmaApplication {
  return {
    lemma: RECOMPUTE_P_LEMMA,
    obligation: RECOMPUTE_P_OBLIGATION,
    // Carried STATISTIC (saved from forward, not a recurrence): the per-row
    // logsumexp L = m + log(ℓ). P is recomputed as exp((Q·K)·scale − L).
    carriedStateRef:
      "carried=(L:logsumexp=m+log(l));recompute=P[i,j]=exp((Q_i.K_j)*scale-L_i)",
  };
}

// ============================================================================
// The D-PRECOMPUTE admitted lemma (§7 P4) — the rowsum(dO∘O) refactor
// ============================================================================

/**
 * The attention-backward gradient of the softmax row needs the term
 * `Σ_k P[i,k] · (dO_i · V_k)` subtracted from every `dO_i · V_j`. Naively that is
 * a full inner sum recomputed per (i,j). The D-PRECOMPUTE refactor observes that
 * this sum equals `dO_i · O_i` (because `O_i = Σ_k P[i,k] V_k`), so it can be
 * PRECOMPUTED ONCE PER ROW as the statistic `D_i = rowsum(dO_i ∘ O_i)` and then
 * carried into the dQ/dKV loops as `ds = P · (dO·V − D)`. This is an admitted
 * rewrite: it replaces the per-(i,j) recomputed inner sum with one saved per-row
 * statistic, licensed by the algebra fact `dO_i · O_i == Σ_k P[i,k](dO_i · V_k)`.
 *
 * CARRIED STATE: the per-row `D = rowsum(dO ∘ O)` (a precomputed reduction — the
 * dedicated `makeDPrecomputeSpec` kernel's output). PROOF OBLIGATION:
 * `rowsum(dO ∘ O) == Σ_k P[i,k]·(dO·V_k)`. The dQ/dKV kernels' `dov.sub(dVar)`
 * sites (attention-kernel.ts) are the consumer of the carried D.
 */
export const D_PRECOMPUTE_LEMMA = uid<LemmaUid>("lemma:attention-D-precompute");
export const D_PRECOMPUTE_OBLIGATION = uid<ObligationId>(
  "obl:attention-D-equals-rowsum-dO-hadamard-O",
);

/** The D-precompute lemma application (carried statistic: D = rowsum(dO∘O)). */
export function dPrecomputeLemma(): LemmaApplication {
  return {
    lemma: D_PRECOMPUTE_LEMMA,
    obligation: D_PRECOMPUTE_OBLIGATION,
    // Carried STATISTIC (precomputed once per row): D_i = Σ_d dO[i,d]·O[i,d].
    carriedStateRef:
      "carried=(D:rowsum(dO.O));refactor=sum_k P[i,k]*(dO_i.V_k)==dO_i.O_i",
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
  const isBackward =
    desc.role === "backwardDQ" ||
    desc.role === "backwardDKV" ||
    desc.role === "dPrecompute";
  return {
    visibility: "opaque",
    kernelRef: kernelRefFor(desc.role),
    refusalReason: isBackward
      ? // §7 P4 CUTOVER-FLIP: the LIVE-PATH flip is DONE — the backward body
        // (lowerBackward*Body / lowerDPrecomputeBody) LOWERS FROM the schedule
        // now; realizeAttentionSpec is the live dispatch chokepoint. The skeleton
        // STAYS opaque because its INTERNALS are still authored: the recompute-
        // from-logsumexp + D-precompute composite is proven in-grammar
        // (tools/fa-backward-derivation-script.ts reaches this kernel byte-
        // identically via the RECOMPUTATION + D-PRECOMPUTE admitted lemmas), but
        // the AUTOMATED internal opaque→derived flip waits on the S3 merge/fuse
        // composite transaction (islands altitude, unbuilt engine deliverable —
        // not a grammar gap). See docs/schedule-state-p4-local-self-hosting-report.md.
        "authored (LIVE-PATH FLIPPED, §7 P4). The backward body lowers FROM this " +
        "schedule module (realizeAttentionSpec is the live dispatch chokepoint); " +
        "the internals stay authored — the recompute-from-logsumexp + D-precompute " +
        "composite is re-derived in-grammar (RECOMPUTATION + D-PRECOMPUTE lemmas) " +
        "but the automated internal opaque→derived flip waits on the S3 merge/fuse " +
        "transaction (unbuilt engine deliverable, not a grammar gap)."
      : // Forward: LIVE-PATH flipped (body lowers from the schedule); the internal
        // online-softmax composite stays authored until S3.
        "authored (LIVE-PATH FLIPPED, §7 P4). The forward body lowers FROM this " +
        "schedule module (realizeAttentionSpec is the live dispatch chokepoint); " +
        "the internals stay authored — the online-softmax composite is re-derived " +
        "in-grammar via the FA-derivation (rungs 0–7: merge naive three islands " +
        "[S3] → tile → stream K/V → recolor accumulator → apply the online-softmax " +
        "admitted lemma; F17) but the automated internal opaque→derived flip waits " +
        "on the S3 merge/fuse composite transaction (unbuilt engine deliverable, " +
        "not a grammar gap).",
    params: ATTENTION_PARAM_SCHEMA,
  };
}

function kernelRefFor(role: AttentionKernelRole): string {
  // The body was ABSORBED into this schedule module (the cutover-flip); the
  // kernelRef names the schedule-owned lowering (single source), not the retired
  // attention-kernel.ts factory.
  switch (role) {
    case "forward":
      return "src/schedule/attention-skeleton.ts::lowerForwardAttentionBody";
    case "dPrecompute":
      return "src/schedule/attention-skeleton.ts::lowerDPrecomputeBody";
    case "backwardDQ":
      return "src/schedule/attention-skeleton.ts::lowerBackwardDQBody";
    case "backwardDKV":
      return "src/schedule/attention-skeleton.ts::lowerBackwardDKVBody";
  }
}

/**
 * `applyAttentionSchedule` for the authored family: assert the opaque skeleton
 * is well-formed at the seam (no loop/staging/role data leaked into it — F3),
 * then LOWER the authored kernel body from the schedule object. Returns the
 * `TileKernelSpec` the differential compiles.
 *
 * CUTOVER-FLIP (§7 P4): the four `make*Spec` kernel bodies (the online-softmax
 * K/V staging loops, the recompute-from-logsumexp backward structure, the
 * register×shared vec4 dot, the D-precompute reduction) were ABSORBED from
 * attention-kernel.ts INTO this schedule module (`lowerAttention*Body`). The
 * schedule object is now the SOLE owner of the attention kernel body at the
 * live dispatch seam (`realizeAttentionSpec`); the retired `make*Spec` factories
 * are gone. The skeleton STAYS opaque (F3) — its internals are still authored (a
 * locked online-softmax composite, not yet reached by the move grammar's `merge`
 * transaction; S3 unbuilt) — but the body they name now lives here, so the
 * schedule module is the single source. The BYTE DIFFERENTIAL proves the lowered
 * WGSL is byte-identical across the modifier corpus.
 */
export function applyAttentionSchedule(
  skeleton: Skeleton,
  desc: AttentionDescriptor,
): TileKernelSpec {
  assertAuthoredSeam(skeleton, desc);
  const mod = desc.modifier;
  switch (desc.role) {
    case "forward":
      return lowerForwardAttentionBody(desc.headDim, mod);
    case "dPrecompute":
      // D-precompute has no score/mask seam sites — one template for all mods.
      return lowerDPrecomputeBody(desc.headDim);
    case "backwardDQ":
      // Backward is inference-first for score modifiers (#64): the paired
      // derivative ("attn_dscore") is designed but unimplemented. The live
      // dispatch/plan path refuses a scoreMod via assertBackwardSupportsModifier
      // — re-call it here so the authored seam preserves the SAME refusal (one
      // owner of the backward-scoreMod legality fact).
      assertBackwardSupportsModifier(mod);
      return lowerBackwardDQBody(desc.headDim, mod);
    case "backwardDKV":
      assertBackwardSupportsModifier(mod);
      return lowerBackwardDKVBody(desc.headDim, mod);
  }
}

/**
 * Realize an attention `TileKernelSpec` THROUGH the schedule object (the live
 * dispatch chokepoint — the P1/reduction/matmul `realize*` pattern). The live
 * attention dispatch/plan sites in attention-kernel.ts route their spec factories
 * here: derive the authored (opaque) skeleton for the role, assert no-second-
 * owner at the seam, and lower the body FROM the state. The schedule object is
 * the sole WGSL writer at the dispatch seam; `lowerAttention*Body` are realizer-
 * internals of `applyAttentionSchedule`, unreachable from live dispatch except
 * through the schedule object.
 */
export function realizeAttentionSpec(
  role: AttentionKernelRole,
  headDim: number,
  mod?: AttnModifierSpec,
): TileKernelSpec {
  const desc: AttentionDescriptor = { role, headDim, modifier: mod };
  return applyAttentionSchedule(deriveAttentionSkeleton(desc), desc);
}

/** Compile the attention WGSL through the schedule object (the differential's
 *  live comparison target; formerly attention-kernel.ts `make*Spec` compiled
 *  directly, relocated here now that the schedule owns the body). */
export function generateAttentionShaderTileIR(
  role: AttentionKernelRole,
  headDim: number,
  mod?: AttnModifierSpec,
): string {
  return compileTileKernel(realizeAttentionSpec(role, headDim, mod));
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

// ============================================================================
// The NAIVE attention BACKWARD composition (§7 P4 — the backward starting position)
// ============================================================================

/**
 * The naive (autograd-composed) attention BACKWARD as a multi-region composition,
 * the P4 starting position for re-deriving the authored dQ/dKV/D kernels. The
 * naive backward MATERIALIZES the [S,S] intermediates:
 *
 *   dV = Pᵀ @ dO                       (matmul TN)
 *   dP = dO @ Vᵀ                       (matmul NT)  — [S,S], materialized
 *   dS = P ∘ (dP − rowsum(dP ∘ P))     (softmax-backward row-program)
 *   dQ = dS @ K,  dK = dSᵀ @ Q         (matmul NN / TN)
 *
 * The two admitted rewrites the derivation applies:
 *   - RECOMPUTATION: dP's materialized P (and P itself) is recomputed from the
 *     saved logsumexp L instead of stored — `materialized_P` → `recompute_P`.
 *   - D-PRECOMPUTE: the `rowsum(dP ∘ P) == rowsum(dO ∘ O)` inner sum is carried out
 *     of the loop as the per-row statistic D — `inline_softmax_grad_innersum` →
 *     `precomputed_D`. This is the dedicated D-precompute kernel's raison d'être.
 *
 * After both discharges + the fuse/tile/stream moves, the backward reaches the
 * authored three-kernel shape (D-precompute + dQ + dKV). Like the forward, the
 * per-region matmul/row-program states round-trip byte-identically via the REUSED
 * family seams; the softmax-backward region carries the two backward markers the
 * lemmas act on.
 */
export interface NaiveAttentionBackwardComposition {
  /** dV = Pᵀ @ dO (matmul TN, no epilogue). */
  readonly dV: {
    region: SemanticRegionUid;
    state: ScheduleState;
    desc: TiledMatmulDescriptor;
  };
  /** dP = dO @ Vᵀ (matmul NT) — the [S,S] materialized intermediate. */
  readonly dP: {
    region: SemanticRegionUid;
    state: ScheduleState;
    desc: TiledMatmulDescriptor;
  };
  /** dS = softmax-backward row-program — carries the P-recompute + D-inner-sum
   *  markers the RECOMPUTATION and D-PRECOMPUTE lemmas discharge. */
  readonly dS: {
    region: SemanticRegionUid;
    state: ScheduleState;
    program: RowProgram;
    /** The softmax-backward body's value UID (the lemma-target for stream). */
    dsValue: ValueUid;
  };
  /** dQ = dS @ K (matmul NN, no epilogue). */
  readonly dQ: {
    region: SemanticRegionUid;
    state: ScheduleState;
    desc: TiledMatmulDescriptor;
  };
  /** dK = dSᵀ @ Q (matmul TN, no epilogue). */
  readonly dK: {
    region: SemanticRegionUid;
    state: ScheduleState;
    desc: TiledMatmulDescriptor;
  };
  /** The island-flow edges: dO→{dV,dP}, dP→dS, dS→{dQ,dK}. */
  readonly islandFlow: readonly {
    readonly from: SemanticRegionUid;
    readonly to: SemanticRegionUid;
    readonly via: string;
  }[];
  readonly headDim: number;
}

/**
 * The softmax-backward RowProgram (the dS region): a numerically-stable softmax
 * gradient over the KV axis. Phase 0 reduces the inner sum `D = Σ_k dP[k]·P[k]`
 * (the term the D-precompute lemma refactors out); the write phase emits
 * `dS[j] = P[j]·(dP[j] − D)`. Same RowProgram shape the graph compiler detects
 * for a real softmax-backward, so it round-trips byte-identically via the
 * row-program family differential.
 */
export function softmaxBackwardRowProgram(): RowProgram {
  const dP = (): RPExprInput => ({ kind: "input", bufferIndex: 0 });
  const P = (): RPExprInput => ({ kind: "input", bufferIndex: 1 });
  const Dsum = (): RPExprRef => ({ kind: "reduceResult", phaseIndex: 0 });
  return {
    inputs: [{ dtype: "f32" }, { dtype: "f32" }],
    output: { dtype: "f32" },
    dim: -1,
    phases: [
      // Phase 0: D = Σ_k dP[k]·P[k]  (== rowsum(dO∘O), the D-precompute statistic)
      { kind: "reduce", reduceOp: "sum", bodyExpr: { op: "mul", inputs: [dP(), P()] } },
      // Write: dS[j] = P[j]·(dP[j] − D)
      {
        kind: "write",
        bodyExpr: {
          op: "mul",
          inputs: [P(), { op: "sub", inputs: [dP(), Dsum()] }],
        },
      },
    ],
    cacheKey: "attn-naive-softmax-backward-f32",
  };
}

/**
 * Build the naive attention-backward composition. The dV/dP/dQ/dK regions reuse
 * the matmul family (`deriveTiledMatmulState`); the dS region reuses the
 * row-program family over `softmaxBackwardRowProgram()`. The island-flow edges
 * connect them. This is P4's starting position for the backward derivation.
 */
export function naiveAttentionBackwardComposition(
  headDim: number,
): NaiveAttentionBackwardComposition {
  const dVRegion = uid<SemanticRegionUid>("region:attn-bwd-naive-dV");
  const dPRegion = uid<SemanticRegionUid>("region:attn-bwd-naive-dP");
  const dSRegion = uid<SemanticRegionUid>("region:attn-bwd-naive-dS");
  const dQRegion = uid<SemanticRegionUid>("region:attn-bwd-naive-dQ");
  const dKRegion = uid<SemanticRegionUid>("region:attn-bwd-naive-dK");

  const mm = (transposeMode: TiledMatmulDescriptor["transposeMode"]) => ({
    config: { ...DEFAULT_CONFIG },
    transposeMode,
    dtype: "f32" as const,
  });

  // dV = Pᵀ @ dO (TN); dP = dO @ Vᵀ (NT); dQ = dS @ K (NN); dK = dSᵀ @ Q (TN).
  const dVDesc = mm("TN");
  const dPDesc = mm("NT");
  const dQDesc = mm("NN");
  const dKDesc = mm("TN");

  const dVState = deriveTiledMatmulState(dVDesc, dVRegion);
  const dPState = deriveTiledMatmulState(dPDesc, dPRegion);
  const dQState = deriveTiledMatmulState(dQDesc, dQRegion);
  const dKState = deriveTiledMatmulState(dKDesc, dKRegion);

  const program = softmaxBackwardRowProgram();
  const dSState = deriveRowProgramState(program, dSRegion);
  const dsValue = uid<ValueUid>("value:attn-bwd:dS");

  return {
    dV: { region: dVRegion, state: dVState, desc: dVDesc },
    dP: { region: dPRegion, state: dPState, desc: dPDesc },
    dS: { region: dSRegion, state: dSState, program, dsValue },
    dQ: { region: dQRegion, state: dQState, desc: dQDesc },
    dK: { region: dKRegion, state: dKState, desc: dKDesc },
    islandFlow: [
      { from: dVRegion, to: dSRegion, via: "dO" },
      { from: dPRegion, to: dSRegion, via: "dP" },
      { from: dSRegion, to: dQRegion, via: "dS" },
      { from: dSRegion, to: dKRegion, via: "dS" },
    ],
    headDim,
  };
}

// ============================================================================
// The ABSORBED authored attention kernel bodies (§7 P4 cutover-flip)
// ============================================================================
//
// The four fused-FlashAttention kernel bodies — RELOCATED from
// attention-kernel.ts `make*Spec` — so the schedule module is the SOLE owner of
// the attention kernel structure at the live dispatch seam (`realizeAttentionSpec`).
// The K/V staging loops, the online-softmax (m,ℓ,o) accumulation with the
// exp-rescale correction, the register×shared vec4 dot, and the backward
// recompute-from-logsumexp structure LOWER FROM here now (the P4 derivations —
// tools/fa-derivation-script.ts + tools/fa-backward-derivation-script.ts — prove
// these states carry everything, operand-residency decorations included). The
// bodies are BYTE-IDENTICAL to the retired factories (the 14-kernel modifier
// differential is the safety net). The skeleton STAYS opaque (F3): the internals
// are still authored (a locked online-softmax composite; S3 merge/fuse — the
// automated opaque→derived flip — is the remaining engine deliverable), but the
// body they name lives here, so there is one owner. The seam parameters
// (scoreMod/maskMod, #64) survive as typed modifier data threaded through
// buildAttentionSeams / modifierUniformFields.

/** Build the seam functions for a modifier (undefined for the null modifier
 *  — applySeam is identity, the kernel is the bare bounds-checked softmax).
 *  RELOCATED from attention-kernel.ts (the body's #64 seam emission). */
function buildAttentionSeams(
  mod: AttnModifierSpec | undefined,
): Record<string, SeamFn> | undefined {
  if (!mod) return undefined;
  const seams: Record<string, SeamFn> = {};
  const maskMods = mod.maskMods ?? [];
  if (maskMods.length > 0) {
    seams.attn_mask = (
      ctx: KernelContext,
      active: BlockExpr,
      args: Record<string, BlockExpr>,
    ) => {
      let a = active;
      for (const m of maskMods) {
        if (m.kind === "causal") {
          a = a.and(args.kvIdx.le(args.qIdx));
        } else if (m.kind === "slidingWindow") {
          // active iff kv > q − window, computed u32-underflow-safe as
          // kv + window > q. The window value is uniform DATA (mod_window)
          // — same template serves any window size.
          a = a.and(args.kvIdx.add(ctx.uniform("mod_window")).gt(args.qIdx));
        } else {
          throw new Error(
            `attention maskMod '${(m as { kind: string }).kind}' not implemented in kernel emission`,
          );
        }
      }
      return a;
    };
  }
  if (mod.scoreMod) {
    if (mod.scoreMod.kind !== "softcap") {
      throw new Error(
        `attention scoreMod '${(mod.scoreMod as { kind: string }).kind}' not implemented in kernel emission`,
      );
    }
    // Logit soft-cap: s' = cap · tanh(s / cap) (Gemma-2). Emitted in f32 —
    // modifier arithmetic stays f32 under f16 QKV (mandatory f16 gate).
    // Backward's paired "attn_dscore" (1 − (s'/cap)²) is inference-first —
    // backward entries throw via assertBackwardSupportsModifier.
    seams.attn_score = (ctx: KernelContext, sVal: BlockExpr) => {
      const cap = ctx.uniform("mod_softcap").bitcastTo("f32");
      return sVal.div(cap).tanh().mul(cap);
    };
  }
  return seams;
}

/** Uniform struct fields for modifier params — paired with the config-buffer's
 *  modifierParamWords (same order). Spread after scale_u32 in each seam-site
 *  spec's uniforms. RELOCATED from attention-kernel.ts (body uniform layout). */
function modifierUniformFields(mod?: AttnModifierSpec): Record<string, "u32"> {
  if (!mod) return {};
  const fields: Record<string, "u32"> = {};
  if (mod.scoreMod) fields.mod_softcap = "u32";
  for (const m of mod.maskMods ?? []) {
    if (m.kind === "slidingWindow") fields.mod_window = "u32";
  }
  return fields;
}

/** WGSL-identifier-safe name fragment ("" for null modifier). RELOCATED from
 *  attention-kernel.ts (the body's spec name). */
function modNameFragment(mod?: AttnModifierSpec): string {
  const k = attnModifierKey(mod);
  return k ? `_${k.replace(/[^A-Za-z0-9]/g, "_")}` : "";
}

/** The fused FlashAttention FORWARD body (online softmax over KV tiles). */
function lowerForwardAttentionBody(
  headDim: number,
  mod?: AttnModifierSpec,
): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BR;

  return {
    name: `tileAttnFwd_D${D}${modNameFragment(mod)}`,
    workgroupSize: WG,
    autoBarriers: true,
    seams: buildAttentionSeams(mod),
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      O: { storage: "read_write", type: "f32" },
      L: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      ...modifierUniformFields(mod),
    },
    grid: tiledGrid({
      x: { uniform: "seq_len", tileSize: BR },
      y: "num_heads",
      z: "batch_size",
    }),

    kernel(ctx) {
      const tidx = ctx.localIndex();
      const qBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const qRow = qBlock.mul(ctx.u32(BR)).add(tidx);
      const valid = qRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(N);
      const qBase = bhOff.add(qRow.mul(Dim));

      const Q = ctx.tileLoad(
        "Q",
        {
          kind: "thread",
          base: qBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const mPrev = ctx.full(1, 1, F32_NEG_MAX);
      const lPrev = ctx.full(1, 1, 0);
      const oAcc = ctx.zeros(1, D);

      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));

        const offsR = ctx.arange(kvStart, BC);
        const offsD = ctx.arange(ctx.u32(0), D);
        const tilePtr = ctx.tilePtr(
          bhOff,
          offsR.outer(Dim),
          offsD.inner(ctx.u32(1)),
        );
        const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
        const K = ctx.load2D("K", tilePtr, tileMask);

        const scores = ctx.dot(Q, K.T());

        ctx.range(0, BC, (j) => {
          const kvPos = kvStart.add(j);
          const seamArgs = {
            qIdx: qRow,
            kvIdx: kvPos,
            head: hIdx,
            batch: bIdx,
          };
          // Seam "attn_mask": mask predicates (incl. causal) are structural
          // modifier emissions AND'ed onto the bounds check.
          const isActive = ctx.applySeam(
            "attn_mask",
            valid.and(kvPos.lt(N)),
            seamArgs,
          );
          // Seam "attn_score": wraps the scaled pre-softmax score.
          const s = ctx.applySeam(
            "attn_score",
            scores.get(j).mul(scale),
            seamArgs,
          );
          scores.set(j, isActive.select(s, ctx.f32(F32_NEG_MAX)));
        });

        const mNew = scores.max(1);
        const mMax = mNew.max(mPrev);
        const correction = mPrev.sub(mMax).exp();

        oAcc.mul_(correction);
        lPrev.mul_(correction);

        scores.sub_(mMax);
        scores.exp_();
        lPrev.add_(scores.sum(1));
        mPrev.assign(mMax);

        const V = ctx.load2D("V", tilePtr, tileMask, { reuseShared: K });
        ctx.dotAccum(scores, V, oAcc);
      });

      ctx.ifThen(valid, () => {
        const l = lPrev.get(ctx.u32(0));
        const invL = l.gt(ctx.f32(0)).select(ctx.f32(1).div(l), ctx.f32(0));
        oAcc.mul_(invL);
        ctx.tileStore("O", oAcc, { base: qBase, stride: ctx.u32(1) });

        const m = mPrev.get(ctx.u32(0));
        const lse = m.add(l.max(ctx.f32(1e-10)).log());
        ctx.emitStore("L", bhOffL.add(qRow), lse);
      });
    },
  };
}

/** The D-precompute body (per-row D = rowsum(dO ∘ O), the D-precompute lemma's
 *  carried statistic). No seam sites — one template for all modifiers. */
function lowerDPrecomputeBody(headDim: number): TileKernelSpec {
  const D = headDim;
  const WG = WORKGROUP_SIZE;

  return {
    name: `tileAttnDPrecompute_D${D}`,
    workgroupSize: WG,
    bindings: {
      dO: { storage: "read", type: "f32" },
      Out: { storage: "read", type: "f32" },
      D_val: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
    },
    grid: (u) => [u.batch_size * u.num_heads * u.seq_len],

    kernel(ctx) {
      const row = ctx.programId(0);
      const tid = ctx.localIndex();
      const Dim = ctx.uniform("head_dim");
      const base = row.mul(Dim);

      const dotProd = ctx.wgReduce("sum", tid, Dim, WG, (i) =>
        ctx.load("dO", base.add(i)).mul(ctx.load("Out", base.add(i))),
      );
      ctx.guardedStore("D_val", tid.eq(ctx.u32(0)), row, dotProd);
    },
  };
}

/** The FlashAttention backward-dQ body (recompute P from saved logsumexp L). */
function lowerBackwardDQBody(
  headDim: number,
  mod?: AttnModifierSpec,
): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BR;

  return {
    name: `tileAttnBwdDQ_D${D}${modNameFragment(mod)}`,
    workgroupSize: WG,
    autoBarriers: true,
    seams: buildAttentionSeams(mod),
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      L_buf: { storage: "read", type: "f32" },
      D_buf: { storage: "read", type: "f32" },
      dO: { storage: "read", type: "f32" },
      dQ: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      ...modifierUniformFields(mod),
    },
    grid: tiledGrid({
      x: { uniform: "seq_len", tileSize: BR },
      y: "num_heads",
      z: "batch_size",
    }),

    kernel(ctx) {
      const tidx = ctx.localIndex();
      const qBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const qRow = qBlock.mul(ctx.u32(BR)).add(tidx);
      const valid = qRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(N);
      const rowBase = bhOff.add(qRow.mul(Dim));

      const Q = ctx.tileLoad(
        "Q",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const dO = ctx.tileLoad(
        "dO",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const lVar = ctx.emitVar("_Li", "f32", ctx.f32(0));
      const dVar = ctx.emitVar("_Di", "f32", ctx.f32(0));
      ctx.ifThen(valid, () => {
        lVar.set(ctx.load("L_buf", bhOffL.add(qRow)));
        dVar.set(ctx.load("D_buf", bhOffL.add(qRow)));
      });

      const dqAcc = ctx.zeros(1, D);

      const numKVTiles = N.add(ctx.u32(BC - 1)).div(ctx.u32(BC));

      ctx.forRange(ctx.u32(0), numKVTiles, (tile) => {
        const kvStart = tile.mul(ctx.u32(BC));

        const offsR = ctx.arange(kvStart, BC);
        const offsD = ctx.arange(ctx.u32(0), D);
        const tilePtr = ctx.tilePtr(
          bhOff,
          offsR.outer(Dim),
          offsD.inner(ctx.u32(1)),
        );
        const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
        const K = ctx.load2D("K", tilePtr, tileMask);
        const V = ctx.load2D("V", tilePtr, tileMask);

        // Fused single-loop: compute score, p, ds per KV-row, accumulate dQ inline
        ctx.range(0, BC, (j) => {
          const kvPos = kvStart.add(j);
          const seamArgs = {
            qIdx: qRow,
            kvIdx: kvPos,
            head: hIdx,
            batch: bIdx,
          };
          const isActive = ctx.applySeam(
            "attn_mask",
            valid.and(kvPos.lt(N)),
            seamArgs,
          );
          // Flash-style recompute: raw score is rebuilt from Q·K, so the
          // score modifier's forward AND its local derivative (the
          // "attn_dscore" seam, chain factor d(modded)/d(raw)) are available
          // inline — no extra saved tensor.
          const s = ctx.dotRow(Q, K, j).mul(scale);
          const sMod = ctx.applySeam("attn_score", s, seamArgs);
          const dov = ctx.dotRow(dO, V, j);
          const p = isActive.select(sMod.sub(lVar.get()).exp(), ctx.f32(0));
          const ds = ctx.applySeam("attn_dscore", p.mul(dov.sub(dVar.get())), {
            ...seamArgs,
            raw: s,
            modded: sMod,
          });
          ctx.accumRow(dqAcc, ds, K, j);
        });
      });

      ctx.ifThen(valid, () => {
        dqAcc.mul_(scale);
        ctx.tileStore("dQ", dqAcc, { base: rowBase, stride: ctx.u32(1) });
      });
    },
  };
}

/** The FlashAttention backward-dK/dV body (recompute P from L; accumulate dK,dV). */
function lowerBackwardDKVBody(
  headDim: number,
  mod?: AttnModifierSpec,
): TileKernelSpec {
  if (headDim % 4 !== 0)
    throw new Error(`headDim must be divisible by 4, got ${headDim}`);
  const D = headDim;
  const WG = BC_BW;

  return {
    name: `tileAttnBwdDKV_D${D}${modNameFragment(mod)}`,
    workgroupSize: WG,
    autoBarriers: true,
    seams: buildAttentionSeams(mod),
    bindings: {
      Q: { storage: "read", type: "f32" },
      K: { storage: "read", type: "f32" },
      V: { storage: "read", type: "f32" },
      L_buf: { storage: "read", type: "f32" },
      D_buf: { storage: "read", type: "f32" },
      dO: { storage: "read", type: "f32" },
      dK: { storage: "read_write", type: "f32" },
      dV: { storage: "read_write", type: "f32" },
    },
    uniforms: {
      batch_size: "u32",
      num_heads: "u32",
      seq_len: "u32",
      head_dim: "u32",
      scale_u32: "u32",
      ...modifierUniformFields(mod),
    },
    grid: tiledGrid({
      x: { uniform: "seq_len", tileSize: BC_BW },
      y: "num_heads",
      z: "batch_size",
    }),

    kernel(ctx) {
      const tidx = ctx.localIndex();
      const kvBlock = ctx.programId(0);
      const hIdx = ctx.programId(1);
      const bIdx = ctx.programId(2);

      const N = ctx.uniform("seq_len");
      const Dim = ctx.u32(D);
      const numHeads = ctx.uniform("num_heads");
      const scale = ctx.uniform("scale_u32").bitcastTo("f32");

      const kvRow = kvBlock.mul(ctx.u32(BC_BW)).add(tidx);
      const valid = kvRow.lt(N);

      const bhOff = bIdx.mul(numHeads).add(hIdx).mul(N).mul(Dim);
      const bhOffL = bIdx.mul(numHeads).add(hIdx).mul(N);
      const rowBase = bhOff.add(kvRow.mul(Dim));

      const K = ctx.tileLoad(
        "K",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const V = ctx.tileLoad(
        "V",
        {
          kind: "thread",
          base: rowBase,
          stride: ctx.u32(1),
        },
        { rows: 1, cols: D, guard: valid },
      );

      const dkAcc = ctx.zeros(1, D);
      const dvAcc = ctx.zeros(1, D);

      const lTile = ctx.sharedArray("L_tile", BQ_BW, "f32");
      const dTile = ctx.sharedArray("D_tile", BQ_BW, "f32");

      const numQTiles = N.add(ctx.u32(BQ_BW - 1)).div(ctx.u32(BQ_BW));

      ctx.forRange(ctx.u32(0), numQTiles, (qt) => {
        const qStart = qt.mul(ctx.u32(BQ_BW));

        // Causal tile-skip: Q-tiles entirely above the diagonal contribute
        // nothing — baked structurally when a causal maskMod is present (the
        // affine-mask → loop-bound rule's precedent; window bounds land with
        // #64 iii). Workgroup-uniform (qStart/kvBlock only).
        const skipTile = hasCausalMask(mod)
          ? qStart.add(ctx.u32(BQ_BW - 1)).lt(kvBlock.mul(ctx.u32(BC_BW)))
          : undefined;

        const emitTileBody = () => {
          const offsR = ctx.arange(qStart, BQ_BW);
          const offsD = ctx.arange(ctx.u32(0), D);
          const tilePtr = ctx.tilePtr(
            bhOff,
            offsR.outer(Dim),
            offsD.inner(ctx.u32(1)),
          );
          const tileMask = ctx.tileMask(offsR.lt(N), offsD.lt(Dim));
          const QTile = ctx.load2D("Q", tilePtr, tileMask);
          const dOTile = ctx.load2D("dO", tilePtr, tileMask);

          ctx.ifThen(tidx.lt(ctx.u32(BQ_BW)), () => {
            const qi = qStart.add(tidx);
            const inBounds = qi.lt(N);
            const lIdx = bhOffL.add(qi);
            lTile.write(
              tidx,
              inBounds.select(ctx.load("L_buf", lIdx), ctx.f32(0)),
            );
            dTile.write(
              tidx,
              inBounds.select(ctx.load("D_buf", lIdx), ctx.f32(0)),
            );
          });

          ctx.barrier();

          // Fused single-loop: compute score, p, ds per Q-row, accumulate dK/dV inline
          ctx.range(0, BQ_BW, (j) => {
            const qi = qStart.add(j);
            const seamArgs = {
              qIdx: qi,
              kvIdx: kvRow,
              head: hIdx,
              batch: bIdx,
            };
            const isActive = ctx.applySeam(
              "attn_mask",
              valid.and(qi.lt(N)),
              seamArgs,
            );
            const s = ctx.dotRow(K, QTile, j).mul(scale);
            const sMod = ctx.applySeam("attn_score", s, seamArgs);
            const p = isActive.select(
              sMod.sub(lTile.read(j)).exp(),
              ctx.f32(0),
            );
            const dov = ctx.dotRow(V, dOTile, j);
            // "attn_dscore" applies BEFORE the trailing d(raw)/d(QK) scale
            // factor — the chain multiplies d(modded)/d(raw) into dS first.
            const ds = ctx
              .applySeam("attn_dscore", p.mul(dov.sub(dTile.read(j))), {
                ...seamArgs,
                raw: s,
                modded: sMod,
              })
              .mul(scale);
            ctx.accumRow(dkAcc, ds, QTile, j);
            ctx.accumRow(dvAcc, p, dOTile, j);
          });

          ctx.barrier();
        };
        if (skipTile) ctx.ifThen(skipTile.not(), emitTileBody);
        else emitTileBody();
      });

      ctx.ifThen(valid, () => {
        ctx.tileStore("dK", dkAcc, { base: rowBase, stride: ctx.u32(1) });
        ctx.tileStore("dV", dvAcc, { base: rowBase, stride: ctx.u32(1) });
      });
    },
  };
}
