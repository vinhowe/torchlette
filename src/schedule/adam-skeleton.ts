/**
 * FUSED ADAM schedule module (§7 P4 LOCAL self-hosting + CUTOVER-FLIP). The live
 * Adam dispatch path now ROUTES THROUGH this module: the fused-Adam kernel body
 * was ABSORBED here from adam-kernel.ts (`lowerAdamStepBody`), so the ScheduleState
 * OWNS the kernel structure and `realizeAdamStepSpec` is the live dispatch
 * chokepoint (the adam-kernel.ts `getAdamDispatcher` builds through it). NO
 * behavior change — the body is byte-identical to the retired `makeAdamStepSpec`
 * factory (the 5-variant differential guards the LIVE path).
 *
 * ------------------------------------------------------------------------
 * THE DERIVED KERNEL + THE HORIZONTAL-PACK DERIVATION
 * ------------------------------------------------------------------------
 * The fused Adam kernel (adamStep, single WGSL dispatch: reads grad/param/m/v/bc/lr,
 * writes param/m/v). R4 (2026-07-22): its per-element update body is DERIVED from
 * ADAMW_PROGRAM via the OptTerm→tile-IR fold (`emitDerivedAdamScalarBody`) — the
 * executed kernel is a theorem of the program (ruling O1); the authored body
 * (emitAdamScalarBody/emitBiasCorrection/emitExpm1) is deleted. Bias correction
 * rides in as a `[2]` `bc` DATA input (fork C: a host-computed live scalar). The
 * skeleton stays a black-box dispatch wrapper (F3) with a TYPED PARAMETER SCHEMA,
 * exactly as the attention kernels do. `applyAdamSchedule` LOWERS the (derived)
 * body FROM the state; the byte differential proves the two entry points agree.
 *
 * The DERIVABLE part — the §7 P4 deliverable-3 tenant of the `pack` move — is the
 * HORIZONTAL PACKING at parameter altitude. The optimizer's own words
 * (optim/adam.ts): "the per-param definition is the semantics, the packed form is
 * the batched execution of the same tensor program." N per-parameter Adam bodies
 * (each an elementwise update over one param's flat) are PACKED into one flat
 * dispatch by the `pack` move: concatenate the N params/grads/m/v into one
 * `[total]` buffer, run ONE adamStep over it. That is the multi-tensor packing the
 * §3 `pack` move expresses — the real tenant of the move the FA forward derivation
 * never exercised (it packed nothing). The packed schedule reaches the SAME
 * authored kernel (one dispatch), and the packed-dispatch COUNT (one adamStep per
 * group, not N) is the perf-parity target (the 8-dispatch/step baseline class).
 */

import { compileTileKernel } from "../backend/webgpu/tile-compiler";
import type { TileKernelSpec } from "../backend/webgpu/tile-ir";
import { reportNoSecondOwner } from "./canonical";
import { applyMove } from "./moves/moves";
import { lowerOptStepBody } from "./opt-step-realizer";
import { ADAM_STEP_SPEC } from "./opt-step-specs";
import type {
  PredicateAstNode,
  ScheduleMove,
  ScheduleState,
  SemanticSchedule,
  Skeleton,
  TypedParamSchema,
  ValueUid,
} from "./types";

const uid = <T>(s: string): T => s as unknown as T;

// ============================================================================
// The typed parameter schema (R10 + F7) — the fused-Adam constraint set
// ============================================================================

function paramLeaf(name: string): PredicateAstNode {
  return { kind: "uniformRef", name };
}

/**
 * The fused-Adam variant is keyed on three booleans (useVec4 × emitF16 ×
 * emitUnscale) — the SAME variant key `getAdamDispatcher` uses (single source).
 * The vec4 path requires the unscale route (atomics prevent auto-vec4 otherwise);
 * the schema's dependent constraint names it: useVec4 ⇒ emitUnscale.
 */
function vec4RequiresUnscaleConstraint(): PredicateAstNode {
  // NOT useVec4  OR  emitUnscale  (useVec4 ⇒ emitUnscale)
  return {
    kind: "or",
    terms: [
      {
        kind: "cmp",
        op: "==",
        lhs: paramLeaf("useVec4"),
        rhs: { kind: "intLit", value: 0 },
      },
      {
        kind: "cmp",
        op: "==",
        lhs: paramLeaf("emitUnscale"),
        rhs: { kind: "intLit", value: 1 },
      },
    ],
  };
}

/** The fused-Adam family's declared parameter schema (§6, R10/F7). The three
 *  variant booleans are the DECLARED editable parameters; the dependent
 *  constraint is the vec4⇒unscale requirement. No device capability predicate
 *  (the kernel runs on any WebGPU device; chunking handles the >128MB case). */
export const ADAM_PARAM_SCHEMA: TypedParamSchema = {
  params: {
    useVec4: { domain: [0, 1], default: 0 },
    emitF16: { domain: [0, 1], default: 0 },
    emitUnscale: { domain: [0, 1], default: 0 },
  },
  constraints: [vec4RequiresUnscaleConstraint()],
  // Trivially-satisfied capability: any WebGPU device (1 == 1).
  capabilityPredicate: {
    kind: "cmp",
    op: "==",
    lhs: { kind: "intLit", value: 1 },
    rhs: { kind: "intLit", value: 1 },
  },
};

// ============================================================================
// The authored (opaque) fused-Adam skeleton (§6 / F3)
// ============================================================================

/** A structured description of ONE authored fused-Adam kernel variant — the
 *  honest reification of the live `makeAdamStepSpec` factory's inputs. */
export interface AdamDescriptor {
  readonly useVec4: boolean;
  readonly emitF16: boolean;
  readonly emitUnscale: boolean;
}

// The body was ABSORBED into this schedule module (the cutover-flip); the
// kernelRef names the schedule-owned lowering (single source), not the retired
// adam-kernel.ts factory.
const KERNEL_REF = "src/schedule/adam-skeleton.ts::lowerAdamStepBody";

/**
 * The canonical cache-key encoder (R10) for an authored Adam kernel: the SAME
 * variant key `getAdamDispatcher` uses (`${useVec4}:${emitF16}:${emitUnscale}`) —
 * delegated to that single source rather than re-derived (no second owner of the
 * template-identity fact).
 */
export function adamCacheKey(desc: AdamDescriptor): string {
  return `adam:${desc.useVec4}:${desc.emitF16}:${desc.emitUnscale}`;
}

/**
 * Derive the AUTHORED (opaque) skeleton for one fused-Adam variant (§6 / F3). The
 * skeleton NAMES the live kernel (kernelRef) + a refusal reason, and carries the
 * TYPED PARAM SCHEMA. It FORBIDS loop/staging/role data (F3). The refusal reason
 * records that the per-element update is a LOCKED numeric formula (authored); the
 * DERIVABLE part is the horizontal packing (`deriveHorizontalPackedAdam`).
 */
export function deriveAdamSkeleton(_desc: AdamDescriptor): Skeleton {
  return {
    visibility: "opaque",
    kernelRef: KERNEL_REF,
    refusalReason:
      "derived (LIVE-PATH FLIPPED, §7 P4; R4 body-derivation 2026-07-22). The " +
      "fused-Adam body lowers FROM this schedule module (realizeAdamStepSpec is the " +
      "live dispatch chokepoint); the per-element update is now DERIVED from " +
      "ADAMW_PROGRAM via the OptTerm→tile-IR fold (lowerAdamStepBody→" +
      "emitDerivedAdamScalarBody) — the executed kernel is a theorem of the program " +
      "(ruling O1), NOT a second authored copy. Bias correction rides in as a [2] " +
      "`bc` DATA input (fork C: a host-computed live scalar). The skeleton stays a " +
      "black-box dispatch wrapper (no loop/staging/role data — F3). The DERIVABLE " +
      "packing is the HORIZONTAL-PACK move (§7 P4 deliverable 3): N per-param " +
      "elementwise Adam bodies pack into one flat dispatch — the per-param " +
      "definition is the semantics, the packed form is the batched execution. §6/§7 P4.",
    params: ADAM_PARAM_SCHEMA,
  };
}

/**
 * `applyAdamSchedule` for the authored family: assert the opaque skeleton is
 * well-formed at the seam (no loop/staging/role leaked — F3), then LOWER the
 * authored fused-Adam body from the schedule object. Returns the `TileKernelSpec`
 * the differential compiles.
 *
 * CUTOVER-FLIP (§7 P4): the `makeAdamStepSpec` body (the bias-corrected step-size
 * via expm1, the L2/decoupled weight decay, the atomic-guarded unscale vec4 path)
 * was ABSORBED from adam-kernel.ts INTO this schedule module (`lowerAdamStepBody`).
 * The schedule object is now the SOLE owner of the fused-Adam kernel body at the
 * live dispatch seam (`realizeAdamStepSpec`); the retired `makeAdamStepSpec`
 * factory is gone. The skeleton STAYS opaque (F3) — the per-element update is a
 * locked numeric formula — but the body it names now lives here. The BYTE
 * DIFFERENTIAL proves the lowered WGSL is byte-identical across all 5 variants.
 */
export function applyAdamSchedule(
  skeleton: Skeleton,
  desc: AdamDescriptor,
): TileKernelSpec {
  assertAuthoredSeam(skeleton, desc);
  return lowerAdamStepBody(desc.useVec4, desc.emitF16, desc.emitUnscale);
}

/**
 * Realize the fused-Adam `TileKernelSpec` THROUGH the schedule object (the live
 * dispatch chokepoint — the P1/matmul/reduction `realize*` pattern). The live
 * Adam dispatcher (adam-kernel.ts `getAdamDispatcher`) routes here: derive the
 * authored (opaque) skeleton for the variant, assert no-second-owner at the seam,
 * and lower the body FROM the state. The schedule object is the sole WGSL writer
 * at the dispatch seam; `lowerAdamStepBody` is a realizer-internal of
 * `applyAdamSchedule`, unreachable from live dispatch except through the schedule.
 */
export function realizeAdamStepSpec(
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): TileKernelSpec {
  // R4 (2026-07-22): the per-element body is DERIVED from ADAMW_PROGRAM
  // (lowerOptTermToTileIR) — the sole path. It flows through the schedule
  // chokepoint (applyAdamSchedule ∘ deriveAdamSkeleton), which now lowers the
  // derived body; the executed kernel is a theorem of the program (ruling O1).
  const desc: AdamDescriptor = { useVec4, emitF16, emitUnscale };
  return applyAdamSchedule(deriveAdamSkeleton(desc), desc);
}

/** Compile the fused-Adam WGSL through the schedule object (the differential's
 *  live comparison target; formerly adam-kernel.ts `makeAdamStepSpec` compiled
 *  directly, relocated here now that the schedule owns the body). */
export function generateAdamShaderTileIR(
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): string {
  return compileTileKernel(
    realizeAdamStepSpec(useVec4, emitF16, emitUnscale),
  );
}

/**
 * Assert the authored fused-Adam skeleton carries no second owner and matches the
 * descriptor at the seam (§12 check 3, family-local). The opaque skeleton MUST
 * name the correct live kernel (kernelRef), MUST carry the single param schema,
 * and MUST satisfy its own vec4⇒unscale constraint (the live factory's atomics
 * path is the SAME fact — one owner).
 */
export function assertAuthoredSeam(
  skeleton: Skeleton,
  desc: AdamDescriptor,
): void {
  if (skeleton.visibility !== "opaque") {
    reportNoSecondOwner(
      `no-second-owner[adam]: the fused Adam kernel MUST wear an opaque skeleton ` +
        `(F3 authored hatch); got "${skeleton.visibility}". The per-element update is a ` +
        `locked numeric formula (not derivable).`,
    );
    return;
  }
  if (skeleton.kernelRef !== KERNEL_REF)
    reportNoSecondOwner(
      `no-second-owner[adam]: skeleton kernelRef "${skeleton.kernelRef}" disagrees ` +
        `with the schedule-owned lowerAdamStepBody (${KERNEL_REF}).`,
    );
  if (skeleton.params !== ADAM_PARAM_SCHEMA)
    reportNoSecondOwner(
      `no-second-owner[adam]: the opaque skeleton must carry the single ` +
        `ADAM_PARAM_SCHEMA — a second typed-param copy is a second owner of the ` +
        `validateConfig-class facts.`,
    );
  // The vec4⇒unscale constraint is owned by the schema AND enforced by the live
  // kernel (vec4 lives on the unscale path). The seam checks agreement.
  if (desc.useVec4 && !desc.emitUnscale)
    reportNoSecondOwner(
      `legality[adam]: useVec4 requires emitUnscale (vec4 lives on the unscale ` +
        `path — atomics prevent auto-vec4 otherwise). The schema's dependent ` +
        `constraint and the live kernel's variant selection are the SAME fact.`,
    );
}

// ============================================================================
// The per-param elementwise Adam BODY (the semantic tensor program)
// ============================================================================

// ============================================================================
// The HORIZONTAL-PACK derivation (§7 P4 deliverable 3 — the `pack` move tenant)
// ============================================================================

/** A per-parameter Adam segment: its value UID + flat element count. */
export interface AdamParamSegment {
  readonly value: ValueUid;
  readonly numElements: number;
}

/**
 * Derive the horizontally-packed fused Adam from N per-param segments. The
 * derivation:
 *   1. builds a per-param loop nest (one loop per segment — the un-packed form:
 *      N separate elementwise Adam applications);
 *   2. applies the `pack` move (moves.ts) at PARAMETER altitude — packing the N
 *      per-param loops into ONE pack loop over the segments (the multi-tensor
 *      packing the §3 pack move expresses);
 *   3. the packed schedule's ONE flat dispatch reaches the authored adamStep
 *      kernel (the packed-dispatch count is 1 per group, not N — the perf target).
 *
 * Returns the packed ScheduleState + the applied `pack` move (for the move-script
 * record) + the total packed element count. Refuses (throws) on an empty pack —
 * a group with no params has nothing to pack.
 */
export function deriveHorizontalPackedAdam(segments: readonly AdamParamSegment[]): {
  packedState: ScheduleState;
  packMove: ScheduleMove;
  totalElements: number;
  perParamLoopCount: number;
} {
  if (segments.length === 0)
    throw new Error(
      "deriveHorizontalPackedAdam: empty group — nothing to pack (a group " +
        "with no grad-bearing params has no work).",
    );

  // The un-packed per-param nest: one sequential loop per segment (each iterates
  // that param's flat elements — the N separate elementwise Adam applications).
  const perParamLoops = segments.map((seg, i) => ({
    uid: uid<SemanticSchedule["loopNest"][number]["uid"]>(
      `loop:adam:param${i}`,
    ),
    entity: uid<SemanticSchedule["loopNest"][number]["entity"]>(
      `ent:loop:adam:param${i}`,
    ),
    axis: uid<SemanticSchedule["loopNest"][number]["axis"]>(
      `axis:adam:param${i}`,
    ),
    kind: "parallel" as const,
    bound: {
      kind: "affineLeaf" as const,
      leaf: { kind: "intLit" as const, value: seg.numElements },
    },
    children: [],
  }));

  const semantic: SemanticSchedule = {
    blockShapes: [],
    loopNest: perParamLoops,
    ordering: { kind: "flat" },
    programGridMap: { kind: "identity" },
    values: segments.map((seg) => ({
      uid: seg.value,
      entity: uid<SemanticSchedule["values"][number]["entity"]>(
        `ent:${seg.value}`,
      ),
      allocation: "global" as const,
      dtype: "f32" as const,
      aliasOf: null,
    })),
    noMaterialization: [],
    stores: [],
    bodies: [],
    roles: [],
    sync: [],
    atoms: [],
    lemmas: [],
  };

  const state: ScheduleState = {
    semantic,
    requests: {
      warpBudget: null,
      pipeline: { kind: "none" },
      placementPreferences: [],
      cachePolicy: [],
    },
    receipts: {},
    region: uid<ScheduleState["region"]>("region:adam:packed"),
  };

  // The `pack` move at PARAMETER altitude: pack the N per-param loops into one
  // pack loop over the segments. `concatenate` is the multi-tensor horizontal
  // pack the foreach optimizer performs (cat the N param/grad/m/v flats into one
  // [total] buffer, one dispatch) — the §3 pack move's real tenant.
  const packMove: ScheduleMove = {
    move: "pack",
    loops: perParamLoops.map((l) => l.uid),
    kind: "concatenate",
  };
  const outcome = applyMove(state, packMove);
  if (outcome.kind !== "applied")
    throw new Error(
      `deriveHorizontalPackedAdam: pack move refused (${outcome.refusal.code}: ${outcome.refusal.reason}).`,
    );

  const totalElements = segments.reduce((a, s) => a + s.numElements, 0);
  return {
    packedState: outcome.state,
    packMove,
    totalElements,
    perParamLoopCount: segments.length,
  };
}


// ============================================================================
// The Adam spec for the PROGRAM-ROLES REALIZER (R5a)
// ============================================================================
//
// R5a (2026-07-22): the fused-Adam kernel body was GENERALIZED off Adam. The
// per-element arithmetic that used to live here (lowerAdamStepBody +
// emitDerivedBiasCorrection + emitDerivedAdamScalarBody + loadAdamUniforms) now
// lives in the generic program-roles realizer (opt-step-realizer.ts): it assembles
// a fused-optimizer kernel from ANY OptimizerProgram + a role schema by folding the
// program's state updates + param update via the OptTerm→tile-IR fold. Adam becomes
// a thin SPEC provider — the executed kernel is bit-for-bit the prior derived body
// (the fold of `oSub(p, SCALED)` is the same tile-IR as the prior `p.sub(SCALED)`;
// same uniform block, same op order). The realizer is the ONE seam Lion/SGD plug
// into for their fused kernels (design ruling O1, §3.4 "one seam, four clients").
//
// bias correction rides in as a `[2]` `bc` DATA input (fork C: a host-computed live
// scalar); `lr` as a `[1]` DATA input (the live-scalar seam). The static hypers stay
// UNIFORMS (ln_beta1/ln_beta2 retained though dead so the setAdamConfigUniforms
// block is byte-unchanged). Surviving guards: the fold-parity differential
// (optterm-fold) + the realizer-parity differential (opt-step-realizer-parity).

/** The fused Adam/AdamW body — DELEGATED to the generic program-roles realizer
 *  over ADAM_STEP_SPEC. Reproduces the prior derived body bit-for-bit. The wrapper
 *  (grid, f16, unscale/atomic-inf, the L2/decoupled wd branches) is the realizer's
 *  program-agnostic dispatch policy. */
function lowerAdamStepBody(
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): TileKernelSpec {
  return lowerOptStepBody(ADAM_STEP_SPEC, useVec4, emitF16, emitUnscale);
}
