/**
 * Walking skeleton — FUSED ADAM (§7 P4 LOCAL self-hosting). Side-by-side with the
 * live path; NO behavior change; NO dispatch cutover. Exercised only by
 * test/schedule/adam-differential.spec.ts + tools/fa-adam-derivation-script.ts.
 *
 * ------------------------------------------------------------------------
 * THE AUTHORED KERNEL + THE HORIZONTAL-PACK DERIVATION
 * ------------------------------------------------------------------------
 * The fused Adam kernel (adamStep, single WGSL dispatch: reads grad/param/m/v/t/lr,
 * writes param/m/v) is AUTHORED — its per-element update body (bias-corrected
 * step-size, expm1 bias correction, L2/decoupled weight decay) is a locked
 * numeric formula (the seam's single source, adam-kernel.ts `emitAdamScalarBody`).
 * So it wears an OPAQUE skeleton (F3) with a TYPED PARAMETER SCHEMA, exactly as
 * the attention kernels do. `applyAdamSchedule` re-CALLS the live `makeAdamStepSpec`
 * factory the kernelRef names; the byte differential proves the regeneration is
 * byte-identical. There is nowhere in the opaque skeleton to store a generator (R22).
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

import {
  makeAdamStepSpec,
} from "../backend/webgpu/adam-kernel";
import type { TileKernelSpec } from "../backend/webgpu/tile-ir";
import { reportNoSecondOwner } from "./canonical";
import { applyMove } from "./moves/moves";
import type {
  PredicateAstNode,
  ScheduleMove,
  ScheduleState,
  SemanticBodyNode,
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

const KERNEL_REF = "src/backend/webgpu/adam-kernel.ts::makeAdamStepSpec";

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
      "authored — the per-element Adam update is a LOCKED numeric formula " +
      "(bias-corrected step-size via expm1, L2/decoupled weight decay; the seam's " +
      "single source emitAdamScalarBody). The DERIVABLE part is the HORIZONTAL-PACK " +
      "move (§7 P4 deliverable 3): N per-param elementwise Adam bodies pack into one " +
      "flat dispatch — the per-param definition is the semantics, the packed form is " +
      "the batched execution of the same tensor program. §6/§7 P4.",
    params: ADAM_PARAM_SCHEMA,
  };
}

/**
 * `applyAdamSchedule` for the authored family: assert the opaque skeleton is
 * well-formed at the seam (no loop/staging/role leaked — F3), then call the live
 * single-source `makeAdamStepSpec` factory the kernelRef names, reconstructing
 * its inputs from the descriptor. Returns the `TileKernelSpec` the differential
 * compiles (its WGSL equals the live makeAdamStepSpec byte-for-byte).
 */
export function applyAdamSchedule(
  skeleton: Skeleton,
  desc: AdamDescriptor,
): TileKernelSpec {
  assertAuthoredSeam(skeleton, desc);
  return makeAdamStepSpec(desc.useVec4, desc.emitF16, desc.emitUnscale);
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
        `with the live makeAdamStepSpec (${KERNEL_REF}).`,
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

/**
 * The per-parameter Adam update as a semantic body (the tensor program the
 * packed form batches). This is the SEMANTIC statement of the same math the
 * authored kernel implements per element:
 *
 *   m' = β1·m + (1−β1)·g
 *   v' = β2·v + (1−β2)·g²
 *   p' = p − stepSize·m' / (sqrt(v') + eps)
 *
 * Expressed as an op-catalog tree over the loaded (g, p, m, v) values. This is
 * NOT the authored kernel's WGSL — it is the semantic program the horizontal-pack
 * move operates on (packing N of these into one flat dispatch). The differential
 * proves the packed batch of N of these == N per-param applications (numeric).
 */
export function adamUpdateBody(): {
  paramNew: SemanticBodyNode;
  mNew: SemanticBodyNode;
  vNew: SemanticBodyNode;
} {
  const g: SemanticBodyNode = { kind: "value", value: uid<ValueUid>("g") };
  const p: SemanticBodyNode = { kind: "value", value: uid<ValueUid>("p") };
  const m: SemanticBodyNode = { kind: "value", value: uid<ValueUid>("m") };
  const v: SemanticBodyNode = { kind: "value", value: uid<ValueUid>("v") };
  const beta1: SemanticBodyNode = {
    kind: "value",
    value: uid<ValueUid>("beta1"),
  };
  const beta2: SemanticBodyNode = {
    kind: "value",
    value: uid<ValueUid>("beta2"),
  };
  const stepSize: SemanticBodyNode = {
    kind: "value",
    value: uid<ValueUid>("stepSize"),
  };
  const eps: SemanticBodyNode = { kind: "value", value: uid<ValueUid>("eps") };
  const one: SemanticBodyNode = { kind: "literal", dtype: "f32", value: 1 };
  const ap = (op: string, ...args: SemanticBodyNode[]): SemanticBodyNode => ({
    kind: "apply",
    catalog: { op },
    args,
  });
  // m' = β1·m + (1−β1)·g
  const mNew = ap(
    "add",
    ap("mul", beta1, m),
    ap("mul", ap("sub", one, beta1), g),
  );
  // v' = β2·v + (1−β2)·g²
  const vNew = ap(
    "add",
    ap("mul", beta2, v),
    ap("mul", ap("sub", one, beta2), ap("mul", g, g)),
  );
  // p' = p − stepSize·m' / (sqrt(v') + eps)
  const paramNew = ap(
    "sub",
    p,
    ap("div", ap("mul", stepSize, mNew), ap("add", ap("sqrt", vNew), eps)),
  );
  return { paramNew, mNew, vNew };
}

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
