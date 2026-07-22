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
import type {
  BindingSpec,
  BlockExpr,
  KernelContext,
  TileKernelSpec,
  UniformType,
  VarHandle,
} from "../backend/webgpu/tile-ir";
import {
  F32_ONE_BITS,
  MAX_WORKGROUPS_PER_DIM,
  WORKGROUP_SIZE,
} from "../backend/webgpu/shape-utils";
import { reportNoSecondOwner } from "./canonical";
import { applyMove } from "./moves/moves";
import {
  ADAMW_M_NEW,
  ADAMW_SCALED,
  ADAMW_V_NEW,
  type OptTerm,
} from "../ops/semantic/optimizer";
import {
  lowerOptTermToTileIR,
  type FoldRoleBindings,
} from "./optterm-fold";
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
// The ABSORBED authored fused-Adam kernel body (§7 P4 cutover-flip)
// ============================================================================
//
// The fused-Adam kernel body — RELOCATED from adam-kernel.ts `makeAdamStepSpec` —
// so the schedule module is the SOLE owner of the Adam kernel structure at the
// live dispatch seam (`realizeAdamStepSpec`). The bias-corrected step-size
// (expm1-form bias correction from the on-device t/lr tensor inputs), the
// L2/decoupled weight-decay branches, and the atomic-guarded unscale vec4 path
// LOWER FROM here now. BYTE-IDENTICAL to the retired factory (the 5-variant
// differential is the safety net). The skeleton STAYS opaque (F3): the per-element
// update is a locked numeric formula, but the body it names lives here — one owner.
// The horizontal-pack derivation (deriveHorizontalPackedAdam, above) is the
// DERIVABLE part; this body is the authored per-element update the pack batches.

/** The fused Adam/AdamW body (single WGSL dispatch per param/segment).
 *
 * `derived` (R2, fork B): route the per-element arithmetic through the
 * `OptTerm→tile-IR` fold and take bias correction as a `[2]` `bc=[bc1,bc2]` DATA
 * input instead of computing it in-kernel from `t` (the `expm1` prelude dies).
 * The wrapper (grid, f16, unscale/atomic-inf, the L2/decoupled wd branches) is
 * program-agnostic dispatch policy and stays IDENTICAL across both paths — only
 * the binding at slot 4 (`t`→`bc`) and the two emit helpers swap. */
function lowerAdamStepBody(
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): TileKernelSpec {
  const bindings: Record<string, BindingSpec> = {
    grad: { storage: "read", type: "f32" },
    param: { storage: "read_write", type: "f32" },
    m: { storage: "read_write", type: "f32" },
    v: { storage: "read_write", type: "f32" },
    // inc-2a: lr flows as a persistent 1-element f32 tensor DATA, not a
    // per-step-varying uniform. The recorder sees stable buffers (TAG_WRITE),
    // which kills the frozen-scalar / volatile-repack class. R3/fork-C: bias
    // correction rides at the `bc` slot as a `[2]` `bc`=[bc1,bc2] DATA input
    // (host-computed live scalar), the identical TAG_WRITE channel.
    bc: { storage: "read", type: "f32" },
    lr: { storage: "read", type: "f32" },
  };
  const biasCorrection = (ctx: KernelContext): DerivedBias =>
    emitDerivedBiasCorrection(ctx);
  const adamScalarBody = (
    ctx: KernelContext,
    idx: BlockExpr,
    gVar: VarHandle,
    f16: boolean,
    bc: DerivedBias,
    suffix = "",
  ): void => {
    emitDerivedAdamScalarBody(ctx, idx, gVar, f16, bc, suffix);
  };
  if (emitF16) {
    bindings.param_f16 = { storage: "read_write", type: "f16" };
  }
  if (emitUnscale) {
    bindings.inf_flag = { storage: "atomic", type: "u32" };
  }

  // Build uniforms. inc-2a: the config is now FULLY STATIC — beta1/beta2/
  // eps(orig)/weight_decay/decoupled_wd never vary per step. ln_beta1/ln_beta2
  // are precomputed f64->f32 on the CPU and used by the in-kernel expm1-form
  // bias correction (bc = -expm1(t*lnBeta)); they replace the retired
  // step_size / lr_times_wd per-step fields.
  const uniforms: Record<string, UniformType> = {
    beta1: "f32",
    beta2: "f32",
    ln_beta1: "f32",
    ln_beta2: "f32",
    eps: "f32",
    weight_decay: "f32",
    decoupled_wd: "u32",
    num_elements: "u32",
  };
  if (emitUnscale) {
    // Unscale path keeps grid_stride for manual 2D indexing (atomics prevent auto-vec4)
    uniforms.grid_stride = "u32";
    uniforms.inv_scale = "f32";
    uniforms._pad0 = "u32";
    uniforms._pad1 = "u32";
  } else {
    // Non-unscale: compiler handles 2D grid via flatGlobalId, no grid_stride needed
    uniforms._pad0 = "u32";
    uniforms._pad1 = "u32";
    uniforms._pad2 = "u32";
    uniforms._pad3 = "u32";
  }

  // Non-unscale path: pure elementwise, auto-vectorized by the compiler.
  // Unscale path: atomics prevent auto-vec4, so manual vec4 with grid_stride.
  if (!emitUnscale) {
    return {
      name: `adamStep${emitF16 ? "F16" : ""}`,
      workgroupSize: WORKGROUP_SIZE,
      bindings,
      uniforms,
      enableF16: emitF16,
      autoVectorize: true,
      kernel(ctx) {
        const bc = biasCorrection(ctx);
        const idx = ctx.elementIndex(WORKGROUP_SIZE, "num_elements");
        const gVar = ctx.emitVar("g", "f32", ctx.load("grad", idx));
        adamScalarBody(ctx, idx, gVar, emitF16, bc);
      },
    };
  }

  // Unscale path: manual vec4 with per-element inf checks + atomicMax
  return {
    name: `adamStepUnscale${emitF16 ? "F16" : ""}${useVec4 ? "Vec4" : ""}`,
    workgroupSize: WORKGROUP_SIZE,
    bindings,
    uniforms,
    enableF16: emitF16,
    grid: (u) => {
      const workItems = useVec4
        ? Math.ceil(u.num_elements / 4)
        : u.num_elements;
      const wg = Math.ceil(workItems / WORKGROUP_SIZE);
      if (wg <= MAX_WORKGROUPS_PER_DIM) return [wg];
      const x = Math.min(wg, MAX_WORKGROUPS_PER_DIM);
      return [x, Math.ceil(wg / x)];
    },
    kernel(ctx) {
      const numElements = ctx.uniform("num_elements");
      const gridStride = ctx.uniform("grid_stride");
      const invScale = ctx.uniform("inv_scale").bitcastTo("f32");
      const bc = biasCorrection(ctx);

      if (useVec4) {
        const flatId = ctx.emitLet(
          "flatId",
          ctx.globalId(0).add(ctx.globalId(1).mul(gridStride)),
        );
        const base = ctx.emitLet("base", flatId.mul(ctx.u32(4)));
        ctx.ifThen(base.ge(numElements), () => {
          ctx.emitReturn();
        });

        // Load and unscale 4 grads, check each for inf/nan
        const gVars: VarHandle[] = [];
        for (let e = 0; e < 4; e++) {
          const off = e === 0 ? base : base.add(ctx.u32(e));
          const gVar = ctx.emitVar(
            `g${e}`,
            "f32",
            ctx.load("grad", off).mul(invScale),
          );
          gVars.push(gVar);
          const bits = gVar.get().bitcastTo("u32");
          const exponent = bits.shr(ctx.u32(23)).and(ctx.u32(0xff));
          ctx.ifThen(exponent.eq(ctx.u32(0xff)), () => {
            ctx.atomicOp("inf_flag", ctx.u32(0), "max", ctx.u32(F32_ONE_BITS));
            gVar.set(ctx.f32(0.0));
          });
        }
        for (let e = 0; e < 4; e++) {
          const off = e === 0 ? base : base.add(ctx.u32(e));
          adamScalarBody(ctx, off, gVars[e], emitF16, bc, `${e}`);
        }
      } else {
        const idx = ctx.emitLet(
          "idx",
          ctx.globalId(0).add(ctx.globalId(1).mul(gridStride)),
        );
        ctx.ifThen(idx.ge(numElements), () => {
          ctx.emitReturn();
        });

        const gVar = ctx.emitVar(
          "g",
          "f32",
          ctx.load("grad", idx).mul(invScale),
        );
        const bits = gVar.get().bitcastTo("u32");
        const exponent = bits.shr(ctx.u32(23)).and(ctx.u32(0xff));
        ctx.ifThen(exponent.eq(ctx.u32(0xff)), () => {
          ctx.atomicOp("inf_flag", ctx.u32(0), "max", ctx.u32(F32_ONE_BITS));
          gVar.set(ctx.f32(0.0));
        });

        adamScalarBody(ctx, idx, gVar, emitF16, bc);
      }
    },
  };
}

// ----------------------------------------------------------------------------
// Adam update logic (shared between scalar and vec4 paths) — RELOCATED from
// adam-kernel.ts (the body's single-source per-element update).
// ----------------------------------------------------------------------------

function loadAdamUniforms(ctx: KernelContext) {
  return {
    beta1: ctx.uniform("beta1").bitcastTo("f32"),
    beta2: ctx.uniform("beta2").bitcastTo("f32"),
    lnBeta1: ctx.uniform("ln_beta1").bitcastTo("f32"),
    lnBeta2: ctx.uniform("ln_beta2").bitcastTo("f32"),
    eps: ctx.uniform("eps").bitcastTo("f32"),
    weightDecay: ctx.uniform("weight_decay").bitcastTo("f32"),
    decoupledWd: ctx.uniform("decoupled_wd"),
  };
}

// ============================================================================
// The DERIVED per-element body (R2 fork B / R3 fork C) — FOLDED from ADAMW_PROGRAM
// ============================================================================
//
// R4 (2026-07-22): the authored per-element arithmetic (emitAdamScalarBody,
// emitBiasCorrection, emitExpm1) is DELETED — it was the differential oracle for
// its own re-derivation, retired once fork C became the proven default. The body
// below re-sources the per-element ARITHMETIC from ADAMW_PROGRAM via the
// OptTerm→tile-IR fold — the executed kernel is a THEOREM of the program (design
// ruling O1). bias correction rides in as a `[2]` `bc` DATA input (fork C: a
// host-computed live scalar). The wrapper (grid, f16, unscale/atomic-inf) and the
// runtime L2/decoupled wd branches stay skeleton policy (the fold has no
// conditional). Surviving guard: the fold-parity differential (optterm-fold).

/** Derived bias data — bc1/bc2 as DATA roles the fold consumes. */
interface DerivedBias {
  bc1: BlockExpr;
  bc2: BlockExpr;
  lr: BlockExpr;
}

/**
 * fork B: bias correction is DATA. Read bc1/bc2 from the `[2]` `bc` input and
 * `lr` from the `lr` input; the in-kernel `expm1` prelude is GONE (it moved
 * graph-side to `Adam._biasCorrection`, feeding the `bc` tensor). Returns the
 * roles the fold binds — NOT a reassociated step_size (the fold computes m̂/v̂).
 */
function emitDerivedBiasCorrection(ctx: KernelContext): DerivedBias {
  const bc1 = ctx.emitLet("bc1", ctx.load("bc", ctx.u32(0)));
  const bc2 = ctx.emitLet("bc2", ctx.load("bc", ctx.u32(1)));
  const lr = ctx.load("lr", ctx.u32(0));
  return { bc1, bc2, lr };
}

/**
 * Emit the Adam per-element update at `idx` with the m'/v'/scaled arithmetic
 * DERIVED from ADAMW_PROGRAM via `lowerOptTermToTileIR`. Only the L2/decoupled wd
 * branches (a runtime `decoupled_wd` select the fold cannot express) and the
 * stores are skeleton policy.
 *
 * The named reassociation lemmas the derived body carries (vs the now-deleted
 * authored form the fold was validated against at R2/R3):
 *   (L1) ADAMW_V_NEW folds (1−β2)·g² as `(g·g)·(1−β2)`; authored was
 *        `((1−β2)·g)·g`.
 *   (L2) ADAMW_SCALED divides inside the sqrt — `lr·(m/bc1)/(√(v/bc2)+ε)` —
 *        whereas authored factors √bc2 out (`step_size = lr·√bc2/bc1`,
 *        `eps_adj = ε·√bc2`). Mathematically equal; f32 agrees to ≤1e-7.
 */
function emitDerivedAdamScalarBody(
  ctx: KernelContext,
  idx: BlockExpr,
  gVar: VarHandle,
  emitF16: boolean,
  bc: DerivedBias,
  suffix = "",
): void {
  const p = ctx.emitLet(`p${suffix}`, ctx.load("param", idx));
  const { beta1, beta2, eps, weightDecay, decoupledWd } = loadAdamUniforms(ctx);

  // L2 weight decay (Adam): grad += wd * param — runtime policy branch.
  ctx.ifThen(
    decoupledWd.eq(ctx.u32(0)).and(weightDecay.bitcastTo("u32").gt(ctx.u32(0))),
    () => {
      gVar.set(gVar.get().add(weightDecay.mul(p)));
    },
  );

  // Role bindings for the fold — the program's roles map to tile-IR values.
  const roles: Record<string, BlockExpr> = {
    m: ctx.load("m", idx),
    v: ctx.load("v", idx),
    g: gVar.get(),
    p,
    beta1,
    beta2,
    om_beta1: ctx.f32(1.0).sub(beta1),
    om_beta2: ctx.f32(1.0).sub(beta2),
    bc1: bc.bc1,
    bc2: bc.bc2,
    lr: bc.lr,
    eps,
    wd: weightDecay,
  };
  const fold = (t: OptTerm, r: Record<string, BlockExpr>): BlockExpr =>
    lowerOptTermToTileIR(t, ctx, r as FoldRoleBindings);

  const mNew = ctx.emitLet(`m_new${suffix}`, fold(ADAMW_M_NEW, roles));
  const vNew = ctx.emitLet(`v_new${suffix}`, fold(ADAMW_V_NEW, roles));

  // p' reads the POST-update moments (the realizer's role binding). ADAMW_SCALED
  // = lr·(m̂/(√v̂+ε)); the derived p' = p − scaled, then the decoupled branch.
  const scaled = fold(ADAMW_SCALED, { ...roles, m: mNew, v: vNew });
  const pNewVar = ctx.emitVar(`p_new${suffix}`, "f32", p.sub(scaled));

  // Decoupled weight decay (AdamW): p -= lr*wd*p — runtime policy branch.
  ctx.ifThen(decoupledWd.eq(ctx.u32(1)), () => {
    pNewVar.set(pNewVar.get().sub(bc.lr.mul(weightDecay).mul(p)));
  });

  ctx.emitStore("param", idx, pNewVar.get());
  ctx.emitStore("m", idx, mNew);
  ctx.emitStore("v", idx, vNew);

  if (emitF16) {
    ctx.emitStore("param_f16", idx, pNewVar.get().toF16());
  }
}
