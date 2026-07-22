/**
 * The PROGRAM-ROLES REALIZER (derived-optimizer-realizer campaign, R5a) — the
 * generic fused-optimizer kernel body, assembled from an `OptimizerProgram` and a
 * ROLE SCHEMA. It is the generalization of the Adam-specific `lowerAdamStepBody`:
 * where that hard-coded Adam's roles ([grad, param, m, v, bc, lr] + the
 * beta/eps/wd uniforms + the m̂/v̂ arithmetic), this reads the roles from the SPEC
 * and folds the per-element arithmetic from the program via the `OptTerm→tile-IR`
 * fold (`lowerOptTermToTileIR`). A new optimizer gets a fused kernel by declaring
 * a spec — zero per-optimizer WGSL (design ruling O1, §3.4 "one seam, four
 * clients").
 *
 * The kernel STRUCTURE the realizer assembles:
 *   bindings  = grad(read) · param(rw) · <stateSlots>(rw) · <scalarInputs>(read)
 *               [· param_f16(rw) · inf_flag(atomic)]
 *   uniforms  = <f32 hypers> · decoupled_wd(u32) · num_elements(u32) · pads
 *   body      = load p ; load hypers ; L2 wd branch (g += wd·p) ;
 *               fold each program.stateUpdate → sNew ; store sNew ;
 *               fold paramUpdateNoWd (state bound OLD or NEW per paramReadsPostState)
 *               → p' ; decoupled wd branch (p' -= lr·wd·p) ; store p' [+ f16].
 *
 * The wrapper (grid, f16, the unscale/atomic-inf vec4 path) is PROGRAM-AGNOSTIC
 * dispatch policy — it stays identical across every optimizer; only the body,
 * bindings, and hyper uniforms are spec-driven. The L2-vs-decoupled weight-decay
 * split is a RUNTIME `decoupled_wd` uniform branch (the same one Adam already
 * carried), so all three of Adam/Lion/SGD select their policy by the config value,
 * not by a name-keyed code path.
 *
 * Adam routes through this realizer with a spec that reproduces its exact prior
 * emission BIT-FOR-BIT (the arithmetic, the op order, the uniform block); the fold
 * of `oSub(p, SCALED)` is the same tile-IR as the prior `p.sub(fold(SCALED))`. The
 * differential (`tools/opt-step-realizer-parity.ts`) proves the assembled kernel
 * matches `evalOptTerm` for Adam/Lion/SGD.
 */

import {
  F32_ONE_BITS,
  MAX_WORKGROUPS_PER_DIM,
  WORKGROUP_SIZE,
} from "../backend/webgpu/shape-utils";
import type {
  BindingSpec,
  BlockExpr,
  KernelContext,
  TileKernelSpec,
  UniformType,
  VarHandle,
} from "../backend/webgpu/tile-ir";
import type { OptimizerProgram, OptTerm } from "../ops/semantic/optimizer";
import { type FoldRoleBindings, lowerOptTermToTileIR } from "./optterm-fold";

/** One scalar-DATA input (a host-computed live scalar): a small `read` f32 buffer
 *  bound once and shared across the group, exposing `roles.length` roles read at
 *  fixed indices. Adam: bc(len 2 → [bc1,bc2]), lr(len 1 → [lr]). */
export interface OptStepScalarInput {
  /** The binding name (a `read` f32 buffer). */
  readonly name: string;
  /** Element count in the buffer (roles are read at index 0..length-1). */
  readonly length: number;
  /** Role name per element (length must equal `length`). */
  readonly roles: readonly string[];
}

/** The declared assembly of ONE fused-optimizer kernel — the program-roles seam. */
export interface OptStepRealizerSpec {
  /** The program whose state updates + param update the body folds. */
  readonly program: OptimizerProgram;
  /** The wd-free param update term (`p'` at wd=0); folded and stored to `param`. */
  readonly paramUpdateNoWd: OptTerm;
  /** true: the param term reads POST-update state (Adam/SGD); false: OLD (Lion). */
  readonly paramReadsPostState: boolean;
  /** Scalar-DATA inputs bound once per group (bc, lr). */
  readonly scalarInputs: readonly OptStepScalarInput[];
  /** Static-hyper f32 uniforms to declare & load; the referenced ones expose
   *  roles (beta1/beta2 also expose om_beta1/om_beta2 = 1−β; weight_decay→wd). */
  readonly f32Uniforms: readonly string[];
  /** Kernel name stem (variant suffixes appended: F16/Unscale/Vec4). */
  readonly kernelName: string;
}

/** Load the declared f32 hyper uniforms + the always-present decoupled_wd u32,
 *  and derive the role bindings the program folds over (om_beta*, wd). */
function loadHyperRoles(
  ctx: KernelContext,
  spec: OptStepRealizerSpec,
): {
  roles: Record<string, BlockExpr>;
  weightDecay: BlockExpr;
  decoupledWd: BlockExpr;
} {
  const roles: Record<string, BlockExpr> = {};
  const loaded: Record<string, BlockExpr> = {};
  for (const name of spec.f32Uniforms) {
    loaded[name] = ctx.uniform(name).bitcastTo("f32");
  }
  // Expose the hyper roles the fold reads. Only the roles the programs name are
  // bound; a program that does not reference one simply never looks it up.
  if (loaded.beta1 !== undefined) {
    roles.beta1 = loaded.beta1;
    roles.om_beta1 = ctx.f32(1.0).sub(loaded.beta1);
  }
  if (loaded.beta2 !== undefined) {
    roles.beta2 = loaded.beta2;
    roles.om_beta2 = ctx.f32(1.0).sub(loaded.beta2);
  }
  if (loaded.eps !== undefined) roles.eps = loaded.eps;
  if (loaded.mu !== undefined) roles.mu = loaded.mu;
  const weightDecay = loaded.weight_decay ?? ctx.f32(0.0);
  roles.wd = weightDecay;
  const decoupledWd = ctx.uniform("decoupled_wd");
  return { roles, weightDecay, decoupledWd };
}

/** Read the scalar-DATA inputs (bc/lr) into their role bindings + the `lr` handle
 *  the decoupled-wd branch needs. */
function loadScalarRoles(
  ctx: KernelContext,
  spec: OptStepRealizerSpec,
): { roles: Record<string, BlockExpr>; lr: BlockExpr } {
  const roles: Record<string, BlockExpr> = {};
  let lr: BlockExpr | undefined;
  for (const inp of spec.scalarInputs) {
    for (let e = 0; e < inp.length; e++) {
      const roleName = inp.roles[e]!;
      // Byte-faithful with the prior Adam body: bc1/bc2 are `emitLet`-bound
      // (reused across the m̂/v̂ folds), lr is read inline.
      const v =
        inp.length > 1
          ? ctx.emitLet(roleName, ctx.load(inp.name, ctx.u32(e)))
          : ctx.load(inp.name, ctx.u32(e));
      roles[roleName] = v;
      if (roleName === "lr") lr = v;
    }
  }
  if (lr === undefined)
    throw new Error(
      `opt-step-realizer: spec '${spec.program.name}' declares no 'lr' scalar role — the decoupled-wd term needs it.`,
    );
  return { roles, lr };
}

/**
 * Emit the per-element optimizer update at `idx`, folding the program's state
 * updates + param update. Mirrors the retired `emitDerivedAdamScalarBody` but
 * driven entirely by the spec (state slots, paramReadsPostState, scalar roles).
 */
function emitOptScalarBody(
  spec: OptStepRealizerSpec,
  ctx: KernelContext,
  idx: BlockExpr,
  gVar: VarHandle,
  emitF16: boolean,
  scalarRoles: Record<string, BlockExpr>,
  lr: BlockExpr,
  suffix = "",
): void {
  const p = ctx.emitLet(`p${suffix}`, ctx.load("param", idx));
  const {
    roles: hyperRoles,
    weightDecay,
    decoupledWd,
  } = loadHyperRoles(ctx, spec);

  // L2 weight decay: grad += wd·param — runtime policy branch (decoupled_wd==0).
  ctx.ifThen(
    decoupledWd.eq(ctx.u32(0)).and(weightDecay.bitcastTo("u32").gt(ctx.u32(0))),
    () => {
      gVar.set(gVar.get().add(weightDecay.mul(p)));
    },
  );

  // Base roles: g, p, the scalar-DATA roles (bc1/bc2/lr), the hyper roles. Old
  // state slots load their current value.
  const roles: Record<string, BlockExpr> = {
    ...hyperRoles,
    ...scalarRoles,
    g: gVar.get(),
    p,
  };
  const oldState: Record<string, BlockExpr> = {};
  for (const slot of spec.program.state) {
    oldState[slot] = ctx.load(slot, idx);
    roles[slot] = oldState[slot]!;
  }

  // Fold each state update over the OLD state (matching the hand foreach order:
  // a later term reads the pre-step value until its own store).
  const newState: Record<string, BlockExpr> = {};
  const fold = (t: OptTerm, r: Record<string, BlockExpr>): BlockExpr =>
    lowerOptTermToTileIR(t, ctx, r as FoldRoleBindings);
  for (const su of spec.program.stateUpdates) {
    newState[su.slot] = ctx.emitLet(
      `${su.slot}_new${suffix}`,
      fold(su.expr, roles),
    );
  }

  // The param term binds state to the POST-update local (Adam/SGD) or keeps the
  // OLD load (Lion). The wd-free term is `p − magnitude`; the fold of that is the
  // same tile-IR as `p.sub(fold(magnitude))`.
  const paramRoles: Record<string, BlockExpr> = { ...roles };
  if (spec.paramReadsPostState) {
    for (const slot of spec.program.state) paramRoles[slot] = newState[slot]!;
  }
  const pNewVar = ctx.emitVar(
    `p_new${suffix}`,
    "f32",
    fold(spec.paramUpdateNoWd, paramRoles),
  );

  // Decoupled weight decay: p' -= lr·wd·p — runtime policy branch (==1).
  ctx.ifThen(decoupledWd.eq(ctx.u32(1)), () => {
    pNewVar.set(pNewVar.get().sub(lr.mul(weightDecay).mul(p)));
  });

  ctx.emitStore("param", idx, pNewVar.get());
  for (const slot of spec.program.state) {
    ctx.emitStore(slot, idx, newState[slot]!);
  }
  if (emitF16) {
    ctx.emitStore("param_f16", idx, pNewVar.get().toF16());
  }
}

/**
 * Build the fused-optimizer `TileKernelSpec` from the program-roles spec. The
 * wrapper (grid, f16, unscale vec4 + atomic-inf) is program-agnostic; the body is
 * folded from `spec.program`. Reproduces the prior Adam kernel bit-for-bit for the
 * Adam spec.
 */
export function lowerOptStepBody(
  spec: OptStepRealizerSpec,
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): TileKernelSpec {
  // Bindings: grad(read) · param(rw) · state slots(rw) · scalar inputs(read).
  const bindings: Record<string, BindingSpec> = {
    grad: { storage: "read", type: "f32" },
    param: { storage: "read_write", type: "f32" },
  };
  for (const slot of spec.program.state) {
    bindings[slot] = { storage: "read_write", type: "f32" };
  }
  for (const inp of spec.scalarInputs) {
    bindings[inp.name] = { storage: "read", type: "f32" };
  }
  if (emitF16) {
    bindings.param_f16 = { storage: "read_write", type: "f16" };
  }
  if (emitUnscale) {
    bindings.inf_flag = { storage: "atomic", type: "u32" };
  }

  // Uniforms: the declared f32 hypers, decoupled_wd, num_elements, + padding.
  const uniforms: Record<string, UniformType> = {};
  for (const name of spec.f32Uniforms) uniforms[name] = "f32";
  uniforms.decoupled_wd = "u32";
  uniforms.num_elements = "u32";
  if (emitUnscale) {
    uniforms.grid_stride = "u32";
    uniforms.inv_scale = "f32";
    uniforms._pad0 = "u32";
    uniforms._pad1 = "u32";
  } else {
    uniforms._pad0 = "u32";
    uniforms._pad1 = "u32";
    uniforms._pad2 = "u32";
    uniforms._pad3 = "u32";
  }

  // Non-unscale path: pure elementwise, auto-vectorized by the compiler.
  if (!emitUnscale) {
    return {
      name: `${spec.kernelName}${emitF16 ? "F16" : ""}`,
      workgroupSize: WORKGROUP_SIZE,
      bindings,
      uniforms,
      enableF16: emitF16,
      autoVectorize: true,
      kernel(ctx) {
        const { roles: scalarRoles, lr } = loadScalarRoles(ctx, spec);
        const idx = ctx.elementIndex(WORKGROUP_SIZE, "num_elements");
        const gVar = ctx.emitVar("g", "f32", ctx.load("grad", idx));
        emitOptScalarBody(spec, ctx, idx, gVar, emitF16, scalarRoles, lr);
      },
    };
  }

  // Unscale path: manual vec4 with per-element inf checks + atomicMax.
  return {
    name: `${spec.kernelName}Unscale${emitF16 ? "F16" : ""}${useVec4 ? "Vec4" : ""}`,
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
      const { roles: scalarRoles, lr } = loadScalarRoles(ctx, spec);

      if (useVec4) {
        const flatId = ctx.emitLet(
          "flatId",
          ctx.globalId(0).add(ctx.globalId(1).mul(gridStride)),
        );
        const base = ctx.emitLet("base", flatId.mul(ctx.u32(4)));
        ctx.ifThen(base.ge(numElements), () => {
          ctx.emitReturn();
        });

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
          emitOptScalarBody(
            spec,
            ctx,
            off,
            gVars[e]!,
            emitF16,
            scalarRoles,
            lr,
            `${e}`,
          );
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

        emitOptScalarBody(spec, ctx, idx, gVar, emitF16, scalarRoles, lr);
      }
    },
  };
}
