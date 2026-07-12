/**
 * Fused Adam/AdamW Optimizer Kernel
 *
 * A single WGSL compute shader that updates param, m, and v in-place in one
 * dispatch per parameter. Supports both Adam (L2 decay on gradient) and
 * AdamW (decoupled decay on parameter). grad is read-only; param/m/v are
 * read_write (each thread touches only its own index → no cross-thread race).
 *
 * Handles large buffers (>maxStorageBufferBindingSize) via tile-IR dispatchChunked.
 */

import type { AdamStepConfig } from "../types";
import { allocateOutputBuffer } from "./buffer-arena";
import { computeFlatChunkLayout } from "./chunked-dispatch";
import { getMaxStorageBufferBindingSize, isF16Supported } from "./gpu-context";
import type { GPUBuffer } from "./gpu-types";
import { profileSubOpBegin, profileSubOpEnd } from "./profiler";
import {
  F32_ONE_BITS,
  MAX_WORKGROUPS_PER_DIM,
  WORKGROUP_SIZE,
} from "./shape-utils";
import {
  createTileKernelDispatcher,
  type TileKernelInstance,
} from "./tile-dispatch";
import type {
  BindingSpec,
  BlockExpr,
  KernelContext,
  TileKernelSpec,
  UniformType,
  VarHandle,
} from "./tile-ir";
import { onTeardown, trackSharedEncoderWrite } from "./webgpu-state";

// ============================================================================
// Tile-IR Adam Spec Factory
// ============================================================================

export function makeAdamStepSpec(
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): TileKernelSpec {
  const bindings: Record<string, BindingSpec> = {
    grad: { storage: "read", type: "f32" },
    param: { storage: "read_write", type: "f32" },
    m: { storage: "read_write", type: "f32" },
    v: { storage: "read_write", type: "f32" },
    // inc-2a: t (step counter) and lr flow as persistent 1-element f32 tensor
    // DATA, not per-step-varying uniforms. The recorder sees stable buffers
    // (TAG_WRITE), which kills the frozen-scalar / volatile-repack class.
    t: { storage: "read", type: "f32" },
    lr: { storage: "read", type: "f32" },
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
        const bc = emitBiasCorrection(ctx);
        const idx = ctx.elementIndex(WORKGROUP_SIZE, "num_elements");
        const gVar = ctx.emitVar("g", "f32", ctx.load("grad", idx));
        emitAdamScalarBody(ctx, idx, gVar, emitF16, bc);
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
      const bc = emitBiasCorrection(ctx);

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
          emitAdamScalarBody(ctx, off, gVars[e], emitF16, bc, `${e}`);
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

        emitAdamScalarBody(ctx, idx, gVar, emitF16, bc);
      }
    },
  };
}

// ============================================================================
// Adam Update Logic (shared between scalar and vec4 paths)
// ============================================================================

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

/**
 * LOCKED formula (single source at the seam; mirrored by
 * `test/optim/adam-biascorrection-formula.spec.ts`'s `expm1F32`):
 *   expm1(y):  |y| < 0.25 → 5-term Horner series
 *                y*(1 + y*(1/2 + y*(1/6 + y*(1/24 + y/120))))
 *              else       → exp(y) - 1
 * y = t * lnBeta (≤ 0), so the naive `1-pow(beta,t)` cancellation is avoided.
 * Returns expm1(y) as a BlockExpr. `select` computes both branches — fine for
 * a scalar derivation evaluated once per thread.
 */
function emitExpm1(ctx: KernelContext, y: BlockExpr): BlockExpr {
  // 5-term Horner series (small-|y| branch).
  let r: BlockExpr = ctx.f32(1 / 120);
  r = ctx.f32(1 / 24).add(y.mul(r));
  r = ctx.f32(1 / 6).add(y.mul(r));
  r = ctx.f32(1 / 2).add(y.mul(r));
  r = ctx.f32(1.0).add(y.mul(r));
  const series = y.mul(r);
  const large = y.exp().sub(ctx.f32(1.0));
  return y.abs().lt(ctx.f32(0.25)).select(series, large);
}

/**
 * Derive the bias-corrected step size and epsilon from the on-device step
 * counter `t` and learning rate `lr` (both read from 1-element storage
 * bindings). Replaces the retired JS-computed `stepSize`/`lrTimesWd` config
 * fields. bc = -expm1(t*lnBeta); the PyTorch-equivalent update is
 *   p -= (lr*sqrt(bc2)/bc1) * m / (sqrt(v) + eps*sqrt(bc2))
 * i.e. p -= lr * (m/bc1) / (sqrt(v/bc2) + eps_orig).
 */
function emitBiasCorrection(ctx: KernelContext): {
  stepSize: BlockExpr;
  epsAdjusted: BlockExpr;
  lr: BlockExpr;
} {
  const t = ctx.load("t", ctx.u32(0));
  const lr = ctx.load("lr", ctx.u32(0));
  const { lnBeta1, lnBeta2, eps } = loadAdamUniforms(ctx);
  const bc1 = ctx.emitLet("bc1", emitExpm1(ctx, t.mul(lnBeta1)).neg());
  const bc2 = ctx.emitLet("bc2", emitExpm1(ctx, t.mul(lnBeta2)).neg());
  const sqrtBc2 = ctx.emitLet("sqrt_bc2", bc2.sqrt());
  const stepSize = ctx.emitLet("step_size", lr.mul(sqrtBc2).div(bc1));
  const epsAdjusted = ctx.emitLet("eps_adj", eps.mul(sqrtBc2));
  return { stepSize, epsAdjusted, lr };
}

/** Emit the Adam update logic for a single element at `idx`. */
function emitAdamScalarBody(
  ctx: KernelContext,
  idx: BlockExpr,
  gVar: VarHandle,
  emitF16: boolean,
  bc: { stepSize: BlockExpr; epsAdjusted: BlockExpr; lr: BlockExpr },
  suffix = "",
): void {
  const p = ctx.emitLet(`p${suffix}`, ctx.load("param", idx));
  const { beta1, beta2, weightDecay, decoupledWd } = loadAdamUniforms(ctx);

  // L2 weight decay (Adam): grad += wd * param
  ctx.ifThen(
    decoupledWd.eq(ctx.u32(0)).and(weightDecay.bitcastTo("u32").gt(ctx.u32(0))),
    () => {
      gVar.set(gVar.get().add(weightDecay.mul(p)));
    },
  );

  const g = gVar.get();
  const mNew = ctx.emitLet(
    `m_new${suffix}`,
    beta1.mul(ctx.load("m", idx)).add(ctx.f32(1.0).sub(beta1).mul(g)),
  );
  const vNew = ctx.emitLet(
    `v_new${suffix}`,
    beta2.mul(ctx.load("v", idx)).add(ctx.f32(1.0).sub(beta2).mul(g).mul(g)),
  );

  const pNewVar = ctx.emitVar(
    `p_new${suffix}`,
    "f32",
    p.sub(bc.stepSize.mul(mNew).div(vNew.sqrt().add(bc.epsAdjusted))),
  );

  // Decoupled weight decay (AdamW): p -= lr*wd*p
  ctx.ifThen(decoupledWd.eq(ctx.u32(1)), () => {
    pNewVar.set(pNewVar.get().sub(bc.lr.mul(weightDecay).mul(p)));
  });

  // Write outputs
  ctx.emitStore("param", idx, pNewVar.get());
  ctx.emitStore("m", idx, mNew);
  ctx.emitStore("v", idx, vNew);

  if (emitF16) {
    ctx.emitStore("param_f16", idx, pNewVar.get().toF16());
  }
}

// ============================================================================
// Dispatcher Cache (keyed by variant: vec4 × f16 × unscale)
// ============================================================================

const adamDispatchers = new Map<string, TileKernelInstance>();

function getAdamDispatcher(
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): TileKernelInstance {
  const key = `${useVec4}:${emitF16}:${emitUnscale}`;
  let d = adamDispatchers.get(key);
  if (!d) {
    d = createTileKernelDispatcher(
      makeAdamStepSpec(useVec4, emitF16, emitUnscale),
    );
    adamDispatchers.set(key, d);
  }
  return d;
}

// ============================================================================
// Dispatch
// ============================================================================

interface AdamStepResult {
  paramBuffer: GPUBuffer;
  mBuffer: GPUBuffer;
  vBuffer: GPUBuffer;
  paramF16Buffer?: GPUBuffer;
}

/**
 * Write the STATIC Adam config uniforms. inc-2a: every field here is now
 * step-invariant — beta1/beta2, eps (ORIGINAL, un-adjusted), weight_decay,
 * decoupled_wd, and the precomputed ln(beta) constants (f64→f32 on the CPU;
 * the in-kernel expm1-form bias correction derives bc1/bc2/step_size/
 * epsAdjusted from the on-device `t` and `lr` tensor inputs). The retired
 * per-step fields (step_size, lr_times_wd) and their volatile-repack are gone:
 * the config buffer is bound once and never rewritten across replays.
 * `invScale` remains a per-step value for the fused-unscale path (GradScaler
 * uses graph-level unscaleGrad, so nothing engages that path here).
 */
function setAdamConfigUniforms(
  uniforms: Record<string, number>,
  config: AdamStepConfig,
  doUnscale: boolean,
): void {
  uniforms.beta1 = config.beta1;
  uniforms.beta2 = config.beta2;
  // Precompute ln(beta) in f64 then narrow to f32 (Math.fround) so the static
  // uniform matches the Gate-2 emulation's `f(Math.log(beta))` exactly.
  uniforms.ln_beta1 = Math.fround(Math.log(config.beta1));
  uniforms.ln_beta2 = Math.fround(Math.log(config.beta2));
  uniforms.eps = config.eps;
  uniforms.weight_decay = config.weightDecay;
  uniforms.decoupled_wd = config.decoupledWd ? 1 : 0;
  if (doUnscale) {
    uniforms.inv_scale = config.invScale ?? 1.0;
  }
}

/**
 * Dispatch the fused Adam/AdamW kernel.
 *
 * Uses tile-IR dispatchChunked for buffers exceeding maxStorageBufferBindingSize.
 * When infFlagBuffer is provided, uses fused unscale+inf-check variants
 * that multiply grad by invScale and detect non-finite values.
 */
export function dispatchAdamStep(
  gradBuffer: GPUBuffer,
  paramBuffer: GPUBuffer,
  mBuffer: GPUBuffer,
  vBuffer: GPUBuffer,
  tBuffer: GPUBuffer,
  lrBuffer: GPUBuffer,
  numElements: number,
  config: AdamStepConfig,
  emitF16 = false,
  infFlagBuffer: GPUBuffer | null = null,
): AdamStepResult {
  // Only emit f16 if requested AND the device actually supports shader-f16
  const doF16 = emitF16 && isF16Supported();
  const doUnscale = infFlagBuffer !== null;

  const bytesPerElement = 4; // f32
  const f16BytesPerElement = 2;
  const totalBytes = numElements * bytesPerElement;
  const maxBindingSize = getMaxStorageBufferBindingSize();
  const needsChunking = totalBytes > maxBindingSize;

  // Track all in-place buffers in the write set so the pool won't hand
  // them back for output allocation within this shared-encoder scope.
  let _st = profileSubOpBegin();
  trackSharedEncoderWrite(gradBuffer);
  trackSharedEncoderWrite(paramBuffer);
  trackSharedEncoderWrite(mBuffer);
  trackSharedEncoderWrite(vBuffer);
  // t/lr are read-only 1-element inputs — track so the pool won't hand them
  // back for output allocation within this shared-encoder scope.
  trackSharedEncoderWrite(tBuffer);
  trackSharedEncoderWrite(lrBuffer);

  // Allocate f16 output buffer if needed.
  let paramF16Out: GPUBuffer | null = null;
  if (doF16) {
    paramF16Out = allocateOutputBuffer(numElements * f16BytesPerElement);
    trackSharedEncoderWrite(paramF16Out);
  }
  profileSubOpEnd("adam.allocBufs", _st);

  // Vec4 coalescing for unscale path (non-unscale uses autoVectorize)
  const useVec4 = doUnscale && numElements % 4 === 0;

  // Build buffers map — all in-place (param/m/v modified in the same buffer).
  const buffers: Record<string, GPUBuffer> = {
    grad: gradBuffer,
    param: paramBuffer,
    m: mBuffer,
    v: vBuffer,
    t: tBuffer,
    lr: lrBuffer,
  };
  if (doF16 && paramF16Out) buffers.param_f16 = paramF16Out;
  if (doUnscale && infFlagBuffer) buffers.inf_flag = infFlagBuffer;

  // Build uniforms
  const uniforms: Record<string, number> = {
    num_elements: numElements,
  };
  setAdamConfigUniforms(uniforms, config, doUnscale);
  if (doUnscale) {
    // Unscale path needs grid_stride for manual 2D indexing
    const epa = doF16 ? 128 : 64;
    const elemPerChunk = needsChunking
      ? computeFlatChunkLayout(
          numElements,
          bytesPerElement,
          maxBindingSize,
          256,
          epa,
        ).elementsPerChunk
      : numElements;
    const workItems = useVec4 ? Math.ceil(elemPerChunk / 4) : elemPerChunk;
    const wg = Math.ceil(workItems / WORKGROUP_SIZE);
    const gridSizeX = Math.min(wg, MAX_WORKGROUPS_PER_DIM);
    uniforms.grid_stride = gridSizeX * WORKGROUP_SIZE;
    uniforms._pad0 = 0;
    uniforms._pad1 = 0;
  } else {
    uniforms._pad0 = 0;
    uniforms._pad1 = 0;
    uniforms._pad2 = 0;
    uniforms._pad3 = 0;
  }

  const dispatcher = getAdamDispatcher(useVec4, doF16, doUnscale);

  // inc-2a: the config is fully STATIC now (bias correction is derived
  // in-kernel from the t/lr tensor DATA inputs), so there is NO volatile
  // repack — the config buffer is bound once and never rewritten across
  // compiled-plan replays. The retired volatile-repack closure lived here.

  _st = profileSubOpBegin();
  if (needsChunking) {
    const epa = doF16 ? 128 : 64;
    const modes: Record<string, "scalar" | "chunked"> = {
      grad: "chunked",
      param: "chunked",
      m: "chunked",
      v: "chunked",
      // t/lr are 1-element inputs read at index 0 by every chunk — bind whole.
      t: "scalar",
      lr: "scalar",
    };
    if (doF16) modes.param_f16 = "chunked";
    if (doUnscale) modes.inf_flag = "scalar";

    const bpe: Record<string, number> | undefined = doF16
      ? { param_f16: f16BytesPerElement }
      : undefined;

    dispatcher.dispatchChunked(buffers, uniforms, {
      modes,
      bytesPerElement: bpe,
      sizeUniform: "num_elements",
      totalElements: numElements,
      maxBytesPerElement: bytesPerElement,
      elementsPerAlignment: epa,
    });
  } else {
    dispatcher.dispatch(buffers, uniforms);
  }
  profileSubOpEnd("adam.dispatch", _st);

  // param/m/v are all updated in-place — return the same input buffers.
  const result: AdamStepResult = {
    paramBuffer,
    mBuffer,
    vBuffer,
  };
  if (paramF16Out) {
    result.paramF16Buffer = paramF16Out;
  }
  return result;
}

/**
 * Stage-4 plan/encode: the adam-step dispatch plan (non-f16 path used by the
 * packed optimizer), derived from the SAME dispatcher instance + uniform
 * mapping dispatchAdamStep uses. Returns null on the f16 / chunked routes.
 * The binding order is [grad, param, m, v, t, lr, (inf_flag?), config].
 *
 * inc-2a: config is fully STATIC (bias correction derived in-kernel from the
 * t/lr tensor inputs), so there is NO volatilePack — the generated stream
 * binds the config buffer once (no TAG_UNIFORM repack).
 */
export function planAdamStepDispatch(
  numElements: number,
  config: AdamStepConfig,
  infFlagBuffer: GPUBuffer | null,
  emitF16 = false,
): {
  plan: import("./tile-dispatch").TileKernelPlan;
  doUnscale: boolean;
  /** f16 weight emission active → the dispatch has a `param_f16` OUTPUT
   *  binding the caller must allocate (numElements*2 bytes, allocKind 1). */
  doF16: boolean;
  f16Bytes: number;
} | null {
  if (numElements * 4 > getMaxStorageBufferBindingSize()) return null; // chunked
  const doF16 = emitF16 && isF16Supported();
  const doUnscale = infFlagBuffer !== null;
  const useVec4 = doUnscale && numElements % 4 === 0;
  const uniforms: Record<string, number> = { num_elements: numElements };
  setAdamConfigUniforms(uniforms, config, doUnscale);
  if (doUnscale) {
    const epa = doF16 ? 128 : 64;
    const workItems = useVec4 ? Math.ceil(numElements / 4) : numElements;
    const wg = Math.ceil(workItems / WORKGROUP_SIZE);
    const gridSizeX = Math.min(wg, MAX_WORKGROUPS_PER_DIM);
    uniforms.grid_stride = gridSizeX * WORKGROUP_SIZE;
    uniforms._pad0 = 0;
    uniforms._pad1 = 0;
    void epa;
  } else {
    uniforms._pad0 = 0;
    uniforms._pad1 = 0;
    uniforms._pad2 = 0;
    uniforms._pad3 = 0;
  }
  const dispatcher = getAdamDispatcher(useVec4, doF16, doUnscale);
  return {
    plan: dispatcher.plan(uniforms),
    doUnscale,
    doF16,
    f16Bytes: numElements * 2,
  };
}

/**
 * Reset all module-local mutable state (dispatcher cache).
 */
export function resetAdamKernelState(): void {
  for (const d of adamDispatchers.values()) d.reset();
  adamDispatchers.clear();
}
onTeardown(resetAdamKernelState);
