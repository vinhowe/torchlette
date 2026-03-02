/**
 * Fused Adam/AdamW Optimizer Kernel
 *
 * A single WGSL compute shader that updates param, m, v in-place in one
 * dispatch per parameter. Supports both Adam (L2 decay on gradient) and
 * AdamW (decoupled decay on parameter). grad is read-only; param/m/v are
 * read_write (no separate output buffers needed).
 *
 * Handles large buffers (>maxStorageBufferBindingSize) via tile-IR dispatchChunked.
 */

import {
  getMaxStorageBufferBindingSize,
  isF16Supported,
  allocateOutputBuffer,
} from "./index";
import { requireContext, trackSharedEncoderWrite } from "./webgpu-state";
import type { AdamStepConfig } from "../types";
import { profileSubOpBegin, profileSubOpEnd } from "./profiler";
import type { GPUBuffer } from "./gpu-types";
import { WORKGROUP_SIZE, MAX_WORKGROUPS_PER_DIM, F32_ONE_BITS } from "./shape-utils";
import { createTileKernelDispatcher, type TileKernelInstance } from "./tile-dispatch";
import { computeFlatChunkLayout } from "./chunked-dispatch";
import type { TileKernelSpec, BindingSpec, UniformType, VarHandle, BlockExpr, KernelContext } from "./tile-ir";

// ============================================================================
// Tile-IR Adam Spec Factory
// ============================================================================

function makeAdamStepSpec(
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): TileKernelSpec {
  const bindings: Record<string, BindingSpec> = {
    grad: { storage: "read", type: "f32" },
    param: { storage: "read_write", type: "f32" },
    m: { storage: "read_write", type: "f32" },
    v: { storage: "read_write", type: "f32" },
  };
  // config uniform goes at binding(4) — set via uniformBindingIndex
  if (emitF16) {
    bindings.param_f16 = { storage: "read_write", type: "f16" };
  }
  if (emitUnscale) {
    bindings.inf_flag = { storage: "atomic", type: "u32" };
  }

  // Build uniforms. grid_stride replaces the compile-time gridSizeX constant.
  const uniforms: Record<string, UniformType> = {
    beta1: "f32",
    beta2: "f32",
    step_size: "f32",
    eps: "f32",
    weight_decay: "f32",
    lr_times_wd: "f32",
    decoupled_wd: "u32",
    num_elements: "u32",
    grid_stride: "u32",
  };
  if (emitUnscale) {
    uniforms.inv_scale = "f32";
    uniforms._pad0 = "u32";
    uniforms._pad1 = "u32";
  } else {
    uniforms._pad0 = "u32";
    uniforms._pad1 = "u32";
    uniforms._pad2 = "u32";
  }

  return {
    name: `adamStep${emitF16 ? "F16" : ""}${emitUnscale ? "Unscale" : ""}${useVec4 ? "Vec4" : ""}`,
    workgroupSize: WORKGROUP_SIZE,
    bindings,
    uniforms,
    uniformBindingIndex: 4,
    enableF16: emitF16,
    grid: (u) => {
      const workItems = useVec4 ? Math.ceil(u.num_elements / 4) : u.num_elements;
      const wg = Math.ceil(workItems / WORKGROUP_SIZE);
      if (wg <= MAX_WORKGROUPS_PER_DIM) return [wg];
      const x = Math.min(wg, MAX_WORKGROUPS_PER_DIM);
      return [x, Math.ceil(wg / x)];
    },
    kernel(ctx) {
      const numElements = ctx.uniform("num_elements");
      const gridStride = ctx.uniform("grid_stride");

      if (useVec4) {
        // Vec4 path: 4 elements per thread.
        // 2D-safe: grid_stride = gridSizeX * WG. For 1D, globalId(1)=0.
        const flatId = ctx.emitLet("flatId", ctx.globalId(0).add(ctx.globalId(1).mul(gridStride)));
        const base = ctx.emitLet("base", flatId.mul(ctx.u32(4)));
        ctx.ifThen(base.ge(numElements), () => { ctx.emitReturn(); });

        if (emitUnscale) {
          const invScale = ctx.uniform("inv_scale").bitcastTo("f32");
          const g0Var = ctx.emitVar("g0", "f32", ctx.load("grad", base).mul(invScale));
          const g1Var = ctx.emitVar("g1", "f32", ctx.load("grad", base.add(ctx.u32(1))).mul(invScale));
          const g2Var = ctx.emitVar("g2", "f32", ctx.load("grad", base.add(ctx.u32(2))).mul(invScale));
          const g3Var = ctx.emitVar("g3", "f32", ctx.load("grad", base.add(ctx.u32(3))).mul(invScale));

          for (let e = 0; e < 4; e++) {
            const gVar = [g0Var, g1Var, g2Var, g3Var][e];
            const bits = gVar.get().bitcastTo("u32");
            const exponent = bits.shr(ctx.u32(23)).and(ctx.u32(0xFF));
            ctx.ifThen(exponent.eq(ctx.u32(0xFF)), () => {
              ctx.atomicOp("inf_flag", ctx.u32(0), "max", ctx.u32(F32_ONE_BITS));
              gVar.set(ctx.f32(0.0));
            });
          }

          emitAdamVec4Body(ctx, base, g0Var, g1Var, g2Var, g3Var, emitF16);
        } else {
          const g0Var = ctx.emitVar("g0", "f32", ctx.load("grad", base));
          const g1Var = ctx.emitVar("g1", "f32", ctx.load("grad", base.add(ctx.u32(1))));
          const g2Var = ctx.emitVar("g2", "f32", ctx.load("grad", base.add(ctx.u32(2))));
          const g3Var = ctx.emitVar("g3", "f32", ctx.load("grad", base.add(ctx.u32(3))));

          emitAdamVec4Body(ctx, base, g0Var, g1Var, g2Var, g3Var, emitF16);
        }
      } else {
        // Scalar path
        const idx = ctx.emitLet("idx", ctx.globalId(0).add(ctx.globalId(1).mul(gridStride)));
        ctx.ifThen(idx.ge(numElements), () => { ctx.emitReturn(); });

        if (emitUnscale) {
          const invScale = ctx.uniform("inv_scale").bitcastTo("f32");
          const gVar = ctx.emitVar("g", "f32", ctx.load("grad", idx).mul(invScale));

          const bits = gVar.get().bitcastTo("u32");
          const exponent = bits.shr(ctx.u32(23)).and(ctx.u32(0xFF));
          ctx.ifThen(exponent.eq(ctx.u32(0xFF)), () => {
            ctx.atomicOp("inf_flag", ctx.u32(0), "max", ctx.u32(F32_ONE_BITS));
            gVar.set(ctx.f32(0.0));
          });

          emitAdamScalarBody(ctx, idx, gVar, emitF16);
        } else {
          const gVar = ctx.emitVar("g", "f32", ctx.load("grad", idx));
          emitAdamScalarBody(ctx, idx, gVar, emitF16);
        }
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
    stepSize: ctx.uniform("step_size").bitcastTo("f32"),
    eps: ctx.uniform("eps").bitcastTo("f32"),
    weightDecay: ctx.uniform("weight_decay").bitcastTo("f32"),
    lrTimesWd: ctx.uniform("lr_times_wd").bitcastTo("f32"),
    decoupledWd: ctx.uniform("decoupled_wd"),
  };
}

/** Emit the Adam update logic for the scalar path. */
function emitAdamScalarBody(
  ctx: KernelContext,
  idx: BlockExpr,
  gVar: VarHandle,
  emitF16: boolean,
): void {
  const p = ctx.emitLet("p", ctx.load("param", idx));
  const { beta1, beta2, stepSize, eps, weightDecay, lrTimesWd, decoupledWd } = loadAdamUniforms(ctx);

  // L2 weight decay (Adam): grad += wd * param
  ctx.ifThen(decoupledWd.eq(ctx.u32(0)).and(weightDecay.bitcastTo("u32").gt(ctx.u32(0))), () => {
    gVar.set(gVar.get().add(weightDecay.mul(p)));
  });

  const g = gVar.get();
  const mNew = ctx.emitLet("m_new",
    beta1.mul(ctx.load("m", idx)).add(ctx.f32(1.0).sub(beta1).mul(g)));
  const vNew = ctx.emitLet("v_new",
    beta2.mul(ctx.load("v", idx)).add(ctx.f32(1.0).sub(beta2).mul(g).mul(g)));

  const pNewVar = ctx.emitVar("p_new", "f32",
    p.sub(stepSize.mul(mNew).div(vNew.sqrt().add(eps))));

  // Decoupled weight decay (AdamW)
  ctx.ifThen(decoupledWd.eq(ctx.u32(1)), () => {
    pNewVar.set(pNewVar.get().sub(lrTimesWd.mul(p)));
  });

  // Write outputs
  ctx.emitStore("param", idx, pNewVar.get());
  ctx.emitStore("m", idx, mNew);
  ctx.emitStore("v", idx, vNew);

  if (emitF16) {
    ctx.emitStore("param_f16", idx, pNewVar.get().toF16());
  }
}

/** Emit the Adam update logic for the vec4 path (4 separate scalars). */
function emitAdamVec4Body(
  ctx: KernelContext,
  base: BlockExpr,
  g0Var: VarHandle, g1Var: VarHandle, g2Var: VarHandle, g3Var: VarHandle,
  emitF16: boolean,
): void {
  const { beta1, beta2, stepSize, eps, weightDecay, lrTimesWd, decoupledWd } = loadAdamUniforms(ctx);

  // Load param, m, v for all 4 elements
  const offsets = [ctx.u32(0), ctx.u32(1), ctx.u32(2), ctx.u32(3)];
  const pVars = offsets.map((o, i) => ctx.emitLet(`p${i}`, ctx.load("param", base.add(o))));
  const mVars = offsets.map((o, i) => ctx.emitLet(`m_old${i}`, ctx.load("m", base.add(o))));
  const vVars = offsets.map((o, i) => ctx.emitLet(`v_old${i}`, ctx.load("v", base.add(o))));
  const gVars = [g0Var, g1Var, g2Var, g3Var];

  // L2 weight decay
  ctx.ifThen(decoupledWd.eq(ctx.u32(0)).and(weightDecay.bitcastTo("u32").gt(ctx.u32(0))), () => {
    for (let i = 0; i < 4; i++) {
      gVars[i].set(gVars[i].get().add(weightDecay.mul(pVars[i])));
    }
  });

  // Moment updates + parameter update for each element
  const pNewVars: VarHandle[] = [];
  const mNewLets: BlockExpr[] = [];
  const vNewLets: BlockExpr[] = [];
  for (let i = 0; i < 4; i++) {
    const g = gVars[i].get();
    const mNew = ctx.emitLet(`m_new${i}`,
      beta1.mul(mVars[i]).add(ctx.f32(1.0).sub(beta1).mul(g)));
    const vNew = ctx.emitLet(`v_new${i}`,
      beta2.mul(vVars[i]).add(ctx.f32(1.0).sub(beta2).mul(g).mul(g)));
    mNewLets.push(mNew);
    vNewLets.push(vNew);
    pNewVars.push(ctx.emitVar(`p_new${i}`, "f32",
      pVars[i].sub(stepSize.mul(mNew).div(vNew.sqrt().add(eps)))));
  }

  // Decoupled weight decay
  ctx.ifThen(decoupledWd.eq(ctx.u32(1)), () => {
    for (let i = 0; i < 4; i++) {
      pNewVars[i].set(pNewVars[i].get().sub(lrTimesWd.mul(pVars[i])));
    }
  });

  // Write outputs
  for (let i = 0; i < 4; i++) {
    const off = base.add(offsets[i]);
    ctx.emitStore("param", off, pNewVars[i].get());
    ctx.emitStore("m", off, mNewLets[i]);
    ctx.emitStore("v", off, vNewLets[i]);
    if (emitF16) {
      ctx.emitStore("param_f16", off, pNewVars[i].get().toF16());
    }
  }
}

// ============================================================================
// Dispatcher Cache (keyed by variant: vec4 × f16 × unscale)
// ============================================================================

const adamDispatchers = new Map<string, TileKernelInstance>();

function getAdamDispatcher(useVec4: boolean, emitF16: boolean, emitUnscale: boolean): TileKernelInstance {
  const key = `${useVec4}:${emitF16}:${emitUnscale}`;
  let d = adamDispatchers.get(key);
  if (!d) {
    d = createTileKernelDispatcher(makeAdamStepSpec(useVec4, emitF16, emitUnscale));
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

  // In-place: param/m/v are read_write, no output buffer allocation needed.
  // Track ALL input buffers (including grad, which is read-only) in the write
  // set BEFORE any allocation, so the pool won't hand back these buffers for
  // the f16 output allocation. Without this, the pool may return grad's buffer
  // for the f16 output, causing a read/read_write conflict on the same buffer.
  let _st = profileSubOpBegin();
  trackSharedEncoderWrite(gradBuffer);
  trackSharedEncoderWrite(paramBuffer);
  trackSharedEncoderWrite(mBuffer);
  trackSharedEncoderWrite(vBuffer);

  // Only allocate f16 output buffer (different size).
  let paramF16Out: GPUBuffer | null = null;
  if (doF16) {
    paramF16Out = allocateOutputBuffer(numElements * f16BytesPerElement);
    trackSharedEncoderWrite(paramF16Out);
  }
  profileSubOpEnd("adam.allocBufs", _st);

  // Vec4 coalescing: use when total elements divisible by 4
  // (elementsPerChunk is aligned to 64 or 128 elements, both multiples of 4)
  const useVec4 = numElements % 4 === 0;

  // Compute grid_stride based on per-chunk element count for 2D-safe indexing
  const epa = doF16 ? 128 : 64; // f16 needs 128-element alignment
  const elemPerChunk = needsChunking
    ? computeFlatChunkLayout(numElements, bytesPerElement, maxBindingSize, 256, epa).elementsPerChunk
    : numElements;
  const workItems = useVec4 ? Math.ceil(elemPerChunk / 4) : elemPerChunk;
  const wg = Math.ceil(workItems / WORKGROUP_SIZE);
  const gridSizeX = Math.min(wg, MAX_WORKGROUPS_PER_DIM);
  const gridStride = gridSizeX * WORKGROUP_SIZE;

  const dispatcher = getAdamDispatcher(useVec4, doF16, doUnscale);

  // Build buffers map
  const buffers: Record<string, GPUBuffer> = {
    grad: gradBuffer, param: paramBuffer, m: mBuffer, v: vBuffer,
  };
  if (doF16 && paramF16Out) buffers.param_f16 = paramF16Out;
  if (doUnscale) buffers.inf_flag = infFlagBuffer!;

  // Build uniforms
  const uniforms: Record<string, number> = {
    beta1: config.beta1,
    beta2: config.beta2,
    step_size: config.stepSize,
    eps: config.eps,
    weight_decay: config.weightDecay,
    lr_times_wd: config.lrTimesWd,
    decoupled_wd: config.decoupledWd ? 1 : 0,
    num_elements: numElements,
    grid_stride: gridStride,
  };
  if (doUnscale) {
    uniforms.inv_scale = config.invScale ?? 1.0;
    uniforms._pad0 = 0;
    uniforms._pad1 = 0;
  } else {
    uniforms._pad0 = 0;
    uniforms._pad1 = 0;
    uniforms._pad2 = 0;
  }

  _st = profileSubOpBegin();
  if (needsChunking) {
    const modes: Record<string, "scalar" | "chunked"> = {
      grad: "chunked", param: "chunked", m: "chunked", v: "chunked",
    };
    if (doF16) modes.param_f16 = "chunked";
    if (doUnscale) modes.inf_flag = "scalar";

    const bpe: Record<string, number> | undefined =
      doF16 ? { param_f16: f16BytesPerElement } : undefined;

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

  // In-place: return the same input buffers (updated in-place by the shader)
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
 * Reset all module-local mutable state (dispatcher cache).
 */
export function resetAdamKernelState(): void {
  for (const d of adamDispatchers.values()) d.reset();
  adamDispatchers.clear();
}
