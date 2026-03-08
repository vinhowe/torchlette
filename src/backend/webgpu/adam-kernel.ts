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
import { trackSharedEncoderWrite } from "./webgpu-state";

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

  // Build uniforms
  const uniforms: Record<string, UniformType> = {
    beta1: "f32",
    beta2: "f32",
    step_size: "f32",
    eps: "f32",
    weight_decay: "f32",
    lr_times_wd: "f32",
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
      uniformBindingIndex: 4,
      enableF16: emitF16,
      autoVectorize: true,
      kernel(ctx) {
        const idx = ctx.elementIndex(WORKGROUP_SIZE, "num_elements");
        const gVar = ctx.emitVar("g", "f32", ctx.load("grad", idx));
        emitAdamScalarBody(ctx, idx, gVar, emitF16);
      },
    };
  }

  // Unscale path: manual vec4 with per-element inf checks + atomicMax
  return {
    name: `adamStepUnscale${emitF16 ? "F16" : ""}${useVec4 ? "Vec4" : ""}`,
    workgroupSize: WORKGROUP_SIZE,
    bindings,
    uniforms,
    uniformBindingIndex: 4,
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
          emitAdamScalarBody(ctx, off, gVars[e], emitF16, `${e}`);
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

        emitAdamScalarBody(ctx, idx, gVar, emitF16);
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

/** Emit the Adam update logic for a single element at `idx`. */
function emitAdamScalarBody(
  ctx: KernelContext,
  idx: BlockExpr,
  gVar: VarHandle,
  emitF16: boolean,
  suffix = "",
): void {
  const p = ctx.emitLet(`p${suffix}`, ctx.load("param", idx));
  const { beta1, beta2, stepSize, eps, weightDecay, lrTimesWd, decoupledWd } =
    loadAdamUniforms(ctx);

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
    p.sub(stepSize.mul(mNew).div(vNew.sqrt().add(eps))),
  );

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

  // Vec4 coalescing for unscale path (non-unscale uses autoVectorize)
  const useVec4 = doUnscale && numElements % 4 === 0;

  // Build buffers map
  const buffers: Record<string, GPUBuffer> = {
    grad: gradBuffer,
    param: paramBuffer,
    m: mBuffer,
    v: vBuffer,
  };
  if (doF16 && paramF16Out) buffers.param_f16 = paramF16Out;
  if (doUnscale && infFlagBuffer) buffers.inf_flag = infFlagBuffer;

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
  };
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
    uniforms.inv_scale = config.invScale ?? 1.0;
    uniforms._pad0 = 0;
    uniforms._pad1 = 0;
  } else {
    uniforms._pad0 = 0;
    uniforms._pad1 = 0;
    uniforms._pad2 = 0;
    uniforms._pad3 = 0;
  }

  const dispatcher = getAdamDispatcher(useVec4, doF16, doUnscale);

  _st = profileSubOpBegin();
  if (needsChunking) {
    const epa = doF16 ? 128 : 64;
    const modes: Record<string, "scalar" | "chunked"> = {
      grad: "chunked",
      param: "chunked",
      m: "chunked",
      v: "chunked",
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
