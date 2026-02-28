/**
 * Fused Adam/AdamW Optimizer Kernel
 *
 * A single WGSL compute shader that updates param, m, v in-place in one
 * dispatch per parameter. Supports both Adam (L2 decay on gradient) and
 * AdamW (decoupled decay on parameter). grad is read-only; param/m/v are
 * read_write (no separate output buffers needed).
 *
 * Handles large buffers (>maxStorageBufferBindingSize) via chunked dispatch.
 */

import {
  dispatchComputePass,
  getMaxStorageBufferBindingSize,
  trackSharedEncoderWrite,
  createParamsBuffer,
  releaseParamsBuffer,
  isF16Supported,
  allocateOutputBuffer,
  cachedCreateBindGroup,
  getPipeline,
} from "./index";
import { requireContext } from "./webgpu-state";
import type { AdamStepConfig } from "../types";
import { profileSubOpBegin, profileSubOpEnd } from "./profiler";
import type { GPUBuffer, GPUBindGroup, GPUDevice } from "./gpu-types";
import { WORKGROUP_SIZE, MAX_WORKGROUPS_PER_DIM } from "./shape-utils";
import { compileTileKernel } from "./tile-compiler";
import type { TileKernelSpec, BindingSpec, UniformType } from "./tile-ir";

// ============================================================================
// Tile-IR Adam Spec Factory
// ============================================================================

/**
 * Build a TileKernelSpec for the Adam/AdamW optimizer kernel.
 *
 * Only used for WGSL generation — the dispatch logic (chunking, config buffer,
 * bind group construction) in dispatchAdamStep() is unchanged.
 */
function makeAdamStepSpec(
  use2D: boolean,
  gridSizeX: number,
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): TileKernelSpec {
  // Build bindings in the same order as the hand-written shader
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

  // Build uniforms matching the config struct layout
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
    uniforms.inv_scale = "f32";
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
      // Grid is handled by dispatchAdamStep, not used via createTileKernelDispatcher
      return [1];
    },
    kernel(ctx) {
      const numElements = ctx.uniform("num_elements");

      if (useVec4) {
        // Vec4 path: 4 elements per thread
        let flatId;
        if (use2D) {
          flatId = ctx.globalId(0).add(ctx.globalId(1).div(ctx.u32(WORKGROUP_SIZE)).mul(ctx.u32(gridSizeX * WORKGROUP_SIZE)));
          // Actually the 2D formula: gid.x + gid.y * gridSizeX * WORKGROUP_SIZE
          // globalId = (workgroup_id * workgroup_size + local_id)
          // We need: gid.x + gid.y * gridSizeX * WORKGROUP_SIZE where gid = global_invocation_id
          // Actually just: flat_id = gid.x + gid.y * gridSizeX * WORKGROUP_SIZE
          flatId = ctx.globalId(0).add(ctx.globalId(1).mul(ctx.u32(gridSizeX * WORKGROUP_SIZE)));
        } else {
          flatId = ctx.globalId(0);
        }
        const base = ctx.emitLet("base", flatId.mul(ctx.u32(4)));
        ctx.ifThen(base.ge(numElements), () => { ctx.emitReturn(); });

        // Load gradient (4 elements)
        if (emitUnscale) {
          const invScale = ctx.uniform("inv_scale").bitcastTo("f32");
          // Load and unscale
          const g0Var = ctx.emitVar("g0", "f32", ctx.load("grad", base).mul(invScale));
          const g1Var = ctx.emitVar("g1", "f32", ctx.load("grad", base.add(ctx.u32(1))).mul(invScale));
          const g2Var = ctx.emitVar("g2", "f32", ctx.load("grad", base.add(ctx.u32(2))).mul(invScale));
          const g3Var = ctx.emitVar("g3", "f32", ctx.load("grad", base.add(ctx.u32(3))).mul(invScale));

          // Check finite via bit pattern for each element
          for (let e = 0; e < 4; e++) {
            const gVar = [g0Var, g1Var, g2Var, g3Var][e];
            const bits = gVar.get().bitcastTo("u32");
            const exponent = bits.shr(ctx.u32(23)).and(ctx.u32(0xFF));
            ctx.ifThen(exponent.eq(ctx.u32(0xFF)), () => {
              ctx.atomicOp("inf_flag", ctx.u32(0), "max", ctx.u32(1065353216));
              gVar.set(ctx.f32(0.0));
            });
          }

          // Now compute with g0..g3
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
        let idx;
        if (use2D) {
          idx = ctx.globalId(0).add(ctx.globalId(1).mul(ctx.u32(gridSizeX * WORKGROUP_SIZE)));
        } else {
          idx = ctx.globalId(0);
        }
        idx = ctx.emitLet("idx", idx);
        ctx.ifThen(idx.ge(numElements), () => { ctx.emitReturn(); });

        if (emitUnscale) {
          const invScale = ctx.uniform("inv_scale").bitcastTo("f32");
          const gVar = ctx.emitVar("g", "f32", ctx.load("grad", idx).mul(invScale));

          // Check finite
          const bits = gVar.get().bitcastTo("u32");
          const exponent = bits.shr(ctx.u32(23)).and(ctx.u32(0xFF));
          ctx.ifThen(exponent.eq(ctx.u32(0xFF)), () => {
            ctx.atomicOp("inf_flag", ctx.u32(0), "max", ctx.u32(1065353216));
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

import type { VarHandle, BlockExpr, KernelContext } from "./tile-ir";

/** Emit the Adam update logic for the scalar path. */
function emitAdamScalarBody(
  ctx: KernelContext,
  idx: BlockExpr,
  gVar: VarHandle,
  emitF16: boolean,
): void {
  const p = ctx.emitLet("p", ctx.load("param", idx));
  const beta1 = ctx.uniform("beta1").bitcastTo("f32");
  const beta2 = ctx.uniform("beta2").bitcastTo("f32");
  const stepSize = ctx.uniform("step_size").bitcastTo("f32");
  const eps = ctx.uniform("eps").bitcastTo("f32");
  const weightDecay = ctx.uniform("weight_decay").bitcastTo("f32");
  const lrTimesWd = ctx.uniform("lr_times_wd").bitcastTo("f32");
  const decoupledWd = ctx.uniform("decoupled_wd");

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
  const beta1 = ctx.uniform("beta1").bitcastTo("f32");
  const beta2 = ctx.uniform("beta2").bitcastTo("f32");
  const stepSize = ctx.uniform("step_size").bitcastTo("f32");
  const eps = ctx.uniform("eps").bitcastTo("f32");
  const weightDecay = ctx.uniform("weight_decay").bitcastTo("f32");
  const lrTimesWd = ctx.uniform("lr_times_wd").bitcastTo("f32");
  const decoupledWd = ctx.uniform("decoupled_wd");

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

// Cache for compiled tile-IR Adam WGSL
const adamTileIRWGSLCache = new Map<string, string>();

function getAdamTileIRWGSL(
  use2D: boolean,
  gridSizeX: number,
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): string {
  const key = `${use2D}:${gridSizeX}:${useVec4}:${emitF16}:${emitUnscale}`;
  let wgsl = adamTileIRWGSLCache.get(key);
  if (!wgsl) {
    wgsl = compileTileKernel(makeAdamStepSpec(use2D, gridSizeX, useVec4, emitF16, emitUnscale));
    adamTileIRWGSLCache.set(key, wgsl);
  }
  return wgsl;
}

// ============================================================================
// Config Buffer
// ============================================================================

// Pre-allocated typed arrays for config buffer construction.
// Eliminates 228 short-lived allocations per step (76 calls × 3 arrays).
const _configData32 = new ArrayBuffer(32);
const _configF32_32 = new Float32Array(_configData32);
const _configU32_32 = new Uint32Array(_configData32);
const _configData48 = new ArrayBuffer(48);
const _configF32_48 = new Float32Array(_configData48);
const _configU32_48 = new Uint32Array(_configData48);

function createConfigBuffer(
  device: GPUDevice,
  config: AdamStepConfig,
  numElements: number,
  includeInvScale: boolean,
): GPUBuffer {
  if (includeInvScale) {
    // 12 x f32/u32 = 48 bytes (padded to 16-byte alignment)
    _configF32_48[0] = config.beta1;
    _configF32_48[1] = config.beta2;
    _configF32_48[2] = config.stepSize;
    _configF32_48[3] = config.eps;
    _configF32_48[4] = config.weightDecay;
    _configF32_48[5] = config.lrTimesWd;
    _configU32_48[6] = config.decoupledWd ? 1 : 0;
    _configU32_48[7] = numElements;
    _configF32_48[8] = config.invScale ?? 1.0;
    _configU32_48[9] = 0; // pad
    _configU32_48[10] = 0; // pad
    _configU32_48[11] = 0; // pad

    return createParamsBuffer(device, _configU32_48);
  }

  // 8 x f32/u32 = 32 bytes (original layout)
  _configF32_32[0] = config.beta1;
  _configF32_32[1] = config.beta2;
  _configF32_32[2] = config.stepSize;
  _configF32_32[3] = config.eps;
  _configF32_32[4] = config.weightDecay;
  _configF32_32[5] = config.lrTimesWd;
  _configU32_32[6] = config.decoupledWd ? 1 : 0;
  _configU32_32[7] = numElements;

  return createParamsBuffer(device, _configU32_32);
}

// ============================================================================
// Output Buffer Allocation
// ============================================================================

/**
 * Allocate an output buffer for Adam via the buffer pool.
 */
function allocateAdamOutputBuffer(sizeBytes: number): GPUBuffer {
  const buf = allocateOutputBuffer(sizeBytes);
  trackSharedEncoderWrite(buf);
  return buf;
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
 * Handles chunking for buffers larger than maxStorageBufferBindingSize.
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
  const ctx = requireContext();
  const device = ctx.device;

  // Only emit f16 if requested AND the device actually supports shader-f16
  const doF16 = emitF16 && isF16Supported();
  const doUnscale = infFlagBuffer !== null;

  const bytesPerElement = 4; // f32
  const f16BytesPerElement = 2;
  const totalBytes = numElements * bytesPerElement;
  const maxBindingSize = getMaxStorageBufferBindingSize();

  // Determine if chunking is needed
  const needsChunking = totalBytes > maxBindingSize;

  // Align chunk size for sub-range bindings.
  // When emitting f16, chunk alignment must satisfy both f32 (4B) and f16 (2B)
  // offset alignment requirements: offset must be a multiple of minAlignment (256).
  // For f32: 256/4 = 64 elements. For f16: 256/2 = 128 elements. Use the larger.
  const minAlignment = 256; // minStorageBufferOffsetAlignment
  const elementsPerAlignment = doF16
    ? minAlignment / f16BytesPerElement  // 128 — ensures f16 offsets are 256-aligned
    : minAlignment / bytesPerElement;    // 64
  const maxElementsPerChunk = Math.floor(maxBindingSize / bytesPerElement);
  const elementsPerChunk = needsChunking
    ? Math.floor(maxElementsPerChunk / elementsPerAlignment) *
      elementsPerAlignment
    : numElements;
  const numChunks = Math.ceil(numElements / elementsPerChunk);

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
  const totalF16Bytes = numElements * f16BytesPerElement;
  const paramF16Out = doF16
    ? allocateAdamOutputBuffer(totalF16Bytes)
    : null;
  profileSubOpEnd("adam.allocBufs", _st);

  // Vec4 coalescing: use when total elements divisible by 4
  // (elementsPerChunk is aligned to 64 or 128 elements, both multiples of 4)
  const useVec4 = numElements % 4 === 0;

  // Determine dispatch dimensions (vec4 processes 4 elements per thread)
  const workItemsPerChunk = useVec4 ? elementsPerChunk / 4 : elementsPerChunk;
  const maxWorkgroups = Math.ceil(workItemsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;
  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;

  // Select shader variant based on f16, unscale, and vec4 flags
  _st = profileSubOpBegin();
  const vec4Tag = useVec4 ? "Vec4" : "";
  const dimTag = use2D ? `2d:${gridSizeX}` : "1d";
  const key = `adamStep${doF16 ? "F16" : ""}${doUnscale ? "Unscale" : ""}${vec4Tag}:${dimTag}:tile`;
  const code = getAdamTileIRWGSL(use2D, gridSizeX, useVec4, doF16, doUnscale);
  const pipeline = getPipeline(ctx, key, code);
  profileSubOpEnd("adam.pipeline", _st);

  for (let chunk = 0; chunk < numChunks; chunk++) {
    const chunkStart = chunk * elementsPerChunk;
    const chunkEnd = Math.min(chunkStart + elementsPerChunk, numElements);
    const chunkSize = chunkEnd - chunkStart;

    // Create config buffer for this chunk
    _st = profileSubOpBegin();
    const configBuf = createConfigBuffer(device, config, chunkSize, doUnscale);
    profileSubOpEnd("adam.configBuf", _st);

    // Build bind group entries with sub-range bindings for chunked access
    _st = profileSubOpBegin();
    const mkBinding = (buf: GPUBuffer, bpe = bytesPerElement) =>
      needsChunking
        ? { buffer: buf, offset: chunkStart * bpe, size: chunkSize * bpe }
        : { buffer: buf };

    // In-place layout: grad(read), param(rw), m(rw), v(rw), config(uniform)
    // Optional: param_f16(rw), inf_flag(rw)
    const entries: Array<{
      binding: number;
      resource: { buffer: GPUBuffer; offset?: number; size?: number };
    }> = [
      { binding: 0, resource: mkBinding(gradBuffer) },
      { binding: 1, resource: mkBinding(paramBuffer) },
      { binding: 2, resource: mkBinding(mBuffer) },
      { binding: 3, resource: mkBinding(vBuffer) },
      { binding: 4, resource: { buffer: configBuf } },
    ];

    let nextBinding = 5;
    if (doF16 && paramF16Out) {
      entries.push({
        binding: nextBinding++,
        resource: mkBinding(paramF16Out, f16BytesPerElement),
      });
    }

    if (doUnscale) {
      entries.push({
        binding: nextBinding++,
        resource: { buffer: infFlagBuffer! }, // always full 4 bytes, not chunked
      });
    }

    profileSubOpEnd("adam.entries", _st);

    _st = profileSubOpBegin();
    let bindGroup: GPUBindGroup;
    if (needsChunking) {
      // Chunked: entries have offset/size, cannot cache
      bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
    } else {
      // Simple bindings: use cache
      const bgBuffers: GPUBuffer[] = [gradBuffer, paramBuffer, mBuffer, vBuffer, configBuf];
      if (doF16 && paramF16Out) bgBuffers.push(paramF16Out);
      if (doUnscale) bgBuffers.push(infFlagBuffer!);
      bindGroup = cachedCreateBindGroup(device, pipeline, bgBuffers);
    }
    profileSubOpEnd("adam.bindGroup", _st);

    const chunkWorkItems = useVec4 ? chunkSize / 4 : chunkSize;
    const chunkWorkgroups = Math.ceil(chunkWorkItems / WORKGROUP_SIZE);
    const dispatchX = use2D
      ? Math.min(chunkWorkgroups, MAX_WORKGROUPS_PER_DIM)
      : chunkWorkgroups;
    const dispatchY = use2D
      ? Math.ceil(chunkWorkgroups / dispatchX)
      : 1;

    _st = profileSubOpBegin();
    dispatchComputePass(
      pipeline,
      bindGroup,
      dispatchX,
      dispatchY,
    );
    profileSubOpEnd("adam.dispatch", _st);

    // Config buffer must NOT be destroyed while shared encoder is active,
    // because the compute pass hasn't been submitted yet.
    // Use releaseParamsBuffer for safe deferred destruction.
    releaseParamsBuffer(configBuf);
  }

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


