/**
 * Fused optimizer-step kernel dispatch (derived-optimizer-realizer R5b — the
 * de-naming). A single WGSL compute shader updates param + optimizer state
 * in-place in one dispatch per parameter. grad is read-only; param and every
 * state slot are read_write (each thread touches only its own index → no
 * cross-thread race). The kernel BODY is folded from the optimizer's program
 * spec (`realizeOptStepSpec` — the schedule chokepoint); this file carries NO
 * optimizer name, keying every decision on `OptStepConfig` structure
 * (`spec`/`stateSlots`/`scalarInputs`/`hypers`). Adam is one client
 * (`spec:"adamw"`, states `[m,v]`, scalars `[bc,lr]`), reproduced bit-for-bit.
 *
 * Handles large buffers (>maxStorageBufferBindingSize) via tile-IR dispatchChunked.
 */

import { realizeOptStepSpec } from "../../schedule/opt-step-specs";
import type { OptStepConfig } from "../types";
import { allocateOutputBuffer } from "./buffer-arena";
import { computeFlatChunkLayout } from "./chunked-dispatch";
import { getMaxStorageBufferBindingSize, isF16Supported } from "./gpu-context";
import type { GPUBuffer } from "./gpu-types";
import { profileSubOpBegin, profileSubOpEnd } from "./profiler";
import { MAX_WORKGROUPS_PER_DIM, WORKGROUP_SIZE } from "./shape-utils";
import {
  createTileKernelDispatcher,
  type TileKernelInstance,
} from "./tile-dispatch";
import { onTeardown, trackSharedEncoderWrite } from "./webgpu-state";

// ============================================================================
// Dispatcher Cache (keyed by spec + variant: vec4 × f16 × unscale)
// ============================================================================

const optStepDispatchers = new Map<string, TileKernelInstance>();

function getOptStepDispatcher(
  spec: string,
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): TileKernelInstance {
  const key = `${spec}:${useVec4}:${emitF16}:${emitUnscale}`;
  let d = optStepDispatchers.get(key);
  if (!d) {
    // Route through the schedule chokepoint: the body FOLDS from the optimizer
    // program spec (realizeOptStepSpec → lowerOptStepBody); the schedule is the
    // sole WGSL writer at the dispatch seam.
    d = createTileKernelDispatcher(
      realizeOptStepSpec(spec, useVec4, emitF16, emitUnscale),
    );
    optStepDispatchers.set(key, d);
  }
  return d;
}

// ============================================================================
// Dispatch
// ============================================================================

interface OptStepResult {
  paramBuffer: GPUBuffer;
  /** Updated state buffers, in spec order (Adam: [m, v]). In-place → same input buffers. */
  stateBuffers: GPUBuffer[];
  paramF16Buffer?: GPUBuffer;
}

/**
 * Write the STATIC optimizer config uniforms. Every field is step-invariant: the
 * spec's f32 hyper VALUES (`config.hypers`, e.g. Adam beta1/beta2/ln_beta1/
 * ln_beta2/eps/weight_decay), the L2-vs-decoupled `decoupled_wd` branch, and (for
 * the fused-unscale path) `inv_scale`. The bias-corrected step size + lr*wd are
 * derived IN-KERNEL from the `bc`/`lr` scalar-DATA inputs, so the config buffer is
 * bound once and never rewritten across replays (no volatile repack).
 */
function setOptStepConfigUniforms(
  uniforms: Record<string, number>,
  config: OptStepConfig,
  doUnscale: boolean,
): void {
  for (const [name, value] of Object.entries(config.hypers)) {
    uniforms[name] = value;
  }
  uniforms.decoupled_wd = config.decoupledWd ? 1 : 0;
  if (doUnscale) {
    uniforms.inv_scale = config.invScale ?? 1.0;
  }
}

/**
 * Dispatch the fused optimizer-step kernel.
 *
 * Uses tile-IR dispatchChunked for buffers exceeding maxStorageBufferBindingSize.
 * When infFlagBuffer is provided, uses fused unscale+inf-check variants that
 * multiply grad by invScale and detect non-finite values. `stateBuffers` and
 * `scalarBuffers` are bound by the spec's `stateSlots` / `scalarInputs` names.
 */
export function dispatchOptStep(
  gradBuffer: GPUBuffer,
  paramBuffer: GPUBuffer,
  stateBuffers: GPUBuffer[],
  scalarBuffers: GPUBuffer[],
  numElements: number,
  config: OptStepConfig,
  emitF16 = false,
  infFlagBuffer: GPUBuffer | null = null,
): OptStepResult {
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
  for (const b of stateBuffers) trackSharedEncoderWrite(b);
  // Scalar-DATA inputs are read-only small shared bindings — track so the pool
  // won't hand them back for output allocation within this shared-encoder scope.
  for (const b of scalarBuffers) trackSharedEncoderWrite(b);

  // Allocate f16 output buffer if needed.
  let paramF16Out: GPUBuffer | null = null;
  if (doF16) {
    paramF16Out = allocateOutputBuffer(numElements * f16BytesPerElement);
    trackSharedEncoderWrite(paramF16Out);
  }
  profileSubOpEnd("optStep.allocBufs", _st);

  // Vec4 coalescing for unscale path (non-unscale uses autoVectorize)
  const useVec4 = doUnscale && numElements % 4 === 0;

  // Build buffers map — grad(read) · param(rw) · state slots(rw) · scalar inputs(read).
  const buffers: Record<string, GPUBuffer> = {
    grad: gradBuffer,
    param: paramBuffer,
  };
  config.stateSlots.forEach((slot, i) => {
    buffers[slot] = stateBuffers[i];
  });
  config.scalarInputs.forEach((name, i) => {
    buffers[name] = scalarBuffers[i];
  });
  if (doF16 && paramF16Out) buffers.param_f16 = paramF16Out;
  if (doUnscale && infFlagBuffer) buffers.inf_flag = infFlagBuffer;

  // Build uniforms
  const uniforms: Record<string, number> = {
    num_elements: numElements,
  };
  setOptStepConfigUniforms(uniforms, config, doUnscale);
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

  const dispatcher = getOptStepDispatcher(config.spec, useVec4, doF16, doUnscale);

  _st = profileSubOpBegin();
  if (needsChunking) {
    const epa = doF16 ? 128 : 64;
    const modes: Record<string, "scalar" | "chunked"> = {
      grad: "chunked",
      param: "chunked",
    };
    for (const slot of config.stateSlots) modes[slot] = "chunked";
    // Scalar-DATA inputs are small shared buffers read at fixed indices by every
    // chunk — bind whole ("scalar" mode = not chunked).
    for (const name of config.scalarInputs) modes[name] = "scalar";
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
  profileSubOpEnd("optStep.dispatch", _st);

  // param + states are all updated in-place — return the same input buffers.
  const result: OptStepResult = {
    paramBuffer,
    stateBuffers,
  };
  if (paramF16Out) {
    result.paramF16Buffer = paramF16Out;
  }
  return result;
}

/**
 * Stage-4 plan/encode: the opt-step dispatch plan (non-f16 path used by the
 * packed optimizer), derived from the SAME dispatcher instance + uniform
 * mapping dispatchOptStep uses. Returns null on the f16 / chunked routes.
 * The binding order is [grad, param, ...states, ...scalars, (inf_flag?), config].
 *
 * The config is fully STATIC (bias correction derived in-kernel from the bc/lr
 * scalar-DATA inputs), so there is NO volatilePack — the generated stream binds
 * the config buffer once (no TAG_UNIFORM repack).
 */
export function planOptStepDispatch(
  numElements: number,
  config: OptStepConfig,
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
  setOptStepConfigUniforms(uniforms, config, doUnscale);
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
  const dispatcher = getOptStepDispatcher(config.spec, useVec4, doF16, doUnscale);
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
export function resetOptStepKernelState(): void {
  for (const d of optStepDispatchers.values()) d.reset();
  optStepDispatchers.clear();
}
onTeardown(resetOptStepKernelState);
