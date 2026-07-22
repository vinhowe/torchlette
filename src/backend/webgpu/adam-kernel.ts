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

import { realizeAdamStepSpec } from "../../schedule/adam-skeleton";
import { ENV } from "../../core/env";
import type { AdamStepConfig } from "../types";

/** R2/fork-B: whether the fused Adam body is DERIVED from ADAMW_PROGRAM (binds a
 *  `[2]` `bc` DATA input at the `t` slot). Gated OFF by default. */
const derivedAdam = (): boolean => ENV.TORCHLETTE_DERIVED_ADAM !== "0"; // R3 FLIP: fork C default
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
// Adam kernel body — ABSORBED into the schedule module (§7 P4 cutover-flip)
// ============================================================================
//
// The fused-Adam kernel body (makeAdamStepSpec + its per-element update helpers)
// was RELOCATED into src/schedule/adam-skeleton.ts, where the ScheduleState now
// OWNS the kernel structure. The dispatcher below routes through
// `realizeAdamStepSpec` (the schedule chokepoint) — the schedule object is the
// sole WGSL writer at the dispatch seam. The retired `makeAdamStepSpec` factory
// (and its emitExpm1 / emitBiasCorrection / emitAdamScalarBody helpers) are gone;
// the byte differential (test/schedule/adam-differential.spec.ts) guards the LIVE
// path.

// ============================================================================
// Dispatcher Cache (keyed by variant: vec4 × f16 × unscale)
// ============================================================================

const adamDispatchers = new Map<string, TileKernelInstance>();

function getAdamDispatcher(
  useVec4: boolean,
  emitF16: boolean,
  emitUnscale: boolean,
): TileKernelInstance {
  // Include the derived bit so an authored and a derived dispatcher never share
  // a cache slot (they have different bindings — `t` vs `bc`).
  const key = `${derivedAdam() ? "d" : "a"}:${useVec4}:${emitF16}:${emitUnscale}`;
  let d = adamDispatchers.get(key);
  if (!d) {
    // Route through the schedule chokepoint: the body LOWERS FROM the schedule
    // object (the cutover-flip); realizeAdamStepSpec is the sole WGSL writer.
    d = createTileKernelDispatcher(
      realizeAdamStepSpec(useVec4, emitF16, emitUnscale),
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
  // R2/fork-B: the DERIVED kernel binds the same slot-4 buffer under the name
  // `bc` (a [2] bias-correction DATA tensor) instead of `t`. `tBuffer` carries
  // whichever the caller (adam.ts _stepFused) produced for this flag state.
  const biasName = derivedAdam() ? "bc" : "t";
  const buffers: Record<string, GPUBuffer> = {
    grad: gradBuffer,
    param: paramBuffer,
    m: mBuffer,
    v: vBuffer,
    [biasName]: tBuffer,
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
      // t/lr (or the derived [2] bc/lr) are small shared inputs read at fixed
      // indices by every chunk — bind whole ("scalar" mode = not chunked).
      [biasName]: "scalar",
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
