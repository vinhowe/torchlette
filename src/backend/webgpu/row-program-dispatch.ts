/**
 * Row-Program Dispatch
 *
 * Compiles RowProgram specs into tile-IR kernels and dispatches them.
 * Kernel cache is structural (keyed by RowProgram.cacheKey, not node IDs).
 */

import type { RowProgram } from "../../compiler/row-program-types";
import { resolveOutputBuffer } from "./buffer-arena";
import {
  describeWgslMismatch,
  makeCacheKeyGuard,
  wgslContentEqual,
} from "./cache-key-guard";
import type { GPUBuffer } from "./gpu-types";
import { realizeRowProgramSpec as rowProgramToSpec } from "../../schedule/reduction-skeleton";
import { dtypeBytes } from "./shape-utils";
import {
  createTileKernelDispatcher,
  type TileKernelInstance,
} from "./tile-dispatch";
import { onTeardown, requireContext } from "./webgpu-state";

// ============================================================================
// Kernel Cache
// ============================================================================

const kernelCache = new Map<string, TileKernelInstance>();

// Cache-key ⇔ content coherence guard (#92). `program.cacheKey` is a cheap
// structural key; the cached kernel's WGSL comes from rowProgramToSpec(program)
// → compileTileKernel. If the key omits a field the spec/codegen bakes in, two
// different row-programs collide on one entry → a stale kernel served silently.
// On a hit we (strict/sampled) recompile the WGSL and assert it matches.
const rowProgramGuard = makeCacheKeyGuard<string>(
  "row-program kernelCache",
  wgslContentEqual,
  describeWgslMismatch,
);

function getOrCreateKernel(program: RowProgram): TileKernelInstance {
  let kernel = kernelCache.get(program.cacheKey);
  if (!kernel) {
    const spec = rowProgramToSpec(program);
    kernel = createTileKernelDispatcher(spec);
    kernelCache.set(program.cacheKey, kernel);
  } else {
    // Seam guard: on a hit, (strict/sampled) recompile the WGSL from the
    // program and assert it matches the cached kernel's WGSL. A mismatch = the
    // cacheKey under-spans rowProgramToSpec's codegen (#92); throw rather than
    // dispatch a stale kernel. getWGSL + createTileKernelDispatcher are pure
    // string codegen (no device/GPU), so this is GPU-free.
    rowProgramGuard.check(program.cacheKey, kernel.getWGSL(), () =>
      createTileKernelDispatcher(rowProgramToSpec(program)).getWGSL(),
    );
  }
  return kernel;
}

// ============================================================================
// Dispatch
// ============================================================================

/**
 * Dispatch a row-program kernel.
 *
 * @param program      - The RowProgram specification
 * @param inputBuffers - GPU buffers for each program input (order matches program.inputs)
 * @param numRows      - Number of rows (product of dims before reduction dim)
 * @param dimSize      - Size of the reduction dimension
 * @returns Output GPUBuffer
 */
export function dispatchRowProgram(
  program: RowProgram,
  inputBuffers: GPUBuffer[],
  numRows: number,
  dimSize: number,
  /**
   * Element count the CONSUMER will read from the output buffer — i.e.
   * sizeOf(outputNode.shape), the single source of truth for the output layout.
   * The buffer is sized from this, and the kernel's own write count is asserted
   * against it (see below).
   */
  expectedOutElements: number,
): GPUBuffer {
  const ctx = requireContext();
  // Scalar output: 1 element per row; element output: dimSize elements per row.
  const lastPhase = program.phases[program.phases.length - 1];
  const isScalar = lastPhase.kind === "write" && lastPhase.scalarOutput;
  const writeElements = isScalar ? numRows : numRows * dimSize;

  // SEAM INVARIANT: the producer (this kernel) and the consumer (the graph node
  // whose shape wraps this buffer) must agree on the output element count. If a
  // per-row scalar [R,1] is mistakenly emitted full-width [R,D] (or vice versa),
  // the consumer reads the wrong layout and every row silently collapses onto
  // row 0's block — a per-row-correct-looking but wrong result (the compile()
  // layernorm bug). Derive the buffer size from the consumer's shape and crash
  // loudly here rather than returning per-row garbage.
  if (writeElements !== expectedOutElements) {
    throw new Error(
      `row-program output layout mismatch: kernel writes ${writeElements} elements ` +
        `(${isScalar ? "scalar [R,1]" : "full [R,D]"}, numRows=${numRows}, dimSize=${dimSize}) ` +
        `but the consuming node expects ${expectedOutElements}. The write phase's ` +
        `scalarOutput flag disagrees with the output shape — see ` +
        `row-program-detect.ts (exprReadsInput) and the "single source of truth at seams" ` +
        `principle in CLAUDE.md.`,
    );
  }

  const outBytes = expectedOutElements * dtypeBytes(program.output.dtype);
  const outBuffer = resolveOutputBuffer(ctx.device, outBytes, inputBuffers);

  const buffers: Record<string, GPUBuffer> = {};
  for (let i = 0; i < inputBuffers.length; i++) {
    buffers[`in${i}`] = inputBuffers[i];
  }
  buffers["output"] = outBuffer;

  const kernel = getOrCreateKernel(program);
  kernel.dispatch(buffers, { num_rows: numRows, feature_dim: dimSize });

  return outBuffer;
}

// ============================================================================
// Stream-generation plan (stage-4 phase 4.4-coverage)
// ============================================================================

/**
 * The dispatch geometry the stream generator needs to emit a row-program as
 * ALLOC(output) + DISPATCH, single-sourced with `dispatchRowProgram`. All of it
 * is STRUCTURAL — the kernel is keyed by `program.cacheKey` (stable across
 * steps), the uniforms {num_rows, feature_dim} come from the lowered action, and
 * the output byte size comes from the consumer's element count. No live buffer
 * is read; the config buffer is the kernel's cached one (present post-execution,
 * shared with the dispatch path → identical pipeline identity for the differ).
 * `bindingOrder` lists the kernel's bind slots: `in0..inN` (input order matching
 * `inputBuffers`), `output`, and `null` for the uniform config position.
 */
export interface RowProgramDispatchPlan {
  pipeline: GPUComputePipeline;
  bindingOrder: (string | null)[];
  grid: [number, number, number];
  configBuffer: GPUBuffer | null;
  outBytes: number;
}

export function planRowProgramDispatch(
  program: RowProgram,
  numRows: number,
  dimSize: number,
  expectedOutElements: number,
): RowProgramDispatchPlan {
  const outBytes = expectedOutElements * dtypeBytes(program.output.dtype);
  const kernel = getOrCreateKernel(program);
  const p = kernel.plan({ num_rows: numRows, feature_dim: dimSize });
  return {
    pipeline: p.pipeline,
    bindingOrder: p.bindingOrder,
    grid: p.grid,
    configBuffer: p.configBuffer,
    outBytes,
  };
}

// ============================================================================
// Teardown
// ============================================================================

export function resetRowProgramKernelState(): void {
  for (const k of kernelCache.values()) k.reset();
  kernelCache.clear();
}

onTeardown(resetRowProgramKernelState);
