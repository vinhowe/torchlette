/**
 * Row-Program Dispatch
 *
 * Compiles RowProgram specs into tile-IR kernels and dispatches them.
 * Kernel cache is structural (keyed by RowProgram.cacheKey, not node IDs).
 */

import type { RowProgram } from "../../compiler/row-program-types";
import { resolveOutputBuffer } from "./buffer-arena";
import type { GPUBuffer } from "./gpu-types";
import { rowProgramToSpec } from "./row-program-codegen";
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

function getOrCreateKernel(program: RowProgram): TileKernelInstance {
  let kernel = kernelCache.get(program.cacheKey);
  if (!kernel) {
    const spec = rowProgramToSpec(program);
    kernel = createTileKernelDispatcher(spec);
    kernelCache.set(program.cacheKey, kernel);
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
// Teardown
// ============================================================================

export function resetRowProgramKernelState(): void {
  for (const k of kernelCache.values()) k.reset();
  kernelCache.clear();
}

onTeardown(resetRowProgramKernelState);
