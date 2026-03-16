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
): GPUBuffer {
  const ctx = requireContext();
  // Scalar output: 1 element per row; element output: dimSize elements per row
  const lastPhase = program.phases[program.phases.length - 1];
  const isScalar = lastPhase.kind === "write" && lastPhase.scalarOutput;
  const outElements = isScalar ? numRows : numRows * dimSize;
  const outBytes = outElements * dtypeBytes(program.output.dtype);
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
