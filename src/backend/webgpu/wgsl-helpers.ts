/**
 * Shared WGSL code-generation helpers.
 *
 * Provides reusable template fragments for common shader patterns:
 * - Tree reductions (sum, max, dual-sum)
 * - Multi-buffer write tracking
 */

import { trackSharedEncoderWrite } from "./index";
import type { GPUBuffer } from "./gpu-types";

// ============================================================================
// WGSL Tree Reduction Templates
// ============================================================================

/**
 * Generate a WGSL tree reduction that sums values in shared memory.
 * Assumes `tid` is the local thread index.
 *
 * @param varName  - shared memory array (e.g. "sdata")
 * @param halfSize - JS expression for initial stride (e.g. `${WORKGROUP_SIZE / 2}`)
 */
export function wgslSumReduction(varName: string, halfSize: string | number): string {
  return `for (var s = ${halfSize}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      ${varName}[tid] += ${varName}[tid + s];
    }
    workgroupBarrier();
  }`;
}

/**
 * Generate a WGSL tree reduction that takes the max in shared memory.
 * Assumes `tid` is the local thread index.
 */
export function wgslMaxReduction(varName: string, halfSize: string | number): string {
  return `for (var s = ${halfSize}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      ${varName}[tid] = max(${varName}[tid], ${varName}[tid + s]);
    }
    workgroupBarrier();
  }`;
}

/**
 * Generate a WGSL tree reduction that sums two shared memory arrays in parallel.
 * Assumes `tid` is the local thread index.
 */
export function wgslDualSumReduction(var1: string, var2: string, halfSize: string | number): string {
  return `for (var s = ${halfSize}u; s > 0u; s >>= 1u) {
    if (tid < s) {
      ${var1}[tid] += ${var1}[tid + s];
      ${var2}[tid] += ${var2}[tid + s];
    }
    workgroupBarrier();
  }`;
}

// ============================================================================
// Multi-Buffer Write Tracking
// ============================================================================

/** Track multiple buffers in the shared encoder write set. */
export function trackBuffers(...buffers: GPUBuffer[]): void {
  for (const buf of buffers) trackSharedEncoderWrite(buf);
}
