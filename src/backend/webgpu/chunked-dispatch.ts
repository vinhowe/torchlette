/**
 * Generic chunking abstractions for ops that exceed maxStorageBufferBindingSize.
 *
 * Two patterns:
 * 1. **Flat-element chunking** (`computeFlatChunkLayout`):
 *    Partitions flat element ranges.  Used by tile-IR `dispatchChunked`.
 * 2. **Dimension-specific chunking** (`computeDimChunkLayout`):
 *    Partitions along a specific tensor dimension with aligned byte offsets.
 *    Used by gather/scatterAdd.
 */

import { gcd, MAX_WORKGROUPS_PER_DIM, WORKGROUP_SIZE } from "./shape-utils";

// ============================================================================
// Types
// ============================================================================

export interface FlatChunkLayout {
  elementsPerChunk: number;
  numChunks: number;
  use2D: boolean;
  gridSizeX: number;
}

// ============================================================================
// Layout computation
// ============================================================================

/**
 * Compute the flat chunk layout for a given total element count and device limits.
 *
 * @param totalElements   Total output elements.
 * @param maxBytesPerElement  Largest bytes-per-element across all chunked bindings
 *                            (determines how many elements fit in one binding).
 * @param maxBindingSize  `device.limits.maxStorageBufferBindingSize`.
 * @param minAlignment    `device.limits.minStorageBufferOffsetAlignment`.
 * @param elementsPerAlignment  Override alignment granularity in elements.
 *                              Defaults to `minAlignment / maxBytesPerElement`.
 *                              Cast uses `lcm(srcEpa, dstEpa)`.
 */
export function computeFlatChunkLayout(
  totalElements: number,
  maxBytesPerElement: number,
  maxBindingSize: number,
  minAlignment: number,
  elementsPerAlignment?: number,
): FlatChunkLayout {
  const epa = elementsPerAlignment ?? minAlignment / maxBytesPerElement;
  const maxElementsPerChunk = Math.floor(maxBindingSize / maxBytesPerElement);
  const elementsPerChunk = Math.floor(maxElementsPerChunk / epa) * epa;

  const numChunks = Math.ceil(totalElements / elementsPerChunk);

  const maxWorkgroups = Math.ceil(elementsPerChunk / WORKGROUP_SIZE);
  const use2D = maxWorkgroups > MAX_WORKGROUPS_PER_DIM;
  const gridSizeX = use2D
    ? Math.min(maxWorkgroups, MAX_WORKGROUPS_PER_DIM)
    : maxWorkgroups;

  return { elementsPerChunk, numChunks, use2D, gridSizeX };
}

// ============================================================================
// Dimension-specific chunking (for gather / scatterAdd)
// ============================================================================

interface DimChunkLayout {
  /** Maximum slices (entries along the chunked dimension) per chunk. */
  maxSlicesPerChunk: number;
  /** Total number of chunks needed to cover the dimension. */
  numChunks: number;
  /** Bytes per slice (elements-after-dim * bytesPerElement). */
  bytesPerSlice: number;
}

/**
 * Compute an aligned chunk layout along a specific tensor dimension.
 *
 * Each "slice" is one index along the chunked dimension.  Chunk boundaries
 * are aligned to `minAlignment` bytes so that WebGPU offset bindings are valid.
 *
 * @param dimSize            Size of the dimension being chunked.
 * @param elementsPerSlice   Product of sizes after the chunked dimension.
 * @param maxBindingSize     `device.limits.maxStorageBufferBindingSize`.
 * @param minAlignment       `device.limits.minStorageBufferOffsetAlignment`.
 * @param bytesPerElement    Bytes per element (default 4 for f32).
 */
export function computeDimChunkLayout(
  dimSize: number,
  elementsPerSlice: number,
  maxBindingSize: number,
  minAlignment: number,
  bytesPerElement: number = 4,
): DimChunkLayout {
  const bytesPerSlice = elementsPerSlice * bytesPerElement;
  let maxSlicesPerChunk = Math.floor(maxBindingSize / bytesPerSlice);

  // If bytesPerSlice isn't a multiple of minAlignment, adjust so that
  // (slicesPerChunk * bytesPerSlice) is always aligned.
  if (bytesPerSlice % minAlignment !== 0) {
    const slicesForAlignment = minAlignment / gcd(bytesPerSlice, minAlignment);
    maxSlicesPerChunk =
      Math.floor(maxSlicesPerChunk / slicesForAlignment) * slicesForAlignment;
    if (maxSlicesPerChunk === 0) {
      maxSlicesPerChunk = slicesForAlignment; // At least one aligned group
    }
  }

  return {
    maxSlicesPerChunk,
    numChunks: Math.ceil(dimSize / maxSlicesPerChunk),
    bytesPerSlice,
  };
}
