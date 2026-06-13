/**
 * Packed Optimizer Dispatch
 *
 * Groups same-element-count optimizer parameters into contiguous packed buffers
 * and dispatches one kernel per size class instead of one per parameter.
 *
 * Flow per group:
 * 1. Get cached packed buffers (allocated on first use, reused across steps)
 * 2. Scatter individual buffers → packed via copyBufferToBuffer (DMA)
 * 3. Call the caller-provided dispatch function on the packed buffers
 * 4. Gather modified buffers → individual via copyBufferToBuffer
 *
 * This is optimizer-generic — the caller provides buffer tuples and a dispatch
 * callback. The utility handles grouping, sub-batching, buffer caching, and
 * scatter/gather encoding.
 */

import type { GPUBuffer } from "../backend/webgpu/gpu-types";
import { STORAGE_BUFFER_USAGE } from "../backend/webgpu/gpu-types";
import { alignBufferSize } from "../backend/webgpu/shape-utils";
import { getSharedEncoderInstance } from "../backend/webgpu/shared-encoder";
import { onTeardown, requireContext } from "../backend/webgpu/webgpu-state";
import { profileSubOpBegin, profileSubOpEnd } from "../graph/profiler";
import { recordedCopyBufferToBuffer } from "../executor/compiled-plan";

/**
 * Maximum packed buffer size before sub-batching (512 MB).
 * Groups larger than this are split to limit temporary memory usage.
 */
const MAX_PACKED_BYTES = 512 * 1024 * 1024;

/** One parameter's set of buffers for packed dispatch. */
export interface PackedOptimizerItem {
  /** All buffers for this param: [grad, param, state0, state1, ...] */
  buffers: GPUBuffer[];
  numElements: number;
}

/** Options for a packed optimizer dispatch. */
export interface PackedOptimizerOpts {
  items: PackedOptimizerItem[];
  /** Which buffer indices to gather back (the read-write ones). E.g. [1, 2, 3] for Adam. */
  gatherIndices: number[];
  /** Called with packed buffers (same order as item.buffers) and total element count. */
  dispatch: (packed: GPUBuffer[], totalElements: number) => void;
  /** Profiler label prefix (default: "packedOptim"). */
  label?: string;
}

// ---------------------------------------------------------------------------
// Packed buffer cache
// ---------------------------------------------------------------------------

/**
 * Persistent cache for packed buffers, keyed by `${bufferCount}:${alignedBytes}`.
 * Each entry holds N buffers reused across steps.
 * Allocated on first use, freed on resetPackedOptimizerCache().
 */
const packedBufferCache = new Map<string, GPUBuffer[]>();

function getPackedBuffers(count: number, alignedBytes: number): GPUBuffer[] {
  const key = `${count}:${alignedBytes}`;
  let bufs = packedBufferCache.get(key);
  if (!bufs) {
    const device = requireContext().device;
    bufs = [];
    for (let i = 0; i < count; i++) {
      bufs.push(
        device.createBuffer({
          size: alignedBytes,
          usage: STORAGE_BUFFER_USAGE,
        }),
      );
    }
    packedBufferCache.set(key, bufs);
  }
  return bufs;
}

/** Stage-4 stream generation: read-only lookup of the cached packed buffers
 *  for a (bufferCount, alignedBytes) key. Returns null when not yet
 *  allocated — the generator runs post-recording, so a hit is guaranteed
 *  for any group the recording dispatched. Never allocates (generation must
 *  not mutate GPU state). */
export function lookupPackedBuffers(
  count: number,
  alignedBytes: number,
): GPUBuffer[] | null {
  return packedBufferCache.get(`${count}:${alignedBytes}`) ?? null;
}

/** Stage-4: the element-count grouping + memory-limit sub-batching the
 *  packed dispatch uses, as pure data (no GPU). The generator replays the
 *  SAME partition so its scatter/dispatch/gather command order matches the
 *  recording exactly. Mirrors dispatchPackedOptimizer's loop. */
export function planPackedGroups(
  items: PackedOptimizerItem[],
): Array<{ numElements: number; indices: number[] }> {
  const out: Array<{ numElements: number; indices: number[] }> = [];
  if (items.length <= 1) return out;
  const groups = new Map<number, number[]>();
  for (let i = 0; i < items.length; i++) {
    const numEl = items[i].numElements;
    const g = groups.get(numEl);
    if (g) g.push(i);
    else groups.set(numEl, [i]);
  }
  for (const [numElements, indices] of groups) {
    if (indices.length <= 1) continue;
    const bufferCount = items[indices[0]].buffers.length;
    const groupBytes = numElements * 4 * indices.length * bufferCount;
    const maxBatchSize =
      groupBytes > MAX_PACKED_BYTES
        ? Math.max(
            2,
            Math.floor(MAX_PACKED_BYTES / (numElements * 4 * bufferCount)),
          )
        : indices.length;
    for (let start = 0; start < indices.length; start += maxBatchSize) {
      const end = Math.min(start + maxBatchSize, indices.length);
      const batchIndices = indices.slice(start, end);
      if (batchIndices.length <= 1) continue;
      out.push({ numElements, indices: batchIndices });
    }
  }
  return out;
}

/** Release all cached packed buffers. */
function resetPackedOptimizerCache(): void {
  for (const bufs of packedBufferCache.values()) {
    for (const buf of bufs) buf.destroy();
  }
  packedBufferCache.clear();
}
onTeardown(resetPackedOptimizerCache);

// ---------------------------------------------------------------------------
// Core dispatch
// ---------------------------------------------------------------------------

/** Dispatch a single group of same-element-count items. */
function dispatchPackedGroup(
  items: PackedOptimizerItem[],
  numElements: number,
  gatherIndices: number[],
  dispatchFn: (packed: GPUBuffer[], totalElements: number) => void,
  label: string,
): void {
  const bufferCount = items[0].buffers.length;
  const totalElements = numElements * items.length;
  const elementBytes = numElements * 4; // f32
  const alignedBytes = alignBufferSize(totalElements * 4);

  // Get or create cached packed buffers
  const _st = profileSubOpBegin();
  const packed = getPackedBuffers(bufferCount, alignedBytes);

  // All scatter → compute → gather commands are recorded on the SAME shared encoder.
  // WebGPU guarantees commands execute in recorded order within a command buffer.
  const enc = getSharedEncoderInstance();
  if (!enc)
    throw new Error("Packed optimizer dispatch requires shared encoder");

  // Scatter: copy all individual buffers into packed buffers. These are
  // INTRA-PLAN copies — the packed-Adam dispatch reads them, so they MUST be
  // replayed by the compiled plan. If unrecorded, compiled replay runs Adam on
  // stale packed grads/params → optimizer-state drift (silent convergence gap).
  for (let i = 0; i < items.length; i++) {
    const bufs = items[i].buffers;
    const offset = i * elementBytes;
    for (let b = 0; b < bufferCount; b++) {
      recordedCopyBufferToBuffer(enc, bufs[b], 0, packed[b], offset, elementBytes);
    }
  }
  profileSubOpEnd(`${label}.scatter`, _st);

  // Dispatch the kernel on packed buffers
  const _st2 = profileSubOpBegin();
  dispatchFn(packed, totalElements);
  profileSubOpEnd(`${label}.dispatch`, _st2);

  // Gather: copy modified packed buffers back to individual buffers. Also
  // INTRA-PLAN — these write the updated params/m/v that the NEXT step reads,
  // so they must be replayed too (else params/optimizer state stay stale).
  const _st3 = profileSubOpBegin();
  for (let i = 0; i < items.length; i++) {
    const bufs = items[i].buffers;
    const offset = i * elementBytes;
    for (const b of gatherIndices) {
      recordedCopyBufferToBuffer(enc, packed[b], offset, bufs[b], 0, elementBytes);
    }
  }
  profileSubOpEnd(`${label}.gather`, _st3);
}

/**
 * Execute a batch of optimizer updates using packed dispatch.
 *
 * Groups items by element count, dispatches one kernel per group.
 * Items with unique element counts (group size = 1) are skipped.
 *
 * @returns Set of indices that were handled by packed dispatch.
 *          The caller should dispatch remaining indices individually.
 */
export function dispatchPackedOptimizer(
  opts: PackedOptimizerOpts,
): Set<number> {
  const {
    items,
    gatherIndices,
    dispatch: dispatchFn,
    label = "packedOptim",
  } = opts;
  const handled = new Set<number>();
  if (items.length <= 1) return handled;

  // Group by element count
  const groups = new Map<number, number[]>();
  for (let i = 0; i < items.length; i++) {
    const numEl = items[i].numElements;
    const group = groups.get(numEl);
    if (group) group.push(i);
    else groups.set(numEl, [i]);
  }

  for (const [numElements, indices] of groups) {
    if (indices.length <= 1) continue;

    const bufferCount = items[indices[0]].buffers.length;

    // Sub-batch if group would exceed memory limit
    const groupBytes = numElements * 4 * indices.length * bufferCount;
    const maxBatchSize =
      groupBytes > MAX_PACKED_BYTES
        ? Math.max(
            2,
            Math.floor(MAX_PACKED_BYTES / (numElements * 4 * bufferCount)),
          )
        : indices.length;

    for (let start = 0; start < indices.length; start += maxBatchSize) {
      const end = Math.min(start + maxBatchSize, indices.length);
      const batchIndices = indices.slice(start, end);

      if (batchIndices.length <= 1) continue;

      const batchItems = batchIndices.map((i) => items[i]);
      dispatchPackedGroup(
        batchItems,
        numElements,
        gatherIndices,
        dispatchFn,
        label,
      );

      for (const i of batchIndices) handled.add(i);
    }
  }

  return handled;
}
