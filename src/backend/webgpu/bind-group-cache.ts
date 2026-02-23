/**
 * Bind Group Cache — Sequence-indexed caching for WebGPU bind groups and params buffers.
 *
 * Plans execute the same ops in the same order each step. Dispatch #i in step N
 * corresponds to dispatch #i in step N+1. Rather than building string keys from
 * unstable buffer IDs, the cache indexes by dispatch sequence position and validates
 * by GPUBuffer/pipeline pointer equality.
 *
 * Also manages the params buffer pool and sequence-indexed params buffer reuse.
 */

import type { GPUBuffer, GPUComputePipeline, GPUBindGroup, GPUDevice } from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { bufferPool } from "./buffer-pool";
import { profileApiCall } from "./profiler";
import {
  activeBatch,
  sharedEncoder as sharedEncoderFlag,
  autoFlushSharedEncoder,
  deferUniformBufferForSharedEncoder,
  getCurrentOpLabel,
} from "./shared-encoder";
import {
  dispatchRecordingBuffer,
  replayPinnedBufferSet,
  setLastBindGroupBuffers,
} from "./dispatch-recording";

import {
  isArenaBuffer,
} from "./buffer-arena";

// ============================================================================
// Injection callbacks for cross-module state that hasn't been extracted yet.
// These are wired up by index.ts (or eventually buffer-arena.ts / output-buffer module).
// ============================================================================

/** Callback to get/set the output sequence index (lives in output buffer section). */
let _getOutputSeqIndex: () => number = () => 0;
let _setOutputSeqIndex: (v: number) => void = () => {};

/** Callback to clear arena state (lives in buffer-arena section). */
let _clearArenaState: () => void = () => {};

/** Callback to reset arena resolve stats (lives in buffer-arena section). */
let _resetArenaStats: () => void = () => {};

/** Callback to clear output buffer state (lives in output buffer section). */
let _clearOutputState: () => void = () => {};

export function _setOutputSeqIndexAccessors(get: () => number, set: (v: number) => void): void {
  _getOutputSeqIndex = get;
  _setOutputSeqIndex = set;
}

export function _setClearArenaStateFn(fn: () => void): void {
  _clearArenaState = fn;
}

export function _setResetArenaStatsFn(fn: () => void): void {
  _resetArenaStats = fn;
}

export function _setClearOutputStateFn(fn: () => void): void {
  _clearOutputState = fn;
}

// ============================================================================
// Params Buffer Pool
// ============================================================================

// General-purpose params buffer pool — pools uniform buffers by size class (4, 8, 16, 32, 48, 64).
// Replaces the old 4-byte-only uniformBufferPool with support for arbitrary param sizes.
export const paramsBufferPools: Map<number, GPUBuffer[]> = new Map();
export const MAX_PARAMS_POOL_SIZE_PER_CLASS = 256;

export function paramsBufferSizeClass(byteLength: number): number {
  if (byteLength <= 4) return 4;
  if (byteLength <= 8) return 8;
  if (byteLength <= 16) return 16;
  if (byteLength <= 32) return 32;
  if (byteLength <= 48) return 48;
  return 64;
}

// Pre-allocated Uint32Array pool to avoid ~700 short-lived allocations per step.
// Each op dispatch creates a params array; reusing pre-allocated arrays reduces GC pressure.
const paramsArrayPool: Uint32Array[] = [];
for (let i = 0; i <= 8; i++) {
  paramsArrayPool.push(new Uint32Array(i));
}

/** Get a reusable Uint32Array of the given word count. Caller must set values before use. */
export function getParamsArray(wordCount: number): Uint32Array {
  if (wordCount <= 8) return paramsArrayPool[wordCount];
  return new Uint32Array(wordCount); // rare: allocate for large params
}

// Convenience helpers that fill and return pooled arrays (avoids new Uint32Array([...]) allocations)
export function params1(a: number): Uint32Array { const p = paramsArrayPool[1]; p[0] = a; return p; }
export function params2(a: number, b: number): Uint32Array { const p = paramsArrayPool[2]; p[0] = a; p[1] = b; return p; }
export function params3(a: number, b: number, c: number): Uint32Array { const p = paramsArrayPool[3]; p[0] = a; p[1] = b; p[2] = c; return p; }
export function params4(a: number, b: number, c: number, d: number): Uint32Array { const p = paramsArrayPool[4]; p[0] = a; p[1] = b; p[2] = c; p[3] = d; return p; }
export function params5(a: number, b: number, c: number, d: number, e: number): Uint32Array { const p = paramsArrayPool[5]; p[0] = a; p[1] = b; p[2] = c; p[3] = d; p[4] = e; return p; }
export function params6(a: number, b: number, c: number, d: number, e: number, f: number): Uint32Array { const p = paramsArrayPool[6]; p[0] = a; p[1] = b; p[2] = c; p[3] = d; p[4] = e; p[5] = f; return p; }
export function params7(a: number, b: number, c: number, d: number, e: number, f: number, g: number): Uint32Array { const p = paramsArrayPool[7]; p[0] = a; p[1] = b; p[2] = c; p[3] = d; p[4] = e; p[5] = f; p[6] = g; return p; }

// ============================================================================
// Sequence-Indexed Params Buffer Cache
// ============================================================================

// Sequence-indexed params buffer cache: reuse the same GPUBuffer at each dispatch
// position across steps. This enables bind group cache hits since the params buffer
// pointer stays stable for a given dispatch position.
let paramsSeqIndex = 0;
const paramsSequenceBuffers: Array<{ buffer: GPUBuffer; sizeClass: number; data: Uint32Array } | null> = [];

export function createParamsBuffer(device: GPUDevice, data: Uint32Array): GPUBuffer {
  const sizeClass = paramsBufferSizeClass(data.byteLength);
  const idx = paramsSeqIndex++;

  // Try to reuse the buffer from the same dispatch position (previous step).
  // This keeps the GPUBuffer pointer stable so bind group caching can hit.
  if (!activeBatch) {
    const cached = paramsSequenceBuffers[idx];
    if (cached !== undefined && cached !== null && cached.sizeClass === sizeClass) {
      // Fast path: skip writeBuffer if data is identical (params derived from
      // tensor shapes which are constant across steps).
      if (cached.data.length === data.length) {
        let same = true;
        for (let i = 0; i < data.length; i++) {
          if (cached.data[i] !== data[i]) { same = false; break; }
        }
        if (same) {
          return cached.buffer;  // Skip writeBuffer entirely
        }
      }
      // Data changed — write new data, update cached copy
      profileApiCall("writeBuffer", () => device.queue.writeBuffer(cached.buffer, 0, data));
      cached.data.set(data);
      return cached.buffer;
    }

    // Fallback: try pool
    const pool = paramsBufferPools.get(sizeClass);
    if (pool && pool.length > 0) {
      const buffer = pool.pop()!;
      profileApiCall("writeBuffer", () => device.queue.writeBuffer(buffer, 0, data));
      paramsSequenceBuffers[idx] = { buffer, sizeClass, data: data.slice() };
      paramsSequenceSet.add(buffer);
      return buffer;
    }
  }

  const buffer = profileApiCall("createBuffer", () => device.createBuffer({
    size: sizeClass,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  }));
  profileApiCall("writeBuffer", () => device.queue.writeBuffer(buffer, 0, data));
  if (!activeBatch) {
    paramsSequenceBuffers[idx] = { buffer, sizeClass, data: data.slice() };
    paramsSequenceSet.add(buffer);
  }
  return buffer;
}

// Track which buffers are pinned to sequence positions (not returnable to pool)
const paramsSequenceSet = new Set<GPUBuffer>();

/** Get the params sequence set (for buffer classification). */
export function getParamsSequenceSet(): Set<GPUBuffer> { return paramsSequenceSet; }

export function releaseParamsBuffer(buffer: GPUBuffer): void {
  // Sequence-cached params buffers are reused across steps — don't return to pool
  if (paramsSequenceSet.has(buffer)) return;
  // Replay-pinned buffers must stay alive — referenced by recorded bind groups
  if (replayPinnedBufferSet !== null && replayPinnedBufferSet.has(buffer)) return;

  if (activeBatch) {
    activeBatch.deferredDestroyBuffers.push(buffer);
    return;
  }
  if (sharedEncoderFlag) {
    deferUniformBufferForSharedEncoder(buffer);
    autoFlushSharedEncoder();
    return;
  }
  const sizeClass = paramsBufferSizeClass(buffer.size);
  const pool = paramsBufferPools.get(sizeClass);
  if (pool) {
    if (pool.length < MAX_PARAMS_POOL_SIZE_PER_CLASS) {
      pool.push(buffer);
    } else {
      bufferPool.deferredDestroyUntracked(buffer);
    }
  } else {
    paramsBufferPools.set(sizeClass, [buffer]);
  }
}

// Backward-compatible wrappers for existing callsites
export function createUniformBuffer(device: GPUDevice, size: number): GPUBuffer {
  return createParamsBuffer(device, params1(size));
}

export function releaseUniformBuffer(buffer: GPUBuffer): void {
  releaseParamsBuffer(buffer);
}

// ============================================================================
// Profiled Bind Group Creation
// ============================================================================

// Profiled helper for hot-path WebGPU API calls
export function profiledCreateBindGroup(device: GPUDevice, descriptor: any): GPUBindGroup {
  const bg = profileApiCall("createBindGroup", () => device.createBindGroup(descriptor));
  // When recording, capture buffer references from the descriptor for replay pinning
  if (dispatchRecordingBuffer && descriptor.entries) {
    const bufs: GPUBuffer[] = [];
    for (const e of descriptor.entries) {
      const r = e.resource;
      if (r && typeof r === "object" && "buffer" in r) bufs.push(r.buffer);
    }
    setLastBindGroupBuffers(bufs);
  }
  return bg;
}

// ============================================================================
// Sequence-Indexed Bind Group Cache
// ============================================================================

// Plans execute the same ops in the same order each step. Dispatch #i in step N
// corresponds to dispatch #i in step N+1. Rather than building string keys from
// unstable buffer IDs, index the cache by dispatch sequence position and validate
// by GPUBuffer/pipeline pointer equality.
let dispatchIndex = 0;
const sequenceEntries: Array<{
  bindGroup: GPUBindGroup;
  pipeline: GPUComputePipeline;
  buffers: GPUBuffer[];
} | null> = [];
let seqCacheHits = 0;
let seqCacheMisses = 0;
let seqCacheMissLog: Array<{ idx: number; reason: string; label: string | null; details: string }> = [];

/**
 * Create or retrieve a cached bind group for simple (no offset/size) buffer bindings.
 * Entries are built internally: binding i -> { buffer: buffers[i] }.
 *
 * Uses sequence-indexed caching: each dispatch position in a step maps to the same
 * position in the next step. Validation is by pointer equality on pipeline + buffers.
 */
export function cachedCreateBindGroup(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  buffers: GPUBuffer[],
): GPUBindGroup {
  const idx = dispatchIndex++;
  const entry = sequenceEntries[idx];

  if (entry !== undefined && entry !== null
      && entry.pipeline === pipeline
      && entry.buffers.length === buffers.length) {
    let match = true;
    for (let i = 0; i < buffers.length; i++) {
      if (entry.buffers[i] !== buffers[i]) { match = false; break; }
    }
    if (match) {
      seqCacheHits++;
      if (dispatchRecordingBuffer) setLastBindGroupBuffers(entry.buffers);
      return entry.bindGroup;
    }
  }

  seqCacheMisses++;
  if (seqCacheMissLog.length < 200) {
    let reason = "new";
    let details = "";
    if (entry !== undefined && entry !== null) {
      if (entry.pipeline !== pipeline) reason = "pipeline";
      else if (entry.buffers.length !== buffers.length) reason = "buf-count";
      else {
        const changed: number[] = [];
        for (let i = 0; i < buffers.length; i++) {
          if (entry.buffers[i] !== buffers[i]) changed.push(i);
        }
        reason = `buf[${changed.join(",")}]`;
        // Log sizes and arena status for changed buffers
        const parts: string[] = [];
        for (const ci of changed) {
          const oldB = entry.buffers[ci];
          const newB = buffers[ci];
          const oldArena = isArenaBuffer(oldB) ? "A" : "P";
          const newArena = isArenaBuffer(newB) ? "A" : "P";
          parts.push(`${ci}:${oldB.size}${oldArena}->${newB.size}${newArena}`);
        }
        details = parts.join(" ");
      }
    }
    seqCacheMissLog.push({ idx, reason, label: getCurrentOpLabel(), details });
  }
  const entries: Array<{ binding: number; resource: { buffer: GPUBuffer } }> = [];
  for (let i = 0; i < buffers.length; i++) {
    entries.push({ binding: i, resource: { buffer: buffers[i] } });
  }
  const bindGroup = profileApiCall("createBindGroup", () =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    })
  );

  const bufCopy = buffers.slice();
  sequenceEntries[idx] = { bindGroup, pipeline, buffers: bufCopy };
  if (dispatchRecordingBuffer) setLastBindGroupBuffers(bufCopy);
  return bindGroup;
}

// ============================================================================
// Dispatch Sequence Management
// ============================================================================

/** Reset the dispatch sequence counter to 0. Call at the start of each step. */
export function resetDispatchSequence(): void {
  dispatchIndex = 0;
  paramsSeqIndex = 0;
  _setOutputSeqIndex(0);
}

/** Set dispatch sequence counters to specific positions (for replay cache). */
export function setDispatchSequenceCounters(dispatch: number, params: number, output: number): void {
  dispatchIndex = dispatch;
  paramsSeqIndex = params;
  _setOutputSeqIndex(output);
}

/** Get current dispatch sequence counters (for recording). */
export function getDispatchSequenceCounters(): { dispatch: number; params: number; output: number } {
  return { dispatch: dispatchIndex, params: paramsSeqIndex, output: _getOutputSeqIndex() };
}

// ============================================================================
// Cache Management
// ============================================================================

export function clearBindGroupCache(): void {
  sequenceEntries.length = 0;
  dispatchIndex = 0;
  seqCacheHits = 0;
  seqCacheMisses = 0;
  paramsSequenceBuffers.length = 0;
  paramsSequenceSet.clear();
  paramsSeqIndex = 0;
  // Clear output buffer state (outputSequenceHints, pinnedOutputBuffers, outputSeqIndex)
  _clearOutputState();
  // Clear arena state (arenas themselves are cleaned up by plan cache eviction)
  _clearArenaState();
}

export function getBindGroupCacheStats(): { hits: number; misses: number; size: number; hitRate: number } {
  const total = seqCacheHits + seqCacheMisses;
  return { hits: seqCacheHits, misses: seqCacheMisses, size: sequenceEntries.length, hitRate: total > 0 ? seqCacheHits / total : 0 };
}

export function resetBindGroupCacheStats(): void {
  seqCacheHits = 0;
  seqCacheMisses = 0;
  seqCacheMissLog = [];
  _resetArenaStats();
}

export function getBindGroupCacheMissLog(): Array<{ idx: number; reason: string; label: string | null; details: string }> {
  return seqCacheMissLog;
}

/**
 * Get the buffer list for a specific dispatch sequence index.
 * Returns null if the index has no cached entry.
 * Used by dispatch replay to collect buffers that need pinning.
 */
export function getSequenceEntryBuffers(idx: number): GPUBuffer[] | null {
  const entry = sequenceEntries[idx];
  return entry ? entry.buffers : null;
}
