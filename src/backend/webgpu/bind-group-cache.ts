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

import { ENV } from "../../core/env";
import {
  invalidateActiveRecording,
  assignSlot,
  isCompilationRecordingActive,
  setLastBindGroupBuffers,
} from "../../executor/compiled-plan";
import {
  isArenaBuffer,
  outputSequenceHints,
  pinnedOutputBuffers,
  resetArenaResolveStats,
  resetArenaState,
} from "./buffer-arena";
import { bufferPool } from "./buffer-pool";
import { bufLogBindings } from "./buffer-debug";
import type {
  GPUBindGroup,
  GPUBuffer,
  GPUComputePipeline,
  GPUDevice,
} from "./gpu-types";
import { GPUBufferUsage } from "./gpu-types";
import { profileApiCall } from "./profiler";
import {
  autoFlushSharedEncoder,
  deferUniformBufferForSharedEncoder,
} from "./shared-encoder";
import {
  activeBatch,
  getCurrentOpLabel,
  getOutputSeqIndex,
  MAX_PARAMS_POOL_SIZE_PER_CLASS,
  paramsBufferPools,
  paramsBufferSizeClass,
  paramsSequenceSet,
  setOutputSeqIndex,
  sharedEncoderActive as sharedEncoderFlag,
} from "./webgpu-state";

// Pre-allocated Uint32Array pool to avoid ~700 short-lived allocations per step.
// Each op dispatch creates a params array; reusing pre-allocated arrays reduces GC pressure.
const paramsArrayPool: Uint32Array[] = [];
for (let i = 0; i <= 8; i++) {
  paramsArrayPool.push(new Uint32Array(i));
}

/** Fill and return a pooled Uint32Array. Avoids allocations on the hot dispatch path. */
export function params(...values: number[]): Uint32Array {
  const n = values.length;
  const p = n <= 8 ? paramsArrayPool[n] : new Uint32Array(n);
  for (let i = 0; i < n; i++) p[i] = values[i];
  return p;
}

// ============================================================================
// Sequence-Indexed Params Buffer Cache
// ============================================================================

/**
 * Bind group cache mutable state — groups dispatch index, sequence entries,
 * hit/miss counters, and params sequence state into a single typed object.
 */
interface BindGroupCacheState {
  /** Current dispatch sequence position. */
  dispatchIndex: number;
  /** Cached bind groups indexed by dispatch position. */
  sequenceEntries: Array<{
    bindGroup: GPUBindGroup;
    pipeline: GPUComputePipeline;
    buffers: GPUBuffer[];
  } | null>;
  /** Cache hit count. */
  hits: number;
  /** Cache miss count. */
  misses: number;
  /** Detailed miss log for diagnostics. */
  missLog: Array<{
    idx: number;
    reason: string;
    label: string | null;
    details: string;
  }>;
  /** Current params sequence position. */
  paramsSeqIndex: number;
  /** Cached params buffers indexed by dispatch position. */
  paramsSequenceBuffers: Array<{
    buffer: GPUBuffer;
    sizeClass: number;
    data: Uint32Array;
  } | null>;
}

const cacheState: BindGroupCacheState = {
  dispatchIndex: 0,
  sequenceEntries: [],
  hits: 0,
  misses: 0,
  missLog: [],
  paramsSeqIndex: 0,
  paramsSequenceBuffers: [],
};

export function createParamsBuffer(
  device: GPUDevice,
  data: Uint32Array,
): GPUBuffer {
  const sizeClass = paramsBufferSizeClass(data.byteLength);
  const idx = cacheState.paramsSeqIndex++;
  const compiling = isCompilationRecordingActive();

  // Try to reuse the buffer from the same dispatch position (previous step).
  // This keeps the GPUBuffer pointer stable so bind group caching can hit.
  if (!activeBatch) {
    const cached = cacheState.paramsSequenceBuffers[idx];
    if (cached != null && cached.sizeClass === sizeClass) {
      // Fast path: skip writeBuffer if data is identical (params derived from
      // tensor shapes which are constant across steps).
      if (cached.data.length === data.length) {
        let same = true;
        for (let i = 0; i < data.length; i++) {
          if (cached.data[i] !== data[i]) {
            same = false;
            break;
          }
        }
        if (same) {
          if (compiling)
            assignSlot(cached.buffer, {
              kind: "params",
              seqIndex: idx,
              data: data.slice(),
            });
          return cached.buffer; // Skip writeBuffer entirely
        }
      }
      // Data changed at a stable dispatch position. If a compiled plan is
      // being RECORDED right now, the recorded "params" slot would bake THIS
      // execution's bytes and every replay would reuse them — but bytes that
      // changed between consecutive executions of one template will keep
      // changing (per-step-varying value reaching a raw params buffer: the
      // frozen-data form of the frozen-scalar disease, previously guarded
      // only for tile-IR configs via getConfigBuffer). Invalidate the
      // recording: the plan falls back to lowered execution, which rewrites
      // params every dispatch. The payload-fingerprint hash (2809588) makes
      // this rare — values reaching params buffers normally derive from
      // hashed payload/shape — so invalidation here is a correctness
      // backstop, not a hot path.
      if (compiling) {
        invalidateActiveRecording(
          `params bytes changed across executions at seq ${idx} (label=${getCurrentOpLabel() ?? "?"}) — raw params buffers cannot carry per-step-varying values into replays`,
        );
      }
      if (ENV.TORCHLETTE_DEBUG_PARAMS_CHANGED) {
        const diffs: string[] = [];
        for (let i = 0; i < data.length; i++) {
          if (cached.data[i] !== data[i]) {
            const of = new Float32Array(Uint32Array.of(cached.data[i]).buffer)[0];
            const nf = new Float32Array(Uint32Array.of(data[i]).buffer)[0];
            diffs.push(`[${i}] ${cached.data[i]}→${data[i]} (f32 ${of}→${nf})`);
            if (diffs.length >= 4) break;
          }
        }
        console.log(
          `[params-changed] seq=${idx} label=${getCurrentOpLabel() ?? "?"} len=${cached.data.length}→${data.length} ${diffs.join(" ")}`,
        );
      }
      profileApiCall("writeBuffer", () =>
        device.queue.writeBuffer(cached.buffer, 0, data),
      );
      // Same size class does NOT imply same length (sizes are bucketed) — the
      // dispatch sequence can shift across steps and put a longer params array
      // at this position. set() would throw "offset is out of bounds".
      if (cached.data.length === data.length) {
        cached.data.set(data);
      } else {
        cached.data = data.slice();
      }
      if (compiling)
        assignSlot(cached.buffer, {
          kind: "params",
          seqIndex: idx,
          data: data.slice(),
        });
      return cached.buffer;
    }

    // Fallback: try pool
    const pool = paramsBufferPools.get(sizeClass);
    if (pool && pool.length > 0) {
      const buffer = pool.pop() as GPUBuffer;
      profileApiCall("writeBuffer", () =>
        device.queue.writeBuffer(buffer, 0, data),
      );
      cacheState.paramsSequenceBuffers[idx] = {
        buffer,
        sizeClass,
        data: data.slice(),
      };
      paramsSequenceSet.add(buffer);
      if (compiling)
        assignSlot(buffer, {
          kind: "params",
          seqIndex: idx,
          data: data.slice(),
        });
      return buffer;
    }
  }

  const buffer = profileApiCall("createBuffer", () =>
    device.createBuffer({
      size: sizeClass,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    }),
  );
  profileApiCall("writeBuffer", () =>
    device.queue.writeBuffer(buffer, 0, data),
  );
  if (!activeBatch) {
    cacheState.paramsSequenceBuffers[idx] = {
      buffer,
      sizeClass,
      data: data.slice(),
    };
    paramsSequenceSet.add(buffer);
  }
  if (compiling)
    assignSlot(buffer, { kind: "params", seqIndex: idx, data: data.slice() });
  return buffer;
}

export function releaseParamsBuffer(buffer: GPUBuffer): void {
  // Sequence-cached params buffers are reused across steps — don't return to pool
  if (paramsSequenceSet.has(buffer)) return;
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

// ============================================================================
// Profiled Bind Group Creation
// ============================================================================

// Profiled helper for hot-path WebGPU API calls
export function profiledCreateBindGroup(
  device: GPUDevice,
  descriptor: {
    layout: unknown;
    entries: Array<{
      binding: number;
      resource: { buffer: GPUBuffer; offset?: number; size?: number };
    }>;
  },
): GPUBindGroup {
  const bg = profileApiCall("createBindGroup", () =>
    device.createBindGroup(descriptor),
  );
  // When recording, capture buffer references from the descriptor for compilation
  if (isCompilationRecordingActive() && descriptor.entries) {
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
// Bind group cache state is consolidated into cacheState above.

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
  const idx = cacheState.dispatchIndex++;
  const entry = cacheState.sequenceEntries[idx];

  // Log dispatch bindings for debug tool (no-op when disabled)
  bufLogBindings(getCurrentOpLabel() ?? "unknown", buffers);

  if (
    entry != null &&
    entry.pipeline === pipeline &&
    entry.buffers.length === buffers.length
  ) {
    let match = true;
    for (let i = 0; i < buffers.length; i++) {
      if (entry.buffers[i] !== buffers[i]) {
        match = false;
        break;
      }
    }
    if (match) {
      cacheState.hits++;
      if (isCompilationRecordingActive())
        setLastBindGroupBuffers(entry.buffers);
      return entry.bindGroup;
    }
  }

  cacheState.misses++;
  if (cacheState.missLog.length < 200) {
    let reason = "new";
    let details = "";
    if (entry != null) {
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
    cacheState.missLog.push({
      idx,
      reason,
      label: getCurrentOpLabel(),
      details,
    });
  }
  const entries: Array<{ binding: number; resource: { buffer: GPUBuffer } }> =
    [];
  for (let i = 0; i < buffers.length; i++) {
    entries.push({ binding: i, resource: { buffer: buffers[i] } });
  }
  const bindGroup = profileApiCall("createBindGroup", () =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries,
    }),
  );

  const bufCopy = buffers.slice();
  cacheState.sequenceEntries[idx] = { bindGroup, pipeline, buffers: bufCopy };
  if (isCompilationRecordingActive()) setLastBindGroupBuffers(bufCopy);
  return bindGroup;
}

// ============================================================================
// Dispatch Sequence Management
// ============================================================================

/** Reset the dispatch sequence counter to 0. Call at the start of each step. */
export function resetDispatchSequence(): void {
  cacheState.dispatchIndex = 0;
  cacheState.paramsSeqIndex = 0;
  setOutputSeqIndex(0);
}

/** Set dispatch sequence counters to specific positions (for replay cache). */
export function setDispatchSequenceCounters(
  dispatch: number,
  params: number,
  output: number,
): void {
  cacheState.dispatchIndex = dispatch;
  cacheState.paramsSeqIndex = params;
  setOutputSeqIndex(output);
}

/** Get current dispatch sequence counters (for recording). */
export function getDispatchSequenceCounters(): {
  dispatch: number;
  params: number;
  output: number;
} {
  return {
    dispatch: cacheState.dispatchIndex,
    params: cacheState.paramsSeqIndex,
    output: getOutputSeqIndex(),
  };
}

// ============================================================================
// Cache Management
// ============================================================================

export function clearBindGroupCache(): void {
  cacheState.sequenceEntries.length = 0;
  cacheState.dispatchIndex = 0;
  cacheState.hits = 0;
  cacheState.misses = 0;
  cacheState.paramsSequenceBuffers.length = 0;
  paramsSequenceSet.clear();
  cacheState.paramsSeqIndex = 0;
  // Clear output buffer state (outputSequenceHints, pinnedOutputBuffers, outputSeqIndex)
  outputSequenceHints.length = 0;
  pinnedOutputBuffers.length = 0;
  setOutputSeqIndex(0);
  // Clear arena state (arenas themselves are cleaned up by plan cache eviction)
  resetArenaState();
}

export function getBindGroupCacheStats(): {
  hits: number;
  misses: number;
  size: number;
  hitRate: number;
} {
  const total = cacheState.hits + cacheState.misses;
  return {
    hits: cacheState.hits,
    misses: cacheState.misses,
    size: cacheState.sequenceEntries.length,
    hitRate: total > 0 ? cacheState.hits / total : 0,
  };
}

export function resetBindGroupCacheStats(): void {
  cacheState.hits = 0;
  cacheState.misses = 0;
  cacheState.missLog = [];
  resetArenaResolveStats();
}

export function getBindGroupCacheMissLog(): Array<{
  idx: number;
  reason: string;
  label: string | null;
  details: string;
}> {
  return cacheState.missLog;
}
