/**
 * Dispatch recording and replay for WebGPU compute passes.
 *
 * Records GPU dispatch sequences on first execution, then replays bind groups
 * directly on subsequent steps — bypassing all JS dispatch logic.
 */

import type { GPUBuffer, GPUComputePipeline, GPUBindGroup } from "./gpu-types";
import {
  getSharedEncoderInstance,
  autoFlushSharedEncoder,
  incrementSharedEncoderPassCount,
} from "./shared-encoder";
import { getTimestampWrites, setProfileModule } from "./profiler";

// ---------------------------------------------------------------------------
// RecordedDispatch interface
// ---------------------------------------------------------------------------

/** Recorded dispatch entry for replay. */
export interface RecordedDispatch {
  pipeline: GPUComputePipeline;
  bindGroup: GPUBindGroup;
  workgroupsX: number;
  workgroupsY: number;
  workgroupsZ: number;
  /** GPUBuffers referenced by this bind group. Populated during recording for pinning. */
  buffers?: GPUBuffer[];
  /** Op label for GPU timestamp profiling during replay. */
  label?: string;
  /** Module label for per-module GPU breakdown during replay. */
  module?: string;
}

// ---------------------------------------------------------------------------
// Recording state
// ---------------------------------------------------------------------------

/** Active recording buffer (null = not recording). */
export let dispatchRecordingBuffer: RecordedDispatch[] | null = null;

/** Last bind group's buffer list — captured during recording for pinning. */
export let lastBindGroupBuffers: GPUBuffer[] | null = null;

/** Set lastBindGroupBuffers from an external module (ESM bindings are read-only). */
export function setLastBindGroupBuffers(bufs: GPUBuffer[] | null): void {
  lastBindGroupBuffers = bufs;
}

// ---------------------------------------------------------------------------
// Replay Buffer Pinning
// ---------------------------------------------------------------------------

// When dispatch replay caches exist, buffers referenced by recorded bind groups
// must not be destroyed between steps. This set accumulates all such buffers
// across all plans (forward, backward, optimizer).
// Checked in deferredDestroy/deferredDestroyUntracked/deferredDestroyBuffer.
export let replayPinnedBufferSet: Set<GPUBuffer> | null = null;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Get and clear the last bind group's buffer list. Used by matmul/fusion recording.
 *  Also immediately pins the buffers to prevent pool recycling during recording. */
export function getAndClearLastBindGroupBuffers(): GPUBuffer[] | undefined {
  const bufs = lastBindGroupBuffers ?? undefined;
  lastBindGroupBuffers = null;
  // Immediately pin buffers during recording to prevent pool recycling.
  // Without this, a params buffer could be released to the pool, reused by a
  // later op, and overwritten — making the recorded bind group reference stale data.
  if (bufs && replayPinnedBufferSet) {
    for (const b of bufs) replayPinnedBufferSet.add(b);
  }
  return bufs;
}

/** Start recording dispatches into the given buffer. */
export function startDispatchRecording(buffer: RecordedDispatch[]): void {
  dispatchRecordingBuffer = buffer;
  // Initialize pin set at recording start so buffers are pinned immediately
  // as they're captured (not deferred until after recording completes).
  if (!replayPinnedBufferSet) {
    replayPinnedBufferSet = new Set();
  }
}

/** Stop recording dispatches. */
export function stopDispatchRecording(): void {
  dispatchRecordingBuffer = null;
}

/**
 * Replay a sequence of recorded dispatches directly onto the shared encoder.
 * Skips all JS-level dispatch logic (pipeline lookup, params, bind group creation).
 * Caller must ensure the shared encoder is active.
 */
export function replayDispatches(dispatches: RecordedDispatch[]): void {
  if (!getSharedEncoderInstance()) {
    throw new Error("replayDispatches requires an active shared encoder");
  }
  for (let i = 0; i < dispatches.length; i++) {
    const d = dispatches[i];
    // Restore module context and attach GPU timestamp queries during profiling
    if (d.module) setProfileModule(d.module);
    const tsWrites = getTimestampWrites(d.label ?? "unknown");
    const pass = getSharedEncoderInstance().beginComputePass(
      tsWrites ? { timestampWrites: tsWrites } : undefined,
    );
    pass.setPipeline(d.pipeline);
    pass.setBindGroup(0, d.bindGroup);
    pass.dispatchWorkgroups(d.workgroupsX, d.workgroupsY, d.workgroupsZ);
    pass.end();
    incrementSharedEncoderPassCount();
    autoFlushSharedEncoder();
  }
}

/** Add buffers to the replay pinned set. Called when a replay cache is built. */
export function addReplayPinnedBuffers(pins: Set<GPUBuffer>): void {
  if (!replayPinnedBufferSet) {
    replayPinnedBufferSet = new Set(pins);
  } else {
    for (const b of pins) replayPinnedBufferSet.add(b);
  }
}
