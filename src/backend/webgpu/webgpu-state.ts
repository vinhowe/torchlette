/**
 * webgpu-state.ts — Shared mutable state leaf module.
 *
 * Zero-dependency module (imports only types from gpu-types) that houses
 * mutable state variables read by multiple WebGPU backend modules. Since it
 * imports nothing from other webgpu modules, any module can safely import
 * from it without creating circular dependencies.
 *
 * This eliminates injection callbacks that were previously used to break
 * circular dependencies between buffer-pool, shared-encoder, gpu-context,
 * buffer-arena, dispatch-recording, and bind-group-cache.
 *
 * All setter functions are called only at lifecycle boundaries (step start/end),
 * not in hot dispatch loops. ESM live bindings make accessing these variables
 * identical in cost to accessing them from their original declaring modules.
 */

import type { GPUBuffer, GPUCommandBuffer, WebGPUContext } from "./gpu-types";

// ============================================================================
// Batch Execution Context (moved from shared-encoder.ts)
// ============================================================================

/**
 * Batch execution context - collects command buffers for deferred submission.
 */
export interface BatchExecutionContext {
  /** Collected command buffers to submit together */
  commandBuffers: GPUCommandBuffer[];
  /** Buffers to destroy after the batch submits (deferred from mid-batch destroy calls) */
  deferredDestroyBuffers: GPUBuffer[];
}

/** Active batch context (null when in immediate mode) */
export let activeBatch: BatchExecutionContext | null = null;

export function setActiveBatch(batch: BatchExecutionContext | null): void {
  activeBatch = batch;
}

// ============================================================================
// Shared Encoder Active Flag (moved from shared-encoder.ts)
// ============================================================================

/** Whether the shared encoder scope is currently open. */
export let sharedEncoderActive: boolean = false;

export function setSharedEncoderActive(active: boolean): void {
  sharedEncoderActive = active;
}

// ============================================================================
// Shared Encoder Write Set (moved from shared-encoder.ts)
// ============================================================================

/** Buffers written during the current shared encoder scope. */
export let sharedEncoderWriteSet: Set<GPUBuffer> = new Set();

export function resetSharedEncoderWriteSet(): void {
  sharedEncoderWriteSet = new Set();
}

/**
 * Track a buffer as written during the current shared encoder scope.
 * The buffer pool must not return this buffer for the rest of the scope.
 */
export function trackSharedEncoderWrite(buffer: GPUBuffer): void {
  if (sharedEncoderActive) {
    sharedEncoderWriteSet.add(buffer);
  }
}

// ============================================================================
// GPU Context (moved from gpu-context.ts)
// ============================================================================

/** The current WebGPU context (device, queue, etc.). */
export let gpuContext: WebGPUContext | null = null;

export function setGpuContext(ctx: WebGPUContext | null): void {
  gpuContext = ctx;
}

export function requireContext(): WebGPUContext {
  if (!gpuContext) {
    throw new Error("WebGPU backend not initialized; call initWebGPU()");
  }
  return gpuContext;
}

// ============================================================================
// Arena Buffer Set (moved from buffer-arena.ts)
// ============================================================================

/** The set of all buffers owned by any active arena (for release interception). */
export const arenaBufferSet = new Set<GPUBuffer>();

// ============================================================================
// Replay Pinned Buffer Set (moved from dispatch-recording.ts)
// ============================================================================

/**
 * When dispatch replay caches exist, buffers referenced by recorded bind groups
 * must not be destroyed between steps. This set accumulates all such buffers.
 */
export let replayPinnedBufferSet: Set<GPUBuffer> | null = null;

export function setReplayPinnedBufferSet(set: Set<GPUBuffer> | null): void {
  replayPinnedBufferSet = set;
}

// ============================================================================
// GPU Submit Counter (moved from shared-encoder.ts)
// ============================================================================

export let gpuSubmitCount = 0;

export function getSubmitCount(): number {
  return gpuSubmitCount;
}

export function incrementSubmitCount(): void {
  gpuSubmitCount++;
}

export function resetSubmitCount(): void {
  gpuSubmitCount = 0;
}

// ============================================================================
// Params Buffer Pool State (moved from bind-group-cache.ts)
// ============================================================================

/** General-purpose params buffer pool — pools uniform buffers by size class (4, 8, 16, 32, 48, 64). */
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

// ============================================================================
// Params Sequence Set (moved from bind-group-cache.ts)
// ============================================================================

/** Track which buffers are pinned to sequence positions (not returnable to pool). */
export const paramsSequenceSet = new Set<GPUBuffer>();

// ============================================================================
// Output Sequence Index (moved from buffer-arena.ts)
// ============================================================================

export let outputSeqIndex = 0;
export function getOutputSeqIndex(): number { return outputSeqIndex; }
export function setOutputSeqIndex(v: number): void { outputSeqIndex = v; }

// ============================================================================
// Current Op Label (moved from shared-encoder.ts)
// ============================================================================

/** Current op label for GPU timestamp profiling (set from lazy.ts). */
let currentOpLabel: string | null = null;
export function setCurrentOpLabel(label: string | null): void { currentOpLabel = label; }
export function getCurrentOpLabel(): string | null { return currentOpLabel; }
