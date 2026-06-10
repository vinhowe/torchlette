/**
 * Compiled Execution Plan
 *
 * A flat sequence of GPU primitives — dispatches, copies, allocations, barriers.
 * Knows nothing about op semantics (Adam, softmax, fusion groups, etc.).
 * Those are concerns of the compiler that builds the plan.
 *
 * Built by post-processing the first normal-path execution: map recorded
 * GPU buffer pointers to abstract slot indices, producing a buffer-identity-
 * independent command stream that works for both training (stable arena) and
 * inference (growing KV cache buffers).
 */

import type { DType } from "../backend/types";
import type {
  GPUBindGroup,
  GPUBuffer,
  GPUCommandEncoder,
  GPUComputePipeline,
} from "../backend/webgpu/gpu-types";
import { setCompilationRecording } from "../backend/webgpu/webgpu-state";

// ============================================================================
// Slot table
// ============================================================================

/** Index into the slot table (GPUBuffer[]). */
export type Slot = number;

/** How a slot gets its buffer at execution time. */
export type SlotSource =
  | { kind: "external"; planNodeIndex: number; inputIndex: number }
  | {
      kind: "params";
      seqIndex: number;
      data: Uint32Array;
      cachedBuffer?: GPUBuffer;
    }
  | { kind: "arena" } // Populated by an alloc command during execution
  | { kind: "write" } // Populated by a write command during execution
  | { kind: "persistent"; buffer: GPUBuffer }; // Cached/singleton buffer (e.g. attention config)

// ============================================================================
// GPU commands
// ============================================================================

export interface AllocCommand {
  tag: 0; // "alloc"
  slot: Slot;
  bytes: number;
  /** Arena allocation path: 0 = resolveOutputBuffer, 1 = allocateOutputBuffer. */
  allocKind: 0 | 1;
  /** Slots whose buffers are inputs to this op (for aliasing check in resolveOutputBuffer). */
  inputSlots: Slot[];
}

export interface DispatchCommand {
  tag: 1; // "dispatch"
  pipeline: GPUComputePipeline;
  bindings: Slot[];
  gx: number;
  gy: number;
  gz: number;
  /** Op label for profiler attribution during replay. */
  label?: string;
  /** Module label for per-module GPU breakdown during replay. */
  module?: string;
  /** Inline bind group cache — bypasses global sequence-indexed cache.
   *  Needed because the normal path mixes cachedCreateBindGroup (advances
   *  dispatchIndex) and profiledCreateBindGroup (doesn't), so the global
   *  sequence counter is misaligned during compiled plan replay. */
  cachedBindGroup?: GPUBindGroup;
  cachedBuffers?: GPUBuffer[];
}

export interface CopyCommand {
  tag: 2; // "copy"
  src: Slot;
  dst: Slot;
  srcOffset: number;
  dstOffset: number;
  bytes: number;
}

export interface WriteCommand {
  tag: 3; // "write" — host→device (tensorFromArray)
  slot: Slot;
  nodeIndex: number;
}

export interface BarrierCommand {
  tag: 4; // "barrier"
}

/**
 * Zero a (reused) buffer. zeros() clears arena/pool buffers via clearBuffer
 * because they may hold stale data; that clear is NOT a dispatch/copy, so it
 * must be recorded explicitly or the compiled replay skips it — leaving stale
 * data that corrupts accumulating ops (e.g. the embedding-grad scatter-add,
 * which reads a zeroed accumulator). See compiled-plan-clearbuffer regression.
 */
export interface ClearCommand {
  tag: 5; // "clear"
  slot: Slot;
  bytes: number;
}

/**
 * Re-write a uniform/config buffer with values RE-DERIVED from the current
 * step's plan node. Tile-IR config buffers are rewritten by the lowered path
 * on every dispatch, but a replayed dispatch binds the buffer as-is — so any
 * per-step-varying uniform (Adam's bias-corrected step_size, GradScaler's
 * inv_scale) would silently replay STALE values (the frozen-step-size bug:
 * compiled training ran with the t-of-recording learning rate forever).
 *
 * The invariant: per-step-varying values must flow into replays as DATA —
 * either a tensor write (TAG_WRITE re-executes the data source node) or a
 * volatile uniform (this command re-packs from the node's fresh payload).
 */
export interface UniformCommand {
  tag: 6; // "uniform" — volatile config re-pack from current node payload
  slot: Slot;
  nodeIndex: number;
  pack: (node: LazyIRNode) => ArrayBufferView;
}

export type GpuCommand =
  | AllocCommand
  | DispatchCommand
  | CopyCommand
  | WriteCommand
  | BarrierCommand
  | ClearCommand
  | UniformCommand;

// Numeric tags for fast switch in hot loop
export const TAG_ALLOC = 0 as const;
export const TAG_DISPATCH = 1 as const;
export const TAG_COPY = 2 as const;
export const TAG_WRITE = 3 as const;
export const TAG_BARRIER = 4 as const;
export const TAG_CLEAR = 5 as const;
export const TAG_UNIFORM = 6 as const;

// ============================================================================
// Node result mapping
// ============================================================================

/** After execution: map a plan node to its output slot + metadata. */
export interface NodeResult {
  nodeIndex: number;
  /** Output index for multi-output ops (0 = primary, 1+ = secondary). */
  outputIndex: number;
  slot: Slot;
  shape: number[];
  strides: number[];
  dtype: DType;
  offset: number;
}

// ============================================================================
// Compiled plan
// ============================================================================

export interface CompiledPlan {
  /** GPU commands in execution order. */
  commands: GpuCommand[];
  /** Slot sources — how each slot gets its buffer. */
  slots: SlotSource[];
  /** Node result assignments (run after GPU execution). */
  results: NodeResult[];
  /** Whether this compiled plan is valid for execution. */
  valid: boolean;
  /** Sequence counter positions at the end of the normal-path recording execution.
   *  Restored after compiled plan execution so subsequent non-compiled plans
   *  see the correct global sequence counter positions. */
  endCounters?: { dispatch: number; params: number; output: number };
  /**
   * Planned-buffer mode (bounded arena): the recorded buffer per alloc slot.
   * Replays bind these directly — the exact lifetime-reusing assignment the
   * recording execution proved safe under the same command order — instead
   * of growing an unbudgeted per-position arena.
   */
  allocBuffers?: (GPUBuffer | undefined)[];
  /** Buffers adopted from the pool by this plan; destroyed on invalidation. */
  adoptedBuffers?: GPUBuffer[];
}

// ============================================================================
// Compilation: build CompiledPlan from recorded dispatches
// ============================================================================

import {
  arenaLivenessEnabled,
  type BufferArena,
} from "../backend/webgpu/buffer-arena";
import { bufferPool } from "../backend/webgpu/buffer-pool";
import { gpuMemoryTracker } from "../backend/webgpu/memory-tracker";
import { arenaBufferSet } from "../backend/webgpu/webgpu-state";
/** Recorded dispatch entry — the subset of fields needed by the compiled plan. */
export interface RecordedDispatch {
  pipeline: GPUComputePipeline;
  bindGroup: import("../backend/webgpu/gpu-types").GPUBindGroup;
  workgroupsX: number;
  workgroupsY: number;
  workgroupsZ: number;
  /** GPUBuffers referenced by this bind group. */
  buffers?: GPUBuffer[];
  /**
   * Slots resolved at RECORD time (parallel to `buffers`, -1 = unmapped).
   * Slot resolution must be temporal: when the allocator reuses one GPUBuffer
   * for two lifetimes (pool reuse under the bounded arena), the final
   * buffer→slot map only knows the LAST lifetime — resolving at build time
   * would wire earlier consumers to the wrong slot.
   */
  slots?: number[];
  /** Op label for GPU timestamp profiling. */
  label?: string;
  /** Module label for per-module GPU breakdown. */
  module?: string;
}

import type { LazyIRNode } from "../graph/types";
import { getInputStorage } from "./op-dispatch";

/** Recorded buffer copy for compilation. */
export interface RecordedCopy {
  src: GPUBuffer;
  dst: GPUBuffer;
  srcOffset: number;
  dstOffset: number;
  bytes: number;
}

/**
 * Build a CompiledPlan from a completed normal-path execution.
 *
 * Maps every GPU buffer pointer encountered during execution to an abstract
 * slot index, producing a buffer-identity-independent command stream.
 *
 * @param dispatches Recorded compute dispatches from the execution
 * @param copies Recorded buffer-to-buffer copies from the execution
 * @param barriers Indices into the interleaved command stream where barriers occurred
 * @param arena The buffer arena used during execution
 * @param planNodes The plan nodes (with results populated after execution)
 * @param allocLog Log of output buffer allocations: {buffer, bytes, kind, inputBuffers}[]
 */
export function buildCompiledPlan(input: {
  /** Interleaved command log: dispatches, copies, allocs, barriers, writes in execution order. */
  commandLog: CommandLogEntry[];
  arena: BufferArena;
  planNodes: LazyIRNode[];
  /** Map from GPUBuffer to the slot index. Built during recording. */
  bufferToSlot: Map<GPUBuffer, Slot>;
  /** Slot sources, indexed by slot. Built during recording. */
  slotSources: SlotSource[];
  /** Node results to harvest after execution. */
  nodeResults: NodeResult[];
}): CompiledPlan {
  const { commandLog, bufferToSlot, slotSources, nodeResults } = input;

  // A recording hook flagged something a frozen replay would get wrong
  // (e.g. step-varying uniform data with no volatile repack) — fall back
  // to the lowered path for this template.
  if (recordingInvalidReason) {
    if (process.env.TORCHLETTE_DEBUG_COMPILED) {
      console.log(`[compiled] FAIL: ${recordingInvalidReason}`);
    }
    recordingInvalidReason = null;
    return { commands: [], slots: [], results: [], valid: false };
  }

  const commands: GpuCommand[] = [];

  // Planned-buffer mode (EXPERIMENTAL, TORCHLETTE_COMPILED_PLANNED=1 under
  // the bounded arena): the recording execution allocated through the
  // budgeted POOL with liveness reuse, so memory stayed bounded — but the
  // recorded buffer identities are then per-LIFETIME, not persistent.
  // Capture the recorded buffer for every alloc slot and pin (adopt) those
  // buffers to the plan, so replays bind the exact assignment the recording
  // proved safe under the same command order.
  //
  // STATUS 2026-06-10: structure validated (fixed-weight probe replays at
  // 1e-8; all allocs bind planned, zero fallbacks), but the full training
  // loop reads STALE grads on replay steps (first dynamic-fallback replay is
  // correct, the second is stale) — a cross-step buffer-reuse/readback
  // ordering issue still to be root-caused. Off by default.
  const planned =
    arenaLivenessEnabled() && process.env.TORCHLETTE_COMPILED_PLANNED === "1";
  const allocBuffers: (GPUBuffer | undefined)[] | undefined = planned
    ? []
    : undefined;

  // Assign a persistent slot at BUILD time for a buffer unmapped at record
  // time (config caches, k-split temps — long-lived singletons, never
  // lifetime-split, so build-time resolution is safe for them).
  const persistentSlot = (buf: GPUBuffer): Slot => {
    let slot = bufferToSlot.get(buf);
    if (slot === undefined) {
      slot = slotSources.length;
      bufferToSlot.set(buf, slot);
      slotSources.push({ kind: "persistent", buffer: buf });
    }
    return slot;
  };

  for (const entry of commandLog) {
    switch (entry.kind) {
      case "dispatch": {
        const d = entry.dispatch;
        if (!d.buffers || d.buffers.length === 0) {
          // Dispatch without recorded buffers — skip (shouldn't happen with arena)
          continue;
        }
        const bindings: Slot[] = [];
        for (let i = 0; i < d.buffers.length; i++) {
          const recorded = d.slots?.[i] ?? -1;
          bindings.push(recorded >= 0 ? recorded : persistentSlot(d.buffers[i]));
        }
        commands.push({
          tag: TAG_DISPATCH,
          pipeline: d.pipeline,
          bindings,
          gx: d.workgroupsX,
          gy: d.workgroupsY,
          gz: d.workgroupsZ,
          label: d.label,
          module: d.module,
        });
        break;
      }
      case "alloc": {
        const slot = entry.slot;
        if (slot < 0) {
          if (process.env.TORCHLETTE_DEBUG_COMPILED) {
            console.log(
              `[compiled] FAIL: alloc buffer unmapped, size=${entry.bytes}, commands so far=${commands.length}`,
            );
          }
          return { commands: [], slots: [], results: [], valid: false };
        }
        if (allocBuffers) allocBuffers[slot] = entry.buffer;
        commands.push({
          tag: TAG_ALLOC,
          slot,
          bytes: entry.bytes,
          allocKind: entry.allocKind,
          inputSlots: entry.inputSlots,
        });
        break;
      }
      case "copy": {
        const srcSlot =
          entry.srcSlot >= 0 ? entry.srcSlot : persistentSlot(entry.copy.src);
        const dstSlot =
          entry.dstSlot >= 0 ? entry.dstSlot : persistentSlot(entry.copy.dst);
        commands.push({
          tag: TAG_COPY,
          src: srcSlot,
          dst: dstSlot,
          srcOffset: entry.copy.srcOffset,
          dstOffset: entry.copy.dstOffset,
          bytes: entry.copy.bytes,
        });
        break;
      }
      case "write": {
        if (entry.slot < 0) {
          return { commands: [], slots: [], results: [], valid: false };
        }
        commands.push({
          tag: TAG_WRITE,
          slot: entry.slot,
          nodeIndex: entry.nodeIndex,
        });
        break;
      }
      case "uniform": {
        // Config buffers live in long-lived per-dispatcher caches, so they are
        // not arena/write slots — assign as persistent if not already mapped.
        commands.push({
          tag: TAG_UNIFORM,
          slot: persistentSlot(entry.buffer),
          nodeIndex: entry.nodeIndex,
          pack: entry.pack,
        });
        break;
      }
      case "clear": {
        if (entry.slot < 0) {
          // The cleared buffer has no slot (not recorded as an alloc/output).
          // Can't replay the zeroing safely — invalidate and fall back to the
          // lowered path rather than ship a plan that skips a required clear.
          if (process.env.TORCHLETTE_DEBUG_COMPILED) {
            console.log(
              `[compiled] FAIL: clear buffer unmapped, size=${entry.bytes}, commands so far=${commands.length}`,
            );
          }
          return { commands: [], slots: [], results: [], valid: false };
        }
        commands.push({ tag: TAG_CLEAR, slot: entry.slot, bytes: entry.bytes });
        break;
      }
      case "barrier": {
        commands.push({ tag: TAG_BARRIER });
        break;
      }
    }
  }

  // Pin the recorded alloc buffers to the plan: take them out of pool
  // circulation (adopt) and shield them from the release chain via
  // arenaBufferSet — the same protection arena buffers get. Adoption is
  // REFCOUNTED because a pool buffer released by one plan's liveness can be
  // re-acquired by a LATER plan in the same step (forward's dead activation
  // buffer reused by backward) — both plans then pin the same buffer and it
  // must survive until the last one is destroyed. Buffers already
  // arena-owned (small, below the spill threshold) are left to the arena.
  let adoptedBuffers: GPUBuffer[] | undefined;
  if (allocBuffers) {
    adoptedBuffers = [];
    const seen = new Set<GPUBuffer>();
    for (const buf of allocBuffers) {
      if (!buf || seen.has(buf)) continue;
      seen.add(buf);
      const rc = adoptedRefCount.get(buf);
      if (rc !== undefined) {
        adoptedRefCount.set(buf, rc + 1);
        adoptedBuffers.push(buf);
      } else if (!arenaBufferSet.has(buf)) {
        bufferPool.adoptBuffer(buf);
        arenaBufferSet.add(buf);
        adoptedRefCount.set(buf, 1);
        adoptedBuffers.push(buf);
      }
    }
  }

  if (process.env.TORCHLETTE_DEBUG_COMPILED) {
    const dispatchCount = commands.filter((c) => c.tag === TAG_DISPATCH).length;
    const allocCount = commands.filter((c) => c.tag === TAG_ALLOC).length;
    const barrierCount = commands.filter((c) => c.tag === TAG_BARRIER).length;
    const writeCount = commands.filter((c) => c.tag === TAG_WRITE).length;
    const copyCount = commands.filter((c) => c.tag === TAG_COPY).length;
    const clearCount = commands.filter((c) => c.tag === TAG_CLEAR).length;
    const uniformCount = commands.filter((c) => c.tag === TAG_UNIFORM).length;
    const slotKinds: Record<string, number> = {};
    for (const s of slotSources) {
      slotKinds[s.kind] = (slotKinds[s.kind] || 0) + 1;
    }
    console.log(
      `[compiled] Built: ${commands.length} cmds (${dispatchCount} dispatch, ${allocCount} alloc, ${barrierCount} barrier, ${writeCount} write, ${copyCount} copy, ${clearCount} clear, ${uniformCount} uniform), ${slotSources.length} slots (${JSON.stringify(slotKinds)}), ${nodeResults.length} results`,
    );
  }

  return {
    commands,
    slots: slotSources,
    results: nodeResults,
    valid: true,
    allocBuffers,
    adoptedBuffers,
  };
}

/** Refcount for buffers adopted by compiled plans (shared across plans when
 *  pool-liveness reuse hands one buffer to multiple plans in a step). */
const adoptedRefCount = new Map<GPUBuffer, number>();

/**
 * Release a plan's adopted buffers (planned-buffer mode). Called when the
 * compiled plan is invalidated or its template evicted. Buffers still owned
 * by a live tensor re-enter the normal release chain once unpinned; the rest
 * are destroyed immediately.
 */
export function destroyCompiledPlanBuffers(compiled: CompiledPlan): void {
  if (!compiled.adoptedBuffers) return;
  for (const buf of compiled.adoptedBuffers) {
    const rc = adoptedRefCount.get(buf);
    if (rc !== undefined && rc > 1) {
      adoptedRefCount.set(buf, rc - 1);
      continue;
    }
    adoptedRefCount.delete(buf);
    arenaBufferSet.delete(buf);
    if (bufferPool.canRecycle(buf)) {
      gpuMemoryTracker.trackDeallocation(buf);
      try {
        buf.destroy();
      } catch {
        /* already destroyed */
      }
    }
  }
  compiled.adoptedBuffers = undefined;
  compiled.allocBuffers = undefined;
}

// ============================================================================
// Command log entry types (used during recording)
// ============================================================================

// Slots are resolved at RECORD time (-1 / undefined = unmapped at record
// time → assigned a persistent slot at build). See RecordedDispatch.slots
// for why temporal resolution matters under buffer-lifetime reuse.
export type CommandLogEntry =
  | { kind: "dispatch"; dispatch: RecordedDispatch }
  | {
      kind: "alloc";
      buffer: GPUBuffer;
      bytes: number;
      allocKind: 0 | 1;
      inputBuffers: GPUBuffer[];
      slot: Slot;
      inputSlots: Slot[];
    }
  | { kind: "copy"; copy: RecordedCopy; srcSlot: Slot; dstSlot: Slot }
  | { kind: "write"; buffer: GPUBuffer; nodeIndex: number; slot: Slot }
  | { kind: "clear"; buffer: GPUBuffer; bytes: number; slot: Slot }
  | {
      kind: "uniform";
      buffer: GPUBuffer;
      nodeIndex: number;
      pack: (node: LazyIRNode) => ArrayBufferView;
    }
  | { kind: "barrier" };

// ============================================================================
// Recording state — module-level mutable state for the recording pass
// ============================================================================

/** Active command log (null = not recording). */
let activeCommandLog: CommandLogEntry[] | null = null;
/** Active buffer→slot mapping (null = not recording). */
let activeBufferToSlot: Map<GPUBuffer, Slot> | null = null;
/** Active slot sources list (null = not recording). */
let activeSlotSources: SlotSource[] | null = null;
/** Next slot index to assign. */
let nextSlot = 0;
/** Plan-node index of the node currently being executed (recording only). */
let recordingNodeIndex = -1;
/**
 * Set when a recording hook detects something a replay cannot faithfully
 * reproduce (e.g. a config buffer whose data changed across executions with
 * no volatile repack). Survives stopCompilationRecording so buildCompiledPlan
 * (which runs after) can read it; reset by startCompilationRecording.
 */
let recordingInvalidReason: string | null = null;

/** Start compilation recording. Call before the normal-path execution. */
export function startCompilationRecording(): {
  commandLog: CommandLogEntry[];
  bufferToSlot: Map<GPUBuffer, Slot>;
  slotSources: SlotSource[];
} {
  activeCommandLog = [];
  activeBufferToSlot = new Map();
  activeSlotSources = [];
  nextSlot = 0;
  recordingNodeIndex = -1;
  recordingInvalidReason = null;
  setCompilationRecording(true);
  return {
    commandLog: activeCommandLog,
    bufferToSlot: activeBufferToSlot,
    slotSources: activeSlotSources,
  };
}

/** Executor hook: the plan-node index about to be executed (for recording). */
export function setRecordingNodeIndex(nodeIndex: number): void {
  recordingNodeIndex = nodeIndex;
}

/**
 * Poison the active recording. buildCompiledPlan returns an invalid plan, so
 * the template keeps using the (always-correct) lowered path. Use when a
 * recording hook observes state a frozen replay would get wrong.
 */
export function invalidateActiveRecording(reason: string): void {
  if (activeCommandLog && !recordingInvalidReason) {
    recordingInvalidReason = reason;
    if (process.env.TORCHLETTE_DEBUG_COMPILED) {
      console.log(`[compiled] recording invalidated: ${reason}`);
    }
  }
}

/** Stop compilation recording. */
export function stopCompilationRecording(): void {
  activeCommandLog = null;
  activeBufferToSlot = null;
  activeSlotSources = null;
  nextSlot = 0;
  setCompilationRecording(false);
}

/** Check if compilation recording is active. */
export function isCompilationRecordingActive(): boolean {
  return activeCommandLog !== null;
}

/**
 * Assign a slot to a buffer if not already assigned.
 * Returns the slot index.
 */
export function assignSlot(buffer: GPUBuffer, source: SlotSource): Slot {
  if (!activeBufferToSlot || !activeSlotSources) return -1;
  const existing = activeBufferToSlot.get(buffer);
  if (existing !== undefined) return existing;
  const slot = nextSlot++;
  activeBufferToSlot.set(buffer, slot);
  activeSlotSources.push(source);
  return slot;
}

/** Get the slot for a buffer (must already be assigned). */
export function getSlot(buffer: GPUBuffer): Slot | undefined {
  return activeBufferToSlot?.get(buffer);
}

// ============================================================================
// Bind group buffer capture — tracks which GPUBuffers a bind group references.
// Set by cachedCreateBindGroup/profiledCreateBindGroup, consumed by
// dispatchComputePass to populate dispatch recordings with buffer lists.
// ============================================================================

/** Last bind group's buffer list — captured during compilation recording. */
let lastBindGroupBuffers: GPUBuffer[] | null = null;

/** Set the last bind group's buffer list. Called from bind-group-cache.ts. */
export function setLastBindGroupBuffers(bufs: GPUBuffer[] | null): void {
  lastBindGroupBuffers = bufs;
}

/** Get and clear the last bind group's buffer list. Called from dispatch.ts. */
export function getAndClearLastBindGroupBuffers(): GPUBuffer[] | undefined {
  const bufs = lastBindGroupBuffers ?? undefined;
  lastBindGroupBuffers = null;
  return bufs;
}

// ============================================================================
// Recording hooks — called from backend modules during normal-path execution
// ============================================================================

/**
 * Record a dispatch. Called from dispatchComputePass when compilation recording
 * is active. The dispatch's buffers must already have slots assigned.
 */
export function recordDispatch(dispatch: RecordedDispatch): void {
  if (!activeCommandLog) return;
  // Resolve binding slots NOW — the buffer→slot map is temporal under
  // buffer-lifetime reuse (see RecordedDispatch.slots).
  if (dispatch.buffers && activeBufferToSlot) {
    const slots = new Array<number>(dispatch.buffers.length);
    for (let i = 0; i < dispatch.buffers.length; i++) {
      slots[i] = activeBufferToSlot.get(dispatch.buffers[i]) ?? -1;
    }
    dispatch.slots = slots;
  }
  activeCommandLog.push({ kind: "dispatch", dispatch });
}

/**
 * Record an output buffer allocation.
 * Called from resolveOutputBuffer/allocateOutputBuffer.
 */
export function recordAlloc(
  buffer: GPUBuffer,
  bytes: number,
  allocKind: 0 | 1,
  inputBuffers: GPUBuffer[],
): void {
  if (!activeCommandLog) return;
  // Lifetime splitting: when the allocator (pool reuse under the bounded
  // arena) hands the SAME GPUBuffer to a second alloc position, the two
  // allocations are distinct LIFETIMES and must get distinct slots — commands
  // recorded before this point keep the old slot (first lifetime), commands
  // after resolve to the new slot (second lifetime). Without this, all
  // references collapse onto the first slot and the replay wires later
  // consumers to the wrong logical value.
  let slot: Slot;
  if (activeBufferToSlot && activeSlotSources) {
    const existing = activeBufferToSlot.get(buffer);
    if (existing !== undefined) {
      // An alloc is always a fresh lifetime, whatever the buffer's previous
      // slot kind was (arena, external, write) — re-point the buffer.
      slot = nextSlot++;
      activeBufferToSlot.set(buffer, slot);
      activeSlotSources.push({ kind: "arena" });
    } else {
      slot = assignSlot(buffer, { kind: "arena" });
    }
  } else {
    slot = -1;
  }
  const inputSlots: Slot[] = [];
  for (const ib of inputBuffers) {
    const is_ = activeBufferToSlot?.get(ib);
    if (is_ !== undefined) inputSlots.push(is_);
  }
  activeCommandLog.push({
    kind: "alloc",
    buffer,
    bytes,
    allocKind,
    inputBuffers,
    slot,
    inputSlots,
  });
}

/** Record a buffer-to-buffer copy. Called from shared encoder wrapper. */
export function recordCopy(copy: RecordedCopy): void {
  if (!activeCommandLog) return;
  activeCommandLog.push({
    kind: "copy",
    copy,
    srcSlot: activeBufferToSlot?.get(copy.src) ?? -1,
    dstSlot: activeBufferToSlot?.get(copy.dst) ?? -1,
  });
}

/**
 * Encode a buffer-to-buffer copy on the shared encoder AND record it for the
 * compiled plan. Use this for any INTRA-PLAN copy — one whose destination is
 * later read by a dispatch in the same plan (scatter-add accumulator, cat
 * assembly, matmul k-split temp, packed-optimizer pack/unpack). If such a copy
 * is issued via the raw `enc.copyBufferToBuffer` it is NOT replayed, so the
 * compiled plan reuses the destination's stale contents — corrupting any op
 * that reads it (silent, e.g. the embedding-grad +1x/replay bug).
 *
 * Do NOT use this for READBACK/staging copies (tensor → mappable buffer for
 * item()/profiler/inf-flag): those are not part of the replayable compute and
 * target buffers outside the plan. Leave those as raw copyBufferToBuffer.
 *
 * recordCopy is a no-op when not recording, so this is safe on the hot path.
 */
export function recordedCopyBufferToBuffer(
  enc: GPUCommandEncoder,
  src: GPUBuffer,
  srcOffset: number,
  dst: GPUBuffer,
  dstOffset: number,
  bytes: number,
): void {
  enc.copyBufferToBuffer(src, srcOffset, dst, dstOffset, bytes);
  recordCopy({ src, dst, srcOffset, dstOffset, bytes });
}

/** Record a host→device write (tensorFromArray). Called from executor. */
export function recordWrite(buffer: GPUBuffer, nodeIndex: number): void {
  if (!activeCommandLog) return;
  const slot = assignSlot(buffer, { kind: "write" });
  activeCommandLog.push({ kind: "write", buffer, nodeIndex, slot });
}

/**
 * Record a VOLATILE uniform/config buffer: one whose contents must be
 * re-derived from the executing node's payload on every replay (per-step
 * values like Adam's bias-corrected step_size or GradScaler's inv_scale).
 * `pack` receives the CURRENT step's plan node and returns the full packed
 * uniform bytes; the replay re-writes the buffer in stream order, matching
 * the lowered path's writeBuffer-at-dispatch-prep semantics.
 */
export function recordVolatileUniform(
  buffer: GPUBuffer,
  pack: (node: LazyIRNode) => ArrayBufferView,
): void {
  if (!activeCommandLog) return;
  if (recordingNodeIndex < 0) {
    invalidateActiveRecording(
      "volatile uniform recorded outside node execution (no node index)",
    );
    return;
  }
  activeCommandLog.push({
    kind: "uniform",
    buffer,
    nodeIndex: recordingNodeIndex,
    pack,
  });
}

/**
 * Record a buffer-zeroing clear. Called from zeros() when it clears a reused
 * arena/pool buffer during recording. The buffer's slot is assigned by the
 * preceding recordAlloc (resolveOutputBuffer). Without this, the compiled
 * replay reuses the (stale) arena buffer without re-zeroing.
 */
export function recordClear(buffer: GPUBuffer, bytes: number): void {
  if (!activeCommandLog) return;
  activeCommandLog.push({
    kind: "clear",
    buffer,
    bytes,
    slot: activeBufferToSlot?.get(buffer) ?? -1,
  });
}

/** Record a barrier (flush). Called from executor. */
export function recordBarrier(): void {
  activeCommandLog?.push({ kind: "barrier" });
}

// ============================================================================
// Execution: run a compiled plan
// ============================================================================

import type { Backend, BackendTensor } from "../backend/types";
import {
  profiledCreateBindGroup,
  setDispatchSequenceCounters,
} from "../backend/webgpu/bind-group-cache";
import { setProfileModule } from "../backend/webgpu/profiler";
import {
  allocateOutputBuffer,
  clearActiveArena,
  clearArenaExternalInputBuffers,
  resolveOutputBuffer,
  setActiveArena,
  setArenaExternalInputBuffers,
} from "../backend/webgpu/buffer-arena";
import { flushBufferPool } from "../backend/webgpu/buffer-pool";
import { dispatchComputePass } from "../backend/webgpu/dispatch";
import { GPUBufferUsage, gpuBuffer } from "../backend/webgpu/gpu-types";
import {
  beginSharedEncoder,
  endSharedEncoder,
  flushSharedEncoder,
  getSharedEncoderInstance,
} from "../backend/webgpu/shared-encoder";
import { createTensor } from "../backend/webgpu/tensor";
import {
  paramsBufferSizeClass,
  requireContext,
  trackSharedEncoderWrite,
} from "../backend/webgpu/webgpu-state";
import { createStorageHandle } from "../graph/node-factory";
import { executeOpSync } from "./op-dispatch";

/**
 * Execute a compiled plan — the tight GPU command loop.
 *
 * Resolves abstract buffer slots to current GPUBuffers, encodes compute passes,
 * buffer copies, and host writes. ~50 lines of core logic.
 */
export async function executeCompiledPlan(
  compiled: CompiledPlan,
  planNodes: LazyIRNode[],
  arena: BufferArena,
  backend: Backend,
  externalInputBuffers: GPUBuffer[],
): Promise<void> {
  const ctx = requireContext();
  const device = ctx.device;
  const slots: GPUBuffer[] = new Array(compiled.slots.length);
  const externalInputSet = compiled.allocBuffers
    ? new Set(externalInputBuffers)
    : null;
  // Planned buffers whose FIRST lifetime this replay failed safety checks
  // (still owned by a live tensor, or external-input conflict) — all their
  // lifetimes fall back to dynamic allocation this replay.
  let plannedFallback: Set<GPUBuffer> | null = null;
  let plannedSeen: Set<GPUBuffer> | null = null;
  let plannedBindCount = 0;
  let plannedFallbackCount = 0;

  if (process.env.TORCHLETTE_DEBUG_COMPILED) {
    console.log(
      `[compiled] Executing: ${compiled.commands.length} cmds, ${compiled.slots.length} slots, ${compiled.results.length} results${compiled.allocBuffers ? " (planned buffers)" : ""}`,
    );
  }

  beginSharedEncoder();
  setActiveArena(arena);
  setArenaExternalInputBuffers(externalInputBuffers);
  // NOTE: Do NOT reset dispatch sequence here. The step-level beginStep() handles
  // the reset. Multiple plans within a step share the sequence counter — resetting
  // here would cause backward plan params to collide with forward plan params.

  try {
    // Phase 1: Pre-populate external + persistent slots
    for (let i = 0; i < compiled.slots.length; i++) {
      const src = compiled.slots[i];
      if (src.kind === "external") {
        const ref = planNodes[src.planNodeIndex].inputs[src.inputIndex];
        const storage = getInputStorage(ref, backend);
        slots[i] = gpuBuffer(storage.backendTensor);
      } else if (src.kind === "persistent") {
        slots[i] = src.buffer;
      }
      // "arena", "write", and "params" slots are populated below
    }

    // Phase 1b: Self-contained params buffers.
    // The compiled plan owns its params GPUBuffers — created once, cached on the
    // SlotSource, reused every step. Bypasses the global createParamsBuffer cache
    // entirely, so paramsSeqIndex is never advanced by the compiled plan.
    for (let i = 0; i < compiled.slots.length; i++) {
      const src = compiled.slots[i];
      if (src.kind === "params") {
        if (src.cachedBuffer) {
          slots[i] = src.cachedBuffer;
        } else {
          const buf = device.createBuffer({
            size: paramsBufferSizeClass(src.data.byteLength),
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(buf, 0, src.data);
          src.cachedBuffer = buf;
          slots[i] = buf;
        }
      }
    }

    // Phase 2: Execute GPU commands
    const cmds = compiled.commands;
    for (let ci = 0; ci < cmds.length; ci++) {
      const cmd = cmds[ci];
      switch (cmd.tag) {
        case TAG_ALLOC: {
          const inputBufs: GPUBuffer[] = [];
          for (let j = 0; j < cmd.inputSlots.length; j++) {
            inputBufs.push(slots[cmd.inputSlots[j]]);
          }
          // Planned-buffer path: bind the recorded buffer for this lifetime.
          // Safety: (1) first use this replay must not be owned by a live
          // tensor (a kept/user tensor would be clobbered — same check the
          // arena does via canRecycle); later lifetimes of the same buffer
          // within this replay are safe by queue order, exactly as recorded.
          // (2) never alias an external input (template-reuse conflict) or a
          // direct input of this op. On failure, fall back to dynamic
          // allocation for ALL of this buffer's lifetimes this replay.
          const planned = compiled.allocBuffers
            ? process.env.TORCHLETTE_PLANNED_BIND === "0"
              ? undefined
              : compiled.allocBuffers[cmd.slot]
            : undefined;
          if (planned) {
            if (!plannedSeen) {
              plannedSeen = new Set();
              plannedFallback = new Set();
            }
            let ok = !plannedFallback!.has(planned);
            if (ok && !plannedSeen.has(planned)) {
              plannedSeen.add(planned);
              if (
                bufferPool.isLive(planned) ||
                externalInputSet?.has(planned)
              ) {
                plannedFallback!.add(planned);
                ok = false;
              }
            }
            if (ok && inputBufs.some((b) => b === planned)) ok = false;
            if (ok) {
              trackSharedEncoderWrite(planned);
              slots[cmd.slot] = planned;
              plannedBindCount++;
              break;
            }
            plannedFallbackCount++;
          }
          slots[cmd.slot] =
            cmd.allocKind === 0
              ? resolveOutputBuffer(device, cmd.bytes, inputBufs)
              : allocateOutputBuffer(cmd.bytes);
          break;
        }
        case TAG_DISPATCH: {
          const bufs: GPUBuffer[] = new Array(cmd.bindings.length);
          for (let j = 0; j < cmd.bindings.length; j++) {
            bufs[j] = slots[cmd.bindings[j]];
            if (bufs[j] === undefined) {
              throw new Error(
                `[compiled] dispatch at cmd ${ci} binds slot ${cmd.bindings[j]} which was never populated (label=${cmd.label ?? "?"})`,
              );
            }
          }
          // Inline bind group cache — bypasses global sequence-indexed cache.
          // The normal path mixes cachedCreateBindGroup (advances dispatchIndex)
          // and profiledCreateBindGroup (doesn't), so the global sequence counter
          // is misaligned during compiled plan replay.
          let bg = cmd.cachedBindGroup;
          if (bg && cmd.cachedBuffers) {
            // Validate cached bind group: pipeline is fixed, check buffers
            let match = cmd.cachedBuffers.length === bufs.length;
            if (match) {
              for (let j = 0; j < bufs.length; j++) {
                if (cmd.cachedBuffers[j] !== bufs[j]) {
                  match = false;
                  break;
                }
              }
            }
            if (!match) bg = undefined;
          }
          if (!bg) {
            const entries: Array<{
              binding: number;
              resource: { buffer: GPUBuffer };
            }> = [];
            for (let j = 0; j < bufs.length; j++) {
              entries.push({ binding: j, resource: { buffer: bufs[j] } });
            }
            bg = profiledCreateBindGroup(device, {
              layout: cmd.pipeline.getBindGroupLayout(0),
              entries,
            });
            cmd.cachedBindGroup = bg;
            cmd.cachedBuffers = bufs.slice();
          }
          if (cmd.module !== undefined) setProfileModule(cmd.module);
          dispatchComputePass(
            cmd.pipeline,
            bg,
            cmd.gx,
            cmd.gy,
            cmd.gz,
            cmd.label,
          );
          break;
        }
        case TAG_COPY: {
          const encoder = getSharedEncoderInstance();
          if (encoder) {
            encoder.copyBufferToBuffer(
              slots[cmd.src],
              cmd.srcOffset,
              slots[cmd.dst],
              cmd.dstOffset,
              cmd.bytes,
            );
          }
          break;
        }
        case TAG_WRITE: {
          // Execute the data source node (tensorFromArray) to produce buffer data
          const writeNode = planNodes[cmd.nodeIndex];
          if (!writeNode.result) {
            const inputs = writeNode.inputs.map((ref) =>
              getInputStorage(ref, backend),
            );
            const backendInputs = inputs.map((s) => s.backendTensor);
            const resultOrPromise = executeOpSync(
              writeNode,
              backendInputs,
              backend,
            );
            const result: BackendTensor =
              resultOrPromise instanceof Promise
                ? await resultOrPromise
                : resultOrPromise;
            writeNode.result = createStorageHandle(writeNode.device, result);
          }
          slots[cmd.slot] = gpuBuffer(writeNode.result!.backendTensor);
          break;
        }
        case TAG_CLEAR: {
          const encoder = getSharedEncoderInstance();
          if (encoder) {
            encoder.clearBuffer(slots[cmd.slot], 0, cmd.bytes);
          }
          break;
        }
        case TAG_UNIFORM: {
          // Re-derive volatile uniform data from the CURRENT step's node
          // payload (fresh per step) and re-write the config buffer. Stream
          // order matches the lowered path's writeBuffer-at-dispatch-prep:
          // queue.writeBuffer is ordered before any later submit, and flush
          // boundaries (TAG_BARRIER) are replayed at the recorded positions.
          const node = planNodes[cmd.nodeIndex];
          if (!node) {
            throw new Error(
              `[compiled] TAG_UNIFORM nodeIndex ${cmd.nodeIndex} out of range (plan has ${planNodes.length} nodes)`,
            );
          }
          device.queue.writeBuffer(slots[cmd.slot], 0, cmd.pack(node));
          break;
        }
        case TAG_BARRIER: {
          flushSharedEncoder();
          flushBufferPool();
          break;
        }
      }
    }

    if (process.env.TORCHLETTE_DEBUG_COMPILED && compiled.allocBuffers) {
      console.log(
        `[compiled] planned binds=${plannedBindCount} fallbacks=${plannedFallbackCount}`,
      );
    }

    // Restore global sequence counters to where the normal path would have
    // left them. Subsequent non-compiled plans see correct counter positions.
    if (compiled.endCounters) {
      setDispatchSequenceCounters(
        compiled.endCounters.dispatch,
        compiled.endCounters.params,
        compiled.endCounters.output,
      );
    }

    // Phase 3: Harvest results — assign to plan nodes for downstream plans
    for (const r of compiled.results) {
      const node = planNodes[r.nodeIndex];
      const tensor = createTensor(
        r.shape,
        slots[r.slot],
        r.strides,
        r.offset,
        r.dtype,
      );
      const sh = createStorageHandle(node.device, tensor);
      if (r.outputIndex === 0) {
        // Primary output
        if (!node.result) {
          node.result = sh;
        }
      }
      // Multi-output: populate node.results array
      if (r.outputIndex > 0 || (node.results && node.results.length > 0)) {
        if (!node.results) {
          node.results = [node.result!];
        }
        node.results[r.outputIndex] = sh;
      }
    }
  } finally {
    clearActiveArena();
    clearArenaExternalInputBuffers();
    endSharedEncoder();
  }
}
