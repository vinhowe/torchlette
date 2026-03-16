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
  GPUBuffer,
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

export type GpuCommand =
  | AllocCommand
  | DispatchCommand
  | CopyCommand
  | WriteCommand
  | BarrierCommand;

// Numeric tags for fast switch in hot loop
export const TAG_ALLOC = 0 as const;
export const TAG_DISPATCH = 1 as const;
export const TAG_COPY = 2 as const;
export const TAG_WRITE = 3 as const;
export const TAG_BARRIER = 4 as const;

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
}

// ============================================================================
// Compilation: build CompiledPlan from recorded dispatches
// ============================================================================

import type { BufferArena } from "../backend/webgpu/buffer-arena";
/** Recorded dispatch entry — the subset of fields needed by the compiled plan. */
export interface RecordedDispatch {
  pipeline: GPUComputePipeline;
  bindGroup: import("../backend/webgpu/gpu-types").GPUBindGroup;
  workgroupsX: number;
  workgroupsY: number;
  workgroupsZ: number;
  /** GPUBuffers referenced by this bind group. */
  buffers?: GPUBuffer[];
  /** Op label for GPU timestamp profiling. */
  label?: string;
  /** Module label for per-module GPU breakdown. */
  module?: string;
}

import { asGPUTensor } from "../backend/webgpu/gpu-types";
import { contiguousStrides } from "../core/shape";
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

  const commands: GpuCommand[] = [];

  for (const entry of commandLog) {
    switch (entry.kind) {
      case "dispatch": {
        const d = entry.dispatch;
        if (!d.buffers || d.buffers.length === 0) {
          // Dispatch without recorded buffers — skip (shouldn't happen with arena)
          continue;
        }
        const bindings: Slot[] = [];
        for (const buf of d.buffers) {
          let slot = bufferToSlot.get(buf);
          if (slot === undefined) {
            // Auto-assign as persistent: cached/singleton buffer (attention config, k-split temp, etc.)
            slot = slotSources.length;
            bufferToSlot.set(buf, slot);
            slotSources.push({ kind: "persistent", buffer: buf });
          }
          bindings.push(slot);
        }
        commands.push({
          tag: TAG_DISPATCH,
          pipeline: d.pipeline,
          bindings,
          gx: d.workgroupsX,
          gy: d.workgroupsY,
          gz: d.workgroupsZ,
        });
        break;
      }
      case "alloc": {
        const slot = bufferToSlot.get(entry.buffer);
        if (slot === undefined) {
          if (process.env.TORCHLETTE_DEBUG_COMPILED) {
            console.log(
              `[compiled] FAIL: alloc buffer unmapped, size=${entry.bytes}, commands so far=${commands.length}`,
            );
          }
          return { commands: [], slots: [], results: [], valid: false };
        }
        const inputSlots: Slot[] = [];
        for (const ib of entry.inputBuffers) {
          const is_ = bufferToSlot.get(ib);
          if (is_ !== undefined) inputSlots.push(is_);
        }
        commands.push({
          tag: TAG_ALLOC,
          slot,
          bytes: entry.bytes,
          allocKind: entry.allocKind,
          inputSlots,
        });
        break;
      }
      case "copy": {
        let srcSlot = bufferToSlot.get(entry.copy.src);
        let dstSlot = bufferToSlot.get(entry.copy.dst);
        if (srcSlot === undefined) {
          srcSlot = slotSources.length;
          bufferToSlot.set(entry.copy.src, srcSlot);
          slotSources.push({ kind: "persistent", buffer: entry.copy.src });
        }
        if (dstSlot === undefined) {
          dstSlot = slotSources.length;
          bufferToSlot.set(entry.copy.dst, dstSlot);
          slotSources.push({ kind: "persistent", buffer: entry.copy.dst });
        }
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
        const slot = bufferToSlot.get(entry.buffer);
        if (slot === undefined) {
          return { commands: [], slots: [], results: [], valid: false };
        }
        commands.push({
          tag: TAG_WRITE,
          slot,
          nodeIndex: entry.nodeIndex,
        });
        break;
      }
      case "barrier": {
        commands.push({ tag: TAG_BARRIER });
        break;
      }
    }
  }

  if (process.env.TORCHLETTE_DEBUG_COMPILED) {
    const dispatchCount = commands.filter((c) => c.tag === TAG_DISPATCH).length;
    const allocCount = commands.filter((c) => c.tag === TAG_ALLOC).length;
    const barrierCount = commands.filter((c) => c.tag === TAG_BARRIER).length;
    const writeCount = commands.filter((c) => c.tag === TAG_WRITE).length;
    const copyCount = commands.filter((c) => c.tag === TAG_COPY).length;
    const slotKinds: Record<string, number> = {};
    for (const s of slotSources) {
      slotKinds[s.kind] = (slotKinds[s.kind] || 0) + 1;
    }
    console.log(
      `[compiled] Built: ${commands.length} cmds (${dispatchCount} dispatch, ${allocCount} alloc, ${barrierCount} barrier, ${writeCount} write, ${copyCount} copy), ${slotSources.length} slots (${JSON.stringify(slotKinds)}), ${nodeResults.length} results`,
    );
  }

  return {
    commands,
    slots: slotSources,
    results: nodeResults,
    valid: true,
  };
}

// ============================================================================
// Command log entry types (used during recording)
// ============================================================================

export type CommandLogEntry =
  | { kind: "dispatch"; dispatch: RecordedDispatch }
  | {
      kind: "alloc";
      buffer: GPUBuffer;
      bytes: number;
      allocKind: 0 | 1;
      inputBuffers: GPUBuffer[];
    }
  | { kind: "copy"; copy: RecordedCopy }
  | { kind: "write"; buffer: GPUBuffer; nodeIndex: number }
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
  setCompilationRecording(true);
  return {
    commandLog: activeCommandLog,
    bufferToSlot: activeBufferToSlot,
    slotSources: activeSlotSources,
  };
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
  activeCommandLog?.push({ kind: "dispatch", dispatch });
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
  // Assign arena slot for this output buffer
  assignSlot(buffer, { kind: "arena" });
  activeCommandLog.push({
    kind: "alloc",
    buffer,
    bytes,
    allocKind,
    inputBuffers,
  });
}

/** Record a buffer-to-buffer copy. Called from shared encoder wrapper. */
export function recordCopy(copy: RecordedCopy): void {
  activeCommandLog?.push({ kind: "copy", copy });
}

/** Record a host→device write (tensorFromArray). Called from executor. */
export function recordWrite(buffer: GPUBuffer, nodeIndex: number): void {
  if (!activeCommandLog) return;
  assignSlot(buffer, { kind: "write" });
  activeCommandLog.push({ kind: "write", buffer, nodeIndex });
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
} from "../backend/webgpu/webgpu-state";
import type { StorageHandle } from "../graph/types";
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

  if (process.env.TORCHLETTE_DEBUG_COMPILED) {
    console.log(
      `[compiled] Executing: ${compiled.commands.length} cmds, ${compiled.slots.length} slots, ${compiled.results.length} results`,
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
          dispatchComputePass(cmd.pipeline, bg, cmd.gx, cmd.gy, cmd.gz);
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
        case TAG_BARRIER: {
          flushSharedEncoder();
          flushBufferPool();
          break;
        }
      }
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
