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

import { ENV } from "../core/env";
import { TAPE_PROFILE, tpAdd } from "../core/tape-profile";
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
  /** Plan-node index that produced this command (segment-aligned stream
   *  diffing, stage-4 phase 2). Not used by replay. */
  nodeIndex?: number;
}

export interface DispatchCommand {
  tag: 1; // "dispatch"
  pipeline: GPUComputePipeline;
  bindings: Slot[];
  /** Subrange per binding (null = whole buffer). Present only when some
   *  binding is a subrange (chunked dispatches over >maxBindingSize tensors). */
  bindingRanges?: Array<{ offset: number; size: number } | null>;
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
  /** Plan-node index (segment-aligned stream diffing). Not used by replay. */
  nodeIndex?: number;
}

export interface CopyCommand {
  tag: 2; // "copy"
  src: Slot;
  dst: Slot;
  srcOffset: number;
  dstOffset: number;
  bytes: number;
  /** Plan-node index (segment-aligned stream diffing). Not used by replay. */
  nodeIndex?: number;
}

export interface WriteCommand {
  tag: 3; // "write" — host→device (tensorFromArray)
  slot: Slot;
  nodeIndex: number;
  /**
   * Plan-owned STABLE upload buffer (f32 fast path). The legacy replay
   * re-executed the tensorFromArray node each step, allocating a fresh
   * pool/arena buffer — the changed identity invalidated the cached bind
   * group of EVERY consumer dispatch (Qwen3 decode: 141 createBindGroup
   * calls/token from 4 upload slots). Replays instead queue.writeBuffer the
   * fresh payload into this buffer (queue-ordered: lands before the step's
   * encoder submits, after the previous step's fence), so consumer bind
   * groups stay byte-identical across replays. Destroyed (deferred) at plan
   * teardown.
   */
  stableBuf?: GPUBuffer;
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
  /** Plan-node index (segment-aligned stream diffing). Not used by replay. */
  nodeIndex?: number;
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

/** TAG_WRITE stable-buffer fast path: only for small per-step host uploads. */
const STABLE_WRITE_MAX_BYTES = 1 << 20; // 1MB

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
  /** Stage-4 memory planner output: alloc-slot → registry entry index.
   *  Entries live in the step-scoped shared registry (cross-plan packing,
   *  phase 1.5); this plan co-owns every entry listed in plannerEntries. */
  plannerAssignment?: Map<number, number>;
  /** Registry entry indices this plan co-owns (released at teardown). */
  plannerEntries?: number[];
  /** Registry generation at build; a mismatch at replay/teardown means the
   *  registry was reset (engine-instance boundary) — indices are stale, all
   *  allocs fall back dynamically and the plan invalidates for re-record. */
  plannerGen?: number;
  /** Storage ids the harvest rcRetained as VIEW bases on the LAST replay.
   *  Released at the start of the next harvest so view-base retains don't
   *  accumulate per replay (the clip+optimizer storage-handle leak). */
  _viewBaseRetains?: number[];
}

// ============================================================================
// Compilation: build CompiledPlan from recorded dispatches
// ============================================================================

import {
  arenaLivenessEnabled,
  compiledPlannedEnabled,
  type BufferArena,
} from "../backend/webgpu/buffer-arena";
import { bufferPool } from "../backend/webgpu/buffer-pool";
import { gpuMemoryTracker } from "../backend/webgpu/memory-tracker";
import { pinnedBufferSet } from "../backend/webgpu/webgpu-state";
/** Recorded dispatch entry — the subset of fields needed by the compiled plan. */
export interface RecordedDispatch {
  pipeline: GPUComputePipeline;
  bindGroup: import("../backend/webgpu/gpu-types").GPUBindGroup;
  workgroupsX: number;
  workgroupsY: number;
  workgroupsZ: number;
  /** Bind-group entries (buffer + optional subrange) referenced, in order. */
  buffers?: CapturedBindEntry[];
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

import type { LazyIRNode, StorageHandle } from "../graph/types";
import { getInputStorage } from "./op-dispatch";
import { planMemory, PlannerRegistry } from "./memory-planner";

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
    if (ENV.TORCHLETTE_DEBUG_COMPILED) {
      console.log(`[compiled] FAIL: ${recordingInvalidReason}`);
    }
    recordingInvalidReason = null;
    return { commands: [], slots: [], results: [], valid: false };
  }

  const commands: GpuCommand[] = [];

  // The stage-4 memory planner (THE compiled-replay buffer assignment under
  // the liveness arena since phase 1.5) runs in finalizeCompiledPlan over the
  // final command stream — see the note there and compiledPlannedEnabled.

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
        let bindingRanges: Array<{ offset: number; size: number } | null> | undefined;
        for (let i = 0; i < d.buffers.length; i++) {
          const entry = d.buffers[i];
          const recorded = d.slots?.[i] ?? -1;
          if (recorded < 0 && ENV.TORCHLETTE_DEBUG_PERSISTENT) {
            const had = bufferToSlot.has(entry.buffer);
            if (!had) {
              console.log(
                `[persistent-slot] NEW slot ${slotSources.length} buf=${dbgBufId(entry.buffer)} for unmapped buffer: dispatch label=${d.label ?? "?"} bindingIdx=${i} size=${(entry.buffer as { size?: number }).size ?? "?"}`,
              );
            }
          }
          bindings.push(recorded >= 0 ? recorded : persistentSlot(entry.buffer));
          if (entry.offset !== undefined || entry.size !== undefined) {
            bindingRanges ??= new Array(d.buffers.length).fill(null);
            bindingRanges[i] = {
              offset: entry.offset ?? 0,
              size: entry.size ?? 0,
            };
          }
        }
        commands.push({
          tag: TAG_DISPATCH,
          pipeline: d.pipeline,
          bindings,
          bindingRanges,
          gx: d.workgroupsX,
          gy: d.workgroupsY,
          gz: d.workgroupsZ,
          label: d.label,
          module: d.module,
          nodeIndex: entry.nodeIndex,
        });
        break;
      }
      case "alloc": {
        const slot = entry.slot;
        if (slot < 0) {
          if (ENV.TORCHLETTE_DEBUG_COMPILED) {
            console.log(
              `[compiled] FAIL: alloc buffer unmapped, size=${entry.bytes}, commands so far=${commands.length}`,
            );
          }
          return { commands: [], slots: [], results: [], valid: false };
        }
        commands.push({
          tag: TAG_ALLOC,
          slot,
          bytes: entry.bytes,
          allocKind: entry.allocKind,
          inputSlots: entry.inputSlots,
          nodeIndex: entry.nodeIndex,
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
          nodeIndex: entry.nodeIndex,
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
          if (ENV.TORCHLETTE_DEBUG_COMPILED) {
            console.log(
              `[compiled] FAIL: clear buffer unmapped, size=${entry.bytes}, commands so far=${commands.length}`,
            );
          }
          return { commands: [], slots: [], results: [], valid: false };
        }
        commands.push({
          tag: TAG_CLEAR,
          slot: entry.slot,
          bytes: entry.bytes,
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

  return finalizeCompiledPlan(commands, slotSources, nodeResults);
}

/**
 * Stage-4 phase-4 cutover: build a CompiledPlan directly from a GENERATED
 * stream (stream-generate.ts) instead of the recorded log. The generator
 * already emits final GpuCommand[] + SlotSource[] (slot assignment done) and a
 * node→slot map, so this path skips the raw-log → command transform entirely
 * and just runs the SAME planner + assembly tail. The generated stream is
 * proven byte-identical to the recording (modulo slot bijection) by the
 * differential gate, so the resulting plan is equivalent; callers gate on
 * gen.fullyCovered (every action generatable) before using this. nodeResults
 * must carry the GENERATED slot for each result node (built from gen.nodeSlot
 * / gen.nodeSlotExtra, metadata from the live node result).
 */
export function buildCompiledPlanFromGenerated(input: {
  commands: GpuCommand[];
  slots: SlotSource[];
  nodeResults: NodeResult[];
}): CompiledPlan {
  return finalizeCompiledPlan(input.commands, input.slots, input.nodeResults);
}

/**
 * Shared tail of plan construction: the optional debug dump, the stage-4
 * memory planner (alloc-slot → registry-entry assignment, derived from
 * lifetimes over the final command stream with cross-plan temp packing; throws
 * on structural overlap — a planner bug is a build failure, not a replay
 * corruption), and final plan assembly + registry ownership. Used by both the
 * recorded build (buildCompiledPlan) and the generated build
 * (buildCompiledPlanFromGenerated) — the planner sees only the command stream
 * + result slots, so it is identical for both.
 */
function finalizeCompiledPlan(
  commands: GpuCommand[],
  slotSources: SlotSource[],
  nodeResults: NodeResult[],
): CompiledPlan {
  const usePlanner = compiledPlannedEnabled();

  if (ENV.TORCHLETTE_DEBUG_COMPILED) {
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

  let plannerAssignment: Map<number, number> | undefined;
  let plannerEntries: number[] | undefined;
  if (usePlanner) {
    const resultSlots = new Set<number>(nodeResults.map((r) => r.slot));
    const memPlan = planMemory(commands, plannerRegistry, resultSlots);
    plannerAssignment = memPlan.assignment;
    plannerEntries = memPlan.entries;
    if (ENV.TORCHLETTE_DEBUG_COMPILED) {
      console.log(
        `[memory-planner] ${memPlan.assignment.size} alloc slots → ${memPlan.entries.length} entries (${(memPlan.newBytes / 1e6).toFixed(1)}MB new, ${(memPlan.reusedBytes / 1e6).toFixed(1)}MB shared cross-plan)`,
      );
    }
  }

  const plan: CompiledPlan = {
    commands,
    slots: slotSources,
    results: nodeResults,
    valid: true,
    plannerAssignment,
    plannerEntries,
    plannerGen: plannerEntries ? plannerRegistry.generation : undefined,
  };
  if (plannerEntries) {
    for (const idx of plannerEntries) {
      plannerRegistry.entries[idx].owners.add(plan);
    }
  }
  return plan;
}

/**
 * Step-scoped shared planner-buffer registry (stage-4 phase 1.5). One per
 * process; engine-instance boundaries reset it via resetPlannerRegistry()
 * alongside the template cache (instance boundaries are cache boundaries —
 * stale cross-instance entries were the class behind the "reshape issue").
 */
const plannerRegistry = new PlannerRegistry();

/** Reset the planner registry, destroying any materialized buffers via the
 *  fence-gated path. Plans built against the old generation fall back to
 *  dynamic allocation at their next replay and self-invalidate. */
export function resetPlannerRegistry(): void {
  for (const e of plannerRegistry.reset()) {
    if (e.buffer) {
      pinnedBufferSet.delete(e.buffer);
      bufferPool.deferredDestroy(
        e.buffer,
        (e.buffer as { size?: number }).size ?? 0,
      );
      e.buffer = undefined;
    }
  }
}

/** Debug: stable small ids for GPUBuffer identity in logs. */
const _dbgBufIds = new WeakMap<GPUBuffer, number>();
let _dbgBufNext = 1;
export function dbgBufId(buf: GPUBuffer | undefined): string {
  if (!buf) return "none";
  let id = _dbgBufIds.get(buf);
  if (id === undefined) {
    id = _dbgBufNext++;
    _dbgBufIds.set(buf, id);
  }
  return `b${id}`;
}

/** Debug: track destroyed GPUBuffers via prototype patch (debug env only). */
const _dbgDestroyed = new WeakSet<GPUBuffer>();
let _dbgDestroyPatched = false;
function dbgPatchDestroy(sample: GPUBuffer): void {
  if (_dbgDestroyPatched) return;
  _dbgDestroyPatched = true;
  const proto = Object.getPrototypeOf(sample) as {
    destroy: (this: GPUBuffer) => void;
  };
  const orig = proto.destroy;
  proto.destroy = function (this: GPUBuffer) {
    _dbgDestroyed.add(this);
    const minMb = ENV.TORCHLETTE_DEBUG_DESTROY_STACK_MB;
    if (
      minMb &&
      ((this as unknown as { size?: number }).size ?? 0) >
        parseFloat(minMb) * 1024 * 1024
    ) {
      const stack = new Error().stack
        ?.split("\n")
        .slice(2, 8)
        .map((l) => l.trim())
        .join(" <- ");
      console.log(
        `[destroy-stack] buf=${dbgBufId(this)} size=${(((this as unknown as { size?: number }).size ?? 0) / 1e6).toFixed(1)}MB pinned=${pinnedBufferSet.has(this)} ${stack}`,
      );
    }
    return orig.call(this);
  };
}
export function dbgIsDestroyed(buf: GPUBuffer): boolean {
  return _dbgDestroyed.has(buf);
}

/**
 * Release a plan's planner-registry entries. Called when the compiled plan
 * is invalidated or its template evicted.
 */
export function destroyCompiledPlanBuffers(compiled: CompiledPlan): void {
  // Release the last replay's view-base retains (normally released at the next
  // harvest, which won't happen once the plan is gone). Safe no-op if the base
  // storages were already freed (monotonic ids, rcRelease returns -1).
  if (compiled._viewBaseRetains) {
    for (const baseId of compiled._viewBaseRetains) {
      rcRelease(baseId, "view.reharvest.teardown");
    }
    compiled._viewBaseRetains = undefined;
  }
  // Release co-owned registry entries (cross-plan packing). An entry's
  // buffer dies with its LAST owner; the entry record itself stays and is
  // re-listed for future plans (the buffer rematerializes on demand).
  // Generation guard: after a registry reset the indices point into a NEW
  // entries array — the reset already destroyed the old buffers.
  if (compiled.plannerEntries) {
    if (compiled.plannerGen === plannerRegistry.generation) {
      for (const idx of compiled.plannerEntries) {
        const e = plannerRegistry.entries[idx];
        e.owners.delete(compiled);
        if (e.owners.size === 0) {
          if (e.buffer) {
            pinnedBufferSet.delete(e.buffer);
            // DEFERRED destruction, never immediate: teardown fires MID-STEP
            // (staleness gates, eviction) while the step encoder holds
            // encoded passes binding these buffers.
            bufferPool.deferredDestroy(
              e.buffer,
              (e.buffer as { size?: number }).size ?? 0,
            );
            e.buffer = undefined;
          }
          plannerRegistry.relist(idx);
        }
      }
    }
    compiled.plannerEntries = undefined;
    compiled.plannerAssignment = undefined;
    compiled.plannerGen = undefined;
  }
  // Destroy plan-owned stable upload buffers (TAG_WRITE fast path). Unpin
  // first (deferredDestroy is pin-guarded), then defer: teardown can fire
  // mid-step with encoded-but-unsubmitted passes binding them.
  for (const cmd of compiled.commands) {
    if (cmd.tag === TAG_WRITE && cmd.stableBuf) {
      pinnedBufferSet.delete(cmd.stableBuf);
      bufferPool.deferredDestroy(
        cmd.stableBuf,
        (cmd.stableBuf as { size?: number }).size ?? 0,
      );
      cmd.stableBuf = undefined;
    }
  }
}

// ============================================================================
// Command log entry types (used during recording)
// ============================================================================

// Slots are resolved at RECORD time (-1 / undefined = unmapped at record
// time → assigned a persistent slot at build). See RecordedDispatch.slots
// for why temporal resolution matters under buffer-lifetime reuse.
export type CommandLogEntry =
  | { kind: "dispatch"; dispatch: RecordedDispatch; nodeIndex?: number }
  | {
      kind: "alloc";
      buffer: GPUBuffer;
      bytes: number;
      allocKind: 0 | 1;
      inputBuffers: GPUBuffer[];
      slot: Slot;
      inputSlots: Slot[];
      nodeIndex?: number;
    }
  | { kind: "copy"; copy: RecordedCopy; srcSlot: Slot; dstSlot: Slot; nodeIndex?: number }
  | { kind: "write"; buffer: GPUBuffer; nodeIndex: number; slot: Slot }
  | { kind: "clear"; buffer: GPUBuffer; bytes: number; slot: Slot; nodeIndex?: number }
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
    if (ENV.TORCHLETTE_DEBUG_COMPILED) {
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

/** A captured bind-group entry: buffer plus optional SUB-RANGE. Chunked
 *  dispatches (embedding-grad where/cast over >maxStorageBufferBindingSize
 *  tensors, chunked adam) bind buffer subranges; recording only the flat
 *  buffer list loses the offsets — invisible while replays reuse the
 *  recorded bind-group OBJECT (pinned buffers match), fatal the moment a
 *  replay must REBUILD bind groups against different buffers (the stage-4
 *  memory planner assigns fresh ones). */
export interface CapturedBindEntry {
  buffer: GPUBuffer;
  offset?: number;
  size?: number;
}

/** Last bind group's entry list — captured during compilation recording. */
let lastBindGroupBuffers: CapturedBindEntry[] | null = null;

/** Set the last bind group's buffer list. Called from bind-group-cache.ts. */
export function setLastBindGroupBuffers(
  bufs: GPUBuffer[] | CapturedBindEntry[] | null,
): void {
  if (bufs === null) {
    lastBindGroupBuffers = null;
    return;
  }
  lastBindGroupBuffers = (bufs as Array<GPUBuffer | CapturedBindEntry>).map(
    (b) => (b && typeof b === "object" && "buffer" in (b as object)
      ? (b as CapturedBindEntry)
      : { buffer: b as GPUBuffer }),
  );
}

/** Get and clear the last bind group's buffer list. Called from dispatch.ts. */
export function getAndClearLastBindGroupBuffers():
  | CapturedBindEntry[]
  | undefined {
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
      slots[i] = activeBufferToSlot.get(dispatch.buffers[i].buffer) ?? -1;
    }
    dispatch.slots = slots;
  }
  activeCommandLog.push({ kind: "dispatch", dispatch, nodeIndex: recordingNodeIndex });
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
    nodeIndex: recordingNodeIndex,
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
    nodeIndex: recordingNodeIndex,
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
    nodeIndex: recordingNodeIndex,
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
  incrementSharedEncoderPassCount,
} from "../backend/webgpu/shared-encoder";
import { gpuTimestampsActive } from "../backend/webgpu/profiler";
import { createTensor } from "../backend/webgpu/tensor";
import {
  paramsBufferSizeClass,
  requireContext,
  trackSharedEncoderWrite,
} from "../backend/webgpu/webgpu-state";
import { createStorageHandle } from "../graph/node-factory";
import { rcRelease, rcRetain } from "../graph/refcount";
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
  /** Lazily evaluated once per replay: registry reset since build? */
  let plannerGenStale: boolean | undefined;

  if (ENV.TORCHLETTE_DEBUG_COMPILED) {
    console.log(
      `[compiled] Executing: ${compiled.commands.length} cmds, ${compiled.slots.length} slots, ${compiled.results.length} results${compiled.plannerAssignment ? " (planner)" : ""}`,
    );
  }

  beginSharedEncoder();
  setActiveArena(arena);
  setArenaExternalInputBuffers(externalInputBuffers);
  // NOTE: Do NOT reset dispatch sequence here. The step-level beginStep() handles
  // the reset. Multiple plans within a step share the sequence counter — resetting
  // here would cause backward plan params to collide with forward plan params.

  // Batched compute pass: consecutive TAG_DISPATCH commands encode into ONE
  // compute pass instead of one pass each (WebGPU synchronizes storage access
  // between dispatches in a pass — each dispatch is its own usage scope), which
  // removes ~2 Dawn beginComputePass/end calls per dispatch (~700/step on the
  // Qwen3 decode plan). The pass MUST be closed before any encoder-level
  // command (copy/clear), before flushes (barriers), and before commands that
  // can encode their own passes (legacy TAG_WRITE). When GPU timestamps are
  // enabled the per-dispatch path is kept — timestampWrites are per-pass.
  let openPass: ReturnType<GPUCommandEncoder["beginComputePass"]> | null = null;
  let openPassEncoder: GPUCommandEncoder | null = null;
  const closeOpenPass = () => {
    if (openPass) {
      openPass.end();
      openPass = null;
      openPassEncoder = null;
    }
  };

  // [tape-1a] G0 replay-loop attribution (src/core/tape-profile.ts; sunset:
  // 1c): Dawn API time inside the loop accumulates into "replay-dawn" /
  // "replay-bindgroup" / "replay-barrier"; slot population and result harvest
  // get their own keys so the pure JS iteration cost is derivable as
  // replay-total minus these.
  try {
    const tpP0 = TAPE_PROFILE ? performance.now() : 0;
    // Phase 1: Pre-populate external + persistent slots
    const dbgSlots = ENV.TORCHLETTE_DEBUG_SLOTS
      ? ENV.TORCHLETTE_DEBUG_SLOTS.split(",").map(Number)
      : null;
    for (let i = 0; i < compiled.slots.length; i++) {
      const src = compiled.slots[i];
      if (src.kind === "external") {
        const ref = planNodes[src.planNodeIndex].inputs[src.inputIndex];
        const storage = getInputStorage(ref, backend);
        slots[i] = gpuBuffer(storage.backendTensor);
        if (dbgSlots?.includes(i)) {
          const refDesc =
            ref.kind === "materialized"
              ? `materialized storage=${ref.storage.id}`
              : ref.kind === "scalar"
                ? "scalar"
                : `pending node=${(ref as { node: { id: number; op: string } }).node.id}:${(ref as { node: { op: string } }).node.op} oi=${(ref as { outputIndex?: number }).outputIndex ?? 0}`;
          console.log(
            `[slot-src] slot ${i} EXTERNAL via planNodes[${src.planNodeIndex}].inputs[${src.inputIndex}] (${refDesc}) -> buf=${dbgBufId(slots[i])}`,
          );
        }
      } else if (src.kind === "persistent") {
        slots[i] = src.buffer;
        if (dbgSlots?.includes(i)) {
          console.log(
            `[slot-src] slot ${i} PERSISTENT buf=${dbgBufId(slots[i])}`,
          );
        }
      } else if (dbgSlots?.includes(i)) {
        console.log(`[slot-src] slot ${i} kind=${src.kind}`);
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

    if (TAPE_PROFILE) tpAdd("replay-slots", performance.now() - tpP0);

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
          // Stage-4 planner path: bind the registry entry's shared buffer
          // for this slot (materialized lazily, destroyed when the entry's
          // last co-owning plan is torn down). No liveness/aliasing checks
          // needed at replay: the per-plan assignment is audited at build
          // (no overlapping lifetimes share an entry), cross-plan sharing
          // covers only intra-plan-dead temps (safe under the strictly
          // sequential plan execution within a step, in any order), and
          // result entries are exclusive to their plan.
          if (compiled.plannerAssignment) {
            if (plannerGenStale === undefined) {
              plannerGenStale =
                compiled.plannerGen !== plannerRegistry.generation;
              if (plannerGenStale) {
                // Registry was reset under this plan (engine-instance
                // boundary): entry indices are meaningless. Fall back to
                // dynamic allocation for the whole replay and re-record.
                console.warn(
                  "[compiled-plan] planner registry generation changed — dynamic-alloc fallback this step; invalidating plan for re-record",
                );
                compiled.valid = false;
              }
            }
            const entryIdx = plannerGenStale
              ? undefined
              : compiled.plannerAssignment.get(cmd.slot);
            if (entryIdx !== undefined) {
              const entry = plannerRegistry.entries[entryIdx];
              let pbuf = entry.buffer;
              if (!pbuf) {
                pbuf = device.createBuffer({
                  size: entry.bytes,
                  usage:
                    GPUBufferUsage.STORAGE |
                    GPUBufferUsage.COPY_SRC |
                    GPUBufferUsage.COPY_DST,
                });
                gpuMemoryTracker.trackAllocation(pbuf, entry.bytes);
                entry.buffer = pbuf;
              }
              // Pin: harvest wraps pinned buffers NON-owning (the registry
              // owns them; entry teardown unpins + destroys), and every
              // destroy/release path skips pinned buffers — result buffers
              // survive until plan teardown regardless of downstream
              // storage lifecycles.
              pinnedBufferSet.add(pbuf);
              trackSharedEncoderWrite(pbuf);
              slots[cmd.slot] = pbuf;
              break;
            }
            if (plannerGenStale) {
              slots[cmd.slot] =
                cmd.allocKind === 0
                  ? resolveOutputBuffer(device, cmd.bytes, inputBufs)
                  : allocateOutputBuffer(cmd.bytes);
              break;
            }
          }
          // No planner assignment (legacy arena mode, or a slot outside
          // the assignment): dynamic allocation through the arena/pool.
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
          if (ENV.TORCHLETTE_DEBUG_DESTROYED) {
            dbgPatchDestroy(bufs[0]);
            for (let j = 0; j < bufs.length; j++) {
              if (_dbgDestroyed.has(bufs[j])) {
                const src = compiled.slots[cmd.bindings[j]];
                let refDesc = "";
                if (src?.kind === "external") {
                  const ref =
                    planNodes[src.planNodeIndex]?.inputs[src.inputIndex];
                  refDesc =
                    ref?.kind === "materialized"
                      ? ` ref=materialized storage=${ref.storage.id}`
                      : ref?.kind === "scalar"
                        ? " ref=scalar"
                        : ref
                          ? ` ref=pending node=${(ref as { node: { id: number; op: string } }).node.id}:${(ref as { node: { op: string } }).node.op} oi=${(ref as { outputIndex?: number }).outputIndex ?? 0}`
                          : " ref=MISSING";
                }
                console.log(
                  `[destroyed-bind] cmd=${ci} label=${cmd.label ?? "?"} bindingIdx=${j} slot=${cmd.bindings[j]} kind=${src?.kind}${refDesc} buf=${dbgBufId(bufs[j])} IS DESTROYED`,
                );
              }
            }
          }
          if (
            ENV.TORCHLETTE_DEBUG_SLOT &&
            cmd.bindings.includes(
              parseInt(ENV.TORCHLETTE_DEBUG_SLOT, 10),
            )
          ) {
            const s = parseInt(ENV.TORCHLETTE_DEBUG_SLOT, 10);
            console.log(
              `[replay-dispatch] cmd=${ci} label=${cmd.label ?? "?"} binds slot ${s} at idx=${cmd.bindings.indexOf(s)} buf=${dbgBufId(slots[s])} (bindings=${cmd.bindings.join(",")})`,
            );
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
                  if (ENV.TORCHLETTE_DEBUG_BINDMISS) {
                    const src = compiled.slots[cmd.bindings[j]];
                    console.log(
                      `[bindmiss] cmd=${ci} label=${cmd.label ?? "?"} bindingIdx=${j} slot=${cmd.bindings[j]} slotKind=${src?.kind}`,
                    );
                  }
                  break;
                }
              }
            }
            if (!match) bg = undefined;
          }
          if (!bg) {
            const tpB0 = TAPE_PROFILE ? performance.now() : 0;
            const entries: Array<{
              binding: number;
              resource: { buffer: GPUBuffer; offset?: number; size?: number };
            }> = [];
            for (let j = 0; j < bufs.length; j++) {
              const range = cmd.bindingRanges?.[j];
              entries.push({
                binding: j,
                resource: range
                  ? { buffer: bufs[j], offset: range.offset, size: range.size || undefined }
                  : { buffer: bufs[j] },
              });
            }
            bg = profiledCreateBindGroup(device, {
              layout: cmd.pipeline.getBindGroupLayout(0),
              entries,
            });
            cmd.cachedBindGroup = bg;
            cmd.cachedBuffers = bufs.slice();
            if (TAPE_PROFILE) tpAdd("replay-bindgroup", performance.now() - tpB0);
          }
          if (cmd.module !== undefined) setProfileModule(cmd.module);
          const enc = getSharedEncoderInstance();
          const tpD0 = TAPE_PROFILE ? performance.now() : 0;
          if (!enc || gpuTimestampsActive()) {
            // Timestamp collection needs per-pass timestampWrites; no shared
            // encoder means dispatchComputePass manages its own encoder.
            closeOpenPass();
            dispatchComputePass(
              cmd.pipeline,
              bg,
              cmd.gx,
              cmd.gy,
              cmd.gz,
              cmd.label,
            );
          } else {
            if (!openPass || openPassEncoder !== enc) {
              closeOpenPass();
              openPass = enc.beginComputePass();
              openPassEncoder = enc;
            }
            openPass.setPipeline(cmd.pipeline);
            openPass.setBindGroup(0, bg as GPUBindGroup);
            openPass.dispatchWorkgroups(cmd.gx, cmd.gy, cmd.gz);
            incrementSharedEncoderPassCount();
          }
          if (TAPE_PROFILE) tpAdd("replay-dawn", performance.now() - tpD0);
          break;
        }
        case TAG_COPY: {
          closeOpenPass(); // copy is an encoder-level command
          const encoder = getSharedEncoderInstance();
          if (ENV.TORCHLETTE_DEBUG_SHAPE) {
            console.log(
              `[replay-copy] cmd=${ci} src=${dbgBufId(slots[cmd.src])} dst=${dbgBufId(slots[cmd.dst])} bytes=${cmd.bytes} encoder=${encoder ? "yes" : "NULL-SKIPPED"}`,
            );
          }
          if (encoder) {
            const tpD0 = TAPE_PROFILE ? performance.now() : 0;
            encoder.copyBufferToBuffer(
              slots[cmd.src],
              cmd.srcOffset,
              slots[cmd.dst],
              cmd.dstOffset,
              cmd.bytes,
            );
            if (TAPE_PROFILE) tpAdd("replay-dawn", performance.now() - tpD0);
          }
          break;
        }
        case TAG_WRITE: {
          // Execute the data source node (tensorFromArray) to produce buffer data
          const writeNode = planNodes[cmd.nodeIndex];
          const hadResult = !!writeNode.result;
          if (!writeNode.result) {
            // STABLE-upload fast path (f32 tensorFromArray): write the fresh
            // payload into a plan-owned buffer via queue.writeBuffer instead
            // of re-executing the node (which allocates a fresh pool/arena
            // buffer whose changed identity invalidates every consumer's
            // cached bind group — 141 createBindGroup/token on Qwen3 decode).
            // Safe ordering: writeBuffer is queue-ordered BEFORE the step's
            // encoder submits (consumers), and the previous step's reads
            // completed at the markStep fence.
            const payload =
              writeNode.op === "tensorFromArray"
                ? (writeNode.payload as
                    | {
                        values?: number[] | Float32Array | Int32Array | Uint32Array;
                        dtype?: DType;
                      }
                    | undefined)
                : undefined;
            const wDtype = payload?.dtype ?? writeNode.dtype;
            const wValues = payload?.values;
            if (
              wValues &&
              wDtype === "f32" &&
              // Small per-step host uploads only (decode: token idx / RoPE
              // slices / masks / scatter indices, all ≤ a few KB). Large
              // uploads (weight-loading plans push 8-50MB per write) stay on
              // the legacy path — a dedicated stable copy would double their
              // memory for no bind-group benefit once loading ends.
              wValues.length * 4 <= STABLE_WRITE_MAX_BYTES &&
              !(wValues instanceof Int32Array) &&
              !(wValues instanceof Uint32Array)
            ) {
              const f32 =
                wValues instanceof Float32Array
                  ? wValues
                  : Float32Array.from(wValues);
              let sbuf = cmd.stableBuf;
              if (!sbuf) {
                sbuf = device.createBuffer({
                  size: Math.max(16, (f32.byteLength + 15) & ~15),
                  usage:
                    GPUBufferUsage.STORAGE |
                    GPUBufferUsage.COPY_SRC |
                    GPUBufferUsage.COPY_DST,
                });
                gpuMemoryTracker.trackAllocation(sbuf, f32.byteLength);
                // PIN: the buffer is wrapped NON-owning (never incRef'd), so
                // without a pin every "is it free?" check sees it as
                // ownerless — resolveOutputBuffer's in-place reuse hands it
                // out as another dispatch's OUTPUT, whose owner later
                // releases it into the POOL, and pool eviction destroys it
                // while this command still writeBuffers into it every replay
                // ("used in submit while destroyed", tokens diverge).
                pinnedBufferSet.add(sbuf);
                cmd.stableBuf = sbuf;
                if (ENV.TORCHLETTE_DEBUG_STABLEBUF) {
                  const orig = sbuf.destroy?.bind(sbuf);
                  console.log(
                    `[stablebuf] CREATE bytes=${f32.byteLength} node=${writeNode.id} cmd=${ci}`,
                  );
                  (sbuf as { destroy?: () => void }).destroy = () => {
                    console.log(
                      `[stablebuf] DESTROY bytes=${f32.byteLength} node=${writeNode.id}\n${new Error().stack}`,
                    );
                    orig?.();
                  };
                }
              }
              const tpD0 = TAPE_PROFILE ? performance.now() : 0;
              device.queue.writeBuffer(sbuf, 0, f32);
              if (TAPE_PROFILE) tpAdd("replay-dawn", performance.now() - tpD0);
              writeNode.result = createStorageHandle(
                writeNode.device,
                createTensor(
                  writeNode.shape,
                  sbuf,
                  undefined,
                  0,
                  "f32",
                  /* ownsBuffer */ false,
                ),
              );
            } else {
              // Legacy path may encode its own passes / flush (chunked ops).
              const tpW0 = TAPE_PROFILE ? performance.now() : 0;
              closeOpenPass();
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
              if (TAPE_PROFILE)
                tpAdd("replay-write-legacy", performance.now() - tpW0);
            }
          }
          slots[cmd.slot] = gpuBuffer(writeNode.result!.backendTensor);
          if (ENV.TORCHLETTE_DEBUG_WRITES) {
            console.log(
              `[replay-write] cmd=${ci} slot=${cmd.slot} node=${writeNode.id} bytes=${(writeNode.result!.backendTensor as { size?: number }).size ?? "?"} buf=${dbgBufId(slots[cmd.slot])} ${hadResult ? "PRE-MATERIALIZED" : "executed"}`,
            );
          }
          break;
        }
        case TAG_CLEAR: {
          closeOpenPass(); // clearBuffer is an encoder-level command
          const encoder = getSharedEncoderInstance();
          if (encoder) {
            const tpD0 = TAPE_PROFILE ? performance.now() : 0;
            encoder.clearBuffer(slots[cmd.slot], 0, cmd.bytes);
            if (TAPE_PROFILE) tpAdd("replay-dawn", performance.now() - tpD0);
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
          const packed = cmd.pack(node);
          const tpD0 = TAPE_PROFILE ? performance.now() : 0;
          device.queue.writeBuffer(slots[cmd.slot], 0, packed);
          if (TAPE_PROFILE) tpAdd("replay-dawn", performance.now() - tpD0);
          break;
        }
        case TAG_BARRIER: {
          closeOpenPass(); // encoder.finish() requires no open pass
          const tpD0 = TAPE_PROFILE ? performance.now() : 0;
          flushSharedEncoder();
          flushBufferPool();
          if (TAPE_PROFILE) tpAdd("replay-barrier", performance.now() - tpD0);
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

    // Phase 3: Harvest results — assign to plan nodes for downstream plans.
    //
    // OWNERSHIP mirrors the lowered path's wrapResultAsStorage:
    //  - IN-PLACE result (the slot buffer IS one of the node's input buffers
    //    — fused adamStep updates param/m/v in their own buffers): the wrap
    //    must be a NON-OWNING view chained to the input storage
    //    (baseStorageId + rcRetain keeps the true owner alive). A naively
    //    owning wrap double-owns the buffer: when the previous step's wrap
    //    dies at markStep it DESTROYS the live state buffer (under a pool
    //    budget release() refuses to pool the 167MB m/v buffers → deferred
    //    destroy → every later adamStep submit binds a destroyed buffer and
    //    is rejected — the 124M DiLoCo "while destroyed" failure).
    //  - PINNED buffer (plan-owned planned/persistent binding): non-owning,
    //    the plan's teardown disposes it.
    //  - Otherwise (default-arena buffers, dynamic fallback allocs): owning,
    //    as before (arena buffers are shielded by arenaBufferSet anyway).
    const tpH0 = TAPE_PROFILE ? performance.now() : 0;
    let dbgHarvestSkipped = 0;
    // Release the PRIOR replay's view-base retains before creating this
    // replay's (see the rationale where they're pushed below). First replay:
    // the field is undefined → nothing to release.
    if (compiled._viewBaseRetains) {
      for (const baseId of compiled._viewBaseRetains) {
        rcRelease(baseId, "view.reharvest");
      }
    }
    const viewBaseRetains: number[] = [];
    const resolveInputStorage = (
      node: LazyIRNode,
      j: number,
    ): StorageHandle | undefined => {
      const ref = node.inputs[j];
      if (!ref || ref.kind === "scalar") return undefined;
      if (ref.kind === "materialized") return ref.storage;
      const oi = ref.outputIndex ?? 0;
      return oi === 0 ? ref.node.result : ref.node.results?.[oi];
    };
    for (const r of compiled.results) {
      const node = planNodes[r.nodeIndex];
      const buf = slots[r.slot];

      // Will this result actually be ASSIGNED anywhere? Storage handles must
      // be created LAZILY, only when assigned: an eagerly-created handle that
      // nobody adopts is an ORPHAN with rc=0 — destroyUnreachable at markStep
      // destroys it and, if it OWNS the buffer, releases a buffer that the
      // node's real storage (e.g. a TAG_WRITE upload) still owns. The double
      // release puts the buffer in the pool TWICE; two later allocations
      // (outer-step uploads in the same size class) then receive THE SAME
      // buffer in one replay — queue.writeBuffer executes immediately while
      // the copies are still encoded, so every copy reads the LAST upload's
      // data (the 124M DiLoCo outer-step corruption: whole params overwritten
      // with another param's update, loss +0.5 nats).
      const assignsPrimary = r.outputIndex === 0 && !node.result;
      const assignsExtra =
        r.outputIndex > 0 || (node.results && node.results.length > 0);
      if (
        ENV.TORCHLETTE_DEBUG_HARVEST_ALL ||
        (ENV.TORCHLETTE_DEBUG_SHAPE &&
          r.shape.join(",") === ENV.TORCHLETTE_DEBUG_SHAPE) ||
        (ENV.TORCHLETTE_DEBUG_OPMATCH &&
          node.op.includes(ENV.TORCHLETTE_DEBUG_OPMATCH))
      ) {
        console.log(
          `[harvest-${r.shape.join("x")}] node[${r.nodeIndex}] op=${node.op} id=${node.id} oi=${r.outputIndex} slot=${r.slot} buf=${dbgBufId(slots[r.slot])} ${assignsPrimary || assignsExtra ? "assign" : "SKIP(existing buf=" + (node.result ? dbgBufId(gpuBuffer(node.result.backendTensor)) : "?") + ")"}`,
        );
      }
      if (!assignsPrimary && !assignsExtra) {
        if (ENV.TORCHLETTE_DEBUG_HARVEST) {
          dbgHarvestSkipped++;
          if (dbgHarvestSkipped <= 5 && node.result) {
            const existing = gpuBuffer(node.result.backendTensor);
            console.log(
              `[harvest-skip] node[${r.nodeIndex}] op=${node.op} shape=${JSON.stringify(node.shape)} existing buf ${existing === buf ? "SAME" : "DIFFERS"} from replay slot`,
            );
          }
        }
        continue;
      }

      let base: StorageHandle | undefined;
      for (let j = 0; j < node.inputs.length; j++) {
        const st = resolveInputStorage(node, j);
        if (st && gpuBuffer(st.backendTensor) === buf) {
          base = st;
          break;
        }
      }
      const tensor = createTensor(
        r.shape,
        buf,
        r.strides,
        r.offset,
        r.dtype,
        /* ownsBuffer */ !base && !pinnedBufferSet.has(buf),
      );
      const sh = createStorageHandle(node.device, tensor);
      if (base) {
        sh.baseStorageId = base.id;
        rcRetain(base.id, "view.baseStorageId");
        // The harvest re-creates VIEW result handles every replay and rcRetains
        // their base each time. The PLAN owns this retain: it releases the PRIOR
        // replay's set at harvest start (below) and at teardown. Mark the handle
        // `planOwnedBaseRetain` so destroyStorageIds does NOT ALSO fire the
        // "view.destroyed" base release when this handle is reaped at markStep —
        // that second release double-frees the base (rc → negative, base handle
        // destroyed while its owning wrapper is still alive; the [lifetime]
        // reclaimed-read false positive on GPT2.posIndices). The plan is the
        // SOLE owner of the base retain, so the base's refcount is decoupled
        // from whether the harvested handle is destroyed (this trajectory) or
        // leak-retained across replays (the 867f6c1 optimizer-state case) —
        // robust either way. Invisible in GPU bytes (bases alias pooled
        // buffers), so it slipped past the profiler and the regression check.
        sh.planOwnedBaseRetain = true;
        viewBaseRetains.push(base.id);
      }
      if (assignsPrimary) {
        node.result = sh;
      }
      // Multi-output: populate node.results array
      if (assignsExtra) {
        if (!node.results) {
          node.results = [node.result!];
        }
        node.results[r.outputIndex] = sh;
      }
    }
    // Commit this replay's view-base retains; the prior set was released at
    // harvest start. Storage ids are monotonic (never reused), so releasing a
    // since-freed base id next replay is a safe no-op (rcRelease returns -1).
    compiled._viewBaseRetains = viewBaseRetains;
    if (TAPE_PROFILE) tpAdd("replay-harvest", performance.now() - tpH0);
    if (ENV.TORCHLETTE_DEBUG_HARVEST && dbgHarvestSkipped > 0) {
      console.log(
        `[harvest-skip] ${dbgHarvestSkipped}/${compiled.results.length} results NOT assigned (node.result already set); planNodes[0].id=${planNodes[0]?.id} planNodes[last].id=${planNodes[planNodes.length - 1]?.id}`,
      );
    }
  } finally {
    closeOpenPass(); // encoder must have no open pass before finish()
    clearActiveArena();
    clearArenaExternalInputBuffers();
    endSharedEncoder();
  }

  // Debug: read back slot contents AFTER the replay's submission, to see
  // which slots carry stale data across steps.
  if (ENV.TORCHLETTE_DEBUG_READSLOTS) {
    // CRITICAL: submit the step-level shared encoder first, else the staging
    // copies below jump the queue and read PRE-replay data (this debug's
    // first incarnation did exactly that — all its readings were invalid).
    flushSharedEncoder();
    const want = ENV.TORCHLETTE_DEBUG_READSLOTS.split(",").map(Number);
    for (const si of want) {
      const buf = slots[si];
      if (!buf) {
        console.log(`[readslot] slot ${si}: not populated`);
        continue;
      }
      try {
        const bytes = Math.min(64, (buf as unknown as { size: number }).size);
        const staging = device.createBuffer({
          size: 256,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(buf, 0, staging, 0, bytes);
        device.queue.submit([enc.finish()]);
        await staging.mapAsync(0x0001 /* MAP_READ */);
        const f = new Float32Array(staging.getMappedRange().slice(0, bytes));
        let sum = 0;
        for (const v of f) sum += v;
        console.log(
          `[readslot] slot ${si} buf=${dbgBufId(buf)} sum16=${sum.toExponential(4)} [0..3]=${Array.from(f.slice(0, 4)).map((v) => v.toPrecision(4))}`,
        );
        staging.destroy();
      } catch (e) {
        console.log(`[readslot] slot ${si}: read failed (${e})`);
      }
    }
  }
}
