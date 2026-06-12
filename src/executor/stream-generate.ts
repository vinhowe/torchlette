/**
 * Stage-4 phase 2: GENERATE the GpuCommand stream from the lowered plan,
 * instead of recording it from an execution.
 *
 * The recorder must observe every effect or the replay silently corrupts
 * (the record-hook bug ledger). Generation inverts the burden: the stream
 * is derived from the plan, and the RECORDING becomes the cross-check —
 * under TORCHLETTE_STREAM_GENERATE=1 the executor builds both and diffs
 * them at the command level (stream-diff.ts), so a divergence is a loud
 * build-time finding instead of a loss-curve archaeology project.
 *
 * Coverage is a counter, not a cliff: an action class without a generator
 * yields null (the plan keeps record/replay), and the uncovered ops are
 * reported. Increment 0 covers the effect-only classes — data sources
 * (tensorFromArray → ALLOC+WRITE, zeros → ALLOC+CLEAR), views, skips, and
 * reclaim barriers. Dispatching classes (tile-IR, fused, matmul) follow,
 * each sharing its spec/codegen with the imperative dispatcher so pipeline
 * identity resolves through the same caches.
 */
import { alignBufferSize } from "../backend/webgpu/shape-utils";
import { gpuBuffer } from "../backend/webgpu/gpu-types";
import type { LazyIRNode, StorageHandle } from "../graph/types";
import { sizeOf } from "../core/shape";
import type { GpuCommand, Slot, SlotSource } from "./compiled-plan";
import { TAG_ALLOC, TAG_BARRIER, TAG_CLEAR, TAG_WRITE } from "./compiled-plan";
import type { LoweredPlan } from "./lowered-plan";

export interface GeneratedStream {
  /** Commands generated for the covered action PREFIX (all actions before
   *  the first one lacking a generator; the whole plan when fullyCovered). */
  commands: GpuCommand[];
  /** Slot sources, parallel to the recording's table over the prefix. */
  slots: SlotSource[];
  /** Whether every action had a generator (full-stream diff vs prefix). */
  fullyCovered: boolean;
  /** Actions covered before generation stopped. */
  coveredActions: number;
  /** op/action label → count of actions WITHOUT a generator (entire plan,
   *  not just the stopping action — coverage telemetry). */
  uncovered: Map<string, number>;
  /** Total actions examined. */
  actionCount: number;
}

const F32_BYTES = 4;

/**
 * Generate the command stream for a lowered plan. Mirrors the recording's
 * slot-numbering protocol exactly (it must — the differential compares
 * canonical streams where buffers ARE slot indices):
 *  1. externals pre-assigned in (planNodeIndex, inputIndex) order, deduped
 *     by buffer identity (executor.ts pre-assignment loop);
 *  2. allocs take fresh slots in action order (recordAlloc lifetime split
 *     always yields a fresh slot for a fresh logical lifetime).
 */
export function generateStream(
  loweredPlan: LoweredPlan,
  planNodes: LazyIRNode[],
): GeneratedStream {
  const commands: GpuCommand[] = [];
  const slots: SlotSource[] = [];
  const uncovered = new Map<string, number>();
  const bufferToSlot = new Map<unknown, Slot>();
  const nodeIndexById = new Map<number, number>();
  for (let i = 0; i < planNodes.length; i++) {
    nodeIndexById.set(planNodes[i].id, i);
  }

  // 1. External pre-assignment, mirroring executor.ts (recording mode).
  for (let i = 0; i < planNodes.length; i++) {
    const node = planNodes[i];
    for (let j = 0; j < node.inputs.length; j++) {
      const ref = node.inputs[j];
      let storage: StorageHandle | undefined;
      if (ref.kind === "materialized") {
        storage = ref.storage;
      } else if (ref.kind === "scalar") {
        continue;
      } else {
        const idx = ref.outputIndex ?? 0;
        storage = idx === 0 ? ref.node.result : ref.node.results?.[idx];
      }
      if (!storage) continue;
      const buf = gpuBuffer(storage.backendTensor) as unknown;
      if (bufferToSlot.has(buf)) continue;
      bufferToSlot.set(buf, slots.length);
      slots.push({ kind: "external", planNodeIndex: i, inputIndex: j });
    }
  }

  const miss = (label: string) => {
    uncovered.set(label, (uncovered.get(label) ?? 0) + 1);
  };

  // 2. Walk actions in execution order, generating until the first action
  //    without a generator (the PREFIX — diffable against the recorded
  //    stream's head because both follow action order). Keep walking after
  //    that for coverage telemetry only.
  let actionCount = 0;
  let coveredActions = 0;
  let generating = true;
  for (const action of loweredPlan.actions) {
    actionCount++;
    switch (action.kind) {
      case "view":
      case "prologue-skip":
        if (generating) coveredActions++;
        break;
      case "reclaim":
        if (generating) {
          commands.push({ tag: TAG_BARRIER });
          coveredActions++;
        }
        break;
      case "data-source": {
        const node = planNodes[action.nodeIndex];
        const gen = generateDataSource(node, action.nodeIndex, slots);
        if (!gen) {
          miss(`data-source:${node.op}${node.dtype !== "f32" ? `:${node.dtype}` : ""}`);
          generating = false;
          break;
        }
        if (generating) {
          commands.push(...gen);
          coveredActions++;
        }
        break;
      }
      default:
        miss(action.kind === "sequential"
          ? `op:${planNodes[(action as { nodeIndex: number }).nodeIndex]?.op ?? "?"}`
          : action.kind);
        generating = false;
        break;
    }
  }

  return {
    commands,
    slots,
    fullyCovered: uncovered.size === 0,
    coveredActions,
    uncovered,
    actionCount,
  };
}

/**
 * Data-source generators (effect-only ops — no pipelines, no params):
 *  - tensorFromArray (f32): ALLOC(kind 0, no inputs) + WRITE(nodeIndex).
 *    The replay's TAG_WRITE re-executes the source node, so per-step data
 *    flows as data — the same volatility contract recording captures.
 *  - zeros (f32): ALLOC + CLEAR(aligned size). The imperative path clears
 *    only when the buffer came from pool/arena; generation emits the clear
 *    UNCONDITIONALLY — replay buffers are always reused, so the clear is
 *    always required (a recording that skipped it is the stale-accumulator
 *    hazard; the differential makes that visible).
 * Non-f32 dtypes take the typed-buffer path (different effects) — uncovered
 * until ported.
 */
function generateDataSource(
  node: LazyIRNode,
  nodeIndex: number,
  slots: SlotSource[],
): GpuCommand[] | null {
  if (node.dtype !== "f32" && node.dtype !== undefined) return null;
  const bytes = sizeOf(node.shape) * F32_BYTES;
  switch (node.op) {
    case "tensorFromArray": {
      const slot = slots.length;
      slots.push({ kind: "arena" });
      return [
        { tag: TAG_ALLOC, slot, bytes, allocKind: 0, inputSlots: [] },
        { tag: TAG_WRITE, slot, nodeIndex },
      ];
    }
    case "zeros": {
      const slot = slots.length;
      slots.push({ kind: "arena" });
      return [
        { tag: TAG_ALLOC, slot, bytes, allocKind: 0, inputSlots: [] },
        { tag: TAG_CLEAR, slot, bytes: alignBufferSize(bytes) },
      ];
    }
    default:
      return null;
  }
}
