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
import { planBinaryDirect } from "../backend/webgpu/dispatch";
import { requireContext } from "../backend/webgpu/gpu-context";
import { getPipeline } from "../backend/webgpu/dispatch";
import { alignBufferSize } from "../backend/webgpu/shape-utils";
import { gpuBuffer } from "../backend/webgpu/gpu-types";
import type { WebGPUTensor } from "../backend/webgpu/tensor";
import type { LazyIRNode, StorageHandle } from "../graph/types";
import { sizeOf } from "../core/shape";
import type { GpuCommand, Slot, SlotSource } from "./compiled-plan";
import {
  TAG_ALLOC,
  TAG_BARRIER,
  TAG_CLEAR,
  TAG_DISPATCH,
  TAG_WRITE,
} from "./compiled-plan";
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
          // Map the node's actual result buffer to its alloc slot so later
          // ops can reference it as an input (build-time shadow generation
          // reads post-execution buffer identity; pure generation will use
          // logical node identity instead).
          mapNodeResult(node, gen[0].tag === TAG_ALLOC ? (gen[0] as { slot: Slot }).slot : -1, bufferToSlot);
          coveredActions++;
        }
        break;
      }
      case "sequential": {
        const node = planNodes[action.nodeIndex];
        const gen = generateSequential(node, slots, bufferToSlot);
        if (gen === null) {
          miss(`op:${node.op}`);
          generating = false;
          break;
        }
        if (generating) {
          commands.push(...gen.commands);
          mapNodeResult(node, gen.outSlot, bufferToSlot);
          coveredActions++;
        }
        break;
      }
      default:
        miss(action.kind);
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

/** Map a generated node's actual result buffer → its alloc slot (temporal:
 *  pool reuse overwrites, mirroring the recording's buffer→slot map). */
function mapNodeResult(
  node: LazyIRNode,
  slot: Slot,
  bufferToSlot: Map<unknown, Slot>,
): void {
  if (slot < 0 || !node.result) return;
  bufferToSlot.set(gpuBuffer(node.result.backendTensor) as unknown, slot);
}

/** node.op → the WGSL symbol dispatchBinary receives from the backend. */
const BINARY_OPS = new Map<string, string>([
  ["add", "+"],
  ["sub", "-"],
  ["mul", "*"],
  ["div", "/"],
]);

/**
 * Sequential-action generator. Increment 1: DIRECT binary elementwise via
 * planBinaryDirect — the SAME plan dispatchBinary's direct tail executes,
 * so the generated pipeline (resolved through the same getPipeline cache),
 * workgroups, and params are the dispatcher's by construction. Recorded
 * shape: ALLOC(out, kind 0, inputs=[a,b]) then DISPATCH(bind=[a,b,out,
 * params]); the params buffer takes the slot AFTER the output (assigned at
 * createParamsBuffer during encoding).
 */
function generateSequential(
  node: LazyIRNode,
  slots: SlotSource[],
  bufferToSlot: Map<unknown, Slot>,
): { commands: GpuCommand[]; outSlot: Slot } | null {
  const wgslOp = BINARY_OPS.get(node.op);
  if (!wgslOp) return null;
  if (node.inputs.length !== 2) return null;
  const ins: WebGPUTensor[] = [];
  const inSlots: Slot[] = [];
  for (const ref of node.inputs) {
    if (ref.kind === "scalar") return null; // scalar-table operand — later
    const storage =
      ref.kind === "materialized"
        ? ref.storage
        : ((ref.outputIndex ?? 0) === 0
            ? ref.node.result
            : ref.node.results?.[ref.outputIndex ?? 0]);
    if (!storage) return null;
    const buf = gpuBuffer(storage.backendTensor) as unknown;
    const slot = bufferToSlot.get(buf);
    if (slot === undefined) return null; // untracked producer — bail
    ins.push(storage.backendTensor as WebGPUTensor);
    inSlots.push(slot);
  }
  const plan = planBinaryDirect(wgslOp, ins[0], ins[1]);
  if (!plan) return null;
  // Resolve the pipeline through the SAME cache the dispatcher used (at
  // build time it is warm — this just interns the identity for the diff).
  const pipeline = getPipeline(requireContext(), plan.key, plan.shader);
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const paramsSlot = slots.length;
  slots.push({
    kind: "params",
    seqIndex: -1,
    data: plan.paramsData,
  });
  return {
    commands: [
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: plan.outputSizeBytes,
        allocKind: 0,
        inputSlots: inSlots,
      },
      {
        tag: TAG_DISPATCH,
        pipeline,
        bindings: [...inSlots, outSlot, paramsSlot],
        gx: plan.dispatchX,
        gy: plan.dispatchY,
        gz: 1,
      },
    ],
    outSlot,
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
