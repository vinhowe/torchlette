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
import {
  planBinaryDirect,
  planUnaryDirect,
} from "../backend/webgpu/dispatch";
import {
  planCastDirect,
  planContiguousDirect,
} from "../backend/webgpu/ops/views";
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
  /** Per-covered-action command segments, in action order. Verified against
   *  the recording by diffSegmentsAligned (slot-bijection comparison), so
   *  every covered action is checked even when other actions are uncovered. */
  segments: Array<{ nodeIndex: number; commands: GpuCommand[] }>;
  /** Flat concatenation of segments (full-stream diff when fullyCovered). */
  commands: GpuCommand[];
  /** Slot sources for generated slots (placeholder kinds; numbering local). */
  slots: SlotSource[];
  /** Whether every action had a generator. */
  fullyCovered: boolean;
  coveredActions: number;
  /** op/action label → count of actions WITHOUT a generator. */
  uncovered: Map<string, number>;
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

  // 2. Walk actions in execution order. Covered actions yield verifiable
  //    segments. Uncovered actions still map their output buffers to fresh
  //    PHANTOM slots, so downstream covered actions can reference their
  //    results — the segment diff compares modulo slot renaming, so phantom
  //    numbering needn't match the recording.
  const segments: Array<{ nodeIndex: number; commands: GpuCommand[] }> = [];
  let actionCount = 0;
  let coveredActions = 0;
  // Logical identity channel: plan-node index -> generated slot. Liveness
  // may have RELEASED an intermediate's result before plan-build (its
  // buffer identity is gone), but its slot is still well-defined — the one
  // the walker assigned when it processed the producer. Views alias to
  // their input's slot (the recording keys slots by buffer, so a view and
  // its base share one slot; a separate logical slot per view node would
  // break the diff bijection).
  const nodeSlot = new Map<number, Slot>();
  const resolveRefSlot = (ref: LazyIRNode["inputs"][number]): Slot | undefined => {
    if (ref.kind === "scalar") return undefined;
    if (ref.kind !== "materialized") {
      const oi = ref.outputIndex ?? 0;
      if (oi === 0) {
        const idx = nodeIndexById.get(ref.node.id);
        if (idx !== undefined) {
          const s = nodeSlot.get(idx);
          if (s !== undefined) return s;
        }
      }
      const storage =
        oi === 0 ? ref.node.result : ref.node.results?.[oi];
      if (!storage) return undefined;
      return bufferToSlot.get(gpuBuffer(storage.backendTensor) as unknown);
    }
    return bufferToSlot.get(
      gpuBuffer(ref.storage.backendTensor) as unknown,
    );
  };
  const phantom = (node: LazyIRNode | undefined, nodeIndex?: number) => {
    if (!node) return;
    const slot = slots.length;
    slots.push({ kind: "arena" });
    mapNodeResult(node, slot, bufferToSlot);
    if (nodeIndex !== undefined) nodeSlot.set(nodeIndex, slot);
  };
  for (const action of loweredPlan.actions) {
    actionCount++;
    switch (action.kind) {
      case "view": {
        const node = planNodes[action.nodeIndex];
        const aliased = node ? resolveRefSlot(node.inputs[0]) : undefined;
        if (aliased !== undefined) nodeSlot.set(action.nodeIndex, aliased);
        coveredActions++;
        break;
      }
      case "prologue-skip":
        coveredActions++;
        break;
      case "reclaim":
        commands.push({ tag: TAG_BARRIER });
        coveredActions++;
        break;
      case "data-source": {
        const node = planNodes[action.nodeIndex];
        const gen = generateDataSource(node, action.nodeIndex, slots);
        if (!gen) {
          miss(`data-source:${node.op}${node.dtype !== "f32" ? `:${node.dtype}` : ""}`);
          phantom(node, action.nodeIndex);
          break;
        }
        commands.push(...gen);
        segments.push({ nodeIndex: action.nodeIndex, commands: gen });
        const dsSlot =
          gen[0].tag === TAG_ALLOC ? (gen[0] as { slot: Slot }).slot : -1;
        mapNodeResult(node, dsSlot, bufferToSlot);
        if (dsSlot >= 0) nodeSlot.set(action.nodeIndex, dsSlot);
        coveredActions++;
        break;
      }
      case "sequential": {
        const node = planNodes[action.nodeIndex];
        const gen = generateSequential(node, slots, resolveRefSlot);
        if (typeof gen === "string") {
          miss(`op:${node.op}${gen ? `[${gen}]` : ""}`);
          phantom(node, action.nodeIndex);
          break;
        }
        commands.push(...gen.commands);
        if (gen.commands.length > 0) {
          segments.push({ nodeIndex: action.nodeIndex, commands: gen.commands });
        }
        mapNodeResult(node, gen.outSlot, bufferToSlot);
        nodeSlot.set(action.nodeIndex, gen.outSlot);
        coveredActions++;
        break;
      }
      default: {
        miss(action.kind);
        // Map every output node the action produces (fused groups,
        // matmul-epilogue, adam batches, row programs, reductions).
        const a = action as {
          nodeIndex?: number;
          nodeIndices?: number[];
          outputNodeIndex?: number;
          additionalOutputNodeIndices?: number[];
        };
        for (const ni of [
          a.nodeIndex,
          a.outputNodeIndex,
          ...(a.nodeIndices ?? []),
          ...(a.additionalOutputNodeIndices ?? []),
        ]) {
          if (ni !== undefined) phantom(planNodes[ni], ni);
        }
        break;
      }
    }
  }

  return {
    segments,
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

/** View ops whose output layout is NOT derivable from shape alone —
 *  metadata synthesis for liveness-released producers excludes them. */
const VIEW_OPS = new Set([
  "reshape",
  "view",
  "transpose",
  "permute",
  "expand",
  "narrow",
  "squeeze",
  "unsqueeze",
  "broadcastTo",
]);

function contiguousStrides(shape: number[]): number[] {
  const s = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i];
  }
  return s;
}

/** node.op → the WGSL symbol dispatchBinary receives from the backend. */
const BINARY_OPS = new Map<string, string>([
  ["add", "+"],
  ["sub", "-"],
  ["mul", "*"],
  ["div", "/"],
  ["pow", "pow"],
  ["minimum", "min"],
  ["maximum", "max"],
]);

/** Table-driven unary ops routed to dispatchUnary with identity opKeys. */
const UNARY_OPS = new Set([
  "sqrt",
  "relu",
  "exp",
  "log",
  "neg",
  "abs",
  "tanh",
  "sigmoid",
  "silu",
  "sin",
  "cos",
  "rsqrt",
  "floor",
  "ceil",
  "round",
  "sign",
  "isfinite",
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
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  const wgslOp = BINARY_OPS.get(node.op);
  const isUnary =
    UNARY_OPS.has(node.op) ||
    node.op === "cast" ||
    node.op === "contiguous" ||
    node.op === "gelu";
  if (!wgslOp && !isUnary) return "";
  if (wgslOp && node.inputs.length !== 2) return "arity";
  if (isUnary && node.inputs.length !== 1) return "arity";
  const ins: WebGPUTensor[] = [];
  const inSlots: Slot[] = [];
  for (const ref of node.inputs) {
    if (ref.kind === "scalar") return "scalar-operand";
    // Tensor METADATA (shape/strides/offset/dtype) comes from the storage
    // when it still exists. Liveness may have RELEASED an intermediate's
    // result before plan-build: for NON-VIEW producers the output layout is
    // structural (fresh allocation: contiguous, offset 0, node shape/dtype)
    // so the metadata is synthesized from the node — and if that assumption
    // is ever wrong, the pipeline-identity diff catches it loudly. View
    // producers' layouts aren't shape-derivable; they stay bailed.
    const storage =
      ref.kind === "materialized"
        ? ref.storage
        : ((ref.outputIndex ?? 0) === 0
            ? ref.node.result
            : ref.node.results?.[ref.outputIndex ?? 0]);
    let meta: WebGPUTensor;
    if (storage) {
      meta = storage.backendTensor as WebGPUTensor;
    } else if (
      ref.kind !== "materialized" &&
      (ref.outputIndex ?? 0) === 0 &&
      !VIEW_OPS.has(ref.node.op)
    ) {
      const shape = ref.node.shape;
      const dtype = (ref.node.dtype ?? "f32") as WebGPUTensor["dtype"];
      const size = sizeOf(shape);
      meta = {
        shape,
        strides: contiguousStrides(shape),
        offset: 0,
        size,
        dtype,
        isContiguous: true,
        // Only .size is consulted by planner guards.
        buffer: { size: alignBufferSize(size * 4) },
      } as unknown as WebGPUTensor;
    } else {
      return "no-storage";
    }
    const slot = resolveRefSlot(ref);
    if (slot === undefined) return "untracked-producer";
    ins.push(meta);
    inSlots.push(slot);
  }
  let plan;
  if (wgslOp) {
    plan = planBinaryDirect(wgslOp, ins[0], ins[1]);
  } else if (node.op === "cast") {
    plan = planCastDirect(ins[0], node.dtype);
  } else if (node.op === "contiguous") {
    // Already-contiguous fast path: a non-owning view, ZERO commands — the
    // node's result shares the input buffer (mapNodeResult will re-map the
    // same buffer to the input's slot; out slot is the input's).
    if (ins[0].isContiguous) {
      return { commands: [], outSlot: inSlots[0] };
    }
    plan = planContiguousDirect(ins[0]);
  } else if (node.op === "gelu") {
    // Backend convention (elementwise.ts gelu): approximate ?? "tanh";
    // tanh -> opKey "gelu", erf -> "gelu_erf".
    const p = node.payload as { approximate?: string } | undefined;
    plan = planUnaryDirect(
      (p?.approximate ?? "tanh") === "tanh" ? "gelu" : "gelu_erf",
      ins[0],
    );
  } else {
    plan = planUnaryDirect(node.op, ins[0]);
  }
  if (!plan) return "non-direct-route";
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
      // The executor records a TAG_WRITE for EVERY data-source node
      // (op-agnostic), so a zeros node is ALLOC+CLEAR+WRITE. The write
      // re-uploads zero data the clear already produced — redundant but
      // recorded reality; mirror it (candidate cleanup: skip the write for
      // zeros at the executor seam, which would also shrink replays).
      return [
        { tag: TAG_ALLOC, slot, bytes, allocKind: 0, inputSlots: [] },
        { tag: TAG_CLEAR, slot, bytes: alignBufferSize(bytes) },
        { tag: TAG_WRITE, slot, nodeIndex },
      ];
    }
    default:
      return null;
  }
}
