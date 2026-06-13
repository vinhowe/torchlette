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
import { planFullReductionDispatch } from "../backend/webgpu/ops/reductions";
import { planLayerNormForwardDispatch } from "../backend/webgpu/layernorm-kernel";
import { planFusedKernel } from "../backend/webgpu/fusion-dispatch";
import {
  lookupKSplitTempBuffer,
  planTiledMatmul,
} from "../backend/webgpu/matmul/dispatch";
import type { BareMatmulPlan } from "../backend/webgpu/dispatch";
import { dtypeBytes } from "../backend/webgpu/shape-utils";
import { planAdamStepDispatch } from "../backend/webgpu/adam-kernel";
import {
  lookupPackedBuffers,
  planPackedGroups,
} from "../optim/packed-dispatch";
import { getInputStorage } from "./op-dispatch";
import { planUnscaleGradDispatch } from "../backend/webgpu/unscale-kernel";
import type { TileKernelPlan } from "../backend/webgpu/tile-dispatch";
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
  TAG_COPY,
  TAG_DISPATCH,
  TAG_UNIFORM,
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
  backend?: import("../backend/types").Backend,
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
  // Slot for a REAL buffer object (config caches, inf flags): one slot per
  // buffer, shared across the stream — mirroring the recording's
  // buffer-keyed persistent slots so the diff bijection stays 1:1.
  const bufferSlot = (buf: unknown, kind: SlotSource["kind"]): Slot => {
    const existing = bufferToSlot.get(buf);
    if (existing !== undefined) return existing;
    const slot = slots.length;
    slots.push(
      kind === "persistent"
        ? { kind: "persistent", buffer: buf as never }
        : { kind: "arena" },
    );
    bufferToSlot.set(buf, slot);
    return slot;
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
        const gen =
          node.op === "matmul"
            ? generateBareMatmul(
                (action as { cachedMatmulPlan?: BareMatmulPlan | string })
                  .cachedMatmulPlan,
                node,
                slots,
                resolveRefSlot,
                bufferSlot,
              )
            : generateSequential(node, slots, resolveRefSlot, bufferSlot);
        if (typeof gen === "string") {
          miss(`op:${node.op}${gen ? `[${gen}]` : ""}`);
          phantom(node, action.nodeIndex);
          break;
        }
        for (const c of gen.commands) {
          if (c.tag === TAG_UNIFORM && c.nodeIndex < 0) {
            c.nodeIndex = action.nodeIndex;
          }
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
      case "fused": {
        const fa = action as FusedActionShape;
        const gen = backend
          ? generateFused(fa, planNodes, slots, resolveRefSlot, bufferSlot, backend)
          : "no-backend";
        if (typeof gen === "string") {
          miss(`fused[${gen}]`);
          for (const ni of [
            fa.outputNodeIndex,
            ...(fa.additionalOutputNodeIndices ?? []),
          ]) {
            if (ni !== undefined) phantom(planNodes[ni], ni);
          }
          break;
        }
        commands.push(...gen.commands);
        segments.push({ nodeIndex: fa.outputNodeIndex, commands: gen.commands });
        for (const o of gen.outputs) {
          mapNodeResult(planNodes[o.nodeIndex], o.slot, bufferToSlot);
          nodeSlot.set(o.nodeIndex, o.slot);
        }
        coveredActions++;
        break;
      }
      case "matmul-epilogue": {
        const me = action as MatmulEpilogueActionShape;
        const gen = backend
          ? generateMatmulEpilogue(me, planNodes, slots, resolveRefSlot, bufferSlot, backend)
          : "no-backend";
        if (typeof gen === "string") {
          miss(`matmul-epilogue[${gen}]`);
          phantom(planNodes[me.outputNodeIndex], me.outputNodeIndex);
          break;
        }
        commands.push(...gen.commands);
        segments.push({ nodeIndex: me.matmulNodeIndex, commands: gen.commands });
        mapNodeResult(planNodes[me.outputNodeIndex], gen.outSlot, bufferToSlot);
        nodeSlot.set(me.outputNodeIndex, gen.outSlot);
        coveredActions++;
        break;
      }
      case "adam-batch": {
        const ab = action as { nodeIndices: number[] };
        const gen = backend
          ? generateAdamBatch(ab, planNodes, slots, resolveRefSlot, bufferSlot, backend)
          : "no-backend";
        if (typeof gen === "string") {
          miss(`adam-batch[${gen}]`);
          for (const ni of ab.nodeIndices) phantom(planNodes[ni], ni);
          break;
        }
        // Leading barrier (executor recordBarrier before adamStepBatch) lives
        // in the flat stream only — non-node-attributed, not segment-checked.
        commands.push({ tag: TAG_BARRIER });
        for (const c of gen.commands) {
          if (c.tag === TAG_UNIFORM && c.nodeIndex < 0) {
            c.nodeIndex = gen.firstNodeIndex;
          }
        }
        commands.push(...gen.commands);
        segments.push({ nodeIndex: gen.firstNodeIndex, commands: gen.commands });
        for (const o of gen.outputs) {
          mapNodeResult(planNodes[o.nodeIndex], o.slot, bufferToSlot);
          nodeSlot.set(o.nodeIndex, o.slot);
        }
        coveredActions++;
        break;
      }
      default: {
        miss(action.kind);
        // Map every output node the action produces (matmul-epilogue, row
        // programs, reductions).
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
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  // Tile-kernel ops with bespoke command patterns.
  if (node.op === "sum") return generateSumFull(node, slots, resolveRefSlot, bufferSlot);
  if (node.op === "unscaleGrad")
    return generateUnscaleGrad(node, slots, resolveRefSlot, bufferSlot);
  if (node.op === "stridedScatterCopy")
    return generateScatterCopyDMA(node, resolveRefSlot);
  if (node.op === "fusedLayerNormForward")
    return generateLayerNormForward(node, slots, resolveRefSlot, bufferSlot);
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

/** Bindings for a TileKernelPlan: name→slot with the config buffer at the
 *  uniform position. Returns null when the plan's config buffer is absent
 *  (uniform key never dispatched — shouldn't happen post-execution). */
function tilePlanBindings(
  plan: TileKernelPlan,
  named: Record<string, Slot>,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): Slot[] | null {
  const out: Slot[] = [];
  for (const name of plan.bindingOrder) {
    if (name === null) {
      if (!plan.configBuffer) return null;
      out.push(bufferSlot(plan.configBuffer, "persistent"));
    } else {
      const s = named[name];
      if (s === undefined) return null;
      out.push(s);
    }
  }
  return out;
}

/** sum with no dim (FULL reduction): ALLOC(4 bytes, kind 0) + tile dispatch
 *  [input, out, config]. Dim reductions / preamble chains stay uncovered. */
function generateSumFull(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  const payload = node.payload as { dim?: number | number[] | null } | undefined;
  if (payload?.dim != null) return "dim-reduction";
  if (node.inputs.length !== 1) return "arity";
  const ref = node.inputs[0];
  if (ref.kind === "scalar") return "scalar-operand";
  const storage =
    ref.kind === "materialized"
      ? ref.storage
      : ((ref.outputIndex ?? 0) === 0
          ? ref.node.result
          : ref.node.results?.[ref.outputIndex ?? 0]);
  const inSlot = resolveRefSlot(ref);
  if (inSlot === undefined) return "untracked-producer";
  let size: number;
  if (storage) {
    const t = storage.backendTensor as WebGPUTensor;
    if (!t.isContiguous) return "non-contiguous";
    size = t.size;
  } else if (ref.kind !== "materialized" && !VIEW_OPS.has(ref.node.op)) {
    size = sizeOf(ref.node.shape);
  } else {
    return "no-storage";
  }
  if (size * 4 > 128 * 1024 * 1024) return "chunked";
  const plan = planFullReductionDispatch("sum", size);
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const bindings = tilePlanBindings(
    plan,
    { input: inSlot, out: outSlot },
    bufferSlot,
  );
  if (!bindings) return "config-missing";
  return {
    commands: [
      { tag: TAG_ALLOC, slot: outSlot, bytes: 4, allocKind: 0, inputSlots: [inSlot] },
      {
        tag: TAG_DISPATCH,
        pipeline: plan.pipeline,
        bindings,
        gx: plan.grid[0],
        gy: plan.grid[1],
        gz: plan.grid[2],
      },
    ],
    outSlot,
  };
}

/** fusedLayerNormForward: ALLOC(output, kind 1) + one tile dispatch
 *  [x, weight, bias, output, config]. Single output, no workspace temp.
 *  payload = FusedLayerNormConfig {numRows, featureDim, eps}. Inputs are
 *  contiguous in practice (x = residual stream, weight/bias = params); a
 *  non-contiguous input would add a recorded contiguous-copy prologue →
 *  command-count mismatch caught by the segment diff. */
function generateLayerNormForward(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  if (node.inputs.length !== 3) return "arity";
  const cfg = node.payload as
    | { numRows?: number; featureDim?: number; eps?: number }
    | undefined;
  if (!cfg || cfg.numRows == null || cfg.featureDim == null || cfg.eps == null) {
    return "payload";
  }
  const inSlots: Slot[] = [];
  for (const ref of node.inputs) {
    if (ref.kind === "scalar") return "scalar-operand";
    const s = resolveRefSlot(ref);
    if (s === undefined) return "untracked-input";
    inSlots.push(s);
  }
  const planned = planLayerNormForwardDispatch(
    cfg.numRows,
    cfg.featureDim,
    cfg.eps,
  );
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const bindings = tilePlanBindings(
    planned.plan,
    { x: inSlots[0], weight: inSlots[1], bias: inSlots[2], output: outSlot },
    bufferSlot,
  );
  if (!bindings) return "config-missing";
  return {
    commands: [
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: planned.outputBytes,
        allocKind: 1,
        inputSlots: [],
      },
      {
        tag: TAG_DISPATCH,
        pipeline: planned.plan.pipeline,
        bindings,
        gx: planned.plan.grid[0],
        gy: planned.plan.grid[1],
        gz: planned.plan.grid[2],
      },
    ],
    outSlot,
  };
}

/** unscaleGrad: ALLOC(kind 1, power-of-two bytes) + volatile UNIFORM repack
 *  + tile dispatch [grad_in, grad_out, inf_flag, config]. */
function generateUnscaleGrad(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  const payload = node.payload as
    | { invScale?: number; infFlagBuffer?: unknown }
    | undefined;
  if (!payload || payload.invScale === undefined || !payload.infFlagBuffer) {
    return "payload";
  }
  const ref = node.inputs[0];
  if (!ref || ref.kind === "scalar") return "arity";
  const storage =
    ref.kind === "materialized"
      ? ref.storage
      : ((ref.outputIndex ?? 0) === 0
          ? ref.node.result
          : ref.node.results?.[ref.outputIndex ?? 0]);
  const inSlot = resolveRefSlot(ref);
  if (inSlot === undefined) return "untracked-producer";
  let size: number;
  if (storage) {
    const t = storage.backendTensor as WebGPUTensor;
    if (!t.isContiguous) return "non-contiguous";
    size = t.size;
  } else if (ref.kind !== "materialized" && !VIEW_OPS.has(ref.node.op)) {
    size = sizeOf(ref.node.shape);
  } else {
    return "no-storage";
  }
  const planned = planUnscaleGradDispatch(size, payload.invScale);
  if (!planned) return "chunked";
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const named: Record<string, Slot> = {
    grad_in: inSlot,
    grad_out: outSlot,
    inf_flag: bufferSlot(payload.infFlagBuffer, "persistent"),
  };
  const bindings = tilePlanBindings(planned.plan, named, bufferSlot);
  if (!bindings) return "config-missing";
  const configSlot = bufferSlot(planned.plan.configBuffer, "persistent");
  return {
    commands: [
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: planned.alignedBytes,
        allocKind: 1,
        inputSlots: [],
      },
      {
        tag: TAG_UNIFORM,
        slot: configSlot,
        nodeIndex: -1, // patched by the caller (walker knows the action index)
        pack: () => new Uint32Array(0),
      },
      {
        tag: TAG_DISPATCH,
        pipeline: planned.plan.pipeline,
        bindings,
        gx: planned.plan.grid[0],
        gy: planned.plan.grid[1],
        gz: planned.plan.grid[2],
      },
    ],
    outSlot,
  };
}

/** Resolve a ref's slot + tensor metadata (real storage, or synthesized
 *  contiguous/offset-0 for liveness-released NON-VIEW producers). */
function refSlotAndMeta(
  ref: LazyIRNode["inputs"][number],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
): { slot: Slot; meta: WebGPUTensor } | string {
  if (ref.kind === "scalar") return "scalar-operand";
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
    const size = sizeOf(shape);
    meta = {
      shape,
      strides: contiguousStrides(shape),
      offset: 0,
      size,
      dtype: (ref.node.dtype ?? "f32") as WebGPUTensor["dtype"],
      isContiguous: true,
      buffer: { size: alignBufferSize(size * 4) },
    } as unknown as WebGPUTensor;
  } else {
    return "no-storage";
  }
  const slot = resolveRefSlot(ref);
  if (slot === undefined) return "untracked-producer";
  return { slot, meta };
}

/** stridedScatterCopy, TRUE IN-PLACE DMA fast path: one TAG_COPY into the
 *  base's EXISTING buffer (no alloc; the node's result keeps the base
 *  slot). The dispatch (copy-on-write) route stays uncovered. Conditions
 *  mirror stridedScatterImpl exactly. */
function generateScatterCopyDMA(
  node: LazyIRNode,
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  const payload = node.payload as
    | { offset: number; viewShape: number[]; viewStrides: number[] }
    | undefined;
  if (!payload) return "payload";
  if (node.inputs.length !== 2) return "arity";
  const base = refSlotAndMeta(node.inputs[0], resolveRefSlot);
  if (typeof base === "string") return base;
  const src = refSlotAndMeta(node.inputs[1], resolveRefSlot);
  if (typeof src === "string") return src;
  const { offset, viewShape, viewStrides } = payload;
  const baseSize = sizeOf(base.meta.shape);
  const viewSize = sizeOf(viewShape);
  const isContig = (shape: number[], strides: number[]) => {
    let acc = 1;
    for (let i = shape.length - 1; i >= 0; i--) {
      if (strides[i] !== acc && shape[i] !== 1) return false;
      acc *= shape[i];
    }
    return true;
  };
  if (
    !(
      offset >= 0 &&
      offset + viewSize <= baseSize &&
      isContig(viewShape, viewStrides) &&
      base.meta.isContiguous &&
      (base.meta.offset ?? 0) === 0 &&
      (src.meta.isContiguous || isContig(src.meta.shape, src.meta.strides)) &&
      viewSize * 4 <=
        (src.meta.buffer as { size: number }).size -
          (src.meta.offset ?? 0) * 4
    )
  ) {
    return "non-dma-route";
  }
  return {
    commands: [
      {
        tag: TAG_COPY,
        src: src.slot,
        srcOffset: (src.meta.offset ?? 0) * 4,
        dst: base.slot,
        dstOffset: offset * 4,
        bytes: viewSize * 4,
      },
    ],
    outSlot: base.slot,
  };
}

/**
 * Bare matmul generator (the dispatchMatmul slow path — no epilogue). Reads
 * the geometry captured at lowering time (action.cachedMatmulPlan; matmul
 * inputs are liveness-released by plan-build, so live-stride transpose
 * detection can't run here). A simple-transpose input is a non-owning view
 * on the SAME buffer → its slot is the input ref's slot. Emits ALLOC(out) +
 * DISPATCH [a, b, out, params]. Bails when geometry wasn't captured (string
 * reason / undefined → uncovered, record/replay stays).
 */
function generateBareMatmul(
  cached: BareMatmulPlan | string | undefined,
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  if (cached === undefined) return "no-cached-plan";
  if (typeof cached === "string") return cached; // bail reason from capture
  if (node.inputs.length !== 2) return "arity";
  const aRef = node.inputs[0];
  const bRef = node.inputs[1];
  if (aRef.kind === "scalar" || bRef.kind === "scalar") return "scalar-operand";
  const aSlot = resolveRefSlot(aRef);
  const bSlot = resolveRefSlot(bRef);
  if (aSlot === undefined || bSlot === undefined) return "untracked-input";
  const outBytes = sizeOf(cached.outShape) * dtypeBytes(cached.outputDtype);
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const m = cached.matmul;

  if (m.kSplit) {
    // K-split: ALLOC(out) then two dispatches over the cached partials temp
    // (a persistent-slot buffer — no ALLOC, looked up by byte size). The
    // generator binds the SAME cached buffer object the recording used.
    const temp = lookupKSplitTempBuffer(m.tempBytes);
    if (!temp) return "ksplit-temp-missing";
    const tempSlot = bufferSlot(temp as unknown, "persistent");
    const ksplitParamsSlot = slots.length;
    slots.push({ kind: "params", seqIndex: -1, data: m.ksplitParamsData });
    const reduceParamsSlot = slots.length;
    slots.push({ kind: "params", seqIndex: -1, data: m.reduceParamsData });
    return {
      commands: [
        {
          tag: TAG_ALLOC,
          slot: outSlot,
          bytes: outBytes,
          allocKind: 0,
          inputSlots: [aSlot, bSlot],
        },
        {
          tag: TAG_DISPATCH,
          pipeline: m.ksplitPipeline,
          bindings: [aSlot, bSlot, tempSlot, ksplitParamsSlot],
          gx: m.ksplitDispatch[0],
          gy: m.ksplitDispatch[1],
          gz: m.ksplitDispatch[2],
        },
        {
          tag: TAG_DISPATCH,
          pipeline: m.reducePipeline,
          bindings: [tempSlot, outSlot, reduceParamsSlot],
          gx: m.reduceDispatch[0],
          gy: m.reduceDispatch[1],
          gz: m.reduceDispatch[2],
        },
      ],
      outSlot,
    };
  }

  const paramsSlot = slots.length;
  slots.push({ kind: "params", seqIndex: -1, data: m.paramsData });
  return {
    commands: [
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: outBytes,
        allocKind: 0,
        inputSlots: [aSlot, bSlot],
      },
      {
        tag: TAG_DISPATCH,
        pipeline: m.pipeline,
        bindings: [aSlot, bSlot, outSlot, paramsSlot],
        gx: m.dispatchX,
        gy: m.dispatchY,
        gz: m.dispatchZ,
      },
    ],
    outSlot,
  };
}

interface PlanNodePath {
  planNodeIndex: number;
  inputIndex: number;
}
interface MatmulEpilogueActionShape {
  matmulNodeIndex: number;
  outputNodeIndex: number;
  cachedDispatchConfig?: {
    inputAPath: PlanNodePath;
    inputBPath: PlanNodePath;
    epilogueInputPaths: PlanNodePath[];
    inputCastA?: "f16" | "f32";
    inputCastB?: "f16" | "f32";
    m: number;
    k: number;
    n: number;
    transA: boolean;
    transB: boolean;
    batchSize: number;
    batchStrideA: number;
    batchStrideB: number;
    batchStrideC: number;
    outShape: number[];
    dtypeA: "f16" | "f32";
    dtypeB?: "f16" | "f32";
    outputDtype: DType;
    epilogueConfig: unknown;
  };
}

/**
 * matmul-epilogue generator. The action's cachedDispatchConfig (populated on
 * the recording execution — the fast path that calls dispatchMatmulDirect)
 * IS the structured plan: geometry, dtypes, epilogue config, and
 * plan-node-relative input paths. Emits ALLOC(out) + the matmul DISPATCH via
 * planTiledMatmul (shared with dispatchTiledMatmul's standard tail). Bails on
 * the K-split path (op-internal temp + reduction — a later increment) and
 * when the config isn't cached yet.
 * Binding order mirrors dispatchTiledMatmul: [a, b, out, params, ...epi].
 */
function generateMatmulEpilogue(
  action: MatmulEpilogueActionShape,
  planNodes: LazyIRNode[],
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
  backend: import("../backend/types").Backend,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  const cfg = action.cachedDispatchConfig;
  if (!cfg) return "no-config";
  const device = (backend as { device?: GPUDevice }).device;
  if (!device) return "no-device";

  const pathSlot = (p: PlanNodePath): Slot | undefined => {
    const ref = planNodes[p.planNodeIndex]?.inputs[p.inputIndex];
    return ref ? resolveRefSlot(ref) : undefined;
  };
  const aSlot = pathSlot(cfg.inputAPath);
  const bSlot = pathSlot(cfg.inputBPath);
  if (aSlot === undefined || bSlot === undefined) return "untracked-input";
  const epiSlots: Slot[] = [];
  for (const p of cfg.epilogueInputPaths) {
    const s = pathSlot(p);
    if (s === undefined) return "untracked-epilogue-input";
    epiSlots.push(s);
  }

  const plan = planTiledMatmul({
    device,
    a: undefined as unknown as GPUBuffer, // planTiledMatmul reads geometry only
    b: undefined as unknown as GPUBuffer,
    out: undefined as unknown as GPUBuffer,
    m: cfg.m,
    n: cfg.n,
    k: cfg.k,
    batchSize: cfg.batchSize,
    batchStrideA: cfg.batchStrideA,
    batchStrideB: cfg.batchStrideB,
    batchStrideC: cfg.batchStrideC,
    transA: cfg.transA,
    transB: cfg.transB,
    dtype: cfg.dtypeA,
    dtypeB: cfg.dtypeB,
    epilogue: cfg.epilogueConfig as never,
    epilogueInputs: epiSlots.map(() => undefined as unknown as never), // count only
    inputCastA: cfg.inputCastA as never,
    inputCastB: cfg.inputCastB as never,
  });
  if (plan.kSplit) return "ksplit";

  const outBytes = sizeOf(cfg.outShape) * dtypeBytes(cfg.outputDtype);
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const paramsSlot = slots.length;
  slots.push({ kind: "params", seqIndex: -1, data: plan.paramsData });
  return {
    commands: [
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: outBytes,
        allocKind: 0,
        inputSlots: [aSlot, bSlot, ...epiSlots],
      },
      {
        tag: TAG_DISPATCH,
        pipeline: plan.pipeline,
        bindings: [aSlot, bSlot, outSlot, paramsSlot, ...epiSlots],
        gx: plan.dispatchX,
        gy: plan.dispatchY,
        gz: plan.dispatchZ,
      },
    ],
    outSlot,
  };
}

interface FusedActionShape {
  coveredNodeIndices: number[];
  outputNodeIndex: number;
  additionalOutputNodeIndices?: number[];
  neededIntermediateNodeIndices?: number[];
  enableVectorization?: boolean;
  cachedExternalInputPattern?: Array<{ nodeLocalIdx: number; inputIdx: number }>;
  recipe: import("../backend/webgpu/fusion-types").FusedKernelRecipe;
  runtimeScalarInputs?: Set<number>;
}

/**
 * Fused-segment generator. Covers the clean case — all external inputs
 * contiguous, no needed-intermediate re-execution — and emits the fused
 * kernel's ALLOC(s) + DISPATCH via planFusedKernel (shared with the
 * dispatcher, so binding order / workgroups can't drift). Donation is
 * detected POST-HOC: the recording already ran, so a donated out0 is the
 * input whose buffer the output node's result aliases (ownsBuffer=false) —
 * no liveness recomputation needed. Bails (uncovered) on contiguous-copy
 * prologues, needed intermediates, and oversized buffers, exactly the
 * branches executeFusedSegment routes away from the plain dispatch.
 */
function generateFused(
  action: FusedActionShape,
  planNodes: LazyIRNode[],
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
  backend: import("../backend/types").Backend,
): { commands: GpuCommand[]; outputs: Array<{ nodeIndex: number; slot: Slot }> } | string {
  if (action.neededIntermediateNodeIndices?.length) return "needed-intermediates";
  const device = (backend as { device?: GPUDevice }).device;
  if (!device) return "no-device";
  if (!action.cachedExternalInputPattern) return "no-input-pattern";

  const groupNodes = action.coveredNodeIndices.map((i) => planNodes[i]);
  const extInputs = action.cachedExternalInputPattern.map(
    (p) => groupNodes[p.nodeLocalIdx].inputs[p.inputIdx],
  );
  const recipe = action.recipe;

  // Resolve the non-inlined inputs in recipe order (mirrors
  // executeFusedSegment). planFusedKernel never dereferences input buffers
  // (only their count + the donated position), so a liveness-RELEASED
  // producer is fine: synthesize its metadata, resolve its slot via the
  // logical channel. Real buffers are kept for donation detection.
  const dispatchInputs: Array<{ buffer: GPUBuffer; shape: number[]; dtype: DType }> = [];
  const inputSlots: Slot[] = [];
  const inputBufs: Array<GPUBuffer | null> = [];
  for (let i = 0; i < recipe.inputs.length; i++) {
    if (recipe.inputs[i]?.isInlinedConstant) continue;
    const ref = extInputs[i];
    if (!ref) return "input-missing";
    let storage: StorageHandle | undefined;
    if (ref.kind === "scalar") {
      storage = getInputStorage(ref, backend); // scalar table (still active)
    } else if (ref.kind === "materialized") {
      storage = ref.storage;
    } else {
      const oi = ref.outputIndex ?? 0;
      storage = oi === 0 ? ref.node.result : ref.node.results?.[oi];
    }
    if (storage) {
      const t = storage.backendTensor as WebGPUTensor;
      if (t.isContiguous === false || (t.offset != null && t.offset > 0)) {
        return "non-contiguous"; // executor inserts a contiguous-copy prologue
      }
      if (t.buffer.size > 128 * 1024 * 1024) return "oversized";
      const slot =
        ref.kind === "scalar"
          ? bufferSlot(t.buffer as unknown, "persistent")
          : resolveRefSlot(ref);
      if (slot === undefined) return "untracked-input";
      dispatchInputs.push({
        buffer: t.buffer,
        shape: t.shape ?? [1],
        dtype: (t.dtype as DType) ?? "f32",
      });
      inputBufs.push(t.buffer);
      inputSlots.push(slot);
    } else if (ref.kind !== "materialized" && !VIEW_OPS.has(ref.node.op)) {
      // Released non-view producer: slot logical, metadata synthesized.
      const slot = resolveRefSlot(ref);
      if (slot === undefined) return "untracked-input";
      dispatchInputs.push({
        buffer: undefined as unknown as GPUBuffer,
        shape: ref.node.shape,
        dtype: (ref.node.dtype ?? "f32") as DType,
      });
      inputBufs.push(null);
      inputSlots.push(slot);
    } else {
      return "no-storage";
    }
  }

  // POST-HOC donation: out0's result buffer aliasing one of the (live) inputs.
  const outNode = planNodes[action.outputNodeIndex];
  let donatedRecipeIdx: number | undefined;
  if (outNode?.result && (outNode.result.backendTensor as WebGPUTensor).buffer) {
    const out0buf = (outNode.result.backendTensor as WebGPUTensor).buffer;
    let pos = 0;
    for (let i = 0; i < recipe.inputs.length; i++) {
      if (recipe.inputs[i]?.isInlinedConstant) continue;
      if (inputBufs[pos] === out0buf) {
        donatedRecipeIdx = i;
        break;
      }
      pos++;
    }
  }

  let plan;
  try {
    plan = planFusedKernel(device, recipe, dispatchInputs, {
      vectorize: action.enableVectorization ?? true,
      donatedInput: donatedRecipeIdx,
    });
  } catch {
    return "plan-throw";
  }

  // Output slots: the donated out0 reuses the donated input's slot; allocated
  // outputs get fresh slots. Maps recipe output index -> stream slot.
  const outputSlots: Slot[] = [];
  const commands: GpuCommand[] = [];
  for (let oi = 0; oi < plan.outputs.length; oi++) {
    const d = plan.outputs[oi];
    if ("donatedPos" in d) {
      outputSlots.push(inputSlots[d.donatedPos]);
    } else {
      const slot = slots.length;
      slots.push({ kind: "arena" });
      // inputSlots are the alloc's aliasing-check inputs (resolveOutputBuffer).
      commands.push({
        tag: TAG_ALLOC,
        slot,
        bytes: d.bytes,
        allocKind: 0,
        inputSlots: [...inputSlots],
      });
      outputSlots.push(slot);
    }
  }

  const paramsSlot = slots.length;
  slots.push({ kind: "params", seqIndex: -1, data: plan.paramsData });

  const bindings: Slot[] = plan.bindings.map((b) => {
    if (b.kind === "input") return inputSlots[b.pos];
    if (b.kind === "output") return outputSlots[b.index];
    return paramsSlot;
  });
  commands.push({
    tag: TAG_DISPATCH,
    pipeline: plan.pipeline,
    bindings,
    gx: plan.workgroups[0],
    gy: plan.workgroups[1],
    gz: plan.workgroups[2],
  });

  // Map output nodes: recipe output 0 -> outputNodeIndex, 1.. -> additional.
  const outputs: Array<{ nodeIndex: number; slot: Slot }> = [
    { nodeIndex: action.outputNodeIndex, slot: outputSlots[0] },
  ];
  const addl = action.additionalOutputNodeIndices ?? [];
  for (let i = 0; i < addl.length; i++) {
    if (outputSlots[i + 1] !== undefined) {
      outputs.push({ nodeIndex: addl[i], slot: outputSlots[i + 1] });
    }
  }
  return { commands, outputs };
}

/**
 * adam-batch generator. Mirrors the executor's adam-batch action exactly:
 * adamStepBatch packs same-element-count params (dispatchPackedOptimizer)
 * then falls back to per-item dispatch for the rest, ALL recorded under the
 * first node's index (one segment). Per packed group: scatter copies
 * (item buffers → cached packed buffers at offset), one volatile-uniform
 * packed adam dispatch, gather copies (packed → item buffers). Per fallback
 * item: an in-place adam dispatch (no alloc, non-f16). Shares the packed
 * grouping (planPackedGroups), packed buffers (lookupPackedBuffers), and
 * adam dispatch plan (planAdamStepDispatch) with the executor — no
 * reimplementation drift. Bails on f16 / chunked / non-contiguous.
 */
function generateAdamBatch(
  action: { nodeIndices: number[] },
  planNodes: LazyIRNode[],
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
  backend: import("../backend/types").Backend,
):
  | { commands: GpuCommand[]; outputs: Array<{ nodeIndex: number; slot: Slot }>; firstNodeIndex: number }
  | string {
  const device = (backend as { device?: GPUDevice }).device;
  if (!device) return "no-device";

  // Resolve each adamStep node's [grad, param, m, v]: buffer, slot, size.
  interface Item {
    nodeIndex: number;
    bufSlots: [Slot, Slot, Slot, Slot];
    numElements: number;
    config: import("../backend/types").AdamStepConfig;
  }
  const itemList: Item[] = [];
  // The recording dispatched every node whose result was unset at that time;
  // at generator time (post-recording) all results are set, so we process
  // ALL nodeIndices (the executor's `if (node.result) continue` is a
  // prior-plan dedup that doesn't apply here). firstNodeIndex is the action
  // anchor — all commands recorded under it (setRecordingNodeIndex).
  const firstNodeIndex = action.nodeIndices[0];
  for (const nodeIdx of action.nodeIndices) {
    const node = planNodes[nodeIdx];
    if (node.inputs.length !== 4) return "arity";
    // adam-batch needs only the input SLOTS + element count — never the
    // input buffers (planAdamStepDispatch reads neither, the scatter/gather
    // copies key on slots). So resolve slots LOGICALLY (resolveRefSlot
    // handles liveness-released grads via the node→slot channel) and take
    // the element count from the node shape. getInputStorage is avoided
    // precisely because a released grad would trip its lifetime guard.
    // Contiguity isn't checked here: grads (from backward) and persistent
    // state are contiguous by construction, and any inserted contiguous
    // copy would change the recorded command count → caught by the diff.
    const bufSlots: Slot[] = [];
    for (let bi = 0; bi < node.inputs.length; bi++) {
      const slot = resolveRefSlot(node.inputs[bi]);
      if (slot === undefined) return "untracked-input";
      bufSlots.push(slot);
    }
    itemList.push({
      nodeIndex: nodeIdx,
      bufSlots: bufSlots as [Slot, Slot, Slot, Slot],
      numElements: sizeOf(node.shape),
      config: node.payload as import("../backend/types").AdamStepConfig,
    });
  }
  if (itemList.length === 0) return "no-items";

  const commands: GpuCommand[] = [];
  const outputs: Array<{ nodeIndex: number; slot: Slot }> = [];
  // param/m/v are updated in place — the node's result reuses the param/m/v
  // input slot. Primary (param) = bufSlots[1].
  for (const it of itemList) {
    outputs.push({ nodeIndex: it.nodeIndex, slot: it.bufSlots[1] });
  }

  // PACKED groups (planPackedGroups mirrors dispatchPackedOptimizer's
  // element-count grouping + memory sub-batching), then per-item fallback.
  // planPackedGroups reads only numElements + buffers.length (4 for Adam).
  const packItems = itemList.map((it) => ({
    buffers: it.bufSlots as unknown as GPUBuffer[],
    numElements: it.numElements,
  }));
  const groups = planPackedGroups(packItems);
  const handled = new Set<number>();

  const adamBinding = (
    plan: import("../backend/webgpu/tile-dispatch").TileKernelPlan,
    named: Record<string, Slot>,
    infFlag: unknown | null,
  ): Slot[] | null => {
    const out: Slot[] = [];
    for (const name of plan.bindingOrder) {
      if (name === null) {
        if (!plan.configBuffer) return null;
        out.push(bufferSlot(plan.configBuffer, "persistent"));
      } else if (name === "inf_flag") {
        if (!infFlag) return null;
        out.push(bufferSlot(infFlag, "persistent"));
      } else {
        const s = named[name];
        if (s === undefined) return null;
        out.push(s);
      }
    }
    return out;
  };

  for (const g of groups) {
    const numElements = g.numElements;
    const totalElements = numElements * g.indices.length;
    const elementBytes = numElements * 4;
    const alignedBytes = alignBufferSize(totalElements * 4);
    const packed = lookupPackedBuffers(4, alignedBytes);
    if (!packed) return "packed-buffers-missing";
    const packedSlots = packed.map((b) => bufferSlot(b as unknown, "persistent"));
    // Scatter: each item's [grad,param,m,v] → packed[b] at i*elementBytes.
    for (let i = 0; i < g.indices.length; i++) {
      const it = itemList[g.indices[i]];
      for (let b = 0; b < 4; b++) {
        commands.push({
          tag: TAG_COPY,
          src: it.bufSlots[b],
          srcOffset: 0,
          dst: packedSlots[b],
          dstOffset: i * elementBytes,
          bytes: elementBytes,
        });
      }
    }
    // Packed dispatch (first item's config; the packed path forces
    // emitF16=false — dispatchAdamStepKernel(...,false,...)).
    const cfg = itemList[g.indices[0]].config;
    const infFlag = (cfg.infFlagBuffer as unknown) ?? null;
    const planned = planAdamStepDispatch(
      totalElements,
      cfg,
      infFlag as GPUBuffer | null,
      false,
    );
    if (!planned) return "adam-plan-null";
    const bindings = adamBinding(
      planned.plan,
      { grad: packedSlots[0], param: packedSlots[1], m: packedSlots[2], v: packedSlots[3] },
      infFlag,
    );
    if (!bindings) return "adam-bind-null";
    const configSlot = bufferSlot(planned.plan.configBuffer, "persistent");
    commands.push({
      tag: TAG_UNIFORM,
      slot: configSlot,
      nodeIndex: -1,
      pack: () => new Uint32Array(0),
    });
    commands.push({
      tag: TAG_DISPATCH,
      pipeline: planned.plan.pipeline,
      bindings,
      gx: planned.plan.grid[0],
      gy: planned.plan.grid[1],
      gz: planned.plan.grid[2],
    });
    // Gather: packed[b] at offset → item buffers, for b in [1,2,3].
    for (let i = 0; i < g.indices.length; i++) {
      const it = itemList[g.indices[i]];
      for (const b of [1, 2, 3]) {
        commands.push({
          tag: TAG_COPY,
          src: packedSlots[b],
          srcOffset: i * elementBytes,
          dst: it.bufSlots[b],
          dstOffset: 0,
          bytes: elementBytes,
        });
      }
      handled.add(g.indices[i]);
    }
  }

  // Fallback: per-item adam dispatch (adamStepInner). In-place param/m/v —
  // no alloc — UNLESS config.emitF16 (large weights kept in f16 for the
  // forward pass): then a param_f16 OUTPUT buffer is allocated (kind 1) and
  // bound. Matches adamStepInner's `emitF16 = config.emitF16 ?? false`.
  for (let idx = 0; idx < itemList.length; idx++) {
    if (handled.has(idx)) continue;
    const it = itemList[idx];
    const infFlag = (it.config.infFlagBuffer as unknown) ?? null;
    const emitF16 = (it.config as { emitF16?: boolean }).emitF16 ?? false;
    const planned = planAdamStepDispatch(
      it.numElements,
      it.config,
      infFlag as GPUBuffer | null,
      emitF16,
    );
    if (!planned) return "adam-plan-null";
    const named: Record<string, Slot> = {
      grad: it.bufSlots[0],
      param: it.bufSlots[1],
      m: it.bufSlots[2],
      v: it.bufSlots[3],
    };
    // f16 weight output: allocateOutputBuffer(numElements*2) = allocKind 1.
    if (planned.doF16) {
      const f16Slot = slots.length;
      slots.push({ kind: "arena" });
      commands.push({
        tag: TAG_ALLOC,
        slot: f16Slot,
        bytes: planned.f16Bytes,
        allocKind: 1,
        inputSlots: [],
      });
      named.param_f16 = f16Slot;
    }
    const bindings = adamBinding(planned.plan, named, infFlag);
    if (!bindings) return "adam-bind-null";
    const configSlot = bufferSlot(planned.plan.configBuffer, "persistent");
    commands.push({
      tag: TAG_UNIFORM,
      slot: configSlot,
      nodeIndex: -1,
      pack: () => new Uint32Array(0),
    });
    commands.push({
      tag: TAG_DISPATCH,
      pipeline: planned.plan.pipeline,
      bindings,
      gx: planned.plan.grid[0],
      gy: planned.plan.grid[1],
      gz: planned.plan.grid[2],
    });
  }

  return { commands, outputs, firstNodeIndex };
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
