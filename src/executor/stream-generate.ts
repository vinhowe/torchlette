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

import { planAdamStepDispatch } from "../backend/webgpu/adam-kernel";
import {
  type AttentionStepPlan,
  planFlashAttentionBackward,
  planFlashAttentionForward,
} from "../backend/webgpu/attention-kernel";
import {
  planCrossEntropyBackwardDispatch,
  planCrossEntropyForwardDispatch,
} from "../backend/webgpu/cross-entropy-kernel";
import type { BareMatmulPlan } from "../backend/webgpu/dispatch";
import {
  getPipeline,
  planBinaryDirect,
  planChunkedBinary,
  planChunkedUnary,
  planUnaryDirect,
} from "../backend/webgpu/dispatch";
import type { TileKernelChunkedPlan } from "../backend/webgpu/tile-dispatch";
import { planFusedKernel } from "../backend/webgpu/fusion-dispatch";
import { requireContext } from "../backend/webgpu/gpu-context";
import { gpuBuffer } from "../backend/webgpu/gpu-types";
import {
  planLayerNormBackwardGradWeightBias,
  planLayerNormBackwardGradXDispatch,
  planLayerNormForwardDispatch,
} from "../backend/webgpu/layernorm-kernel";
import {
  lookupKSplitTempBuffer,
  planTiledMatmul,
} from "../backend/webgpu/matmul/dispatch";
import {
  planGatherDirect,
  planScatterAddDirect,
} from "../backend/webgpu/ops/gather-scatter";
import {
  fullReductionNeedsChunking,
  planChunkedFullReduction,
  planDimReductionDispatch,
  planFullReductionDispatch,
  planMeanDivDispatch,
} from "../backend/webgpu/ops/reductions";
import {
  deriveNodeOffset,
  type MetaNodeLike,
  VIEW_META_OPS,
  type ViewMeta,
} from "../backend/webgpu/ops/view-meta";
import {
  planCastDirect,
  planContiguousDirect,
  planNarrowBackward,
} from "../backend/webgpu/ops/views";
import { planWhereDirect } from "../backend/webgpu/ops/where";
import { planRowProgramDispatch } from "../backend/webgpu/row-program-dispatch";
import { alignBufferSize, dtypeBytes } from "../backend/webgpu/shape-utils";
import type { WebGPUTensor } from "../backend/webgpu/tensor";
import type { TileKernelPlan } from "../backend/webgpu/tile-dispatch";
import { planUnscaleGradDispatch } from "../backend/webgpu/unscale-kernel";
import { sizeOf } from "../core/shape";
import type { LazyIRNode, StorageHandle } from "../graph/types";
import {
  lookupPackedBuffers,
  planPackedGroups,
} from "../optim/packed-dispatch";
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
import { operandRequiresContiguous } from "./contiguous-operands";
import type { AttnInputContig, LoweredPlan } from "./lowered-plan";
import { getInputStorage } from "./op-dispatch";
import { lookupScalarStorage } from "./scalar-table";

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
  /** plan-node index → its primary-output (outputIndex 0) generated slot.
   *  Used by the phase-4 cutover to build NodeResult slots from the generated
   *  stream instead of the recording's buffer→slot map. */
  nodeSlot: Map<number, Slot>;
  /** `${nodeIndex}:${outputIndex}` → generated slot, for multi-output extras. */
  nodeSlotExtra: Map<string, Slot>;
}

const F32_BYTES = 4;

/**
 * Count of GEMV (M=1) matmul dispatches emitted into GENERATED command streams
 * (bare matmul + matmul-epilogue directive). The `getGemvDispatchCount` counter
 * in matmul/dispatch.ts only ticks on the LOWERED path (dispatchTiledMatmul) —
 * it is REPLAY-BLIND, so it reads 0 once a decode template cuts over to the
 * compiled/generated plan even when the GEMV route engages perfectly. This
 * counter is the route-engagement signal that SURVIVES the compiled-plan
 * activation threshold: it ticks each time the generator bakes a `_gemv`-labeled
 * plan into a TAG_DISPATCH. A route-engagement gate must read THIS (not the
 * lowered counter) to detect a #93/#95-class bypass in the generated stream.
 * See test/gemv-generated-route.spec.ts.
 */
let generatedGemvDispatchCount = 0;
export function getGeneratedGemvDispatchCount(): number {
  return generatedGemvDispatchCount;
}

/**
 * Generate the command stream for a lowered plan. Mirrors the recording's
 * slot-numbering protocol exactly (it must — the differential compares
 * canonical streams where buffers ARE slot indices):
 *  1. externals pre-assigned in (planNodeIndex, inputIndex) order, deduped
 *     by buffer identity (executor.ts pre-assignment loop);
 *  2. allocs take fresh slots in action order (recordAlloc lifetime split
 *     always yields a fresh slot for a fresh logical lifetime).
 */
export interface GenerateStreamOptions {
  /** When true (RECURRING template, build-reach ≥2) a compile-time-constant
   *  `full([...], const)` is covered via a plan-owned constFill slot instead
   *  of bailing. Transient warmup plans (reach 1) pass false so they stay
   *  lowered (a one-shot plan-owned buffer would sit until idle-retire). */
  coverConstFill?: boolean;
  /** When true (the BUILD path), a row-program whose 0-d materialized
   *  cross-plan input (clipGradNorm_'s clipCoef) is STAMPED emits the fused
   *  kernel — the external slot resolves the CURRENT step's ref at replay and
   *  the stamp is asserted at the bind seam. The VERIFY path
   *  (TORCHLETTE_STREAM_GENERATE=1) passes false: it diffs against a RECORDING
   *  whose executor took the sequential fallback (mul + sum, two dispatches)
   *  for exactly this input class, so generating the fused kernel there would
   *  be a spurious segment DIVERGE, not a bug. */
  fuseStampedScalarExternals?: boolean;
}

export function generateStream(
  loweredPlan: LoweredPlan,
  planNodes: LazyIRNode[],
  backend?: import("../backend/types").Backend,
  opts: GenerateStreamOptions = {},
): GeneratedStream {
  const coverConstFill = opts.coverConstFill ?? false;
  const fuseStampedScalarExternals = opts.fuseStampedScalarExternals ?? false;
  const commands: GpuCommand[] = [];
  const slots: SlotSource[] = [];
  const uncovered = new Map<string, number>();
  const bufferToSlot = new Map<unknown, Slot>();
  const nodeIndexById = new Map<number, number>();
  for (let i = 0; i < planNodes.length; i++) {
    nodeIndexById.set(planNodes[i].id, i);
  }

  // Nodes PRODUCED within this plan (executed by some action — every index an
  // action emits/covers). The recording's external pre-assignment runs
  // PRE-execution and assigns external slots only to refs whose storage already
  // exists, i.e. NOT produced by this plan. The generator runs POST-execution
  // (all nodes have results), so it can't use "result exists"; instead it skips
  // exactly the produced-within nodes. Using "in planNodes" as the proxy was
  // WRONG — planNodes also holds LEAF inputs (e.g. the i32 input tokens) that
  // are in-plan yet have NO producing action; those are true externals and must
  // get an external slot, or their (and their consumers') slots go untracked.
  const producedNodes = new Set<number>();
  const addProduced = (...idxs: (number | undefined)[]) => {
    for (const x of idxs) if (x !== undefined) producedNodes.add(x);
  };
  for (const a of loweredPlan.actions) {
    const aa = a as {
      nodeIndex?: number;
      nodeIndices?: number[];
      outputNodeIndex?: number;
      matmulNodeIndex?: number;
      coveredNodeIndices?: number[];
      additionalOutputNodeIndices?: number[];
      neededIntermediateNodeIndices?: number[];
    };
    addProduced(aa.nodeIndex, aa.outputNodeIndex, aa.matmulNodeIndex);
    for (const arr of [
      aa.nodeIndices,
      aa.coveredNodeIndices,
      aa.additionalOutputNodeIndices,
      aa.neededIntermediateNodeIndices,
    ])
      if (arr) for (const x of arr) producedNodes.add(x);
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
        // Skip refs PRODUCED within this plan (they get their slot from their
        // own action). A ref whose producer is NOT produced-within is a true
        // external (matched to the recording's "storage exists pre-execution":
        // leaf inputs + prior-plan results), and gets an external slot. Using
        // produced-within (not merely in-plan) is essential — in-plan LEAF
        // inputs (the i32 tokens) have no action and ARE external.
        const pidx = nodeIndexById.get(ref.node.id);
        if (pidx !== undefined && producedNodes.has(pidx)) continue;
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
  // Extra-output channel: `${nodeIndex}:${outputIndex}` → slot, for
  // multi-output producers (attention dQ/dK/dV, logsumexp) whose non-primary
  // outputs are released by plan-build and have no recoverable buffer.
  const nodeSlotExtra = new Map<string, Slot>();
  const resolveRefSlot = (
    ref: LazyIRNode["inputs"][number],
  ): Slot | undefined => {
    if (ref.kind === "scalar") return undefined;
    if (ref.kind !== "materialized") {
      const oi = ref.outputIndex ?? 0;
      const idx = nodeIndexById.get(ref.node.id);
      if (idx !== undefined) {
        const s =
          oi === 0 ? nodeSlot.get(idx) : nodeSlotExtra.get(`${idx}:${oi}`);
        if (s !== undefined) return s;
      }
      const storage = oi === 0 ? ref.node.result : ref.node.results?.[oi];
      if (!storage) return undefined;
      return bufferToSlot.get(gpuBuffer(storage.backendTensor) as unknown);
    }
    return bufferToSlot.get(gpuBuffer(ref.storage.backendTensor) as unknown);
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
        // reshape (uniquely among the view ops) MATERIALIZES a contiguous
        // copy when its input is non-contiguous (e.g. reshape-of-permute in
        // attention's grad reshaping) — the backend does contiguous()+reshape.
        // transpose/permute/expand/narrow are pure stride math, never copy.
        // The copy is recorded under THIS node; mirror it from the captured
        // input layout. When the input is contiguous reshape is a free view
        // (falls through to the alias path below).
        const vi = (action as { cachedViewInput?: AttnInputContig })
          .cachedViewInput;
        if (node && node.op === "reshape" && vi && !vi.contiguous) {
          const inSlot = resolveRefSlot(node.inputs[0]);
          if (inSlot === undefined) {
            miss("view:reshape-copy[untracked-input]");
            phantom(node, action.nodeIndex);
            break;
          }
          const cc = planContigCopy(vi, inSlot, slots);
          if (typeof cc === "string") {
            miss(`view:reshape-copy[${cc}]`);
            phantom(node, action.nodeIndex);
            break;
          }
          commands.push(...cc.commands);
          segments.push({ nodeIndex: action.nodeIndex, commands: cc.commands });
          mapNodeResult(node, cc.outSlot, bufferToSlot);
          nodeSlot.set(action.nodeIndex, cc.outSlot);
          coveredActions++;
          break;
        }
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
        const gen = generateDataSource(
          node,
          action.nodeIndex,
          slots,
          coverConstFill,
        );
        if (!gen) {
          miss(
            `data-source:${node.op}${node.dtype !== "f32" ? `:${node.dtype}` : ""}`,
          );
          phantom(node, action.nodeIndex);
          break;
        }
        commands.push(...gen.commands);
        // A constFill data-source emits NO commands (its buffer is born at
        // build and populated at replay from the slot source, replacing the
        // recorded TAG_WRITE). Emitting an empty segment for it would compare
        // 0 gen commands against the recorded 1 write → a spurious count
        // divergence. Instead emit no segment (like an alias view) and let the
        // constFill-aware flat-count check reconcile the delta.
        if (gen.commands.length > 0)
          segments.push({
            nodeIndex: action.nodeIndex,
            commands: gen.commands,
          });
        mapNodeResult(node, gen.outSlot, bufferToSlot);
        nodeSlot.set(action.nodeIndex, gen.outSlot);
        coveredActions++;
        break;
      }
      case "sequential": {
        const node = planNodes[action.nodeIndex];
        // Multi-output attention (output+logsumexp; dQ/dK/dV) — dedicated
        // generators that return per-output slot mappings.
        if (
          node.op === "fusedAttentionForward" ||
          node.op === "fusedAttentionBackward"
        ) {
          // Non-contiguous inputs get a contiguous-copy prologue (the op
          // asContiguous's them); generateAttention emits it from the
          // per-input layout captured at lowering.
          const ag = generateAttention(
            node,
            slots,
            resolveRefSlot,
            bufferSlot,
            (action as { cachedInputContig?: AttnInputContig[] })
              .cachedInputContig,
          );
          if (typeof ag === "string") {
            miss(`op:${node.op}[${ag}]`);
            phantom(node, action.nodeIndex);
            break;
          }
          commands.push(...ag.commands);
          segments.push({ nodeIndex: action.nodeIndex, commands: ag.commands });
          for (const o of ag.outputs) {
            if (o.outputIndex === 0) {
              mapNodeResult(node, o.slot, bufferToSlot);
              nodeSlot.set(action.nodeIndex, o.slot);
            } else {
              nodeSlotExtra.set(`${action.nodeIndex}:${o.outputIndex}`, o.slot);
            }
          }
          coveredActions++;
          break;
        }
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
            : generateSequential(
                node,
                slots,
                resolveRefSlot,
                bufferSlot,
                (action as { cachedInputShapes?: number[][] })
                  .cachedInputShapes,
                (action as { cachedStridedInputs?: (AttnInputContig | null)[] })
                  .cachedStridedInputs,
              );
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
          segments.push({
            nodeIndex: action.nodeIndex,
            commands: gen.commands,
          });
        }
        mapNodeResult(node, gen.outSlot, bufferToSlot);
        nodeSlot.set(action.nodeIndex, gen.outSlot);
        // Multi-output sequential ops (e.g. layernorm-bwd grad_weight/grad_bias)
        // expose their non-primary outputs so the harvest carries them — else
        // the cutover drops those results and downstream consumers replay stale
        // (the optimizer reading grad_bias). Extras go to nodeSlotExtra only —
        // same convention as the adam-batch / attention multi-output paths;
        // resolveRefSlot checks nodeSlotExtra first for outputIndex != 0.
        if (gen.extraOutputs) {
          for (const eo of gen.extraOutputs) {
            nodeSlotExtra.set(`${action.nodeIndex}:${eo.outputIndex}`, eo.slot);
          }
        }
        coveredActions++;
        break;
      }
      case "fused": {
        const fa = action as FusedActionShape;
        const gen = backend
          ? generateFused(
              fa,
              planNodes,
              slots,
              resolveRefSlot,
              bufferSlot,
              backend,
            )
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
        segments.push({
          nodeIndex: fa.outputNodeIndex,
          commands: gen.commands,
        });
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
          ? generateMatmulEpilogue(
              me,
              planNodes,
              slots,
              resolveRefSlot,
              bufferSlot,
              backend,
            )
          : "no-backend";
        if (typeof gen === "string") {
          miss(`matmul-epilogue[${gen}]`);
          phantom(planNodes[me.outputNodeIndex], me.outputNodeIndex);
          break;
        }
        commands.push(...gen.commands);
        segments.push({
          nodeIndex: me.matmulNodeIndex,
          commands: gen.commands,
        });
        mapNodeResult(planNodes[me.outputNodeIndex], gen.outSlot, bufferToSlot);
        nodeSlot.set(me.outputNodeIndex, gen.outSlot);
        coveredActions++;
        break;
      }
      case "adam-batch": {
        const ab = action as { nodeIndices: number[] };
        const gen = backend
          ? generateAdamBatch(
              ab,
              planNodes,
              slots,
              resolveRefSlot,
              bufferSlot,
              backend,
            )
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
        segments.push({
          nodeIndex: gen.firstNodeIndex,
          commands: gen.commands,
        });
        for (const o of gen.outputs) {
          // adamStep is multi-output (param oi0, m oi1, v oi2). Primary → the
          // node→slot + buffer→slot channels; m/v extras → the extra channel so
          // the harvest can carry persistent optimizer state across steps.
          if (o.outputIndex === 0) {
            mapNodeResult(planNodes[o.nodeIndex], o.slot, bufferToSlot);
            nodeSlot.set(o.nodeIndex, o.slot);
          } else {
            nodeSlotExtra.set(`${o.nodeIndex}:${o.outputIndex}`, o.slot);
          }
        }
        coveredActions++;
        break;
      }
      case "batched-reduction": {
        const br = action as {
          nodeIndices: number[];
          reduceOp: string;
          payload: { dim: number | number[]; keepdim?: boolean };
        };
        const gen = generateBatchedReduction(
          br,
          planNodes,
          slots,
          resolveRefSlot,
          bufferSlot,
        );
        if (typeof gen === "string") {
          miss(`batched-reduction[${gen}]`);
          for (const ni of br.nodeIndices) phantom(planNodes[ni], ni);
          break;
        }
        commands.push(...gen.commands);
        segments.push({
          nodeIndex: gen.firstNodeIndex,
          commands: gen.commands,
        });
        for (const o of gen.outputs) {
          mapNodeResult(planNodes[o.nodeIndex], o.slot, bufferToSlot);
          nodeSlot.set(o.nodeIndex, o.slot);
        }
        coveredActions++;
        break;
      }
      case "row-program": {
        const rp = action as {
          program: import("../compiler/row-program-types").RowProgram;
          inputRefs: LazyIRNode["inputs"];
          inputRefConsumerPositions: Array<{ pos: number; inputIndex: number }>;
          outputNodeIndex: number;
          numRows: number;
          dimSize: number;
        };
        const gen = generateRowProgram(
          rp,
          planNodes,
          slots,
          resolveRefSlot,
          bufferSlot,
          fuseStampedScalarExternals,
        );
        if (typeof gen === "string") {
          miss(`row-program[${gen}]`);
          phantom(planNodes[rp.outputNodeIndex], rp.outputNodeIndex);
          break;
        }
        commands.push(...gen.commands);
        segments.push({
          nodeIndex: rp.outputNodeIndex,
          commands: gen.commands,
        });
        mapNodeResult(planNodes[rp.outputNodeIndex], gen.outSlot, bufferToSlot);
        nodeSlot.set(rp.outputNodeIndex, gen.outSlot);
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
    nodeSlot,
    nodeSlotExtra,
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

/**
 * Emit a contiguous-copy (ensureContiguous / reshape-of-strided / attention
 * asContiguous) from a captured input LAYOUT: resolveOutputBuffer ALLOC
 * (kind 0, the original as the aliasing input) + planContiguousDirect dispatch
 * [orig, copy, params]. Returns the copy's output slot, or a bail string when
 * the buffer exceeds maxBindingSize (chunked contiguous — a different command
 * pattern). The plan is the SAME one the live path uses (shared getPipeline
 * cache → identical pipeline identity for the diff).
 */
function planContigCopy(
  info: AttnInputContig,
  inSlot: Slot,
  slots: SlotSource[],
): { commands: GpuCommand[]; outSlot: Slot } | string {
  const meta = {
    shape: info.shape,
    strides: info.strides,
    offset: info.offset,
    dtype: info.dtype,
    size: sizeOf(info.shape),
    isContiguous: false,
    buffer: { size: info.bufferSize },
  } as unknown as WebGPUTensor;
  const cp = planContiguousDirect(meta);
  if (!cp) return "contig-chunked";
  const pipeline = getPipeline(requireContext(), cp.key, cp.shader);
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const paramsSlot = slots.length;
  slots.push({ kind: "params", seqIndex: -1, data: cp.paramsData.slice() });
  return {
    commands: [
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: cp.outputSizeBytes,
        allocKind: 0,
        inputSlots: [inSlot],
      },
      {
        tag: TAG_DISPATCH,
        pipeline,
        bindings: [inSlot, outSlot, paramsSlot],
        gx: cp.dispatchX,
        gy: cp.dispatchY,
        gz: 1,
      },
    ],
    outSlot,
  };
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
/** A sequential generator's result. `outSlot` is the primary (outputIndex 0)
 *  result slot; `extraOutputs` carries non-primary outputs for multi-output
 *  sequential ops (e.g. fusedLayerNormBackwardGradWeightBias's grad_bias at
 *  outputIndex 1) so the walker can map them into nodeSlotExtra — without this
 *  the cutover harvest drops those results and downstream consumers (the
 *  optimizer reading grad_bias) silently replay stale values. */
interface SequentialGen {
  commands: GpuCommand[];
  outSlot: Slot;
  extraOutputs?: Array<{ outputIndex: number; slot: Slot }>;
}

function generateSequential(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
  cachedInputShapes?: number[][],
  cachedStridedInputs?: (AttnInputContig | null)[],
): SequentialGen | string {
  // Tile-kernel ops with bespoke command patterns.
  if (node.op === "sum")
    return generateFullReduction(node, slots, resolveRefSlot, bufferSlot);
  if (node.op === "unscaleGrad")
    return generateUnscaleGrad(node, slots, resolveRefSlot, bufferSlot);
  if (node.op === "stridedScatterCopy")
    return generateScatterCopyDMA(node, resolveRefSlot, bufferSlot);
  if (node.op === "fusedLayerNormForward")
    return generateLayerNormForward(node, slots, resolveRefSlot, bufferSlot);
  if (node.op === "gather") return generateGather(node, slots, resolveRefSlot);
  if (node.op === "cat") return generateCat(node, slots, resolveRefSlot);
  if (node.op === "scatterAdd")
    return generateScatterAdd(node, slots, resolveRefSlot);
  if (node.op === "fusedLayerNormBackwardGradX")
    return generateLayerNormGradX(node, slots, resolveRefSlot, bufferSlot);
  if (node.op === "fusedLayerNormBackwardGradWeightBias")
    return generateLayerNormGradWB(node, slots, resolveRefSlot, bufferSlot);
  if (node.op === "narrowBackward")
    return generateNarrowBackward(
      node,
      slots,
      resolveRefSlot,
      cachedInputShapes,
    );
  if (node.op === "fusedCrossEntropyForward")
    return generateCrossEntropyForward(node, slots, resolveRefSlot, bufferSlot);
  if (node.op === "fusedCrossEntropyBackward")
    return generateCrossEntropyBackward(
      node,
      slots,
      resolveRefSlot,
      bufferSlot,
    );
  if (node.op === "mean")
    return generateMean(node, slots, resolveRefSlot, bufferSlot);
  const wgslOp = BINARY_OPS.get(node.op);
  const isUnary =
    UNARY_OPS.has(node.op) ||
    node.op === "cast" ||
    node.op === "contiguous" ||
    node.op === "gelu";
  // `where` is a 3-input elementwise select (whereDirect): the input-resolution
  // loop below is already arity-generic, so covering it is a plan-builder branch
  // (planWhereDirect) — no new resolution machinery.
  const isWhere = node.op === "where";
  if (!wgslOp && !isUnary && !isWhere) return "";
  if (wgslOp && node.inputs.length !== 2) return "arity";
  if (isUnary && node.inputs.length !== 1) return "arity";
  if (isWhere && node.inputs.length !== 3) return "arity";
  const ins: WebGPUTensor[] = [];
  const inSlots: Slot[] = [];
  // Aliasing candidates for the output ALLOC: only POOLABLE operands. Scalar-
  // table buffers are persistent and resolveOutputBuffer excludes them, so
  // they must NOT appear in the recorded ALLOC's inputSlots (caught by diff).
  const aliasInSlots: Slot[] = [];
  for (let inputIdx = 0; inputIdx < node.inputs.length; inputIdx++) {
    const ref = node.inputs[inputIdx];
    if (ref.kind === "scalar") {
      // Scalar operand: resolves through the plan's scalar table to a
      // PERSISTENT buffer (stable identity; the executor refreshes its value
      // from the current step before execution — so it binds as a persistent
      // slot, not a stream write). The legacy CPU/non-f32 fallback path
      // (lookupScalarStorage miss) materializes a fresh full([],v) and isn't
      // generatable — bail.
      const sh = lookupScalarStorage(ref);
      if (!sh) return "scalar-no-table";
      const t = sh.backendTensor as WebGPUTensor;
      ins.push(t);
      inSlots.push(bufferSlot(gpuBuffer(t), "persistent"));
      continue;
    }
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
        : (ref.outputIndex ?? 0) === 0
          ? ref.node.result
          : ref.node.results?.[ref.outputIndex ?? 0];
    let meta: WebGPUTensor;
    if (storage) {
      meta = storage.backendTensor as WebGPUTensor;
    } else if (
      ref.kind !== "materialized" &&
      (ref.outputIndex ?? 0) === 0 &&
      // Non-view producers allocate fresh (contiguous, offset 0). Contiguous-
      // view producers (reshape/view/flatten/squeeze/unsqueeze) share the
      // base buffer but their logical layout is ALSO contiguous-offset-0, so
      // the synthesized metadata is correct; the binding's buffer comes from
      // the resolved slot. Strided views stay bailed.
      (!VIEW_OPS.has(ref.node.op) || CONTIGUOUS_VIEW_OPS.has(ref.node.op))
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
    } else if (cachedStridedInputs?.[inputIdx]) {
      // Released STRIDED-view producer (expand/transpose/permute/narrow): its
      // stride-bearing layout (incl. broadcast stride-0) was captured live at
      // lowering, so synthesize the real strided metadata — planBinaryDirect/
      // planUnaryDirect then emit the matching strided/broadcast kernel (the
      // cache key includes strides, so the pipeline identity matches the
      // recording). The binding's buffer comes from the resolved slot (the
      // view aliases its base).
      const cap = cachedStridedInputs[inputIdx];
      meta = {
        shape: cap.shape,
        strides: cap.strides,
        offset: cap.offset,
        dtype: cap.dtype as WebGPUTensor["dtype"],
        size: sizeOf(cap.shape),
        isContiguous: false,
        buffer: { size: cap.bufferSize },
      } as unknown as WebGPUTensor;
    } else {
      // Remaining no-storage producers are strided views with no captured
      // layout — not shape-derivable, stay bailed.
      return "no-storage";
    }
    const slot = resolveRefSlot(ref);
    if (slot === undefined) return "untracked-producer";
    ins.push(meta);
    inSlots.push(slot);
    aliasInSlots.push(slot);
  }
  // Oversized (>maxStorageBufferBindingSize) elementwise → chunked dispatch.
  // dispatchBinary/dispatchUnary route these to the chunked path (per-chunk
  // sub-range bindings); planBinaryDirect/planUnaryDirect BAIL at >128 MB by
  // construction. Mirror the routing exactly and derive the per-chunk commands
  // from the SAME plan the execution path dispatches (planChunkedBinary/Unary →
  // planChunked → computeChunkGeometry). This is the foreach optimizer's
  // oversized packed-buffer decomposition (add/div/sub/mul/sqrt/neg over the
  // 320 MB m/v-class buffer). Only the contiguous-offset-0 case is chunkable
  // (the dispatch forces contiguity otherwise, inserting uncaptured copies), so
  // a strided/offset input stays an honest miss.
  const dtype = (node.dtype ?? "f32") as WebGPUTensor["dtype"];
  const outSize = sizeOf(node.shape);
  const maxBinding =
    requireContext().device.limits?.maxStorageBufferBindingSize ??
    128 * 1024 * 1024;
  const outOversized = outSize * dtypeBytes(dtype) > maxBinding;
  if (outOversized && (wgslOp || (isUnary && node.op !== "cast"))) {
    const contigOk = (m: WebGPUTensor) =>
      m.isContiguous !== false && (m.offset ?? 0) === 0;
    if (!ins.every(contigOk)) return "chunked-noncontig";
    if (wgslOp) {
      const aIsScalar = ins[0].size === 1;
      const bIsScalar = ins[1].size === 1;
      return generateChunkedBinary(
        wgslOp,
        aIsScalar,
        bIsScalar,
        outSize,
        dtype,
        inSlots,
        aliasInSlots,
        slots,
      );
    }
    // Unary chunked (sqrt/neg/…): gelu maps to its opKey; plain unary uses node.op.
    const opKey =
      node.op === "gelu"
        ? ((node.payload as { approximate?: string } | undefined)?.approximate ??
            "tanh") === "tanh"
          ? "gelu"
          : "gelu_erf"
        : node.op;
    return generateChunkedUnary(
      opKey,
      outSize,
      dtype,
      inSlots[0],
      aliasInSlots,
      slots,
    );
  }
  let plan;
  if (wgslOp) {
    plan = planBinaryDirect(wgslOp, ins[0], ins[1]);
  } else if (isWhere) {
    // The direct `where` carries frozen offsets in paramsData with NO volatile
    // repack (buildDirectOffsetRepack excludes it — clip/scaler run it on
    // offset-0 tensors). A narrow-fed input's offset could vary across replays,
    // stranding the frozen params stale → keep such a `where` uncovered.
    if (node.inputs.some((r) => inputHasNarrow(r))) return "where-narrow";
    plan = planWhereDirect(ins[0], ins[1], ins[2]);
  } else if (node.op === "cast") {
    plan = planCastDirect(ins[0], node.dtype);
  } else if (node.op === "contiguous") {
    // Already-contiguous fast path: a non-owning view, ZERO commands — the
    // node's result shares the input buffer (mapNodeResult will re-map the
    // same buffer to the input's slot; out slot is the input's).
    if (ins[0].isContiguous && (ins[0].offset ?? 0) === 0) {
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
    data: plan.paramsData.slice(),
  });
  const commands: GpuCommand[] = [
    {
      tag: TAG_ALLOC,
      slot: outSlot,
      bytes: plan.outputSizeBytes,
      allocKind: 0,
      // Only poolable operands are aliasing candidates; scalar-table buffers
      // are persistent and excluded by resolveOutputBuffer.
      inputSlots: aliasInSlots,
    },
  ];
  // Task #71: if any input's offset can VARY across replays (its view chain
  // contains a narrow), the frozen paramsData holds only the record-time
  // offset — a start-N replay of a start-0-built template would read the wrong
  // region (the falsified shortcut). Emit a TAG_UNIFORM that re-derives every
  // input offset from the CURRENT step's node and rewrites the params buffer
  // (params layout: [size, ...offsets] in uniform-declaration order). Emitted
  // BEFORE the dispatch to match the recording order (createParamsBuffer runs
  // the repack at params-buffer creation, before dispatchComputePass encodes).
  // nodeIndex -1 is patched to this node's plan index by the caller.
  const offsetRepack = buildDirectOffsetRepack(node);
  if (offsetRepack) {
    commands.push({
      tag: TAG_UNIFORM,
      slot: paramsSlot,
      nodeIndex: -1,
      pack: offsetRepack,
    });
  }
  commands.push({
    tag: TAG_DISPATCH,
    pipeline,
    bindings: [...inSlots, outSlot, paramsSlot],
    gx: plan.dispatchX,
    gy: plan.dispatchY,
    gz: 1,
  });
  return { commands, outSlot };
}

/**
 * Emit the ALLOC + per-chunk DISPATCH commands for a chunked elementwise op
 * (binary/unary >maxStorageBufferBindingSize). Consumes a TileKernelChunkedPlan
 * derived from the SAME chunkedBinaryConfig/chunkedUnaryConfig the execution
 * path dispatches (planChunkedBinary/Unary → planChunked → computeChunkGeometry)
 * — the single source for the capability-forced split. Mirrors
 * generateChunkedFullReduction's command shape: one ALLOC then one DISPATCH per
 * chunk with sub-range bindings + a FRESH params slot per chunk. One-per-chunk
 * (no dedup) mirrors the execution path's createParamsBuffer, which advances its
 * sequence index per chunk → a distinct recorded `params` slot per chunk even
 * when same-size chunks carry identical bytes; the params-data MULTISET still
 * matches (same byte pattern, same count).
 */
function emitChunkedElementwise(
  plan: TileKernelChunkedPlan,
  inputNames: string[],
  inSlots: Slot[],
  aliasInSlots: Slot[],
  outBytes: number,
  slots: SlotSource[],
): SequentialGen {
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const commands: GpuCommand[] = [
    {
      tag: TAG_ALLOC,
      slot: outSlot,
      bytes: outBytes,
      allocKind: 0,
      inputSlots: aliasInSlots,
    },
  ];
  const nameToSlot: Record<string, Slot> = {};
  for (let i = 0; i < inputNames.length; i++) nameToSlot[inputNames[i]] = inSlots[i];
  for (const chunk of plan.chunks) {
    const pslot = slots.length;
    // params data is a u32 view of the packed uniform bytes (matches the
    // createParamsBuffer(Uint32Array) the execution path writes).
    slots.push({
      kind: "params",
      seqIndex: -1,
      data: new Uint32Array(
        chunk.paramsData.buffer.slice(
          chunk.paramsData.byteOffset,
          chunk.paramsData.byteOffset + chunk.paramsData.byteLength,
        ),
      ),
    });
    const bindings: Slot[] = plan.bindingOrder.map((name) =>
      name === null ? pslot : name === "out" ? outSlot : nameToSlot[name],
    );
    commands.push({
      tag: TAG_DISPATCH,
      pipeline: plan.pipeline,
      bindings,
      bindingRanges: chunk.ranges,
      gx: chunk.grid[0],
      gy: chunk.grid[1],
      gz: chunk.grid[2],
    });
  }
  return { commands, outSlot };
}

/**
 * cat generator — mirrors the backend `cat` (ops/gather-scatter.ts) byte-for-
 * byte: resolveOutputBuffer's ALLOC then one recordedCopyBufferToBuffer per
 * (input, outer-position). The foreach optimizer packs per-param flats into one
 * [total] buffer (dim 0, rank-1 → outerSize 1, one copy per input); the general
 * strided loop is reproduced so any cat matches. An input that the backend would
 * ensureContiguous (non-contiguous, or an offset not 4-byte-aligned) inserts an
 * uncaptured copy → stays an honest miss. A released view producer (no
 * recoverable layout) also bails.
 */
function generateCat(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  const payload = node.payload as { dim?: number } | undefined;
  const outShape = node.shape;
  const rank = outShape.length;
  const dim = payload?.dim ?? 0;
  const dtype = (node.dtype ?? "f32") as WebGPUTensor["dtype"];
  const bpe = dtypeBytes(dtype);

  // Resolve each input's slot + (shape, offset). Live storage gives the real
  // layout; a released NON-VIEW producer allocated fresh (contiguous, offset 0)
  // so its shape is the node shape; strided/offset views stay bailed.
  interface CatIn {
    slot: Slot;
    shape: number[];
    offset: number;
  }
  const ins: CatIn[] = [];
  for (const ref of node.inputs) {
    if (ref.kind === "scalar") return "scalar-operand";
    const slot = resolveRefSlot(ref);
    if (slot === undefined) return "untracked-input";
    const storage =
      ref.kind === "materialized"
        ? ref.storage
        : (ref.outputIndex ?? 0) === 0
          ? ref.node.result
          : ref.node.results?.[ref.outputIndex ?? 0];
    if (storage) {
      const t = storage.backendTensor as WebGPUTensor;
      const off = t.offset ?? 0;
      // The backend ensureContiguous's a non-contiguous input or an offset not
      // 4-byte-aligned (inserting an uncaptured copy) — bail to match.
      if (t.isContiguous === false || (off * bpe) % 4 !== 0)
        return "cat-noncontig";
      ins.push({ slot, shape: t.shape, offset: off });
    } else if (ref.kind !== "materialized" && !VIEW_OPS.has(ref.node.op)) {
      ins.push({ slot, shape: ref.node.shape, offset: 0 });
    } else if (
      ref.kind !== "materialized" &&
      (ref.outputIndex ?? 0) === 0 &&
      CONTIGUOUS_VIEW_OPS.has(ref.node.op)
    ) {
      // [D4 #12] Released contiguous-view input (reshape[/narrow-dim0]) —
      // IR-derivable element offset into the aliased base buffer.
      const eoff = releasedContigViewOffset(ref);
      if (eoff === undefined) return "no-storage";
      ins.push({ slot, shape: ref.node.shape, offset: eoff });
    } else {
      return "no-storage";
    }
  }
  if (ins.length === 0) return "no-inputs";
  if (ins.length === 1) return { commands: [], outSlot: ins[0].slot };

  const outSize = sizeOf(outShape);
  const innerSize = dim === rank - 1 ? 1 : sizeOf(outShape.slice(dim + 1));
  const outerSize = dim === 0 ? 1 : sizeOf(outShape.slice(0, dim));
  const outDimSize = outShape[dim];

  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const commands: GpuCommand[] = [
    {
      tag: TAG_ALLOC,
      slot: outSlot,
      bytes: outSize * bpe,
      allocKind: 0,
      inputSlots: ins.map((i) => i.slot),
    },
  ];
  let dimOffset = 0;
  for (const t of ins) {
    const tDimSize = t.shape[dim];
    const copyBytes = tDimSize * innerSize * bpe;
    const tBaseByteOffset = t.offset * bpe;
    for (let o = 0; o < outerSize; o++) {
      commands.push({
        tag: TAG_COPY,
        src: t.slot,
        dst: outSlot,
        srcOffset: tBaseByteOffset + o * tDimSize * innerSize * bpe,
        dstOffset: (o * outDimSize + dimOffset) * innerSize * bpe,
        bytes: copyBytes,
      });
    }
    dimOffset += tDimSize;
  }
  return { commands, outSlot };
}

/** Chunked binary elementwise generator (add/sub/mul/div over >128 MB buffers).
 *  outBytes uses the binary output dtype (== input dtype). */
function generateChunkedBinary(
  wgslOp: string,
  aIsScalar: boolean,
  bIsScalar: boolean,
  outSize: number,
  dtype: WebGPUTensor["dtype"],
  inSlots: Slot[],
  aliasInSlots: Slot[],
  slots: SlotSource[],
): SequentialGen {
  const plan = planChunkedBinary(wgslOp, aIsScalar, bIsScalar, outSize, dtype);
  return emitChunkedElementwise(
    plan,
    ["a", "b"],
    inSlots,
    aliasInSlots,
    outSize * dtypeBytes(dtype),
    slots,
  );
}

/** Chunked unary elementwise generator (sqrt/neg/… over >128 MB buffers). */
function generateChunkedUnary(
  opKey: string,
  outSize: number,
  dtype: WebGPUTensor["dtype"],
  inSlot: Slot,
  aliasInSlots: Slot[],
  slots: SlotSource[],
): SequentialGen {
  const plan = planChunkedUnary(opKey, outSize, dtype);
  return emitChunkedElementwise(
    plan,
    ["a"],
    [inSlot],
    aliasInSlots,
    outSize * dtypeBytes(dtype),
    slots,
  );
}

/** The narrow-offset param word count per direct-elementwise op: one offset
 *  per strided input (binary = 2, unary/cast/contiguous = 1), following `size`
 *  in paramsData. `where` is not on the generated direct path (clip/scaler run
 *  it on offset-0 tensors). */
function directOpInputCount(node: LazyIRNode): number {
  if (BINARY_OPS.has(node.op)) return 2;
  return 1; // cast / contiguous / unary / gelu
}

/** Ops that dispatch through the direct-elementwise params path (the only ops
 *  whose paramsData carries the #71 base_offset uniform). */
function isDirectElementwiseOp(op: string): boolean {
  return (
    BINARY_OPS.has(op) ||
    UNARY_OPS.has(op) ||
    op === "cast" ||
    op === "contiguous" ||
    op === "gelu"
  );
}

/** Does this node's input view chain contain a narrow (→ potentially-varying
 *  element offset)? Only then must the compiled replay re-derive the offset.
 *  Walks ONLY through VIEW_META_OPS — the exact set deriveNodeViewMeta handles
 *  — so a detected narrow is GUARANTEED offset-derivable (a chain through an op
 *  the deriver can't model would otherwise silently repack offset 0). The real
 *  view-node ops (narrow/permute/transpose/reshape/view/expand) are all in this
 *  set; squeeze/unsqueeze/broadcastTo lower to reshape/expand and never appear
 *  as node ops. */
function inputHasNarrow(ref: LazyIRNode["inputs"][number]): boolean {
  if (ref.kind !== "pending") return false;
  let n: LazyIRNode | undefined = ref.node;
  const seen = new Set<number>();
  while (n && VIEW_META_OPS.has(n.op) && !seen.has(n.id)) {
    if (n.op === "narrow") return true;
    seen.add(n.id);
    const inp = n.inputs[0];
    n = inp && inp.kind === "pending" ? inp.node : undefined;
  }
  return false;
}

/** resolveLive: the input ref's live output ViewMeta, when its result tensor
 *  still exists (a materialized base, or a view not yet released). Full
 *  shape/strides/offset so the deriver can bottom out on a materialized base. */
function refLiveMeta(
  ref: MetaNodeLike["inputs"][number],
): ViewMeta | undefined {
  const r = ref as unknown as LazyIRNode["inputs"][number];
  let bt: { shape: number[]; strides: number[]; offset?: number } | undefined;
  if (r.kind === "materialized") {
    bt = r.storage.backendTensor as typeof bt;
  } else if (r.kind === "pending") {
    const oi = r.outputIndex ?? 0;
    const st = oi === 0 ? r.node.result : r.node.results?.[oi];
    if (st) bt = st.backendTensor as typeof bt;
  }
  if (!bt) return undefined;
  return { shape: bt.shape, strides: bt.strides, offset: bt.offset ?? 0 };
}

/** The offset of a direct-op input, re-derived from the CURRENT step's graph:
 *  prefer the input's own live layout, else walk its view chain. */
function directInputOffset(ref: LazyIRNode["inputs"][number]): number {
  const mref = ref as unknown as MetaNodeLike["inputs"][number];
  const live = refLiveMeta(mref);
  if (live) return live.offset;
  if (ref.kind === "pending" && ref.node) {
    return deriveNodeOffset(ref.node, refLiveMeta) ?? 0;
  }
  return 0;
}

/**
 * Task #71: build a volatile-repack closure for a direct-elementwise params
 * buffer whose input offset(s) can vary across replays. Returns null when no
 * input traces through a narrow (offset is a compile-time constant → the frozen
 * paramsData is already correct, no repack overhead). The closure re-derives
 * each input offset from the CURRENT step's node via deriveNodeOffset
 * (single-sourced with view-meta), recomputes `size` from the output shape
 * (template-invariant), and packs EXACTLY [size, ...offsets] — the same
 * un-padded word count the direct path's params(size, ...offsets) writes (the
 * params buffer is sized from that byteLength; over-writing would overrun it).
 * Shared by the GENERATED path (this file) AND the RECORDING path
 * (bind-group-cache via the executor's pending-pack hook) so the two command
 * streams carry the identical TAG_UNIFORM and the segment diff matches.
 */
export function buildDirectOffsetRepack(
  node: LazyIRNode,
): ((node: LazyIRNode) => ArrayBufferView) | null {
  // ONLY the ops that dispatch through the direct-elementwise params path
  // (plan{Binary,Unary,Cast,Contiguous}Direct) carry the base_offset uniform.
  // Guard tightly: matmul / reductions / fused kernels / views also take
  // narrow inputs but use their OWN configs — attaching this repack to their
  // params buffer would rewrite the wrong bytes (silent corruption). Mirrors
  // generateSequential's isUnary/wgslOp direct-path classification.
  if (!isDirectElementwiseOp(node.op)) return null;
  const nInputs = directOpInputCount(node);
  let anyNarrow = false;
  for (let i = 0; i < nInputs; i++) {
    if (node.inputs[i] && inputHasNarrow(node.inputs[i])) anyNarrow = true;
  }
  if (!anyNarrow) return null;
  return (curNode: LazyIRNode): ArrayBufferView => {
    // EXACTLY the direct-path params width: [size, ...offsets]. The direct
    // dispatch sizes its params buffer from params(size, ...offsets).byteLength
    // (no 16-byte pad — unlike packUniforms), so the repack must write the same
    // word count or the writeBuffer overruns the buffer.
    const out = new Uint32Array(1 + nInputs);
    out[0] = sizeOf(curNode.shape);
    for (let i = 0; i < nInputs; i++) {
      const ref = curNode.inputs[i];
      out[1 + i] = ref ? directInputOffset(ref) : 0;
    }
    return out;
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

/**
 * Row-program (multi-reduction + elementwise → single perRowKernel): ONE
 * dispatch. Single-sourced with `dispatchRowProgram` via `planRowProgramDispatch`
 * — the kernel is keyed by `program.cacheKey` (structural, stable across steps),
 * the uniforms {num_rows, feature_dim} come from the action, and the output size
 * comes from the consumer's shape (`sizeOf(outputNode.shape)` — the same seam the
 * dispatcher asserts). Bindings are `in0..inN` (input order matching inputRefs),
 * `output`, and the cached uniform config slot. The absorbed `coveredNodeIndices`
 * produce NO individual result (only the output node), mirroring the fused case;
 * they are already excluded from the harvest (covered-internal). Bails when an
 * input isn't tracked (a released strided producer) or the config buffer is
 * absent (uniform key never dispatched — shouldn't happen post-execution). */
function generateRowProgram(
  action: {
    program: import("../compiler/row-program-types").RowProgram;
    inputRefs: LazyIRNode["inputs"];
    inputRefConsumerPositions: Array<{ pos: number; inputIndex: number }>;
    outputNodeIndex: number;
    numRows: number;
    dimSize: number;
  },
  planNodes: LazyIRNode[],
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
  fuseStampedScalarExternals: boolean,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  const inSlots: Slot[] = [];
  for (let ri = 0; ri < action.inputRefs.length; ri++) {
    // Resolve the CURRENT step's ref through consumer provenance — never the
    // action's lowering-time snapshot. The snapshot's materialized refs point
    // at the DETECTION step's storages (swept temps on template reuse: the
    // clipGradNorm_ clipCoef class — same frozen storage id on every reach),
    // while planNodes are re-created fresh every step. Single source: the node
    // graph. Falls back to the snapshot only when the consumer position is
    // unavailable (pos -1: consumer outside this plan — should not happen for
    // covered subgraph nodes, but never bind blind).
    const cp = action.inputRefConsumerPositions?.[ri];
    const freshNode = cp && cp.pos >= 0 ? planNodes[cp.pos] : undefined;
    const ref =
      (cp ? freshNode?.inputs[cp.inputIndex] : undefined) ??
      action.inputRefs[ri];
    if (ref.kind === "scalar") return "scalar-input";
    // A MATERIALIZED 0-d scalar cross-plan input (clipGradNorm_'s per-step
    // clipCoef feeding the `mul(g, clipCoef)` reduction preamble) is the
    // stale-external class executeRowProgram's sequential fallback exists for.
    // Task #96: when the FRESH ref's storage is STAMPED — the harvested result
    // of a prior COMPILED plan, re-harvested into node-visible results every
    // step — the fused kernel is safe: its external slot resolves the CURRENT
    // step's ref at every replay, and the bind seam asserts the resolved
    // storage still carries this stamp identity (compiled-plan.ts). Emit it
    // (build path only; the verify path mirrors the recorded sequential
    // fallback). If UNSTAMPED (producer lowered → swept pool temp with no
    // cross-plan identity), keep bailing.
    if (
      ref.kind === "materialized" &&
      (ref.storage.backendTensor as WebGPUTensor).shape.length === 0 &&
      (!fuseStampedScalarExternals || !ref.storage.stamp)
    ) {
      return "scalar-steptemp-input";
    }
    const s = resolveRefSlot(ref);
    if (s === undefined) return "untracked-input";
    inSlots.push(s);
  }
  const outputNode = planNodes[action.outputNodeIndex];
  if (!outputNode) return "no-output-node";
  const expectedOutElements = sizeOf(outputNode.shape);
  const plan = planRowProgramDispatch(
    action.program,
    action.numRows,
    action.dimSize,
    expectedOutElements,
  );
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  // Bindings: in0..inN → their slots, output → outSlot, config (null) → uniform.
  const named: Record<string, Slot> = { output: outSlot };
  for (let i = 0; i < inSlots.length; i++) named[`in${i}`] = inSlots[i];
  const bindings = tilePlanBindings(plan as TileKernelPlan, named, bufferSlot);
  if (!bindings) return "config-missing";
  return {
    commands: [
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: plan.outBytes,
        allocKind: 0,
        inputSlots: inSlots,
      },
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

/** sum with no dim (FULL reduction): ALLOC(4 bytes, kind 0) + tile dispatch
 *  [input, out, config], using the SAME cached dispatcher (→ shared config
 *  buffer) fullReduction() uses. Dim reductions / preamble chains stay
 *  uncovered. `mean` is NOT covered here: it lowers to sum + an invCount
 *  epilogue (mul by 1/count) dispatched through a FRESH uncached dispatcher
 *  with a per-call invCount buffer — neither the config nor that buffer is
 *  shape-derivable, so it belongs to the captured-state class, not this one. */
function generateFullReduction(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  const reduceOp = "sum";
  const payload = node.payload as
    | { dim?: number | number[] | null; keepdim?: boolean }
    | undefined;
  if (payload?.dim != null) {
    // A `sum(dim=…)` whose dim list reduces EVERY axis with keepdim=false is
    // what the DISPATCH routes to the plain full-reduction path — reductions.ts
    // prepareDimReduction returns null (outShape empty), so `reduction()` calls
    // fullReduction, NOT the dim-reduction kernel. planDimReductionDispatch
    // mirrors that null → generateDimReduction would bail `all-dims-or-epilogue`
    // and strand the plan lowered. Detect the all-dims/no-keepdim case and fall
    // through to the full-reduction body below (byte-identical to fullReduction);
    // anything else is a genuine dim reduction. (keepdim=true over all axes keeps
    // a [1,…,1] output → prepareDimReduction returns a setup → a real dim
    // reduction, so it must still route to generateDimReduction.)
    const ref0 = node.inputs[0];
    const inRank =
      ref0.kind === "materialized"
        ? (ref0.storage.backendTensor as WebGPUTensor).shape.length
        : ref0.kind === "scalar"
          ? 0
          : ref0.node.shape.length;
    const dims = Array.isArray(payload.dim) ? payload.dim : [payload.dim];
    const norm = new Set(dims.map((d) => (d < 0 ? d + inRank : d)));
    const allDims = !(payload.keepdim ?? false) && norm.size === inRank;
    if (!allDims)
      return generateDimReduction(node, slots, resolveRefSlot, bufferSlot);
  }
  if (node.inputs.length !== 1) return "arity";
  const ref = node.inputs[0];
  if (ref.kind === "scalar") return "scalar-operand";
  const storage =
    ref.kind === "materialized"
      ? ref.storage
      : (ref.outputIndex ?? 0) === 0
        ? ref.node.result
        : ref.node.results?.[ref.outputIndex ?? 0];
  const prologue: GpuCommand[] = [];
  let inSlot: Slot;
  let size: number;
  let bytesPerElement = 4;
  if (storage) {
    const t = storage.backendTensor as WebGPUTensor;
    // Contiguity-forcing resolve (declared input(0)); a live strided input gets
    // a synthesized contiguous-copy prologue, then the reduction binds the copy.
    const r = resolveContiguousOperand(node.op, 0, ref, resolveRefSlot, slots);
    if (typeof r === "string") return r;
    prologue.push(...r.prologue);
    inSlot = r.slot;
    size = t.size; // logical element count is copy-invariant
    bytesPerElement = dtypeBytes(t.dtype);
  } else if (ref.kind !== "materialized" && !VIEW_OPS.has(ref.node.op)) {
    const s = resolveRefSlot(ref);
    if (s === undefined) return "untracked-producer";
    inSlot = s;
    size = sizeOf(ref.node.shape);
    const sd = refShapeDtype(ref);
    if (sd) bytesPerElement = dtypeBytes(sd.dtype);
  } else {
    return "no-storage";
  }
  // Input exceeds maxStorageBufferBindingSize → chunked partials path (the
  // generated commands mirror sumFullReductionChunked, sharing its plan).
  if (fullReductionNeedsChunking(size, bytesPerElement)) {
    const chunked = generateChunkedFullReduction(
      size,
      bytesPerElement,
      inSlot,
      slots,
    );
    return {
      commands: [...prologue, ...chunked.commands],
      outSlot: chunked.outSlot,
    };
  }
  const plan = planFullReductionDispatch(reduceOp, size);
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
      ...prologue,
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: 4,
        allocKind: 0,
        inputSlots: [inSlot],
      },
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

/**
 * Chunked full-reduction sum (input > maxStorageBufferBindingSize). Mirrors
 * sumFullReductionChunked command-for-command, deriving identical geometry +
 * pipelines from the shared planChunkedFullReduction (single source). Both temp
 * buffers (partials, out) are FRESH arena allocs (kind 1) — the execution path
 * routes them through allocateOutputBuffer for exactly this reason, so they
 * record as real ALLOC commands the generator can reproduce here. Per-chunk
 * dispatches bind the input at a subrange (bindingRanges[0]); the partials
 * buffer and uniform params bind whole (null). When numChunks === 1 the
 * partials buffer IS the scalar output (no final reduction).
 */
function generateChunkedFullReduction(
  size: number,
  bytesPerElement: number,
  inSlot: Slot,
  slots: SlotSource[],
): { commands: GpuCommand[]; outSlot: Slot } {
  const plan = planChunkedFullReduction(
    size,
    bytesPerElement,
    requireContext(),
  );
  const commands: GpuCommand[] = [];

  const partialsSlot = slots.length;
  slots.push({ kind: "arena" });
  commands.push({
    tag: TAG_ALLOC,
    slot: partialsSlot,
    bytes: plan.partialsBytes,
    allocKind: 1,
    inputSlots: [],
  });

  for (const c of plan.chunks) {
    const pSlot = slots.length;
    slots.push({ kind: "params", seqIndex: -1, data: c.paramsData.slice() });
    commands.push({
      tag: TAG_DISPATCH,
      pipeline: plan.pipeline,
      bindings: [inSlot, partialsSlot, pSlot],
      bindingRanges: [{ offset: c.byteOffset, size: c.byteSize }, null, null],
      gx: 1,
      gy: 1,
      gz: 1,
    });
  }

  if (plan.numChunks === 1) {
    return { commands, outSlot: partialsSlot };
  }

  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  commands.push({
    tag: TAG_ALLOC,
    slot: outSlot,
    bytes: 4,
    allocKind: 1,
    inputSlots: [],
  });
  const fpSlot = slots.length;
  slots.push({
    kind: "params",
    seqIndex: -1,
    data: plan.finalParamsData!.slice(),
  });
  commands.push({
    tag: TAG_DISPATCH,
    pipeline: plan.finalPipeline!,
    bindings: [partialsSlot, outSlot, fpSlot],
    gx: 1,
    gy: 1,
    gz: 1,
  });
  return { commands, outSlot };
}

/** sum over dim(s) — NON-epilogue (e.g. a bias gradient): ALLOC(outSize*4,
 *  kind 0) + tile dispatch [input, out, config], using the SAME cached
 *  dispatcher reduction() uses (planDimReductionDispatch). Input must be
 *  contiguous (the dispatch forces it otherwise, inserting an uncaptured copy)
 *  and live or a non-view/contiguous-view producer for its shape. mean-over-dim
 *  (invCount epilogue) is excluded by planDimReductionDispatch's non-epilogue
 *  contract — it returns a plan only for plain sum/max/min. */
function generateDimReduction(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  if (node.inputs.length !== 1) return "arity";
  const payload = node.payload as
    | { dim?: number | number[] | null; keepdim?: boolean }
    | undefined;
  if (payload?.dim == null) return "no-dim";
  const ref = node.inputs[0];
  if (ref.kind === "scalar") return "scalar-operand";
  // The dim-reduction kernel reads its input as a contiguous buffer. A live
  // strided input gets a synthesized contiguous-copy prologue (declared
  // input(0)); the reduction then binds the copy.
  const storage =
    ref.kind === "materialized"
      ? ref.storage
      : (ref.outputIndex ?? 0) === 0
        ? ref.node.result
        : ref.node.results?.[ref.outputIndex ?? 0];
  const prologue: GpuCommand[] = [];
  let inSlot: Slot;
  let inShape: number[];
  if (storage) {
    const t = storage.backendTensor as WebGPUTensor;
    const r = resolveContiguousOperand(node.op, 0, ref, resolveRefSlot, slots);
    if (typeof r === "string") return r;
    prologue.push(...r.prologue);
    inSlot = r.slot;
    inShape = t.shape;
  } else {
    const s = resolveRefSlot(ref);
    if (s === undefined) return "untracked-producer";
    inSlot = s;
    const sd = refShapeDtype(ref) ?? contiguousViewShapeDtype(ref);
    if (!sd) return "no-shape";
    inShape = sd.shape;
  }
  const plan = planDimReductionDispatch(
    "sum",
    inShape,
    payload.dim,
    payload.keepdim ?? false,
  );
  if (!plan) return "all-dims-or-epilogue";
  const outShape = node.shape;
  const outBytes = sizeOf(outShape) * 4;
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
      ...prologue,
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: outBytes,
        allocKind: 0,
        inputSlots: [inSlot],
      },
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

/** mean = sum (full or dim) ÷ count, as backend.mean dispatches it: a sum
 *  through the cached reduction dispatcher into an intermediate buffer, then a
 *  meanDiv dispatch (cached "meanDiv", count in a UNIFORM not a buffer) → ONE
 *  node, TWO ALLOC+dispatch pairs recorded under it. The user-epilogue mean
 *  path (meanWithEpilogue, fresh invCount buffer) is a different op and not
 *  reached here. Input must be contiguous (sum forces it otherwise). */
function generateMean(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  if (node.inputs.length !== 1) return "arity";
  const payload = node.payload as
    | { dim?: number | number[] | null; keepdim?: boolean }
    | undefined;
  const ref = node.inputs[0];
  if (ref.kind === "scalar") return "scalar-operand";
  const storage =
    ref.kind === "materialized"
      ? ref.storage
      : (ref.outputIndex ?? 0) === 0
        ? ref.node.result
        : ref.node.results?.[ref.outputIndex ?? 0];
  const prologue: GpuCommand[] = [];
  let inSlot: Slot;
  let inShape: number[];
  if (storage) {
    const t = storage.backendTensor as WebGPUTensor;
    // The sum reads its input contiguous; a live strided input gets a
    // synthesized copy prologue (declared input(0)), then binds the copy.
    const r = resolveContiguousOperand(node.op, 0, ref, resolveRefSlot, slots);
    if (typeof r === "string") return r;
    prologue.push(...r.prologue);
    inSlot = r.slot;
    inShape = t.shape;
  } else {
    const s = resolveRefSlot(ref);
    if (s === undefined) return "untracked-producer";
    inSlot = s;
    const sd = refShapeDtype(ref) ?? contiguousViewShapeDtype(ref);
    if (!sd) return "no-shape";
    inShape = sd.shape;
  }
  const outSize = sizeOf(node.shape);
  const dims =
    payload?.dim == null
      ? null
      : (Array.isArray(payload.dim) ? payload.dim : [payload.dim]).map((d) =>
          d < 0 ? d + inShape.length : d,
        );
  const count =
    dims == null
      ? sizeOf(inShape)
      : dims.reduce((acc, d) => acc * inShape[d], 1);

  // --- sum part (cached full or dim reduction) ---
  let sumPlan: TileKernelPlan;
  let sumOutBytes: number;
  if (dims == null) {
    const inSize = sizeOf(inShape);
    if (inSize * 4 > 128 * 1024 * 1024) return "chunked";
    sumPlan = planFullReductionDispatch("sum", inSize);
    sumOutBytes = 4;
  } else {
    const p = planDimReductionDispatch(
      "sum",
      inShape,
      payload!.dim!,
      payload?.keepdim ?? false,
    );
    if (!p) return "all-dims-or-epilogue";
    sumPlan = p;
    sumOutBytes = outSize * 4;
  }
  const sumSlot = slots.length;
  slots.push({ kind: "arena" });
  const sumBindings = tilePlanBindings(
    sumPlan,
    { input: inSlot, out: sumSlot },
    bufferSlot,
  );
  if (!sumBindings) return "config-missing";

  // --- meanDiv part (cached "meanDiv", size+count uniforms) ---
  const divPlan = planMeanDivDispatch(outSize, count);
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const divBindings = tilePlanBindings(
    divPlan,
    { input: sumSlot, out: outSlot },
    bufferSlot,
  );
  if (!divBindings) return "config-missing";

  return {
    commands: [
      ...prologue,
      {
        tag: TAG_ALLOC,
        slot: sumSlot,
        bytes: sumOutBytes,
        allocKind: 0,
        inputSlots: [inSlot],
      },
      {
        tag: TAG_DISPATCH,
        pipeline: sumPlan.pipeline,
        bindings: sumBindings,
        gx: sumPlan.grid[0],
        gy: sumPlan.grid[1],
        gz: sumPlan.grid[2],
      },
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: outSize * 4,
        allocKind: 0,
        inputSlots: [sumSlot],
      },
      {
        tag: TAG_DISPATCH,
        pipeline: divPlan.pipeline,
        bindings: divBindings,
        gx: divPlan.grid[0],
        gy: divPlan.grid[1],
        gz: divPlan.grid[2],
      },
    ],
    outSlot,
  };
}

/** The batched-reduction ACTION as executed by backend.batchedReduction:
 *  for reductionSize > 64 it falls back to N INDIVIDUAL reduction() dispatches
 *  (the true multi-in/out batched kernel only runs for small reductionSize),
 *  all recorded under nodeIndices[0] (one setRecordingNodeIndex). So generate
 *  one ALLOC+dispatch per node via the SAME cached dim-reduction dispatcher,
 *  exactly mirroring the fallback. Bails (whole action) when the executor
 *  would take the true-batched path (reductionSize ≤ 64) — a different command
 *  shape — or when any member isn't generatable. */
function generateBatchedReduction(
  action: {
    nodeIndices: number[];
    reduceOp: string;
    payload: { dim: number | number[]; keepdim?: boolean };
  },
  planNodes: LazyIRNode[],
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
):
  | {
      commands: GpuCommand[];
      outputs: Array<{ nodeIndex: number; slot: Slot }>;
      firstNodeIndex: number;
    }
  | string {
  if (action.nodeIndices.length === 0) return "empty";
  const op = action.reduceOp;
  if (op !== "sum" && op !== "max" && op !== "min") return "epilogue-op";
  const dim = action.payload.dim;
  const keepdim = action.payload.keepdim ?? false;
  const firstNodeIndex = action.nodeIndices[0];
  const commands: GpuCommand[] = [];
  const outputs: Array<{ nodeIndex: number; slot: Slot }> = [];
  for (const nodeIdx of action.nodeIndices) {
    const node = planNodes[nodeIdx];
    if (node.inputs.length !== 1) return "arity";
    const ref = node.inputs[0];
    if (ref.kind === "scalar") return "scalar-operand";
    const storage =
      ref.kind === "materialized"
        ? ref.storage
        : (ref.outputIndex ?? 0) === 0
          ? ref.node.result
          : ref.node.results?.[ref.outputIndex ?? 0];
    let inSlot: Slot;
    let inShape: number[];
    if (storage) {
      const t = storage.backendTensor as WebGPUTensor;
      // Each reduction reads its input contiguous; a live strided input gets a
      // synthesized copy prologue, then this node binds the copy.
      const r = resolveContiguousOperand(
        node.op,
        0,
        ref,
        resolveRefSlot,
        slots,
      );
      if (typeof r === "string") return r;
      commands.push(...r.prologue);
      inSlot = r.slot;
      inShape = t.shape;
    } else {
      const s = resolveRefSlot(ref);
      if (s === undefined) return "untracked-input";
      inSlot = s;
      const sd = refShapeDtype(ref) ?? contiguousViewShapeDtype(ref);
      if (!sd) return "no-shape";
      inShape = sd.shape;
    }
    // Which path did the executor take? reductionSize > 64 ⇒ individual
    // fallback (what we generate); ≤ 64 ⇒ true batched dispatch (bail).
    const dims = (Array.isArray(dim) ? dim : [dim]).map((d) =>
      d < 0 ? d + inShape.length : d,
    );
    const reductionSize = dims.reduce((acc, d) => acc * inShape[d], 1);
    if (reductionSize <= 64) return "true-batched";
    const plan = planDimReductionDispatch(op, inShape, dim, keepdim);
    if (!plan) return "all-dims-or-epilogue";
    const outSlot = slots.length;
    slots.push({ kind: "arena" });
    const bindings = tilePlanBindings(
      plan,
      { input: inSlot, out: outSlot },
      bufferSlot,
    );
    if (!bindings) return "config-missing";
    commands.push(
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: sizeOf(node.shape) * 4,
        allocKind: 0,
        inputSlots: [inSlot],
      },
      {
        tag: TAG_DISPATCH,
        pipeline: plan.pipeline,
        bindings,
        gx: plan.grid[0],
        gy: plan.grid[1],
        gz: plan.grid[2],
      },
    );
    outputs.push({ nodeIndex: nodeIdx, slot: outSlot });
  }
  return { commands, outputs, firstNodeIndex };
}

/** Resolve a ref's [shape, dtype] from live storage, else the producer
 *  node (non-view) — for ops whose geometry needs an input's shape/dtype. */
function refShapeDtype(
  ref: LazyIRNode["inputs"][number],
): { shape: number[]; dtype: DType } | undefined {
  if (ref.kind === "scalar") return undefined;
  if (ref.kind === "materialized") {
    const t = ref.storage.backendTensor as WebGPUTensor;
    return { shape: t.shape, dtype: (t.dtype as DType) ?? "f32" };
  }
  const oi = ref.outputIndex ?? 0;
  const storage = oi === 0 ? ref.node.result : ref.node.results?.[oi];
  if (storage) {
    const t = storage.backendTensor as WebGPUTensor;
    return { shape: t.shape, dtype: (t.dtype as DType) ?? "f32" };
  }
  if (oi === 0 && !VIEW_OPS.has(ref.node.op)) {
    return { shape: ref.node.shape, dtype: (ref.node.dtype ?? "f32") as DType };
  }
  return undefined;
}

/** Contiguous-view ops whose logical node.shape == physical layout: a flat
 *  kernel binding the shared base buffer reads them correctly. Strided views
 *  (narrow/transpose/slice/expand) are deliberately excluded. */
const CONTIGUOUS_VIEW_OPS = new Set([
  "reshape",
  "view",
  "flatten",
  "squeeze",
  "unsqueeze",
]);

/** [D4 #12 EXPERIMENT] Element offset (into the aliased base buffer) of a
 *  RELEASED view chain built ONLY from contiguous views (reshape/view/…) and
 *  dim-0 narrows — the foreach packed-optimizer scatter-back src
 *  `reshape(narrow(pNew, 0, off, len), shape)` and the cat-input
 *  `reshape(param, [size])`. Walks node.inputs[0] accumulating dim-0-narrow
 *  starts (offset += start·innerSize; dim-0 keeps contiguity by construction),
 *  passing through contiguous views unchanged, until a non-view / live-storage
 *  base. resolveRefSlot already aliases the whole chain to the base slot; this
 *  recovers the offset the slot can't carry. Returns undefined (→ keep bailing)
 *  for ANY strided hop (transpose/permute/expand/narrow-dim≠0) or a
 *  materialized base with a nonzero own-offset we can't confirm contiguous. */
function releasedContigViewOffset(
  ref: LazyIRNode["inputs"][number],
): number | undefined {
  if (ref.kind === "scalar" || ref.kind === "materialized") return undefined;
  if ((ref.outputIndex ?? 0) !== 0) return undefined;
  let node = ref.node;
  let offset = 0;
  // Bound the walk (defensive; chains are 1–2 deep in practice).
  for (let guard = 0; guard < 64; guard++) {
    // Live storage on this hop: fold its own (confirmed-contiguous) offset.
    const storage = node.result;
    if (storage) {
      const t = storage.backendTensor as WebGPUTensor;
      if (t.isContiguous === false) return undefined;
      return offset + (t.offset ?? 0);
    }
    if (CONTIGUOUS_VIEW_OPS.has(node.op)) {
      const inp = node.inputs[0];
      if (inp.kind === "scalar") return undefined;
      if (inp.kind === "materialized") {
        const t = inp.storage.backendTensor as WebGPUTensor;
        if (t.isContiguous === false) return undefined;
        return offset + (t.offset ?? 0);
      }
      node = inp.node;
      continue;
    }
    if (node.op === "narrow") {
      const p = node.payload as
        | { dim?: number; start?: number; length?: number }
        | undefined;
      if (!p || p.dim !== 0 || p.start == null) return undefined;
      const inp = node.inputs[0];
      const inShape =
        inp.kind === "materialized"
          ? (inp.storage.backendTensor as WebGPUTensor).shape
          : inp.kind === "scalar"
            ? undefined
            : inp.node.shape;
      if (!inShape) return undefined;
      // dim-0 narrow of a contiguous base: element offset = start · innerSize.
      let inner = 1;
      for (let d = 1; d < inShape.length; d++) inner *= inShape[d];
      offset += p.start * inner;
      if (inp.kind === "materialized") {
        const t = inp.storage.backendTensor as WebGPUTensor;
        if (t.isContiguous === false) return undefined;
        return offset + (t.offset ?? 0);
      }
      if (inp.kind === "scalar") return undefined;
      node = inp.node;
      continue;
    }
    // Non-view producer base (contiguous, offset 0 relative to its buffer).
    return offset;
  }
  return undefined;
}

/** Logical shape/dtype for a released contiguous-view producer (oi 0). Returns
 *  undefined for non-views, strided views, or multi-output extras. */
function contiguousViewShapeDtype(
  ref: LazyIRNode["inputs"][number],
): { shape: number[]; dtype: DType } | undefined {
  if (ref.kind === "scalar" || ref.kind === "materialized") return undefined;
  if ((ref.outputIndex ?? 0) !== 0) return undefined;
  if (!CONTIGUOUS_VIEW_OPS.has(ref.node.op)) return undefined;
  return { shape: ref.node.shape, dtype: (ref.node.dtype ?? "f32") as DType };
}

/** scatterAdd (embedding backward): ALLOC(out,kind0,[a,index,src]) +
 *  COPY(a→out) + DISPATCH[index,src,out,params]. The copy seeds the
 *  accumulator (recorded, replayed — the +1x-grad bug class). Geometry from
 *  a.shape(=out) + src.shape + payload{dim} + index dtype. Chunked bails;
 *  if index/src needed a contiguous copy the recording diverges (caught). */
function generateScatterAdd(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  if (node.inputs.length !== 3) return "arity";
  const cfg = node.payload as { dim?: number } | undefined;
  if (!cfg || cfg.dim == null) return "payload";
  const aSlot = resolveRefSlot(node.inputs[0]);
  const idxSlot = resolveRefSlot(node.inputs[1]);
  const srcSlot = resolveRefSlot(node.inputs[2]);
  if (aSlot === undefined || idxSlot === undefined || srcSlot === undefined) {
    return "untracked-input";
  }
  const a = refShapeDtype(node.inputs[0]);
  const idx = refShapeDtype(node.inputs[1]);
  // src (the indexed grad) is typically a released `reshape` view of the
  // upstream grad. A contiguous-view's logical node.shape/dtype ARE the
  // geometry scatterAdd needs (the buffer binding comes from srcSlot, resolved
  // above); refShapeDtype skips all VIEW_OPS, so derive it here for the
  // contiguous-reshape class only — strided views (narrow/transpose) would
  // mislead the flat-indexed kernel and must keep bailing.
  const src =
    refShapeDtype(node.inputs[2]) ?? contiguousViewShapeDtype(node.inputs[2]);
  if (!a || !idx || !src) return "no-shape";
  const plan = planScatterAddDirect(a.shape, src.shape, cfg.dim, idx.dtype);
  if (!plan) return "chunked";
  const pipeline = getPipeline(requireContext(), plan.key, plan.shader);
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const paramsSlot = slots.length;
  slots.push({ kind: "params", seqIndex: -1, data: plan.paramsData.slice() });
  return {
    commands: [
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: plan.outputBytes,
        allocKind: 0,
        inputSlots: [aSlot, idxSlot, srcSlot],
      },
      {
        tag: TAG_COPY,
        src: aSlot,
        srcOffset: 0,
        dst: outSlot,
        dstOffset: 0,
        bytes: plan.outputBytes,
      },
      {
        tag: TAG_DISPATCH,
        pipeline,
        bindings: [idxSlot, srcSlot, outSlot, paramsSlot],
        gx: plan.dispatchX,
        gy: plan.dispatchY,
        gz: 1,
      },
    ],
    outSlot,
  };
}

/** gather: ALLOC(out, kind 0, [a,index]) + DISPATCH [a,index,out,params].
 *  Geometry from a.shape + index.shape/dtype + payload {dim}. Single output,
 *  no copy. Chunked (table > maxBindingSize) bails. */
function generateGather(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  if (node.inputs.length !== 2) return "arity";
  const cfg = node.payload as { dim?: number } | undefined;
  if (!cfg || cfg.dim == null) return "payload";
  const aRef = node.inputs[0];
  const idxRef = node.inputs[1];
  const aSlot = resolveRefSlot(aRef);
  const idxSlot = resolveRefSlot(idxRef);
  if (aSlot === undefined || idxSlot === undefined) return "untracked-input";
  const a = refShapeDtype(aRef);
  const idx = refShapeDtype(idxRef);
  if (!a || !idx) return "no-shape";
  const plan = planGatherDirect(
    a.shape,
    idx.shape,
    cfg.dim,
    idx.dtype,
    a.dtype,
  );
  if (!plan) return "chunked";
  const pipeline = getPipeline(requireContext(), plan.key, plan.shader);
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const paramsSlot = slots.length;
  slots.push({ kind: "params", seqIndex: -1, data: plan.paramsData.slice() });
  return {
    commands: [
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: plan.outputBytes,
        allocKind: 0,
        inputSlots: [aSlot, idxSlot],
      },
      {
        tag: TAG_DISPATCH,
        pipeline,
        bindings: [aSlot, idxSlot, outSlot, paramsSlot],
        gx: plan.dispatchX,
        gy: plan.dispatchY,
        gz: 1,
      },
    ],
    outSlot,
  };
}

/**
 * Resolve an input that the kernel reads as a raw-bindable (contiguous strides,
 * element offset 0) buffer, SYNTHESIZING the contiguous-copy prologue the
 * dispatch layer would insert (`asContiguous`/`ensureContiguous`) whenever the
 * operand resolves to LIVE strided storage.
 *
 * This is the single generic mechanism that closes the `[non-contiguous]`
 * uncovered class (docs/operand-layout-metadata-design.md): the per-generator
 * bail is replaced by a call here, driven by the CONTIGUOUS_OPERANDS
 * declaration — the caller passes `(op, operandIndex)` and the table (not the
 * generator's own hardcoded knowledge) decides whether the operand is
 * contiguity-required. The synthesized prologue is byte-identical to the
 * recorded copy (same `planContigCopy` → `planContiguousDirectCore` → shared
 * getPipeline cache), so the t-stream-generate differential ASSERTS agreement
 * with the dispatcher at the recording seam.
 *
 * Returns { slot, prologue } (prologue may be empty when already contiguous),
 * or a bail string. A RELEASED strided view with no live layout stays bailed —
 * its stride-bearing layout isn't recoverable post-hoc (unchanged; not the #99
 * case, whose operand is live). A LIVE strided operand at a position the table
 * does NOT declare contiguity-required bails `undeclared-contiguous`: the
 * declaration is load-bearing, so an undeclared-but-forced operand is a
 * coverage gap the differential surfaces rather than a silent wrong copy.
 */
function resolveContiguousOperand(
  op: string,
  operandIndex: number,
  ref: LazyIRNode["inputs"][number],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  slots: SlotSource[],
): { slot: Slot; prologue: GpuCommand[] } | string {
  if (ref.kind === "scalar") return "scalar-operand";
  const slot = resolveRefSlot(ref);
  if (slot === undefined) return "untracked-input";
  const storage =
    ref.kind === "materialized"
      ? ref.storage
      : (ref.outputIndex ?? 0) === 0
        ? ref.node.result
        : ref.node.results?.[ref.outputIndex ?? 0];
  if (storage) {
    const sbt = storage.backendTensor as WebGPUTensor;
    if (!sbt.isContiguous || (sbt.offset ?? 0) !== 0) {
      // The declaration is the single source: only synthesize the copy for a
      // position the table names contiguity-required. Undeclared-but-strided =
      // coverage gap (bail loudly), never a guessed copy.
      if (!operandRequiresContiguous(op, operandIndex))
        return "undeclared-contiguous";
      // LIVE strided operand: synthesize the contiguous copy from the live
      // layout (shape/strides/offset/dtype/bufferSize), exactly as the
      // attention prologue does from a lowering capture. The copy's output
      // slot replaces the operand's slot in the kernel binding.
      const cc = planContigCopy(
        {
          contiguous: false,
          shape: sbt.shape,
          strides: sbt.strides,
          offset: sbt.offset ?? 0,
          dtype: sbt.dtype,
          bufferSize: sbt.buffer.size,
        },
        slot,
        slots,
      );
      if (typeof cc === "string") return cc;
      return { slot: cc.outSlot, prologue: cc.commands };
    }
  } else if (!(refShapeDtype(ref) ?? contiguousViewShapeDtype(ref))) {
    return "no-storage";
  }
  return { slot, prologue: [] };
}

/** Contiguity-forcing variant that returns only the slot, appending any
 *  synthesized copy prologue to `out`. Thin adapter for the many generators
 *  that emit their commands into a single array in operand order. */
function resolveContiguousInto(
  op: string,
  operandIndex: number,
  ref: LazyIRNode["inputs"][number],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  slots: SlotSource[],
  out: GpuCommand[],
): Slot | string {
  const r = resolveContiguousOperand(
    op,
    operandIndex,
    ref,
    resolveRefSlot,
    slots,
  );
  if (typeof r === "string") return r;
  out.push(...r.prologue);
  return r.slot;
}

/** fusedCrossEntropyForward: ALLOC(loss [B], kind 0) + one tile dispatch
 *  [logits, targets, loss, config] via the module-level cached ceFwd dispatcher
 *  (planCrossEntropyForwardDispatch). Geometry is the payload config
 *  {batchSize, vocabSize, ignoreIndex} — no shape derivation. Targets must be
 *  i32/u32 (else ensureI32Targets inserts an uncaptured cast) and logits
 *  contiguous (else asContiguous inserts an uncaptured copy). */
function generateCrossEntropyForward(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  if (node.inputs.length !== 2) return "arity";
  const cfg = node.payload as
    | { batchSize?: number; vocabSize?: number; ignoreIndex?: number }
    | undefined;
  if (!cfg || cfg.batchSize == null || cfg.vocabSize == null) return "payload";
  const prologue: GpuCommand[] = [];
  const logitsSlot = resolveContiguousInto(
    node.op,
    0,
    node.inputs[0],
    resolveRefSlot,
    slots,
    prologue,
  );
  if (typeof logitsSlot === "string") return logitsSlot;
  const targetsSlot = resolveRefSlot(node.inputs[1]);
  if (targetsSlot === undefined) return "untracked-input";
  const tgt = refShapeDtype(node.inputs[1]);
  if (!tgt) return "no-shape";
  if (tgt.dtype !== "i32" && tgt.dtype !== "u32") return "targets-not-i32";
  const plan = planCrossEntropyForwardDispatch(
    cfg.batchSize,
    cfg.vocabSize,
    cfg.ignoreIndex ?? -100,
  );
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const bindings = tilePlanBindings(
    plan,
    { logits: logitsSlot, targets: targetsSlot, loss: outSlot },
    bufferSlot,
  );
  if (!bindings) return "config-missing";
  return {
    commands: [
      ...prologue,
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: cfg.batchSize * 4,
        allocKind: 1,
        inputSlots: [],
      },
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

/** fusedCrossEntropyBackward: ALLOC(grad_logits [B,V], kind 0) + one tile
 *  dispatch [logits, targets, grad_output, grad_logits, config] via the cached
 *  ceBwd dispatcher. Same guards as forward, plus grad_output contiguous. */
function generateCrossEntropyBackward(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  if (node.inputs.length !== 3) return "arity";
  const cfg = node.payload as
    | { batchSize?: number; vocabSize?: number; ignoreIndex?: number }
    | undefined;
  if (!cfg || cfg.batchSize == null || cfg.vocabSize == null) return "payload";
  const prologue: GpuCommand[] = [];
  const logitsSlot = resolveContiguousInto(
    node.op,
    0,
    node.inputs[0],
    resolveRefSlot,
    slots,
    prologue,
  );
  if (typeof logitsSlot === "string") return logitsSlot;
  const targetsSlot = resolveRefSlot(node.inputs[1]);
  if (targetsSlot === undefined) return "untracked-input";
  const tgt = refShapeDtype(node.inputs[1]);
  if (!tgt) return "no-shape";
  if (tgt.dtype !== "i32" && tgt.dtype !== "u32") return "targets-not-i32";
  const gradSlot = resolveContiguousInto(
    node.op,
    2,
    node.inputs[2],
    resolveRefSlot,
    slots,
    prologue,
  );
  if (typeof gradSlot === "string") return gradSlot;
  const plan = planCrossEntropyBackwardDispatch(
    cfg.batchSize,
    cfg.vocabSize,
    cfg.ignoreIndex ?? -100,
  );
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const bindings = tilePlanBindings(
    plan,
    {
      logits: logitsSlot,
      targets: targetsSlot,
      grad_output: gradSlot,
      grad_logits: outSlot,
    },
    bufferSlot,
  );
  if (!bindings) return "config-missing";
  return {
    commands: [
      ...prologue,
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: cfg.batchSize * cfg.vocabSize * 4,
        allocKind: 1,
        inputSlots: [],
      },
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
  if (
    !cfg ||
    cfg.numRows == null ||
    cfg.featureDim == null ||
    cfg.eps == null
  ) {
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

/** fusedLayerNormBackwardGradX: ALLOC(grad_x, kind 1) + one tile dispatch
 *  [grad_output, x, weight, grad_x, config]. Single output, no workspace —
 *  structurally identical to the forward. payload = FusedLayerNormConfig. */
function generateLayerNormGradX(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  if (node.inputs.length !== 3) return "arity";
  const cfg = node.payload as
    | { numRows?: number; featureDim?: number; eps?: number }
    | undefined;
  if (
    !cfg ||
    cfg.numRows == null ||
    cfg.featureDim == null ||
    cfg.eps == null
  ) {
    return "payload";
  }
  const inSlots: Slot[] = [];
  for (const ref of node.inputs) {
    if (ref.kind === "scalar") return "scalar-operand";
    const s = resolveRefSlot(ref);
    if (s === undefined) return "untracked-input";
    inSlots.push(s);
  }
  const planned = planLayerNormBackwardGradXDispatch(
    cfg.numRows,
    cfg.featureDim,
    cfg.eps,
  );
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const bindings = tilePlanBindings(
    planned.plan,
    {
      grad_output: inSlots[0],
      x: inSlots[1],
      weight: inSlots[2],
      grad_x: outSlot,
    },
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

/**
 * Flash-attention fwd/bwd generator (multi-output). dispatchAttention binds
 * [...buffers, config] POSITIONALLY (not via the tile spec), so bindings are
 * built positionally here. The shared config buffer is a cached persistent
 * slot (lookup). Forward: ALLOC(output) + ALLOC(lse) + 1 dispatch
 * [q,k,v,output,lse,config], outputs {0:output, 1:logsumexp}. Backward: D
 * (ALLOC dBuf + dispatch [dO,o,dBuf,config]) → DQ (ALLOC dQ + dispatch
 * [q,k,v,lse,dBuf,dO,dQ,config]) → DKV (ALLOC dK, ALLOC dV + dispatch
 * [q,k,v,lse,dBuf,dO,dK,dV,config]); dBuf is an ephemeral recorded alloc,
 * outputs {0:dQ, 1:dK, 2:dV}. Inputs q/k/v/output/logsumexp are saved (live);
 * dO is a grad (logical slot if released).
 */
function generateAttention(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
  inputContig: AttnInputContig[] | undefined,
):
  | {
      commands: GpuCommand[];
      outputs: Array<{ outputIndex: number; slot: Slot }>;
    }
  | string {
  const cfg = node.payload as
    | {
        batchSize?: number;
        numHeads?: number;
        seqLen?: number;
        headDim?: number;
        scale?: number;
        isCausal?: boolean;
        modifier?: import("../backend/types").AttnModifierSpec;
      }
    | undefined;
  if (
    !cfg ||
    cfg.batchSize == null ||
    cfg.numHeads == null ||
    cfg.seqLen == null ||
    cfg.headDim == null ||
    cfg.scale == null
  ) {
    return "payload";
  }
  const inSlots: Slot[] = [];
  for (const ref of node.inputs) {
    if (ref.kind === "scalar") return "scalar-operand";
    const s = resolveRefSlot(ref);
    if (s === undefined) return "untracked-input";
    inSlots.push(s);
  }
  const newSlot = (): Slot => {
    const s = slots.length;
    slots.push({ kind: "arena" });
    return s;
  };

  // Contiguous-copy prologue: the op asContiguous's each input (fused.ts),
  // copying any whose isContiguous === false. We mirror that exactly from the
  // per-input layout captured at lowering — for each non-contiguous input, one
  // resolveOutputBuffer ALLOC (kind 0, the original as aliasing input) + one
  // planContiguousDirect dispatch [orig, copy, params], then rebind the kernel
  // input to the copy's slot. Copies are emitted in input order, before the
  // attention dispatches (asContiguous runs on all inputs up front). The
  // captured `contiguous` flag matches ensureContiguous, so the command count
  // matches the recording. Without the capture we can't replay the copy → bail.
  const prologue: GpuCommand[] = [];
  if (inputContig) {
    if (inputContig.length !== inSlots.length) return "contig-arity";
    for (let i = 0; i < inSlots.length; i++) {
      const info = inputContig[i];
      if (info.contiguous) continue;
      const cc = planContigCopy(info, inSlots[i], slots);
      if (typeof cc === "string") return cc;
      prologue.push(...cc.commands);
      inSlots[i] = cc.outSlot;
    }
  } else if (node.op === "fusedAttentionBackward") {
    // Forward inputs (q/k/v) are contiguous in the covered traces, so a
    // missing capture there is benign; backward relies on the capture.
    return "no-contig-capture";
  }
  const causal = cfg.isCausal ?? false;

  if (node.op === "fusedAttentionForward") {
    if (inSlots.length !== 3) return "arity";
    const p = planFlashAttentionForward(
      cfg.batchSize,
      cfg.numHeads,
      cfg.seqLen,
      cfg.headDim,
      cfg.scale,
      causal,
      cfg.modifier,
    );
    if (!p.plan.configBuffer) return "config-missing";
    const cfgSlot = bufferSlot(p.plan.configBuffer as unknown, "persistent");
    const outSlot = newSlot();
    const lseSlot = newSlot();
    return {
      commands: [
        ...prologue,
        {
          tag: TAG_ALLOC,
          slot: outSlot,
          bytes: p.outBytes,
          allocKind: 1,
          inputSlots: [],
        },
        {
          tag: TAG_ALLOC,
          slot: lseSlot,
          bytes: p.lseBytes,
          allocKind: 1,
          inputSlots: [],
        },
        {
          tag: TAG_DISPATCH,
          pipeline: p.plan.pipeline,
          bindings: [
            inSlots[0],
            inSlots[1],
            inSlots[2],
            outSlot,
            lseSlot,
            cfgSlot,
          ],
          gx: p.plan.grid[0],
          gy: p.plan.grid[1],
          gz: p.plan.grid[2],
        },
      ],
      outputs: [
        { outputIndex: 0, slot: outSlot },
        { outputIndex: 1, slot: lseSlot },
      ],
    };
  }

  // fusedAttentionBackward — inputs [q, k, v, logsumexp, dO, output]
  if (inSlots.length !== 6) return "arity";
  const [qS, kS, vS, lseS, dOS, oS] = inSlots;
  const p = planFlashAttentionBackward(
    cfg.batchSize,
    cfg.numHeads,
    cfg.seqLen,
    cfg.headDim,
    cfg.scale,
    causal,
    cfg.modifier,
  );
  if (!p.dPlan.configBuffer) return "config-missing";
  const cfgSlot = bufferSlot(p.dPlan.configBuffer as unknown, "persistent");
  const dBufSlot = newSlot();
  const dQSlot = newSlot();
  const dKSlot = newSlot();
  const dVSlot = newSlot();
  const dispatch = (plan: AttentionStepPlan, bindings: Slot[]): GpuCommand => ({
    tag: TAG_DISPATCH,
    pipeline: plan.pipeline,
    bindings,
    gx: plan.grid[0],
    gy: plan.grid[1],
    gz: plan.grid[2],
  });
  return {
    commands: [
      ...prologue,
      {
        tag: TAG_ALLOC,
        slot: dBufSlot,
        bytes: p.dBytes,
        allocKind: 1,
        inputSlots: [],
      },
      dispatch(p.dPlan, [dOS, oS, dBufSlot, cfgSlot]),
      {
        tag: TAG_ALLOC,
        slot: dQSlot,
        bytes: p.dqBytes,
        allocKind: 1,
        inputSlots: [],
      },
      dispatch(p.dqPlan, [qS, kS, vS, lseS, dBufSlot, dOS, dQSlot, cfgSlot]),
      {
        tag: TAG_ALLOC,
        slot: dKSlot,
        bytes: p.dkvBytes,
        allocKind: 1,
        inputSlots: [],
      },
      {
        tag: TAG_ALLOC,
        slot: dVSlot,
        bytes: p.dkvBytes,
        allocKind: 1,
        inputSlots: [],
      },
      dispatch(p.dkvPlan, [
        qS,
        kS,
        vS,
        lseS,
        dBufSlot,
        dOS,
        dKSlot,
        dVSlot,
        cfgSlot,
      ]),
    ],
    outputs: [
      { outputIndex: 0, slot: dQSlot },
      { outputIndex: 1, slot: dKSlot },
      { outputIndex: 2, slot: dVSlot },
    ],
  };
}

/** fusedLayerNormBackwardGradWeightBias: the workspace case. Three tile
 *  dispatches over CACHED workspace buffers (mean/invStd, partialGW/GB —
 *  persistent slots, looked up by the plan; no ALLOC) then two output
 *  ALLOCs (grad_weight, grad_bias) and the reduce dispatch. Command order
 *  mirrors the dispatcher: rowStats DISPATCH, partial DISPATCH, ALLOC gw,
 *  ALLOC gb, reduce DISPATCH. Multi-output (primary grad_weight); the
 *  outSlot returned is grad_weight, grad_bias is allocated in-stream (the
 *  segment diff verifies both ALLOCs). Inputs [grad_output, x]. */
function generateLayerNormGradWB(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): SequentialGen | string {
  if (node.inputs.length !== 2) return "arity";
  const cfg = node.payload as
    | { numRows?: number; featureDim?: number; eps?: number }
    | undefined;
  if (
    !cfg ||
    cfg.numRows == null ||
    cfg.featureDim == null ||
    cfg.eps == null
  ) {
    return "payload";
  }
  if (node.inputs[0].kind === "scalar" || node.inputs[1].kind === "scalar") {
    return "scalar-operand";
  }
  const goSlot = resolveRefSlot(node.inputs[0]);
  const xSlot = resolveRefSlot(node.inputs[1]);
  if (goSlot === undefined || xSlot === undefined) return "untracked-input";
  const p = planLayerNormBackwardGradWeightBias(
    cfg.numRows,
    cfg.featureDim,
    cfg.eps,
  );
  if (!p.meanBuffer || !p.invStdBuffer || !p.partialGW || !p.partialGB) {
    return "workspace-missing";
  }
  const meanSlot = bufferSlot(p.meanBuffer as unknown, "persistent");
  const invStdSlot = bufferSlot(p.invStdBuffer as unknown, "persistent");
  const gwPartialSlot = bufferSlot(p.partialGW as unknown, "persistent");
  const gbPartialSlot = bufferSlot(p.partialGB as unknown, "persistent");

  const rowStatsBindings = tilePlanBindings(
    p.rowStatsPlan,
    { x: xSlot, row_mean: meanSlot, row_inv_std: invStdSlot },
    bufferSlot,
  );
  const partialBindings = tilePlanBindings(
    p.partialPlan,
    {
      grad_output: goSlot,
      x: xSlot,
      row_mean: meanSlot,
      row_inv_std: invStdSlot,
      partial_gw: gwPartialSlot,
      partial_gb: gbPartialSlot,
    },
    bufferSlot,
  );
  if (!rowStatsBindings || !partialBindings) return "config-missing";

  const gwSlot = slots.length;
  slots.push({ kind: "arena" });
  const gbSlot = slots.length;
  slots.push({ kind: "arena" });

  const reduceBindings = tilePlanBindings(
    p.reducePlan,
    {
      partial_gw: gwPartialSlot,
      partial_gb: gbPartialSlot,
      grad_weight: gwSlot,
      grad_bias: gbSlot,
    },
    bufferSlot,
  );
  if (!reduceBindings) return "config-missing";

  const dispatch = (plan: TileKernelPlan, bindings: Slot[]): GpuCommand => ({
    tag: TAG_DISPATCH,
    pipeline: plan.pipeline,
    bindings,
    gx: plan.grid[0],
    gy: plan.grid[1],
    gz: plan.grid[2],
  });

  return {
    commands: [
      dispatch(p.rowStatsPlan, rowStatsBindings),
      dispatch(p.partialPlan, partialBindings),
      {
        tag: TAG_ALLOC,
        slot: gwSlot,
        bytes: p.featureBytes,
        allocKind: 1,
        inputSlots: [],
      },
      {
        tag: TAG_ALLOC,
        slot: gbSlot,
        bytes: p.featureBytes,
        allocKind: 1,
        inputSlots: [],
      },
      dispatch(p.reducePlan, reduceBindings),
    ],
    outSlot: gwSlot,
    // grad_bias is outputIndex 1 — expose it so the harvest maps it into
    // nodeSlotExtra (the optimizer consumes it; dropping it replays stale bias
    // gradients → slow trajectory divergence).
    extraOutputs: [{ outputIndex: 1, slot: gbSlot }],
  };
}

/** narrowBackward: ALLOC(out, kind 0, inputs=[grad]) + one dispatch
 *  [grad, out, params]. Geometry from the grad input SHAPE + payload
 *  {dim,start,originalLength} + node dtype (no live buffer). The grad shape
 *  comes from the input ref's producer node (works post-release). */
function generateNarrowBackward(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  cachedInputShapes?: number[][],
): { commands: GpuCommand[]; outSlot: Slot } | string {
  if (node.inputs.length !== 1) return "arity";
  const cfg = node.payload as
    | { dim?: number; start?: number; originalLength?: number }
    | undefined;
  if (
    !cfg ||
    cfg.dim == null ||
    cfg.start == null ||
    cfg.originalLength == null
  ) {
    return "payload";
  }
  const ref = node.inputs[0];
  if (ref.kind === "scalar") return "scalar-operand";
  const gradSlot = resolveRefSlot(ref);
  if (gradSlot === undefined) return "untracked-input";
  // Grad shape: the lowering-captured shape (cachedInputShapes[0]) is
  // authoritative — the grad is a released multi-output extra (attention
  // dQ/dK/dV) at plan-build, with no recoverable per-output shape. Fall
  // back to live storage / a non-view producer's node shape.
  let gradShape: number[] | undefined = cachedInputShapes?.[0];
  if (!gradShape) {
    if (ref.kind === "materialized") {
      gradShape = ref.storage.backendTensor.shape;
    } else {
      const oi = ref.outputIndex ?? 0;
      const storage = oi === 0 ? ref.node.result : ref.node.results?.[oi];
      gradShape = storage
        ? storage.backendTensor.shape
        : oi === 0 && !VIEW_OPS.has(ref.node.op)
          ? ref.node.shape
          : undefined;
    }
  }
  if (!gradShape) return "no-grad-shape";
  const dtype = (node.dtype ?? "f32") as DType;
  const plan = planNarrowBackward(
    gradShape,
    cfg.dim,
    cfg.start,
    cfg.originalLength,
    dtype,
  );
  const pipeline = getPipeline(requireContext(), plan.key, plan.shader);
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const paramsSlot = slots.length;
  slots.push({ kind: "params", seqIndex: -1, data: plan.paramsData.slice() });
  return {
    commands: [
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: plan.outputSizeBytes,
        allocKind: 0,
        inputSlots: [gradSlot],
      },
      {
        tag: TAG_DISPATCH,
        pipeline,
        bindings: [gradSlot, outSlot, paramsSlot],
        gx: plan.dispatchX,
        gy: plan.dispatchY,
        gz: 1,
      },
    ],
    outSlot,
  };
}

/** unscaleGrad: ALLOC(kind 1, power-of-two bytes) + tile dispatch
 *  [grad_in, grad_out, inf_flag, scale, config].
 *  scaler-as-tensor: scale is a persistent 1-element storage input (the
 *  LiveScalar buffer, node.inputs[1]) read LIVE (invScale reciprocated
 *  in-kernel), and the config is fully STATIC — so there is NO TAG_UNIFORM
 *  volatile repack (the retired inv_scale-frozen class). The config buffer is
 *  bound once via `tilePlanBindings` (name===null → planned.plan.configBuffer,
 *  baked at plan() time). */
function generateUnscaleGrad(
  node: LazyIRNode,
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  const payload = node.payload as { infFlagBuffer?: unknown } | undefined;
  if (!payload || !payload.infFlagBuffer) {
    return "payload";
  }
  const ref = node.inputs[0];
  if (!ref || ref.kind === "scalar") return "arity";
  const scaleRef = node.inputs[1];
  if (!scaleRef || scaleRef.kind === "scalar") return "arity";
  const scaleSlot = resolveRefSlot(scaleRef);
  if (scaleSlot === undefined) return "untracked-scale";
  const storage =
    ref.kind === "materialized"
      ? ref.storage
      : (ref.outputIndex ?? 0) === 0
        ? ref.node.result
        : ref.node.results?.[ref.outputIndex ?? 0];
  const prologue: GpuCommand[] = [];
  let inSlot: Slot;
  let size: number;
  if (storage) {
    const t = storage.backendTensor as WebGPUTensor;
    // grad_in is raw-bound (declared input(0)); a live strided grad gets a
    // synthesized contiguous-copy prologue, then unscale binds the copy.
    const r = resolveContiguousOperand(node.op, 0, ref, resolveRefSlot, slots);
    if (typeof r === "string") return r;
    prologue.push(...r.prologue);
    inSlot = r.slot;
    size = t.size;
  } else if (ref.kind !== "materialized" && !VIEW_OPS.has(ref.node.op)) {
    const s = resolveRefSlot(ref);
    if (s === undefined) return "untracked-producer";
    inSlot = s;
    size = sizeOf(ref.node.shape);
  } else {
    return "no-storage";
  }
  const planned = planUnscaleGradDispatch(size);
  if (!planned) return "chunked";
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const named: Record<string, Slot> = {
    grad_in: inSlot,
    grad_out: outSlot,
    inf_flag: bufferSlot(payload.infFlagBuffer, "persistent"),
    scale: scaleSlot,
  };
  const bindings = tilePlanBindings(planned.plan, named, bufferSlot);
  if (!bindings) return "config-missing";
  return {
    commands: [
      ...prologue,
      {
        tag: TAG_ALLOC,
        slot: outSlot,
        bytes: planned.alignedBytes,
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
      : (ref.outputIndex ?? 0) === 0
        ? ref.node.result
        : ref.node.results?.[ref.outputIndex ?? 0];
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
  } else if (
    ref.kind !== "materialized" &&
    (ref.outputIndex ?? 0) === 0 &&
    CONTIGUOUS_VIEW_OPS.has(ref.node.op)
  ) {
    // [D4 #12] A released contiguous-view chain (reshape[/narrow-dim0]) whose
    // offset is IR-derivable — the packed-optimizer scatter-back src. slot
    // aliases the base buffer; meta is contiguous at the recovered offset.
    const eoff = releasedContigViewOffset(ref);
    if (eoff === undefined) return "no-storage";
    const shape = ref.node.shape;
    const size = sizeOf(shape);
    meta = {
      shape,
      strides: contiguousStrides(shape),
      offset: eoff,
      size,
      dtype: (ref.node.dtype ?? "f32") as WebGPUTensor["dtype"],
      isContiguous: true,
      buffer: { size: alignBufferSize((size + eoff) * 4) },
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
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
): { commands: GpuCommand[]; outSlot: Slot } | string {
  const payload = node.payload as
    | { offset: number; viewShape: number[]; viewStrides: number[] }
    | undefined;
  if (!payload) return "payload";
  if (node.inputs.length !== 2) return "arity";
  const base = refSlotAndMeta(node.inputs[0], resolveRefSlot);
  if (typeof base === "string") return base;
  const srcRef = node.inputs[1];
  let src: { slot: Slot; meta: WebGPUTensor };
  if (srcRef.kind === "scalar") {
    // [D4 #13] SCALAR-source scatter (a host scalar written into a slice of a
    // state tensor, e.g. into shape [1]). The scalar resolves through the
    // plan's scalar table to a PERSISTENT buffer — stable identity, refreshed
    // from the CURRENT step's value by the executor before every execution —
    // so the value flows as DATA (never frozen into the plan; the
    // frozen-scalar class). Same shape as the direct-elementwise scalar
    // operand. Scalar-table storage is f32; a non-f32 base takes the
    // execution's dtype-guarded dispatch route → keep it uncovered.
    const sh = lookupScalarStorage(srcRef);
    if (!sh) return "scalar-no-table";
    if ((node.dtype ?? "f32") !== "f32") return "scalar-non-f32";
    const t = sh.backendTensor as WebGPUTensor;
    src = { slot: bufferSlot(gpuBuffer(t) as unknown, "persistent"), meta: t };
  } else {
    const resolved = refSlotAndMeta(srcRef, resolveRefSlot);
    if (typeof resolved === "string") return resolved;
    src = resolved;
  }
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
        (src.meta.buffer as { size: number }).size - (src.meta.offset ?? 0) * 4
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
  // Route-engagement signal that survives compiled-plan cutover (see counter
  // doc). Read from the R9 SelectionReceipt the plan carries (single source),
  // not a re-parse of the profiler label.
  if (m.selection?.gemvEngaged) generatedGemvDispatchCount++;

  if (m.kSplit) {
    // K-split: ALLOC(out) then two dispatches over the cached partials temp
    // (a persistent-slot buffer — no ALLOC, looked up by byte size). The
    // generator binds the SAME cached buffer object the recording used.
    const temp = lookupKSplitTempBuffer(m.tempBytes);
    if (!temp) return "ksplit-temp-missing";
    const tempSlot = bufferSlot(temp as unknown, "persistent");
    const ksplitParamsSlot = slots.length;
    slots.push({
      kind: "params",
      seqIndex: -1,
      data: m.ksplitParamsData.slice(),
    });
    const reduceParamsSlot = slots.length;
    slots.push({
      kind: "params",
      seqIndex: -1,
      data: m.reduceParamsData.slice(),
    });
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
  slots.push({ kind: "params", seqIndex: -1, data: m.paramsData.slice() });
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
  // Route-engagement signal that survives compiled-plan cutover (see counter doc).
  if (plan.label?.startsWith("_gemv")) generatedGemvDispatchCount++;

  const outBytes = sizeOf(cfg.outShape) * dtypeBytes(cfg.outputDtype);
  const outSlot = slots.length;
  slots.push({ kind: "arena" });
  const paramsSlot = slots.length;
  slots.push({ kind: "params", seqIndex: -1, data: plan.paramsData.slice() });
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
  cachedExternalInputPattern?: Array<{
    nodeLocalIdx: number;
    inputIdx: number;
  }>;
  recipe: import("../backend/webgpu/fusion-types").FusedKernelRecipe;
  runtimeScalarInputs?: Set<number>;
}

/**
 * Fused-segment generator. Covers the clean case — all external inputs
 * contiguous — and emits the fused kernel's ALLOC(s) + DISPATCH via
 * planFusedKernel (shared with the dispatcher, so binding order / workgroups
 * can't drift). Donation is detected POST-HOC: the recording already ran, so
 * a donated out0 is the input whose buffer the output node's result aliases
 * (ownsBuffer=false) — no liveness recomputation needed.
 *
 * NEEDED INTERMEDIATES: group-internal elementwise nodes consumed OUTSIDE the
 * group that couldn't be promoted to additional fused outputs (shape mismatch
 * with the primary, or out of binding slots). The fused kernel computes them
 * inline but doesn't write them out; the executor re-executes just those nodes
 * sequentially (executeFusedSegment → executeSequentialSegment) so external
 * consumers can read them. That re-execution stays under the SAME recording
 * node index (action.outputNodeIndex — never reset by executeNode), so the
 * recorder attributes those commands to THIS segment. We mirror it exactly:
 * after the fused dispatch, generate each needed-intermediate's plain
 * ALLOC+DISPATCH via generateSequential (the same elementwise path executeNode
 * takes), resolving its inputs against this group's own outputs (assigned
 * locally here, not yet in the outer nodeSlot) and earlier intermediates.
 *
 * Bails (uncovered) on contiguous-copy prologues and oversized buffers,
 * exactly the branches executeFusedSegment routes away from the plain
 * dispatch; a needed-intermediate that itself can't be generated bails the
 * whole action (kept atomic — partial segments would break the diff).
 */
/**
 * Oversized fused group → sequential decomposition. Mirrors the executor's
 * executeSequentialSegment fallback (taken for a fused group whose external
 * input buffers exceed maxStorageBufferBindingSize): run each covered node's
 * plain elementwise dispatch in coveredNodeIndices order, all recorded under
 * this one action (setRecordingNodeIndex(outputNodeIndex)). Each node is the
 * plain dispatch executeNode would run, so generateSequential reproduces it —
 * and at >128 MB generateSequential now routes to the chunked-elementwise path.
 * Group-internal producers feed later nodes via a local slot channel; all
 * covered nodes are mapped for the outer walker so any external consumer of an
 * intermediate resolves. Kept atomic — a node that can't be generated bails the
 * whole action (a partial segment would break the diff).
 */
function generateFusedDecomposed(
  action: FusedActionShape,
  planNodes: LazyIRNode[],
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
):
  | {
      commands: GpuCommand[];
      outputs: Array<{ nodeIndex: number; slot: Slot }>;
    }
  | string {
  const commands: GpuCommand[] = [];
  const outputs: Array<{ nodeIndex: number; slot: Slot }> = [];
  const localById = new Map<number, Slot>();
  const localResolve = (
    ref: LazyIRNode["inputs"][number],
  ): Slot | undefined => {
    if (
      ref.kind !== "materialized" &&
      ref.kind !== "scalar" &&
      (ref.outputIndex ?? 0) === 0
    ) {
      const s = localById.get(ref.node.id);
      if (s !== undefined) return s;
    }
    return resolveRefSlot(ref);
  };
  for (const niIdx of action.coveredNodeIndices) {
    const niNode = planNodes[niIdx];
    const g = generateSequential(niNode, slots, localResolve, bufferSlot);
    if (typeof g === "string") {
      return `oversized-decomp:${niNode.op}[${g || "?"}]`;
    }
    // The whole segment records under action.outputNodeIndex; patch any
    // node-local TAG_UNIFORM placeholder (offset repack) to it. Chunked
    // elementwise carries no volatile uniform, so this is defensive.
    for (const c of g.commands) {
      if (c.tag === TAG_UNIFORM && c.nodeIndex < 0) {
        c.nodeIndex = action.outputNodeIndex;
      }
    }
    commands.push(...g.commands);
    localById.set(niNode.id, g.outSlot);
    outputs.push({ nodeIndex: niIdx, slot: g.outSlot });
  }
  return { commands, outputs };
}

function generateFused(
  action: FusedActionShape,
  planNodes: LazyIRNode[],
  slots: SlotSource[],
  resolveRefSlot: (ref: LazyIRNode["inputs"][number]) => Slot | undefined,
  bufferSlot: (buf: unknown, kind: SlotSource["kind"]) => Slot,
  backend: import("../backend/types").Backend,
):
  | {
      commands: GpuCommand[];
      outputs: Array<{ nodeIndex: number; slot: Slot }>;
    }
  | string {
  const device = (backend as { device?: GPUDevice }).device;
  if (!device) return "no-device";
  if (!action.cachedExternalInputPattern) return "no-input-pattern";

  const groupNodes = action.coveredNodeIndices.map((i) => planNodes[i]);
  const extInputs = action.cachedExternalInputPattern.map(
    (p) => groupNodes[p.nodeLocalIdx].inputs[p.inputIdx],
  );
  const recipe = action.recipe;

  // Oversized fused group → the executor routes to executeSequentialSegment
  // (its `hasOversizedBuffer` check on the external input buffers), running each
  // group node's plain elementwise dispatch, which at >128 MB takes the chunked
  // path. Mirror it: DECOMPOSE the group and generate each covered node via
  // generateSequential (now chunked-aware) under this one segment. This is the
  // foreach optimizer's 320 MB packed-buffer decomposition. Detected the same
  // way the executor detects it — any external input buffer > maxBinding.
  const maxBindingSize =
    device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const extInputOversized = extInputs.some((ref) => {
    if (!ref) return false;
    const oi = ref.kind === "materialized" ? 0 : (ref.outputIndex ?? 0);
    const storage =
      ref.kind === "materialized"
        ? ref.storage
        : ref.kind === "scalar"
          ? undefined
          : oi === 0
            ? ref.node.result
            : ref.node.results?.[oi];
    let bytes: number;
    if (storage) {
      bytes = (storage.backendTensor as WebGPUTensor).buffer.size;
    } else if (ref.kind !== "materialized" && ref.kind !== "scalar") {
      bytes = sizeOf(ref.node.shape) * dtypeBytes(ref.node.dtype ?? "f32");
    } else {
      return false;
    }
    return bytes > maxBindingSize;
  });
  if (extInputOversized) {
    return generateFusedDecomposed(
      action,
      planNodes,
      slots,
      resolveRefSlot,
      bufferSlot,
    );
  }

  // Resolve the non-inlined inputs in recipe order (mirrors
  // executeFusedSegment). planFusedKernel never dereferences input buffers
  // (only their count + the donated position), so a liveness-RELEASED
  // producer is fine: synthesize its metadata, resolve its slot via the
  // logical channel. Real buffers are kept for donation detection.
  const dispatchInputs: Array<{
    buffer: GPUBuffer;
    shape: number[];
    dtype: DType;
  }> = [];
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
      // Oversized is DERIVED from the device limit (single source with
      // executeFusedSegment's hasOversizedBuffer check + dispatchBinary/Unary),
      // NOT a hardcoded 128 MB — on a device whose maxStorageBufferBindingSize
      // exceeds the buffer (e.g. 2 GB), the executor runs the fused kernel
      // WHOLE (no sequential fallback), so this must generate the fused path,
      // not bail. The genuinely-oversized case is caught earlier by
      // extInputOversized → generateFusedDecomposed (chunked elementwise).
      if (t.buffer.size > maxBindingSize) return "oversized";
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

  // Donation: prefer the executor's cached decision (the recording's actual
  // donation), which is authoritative and survives the primary output's result
  // being cleared (step-scoped in-place outputs — the foreach optimizer's m/v
  // update). Fall back to POST-HOC detection (out0's result buffer aliasing an
  // input) for actions not executed under recording (e.g. isolated unit tests).
  const cachedDonated = (action as { cachedDonatedRecipeIdx?: number })
    .cachedDonatedRecipeIdx;
  const outNode = planNodes[action.outputNodeIndex];
  let donatedRecipeIdx: number | undefined = cachedDonated;
  if (
    donatedRecipeIdx === undefined &&
    outNode?.result &&
    (outNode.result.backendTensor as WebGPUTensor).buffer
  ) {
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
  slots.push({ kind: "params", seqIndex: -1, data: plan.paramsData.slice() });

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

  // Needed-intermediate re-execution (see header). Resolve each one's inputs
  // against this group's own outputs (primary + promoted additionals, whose
  // slots are assigned right here and aren't in the outer nodeSlot yet) and
  // against earlier intermediates in this same loop — falling back to the
  // outer channel for true external inputs. Each is the plain elementwise
  // dispatch executeNode would run, so generateSequential reproduces it.
  const needed = action.neededIntermediateNodeIndices ?? [];
  if (needed.length > 0) {
    const localById = new Map<number, Slot>();
    localById.set(planNodes[action.outputNodeIndex].id, outputSlots[0]);
    for (let i = 0; i < addl.length; i++) {
      if (outputSlots[i + 1] !== undefined) {
        localById.set(planNodes[addl[i]].id, outputSlots[i + 1]);
      }
    }
    const localResolve = (
      ref: LazyIRNode["inputs"][number],
    ): Slot | undefined => {
      if (ref.kind !== "materialized" && ref.kind !== "scalar") {
        if ((ref.outputIndex ?? 0) === 0) {
          const s = localById.get(ref.node.id);
          if (s !== undefined) return s;
        }
      }
      return resolveRefSlot(ref);
    };
    for (const niIdx of needed) {
      const niNode = planNodes[niIdx];
      const g = generateSequential(niNode, slots, localResolve, bufferSlot);
      if (typeof g === "string") {
        return `needed-intermediate:${niNode.op}[${g || "?"}]`;
      }
      commands.push(...g.commands);
      localById.set(niNode.id, g.outSlot);
      outputs.push({ nodeIndex: niIdx, slot: g.outSlot });
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
  | {
      commands: GpuCommand[];
      outputs: Array<{ nodeIndex: number; outputIndex: number; slot: Slot }>;
      firstNodeIndex: number;
    }
  | string {
  const device = (backend as { device?: GPUDevice }).device;
  if (!device) return "no-device";

  // Resolve each adamStep node's [grad, param, m, v, t, lr]: buffer, slot, size.
  // inc-2a: t (slot 4) and lr (slot 5) are 1-element persistent tensor DATA
  // inputs (NOT scatter/gathered — they are shared bindings bound once).
  interface Item {
    nodeIndex: number;
    bufSlots: [Slot, Slot, Slot, Slot, Slot, Slot];
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
    if (node.inputs.length !== 6) return "arity";
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
      bufSlots: bufSlots as [Slot, Slot, Slot, Slot, Slot, Slot],
      numElements: sizeOf(node.shape),
      config: node.payload as import("../backend/types").AdamStepConfig,
    });
  }
  if (itemList.length === 0) return "no-items";

  const commands: GpuCommand[] = [];
  const outputs: Array<{ nodeIndex: number; outputIndex: number; slot: Slot }> =
    [];
  // adamStep is multi-output: param (oi 0), m (oi 1), v (oi 2). All three are
  // updated IN PLACE — the node's result reuses the param/m/v INPUT slot
  // (bufSlots = [grad, param, m, v]). The harvest needs slots for ALL three or
  // the persistent m/v optimizer state can't be carried across steps (the
  // genOk bail that kept backward+optimizer plans on the recorded path).
  for (const it of itemList) {
    outputs.push({
      nodeIndex: it.nodeIndex,
      outputIndex: 0,
      slot: it.bufSlots[1],
    });
    outputs.push({
      nodeIndex: it.nodeIndex,
      outputIndex: 1,
      slot: it.bufSlots[2],
    });
    outputs.push({
      nodeIndex: it.nodeIndex,
      outputIndex: 2,
      slot: it.bufSlots[3],
    });
  }

  // PACKED groups (planPackedGroups mirrors dispatchPackedOptimizer's
  // element-count grouping + memory sub-batching), then per-item fallback.
  // planPackedGroups reads only numElements + buffers.length (4 for Adam —
  // grad/param/m/v; t/lr are 1-element shared bindings, NOT packed).
  const packItems = itemList.map((it) => ({
    buffers: it.bufSlots.slice(0, 4) as unknown as GPUBuffer[],
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
    const packedSlots = packed.map((b) =>
      bufferSlot(b as unknown, "persistent"),
    );
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
    // inc-2a: t/lr are the group representative's 1-element persistent slots
    // (bound ONCE for the packed dispatch — the grouping key guarantees every
    // item in this group shares the same t/lr tensors).
    const repItem = itemList[g.indices[0]];
    const bindings = adamBinding(
      planned.plan,
      {
        grad: packedSlots[0],
        param: packedSlots[1],
        m: packedSlots[2],
        v: packedSlots[3],
        t: repItem.bufSlots[4],
        lr: repItem.bufSlots[5],
      },
      infFlag,
    );
    if (!bindings) return "adam-bind-null";
    // inc-2a: config is STATIC (bias correction derived in-kernel from t/lr) —
    // NO TAG_UNIFORM repack. The config buffer is bound once via adamBinding
    // (name === null → planned.plan.configBuffer, baked at plan() time).
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
      t: it.bufSlots[4], // inc-2a: persistent step counter
      lr: it.bufSlots[5], // inc-2a: persistent learning rate
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
    // inc-2a: STATIC config — no TAG_UNIFORM repack (see packed path above).
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
  coverConstFill: boolean,
): { commands: GpuCommand[]; outSlot: Slot } | null {
  if (node.dtype !== "f32" && node.dtype !== undefined) return null;
  const elements = sizeOf(node.shape);
  const bytes = elements * F32_BYTES;
  switch (node.op) {
    case "full": {
      // Compile-time-CONSTANT full([...], const) — the grad-clip `1.0` ceiling
      // (clipCoef = minimum(., 1.0)), GradScaler no-op scales. Emit a plan-owned
      // constFill slot: a FIXED buffer born at build, pre-filled with the host
      // constant, reused byte-identical every replay. NO alloc/write command —
      // it is neither a pool acquire nor a harvested NodeResult, so it cannot
      // repeat the reverted generateDataSource(full) ALLOC+WRITE over-harvest.
      // Only for RECURRING templates (coverConstFill): a transient warmup plan
      // stays lowered so its one-shot buffer never leaks. A full whose
      // fillValue is not a finite host number falls through to bail.
      const fillValue = (node.payload as { fillValue?: number } | undefined)
        ?.fillValue;
      if (
        !coverConstFill ||
        fillValue === undefined ||
        !Number.isFinite(fillValue)
      )
        return null;
      const slot = slots.length;
      slots.push({ kind: "constFill", elements, fillValue, nodeIndex });
      return { commands: [], outSlot: slot };
    }
    case "tensorFromArray": {
      const slot = slots.length;
      slots.push({ kind: "arena" });
      return {
        commands: [
          { tag: TAG_ALLOC, slot, bytes, allocKind: 0, inputSlots: [] },
          { tag: TAG_WRITE, slot, nodeIndex },
        ],
        outSlot: slot,
      };
    }
    case "arange": {
      // arange is a compile-time-CONSTANT index ramp (start + i*step), fully
      // determined by shape + payload — NOT rng state (unlike rand/randn/
      // bernoulli, which stay bailed). Emit a plan-owned constant buffer (the
      // constFill mechanism, array form): the ramp is materialized at build and
      // reused byte-identical every replay, sitting OUTSIDE the arena AND the
      // harvest ledger — so it sidesteps the legacy TAG_WRITE-allocates-fresh
      // problem that keeps rng data-sources uncovered. Only for RECURRING
      // templates (coverConstFill); a transient warmup arange stays lowered so
      // its one-shot buffer never leaks (same gate as `full`).
      if (!coverConstFill) return null;
      const ap = node.payload as { start?: number; step?: number } | undefined;
      const start = ap?.start ?? 0;
      const step = ap?.step ?? 1;
      // Match the arange_tile WGSL bit-for-bit: `start + f32(i)*step`, every op
      // in f32 (Math.fround), so a non-integer step can't drift 1 ULP from the
      // GPU (the pre-fill is the elided dispatch's exact output). Float32Array
      // assignment already rounds the final add.
      const data = new Float32Array(elements);
      for (let i = 0; i < elements; i++)
        data[i] = start + Math.fround(Math.fround(i) * step);
      const slot = slots.length;
      slots.push({ kind: "constFill", elements, data, nodeIndex });
      return { commands: [], outSlot: slot };
    }
    // NOTE (2026-07-12): `full` IS now generated — as a constFill slot (see the
    // `full` case above), NOT via the TAG_WRITE legacy path. The reason the old
    // TAG_WRITE approach leaked is exactly why constFill sidesteps it: the
    // legacy executeOpSync path ALLOCATES A FRESH buffer each replay (designed
    // for one-time weight loading, not per-step recurring creation) AND that
    // storage got harvested — a storage-ledger LEAK (rc drift reachable+13/
    // total+8 on the clip/scaler trainer; tools/t-ledger-attack-probe.ts). A
    // constFill slot is a plan-owned FIXED buffer (no fresh alloc) that is NOT
    // harvested (no NodeResult), so it leaks nothing.
    //
    // arange IS now generated — as a constFill slot (array form; see the
    // `arange` case above), because it is a compile-time-CONSTANT index ramp
    // (start + i*step, deterministic from shape+payload). rand / randn /
    // bernoulli are STILL NOT generated: they carry rng STATE (not a pure
    // function of shape/payload) and would need the TAG_WRITE legacy path to
    // write INTO a planned slot instead of allocating fresh. tensorFromArray is
    // safe (small stable-upload fast path, plan-owned buffer) and zeros is safe
    // (TAG_CLEAR into the slot). Covering rand/etc. needs the TAG_WRITE legacy
    // path to write INTO the planned slot instead of allocating fresh — a
    // separate executor change. They stay bailed (lowered).
    case "zeros": {
      const slot = slots.length;
      slots.push({ kind: "arena" });
      // The executor records a TAG_WRITE for EVERY data-source node
      // (op-agnostic), so a zeros node is ALLOC+CLEAR+WRITE. The write
      // re-uploads zero data the clear already produced — redundant but
      // recorded reality; mirror it (candidate cleanup: skip the write for
      // zeros at the executor seam, which would also shrink replays).
      return {
        commands: [
          { tag: TAG_ALLOC, slot, bytes, allocKind: 0, inputSlots: [] },
          { tag: TAG_CLEAR, slot, bytes: alignBufferSize(bytes) },
          { tag: TAG_WRITE, slot, nodeIndex },
        ],
        outSlot: slot,
      };
    }
    default:
      return null;
  }
}
