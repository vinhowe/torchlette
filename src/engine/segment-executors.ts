import { getBackend } from "../backend/registry";
import type { Backend, BackendTensor, DType } from "../backend/types";
import {
  beginSharedEncoder,
  endSharedEncoder,
  flushBufferPool,
  flushSharedEncoder,
  setAdamBatchMode,
  setCurrentOpLabel,
} from "../backend/webgpu";
import { asGPUTensor, type GPUBuffer } from "../backend/webgpu/gpu-types";
import {
  recordFusionFallback,
  setProfileModule,
} from "../backend/webgpu/profiler";
import { contiguousStrides, sizeOf } from "../core/shape";
import { executeNode } from "./executor-sequential";
import {
  type FusionGroup,
  type groupToRecipe,
  isFusibleOp,
} from "./fusion-detect";
import type { LazyIRNode, StorageHandle } from "./lazy-types";
import type { TensorLifetime } from "./lifetime-analysis";
import {
  isDataSourceOp,
  isViewOp,
  type LoweredPlanBuilder,
} from "./lowered-plan";
import type { MatmulPrologueInfo } from "./matmul-epilogue";
import {
  detectMatmulEpilogueCore,
  executeMatmulWithEpilogue,
  formatEpilogueLabel,
} from "./matmul-epilogue";
import {
  _webgpuMatmulImports,
  createStorageHandle,
  ensureWebGPUMatmulImports,
  wrapResultAsStorage,
} from "./node-factory";
import { executeOp, getInputStorage, withProfileContext } from "./op-dispatch";
import {
  detectReductionEpilogue,
  detectReductionFusion,
  detectReductionPreamble,
  executeReductionWithEpilogue,
  executeReductionWithFusion,
  executeReductionWithPreamble,
  type ReductionDirective,
} from "./reduction-preamble";
import { releaseDeadTensors } from "./storage-tracker";

/**
 * Execute a compound softmax/log_softmax pattern as a single fused kernel.
 * Shared by both the first-time execution path (segment-executors) and the
 * lowered plan replay path (executor-lowered).
 */
export async function executeCompoundSoftmax(
  inputNode: LazyIRNode,
  outputNode: LazyIRNode,
  dim: number,
  name: string,
  backend: Backend,
): Promise<void> {
  const { dispatchFusedSoftmax } = await import(
    "../backend/webgpu/softmax-kernel"
  );
  const nodeBackend = getBackend(inputNode.device) ?? backend;
  const inputStorage = getInputStorage(inputNode.inputs[0], nodeBackend);
  const inputBT = asGPUTensor(inputStorage.backendTensor);
  const shape = inputBT.shape;
  const normDim = dim < 0 ? shape.length + dim : dim;
  const dimSize = shape[normDim];
  let numRows = 1;
  for (let d = 0; d < normDim; d++) numRows *= shape[d];
  const isLog = name === "log_softmax";
  const outBuffer = dispatchFusedSoftmax(
    inputBT.buffer,
    numRows,
    dimSize,
    isLog,
  );
  const outShape = shape.slice();
  outputNode.result = createStorageHandle(inputNode.device, {
    buffer: outBuffer,
    shape: outShape,
    dtype: "f32",
    size: sizeOf(outShape),
    strides: contiguousStrides(outShape),
    offset: 0,
    isContiguous: true,
    ownsBuffer: true,
  });
}

/** Build a map of nodeId → consumer count from plan nodes. */
export function buildConsumerCount(nodes: LazyIRNode[]): Map<number, number> {
  const counts = new Map<number, number>();
  for (const n of nodes) {
    for (const ref of n.inputs) {
      if (ref.kind === "pending") {
        counts.set(ref.node.id, (counts.get(ref.node.id) ?? 0) + 1);
      }
    }
  }
  return counts;
}

/** Collect final positions for a range of nodes. */
function collectNodePositions(
  nodes: LazyIRNode[],
  startIdx: number,
  count: number,
  posMap: Map<number, number>,
): number[] {
  const result: number[] = [];
  for (let i = 0; i < count; i++) {
    result.push(posMap.get(nodes[startIdx + i].id) as number);
  }
  return result;
}

/**
 * Execution descriptor for a compound pattern match.
 * Built from CompoundMatch by the executor before passing to segment executors.
 */
export interface CompoundMatchExec {
  /** Pattern name: "softmax" or "log_softmax". */
  name: string;
  /** Set of covered node IDs (intermediates + output). */
  coveredNodeIds: Set<number>;
  /** Node ID of the final output. */
  outputNodeId: number;
  /** Reduction dimension (normalized). */
  dim: number;
}

/** Default reclaim interval, overridable via TORCHLETTE_RECLAIM_INTERVAL env var. */
export const DEFAULT_RECLAIM_INTERVAL =
  typeof process !== "undefined" && process.env?.TORCHLETTE_RECLAIM_INTERVAL
    ? parseInt(process.env.TORCHLETTE_RECLAIM_INTERVAL, 10)
    : 10000;

/**
 * Tracks node count since last buffer reclamation and flushes when threshold is reached.
 * Used by both executor-optimized (batch-level) and segment-executors (per-node).
 */
export function createReclaimController(
  interval: number,
  builder: LoweredPlanBuilder | null,
) {
  let count = 0;
  return {
    advance(n: number) {
      count += n;
    },
    /** Flush if count >= interval and active is true. Returns whether it flushed. */
    maybeFlush(active: boolean): boolean {
      if (active && interval > 0 && count >= interval) {
        flushSharedEncoder();
        flushBufferPool();
        if (builder) builder.recordReclaim();
        count = 0;
        return true;
      }
      return false;
    },
  };
}

/**
 * Execute a fused segment using a fused kernel.
 * For WebGPU, dispatches a generated kernel. For other backends, falls back to sequential.
 */
export async function executeFusedSegment(
  group: FusionGroup,
  recipe: ReturnType<typeof groupToRecipe>,
  backend: Backend,
  enableVectorization: boolean,
): Promise<void> {
  // For CPU or other backends, fall back to sequential execution
  if (backend.name !== "webgpu" || !("dispatchFusedKernel" in backend)) {
    await executeSequentialSegment(group.nodes, backend);
    return;
  }
  // Import fusion dispatch and buffer lifecycle helpers (cached on first call)
  const fusionDispatch = await import("../backend/webgpu/fusion-dispatch");
  const { dispatchFusedKernel } = fusionDispatch;
  await ensureWebGPUMatmulImports();
  const { deferredDestroyBuffer } = _webgpuMatmulImports as NonNullable<
    typeof _webgpuMatmulImports
  >;

  /** Wrap a fusion output buffer into a StorageHandle with deferred destroy. */
  const wrapFusionOutput = (
    device: string,
    output: { buffer: unknown; shape: number[]; dtype: DType },
  ): StorageHandle => {
    const buf = output.buffer as GPUBuffer;
    const bufSize = buf.size;
    let destroyed = false;
    return createStorageHandle(device, {
      buffer: output.buffer,
      shape: output.shape,
      dtype: output.dtype,
      size: sizeOf(output.shape),
      strides: contiguousStrides(output.shape),
      offset: 0,
      isContiguous: true,
      ownsBuffer: true,
      destroy() {
        if (destroyed) return;
        destroyed = true;
        deferredDestroyBuffer(buf, bufSize);
      },
    } as BackendTensor);
  };

  // Get WebGPU device from backend (narrowed by the webgpu check above)
  const device = (
    backend as Backend & {
      device?: { limits?: { maxStorageBuffersPerShaderStage?: number } };
    }
  ).device;
  if (!device) {
    // No device available - fall back to sequential
    recordFusionFallback("no_device", group.nodes.length);
    await executeSequentialSegment(group.nodes, backend);
    return;
  }

  // Check storage buffer limit before attempting fusion.
  // Each fused kernel needs inputs + outputs storage bindings.
  // Inlined constants don't consume binding slots.
  // If we'd exceed the device limit, skip fusion silently (no console.warn spam).
  const maxStorageBuffers = device.limits?.maxStorageBuffersPerShaderStage ?? 8;
  const numOutputs = recipe.outputs?.length ?? 1;
  const nonInlinedInputCount = recipe.inputs.filter(
    (inp) => !inp.isInlinedConstant,
  ).length;
  const requiredBindings = nonInlinedInputCount + numOutputs;
  if (requiredBindings > maxStorageBuffers) {
    recordFusionFallback("binding_limit", group.nodes.length, {
      required: requiredBindings,
      max: maxStorageBuffers,
    });
    await executeSequentialSegment(group.nodes, backend);
    return;
  }

  // Prepare inputs from external refs, skipping inlined constants
  const inputs: Array<{ buffer: GPUBuffer; shape: number[]; dtype: DType }> =
    [];
  const tempContiguousCopies: Array<{ destroy?: () => void }> = [];
  for (let inputIdx = 0; inputIdx < group.externalInputs.length; inputIdx++) {
    // Skip inlined constants — their values are baked into the shader
    // This handles both scalar LazyRefs and pending nodes detected as inlinable
    if (recipe.inputs[inputIdx]?.isInlinedConstant) {
      continue;
    }

    const inputRef = group.externalInputs[inputIdx];
    // Scalar refs should always be inlined — this is a safety fallback
    if (inputRef.kind === "scalar") {
      continue;
    }
    const storage =
      inputRef.kind === "materialized"
        ? inputRef.storage
        : inputRef.node.result;

    if (!storage) {
      // Input not materialized - fall back to sequential
      recordFusionFallback("not_materialized", group.nodes.length);
      await executeSequentialSegment(group.nodes, backend);
      return;
    }

    const tensor = asGPUTensor(storage.backendTensor);
    // Fusion requires contiguous inputs — strided/offset layouts not supported by codegen
    if (
      tensor.isContiguous === false ||
      (tensor.offset != null && tensor.offset > 0)
    ) {
      // Auto-materialize to contiguous rather than abandoning fusion
      if (backend.ops.contiguous) {
        const contig = asGPUTensor(backend.ops.contiguous(tensor));
        tempContiguousCopies.push(contig);
        inputs.push({
          buffer: contig.buffer,
          shape: contig.shape ?? tensor.shape ?? [1],
          dtype: (contig.dtype as DType) ?? (tensor.dtype as DType) ?? "f32",
        });
        continue;
      }
      // No contiguous op — fall back
      recordFusionFallback("non_contiguous", group.nodes.length, {
        shape: tensor.shape,
        isContiguous: tensor.isContiguous,
        offset: tensor.offset,
      });
      await executeSequentialSegment(group.nodes, backend);
      return;
    }
    inputs.push({
      buffer: tensor.buffer,
      shape: tensor.shape ?? [1],
      dtype: (tensor.dtype as DType) ?? "f32",
    });
  }

  // Check if any input buffer exceeds maxStorageBufferBindingSize
  const maxBindingSize =
    device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const hasOversizedBuffer = inputs.some(
    (inp) => inp.buffer.size > maxBindingSize,
  );
  if (hasOversizedBuffer) {
    recordFusionFallback("oversized_buffer", group.nodes.length, {
      maxBindingSize,
    });
    await executeSequentialSegment(group.nodes, backend);
    return;
  }

  try {
    // Set module context for profiling from the output node
    setProfileModule(group.outputNode.module ?? "unknown");
    // Build a label from the group's unique op names (e.g. "add+mul+relu")
    const fusionLabel = [...new Set(group.nodes.map((n) => n.op))].join("+");
    setCurrentOpLabel(fusionLabel);
    // Dispatch the fused kernel
    const result = dispatchFusedKernel(device, recipe, inputs, {
      vectorize: enableVectorization,
    });

    // Store results on output nodes
    group.outputNode.result = wrapFusionOutput(group.outputNode.device, result);
    if (group.additionalOutputNodes && result.outputs) {
      for (let i = 0; i < group.additionalOutputNodes.length; i++) {
        const addNode = group.additionalOutputNodes[i];
        const addOutput = result.outputs[i + 1]; // +1: primary is at index 0
        if (addOutput)
          addNode.result = wrapFusionOutput(addNode.device, addOutput);
      }
    }
    // Re-execute intermediates that are consumed outside the group but
    // couldn't be promoted to additional outputs (shape mismatch / binding limit).
    // The fused kernel computed the chain inline; we re-execute just the needed
    // nodes so external consumers can access their results.
    if (group.neededIntermediates && group.neededIntermediates.length > 0) {
      await executeSequentialSegment(group.neededIntermediates, backend);
    }
  } catch (e) {
    // Fusion failed - fall back to sequential
    recordFusionFallback("exception", group.nodes.length, { error: String(e) });
    console.warn("Fusion dispatch failed, falling back to sequential:", e);
    await executeSequentialSegment(group.nodes, backend);
  } finally {
    for (const temp of tempContiguousCopies) {
      temp.destroy?.();
    }
    setCurrentOpLabel(null);
  }
}

/**
 * Execute nodes sequentially (standard execution).
 * Used as a fallback by fusion dispatch when fused execution can't proceed
 * (e.g., binding limits, non-contiguous inputs, oversized buffers).
 * Fusion groups contain only elementwise ops, so no pattern matching needed.
 */
async function executeSequentialSegment(
  nodes: LazyIRNode[],
  backend: Backend,
): Promise<void> {
  const useSharedEncoder = backend.name === "webgpu";
  if (useSharedEncoder) beginSharedEncoder();

  try {
    for (const node of nodes) {
      if (node.result) continue;
      await executeNode(node, backend);
    }
  } finally {
    if (useSharedEncoder) endSharedEncoder();
  }
}

/**
 * Options for sequential segment execution with early release and lowered plan recording.
 */
interface SegmentExecOptions {
  enableEarlyRelease: boolean;
  lifetimes: Map<number, TensorLifetime> | null;
  outputNodeIds: Set<number> | null;
  alreadyReleased: Set<number>;
  nodeToStorage: Map<number, StorageHandle>;
  startStep: number;
  externalNodeIds?: Set<number>;
  matmulPrologueMap?: Map<number, MatmulPrologueInfo[]>;
  prologueSkipIds?: Set<number>;
  prebuiltConsumerCount?: Map<number, number>;
  loweredPlanBuilder?: LoweredPlanBuilder | null;
  nodeIdToFinalPos?: Map<number, number>;
  compoundMatchMap?: Map<number, CompoundMatchExec>;
  reductionDirectives?: Map<number, ReductionDirective>;
}

/**
 * Execute nodes sequentially with early buffer release support.
 */
export async function executeSequentialSegmentWithEarlyRelease(
  nodes: LazyIRNode[],
  backend: Backend,
  options: SegmentExecOptions,
): Promise<void> {
  const {
    enableEarlyRelease,
    lifetimes,
    outputNodeIds,
    alreadyReleased,
    nodeToStorage,
    startStep,
    externalNodeIds,
    matmulPrologueMap,
    prologueSkipIds,
    prebuiltConsumerCount,
    loweredPlanBuilder,
    nodeIdToFinalPos,
    compoundMatchMap,
    reductionDirectives,
  } = options;
  const useSharedEncoder = backend.name === "webgpu";
  if (useSharedEncoder) beginSharedEncoder();

  try {
    // Use pre-built consumer count if provided, otherwise build from local nodes.
    const reductionConsumerCount =
      prebuiltConsumerCount ??
      (backend.name === "webgpu"
        ? buildConsumerCount(nodes)
        : new Map<number, number>());

    // Intra-segment periodic reclamation: flush pending buffers to main pool
    // every N nodes so freed intermediates can be reused within the same segment.
    const reclaim = createReclaimController(
      DEFAULT_RECLAIM_INTERVAL,
      loweredPlanBuilder ?? null,
    );

    let step = startStep;

    /** Track storages for consumed nodes and release dead buffers. */
    const advanceConsumed = (nodeIdx: number, count: number) => {
      if (enableEarlyRelease) {
        for (let c = 0; c < count; c++) {
          const n = nodes[nodeIdx + c];
          if (n.result) nodeToStorage.set(n.id, n.result);
          step++;
          releaseDeadTensors(
            lifetimes,
            step,
            outputNodeIds,
            alreadyReleased,
            nodeToStorage,
          );
        }
      } else {
        step += count;
      }
    };

    for (let nodeIdx = 0; nodeIdx < nodes.length; nodeIdx++) {
      const node = nodes[nodeIdx];
      if (node.result) {
        if (enableEarlyRelease) {
          nodeToStorage.set(node.id, node.result);
        }
        step++;
        continue;
      }

      // Skip prologue-claimed cast nodes — their work is absorbed into the
      // matmul tile load. They stay in the plan for topological ordering but
      // don't execute. No result is set since the only consumer (the matmul)
      // uses the pre-cast input via prologue info.
      if (prologueSkipIds?.has(node.id)) {
        if (loweredPlanBuilder && nodeIdToFinalPos) {
          loweredPlanBuilder.recordNode(
            "prologue-skip",
            nodeIdToFinalPos.get(node.id) as number,
          );
        }
        step++;
        continue;
      }

      // Handle compound patterns (softmax, log_softmax). The compoundMatchMap
      // maps the FIRST covered node ID (in topological order) to the match.
      // When we encounter this first node, execute the fused kernel and set
      // results on the output node. Remaining covered nodes are skipped
      // (their IDs are also in the map as skip entries with name === "").
      if (compoundMatchMap?.has(node.id)) {
        const match = compoundMatchMap.get(node.id) as CompoundMatchExec;
        if (match.name === "") {
          // This is an intermediate/non-first covered node — skip it
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            loweredPlanBuilder.recordNode(
              "prologue-skip",
              nodeIdToFinalPos.get(node.id) as number,
            );
          }
          step++;
          continue;
        }

        const outputNode =
          nodes[nodes.findIndex((n) => n.id === match.outputNodeId)];
        await executeCompoundSoftmax(
          node,
          outputNode,
          match.dim,
          match.name,
          backend,
        );

        const coveredCount = match.coveredNodeIds.size;
        advanceConsumed(nodeIdx, coveredCount);

        // Record compound action in lowered plan builder
        if (loweredPlanBuilder && nodeIdToFinalPos) {
          const coveredPoss = collectNodePositions(
            nodes,
            nodeIdx,
            coveredCount,
            nodeIdToFinalPos,
          );
          loweredPlanBuilder.recordCompound(
            match.name,
            coveredPoss,
            nodeIdToFinalPos.get(match.outputNodeId) as number,
            match.dim,
          );
        }

        nodeIdx += coveredCount - 1;
        continue;
      }

      // Try matmul epilogue/prologue fusion (Phase 1)
      if (node.op === "matmul" && backend.name === "webgpu") {
        let epiloguePlan = detectMatmulEpilogueCore(
          nodes,
          nodeIdx,
          reductionConsumerCount,
          externalNodeIds,
        );
        const prologues = matmulPrologueMap?.get(node.id);

        // If we have prologues but no epilogue, create a minimal (empty) epilogue plan
        // so the matmul goes through the epilogue dispatch path with prologue support.
        if (!epiloguePlan && prologues && prologues.length > 0) {
          epiloguePlan = {
            consumedCount: 1, // just the matmul itself
            epilogueOps: [],
            epilogueInputRefs: [],
            outputDtype: node.dtype,
            outputNode: node,
          };
        }

        // Attach prologues to the plan
        if (epiloguePlan && prologues && prologues.length > 0) {
          epiloguePlan.prologues = prologues;
        }

        if (epiloguePlan) {
          const prologueLabel = epiloguePlan.prologues ? "prologue+" : "";
          const epilogueLabel =
            epiloguePlan.epilogueOps.length > 0
              ? "+" + epiloguePlan.epilogueOps.map((o) => o.kind).join("+")
              : "";
          const epLabel = `matmul+${prologueLabel}${epilogueLabel}`.replace(
            /\+$/,
            "",
          );
          await withProfileContext(epLabel, node.module, () =>
            executeMatmulWithEpilogue(node, epiloguePlan),
          );

          advanceConsumed(nodeIdx, epiloguePlan.consumedCount);

          // Record matmul epilogue action in lowered plan builder
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            const covered = collectNodePositions(
              nodes,
              nodeIdx,
              epiloguePlan.consumedCount,
              nodeIdToFinalPos,
            );
            loweredPlanBuilder.recordMatmulEpilogue(
              nodeIdToFinalPos.get(node.id) as number,
              covered,
              nodeIdToFinalPos.get(epiloguePlan.outputNode.id) as number,
              epiloguePlan.epilogueOps,
              epiloguePlan.outputDtype,
              epiloguePlan.consumedCount,
              epiloguePlan.prologues?.map((p) => ({
                inputIndex: p.inputIndex,
                castNodeIndex: nodeIdToFinalPos.get(p.castNodeId) as number,
                fromDtype: p.fromDtype,
                toDtype: p.toDtype,
              })),
            );
          }

          nodeIdx += epiloguePlan.consumedCount - 1;
          continue;
        }
      }

      // Try combined preamble + epilogue reduction fusion (Phase 5)
      if (isFusibleOp(node.op) && backend.name === "webgpu") {
        const rdDirective = reductionDirectives?.get(node.id);
        const fusionPlan =
          rdDirective?.kind === "fusion"
            ? rdDirective.plan
            : detectReductionFusion(
                nodes,
                nodeIdx,
                reductionConsumerCount,
                externalNodeIds,
              );
        if (fusionPlan) {
          const fusionLabel = `${fusionPlan.isMean ? "mean" : "sum"}+${fusionPlan.preambleChain
            .map((n) => n.op)
            .join("+")}+${formatEpilogueLabel(fusionPlan.epilogueOps)}`;
          await withProfileContext(fusionLabel, node.module, () =>
            executeReductionWithFusion(fusionPlan, backend),
          );

          const consumed = fusionPlan.consumedCount;
          advanceConsumed(nodeIdx, consumed);

          if (loweredPlanBuilder && nodeIdToFinalPos) {
            const preambleIndices = fusionPlan.preambleChain.map(
              (n) => nodeIdToFinalPos.get(n.id) as number,
            );
            const epilogueIndices = fusionPlan.epilogueChain.map(
              (n) => nodeIdToFinalPos.get(n.id) as number,
            );
            loweredPlanBuilder.recordReductionFusion(
              preambleIndices,
              nodeIdToFinalPos.get(fusionPlan.reductionNode.id) as number,
              epilogueIndices,
              nodeIdToFinalPos.get(fusionPlan.outputNode.id) as number,
              fusionPlan.preambleOps,
              fusionPlan.preambleInputDtypes,
              fusionPlan.epilogueOps,
              fusionPlan.outputDtype,
              fusionPlan.consumedCount,
              fusionPlan.isMean,
            );
          }

          nodeIdx += consumed - 1;
          continue;
        }

        // Fall back to preamble-only fusion (Phase 3)
        const reductionPlan =
          rdDirective?.kind === "preamble"
            ? rdDirective.plan
            : detectReductionPreamble(nodes, nodeIdx, reductionConsumerCount);
        if (reductionPlan) {
          const rpLabel = `${reductionPlan.isMean ? "mean" : "sum"}+${reductionPlan.preambleChain.map((n) => n.op).join("+")}`;
          await withProfileContext(rpLabel, node.module, () =>
            executeReductionWithPreamble(reductionPlan, backend),
          );

          const consumed = reductionPlan.consumedCount;
          advanceConsumed(nodeIdx, consumed);

          if (loweredPlanBuilder && nodeIdToFinalPos) {
            loweredPlanBuilder.recordReductionPreamble(
              nodeIdToFinalPos.get(node.id) as number,
              nodeIdToFinalPos.get(reductionPlan.reductionNode.id) as number,
              reductionPlan.preambleChain.map(
                (n) => nodeIdToFinalPos.get(n.id) as number,
              ),
              reductionPlan.chainOps,
              reductionPlan.chainInputDtypes,
              reductionPlan.consumedCount,
            );
          }

          nodeIdx += consumed - 1;
          continue;
        }
      }

      // Try reduction epilogue fusion (Phase 4)
      if (
        (node.op === "sum" || node.op === "mean" || node.op === "max") &&
        backend.name === "webgpu"
      ) {
        const reDirective = reductionDirectives?.get(node.id);
        const epiloguePlan =
          reDirective?.kind === "epilogue"
            ? reDirective.plan
            : detectReductionEpilogue(
                nodes,
                nodeIdx,
                reductionConsumerCount,
                externalNodeIds,
              );
        if (epiloguePlan) {
          const reLabel = `${node.op}+${formatEpilogueLabel(epiloguePlan.epilogueOps)}`;
          await withProfileContext(reLabel, node.module, () =>
            executeReductionWithEpilogue(epiloguePlan, backend),
          );

          advanceConsumed(nodeIdx, epiloguePlan.consumedCount);

          // Record reduction epilogue action in lowered plan builder
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            const covered = collectNodePositions(
              nodes,
              nodeIdx,
              epiloguePlan.consumedCount,
              nodeIdToFinalPos,
            );
            loweredPlanBuilder.recordReductionEpilogue(
              nodeIdToFinalPos.get(node.id) as number,
              covered,
              nodeIdToFinalPos.get(epiloguePlan.outputNode.id) as number,
              epiloguePlan.epilogueOps,
              epiloguePlan.outputDtype,
              epiloguePlan.consumedCount,
            );
          }

          nodeIdx += epiloguePlan.consumedCount - 1;
          continue;
        }
      }

      // Batch consecutive adamStep nodes: flush once before the batch, then
      // execute all Adam nodes without per-op flushes. All 76 params are
      // independent — a single pre-flush resolves all read→read_write conflicts.
      if (node.op === "adamStep" && useSharedEncoder) {
        // Count consecutive adamStep nodes
        let adamCount = 1;
        for (let j = nodeIdx + 1; j < nodes.length; j++) {
          if (nodes[j].op === "adamStep" && !nodes[j].result) adamCount++;
          else break;
        }

        if (adamCount > 1) {
          // Single flush before the entire Adam batch
          flushSharedEncoder();
          flushBufferPool(); // Make released fwd/bwd buffers available for Adam allocs

          setAdamBatchMode(true);
          try {
            for (let a = 0; a < adamCount; a++) {
              const adamNode = nodes[nodeIdx + a];
              if (adamNode.result) {
                if (enableEarlyRelease)
                  nodeToStorage.set(adamNode.id, adamNode.result);
                step++;
                continue;
              }

              const adamBackend = getBackend(adamNode.device) ?? backend;
              const adamInputs = adamNode.inputs.map((ref) =>
                getInputStorage(ref, adamBackend),
              );
              const adamBackendInputs = adamInputs.map((s) => s.backendTensor);

              const adamResult = await withProfileContext(
                "adamStep",
                adamNode.module,
                () => executeOp(adamNode, adamBackendInputs, adamBackend),
              );
              adamNode.result = wrapResultAsStorage(
                adamNode.device,
                adamResult,
                adamBackendInputs,
                adamInputs,
              );

              advanceConsumed(nodeIdx + a, 1);
            }
          } finally {
            setAdamBatchMode(false);
          }

          // Record adam batch action in lowered plan builder
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            const adamIndices = collectNodePositions(
              nodes,
              nodeIdx,
              adamCount,
              nodeIdToFinalPos,
            );
            loweredPlanBuilder.recordAdamBatch(adamIndices);
          }

          nodeIdx += adamCount - 1;
          continue;
        }
      }

      await executeNode(node, backend);

      // Record action in lowered plan builder
      if (loweredPlanBuilder && nodeIdToFinalPos) {
        const finalPos = nodeIdToFinalPos.get(node.id) as number;
        const kind = isDataSourceOp(node.op)
          ? ("data-source" as const)
          : isViewOp(node.op)
            ? ("view" as const)
            : ("sequential" as const);
        loweredPlanBuilder.recordNode(kind, finalPos);
      }

      advanceConsumed(nodeIdx, 1);

      // Periodic intra-segment reclamation
      reclaim.advance(1);
      reclaim.maybeFlush(useSharedEncoder && enableEarlyRelease);
    }
  } finally {
    if (useSharedEncoder) endSharedEncoder();
  }
}
