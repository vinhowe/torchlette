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
  detectMatmulEpilogue,
  detectMatmulEpilogueCore,
  executeMatmulWithEpilogue,
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
} from "./reduction-preamble";
import { releaseDeadTensors } from "./storage-tracker";

/** Build a map of nodeId → consumer count from plan nodes. */
function buildConsumerCount(nodes: LazyIRNode[]): Map<number, number> {
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
    // Clean up temporary contiguous copies (deferred destroy for GPU fence)
    for (const temp of tempContiguousCopies) {
      if (temp.destroy) temp.destroy();
    }

    setCurrentOpLabel(null);

    // Re-execute intermediates that are consumed outside the group but
    // couldn't be promoted to additional outputs (shape mismatch / binding limit).
    // The fused kernel computed the chain inline; we re-execute just the needed
    // nodes so external consumers can access their results.
    if (group.neededIntermediates && group.neededIntermediates.length > 0) {
      await executeSequentialSegment(group.neededIntermediates, backend);
    }
  } catch (e) {
    // Clean up temporary contiguous copies on failure too
    for (const temp of tempContiguousCopies) {
      if (temp.destroy) temp.destroy();
    }
    setCurrentOpLabel(null);
    // Fusion failed - fall back to sequential
    recordFusionFallback("exception", group.nodes.length, { error: String(e) });
    console.warn("Fusion dispatch failed, falling back to sequential:", e);
    await executeSequentialSegment(group.nodes, backend);
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

      const nodeBackend = getBackend(node.device) ?? backend;
      setProfileModule(node.module ?? "unknown");
      const inputs = node.inputs.map((ref) =>
        getInputStorage(ref, nodeBackend),
      );
      const backendInputs = inputs.map((s) => s.backendTensor);

      const resultTensor = await executeOp(node, backendInputs, nodeBackend);
      node.result = wrapResultAsStorage(
        node.device,
        resultTensor,
        backendInputs,
        inputs,
      );
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
  allPlanNodes?: LazyIRNode[];
  matmulPrologueMap?: Map<number, MatmulPrologueInfo[]>;
  prologueSkipIds?: Set<number>;
  prebuiltConsumerCount?: Map<number, number>;
  loweredPlanBuilder?: LoweredPlanBuilder | null;
  nodeIdToFinalPos?: Map<number, number>;
  compoundMatchMap?: Map<number, CompoundMatchExec>;
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
    allPlanNodes,
    matmulPrologueMap,
    prologueSkipIds,
    prebuiltConsumerCount,
    loweredPlanBuilder,
    nodeIdToFinalPos,
    compoundMatchMap,
  } = options;
  const useSharedEncoder = backend.name === "webgpu";
  if (useSharedEncoder) beginSharedEncoder();

  try {
    // Use pre-built consumer count if provided, otherwise build it.
    const reductionConsumerCount =
      prebuiltConsumerCount ??
      (backend.name === "webgpu"
        ? buildConsumerCount(allPlanNodes ?? nodes)
        : new Map<number, number>());

    // Intra-segment periodic reclamation: flush pending buffers to main pool
    // every N nodes so freed intermediates can be reused within the same segment.
    // Uses the safe pattern: flushSharedEncoder() submits all encoded work, then
    // flushBufferPool() moves pending→pool. Subsequent dispatches encode on a
    // fresh encoder, and WebGPU queue ordering guarantees prior work completes first.
    const INTRA_SEGMENT_RECLAIM_INTERVAL = DEFAULT_RECLAIM_INTERVAL;
    let nodesSinceIntraReclaim = 0;

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
          loweredPlanBuilder.recordPrologueSkip(
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
            loweredPlanBuilder.recordPrologueSkip(
              nodeIdToFinalPos.get(node.id) as number,
            );
          }
          step++;
          continue;
        }

        // Execute compound kernel. We lazy-import to avoid circular deps.
        const { dispatchFusedSoftmax } = await import(
          "../backend/webgpu/softmax-kernel"
        );

        // Resolve input: the max node's input[0]
        const nodeBackend = getBackend(node.device) ?? backend;
        const inputStorage = getInputStorage(node.inputs[0], nodeBackend);
        const inputBT = asGPUTensor(inputStorage.backendTensor);

        // Compute reduction geometry: numRows = product of dims before dim,
        // dimSize = shape[dim]
        const outputNodeIdx = nodes.findIndex(
          (n) => n.id === match.outputNodeId,
        );
        const outputNode = nodes[outputNodeIdx];
        const shape = inputBT.shape;
        const dim = match.dim < 0 ? shape.length + match.dim : match.dim;
        const dimSize = shape[dim];
        let numRows = 1;
        for (let d = 0; d < dim; d++) numRows *= shape[d];

        const isLog = match.name === "log_softmax";
        const outBuffer = dispatchFusedSoftmax(
          inputBT.buffer,
          numRows,
          dimSize,
          isLog,
        );

        // Create result storage on the output node
        const outShape = shape.slice();
        const outStrides = contiguousStrides(outShape);
        outputNode.result = createStorageHandle(node.device, {
          buffer: outBuffer,
          shape: outShape,
          dtype: "f32",
          size: sizeOf(outShape),
          strides: outStrides,
          offset: 0,
          isContiguous: true,
          ownsBuffer: true,
        });

        const coveredCount = match.coveredNodeIds.size;
        advanceConsumed(nodeIdx, coveredCount);

        // Record compound action in lowered plan builder
        if (loweredPlanBuilder && nodeIdToFinalPos) {
          const coveredPoss: number[] = [];
          for (let c = 0; c < coveredCount; c++) {
            coveredPoss.push(
              nodeIdToFinalPos.get(nodes[nodeIdx + c].id) as number,
            );
          }
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
        let epiloguePlan = prebuiltConsumerCount
          ? detectMatmulEpilogueCore(
              nodes,
              nodeIdx,
              prebuiltConsumerCount,
              externalNodeIds,
            )
          : detectMatmulEpilogue(
              nodes,
              nodeIdx,
              allPlanNodes ?? nodes,
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
            executeMatmulWithEpilogue(node, epiloguePlan, backend),
          );

          advanceConsumed(nodeIdx, epiloguePlan.consumedCount);

          // Record matmul epilogue action in lowered plan builder
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            const covered: number[] = [];
            for (let skip = 0; skip < epiloguePlan.consumedCount; skip++) {
              covered.push(
                nodeIdToFinalPos.get(nodes[nodeIdx + skip].id) as number,
              );
            }
            loweredPlanBuilder.recordMatmulEpilogue(
              nodeIdToFinalPos.get(node.id) as number,
              covered,
              nodeIdToFinalPos.get(epiloguePlan.outputNode.id) as number,
              epiloguePlan.epilogueOps,
              epiloguePlan.epilogueInputRefs.length,
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
        const fusionPlan = detectReductionFusion(
          nodes,
          nodeIdx,
          reductionConsumerCount,
          externalNodeIds,
        );
        if (fusionPlan) {
          const fusionLabel = `${fusionPlan.isMean ? "mean" : "sum"}+${fusionPlan.preambleChain
            .map((n) => n.op)
            .join(
              "+",
            )}+${fusionPlan.epilogueOps.map((o) => (o.kind === "binary" ? o.op : o.kind === "cast" ? "cast" : o.op || o.kind)).join("+")}`;
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
              fusionPlan.epilogueInputRefs.length,
              fusionPlan.outputDtype,
              fusionPlan.consumedCount,
              fusionPlan.isMean,
            );
          }

          nodeIdx += consumed - 1;
          continue;
        }

        // Fall back to preamble-only fusion (Phase 3)
        const reductionPlan = detectReductionPreamble(
          nodes,
          nodeIdx,
          reductionConsumerCount,
        );
        if (reductionPlan) {
          const rpLabel = `${reductionPlan.isMean ? "mean" : "sum"}+${
            reductionPlan.preambleChain
              ? reductionPlan.preambleChain.map((n) => n.op).join("+")
              : reductionPlan.op
          }`;
          await withProfileContext(rpLabel, node.module, () =>
            executeReductionWithPreamble(reductionPlan, backend),
          );

          const consumed = reductionPlan.consumedCount;
          advanceConsumed(nodeIdx, consumed);

          if (loweredPlanBuilder && nodeIdToFinalPos) {
            if (
              reductionPlan.preambleChain &&
              reductionPlan.chainOps &&
              reductionPlan.chainInputDtypes
            ) {
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
            } else {
              loweredPlanBuilder.recordReductionPreamble(
                nodeIdToFinalPos.get(node.id) as number,
                nodeIdToFinalPos.get(reductionPlan.reductionNode.id) as number,
              );
            }
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
        const epiloguePlan = detectReductionEpilogue(
          nodes,
          nodeIdx,
          reductionConsumerCount,
          externalNodeIds,
        );
        if (epiloguePlan) {
          const reLabel =
            node.op +
            "+" +
            epiloguePlan.epilogueOps
              .map((o) =>
                o.kind === "binary"
                  ? o.op
                  : o.kind === "cast"
                    ? "cast"
                    : o.op || o.kind,
              )
              .join("+");
          await withProfileContext(reLabel, node.module, () =>
            executeReductionWithEpilogue(epiloguePlan, backend),
          );

          advanceConsumed(nodeIdx, epiloguePlan.consumedCount);

          // Record reduction epilogue action in lowered plan builder
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            const covered: number[] = [];
            for (let c = 0; c < epiloguePlan.consumedCount; c++) {
              covered.push(
                nodeIdToFinalPos.get(nodes[nodeIdx + c].id) as number,
              );
            }
            loweredPlanBuilder.recordReductionEpilogue(
              nodeIdToFinalPos.get(node.id) as number,
              covered,
              nodeIdToFinalPos.get(epiloguePlan.outputNode.id) as number,
              epiloguePlan.epilogueOps,
              epiloguePlan.epilogueInputRefs.length,
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
            const adamIndices: number[] = [];
            for (let a = 0; a < adamCount; a++) {
              adamIndices.push(
                nodeIdToFinalPos.get(nodes[nodeIdx + a].id) as number,
              );
            }
            loweredPlanBuilder.recordAdamBatch(adamIndices);
          }

          nodeIdx += adamCount - 1;
          continue;
        }
      }

      const nodeBackend = getBackend(node.device) ?? backend;
      setProfileModule(node.module ?? "unknown");
      const inputs = node.inputs.map((ref) =>
        getInputStorage(ref, nodeBackend),
      );
      const backendInputs = inputs.map((s) => s.backendTensor);

      const resultTensor = await executeOp(node, backendInputs, nodeBackend);
      node.result = wrapResultAsStorage(
        node.device,
        resultTensor,
        backendInputs,
        inputs,
      );

      // Record action in lowered plan builder
      if (loweredPlanBuilder && nodeIdToFinalPos) {
        const finalPos = nodeIdToFinalPos.get(node.id) as number;
        if (isDataSourceOp(node.op)) {
          loweredPlanBuilder.recordDataSource(finalPos);
        } else if (isViewOp(node.op)) {
          loweredPlanBuilder.recordView(finalPos);
        } else {
          loweredPlanBuilder.recordSequential(finalPos);
        }
      }

      advanceConsumed(nodeIdx, 1);

      // Periodic intra-segment reclamation
      nodesSinceIntraReclaim++;
      if (
        useSharedEncoder &&
        enableEarlyRelease &&
        nodesSinceIntraReclaim >= INTRA_SEGMENT_RECLAIM_INTERVAL
      ) {
        flushSharedEncoder();
        flushBufferPool();
        if (loweredPlanBuilder) loweredPlanBuilder.recordReclaim();
        nodesSinceIntraReclaim = 0;
      }
    }
  } finally {
    if (useSharedEncoder) endSharedEncoder();
  }
}
