import type { Backend, BackendTensor, DType } from "../backend/types";
import { getBackend } from "../backend/registry";
import {
  flushBufferPool,
  flushSharedEncoder,
  beginSharedEncoder,
  endSharedEncoder,
  setCurrentOpLabel,
  setAdamBatchMode,
} from "../backend/webgpu";
import { profileOpBegin, profileOpEnd, isProfilingEnabled, setProfileModule, recordFusionFallback } from "../backend/webgpu/profiler";
import {
  isFusibleOp,
  groupToRecipe,
  type FusionGroup,
} from "./fusion-detect";
import {
  findDeadTensorsAtStep,
  type TensorLifetime,
} from "./memory-planning";
import {
  LoweredPlanBuilder,
  isDataSourceOp,
  isViewOp,
} from "./lowered-plan";
import type { LazyIRNode, LazyRef, StorageHandle, ExecutionPlan } from "./lazy-types";
import { createStorageHandle, ensureWebGPUMatmulImports, _webgpuMatmulImports } from "./node-factory";
import { storageTracker, canSafelyRelease, releaseBufferImmediate } from "./storage-tracker";
import { computeContiguousStrides } from "../backend/types";
import { getInputStorage, executeOp } from "./op-dispatch";
import type { MatmulPrologueInfo, MatmulEpiloguePlan } from "./matmul-epilogue";
import { detectMatmulEpilogue, detectMatmulEpilogueCore, executeMatmulWithEpilogue } from "./matmul-epilogue";
import { detectReductionPreamble, executeReductionWithPreamble } from "./reduction-preamble";

/** Default reclaim interval, overridable via TORCHLETTE_RECLAIM_INTERVAL env var. */
export const DEFAULT_RECLAIM_INTERVAL =
  typeof process !== "undefined" && process.env?.TORCHLETTE_RECLAIM_INTERVAL
    ? parseInt(process.env.TORCHLETTE_RECLAIM_INTERVAL, 10)
    : 100;

/**
 * Execute a fused segment using a fused kernel.
 */
export async function executeFusedSegment(
  group: FusionGroup,
  recipe: ReturnType<typeof groupToRecipe>,
  backend: Backend,
  enableVectorization: boolean,
): Promise<void> {
  // For WebGPU, use the fused kernel dispatcher
  if (backend.name === "webgpu" && "dispatchFusedKernel" in backend) {
    await executeFusedWebGPU(
      group,
      recipe,
      backend as any,
      enableVectorization,
    );
    return;
  }

  // For CPU or other backends, fall back to sequential execution
  await executeSequentialSegment(group.nodes, backend);
}

/**
 * Execute a fused segment on WebGPU using generated kernels.
 */
export async function executeFusedWebGPU(
  group: FusionGroup,
  recipe: ReturnType<typeof groupToRecipe>,
  backend: Backend & { device?: unknown },
  enableVectorization: boolean,
): Promise<void> {
  // Import fusion dispatch and buffer lifecycle helpers (cached on first call)
  const fusionDispatch = await import("../backend/webgpu/fusion-dispatch");
  const { dispatchFusedKernel } = fusionDispatch;
  await ensureWebGPUMatmulImports();
  const { deferredDestroyBuffer } = _webgpuMatmulImports!;

  // Get WebGPU device from backend
  const device = (backend as any).device;
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
  const nonInlinedInputCount = recipe.inputs.filter((inp: any) => !inp.isInlinedConstant).length;
  const requiredBindings = nonInlinedInputCount + numOutputs;
  if (requiredBindings > maxStorageBuffers) {
    recordFusionFallback("binding_limit", group.nodes.length, { required: requiredBindings, max: maxStorageBuffers });
    await executeSequentialSegment(group.nodes, backend);
    return;
  }

  // Prepare inputs from external refs, skipping inlined constants
  const inputs: Array<{ buffer: unknown; shape: number[]; dtype: DType }> = [];
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

    const tensor = storage.backendTensor as any;
    // Fusion requires contiguous inputs — strided/offset layouts not supported by codegen
    if (tensor.isContiguous === false || (tensor.offset != null && tensor.offset > 0)) {
      // Auto-materialize to contiguous rather than abandoning fusion
      if (backend.ops.contiguous) {
        const contig = backend.ops.contiguous(tensor) as any;
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
        shape: tensor.shape, isContiguous: tensor.isContiguous, offset: tensor.offset,
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
  const maxBindingSize = device.limits?.maxStorageBufferBindingSize ?? 128 * 1024 * 1024;
  const hasOversizedBuffer = inputs.some(
    (inp) => (inp.buffer as { size?: number }).size! > maxBindingSize,
  );
  if (hasOversizedBuffer) {
    recordFusionFallback("oversized_buffer", group.nodes.length, { maxBindingSize });
    await executeSequentialSegment(group.nodes, backend);
    return;
  }

  try {
    // Set module context for profiling from the output node
    setProfileModule(group.outputNode.module ?? "unknown");
    // Dispatch the fused kernel
    const result = dispatchFusedKernel(device, recipe, inputs as any, {
      vectorize: enableVectorization,
    });

    // Store the result in the output node
    const outputNode = group.outputNode;
    const fusionBuffer = result.buffer as GPUBuffer;
    const fusionBufferSize = (fusionBuffer as unknown as { size: number }).size ?? 0;
    let fusionDestroyed = false;
    outputNode.result = createStorageHandle(outputNode.device, {
      buffer: result.buffer,
      shape: result.shape,
      dtype: result.dtype,
      size: result.shape.reduce((a, b) => a * b, 1),
      strides: computeContiguousStrides(result.shape),
      offset: 0,
      isContiguous: true,
      ownsBuffer: true,
      destroy() {
        if (fusionDestroyed) return;
        fusionDestroyed = true;
        deferredDestroyBuffer(fusionBuffer, fusionBufferSize);
      },
    } as BackendTensor);

    // Multi-output: store results for additional output nodes (§15.2)
    if (group.additionalOutputNodes && result.outputs) {
      for (let i = 0; i < group.additionalOutputNodes.length; i++) {
        const addNode = group.additionalOutputNodes[i];
        const addOutput = result.outputs[i + 1]; // +1: primary is at index 0
        if (addOutput) {
          const addBuffer = addOutput.buffer as GPUBuffer;
          const addBufferSize = (addBuffer as unknown as { size: number }).size ?? 0;
          let addDestroyed = false;
          addNode.result = createStorageHandle(addNode.device, {
            buffer: addOutput.buffer,
            shape: addOutput.shape,
            dtype: addOutput.dtype,
            size: addOutput.shape.reduce((a, b) => a * b, 1),
            strides: computeContiguousStrides(addOutput.shape),
            offset: 0,
            isContiguous: true,
            ownsBuffer: true,
            destroy() {
              if (addDestroyed) return;
              addDestroyed = true;
              deferredDestroyBuffer(addBuffer, addBufferSize);
            },
          } as BackendTensor);
        }
      }
    }
    // Clean up temporary contiguous copies (deferred destroy for GPU fence)
    for (const temp of tempContiguousCopies) {
      if (temp.destroy) temp.destroy();
    }

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
    // Fusion failed - fall back to sequential
    recordFusionFallback("exception", group.nodes.length, { error: String(e) });
    console.warn("Fusion dispatch failed, falling back to sequential:", e);
    await executeSequentialSegment(group.nodes, backend);
  }
}

/**
 * Execute nodes sequentially (standard execution).
 */
export async function executeSequentialSegment(
  nodes: LazyIRNode[],
  backend: Backend,
  externalNodeIds?: Set<number>,
  allPlanNodes?: LazyIRNode[],
): Promise<void> {
  const useSharedEncoder = backend.name === "webgpu";
  if (useSharedEncoder) beginSharedEncoder();

  try {
    // Build consumer count map for reduction preamble detection
    const reductionConsumerCount = new Map<number, number>();
    if (backend.name === "webgpu") {
      for (const n of (allPlanNodes ?? nodes)) {
        for (const ref of n.inputs) {
          if (ref.kind === "pending") {
            reductionConsumerCount.set(ref.node.id, (reductionConsumerCount.get(ref.node.id) ?? 0) + 1);
          }
        }
      }
    }

    for (let nodeIdx = 0; nodeIdx < nodes.length; nodeIdx++) {
      const node = nodes[nodeIdx];
      if (node.result) {
        continue;
      }

      // Try matmul epilogue fusion (Phase 1)
      if (node.op === "matmul" && backend.name === "webgpu") {
        const epiloguePlan = detectMatmulEpilogue(nodes, nodeIdx, allPlanNodes ?? nodes, externalNodeIds);
        if (epiloguePlan) {
          const epLabel = "matmul+" + epiloguePlan.epilogueOps.map(o => o.kind).join("+");
          setCurrentOpLabel(epLabel);
          setProfileModule(node.module ?? "unknown");
          const _profT0 = profileOpBegin(epLabel);
          try {
            await executeMatmulWithEpilogue(node, epiloguePlan, backend);
          } finally {
            profileOpEnd(epLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }
          nodeIdx += epiloguePlan.consumedCount - 1;
          continue;
        }
      }

      // Try reduction preamble fusion (Phase 3)
      if (isFusibleOp(node.op) && backend.name === "webgpu") {
        const reductionPlan = detectReductionPreamble(nodes, nodeIdx, reductionConsumerCount);
        if (reductionPlan) {
          const rpLabel = `${reductionPlan.isMean ? "mean" : "sum"}+${reductionPlan.op}`;
          setCurrentOpLabel(rpLabel);
          setProfileModule(node.module ?? "unknown");
          const _profT0 = profileOpBegin(rpLabel);
          try {
            await executeReductionWithPreamble(reductionPlan, backend);
          } finally {
            profileOpEnd(rpLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }
          nodeIdx += 1; // Skip the reduction node (consumed 2 nodes total)
          continue;
        }
      }

      const nodeBackend = getBackend(node.device) ?? backend;
      const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
      const backendInputs = inputs.map((s) => s.backendTensor);

      let resultTensor = await executeOp(node, backendInputs, nodeBackend);
      const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
      if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
        resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
      }
      const isView =
        (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
      const baseStorageId =
        isView && inputs.length > 0
          ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
          : undefined;
      node.result = createStorageHandle(node.device, resultTensor, baseStorageId);
    }
  } finally {
    if (useSharedEncoder) endSharedEncoder();
  }
}

/**
 * Execute nodes sequentially with early buffer release support.
 */
export async function executeSequentialSegmentWithEarlyRelease(
  nodes: LazyIRNode[],
  backend: Backend,
  enableEarlyRelease: boolean,
  lifetimes: Map<number, TensorLifetime> | null,
  outputNodeIds: Set<number> | null,
  alreadyReleased: Set<number>,
  nodeToStorage: Map<number, StorageHandle>,
  startStep: number,
  externalNodeIds?: Set<number>,
  allPlanNodes?: LazyIRNode[],
  matmulPrologueMap?: Map<number, MatmulPrologueInfo[]>,
  prologueSkipIds?: Set<number>,
  prebuiltConsumerCount?: Map<number, number>,
  loweredPlanBuilder?: LoweredPlanBuilder | null,
  nodeIdToFinalPos?: Map<number, number>,
): Promise<void> {
  const useSharedEncoder = backend.name === "webgpu";
  if (useSharedEncoder) beginSharedEncoder();

  try {
    // Use pre-built consumer count if provided, otherwise build it.
    // The consumer count is used for both reduction preamble and matmul epilogue detection.
    // Building it once per plan (in the caller) instead of once per segment saves ~15ms.
    const reductionConsumerCount = prebuiltConsumerCount ?? new Map<number, number>();
    if (!prebuiltConsumerCount && backend.name === "webgpu") {
      for (const n of (allPlanNodes ?? nodes)) {
        for (const ref of n.inputs) {
          if (ref.kind === "pending") {
            reductionConsumerCount.set(ref.node.id, (reductionConsumerCount.get(ref.node.id) ?? 0) + 1);
          }
        }
      }
    }

    // Intra-segment periodic reclamation: flush pending buffers to main pool
    // every N nodes so freed intermediates can be reused within the same segment.
    // Uses the safe pattern: flushSharedEncoder() submits all encoded work, then
    // flushBufferPool() moves pending→pool. Subsequent dispatches encode on a
    // fresh encoder, and WebGPU queue ordering guarantees prior work completes first.
    const INTRA_SEGMENT_RECLAIM_INTERVAL = DEFAULT_RECLAIM_INTERVAL;
    let nodesSinceIntraReclaim = 0;

    let step = startStep;
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
          loweredPlanBuilder.recordPrologueSkip(nodeIdToFinalPos.get(node.id)!);
        }
        step++;
        continue;
      }

      // Try matmul epilogue/prologue fusion (Phase 1)
      if (node.op === "matmul" && backend.name === "webgpu") {
        let epiloguePlan = prebuiltConsumerCount
          ? detectMatmulEpilogueCore(nodes, nodeIdx, prebuiltConsumerCount, externalNodeIds)
          : detectMatmulEpilogue(nodes, nodeIdx, allPlanNodes ?? nodes, externalNodeIds);
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
          const epilogueLabel = epiloguePlan.epilogueOps.length > 0
            ? "+" + epiloguePlan.epilogueOps.map(o => o.kind).join("+")
            : "";
          const epLabel = `matmul+${prologueLabel}${epilogueLabel}`.replace(/\+$/, "");
          setCurrentOpLabel(epLabel);
          setProfileModule(node.module ?? "unknown");
          const _profT0 = profileOpBegin(epLabel);
          try {
            await executeMatmulWithEpilogue(node, epiloguePlan, backend);
          } finally {
            profileOpEnd(epLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }

          // Track storages for all consumed nodes and release dead buffers
          if (enableEarlyRelease) {
            for (let skip = 0; skip < epiloguePlan.consumedCount; skip++) {
              const consumedNode = nodes[nodeIdx + skip];
              if (consumedNode.result) {
                nodeToStorage.set(consumedNode.id, consumedNode.result);
              }
              step++;
              if (lifetimes && outputNodeIds) {
                const deadNodeIds = findDeadTensorsAtStep(
                  lifetimes,
                  step,
                  outputNodeIds,
                  alreadyReleased,
                );
                for (const deadId of deadNodeIds) {
                  const storage = nodeToStorage.get(deadId);
                  if (storage && canSafelyRelease(storage, nodeToStorage)) {
                    releaseBufferImmediate(storage);
                    nodeToStorage.delete(deadId);
                    alreadyReleased.add(deadId);
                  }
                }
              }
            }
          } else {
            step += epiloguePlan.consumedCount;
          }

          // Record matmul epilogue action in lowered plan builder
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            const covered: number[] = [];
            for (let skip = 0; skip < epiloguePlan.consumedCount; skip++) {
              covered.push(nodeIdToFinalPos.get(nodes[nodeIdx + skip].id)!);
            }
            loweredPlanBuilder.recordMatmulEpilogue(
              nodeIdToFinalPos.get(node.id)!,
              covered,
              nodeIdToFinalPos.get(epiloguePlan.outputNode.id)!,
              epiloguePlan.epilogueOps,
              epiloguePlan.epilogueInputRefs.length,
              epiloguePlan.outputDtype,
              epiloguePlan.consumedCount,
              epiloguePlan.prologues?.map(p => ({
                inputIndex: p.inputIndex,
                castNodeIndex: nodeIdToFinalPos.get(p.castNodeId)!,
                fromDtype: p.fromDtype,
                toDtype: p.toDtype,
              })),
            );
          }

          nodeIdx += epiloguePlan.consumedCount - 1;
          continue;
        }
      }

      // Try reduction preamble fusion (Phase 3)
      if (isFusibleOp(node.op) && backend.name === "webgpu") {
        const reductionPlan = detectReductionPreamble(nodes, nodeIdx, reductionConsumerCount);
        if (reductionPlan) {
          const rpLabel = `${reductionPlan.isMean ? "mean" : "sum"}+${reductionPlan.op}`;
          setCurrentOpLabel(rpLabel);
          setProfileModule(node.module ?? "unknown");
          const _profT0 = profileOpBegin(rpLabel);
          try {
            await executeReductionWithPreamble(reductionPlan, backend);
          } finally {
            profileOpEnd(rpLabel, _profT0);
            setCurrentOpLabel(null);
            setProfileModule("unknown");
          }

          // Track storages for both consumed nodes (preamble + reduction)
          if (enableEarlyRelease) {
            // Preamble node: no result (consumed), but step still advances
            step++;
            if (lifetimes && outputNodeIds) {
              const deadNodeIds = findDeadTensorsAtStep(lifetimes, step, outputNodeIds, alreadyReleased);
              for (const deadId of deadNodeIds) {
                const storage = nodeToStorage.get(deadId);
                if (storage && canSafelyRelease(storage, nodeToStorage)) {
                  releaseBufferImmediate(storage);
                  nodeToStorage.delete(deadId);
                  alreadyReleased.add(deadId);
                }
              }
            }
            // Reduction node: has the result
            const reductionNode = nodes[nodeIdx + 1];
            if (reductionNode.result) {
              nodeToStorage.set(reductionNode.id, reductionNode.result);
            }
            step++;
            if (lifetimes && outputNodeIds) {
              const deadNodeIds = findDeadTensorsAtStep(lifetimes, step, outputNodeIds, alreadyReleased);
              for (const deadId of deadNodeIds) {
                const storage = nodeToStorage.get(deadId);
                if (storage && canSafelyRelease(storage, nodeToStorage)) {
                  releaseBufferImmediate(storage);
                  nodeToStorage.delete(deadId);
                  alreadyReleased.add(deadId);
                }
              }
            }
          } else {
            step += 2;
          }

          // Record reduction preamble action in lowered plan builder
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            loweredPlanBuilder.recordReductionPreamble(
              nodeIdToFinalPos.get(node.id)!,
              nodeIdToFinalPos.get(nodes[nodeIdx + 1].id)!,
            );
          }

          nodeIdx += 1; // Skip the reduction node (consumed 2 nodes total)
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
                if (enableEarlyRelease) nodeToStorage.set(adamNode.id, adamNode.result);
                step++;
                continue;
              }

              const adamBackend = getBackend(adamNode.device) ?? backend;
              const adamInputs = adamNode.inputs.map(ref => getInputStorage(ref, adamBackend));
              const adamBackendInputs = adamInputs.map((s) => s.backendTensor);

              setCurrentOpLabel("adamStep");
              setProfileModule(adamNode.module ?? "unknown");
              const _profT0 = profileOpBegin("adamStep");
              let adamResult: BackendTensor;
              try {
                adamResult = await executeOp(adamNode, adamBackendInputs, adamBackend);
              } finally {
                profileOpEnd("adamStep", _profT0);
                setCurrentOpLabel(null);
                setProfileModule("unknown");
              }

              const adamAliasedIdx = adamBackendInputs.findIndex(b => b === adamResult);
              if (adamAliasedIdx >= 0 && (adamResult as { ownsBuffer?: boolean }).ownsBuffer === true) {
                adamResult = { ...adamResult, ownsBuffer: false } as BackendTensor;
              }
              const adamIsView = (adamResult as { ownsBuffer?: boolean }).ownsBuffer === false;
              const adamBaseId = adamIsView && adamInputs.length > 0
                ? adamInputs[adamAliasedIdx >= 0 ? adamAliasedIdx : 0].id
                : undefined;
              adamNode.result = createStorageHandle(adamNode.device, adamResult, adamBaseId);

              if (enableEarlyRelease) {
                nodeToStorage.set(adamNode.id, adamNode.result);
                step++;
                if (lifetimes && outputNodeIds) {
                  const deadNodeIds = findDeadTensorsAtStep(lifetimes, step, outputNodeIds, alreadyReleased);
                  for (const deadId of deadNodeIds) {
                    const storage = nodeToStorage.get(deadId);
                    if (storage && canSafelyRelease(storage, nodeToStorage)) {
                      releaseBufferImmediate(storage);
                      nodeToStorage.delete(deadId);
                      alreadyReleased.add(deadId);
                    }
                  }
                }
              } else {
                step++;
              }
            }
          } finally {
            setAdamBatchMode(false);
          }

          // Record adam batch action in lowered plan builder
          if (loweredPlanBuilder && nodeIdToFinalPos) {
            const adamIndices: number[] = [];
            for (let a = 0; a < adamCount; a++) {
              adamIndices.push(nodeIdToFinalPos.get(nodes[nodeIdx + a].id)!);
            }
            loweredPlanBuilder.recordAdamBatch(adamIndices);
          }

          nodeIdx += adamCount - 1;
          continue;
        }
      }

      const nodeBackend = getBackend(node.device) ?? backend;
      const inputs = node.inputs.map(ref => getInputStorage(ref, nodeBackend));
      const backendInputs = inputs.map((s) => s.backendTensor);

      let resultTensor = await executeOp(node, backendInputs, nodeBackend);
      const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
      if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
        resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
      }
      const isView =
        (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
      const baseStorageId =
        isView && inputs.length > 0
          ? inputs[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
          : undefined;
      node.result = createStorageHandle(node.device, resultTensor, baseStorageId);

      // Record action in lowered plan builder
      if (loweredPlanBuilder && nodeIdToFinalPos) {
        const finalPos = nodeIdToFinalPos.get(node.id)!;
        if (isDataSourceOp(node.op)) {
          loweredPlanBuilder.recordDataSource(finalPos);
        } else if (isViewOp(node.op)) {
          loweredPlanBuilder.recordView(finalPos);
        } else {
          loweredPlanBuilder.recordSequential(finalPos);
        }
      }

      // Track storage and release dead buffers
      if (enableEarlyRelease) {
        nodeToStorage.set(node.id, node.result);
        step++;

        if (lifetimes && outputNodeIds) {
          const deadNodeIds = findDeadTensorsAtStep(
            lifetimes,
            step,
            outputNodeIds,
            alreadyReleased,
          );
          for (const deadId of deadNodeIds) {
            const storage = nodeToStorage.get(deadId);
            if (storage && canSafelyRelease(storage, nodeToStorage)) {
              releaseBufferImmediate(storage);
              nodeToStorage.delete(deadId);
              alreadyReleased.add(deadId);
            }
          }
        }
      } else {
        step++;
      }

      // Periodic intra-segment reclamation
      nodesSinceIntraReclaim++;
      if (useSharedEncoder && enableEarlyRelease && nodesSinceIntraReclaim >= INTRA_SEGMENT_RECLAIM_INTERVAL) {
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
