import { getBackend } from "../backend/registry";
import type {
  Backend,
  BackendTensor,
  DeviceKind,
  DType,
} from "../backend/types";
import { isFusedBackend } from "../backend/types";
import {
  asGPUTensor,
  type GPUBuffer,
  type GPUDevice,
} from "../backend/webgpu/gpu-types";
import { contiguousStrides, sizeOf } from "../core/shape";
import { executeNode } from "./executor-sequential";
import type { FusionGroup, groupToRecipe } from "./fusion-detect";
import type { LazyIRNode, StorageHandle } from "./lazy-types";
import {
  _webgpuMatmulImports,
  createStorageHandle,
  ensureWebGPUMatmulImports,
} from "./node-factory";
import { getInputStorage } from "./op-dispatch";
import {
  recordFusionFallback,
  setCurrentOpLabel,
  setProfileModule,
} from "./profiler";
import type { ReductionGroup } from "./reduction-detect";

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
  } as unknown as BackendTensor);
}

// Module-level cached imports to avoid per-call dynamic import overhead.
// After first call, these are resolved and reused synchronously.
let _cachedDispatchFusedKernel:
  | typeof import("../backend/webgpu/fusion-dispatch").dispatchFusedKernel
  | null = null;
let _cachedDeferredDestroyBuffer:
  | NonNullable<typeof _webgpuMatmulImports>["deferredDestroyBuffer"]
  | null = null;

/** Ensure fusion dispatch imports are resolved. Call once; subsequent calls are no-ops. */
export async function ensureFusionImports(): Promise<void> {
  if (_cachedDispatchFusedKernel) return;
  const fusionDispatch = await import("../backend/webgpu/fusion-dispatch");
  _cachedDispatchFusedKernel = fusionDispatch.dispatchFusedKernel;
  await ensureWebGPUMatmulImports();
  _cachedDeferredDestroyBuffer = (
    _webgpuMatmulImports as NonNullable<typeof _webgpuMatmulImports>
  ).deferredDestroyBuffer;
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
  // For CPU or other backends without fusion support, fall back to sequential execution
  if (!isFusedBackend(backend) || !("dispatchFusedKernel" in backend)) {
    await executeSequentialSegment(group.nodes, backend);
    return;
  }
  // Ensure imports are resolved (no-op after first call)
  await ensureFusionImports();
  const dispatchFusedKernel = _cachedDispatchFusedKernel!;
  const deferredDestroyBuffer = _cachedDeferredDestroyBuffer!;

  /** Wrap a fusion output buffer into a StorageHandle with deferred destroy. */
  const wrapFusionOutput = (
    device: string,
    output: { buffer: unknown; shape: number[]; dtype: DType },
  ): StorageHandle => {
    const buf = output.buffer as GPUBuffer;
    const bufSize = buf.size;
    let destroyed = false;
    return createStorageHandle(
      device as DeviceKind,
      {
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
      } as unknown as BackendTensor,
    );
  };

  // Get WebGPU device from backend (narrowed by the webgpu check above)
  const device = (
    backend as Backend & {
      device?: GPUDevice;
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
    let storage: StorageHandle | undefined;
    if (inputRef.kind === "materialized") {
      storage = inputRef.storage;
    } else {
      const idx = inputRef.outputIndex ?? 0;
      storage = idx === 0 ? inputRef.node.result : inputRef.node.results?.[idx];
    }

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
  const fused = isFusedBackend(backend) ? backend : null;
  if (fused) fused.beginSharedEncoder();

  try {
    for (const node of nodes) {
      if (node.result) continue;
      await executeNode(node, backend);
    }
  } finally {
    if (fused) fused.endSharedEncoder();
  }
}

// ============================================================================
// Reduction Segment Execution
// ============================================================================

/** Reduction payload shape shared across sum/mean/max. */
type ReductionPayload = { dim?: number | number[] | null; keepdim?: boolean };

/**
 * Execute a reduction segment: preamble → reduction → epilogue.
 *
 * Routes to the appropriate backend dispatch based on which parts are present:
 * - Preamble + epilogue → sumWithPreambleEpilogue
 * - Preamble only → sumDimWithPreambleChain
 * - Epilogue only → reduction / meanWithEpilogue
 */
export async function executeReductionSegment(
  group: ReductionGroup,
  backend: Backend,
): Promise<void> {
  // Fused reduction kernels are WebGPU-only; fall back to sequential on CPU
  if (group.reductionNode.device === "cpu") {
    for (const node of group.nodes) {
      if (!node.result) await executeNode(node, backend);
    }
    return;
  }

  const payload = group.reductionNode.payload as ReductionPayload | undefined;
  const hasPreamble = group.preambleNodes.length > 0;
  const hasEpilogue = group.epilogueOps.length > 0;

  if (hasPreamble && hasEpilogue) {
    // Combined preamble + epilogue
    const { sumWithPreambleEpilogue } = await import("../backend/webgpu/index");
    const preambleInputTensors = group.preambleInputRefs.map(
      (ref) => getInputStorage(ref, backend).backendTensor,
    );
    const epilogueInputTensors = group.epilogueInputRefs.map(
      (ref) => getInputStorage(ref, backend).backendTensor,
    );

    const resultTensor = sumWithPreambleEpilogue(
      preambleInputTensors,
      group.preambleOps,
      group.preambleInputDtypes,
      group.epilogueOps,
      epilogueInputTensors,
      group.outputDtype,
      payload ?? {},
      group.isMean,
    );

    group.outputNode.result = createStorageHandle(
      group.outputNode.device,
      resultTensor,
    );
  } else if (hasPreamble) {
    // Preamble only
    const { sumDimWithPreambleChain } = await import("../backend/webgpu/index");
    const inputTensors = group.preambleInputRefs.map(
      (ref) => getInputStorage(ref, backend).backendTensor,
    );

    let resultTensor = sumDimWithPreambleChain(
      inputTensors,
      group.preambleOps,
      group.preambleInputDtypes,
      payload ?? {},
    );

    // If this is a mean, divide by reduction size
    if (group.isMean) {
      const { normalizeDim } = await import("../backend/types");
      const inputShape = group.preambleNodes[0].shape;
      const dim = payload?.dim;
      let reductionSize: number;
      if (dim == null) {
        reductionSize = inputShape.reduce((a, b) => a * b, 1);
      } else {
        const dims = Array.isArray(dim) ? dim : [dim];
        const rank = inputShape.length;
        reductionSize = dims.reduce(
          (acc, d) => acc * inputShape[normalizeDim(d, rank)],
          1,
        );
      }
      const invSize = 1.0 / reductionSize;
      const sumResult = resultTensor;
      const invSizeTensor = backend.ops.full
        ? backend.ops.full([], invSize)
        : backend.ops.tensorFromArray([invSize], []);
      resultTensor = backend.ops.mul(sumResult, invSizeTensor);
      (sumResult as { destroy?: () => void }).destroy?.();
      (invSizeTensor as { destroy?: () => void }).destroy?.();
    }

    group.outputNode.result = createStorageHandle(
      group.outputNode.device,
      resultTensor,
    );
  } else if (hasEpilogue) {
    // Epilogue only
    const { reduction, meanWithEpilogue } = await import(
      "../backend/webgpu/index"
    );
    const reductionInputStorage = getInputStorage(
      group.reductionNode.inputs[0],
      backend,
    );
    const reductionInputTensor = reductionInputStorage.backendTensor;
    const epilogueInputTensors = group.epilogueInputRefs.map(
      (ref) => getInputStorage(ref, backend).backendTensor,
    );

    const resultTensor =
      group.reductionNode.op === "mean"
        ? meanWithEpilogue(
            reductionInputTensor,
            payload ?? {},
            group.epilogueOps,
            epilogueInputTensors,
            group.outputDtype,
          )
        : reduction(
            group.reductionNode.op as "sum" | "max",
            reductionInputTensor,
            payload ?? {},
            group.epilogueOps,
            epilogueInputTensors,
            group.outputDtype,
          );

    group.outputNode.result = createStorageHandle(
      group.outputNode.device,
      resultTensor,
    );
  }
}
