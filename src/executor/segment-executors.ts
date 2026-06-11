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
import type { FusionGroup, groupToRecipe } from "../compiler/fusion-detect";
import type { RowProgram } from "../compiler/row-program-types";
import { contiguousStrides, sizeOf } from "../core/shape";
import {
  _webgpuMatmulImports,
  createStorageHandle,
  ensureWebGPUMatmulImports,
} from "../graph/node-factory";
import {
  recordFusionFallback,
  setCurrentOpLabel,
  setProfileModule,
} from "../graph/profiler";
import {
  arenaBufferSet,
  pinnedBufferSet,
} from "../backend/webgpu/webgpu-state";
import { rcRetain } from "../graph/refcount";
import type { LazyIRNode, LazyRef, StorageHandle } from "../graph/types";
import { getInputStorage } from "./op-dispatch";
import { executeNode } from "./sequential";

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
  donatableInputIds?: Set<number>,
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
        toArray() {
          return [];
        },
        destroy() {
          if (destroyed) return;
          destroyed = true;
          deferredDestroyBuffer(buf, bufSize);
        },
      } as BackendTensor,
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
  // Parallel to `inputs` (non-inlined order): the source StorageHandle for
  // donation eligibility + base-chaining; null for scalars and contiguous
  // copies (not donatable). Also the recipe-input index per entry.
  const inputStorages: Array<StorageHandle | null> = [];
  const inputRecipeIdx: number[] = [];
  const tempContiguousCopies: Array<{ destroy?: () => void }> = [];
  for (let inputIdx = 0; inputIdx < group.externalInputs.length; inputIdx++) {
    // Skip inlined constants — their values are baked into the shader
    // This handles both scalar LazyRefs and pending nodes detected as inlinable
    if (recipe.inputs[inputIdx]?.isInlinedConstant) {
      continue;
    }

    const inputRef = group.externalInputs[inputIdx];
    // Non-inlined scalar ref: demoted to a runtime input by the
    // frozen-scalar adaptation. Resolve through getInputStorage — it hits
    // the plan's scalar table (persistent 0-d buffer refreshed from the
    // CURRENT step's value), so the fused kernel reads fresh data while the
    // cached recipe/kernel stay value-independent.
    if (inputRef.kind === "scalar") {
      const scalarStorage = getInputStorage(inputRef, backend);
      const st = asGPUTensor(scalarStorage.backendTensor);
      inputs.push({
        buffer: st.buffer,
        shape: [],
        dtype: (st.dtype as DType) ?? "f32",
      });
      inputStorages.push(null);
      inputRecipeIdx.push(inputIdx);
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
        inputStorages.push(null);
        inputRecipeIdx.push(inputIdx);
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
    inputStorages.push(storage);
    inputRecipeIdx.push(inputIdx);
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

  // BUFFER DONATION: pick a dying input whose buffer the primary output can
  // overwrite in place. Eligibility (every condition is load-bearing):
  //  - liveness says this segment is the input node's LAST reader and the
  //    node is not a plan output / externally referenced (donatableInputIds);
  //  - single-output recipe, no externally-needed intermediates (those
  //    re-execute group nodes AFTER the dispatch and would read the
  //    clobbered input);
  //  - same element count + dtype as out0, non-scalar, non-broadcast;
  //  - buffer not arena-owned or plan-pinned (their identity is managed
  //    elsewhere; donating transfers WRITE ownership to the output);
  //  - buffer not bound twice in this kernel (read + rw of one buffer in a
  //    single dispatch is a WebGPU validation error).
  let donatedRecipeIdx: number | undefined;
  let donatedStorage: StorageHandle | null = null;
  if (
    donatableInputIds &&
    donatableInputIds.size > 0 &&
    process.env.TORCHLETTE_DONATION !== "0" &&
    // Multi-output groups donate into out0 only; additional outputs keep
    // their own allocations (the earlier corruption was a dispatch bug:
    // EVERY output was handed the donated buffer — one buffer at multiple
    // writable binding indices is a validation error that drops the whole
    // submit; pinned by tools/test-donation-multiout.ts). Hard requirement:
    // NO externally-needed intermediates — those re-execute group nodes
    // after the dispatch and would read the clobbered donated input.
    (!group.neededIntermediates || group.neededIntermediates.length === 0)
  ) {
    const out0 = recipe.outputs[0];
    const outElems = sizeOf(out0.shape);
    for (let pos = 0; pos < inputs.length; pos++) {
      const storage = inputStorages[pos];
      if (!storage) continue;
      const rIdx = inputRecipeIdx[pos];
      const rin = recipe.inputs[rIdx];
      if (!rin || rin.isInlinedConstant || rin.isScalar) continue;
      const ref = group.externalInputs[rIdx];
      if (!ref || ref.kind !== "pending") continue;
      if (!donatableInputIds.has(ref.node.id)) continue;
      if (sizeOf(rin.shape) !== outElems) continue;
      if (rin.dtype !== out0.dtype) continue;
      const buf = inputs[pos].buffer;
      if (arenaBufferSet.has(buf) || pinnedBufferSet.has(buf)) continue;
      // No duplicate binding of the same buffer in this dispatch
      let dup = false;
      for (let q = 0; q < inputs.length; q++) {
        if (q !== pos && inputs[q].buffer === buf) {
          dup = true;
          break;
        }
      }
      if (dup) continue;
      donatedRecipeIdx = rIdx;
      donatedStorage = storage;
      break;
    }
  }
  if (process.env.TORCHLETTE_DEBUG_DONATION) {
    const outElems = sizeOf(recipe.outputs[0].shape);
    if (outElems > 1_000_000) {
      console.log(
        `[donation] group out=${recipe.outputs[0].shape.join("x")} outs=${recipe.outputs.length} needed=${group.neededIntermediates?.length ?? 0} addl=${group.additionalOutputNodes?.length ?? 0} donatable=${donatableInputIds ? [...donatableInputIds].length : "none"} -> ${donatedRecipeIdx !== undefined ? "DONATED in" + donatedRecipeIdx : "no"}`,
      );
    }
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
      donatedInput: donatedRecipeIdx,
    });

    // Store results on output nodes
    if (donatedRecipeIdx !== undefined && donatedStorage) {
      // In-place result: a NON-OWNING view base-chained to the donated
      // input's storage (mirrors wrapResultAsStorage) — the input storage
      // remains the buffer's owner and survives (rcRetain) until the output
      // dies; liveness release of the input is deferred by the same retain.
      const out0 = result.outputs[0];
      const sh = createStorageHandle(group.outputNode.device as DeviceKind, {
        buffer: out0.buffer,
        shape: out0.shape,
        dtype: out0.dtype,
        size: sizeOf(out0.shape),
        strides: contiguousStrides(out0.shape),
        offset: 0,
        isContiguous: true,
        ownsBuffer: false,
        toArray() {
          return [];
        },
      } as BackendTensor);
      sh.baseStorageId = donatedStorage.id;
      rcRetain(donatedStorage.id, "view.baseStorageId");
      group.outputNode.result = sh;
    } else {
      group.outputNode.result = wrapFusionOutput(
        group.outputNode.device,
        result,
      );
    }
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
// Row-Program Execution
// ============================================================================

/**
 * Execute a row-program fusion: multi-reduction subgraph → single perRowKernel.
 * Falls back to sequential execution on CPU or if dispatch fails.
 */
export async function executeRowProgram(
  program: RowProgram,
  inputRefs: LazyRef[],
  outputNode: LazyIRNode,
  numRows: number,
  dimSize: number,
  coveredNodes: LazyIRNode[],
  backend: Backend,
): Promise<void> {
  // CPU fallback
  if (outputNode.device === "cpu") {
    for (const node of coveredNodes) {
      if (!node.result) await executeNode(node, backend);
    }
    return;
  }

  try {
    const { dispatchRowProgram } = await import(
      "../backend/webgpu/row-program-dispatch"
    );

    // Force-execute any unresolved pending inputs (and their transitive
    // dependencies) before dispatch. The lowered plan may place the
    // row-program action before data-source or intermediate nodes that
    // feed into it (e.g., full → triu for attention masks).
    const forceWithDeps = async (node: LazyIRNode): Promise<void> => {
      if (node.result) return;
      for (const inp of node.inputs) {
        if (inp.kind === "pending" && !inp.node.result) {
          await forceWithDeps(inp.node);
        }
      }
      await executeNode(node, backend);
    };
    for (const ref of inputRefs) {
      if (ref.kind === "pending" && !ref.node.result) {
        await forceWithDeps(ref.node);
      }
    }

    // Resolve input buffers from inputRefs
    const inputBuffers: GPUBuffer[] = [];
    for (const ref of inputRefs) {
      const storage = getInputStorage(ref, backend);
      const tensor = asGPUTensor(storage.backendTensor);
      inputBuffers.push(tensor.buffer);
    }

    setCurrentOpLabel("row-program");
    const shape = outputNode.shape;
    // Pass the consumer's element count (sizeOf(outputNode.shape)) as the single
    // source of truth for the output layout — dispatchRowProgram sizes the buffer
    // from it and asserts the kernel's write count matches (see SEAM INVARIANT).
    const outBuffer = dispatchRowProgram(
      program,
      inputBuffers,
      numRows,
      dimSize,
      sizeOf(shape),
    );

    outputNode.result = createStorageHandle(outputNode.device, {
      buffer: outBuffer,
      shape: shape.slice(),
      dtype: program.output.dtype,
      size: sizeOf(shape),
      strides: contiguousStrides(shape),
      offset: 0,
      isContiguous: true,
      ownsBuffer: true,
      toArray: () => [],
    } as BackendTensor);
  } catch (e) {
    console.warn("Row-program dispatch failed, falling back to sequential:", e);
    for (const node of coveredNodes) {
      if (!node.result) await executeNode(node, backend);
    }
  } finally {
    setCurrentOpLabel(null);
  }
}
