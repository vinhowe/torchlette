import type { BackendTensor, DeviceKind, DType } from "../backend/types";
import { getProfileModule } from "./profiler";
import { rcRelease, rcRetain } from "./refcount";
import { storageTracker } from "./storage-tracker";
import type { LazyIRNode, LazyOpCode, LazyRef, StorageHandle } from "./types";

// ============================================================================
// Node and Storage ID Counters
// ============================================================================

let nextNodeId = 1;

export function resetNodeIdCounter(): void {
  nextNodeId = 1;
}

export function createLazyIRNode(
  op: LazyOpCode,
  inputs: LazyRef[],
  shape: number[],
  dtype: DType,
  device: DeviceKind,
  payload?: unknown,
): LazyIRNode {
  const node: LazyIRNode = {
    id: nextNodeId++,
    op,
    inputs,
    shape,
    dtype,
    device,
    payload,
  };
  // Capture module context for profiling (zero-cost when profiling disabled)
  const mod = getProfileModule();
  if (mod !== "unknown") node.module = mod;
  return node;
}

/**
 * Retain rc on all materialized inputs of the nodes in a plan.
 * Keeps those storages alive through plan execution, even if their owning
 * tensors are disposed mid-step. Pair with releasePlanInputRefs.
 */
export function retainPlanInputRefs(nodes: readonly LazyIRNode[]): void {
  for (const node of nodes) {
    if (node._inputsRetained) continue;
    node._inputsRetained = true;
    for (const input of node.inputs) {
      if (input.kind === "materialized") {
        rcRetain(input.storage.id, "plan.input");
      }
    }
  }
}

/** Release retained refs on a node's materialized inputs (idempotent). */
export function releaseNodeInputRefs(node: LazyIRNode): void {
  if (!node._inputsRetained) return;
  node._inputsRetained = false;
  for (const input of node.inputs) {
    if (input.kind === "materialized") {
      rcRelease(input.storage.id, "plan.inputConsumed");
    }
  }
}

let nextStorageId = 1;

export function resetStorageIdCounter(): void {
  nextStorageId = 1;
}

export function getNextStorageId(): number {
  return nextStorageId;
}

/**
 * Create a lightweight BackendTensor for GPU dispatch results.
 * Centralizes the type cast for object literals with GPU-specific fields
 * (buffer, size, strides, offset, isContiguous) that satisfy BackendTensor
 * at runtime but don't match the interface structurally.
 */
export function createGPUBackendTensor(fields: {
  buffer: unknown;
  shape: number[];
  dtype: DType;
  size: number;
  strides: number[];
  offset: number;
  isContiguous: boolean;
  ownsBuffer: boolean;
  destroy: () => void;
}): BackendTensor {
  return fields as BackendTensor;
}

export function createStorageHandle(
  device: DeviceKind,
  backendTensor: BackendTensor,
): StorageHandle {
  const storage: StorageHandle = {
    id: nextStorageId++,
    device,
    backendTensor,
  };
  // Register in the global tracker
  storageTracker.register(storage);
  // Note: NO rcRetain here. rc starts at 0 (no owner yet).
  // Ownership is claimed by tensor._materialize/constructor (retain)
  // and by view.baseStorageId (retain on base).
  return storage;
}

/**
 * Wrap a backend op result as a StorageHandle, detecting aliased returns.
 *
 * Many backend ops (e.g. contiguous on a contiguous tensor) may return the
 * exact same tensor object as one of their inputs. Creating a separate owning
 * StorageHandle would double-free the underlying buffer. This helper detects
 * the alias, marks the result as a non-owning view, and sets the correct
 * baseStorageId.
 */
export function wrapResultAsStorage(
  device: DeviceKind,
  resultTensor: BackendTensor,
  backendInputs: BackendTensor[],
  inputStorages: StorageHandle[],
): StorageHandle {
  const aliasedInputIdx = backendInputs.indexOf(resultTensor);
  if (aliasedInputIdx >= 0 && resultTensor.ownsBuffer === true) {
    resultTensor = { ...resultTensor, ownsBuffer: false };
  }
  const storage = createStorageHandle(device, resultTensor);
  // Set baseStorageId when the result shares a buffer with an input.
  // Two checks: (1) same object (indexOf match), or (2) same underlying
  // buffer (for contiguous() which returns a new object wrapping the same buffer).
  // Do NOT chain results that have ownsBuffer=false but use an independent
  // buffer (e.g., f16 weight cache hits) — those would create fake view
  // chains that leak storages.
  if (inputStorages.length > 0) {
    let baseIdx = aliasedInputIdx;
    if (baseIdx < 0 && resultTensor.ownsBuffer === false) {
      // Check buffer-level aliasing for non-object-aliased views
      const resultBuf = (resultTensor as { buffer?: unknown }).buffer;
      if (resultBuf) {
        for (let i = 0; i < backendInputs.length; i++) {
          if ((backendInputs[i] as { buffer?: unknown }).buffer === resultBuf) {
            baseIdx = i;
            break;
          }
        }
      }
    }
    if (baseIdx >= 0) {
      storage.baseStorageId = inputStorages[baseIdx].id;
      rcRetain(inputStorages[baseIdx].id, "view.baseStorageId");
    }
  }
  return storage;
}

// ============================================================================
// Lazy-initialized WebGPU imports (avoids circular deps; loaded once on first use)
// ============================================================================

export let _webgpuMatmulImports: {
  dispatchMatmul: typeof import("../backend/webgpu/index").dispatchMatmul;
  dispatchMatmulDirect: typeof import("../backend/webgpu/index").dispatchMatmulDirect;
  deferredDestroyBuffer: typeof import("../backend/webgpu/index").deferredDestroyBuffer;
} | null = null;

export let _webgpuMatmulGeomImports: {
  computeMatmulOutputShape: typeof import("../backend/webgpu/matmul/dispatch").computeMatmulOutputShape;
  computeBatchSize: typeof import("../backend/webgpu/matmul/dispatch").computeBatchSize;
  computeBatchStrides: typeof import("../backend/webgpu/matmul/dispatch").computeBatchStrides;
} | null = null;

export async function ensureWebGPUMatmulImports() {
  if (!_webgpuMatmulImports) {
    const mod = await import("../backend/webgpu/index");
    _webgpuMatmulImports = {
      dispatchMatmul: mod.dispatchMatmul,
      dispatchMatmulDirect: mod.dispatchMatmulDirect,
      deferredDestroyBuffer: mod.deferredDestroyBuffer,
    };
  }
  if (!_webgpuMatmulGeomImports) {
    const mod = await import("../backend/webgpu/matmul/dispatch");
    _webgpuMatmulGeomImports = {
      computeMatmulOutputShape: mod.computeMatmulOutputShape,
      computeBatchSize: mod.computeBatchSize,
      computeBatchStrides: mod.computeBatchStrides,
    };
  }
}
