import type { BackendTensor, DeviceKind, DType } from "../backend/types";
import { getProfileModule } from "../backend/webgpu/profiler";
import type { LazyOpCode, LazyIRNode, LazyRef, StorageHandle } from "./lazy-types";
import { storageTracker } from "./storage-tracker";

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

let nextStorageId = 1;

export function resetStorageIdCounter(): void {
  nextStorageId = 1;
}

export function getNextStorageId(): number {
  return nextStorageId;
}

export function createStorageHandle(
  device: DeviceKind,
  backendTensor: BackendTensor,
  baseStorageId?: number,
): StorageHandle {
  const storage: StorageHandle = {
    id: nextStorageId++,
    device,
    backendTensor,
    baseStorageId,
  };
  // Register in the global tracker
  storageTracker.register(storage);
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
  const aliasedInputIdx = backendInputs.findIndex(b => b === resultTensor);
  if (aliasedInputIdx >= 0 && (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === true) {
    resultTensor = { ...resultTensor, ownsBuffer: false } as BackendTensor;
  }
  const isView = (resultTensor as { ownsBuffer?: boolean }).ownsBuffer === false;
  const baseStorageId = isView && inputStorages.length > 0
    ? inputStorages[aliasedInputIdx >= 0 ? aliasedInputIdx : 0].id
    : undefined;
  return createStorageHandle(device, resultTensor, baseStorageId);
}

// ============================================================================
// Lazy-initialized WebGPU imports (avoids circular deps; loaded once on first use)
// ============================================================================

export let _webgpuMatmulImports: {
  dispatchMatmulWithEpilogue: typeof import("../backend/webgpu/index").dispatchMatmulWithEpilogue;
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
      dispatchMatmulWithEpilogue: mod.dispatchMatmulWithEpilogue,
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
