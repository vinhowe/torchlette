/**
 * Cross-Device Execution (§13)
 *
 * Implements:
 * - Device transfer operations (lazy)
 * - Multi-device graph execution
 * - Efficient transfer paths between backends
 * - Auto-transfer insertion for cross-device ops
 */

import type { BackendTensor, DeviceKind } from "../backend/types";
import { getBackend } from "../backend/registry";
import type { LazyIRNode, LazyRef, StorageHandle } from "./lazy";
import { createStorageHandle } from "./lazy";

// ============================================================================
// Transfer Path Types
// ============================================================================

/**
 * Describes how to transfer data between two devices.
 */
export interface TransferPath {
  sourceDevice: DeviceKind;
  targetDevice: DeviceKind;
  // Transfer method: "direct" for same-type devices, "via_cpu" for different types
  method: "direct" | "via_cpu" | "noop";
}

/**
 * Transfer execution result.
 */
export interface TransferResult {
  storage: StorageHandle;
  // Statistics for performance monitoring
  stats: {
    bytesTransferred: number;
    path: TransferPath;
  };
}

// ============================================================================
// Transfer Path Resolution
// ============================================================================

/**
 * Determine the optimal transfer path between two devices.
 */
export function resolveTransferPath(
  sourceDevice: DeviceKind,
  targetDevice: DeviceKind,
): TransferPath {
  if (sourceDevice === targetDevice) {
    return { sourceDevice, targetDevice, method: "noop" };
  }

  // CPU <-> WebGPU: must go via CPU (read to host, then write to device)
  // In the future, could support direct GPU copies for same-GPU-type transfers
  if (sourceDevice === "cpu" || targetDevice === "cpu") {
    return { sourceDevice, targetDevice, method: "via_cpu" };
  }

  // WebGPU <-> WebGPU on same adapter could be "direct" in future
  // For now, all cross-device transfers go via CPU
  return { sourceDevice, targetDevice, method: "via_cpu" };
}

/**
 * Check if a transfer is needed between two devices.
 */
export function needsTransfer(
  sourceDevice: DeviceKind,
  targetDevice: DeviceKind,
): boolean {
  return sourceDevice !== targetDevice;
}

// ============================================================================
// Transfer Execution
// ============================================================================

/**
 * Execute a transfer between devices.
 * This is the low-level implementation called during plan execution.
 */
export async function executeTransfer(
  sourceTensor: BackendTensor,
  sourceDevice: DeviceKind,
  targetDevice: DeviceKind,
): Promise<TransferResult> {
  const path = resolveTransferPath(sourceDevice, targetDevice);

  if (path.method === "noop") {
    // No transfer needed, return as-is
    return {
      storage: createStorageHandle(targetDevice, sourceTensor),
      stats: {
        bytesTransferred: 0,
        path,
      },
    };
  }

  // Get backends
  const sourceBackend = getBackend(sourceDevice);
  const targetBackend = getBackend(targetDevice);

  if (!sourceBackend || !targetBackend) {
    throw new Error(
      `Transfer failed: backend not available for ${sourceDevice} or ${targetDevice}`,
    );
  }

  // For now, all transfers go via CPU (read values, create on target)
  const values = await sourceBackend.ops.read(sourceTensor);
  const shape = sourceTensor.shape;
  const targetTensor = targetBackend.ops.tensorFromArray(values, shape);

  const bytesTransferred = values.length * 4; // Assume f32

  return {
    storage: createStorageHandle(targetDevice, targetTensor),
    stats: {
      bytesTransferred,
      path,
    },
  };
}

// ============================================================================
// Multi-Device Plan Analysis
// ============================================================================

/**
 * Analyze a plan for cross-device operations.
 */
export interface CrossDeviceAnalysis {
  // All devices used in the plan
  devices: Set<DeviceKind>;
  // Nodes that require cross-device transfers
  transferPoints: Array<{
    nodeId: number;
    inputIndex: number;
    sourceDevice: DeviceKind;
    targetDevice: DeviceKind;
  }>;
  // Whether the plan is single-device (no transfers needed)
  isSingleDevice: boolean;
}

/**
 * Analyze a lazy IR graph for cross-device operations.
 */
export function analyzeCrossDeviceOps(
  nodes: LazyIRNode[],
): CrossDeviceAnalysis {
  const devices = new Set<DeviceKind>();
  const transferPoints: CrossDeviceAnalysis["transferPoints"] = [];

  for (const node of nodes) {
    devices.add(node.device);

    // Check each input for device mismatch
    for (let i = 0; i < node.inputs.length; i++) {
      const input = node.inputs[i];
      const inputDevice = getRefDevice(input);

      if (inputDevice && inputDevice !== node.device) {
        transferPoints.push({
          nodeId: node.id,
          inputIndex: i,
          sourceDevice: inputDevice,
          targetDevice: node.device,
        });
      }
    }
  }

  return {
    devices,
    transferPoints,
    isSingleDevice: devices.size <= 1,
  };
}

/**
 * Get the device of a LazyRef.
 */
function getRefDevice(ref: LazyRef): DeviceKind | null {
  if (ref.kind === "materialized") {
    return ref.storage.device;
  }
  return ref.node.device;
}

// ============================================================================
// Transfer Statistics Tracking
// ============================================================================

/**
 * Aggregated transfer statistics for performance monitoring.
 */
export interface TransferStats {
  totalTransfers: number;
  totalBytesTransferred: number;
  transfersByPath: Map<string, { count: number; bytes: number }>;
}

/**
 * Create an empty transfer stats object.
 */
export function createTransferStats(): TransferStats {
  return {
    totalTransfers: 0,
    totalBytesTransferred: 0,
    transfersByPath: new Map(),
  };
}

/**
 * Record a transfer in the stats.
 */
export function recordTransfer(
  stats: TransferStats,
  result: TransferResult,
): void {
  if (result.stats.bytesTransferred === 0) {
    return; // Noop transfer
  }

  stats.totalTransfers++;
  stats.totalBytesTransferred += result.stats.bytesTransferred;

  const pathKey = `${result.stats.path.sourceDevice}->${result.stats.path.targetDevice}`;
  const existing = stats.transfersByPath.get(pathKey) ?? { count: 0, bytes: 0 };
  existing.count++;
  existing.bytes += result.stats.bytesTransferred;
  stats.transfersByPath.set(pathKey, existing);
}

// ============================================================================
// Device Placement Utilities
// ============================================================================

/**
 * Determine the preferred device for an operation given its inputs.
 * Uses simple heuristics:
 * - If all inputs are on the same device, use that device
 * - If inputs are on different devices, prefer GPU over CPU
 * - If no inputs, use the specified default device
 */
export function inferOperationDevice(
  inputDevices: DeviceKind[],
  defaultDevice: DeviceKind = "cpu",
): DeviceKind {
  if (inputDevices.length === 0) {
    return defaultDevice;
  }

  const deviceSet = new Set(inputDevices);

  if (deviceSet.size === 1) {
    return inputDevices[0];
  }

  // Multiple devices - prefer GPU
  if (deviceSet.has("webgpu")) {
    return "webgpu";
  }

  return inputDevices[0];
}

/**
 * Check if auto-transfer should be inserted for a cross-device operation.
 * Returns the device to transfer to, or null if no transfer needed.
 */
export function shouldAutoTransfer(
  inputDevices: DeviceKind[],
  targetDevice: DeviceKind,
): DeviceKind | null {
  for (const device of inputDevices) {
    if (device !== targetDevice) {
      return targetDevice;
    }
  }
  return null;
}
