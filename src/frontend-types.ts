import type { AMPPolicy } from "./engine/amp";
import type { SavedTensorRecord } from "./engine/engine";
import type { DeviceKind } from "./backend/types";
import type { Tensor as RuntimeTensor } from "./runtime/tensor";

export type TensorCreateOptions = {
  requiresGrad?: boolean;
  device?: DeviceKind;
};

/**
 * Accessor function for retrieving saved tensors during backward pass.
 * May trigger recomputation if checkpointing hooks are active.
 */
export type GetSavedFn = (idx: number) => import("./frontend-tensor").Tensor;

/**
 * Backward function that computes gradients with respect to inputs.
 * @param grad - Gradient of the loss with respect to the output
 * @param getSaved - Accessor for saved tensors (may trigger recomputation)
 * @returns Array of gradients for each input (null if input doesn't require grad)
 */
export type GradFn = (
  grad: RuntimeTensor,
  getSaved: GetSavedFn,
) => Array<RuntimeTensor | null>;

export type AutogradNode = {
  inputs: import("./frontend-tensor").Tensor[];
  output: import("./frontend-tensor").Tensor;
  backward: GradFn;
  /** Saved tensor slots with pack/unpack hooks support */
  savedSlots: SavedTensorSlot[];
  /** Opaque annotation captured during forward, available to backward hooks */
  label?: string;
};

// ============================================================================
// Saved Tensor Hooks (for checkpointing, §10)
// ============================================================================

/**
 * Pack hook: transforms a tensor into a packed representation during forward.
 * For checkpointing, this replaces the tensor with a lightweight placeholder.
 */
export type PackHook = (tensor: import("./frontend-tensor").Tensor) => unknown;

/**
 * Unpack hook: restores a tensor from its packed representation during backward.
 * For checkpointing, this triggers recomputation to reconstruct the tensor.
 */
export type UnpackHook = (packed: unknown) => import("./frontend-tensor").Tensor;

/**
 * Context for saved tensor hooks.
 */
export interface SavedTensorHooksContext {
  packHook: PackHook;
  unpackHook: UnpackHook;
}

/**
 * Slot for a saved tensor with lazy unpacking support.
 * Used internally to defer tensor restoration until backward pass.
 */
export interface SavedTensorSlot {
  /** Packed representation (result of packHook, may be placeholder or tensor) */
  packed: unknown;
  /** Hook to restore the tensor */
  unpackHook: UnpackHook;
  /** Metadata for validation */
  record: SavedTensorRecord;
}

/**
 * Options for the autocast context.
 */
export type AutocastOptions = {
  /** Whether autocast is enabled (default: true) */
  enabled?: boolean;
  /** The AMP policy to use (default: DEFAULT_AMP_POLICY) */
  policy?: AMPPolicy;
  /** Device type hint for AMP (default: inferred from current backend) */
  deviceType?: "cpu" | "webgpu";
};

export type TorchletteOptions = {
  /** Enable fusion optimizations (§15). Default: false */
  enableFusion?: boolean;
  /** Enable memory planning for buffer reuse. Default: false */
  enableMemoryPlanning?: boolean;
  /** Maximum memory limit in bytes. Default: 10GB */
  memoryLimitBytes?: number;
  /** Enable early buffer release during execution for memory savings. Default: false */
  enableEarlyRelease?: boolean;
  /**
   * Enable segmented execution at checkpoint boundaries.
   * This enables memory savings for large models by executing checkpoint
   * segments separately and flushing buffers between them.
   * Default: false
   */
  enableCheckpointSegmentation?: boolean;
  /**
   * Enable true segmented execution with GPU synchronization.
   * Provides actual memory savings for checkpointed models by waiting for
   * GPU completion between segments before releasing buffers.
   * More effective than enableCheckpointSegmentation but slower.
   * Default: false
   */
  enableTrueSegmentation?: boolean;
};
