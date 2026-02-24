import type { DType } from "../backend/types";
import type { LazyRef } from "../engine/lazy";
import { type BaseId, Tensor } from "./tensor";

/** A tensor or a numeric scalar (will be inlined as a constant in fused kernels). */
export type TensorOrScalar = Tensor | number;

/**
 * Lightweight reference to a lazy computation with shape/dtype metadata.
 * Used internally by ensureDtypeSafety to build cast nodes without creating
 * full Tensor objects (which would register in pendingTensorsByNodeId and leak).
 * Tensor satisfies this interface, so existing Tensors can be used directly.
 */
export type OpInput = Pick<Tensor, "lazyRef" | "shape" | "dtype" | "device">;

/**
 * Dispatch mode: receives notifications when RuntimeTensors are created or escape a scope.
 * Pushed/popped on a stack on RuntimeEngine. All active modes are notified.
 */
export interface DispatchMode {
  onTensorCreated(tensor: Tensor): void;
  onTensorEscaped?(tensor: Tensor): void;
}

/**
 * Dispatch mode that tracks RuntimeTensors created within a tidy scope.
 * Tensors that are wrapped (via markEscaped) are exempt from disposal.
 * At scope exit, disposeNonEscaped() cleans up unwrapped intermediates.
 */
export class TidyDispatchMode implements DispatchMode {
  readonly tracked = new Set<Tensor>();
  readonly escaped = new Set<Tensor>();

  onTensorCreated(tensor: Tensor): void {
    this.tracked.add(tensor);
  }

  onTensorEscaped(tensor: Tensor): void {
    this.escaped.add(tensor);
  }

  /** Dispose all RuntimeTensors that weren't wrapped (didn't escape). */
  disposeNonEscaped(): void {
    for (const t of this.tracked) {
      if (!this.escaped.has(t) && !t.disposed) {
        t.dispose();
      }
    }
  }
}

export interface RuntimeEngineOptions {
  /** Enable memory planning for buffer pooling and reuse */
  enableMemoryPlanning?: boolean;
  /** Enable buffer donation (reuse input buffers for outputs) */
  enableDonation?: boolean;
  /** Track memory planning statistics */
  trackStats?: boolean;
  /** Enable automatic fusion of elementwise ops (§15) */
  enableFusion?: boolean;
  /** Enable vectorization for fused kernels */
  enableVectorization?: boolean;
  /** Enable early buffer release during execution for memory savings */
  enableEarlyRelease?: boolean;
  /**
   * Enable segmented execution at checkpoint boundaries.
   * This enables memory savings for large models by executing segments
   * separately and flushing buffers between them.
   */
  enableCheckpointSegmentation?: boolean;
  /**
   * Enable true segmented execution with GPU synchronization.
   * Provides actual memory savings for checkpointed models at cost of
   * increased latency due to GPU sync points between segments.
   * This is more effective than enableCheckpointSegmentation but slower.
   */
  enableTrueSegmentation?: boolean;
}

/** Internal dispatch mode used by startIntermediateTracking/stopIntermediateTracking. */
export class IntermediateTrackingMode implements DispatchMode {
  readonly tracked = new Set<Tensor>();
  onTensorCreated(tensor: Tensor): void {
    this.tracked.add(tensor);
  }
}
