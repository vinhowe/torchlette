/**
 * Gradient Checkpointing API (§10)
 *
 * PyTorch-like activation checkpointing for trading compute for memory.
 * During forward, intermediate activations are replaced with lightweight
 * placeholders via pack hooks. During backward, the unpack hook triggers
 * recomputation to restore tensors on-demand.
 *
 * This implementation uses saved_tensors_hooks (non-reentrant checkpointing),
 * matching PyTorch's modern use_reentrant=False approach.
 *
 * Usage:
 * ```typescript
 * import { checkpoint } from "torchlette/nn";
 *
 * // Without checkpointing - all activations saved
 * const output = block.forward(input);
 *
 * // With checkpointing - activations recomputed during backward
 * const output = checkpoint(api, (x) => block.forward(x), [input]);
 * ```
 *
 * This is especially useful for transformer layers where memory grows
 * linearly with sequence length and layer count.
 */

import type { PackHook, Tensor, Torchlette, UnpackHook } from "../frontend";
import { markAsCheckpointBoundary } from "../engine/lazy";

// ============================================================================
// Types
// ============================================================================

export type CheckpointOptions = {
  /**
   * Whether to preserve RNG state for deterministic recomputation.
   * Default: true
   */
  preserveRngState?: boolean;

  /**
   * Whether to use reentrant checkpointing (PyTorch-style).
   * When false, uses simpler non-reentrant implementation.
   * Default: false (non-reentrant is more efficient and recommended)
   */
  useReentrant?: boolean;
};

/**
 * Placeholder stored instead of actual tensor during checkpointed forward.
 * Contains minimal metadata needed to identify the tensor for recomputation.
 */
interface CheckpointPlaceholder {
  /** Index of this tensor in the save order */
  checkpointIndex: number;
  /** Original tensor's baseId for validation */
  baseId: number;
}

// ============================================================================
// Checkpoint Implementation
// ============================================================================

/**
 * Apply gradient checkpointing to a function.
 *
 * During the forward pass, the function is executed normally but intermediate
 * activations are replaced with placeholders via pack hooks. During the
 * backward pass, the unpack hook triggers recomputation to restore tensors.
 *
 * @param api - The Torchlette API instance
 * @param fn - The function to checkpoint (typically a forward method of a module)
 * @param inputs - Input tensors to the function
 * @param options - Checkpointing options
 * @returns The output tensor with checkpointed backward
 *
 * @example
 * ```typescript
 * // Checkpoint a transformer block
 * const output = checkpoint(api, (x) => block.forward(x), [input]);
 *
 * // Checkpoint multiple layers
 * let hidden = input;
 * for (const layer of layers) {
 *   hidden = checkpoint(api, (x) => layer.forward(x), [hidden]);
 * }
 * ```
 */
export function checkpoint<T extends Tensor>(
  api: Torchlette,
  fn: (...args: Tensor[]) => T,
  inputs: Tensor[],
  options: CheckpointOptions = {},
): T {
  const { preserveRngState = true, useReentrant = false } = options;

  if (useReentrant) {
    // Reentrant falls back to non-reentrant for now
    return checkpointNonReentrant(api, fn, inputs, preserveRngState);
  }
  return checkpointNonReentrant(api, fn, inputs, preserveRngState);
}

/**
 * Non-reentrant checkpoint implementation using pack/unpack hooks.
 *
 * This is the modern, efficient implementation that intercepts tensor
 * saving/restoring at the autograd level. It's sufficient for most
 * use cases like checkpointing individual transformer layers.
 *
 * How it works:
 * 1. During forward: pack hook replaces each saved tensor with a placeholder
 * 2. During backward: unpack hook triggers recomputation of the forward
 *    function, returning the recomputed tensor for the requested index
 */
function checkpointNonReentrant<T extends Tensor>(
  api: Torchlette,
  fn: (...args: Tensor[]) => T,
  inputs: Tensor[],
  preserveRngState: boolean,
): T {
  // Get the underlying engine for checkpoint primitives
  const engine = (api as any).engine;

  // Keep inputs alive for recomputation
  for (const input of inputs) {
    api.keep(input);
  }

  // Record RNG state if needed for deterministic recomputation
  let rngDraws: unknown[] | null = null;
  if (preserveRngState) {
    engine._debug_startCheckpointRecord();
  }

  // State for recomputation
  let recomputedTensors: Map<number, Tensor> | null = null;
  let tensorIndex = 0;

  // Pack hook: replace tensor with placeholder during forward
  const packHook: PackHook = (tensor: Tensor): CheckpointPlaceholder => {
    const idx = tensorIndex++;
    return { checkpointIndex: idx, baseId: tensor.baseId };
  };

  // Unpack hook: trigger recomputation and return the requested tensor
  const unpackHook: UnpackHook = (packed: unknown): Tensor => {
    const { checkpointIndex } = packed as CheckpointPlaceholder;

    if (!recomputedTensors) {
      // First unpack call - trigger recomputation
      recomputedTensors = new Map();

      // Enter recompute mode with RNG replay if needed
      if (rngDraws && rngDraws.length > 0) {
        engine._debug_startCheckpointReplay(rngDraws);
      }

      // Run forward again, but this time WITHOUT hooks
      // so tensors are saved normally and captured
      let captureIndex = 0;
      let lastCapturedTensor: Tensor | null = null;

      // Capture tensors during recomputation
      const captureHook: PackHook = (t: Tensor): Tensor => {
        recomputedTensors!.set(captureIndex++, t);
        lastCapturedTensor = t;
        return t; // Return tensor directly during recompute (no placeholder)
      };

      // Recompute with capture hooks inside tidy() to dispose non-captured
      // intermediates. Without this, recomputed forward creates frontend Tensors
      // whose autograd graph chains keep RuntimeTensors alive until GC,
      // causing ~150 StorageHandle leaks per training step.
      api.tidy(() => {
        api.saved_tensors_hooks(captureHook, (t) => t as Tensor, () => {
          fn(...inputs);
        });
        // Keep captured tensors so they survive tidy disposal
        for (const t of recomputedTensors!.values()) {
          api.keep(t);
        }
      });

      // Exit RNG replay mode
      if (rngDraws && rngDraws.length > 0) {
        engine._debug_finishCheckpointReplay();
      }

      // Mark the last captured tensor as a checkpoint boundary.
      // This tells the segmented executor to flush buffers after this
      // checkpoint region, enabling memory reuse for subsequent segments.
      if (lastCapturedTensor) {
        const lazyRef = (lastCapturedTensor as any).lazyRef;
        if (lazyRef?.kind === "pending" && lazyRef.node) {
          markAsCheckpointBoundary(lazyRef.node);
        }
      }
    }

    const result = recomputedTensors.get(checkpointIndex);
    if (!result) {
      throw new Error(`Checkpoint: no tensor at index ${checkpointIndex} after recomputation`);
    }
    return result;
  };

  // Run forward with pack/unpack hooks inside tidy() to dispose intermediates
  // The output tensor is kept, but all other tensors created inside fn() are disposed
  const output = api.tidy(() => {
    const result = api.saved_tensors_hooks(packHook, unpackHook, () => {
      return fn(...inputs);
    });
    // Keep the output so it survives tidy()
    api.keep(result);
    return result;
  });

  // Capture RNG draws for replay during backward
  if (preserveRngState) {
    rngDraws = engine._debug_finishCheckpointRecord();
  }

  return output;
}

// ============================================================================
// Sequential Checkpointing
// ============================================================================

/**
 * Checkpoint a sequence of modules, applying checkpointing to each.
 *
 * This is useful for stacks of layers like transformer blocks where
 * you want to checkpoint each layer individually.
 *
 * @param api - The Torchlette API instance
 * @param modules - Array of modules with forward() methods
 * @param input - Input tensor to the first module
 * @param options - Checkpointing options applied to each module
 * @returns Output tensor from the last module
 *
 * @example
 * ```typescript
 * const output = checkpointSequential(api, transformerBlocks, embeddings);
 * ```
 */
export function checkpointSequential<M extends { forward(x: Tensor): Tensor }>(
  api: Torchlette,
  modules: M[],
  input: Tensor,
  options: CheckpointOptions = {},
): Tensor {
  let hidden = input;
  for (const module of modules) {
    hidden = checkpoint(api, (x) => module.forward(x), [hidden], options);
  }
  return hidden;
}

// ============================================================================
// Layer-wise Checkpointing Helper
// ============================================================================

/**
 * Options for layer-wise checkpointing.
 */
export type LayerCheckpointPolicy = {
  /**
   * Which layers to checkpoint. Options:
   * - "all": Checkpoint every layer
   * - "alternate": Checkpoint every other layer (0, 2, 4, ...)
   * - "none": No checkpointing
   * - number[]: Specific layer indices to checkpoint
   */
  policy: "all" | "alternate" | "none" | number[];

  /**
   * Preserve RNG state during recomputation.
   */
  preserveRngState?: boolean;
};

/**
 * Create a checkpointing wrapper based on policy.
 *
 * Returns a function that conditionally applies checkpointing based
 * on the layer index and policy.
 *
 * @param api - The Torchlette API instance
 * @param policy - Checkpointing policy configuration
 * @returns A function that wraps layer forward calls with optional checkpointing
 *
 * @example
 * ```typescript
 * const maybeCheckpoint = createLayerCheckpointer(api, { policy: "alternate" });
 *
 * let hidden = input;
 * for (let i = 0; i < layers.length; i++) {
 *   hidden = maybeCheckpoint(i, () => layers[i].forward(hidden));
 * }
 * ```
 */
export function createLayerCheckpointer(
  api: Torchlette,
  policy: LayerCheckpointPolicy,
): (layerIdx: number, fn: () => Tensor) => Tensor {
  const shouldCheckpoint = (idx: number): boolean => {
    if (policy.policy === "none") return false;
    if (policy.policy === "all") return true;
    if (policy.policy === "alternate") return idx % 2 === 0;
    if (Array.isArray(policy.policy)) return policy.policy.includes(idx);
    return false;
  };

  return (layerIdx: number, fn: () => Tensor): Tensor => {
    if (shouldCheckpoint(layerIdx)) {
      // Wrap the function to work with checkpoint API
      return checkpoint(api, fn, [], {
        preserveRngState: policy.preserveRngState ?? true,
      });
    }
    return fn();
  };
}

// ============================================================================
// Memory Estimation Helpers
// ============================================================================

/**
 * Estimate memory savings from checkpointing.
 *
 * @param numLayers - Number of layers in the model
 * @param hiddenSize - Hidden dimension size
 * @param seqLength - Sequence length
 * @param batchSize - Batch size
 * @param policy - Checkpointing policy
 * @returns Estimated memory stats in bytes
 */
export function estimateCheckpointSavings(
  numLayers: number,
  hiddenSize: number,
  seqLength: number,
  batchSize: number,
  policy: LayerCheckpointPolicy["policy"],
): {
  withoutCheckpoint: number;
  withCheckpoint: number;
  savings: number;
  savingsPercent: number;
} {
  // Rough estimate: each layer stores activations of size [batch, seq, hidden]
  const activationBytes = batchSize * seqLength * hiddenSize * 4; // f32

  // Additional activations in attention (QKV, attention weights)
  const attnActivations = activationBytes * 4; // Q, K, V, attention scores

  const perLayerActivations = activationBytes + attnActivations;
  const totalWithoutCheckpoint = numLayers * perLayerActivations;

  let checkpointedLayers = 0;
  if (policy === "all") {
    checkpointedLayers = numLayers;
  } else if (policy === "alternate") {
    checkpointedLayers = Math.ceil(numLayers / 2);
  } else if (Array.isArray(policy)) {
    checkpointedLayers = policy.length;
  }

  // With checkpointing, we only store activations for non-checkpointed layers
  // plus the inputs to each checkpointed segment
  const nonCheckpointedLayers = numLayers - checkpointedLayers;
  const inputsPerCheckpoint = activationBytes; // Store input to recompute segment
  const totalWithCheckpoint =
    nonCheckpointedLayers * perLayerActivations +
    checkpointedLayers * inputsPerCheckpoint;

  const savings = totalWithoutCheckpoint - totalWithCheckpoint;
  const savingsPercent = (savings / totalWithoutCheckpoint) * 100;

  return {
    withoutCheckpoint: totalWithoutCheckpoint,
    withCheckpoint: totalWithCheckpoint,
    savings,
    savingsPercent,
  };
}
