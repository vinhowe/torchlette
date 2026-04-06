/**
 * Tensor Lifetime Analysis
 *
 * Analyzes when tensors are first/last used in an execution plan,
 * enabling early buffer release during plan execution.
 *
 * Also provides size-class utilities used by the WebGPU buffer pool and arena.
 */

// ============================================================================
// Types
// ============================================================================

export interface TensorLifetime {
  nodeId: number;
  firstUse: number; // Step index
  lastUse: number; // Step index
  isOutput: boolean;
  isInput: boolean;
  bufferSize: number;
}

// ============================================================================
// Size Class Utilities
// ============================================================================

const MIN_BUFFER_SIZE = 256;

// Binning scheme:
//   sizes <= 128MB: log2 classes (8=256B, 9=512B, ..., 27=128MB) — doubles each step.
//   sizes > 128MB:  linear 16MB steps (28=144MB, 29=160MB, ...) — reduces internal
//     fragmentation for large buffers. At distilgpt2 batch=4 seq=512 this recovers
//     ~2GB of peak that was lost rounding 589MB→1024MB, 393MB→512MB, 154MB→256MB
//     in the prior log2-only scheme.
const LARGE_THRESHOLD_CLASS = 27; // 2^27 = 128MB
const LARGE_THRESHOLD = 1 << LARGE_THRESHOLD_CLASS;
const LARGE_STEP_BITS = 24; // 2^24 = 16MB
const LARGE_STEP = 1 << LARGE_STEP_BITS;

export function getSizeClass(sizeBytes: number): number {
  const size = Math.max(sizeBytes, MIN_BUFFER_SIZE);
  if (size <= LARGE_THRESHOLD) {
    return Math.ceil(Math.log2(size));
  }
  return (
    LARGE_THRESHOLD_CLASS + Math.ceil((size - LARGE_THRESHOLD) / LARGE_STEP)
  );
}

export function getSizeForClass(sizeClass: number): number {
  if (sizeClass <= LARGE_THRESHOLD_CLASS) {
    return 2 ** sizeClass;
  }
  return LARGE_THRESHOLD + (sizeClass - LARGE_THRESHOLD_CLASS) * LARGE_STEP;
}

// ============================================================================
// Lifetime Analysis
// ============================================================================

/**
 * Analyze tensor lifetimes in an execution plan.
 */
export function analyzeLifetimes(
  nodeOrder: number[],
  nodeInputs: Map<number, number[]>,
  nodeOutputs: Set<number>,
  nodeSizes: Map<number, number>,
): Map<number, TensorLifetime> {
  const lifetimes = new Map<number, TensorLifetime>();
  const stepIndex = new Map<number, number>();

  nodeOrder.forEach((nodeId, idx) => {
    stepIndex.set(nodeId, idx);
  });

  for (const nodeId of nodeOrder) {
    const step = stepIndex.get(nodeId) as number;
    lifetimes.set(nodeId, {
      nodeId,
      firstUse: step,
      lastUse: step,
      isOutput: nodeOutputs.has(nodeId),
      isInput: !nodeInputs.has(nodeId) || nodeInputs.get(nodeId)?.length === 0,
      bufferSize: nodeSizes.get(nodeId) ?? 0,
    });
  }

  for (const [nodeId, inputs] of nodeInputs) {
    const nodeStep = stepIndex.get(nodeId) as number;
    for (const inputId of inputs) {
      const lifetime = lifetimes.get(inputId);
      if (lifetime && nodeStep > lifetime.lastUse) {
        lifetime.lastUse = nodeStep;
      }
    }
  }

  const lastStep = nodeOrder.length - 1;
  for (const outputId of nodeOutputs) {
    const lifetime = lifetimes.get(outputId);
    if (lifetime) {
      lifetime.lastUse = lastStep;
    }
  }

  return lifetimes;
}

/**
 * Find tensors that become dead at a specific step during execution.
 * Used for incremental buffer release during plan execution.
 */
export function findDeadTensorsAtStep(
  lifetimes: Map<number, TensorLifetime>,
  currentStep: number,
  outputNodeIds: Set<number>,
  alreadyReleased: Set<number>,
): number[] {
  const dead: number[] = [];
  for (const [nodeId, lifetime] of lifetimes) {
    if (outputNodeIds.has(nodeId)) continue;
    if (alreadyReleased.has(nodeId)) continue;
    if (lifetime.lastUse < currentStep) {
      dead.push(nodeId);
    }
  }
  return dead;
}
