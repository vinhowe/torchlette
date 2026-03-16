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

export function getSizeClass(sizeBytes: number): number {
  const size = Math.max(sizeBytes, MIN_BUFFER_SIZE);
  return Math.ceil(Math.log2(size));
}

export function getSizeForClass(sizeClass: number): number {
  return 2 ** sizeClass;
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
