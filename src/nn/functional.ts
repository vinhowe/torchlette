/**
 * Functional neural network operations.
 * Similar to PyTorch's torch.nn.functional.
 */

import type { Tensor, Torchlette } from "../frontend";

/**
 * Apply dropout to a tensor.
 *
 * During training, randomly zeroes elements with probability p using a Bernoulli
 * distribution, and scales the remaining elements by 1/(1-p).
 *
 * During evaluation (training=false), returns the input unchanged.
 *
 * @param api - Torchlette instance
 * @param input - Input tensor
 * @param p - Probability of an element to be zeroed (default: 0.5)
 * @param training - Whether in training mode (default: true)
 * @returns Tensor with dropout applied
 */
export function dropout(
  api: Torchlette,
  input: Tensor,
  p = 0.5,
  training = true,
): Tensor {
  if (!training || p === 0) {
    return input;
  }
  if (p === 1) {
    // All zeros
    return api.mul(input, 0);
  }
  if (p < 0 || p > 1) {
    throw new Error(`dropout probability must be between 0 and 1, got ${p}`);
  }

  // Generate bernoulli mask: 1 with probability (1-p), 0 with probability p
  const mask = api.bernoulli(input.shape, 1 - p, { device: input.device });

  // Scale by 1/(1-p) to maintain expected value
  const scale = 1 / (1 - p);
  const scaled = api.mul(input, mask);
  return api.mul(scaled, scale);
}

/**
 * Compute cross-entropy loss.
 *
 * For multi-class classification:
 *   loss = -log(softmax(logits)[target])
 *
 * More numerically stable version:
 *   loss = -logits[target] + log(sum(exp(logits)))
 *
 * @param api - Torchlette instance
 * @param logits - Unnormalized scores, shape [batch, num_classes] or [num_classes]
 * @param targets - Target class indices, shape [batch] or scalar
 * @param options - Options
 * @returns Loss tensor (scalar if reduction='mean' or 'sum', otherwise [batch])
 */
export function crossEntropy(
  api: Torchlette,
  logits: Tensor,
  targets: Tensor,
  options?: {
    reduction?: "none" | "mean" | "sum";
  },
): Tensor {
  const reduction = options?.reduction ?? "mean";

  // Ensure logits has at least 1 dimension
  if (logits.shape.length === 0) {
    throw new Error("crossEntropy: logits must have at least 1 dimension");
  }

  // Fused path for WebGPU (single kernel instead of 9 ops)
  if (logits.device === "webgpu" && logits.shape.length === 2) {
    const perSampleLoss = api._crossEntropyFused(logits, targets);
    if (reduction === "none") return perSampleLoss;
    if (reduction === "sum") {
      const s = perSampleLoss.sum();
      if (typeof s === "number") return api.tensorFromArray([s], [], { device: logits.device });
      return s;
    }
    const m = perSampleLoss.mean();
    if (typeof m === "number") return api.tensorFromArray([m], [], { device: logits.device });
    return m;
  }

  // f16 inputs are automatically upcast to f32 by the RuntimeEngine's
  // ensureDtypeSafety when max/sum/exp/log are called below.

  // Get dimensions
  const isBatched = logits.shape.length >= 2;
  const numClasses = logits.shape[logits.shape.length - 1];

  // For numerical stability, compute: -logits[target] + log(sum(exp(logits - max(logits))))
  // This is equivalent to: -log(softmax(logits)[target])

  // Step 1: Compute log-softmax along the last dimension
  // log_softmax = logits - max(logits) - log(sum(exp(logits - max(logits))))
  const dim = -1;
  const maxLogits = logits.max({ dim, keepdim: true });
  if (typeof maxLogits === "number") {
    throw new Error("crossEntropy: max with keepdim should return tensor");
  }
  const shifted = api.sub(logits, maxLogits);
  const expShifted = shifted.exp();
  const sumExp = expShifted.sum({ dim, keepdim: true });
  if (typeof sumExp === "number") {
    throw new Error("crossEntropy: sum with keepdim should return tensor");
  }
  const logSumExp = sumExp.log();
  const logSoftmax = api.sub(shifted, logSumExp);

  // Step 2: Gather the log-softmax values at target indices
  // This gives us -log(softmax[target])
  // For batch inputs, targets is [batch] but gather needs same rank as input [batch, classes]
  // So we unsqueeze targets to [batch, 1], gather gives [batch, 1], then reshape to [batch]
  let targetsForGather = targets;
  if (isBatched && targets.shape.length === logits.shape.length - 1) {
    // Unsqueeze targets to add dimension for gather
    targetsForGather = targets.reshape([...targets.shape, 1]);
  }
  const gatheredLogProbs = api.gather(logSoftmax, targetsForGather, {
    dim: logits.shape.length - 1,
  });

  // Squeeze the gathered result if we added a dimension
  let gatheredSqueezed = gatheredLogProbs;
  if (isBatched && targets.shape.length === logits.shape.length - 1) {
    // Remove the last dimension we added (shape goes from [batch, 1] to [batch])
    gatheredSqueezed = gatheredLogProbs.reshape(targets.shape);
  }

  // Step 3: Negate to get the loss
  const loss = api.neg(gatheredSqueezed);

  // Step 4: Apply reduction
  if (reduction === "none") {
    return loss;
  } else if (reduction === "sum") {
    const sumLoss = loss.sum();
    if (typeof sumLoss === "number") {
      return api.tensorFromArray([sumLoss], [], { device: logits.device });
    }
    return sumLoss;
  } else {
    // mean
    const meanLoss = loss.mean();
    if (typeof meanLoss === "number") {
      return api.tensorFromArray([meanLoss], [], { device: logits.device });
    }
    return meanLoss;
  }
}

/**
 * Compute negative log likelihood loss.
 *
 * Takes log-probabilities (output of log_softmax) and target indices.
 *   loss = -log_probs[target]
 *
 * @param api - Torchlette instance
 * @param logProbs - Log-probabilities, shape [batch, num_classes]
 * @param targets - Target class indices, shape [batch]
 * @param options - Options
 * @returns Loss tensor
 */
export function nllLoss(
  api: Torchlette,
  logProbs: Tensor,
  targets: Tensor,
  options?: {
    reduction?: "none" | "mean" | "sum";
  },
): Tensor {
  const reduction = options?.reduction ?? "mean";

  // Gather log probs at target indices and negate
  // For batch inputs, targets is [batch] but gather needs same rank as input [batch, classes]
  const isBatched = logProbs.shape.length >= 2;
  let targetsForGather = targets;
  if (isBatched && targets.shape.length === logProbs.shape.length - 1) {
    targetsForGather = targets.reshape([...targets.shape, 1]);
  }
  const gatheredLogProbs = api.gather(logProbs, targetsForGather, {
    dim: logProbs.shape.length - 1,
  });

  // Squeeze if we added a dimension
  let gatheredSqueezed = gatheredLogProbs;
  if (isBatched && targets.shape.length === logProbs.shape.length - 1) {
    gatheredSqueezed = gatheredLogProbs.reshape(targets.shape);
  }
  const loss = api.neg(gatheredSqueezed);

  if (reduction === "none") {
    return loss;
  } else if (reduction === "sum") {
    const sumLoss = loss.sum();
    if (typeof sumLoss === "number") {
      return api.tensorFromArray([sumLoss], [], { device: logProbs.device });
    }
    return sumLoss;
  } else {
    const meanLoss = loss.mean();
    if (typeof meanLoss === "number") {
      return api.tensorFromArray([meanLoss], [], { device: logProbs.device });
    }
    return meanLoss;
  }
}

/**
 * Compute log-softmax along a dimension.
 *
 * log_softmax(x) = x - max(x) - log(sum(exp(x - max(x))))
 *
 * @param api - Torchlette instance
 * @param input - Input tensor
 * @param dim - Dimension to compute log-softmax along
 * @returns Log-softmax tensor
 */
export function logSoftmax(
  api: Torchlette,
  input: Tensor,
  dim: number,
): Tensor {
  const maxVal = input.max({ dim, keepdim: true });
  if (typeof maxVal === "number") {
    throw new Error("logSoftmax: max with keepdim should return tensor");
  }
  const shifted = api.sub(input, maxVal);
  const expShifted = shifted.exp();
  const sumExp = expShifted.sum({ dim, keepdim: true });
  if (typeof sumExp === "number") {
    throw new Error("logSoftmax: sum with keepdim should return tensor");
  }
  const logSumExp = sumExp.log();
  return api.sub(shifted, logSumExp);
}
