/**
 * Functional neural network operations.
 * Similar to PyTorch's torch.nn.functional.
 */

import type { DeviceKind } from "../backend/types";
import type { Tensor, Torchlette } from "../frontend/torchlette";

/** Apply none/sum/mean reduction to a per-sample loss tensor. */
function applyReduction(
  api: Torchlette,
  loss: Tensor,
  reduction: "none" | "mean" | "sum",
  device: DeviceKind,
): Tensor {
  if (reduction === "none") return loss;
  if (reduction === "sum") {
    const s = loss.sum();
    return typeof s === "number" ? api.full([], s, { device }) : s;
  }
  const m = loss.mean();
  return typeof m === "number" ? api.full([], m, { device }) : m;
}

/** Gather values at target indices along the last dim, handling unsqueeze/squeeze. */
function gatherTargets(
  api: Torchlette,
  input: Tensor,
  targets: Tensor,
): Tensor {
  const isBatched = input.shape.length >= 2;
  // gather kernel reads indices from an f32 binding; cast i32/u32 → f32 if needed.
  let targetsForGather =
    targets.dtype === "f32" ? targets : api.toDtype(targets, "f32");
  if (isBatched && targets.shape.length === input.shape.length - 1) {
    targetsForGather = targetsForGather.reshape([...targets.shape, 1]);
  }
  const gathered = api.gather(input, targetsForGather, {
    dim: input.shape.length - 1,
  });
  if (isBatched && targets.shape.length === input.shape.length - 1) {
    return gathered.reshape(targets.shape);
  }
  return gathered;
}

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
    ignoreIndex?: number;
  },
): Tensor {
  const reduction = options?.reduction ?? "mean";
  const ignoreIndex = options?.ignoreIndex;

  // Ensure logits has at least 1 dimension
  if (logits.shape.length === 0) {
    throw new Error("crossEntropy: logits must have at least 1 dimension");
  }

  // Fused path for WebGPU (single kernel instead of 9 ops)
  if (logits.device === "webgpu" && logits.shape.length === 2) {
    const perSample = api._crossEntropyFused(logits, targets, ignoreIndex);
    if (ignoreIndex !== undefined && reduction === "mean") {
      // Mean over non-ignored samples: sum(loss) / count_valid.
      // The kernel writes 0 for ignored rows, so sum gives the correct total.
      // Use 'sum' reduction and divide by valid count on the GPU.
      const lossSum = perSample.sum();
      // Count valid (non-ignored) targets
      const ignoreT = api.full(targets.shape, ignoreIndex, { device: logits.device, dtype: targets.dtype });
      const mask = api.ne(targets, ignoreT);
      const validCount = api.mul(mask, 1.0).sum();
      return api.div(lossSum, api.add(validCount, 1e-8));
    }
    return applyReduction(api, perSample, reduction, logits.device);
  }

  // Fallback: decomposed cross-entropy via log-softmax + gather
  const logProbs = logSoftmax(api, logits, -1);
  const loss = api.neg(gatherTargets(api, logProbs, targets));

  // Step 3: Apply reduction
  return applyReduction(api, loss, reduction, logits.device);
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

  const loss = api.neg(gatherTargets(api, logProbs, targets));
  return applyReduction(api, loss, reduction, logProbs.device);
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
