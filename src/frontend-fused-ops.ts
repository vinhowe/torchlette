import type { Tensor as RuntimeTensor } from "./runtime/tensor";
import type { Torchlette } from "./frontend";
import type { Tensor } from "./frontend-tensor";
import { autocastCastImpl, applyAutocastImpl } from "./frontend-autocast";

/**
 * Softmax along a dimension.
 * softmax(x, dim) = exp(x - max(x, dim, keepdim=true)) / sum(exp(...), dim, keepdim=true)
 */
export function softmaxImpl(torch: Torchlette, a: Tensor, dim: number): Tensor {
  torch._assertUsable(a);
  // Autocast: softmax is F32-required for numerical stability
  const [castA] = applyAutocastImpl(torch, "softmax", [a]);
  const rank = castA.shape.length;
  const normalizedDim = dim < 0 ? dim + rank : dim;
  if (normalizedDim < 0 || normalizedDim >= rank) {
    throw new Error(`softmax: dim ${dim} out of range for tensor of rank ${rank}`);
  }

  // Numerical stability: subtract max
  const maxResult = torch.runtime.max(castA._unwrap(), { dim: normalizedDim, keepdim: true });
  if (typeof maxResult === "number") {
    throw new Error("softmax: max with keepdim=true should return tensor");
  }
  const shifted = torch.runtime.sub(castA._unwrap(), maxResult);
  const exps = torch.runtime.exp(shifted);
  const sumResult = torch.runtime.sum(exps, { dim: normalizedDim, keepdim: true });
  if (typeof sumResult === "number") {
    throw new Error("softmax: sum with keepdim=true should return tensor");
  }
  const result = torch.runtime.div(exps, sumResult);

  const tensorsToSave = a.requiresGrad ? [castA] : [];
  // Softmax backward: grad_input = softmax * (grad_output - sum(softmax * grad_output, dim, keepdim=true))
  return torch._wrapWithGrad(result, [a], (grad, getSaved) => {
    // Recompute softmax from saved input for checkpointing support
    const savedA = getSaved(0);
    const savedMax = torch.runtime.max(savedA._unwrap(), { dim: normalizedDim, keepdim: true });
    if (typeof savedMax === "number") {
      throw new Error("softmax backward: max with keepdim=true should return tensor");
    }
    const savedShifted = torch.runtime.sub(savedA._unwrap(), savedMax);
    const savedExps = torch.runtime.exp(savedShifted);
    const savedSum = torch.runtime.sum(savedExps, { dim: normalizedDim, keepdim: true });
    if (typeof savedSum === "number") {
      throw new Error("softmax backward: sum with keepdim=true should return tensor");
    }
    const softmaxResult = torch.runtime.div(savedExps, savedSum);

    const softmaxMulGrad = torch.runtime.mul(softmaxResult, grad);
    const sumGradResult = torch.runtime.sum(softmaxMulGrad, { dim: normalizedDim, keepdim: true });
    if (typeof sumGradResult === "number") {
      throw new Error("softmax backward: sum with keepdim=true should return tensor");
    }
    const gradMinusSum = torch.runtime.sub(grad, sumGradResult);
    const gradInput = torch.runtime.mul(softmaxResult, gradMinusSum);
    return [gradInput];
  }, tensorsToSave);
}

/**
 * Fused cross-entropy forward + backward for WebGPU.
 * logits [B, V] + targets [B] → per-sample loss [B]
 * Backward: fused kernel → grad_logits [B, V]
 */
export function crossEntropyFusedImpl(torch: Torchlette, logits: Tensor, targets: Tensor): Tensor {
  torch._assertUsable(logits, targets);

  // Upcast f16 logits to f32 for numerical stability (same as softmax)
  let castLogits = logits;
  if (logits.dtype === "f16") {
    castLogits = autocastCastImpl(torch, logits, "f32");
  }

  const B = castLogits.shape[0];
  const V = castLogits.shape[1];
  const config = { batchSize: B, vocabSize: V };

  const result = torch.runtime.fusedCrossEntropyForward(
    castLogits._unwrap(), targets._unwrap(), config,
  );

  // Save logits for backward (needs recomputation of softmax).
  // Targets are captured via closure — they don't require grad and aren't
  // modified between forward and backward, so keep() is unnecessary.
  const tensorsToSave = logits.requiresGrad ? [castLogits] : [];
  const targetsInner = targets._unwrap();
  return torch._wrapWithGrad(result, [logits], (grad, getSaved) => {
    const savedLogits = getSaved(0);
    const gradLogits = torch.runtime.fusedCrossEntropyBackward(
      savedLogits._unwrap(), targetsInner, grad, config,
    );
    return [gradLogits];
  }, tensorsToSave);
}

/**
 * Scaled dot-product attention with optional causal mask.
 * q, k, v: [batch, heads, seq_len, head_dim]
 * Returns: [batch, heads, seq_len, head_dim]
 *
 * On WebGPU, uses fused FlashAttention kernel (single dispatch).
 * On CPU, falls back to decomposed matmul + softmax + matmul.
 */
export function scaledDotProductAttentionImpl(
  torch: Torchlette,
  q: Tensor, k: Tensor, v: Tensor, scale?: number, isCausal = false,
): Tensor {
  torch._assertUsable(q, k, v);
  const [batch, heads, seq, hd] = q.shape;
  const actualScale = scale ?? (1.0 / Math.sqrt(hd));
  const config = {
    batchSize: batch, numHeads: heads, seqLen: seq, headDim: hd,
    scale: actualScale, isCausal,
  };

  if (q.device === "webgpu") {
    // Fused FlashAttention path
    const fwdResult = torch.runtime.fusedAttentionForward(
      q._unwrap(), k._unwrap(), v._unwrap(), config,
    );
    const logsumexp = torch.runtime.extractAttentionLogsumexp(fwdResult, config);

    // Save Q, K, V, logsumexp, and output O for backward.
    // IMPORTANT: Create reshape views with independent RuntimeTensors for saved
    // tensors. Without this, wrap(fwdResult) shares the same RuntimeTensor as the
    // wrapWithGrad output. When checkpoint tidy disposes these wrappers, it calls
    // dispose() on the shared RuntimeTensor BEFORE materialization, setting _disposed=true.
    // Later disposal of the real output hits the idempotency guard and skips cleanup,
    // leaking the GPU buffer. Similarly, disposing the logsumexp wrapper unregisters
    // its pending node, preventing extractAttentionLogsumexp from executing and
    // orphaning the side-output buffer. Reshape-to-same-shape creates a new
    // RuntimeTensor that can be safely disposed without poisoning the original.
    const logsumexpRef = torch.runtime.reshape(logsumexp, [batch, heads, seq]);
    const logsumexpTensor = torch._wrap(logsumexpRef);
    const outputRef = torch.runtime.reshape(fwdResult, [batch, heads, seq, hd]);
    const outputTensor = torch._wrap(outputRef);

    const tensorsToSave = (q.requiresGrad || k.requiresGrad || v.requiresGrad)
      ? [q, k, v, logsumexpTensor, outputTensor]
      : [];

    return torch._wrapWithGrad(fwdResult, [q, k, v], (dO, getSaved) => {
      const sQ = getSaved(0);
      const sK = getSaved(1);
      const sV = getSaved(2);
      const sL = getSaved(3);
      const sO = getSaved(4);

      // O is the 6th input to backward for D precomputation
      const bwdDQ = torch.runtime.fusedAttentionBackward(
        sQ._unwrap(), sK._unwrap(), sV._unwrap(), sL._unwrap(), dO, sO._unwrap(), config,
      );
      const dK = torch.runtime.extractAttentionDK(bwdDQ, config);
      const dV = torch.runtime.extractAttentionDV(bwdDQ, config);
      return [bwdDQ, dK, dV]; // dQ, dK, dV for inputs [q, k, v]
    }, tensorsToSave);
  }

  // CPU fallback: decomposed matmul + softmax + matmul
  const kT = torch.runtime.transpose(k._unwrap(), { dim0: 2, dim1: 3 });
  const scores = torch.runtime.matmul(q._unwrap(), kT);
  const scaleTensor = torch.runtime.full([], actualScale, q.device);
  const scaledScores = torch.runtime.mul(scores, scaleTensor);

  let finalScores: RuntimeTensor;
  if (isCausal) {
    // Create causal mask: -1e9 where j > i, 0 elsewhere
    const negInf = torch.runtime.full([1, 1, seq, seq], -1e9, q.device);
    const mask = torch.runtime.triu(negInf, 1);
    finalScores = torch.runtime.add(scaledScores, mask);
  } else {
    finalScores = scaledScores;
  }

  // Softmax along last dim (using the public softmax which handles autograd)
  const softmaxResult = softmaxImpl(torch, torch._wrap(finalScores), -1);
  const output = torch.runtime.matmul(softmaxResult._unwrap(), v._unwrap());

  // Wrap with autograd
  const tensorsToSave = (q.requiresGrad || k.requiresGrad || v.requiresGrad)
    ? [q, k, v, softmaxResult]
    : [];

  return torch._wrapWithGrad(output, [q, k, v], (dO, getSaved) => {
    const sQ = getSaved(0);
    const sK = getSaved(1);
    const sV = getSaved(2);
    const sSoftmax = getSaved(3);

    // dV = attn_weights^T @ dO
    const attnT = torch.runtime.transpose(sSoftmax._unwrap(), { dim0: 2, dim1: 3 });
    const dV = torch.runtime.matmul(attnT, dO);

    // dAttn = dO @ V^T
    const vT = torch.runtime.transpose(sV._unwrap(), { dim0: 2, dim1: 3 });
    const dAttn = torch.runtime.matmul(dO, vT);

    // dScores = softmax_backward(dAttn, softmax_out)
    const dAttnTimesSoftmax = torch.runtime.mul(dAttn, sSoftmax._unwrap());
    const sumDAttnSoftmax = torch.runtime.sum(dAttnTimesSoftmax, { dim: -1, keepdim: true }) as RuntimeTensor;
    const dScoresSub = torch.runtime.sub(dAttn, sumDAttnSoftmax);
    const dScores = torch.runtime.mul(sSoftmax._unwrap(), dScoresSub);

    // Scale gradients
    const scaleT = torch.runtime.full([], actualScale, sQ.device);
    const dScoresScaled = torch.runtime.mul(dScores, scaleT);

    // dQ = dScoresScaled @ K
    const dQ = torch.runtime.matmul(dScoresScaled, sK._unwrap());

    // dK = dScoresScaled^T @ Q
    const dScoresT = torch.runtime.transpose(dScoresScaled, { dim0: 2, dim1: 3 });
    const dK = torch.runtime.matmul(dScoresT, sQ._unwrap());

    return [dQ, dK, dV];
  }, tensorsToSave);
}

/**
 * Layer normalization along the last dimension.
 * layernorm(x, weight, bias, eps) = (x - mean) / sqrt(var + eps) * weight + bias
 */
export function layernormImpl(
  torch: Torchlette,
  x: Tensor, weight: Tensor, bias: Tensor, eps = 1e-5,
): Tensor {
  torch._assertUsable(x, weight, bias);

  // Forward pass: normalize along last dimension
  const xShape = x.shape;
  const rank = xShape.length;
  const dim = -1; // normalize along last dim
  const normalizedDim = dim < 0 ? dim + rank : dim;
  const lastDimSize = xShape[xShape.length - 1];

  // Save inputs for backward
  const tensorsToSave = x.requiresGrad || weight.requiresGrad || bias.requiresGrad
    ? [x, weight, bias]
    : [];

  // Use fused forward kernel on WebGPU
  if (x.device === "webgpu") {
    const numRows = xShape.slice(0, rank - 1).reduce((a, b) => a * b, 1);
    const config = { numRows, featureDim: lastDimSize, eps };
    const result = torch.runtime.fusedLayerNormForward(
      x._unwrap(), weight._unwrap(), bias._unwrap(), config,
    );

    // Backward stays decomposed (recomputes forward from saved x, weight, bias)
    return torch._wrapWithGrad(result, [x, weight, bias], (grad, getSaved) => {
      return layernormBackwardImpl(torch, grad, getSaved, normalizedDim, lastDimSize, rank, eps);
    }, tensorsToSave);
  }

  // CPU: decomposed forward
  // mean(x, dim=-1, keepdim=true)
  const meanResult = torch.runtime.mean(x._unwrap(), { dim: normalizedDim, keepdim: true });
  if (typeof meanResult === "number") {
    throw new Error("layernorm: mean with keepdim=true should return tensor");
  }

  // centered = x - mean
  const centered = torch.runtime.sub(x._unwrap(), meanResult);

  // variance = mean(centered^2, dim=-1, keepdim=true)
  const centeredSq = torch.runtime.mul(centered, centered);
  const varianceResult = torch.runtime.mean(centeredSq, { dim: normalizedDim, keepdim: true });
  if (typeof varianceResult === "number") {
    throw new Error("layernorm: variance mean with keepdim=true should return tensor");
  }

  // std = sqrt(variance + eps)
  const variancePlusEps = torch.runtime.add(varianceResult, eps);
  const std = torch.runtime.sqrt(variancePlusEps);

  // normalized = centered / std
  const normalized = torch.runtime.div(centered, std);

  // output = normalized * weight + bias
  const scaled = torch.runtime.mul(normalized, weight._unwrap());
  const result = torch.runtime.add(scaled, bias._unwrap());

  // Autograd - recompute intermediates from saved inputs for checkpointing support
  return torch._wrapWithGrad(result, [x, weight, bias], (grad, getSaved) => {
    return layernormBackwardImpl(torch, grad, getSaved, normalizedDim, lastDimSize, rank, eps);
  }, tensorsToSave);
}

/** Shared backward for LayerNorm. Uses fused gradX kernel on WebGPU. */
function layernormBackwardImpl(
  torch: Torchlette,
  grad: RuntimeTensor,
  getSaved: (i: number) => Tensor,
  normalizedDim: number,
  lastDimSize: number,
  rank: number,
  eps = 1e-5,
): RuntimeTensor[] {
  const savedX = getSaved(0);
  const savedWeight = getSaved(1);

  let gradWeight: RuntimeTensor;
  let gradBias: RuntimeTensor;
  let gradX: RuntimeTensor;

  if (savedX.device === "webgpu") {
    // Fully fused path: 2 dispatches total (gradX + gradWeightBias)
    const numRows = savedX.shape.slice(0, rank - 1).reduce((a: number, b: number) => a * b, 1);
    const config = { numRows, featureDim: lastDimSize, eps };

    gradX = torch.runtime.fusedLayerNormBackwardGradX(
      grad, savedX._unwrap(), savedWeight._unwrap(), config,
    );

    const gradWeightTensor = torch.runtime.fusedLayerNormBackwardGradWeightBias(
      grad, savedX._unwrap(), config,
    );
    gradWeight = gradWeightTensor;
    gradBias = torch.runtime.extractLnBwdGradBias(gradWeightTensor, lastDimSize);
  } else {
    // CPU decomposed path
    const recomputeMean = torch.runtime.mean(savedX._unwrap(), { dim: normalizedDim, keepdim: true });
    if (typeof recomputeMean === "number") {
      throw new Error("layernorm backward: mean should return tensor");
    }
    const recomputeCentered = torch.runtime.sub(savedX._unwrap(), recomputeMean);
    const recomputeCenteredSq = torch.runtime.mul(recomputeCentered, recomputeCentered);
    const recomputeVariance = torch.runtime.mean(recomputeCenteredSq, { dim: normalizedDim, keepdim: true });
    if (typeof recomputeVariance === "number") {
      throw new Error("layernorm backward: variance mean should return tensor");
    }
    const recomputeVarPlusEps = torch.runtime.add(recomputeVariance, eps);
    const recomputeStd = torch.runtime.sqrt(recomputeVarPlusEps);
    const recomputeNormalized = torch.runtime.div(recomputeCentered, recomputeStd);

    const sumDims = Array.from({ length: rank - 1 }, (_, i) => i);

    let gradBiasReduced = grad;
    for (let i = sumDims.length - 1; i >= 0; i--) {
      const sumResult = torch.runtime.sum(gradBiasReduced, { dim: sumDims[i], keepdim: false });
      if (typeof sumResult === "number") {
        throw new Error("layernorm backward: sum for gradBias should return tensor");
      }
      gradBiasReduced = sumResult;
    }
    gradBias = gradBiasReduced;

    let gradWeightReduced = torch.runtime.mul(grad, recomputeNormalized);
    for (let i = sumDims.length - 1; i >= 0; i--) {
      const sumResult = torch.runtime.sum(gradWeightReduced, { dim: sumDims[i], keepdim: false });
      if (typeof sumResult === "number") {
        throw new Error("layernorm backward: sum for gradWeight should return tensor");
      }
      gradWeightReduced = sumResult;
    }
    gradWeight = gradWeightReduced;

    // Decomposed gradX for CPU
    const gradNormalized = torch.runtime.mul(grad, savedWeight._unwrap());
    const gradCentered = torch.runtime.div(gradNormalized, recomputeStd);

    const gradNormCentered = torch.runtime.mul(gradNormalized, recomputeCentered);
    const sumGradNormCentered = torch.runtime.sum(gradNormCentered, { dim: normalizedDim, keepdim: true });
    if (typeof sumGradNormCentered === "number") {
      throw new Error("layernorm backward: sum should return tensor");
    }
    const varStd = torch.runtime.mul(recomputeVarPlusEps, recomputeStd);
    const gradVariance = torch.runtime.mul(
      -0.5,
      torch.runtime.div(sumGradNormCentered, varStd),
    );

    const gradCenteredFromVar = torch.runtime.div(
      torch.runtime.mul(
        torch.runtime.mul(2, gradVariance),
        recomputeCentered,
      ),
      lastDimSize,
    );

    const totalGradCentered = torch.runtime.add(gradCentered, gradCenteredFromVar);
    const sumTotalGradCentered = torch.runtime.sum(totalGradCentered, { dim: normalizedDim, keepdim: true });
    if (typeof sumTotalGradCentered === "number") {
      throw new Error("layernorm backward: sum should return tensor");
    }
    const gradMean = torch.runtime.neg(torch.runtime.div(sumTotalGradCentered, lastDimSize));
    gradX = torch.runtime.add(totalGradCentered, gradMean);
  }

  return [gradX, gradWeight, gradBias];
}
