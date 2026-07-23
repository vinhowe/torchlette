import type { AttnModifierSpec } from "../backend/types";
import { sizeOf } from "../core/shape";
import {
  type CompositeDef,
  interpretComposition,
  LAYERNORM_DEF,
  RMSNORM_DEF,
  SOFTMAX_BWD_LEMMA,
  vjpComposition,
} from "../ops/semantic";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import { applyAutocastImpl, autocastCastImpl } from "./autocast";
import type { Tensor } from "./tensor";
import type { Torchlette } from "./torchlette";

/**
 * COMPOSITE-CLOSURE C3 — the CPU / non-fused composite backwards DERIVE through
 * the reverse-mode reference (`vjpComposition`, F1) instead of a hand closure.
 * The C2 soak flag has fired (soak → default → removed): the derived path is now
 * the sole CPU realization for rmsnorm / layernorm, and the hand
 * `rmsnormBackwardImpl` / `layernormBackwardImpl` CPU branch are DELETED. The
 * derived VJP is pinned by torch
 * (semantic-composite-backward.spec) and the fused GPU kernels are asserted
 * against the SAME reference (composite-fused-vs-derived.spec).
 *
 * The softmax backward is the DECLARED SIMPLIFICATION LEMMA `SOFTMAX_BWD_LEMMA`
 * (T1, RESOLVED-BY-LEMMA 2026-07-23): the collapsed closed form the C1 cost probe
 * kept (13 nodes vs the reverse-mode graph's 22) STOPPED being trusted hand code
 * and became data in the semantic stratum, realized here byte-identically via the
 * memoized composition interpreter. Its proof obligation — equality to the honest
 * `vjpComposition(SOFTMAX_DEF)` — is the permanent machine-checked witness
 * (semantic-composite-backward.spec, L-COMP).
 */

/** Realize a composite's derived VJP over `rt` and return grads in role order. */
function derivedCompositeGrads(
  torch: Torchlette,
  def: CompositeDef,
  grad: RuntimeTensor,
  dim: number,
  roleInputs: Readonly<Record<string, number | RuntimeTensor>>,
  outRoles: readonly string[],
): RuntimeTensor[] {
  const grads = vjpComposition(def, torch.runtime, dim, roleInputs, grad);
  return outRoles.map((r) => {
    const g = grads[r];
    if (g === undefined)
      throw new Error(`derivedCompositeGrads: ${def.name} missing role '${r}'`);
    return g;
  });
}

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
    throw new Error(
      `softmax: dim ${dim} out of range for tensor of rank ${rank}`,
    );
  }

  // Numerical stability: subtract max
  const maxResult = torch.runtime.max(castA._unwrap(), {
    dim: normalizedDim,
    keepdim: true,
  });
  if (typeof maxResult === "number") {
    throw new Error("softmax: max with keepdim=true should return tensor");
  }
  const shifted = torch.runtime.sub(castA._unwrap(), maxResult);
  const exps = torch.runtime.exp(shifted);
  const sumResult = torch.runtime.sum(exps, {
    dim: normalizedDim,
    keepdim: true,
  });
  if (typeof sumResult === "number") {
    throw new Error("softmax: sum with keepdim=true should return tensor");
  }
  const result = torch.runtime.div(exps, sumResult);

  const tensorsToSave = a.requiresGrad ? [castA] : [];
  // Softmax backward = the DECLARED simplification lemma (SOFTMAX_BWD_LEMMA):
  //   grad_input = y ⊙ (g − Σ(y ⊙ g)),  y = softmax(savedA) (recomputed for
  //   checkpointing). The memoized interpreter folds the shared forward `y` once,
  //   so this emits max, sub, exp, sum, div, mul, sum, sub, mul — byte-identical
  //   to the deleted hand closure. Proof obligation: == vjpComposition(SOFTMAX_DEF)
  //   (semantic-composite-backward.spec, L-COMP).
  return torch._wrapWithGrad(
    result,
    [a],
    (grad, getSaved) => [
      interpretComposition(SOFTMAX_BWD_LEMMA, torch.runtime, normalizedDim, {
        x: getSaved(0)._unwrap(),
        g: grad,
      }),
    ],
    tensorsToSave,
  );
}

/**
 * Fused cross-entropy forward + backward for WebGPU.
 * logits [B, V] + targets [B] → per-sample loss [B]
 * Backward: fused kernel → grad_logits [B, V]
 */
export function crossEntropyFusedImpl(
  torch: Torchlette,
  logits: Tensor,
  targets: Tensor,
  ignoreIndex?: number,
): Tensor {
  torch._assertUsable(logits, targets);

  // Upcast f16 logits to f32 for numerical stability (same as softmax)
  let castLogits = logits;
  if (logits.dtype === "f16") {
    castLogits = autocastCastImpl(torch, logits, "f32");
  }

  // Fused CE kernels read targets as native i32. If the caller passed f32
  // targets (e.g. legacy code that predates the dtype creation option),
  // insert a lazy cast so the kernel sees i32.
  const i32Targets =
    targets.dtype === "i32" ? targets : torch.toDtype(targets, "i32");

  const B = castLogits.shape[0];
  const V = castLogits.shape[1];
  const config = {
    batchSize: B,
    vocabSize: V,
    ignoreIndex: ignoreIndex ?? -100,
  };

  const result = torch.runtime.fusedCrossEntropyForward(
    castLogits._unwrap(),
    i32Targets._unwrap(),
    config,
  );

  // Save logits for backward (needs recomputation of softmax).
  // Targets are captured via closure — they don't require grad and aren't
  // modified between forward and backward, so keep() is unnecessary.
  const tensorsToSave = logits.requiresGrad ? [castLogits] : [];
  const targetsInner = i32Targets._unwrap();
  return torch._wrapWithGrad(
    result,
    [logits],
    (grad, getSaved) => {
      const savedLogits = getSaved(0);
      const gradLogits = torch.runtime.fusedCrossEntropyBackward(
        savedLogits._unwrap(),
        targetsInner,
        grad,
        config,
      );
      return [gradLogits];
    },
    tensorsToSave,
  );
}

/**
 * RMS normalization along the last dimension.
 * rmsnorm(x, weight, eps) = x * rsqrt(mean(x², dim=-1, keepdim=true) + eps) * weight
 */
export function rmsnormImpl(
  torch: Torchlette,
  x: Tensor,
  weight: Tensor,
  eps = 1e-6,
): Tensor {
  torch._assertUsable(x, weight);

  const xShape = x.shape;
  const rank = xShape.length;
  const normalizedDim = rank - 1;
  const lastDimSize = xShape[xShape.length - 1];

  const tensorsToSave =
    x.requiresGrad || weight.requiresGrad ? [x, weight] : [];

  // Use fused forward + backward kernels on WebGPU
  if (x.device === "webgpu") {
    const numRows = sizeOf(xShape.slice(0, rank - 1));
    const config = { numRows, featureDim: lastDimSize, eps };
    const result = torch.runtime.fusedRMSNormForward(
      x._unwrap(),
      weight._unwrap(),
      config,
    );

    return torch._wrapWithGrad(
      result,
      [x, weight],
      (grad, getSaved) => {
        const savedX = getSaved(0);
        const savedWeight = getSaved(1);
        const gradX = torch.runtime.fusedRMSNormBackwardGradX(
          grad,
          savedX._unwrap(),
          savedWeight._unwrap(),
          config,
        );
        const gradWeight = torch.runtime.fusedRMSNormBackwardGradWeight(
          grad,
          savedX._unwrap(),
          savedWeight._unwrap(),
          config,
        );
        return [gradX, gradWeight];
      },
      tensorsToSave,
    );
  }

  // CPU: decomposed forward
  // x_sq = x * x
  const xSq = torch.runtime.mul(x._unwrap(), x._unwrap());
  // mean_sq = mean(x², dim=-1, keepdim=true)
  const meanSq = torch.runtime.mean(xSq, { dim: normalizedDim, keepdim: true });
  if (typeof meanSq === "number") {
    throw new Error("rmsnorm: mean with keepdim=true should return tensor");
  }
  // inv_rms = rsqrt(mean_sq + eps)
  const meanSqPlusEps = torch.runtime.add(meanSq, eps);
  const invRms = torch.runtime.rsqrt(meanSqPlusEps);
  // normalized = x * inv_rms
  const normalized = torch.runtime.mul(x._unwrap(), invRms);
  // output = normalized * weight
  const result = torch.runtime.mul(normalized, weight._unwrap());

  // CPU backward DERIVES (COMPOSITE-CLOSURE C3): one reverse pass over
  // RMSNORM_DEF, not a hand closed form. The hand `rmsnormBackwardImpl`
  // (`inv_rms·(g·w − norm·mean(g·w·norm))`, `dW=Σ(g·norm)`) is DELETED — the
  // derived VJP is pinned by torch (semantic-composite-backward.spec) and the
  // fused GPU kernel is asserted against it (composite-fused-vs-derived.spec).
  return torch._wrapWithGrad(
    result,
    [x, weight],
    (grad, getSaved) =>
      derivedCompositeGrads(
        torch,
        RMSNORM_DEF,
        grad,
        normalizedDim,
        {
          x: getSaved(0)._unwrap(),
          w: getSaved(1)._unwrap(),
          eps,
        },
        ["x", "w"],
      ),
    tensorsToSave,
  );
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
  q: Tensor,
  k: Tensor,
  v: Tensor,
  scale?: number,
  isCausal = false,
  modifier?: AttnModifierSpec,
): Tensor {
  torch._assertUsable(q, k, v);
  const [batch, heads, seq, hd] = q.shape;
  const actualScale = scale ?? 1.0 / Math.sqrt(hd);
  // Omit the modifier FIELD entirely when unused: the payload participates in
  // CSE structural keys and plan fingerprints, so existing (null-modifier)
  // payload hashes stay byte-stable.
  const config = {
    batchSize: batch,
    numHeads: heads,
    seqLen: seq,
    headDim: hd,
    scale: actualScale,
    isCausal,
    ...(modifier ? { modifier } : {}),
  };

  // Inference-first (#64): scoreMod backward is not implemented on EITHER
  // path — the fused kernels lack the paired "attn_dscore" chain factor and
  // the CPU decomposed backward assumes plain softmax attention (it would be
  // SILENTLY wrong, the worst failure mode). Fail BEFORE creating any lazy
  // node (a post-hoc throw leaves a poisoned node that still executes at the
  // next force/markStep), and at FORWARD time, when the user can still
  // restructure, not deep in backward().
  if (
    modifier?.scoreMod &&
    torch.isGradEnabled() &&
    (q.requiresGrad || k.requiresGrad || v.requiresGrad)
  ) {
    throw new Error(
      `scaledDotProductAttention: backward with scoreMod ` +
        `'${modifier.scoreMod.kind}' is not implemented (inference-first); ` +
        `wrap in noGrad() or drop the scoreMod`,
    );
  }

  if (q.device === "webgpu") {
    // Fused FlashAttention path
    const fwdResult = torch.runtime.fusedAttentionForward(
      q._unwrap(),
      k._unwrap(),
      v._unwrap(),
      config,
    );

    const needsGrad =
      torch.isGradEnabled() &&
      (q.requiresGrad || k.requiresGrad || v.requiresGrad);

    // Only extract logsumexp and save tensors when backward is needed.
    // In noGrad/eval mode, extracting logsumexp creates dangling side-output
    // references that become stale across generation steps.
    let tensorsToSave: Tensor[] = [];
    if (needsGrad) {
      const logsumexp = torch.runtime.extractAttentionLogsumexp(
        fwdResult,
        config,
      );

      // Create reshape views with independent RuntimeTensors for saved tensors.
      // Without this, wrap(fwdResult) shares the same RuntimeTensor as the
      // wrapWithGrad output. When checkpoint tidy disposes these wrappers, it
      // calls dispose() on the shared RuntimeTensor BEFORE materialization,
      // setting _disposed=true. Reshape-to-same-shape creates a new
      // RuntimeTensor that can be safely disposed without poisoning the original.
      const logsumexpRef = torch.runtime.reshape(logsumexp, [
        batch,
        heads,
        seq,
      ]);
      const logsumexpTensor = torch._wrap(logsumexpRef);
      const outputRef = torch.runtime.reshape(fwdResult, [
        batch,
        heads,
        seq,
        hd,
      ]);
      const outputTensor = torch._wrap(outputRef);
      tensorsToSave = [q, k, v, logsumexpTensor, outputTensor];
    }

    return torch._wrapWithGrad(
      fwdResult,
      [q, k, v],
      (dO, getSaved) => {
        const sQ = getSaved(0);
        const sK = getSaved(1);
        const sV = getSaved(2);
        const sL = getSaved(3);
        const sO = getSaved(4);

        // O is the 6th input to backward for D precomputation
        const bwdDQ = torch.runtime.fusedAttentionBackward(
          sQ._unwrap(),
          sK._unwrap(),
          sV._unwrap(),
          sL._unwrap(),
          dO,
          sO._unwrap(),
          config,
        );
        const dK = torch.runtime.extractAttentionDK(bwdDQ, config);
        const dV = torch.runtime.extractAttentionDV(bwdDQ, config);
        return [bwdDQ, dK, dV]; // dQ, dK, dV for inputs [q, k, v]
      },
      tensorsToSave,
    );
  }

  // CPU fallback: decomposed matmul + softmax + matmul.
  // Modifiers are interpreted here as tensor ops — the cross-path reference
  // the fused kernels are diffed against. Order matches the kernel seams:
  // scale → scoreMod → masks (additive −1e9; softcapped scores are bounded
  // by ±cap, so −1e9 dominates).
  const cpuMaskMods = modifier?.maskMods ?? [];
  const cpuCausal = isCausal || cpuMaskMods.some((m) => m.kind === "causal");
  const kT = torch.runtime.transpose(k._unwrap(), { dim0: 2, dim1: 3 });
  const scores = torch.runtime.matmul(q._unwrap(), kT);
  const scaleTensor = torch.runtime.full([], actualScale, q.device);
  const scaledScores = torch.runtime.mul(scores, scaleTensor);

  let finalScores: RuntimeTensor = scaledScores;
  if (modifier?.scoreMod) {
    if (modifier.scoreMod.kind !== "softcap") {
      throw new Error(
        `scaledDotProductAttention (cpu): scoreMod '${(modifier.scoreMod as { kind: string }).kind}' not implemented`,
      );
    }
    // softcap: cap · tanh(s / cap)
    const capT = torch.runtime.full([], modifier.scoreMod.cap, q.device);
    finalScores = torch.runtime.mul(
      torch.runtime.tanh(torch.runtime.div(finalScores, capT)),
      capT,
    );
  }
  if (cpuCausal) {
    // Causal mask: -1e9 where j > i, 0 elsewhere
    const negInf = torch.runtime.full([1, 1, seq, seq], -1e9, q.device);
    const mask = torch.runtime.triu(negInf, 1);
    finalScores = torch.runtime.add(finalScores, mask);
  }
  for (const m of cpuMaskMods) {
    if (m.kind === "causal") continue;
    if (m.kind === "slidingWindow") {
      // Window recency bound: -1e9 where j <= i - window (tril at -window)
      const negInf = torch.runtime.full([1, 1, seq, seq], -1e9, q.device);
      const mask = torch.runtime.tril(negInf, -m.window);
      finalScores = torch.runtime.add(finalScores, mask);
    } else {
      throw new Error(
        `scaledDotProductAttention (cpu): maskMod '${(m as { kind: string }).kind}' not implemented`,
      );
    }
  }

  // Softmax along last dim (using the public softmax which handles autograd)
  const softmaxResult = softmaxImpl(torch, torch._wrap(finalScores), -1);
  const output = torch.runtime.matmul(softmaxResult._unwrap(), v._unwrap());

  // Wrap with autograd
  const tensorsToSave =
    q.requiresGrad || k.requiresGrad || v.requiresGrad
      ? [q, k, v, softmaxResult]
      : [];

  return torch._wrapWithGrad(
    output,
    [q, k, v],
    (dO, getSaved) => {
      const sQ = getSaved(0);
      const sK = getSaved(1);
      const sV = getSaved(2);
      const sSoftmax = getSaved(3);

      // dV = attn_weights^T @ dO
      const attnT = torch.runtime.transpose(sSoftmax._unwrap(), {
        dim0: 2,
        dim1: 3,
      });
      const dV = torch.runtime.matmul(attnT, dO);

      // dAttn = dO @ V^T
      const vT = torch.runtime.transpose(sV._unwrap(), { dim0: 2, dim1: 3 });
      const dAttn = torch.runtime.matmul(dO, vT);

      // dScores = softmax_backward(dAttn, softmax_out)
      const dAttnTimesSoftmax = torch.runtime.mul(dAttn, sSoftmax._unwrap());
      const sumDAttnSoftmax = torch.runtime.sum(dAttnTimesSoftmax, {
        dim: -1,
        keepdim: true,
      }) as RuntimeTensor;
      const dScoresSub = torch.runtime.sub(dAttn, sumDAttnSoftmax);
      const dScores = torch.runtime.mul(sSoftmax._unwrap(), dScoresSub);

      // Scale gradients
      const scaleT = torch.runtime.full([], actualScale, sQ.device);
      const dScoresScaled = torch.runtime.mul(dScores, scaleT);

      // dQ = dScoresScaled @ K
      const dQ = torch.runtime.matmul(dScoresScaled, sK._unwrap());

      // dK = dScoresScaled^T @ Q
      const dScoresT = torch.runtime.transpose(dScoresScaled, {
        dim0: 2,
        dim1: 3,
      });
      const dK = torch.runtime.matmul(dScoresT, sQ._unwrap());

      return [dQ, dK, dV];
    },
    tensorsToSave,
  );
}

/**
 * Layer normalization along the last dimension.
 * layernorm(x, weight, bias, eps) = (x - mean) / sqrt(var + eps) * weight + bias
 */
export function layernormImpl(
  torch: Torchlette,
  x: Tensor,
  weight: Tensor,
  bias: Tensor,
  eps = 1e-5,
): Tensor {
  torch._assertUsable(x, weight, bias);

  // Forward pass: normalize along last dimension
  const xShape = x.shape;
  const rank = xShape.length;
  const dim = -1; // normalize along last dim
  const normalizedDim = dim < 0 ? dim + rank : dim;
  const lastDimSize = xShape[xShape.length - 1];

  // Save inputs for backward
  const tensorsToSave =
    x.requiresGrad || weight.requiresGrad || bias.requiresGrad
      ? [x, weight, bias]
      : [];

  // Use fused forward kernel on WebGPU
  if (x.device === "webgpu") {
    const numRows = sizeOf(xShape.slice(0, rank - 1));
    const config = { numRows, featureDim: lastDimSize, eps };
    const result = torch.runtime.fusedLayerNormForward(
      x._unwrap(),
      weight._unwrap(),
      bias._unwrap(),
      config,
    );

    // Backward stays decomposed (recomputes forward from saved x, weight, bias)
    return torch._wrapWithGrad(
      result,
      [x, weight, bias],
      (grad, getSaved) => {
        return layernormBackwardImpl(
          torch,
          grad,
          getSaved,
          normalizedDim,
          lastDimSize,
          rank,
          eps,
        );
      },
      tensorsToSave,
    );
  }

  // CPU: decomposed forward
  // mean(x, dim=-1, keepdim=true)
  const meanResult = torch.runtime.mean(x._unwrap(), {
    dim: normalizedDim,
    keepdim: true,
  });
  if (typeof meanResult === "number") {
    throw new Error("layernorm: mean with keepdim=true should return tensor");
  }

  // centered = x - mean
  const centered = torch.runtime.sub(x._unwrap(), meanResult);

  // variance = mean(centered^2, dim=-1, keepdim=true)
  const centeredSq = torch.runtime.mul(centered, centered);
  const varianceResult = torch.runtime.mean(centeredSq, {
    dim: normalizedDim,
    keepdim: true,
  });
  if (typeof varianceResult === "number") {
    throw new Error(
      "layernorm: variance mean with keepdim=true should return tensor",
    );
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
  return torch._wrapWithGrad(
    result,
    [x, weight, bias],
    (grad, getSaved) => {
      return layernormBackwardImpl(
        torch,
        grad,
        getSaved,
        normalizedDim,
        lastDimSize,
        rank,
        eps,
      );
    },
    tensorsToSave,
  );
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
    const numRows = sizeOf(savedX.shape.slice(0, rank - 1));
    const config = { numRows, featureDim: lastDimSize, eps };

    gradX = torch.runtime.fusedLayerNormBackwardGradX(
      grad,
      savedX._unwrap(),
      savedWeight._unwrap(),
      config,
    );

    const gradWeightTensor = torch.runtime.fusedLayerNormBackwardGradWeightBias(
      grad,
      savedX._unwrap(),
      config,
    );
    gradWeight = gradWeightTensor;
    gradBias = torch.runtime.extractLnBwdGradBias(
      gradWeightTensor,
      lastDimSize,
    );
  } else {
    // CPU backward DERIVES (COMPOSITE-CLOSURE C3): the layernorm VJP is one
    // reverse pass over LAYERNORM_DEF. The hand naive `gradVar`/`gradMean`
    // expansion is DELETED — the derived VJP is pinned by torch
    // (semantic-composite-backward.spec) and the fused GPU kernel is asserted
    // against the SAME derived reference (composite-fused-vs-derived.spec).
    [gradX, gradWeight, gradBias] = derivedCompositeGrads(
      torch,
      LAYERNORM_DEF,
      grad,
      normalizedDim,
      {
        x: savedX._unwrap(),
        w: savedWeight._unwrap(),
        b: getSaved(2)._unwrap(),
        eps,
      },
      ["x", "w", "b"],
    ) as [RuntimeTensor, RuntimeTensor, RuntimeTensor];
  }

  return [gradX, gradWeight, gradBias];
}
