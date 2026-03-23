/**
 * Gradient clipping utilities, matching PyTorch's torch.nn.utils.clip_grad_norm_
 * and torch.nn.utils.clip_grad_value_.
 */

import type { Tensor, Torchlette } from "../frontend/torchlette";

/**
 * Clip gradients of an iterable of parameters by total norm.
 *
 * The norm is computed over all gradients together, as if they were
 * concatenated into a single vector. Gradients are modified in-place.
 *
 * @param api - The Torchlette instance
 * @param parameters - Iterable of Tensors that may have gradients
 * @param maxNorm - Maximum allowed total norm
 * @param normType - Type of the p-norm (default: 2.0). Use Infinity for max norm.
 * @returns The total norm of the parameter gradients (before clipping)
 */
export async function clipGradNorm_(
  api: Torchlette,
  parameters: Tensor[],
  maxNorm: number,
  normType: number = 2,
): Promise<number> {
  // Collect parameters that have gradients
  const grads: Tensor[] = [];
  for (const p of parameters) {
    if (p.grad != null) {
      grads.push(p.grad);
    }
  }

  if (grads.length === 0) {
    return 0;
  }

  // Compute total norm inside tidy to dispose intermediate tensors
  // (pow, abs, sum, add) immediately. Without this, ~3 tensors per
  // parameter leak until GC — causing memory oscillation in training.
  let totalNorm: number;

  // Compute total norm lazily on GPU — no item() or materialization.
  // PyTorch-style: the entire clip operation stays in the lazy graph.
  // Track intermediates so we can dispose them after item() reads the result.
  // Without explicit disposal, ~48 frontend Tensor objects per step accumulate
  // until GC, causing progressive slowdown in long training runs.
  const intermediates: Tensor[] = [];
  let totalNormTensor: Tensor;

  if (normType === Infinity) {
    let maxTensor: Tensor | null = null;
    for (const g of grads) {
      const absMax = api.max(api.abs(g));
      intermediates.push(absMax);
      if (maxTensor === null) {
        maxTensor = absMax;
      } else {
        const gt1 = api.gt(absMax, maxTensor);
        const gt2 = api.gt(maxTensor, absMax);
        const m1 = api.mul(gt1, absMax);
        const m2 = api.mul(gt2, maxTensor);
        const added = api.add(m1, m2);
        const newMax = api.max(added);
        intermediates.push(gt1, gt2, m1, m2, added, newMax);
        maxTensor = newMax;
      }
    }
    totalNormTensor = maxTensor!;
  } else if (normType === 2) {
    let totalSumSq: Tensor | null = null;
    for (const g of grads) {
      const sq = api.pow(g, 2);
      const sumSq = api.sum(sq);
      intermediates.push(sq, sumSq);
      if (totalSumSq === null) {
        totalSumSq = sumSq;
      } else {
        const added = api.add(totalSumSq, sumSq);
        intermediates.push(added);
        totalSumSq = added;
      }
    }
    totalNormTensor = api.sqrt(totalSumSq!);
    intermediates.push(totalNormTensor);
  } else {
    let totalSumP: Tensor | null = null;
    for (const g of grads) {
      const absG = api.abs(g);
      const powG = api.pow(absG, normType);
      const sumP = api.sum(powG);
      intermediates.push(absG, powG, sumP);
      if (totalSumP === null) {
        totalSumP = sumP;
      } else {
        const added = api.add(totalSumP, sumP);
        intermediates.push(added);
        totalSumP = added;
      }
    }
    totalNormTensor = api.pow(totalSumP!, 1 / normType);
    intermediates.push(totalNormTensor);
  }

  // Read the norm and clip coefficient — one materialization point.
  // The norm computation was lazy; item() forces it.
  totalNorm = await totalNormTensor.item();

  // Dispose all intermediates now that the scalar value is read.
  for (const t of intermediates) {
    if (!t.disposed) t.dispose();
  }
  const clipCoef = maxNorm / (totalNorm + 1e-6);
  if (clipCoef < 1) {
    for (const g of grads) {
      api.mul_(g, clipCoef);
    }
  }

  return totalNorm;
}

/**
 * Clip gradients of an iterable of parameters by value.
 *
 * Gradients are modified in-place by clamping all values to
 * [-clipValue, clipValue].
 *
 * @param api - The Torchlette instance
 * @param parameters - Iterable of Tensors that may have gradients
 * @param clipValue - Maximum allowed absolute value for each gradient element
 */
export function clipGradValue_(
  api: Torchlette,
  parameters: Tensor[],
  clipValue: number,
): void {
  for (const p of parameters) {
    if (p.grad != null) {
      api.copy_(p.grad, api.clamp(p.grad, -clipValue, clipValue));
    }
  }
}
