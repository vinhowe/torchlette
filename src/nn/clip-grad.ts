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

  if (normType === Infinity) {
    let maxVal = 0;
    for (const g of grads) {
      const absMax = await api
        .tidy(() => {
          const result = api.max(api.abs(g));
          api.keep(result);
          return result;
        })
        .item();
      if (absMax > maxVal) maxVal = absMax;
    }
    totalNorm = maxVal;
  } else if (normType === 2) {
    const totalSumSqVal = await api
      .tidy(() => {
        let totalSumSq: Tensor | null = null;
        for (const g of grads) {
          const sq = api.pow(g, 2);
          const sumSq = api.sum(sq);
          totalSumSq = totalSumSq === null ? sumSq : api.add(totalSumSq, sumSq);
        }
        api.keep(totalSumSq!);
        return totalSumSq!;
      })
      .item();
    totalNorm = Math.sqrt(totalSumSqVal);
  } else {
    const totalSumPVal = await api
      .tidy(() => {
        let totalSumP: Tensor | null = null;
        for (const g of grads) {
          const absG = api.abs(g);
          const powG = api.pow(absG, normType);
          const sumP = api.sum(powG);
          totalSumP = totalSumP === null ? sumP : api.add(totalSumP, sumP);
        }
        api.keep(totalSumP!);
        return totalSumP!;
      })
      .item();
    // biome-ignore lint/style/useExponentiationOperator: Math.pow avoids `await x ** y` SyntaxError after bundler inlining
    totalNorm = Math.pow(totalSumPVal, 1 / normType);
  }

  // Clip: scale gradients if total norm exceeds maxNorm
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
