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

  // Build the norm tensor inside tidy — all intermediates (pow, sum, add,
  // sqrt) are disposed automatically when tidy exits. Only the kept result
  // survives. Without tidy, ~48 frontend Tensor objects per step accumulate
  // until GC, causing progressive slowdown in long training runs.
  const totalNormTensor = api.tidy(() => {
    let result: Tensor;

    if (normType === Infinity) {
      let maxTensor: Tensor | null = null;
      for (const g of grads) {
        const absMax = api.max(api.abs(g));
        maxTensor =
          maxTensor === null
            ? absMax
            : api.max(
                api.add(
                  api.mul(api.gt(absMax, maxTensor), absMax),
                  api.mul(api.gt(maxTensor, absMax), maxTensor),
                ),
              );
      }
      result = maxTensor!;
    } else if (normType === 2) {
      let totalSumSq: Tensor | null = null;
      for (const g of grads) {
        const sq = api.pow(g, 2);
        const sumSq = api.sum(sq);
        totalSumSq = totalSumSq === null ? sumSq : api.add(totalSumSq, sumSq);
      }
      result = api.sqrt(totalSumSq!);
    } else {
      let totalSumP: Tensor | null = null;
      for (const g of grads) {
        const absG = api.abs(g);
        const powG = api.pow(absG, normType);
        const sumP = api.sum(powG);
        totalSumP = totalSumP === null ? sumP : api.add(totalSumP, sumP);
      }
      result = api.pow(totalSumP!, 1 / normType);
    }

    api.keep(result);
    return result;
  });

  // One materialization point — forces the lazy norm computation.
  const totalNorm = await totalNormTensor.item();
  totalNormTensor.dispose();

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
