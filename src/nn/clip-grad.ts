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
 * Fully lazy on GPU — no CPU readback or GPU fence. The norm computation,
 * clip coefficient, and conditional scaling all stay in the lazy graph.
 * This avoids an ~800ms onSubmittedWorkDone stall in Full FT training.
 *
 * @param api - The Torchlette instance
 * @param parameters - Iterable of Tensors that may have gradients
 * @param maxNorm - Maximum allowed total norm
 * @param normType - Type of the p-norm (default: 2.0). Use Infinity for max norm.
 */
export function clipGradNorm_(
  api: Torchlette,
  parameters: Tensor[],
  maxNorm: number,
  normType: number = 2,
): void {
  // Collect parameters that have gradients
  const grads: Tensor[] = [];
  for (const p of parameters) {
    if (p.grad != null) {
      grads.push(p.grad);
    }
  }

  if (grads.length === 0) {
    return;
  }

  // Build norm + clip coefficient + scaled gradients all as lazy GPU ops.
  // tidy() disposes all intermediates (pow, sum, add, sqrt, div, clamp).
  api.tidy(() => {
    let totalNormTensor: Tensor;

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
      totalNormTensor = maxTensor!;
    } else if (normType === 2) {
      let totalSumSq: Tensor | null = null;
      for (const g of grads) {
        const sq = api.pow(g, 2);
        const sumSq = api.sum(sq);
        totalSumSq = totalSumSq === null ? sumSq : api.add(totalSumSq, sumSq);
      }
      totalNormTensor = api.sqrt(totalSumSq!);
    } else {
      let totalSumP: Tensor | null = null;
      for (const g of grads) {
        const absG = api.abs(g);
        const powG = api.pow(absG, normType);
        const sumP = api.sum(powG);
        totalSumP = totalSumP === null ? sumP : api.add(totalSumP, sumP);
      }
      totalNormTensor = api.pow(totalSumP!, 1 / normType);
    }

    // clipCoef = clamp(maxNorm / (totalNorm + eps), max=1.0)
    // When clipCoef = 1.0 (norm within budget), multiplication is a no-op.
    // clipCoef = min(maxNorm / (norm + eps), 1.0) — cap at 1.0 so we never
    // scale gradients UP. Implemented with fusible ops (clamp isn't in the
    // tile-IR op registry): rawCoef * le(rawCoef, one) + gt(rawCoef, one).
    const rawCoef = api.div(maxNorm, api.add(totalNormTensor, 1e-6));
    const one = api.full([], 1.0);
    const clipCoef = api.add(
      api.mul(rawCoef, api.le(rawCoef, one)),
      api.gt(rawCoef, one),
    );

    // Scale all gradients by clipCoef (always — clipCoef=1 is identity)
    for (const g of grads) {
      api.copy_(g, api.mul(g, clipCoef));
    }
  });
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
