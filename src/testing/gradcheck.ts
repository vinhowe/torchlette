/**
 * Numerical gradient checking for autograd verification.
 *
 * Compares analytically computed gradients (from backward()) against
 * numerically estimated gradients via central finite differences:
 *   numerical_grad[i] ≈ (f(x + h*e_i) - f(x - h*e_i)) / (2h)
 *
 * Usage:
 *   const result = await gradcheck(api, (x) => api.sum(api.mul(x, x)), inputs);
 *   // result.pass === true if all gradients match within tolerance
 *
 * Runs on CPU with f32. Use small tensors (e.g. [3,4]) — cost is O(n) forward
 * passes per input element.
 */

import type { Tensor, Torchlette } from "../frontend/torchlette";

export interface GradcheckOptions {
  /** Perturbation size for finite differences (default: 1e-3) */
  h?: number;
  /** Absolute tolerance for gradient comparison (default: 1e-2) */
  atol?: number;
  /** Relative tolerance for gradient comparison (default: 1e-2) */
  rtol?: number;
}

export interface GradcheckResult {
  /** Whether all gradients matched within tolerance */
  pass: boolean;
  /** Per-input results */
  inputs: Array<{
    /** Shape of the input */
    shape: number[];
    /** Maximum absolute error across all elements */
    maxAbsErr: number;
    /** Maximum relative error across all elements */
    maxRelErr: number;
    /** Index of the worst element */
    worstIndex: number;
    /** Analytical gradient value at worst index */
    analyticalAtWorst: number;
    /** Numerical gradient value at worst index */
    numericalAtWorst: number;
  }>;
  /** Human-readable summary */
  message: string;
}

/**
 * Check gradients of a scalar-valued function via finite differences.
 *
 * @param api - Torchlette instance (should use CPU device for determinism)
 * @param fn - Function that takes input tensors and returns a SCALAR tensor
 * @param inputs - Input tensors (will be cloned; originals not modified)
 * @param options - Tolerance and perturbation settings
 */
export async function gradcheck(
  api: Torchlette,
  fn: (...inputs: Tensor[]) => Tensor,
  inputs: Tensor[],
  options?: GradcheckOptions,
): Promise<GradcheckResult> {
  const h = options?.h ?? 1e-3;
  const atol = options?.atol ?? 1e-2;
  const rtol = options?.rtol ?? 1e-2;

  // Step 1: Get analytical gradients via backward.
  // Collect input data from originals (before any grad tracking).
  const inputData: number[][] = [];
  const inputShapes: number[][] = [];
  for (const inp of inputs) {
    inputData.push(Array.from(await inp.cpu()));
    inputShapes.push(inp.shape);
  }

  // Run forward + backward to get analytical gradients
  const analyticalGrads: number[][] = [];
  {
    const gradInputs = inputData.map((data, i) =>
      api.tensorFromArray(data, inputShapes[i], { requiresGrad: true }),
    );
    const output = fn(...gradInputs);
    const outputData = await output.cpu();
    if (outputData.length !== 1) {
      throw new Error(
        `gradcheck requires scalar output, got ${outputData.length} elements. ` +
          `Wrap with api.sum() if needed.`,
      );
    }
    await output.backward();
    for (const inp of gradInputs) {
      const grad = inp.grad;
      if (!grad) {
        analyticalGrads.push(
          new Array(inputData[analyticalGrads.length].length).fill(0),
        );
      } else {
        analyticalGrads.push(Array.from(await grad.cpu()));
      }
    }
    // Dispose to release engine state
    output.dispose();
    for (const inp of gradInputs) inp.dispose();
  }

  // Step 2: Numerical gradients via central finite differences.
  // Run inside noGrad to avoid autograd overhead and engine lock conflicts.
  const inputResults: GradcheckResult["inputs"] = [];
  let allPass = true;

  for (let inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
    const data = Array.from(inputData[inputIdx]);
    const shape = inputShapes[inputIdx];
    const analyticalGrad = analyticalGrads[inputIdx];
    const numericalGrad = new Array<number>(data.length);

    for (let elemIdx = 0; elemIdx < data.length; elemIdx++) {
      const original = data[elemIdx];

      // f(x + h) — no grad tracking needed
      data[elemIdx] = original + h;
      const fPlus = await api
        .noGrad(() => {
          const plusInputs = inputData.map((d, i) =>
            i === inputIdx
              ? api.tensorFromArray(data, shape)
              : api.tensorFromArray(Array.from(d), inputShapes[i]),
          );
          return fn(...plusInputs);
        })
        .item();

      // f(x - h)
      data[elemIdx] = original - h;
      const fMinus = await api
        .noGrad(() => {
          const minusInputs = inputData.map((d, i) =>
            i === inputIdx
              ? api.tensorFromArray(data, shape)
              : api.tensorFromArray(Array.from(d), inputShapes[i]),
          );
          return fn(...minusInputs);
        })
        .item();

      numericalGrad[elemIdx] = (fPlus - fMinus) / (2 * h);

      // Restore
      data[elemIdx] = original;
    }

    // Compare
    let maxAbsErr = 0;
    let maxRelErr = 0;
    let worstIndex = 0;

    for (let j = 0; j < data.length; j++) {
      const a = analyticalGrad[j];
      const n = numericalGrad[j];
      const absErr = Math.abs(a - n);
      const denom = Math.max(Math.abs(a), Math.abs(n), 1e-8);
      const relErr = absErr / denom;

      if (absErr > maxAbsErr) {
        maxAbsErr = absErr;
        worstIndex = j;
      }
      if (relErr > maxRelErr) {
        maxRelErr = relErr;
      }

      // Tolerance check: pass if EITHER absolute OR relative error is small
      if (absErr > atol && relErr > rtol) {
        allPass = false;
      }
    }

    inputResults.push({
      shape,
      maxAbsErr,
      maxRelErr,
      worstIndex,
      analyticalAtWorst: analyticalGrad[worstIndex],
      numericalAtWorst: numericalGrad[worstIndex],
    });
  }

  const message = inputResults
    .map(
      (r, i) =>
        `input[${i}] ${JSON.stringify(r.shape)}: maxAbsErr=${r.maxAbsErr.toExponential(3)} maxRelErr=${(r.maxRelErr * 100).toFixed(2)}%` +
        (r.maxAbsErr > atol && r.maxRelErr > rtol
          ? ` FAIL at [${r.worstIndex}]: analytical=${r.analyticalAtWorst.toFixed(6)} numerical=${r.numericalAtWorst.toFixed(6)}`
          : ` OK`),
    )
    .join("\n");

  return { pass: allPass, inputs: inputResults, message };
}
