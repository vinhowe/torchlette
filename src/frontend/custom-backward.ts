/**
 * Custom backward functions for non-elementwise ops.
 * Extracted from torchlette.ts to reduce the monolithic frontend.
 */

import type { RuntimeEngine } from "../runtime/engine";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import type { GradFn } from "./types";

/** Context available to custom backward functions. */
export interface BackwardContext {
  rt: RuntimeEngine;
  sumToShape: (grad: RuntimeTensor, shape: number[]) => RuntimeTensor;
}

/**
 * Matmul backward: dA = dC @ B^T, dB = A^T @ dC
 */
export function matmulBackward(
  ctx: BackwardContext,
  aShape: number[],
  bShape: number[],
): GradFn {
  return (grad, getSaved) => {
    if (aShape.length < 2 || bShape.length < 2) {
      throw new Error("matmul backward requires rank >= 2");
    }
    const savedA = getSaved(0);
    const savedB = getSaved(1);
    const savedAShape = savedA.shape;
    const savedBShape = savedB.shape;
    const savedAInner = savedA._unwrap();
    const savedBInner = savedB._unwrap();
    const aT = ctx.rt.transpose(savedAInner, {
      dim0: savedAShape.length - 2,
      dim1: savedAShape.length - 1,
    });
    const bT = ctx.rt.transpose(savedBInner, {
      dim0: savedBShape.length - 2,
      dim1: savedBShape.length - 1,
    });
    const gradA = ctx.rt.matmul(grad, bT);
    const gradB = ctx.rt.matmul(aT, grad);
    return [ctx.sumToShape(gradA, aShape), ctx.sumToShape(gradB, bShape)];
  };
}

/**
 * Linear backward: dX = dY @ W, dW = dY^T @ X, dBias = sum(dY)
 * More efficient than generic matmul backward — computes dW directly
 * in weight's shape without an extra transpose.
 */
export function linearBackward(
  ctx: BackwardContext,
  inputShape: number[],
  weightShape: number[],
  needsInputGrad: boolean,
  needsWeightGrad: boolean,
  hasBias: boolean,
): GradFn {
  return (grad, getSaved) => {
    if (inputShape.length < 2 || weightShape.length < 2)
      throw new Error("linear backward requires rank >= 2");

    let savedIdx = 0;
    // dX = dY @ W  (weight is [out, in])
    let resultInput: RuntimeTensor | null = null;
    if (needsInputGrad) {
      const savedWeight = getSaved(savedIdx++)._unwrap();
      const gradInput = ctx.rt.matmul(grad, savedWeight);
      resultInput = ctx.sumToShape(gradInput, inputShape);
    }

    // dW = dY^T @ X → [out, in] = weight's shape directly
    let resultWeight: RuntimeTensor | null = null;
    if (needsWeightGrad) {
      const savedInput = getSaved(savedIdx++)._unwrap();
      const gradT = ctx.rt.transpose(grad, {
        dim0: grad.shape.length - 2,
        dim1: grad.shape.length - 1,
      });
      const gradWeight = ctx.rt.matmul(gradT, savedInput);
      resultWeight = ctx.sumToShape(gradWeight, weightShape);
    }

    // dBias = sum(dY, all dims except last)
    let resultBias: RuntimeTensor | null = null;
    if (hasBias) {
      const biasShape = getSaved(savedIdx).shape;
      resultBias = ctx.sumToShape(grad, biasShape);
    }

    return hasBias
      ? [resultInput, resultWeight, resultBias as RuntimeTensor]
      : [resultInput, resultWeight];
  };
}

/**
 * GELU backward (tanh approximation).
 * gelu(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
 */
export function geluTanhBackward(ctx: BackwardContext): GradFn {
  return (grad, getSaved) => {
    const x = getSaved(0)._unwrap();
    const { rt } = ctx;

    const x2 = rt.mul(x, x);
    const x3 = rt.mul(x2, x);
    const term = rt.add(x, rt.mul(0.044715, x3));
    const innerVal = rt.mul(0.7978845608, term);

    const clampedInner = rt.where(
      rt.lt(innerVal, -10),
      -10,
      rt.where(rt.gt(innerVal, 10), 10, innerVal),
    );

    const tanhInner = rt.tanh(clampedInner);
    const cdf = rt.mul(0.5, rt.add(1, tanhInner));
    const tanh2 = rt.mul(tanhInner, tanhInner);
    const sech2 = rt.sub(1, tanh2);

    const pdfTerm = rt.add(1, rt.mul(0.134145, x2));
    const pdf = rt.mul(rt.mul(0.7978845608, pdfTerm), sech2);

    const xPdfHalf = rt.mul(rt.mul(x, pdf), 0.5);
    const geluGrad = rt.add(cdf, xPdfHalf);
    return [rt.mul(grad, geluGrad)];
  };
}

/**
 * GELU backward (erf approximation using Horner's method).
 * gelu(x) = 0.5x(1 + erf(x/√2))
 */
export function geluErfBackward(ctx: BackwardContext): GradFn {
  return (grad, getSaved) => {
    const x = getSaved(0)._unwrap();
    const { rt } = ctx;

    const z = rt.mul(x, Math.SQRT1_2);
    const absZ = rt.abs(z);

    const t = rt.div(1, rt.add(1, rt.mul(0.3275911, absZ)));
    const t2 = rt.mul(t, t);
    const t3 = rt.mul(t2, t);
    const t4 = rt.mul(t3, t);
    const t5 = rt.mul(t4, t);

    const poly = rt.add(
      rt.mul(0.254829592, t),
      rt.add(
        rt.mul(-0.284496736, t2),
        rt.add(
          rt.mul(1.421413741, t3),
          rt.add(rt.mul(-1.453152027, t4), rt.mul(1.061405429, t5)),
        ),
      ),
    );

    const negZ2 = rt.mul(-0.5, rt.mul(x, x));
    const expNegZ2 = rt.exp(negZ2);

    const erfAbs = rt.sub(1, rt.mul(poly, expNegZ2));
    const xGe0 = rt.ge(x, 0);
    const erfPos = rt.add(1, erfAbs);
    const erfNeg = rt.sub(1, erfAbs);
    const erfTerm = rt.where(xGe0, erfPos, erfNeg);
    const cdf = rt.mul(0.5, erfTerm);

    const pdf = rt.mul(expNegZ2, 0.3989422804014327);
    const xPdf = rt.mul(x, pdf);
    const geluGrad = rt.add(cdf, xPdf);
    return [rt.mul(grad, geluGrad)];
  };
}
