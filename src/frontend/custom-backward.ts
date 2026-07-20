/**
 * Custom backward functions for non-elementwise ops.
 * Extracted from torchlette.ts to reduce the monolithic frontend.
 *
 * Only the TWO admitted contraction adjoints live here now (design §4.2):
 * `matmulBackward` and its `linearBackward` specialization — the index-transpose
 * facts about contraction that do NOT derive from the elementwise chain-rule.
 * The GELU tanh/erf custom backwards were DELETED (Crystal Campaign 3, P2): they
 * are the adjoint of the GELU composition and now derive via
 * `makeUnaryGrad(GELU_*_DEF)` (ops/semantic/composite.ts).
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
