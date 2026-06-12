import type { Tensor, Torchlette } from "../frontend/torchlette";
import { Module } from "./module";

export type RMSNormOptions = {
  /** Numerical stability epsilon. Default: 1e-5 (PyTorch default 1e-6 for LLaMA-family configs — pass explicitly). */
  eps?: number;
  /** Whether to learn a scale weight. Default: true. */
  elementwiseAffine?: boolean;
  device?: import("../backend/types").DeviceKind;
};

/**
 * Root-mean-square normalization (Zhang & Sennrich 2019) — the LLaMA-family
 * norm: `rmsnorm(x) = x / sqrt(mean(x², -1) + eps) * weight`. No mean
 * subtraction, no bias. Thin module wrapper over `tensor.rmsnorm()`, which
 * lowers to the fused RMSNorm kernel on WebGPU and the decomposed graph
 * elsewhere.
 */
export class RMSNorm extends Module {
  readonly normalizedShape: number;
  readonly eps: number;
  declare readonly weight: Tensor | null;

  constructor(
    api: Torchlette,
    normalizedShape: number,
    options?: RMSNormOptions,
  ) {
    super(api);
    this.normalizedShape = normalizedShape;
    this.eps = options?.eps ?? 1e-5;
    const elementwiseAffine = options?.elementwiseAffine ?? true;
    this.registerParameter(
      "weight",
      elementwiseAffine
        ? api.ones([normalizedShape], {
            requiresGrad: true,
            device: options?.device,
          })
        : null,
    );
  }

  forward(input: Tensor): Tensor {
    if (this.weight === null) {
      throw new Error("RMSNorm without elementwiseAffine is not yet supported");
    }
    return input.rmsnorm(this.weight, this.eps);
  }
}
