/**
 * Layer Normalization module.
 * Similar to PyTorch's nn.LayerNorm.
 */

import type { DeviceKind, Tensor, Torchlette } from "../frontend";
import { Module } from "./module";

export type LayerNormOptions = {
  /** Small constant for numerical stability. Default: 1e-5 */
  eps?: number;
  /** Whether to include learnable affine parameters. Default: true */
  elementwiseAffine?: boolean;
  /** Device to create parameters on */
  device?: DeviceKind;
};

/**
 * Applies Layer Normalization over the last dimension.
 *
 * LayerNorm normalizes the input across the features (last dimension):
 *   y = (x - mean) / sqrt(var + eps) * weight + bias
 *
 * @example
 * ```ts
 * const ln = new LayerNorm(api, 768);  // normalize 768-dim features
 * const output = ln.forward(input);    // input: [..., 768] -> output: [..., 768]
 * ```
 */
export class LayerNorm extends Module {
  readonly normalizedShape: number;
  readonly eps: number;
  readonly weight: Tensor | null;
  readonly bias: Tensor | null;

  constructor(
    api: Torchlette,
    normalizedShape: number,
    options?: LayerNormOptions,
  ) {
    super(api);
    this.normalizedShape = normalizedShape;
    this.eps = options?.eps ?? 1e-5;

    const elementwiseAffine = options?.elementwiseAffine ?? true;
    const device = options?.device;

    if (elementwiseAffine) {
      // Initialize weight to ones and bias to zeros (like PyTorch)
      this.weight = api.ones([normalizedShape], { requiresGrad: true, device });
      this.bias = api.zeros([normalizedShape], { requiresGrad: true, device });
    } else {
      this.weight = null;
      this.bias = null;
    }
  }

  /**
   * Forward pass: apply layer normalization.
   *
   * @param input - Input tensor of shape [..., normalizedShape]
   * @returns Normalized tensor of same shape
   */
  forward(input: Tensor): Tensor {
    if (this.weight === null || this.bias === null) {
      // Without affine parameters, just normalize
      // This requires implementing normalization without affine transform
      // For now, we'll require affine=true
      throw new Error(
        "LayerNorm without elementwiseAffine is not yet supported",
      );
    }

    return input.layernorm(this.weight, this.bias, this.eps);
  }

  /**
   * Get all learnable parameters.
   */
  parameters(): Tensor[] {
    if (this.weight !== null && this.bias !== null) {
      return [this.weight, this.bias];
    }
    return [];
  }
}
