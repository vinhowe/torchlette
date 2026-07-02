/**
 * Linear (fully connected) layer.
 * Similar to PyTorch's nn.Linear.
 */

import type { DeviceKind, Tensor, Torchlette } from "../frontend/torchlette";
import { uniform_ } from "./init";
import { Module } from "./module";

export type LinearOptions = {
  /** Whether to include a bias term. Default: true */
  bias?: boolean;
  /** Device to create parameters on */
  device?: DeviceKind;
  /**
   * Parameter dtype. Default: "f32". Non-f32 skips random init (parameters
   * are zeros) — intended for pretrained-weight loading, e.g. f16 weights
   * with f32 activations via mixed-dtype matmul.
   */
  dtype?: import("../backend/types").DType;
};

/**
 * Linear transformation: y = x @ W^T + b
 *
 * @example
 * ```ts
 * const linear = new Linear(api, 768, 3072);  // [batch, 768] -> [batch, 3072]
 * const output = linear.forward(input);
 * ```
 */
export class Linear extends Module {
  readonly inFeatures: number;
  readonly outFeatures: number;
  declare readonly weight: Tensor;
  declare readonly bias: Tensor | null;

  constructor(
    api: Torchlette,
    inFeatures: number,
    outFeatures: number,
    options?: LinearOptions,
  ) {
    super(api);
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;

    const hasBias = options?.bias ?? true;
    const device = options?.device;
    const dtype = options?.dtype ?? "f32";

    // PyTorch default init:
    //   weight ~ kaiming_uniform(a=sqrt(5)) ≡ U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    //   bias   ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    // See https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
    // Note: for pretrained models, weights are overwritten by copy_ during loading.
    // Non-f32 dtypes skip init (pretrained-loading path).
    const bound = inFeatures > 0 && dtype === "f32" ? 1 / Math.sqrt(inFeatures) : 0;
    const weight = api.zeros([outFeatures, inFeatures], {
      requiresGrad: true,
      device,
      dtype,
    });
    if (bound > 0) uniform_(api, weight, -bound, bound);
    this.registerParameter("weight", weight);

    // Bias shape: [outFeatures]
    let bias: Tensor | null = null;
    if (hasBias) {
      bias = api.zeros([outFeatures], { requiresGrad: true, device, dtype });
      if (bound > 0) uniform_(api, bias, -bound, bound);
    }
    this.registerParameter("bias", bias);
  }

  /**
   * Forward pass: y = x @ W^T + b
   *
   * @param input - Input tensor of shape [..., inFeatures]
   * @returns Output tensor of shape [..., outFeatures]
   */
  forward(input: Tensor): Tensor {
    return this.api.linear(input, this.weight, this.bias);
  }
}
