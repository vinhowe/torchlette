/**
 * Linear (fully connected) layer.
 * Similar to PyTorch's nn.Linear.
 */

import type { DeviceKind, Tensor, Torchlette } from "../frontend";
import { Module } from "./module";

export type LinearOptions = {
  /** Whether to include a bias term. Default: true */
  bias?: boolean;
  /** Device to create parameters on */
  device?: DeviceKind;
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

    // Initialize weight with standard normal distribution.
    // Uses lazy GPU-side randn to avoid allocating large JS arrays on CPU.
    // Note: for pretrained models, weights are overwritten by copy_ during loading.
    this.registerParameter(
      "weight",
      api.randn([outFeatures, inFeatures], { requiresGrad: true, device }),
    );

    // Bias shape: [outFeatures]
    this.registerParameter(
      "bias",
      hasBias ? api.zeros([outFeatures], { requiresGrad: true, device }) : null,
    );
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
