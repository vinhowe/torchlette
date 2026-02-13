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
  readonly weight: Tensor;
  readonly bias: Tensor | null;

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

    // Initialize weight with scaled normal distribution (Kaiming-like)
    // Scale by 1/sqrt(in_features) for better gradient flow
    const scale = 1 / Math.sqrt(inFeatures);
    const weightData = new Array(outFeatures * inFeatures);
    for (let i = 0; i < weightData.length; i += 2) {
      // Box-Muller for standard normal
      const u1 = Math.random();
      const u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1 || 1e-10));
      const theta = 2 * Math.PI * u2;
      weightData[i] = r * Math.cos(theta) * scale;
      if (i + 1 < weightData.length) {
        weightData[i + 1] = r * Math.sin(theta) * scale;
      }
    }

    // Weight shape: [outFeatures, inFeatures]
    this.weight = api.tensorFromArray(weightData, [outFeatures, inFeatures], {
      requiresGrad: true,
      device,
    });

    // Bias shape: [outFeatures]
    if (hasBias) {
      this.bias = api.zeros([outFeatures], { requiresGrad: true, device });
    } else {
      this.bias = null;
    }
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

  /**
   * Get all learnable parameters.
   */
  parameters(): Tensor[] {
    if (this.bias !== null) {
      return [this.weight, this.bias];
    }
    return [this.weight];
  }
}
