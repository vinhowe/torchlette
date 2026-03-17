/**
 * LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.
 *
 * LoRA adds trainable low-rank decomposition matrices to frozen base weights:
 * y = Wx + b + (alpha/rank) * (BA)x
 *
 * Where:
 * - W: Frozen base weight [out_features, in_features]
 * - B: LoRA down-projection [out_features, rank] - initialized to zeros
 * - A: LoRA up-projection [rank, in_features] - initialized with random normal
 * - alpha: Scaling factor (typically equal to rank)
 */

import type { FrontendTensor as Tensor, Torchlette } from "torchlette";

export type LoRAConfig = {
  rank: number;
  alpha: number;
  dropout?: number;
};

/**
 * LoRA-wrapped Linear layer.
 *
 * Wraps a frozen base Linear layer with trainable LoRA parameters.
 */
export class LoRALinear {
  readonly api: Torchlette;
  readonly inFeatures: number;
  readonly outFeatures: number;
  readonly rank: number;
  readonly alpha: number;
  readonly scaling: number;

  // Base weights (frozen - requiresGrad: false)
  readonly baseWeight: Tensor;
  readonly baseBias: Tensor | null;

  // LoRA parameters (trainable - requiresGrad: true)
  readonly loraA: Tensor; // [rank, in_features]
  readonly loraB: Tensor; // [out_features, rank]

  constructor(
    api: Torchlette,
    inFeatures: number,
    outFeatures: number,
    config: LoRAConfig,
    options?: { device?: "cpu" | "webgpu" },
  ) {
    this.api = api;
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.rank = config.rank;
    this.alpha = config.alpha;
    this.scaling = config.alpha / config.rank;

    const device = options?.device ?? "webgpu";

    // Initialize base weights (will be loaded from pretrained)
    // These are placeholders - actual values come from loadBaseWeights()
    this.baseWeight = api.zeros([outFeatures, inFeatures], {
      device,
      requiresGrad: false,
    });
    this.baseBias = api.zeros([outFeatures], {
      device,
      requiresGrad: false,
    });

    // Initialize LoRA A with small random normal (std=0.01).
    // This matches the standard LoRA init and produces stable training
    // dynamics with Adam. Kaiming uniform gives 2x larger values which
    // can cause Adam to overshoot on small datasets.
    this.loraA = api.randn([config.rank, inFeatures], {
      device,
      requiresGrad: true,
    });
    // Scale to std=0.01
    const runtime = api._runtime();
    runtime.copy_(
      this.loraA._unwrap(),
      runtime.mul(this.loraA._unwrap(), 0.01),
    );

    // Initialize LoRA B to zeros (so LoRA starts as identity)
    // B is [out_features, rank]
    this.loraB = api.zeros([outFeatures, config.rank], {
      device,
      requiresGrad: true,
    });
  }

  /**
   * Load base weights from pretrained model.
   * The base weights are frozen (requiresGrad: false).
   */
  loadBaseWeights(weight: Tensor, bias: Tensor | null): void {
    // Copy pretrained weights to base (keeping them frozen)
    const runtime = this.api._runtime();
    runtime.copy_(this.baseWeight._unwrap(), weight._unwrap());
    if (bias && this.baseBias) {
      runtime.copy_(this.baseBias._unwrap(), bias._unwrap());
    }
  }

  /**
   * Forward pass: y = Wx + b + scaling * (x @ A^T @ B^T)
   */
  forward(x: Tensor): Tensor {
    // Base linear: y = x @ W^T + b
    // W is [out, in], so W^T is [in, out]
    const baseOut = this.api.matmul(
      x,
      this.baseWeight.transpose({ dim0: 0, dim1: 1 }),
    );

    const withBias = this.baseBias
      ? this.api.add(baseOut, this.baseBias)
      : baseOut;

    // LoRA path: scaling * (x @ A^T @ B^T)
    const loraOut = this.api.matmul(
      this.api.matmul(x, this.loraA.transpose({ dim0: 0, dim1: 1 })),
      this.loraB.transpose({ dim0: 0, dim1: 1 }),
    );
    const scalingTensor = this.api.tensorFromArray([this.scaling], []);
    const scaledLora = this.api.mul(loraOut, scalingTensor);

    // Detach base output — base weights are frozen (requiresGrad=false).
    // This prevents building autograd graph through base weights, saving memory.
    // Gradient for x flows through the LoRA path only, which is sufficient
    // since the LoRA adapter is the only thing being optimized.
    return this.api.add(withBias.detach(), scaledLora);
  }

  /**
   * Get trainable LoRA parameters only.
   */
  getLoRAParameters(): Tensor[] {
    return [this.loraA, this.loraB];
  }

  /**
   * Get all parameters (for state dict).
   */
  parameters(): Tensor[] {
    const params = [this.baseWeight, this.loraA, this.loraB];
    if (this.baseBias) params.push(this.baseBias);
    return params;
  }
}

/**
 * Create LoRA config with sensible defaults.
 */
export function createLoRAConfig(rank: number = 8, alpha?: number): LoRAConfig {
  return {
    rank,
    alpha: alpha ?? rank,
  };
}
