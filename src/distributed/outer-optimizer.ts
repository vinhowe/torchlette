/**
 * DiLoCo Outer Optimizer: Nesterov Momentum SGD
 *
 * Applies averaged pseudo-gradients to global parameters using Nesterov
 * momentum. This is the "outer loop" of DiLoCo — called once every H
 * inner training steps after all-reducing pseudo-gradients across workers.
 *
 * Algorithm per outer step:
 *   v_t = mu * v_{t-1} + delta_avg        (momentum update)
 *   theta_{t+1} = theta_t + lr * v_t      (parameter update)
 *
 * Where delta_avg is the averaged pseudo-gradient: (local_params - global_params)
 * averaged across all workers.
 *
 * The update runs on CPU (read params, compute, write back). This avoids
 * GPU buffer lifecycle issues and is fine since it runs once per H steps.
 *
 * Default hyperparameters from DiLoCo paper:
 *   lr = 0.7, mu = 0.9
 */

import type { Tensor } from "../frontend/tensor";
import type { Torchlette } from "../frontend/torchlette";

export interface OuterOptimizerConfig {
  /** Outer learning rate (default: 0.7) */
  lr?: number;
  /** Nesterov momentum coefficient (default: 0.9) */
  momentum?: number;
}

/**
 * Nesterov SGD outer optimizer for DiLoCo.
 *
 * Holds momentum buffers (CPU Float32Arrays) for each parameter.
 * Call `step()` with the averaged pseudo-gradients after all-reduce.
 */
export class NesterovOuterOptimizer {
  private readonly lr: number;
  private readonly mu: number;
  private readonly velocities: Map<Tensor, Float32Array> = new Map();
  private readonly api: Torchlette;

  constructor(api: Torchlette, config: OuterOptimizerConfig = {}) {
    this.api = api;
    this.lr = config.lr ?? 0.7;
    this.mu = config.momentum ?? 0.9;
  }

  /**
   * Apply one outer optimization step.
   *
   * Reads params and pseudo-grads to CPU, computes Nesterov update,
   * writes updated params back to GPU.
   *
   * @param params - Model parameters
   * @param pseudoGrads - Averaged pseudo-gradients (same order as params).
   *   Each pseudo-gradient is: local_params - global_params_at_sync_start.
   */
  async step(params: Tensor[], pseudoGrads: Tensor[]): Promise<void> {
    if (params.length !== pseudoGrads.length) {
      throw new Error(
        `Outer optimizer: params (${params.length}) and pseudoGrads (${pseudoGrads.length}) must match`,
      );
    }

    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const deltaData = await pseudoGrads[i].cpu();

      // Get or create velocity (zero-initialized on first use)
      let v = this.velocities.get(param);
      if (!v) {
        v = new Float32Array(deltaData.length);
        this.velocities.set(param, v);
      }

      // Read current param values
      const paramData = await param.cpu();

      // Nesterov momentum: v = mu * v + delta, theta = theta + lr * v
      const updated = new Float32Array(deltaData.length);
      for (let j = 0; j < deltaData.length; j++) {
        v[j] = this.mu * v[j] + deltaData[j];
        updated[j] = paramData[j] + this.lr * v[j];
      }

      // Write updated params back to GPU
      const t = this.api.tensorFromArray(Array.from(updated), param.shape, {
        device: param.device,
      });
      this.api.copy_(param, t);
    }
  }

  /** Dispose all momentum buffers. */
  dispose(): void {
    this.velocities.clear();
  }
}
