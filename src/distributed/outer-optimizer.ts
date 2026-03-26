/**
 * DiLoCo Outer Optimizer: Nesterov Momentum SGD
 *
 * Applies averaged pseudo-gradients to global parameters using Nesterov
 * momentum. This is the "outer loop" of DiLoCo — called once every H
 * inner training steps after all-reducing pseudo-gradients across workers.
 *
 * Algorithm per outer step:
 *   v_t = mu * v_{t-1} + delta_avg        (momentum update)
 *   theta_{t+1} = theta_t - lr * v_t      (parameter update with Nesterov)
 *
 * Where delta_avg is the averaged pseudo-gradient: (local_params - global_params)
 * averaged across all workers.
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
 * Holds momentum buffers for each parameter. Call `step()` with the
 * averaged pseudo-gradients after all-reduce.
 */
export class NesterovOuterOptimizer {
  private readonly lr: number;
  private readonly mu: number;
  private readonly velocities: Map<Tensor, Tensor> = new Map();
  private readonly api: Torchlette;

  constructor(api: Torchlette, config: OuterOptimizerConfig = {}) {
    this.api = api;
    this.lr = config.lr ?? 0.7;
    this.mu = config.momentum ?? 0.9;
  }

  /**
   * Apply one outer optimization step.
   *
   * @param params - Model parameters (same tensors as inner optimizer)
   * @param pseudoGrads - Averaged pseudo-gradients (same order as params).
   *   Each pseudo-gradient is: local_params - global_params_at_sync_start.
   */
  step(params: Tensor[], pseudoGrads: Tensor[]): void {
    if (params.length !== pseudoGrads.length) {
      throw new Error(
        `Outer optimizer: params (${params.length}) and pseudoGrads (${pseudoGrads.length}) must match`,
      );
    }

    const api = this.api;

    api.tidy(() => {
      for (let i = 0; i < params.length; i++) {
        const param = params[i];
        const delta = pseudoGrads[i];

        // Get or create velocity buffer (lazy — zero-initialized on first use)
        let v = this.velocities.get(param);
        if (!v) {
          v = api.zeros(param.shape, { device: param.device });
          this.velocities.set(param, v);
        }

        // v = mu * v + delta
        const newV = api.add(api.mul(v, this.mu), delta);

        // theta = theta - lr * v (Nesterov: use the NEW velocity)
        api.copy_(param, api.sub(param, api.mul(newV, this.lr)));

        // Update stored velocity (keep it alive across tidy)
        api.keep(newV);
        this.velocities.set(param, newV);

        // Dispose old velocity
        v.dispose();
      }
    });
  }

  /** Dispose all momentum buffers. */
  dispose(): void {
    for (const v of this.velocities.values()) {
      v.dispose();
    }
    this.velocities.clear();
  }
}
