/**
 * DiLoCo Outer Optimizer: Nesterov Momentum SGD
 *
 * Applies averaged pseudo-gradients to global parameters using Nesterov
 * momentum. Velocity state lives on CPU (Float32Arrays). The param
 * update is computed on CPU and written to GPU via tensorFromArray + copy_.
 *
 * This avoids GPU buffer lifecycle issues (lazy tensors freed by step-scoped
 * cleanup) and is fine since the outer step runs once per H inner steps.
 *
 * Algorithm per outer step:
 *   v_t = mu * v_{t-1} + delta_avg
 *   theta_{t+1} = theta_t + lr * v_t
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
 * Velocity buffers are CPU Float32Arrays. Param update is computed on CPU
 * and written back to GPU in one copy. Must be called within a
 * beginStep()/endStep() pair.
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
   * Reads current params and pseudo-grads to CPU, computes Nesterov
   * update, writes result back to GPU via copy_.
   *
   * Must be called within beginStep()/endStep().
   */
  async step(params: Tensor[], pseudoGrads: Tensor[]): Promise<void> {
    if (params.length !== pseudoGrads.length) {
      throw new Error(
        `Outer optimizer: params (${params.length}) and pseudoGrads (${pseudoGrads.length}) must match`,
      );
    }

    const api = this.api;

    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const paramData = await param.cpu();
      const deltaData = await pseudoGrads[i].cpu();
      const n = paramData.length;

      // Get or create velocity
      let v = this.velocities.get(param);
      if (!v) {
        v = new Float32Array(n);
        this.velocities.set(param, v);
      }

      // Nesterov: v = mu * v + delta, theta = theta + lr * v
      const updated = new Float32Array(n);
      for (let j = 0; j < n; j++) {
        v[j] = this.mu * v[j] + deltaData[j];
        updated[j] = paramData[j] + this.lr * v[j];
      }

      // Write back to GPU
      const t = api.tensorFromArray(Array.from(updated), param.shape, {
        device: param.device,
      });
      api.copy_(param, t);
    }
  }

  /** Dispose all momentum buffers. */
  dispose(): void {
    this.velocities.clear();
  }
}
