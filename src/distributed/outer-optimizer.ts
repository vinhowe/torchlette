/**
 * DiLoCo Outer Optimizer: Nesterov Momentum SGD
 *
 * Applies averaged pseudo-gradients to global parameters using Nesterov
 * momentum. Velocity state lives on CPU (Float32Arrays). The param update
 * is computed on CPU and uploaded to GPU via tensorFromArray + copy_.
 *
 * Algorithm per outer step:
 *   v_t = mu * v_{t-1} + delta_avg
 *   theta_{t+1} = theta_t + lr * v_t
 *
 * Default hyperparameters from DiLoCo paper: lr = 0.7, mu = 0.9.
 */

import type { Tensor } from "../frontend/tensor";
import type { Torchlette } from "../frontend/torchlette";

export interface OuterOptimizerConfig {
  /** Outer learning rate (default: 0.7) */
  lr?: number;
  /** Nesterov momentum coefficient (default: 0.9) */
  momentum?: number;
}

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

  /** Apply one outer optimization step using a CPU snapshot + averaged grads. */
  async stepFromCpu(
    params: Tensor[],
    snapshot: Float32Array[],
    avgGrads: Float32Array[],
  ): Promise<void> {
    if (
      params.length !== snapshot.length ||
      params.length !== avgGrads.length
    ) {
      throw new Error(
        `stepFromCpu: length mismatch (params=${params.length}, snapshot=${snapshot.length}, avgGrads=${avgGrads.length})`,
      );
    }
    const api = this.api;
    await api.beginStep();
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const snap = snapshot[i];
      const grad = avgGrads[i];
      const n = snap.length;
      let v = this.velocities.get(param);
      if (!v) {
        v = new Float32Array(n);
        this.velocities.set(param, v);
      }
      const updated = new Float32Array(n);
      for (let j = 0; j < n; j++) {
        v[j] = this.mu * v[j] + grad[j];
        updated[j] = snap[j] + this.lr * v[j];
      }
      api.copy_(
        param,
        api.tensorFromArray(updated, param.shape, { device: param.device }),
      );
    }
    api.endStep();
    await api.markStep();
  }

  /**
   * Legacy: apply outer step using GPU pseudo-grad tensors. Reads grads + params
   * via .cpu() (slow, two host round-trips per param). Prefer stepFromCpu.
   */
  async step(params: Tensor[], pseudoGrads: Tensor[]): Promise<void> {
    if (params.length !== pseudoGrads.length) {
      throw new Error(
        `Outer optimizer: params (${params.length}) and pseudoGrads (${pseudoGrads.length}) must match`,
      );
    }
    const api = this.api;
    await api.beginStep();
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const paramData = await param.cpu();
      const deltaData = await pseudoGrads[i].cpu();
      const n = paramData.length;
      let v = this.velocities.get(param);
      if (!v) {
        v = new Float32Array(n);
        this.velocities.set(param, v);
      }
      const updated = new Float32Array(n);
      for (let j = 0; j < n; j++) {
        v[j] = this.mu * v[j] + deltaData[j];
        updated[j] = paramData[j] + this.lr * v[j];
      }
      api.copy_(
        param,
        api.tensorFromArray(updated, param.shape, { device: param.device }),
      );
    }
    api.endStep();
    await api.markStep();
  }

  /** Dispose all momentum buffers. */
  dispose(): void {
    this.velocities.clear();
  }

  /**
   * Zero out momentum buffers. Use after an F16W resync — the velocity from
   * before the resync was accumulated against an anchor that is no longer the
   * one we're anchored to, so it would push params in a direction that is
   * meaningless relative to the new anchor.
   */
  reset(): void {
    for (const v of this.velocities.values()) v.fill(0);
  }
}
