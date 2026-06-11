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

import { ENV } from "../core/env";
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

  /** Apply one outer optimization step using a CPU snapshot + averaged grads.
   *  Returns the updated parameter values — these are EXACTLY the bytes
   *  uploaded into the GPU params (f32, bit-faithful), so the caller can use
   *  them as the new anchor without reading the whole model back. */
  async stepFromCpu(
    params: Tensor[],
    snapshot: Float32Array[],
    avgGrads: Float32Array[],
  ): Promise<Float32Array[]> {
    if (
      params.length !== snapshot.length ||
      params.length !== avgGrads.length
    ) {
      throw new Error(
        `stepFromCpu: length mismatch (params=${params.length}, snapshot=${snapshot.length}, avgGrads=${avgGrads.length})`,
      );
    }
    const api = this.api;
    const updatedAll: Float32Array[] = [];
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
      updatedAll.push(updated);
      api.copy_(
        param,
        api.tensorFromArray(updated, param.shape, { device: param.device }),
      );
    }
    api.endStep();
    await api.markStep();

    // Debug: verify the GPU params actually received the CPU-computed values
    // (this plan is upload+copy only, so JS holds exact ground truth).
    if (ENV.TORCHLETTE_DEBUG_OUTER_VERIFY === "1") {
      let worst = 0;
      let worstIdx = -1;
      let worstElem = -1;
      const TOL = 1e-5;
      const corrupt: string[] = [];
      for (let i = 0; i < params.length; i++) {
        const snap = snapshot[i];
        const v = this.velocities.get(params[i])!;
        const got = await params[i].cpu();
        let nBad = 0;
        let first = -1;
        let last = -1;
        let sampleExp = 0;
        let sampleGot = 0;
        for (let j = 0; j < snap.length; j++) {
          const expected = snap[j] + this.lr * v[j];
          const d = Math.abs(got[j] - expected);
          if (d > worst) {
            worst = d;
            worstIdx = i;
            worstElem = j;
          }
          if (d > TOL) {
            if (nBad === 0) {
              first = j;
              sampleExp = expected;
              sampleGot = got[j];
            }
            last = j;
            nBad++;
          }
        }
        if (nBad > 0) {
          corrupt.push(
            `param ${i} (${params[i].shape.join("x")}, n=${snap.length}): ${nBad} bad elems in [${first}..${last}] e.g. expected=${sampleExp.toPrecision(6)} got=${sampleGot.toPrecision(6)}`,
          );
        }
      }
      console.log(
        `[outer-verify] worst |gpu - expected| = ${worst.toExponential(3)} (param ${worstIdx} elem ${worstElem})${corrupt.length ? `\n[outer-verify] CORRUPT: ${corrupt.join("\n[outer-verify] CORRUPT: ")}` : ""}`,
      );
    }
    return updatedAll;
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
