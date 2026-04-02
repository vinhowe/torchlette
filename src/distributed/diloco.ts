/**
 * DiLoCo Training Loop
 *
 * Coordinates inner optimization (AdamW) with outer synchronization
 * (Nesterov SGD on pseudo-gradients). This module handles the local
 * training loop — networking is handled separately by the peer layer.
 *
 * Usage:
 * ```typescript
 * const diloco = new DiLoCoTrainer(api, model.parameters(), {
 *   innerSteps: 500,
 *   innerLR: 4e-4,
 *   outerLR: 0.7,
 *   outerMomentum: 0.9,
 * });
 *
 * // Each call runs H inner steps and returns pseudo-gradients
 * const pseudoGrads = await diloco.innerLoop(dataLoader);
 *
 * // After all-reduce across peers:
 * diloco.outerStep(averagedPseudoGrads);
 * ```
 */

import type { Tensor } from "../frontend/tensor";
import type { Torchlette } from "../frontend/torchlette";
import { Adam } from "../optim";
import { NesterovOuterOptimizer } from "./outer-optimizer";

export interface DiLoCoConfig {
  /** Number of inner training steps between syncs (default: 500) */
  innerSteps?: number;
  /** Inner optimizer learning rate (default: 4e-4) */
  innerLR?: number;
  /** Inner optimizer weight decay (default: 0.1) */
  innerWeightDecay?: number;
  /** Outer optimizer learning rate (default: 0.7) */
  outerLR?: number;
  /** Outer optimizer Nesterov momentum (default: 0.9) */
  outerMomentum?: number;
}

export interface InnerLoopResult {
  /** Pseudo-gradients: local_params - global_params for each parameter */
  pseudoGrads: Float32Array[];
  /** Average training loss over the inner loop */
  avgLoss: number;
  /** Number of steps completed */
  steps: number;
}

/**
 * DiLoCo trainer: manages inner AdamW loop and outer Nesterov updates.
 *
 * The trainer snapshots global parameters at the start of each inner loop.
 * After H inner steps, it computes pseudo-gradients (local - snapshot) and
 * returns them for communication. After all-reduce, `outerStep()` applies
 * the averaged pseudo-gradients via Nesterov momentum.
 */
export class DiLoCoTrainer {
  readonly api: Torchlette;
  readonly params: Tensor[];
  readonly innerSteps: number;

  private readonly innerOptimizer: Adam;
  private readonly outerOptimizer: NesterovOuterOptimizer;
  private globalSnapshot: Float32Array[] | null = null;

  constructor(api: Torchlette, params: Tensor[], config: DiLoCoConfig = {}) {
    this.api = api;
    this.params = params;
    this.innerSteps = config.innerSteps ?? 500;

    this.innerOptimizer = new Adam(
      params,
      {
        lr: config.innerLR ?? 4e-4,
        weightDecay: config.innerWeightDecay ?? 0.1,
      },
      api,
    );

    this.outerOptimizer = new NesterovOuterOptimizer(api, {
      lr: config.outerLR ?? 0.7,
      momentum: config.outerMomentum ?? 0.9,
    });
  }

  /**
   * Snapshot current parameters as the "global" state before inner loop.
   * Call this at the start of each DiLoCo round.
   */
  async snapshotGlobalParams(): Promise<void> {
    this.globalSnapshot = [];
    for (const param of this.params) {
      const data = await param.cpu();
      this.globalSnapshot.push(new Float32Array(data));
    }
  }

  /**
   * Compute pseudo-gradients: local_params - global_snapshot.
   * Call this after the inner loop completes.
   *
   * @returns Array of Float32Array pseudo-gradients (one per parameter)
   */
  async computePseudoGrads(): Promise<Float32Array[]> {
    if (!this.globalSnapshot) {
      throw new Error(
        "Must call snapshotGlobalParams() before computing pseudo-grads",
      );
    }

    const pseudoGrads: Float32Array[] = [];
    for (let i = 0; i < this.params.length; i++) {
      const localData = await this.params[i].cpu();
      const globalData = this.globalSnapshot[i];
      const delta = new Float32Array(localData.length);
      for (let j = 0; j < delta.length; j++) {
        delta[j] = localData[j] - globalData[j];
      }
      pseudoGrads.push(delta);
    }

    return pseudoGrads;
  }

  /**
   * Apply averaged pseudo-gradients via outer Nesterov optimizer.
   * Call this after all-reduce across peers.
   *
   * First resets params to global snapshot, then applies outer update.
   *
   * @param avgPseudoGrads - Averaged pseudo-gradients from all workers
   */
  async outerStep(avgPseudoGrads: Tensor[]): Promise<void> {
    if (!this.globalSnapshot) {
      throw new Error("Must call snapshotGlobalParams() before outerStep");
    }

    // Reset parameters to global snapshot (before inner training modified them)
    for (let i = 0; i < this.params.length; i++) {
      const snapTensor = this.api.tensorFromArray(
        Array.from(this.globalSnapshot[i]),
        this.params[i].shape,
        { device: this.params[i].device },
      );
      this.api.copy_(this.params[i], snapTensor);
      snapTensor.dispose();
    }

    // Apply outer optimizer update
    await this.outerOptimizer.step(this.params, avgPseudoGrads);

    // Reset inner optimizer state (momentum from previous inner loop is stale)
    this.innerOptimizer.zeroGrad();

    this.globalSnapshot = null;
  }

  /** Get the inner optimizer for use in the training loop. */
  getInnerOptimizer(): Adam {
    return this.innerOptimizer;
  }

  /** Dispose all resources. */
  dispose(): void {
    this.outerOptimizer.dispose();
    this.globalSnapshot = null;
  }
}
