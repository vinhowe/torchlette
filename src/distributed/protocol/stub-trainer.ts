/**
 * Synthetic CPU trainer for protocol smoke tests.
 *
 * Holds Float32Array params, runs deterministic gradient descent toward a
 * per-tensor "target" vector. Implements the same Trainer contract the
 * real WebGPU trainer will — so a smoke test that exercises the protocol
 * against StubTrainer also exercises the same call sequence the production
 * trainer will see.
 *
 * Convergence properties used by tests:
 *   - With shared targets across peers AND identical inner-step trajectory,
 *     all peers produce identical pseudograds → the outer step is trivially
 *     consistent.
 *   - With different per-peer targets (simulating different data shards),
 *     pseudograds differ; the protocol must still produce identical
 *     post-outer-step params on every participating peer because the
 *     averaged grad is the same set on every peer.
 *
 * Loss is a simple L2 to the target — useful for asserting "loss
 * descended after outer step" in tests.
 */

import type { ParamShapes, Trainer } from "./trainer.ts";

export interface StubTrainerOptions {
  paramShapes: ParamShapes;
  /** Per-tensor target vectors. If omitted, defaults to zeros. */
  targets?: Float32Array[];
  /** Deterministic seed for initial params. */
  seed?: number;
  innerSteps?: number;
  innerLr?: number;
  outerLr?: number;
  outerMu?: number;
}

/** Tiny deterministic PRNG so init is reproducible across peers. */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function flatSize(shape: readonly number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

function copyOf(src: readonly Float32Array[]): Float32Array[] {
  return src.map((a) => new Float32Array(a));
}

function zerosLike(src: readonly Float32Array[]): Float32Array[] {
  return src.map((a) => new Float32Array(a.length));
}

export class StubTrainer implements Trainer {
  private readonly shapes: ParamShapes;
  private readonly innerStepsPerRound: number;
  private readonly innerLr: number;
  private readonly outerLr: number;
  private readonly outerMu: number;
  private readonly targets: Float32Array[];

  private params: Float32Array[];
  private anchor: Float32Array[];
  /** Outer Nesterov velocity, per param tensor. */
  private outerV: Float32Array[];

  constructor(opts: StubTrainerOptions) {
    this.shapes = opts.paramShapes;
    this.innerStepsPerRound = opts.innerSteps ?? 10;
    this.innerLr = opts.innerLr ?? 0.1;
    this.outerLr = opts.outerLr ?? 0.7;
    this.outerMu = opts.outerMu ?? 0.9;

    const rng = mulberry32(opts.seed ?? 42);
    this.params = this.shapes.map((shape) => {
      const n = flatSize(shape);
      const a = new Float32Array(n);
      for (let i = 0; i < n; i++) a[i] = rng() * 0.04 - 0.02;
      return a;
    });
    this.targets =
      opts.targets ??
      this.shapes.map((shape) => new Float32Array(flatSize(shape)));

    // Sanity-check target shapes
    if (this.targets.length !== this.params.length) {
      throw new Error(
        `StubTrainer: targets count ${this.targets.length} != params count ${this.params.length}`,
      );
    }
    for (let i = 0; i < this.params.length; i++) {
      if (this.targets[i].length !== this.params[i].length) {
        throw new Error(
          `StubTrainer: target[${i}].length=${this.targets[i].length} != params[${i}].length=${this.params[i].length}`,
        );
      }
    }

    this.anchor = copyOf(this.params);
    this.outerV = zerosLike(this.params);
  }

  paramShapes(): ParamShapes {
    return this.shapes;
  }

  setAnchor(): void {
    this.anchor = copyOf(this.params);
  }

  async innerSteps(_round: number): Promise<void> {
    for (let s = 0; s < this.innerStepsPerRound; s++) {
      for (let p = 0; p < this.params.length; p++) {
        const cur = this.params[p];
        const tgt = this.targets[p];
        const lr = this.innerLr;
        for (let i = 0; i < cur.length; i++) {
          cur[i] += lr * (tgt[i] - cur[i]);
        }
      }
    }
  }

  pseudograd(): Float32Array[] {
    const out: Float32Array[] = [];
    for (let p = 0; p < this.params.length; p++) {
      const cur = this.params[p];
      const anc = this.anchor[p];
      const delta = new Float32Array(cur.length);
      for (let i = 0; i < cur.length; i++) delta[i] = cur[i] - anc[i];
      out.push(delta);
    }
    return out;
  }

  applyOuterStep(avgGrad: Float32Array[]): void {
    for (let p = 0; p < this.params.length; p++) {
      const v = this.outerV[p];
      const g = avgGrad[p];
      const anc = this.anchor[p];
      const cur = this.params[p];
      for (let i = 0; i < v.length; i++) {
        v[i] = this.outerMu * v[i] + g[i];
        cur[i] = anc[i] + this.outerLr * v[i];
      }
    }
    // New anchor = new params.
    this.anchor = copyOf(this.params);
  }

  revertToAnchor(): void {
    this.params = copyOf(this.anchor);
  }

  snapshotAnchor(): Float32Array[] {
    return copyOf(this.anchor);
  }

  applyF16W(params: Float32Array[]): void {
    if (params.length !== this.params.length) {
      throw new Error(
        `applyF16W: payload tensor count ${params.length} != expected ${this.params.length}`,
      );
    }
    this.params = copyOf(params);
    this.anchor = copyOf(params);
  }

  resetOptimState(): void {
    this.outerV = zerosLike(this.outerV);
  }

  // ─── Test helpers (not part of the Trainer interface) ───────────────

  /** Current params snapshot. */
  currentParams(): Float32Array[] {
    return copyOf(this.params);
  }

  /** Anchor params snapshot. */
  anchorParams(): Float32Array[] {
    return copyOf(this.anchor);
  }

  /** L2 distance to each target — proxy for "loss." */
  lossToTarget(): number {
    let total = 0;
    for (let p = 0; p < this.params.length; p++) {
      const cur = this.params[p];
      const tgt = this.targets[p];
      for (let i = 0; i < cur.length; i++) {
        const d = cur[i] - tgt[i];
        total += d * d;
      }
    }
    return Math.sqrt(total);
  }
}
