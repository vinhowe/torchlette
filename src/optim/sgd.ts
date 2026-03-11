import type { Tensor, Torchlette } from "../frontend";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import { validateOptimizerParams } from "./validate";

export type SGDOptions = {
  lr: number;
  momentum?: number;
  weightDecay?: number;
};

export class SGD {
  private params: Tensor[];
  private readonly api: Torchlette;
  private _lr: number;
  private readonly momentum: number;
  private readonly weightDecay: number;
  private velocity: Array<RuntimeTensor | null>;

  constructor(params: Tensor[], options: SGDOptions, api?: Torchlette) {
    const { api: engine } = validateOptimizerParams("SGD", params, api);
    if (options.lr <= 0) {
      throw new Error("SGD learning rate must be > 0");
    }
    this.api = engine;
    this._lr = options.lr;
    this.params = params.slice();
    this.momentum = options.momentum ?? 0;
    this.weightDecay = options.weightDecay ?? 0;
    this.velocity = new Array(params.length).fill(null);
  }

  getParams(): Tensor[] {
    return this.params.slice();
  }

  getLR(): number {
    return this._lr;
  }

  setLR(lr: number): void {
    this._lr = lr;
  }

  step(): Tensor[] {
    const runtime = this.api._runtime();
    const updated: Tensor[] = [];
    for (let i = 0; i < this.params.length; i += 1) {
      const param = this.params[i];
      const grad = param.grad?._unwrap() ?? null;
      if (!grad) {
        updated.push(param);
        continue;
      }
      let gradAdj = grad;
      if (this.weightDecay !== 0) {
        const decay = runtime.mul(param._unwrap(), this.weightDecay);
        gradAdj = runtime.add(gradAdj, decay);
      }
      let update = gradAdj;
      if (this.momentum !== 0) {
        const prev = this.velocity[i];
        if (prev) {
          const mom = runtime.mul(prev, this.momentum);
          update = runtime.add(mom, gradAdj);
        } else {
          update = gradAdj;
        }
        this.velocity[i] = update;
      }
      const next = runtime.sub(param._unwrap(), update, {
        alpha: this._lr,
      });
      // Update parameter IN-PLACE to preserve tensor identity
      // This is critical for GradScaler - backward() writes to the same tensor
      runtime.copy_(param._unwrap(), next);
      updated.push(param);
    }
    // Don't replace this.params - keep original tensor objects
    return updated;
  }

  zeroGrad(): void {
    for (const param of this.params) {
      param.zeroGrad();
    }
  }
}
