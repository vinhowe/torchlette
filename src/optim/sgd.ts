import type { Tensor, Torchlette } from "../frontend/torchlette";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import { validateOptimizerParams } from "./validate";

export type SGDOptions = {
  lr: number;
  momentum?: number;
  weightDecay?: number;
};

/** Per-group overrides for SGD. Unset fields inherit from defaults. */
export type SGDParamGroup = {
  params: Tensor[];
  lr?: number;
  weightDecay?: number;
};

type ResolvedSGDGroup = {
  params: Tensor[];
  lr: number;
  weightDecay: number;
};

export class SGD {
  private params: Tensor[];
  private readonly api: Torchlette;
  private readonly momentum: number;
  private readonly device: import("../backend/types").DeviceKind;
  private _groups: ResolvedSGDGroup[];
  private _groupIndex: number[];
  private velocity: Array<RuntimeTensor | null>;

  constructor(
    params: Tensor[] | SGDParamGroup[],
    options: SGDOptions,
    api?: Torchlette,
  ) {
    const isGroups =
      params.length > 0 &&
      typeof params[0] === "object" &&
      "params" in params[0] &&
      Array.isArray((params[0] as SGDParamGroup).params);

    const flatParams: Tensor[] = isGroups
      ? (params as SGDParamGroup[]).flatMap((g) => g.params)
      : (params as Tensor[]);

    const { api: engine, device } = validateOptimizerParams(
      "SGD",
      flatParams,
      api,
    );
    this.device = device;
    if (options.lr <= 0) {
      throw new Error("SGD learning rate must be > 0");
    }
    this.api = engine;
    this.params = flatParams;
    this.momentum = options.momentum ?? 0;

    if (isGroups) {
      const groups = params as SGDParamGroup[];
      this._groups = groups.map((g) => ({
        params: g.params,
        lr: g.lr ?? options.lr,
        weightDecay: g.weightDecay ?? options.weightDecay ?? 0,
      }));
      this._groupIndex = [];
      for (let gi = 0; gi < groups.length; gi++) {
        for (let pi = 0; pi < groups[gi].params.length; pi++) {
          this._groupIndex.push(gi);
        }
      }
    } else {
      this._groups = [
        {
          params: flatParams,
          lr: options.lr,
          weightDecay: options.weightDecay ?? 0,
        },
      ];
      this._groupIndex = flatParams.map(() => 0);
    }

    this.velocity = new Array(flatParams.length).fill(null);
  }

  getParams(): Tensor[] {
    return this.params.slice();
  }

  getLR(): number {
    return this._groups[0].lr;
  }

  setLR(lr: number): void {
    for (const g of this._groups) g.lr = lr;
  }

  getParamGroupLRs(): number[] {
    return this._groups.map((g) => g.lr);
  }

  setGroupLR(groupIndex: number, lr: number): void {
    this._groups[groupIndex].lr = lr;
  }

  get numGroups(): number {
    return this._groups.length;
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
      const group = this._groups[this._groupIndex[i]];
      let gradAdj = grad;
      if (group.weightDecay !== 0) {
        const decay = runtime.mul(param._unwrap(), group.weightDecay);
        gradAdj = runtime.add(gradAdj, decay);
      }
      let update = gradAdj;
      if (this.momentum !== 0) {
        let v = this.velocity[i];
        if (!v) {
          // Lazy persistent state: created MID-STEP and held across steps —
          // persist() adopts it into the step snapshot (without it, markStep
          // demotes the tensor as a temporary and its buffer is pooled while
          // live: the silent-UAF class that corrupted per-param Adam, and
          // corrupted multi-param SGD here via the old replace-and-hold
          // `this.velocity[i] = update`).
          v = runtime.registerState(runtime.zeros(param.shape, this.device));
          this.velocity[i] = v;
        }
        // v = momentum*v + gradAdj, IN PLACE. `update` reads the POST-copy_
        // ref so the param update below depends on the state write (a
        // dangling-root copy_ defers to a later plan and reads freed grads).
        runtime.copy_(
          v,
          runtime.add(runtime.mul(v, this.momentum), gradAdj),
        );
        update = v;
      }
      // lr rides sub's alpha, which the runtime LOWERS to mul(update, lr) —
      // a graph scalar on the principled path (inlined while constant,
      // demoted to scalar-table data when an LR schedule changes it).
      const next = runtime.sub(param._unwrap(), update, {
        alpha: group.lr,
      });
      runtime.copy_(param._unwrap(), next);
      updated.push(param);
    }
    // Implied step boundary (minimal training loops): commits at the next
    // backward() or explicit markStep(). See Torchlette.queueStepBoundary.
    this.api.queueStepBoundary();
    return updated;
  }

  zeroGrad(): void {
    for (const param of this.params) {
      param.zeroGrad();
    }
  }
}
