/**
 * Lion (Chen et al. 2023, "Symbolic Discovery of Optimization Algorithms").
 *
 * The GENERALITY DIVIDEND of P5 (docs/semantic-derivation-design.md §5, §14): a
 * complete optimizer realized from a definition ALONE — `LION_PROGRAM`
 * (src/ops/semantic/optimizer.ts) — with NO hand kernel and NO hand grads. Its
 * sign-based step falls out of the same primitive algebra AdamW derives from;
 * this realizer only sequences the in-place m-state + step-boundary EFFECTS
 * (design §4.6). That a brand-new optimizer needed no new engine is the payoff.
 *
 *   c   = β1·m + (1−β1)·g              (the step direction — a β1 interpolation)
 *   p'  = p − lr·( sign(c) + wd·p )     (decoupled weight decay)
 *   m'  = β2·m + (1−β2)·g              (the stored EMA — a DIFFERENT, β2, average)
 *
 * The param update reads the OLD momentum (its direction is the β1-interp of it);
 * the stored `m'` is the separate β2-EMA — so `paramUpdate` binds `m` to the
 * pre-update state, and `m` is written in place afterwards (its copy_ is forced
 * at the step boundary's `forceAllPending`, exactly as SGD's velocity copy_ is).
 */

import type { DeviceKind } from "../backend/types";
import type { Tensor, Torchlette } from "../frontend/torchlette";
import {
  evalOptTensor,
  LION_M_NEW,
  LION_P_NEW,
  LION_STEP,
  type OptRoles,
} from "../ops/semantic/optimizer";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import { validateOptimizerParams } from "./validate";

export type LionOptions = {
  lr: number;
  /** [β1, β2]. β1 weights the SIGN step direction; β2 the stored EMA. */
  betas?: [number, number];
  /** Decoupled weight decay (Lion has no L2 variant — wd is always decoupled). */
  weightDecay?: number;
};

/** Per-group overrides for Lion. Unset fields inherit from defaults. */
export type LionParamGroup = {
  params: Tensor[];
  lr?: number;
  weightDecay?: number;
};

type ResolvedLionGroup = {
  params: Tensor[];
  lr: number;
  weightDecay: number;
};

export class Lion {
  private params: Tensor[];
  private readonly api: Torchlette;
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly device: DeviceKind;
  private _groups: ResolvedLionGroup[];
  private _groupIndex: number[];
  /** Per-param β2-EMA momentum `m`. Lazily created (persistent state). */
  private momentum: Array<RuntimeTensor | null>;

  constructor(
    params: Tensor[] | LionParamGroup[],
    options: LionOptions,
    api?: Torchlette,
  ) {
    const isGroups =
      params.length > 0 &&
      typeof params[0] === "object" &&
      "params" in params[0] &&
      Array.isArray((params[0] as LionParamGroup).params);

    const flatParams: Tensor[] = isGroups
      ? (params as LionParamGroup[]).flatMap((g) => g.params)
      : (params as Tensor[]);

    const { api: engine, device } = validateOptimizerParams(
      "Lion",
      flatParams,
      api,
    );
    if (options.lr <= 0) throw new Error("Lion learning rate must be > 0");
    const betas = options.betas ?? [0.9, 0.99];
    if (betas.length !== 2) throw new Error("Lion betas must have two entries");
    const [beta1, beta2] = betas;
    if (beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1)
      throw new Error("Lion betas must be in the range [0, 1)");

    this.api = engine;
    this.device = device;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.params = flatParams;

    if (isGroups) {
      const groups = params as LionParamGroup[];
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

    this.momentum = new Array(flatParams.length).fill(null);
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
    for (let i = 0; i < this.params.length; i++) {
      const param = this.params[i];
      const grad = param.grad?._unwrap() ?? null;
      if (!grad) {
        updated.push(param);
        continue;
      }
      const group = this._groups[this._groupIndex[i]];

      // Lazy persistent momentum. registerState adopts it into the step
      // snapshot (without it, markStep demotes the mid-step-created tensor and
      // pools its buffer while live — the silent-UAF class SGD/Adam hit).
      let m = this.momentum[i];
      if (!m) {
        m = runtime.registerState(runtime.zeros(param.shape, this.device));
        this.momentum[i] = m;
      }

      // Roles bind `m` to the OLD momentum (the param term reads the β1-interp
      // direction of it; the stored EMA is the separate β2 update).
      const roles: OptRoles = {
        p: param._unwrap(),
        m,
        g: grad,
        lr: group.lr,
        wd: group.weightDecay,
        beta1: this.beta1,
        om_beta1: 1 - this.beta1,
        beta2: this.beta2,
        om_beta2: 1 - this.beta2,
      };

      // p' = p − lr·(sign(c) + wd·p) — the full term when wd fires; else derive
      // only the signed step (LION_STEP) and subtract (no full-size `+0`).
      const pNew =
        group.weightDecay !== 0
          ? evalOptTensor(LION_P_NEW, runtime, roles)
          : runtime.sub(
              param._unwrap(),
              evalOptTensor(LION_STEP, runtime, roles),
            );

      // m' = β2·m + (1−β2)·g, IN PLACE. Both pNew and mNew read the OLD m (built
      // before this copy_ reassigns m's lazy ref); the copy_ is forced at the
      // step boundary's forceAllPending (SGD's velocity discipline).
      const mNew = evalOptTensor(LION_M_NEW, runtime, roles);
      runtime.copy_(m, mNew);
      runtime.copy_(param._unwrap(), pNew);
      updated.push(param);
    }
    // Implied step boundary (minimal training loops): commits at the next
    // backward() or explicit markStep().
    this.api.queueStepBoundary();
    return updated;
  }

  zeroGrad(): void {
    for (const param of this.params) param.zeroGrad();
  }
}
