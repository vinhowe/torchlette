import { ENV } from "../core/env";
import type { Tensor, Torchlette } from "../frontend/torchlette";
import {
  SGD_MOMENTUM_PROGRAM,
  SGD_PROGRAM,
} from "../ops/semantic/optimizer";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import {
  type PackedOptState,
  packOptimizerClass,
} from "./pack-optimizer";
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
  /** Packed foreach state, keyed by param-group index (chain-packing P2). */
  private _packState = new Map<number, PackedOptState[]>();

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
    // Packed path (chain-packing P2): one flat chain per isomorphism class via
    // packOptimizerProgram (the SAME SGD program the per-param path interprets).
    // Opt out with TORCHLETTE_PACK_OPTIM=0; a single param has no class to pack.
    let updated: Tensor[];
    if (ENV.TORCHLETTE_PACK_OPTIM !== "0" && this.params.length > 1) {
      updated = this._stepPacked(runtime);
    } else {
      updated = this._stepPerParam(runtime);
    }
    // Implied step boundary (minimal training loops): commits at the next
    // backward() or explicit markStep(). See Torchlette.queueStepBoundary.
    this.api.queueStepBoundary();
    return updated;
  }

  /**
   * Packed SGD step: group grad-bearing params by isomorphism class (param
   * group → shared lr/wd), then emit ONE flat chain per class. With momentum the
   * velocity is packed state (SGD_MOMENTUM_PROGRAM, v' read POST-copy_); without
   * it, a stateless `p' = p − lr·g` (SGD_PROGRAM). L2 weight decay folds into `g`
   * (adjustGrad) — the realizer's policy, matching the per-param path.
   */
  private _stepPacked(runtime: ReturnType<Torchlette["_runtime"]>): Tensor[] {
    const updated: Tensor[] = [];
    const hasMomentum = this.momentum !== 0;
    const program = hasMomentum ? SGD_MOMENTUM_PROGRAM : SGD_PROGRAM;
    const groups = new Map<number, number[]>();
    for (let i = 0; i < this.params.length; i++) {
      updated.push(this.params[i]);
      const grad = this.params[i].grad?._unwrap() ?? null;
      if (!grad) continue;
      if (hasMomentum && !this.velocity[i])
        this.velocity[i] = runtime.registerState(
          runtime.zeros(this.params[i].shape, this.device),
        );
      const gi = this._groupIndex[i];
      const list = groups.get(gi);
      if (list) list.push(i);
      else groups.set(gi, [i]);
    }
    for (const [gi, idxs] of groups) {
      const group = this._groups[gi];
      const wd = group.weightDecay;
      const st = packOptimizerClass(
        runtime,
        {
          program,
          items: idxs.map((i) => ({
            id: i,
            param: this.params[i]._unwrap(),
            grad: this.params[i].grad!._unwrap(),
            state: hasMomentum ? [this.velocity[i]!] : [],
          })),
          sharedRoles: { lr: group.lr, mu: this.momentum },
          // L2 folds wd into g (affects the velocity too, matching the per-param
          // path); the param term carries no wd.
          adjustGrad:
            wd !== 0 ? (rt, g, p) => rt.add(g, rt.mul(p, wd)) : undefined,
          paramReadsPostState: true,
        },
        this._packState.get(gi),
      );
      this._packState.set(gi, st);
    }
    return updated;
  }

  /** Per-param SGD step: the reference definition (also the >1-param opt-out). */
  private _stepPerParam(runtime: ReturnType<Torchlette["_runtime"]>): Tensor[] {
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
    return updated;
  }

  zeroGrad(): void {
    for (const param of this.params) {
      param.zeroGrad();
    }
  }
}
