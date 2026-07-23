import type { OptStepConfig } from "../backend/types";
import { ENV } from "../core/env";
import { LiveScalar } from "../core/live-scalar";
import type { Tensor, Torchlette } from "../frontend/torchlette";
import { createLazyIRNode } from "../graph/node-factory";
import { createPendingRef } from "../graph/types";
import {
  SGD_MOMENTUM_PROGRAM,
  SGD_PROGRAM,
} from "../ops/semantic/optimizer";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import { validateOptimizerParams } from "./validate";

// Structural identity of the fused SGD `optStep` nodes (derived-optimizer-realizer
// R5c), mirroring SGD_MOMENTUM_STEP_SPEC / SGD_STEP_SPEC in
// schedule/opt-step-specs.ts. With momentum the state is the velocity `v`
// (SGD_MOMENTUM_PROGRAM); without it there is no state (plain SGD). L2 weight
// decay folds into `g` via the realizer's `decoupled_wd==0` branch — so
// `decoupledWd` is always false. lr is the only scalar-DATA input.
const SGD_MOMENTUM_STATE_SLOTS: readonly string[] = SGD_MOMENTUM_PROGRAM.state; // ["v"]
const SGD_STATE_SLOTS: readonly string[] = SGD_PROGRAM.state; // []
const SGD_SCALAR_INPUTS: readonly string[] = ["lr"];

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
  /**
   * Per-group learning rate as a persistent on-device LIVE SCALAR
   * (core/live-scalar.ts) — the SAME primitive Adam's lr rides. The fused
   * `optStep` kernel reads it as scalar-DATA (a stable f32[1] buffer written
   * IN-PLACE per step), so an LR schedule flows through compiled replay as DATA
   * (the historical lr=1.0 frozen-scalar bug class is structurally impossible).
   * Created eagerly (device known at ctor); only read on the fused path.
   */
  private _lrLive: (LiveScalar | null)[];

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
    this._lrLive = this._groups.map(
      (g) => new LiveScalar(engine, g.lr, device as "cpu" | "webgpu"),
    );
  }

  getParams(): Tensor[] {
    return this.params.slice();
  }

  /** The live lr scalar for a group (created on demand). */
  private _lrScalar(gi: number): LiveScalar {
    let s = this._lrLive[gi];
    if (!s) {
      s = new LiveScalar(
        this.api,
        this._groups[gi].lr,
        this.device as "cpu" | "webgpu",
      );
      this._lrLive[gi] = s;
    }
    return s;
  }

  /** The persistent lr tensor (runtime) for a group — read by the fused optStep
   *  kernel as scalar-DATA. */
  private _lrTensor(gi: number): RuntimeTensor {
    return this._lrScalar(gi).tensor._unwrap();
  }

  /** Check whether the fused optimizer kernel is available on this backend. */
  hasFusedKernel(): boolean {
    const backend = this.api._runtime().getBackend(this.device);
    return !!backend.ops.optStep;
  }

  getLR(): number {
    return this._groups[0].lr;
  }

  /** Set learning rate for all groups. Writes the persistent lr LiveScalar
   *  IN-PLACE so a compiled replay sees the new value as DATA (the LR-schedule-
   *  exactness seam; the historical lr=1.0 bug class). */
  setLR(lr: number): void {
    for (let gi = 0; gi < this._groups.length; gi++) {
      this._groups[gi].lr = lr;
      this._lrScalar(gi).set(lr);
    }
  }

  getParamGroupLRs(): number[] {
    return this._groups.map((g) => g.lr);
  }

  setGroupLR(groupIndex: number, lr: number): void {
    this._groups[groupIndex].lr = lr;
    this._lrScalar(groupIndex).set(lr);
  }

  get numGroups(): number {
    return this._groups.length;
  }

  step(): Tensor[] {
    const runtime = this.api._runtime();
    // Path selection (pinned by tools/parity-packed-vs-unpacked.ts):
    //  - fused optStep kernel: the WebGPU default — SGD(+momentum) folded into the
    //    generic program-roles realizer (opt-step-realizer.ts), one DMA-packed
    //    in-place dispatch per size class. Requires >1 param.
    //  - per-param: the reference definition AND the CPU / non-fused fallback AND
    //    the TORCHLETTE_PACK_OPTIM=0 opt-out (the differential's unpacked arm).
    let updated: Tensor[];
    if (
      ENV.TORCHLETTE_PACK_OPTIM !== "0" &&
      this.params.length > 1 &&
      this.hasFusedKernel()
    ) {
      updated = this._stepFused(runtime);
    } else {
      updated = this._stepPerParam(runtime);
    }
    // Implied step boundary (minimal training loops): commits at the next
    // backward() or explicit markStep(). See Torchlette.queueStepBoundary.
    this.api.queueStepBoundary();
    return updated;
  }

  /**
   * Fused SGD step: one generic `optStep` node per grad-bearing param, folded
   * from SGD_MOMENTUM_PROGRAM (with momentum) or SGD_PROGRAM (without) by the
   * program-roles realizer (design ruling O1). The node carries the de-named
   * `OptStepConfig`; the executor batches same-size nodes into ONE packed DMA
   * dispatch and the backend derives every in-place / ownership decision from the
   * config's STRUCTURE (stateSlots `["v"]` with momentum, `[]` without). L2 weight
   * decay folds into `g` (the realizer's `decoupled_wd==0` branch), matching the
   * per-param path; lr rides as scalar-DATA.
   */
  private _stepFused(runtime: ReturnType<Torchlette["_runtime"]>): Tensor[] {
    const updated: Tensor[] = [];
    const hasMomentum = this.momentum !== 0;
    for (let i = 0; i < this.params.length; i++) {
      const param = this.params[i];
      const grad = param.grad?._unwrap() ?? null;
      if (!grad) {
        updated.push(param);
        continue;
      }

      const gi = this._groupIndex[i];
      const wd = this._groups[gi].weightDecay;

      // Lazy persistent velocity (momentum only). registerState adopts it into
      // the step snapshot (the silent-UAF discipline). The optStep kernel writes
      // `v` IN PLACE and the backend transfers its buffer ownership to the fresh
      // output at execution — never replace-and-hold.
      let v: RuntimeTensor | null = null;
      if (hasMomentum) {
        v = this.velocity[i];
        if (!v) {
          v = runtime.registerState(runtime.zeros(param.shape, this.device));
          this.velocity[i] = v;
        }
      }

      // emitF16:false — do NOT dual-write the f16 param shadow (unlike Adam); the
      // autocast forward recasts f32→f16 (as SGD's graph-cat predecessor did). A
      // per-step wash (the packed dispatch is emitF16=false regardless), and
      // dual-writing is incompatible with the differential's cross-optimizer
      // cache-reset (the shadow is an arena buffer bound by the persistent forward
      // compiled plan — see the note in lion.ts).
      const config: OptStepConfig = hasMomentum
        ? {
            spec: "sgd_momentum",
            stateSlots: SGD_MOMENTUM_STATE_SLOTS,
            scalarInputs: SGD_SCALAR_INPUTS,
            hypers: { mu: this.momentum, weight_decay: wd },
            decoupledWd: false,
            emitF16: false,
          }
        : {
            spec: "sgd",
            stateSlots: SGD_STATE_SLOTS,
            scalarInputs: SGD_SCALAR_INPUTS,
            hypers: { weight_decay: wd },
            decoupledWd: false,
            emitF16: false,
          };

      // Node inputs [grad, param, (v,) lr]. lr flows as a persistent scalar-DATA
      // tensor (stable buffer → TAG_WRITE, no repack).
      const lrRt = this._lrTensor(gi);
      const inputs = hasMomentum
        ? [grad.lazyRef, param._unwrap().lazyRef, v!.lazyRef, lrRt.lazyRef]
        : [grad.lazyRef, param._unwrap().lazyRef, lrRt.lazyRef];
      const node = createLazyIRNode(
        "optStep",
        inputs,
        param.shape,
        "f32",
        this.device,
        config,
      );

      // param (output 0) — and v (output 1) when momentum is on — point at the
      // node's in-place results.
      param._unwrap()._updateLazyRef(createPendingRef(node, 0));
      if (hasMomentum && v) {
        v._updateLazyRef(createPendingRef(node, 1));
        runtime.registerState(v);
      }
      updated.push(param);
    }

    // Persist the live lr scalar buffers across the step boundary.
    for (const s of this._lrLive)
      if (s) runtime.registerState(s.tensor._unwrap());
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
