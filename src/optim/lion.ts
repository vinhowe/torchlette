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

import type { DeviceKind, OptStepConfig } from "../backend/types";
import { ENV } from "../core/env";
import { LiveScalar } from "../core/live-scalar";
import type { Tensor, Torchlette } from "../frontend/torchlette";
import { createLazyIRNode } from "../graph/node-factory";
import { createPendingRef } from "../graph/types";
import {
  evalOptTensor,
  LION_M_NEW,
  LION_P_NEW,
  LION_PROGRAM,
  LION_STEP,
  type OptRoles,
  oSub,
  role,
} from "../ops/semantic/optimizer";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import { validateOptimizerParams } from "./validate";

/** The wd=0 param-update term `p' = p − lr·sign(c)` (no full-size `+0` at wd=0). */
const LION_P_NO_WD = oSub(role("p"), LION_STEP);

// Structural identity of the fused Lion `optStep` node (derived-optimizer-realizer
// R5c), mirroring LION_STEP_SPEC in schedule/opt-step-specs.ts — the SINGLE
// conceptual source. Lion's state is the single β2-EMA `m`; lr is its only scalar-
// DATA input (no bias correction). The backend resolves the realizer by `spec` and
// binds by these slots, so any divergence throws at the binding seam.
const LION_STATE_SLOTS: readonly string[] = LION_PROGRAM.state; // ["m"]
const LION_SCALAR_INPUTS: readonly string[] = ["lr"];

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
  /**
   * Per-group learning rate as a persistent on-device LIVE SCALAR
   * (core/live-scalar.ts) — the SAME primitive Adam's lr rides. The fused
   * `optStep` kernel reads it as scalar-DATA (a stable f32[1] buffer written
   * IN-PLACE per step), so an LR schedule flows through compiled replay as DATA
   * (the frozen-scalar class is structurally impossible). Created eagerly (device
   * known at ctor); only materialized/read on the fused path.
   */
  private _lrLive: (LiveScalar | null)[];

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
   *  exactness seam). */
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
    //  - fused optStep kernel: the WebGPU default — Lion's LION_PROGRAM folded
    //    into the generic program-roles realizer (opt-step-realizer.ts), one
    //    DMA-packed in-place dispatch per size class (memory/speed of the fused
    //    Adam path). Requires >1 param (nothing to pack otherwise).
    //  - per-param: the reference definition AND the CPU / non-fused fallback AND
    //    the TORCHLETTE_PACK_OPTIM=0 opt-out (the differential's unpacked arm).
    // TORCHLETTE_PACK_OPTIM=0 forces the per-param reference on every backend.
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
    // backward() or explicit markStep().
    this.api.queueStepBoundary();
    return updated;
  }

  /**
   * Fused Lion step: one generic `optStep` node per grad-bearing param, folded
   * from LION_PROGRAM by the program-roles realizer (design ruling O1). The node
   * carries the de-named `OptStepConfig` (`spec:"lion"`, `stateSlots:["m"]`,
   * `scalarInputs:["lr"]`); the executor batches same-size nodes into ONE packed
   * DMA dispatch and the backend derives every in-place / ownership decision from
   * the config's STRUCTURE. Lion's param term reads the OLD momentum
   * (paramReadsPostState=false in the spec); the stored β2-EMA `m'` is the state
   * output. wd is always decoupled (`decoupledWd:true`), lr rides as scalar-DATA.
   */
  private _stepFused(runtime: ReturnType<Torchlette["_runtime"]>): Tensor[] {
    const updated: Tensor[] = [];
    for (let i = 0; i < this.params.length; i++) {
      const param = this.params[i];
      const grad = param.grad?._unwrap() ?? null;
      if (!grad) {
        updated.push(param);
        continue;
      }

      // Lazy persistent momentum. registerState adopts it into the step snapshot
      // (the silent-UAF discipline — without it markStep demotes the mid-step-
      // created tensor and pools its live buffer). This is THE UAF surface: the
      // state slot must be copy_-in-place OR a properly registered replacement,
      // never replace-and-hold. The optStep kernel writes `m` IN PLACE and the
      // backend transfers its buffer ownership to the fresh output at execution.
      let m = this.momentum[i];
      if (!m) {
        m = runtime.registerState(runtime.zeros(param.shape, this.device));
        this.momentum[i] = m;
      }

      const gi = this._groupIndex[i];
      const wd = this._groups[gi].weightDecay;
      const config: OptStepConfig = {
        spec: "lion",
        stateSlots: LION_STATE_SLOTS,
        scalarInputs: LION_SCALAR_INPUTS,
        hypers: {
          beta1: this.beta1,
          beta2: this.beta2,
          weight_decay: wd,
        },
        decoupledWd: true,
        // emitF16:false — do NOT dual-write the f16 param shadow (unlike Adam).
        // The autocast forward recasts f32→f16 (exactly as Lion's graph-cat
        // predecessor did), which is a per-step WASH here anyway: the packed
        // dispatch hardcodes emitF16=false so only size-unique params would ever
        // get the kernel shadow. Dual-writing is also INCOMPATIBLE with the
        // cross-optimizer cache-reset the differential needs — the shadow is an
        // arena buffer bound by the persistent model-forward compiled plan;
        // deferredDestroy skips arena buffers (can't invalidate) and an immediate
        // destroy poisons the plan's next replay ("used in submit while destroyed").
        emitF16: false,
      };

      // 4-input optStep node [grad, param, m, lr]. lr flows as a persistent
      // scalar-DATA tensor (stable buffer → TAG_WRITE, no repack).
      const lrRt = this._lrTensor(gi);
      const node = createLazyIRNode(
        "optStep",
        [
          grad.lazyRef,
          param._unwrap().lazyRef,
          m.lazyRef,
          lrRt.lazyRef,
        ],
        param.shape,
        "f32",
        this.device,
        config,
      );

      // param (output 0) and m (output 1) point at the node's in-place results.
      param._unwrap()._updateLazyRef(createPendingRef(node, 0));
      m._updateLazyRef(createPendingRef(node, 1));

      // Persist m (lazily materializes mid-step — same adoption rationale as the
      // per-param path).
      runtime.registerState(m);
      updated.push(param);
    }

    // Persist the live lr scalar buffers across the step boundary.
    for (const s of this._lrLive)
      if (s) runtime.registerState(s.tensor._unwrap());
    return updated;
  }

  /** Per-param Lion step: the reference definition (also the >1-param opt-out). */
  private _stepPerParam(runtime: ReturnType<Torchlette["_runtime"]>): Tensor[] {
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
    return updated;
  }

  zeroGrad(): void {
    for (const param of this.params) param.zeroGrad();
  }
}
