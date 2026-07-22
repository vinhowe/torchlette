import type { AdamStepConfig, DeviceKind } from "../backend/types";
import { ENV } from "../core/env";
import { LiveScalar } from "../core/live-scalar";
import type { Tensor, Torchlette } from "../frontend/torchlette";
import { createLazyIRNode } from "../graph/node-factory";
import { createPendingRef } from "../graph/types";
import {
  ADAMW_M_NEW,
  ADAMW_P_NEW,
  ADAMW_PROGRAM,
  ADAMW_SCALED,
  ADAMW_V_NEW,
  evalOptTensor,
  type OptRoles,
  oSub,
  role,
} from "../ops/semantic/optimizer";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import {
  type PackedOptState,
  packOptimizerClass,
} from "./pack-optimizer";
import { validateOptimizerParams } from "./validate";

/**
 * The wd=0 param-update term: `p' = p − lr·m̂/(√v̂+ε)` (design §2.5). Evaluating
 * `oSub(role("p"), ADAMW_SCALED)` is byte-identical to the hand
 * `runtime.sub(P, eval(ADAMW_SCALED))` the foreach path used — avoiding the
 * full-model-size `+0` the decoupled term would materialize at wd=0.
 */
const ADAMW_P_NO_WD = oSub(role("p"), ADAMW_SCALED);

export type AdamOptions = {
  lr: number;
  betas?: [number, number];
  eps?: number;
  weightDecay?: number;
  /** Use decoupled weight decay (AdamW). Default: false (L2 regularization). */
  adamW?: boolean;
};

/** Per-group overrides for Adam/AdamW. Unset fields inherit from defaults. */
export type AdamParamGroup = {
  params: Tensor[];
  lr?: number;
  weightDecay?: number;
};

/** Resolved internal group with all fields populated. */
type ResolvedAdamGroup = {
  params: Tensor[];
  lr: number;
  weightDecay: number;
};

export class Adam {
  private params: Tensor[];
  private readonly api: Torchlette;
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly eps: number;
  private readonly adamW: boolean;
  private readonly device: DeviceKind;
  /** Per-group hyperparameters. Single-group mode has exactly one entry. */
  private _groups: ResolvedAdamGroup[];
  /** Maps flat param index → group index. */
  private _groupIndex: number[];
  private expAvg: RuntimeTensor[];
  private expAvgSq: RuntimeTensor[];
  private steps: number[];
  /**
   * inc-2a (capturable-optimizer contract): the step counter `t` and the
   * per-group learning rate flow as persistent on-device f32 [1] tensors
   * (DATA), not as per-step-varying node payload scalars. The fused kernel
   * and the graph paths derive the bias-corrected step size from `t`/`lr`
   * (expm1-form; see `_biasCorrection`). `t` is advanced IN-PLAN by
   * copy_(t, add(t,1)) inside the optimizer plan so replays advance it.
   * Per-param `steps[]` collapses to the shared `t` (PyTorch capturable
   * semantics: a param whose grad is absent skips its update but `t` still
   * advances). `_t`/`_lrLive` are lazily created on first step (device
   * known once a real param is stepped) — mirrors m/v lazy init. lr rides the
   * LIVE SCALAR SLOT primitive (see `_lrLive`).
   */
  private _t: RuntimeTensor | null = null;
  /** inc-2a lr now rides the LIVE SCALAR SLOT primitive (core/live-scalar.ts):
   *  a persistent f32[1] whose per-step value the scheduler delivers via an
   *  in-place, graph-ordered write into its fixed buffer (buffer-stable, live
   *  across replays). The primitive owns what setLR used to open-code (the
   *  copy_ scatter + scalar-slots note). Lazily created (device known). */
  private _lrLive: (LiveScalar | null)[];
  /**
   * R3/fork-C: the derived fused body takes bias correction as a `[2]`
   * `bc`=[bc1,bc2] DATA input delivered as a HOST-computed LIVE SCALAR — the
   * exact primitive `lr` rides (core/live-scalar.ts), a persistent [2] buffer
   * host-written each step through the graph-ordered in-plan scatter, live under
   * compiled replay via the scalar-slots re-dress. bc is computed on the HOST
   * from the JS step counter (`_tHost`) via `_bcHost` (`expm1F32`, the SAME
   * 5-term Horner / exp branch the authored kernel emits) — so the derived path
   * adds ZERO optimizer-plan expm1 nodes (fork B's ~18-node graph prelude is
   * gone; the reclaim-boundary quantization it tripped on gpt2-medium is dodged
   * by construction). Precision blocker CLEARED: host-vs-GPU bc agrees to ≤1 ULP
   * (the exp intrinsic's divergence is absorbed by bc=1−exp(y) → saturates to 1
   * in the deep tail); propagated Δparam ≤5.96e-8, Δm=Δv bit-exact. Lazily
   * created (device known on first step). Only allocated when
   * TORCHLETTE_DERIVED_ADAM is on.
   */
  private _bcLive: LiveScalar | null = null;
  /** HOST mirror of the shared on-device step counter `t`, advanced in lockstep
   *  by `_advanceT`. Fork C computes bc from THIS (host expm1F32), so no graph
   *  read of the device `t` is needed on the derived path. */
  private _tHost = 0;
  /**
   * Packed foreach state, keyed by group index. While the foreach path is
   * active, m/v live as ONE flat tensor per group (the per-param expAvg
   * arrays are consumed at first pack and become stale). Mirrors PyTorch's
   * foreach optimizers: the per-param definition is the semantics, the
   * packed form is the batched execution of the same tensor program.
   */
  private _foreachState = new Map<number, PackedOptState[]>();
  constructor(
    params: Tensor[] | AdamParamGroup[],
    options: AdamOptions,
    api?: Torchlette,
  ) {
    // Detect whether first arg is param groups or flat params
    const isGroups =
      params.length > 0 &&
      typeof params[0] === "object" &&
      "params" in params[0] &&
      Array.isArray((params[0] as AdamParamGroup).params);

    const flatParams: Tensor[] = isGroups
      ? (params as AdamParamGroup[]).flatMap((g) => g.params)
      : (params as Tensor[]);

    const { api: engine, device } = validateOptimizerParams(
      "Adam",
      flatParams,
      api,
    );
    if (options.lr <= 0) {
      throw new Error("Adam learning rate must be > 0");
    }
    const betas = options.betas ?? [0.9, 0.999];
    if (betas.length !== 2) {
      throw new Error("Adam betas must have two entries");
    }
    const [beta1, beta2] = betas;
    if (beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1) {
      throw new Error("Adam betas must be in the range [0, 1)");
    }
    const eps = options.eps ?? 1e-8;
    if (eps <= 0) {
      throw new Error("Adam eps must be > 0");
    }

    this.api = engine;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.params = flatParams;
    this.adamW = options.adamW ?? false;
    this.device = device;

    // Build groups
    if (isGroups) {
      const groups = params as AdamParamGroup[];
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

    const runtime = engine._runtime();
    this.expAvg = flatParams.map((p) => runtime.zeros(p.shape, device));
    this.expAvgSq = flatParams.map((p) => runtime.zeros(p.shape, device));
    this.steps = new Array(flatParams.length).fill(0);
    // inc-2a: persistent step counter (shared) + per-group lr tensors.
    // Created eagerly (device known); f32 [1]. Materialize on first force.
    this._t = runtime.full([1], 0, device, "f32");
    this._lrLive = this._groups.map(
      (g) => new LiveScalar(engine, g.lr, device as "cpu" | "webgpu"),
    );
  }

  /** The persistent step-counter tensor (lazily kept non-null after ctor). */
  private _tTensor(): RuntimeTensor {
    if (!this._t) {
      this._t = this.api._runtime().full([1], 0, this.device, "f32");
    }
    return this._t;
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

  /** The persistent lr tensor (runtime) for a group — read by adamStep / the
   *  elementwise update as an ordinary graph input. */
  private _lrTensor(gi: number): RuntimeTensor {
    return this._lrScalar(gi).tensor._unwrap();
  }

  getParams(): Tensor[] {
    return this.params.slice();
  }

  /**
   * Return all tensors that must survive across step boundaries:
   * parameters + optimizer state (momentum m, variance v).
   * Used by remote engines for markStep() handle retention.
   */
  getAllKeepTensors(): Tensor[] {
    const keep: Tensor[] = [...this.params];
    for (let i = 0; i < this.expAvg.length; i++) {
      keep.push(this.api._wrapRuntime(this.expAvg[i], false));
      keep.push(this.api._wrapRuntime(this.expAvgSq[i], false));
    }
    // inc-2a: persistent t/lr must survive step boundaries too.
    if (this._t) keep.push(this.api._wrapRuntime(this._t, false));
    for (const s of this._lrLive) {
      if (s) keep.push(s.tensor);
    }
    // R3/fork-C derived-path persistent bc live buffer.
    if (this._bcLive) keep.push(this._bcLive.tensor);
    return keep;
  }

  /** Get the default (first group) learning rate. */
  getLR(): number {
    return this._groups[0].lr;
  }

  /** Set learning rate for all parameter groups. Writes the persistent lr
   *  tensors IN-PLACE (copy_) so a compiled replay sees the new value as DATA
   *  — the LR-schedule-exactness seam (schedulers funnel through here). */
  setLR(lr: number): void {
    for (let gi = 0; gi < this._groups.length; gi++) {
      this._groups[gi].lr = lr;
      // LIVE SCALAR SLOT delivery (core/live-scalar.ts): an in-place,
      // graph-ordered write into the lr tensor's FIXED buffer, re-dressed per
      // replay from the registry — the single primitive that carries the
      // per-step lr through compiled replays as DATA.
      this._lrScalar(gi).set(lr);
    }
  }

  /** Get per-group learning rates. */
  getParamGroupLRs(): number[] {
    return this._groups.map((g) => g.lr);
  }

  /** Set learning rate for a specific parameter group. Writes the persistent
   *  lr tensor IN-PLACE (copy_). */
  setGroupLR(groupIndex: number, lr: number): void {
    this._groups[groupIndex].lr = lr;
    // LIVE SCALAR SLOT delivery — see setLR.
    this._lrScalar(groupIndex).set(lr);
  }

  /** Get the number of parameter groups. */
  get numGroups(): number {
    return this._groups.length;
  }

  /** Get per-param weight decay for the fused kernel. */
  private _getParamWeightDecay(i: number): number {
    return this._groups[this._groupIndex[i]].weightDecay;
  }

  /**
   * Check if the fused Adam kernel is available on the current backend.
   */
  hasFusedKernel(): boolean {
    const runtime = this.api._runtime();
    const backend = runtime.getBackend(this.device);
    return !!backend.ops.adamStep;
  }

  step(): Tensor[] {
    const runtime = this.api._runtime();

    // Path selection (each pinned to the others by
    // test/optim/fused-vs-elementwise.spec.ts):
    //  - fused WGSL kernel (WebGPU default; TORCHLETTE_FUSED_ADAM=0 disables)
    //  - foreach: the per-param tensor program batched over ONE packed flat
    //    tensor per group — pure graph ops, so fusion/compiled-plan/scalar
    //    table all apply, at ~constant op count instead of ~13 ops per param
    //    (TORCHLETTE_FOREACH_ADAM=0 disables)
    //  - per-param elementwise: the reference definition.
    //
    // Foreach-as-default is BLOCKED by the buffer arena, not by foreach:
    // measured 2026-06-10 (distilgpt2@512), foreach == fused to 1.5e-5/30
    // steps fp32 fullstack, but the default arena gives each of foreach's
    // ~30 full-model-size graph intermediates a PERSISTENT per-position
    // slot: 20.3GB vs 9.1GB fused (under TORCHLETTE_ARENA_LIVENESS=1 it's
    // 2.5GB current — the program is fine, the memory policy isn't). Flip
    // the default once bounded-memory compiled execution lands
    // (docs/architecture-debt.md, planned compiled buffers).
    let updated: Tensor[];
    if (this.hasFusedKernel() && ENV.TORCHLETTE_FUSED_ADAM !== "0") {
      updated = this._stepFused(runtime);
    } else if (ENV.TORCHLETTE_FOREACH_ADAM !== "0" && this.params.length > 1) {
      updated = this._stepForeach(runtime);
    } else {
      updated = this._stepElementwise(runtime);
    }
    // Implied step boundary (minimal training loops): commits at the next
    // backward() or explicit markStep(). See Torchlette.queueStepBoundary.
    this.api.queueStepBoundary();
    return updated;
  }

  /**
   * Foreach Adam step: the SAME tensor program as `_updateParamElementwise`,
   * executed once per parameter GROUP over packed flat tensors instead of
   * once per parameter. Packing is graph-level (reshape + cat + narrow +
   * copy_), so every downstream system sees ordinary tensor ops: vertical
   * fusion collapses the arithmetic chain, the scalar table keeps the
   * per-step coefficients honest under template/compiled caching, and the
   * compiled plan replays the whole thing without optimizer-specific hooks.
   *
   * m/v state lives permanently packed (no per-step state copies); only the
   * grads and params are packed in (N segment copies) and the updated params
   * copied back out (N segment copies).
   */
  private _stepForeach(runtime: ReturnType<Torchlette["_runtime"]>): Tensor[] {
    const updated: Tensor[] = [];
    const groups = new Map<number, number[]>();
    for (let i = 0; i < this.params.length; i++) {
      updated.push(this.params[i]);
      const grad = this.params[i].grad?._unwrap() ?? null;
      if (!grad) continue;
      const gi = this._groupIndex[i];
      const list = groups.get(gi);
      if (list) list.push(i);
      else groups.set(gi, [i]);
    }
    // inc-2a: advance the shared on-device `t` ONCE per step (before any
    // group's update graph reads it).
    this._advanceT(runtime);
    for (const [gi, idxs] of groups) {
      this._foreachGroupStep(runtime, gi, idxs);
    }
    // Persist t/lr (materialize mid-step).
    runtime.registerState(this._tTensor());
    for (const s of this._lrLive)
      if (s) runtime.registerState(s.tensor._unwrap());
    return updated;
  }

  private _foreachGroupStep(
    runtime: ReturnType<Torchlette["_runtime"]>,
    gi: number,
    idxs: number[],
  ): void {
    // JS step mirror (diagnostics + intermittent-missing-grad detection). The
    // packer re-derives the same guard from item `id`s (sig mismatch throws),
    // but the JS mirror also catches diverging step counts BEFORE the pack.
    for (const i of idxs) this._advanceStep(i);
    const tScalar = this.steps[idxs[0]];
    for (const i of idxs) {
      if (this.steps[i] !== tScalar) {
        throw new Error(
          "Adam foreach: params in one group have diverging step counts " +
            "(gradients intermittently missing for some params). Set " +
            "TORCHLETTE_FOREACH_ADAM=0 to use the per-param path.",
        );
      }
    }
    // inc-2a: lr flows as the persistent per-group tensor; wd stays a JS scalar.
    const lrTensor = this._lrTensor(gi);
    const wd = this._groups[gi].weightDecay;
    // inc-2a: bias correction rides in as bc1/bc2 (the step as DATA); [1] tensors
    // that broadcast over the packed [total]. Disposed by the packer.
    const { bc1, bc2 } = this._biasCorrection(runtime, this._tTensor());
    const isL2 = wd !== 0 && !this.adamW;

    // Declare the class to the packer (design §2): AdamW's program + the shared
    // per-step DATA + the per-param state; L2-vs-decoupled weight decay is the
    // realizer's POLICY (L2 folds wd into g via adjustGrad; AdamW binds it in the
    // param term). The pack/copy_/persist/dispose EFFECTS are the packer's.
    const st = packOptimizerClass(
      runtime,
      {
        program: ADAMW_PROGRAM,
        items: idxs.map((i) => ({
          id: i,
          param: this.params[i]._unwrap(),
          grad: this.params[i].grad!._unwrap(),
          state: [this.expAvg[i], this.expAvgSq[i]],
        })),
        sharedRoles: {
          lr: lrTensor,
          eps: this.eps,
          wd: this.adamW ? wd : 0,
          beta1: this.beta1,
          om_beta1: 1 - this.beta1,
          beta2: this.beta2,
          om_beta2: 1 - this.beta2,
          bc1,
          bc2,
        },
        // The full p' (ADAMW_P_NEW) folds in the decoupled `lr·wd·p`; the no-wd
        // path derives just the update magnitude (ADAMW_SCALED) and subtracts.
        paramUpdate: this.adamW && wd !== 0 ? ADAMW_P_NEW : ADAMW_P_NO_WD,
        adjustGrad: isL2
          ? (rt, g, p) => rt.add(g, rt.mul(p, wd))
          : undefined,
        paramReadsPostState: true,
        disposeExtra: [bc1, bc2],
      },
      this._foreachState.get(gi),
    );
    this._foreachState.set(gi, st);
  }

  /**
   * Fused Adam step: one dispatch per parameter.
   * (The kernel also supports a fused unscale+inf-check variant via
   * AdamStepConfig.invScale/infFlagBuffer; GradScaler unscales through
   * graph-level unscaleGrad nodes instead, so nothing engages it here.)
   */
  private _stepFused(runtime: ReturnType<Torchlette["_runtime"]>): Tensor[] {
    const updated: Tensor[] = [];

    // inc-2a: advance the SHARED on-device step counter ONCE per step, before
    // the param loop, so every param's adamStep node reads the post-increment
    // `t` (the kernel derives bias correction from it). t is advanced in-plan.
    this._advanceT(runtime);
    const tRt = this._tTensor();

    // R3/fork-C: when the DERIVED body is selected, bias correction rides into
    // the kernel as a [2] `bc`=[bc1,bc2] DATA input at the `t` slot, delivered as
    // a HOST-computed LIVE SCALAR (the `lr` primitive). It is computed on the
    // HOST from `_tHost` (`_bcHost`) and written into the persistent [2] buffer
    // via the in-plan graph-ordered scatter — ZERO expm1 graph nodes (fork B's
    // ~18-node prelude gone). The buffer is SHARED by every param's node (like
    // `t`/`lr`), so the adam-batch grouping key (shared input[4] identity) still
    // packs the group; the scatter re-executes under compiled replay from the
    // re-noted host value (scalar-slots re-dress), exactly the `lr` discipline.
    // R3 FLIP (2026-07-22): fork C is the DEFAULT. Opt out with
    // TORCHLETTE_DERIVED_ADAM=0 (sunset at R4, when the authored body is deleted).
    const derived = ENV.TORCHLETTE_DERIVED_ADAM !== "0";
    let biasRt = tRt;
    if (derived) {
      if (!this._bcLive) {
        // Seed the persistent [2] buffer; `.set()` below records the in-plan
        // scatter EVERY step (bc changes each step, so — unlike a constant lr —
        // the scatter must be in the recorded template for the replay re-dress).
        this._bcLive = new LiveScalar(
          this.api,
          [0, 0],
          this.device as "cpu" | "webgpu",
        );
      }
      this._bcLive.set(this._bcHost());
      biasRt = this._bcLive.tensor._unwrap();
    }

    for (let i = 0; i < this.params.length; i++) {
      const param = this.params[i];
      const grad = param.grad?._unwrap() ?? null;
      if (!grad) {
        updated.push(param);
        continue;
      }

      // JS mirror advance (diagnostics + foreach divergence assert only).
      this._advanceStep(i);

      const gi = this._groupIndex[i];
      const wd = this._getParamWeightDecay(i);
      // inc-2a: config is FULLY STATIC — no stepSize/lrTimesWd. eps is the
      // ORIGINAL value; the kernel derives eps*sqrt(bc2) and lr*wd from the
      // t/lr tensor inputs.
      const config: AdamStepConfig = {
        beta1: this.beta1,
        beta2: this.beta2,
        eps: this.eps,
        weightDecay: wd,
        decoupledWd: this.adamW,
        emitF16: true,
      };

      // inc-2a: 6-input adamStep node [grad, param, m, v, t, lr]. t/lr flow as
      // persistent tensor DATA (stable buffers → TAG_WRITE, no volatile repack).
      const lrRt = this._lrTensor(gi);
      const adamNode = createLazyIRNode(
        "adamStep",
        [
          grad.lazyRef,
          param._unwrap().lazyRef,
          this.expAvg[i].lazyRef,
          this.expAvgSq[i].lazyRef,
          biasRt.lazyRef,
          lrRt.lazyRef,
        ],
        param.shape,
        "f32",
        this.device,
        config,
      );

      const paramRt = param._unwrap();

      // Update param (output 0), m (output 1), v (output 2) to point at
      // the adamStep node's results. The kernel writes param/m/v in place;
      // the backend transfers each input buffer's ownership to the fresh
      // output storage AT EXECUTION (neuters the old backendTensor's destroy
      // + decRefs the buffer — src/backend/webgpu/ops/fused.ts). So no
      // optimizer-level buffer-preservation hack is needed here: the old
      // param storage's destroy is already a no-op by the time
      // destroyUnreachable can run (post-force), and plan inputs are rc-held
      // through execution (retainPlanInputRefs). The neuter trick this code
      // used to carry was redundant with that backend step and is gone.
      paramRt._updateLazyRef(createPendingRef(adamNode, 0));
      this.expAvg[i]._updateLazyRef(createPendingRef(adamNode, 1));
      this.expAvgSq[i]._updateLazyRef(createPendingRef(adamNode, 2));

      // Persist m/v (design: docs/scoped-memory-design.md §6 — "optimizer
      // state adopts at creation"). m/v are lazily created (constructor
      // zeros) and first MATERIALIZE mid-step/mid-scope, so they are not in
      // the step/scope snapshot; without adoption releaseStepTemps demotes
      // them at the boundary (buffer pooled while the optimizer still points
      // at it → silent UAF → NaN on the next step). Under markStep the next
      // beginStep re-snapshots them; under api.scope() the snapshot is fixed
      // at scope entry, so the adoption is what makes scope() a real
      // beginStep/endStep replacement. Idempotent (WeakSet add); mirrors the
      // foreach path's runtime.registerState() of its packed state.
      runtime.registerState(this.expAvg[i]);
      runtime.registerState(this.expAvgSq[i]);

      updated.push(param);
    }

    // Persist t/lr (lazily materialize mid-step; same rationale as m/v).
    runtime.registerState(tRt);
    for (const s of this._lrLive)
      if (s) runtime.registerState(s.tensor._unwrap());
    // Persist the derived-path bc live buffer (same lazy-mid-step adoption
    // rationale as t/lr/m/v — buffer-stable across replays).
    if (this._bcLive) runtime.registerState(this._bcLive.tensor._unwrap());

    return updated;
  }

  /** Advance step counter. */
  private _advanceStep(i: number): void {
    this.steps[i] += 1;
  }

  /**
   * inc-2a: advance the persistent on-device step counter `t` by 1, IN-PLAN
   * (copy_(t, add(t,1))) so compiled-plan replays advance it too. Idempotent
   * per step-boundary: call ONCE per optimizer step (before building the
   * update graph that reads t). The JS `steps[]` mirror is kept for the
   * foreach-group divergence assert and diagnostics only.
   */
  private _advanceT(runtime: ReturnType<Torchlette["_runtime"]>): void {
    const t = this._tTensor();
    runtime.copy_(t, runtime.add(t, 1));
    // Host mirror advanced in lockstep (fork C reads _tHost for host-side bc).
    this._tHost += 1;
  }

  /**
   * Fork C host-side bias correction: bc=[bc1,bc2] = [-expm1(t·lnβ1),
   * -expm1(t·lnβ2)] computed in f32 (Math.fround discipline) via the SAME
   * 5-term Horner (|y|<0.25) / exp(y)−1 (else) branch the authored kernel emits
   * in `emitExpm1` and the graph `_biasCorrection` uses — single source of the
   * bc formula across all three realizations. `t = _tHost` (the post-advance
   * shared step counter). Named precision lemma (host-vs-GPU exp, bc-absorbed):
   * the Horner branch is bit-exact host-vs-GPU; the exp branch's ≤ few-ULP
   * intrinsic divergence is absorbed by bc=1−exp(y) (saturates to 1.0 in the
   * deep tail) → bc agrees to ≤1 ULP, Δparam ≤5.96e-8.
   */
  private _bcHost(): [number, number] {
    const f = Math.fround;
    const expm1F32 = (y: number): number => {
      if (Math.abs(y) < 0.25) {
        let r = f(1 / 120);
        r = f(f(1 / 24) + f(y * r));
        r = f(f(1 / 6) + f(y * r));
        r = f(f(1 / 2) + f(y * r));
        r = f(1 + f(y * r));
        return f(y * r);
      }
      return f(Math.exp(y) - 1);
    };
    const t = this._tHost;
    const bc1 = f(-expm1F32(f(t * f(Math.log(this.beta1)))));
    const bc2 = f(-expm1F32(f(t * f(Math.log(this.beta2)))));
    return [bc1, bc2];
  }

  /**
   * Shared expm1-form bias-correction subgraph. ONE derivation source across
   * foreach and elementwise (the fused kernel emits the identical WGSL in
   * `emitExpm1`/`emitBiasCorrection`; Gate-2 pins the numerics). Given the
   * persistent `t` tensor, returns bc1 = -expm1(t*lnB1), bc2 = -expm1(t*lnB2)
   * as graph tensors.
   *
   *   expm1(y):  |y| < 0.25 → 5-term Horner series
   *                y*(1 + y*(1/2 + y*(1/6 + y*(1/24 + y/120))))
   *              else       → exp(y) - 1
   *
   * lnB* are precomputed f64→f32 (Math.fround) to match the kernel's static
   * uniforms. `t` (≤ 2^24 steps) is exact.
   */
  private _biasCorrection(
    runtime: ReturnType<Torchlette["_runtime"]>,
    t: RuntimeTensor,
  ): { bc1: RuntimeTensor; bc2: RuntimeTensor } {
    const expm1 = (lnBeta: number): RuntimeTensor => {
      const y = runtime.mul(t, lnBeta); // t * lnBeta  (≤ 0)
      // 5-term Horner series (small-|y| branch).
      let r: RuntimeTensor = runtime.full([1], 1 / 120, this.device, "f32");
      r = runtime.add(runtime.mul(r, y), 1 / 24);
      r = runtime.add(runtime.mul(r, y), 1 / 6);
      r = runtime.add(runtime.mul(r, y), 1 / 2);
      r = runtime.add(runtime.mul(r, y), 1);
      const series = runtime.mul(y, r);
      const large = runtime.sub(runtime.exp(y), 1);
      const cond = runtime.lt(runtime.abs(y), 0.25);
      return runtime.where(cond, series, large);
    };
    const lnB1 = Math.fround(Math.log(this.beta1));
    const lnB2 = Math.fround(Math.log(this.beta2));
    const bc1 = runtime.neg(expm1(lnB1));
    const bc2 = runtime.neg(expm1(lnB2));
    return { bc1, bc2 };
  }

  /**
   * Update a single parameter using elementwise ops (shared by sync and async paths).
   */
  private _updateParamElementwise(
    runtime: ReturnType<Torchlette["_runtime"]>,
    i: number,
    param: Tensor,
    grad: RuntimeTensor,
  ): void {
    if (ENV.TORCHLETTE_DEBUG_ADAM_BUFS) {
      const bid = (t: RuntimeTensor): string => {
        const bt = (t as unknown as { backendTensor?: { buffer?: object } })
          .backendTensor;
        if (!bt?.buffer) return "pending";
        let id = _adamDbgIds.get(bt.buffer);
        if (id === undefined) {
          id = _adamDbgNext++;
          _adamDbgIds.set(bt.buffer, id);
        }
        return `b${id}`;
      };
      console.log(
        `[adamdbg] t=${this.steps[i] + 1} i=${i} param=${bid(param._unwrap())} grad=${bid(grad)} m=${bid(this.expAvg[i])} v=${bid(this.expAvgSq[i])}`,
      );
    }
    this._advanceStep(i);

    let gradAdj = grad;
    const wd = this._getParamWeightDecay(i);
    // Classic Adam: L2 regularization THROUGH the gradient (affects m/v).
    // AdamW: weight decay is DECOUPLED — applied directly to the param in the
    // final update (below), never entering the moment estimates. This must
    // match the fused kernel's `decoupled_wd` semantics; the two paths had
    // silently forked (elementwise did L2 even with adamW=true) until the
    // fused-vs-elementwise differential test pinned them together.
    if (wd !== 0 && !this.adamW) {
      const paramW = runtime.mul(param._unwrap(), wd);
      gradAdj = runtime.add(gradAdj, paramW);
    }

    const prevAvg = this.expAvg[i];
    const prevAvgSq = this.expAvgSq[i];

    // The moment update GENERATES from ADAMW_PROGRAM — the SAME source the
    // foreach path interprets (P5, design §4.6). State is written IN PLACE into
    // the persistent (snapshot-protected) m/v; replacement was the silent-UAF
    // pattern (the new tensor is demoted as a step temporary at markStep, its
    // buffer pooled while this.expAvg still points at it — the multi-param
    // first-param m/v bug).
    const stateRoles: OptRoles = {
      m: prevAvg,
      v: prevAvgSq,
      g: gradAdj,
      beta1: this.beta1,
      om_beta1: 1 - this.beta1,
      beta2: this.beta2,
      om_beta2: 1 - this.beta2,
    };
    runtime.copy_(prevAvg, evalOptTensor(ADAMW_M_NEW, runtime, stateRoles));
    runtime.copy_(prevAvgSq, evalOptTensor(ADAMW_V_NEW, runtime, stateRoles));

    // PyTorch-style bias-corrected Adam: apply bias correction to v directly
    // in the denominator, not via the step size. This matters because the
    // epsilon term must NOT be scaled by 1/sqrt(bc2).
    // PyTorch: param -= lr * (m / bc1) / (sqrt(v / bc2) + eps)
    //
    // Read m/v through the POST-copy_ state refs (prevAvg's lazy ref now points
    // at the copy_ result) by binding roles `m`/`v` to prevAvg/prevAvgSq AFTER
    // the copy_. This makes the param-update chain DEPEND on the copies, so any
    // force of the param (stepAsync's `force(param)`, partial forces) executes
    // them. Reading the pre-copy moment tensors left the copy_ nodes as DANGLING
    // ROOTS: not in the param's dependency chain, they deferred to whatever plan
    // next touched the state — one step LATE, after zeroGrad() had already
    // zeroed/freed the grad buffer their pending source chain reads (the
    // gpt2-memorization stepAsync NaN regression). inc-2a: bias correction rides
    // in as bc1/bc2 (the step as DATA); [1] tensors broadcasting over the param.
    const { bc1, bc2 } = this._biasCorrection(runtime, this._tTensor());
    const lrTensor = this._lrTensor(this._groupIndex[i]);
    const paramRoles: OptRoles = {
      p: param._unwrap(),
      m: prevAvg,
      v: prevAvgSq,
      lr: lrTensor,
      eps: this.eps,
      wd: this.adamW ? wd : 0,
      bc1,
      bc2,
    };
    // The full p' (ADAMW_P_NEW) folds in the decoupled `lr·wd·p`; the no-wd path
    // derives just the update magnitude (ADAMW_SCALED) and subtracts.
    const pNew =
      this.adamW && wd !== 0
        ? evalOptTensor(ADAMW_P_NEW, runtime, paramRoles)
        : runtime.sub(
            param._unwrap(),
            evalOptTensor(ADAMW_SCALED, runtime, paramRoles),
          );
    runtime.copy_(param._unwrap(), pNew);
  }

  /**
   * Elementwise Adam step: fallback for backends without fused kernel (e.g., CPU).
   */
  private _stepElementwise(
    runtime: ReturnType<Torchlette["_runtime"]>,
  ): Tensor[] {
    const updated: Tensor[] = [];
    // inc-2a: advance the shared on-device `t` ONCE per step.
    this._advanceT(runtime);
    for (let i = 0; i < this.params.length; i++) {
      const param = this.params[i];
      const grad = param.grad?._unwrap() ?? null;
      if (!grad) {
        updated.push(param);
        continue;
      }
      this._updateParamElementwise(runtime, i, param, grad);
      updated.push(param);
    }
    runtime.registerState(this._tTensor());
    for (const s of this._lrLive)
      if (s) runtime.registerState(s.tensor._unwrap());
    return updated;
  }

  /**
   * Async version of step() that forces each parameter update immediately.
   * This prevents peak memory spikes from building up a huge lazy graph.
   * Use this for large models where memory is a concern.
   */
  async stepAsync(): Promise<Tensor[]> {
    const runtime = this.api._runtime();
    const updated: Tensor[] = [];
    // inc-2a: advance the shared on-device `t` ONCE per step, then force it so
    // the per-param forces below read the settled post-increment value.
    this._advanceT(runtime);
    await runtime.force(this._tTensor());
    for (let i = 0; i < this.params.length; i++) {
      const param = this.params[i];
      const grad = param.grad?._unwrap() ?? null;
      if (!grad) {
        updated.push(param);
        continue;
      }
      this._updateParamElementwise(runtime, i, param, grad);
      await runtime.force(param._unwrap());
      updated.push(param);
    }
    runtime.registerState(this._tTensor());
    for (const s of this._lrLive)
      if (s) runtime.registerState(s.tensor._unwrap());
    this.api.queueStepBoundary();
    return updated;
  }

  zeroGrad(): void {
    for (const param of this.params) {
      param.zeroGrad();
    }
  }

  /**
   * Reset moment estimates and step counts. Used when the optimizer's prior
   * trajectory becomes meaningless — e.g., after a DiLoCo F16W resync where
   * params jumped to a peer's anchor state and the m/v moments built up
   * against the now-discarded local trajectory would push subsequent updates
   * in a stale direction.
   */
  resetState(): void {
    const runtime = this.api._runtime();
    for (let i = 0; i < this.params.length; i++) {
      this.expAvg[i] = runtime.zeros(this.params[i].shape, this.device);
      this.expAvgSq[i] = runtime.zeros(this.params[i].shape, this.device);
      this.steps[i] = 0;
    }
    // inc-2a: reset the on-device step counter IN-PLACE (preserve the
    // persistent buffer identity; never replace-and-hold).
    if (this._t)
      runtime.copy_(this._t, runtime.full([1], 0, this.device, "f32"));
    this._tHost = 0; // fork C host mirror resets with the device counter
    // Packed foreach state is built FROM the per-param state on the next
    // step — dispose it so the reset takes effect there too.
    for (const st of this._foreachState.values()) {
      for (const slot of st.slots) slot.dispose();
    }
    this._foreachState.clear();
  }
}

// Debug buffer-identity bookkeeping for TORCHLETTE_DEBUG_ADAM_BUFS.
const _adamDbgIds = new WeakMap<object, number>();
let _adamDbgNext = 1;

// Debug accessor for state-value probes (tools/adam-trajectory-probe.ts).
export function _debugAdamState(
  opt: Adam,
  i: number,
): { m: RuntimeTensor; v: RuntimeTensor } {
  const o = opt as unknown as {
    expAvg: RuntimeTensor[];
    expAvgSq: RuntimeTensor[];
  };
  return { m: o.expAvg[i], v: o.expAvgSq[i] };
}
