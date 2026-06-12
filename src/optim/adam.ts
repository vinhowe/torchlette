import { ENV } from "../core/env";
import type { AdamStepConfig, DeviceKind } from "../backend/types";
import type { Tensor, Torchlette } from "../frontend/torchlette";
import { createLazyIRNode } from "../graph/node-factory";
import { createPendingRef } from "../graph/types";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";
import { validateOptimizerParams } from "./validate";

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
  /** Intermediate tensors from the last step — disposed after markStep(). */
  private _intermediates: RuntimeTensor[] = [];
  private steps: number[];
  /**
   * Packed foreach state, keyed by group index. While the foreach path is
   * active, m/v live as ONE flat tensor per group (the per-param expAvg
   * arrays are consumed at first pack and become stale). Mirrors PyTorch's
   * foreach optimizers: the per-param definition is the semantics, the
   * packed form is the batched execution of the same tensor program.
   */
  private _foreachState = new Map<
    number,
    { m: RuntimeTensor; v: RuntimeTensor; sig: string }
  >();
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
    return keep;
  }

  /** Get the default (first group) learning rate. */
  getLR(): number {
    return this._groups[0].lr;
  }

  /** Set learning rate for all parameter groups. */
  setLR(lr: number): void {
    for (const g of this._groups) g.lr = lr;
  }

  /** Get per-group learning rates. */
  getParamGroupLRs(): number[] {
    return this._groups.map((g) => g.lr);
  }

  /** Set learning rate for a specific parameter group. */
  setGroupLR(groupIndex: number, lr: number): void {
    this._groups[groupIndex].lr = lr;
  }

  /** Get the number of parameter groups. */
  get numGroups(): number {
    return this._groups.length;
  }

  /** Get per-param LR for the fused kernel. */
  private _getParamLR(i: number): number {
    return this._groups[this._groupIndex[i]].lr;
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
    } else if (
      ENV.TORCHLETTE_FOREACH_ADAM !== "0" &&
      this.params.length > 1
    ) {
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
    for (const [gi, idxs] of groups) {
      this._foreachGroupStep(runtime, gi, idxs);
    }
    return updated;
  }

  private _foreachGroupStep(
    runtime: ReturnType<Torchlette["_runtime"]>,
    gi: number,
    idxs: number[],
  ): void {
    const sizes = idxs.map((i) =>
      this.params[i].shape.reduce((a, b) => a * b, 1),
    );
    const offsets: number[] = [];
    let total = 0;
    for (const s of sizes) {
      offsets.push(total);
      total += s;
    }
    const sig = idxs.map((i, k) => `${i}:${sizes[k]}`).join(",");

    for (const i of idxs) this._advanceStep(i);
    const t = this.steps[idxs[0]];
    for (const i of idxs) {
      if (this.steps[i] !== t) {
        throw new Error(
          "Adam foreach: params in one group have diverging step counts " +
            "(gradients intermittently missing for some params). Set " +
            "TORCHLETTE_FOREACH_ADAM=0 to use the per-param path.",
        );
      }
    }
    const lr = this._groups[gi].lr;
    const wd = this._groups[gi].weightDecay;

    // Pack grads and params: [size_i] flats concatenated to one [total].
    const gFlat = idxs.map((i, k) =>
      runtime.reshape(this.params[i].grad!._unwrap(), [sizes[k]]),
    );
    const G = runtime.cat(gFlat);
    const pFlat = idxs.map((i, k) =>
      runtime.reshape(this.params[i]._unwrap(), [sizes[k]]),
    );
    const P = runtime.cat(pFlat);

    // Packed m/v: initialized ONCE by packing the current per-param state
    // (zeros at start; real moments when switching paths or after
    // resetState), then updated IN PLACE every step via copy_. Stable
    // storage matters structurally: re-creating state tensors each step
    // ping-pongs against the buffer arena (last step's m is still a live
    // input while this step's m wants the same plan position) — fresh
    // allocations every step, unbounded growth. In-place state also keeps
    // the steady-state plan structure identical every step (one template,
    // one compiled plan).
    let st = this._foreachState.get(gi);
    if (st && st.sig !== sig) {
      throw new Error(
        "Adam foreach: the set of grad-bearing params in a group changed " +
          "across steps; packed state cannot be remapped. Set " +
          "TORCHLETTE_FOREACH_ADAM=0 to use the per-param path.",
      );
    }
    if (!st) {
      const mFlat = idxs.map((i, k) =>
        runtime.reshape(this.expAvg[i], [sizes[k]]),
      );
      const vFlat = idxs.map((i, k) =>
        runtime.reshape(this.expAvgSq[i], [sizes[k]]),
      );
      // persist(): the packed state is created MID-STEP and held across
      // steps — without adoption into the step snapshot it would be demoted
      // as a temporary at markStep (buffer pooled while live → silent UAF).
      st = {
        m: runtime.persist(runtime.cat(mFlat)),
        v: runtime.persist(runtime.cat(vFlat)),
        sig,
      };
      this._foreachState.set(gi, st);
    }

    // The exact tensor program of _updateParamElementwise, batched.
    let gAdj = G;
    if (wd !== 0 && !this.adamW) {
      gAdj = runtime.add(gAdj, runtime.mul(P, wd));
    }
    const mNew = runtime.add(
      runtime.mul(st.m, this.beta1),
      runtime.mul(gAdj, 1 - this.beta1),
    );
    const vNew = runtime.add(
      runtime.mul(st.v, this.beta2),
      runtime.mul(runtime.mul(gAdj, gAdj), 1 - this.beta2),
    );
    runtime.copy_(st.m, mNew);
    runtime.copy_(st.v, vNew);

    // Read m/v through the POST-copy_ state refs so the param-update chain
    // depends on the copies and any force of the params executes them.
    // Reading mNew/vNew directly leaves the copy_ nodes as dangling roots
    // that defer to a LATER plan — after zeroGrad has zeroed/freed the grad
    // buffer their pending source reads (see _updateParamElementwise).
    const bc1 = 1 - this.beta1 ** t;
    const bc2 = 1 - this.beta2 ** t;
    const mHat = runtime.div(st.m, bc1);
    const vHat = runtime.div(st.v, bc2);
    const denom = runtime.add(runtime.sqrt(vHat), this.eps);
    let scaled = runtime.mul(runtime.div(mHat, denom), lr);
    if (wd !== 0 && this.adamW) {
      // Decoupled weight decay: p -= lr*wd*p (PyTorch AdamW).
      scaled = runtime.add(scaled, runtime.mul(P, lr * wd));
    }
    const pNew = runtime.sub(P, scaled);

    // Unpack: copy each segment back into its (persistent) param storage.
    for (let k = 0; k < idxs.length; k++) {
      const seg = runtime.reshape(
        runtime.narrow(pNew, 0, offsets[k], sizes[k]),
        this.params[idxs[k]].shape,
      );
      runtime.copy_(this.params[idxs[k]]._unwrap(), seg);
    }

    // Dispose the big packed intermediates NOW that the update graph is
    // built. The IR nodes survive (downstream nodes reference nodes, not
    // wrappers); disposal removes them from the live-pending registry so
    // the executor's liveness analysis can release — and the fused kernels
    // DONATE — their buffers. Without this, every full-group-size
    // intermediate (328MB each at 124M) is conservatively protected as
    // "user-held" and the packed chain costs ~2x the fused path's memory.
    // st.m / st.v / params are NOT disposed (persistent state).
    for (const t of [G, gAdj, mNew, vNew, mHat, vHat, denom, scaled, pNew, P]) {
      if (
        t !== (st.m as unknown) &&
        t !== (st.v as unknown)
      ) {
        (t as { dispose?: () => void }).dispose?.();
      }
    }
  }

  /**
   * Fused Adam step: one dispatch per parameter.
   * (The kernel also supports a fused unscale+inf-check variant via
   * AdamStepConfig.invScale/infFlagBuffer; GradScaler unscales through
   * graph-level unscaleGrad nodes instead, so nothing engages it here.)
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

      this._advanceStep(i);
      const step = this.steps[i];
      const bc1 = 1 - this.beta1 ** step;
      const bc2 = 1 - this.beta2 ** step;
      const lr = this._getParamLR(i);
      const stepSize = (lr * Math.sqrt(bc2)) / bc1;
      // Adjust epsilon so the kernel formula p -= stepSize * m / (sqrt(v) + eps)
      // becomes equivalent to PyTorch's p -= lr * (m/bc1) / (sqrt(v/bc2) + eps_orig).
      // The adjustment: eps_adjusted = eps_orig * sqrt(bc2)
      const epsAdjusted = this.eps * Math.sqrt(bc2);

      const wd = this._getParamWeightDecay(i);
      const config: AdamStepConfig = {
        beta1: this.beta1,
        beta2: this.beta2,
        stepSize,
        eps: epsAdjusted,
        weightDecay: wd,
        lrTimesWd: lr * wd,
        decoupledWd: this.adamW,
        emitF16: true,
      };

      // Create a single adamStep lazy node with 4 inputs: grad, param, m, v
      const adamNode = createLazyIRNode(
        "adamStep",
        [
          grad.lazyRef,
          param._unwrap().lazyRef,
          this.expAvg[i].lazyRef,
          this.expAvgSq[i].lazyRef,
        ],
        param.shape,
        "f32",
        this.device,
        config,
      );

      // For in-place execution: prevent the old param storage's backend tensor
      // from releasing the GPU buffer when destroyUnreachable() runs (before
      // the adam node executes). The adam kernel writes param in-place, so the
      // buffer must NOT be returned to the pool.
      const paramRt = param._unwrap();
      if (paramRt.isMaterialized()) {
        const oldBt = paramRt.backendTensor as { destroy?: () => void };
        if (oldBt.destroy) {
          oldBt.destroy = () => {};
        }
      }

      // Update param (output 0), m (output 1), v (output 2) to point at
      // the adamStep node's results. m/v are fresh output buffers.
      paramRt._updateLazyRef(createPendingRef(adamNode, 0));
      this.expAvg[i]._updateLazyRef(createPendingRef(adamNode, 1));
      this.expAvgSq[i]._updateLazyRef(createPendingRef(adamNode, 2));

      updated.push(param);
    }

    return updated;
  }

  /** Advance step counter. */
  private _advanceStep(i: number): void {
    this.steps[i] += 1;
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

    const avg = runtime.add(
      runtime.mul(prevAvg, this.beta1),
      runtime.mul(gradAdj, 1 - this.beta1),
    );

    const gradSq = runtime.mul(gradAdj, gradAdj);

    const avgSq = runtime.add(
      runtime.mul(prevAvgSq, this.beta2),
      runtime.mul(gradSq, 1 - this.beta2),
    );

    // Update state IN PLACE into the persistent (snapshot-protected) m/v
    // tensors instead of replacing them with the mid-step-created results.
    // Replacement was the silent-UAF pattern: the new tensor is demoted as a
    // step temporary at markStep (it was not in the beginStep snapshot), its
    // buffer returns to the pool while this.expAvg still points at it, and a
    // later allocation corrupts it — the multi-param first-param m/v bug.
    runtime.copy_(prevAvg, avg);
    runtime.copy_(prevAvgSq, avgSq);

    // PyTorch-style bias-corrected Adam: apply bias correction to v directly
    // in the denominator, not via the step size. This matters because the
    // epsilon term must NOT be scaled by 1/sqrt(bc2).
    // PyTorch: param -= lr * (m / bc1) / (sqrt(v / bc2) + eps)
    //
    // Read m/v through the POST-copy_ state refs (prevAvg's lazy ref now
    // points at the copy_ result), NOT through `avg`/`avgSq` directly. This
    // makes the param-update chain DEPEND on the copies, so any force of the
    // param (stepAsync's `force(param)`, partial forces) executes them.
    // Reading `avg` left the copy_ nodes as DANGLING ROOTS: not in the
    // param's dependency chain, they deferred to whatever plan next touched
    // the state — one step LATE, after zeroGrad() had already zeroed/freed
    // the grad buffer their pending source chain reads, and after freed
    // intermediates were reacquired (src==dst in the scatter kernel is a
    // Dawn read-write usage validation error → dropped command buffer).
    // That was the gpt2-memorization stepAsync NaN regression.
    const bc1 = 1 - this.beta1 ** this.steps[i];
    const bc2 = 1 - this.beta2 ** this.steps[i];
    const mHat = runtime.div(prevAvg, bc1);
    const vHat = runtime.div(prevAvgSq, bc2);
    const denom = runtime.add(runtime.sqrt(vHat), this.eps);
    const lr = this._getParamLR(i);
    let scaled = runtime.mul(runtime.div(mHat, denom), lr);
    if (wd !== 0 && this.adamW) {
      // Decoupled weight decay: p -= lr*wd*p (PyTorch AdamW).
      scaled = runtime.add(scaled, runtime.mul(param._unwrap(), lr * wd));
    }
    runtime.copy_(param._unwrap(), runtime.sub(param._unwrap(), scaled));
  }

  /**
   * Elementwise Adam step: fallback for backends without fused kernel (e.g., CPU).
   */
  private _stepElementwise(
    runtime: ReturnType<Torchlette["_runtime"]>,
  ): Tensor[] {
    const updated: Tensor[] = [];
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
    // Packed foreach state is built FROM the per-param state on the next
    // step — dispose it so the reset takes effect there too.
    for (const st of this._foreachState.values()) {
      st.m.dispose();
      st.v.dispose();
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
