import type { AdamStepConfig, DeviceKind } from "../backend/types";
import type { Tensor, Torchlette } from "../frontend/torchlette";
import { createLazyIRNode } from "../graph/node-factory";
import type { LazyIRNode } from "../graph/types";
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
  private expAvg: Array<RuntimeTensor | null>;
  private expAvgSq: Array<RuntimeTensor | null>;
  private steps: number[];
  /** Pending adamStep nodes from the last step, for side output extraction */
  private _pendingNodes: Array<LazyIRNode | null>;
  /** Pending unscale config from GradScaler (fused unscale+Adam path) */
  private _pendingUnscale: { invScale: number; infFlagBuffer: unknown } | null =
    null;

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

    this.expAvg = new Array(flatParams.length).fill(null);
    this.expAvgSq = new Array(flatParams.length).fill(null);
    this.steps = new Array(flatParams.length).fill(0);
    this._pendingNodes = new Array(flatParams.length).fill(null);
  }

  getParams(): Tensor[] {
    return this.params.slice();
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

  /**
   * Set pending unscale config for fused unscale+Adam step.
   * Called by GradScaler._unscaleFused() to pass invScale and infFlagBuffer
   * so that the Adam kernel can fuse gradient unscaling with the optimizer step.
   */
  setUnscaleConfig(invScale: number, infFlagBuffer: unknown): void {
    this._pendingUnscale = { invScale, infFlagBuffer };
  }

  /**
   * Resolve side outputs (m, v) from pending adamStep nodes.
   * Called at the start of each step() to extract m/v from the previous step's
   * fused kernel execution.
   */
  private _resolvePendingState(): void {
    const runtime = this.api._runtime();
    let resolved = 0;
    let missed = 0;
    for (let i = 0; i < this._pendingNodes.length; i++) {
      const node = this._pendingNodes[i];
      if (!node) continue;

      // Multi-output: results[1] = m, results[2] = v
      const mStorage = node.results?.[1];
      const vStorage = node.results?.[2];

      if (mStorage && vStorage) {
        resolved++;
        // Dispose old state
        if (this.expAvg[i]) this.expAvg[i]?.dispose();
        if (this.expAvgSq[i]) this.expAvgSq[i]?.dispose();

        // Wrap existing StorageHandles into tracked RuntimeTensors
        this.expAvg[i] = runtime.createFromStorageHandle(
          mStorage,
          this.params[i].shape,
          this.device,
        );
        this.expAvgSq[i] = runtime.createFromStorageHandle(
          vStorage,
          this.params[i].shape,
          this.device,
        );
      } else {
        missed++;
      }
      this._pendingNodes[i] = null;
    }
    // resolved/missed counters available for debug if needed
  }

  step(): Tensor[] {
    const runtime = this.api._runtime();

    // Resolve side outputs from previous fused step
    this._resolvePendingState();

    // TODO: Fused Adam kernel has a buffer aliasing bug where gradients
    // are read as scalar instead of element-wise, producing wrong updates.
    // Use elementwise path until the fused kernel is fixed.
    // if (this.hasFusedKernel()) {
    //   return this._stepFused(runtime);
    // }
    return this._stepElementwise(runtime);
  }

  /**
   * Fused Adam step: one dispatch per parameter.
   * When _pendingUnscale is set, the kernel also fuses gradient unscaling
   * and inf detection (eliminating separate unscaleGrad dispatches).
   */
  private _stepFused(runtime: ReturnType<Torchlette["_runtime"]>): Tensor[] {
    const updated: Tensor[] = [];

    // Consume pending unscale config (set by GradScaler)
    const unscale = this._pendingUnscale;
    this._pendingUnscale = null;

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

      // Initialize m, v as zeros on first step
      if (!this.expAvg[i]) {
        this.expAvg[i] = runtime.zeros(param.shape, this.device);
      }
      if (!this.expAvgSq[i]) {
        this.expAvgSq[i] = runtime.zeros(param.shape, this.device);
      }

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
        invScale: unscale?.invScale,
        infFlagBuffer: unscale?.infFlagBuffer,
      };

      // Create a single adamStep lazy node with 4 inputs: grad, param, m, v
      const adamNode = createLazyIRNode(
        "adamStep",
        [
          grad.lazyRef,
          param._unwrap().lazyRef,
          this.expAvg[i]!.lazyRef,
          this.expAvgSq[i]!.lazyRef,
        ],
        param.shape,
        "f32",
        this.device,
        config,
      );

      // For in-place execution: prevent the old storage's backend tensor from
      // releasing the GPU buffer when destroyUnreachable() runs (before the adam
      // node executes). The adam kernel writes in-place to the same buffer, so
      // the buffer must NOT be returned to the pool. _updateLazyRef marks the
      // old storage unreachable, and destroyUnreachable can run before execution.
      const paramRt = param._unwrap();
      if (paramRt.isMaterialized()) {
        const oldBt = paramRt.backendTensor as { destroy?: () => void };
        if (oldBt.destroy) {
          oldBt.destroy = () => {};
        }
      }

      // Update param in-place to point at the adamStep node's result
      paramRt._updateLazyRef(createPendingRef(adamNode));

      // Store node ref for side output extraction on next step
      this._pendingNodes[i] = adamNode;

      // Keep expAvg[i] / expAvgSq[i] alive — they're inputs to the lazy node
      // and can't be disposed until execution completes. _resolvePendingState()
      // on the NEXT step will dispose them (after markStep() has executed the node).

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
    this._advanceStep(i);

    let gradAdj = grad;
    const wd = this._getParamWeightDecay(i);
    if (wd !== 0) {
      const paramW = runtime.mul(param._unwrap(), wd);
      gradAdj = runtime.add(gradAdj, paramW);
    }

    const prevAvg = this.expAvg[i];
    const prevAvgSq = this.expAvgSq[i];

    let avg: RuntimeTensor;
    if (prevAvg) {
      avg = runtime.add(
        runtime.mul(prevAvg, this.beta1),
        runtime.mul(gradAdj, 1 - this.beta1),
      );
    } else {
      avg = runtime.mul(gradAdj, 1 - this.beta1);
    }

    const gradSq = runtime.mul(gradAdj, gradAdj);

    let avgSq: RuntimeTensor;
    if (prevAvgSq) {
      avgSq = runtime.add(
        runtime.mul(prevAvgSq, this.beta2),
        runtime.mul(gradSq, 1 - this.beta2),
      );
    } else {
      avgSq = runtime.mul(gradSq, 1 - this.beta2);
    }

    prevAvg?.dispose();
    prevAvgSq?.dispose();
    this.expAvg[i] = avg;
    this.expAvgSq[i] = avgSq;

    // PyTorch-style bias-corrected Adam: apply bias correction to v directly
    // in the denominator, not via the step size. This matters because the
    // epsilon term must NOT be scaled by 1/sqrt(bc2).
    // PyTorch: param -= lr * (m / bc1) / (sqrt(v / bc2) + eps)
    const bc1 = 1 - this.beta1 ** this.steps[i];
    const bc2 = 1 - this.beta2 ** this.steps[i];
    const mHat = runtime.div(avg, bc1);
    const vHat = runtime.div(avgSq, bc2);
    const denom = runtime.add(runtime.sqrt(vHat), this.eps);
    const scaled = runtime.mul(runtime.div(mHat, denom), this._getParamLR(i));
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
    this._resolvePendingState();
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
    return updated;
  }

  zeroGrad(): void {
    for (const param of this.params) {
      param.zeroGrad();
    }
  }
}
