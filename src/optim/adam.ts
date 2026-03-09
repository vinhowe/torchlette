import type { AdamStepConfig, DeviceKind } from "../backend/types";
import type { LazyIRNode } from "../engine/lazy-types";
import { createPendingRef } from "../engine/lazy-types";
import { createLazyIRNode } from "../engine/node-factory";
import type { Tensor, Torchlette } from "../frontend";
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

export class Adam {
  private params: Tensor[];
  private readonly api: Torchlette;
  private readonly lr: number;
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly eps: number;
  private readonly weightDecay: number;
  private readonly adamW: boolean;
  private readonly device: DeviceKind;
  private expAvg: Array<RuntimeTensor | null>;
  private expAvgSq: Array<RuntimeTensor | null>;
  private steps: number[];
  /** Pending adamStep nodes from the last step, for side output extraction */
  private _pendingNodes: Array<LazyIRNode | null>;
  /** Pending unscale config from GradScaler (fused unscale+Adam path) */
  private _pendingUnscale: { invScale: number; infFlagBuffer: unknown } | null =
    null;

  constructor(params: Tensor[], options: AdamOptions, api?: Torchlette) {
    const { api: engine, device } = validateOptimizerParams(
      "Adam",
      params,
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
    this.lr = options.lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.params = params.slice();
    this.weightDecay = options.weightDecay ?? 0;
    this.adamW = options.adamW ?? false;
    this.device = device;
    this.expAvg = new Array(params.length).fill(null);
    this.expAvgSq = new Array(params.length).fill(null);
    this.steps = new Array(params.length).fill(0);
    this._pendingNodes = new Array(params.length).fill(null);
  }

  getParams(): Tensor[] {
    return this.params.slice();
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
    for (let i = 0; i < this._pendingNodes.length; i++) {
      const node = this._pendingNodes[i];
      if (!node) continue;

      // Side outputs are StorageHandles created during execution
      const sideOutputs = node._sideOutputs?.adamMV;

      if (sideOutputs) {
        // Dispose old state
        if (this.expAvg[i]) this.expAvg[i]?.dispose();
        if (this.expAvgSq[i]) this.expAvgSq[i]?.dispose();

        // Wrap existing StorageHandles into tracked RuntimeTensors
        this.expAvg[i] = runtime.createFromStorageHandle(
          sideOutputs.m,
          this.params[i].shape,
          this.device,
        );
        this.expAvgSq[i] = runtime.createFromStorageHandle(
          sideOutputs.v,
          this.params[i].shape,
          this.device,
        );
      }
      this._pendingNodes[i] = null;
    }
  }

  step(): Tensor[] {
    const runtime = this.api._runtime();

    // Resolve side outputs from previous fused step
    this._resolvePendingState();

    // Check if fused kernel is available
    if (this.hasFusedKernel()) {
      return this._stepFused(runtime);
    }
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

      const stepSize = this._advanceStep(i);

      // Initialize m, v as zeros on first step
      if (!this.expAvg[i]) {
        this.expAvg[i] = runtime.zeros(param.shape, this.device);
      }
      if (!this.expAvgSq[i]) {
        this.expAvgSq[i] = runtime.zeros(param.shape, this.device);
      }

      const config: AdamStepConfig = {
        beta1: this.beta1,
        beta2: this.beta2,
        stepSize,
        eps: this.eps,
        weightDecay: this.weightDecay,
        lrTimesWd: this.lr * this.weightDecay,
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

  /** Advance step counter and return bias-corrected step size. */
  private _advanceStep(i: number): number {
    const step = this.steps[i] + 1;
    this.steps[i] = step;
    const bc1 = 1 - this.beta1 ** step;
    const bc2 = 1 - this.beta2 ** step;
    return (this.lr * Math.sqrt(bc2)) / bc1;
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
    const stepSize = this._advanceStep(i);

    let gradAdj = grad;
    if (this.weightDecay !== 0) {
      const paramW = runtime.mul(param._unwrap(), this.weightDecay);
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

    const denom = runtime.add(runtime.sqrt(avgSq), this.eps);
    const scaled = runtime.mul(runtime.div(avg, denom), stepSize);
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
