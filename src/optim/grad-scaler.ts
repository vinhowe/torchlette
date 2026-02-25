/**
 * GradScaler for Automatic Mixed Precision (AMP) training.
 *
 * Handles gradient scaling to prevent underflow when using f16 gradients,
 * and automatically adjusts the scale factor based on NaN/Inf detection.
 *
 * When a fused unscale kernel is available (WebGPU), uses a single dispatch
 * per parameter that combines unscale + inf-check + zero-mask. Falls back
 * to elementwise tensor ops on CPU.
 *
 * Usage:
 * ```typescript
 * const scaler = new GradScaler(api);
 *
 * // Training loop
 * for (let step = 0; step < numSteps; step++) {
 *   await scaler.resolveDeferred(); // reads back inf from previous step, adjusts scale
 *   const loss = model.forward(input);
 *   const scaledLoss = scaler.scale(loss);
 *   await scaledLoss.backward();
 *
 *   scaler.unscale_(optimizer);    // SYNC — builds lazy graph, no GPU force
 *   scaler.step(optimizer);        // SYNC — optimizer builds more lazy ops
 *   scaler.update();
 *   optimizer.zeroGrad();
 *   await api.markStep();          // forces entire graph on GPU in one plan
 * }
 * ```
 */
import type { Tensor, Torchlette } from "../frontend";
import type { Adam } from "./adam";
import type { SGD } from "./sgd";
import { createLazyIRNode, createPendingRef } from "../engine/lazy";
import type { Backend, DeviceKind } from "../backend/types";

export type Optimizer = Adam | SGD;

/** Duck-type interface for optimizers with fused GPU kernels (e.g., Adam). */
interface FusedOptimizer {
  hasFusedKernel(): boolean;
  setUnscaleConfig(invScale: number, infFlagBuffer: unknown): void;
}

function isFusedOptimizer(opt: Optimizer): opt is Optimizer & FusedOptimizer {
  return typeof (opt as FusedOptimizer).hasFusedKernel === "function";
}

export type GradScalerOptions = {
  /** Initial scale factor. Default: 65536.0 (2^16) */
  initScale?: number;
  /** Factor to multiply scale by when no inf/nan. Default: 2.0 */
  growthFactor?: number;
  /** Factor to multiply scale by when inf/nan found. Default: 0.5 */
  backoffFactor?: number;
  /** Number of consecutive non-inf/nan steps before growing. Default: 2000 */
  growthInterval?: number;
  /** Whether scaler is enabled. Default: true */
  enabled?: boolean;
};

export type GradScalerState = {
  scale: number;
  growthTracker: number;
};

export class GradScaler {
  private readonly api: Torchlette;
  private readonly growthFactor: number;
  private readonly backoffFactor: number;
  private readonly growthInterval: number;
  private readonly enabled: boolean;

  private _scale: number;
  private _growthTracker: number;
  private _foundInfThisStep: boolean;
  private _unscaleCalled: boolean;
  private _pendingInfAccum: Tensor | null = null;

  // Fused kernel state
  private _pendingInfBuffer: unknown = null;
  private _pendingInfBackend: Backend | null = null;

  constructor(api: Torchlette, options: GradScalerOptions = {}) {
    this.api = api;
    this._scale = options.initScale ?? 65536.0;
    this.growthFactor = options.growthFactor ?? 2.0;
    this.backoffFactor = options.backoffFactor ?? 0.5;
    this.growthInterval = options.growthInterval ?? 2000;
    this.enabled = options.enabled ?? true;
    this._growthTracker = 0;
    this._foundInfThisStep = false;
    this._unscaleCalled = false;
  }

  /**
   * Get the current scale factor.
   */
  getScale(): number {
    return this._scale;
  }

  /**
   * Check if inf/nan was found in the current step.
   * Only valid after unscale_() has been called.
   */
  get foundInf(): boolean {
    return this._foundInfThisStep;
  }

  /**
   * Get the state for serialization.
   */
  stateDict(): GradScalerState {
    return {
      scale: this._scale,
      growthTracker: this._growthTracker,
    };
  }

  /**
   * Load state from a previous save.
   */
  loadStateDict(state: GradScalerState): void {
    this._scale = state.scale;
    this._growthTracker = state.growthTracker;
  }

  /**
   * Resolve the deferred inf readback from the previous step.
   *
   * This reads back the inf count from the GPU. Because the GPU work from
   * the previous step has already completed by the time the next step starts,
   * this readback is nearly instant (< 1ms).
   *
   * Must be called at the **start** of each training step, before scale().
   * On the first step (no pending inf), this is a no-op.
   */
  async resolveDeferred(): Promise<void> {
    if (this._pendingInfBuffer) {
      // Fused path: force all pending GPU work so the unscaleGrad kernels
      // have written to the infFlagBuffer before we read it back.
      // In the normal training loop, markStep() is called before resolveDeferred(),
      // but we handle the case where it wasn't called explicitly.
      await this.api.markStep();
      const val = await this._pendingInfBackend!.ops.readAndDestroyInfCount!(this._pendingInfBuffer);
      this._foundInfThisStep = val > 0.5;
      this._pendingInfBuffer = null;
      this._pendingInfBackend = null;
    } else if (this._pendingInfAccum) {
      // Elementwise path: read tensor
      const totalInfCount = await this._pendingInfAccum.item();
      this._foundInfThisStep = totalInfCount > 0.5;
      this._pendingInfAccum.dispose();
      this._pendingInfAccum = null;
    } else {
      return; // First step, no-op
    }

    // Apply scale adjustment (deferred from the previous step's update())
    if (this._foundInfThisStep) {
      this._scale *= this.backoffFactor;
      this._growthTracker = 0;
    } else {
      this._growthTracker += 1;
      if (this._growthTracker >= this.growthInterval) {
        this._scale *= this.growthFactor;
        this._growthTracker = 0;
      }
    }
  }

  /**
   * Scale the loss tensor before calling backward().
   * This multiplies the loss by the current scale factor.
   *
   * @param loss The loss tensor to scale
   * @returns Scaled loss tensor
   */
  scale(loss: Tensor): Tensor {
    if (!this.enabled) {
      return loss;
    }
    return this.api.mul(loss, this._scale);
  }

  /**
   * Unscale gradients in-place for all parameters in the optimizer.
   * Also checks for inf/nan in gradients.
   *
   * Uses fused kernel when available (WebGPU), falls back to elementwise ops (CPU).
   *
   * Must be called before step() and after backward().
   *
   * @param optimizer The optimizer containing parameters to unscale
   */
  unscale_(optimizer: Optimizer): void {
    if (!this.enabled) {
      this._unscaleCalled = true;
      this._foundInfThisStep = false;
      return;
    }

    this._foundInfThisStep = false;
    this._unscaleCalled = true;

    const params = optimizer.getParams();
    const invScale = 1.0 / this._scale;
    const runtime = this.api._runtime();

    // Determine device from first param with a grad
    let device: DeviceKind = "cpu";
    for (const p of params) {
      if (p.grad) { device = p.device; break; }
    }

    const backend = runtime.getBackend(device);

    // Fused Adam+unscale path: pass invScale/infFlagBuffer to Adam so it
    // handles unscaling inside the adamStep kernel (one dispatch per param
    // instead of two). Adam.step() will consume the pending unscale config.
    if (
      backend.ops.adamStep &&
      isFusedOptimizer(optimizer) &&
      optimizer.hasFusedKernel()
    ) {
      const infFlagBuffer = backend.ops.createInfCountBuffer!();
      optimizer.setUnscaleConfig(invScale, infFlagBuffer);
      this._pendingInfBuffer = infFlagBuffer;
      this._pendingInfBackend = backend;
      return;
    }

    if (backend.ops.unscaleGrad) {
      this._unscaleFused(params, invScale, backend, device);
    } else {
      this._unscaleElementwise(params, invScale, runtime, device);
    }
  }

  /**
   * Fused unscale path: one kernel dispatch per parameter.
   * All dispatches share a single infFlagBuffer (4 bytes, atomic).
   */
  private _unscaleFused(
    params: Tensor[],
    invScale: number,
    backend: Backend,
    device: DeviceKind,
  ): void {
    const infFlagBuffer = backend.ops.createInfCountBuffer!();
    const sharedPayload = { invScale, infFlagBuffer };

    for (const param of params) {
      const grad = param.grad;
      if (!grad) continue;

      const gradInner = grad._unwrap();

      // Create unscaleGrad lazy node: single input (grad), single output (unscaled grad)
      const node = createLazyIRNode(
        "unscaleGrad",
        [gradInner.lazyRef],
        grad.shape,
        "f32",
        device,
        sharedPayload,
      );

      // Update grad's lazyRef to point at the unscaleGrad result
      gradInner._updateLazyRef(createPendingRef(node));
    }

    // Store for deferred readback
    this._pendingInfBuffer = infFlagBuffer;
    this._pendingInfBackend = backend;
  }

  /**
   * Elementwise unscale path: CPU fallback using standard tensor ops.
   */
  private _unscaleElementwise(
    params: Tensor[],
    invScale: number,
    runtime: ReturnType<Torchlette["_runtime"]>,
    device: DeviceKind,
  ): void {
    let infAccum = this.api.full([], 0.0, { device });

    const toDispose: Tensor[] = [];
    const unscaledGrads: Tensor[] = [];
    const gradTensors: Tensor[] = [];

    // Loop 1: Unscale + accumulate inf count (all lazy)
    for (const param of params) {
      const grad = param.grad;
      if (!grad) continue;

      // Unscale gradient: grad * invScale
      const unscaledGrad = this.api.mul(grad, invScale);
      unscaledGrads.push(unscaledGrad);
      gradTensors.push(grad);

      // Count non-finite elements: (1 - isfinite(x)) gives 1.0 for inf/nan, 0.0 for finite.
      const finiteFlags = this.api.isfinite(unscaledGrad);
      const nonFiniteFlags = this.api.sub(1.0, finiteFlags);
      const paramInfCount = this.api.sum(nonFiniteFlags);

      // Accumulate on GPU — no item() call in the loop
      const prevAccum = infAccum;
      infAccum = this.api.add(infAccum, paramInfCount);

      toDispose.push(finiteFlags, nonFiniteFlags, paramInfCount, prevAccum);
    }

    // Build shouldZero flag from final infAccum (lazy, 0-d tensor)
    const shouldZero = this.api.gt(infAccum, 0.5);
    toDispose.push(shouldZero);

    // Loop 2: Mask grads and write back (all lazy)
    for (let i = 0; i < unscaledGrads.length; i++) {
      const maskedGrad = this.api.where(shouldZero, 0.0, unscaledGrads[i]);
      runtime.copy_(gradTensors[i]._unwrap(), maskedGrad._unwrap());
      toDispose.push(maskedGrad);
    }

    // Store infAccum for deferred readback in resolveDeferred()
    this._pendingInfAccum = infAccum;

    // Dispose all intermediates except infAccum (needed by resolveDeferred())
    for (const ug of unscaledGrads) {
      ug.dispose();
    }
    for (const t of toDispose) {
      t.dispose();
    }
  }

  /**
   * Step the optimizer.
   *
   * Gradients have already been unscaled (and non-finite elements zeroed)
   * by unscale_(), so the optimizer always runs safely.
   *
   * Fully synchronous — just builds more lazy ops on top of the unscaled grads.
   * Must be called after unscale_().
   *
   * @param optimizer The optimizer to step
   * @returns Always true (optimizer always runs)
   */
  step(optimizer: Optimizer): boolean {
    if (!this._unscaleCalled) {
      throw new Error(
        "GradScaler.step() called before unscale_(). Call unscale_() first.",
      );
    }

    if (!this.enabled) {
      optimizer.step();
      return true;
    }

    // Grads were already unscaled by unscale_().
    // Optimizer builds more lazy ops on the unscaled grads.
    optimizer.step();

    return true;
  }

  /**
   * Update flags after step(). Scale adjustment is deferred to resolveDeferred().
   *
   * Call after step() at the end of each training iteration.
   */
  update(): void {
    if (!this.enabled) {
      this._unscaleCalled = false;
      return;
    }

    if (!this._unscaleCalled) {
      throw new Error(
        "GradScaler.update() called before unscale_(). Call unscale_() first.",
      );
    }

    // Reset for next step — scale adjustment happens in resolveDeferred()
    this._unscaleCalled = false;
  }

  /**
   * Check if the scaler is enabled.
   */
  isEnabled(): boolean {
    return this.enabled;
  }
}
