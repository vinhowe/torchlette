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

import type { Backend, DeviceKind } from "../backend/types";
import { LiveScalar } from "../core/live-scalar";
import type { Tensor, Torchlette } from "../frontend/torchlette";
import { createLazyIRNode } from "../graph/node-factory";
import { createPendingRef } from "../graph/types";
import type { Adam } from "./adam";
import type { SGD } from "./sgd";

export type Optimizer = Adam | SGD;

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

  // [scaler-as-tensor] scale as a LIVE persistent tensor. The forward
  // `scale(loss)` reads `_scaleLive.tensor` in-graph; the fused `unscale_`
  // feeds the SAME tensor to unscaleGrad as node.inputs[1] (invScale = 1/scale
  // reciprocated IN-KERNEL) — so a growth/backoff rescale flows through a
  // step-tape HIT as DATA via the SINGLE scale write. A second live invScale
  // tensor would need its own driver scatter, and the per-step re-dress only
  // cleanly covers one such write per body (the second's recorded
  // scatter-source is left stale); an in-graph `div(1, scale)` node instead
  // added compiled-plan fp drift over a trajectory (~5x on the train-capture
  // gate). ONE tensor, reciprocal in-shader. The CPU `_scale` number is
  // DEMOTED to a stats-only mirror (getScale / stateDict / bookkeeping).
  private _scaleLive: LiveScalar | null = null;
  private _liveDevice: "cpu" | "webgpu" | null = null;

  // [scaler-as-tensor] DRIVER-RESOLVED persistent found-inf. Once the fused
  // path initializes, the replayed plan writes the persistent inf flag every
  // step (hit or miss). The DRIVER (`resolveDeferred`) snapshots + reads that
  // flag at settle time so the CPU scale mirror advances on HITS too (where the
  // body — hence the old `_pendingInfBuffer` set in `unscale_` — never runs).
  private _fusedInfBackend: Backend | null = null;

  // [inc-3 runahead ring] Per-step found-inf snapshots (fused path), oldest
  // first. Each entry isolates ONE step's report: the shared flag buffer is
  // snapshotted + re-zeroed in queue order by `snapshotDeferred()`, so a later
  // step's zero-write (or a replayed step's kernels, which never re-zero)
  // cannot clobber an unread report. `handle: null` = the fused path hadn't
  // initialized that step (reads as "no inf").
  private _pendingInfSnapshots: Array<{
    handle: unknown;
    backend: Backend | null;
  }> = [];

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
    // [scaler-as-tensor] Create the LIVE scale/invScale tensors EAGERLY on the
    // api's default device (like Adam's `_lrLive` in its ctor). Mid-step lazy
    // creation demotes them as step temporaries (they're born inside the first
    // captured body, after that step's snapshot) → reclaimed-storage corruption.
    // Constructing here (outside any step) lets the next beginStep snapshot
    // capture them as persistent model-level state.
    if (this.enabled) this._ensureLive(api.getDefaultDevice());
  }

  /**
   * Get the current scale factor.
   */
  getScale(): number {
    return this._scale;
  }

  /** [scaler-as-tensor] Ensure the scale/invScale LiveScalars exist on `device`.
   *  Idempotent; created once the first consumer (scale/unscale) knows the
   *  device. The tensors carry the CURRENT scale/invScale as DATA. */
  private _ensureLive(device: DeviceKind): void {
    if (this._scaleLive && this._liveDevice === device) return;
    // Device is fixed for the scaler's life on first materialization; a mid-run
    // device flip is not a real training pattern.
    const dev = device as "cpu" | "webgpu";
    this._liveDevice = dev;
    this._scaleLive = new LiveScalar(this.api, this._scale, dev);
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
      // MISS path (body ran `unscale_` last step): read the flag it set. A
      // markStep here commits the prior (possibly implied) boundary + submits
      // the unscale writes, forming the tape during warmup — the same behavior
      // as before scaler-as-tensor. Hits take the branch below (NO markStep, so
      // the implied-boundary regime and its fp are undisturbed).
      await this.api.markStep();
      const val = await this._pendingInfBackend?.ops.readAndDestroyInfCount?.(
        this._pendingInfBuffer,
      );
      this._foundInfThisStep = val! > 0.5;
      this._pendingInfBuffer = null;
      this._pendingInfBackend = null;
      this._applyScaleAdjustment();
    } else if (this._fusedInfBackend) {
      // [scaler-as-tensor] HIT path (fused initialized, but the body did NOT run
      // last step — a tape HIT). The scale mirror must still ADVANCE so a
      // growth/backoff flows through hits as DATA. Do it WITHOUT a markStep: an
      // extra boundary on a hit perturbs the implied-boundary fp regime (and
      // resets the recorder). If the loop opted into per-step `snapshotDeferred()`
      // (the growth/ring probes), drain the oldest found-inf report via the
      // hit-safe ring (self-synchronizing mapAsync, no boundary); otherwise
      // advance with foundInf=false (growth still flows; inf-on-hit detection
      // then needs the loop to snapshot). The ring path (runahead) resolves via
      // resolveOldestDeferred directly.
      if (this._pendingInfSnapshots.length > 0) {
        await this.resolveOldestDeferred();
      } else {
        this._foundInfThisStep = false;
        this._applyScaleAdjustment();
      }
    } else if (this._pendingInfAccum) {
      // Elementwise path: should have been resolved in update().
      // Fallback if update() wasn't called or wasn't awaited.
      const totalInfCount = await this._pendingInfAccum.item();
      this._foundInfThisStep = totalInfCount > 0.5;
      this._pendingInfAccum.dispose();
      this._pendingInfAccum = null;
      this._applyScaleAdjustment();
    }
  }

  /**
   * [inc-3 runahead ring] Snapshot THIS step's found-inf report. Called by a
   * RUNAHEAD driver once per step, right after the captured call returns (the
   * step's unscale kernels are submitted; the next step hasn't). Snapshots the
   * shared inf flag into a pool-excluded staging buffer and re-zeroes it, both
   * in queue order — isolating the report per step where the serial single-slot
   * path would lose it (a later `unscale_` zero-write clobbers an unread flag,
   * and a tape-HIT step replays the kernels without ever re-zeroing).
   * Subsumes `_pendingInfBuffer` for that step (cleared here) — a runahead
   * driver resolves via `resolveOldestDeferred()` at its K-behind cadence (or
   * drains after the ring's `drain()`), NEVER via per-step `resolveDeferred()`
   * (its `markStep` is a full non-gen boundary — illegal mid-ring).
   */
  snapshotDeferred(): void {
    if (!this.enabled) return;
    const backend =
      this._pendingInfBackend ??
      this._pendingInfSnapshots.find((e) => e.backend)?.backend ??
      null;
    const handle = backend?.ops.snapshotInfFlag?.() ?? null;
    this._pendingInfSnapshots.push({ handle, backend });
    // The single-slot pending readback is subsumed by the snapshot (reading
    // the shared flag later would see the re-zeroed value anyway).
    this._pendingInfBuffer = null;
  }

  /**
   * [inc-3 runahead ring] Resolve the OLDEST deferred found-inf snapshot and
   * apply its scale adjustment (in step order). Self-synchronizing mapAsync —
   * no step boundary, no shared fence, no pool bookkeeping — so it is legal
   * mid-ring. A K-behind cadence gives the charter's bookkeeping-lag bound: the
   * CPU scale mirror lags the GPU trajectory by exactly ≤K steps. Returns false
   * when no snapshot is pending.
   */
  async resolveOldestDeferred(): Promise<boolean> {
    const entry = this._pendingInfSnapshots.shift();
    if (!entry) return false;
    let val = 0;
    if (entry.handle !== null && entry.backend?.ops.readInfSnapshot) {
      val = await entry.backend.ops.readInfSnapshot(entry.handle);
    }
    this._foundInfThisStep = val > 0.5;
    this._applyScaleAdjustment();
    return true;
  }

  /** Adjust scale based on whether inf was found this step. */
  private _applyScaleAdjustment(): void {
    const prev = this._scale;
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
    // [scaler-as-tensor] Push the new scale into the LIVE tensor so the NEXT
    // replayed step reads it as DATA (growth/backoff flows through hits; invScale
    // is derived in-graph from it). The JS `_scale` is now a stats-only mirror.
    //
    // Only write on an ACTUAL change. Every `.set()` mints a per-step scatter
    // that CHAINS onto the scale tensor's ref (copy-on-write when the tensor is
    // still pending) — during the lowered warmup that chain strands an earlier
    // scatter's source, read after reclaim (STRICT_LIFETIME; the setLR
    // driver-scalar KNOWN OPEN's warmup transient). Since the scale holds
    // constant across the vast majority of steps (growthInterval apart), gating
    // on change makes the write RARE — the scale tensor is materialized by the
    // time the next change lands, so its scatter is a clean true-in-place DMA
    // with a same-step source. The buffer already holds `prev` (the live tensor
    // is the single source), so skipping an unchanged write is exact.
    if (this._scaleLive && this._scale !== prev) this._scaleLive.set(this._scale);
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
    // [scaler-as-tensor] Multiply by the LIVE scale TENSOR (not the JS number),
    // so a growth/backoff rescale flows through a replayed step as DATA. The
    // 1-element scale broadcasts against the (scalar) loss.
    this._ensureLive(loss.device);
    return this.api.mul(loss, this._scaleLive!.tensor);
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
    const runtime = this.api._runtime();

    // Determine device from first param with a grad
    let device: DeviceKind = "cpu";
    for (const p of params) {
      if (p.grad) {
        device = p.device;
        break;
      }
    }

    // [scaler-as-tensor] the scale flows as the LIVE tensor; keep the JS number
    // for the elementwise (CPU) fallback path only.
    this._ensureLive(device);
    const invScale = 1.0 / this._scale;

    const backend = runtime.getBackend(device);

    // Always update param.grad's lazy ref to point at the unscaled value
    // via the `unscaleGrad` op. The graph is the source of truth: anything
    // that reads param.grad between here and the optimizer step (e.g.
    // clip_grad_norm_) sees the unscaled value, matching PyTorch's
    // unscale_() -> clip -> step() pattern. The IR fuser can still combine
    // unscaleGrad -> adamStep into a single dispatch when that's optimal,
    // but it does so from the graph instead of via an out-of-band optimizer
    // config that broke param.grad's semantics.
    if (backend.ops.unscaleGrad) {
      this._unscaleFused(params, backend, device);
    } else {
      this._unscaleElementwise(params, invScale, runtime, device);
    }
  }

  /**
   * Fused unscale path: one kernel dispatch per parameter.
   * All dispatches share a single infFlagBuffer (4 bytes, atomic).
   *
   * [scaler-as-tensor] The SCALE is delivered as a graph INPUT (node.inputs[1] =
   * the LiveScalar's persistent 1-element tensor, read LIVE from a storage
   * binding; invScale = 1/scale reciprocated in-kernel), not a frozen payload
   * number — so a growth/backoff rescale flows through a replayed step as DATA.
   * Passing the scale tensor DIRECTLY (no in-graph `div` node) avoids the div's
   * step-temporary output, which added compiled-plan fp drift over a trajectory.
   * infFlagBuffer stays on the payload (a raw GPU buffer, not a graph tensor).
   */
  private _unscaleFused(
    params: Tensor[],
    backend: Backend,
    device: DeviceKind,
  ): void {
    const infFlagBuffer = backend.ops.createInfCountBuffer?.();
    const sharedPayload = { infFlagBuffer };
    const scaleRef = this._scaleLive!.tensor._unwrap().lazyRef;

    for (const param of params) {
      const grad = param.grad;
      if (!grad) continue;

      const gradInner = grad._unwrap();

      // Create unscaleGrad lazy node: [grad, scale] inputs, single output.
      const node = createLazyIRNode(
        "unscaleGrad",
        [gradInner.lazyRef, scaleRef],
        grad.shape,
        "f32",
        device,
        sharedPayload,
      );

      // Update grad's lazyRef to point at the unscaleGrad result
      gradInner._updateLazyRef(createPendingRef(node));
    }

    // Store for deferred readback. `_fusedInfBackend` is the PERSISTENT ref the
    // driver uses to advance the CPU mirror on HITS (where this body never runs).
    this._pendingInfBuffer = infFlagBuffer;
    this._pendingInfBackend = backend;
    this._fusedInfBackend = backend;
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
      const one = this.api.full([], 1);
      const nonFiniteFlags = this.api.sub(one, finiteFlags);
      const paramInfCount = this.api.sum(nonFiniteFlags);

      // Accumulate on GPU — no item() call in the loop
      const prevAccum = infAccum;
      infAccum = this.api.add(infAccum, paramInfCount);

      toDispose.push(
        one,
        finiteFlags,
        nonFiniteFlags,
        paramInfCount,
        prevAccum,
      );
    }

    // Build shouldZero flag from final infAccum (lazy, 0-d tensor)
    const threshold = this.api.full([], 0.5);
    const shouldZero = this.api.gt(infAccum, threshold);
    const zeroTensor = this.api.full([], 0);
    toDispose.push(shouldZero, threshold, zeroTensor);

    // Loop 2: Mask grads and write back (all lazy)
    for (let i = 0; i < unscaledGrads.length; i++) {
      const maskedGrad = this.api.where(
        shouldZero,
        zeroTensor,
        unscaledGrads[i],
      );
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
   * Update scale after step(). Reads back the inf detection result for the
   * elementwise unscale path immediately (not deferred) so the infAccum
   * tensor doesn't need to survive across step boundaries.
   *
   * Call after step() at the end of each training iteration, before markStep().
   */
  async update(): Promise<void> {
    if (!this.enabled) {
      this._unscaleCalled = false;
      return;
    }

    if (!this._unscaleCalled) {
      throw new Error(
        "GradScaler.update() called before unscale_(). Call unscale_() first.",
      );
    }

    // Elementwise path: read infAccum NOW (before markStep destroys it).
    // The fused path uses a raw GPU buffer (_pendingInfBuffer) which is not
    // affected by step-scoped cleanup — it's resolved in resolveDeferred().
    if (this._pendingInfAccum) {
      const totalInfCount = await this._pendingInfAccum.item();
      this._foundInfThisStep = totalInfCount > 0.5;
      this._pendingInfAccum.dispose();
      this._pendingInfAccum = null;
      this._applyScaleAdjustment();
    }

    this._unscaleCalled = false;
  }

  /**
   * Check if the scaler is enabled.
   */
  isEnabled(): boolean {
    return this.enabled;
  }
}
