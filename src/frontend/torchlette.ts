import type {
  ArgReduceOptions,
  DeviceKind,
  DType,
  GatherOptions,
  GeluOptions,
  MaxOptions,
  MeanOptions,
  MinOptions,
  ScatterAddOptions,
  SubOptions,
  SumOptions,
  TransposeOptions,
} from "../backend/types";
import {
  destroyPendingGPUBuffers,
  evictAllPoolBuffers,
  flushBufferPool,
  flushSharedEncoder,
  isAutotuneEnabled,
  setAutotuneEnabled,
} from "../backend/webgpu";
import {
  awaitDeferredFence,
  issueDeferredFence,
} from "../backend/webgpu/buffer-pool";
import {
  getGPUMemoryLimit,
  getGPUMemoryStats,
  setGPUMemoryLimit,
} from "../backend/webgpu/memory-tracker";
import {
  type AutocastConfig,
  type AutocastContext,
  createAutocastContext,
} from "../compiler/amp";
import { shapesEqual, sizeOf } from "../core/shape";
import { currentEpoch } from "../core/epoch";
import {
  STEP_TAPE_RECORD,
  STEP_TAPE_REPLAY,
  stEndStep,
  stNoteBoundary,
  stStats,
} from "../core/step-tape";
import {
  stDropSkeleton,
  stPromoteEligibleSkeleton,
  stReplayStats,
  stSetTapeContext,
  stTapeReadyFor,
  stTapeReplayValid,
  stTryReplay,
} from "../executor/step-tape-replay";
import { getNextNodeId } from "../graph/node-factory";
import { storageTracker } from "../graph/storage-tracker";
import {
  type EngineTensor,
  RuntimeEngine,
  TidyDispatchMode,
} from "../runtime/engine";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";

// Re-export the Tensor class and DisposedTensorError from their new home
export { DisposedTensorError, Tensor } from "./tensor";

import { Tensor } from "./tensor";
import { CapturedFn, type CaptureOptions } from "./capture";

// Re-export types from frontend-types
export type {
  AutocastOptions,
  PackHook,
  TensorCreateOptions,
  TorchletteOptions,
  UnpackHook,
} from "./types";

import type {
  AutocastOptions,
  GradFn,
  SavedTensorHooksContext,
  SavedTensorSlot,
  TensorCreateOptions,
  TorchletteOptions,
} from "./types";

// Re-export backend types
export type {
  DeviceKind,
  DType,
  GatherOptions,
  MeanOptions,
  ScatterAddOptions,
  SubOptions,
  SumOptions,
  TransposeOptions,
};

import {
  BINARY_AUTOGRAD_OPS,
  OP_REGISTRY,
  UNARY_AUTOGRAD_OPS,
} from "../ops/registry";
// Import extracted modules
import {
  applyAutocastImpl,
  autocastAsyncImpl,
  autocastCastImpl,
  autocastImpl,
  savedTensorHooksImpl,
} from "./autocast";
import { backwardImpl } from "./autograd";
import {
  type BackwardContext,
  geluErfBackward,
  geluTanhBackward,
  linearBackward,
  matmulBackward,
} from "./custom-backward";
import {
  crossEntropyFusedImpl,
  layernormImpl,
  rmsnormImpl,
  scaledDotProductAttentionImpl,
  softmaxImpl,
} from "./decomposed-ops";

export class Torchlette {
  readonly runtime: RuntimeEngine;
  private readonly autocastContext: AutocastContext;
  private inCompileRegion = false;
  /** Tensors created inside compile regions — eligible for disposal after backward */
  readonly _compileCreatedTensors = new WeakSet<Tensor>();
  /** Stack of saved tensor hooks for checkpointing (§10) */
  readonly _savedTensorHooksStack: SavedTensorHooksContext[] = [];
  /** Label to capture on subsequent autograd nodes (for backward attribution) */
  private _currentNodeLabel: string | null = null;
  /** Depth counter for noGrad context. When > 0, autograd recording is disabled. */
  private _noGradDepth = 0;
  /** When true, _wrap skips markEscaped — used during checkpoint recomputation
   *  so recomputed RuntimeTensors stay tracked (not escaped) in TidyDispatchMode. */
  _inCheckpointRecompute = false;
  /** Hooks fired before each backward op */
  readonly _backwardDispatchHooks: Array<
    (info: { output: Tensor; inputs: Tensor[]; label?: string }) => void
  > = [];

  /** [capture 2a] Active upload interceptor (null = not capturing). A
   *  CapturedFn installs it for the duration of one call so it can (a) collect
   *  the fn's internal tensorFromArray uploads (the dynamic slots built inside
   *  the body) and short-circuit the body on a replay hit, and (b) track every
   *  Tensor wrapped in the window so a short-circuited partial body can be
   *  reclaimed by disposing exactly those wrappers (cheaper than a full
   *  reclamation scope on the per-token hot path). Single-slot by phase-1
   *  scope (captured calls never nest). */
  private _captureInterceptor: {
    onUpload: (shape: number[], values: Float32Array) => void;
    onWrap?: (t: Tensor) => void;
  } | null = null;

  constructor(backendName?: DeviceKind, options?: TorchletteOptions) {
    // Configure memory limit if provided (applies to GPU memory tracker)
    if (options?.memoryLimitBytes !== undefined) {
      setGPUMemoryLimit(options.memoryLimitBytes);
    }
    this.runtime = new RuntimeEngine(backendName, {
      enableFusion: options?.enableFusion ?? false,
      enableEarlyRelease: options?.enableEarlyRelease ?? false,
      enableCheckpointSegmentation:
        options?.enableCheckpointSegmentation ?? false,
      enableTrueSegmentation: options?.enableTrueSegmentation ?? false,
      executionHook: options?.executionHook,
      readHook: options?.readHook,
    });
    this.autocastContext = createAutocastContext();
    if (options?.stepScopedCleanup) {
      this.setStepScopedCleanup(true);
    }
  }

  /** Step-scoped cleanup on bare markStep() — see setStepScopedCleanup(). */
  private _stepScopedCleanup = false;

  /**
   * Enable/disable step-scoped cleanup on bare markStep() calls — the
   * ceremony-free equivalent of beginStep()/endStep() for inference loops.
   *
   * When enabled, every markStep() ends by snapshotting the tensors alive at
   * that moment; the NEXT markStep() releases everything created in between
   * and not in the snapshot. This reclaims per-step graph temporaries
   * deterministically (JS GC otherwise collects their wrappers lazily —
   * a decode loop leaks ~1 storage handle per graph node per step and the
   * markStep sweep cost grows unboundedly).
   *
   * SEMANTIC CONTRACT (same rule as the explicit ceremony, applied per
   * markStep-to-markStep interval): a tensor CREATED between two markSteps
   * and held across the second one is reclaimed even if user code still
   * references it — JS offers no reliable way to distinguish "user still
   * holds this wrapper" from "graph temporary whose wrapper GC hasn't run
   * yet". Tensors alive at the moment of enabling (or at any markStep end)
   * are persistent. For legitimately long-lived state created inside an
   * interval (e.g. a KV cache built from cat() and carried forward), call
   * persist(t) before markStep — or leave this flag off (the default), which
   * keeps the exact historical markStep semantics.
   *
   * Explicit beginStep() supersedes the implicit snapshot (it re-snapshots),
   * and the optimizer-queued implied boundaries are unaffected.
   *
   * TIMING: enabling ARMS the mechanism — the first markStep() after
   * enabling is the baseline boundary (its end-snapshot runs after
   * forceAllPending, so lazily-created state like preallocated KV slots is
   * materialized and captured); reclamation applies from the SECOND
   * markStep(). (A snapshot taken directly at enable time would miss
   * still-lazy tensors — they only enter the tracker when their storage
   * materializes — and would reclaim them after their first force.)
   *
   * Returns the previous value so callers can scope the setting:
   *   const prev = api.setStepScopedCleanup(true);
   *   try { ... } finally { api.setStepScopedCleanup(prev); }
   */
  setStepScopedCleanup(enabled: boolean): boolean {
    const prev = this._stepScopedCleanup;
    this._stepScopedCleanup = enabled;
    if (STEP_TAPE_RECORD && prev !== enabled) {
      stNoteBoundary("regime-toggle");
    }
    if (prev && !enabled) {
      // Disarm: drop any implicit end-of-markStep snapshot so a later bare
      // markStep (back on historical semantics) doesn't consume it and
      // reclaim tensors created after the disable.
      storageTracker.clearStepSnapshot();
    }
    return prev;
  }

  /**
   * Mark a tensor created mid-step (or mid markStep-interval under
   * setStepScopedCleanup) as persistent: step-boundary cleanup will not
   * reclaim its storage. THE escape hatch for long-lived state created
   * inside a step — optimizer state, EMA shadows, KV caches built from
   * graph ops and carried across markStep.
   */
  persist(a: Tensor): Tensor {
    this._assertUsable(a);
    const rt = a._unwrap();
    this.runtime.persist(rt); // adopt into the active (step or innermost-scope) snapshot
    // Scope-escape to ROOT: also adopt into every ancestor scope's snapshot so
    // the tensor survives each LIFO restore (no-op when no scope is open — the
    // pre-existing step-persist behavior).
    for (const record of this._scopeStack) {
      storageTracker.adoptIntoSnapshotToken(record.parentSnapshot, rt);
    }
    return a;
  }

  /**
   * Get the current GPU memory limit in bytes.
   */
  static getGPUMemoryLimit(): number {
    return getGPUMemoryLimit();
  }

  /**
   * Set the GPU memory limit in bytes.
   * @param limitBytes Maximum memory limit (default: 10GB)
   */
  static setGPUMemoryLimit(limitBytes: number): void {
    setGPUMemoryLimit(limitBytes);
  }

  /**
   * Get GPU memory statistics.
   */
  static getGPUMemoryStats(): ReturnType<typeof getGPUMemoryStats> {
    return getGPUMemoryStats();
  }

  /**
   * Enable or disable fusion optimizations.
   */
  setFusionEnabled(enabled: boolean): void {
    this.runtime.setFusionEnabled(enabled);
  }

  /**
   * Check if fusion is enabled.
   */
  isFusionEnabled(): boolean {
    return this.runtime.isFusionEnabled();
  }

  /**
   * Options for compile().
   */
  static CompileOptions: {
    /**
     * Enable autotuning for matmul operations within the compiled region.
     * When enabled, matmul kernels will be benchmarked and optimized for
     * the specific shapes encountered, including subgroup variants if the
     * hardware supports them.
     */
    autotune?: boolean;
  };

  /**
   * Compile a function for optimized execution.
   *
   * Inside compile regions, operations are traced and optimized with:
   * - Elementwise fusion (§15.1)
   * - CSE elimination
   * - DCE elimination
   * - Matmul autotuning (when options.autotune is true)
   *
   * @param fn Function to compile (must be synchronous)
   * @param options Optional compile options (e.g., { autotune: true })
   * @returns Compiled function that can be called repeatedly
   */
  compile<Args extends Tensor[], R extends Tensor>(
    fn: (...args: Args) => R,
    options?: { autotune?: boolean },
  ): (...args: Args) => R {
    const enableAutotune = options?.autotune ?? false;

    // Enter compile staging mode on engine
    const compiledFn = this.runtime.compile((..._args: unknown[]) => {
      // We don't use engine's TraceTensors directly,
      // instead we enable fusion and run the actual function
      return undefined;
    });

    return (...args: Args): R => {
      // Signal we're in a compile region
      this.inCompileRegion = true;
      const wasFusionEnabled = this.runtime.isFusionEnabled();
      const wasAutotuneEnabled = isAutotuneEnabled();

      // Per spec §0.1 goal #2: fusion runs ONLY inside compiled regions
      this.runtime.setFusionEnabled(true);

      // Enable autotune if requested
      if (enableAutotune) {
        setAutotuneEnabled(true);
      }

      try {
        // Run the engine compile wrapper (for staging tracking)
        compiledFn();

        // Per spec §1.6: "non-returned trace tensors are disposed at epoch end"
        // Wrap execution in tidy() so intermediate tensors are auto-disposed.
        // Only returned tensors escape the scope.
        const result = this.tidy(() => fn(...args));
        // The returned tensor has been promoted to user-held — it may be reused
        // as an external input to a later compile call, so it must not be treated
        // as an internal intermediate during backward cleanup.
        this._compileCreatedTensors.delete(result);
        return result;
      } finally {
        // Restore previous states
        this.runtime.setFusionEnabled(wasFusionEnabled);
        // NOTE: Don't reset autotune here. Lazy execution happens AFTER this
        // finally block returns, so we need to keep autotune enabled.
        // The flag is reset via setAutotuneEnabled(false) in tests or by
        // the user when they're done with autotuning.
        // Only restore if autotune wasn't enabled by this compile call:
        if (!enableAutotune) {
          setAutotuneEnabled(wasAutotuneEnabled);
        }
        this.inCompileRegion = false;
      }
    };
  }

  /**
   * Check if autocast is currently enabled.
   */
  get isAutocastEnabled(): boolean {
    return this.autocastContext.current.enabled;
  }

  /**
   * Get the current autocast configuration.
   */
  get currentAutocastConfig(): AutocastConfig {
    return { ...this.autocastContext.current };
  }

  setBackend(name: DeviceKind): void {
    this.runtime.setBackend(name);
  }

  /**
   * Set the default device for tensor creation (like PyTorch's torch.set_default_device).
   * All tensor factory calls (zeros, randn, tensorFromArray, etc.) and nn.Module
   * constructors will use this device when no explicit device is provided.
   */
  setDefaultDevice(device: DeviceKind): void {
    this.runtime.setBackend(device);
  }

  /**
   * Get the current default device.
   */
  getDefaultDevice(): DeviceKind {
    return this.runtime.currentDefaultDevice;
  }

  // ============================================================================
  // Autocast (§12) — delegated to frontend-autocast.ts
  // ============================================================================

  autocast<T>(fn: () => T, options?: AutocastOptions): T {
    return autocastImpl(this, fn, options);
  }

  async autocastAsync<T>(
    fn: () => Promise<T>,
    options?: AutocastOptions,
  ): Promise<T> {
    return autocastAsyncImpl(this, fn, options);
  }

  saved_tensors_hooks<T>(
    packHook: (tensor: Tensor) => unknown,
    unpackHook: (packed: unknown) => Tensor,
    fn: () => T,
  ): T {
    return savedTensorHooksImpl(this, packHook, unpackHook, fn);
  }

  /**
   * Get the current saved tensor hooks context (if any).
   * Returns the topmost hooks on the stack, or null if none.
   */
  _getSavedTensorHooks(): SavedTensorHooksContext | null {
    return this._savedTensorHooksStack.length > 0
      ? this._savedTensorHooksStack[this._savedTensorHooksStack.length - 1]
      : null;
  }

  /** Set an opaque label captured on subsequent autograd nodes. */
  setNodeLabel(label: string | null): void {
    this._currentNodeLabel = label;
  }

  /**
   * Execute `fn` with autograd recording disabled.
   * Inside noGrad, ops never create grad nodes or save tensors for backward,
   * even when inputs have requiresGrad=true. Matches PyTorch's torch.no_grad().
   */
  noGrad<T>(fn: () => T): T {
    this._noGradDepth++;
    try {
      return fn();
    } finally {
      this._noGradDepth--;
    }
  }

  /** Returns true if autograd recording is currently enabled. */
  isGradEnabled(): boolean {
    return this._noGradDepth === 0;
  }

  /** Register a hook that fires before each backward op. Returns unregister function. */
  onBackwardDispatch(
    hook: (info: { output: Tensor; inputs: Tensor[]; label?: string }) => void,
  ): () => void {
    this._backwardDispatchHooks.push(hook);
    return () => {
      const idx = this._backwardDispatchHooks.indexOf(hook);
      if (idx >= 0) this._backwardDispatchHooks.splice(idx, 1);
    };
  }

  /**
   * Get the autocast context for use in compiled regions.
   * This is used internally for select-gated commit logic.
   */
  _getAutocastContext(): AutocastContext {
    return this.autocastContext;
  }

  // ============================================================================
  // Creation ops
  // ============================================================================

  tensorFromArray(
    values: number[] | Float32Array | Int32Array | Uint32Array | Uint16Array,
    shape: number[],
    options?: TensorCreateOptions,
  ): Tensor {
    // [capture 2a] a captured call's per-step upload — record it for skeleton
    // re-dressing (the derived upload slots). f32 values only (the decode
    // uploads are f32); other dtypes are not per-step decode uploads.
    if (this._captureInterceptor) {
      this._captureInterceptor.onUpload(
        shape,
        values instanceof Float32Array ? values : Float32Array.from(values),
      );
    }
    return this._wrap(
      this.runtime.tensorFromArray(
        values,
        shape,
        options?.device,
        options?.dtype,
      ),
      options?.requiresGrad ?? false,
    );
  }

  /**
   * Set the global random seed for all subsequent random ops (rand, randn, bernoulli).
   * All Torchlette instances sharing the same runtime use the same RNG state,
   * so this affects all random generation globally.
   *
   * Usage: `api.manualSeed(42)` — equivalent to PyTorch's `torch.manual_seed(42)`.
   */
  manualSeed(seed: number): void {
    this.runtime.setRngSeed(seed);
  }

  rand(shape: number[], options?: TensorCreateOptions): Tensor {
    return this._wrap(
      this.runtime.rand(shape, options?.device),
      options?.requiresGrad ?? false,
    );
  }

  randn(shape: number[], options?: TensorCreateOptions): Tensor {
    return this._wrap(
      this.runtime.randn(shape, options?.device),
      options?.requiresGrad ?? false,
    );
  }

  bernoulli(shape: number[], p = 0.5, options?: TensorCreateOptions): Tensor {
    if (p < 0 || p > 1)
      throw new Error(
        `Bernoulli probability must be between 0 and 1, got ${p}`,
      );
    return this._wrap(
      this.runtime.bernoulli(shape, p, options?.device),
      options?.requiresGrad ?? false,
    );
  }

  zeros(shape: number[], options?: TensorCreateOptions): Tensor {
    return this._wrap(
      this.runtime.zeros(shape, options?.device, options?.dtype),
      options?.requiresGrad ?? false,
    );
  }

  ones(shape: number[], options?: TensorCreateOptions): Tensor {
    return this.full(shape, 1, options);
  }

  full(
    shape: number[],
    fillValue: number,
    options?: TensorCreateOptions,
  ): Tensor {
    return this._wrap(
      this.runtime.full(shape, fillValue, options?.device, options?.dtype),
      options?.requiresGrad ?? false,
    );
  }

  arange(
    end: number,
    options?: {
      start?: number;
      step?: number;
      device?: DeviceKind;
      requiresGrad?: boolean;
    },
  ): Tensor {
    const start = options?.start ?? 0;
    const step = options?.step ?? 1;
    return this._wrap(
      this.runtime.arange(end, start, step, options?.device),
      options?.requiresGrad ?? false,
    );
  }

  tril(a: Tensor, k = 0): Tensor {
    this._assertUsable(a);
    return this._wrap(this.runtime.tril(a._unwrap(), k), false);
  }

  triu(a: Tensor, k = 0): Tensor {
    this._assertUsable(a);
    return this._wrap(this.runtime.triu(a._unwrap(), k), false);
  }

  // ============================================================================
  // Autocast dispatch helpers (§12) — delegated to frontend-autocast.ts
  // ============================================================================

  _autocastCast(a: Tensor, targetDtype: DType): Tensor {
    return autocastCastImpl(this, a, targetDtype);
  }

  _applyAutocast(op: string, inputs: Tensor[]): Tensor[] {
    return applyAutocastImpl(this, op, inputs);
  }

  // ============================================================================
  // Math ops (binary, unary, reductions — stay in hub)
  // ============================================================================

  /** Generic dispatcher for registry-driven unary ops. */
  _dispatchUnary(opName: string, a: Tensor): Tensor {
    const def = OP_REGISTRY[opName];
    this._assertUsable(a);
    const castA = def.autocast ? this._applyAutocast(def.autocast, [a])[0] : a;
    const inner = (
      this.runtime as unknown as Record<
        string,
        (a: RuntimeTensor) => RuntimeTensor
      >
    )[opName](castA._unwrap());
    if (!def.grad) return this._wrap(inner);
    const needsSave = def.needsSave !== false;
    const tensorsToSave = needsSave && a.requiresGrad ? [castA] : [];
    return this._wrapWithGrad(
      inner,
      [a],
      (grad, getSaved) => {
        return [
          def.grad?.(
            this.runtime,
            grad,
            needsSave ? getSaved(0)?._unwrap() : undefined,
          ) ?? null,
        ];
      },
      tensorsToSave,
    );
  }

  /**
   * Shared dispatch for binary ops that accept Tensor|number operands.
   * Handles tensor+tensor (with autocast) and tensor+scalar branches,
   * including sumToShape gradient reduction and tensor saving for backward.
   *
   * @param ttGrad - Tensor+Tensor grad: return [gradA, gradB] (before sumToShape)
   * @param tsGrad - Tensor+Scalar grad: return [gradTensor] (before sumToShape).
   *                 If omitted, gradient passes through unchanged.
   * @param save - Whether to save input tensors for backward (for getSaved access in grad)
   */
  private _dispatchBinary(
    opName: string,
    a: Tensor | number,
    b: Tensor | number,
    ttGrad: (
      g: RuntimeTensor,
      getSaved: (i: number) => Tensor,
    ) => [RuntimeTensor, RuntimeTensor],
    tsGrad?: (
      g: RuntimeTensor,
      getSaved: (i: number) => Tensor,
      scalar: number,
      scalarIsA: boolean,
    ) => RuntimeTensor[],
    save = false,
  ): Tensor {
    const tensors = [a, b].filter((x): x is Tensor => typeof x !== "number");
    this._assertUsable(...tensors);
    const rt = this.runtime;

    if (typeof a !== "number" && typeof b !== "number") {
      [a, b] = this._applyAutocast(opName, [a, b]) as [Tensor, Tensor];
      const inner = (rt as any)[opName](a._unwrap(), b._unwrap());
      const aShape = a.shape,
        bShape = b.shape;
      const tensorsToSave =
        save && (a.requiresGrad || b.requiresGrad) ? [a, b] : [];
      return this._wrapWithGrad(
        inner,
        [a, b],
        (grad, getSaved) => {
          const [gA, gB] = ttGrad(grad, getSaved);
          return [this._sumToShape(gA, aShape), this._sumToShape(gB, bShape)];
        },
        tensorsToSave,
      );
    }

    // Tensor + Scalar
    const tensorInput = typeof a !== "number" ? a : (b as Tensor);
    const scalarVal = typeof a === "number" ? a : (b as number);
    const scalarIsA = typeof a === "number";
    const inner = (rt as any)[opName](
      typeof a === "number" ? a : a._unwrap(),
      typeof b === "number" ? b : b._unwrap(),
    );
    const tensorsToSave = save && tensorInput.requiresGrad ? [tensorInput] : [];
    const gFn = tsGrad ?? ((g: RuntimeTensor) => [g]);
    return this._wrapWithGrad(
      inner,
      [tensorInput],
      (grad, getSaved) =>
        gFn(grad, getSaved, scalarVal, scalarIsA).map((g) =>
          this._sumToShape(g, tensorInput.shape),
        ),
      tensorsToSave,
    );
  }

  /** Generic dispatcher for registry-driven binary ops (gradient specs in OP_REGISTRY). */
  _dispatchBinaryFromTable(
    opName: string,
    a: Tensor | number,
    b: Tensor | number,
  ): Tensor {
    const def = OP_REGISTRY[opName];
    const rt = this.runtime;
    return this._dispatchBinary(
      opName,
      a,
      b,
      (g, gs) => def.ttGrad!(rt, g, (i) => gs(i)._unwrap()),
      def.tsGrad
        ? (g, gs, s, isA) => def.tsGrad!(rt, g, (i) => gs(i)._unwrap(), s, isA)
        : undefined,
      def.saveBinary ?? false,
    );
  }

  sub(a: Tensor, b: Tensor | number, options?: SubOptions): Tensor {
    if (typeof b === "number") {
      // scalar sub: a - b*alpha => a + (-b*alpha)
      const alpha = options?.alpha ?? 1;
      return this.add(a, -(b * alpha));
    }
    this._assertUsable(a, b);
    [a, b] = this._applyAutocast("sub", [a, b]) as [Tensor, Tensor];
    const inner = this.runtime.sub(a._unwrap(), b._unwrap(), options);
    const aShape = a.shape;
    const bShape = b.shape;
    return this._wrapWithGrad(inner, [a, b], (grad, _getSaved) => {
      const alpha = options?.alpha ?? 1;
      const gradA = this._sumToShape(grad, aShape);
      const scaled = this.runtime.mul(grad, -alpha);
      const gradB = this._sumToShape(scaled, bShape);
      return [gradA, gradB];
    });
  }

  matmul(a: Tensor, b: Tensor): Tensor {
    this._assertUsable(a, b);
    const [castA, castB] = this._applyAutocast("matmul", [a, b]) as [
      Tensor,
      Tensor,
    ];
    const inner = this.runtime.matmul(castA._unwrap(), castB._unwrap());
    const aShape = a.shape;
    const bShape = b.shape;
    const tensorsToSave =
      a.requiresGrad || b.requiresGrad ? [castA, castB] : [];
    return this._wrapWithGrad(
      inner,
      [a, b],
      matmulBackward(this._backwardCtx(), aShape, bShape),
      tensorsToSave,
    );
  }

  /**
   * Linear transformation: Y = input @ weight.T + bias
   * Custom backward computes dW = dY.T @ X directly in weight's shape,
   * eliminating the transpose op that generic matmul backward would produce.
   */
  linear(input: Tensor, weight: Tensor, bias: Tensor | null = null): Tensor {
    this._assertUsable(input, weight);
    if (bias) this._assertUsable(bias);
    const [castInput, castWeight] = this._applyAutocast("matmul", [
      input,
      weight,
    ]) as [Tensor, Tensor];

    // Forward: Y = input @ weight.T (+ bias)
    const wT = this.runtime.transpose(castWeight._unwrap(), {
      dim0: castWeight.shape.length - 2,
      dim1: castWeight.shape.length - 1,
    });
    let inner = this.runtime.matmul(castInput._unwrap(), wT);
    if (bias) {
      inner = this.runtime.add(inner, bias._unwrap());
    }

    const inputShape = input.shape;
    const weightShape = weight.shape;
    const needsInputGrad = input.requiresGrad;
    const needsWeightGrad = weight.requiresGrad;
    const allInputs = bias ? [input, weight, bias] : [input, weight];
    // Save only what's needed: weight for dX, input for dW, bias for dBias
    const toSave: Tensor[] = [];
    if (needsInputGrad || needsWeightGrad) {
      if (needsInputGrad) toSave.push(castWeight); // saved[0] = weight (for dX)
      if (needsWeightGrad) toSave.push(castInput); // saved[1 or 0] = input (for dW)
      if (bias) toSave.push(bias);
    }

    return this._wrapWithGrad(
      inner,
      allInputs,
      linearBackward(
        this._backwardCtx(),
        inputShape,
        weightShape,
        needsInputGrad,
        needsWeightGrad,
        !!bias,
      ),
      toSave,
    );
  }

  gelu(a: Tensor, options?: GeluOptions): Tensor {
    this._assertUsable(a);
    const approximate = options?.approximate ?? "tanh";
    const inner = this.runtime.gelu(a._unwrap(), options);
    const tensorsToSave = a.requiresGrad ? [a] : [];

    const bwdFn =
      approximate === "tanh"
        ? geluTanhBackward(this._backwardCtx())
        : geluErfBackward(this._backwardCtx());
    return this._wrapWithGrad(inner, [a], bwdFn, tensorsToSave);
  }

  clamp(a: Tensor, min: number | null, max: number | null): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.clamp(a._unwrap(), min, max);
    const tensorsToSave = a.requiresGrad ? [a] : [];
    // d/dx clamp(x, min, max) = 1 where min <= x <= max, 0 elsewhere
    return this._wrapWithGrad(
      inner,
      [a],
      (grad, getSaved) => {
        const savedA = getSaved(0);
        const x = savedA._unwrap();
        // Build a mask: 1.0 where x is within [min, max], 0.0 elsewhere
        let mask = grad;
        if (min !== null) {
          const geMin = this.runtime.ge(x, min);
          mask = this.runtime.mul(mask, geMin);
        }
        if (max !== null) {
          const leMax = this.runtime.le(x, max);
          mask = this.runtime.mul(mask, leMax);
        }
        return [mask];
      },
      tensorsToSave,
    );
  }

  softplus(a: Tensor): Tensor {
    this._assertUsable(a);
    // softplus(x) = log(1 + exp(x))
    const one = this.runtime.full(a.shape, 1, a.device);
    const expA = this.runtime.exp(a._unwrap());
    const inner = this.runtime.log(this.runtime.add(one, expA));
    // d/dx softplus(x) = sigmoid(x)
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => {
      const sigA = this.runtime.sigmoid(a._unwrap());
      return [this.runtime.mul(grad, sigA)];
    });
  }

  fmod(a: Tensor, b: Tensor): Tensor {
    this._assertUsable(a, b);
    // fmod(a, b) = a - b * floor(a / b)
    const quotient = this.runtime.div(a._unwrap(), b._unwrap());
    const floored = this.runtime.floor(quotient);
    const inner = this.runtime.sub(
      a._unwrap(),
      this.runtime.mul(b._unwrap(), floored),
    );
    return this._wrap(inner);
  }

  /**
   * Embedding lookup: weight[indices] for each index.
   * Like PyTorch's F.embedding(input, weight).
   *
   * @param weight - Embedding table of shape [numEmbeddings, embeddingDim]
   * @param indices - Index tensor of arbitrary shape [...]
   * @returns Tensor of shape [..., embeddingDim]
   */
  embedding(weight: Tensor, indices: Tensor): Tensor {
    this._assertUsable(weight, indices);
    const inputShape = indices.shape;
    const embDim = weight.shape[1];
    const numElements = sizeOf(inputShape);

    // Gather kernel accepts f32/i32/u32 index tensors natively — no cast needed.
    // Flatten → expand → contiguous → gather → reshape
    // Uses existing gather autograd (scatterAdd backward)
    const flat = this.reshape(indices, [numElements]);
    const expanded = this.expand(this.reshape(flat, [numElements, 1]), [
      numElements,
      embDim,
    ]);
    const contig = this.contiguous(expanded);
    const gathered = this.gather(weight, contig, { dim: 0 });
    return this.reshape(gathered, [...inputShape, embDim]);
  }

  expand(a: Tensor, shape: number[]): Tensor {
    this._assertUsable(a);
    const aShape = a.shape;
    const inner = this.runtime.expand(a._unwrap(), shape);
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this._sumToShape(grad, aShape),
    ]);
  }

  /** Generic dispatcher for reduction ops with autograd. */
  private _dispatchReduction(
    opName: string,
    a: Tensor,
    options?: { dim?: number | number[] | null; keepdim?: boolean },
  ): number | Tensor {
    this._assertUsable(a);
    const autocastOp = opName === "sum" || opName === "mean" ? opName : null;
    const castA = autocastOp ? this._applyAutocast(autocastOp, [a])[0] : a;
    const result = (this.runtime as any)[opName](castA._unwrap(), options);
    if (typeof result === "number") {
      if (a.requiresGrad && opName === "mean") {
        throw new Error("mean with requiresGrad must specify dim");
      }
      return typeof result === "number" && !a.requiresGrad
        ? result
        : this._wrap(
            typeof result === "number" ? this.runtime.full([], result) : result,
          );
    }
    // Non-differentiable reductions (max, min without dim grad)
    if (opName === "max" || opName === "min") {
      return this._wrap(result);
    }
    const aShape = a.shape;
    const dims = this._normalizeDims(options?.dim ?? null, aShape.length);
    const keepdim = options?.keepdim ?? false;
    return this._wrapWithGrad(result, [a], (grad, _getSaved) => {
      const expanded = this._expandGrad(grad, aShape, dims, keepdim);
      if (opName === "mean") {
        const reduceCount =
          dims.length === 0 ? 1 : dims.reduce((acc, d) => acc * aShape[d], 1);
        return [this.runtime.mul(expanded, 1 / reduceCount)];
      }
      return [expanded];
    });
  }

  sum(a: Tensor, options?: SumOptions): Tensor {
    return this._dispatchReduction("sum", a, options) as Tensor;
  }
  max(a: Tensor, options?: MaxOptions): number | Tensor {
    return this._dispatchReduction("max", a, options);
  }
  min(a: Tensor, options?: MinOptions): number | Tensor {
    return this._dispatchReduction("min", a, options);
  }
  mean(a: Tensor, options?: MeanOptions): number | Tensor {
    return this._dispatchReduction("mean", a, options);
  }

  argmax(a: Tensor, options: ArgReduceOptions): Tensor {
    this._assertUsable(a);
    return this._wrap(this.runtime.argmax(a._unwrap(), options));
  }
  argmin(a: Tensor, options: ArgReduceOptions): Tensor {
    this._assertUsable(a);
    return this._wrap(this.runtime.argmin(a._unwrap(), options));
  }

  variance(
    a: Tensor,
    options?: {
      dim?: number | number[] | null;
      correction?: number;
      keepdim?: boolean;
    },
  ): Tensor {
    this._assertUsable(a);
    const dim = options?.dim ?? null;
    const correction = options?.correction ?? 1;
    const keepdim = options?.keepdim ?? false;

    const meanVal = this.mean(a, { dim, keepdim: true }) as Tensor;
    const diff = this.sub(a, meanVal);
    const sq = this.mul(diff, diff);
    const sumSq = this.sum(sq, { dim, keepdim });

    const aShape = a.shape;
    const dims = this._normalizeDims(dim, aShape.length);
    const reduceCount =
      dims.length === 0
        ? aShape.reduce((acc, s) => acc * s, 1)
        : dims.reduce((acc, d) => acc * aShape[d], 1);
    const denom = Math.max(reduceCount - correction, 0);
    if (denom === 0) {
      throw new Error(
        "variance: correction >= number of elements in reduction",
      );
    }
    return this.div(sumSq, denom);
  }

  std(
    a: Tensor,
    options?: {
      dim?: number | number[] | null;
      correction?: number;
      keepdim?: boolean;
    },
  ): Tensor {
    return this.sqrt(this.variance(a, options));
  }

  // ============================================================================
  // Comparison ops
  // ============================================================================

  _cmpOp(
    op: "gt" | "lt" | "ge" | "le" | "eq" | "ne",
    a: Tensor,
    b: Tensor,
  ): Tensor {
    this._assertUsable(a, b);
    return this._wrap(this.runtime[op](a._unwrap(), b._unwrap()));
  }

  // ============================================================================
  // Fused ops — delegated to frontend-fused-ops.ts
  // ============================================================================

  softmax(a: Tensor, dim: number): Tensor {
    return softmaxImpl(this, a, dim);
  }

  _crossEntropyFused(
    logits: Tensor,
    targets: Tensor,
    ignoreIndex?: number,
  ): Tensor {
    return crossEntropyFusedImpl(this, logits, targets, ignoreIndex);
  }

  /**
   * Apply Rotary Position Embedding (RoPE) to a Q or K tensor.
   * Input `qk` has shape [..., seqLen, headDim] (contiguous, headDim even).
   * `cos`/`sin` are precomputed tables of shape [seqLen, headDim/2].
   *
   * Uses the half-split convention: first half rotates with -sin, second
   * half rotates with +sin. Backward is the same kernel with sin negated.
   *
   * Single fused GPU dispatch.
   */
  applyRoPE(qk: Tensor, cos: Tensor, sin: Tensor): Tensor {
    this._assertUsable(qk, cos, sin);
    const shape = qk.shape;
    const headDim = shape[shape.length - 1];
    const seqLen = shape[shape.length - 2];
    if (headDim % 2 !== 0) {
      throw new Error(`applyRoPE: headDim must be even, got ${headDim}`);
    }
    const total = shape.reduce((a, b) => a * b, 1);
    const fwdConfig = { total, seqLen, headDim, sinScale: 1 };
    const bwdConfig = { total, seqLen, headDim, sinScale: -1 };
    const result = this.runtime.fusedRoPE(
      qk._unwrap(),
      cos._unwrap(),
      sin._unwrap(),
      fwdConfig,
    );
    const cosInner = cos._unwrap();
    const sinInner = sin._unwrap();
    return this._wrapWithGrad(result, [qk], (grad, _getSaved) => {
      const gradQK = this.runtime.fusedRoPE(
        grad,
        cosInner,
        sinInner,
        bwdConfig,
      );
      return [gradQK];
    });
  }

  scaledDotProductAttention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale?: number,
    isCausal = false,
  ): Tensor {
    return scaledDotProductAttentionImpl(this, q, k, v, scale, isCausal);
  }

  layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps = 1e-5): Tensor {
    return layernormImpl(this, x, weight, bias, eps);
  }

  rmsnorm(x: Tensor, weight: Tensor, eps = 1e-5): Tensor {
    return rmsnormImpl(this, x, weight, eps);
  }

  // ============================================================================
  // Data access
  // ============================================================================

  async cpu(a: Tensor): Promise<number[]> {
    this._assertUsable(a);
    return this._runEntryPoint(async () => {
      this.runtime.forceRead(a.baseId);
      return this.runtime.cpu(a._unwrap());
    });
  }

  async item(a: Tensor): Promise<number> {
    this._assertUsable(a);
    return this._runEntryPoint(async () => {
      this.runtime.forceRead(a.baseId);
      return this.runtime.item(a._unwrap());
    });
  }

  /**
   * Top-K readback for sampling: the top-k (value, index) pairs of a 1-D
   * slice of `a` (default: the whole flattened tensor), sorted by
   * (value desc, index asc). On WebGPU this runs a GPU prefilter kernel and
   * reads back only ~2k*4 bytes instead of the full tensor — the fast path
   * for decode-time sampling over large vocab logits. `indices[0]` is the
   * greedy argmax (bit-identical to a full-logits first-max linear scan).
   * `offset`/`length` are in elements of the flattened tensor.
   */
  async readTopK(
    a: Tensor,
    k: number,
    opts?: { offset?: number; length?: number },
  ): Promise<{ values: Float32Array; indices: Int32Array }> {
    this._assertUsable(a);
    return this._runEntryPoint(async () => {
      this.runtime.forceRead(a.baseId);
      return this.runtime.readTopK(a._unwrap(), k, opts);
    });
  }

  // ============================================================================
  // Device transfer
  // ============================================================================

  to(a: Tensor, device: DeviceKind): Tensor {
    this._assertUsable(a);
    if (a.device === device) {
      return a;
    }
    const inner = this.runtime.transfer(a._unwrap(), device);
    return this._wrap(inner, a.requiresGrad);
  }

  async toNow(a: Tensor, device: DeviceKind): Promise<Tensor> {
    this._assertUsable(a);
    return this._runEntryPoint(async () => {
      this.runtime.forceRead(a.baseId);
      const inner = await this.runtime.transferNow(a._unwrap(), device);
      return this._wrap(inner, a.requiresGrad);
    });
  }

  // ============================================================================
  // In-place operations (§4.3-4.4)
  // ============================================================================

  private _inPlace(dst: Tensor, fn: () => void, ...extra: Tensor[]): Tensor {
    this._assertUsable(dst, ...extra);
    fn();
    this._debug_baseCommit(dst.baseId, this.runtime.nextMutId());
    return dst;
  }

  copy_(dst: Tensor, src: Tensor): Tensor {
    return this._inPlace(
      dst,
      () => this.runtime.copy_(dst._unwrap(), src._unwrap()),
      src,
    );
  }
  add_(dst: Tensor, src: Tensor): Tensor {
    return this._inPlace(
      dst,
      () => this.runtime.add_(dst._unwrap(), src._unwrap()),
      src,
    );
  }
  zero_(dst: Tensor): Tensor {
    return this._inPlace(dst, () => this.runtime.zero_(dst._unwrap()));
  }
  fill_(dst: Tensor, value: number): Tensor {
    return this._inPlace(dst, () => this.runtime.fill_(dst._unwrap(), value));
  }
  mul_(dst: Tensor, value: number): Tensor {
    return this._inPlace(dst, () => this.runtime.mul_(dst._unwrap(), value));
  }

  // ============================================================================
  // Gather/scatter/where
  // ============================================================================

  gather(a: Tensor, index: Tensor, options: GatherOptions): Tensor {
    this._assertUsable(a, index);
    const inner = this.runtime.gather(a._unwrap(), index._unwrap(), options);
    const aShape = a.shape;
    const indexInner = index._unwrap();
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => {
      const z = this.runtime.zeros(aShape);
      return [this.runtime.scatterAdd(z, indexInner, grad, options)];
    });
  }

  scatterAdd(
    a: Tensor,
    index: Tensor,
    src: Tensor,
    options: ScatterAddOptions,
  ): Tensor {
    this._assertUsable(a, index, src);
    const inner = this.runtime.scatterAdd(
      a._unwrap(),
      index._unwrap(),
      src._unwrap(),
      options,
    );
    const indexInner = index._unwrap();
    return this._wrapWithGrad(inner, [a, src], (grad, _getSaved) => {
      const grad_a = grad;
      const grad_src = this.runtime.gather(grad, indexInner, options);
      return [grad_a, grad_src];
    });
  }

  where(condition: Tensor, x: Tensor, y: Tensor): Tensor {
    this._assertUsable(condition, x, y);
    const inner = this.runtime.where(
      condition._unwrap(),
      x._unwrap(),
      y._unwrap(),
    );
    const xShape = x.shape;
    const yShape = y.shape;
    const conditionInner = condition._unwrap();
    return this._wrapWithGrad(inner, [x, y], (grad, _getSaved) => {
      const zerosTensor = this.runtime.zeros(grad.shape, grad.device);
      const grad_x = this.runtime.where(conditionInner, grad, zerosTensor);
      const grad_y = this.runtime.where(conditionInner, zerosTensor, grad);
      return [
        this._sumToShape(grad_x, xShape),
        this._sumToShape(grad_y, yShape),
      ];
    });
  }

  // ============================================================================
  // View/reshape/transpose ops
  // ============================================================================

  view(a: Tensor, shape: number[]): Tensor {
    return this.reshape(a, shape);
  }

  reshape(a: Tensor, shape: number[]): Tensor {
    this._assertUsable(a);
    const aShape = a.shape;
    const inner = this.runtime.reshape(a._unwrap(), shape);
    return this._wrapWithGrad(inner, [a], (grad) => [
      this.runtime.reshape(grad, aShape),
    ]);
  }

  squeeze(a: Tensor, dim?: number): Tensor {
    this._assertUsable(a);
    const aShape = a.shape;
    let newShape: number[];
    if (dim !== undefined) {
      const d = dim < 0 ? dim + aShape.length : dim;
      newShape =
        aShape[d] === 1
          ? [...aShape.slice(0, d), ...aShape.slice(d + 1)]
          : [...aShape];
    } else {
      newShape = aShape.filter((s) => s !== 1);
    }
    if (newShape.length === aShape.length) return a;
    const inner = this.runtime.reshape(a._unwrap(), newShape);
    return this._wrapWithGrad(inner, [a], (grad) => [
      this.runtime.reshape(grad, aShape),
    ]);
  }

  unsqueeze(a: Tensor, dim: number): Tensor {
    this._assertUsable(a);
    const aShape = a.shape;
    const d = dim < 0 ? dim + aShape.length + 1 : dim;
    const newShape = [...aShape.slice(0, d), 1, ...aShape.slice(d)];
    const inner = this.runtime.reshape(a._unwrap(), newShape);
    return this._wrapWithGrad(inner, [a], (grad) => [
      this.runtime.reshape(grad, aShape),
    ]);
  }

  transpose(a: Tensor, options: TransposeOptions): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.transpose(a._unwrap(), options);
    return this._wrapWithGrad(inner, [a], (grad) => [
      this.runtime.transpose(grad, options),
    ]);
  }

  permute(a: Tensor, dims: number[]): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.permute(a._unwrap(), dims);
    const inverseDims = new Array<number>(dims.length);
    for (let i = 0; i < dims.length; i++) inverseDims[dims[i]] = i;
    return this._wrapWithGrad(inner, [a], (grad) => [
      this.runtime.permute(grad, inverseDims),
    ]);
  }

  contiguous(a: Tensor): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.contiguous(a._unwrap());
    return this._wrapWithGrad(inner, [a], (grad) => [grad]);
  }

  narrow(a: Tensor, dim: number, start: number, length: number): Tensor {
    this._assertUsable(a);
    const originalLength = a.shape[dim];
    const inner = this.runtime.narrow(a._unwrap(), dim, start, length);
    return this._wrapWithGrad(inner, [a], (grad) => [
      this.runtime.narrowBackward(grad, dim, start, originalLength),
    ]);
  }

  chunk(a: Tensor, chunks: number, dim = 0): Tensor[] {
    this._assertUsable(a);
    const rank = a.shape.length;
    const d = dim < 0 ? dim + rank : dim;
    if (d < 0 || d >= rank)
      throw new Error(`chunk: dim ${dim} out of range for rank ${rank}`);
    if (chunks <= 0) throw new Error("chunk: chunks must be positive");
    const dimSize = a.shape[d];
    const chunkSize = Math.ceil(dimSize / chunks);
    const results: Tensor[] = [];
    for (let i = 0; i < chunks; i++) {
      const start = i * chunkSize;
      if (start >= dimSize) break;
      const length = Math.min(chunkSize, dimSize - start);
      results.push(this.narrow(a, d, start, length));
    }
    return results;
  }

  split(a: Tensor, splitSizeOrSections: number | number[], dim = 0): Tensor[] {
    this._assertUsable(a);
    const rank = a.shape.length;
    const d = dim < 0 ? dim + rank : dim;
    if (d < 0 || d >= rank)
      throw new Error(`split: dim ${dim} out of range for rank ${rank}`);
    const dimSize = a.shape[d];
    let sizes: number[];
    if (typeof splitSizeOrSections === "number") {
      sizes = [];
      for (let pos = 0; pos < dimSize; pos += splitSizeOrSections) {
        sizes.push(Math.min(splitSizeOrSections, dimSize - pos));
      }
    } else {
      sizes = splitSizeOrSections;
      const total = sizes.reduce((s, v) => s + v, 0);
      if (total !== dimSize) {
        throw new Error(
          `split: sizes sum ${total} != dimension size ${dimSize}`,
        );
      }
    }
    const results: Tensor[] = [];
    let start = 0;
    for (const size of sizes) {
      results.push(this.narrow(a, d, start, size));
      start += size;
    }
    return results;
  }

  cat(tensors: Tensor[], dim = 0): Tensor {
    if (tensors.length === 0) throw new Error("cat: empty tensor list");
    for (const t of tensors) this._assertUsable(t);
    const d = dim < 0 ? dim + tensors[0].shape.length : dim;
    const sizes = tensors.map((t) => t.shape[d]);
    const inner = this.runtime.cat(
      tensors.map((t) => t._unwrap()),
      { dim: d },
    );
    return this._wrapWithGrad(inner, tensors, (grad, _getSaved) => {
      const grads: ReturnType<typeof this.runtime.narrow>[] = [];
      let offset = 0;
      for (const size of sizes) {
        grads.push(this.runtime.narrow(grad, d, offset, size));
        offset += size;
      }
      return grads;
    });
  }

  stack(tensors: Tensor[], dim = 0): Tensor {
    if (tensors.length === 0) throw new Error("stack: empty tensor list");
    const unsqueezed = tensors.map((t) => this.unsqueeze(t, dim));
    return this.cat(unsqueezed, dim);
  }

  toDtype(a: Tensor, dtype: DType): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.cast(a._unwrap(), dtype);
    return this._wrap(inner);
  }

  detach(a: Tensor): Tensor {
    this._assertUsable(a);
    return this._wrap(a._unwrap());
  }

  // ============================================================================
  // Autograd backward — delegated to frontend-autograd.ts
  // ============================================================================

  async backward(a: Tensor, grad?: Tensor): Promise<void> {
    // A queued implied step boundary (optimizer.step() with no explicit
    // markStep) commits here: backward is the first point in the NEXT
    // iteration that is async, always present in a training loop, and
    // early enough to keep memory flat. The previous step's residue is
    // forced and its temporaries demoted before the new backward runs;
    // gen-scoping protects this iteration's already-built forward graph.
    await this._commitPendingStepBoundary();
    return backwardImpl(this, a, grad);
  }

  // ============================================================================
  // Public helper methods (promoted from private for extracted modules)
  // ============================================================================

  /** Tensors created during checkpoint recomputation — disposable after backward. */
  _checkpointRecomputeTensors: Set<Tensor> | null = null;

  _wrap(inner: RuntimeTensor, requiresGrad = false): Tensor {
    if (!this._inCheckpointRecompute) {
      this.runtime.markEscaped(inner);
    }
    const handle = this.runtime.createTensor(inner.baseId);
    const tensor = new Tensor(this, inner, handle, { requiresGrad });
    if (this.inCompileRegion) {
      this._compileCreatedTensors.add(tensor);
    }
    if (this._checkpointRecomputeTensors) {
      this._checkpointRecomputeTensors.add(tensor);
    }
    // [capture 2a] track tensors created inside a captured call's window so a
    // short-circuited partial body can be reclaimed by disposing exactly them.
    this._captureInterceptor?.onWrap?.(tensor);
    return tensor;
  }

  _wrapWithGrad(
    inner: RuntimeTensor,
    inputs: Tensor[],
    backward: GradFn,
    tensorsToSave: Tensor[] = [],
  ): Tensor {
    const requiresGrad =
      this._noGradDepth === 0 && inputs.some((tensor) => tensor.requiresGrad);
    const output = this._wrap(inner, requiresGrad);

    if (requiresGrad) {
      // Keep autograd input tensors alive through tidy — their RuntimeTensors
      // hold GPU buffers needed for backward. cleanupAutogradGraph disposes
      // them after backward completes.
      for (const tensor of inputs) {
        this.keep(tensor);
      }

      const savedSlots: SavedTensorSlot[] = [];
      const hooks = this._getSavedTensorHooks();

      for (const tensor of tensorsToSave) {
        if (hooks) {
          const packed = hooks.packHook(tensor);
          const record = this.runtime._debug_saveForBackward(tensor.baseId);
          savedSlots.push({
            packed,
            unpackHook: hooks.unpackHook,
            record,
          });
        } else {
          // Graph-owned retention: the autograd node takes an INDEPENDENT
          // storage-level reference on the saved value (PyTorch semantics), so
          // it survives tidy/step scope exit AND disposal of the user's handle.
          // Released in cleanupAutogradGraph (and the backward error-path
          // finally) for a symmetric rc release. Replaces the old
          // escapes-flag `keep(tensor)`, which only survived tidy, not scope
          // reclamation or manual dispose (the "disposing intermediates breaks
          // autograd" footgun).
          const retained = tensor._unwrap()._cloneForRetention();
          const record = this.runtime._debug_saveForBackward(tensor.baseId);
          savedSlots.push({
            packed: tensor,
            unpackHook: (t) => t as Tensor,
            record,
            retained,
          });
        }
      }

      if (savedSlots.length > 0) {
        this.runtime._debug_publishSave(output.baseId);
      }
      output._setGradNode({
        inputs,
        output,
        backward,
        savedSlots,
        label: this._currentNodeLabel ?? undefined,
      });
    }
    return output;
  }

  _seedGrad(output: Tensor): RuntimeTensor {
    const size = sizeOf(output.shape);
    if (size !== 1) {
      throw new Error("backward requires an explicit grad for non-scalars");
    }
    return this.runtime.full([], 1.0, output.device);
  }

  /** Create a context object for extracted backward functions. */
  _backwardCtx(): BackwardContext {
    return {
      rt: this.runtime,
      sumToShape: (grad, shape) => this._sumToShape(grad, shape),
    };
  }

  _sumToShape(grad: RuntimeTensor, shape: number[]): RuntimeTensor {
    if (shapesEqual(grad.shape, shape)) {
      return grad;
    }
    const gradRank = grad.shape.length;
    const targetRank = shape.length;
    const pad = Math.max(0, gradRank - targetRank);
    const paddedTarget = new Array(pad).fill(1).concat(shape);
    // Sum dims where target is 1 but grad is not (broadcast reduction),
    // AND all leading padded dims (rank reduction — target has fewer dims).
    // Summing a size-1 dim with keepdim=false is a no-op that drops the dim.
    const dims: number[] = [];
    for (let axis = 0; axis < gradRank; axis += 1) {
      if (axis < pad) {
        // Padded leading dim — always include (drops it from output)
        dims.push(axis);
      } else if (paddedTarget[axis] === 1 && grad.shape[axis] !== 1) {
        dims.push(axis);
      }
    }
    let reduced = grad;
    if (dims.length > 0) {
      const summed = this.runtime.sum(reduced, { dim: dims, keepdim: false });
      if (typeof summed === "number") {
        reduced = this.runtime.full([], summed, grad.device);
      } else {
        reduced = summed;
      }
    }
    // Reshape only for rare cases (e.g., target has explicit 1-dims)
    if (!shapesEqual(reduced.shape, shape)) {
      reduced = this.runtime.reshape(reduced, shape);
    }
    return reduced;
  }

  _expandGrad(
    grad: RuntimeTensor,
    inputShape: number[],
    dims: number[],
    keepdim: boolean,
  ): RuntimeTensor {
    let expanded = grad;
    if (!keepdim && dims.length > 0) {
      const rank = inputShape.length;
      const reduceSet = new Set(dims);
      const nextShape = new Array<number>(rank);
      let gradAxis = 0;
      for (let axis = 0; axis < rank; axis += 1) {
        if (reduceSet.has(axis)) {
          nextShape[axis] = 1;
        } else {
          nextShape[axis] = grad.shape[gradAxis];
          gradAxis += 1;
        }
      }
      expanded = this.runtime.reshape(expanded, nextShape);
    }
    return this.runtime.expand(expanded, inputShape);
  }

  _normalizeDims(dim: number | number[] | null, rank: number): number[] {
    if (dim == null) {
      return Array.from({ length: rank }, (_, index) => index);
    }
    const dims = Array.isArray(dim) ? dim.slice() : [dim];
    const normalized = dims.map((value) => (value < 0 ? rank + value : value));
    const unique = new Set<number>();
    for (const value of normalized) {
      if (value < 0 || value >= rank) {
        throw new Error(`dim out of range: ${value}`);
      }
      if (unique.has(value)) {
        continue;
      }
      unique.add(value);
    }
    return Array.from(unique).sort((a, b) => a - b);
  }

  _assertUsable(...tensors: Tensor[]): void {
    this.assertSameEngine(...tensors);
    this.assertSameDevice(...tensors);
    for (const tensor of tensors) {
      tensor._ensureNotDisposed();
    }
  }

  _runEntryPoint<T>(fn: () => Promise<T>): Promise<T> {
    return this.runtime.runEntryPoint(fn);
  }

  // ============================================================================
  // Lifecycle management
  // ============================================================================

  tidy<T>(fn: () => T): T {
    const tidyMode = new TidyDispatchMode();
    this.runtime.pushDispatchMode(tidyMode);
    let result: T | undefined;
    try {
      this.runtime.tidy(() => {
        result = fn();
        return collectEngineTensors(result);
      });
    } finally {
      this.runtime.popDispatchMode();
      tidyMode.disposeNonEscaped();
    }
    return result as T;
  }

  async asyncTidy<T>(fn: () => Promise<T>): Promise<T> {
    const tidyMode = new TidyDispatchMode();
    this.runtime.pushDispatchMode(tidyMode);
    try {
      return await this.runtime.runWithAsyncScope(async () => {
        const result = await fn();
        for (const et of collectEngineTensors(result)) {
          this.runtime.keep(et);
        }
        return result;
      });
    } finally {
      this.runtime.popDispatchMode();
      tidyMode.disposeNonEscaped();
    }
  }

  async runWithAsyncScope<T>(fn: () => Promise<T>): Promise<T> {
    return this.runtime.runWithAsyncScope(fn);
  }

  // ── Async scope surface (docs/scoped-memory-design.md §2-4) ───────────────
  // A thin layer over the step path's snapshot/releaseStepTemps reclamation.
  // No async_hooks / AsyncLocalStorage / AsyncContext — the whole mechanism is
  // a synchronous module-level scope stack + a "synchronous region" pointer,
  // so it behaves identically in the browser.

  /** Open scope records (LIFO). Innermost is last. */
  private _scopeStack: ScopeRecord[] = [];
  /** The scope whose fn() body is CURRENTLY executing synchronously (up to its
   *  first await), or null at the event-loop top level. A scope opened while
   *  this is non-null is a legitimate SYNCHRONOUS nesting; a second ambient
   *  root scope opened while this is null (i.e. a prior ambient scope is
   *  suspended at an await) is a concurrent OVERLAP → throw (§4). */
  private _syncRegionScope: ScopeRecord | null = null;
  private _openAmbientCount = 0;
  private _nextScopeSurfaceId = 1;

  /**
   * Run `fn` inside a reclamation scope. Tensors created during the call that
   * are NOT returned (recursively through arrays/objects) are reclaimed when
   * `fn` returns (sync) or its promise resolves (async); returned tensors — and
   * anything their storage graph reaches, e.g. a returned view's base — are
   * re-parented to the enclosing scope and survive. Works for sync and async
   * `fn`; returns the fn's value (or a promise of it for async fn).
   *
   * Single-flight (§4): a second ambient `scope()` entered while another is
   * open across an await (concurrent, not synchronously nested) THROWS. Use
   * `openScope()` handles for library-internal scopes that must compose freely.
   */
  scope<T>(fn: () => T): T {
    const handle = this._openScopeInternal("ambient");
    const prevSync = this._syncRegionScope;
    this._syncRegionScope = handle;
    let result: T;
    try {
      result = fn();
    } catch (err) {
      this._syncRegionScope = prevSync;
      handle.abort();
      throw err;
    }
    // fn() has returned (sync return, or an async fn that hit its first await
    // and handed back a promise): its synchronous region is over.
    this._syncRegionScope = prevSync;
    const maybe = result as unknown as { then?: unknown } | null;
    if (maybe != null && typeof maybe.then === "function") {
      return (result as unknown as Promise<unknown>).then(
        async (value) => {
          // Materialize pending work (like markStep does before its release)
          // so lazy escapees — e.g. a returned view — exist as storages and
          // their view-base rc protections are established before reclamation.
          // Destruction stays fence-gated (deferredDestroy), so this forces
          // but does not fence: scopes stay cheap (§10).
          await this.runtime.forceAllPending();
          handle.close(value);
          return value;
        },
        (err) => {
          handle.abort();
          throw err;
        },
      ) as unknown as T;
    }
    handle.close(result);
    return result;
  }

  /**
   * Low-level scope handle: `const s = api.openScope(); … s.close(returnValue)`.
   * The primitive `scope()` and the step machinery compile to. Handles are
   * EXEMPT from single-flight overlap detection (they carry no ambient magic),
   * so libraries can open their own scopes inside an app scope freely; they
   * must still close in LIFO order (a mismatched close throws).
   */
  openScope(): ScopeHandle {
    return this._openScopeInternal("handle");
  }

  private _openScopeInternal(kind: "ambient" | "handle"): ScopeRecord {
    // Overlap detection (ambient scopes only): a new ambient scope while
    // another ambient scope is open AND we are not synchronously nested inside
    // an open scope's body means two async tasks interleaved.
    if (
      kind === "ambient" &&
      this._openAmbientCount > 0 &&
      this._syncRegionScope === null
    ) {
      throw new Error(
        "overlapping async scope — serialize your compute; the engine is " +
          "single-flight. Two api.scope(...) tasks are interleaved across an " +
          "await; await one before starting the next, nest them synchronously, " +
          "or use api.openScope() handles for library-internal scopes.",
      );
    }
    // Capture the enclosing scope's snapshot, then install a fresh one that
    // captures everything alive NOW as this scope's persistent baseline.
    const parentSnapshot = storageTracker.peekSnapshot();
    storageTracker.snapshotForStep();
    const record = new ScopeRecord(
      this,
      this._nextScopeSurfaceId++,
      kind,
      parentSnapshot,
    );
    this._scopeStack.push(record);
    if (kind === "ambient") this._openAmbientCount++;
    return record;
  }

  /** Internal: close bookkeeping shared by ScopeRecord.close/abort. */
  _closeScopeRecord(record: ScopeRecord, returnValue: unknown): void {
    const top = this._scopeStack[this._scopeStack.length - 1];
    if (top !== record) {
      throw new Error(
        "overlapping async scope — serialize your compute; the engine is " +
          "single-flight. A scope closed out of order (not the innermost open " +
          "scope); this happens when two scopes interleave across an await.",
      );
    }
    this._scopeStack.pop();
    if (record.kind === "ambient") this._openAmbientCount--;
    // Structural escape (§3.1): re-parent returned tensors to THIS scope's
    // snapshot so releaseStepTemps keeps them; the restore below then hands
    // them to the parent scope's interval (they're not in the parent's older
    // snapshot, so the parent reclaims them unless they escape it too).
    if (returnValue !== undefined) {
      for (const t of collectFrontendTensors(returnValue)) {
        if (!t.disposed) storageTracker.adoptIntoSnapshot(t._unwrap());
      }
    }
    // Reclaim in-scope non-escapees via the step path. Reachability escape
    // (§3.2 — a returned view's in-scope base) is preserved by rc: the live
    // view retains its base, so destroyUnreachable's protected-bases walk and
    // the rc keep it. Destruction is fence-gated (deferredDestroy) so no fence
    // is needed here — scopes stay cheap (§10).
    storageTracker.releaseStepTemps();
    storageTracker.destroyUnreachable();
    // Restore the enclosing scope's snapshot.
    storageTracker.installSnapshot(record.parentSnapshot);
  }

  keep(tensor: Tensor): void {
    this._assertUsable(tensor);
    // Tidy-escape (EngineTensor.escapes) for the synchronous tidy() path.
    this.runtime.keep(tensor._engineTensor());
    // Scope-escape to ROOT (§3.3): survive every enclosing scope. Adopt into
    // the active (innermost scope) snapshot AND every ancestor scope's saved
    // snapshot, so each LIFO restore keeps it. GUARDED on an open scope: with
    // only a step active (no scope), keep() must NOT persist the tensor into
    // the step snapshot — the historical keep() is tidy-escape only, and
    // callers such as forwardLoss's keep(loss) rely on the loss still being
    // reclaimed at markStep (persisting it would leak one storage/step).
    if (this._scopeStack.length > 0) {
      const rt = tensor._unwrap();
      storageTracker.adoptIntoSnapshot(rt);
      for (const record of this._scopeStack) {
        storageTracker.adoptIntoSnapshotToken(record.parentSnapshot, rt);
      }
    }
  }

  dispose(tensor: Tensor): void {
    this.assertSameEngine(tensor);
    // NOTE(scoped-memory stage 1): disposing a tensor releases only ITS OWN
    // handle. It no longer tears down the upstream autograd graph — the graph
    // now INDEPENDENTLY owns its saved values (per-slot `retained` rc), so a
    // downstream loss can still be backwarded after an intermediate handle is
    // disposed (the old "disposing intermediates breaks autograd" footgun).
    // The graph and its retained rcs are released by cleanupAutogradGraph at
    // backward, or by GC/FinalizationRegistry if a graph is abandoned without
    // backward (dead lazy nodes are dropped unforced — scoped-memory §10).
    this.runtime.dispose(tensor._engineTensor());
  }

  /**
   * Queue an IMPLIED step boundary (the minimal-training-loop API).
   * Called by optimizer step() implementations — built-ins do this
   * automatically; custom optimizers should call it at the end of their
   * step(). The boundary is DEFERRED: it commits at the next backward()
   * (or explicit markStep()/beginStep()), so reading loss.item() after
   * optimizer.step() still works, and multiple optimizers stepping
   * back-to-back share one boundary (each bump supersedes the last —
   * everything up to the final step() is one step's work).
   *
   * With this, a training loop needs NO beginStep/endStep/markStep:
   *   forward → backward → optimizer.step()  — cleanup is automatic.
   */
  queueStepBoundary(): void {
    // Under an active api.scope(), the SCOPE is the reclamation boundary
    // (docs/scoped-memory-design.md §9) — its close runs releaseStepTemps.
    // A queued implied boundary would additionally fire a markStep-style
    // commit at the NEXT backward (inside the next scope), duplicating the
    // boundary work the scope already owns. No-op so the scope is the single
    // boundary; optimizer.step() inside a scope needs no implied commit.
    if (this._scopeStack.length > 0) return;
    this._pendingStepBoundary = storageTracker.bumpStepGen();
  }

  /** Pending implied boundary: the generation that closes when it commits. */
  private _pendingStepBoundary: number | null = null;

  /** Commit a queued implied step boundary: force the closing step's
   *  residue (optimizer update chains), demote its temporaries (gen-scoped
   *  — work the next iteration already built lazily is untouched), fence,
   *  and snapshot persistents for the new step. Mirrors the explicit
   *  endStep(); markStep(); beginStep() sequence. */
  private async _commitPendingStepBoundary(): Promise<void> {
    if (this._pendingStepBoundary === null) return;
    const gen = this._pendingStepBoundary;
    this._pendingStepBoundary = null;
    this.runtime.endStep();
    try {
      await awaitDeferredFence();
    } catch {
      // CPU-only usage: no WebGPU fence to await.
    }
    await this.runtime.markStep();
    await this.runtime.forceAllPending();
    // QUIESCE BEFORE DESTROYING — order differs from explicit markStep
    // deliberately. The force above may encode passes for the closing
    // step's optimizer chain; a fence issued AFTER the demotion sweep
    // executes the sweep's deferred destroys while those passes can still
    // be pending — "used in submit while destroyed", the silent-corruption
    // class (whole submit rejected, outputs frozen). Fencing first means
    // everything the sweep destroys is past its last GPU use, and the
    // deferred destruction fires at some far-future fence, long after
    // submission.
    try {
      issueDeferredFence();
      await awaitDeferredFence();
    } catch {
      // CPU-only usage.
    }
    storageTracker.destroyUnreachable();
    storageTracker.releaseStepTemps(gen);
    storageTracker.destroyUnreachable();
    this.runtime.resetCumulativeFusionStats();
    // beginStep equivalent: residue is already forced above; snapshot the
    // persistents (gen-scoped: the next iteration's lazily-built tensors
    // must NOT be classified persistent or they'd never be cleaned up).
    storageTracker.snapshotForStep(gen);
    await this.runtime.beginStep();
    // [step-tape 1b] implied boundaries (training loops) are a different
    // regime than the bare-markStep decode loop — reset the comparator.
    if (STEP_TAPE_RECORD) stNoteBoundary("implied-boundary");
  }

  async markStep(): Promise<void> {
    // An explicit markStep IS the step boundary — supersede any queued
    // implied one (full, un-scoped cleanup: everything created since the
    // snapshot is this step's, exactly the pre-existing semantics).
    this._pendingStepBoundary = null;
    // Step 0: If there's a deferred fence from the previous markStep, await it now.
    try {
      await awaitDeferredFence();
    } catch {
      // Safe to ignore if WebGPU backend is not initialized (CPU-only usage).
    }

    // Step 1: Run the engine's markStep (token reset, finalization, etc.)
    await this.runtime.markStep();

    // Step 2: Force all pending tensors to materialize
    await this.runtime.forceAllPending();

    // Step 3: Destroy all storages with rc <= 0.
    storageTracker.destroyUnreachable();

    // Step 3.5: Release refs for step-scoped temporaries, then destroy.
    storageTracker.releaseStepTemps();

    // Step 3.6: Final cleanup after releasing step-scoped refs.
    storageTracker.destroyUnreachable();

    // Step 4: Reset cumulative fusion stats for the next step
    this.runtime.resetCumulativeFusionStats();

    // Step 4: Issue GPU fence and await it. Previously the fence was deferred
    // to beginStep, which made the "forward" phase timing include the previous
    // step's GPU execution time. Awaiting here keeps the GPU cost in "cleanup"
    // where it belongs, giving honest phase breakdowns.
    try {
      issueDeferredFence();
      await awaitDeferredFence();
    } catch {
      // Safe to ignore if WebGPU backend is not initialized (CPU-only usage).
    }

    // Step 5 (opt-in): implicit step boundary for ceremony-free loops.
    // Snapshot the survivors — the NEXT markStep's releaseStepTemps (Step 3.5)
    // reclaims everything created after this point. An explicit beginStep()
    // before then simply re-snapshots (supersedes). See setStepScopedCleanup.
    if (this._stepScopedCleanup) {
      storageTracker.snapshotForStep();
    }

    // [step-tape 1b] markStep IS the step delimiter: finalize the observed
    // step record (guard-3 eligibility diff against the previous step) with
    // the guard-1 op-sequence counter and guard-5 epoch/regime bookkeeping.
    if (STEP_TAPE_RECORD) {
      stEndStep({
        opSeq: getNextNodeId(),
        epoch: currentEpoch(),
        stepScopedCleanup: this._stepScopedCleanup,
      });
      // [step-tape 1c] promote a captured skeleton if the just-ended step
      // became eligible (two consecutive structurally-identical compiled
      // steps under one appKey).
      if (STEP_TAPE_REPLAY) stPromoteEligibleSkeleton();
    }
  }

  // ==========================================================================
  // Step-tape (phase 1c program-level replay) — docs/staged-execution-phase1.md
  // ==========================================================================

  /** Declare the app-level bucket key + scalar-slot values (α) for the
   *  UPCOMING decode step. The appKey is the structural declaration (guards
   *  1/2); scalarValues carry guard-3's value-level coverage. No-op unless
   *  TORCHLETTE_STEP_TAPE=1. */
  setTapeContext(appKey: string, scalarValues: number[] = []): void {
    if (!STEP_TAPE_REPLAY) return;
    stSetTapeContext(
      appKey,
      scalarValues,
      currentEpoch(),
      this._stepScopedCleanup,
    );
  }

  /** Whether a replayable skeleton exists for `appKey`. */
  tapeReadyFor(appKey: string): boolean {
    return STEP_TAPE_REPLAY && stTapeReadyFor(appKey);
  }

  /** Replay the current step's tape (all guards enforced inside). Returns the
   *  logits tensor on a hit, or null on ANY guard miss (the caller then runs
   *  the normal path, which re-records). `uploads` are the fresh per-token
   *  payloads keyed by shape (unique per bucket). */
  async tapeReplay(
    uploads: Array<{ shape: number[]; values: Float32Array }>,
  ): Promise<Tensor | null> {
    if (!STEP_TAPE_REPLAY) return null;
    const backend = this.runtime.getBackend("webgpu");
    const r = await stTryReplay(uploads, backend);
    if (!r.hit || !r.resultHandle) return null;
    const rt = this.runtime.createFromStorageHandle(
      r.resultHandle,
      r.shape ?? [],
      r.device ?? "webgpu",
      r.dtype ?? "f32",
    );
    return this._wrap(rt);
  }

  /** Guard-miss observability (§6): merged recorder + replay counters. */
  getStepTapeStats(): {
    recorder: ReturnType<typeof stStats>;
    replay: ReturnType<typeof stReplayStats>;
  } {
    return { recorder: stStats(), replay: stReplayStats() };
  }

  // ==========================================================================
  // capture() — phase 2a user-declared staging (docs/staged-execution-phase2a)
  // ==========================================================================

  /**
   * Wrap an inference loop body (noGrad, no optimizer) in a CapturedFn that
   * traces on early calls and replays via the step-tape once a skeleton is
   * ready and every guard passes.
   *
   * THE ARG-BOUNDARY CONTRACT — everything that varies must cross the argument
   * list (see src/frontend/capture.ts header + docs/staged-execution-phase2a.md
   * §2): TENSOR args are dynamic slots (warm — fresh values every call);
   * PLAIN-VALUE args are hashed into the bucket key (cold — a changed value is
   * a counted miss + re-record); VALUES CAPTURED BY CLOSURE ARE FROZEN AT
   * RECORD TIME (jax.jit semantics) — pass anything that varies as an argument
   * (a tensor for scrubbed knobs, a plain value for occasional config).
   *
   * When TORCHLETTE_STEP_TAPE is off the CapturedFn is a transparent
   * pass-through (just calls fn).
   */
  capture<A extends unknown[]>(
    fn: (...args: A) => Tensor | Tensor[],
    opts?: CaptureOptions,
  ): {
    (...args: A): Promise<Tensor | Tensor[]>;
    invalidate(): void;
    stats(): ReturnType<CapturedFn["stats_"]>;
  } {
    const cf = new CapturedFn(this, fn as (...a: never[]) => Tensor | Tensor[], opts);
    const callable = ((...args: A) => cf.call(args)) as {
      (...args: A): Promise<Tensor | Tensor[]>;
      invalidate(): void;
      stats(): ReturnType<CapturedFn["stats_"]>;
    };
    callable.invalidate = () => cf.invalidate();
    callable.stats = () => cf.stats_();
    return callable;
  }

  // --- CapturedFn support seams (used only by src/frontend/capture.ts) ---

  _tapeActive(): boolean {
    return STEP_TAPE_REPLAY;
  }

  _setCaptureTapeContext(appKey: string, scalars: number[]): void {
    if (STEP_TAPE_REPLAY) {
      stSetTapeContext(appKey, scalars, currentEpoch(), this._stepScopedCleanup);
    }
  }

  _tapeReadyFor(appKey: string): boolean {
    // Validity-checked readiness: capture commits to a hit only if the skeleton
    // AND its compiled plan are valid, so a post-commit replay decline (which
    // would strand the captured fn's already-advanced in-place state) cannot
    // happen in steady state.
    return STEP_TAPE_REPLAY && stTapeReplayValid(appKey);
  }

  /** Replay the phase-1 skeleton for the active appKey with fresh uploads (from
   *  the capture interceptor). Returns the logits tensor on a hit, null on a
   *  guard miss. */
  async _captureReplay(
    uploads: Array<{ shape: number[]; values: Float32Array }>,
  ): Promise<Tensor | null> {
    return this.tapeReplay(uploads);
  }

  /** Install the capture interceptor for the duration of one captured call.
   *  Returns a restore fn. Not reentrant (captured calls never nest). */
  _installCaptureInterceptor(hooks: {
    onUpload: (shape: number[], values: Float32Array) => void;
  }): () => void {
    const prevHooks = this._captureInterceptor;
    this._captureInterceptor = hooks;
    return () => {
      this._captureInterceptor = prevHooks;
    };
  }

  _invalidateCapture(appKey: string): void {
    if (STEP_TAPE_REPLAY) stDropSkeleton(appKey);
  }

  /** Fresh values of a captured call's TENSOR args that are caller-built
   *  PENDING tensorFromArray uploads (their values read synchronously from the
   *  un-forced node payload — no GPU roundtrip). Keyed by shape, matching the
   *  phase-1 replay's shape-keyed re-dressing. MATERIALIZED/persistent tensor
   *  args are skipped: they are external plan inputs whose stable buffer the
   *  replay reads live (the warm in-place knob). Derived args (pending
   *  compute) are skipped too — if their leaves vary, the phase-1 shape guard
   *  misses the replay (safe fallback), never silently reuses stale data. */
  _captureArgUploads(args: unknown[]): {
    uploads: Array<{ shape: number[]; values: Float32Array }>;
    /** The pending upload arg tensors themselves — DONATED on a replay hit
     *  (disposed by capture): the replay consumed their VALUES, and the
     *  never-consumed pending node would otherwise be force-executed as a
     *  wasted mini-plan at every markStep. */
    donatable: Tensor[];
  } {
    const uploads: Array<{ shape: number[]; values: Float32Array }> = [];
    const donatable: Tensor[] = [];
    for (const a of args) {
      const t = a as Tensor;
      if (!t || typeof t !== "object" || !Array.isArray(t.shape)) continue;
      const ref = t._unwrap().lazyRef as {
        kind: string;
        node?: { op?: string; payload?: unknown };
      };
      if (ref.kind !== "pending" || ref.node?.op !== "tensorFromArray") continue;
      const p = ref.node.payload as { values?: ArrayLike<number> } | undefined;
      if (!p?.values) continue;
      uploads.push({
        shape: t.shape,
        values:
          p.values instanceof Float32Array
            ? p.values
            : Float32Array.from(p.values as ArrayLike<number>),
      });
      donatable.push(t);
    }
    return { uploads, donatable };
  }

  /** Mark a captured output as past its staging-ring validity window (§4). A
   *  subsequent read (via _assertUsable) throws a LOUD error naming the step. */
  _markCaptureExpired(t: Tensor, step: number, now: number, k: number): void {
    t._captureExpired = { step, now, k };
  }

  _debug_baseCommit(baseId: number, mutId: number): void {
    this.runtime._debug_baseCommit(baseId, mutId);
  }

  async beginStep(): Promise<void> {
    // Mixed explicit/implied usage: a queued boundary must close the old
    // step before a new one begins.
    if (this._pendingStepBoundary !== null) {
      await this.markStep();
    }
    // Force any unmaterialized tensors (e.g., model weights from init) so their
    // storages exist before the snapshot. Without this, lazy model-init tensors
    // would materialize during the step and be treated as step-scoped.
    await this.runtime.forceAllPending();
    storageTracker.snapshotForStep();
    await this.runtime.beginStep();
    // [step-tape 1b] explicit ceremony is a different boundary regime than
    // the bare-markStep loop — reset the consecutive-step comparator
    // (guard 5); tapes record the regime they were captured under.
    if (STEP_TAPE_RECORD) stNoteBoundary("beginStep");
  }

  endStep(): void {
    this.runtime.endStep();
    if (STEP_TAPE_RECORD) stNoteBoundary("endStep");
  }

  /**
   * Flush GPU work and reclaim buffers mid-step without destroying step-scoped
   * tensors. Use between gradient accumulation micro-batches to prevent
   * GPU memory from growing unbounded.
   *
   * Unlike markStep(), this does NOT run step-scoped cleanup, so gradient
   * tensors on parameters survive across flushes.
   *
   * The flush sequence:
   * 1. Force pending tensors → destroy unreachable storages
   * 2. Flush shared encoder → submit command buffer to GPU
   * 3. Issue + await GPU fence → wait for GPU to finish
   * 4. Flush pending buffers to pool → destroy pending-destroy queue
   * 5. Evict pool buffers → free GPU memory held by the cache
   */
  async flushStep(): Promise<void> {
    // Force all pending lazy tensors + destroy unreachable
    await this.runtime.forceAllPending();
    storageTracker.destroyUnreachable();
    try {
      // Submit GPU work
      flushSharedEncoder();
      // Wait for GPU completion so buffers are safe to reuse/destroy
      issueDeferredFence();
      await awaitDeferredFence();
      // Promote pending → pool (makes them reusable), destroy pending-destroy
      flushBufferPool();
      destroyPendingGPUBuffers();
    } catch {
      // Safe to ignore if WebGPU backend not initialized
    }
  }

  /**
   * Evict all buffer arenas and compiled plans, freeing GPU memory.
   * Call between training rounds to prevent unbounded growth in long sessions.
   * One-time cost: next step rebuilds the arena (~100ms), then back to full speed.
   */
  async evictArenas(): Promise<void> {
    const { evictAllArenas } = await import("../executor/executor");
    evictAllArenas();
  }

  _runtime(): RuntimeEngine {
    return this.runtime;
  }

  _wrapRuntime(inner: RuntimeTensor, requiresGrad: boolean): Tensor {
    return this._wrap(inner, requiresGrad);
  }

  // ============================================================================
  // Private helpers (not exposed to extracted modules)
  // ============================================================================

  private assertSameEngine(...tensors: Tensor[]): void {
    for (const tensor of tensors) {
      if (tensor._engine() !== this) {
        throw new Error("Tensor belongs to a different Torchlette instance");
      }
    }
  }

  private assertSameDevice(...tensors: Tensor[]): void {
    if (tensors.length <= 1) {
      return;
    }
    const device = tensors[0].device;
    for (const tensor of tensors) {
      if (tensor.device !== device) {
        throw new Error("Tensors must be on the same device");
      }
    }
  }
}

// ============================================================================
// Table-driven method generation — typed declarations + prototype installation
// ============================================================================

// Unary ops: gradient specs live in OP_REGISTRY, dispatch via _dispatchUnary.
export interface Torchlette {
  sqrt(a: Tensor): Tensor;
  relu(a: Tensor): Tensor;
  exp(a: Tensor): Tensor;
  log(a: Tensor): Tensor;
  neg(a: Tensor): Tensor;
  abs(a: Tensor): Tensor;
  tanh(a: Tensor): Tensor;
  sigmoid(a: Tensor): Tensor;
  silu(a: Tensor): Tensor;
  sin(a: Tensor): Tensor;
  cos(a: Tensor): Tensor;
  rsqrt(a: Tensor): Tensor;
  floor(a: Tensor): Tensor;
  ceil(a: Tensor): Tensor;
  round(a: Tensor): Tensor;
  sign(a: Tensor): Tensor;
  isfinite(a: Tensor): Tensor;
}

for (const opName of UNARY_AUTOGRAD_OPS) {
  (Torchlette.prototype as any)[opName] = function (
    this: Torchlette,
    a: Tensor,
  ) {
    return this._dispatchUnary(opName, a);
  };
}

// Binary ops: gradient specs live in OP_REGISTRY, dispatch via _dispatchBinaryFromTable.
export interface Torchlette {
  add(a: Tensor | number, b: Tensor | number): Tensor;
  mul(a: Tensor | number, b: Tensor | number): Tensor;
  div(a: Tensor | number, b: Tensor | number): Tensor;
  pow(a: Tensor | number, b: Tensor | number): Tensor;
  minimum(a: Tensor | number, b: Tensor | number): Tensor;
  maximum(a: Tensor | number, b: Tensor | number): Tensor;
}

for (const opName of BINARY_AUTOGRAD_OPS) {
  (Torchlette.prototype as any)[opName] = function (
    this: Torchlette,
    a: Tensor | number,
    b: Tensor | number,
  ) {
    return this._dispatchBinaryFromTable(opName, a, b);
  };
}

// Override pow: lower pow(x, k) for a non-negative INTEGER exponent k to a
// multiplication chain (exponentiation by squaring) instead of the generic
// pow node. WGSL's pow(x, y) = exp2(y * log2(x)) returns NaN for x < 0, so
// pow(signedTensor, 2) silently poisons results (this bit clipGradNorm_'s L2
// norm). x*x is exact for any sign, needs no transcendental, and its autograd
// (gradient accumulated through both mul inputs) reproduces d/dx x^k = k*x^(k-1)
// exactly. Non-integer, negative, or tensor exponents keep the real pow (whose
// log-based gradient is genuinely needed and matches PyTorch's domain limits).
const _powTranscendental = Torchlette.prototype.pow;
(Torchlette.prototype as any).pow = function (
  this: Torchlette,
  a: Tensor | number,
  b: Tensor | number,
): Tensor {
  if (
    typeof a !== "number" &&
    typeof b === "number" &&
    Number.isInteger(b) &&
    b >= 0
  ) {
    // x^0 = 1 (a constant; pow would give NaN for x<0 via 0*log2(x)=NaN).
    if (b === 0) return this.ones(a.shape, { device: a.device });
    let result: Tensor | null = null;
    let base: Tensor = a;
    let e = b;
    while (e > 0) {
      if (e % 2 === 1) result = result === null ? base : this.mul(result, base);
      e = Math.floor(e / 2);
      if (e > 0) base = this.mul(base, base);
    }
    return result as Tensor;
  }
  return _powTranscendental.call(this, a, b);
};

// Comparison ops: non-differentiable, dispatch via _cmpOp.
const COMPARISON_OPS = ["gt", "lt", "ge", "le", "eq", "ne"] as const;

export interface Torchlette {
  gt(a: Tensor, b: Tensor): Tensor;
  lt(a: Tensor, b: Tensor): Tensor;
  ge(a: Tensor, b: Tensor): Tensor;
  le(a: Tensor, b: Tensor): Tensor;
  eq(a: Tensor, b: Tensor): Tensor;
  ne(a: Tensor, b: Tensor): Tensor;
}

for (const op of COMPARISON_OPS) {
  (Torchlette.prototype as any)[op] = function (
    this: Torchlette,
    a: Tensor,
    b: Tensor,
  ) {
    return this._cmpOp(op, a, b);
  };
}

export const torch = new Torchlette();

/**
 * Public handle for a scope opened via api.openScope(). Close it exactly once,
 * in LIFO order relative to nested scopes.
 */
export interface ScopeHandle {
  readonly id: number;
  /** Close the scope, escaping any tensors reachable from `returnValue`. */
  close(returnValue?: unknown): void;
}

/** Concrete scope record — carries the parent snapshot to restore on close and
 *  the close/abort bookkeeping (delegated to the owning Torchlette). */
class ScopeRecord implements ScopeHandle {
  private _closed = false;
  constructor(
    private readonly engine: Torchlette,
    readonly id: number,
    readonly kind: "ambient" | "handle",
    readonly parentSnapshot: object | null,
  ) {}
  close(returnValue?: unknown): void {
    if (this._closed) return;
    this._closed = true;
    this.engine._closeScopeRecord(this, returnValue);
  }
  /** Close with no escapees (error path). */
  abort(): void {
    if (this._closed) return;
    this._closed = true;
    this.engine._closeScopeRecord(this, undefined);
  }
}

/** Walk a scope/tidy return value, collecting frontend Tensors (recursively
 *  through arrays/plain objects) — the structural escape set (§3.1). */
function collectFrontendTensors(value: unknown): Tensor[] {
  const out: Tensor[] = [];
  const seen = new Set<unknown>();
  const visit = (current: unknown) => {
    if (current == null) return;
    if (current instanceof Tensor) {
      out.push(current);
      return;
    }
    if (typeof current !== "object") return;
    if (seen.has(current)) return;
    seen.add(current);
    if (Array.isArray(current)) {
      for (const entry of current) visit(entry);
      return;
    }
    for (const entry of Object.values(current as Record<string, unknown>)) {
      visit(entry);
    }
  };
  visit(value);
  return out;
}

function collectEngineTensors(value: unknown): EngineTensor[] {
  const out: EngineTensor[] = [];
  const seen = new Set<unknown>();

  const visit = (current: unknown) => {
    if (current == null) {
      return;
    }
    if (current instanceof Tensor) {
      out.push(current._engineTensor());
      return;
    }
    if (typeof current !== "object") {
      return;
    }
    if (seen.has(current)) {
      return;
    }
    seen.add(current);
    if (Array.isArray(current)) {
      for (const entry of current) {
        visit(entry);
      }
      return;
    }
    for (const entry of Object.values(current as Record<string, unknown>)) {
      visit(entry);
    }
  };

  visit(value);
  return out;
}

// sizeOf and shapesEqual imported from core/shape
