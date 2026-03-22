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
import { isAutotuneEnabled, setAutotuneEnabled } from "../backend/webgpu";
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

  constructor(backendName?: DeviceKind, options?: TorchletteOptions) {
    // Configure memory limit if provided (applies to GPU memory tracker)
    if (options?.memoryLimitBytes !== undefined) {
      setGPUMemoryLimit(options.memoryLimitBytes);
    }
    this.runtime = new RuntimeEngine(backendName, {
      enableFusion: options?.enableFusion ?? false,
      // Early release can be enabled globally for memory savings
      enableEarlyRelease: options?.enableEarlyRelease ?? false,
      // Checkpoint segmentation for large model memory savings
      enableCheckpointSegmentation:
        options?.enableCheckpointSegmentation ?? false,
      // True segmentation with GPU sync for actual memory savings
      enableTrueSegmentation: options?.enableTrueSegmentation ?? false,
    });
    this.autocastContext = createAutocastContext();
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
    values: number[] | Float32Array,
    shape: number[],
    options?: TensorCreateOptions,
  ): Tensor {
    return this._wrap(
      this.runtime.tensorFromArray(values, shape, options?.device),
      options?.requiresGrad ?? false,
    );
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
      this.runtime.zeros(shape, options?.device),
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
      this.runtime.full(shape, fillValue, options?.device),
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
    // Apply autocast: cast f32 inputs to f16 for compute-bound matmul
    const [castA, castB] = this._applyAutocast("matmul", [a, b]) as [
      Tensor,
      Tensor,
    ];
    const inner = this.runtime.matmul(castA._unwrap(), castB._unwrap());
    // Capture shapes of the ORIGINAL inputs for backward gradient shapes
    const aShape = a.shape;
    const bShape = b.shape;
    // Save cast tensors for backward so backward matmuls also run in the cast dtype
    const tensorsToSave =
      a.requiresGrad || b.requiresGrad ? [castA, castB] : [];
    return this._wrapWithGrad(
      inner,
      [a, b],
      (grad, getSaved) => {
        if (aShape.length < 2 || bShape.length < 2) {
          throw new Error("matmul backward requires rank >= 2");
        }
        const savedA = getSaved(0);
        const savedB = getSaved(1);
        const savedAShape = savedA.shape;
        const savedBShape = savedB.shape;
        // Mixed-dtype matmul: the backend handles f16/f32 input combinations natively
        // by promoting to f32 accumulation. No need to cast saved f16 tensors to f32.
        const savedAInner = savedA._unwrap();
        const savedBInner = savedB._unwrap();
        // Transpose creates strided views; the backend detects simple transposes
        // and handles them natively via flipped transpose flags (no contiguous needed).
        const aT = this.runtime.transpose(savedAInner, {
          dim0: savedAShape.length - 2,
          dim1: savedAShape.length - 1,
        });
        const bT = this.runtime.transpose(savedBInner, {
          dim0: savedBShape.length - 2,
          dim1: savedBShape.length - 1,
        });
        const gradA = this.runtime.matmul(grad, bT);
        const gradB = this.runtime.matmul(aT, grad);
        // Sum to the ORIGINAL input shapes (before autocast)
        const resultA = this._sumToShape(gradA, aShape);
        const resultB = this._sumToShape(gradB, bShape);
        return [resultA, resultB];
      },
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
      (grad, getSaved) => {
        if (inputShape.length < 2 || weightShape.length < 2)
          throw new Error("linear backward requires rank >= 2");

        let savedIdx = 0;
        // dX = dY @ W  (weight is [out, in], so dY @ W = [..., out] @ [out, in] = [..., in])
        let resultInput: RuntimeTensor | null = null;
        if (needsInputGrad) {
          const savedWeight = getSaved(savedIdx++)._unwrap();
          const gradInput = this.runtime.matmul(grad, savedWeight);
          resultInput = this._sumToShape(gradInput, inputShape);
        }

        // dW = dY^T @ X → [out, in] = weight's shape directly (no transpose needed!)
        // Skipped when weight doesn't require grad (e.g. detached weight).
        let resultWeight: RuntimeTensor | null = null;
        if (needsWeightGrad) {
          const savedInput = getSaved(savedIdx++)._unwrap();
          const gradT = this.runtime.transpose(grad, {
            dim0: grad.shape.length - 2,
            dim1: grad.shape.length - 1,
          });
          const gradWeight = this.runtime.matmul(gradT, savedInput);
          resultWeight = this._sumToShape(gradWeight, weightShape);
        }

        // dBias = sum(dY, all dims except last)
        let resultBias: RuntimeTensor | null = null;
        if (bias) {
          const biasShape = getSaved(savedIdx).shape;
          resultBias = this._sumToShape(grad, biasShape);
        }

        return bias
          ? [resultInput, resultWeight, resultBias as RuntimeTensor]
          : [resultInput, resultWeight];
      },
      toSave,
    );
  }

  gelu(a: Tensor, options?: GeluOptions): Tensor {
    this._assertUsable(a);
    const approximate = options?.approximate ?? "tanh";
    const inner = this.runtime.gelu(a._unwrap(), options);
    const tensorsToSave = a.requiresGrad ? [a] : [];

    if (approximate === "tanh") {
      return this._wrapWithGrad(
        inner,
        [a],
        (grad, getSaved) => {
          const savedA = getSaved(0);
          const x = savedA._unwrap();
          const x2 = this.runtime.mul(x, x);
          const x3 = this.runtime.mul(x2, x);

          const term = this.runtime.add(x, this.runtime.mul(0.044715, x3));
          const innerVal = this.runtime.mul(0.7978845608, term);

          const clampedInner = this.runtime.where(
            this.runtime.lt(innerVal, -10),
            -10,
            this.runtime.where(this.runtime.gt(innerVal, 10), 10, innerVal),
          );

          const tanhInner = this.runtime.tanh(clampedInner);
          const cdf = this.runtime.mul(0.5, this.runtime.add(1, tanhInner));
          const tanh2 = this.runtime.mul(tanhInner, tanhInner);
          const sech2 = this.runtime.sub(1, tanh2);

          const pdfTerm = this.runtime.add(1, this.runtime.mul(0.134145, x2));
          const pdf = this.runtime.mul(
            this.runtime.mul(0.7978845608, pdfTerm),
            sech2,
          );

          const xPdfHalf = this.runtime.mul(this.runtime.mul(x, pdf), 0.5);
          const geluGrad = this.runtime.add(cdf, xPdfHalf);
          const gradInput = this.runtime.mul(grad, geluGrad);

          return [gradInput];
        },
        tensorsToSave,
      );
    } else {
      return this._wrapWithGrad(
        inner,
        [a],
        (grad, getSaved) => {
          const savedA = getSaved(0);
          const x = savedA._unwrap();

          const z = this.runtime.mul(x, Math.SQRT1_2);
          const absZ = this.runtime.abs(z);

          const t = this.runtime.div(
            1,
            this.runtime.add(1, this.runtime.mul(0.3275911, absZ)),
          );
          const t2 = this.runtime.mul(t, t);
          const t3 = this.runtime.mul(t2, t);
          const t4 = this.runtime.mul(t3, t);
          const t5 = this.runtime.mul(t4, t);

          const poly = this.runtime.add(
            this.runtime.mul(0.254829592, t),
            this.runtime.add(
              this.runtime.mul(-0.284496736, t2),
              this.runtime.add(
                this.runtime.mul(1.421413741, t3),
                this.runtime.add(
                  this.runtime.mul(-1.453152027, t4),
                  this.runtime.mul(1.061405429, t5),
                ),
              ),
            ),
          );

          const negZ2 = this.runtime.mul(-0.5, this.runtime.mul(x, x));
          const expNegZ2 = this.runtime.exp(negZ2);

          const erfAbs = this.runtime.sub(1, this.runtime.mul(poly, expNegZ2));
          const xGe0 = this.runtime.ge(x, 0);
          const erfPos = this.runtime.add(1, erfAbs);
          const erfNeg = this.runtime.sub(1, erfAbs);
          const erfTerm = this.runtime.where(xGe0, erfPos, erfNeg);
          const cdf = this.runtime.mul(0.5, erfTerm);

          const pdf = this.runtime.mul(expNegZ2, 0.3989422804014327);
          const xPdf = this.runtime.mul(x, pdf);
          const geluGrad = this.runtime.add(cdf, xPdf);
          const gradInput = this.runtime.mul(grad, geluGrad);

          return [gradInput];
        },
        tensorsToSave,
      );
    }
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

  _crossEntropyFused(logits: Tensor, targets: Tensor): Tensor {
    return crossEntropyFusedImpl(this, logits, targets);
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
    return backwardImpl(this, a, grad);
  }

  // ============================================================================
  // Public helper methods (promoted from private for extracted modules)
  // ============================================================================

  _wrap(inner: RuntimeTensor, requiresGrad = false): Tensor {
    if (!this._inCheckpointRecompute) {
      this.runtime.markEscaped(inner);
    }
    const handle = this.runtime.createTensor(inner.baseId);
    const tensor = new Tensor(this, inner, handle, { requiresGrad });
    if (this.inCompileRegion) {
      this._compileCreatedTensors.add(tensor);
    }
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
          this.keep(tensor);
          const record = this.runtime._debug_saveForBackward(tensor.baseId);
          savedSlots.push({
            packed: tensor,
            unpackHook: (t) => t as Tensor,
            record,
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

  _sumToShape(grad: RuntimeTensor, shape: number[]): RuntimeTensor {
    if (shapesEqual(grad.shape, shape)) {
      return grad;
    }
    const gradRank = grad.shape.length;
    const targetRank = shape.length;
    const pad = Math.max(0, gradRank - targetRank);
    const paddedTarget = new Array(pad).fill(1).concat(shape);
    const dims: number[] = [];
    for (let axis = 0; axis < gradRank; axis += 1) {
      const targetDim = paddedTarget[axis];
      const gradDim = grad.shape[axis];
      if (targetDim === 1 && gradDim !== 1) {
        dims.push(axis);
      }
    }
    let reduced = grad;
    if (dims.length > 0) {
      const summed = this.runtime.sum(reduced, { dim: dims, keepdim: true });
      if (typeof summed === "number") {
        reduced = this.runtime.full([], summed, grad.device);
      } else {
        reduced = summed;
      }
    }
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

  keep(tensor: Tensor): void {
    this._assertUsable(tensor);
    this.runtime.keep(tensor._engineTensor());
  }

  dispose(tensor: Tensor): void {
    this.assertSameEngine(tensor);
    this._disposeAutogradChain(tensor);
    this.runtime.dispose(tensor._engineTensor());
  }

  /**
   * Walk the autograd graph rooted at `tensor` and deterministically clean up
   * all pending saved tensors. Without this, saved tensors deep in the chain
   * (e.g., attention logsumexp reshapes) survive as zombie pending RuntimeTensors
   * until GC, causing stale-graph crashes in forceAllPending().
   */
  private _disposeAutogradChain(tensor: Tensor): void {
    const visited = new Set<Tensor>();
    const stack: Tensor[] = [tensor];
    while (stack.length > 0) {
      const t = stack.pop()!;
      if (visited.has(t)) continue;
      visited.add(t);
      const gradNode = t._gradNode();
      if (!gradNode) continue;
      // Dispose pending (unmaterialized) saved tensors.
      // Materialized tensors (e.g., model params) are shared and must not be disposed.
      for (const slot of gradNode.savedSlots) {
        const saved = slot.packed;
        if (
          saved &&
          typeof (saved as Tensor).disposed === "boolean" &&
          !(saved as Tensor).disposed &&
          typeof (saved as Tensor)._unwrap === "function"
        ) {
          const rt = (saved as Tensor)._unwrap();
          if (!rt.isMaterialized()) {
            this.runtime.dispose((saved as Tensor)._engineTensor());
          }
        }
      }
      gradNode.savedSlots.length = 0;
      // Recurse into autograd inputs
      for (const input of gradNode.inputs) {
        stack.push(input);
      }
      gradNode.inputs.length = 0;
      t._setGradNode(null);
    }
  }

  async markStep(): Promise<void> {
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

    // Step 3: Destroy all unreachable storages (intermediate buffers).
    storageTracker.destroyUnreachable();

    // Step 3.5: Deterministic step-scoped cleanup. Demotes reachable storages
    // for tensors created during this step (not in the beginStep snapshot),
    // so they're destroyed immediately rather than lingering until GC runs.
    storageTracker.destroyStepScoped();

    // Step 4: Reset cumulative fusion stats for the next step
    this.runtime.resetCumulativeFusionStats();

    // Step 4: Issue a deferred GPU fence.
    try {
      issueDeferredFence();
    } catch {
      // Safe to ignore if WebGPU backend is not initialized (CPU-only usage).
    }
  }

  _debug_baseCommit(baseId: number, mutId: number): void {
    this.runtime._debug_baseCommit(baseId, mutId);
  }

  async beginStep(): Promise<void> {
    // Force any unmaterialized tensors (e.g., model weights from init) so their
    // storages exist before the snapshot. Without this, lazy model-init tensors
    // would materialize during the step and be treated as step-scoped.
    await this.runtime.forceAllPending();
    // Snapshot which RuntimeTensor objects are alive now. These are persistent
    // (model params, optimizer state) and their storages survive step cleanup.
    storageTracker.snapshotForStep();
    await this.runtime.beginStep();
  }

  endStep(): void {
    this.runtime.endStep();
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
