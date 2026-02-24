import type {
  ArgReduceOptions,
  DeviceKind,
  DType,
  GatherOptions,
  GeluOptions,
  MaxOptions,
  MeanOptions,
  ScatterAddOptions,
  SubOptions,
  SumOptions,
  TransposeOptions,
} from "./backend/types";
import {
  Engine,
  type EngineTensor,
  type SavedTensorRecord,
} from "./engine/engine";
import {
  type AutocastConfig,
  type AutocastContext,
  createAutocastContext,
} from "./engine/amp";
import { promoteDtype } from "./engine/dtype-rules";
import { RuntimeEngine, TidyDispatchMode, type TensorOrScalar } from "./runtime/engine";
import { setMemoryLimit } from "./engine/memory-planned-executor";
import {
  setGPUMemoryLimit,
  getGPUMemoryLimit,
  getGPUMemoryStats,
} from "./backend/webgpu/memory-tracker";
import {
  flushBufferPool,
  flushBufferPoolWithSync,
  isAutotuneEnabled,
  setAutotuneEnabled,
  issueDeferredFence,
  awaitDeferredFence,
} from "./backend/webgpu";
import { sizeOf, shapesEqual } from "./core/shape";
import { storageTracker } from "./engine/lazy";
import type { Tensor as RuntimeTensor } from "./runtime/tensor";

// Re-export the Tensor class and DisposedTensorError from their new home
export { Tensor, DisposedTensorError } from "./frontend-tensor";
import { Tensor } from "./frontend-tensor";

// Re-export types from frontend-types
export type {
  TensorCreateOptions,
  AutocastOptions,
  TorchletteOptions,
  PackHook,
  UnpackHook,
} from "./frontend-types";
import type {
  TensorCreateOptions,
  AutocastOptions,
  TorchletteOptions,
  GradFn,
  SavedTensorHooksContext,
  SavedTensorSlot,
} from "./frontend-types";

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

// Import extracted modules
import {
  tensorFromArrayImpl,
  randImpl,
  randnImpl,
  bernoulliImpl,
  zerosImpl,
  onesImpl,
  fullImpl,
  arangeImpl,
  trilImpl,
  triuImpl,
} from "./frontend-creation";
import {
  autocastImpl,
  autocastAsyncImpl,
  savedTensorHooksImpl,
  autocastCastImpl,
  applyAutocastImpl,
} from "./frontend-autocast";
import {
  softmaxImpl,
  crossEntropyFusedImpl,
  scaledDotProductAttentionImpl,
  layernormImpl,
} from "./frontend-fused-ops";
import { backwardImpl } from "./frontend-autograd";

export class Torchlette {
  readonly engine: Engine;
  readonly runtime: RuntimeEngine;
  private readonly autocastContext: AutocastContext;
  private inCompileRegion = false;
  /** Tensors created inside compile regions — eligible for disposal after backward */
  readonly _compileCreatedTensors = new WeakSet<Tensor>();
  /** Whether memory planning is available for compile regions (§0.1 goal #2) */
  private readonly memoryPlanningAvailable: boolean;
  /** Stack of saved tensor hooks for checkpointing (§10) */
  readonly _savedTensorHooksStack: SavedTensorHooksContext[] = [];
  /** Label to capture on subsequent autograd nodes (for backward attribution) */
  private _currentNodeLabel: string | null = null;
  /** Hooks fired before each backward op */
  readonly _backwardDispatchHooks: Array<(info: { output: Tensor; inputs: Tensor[]; label?: string }) => void> = [];

  constructor(backendName?: DeviceKind, options?: TorchletteOptions) {
    this.engine = new Engine();
    // Configure memory limit if provided (applies to both BufferPool and GPU memory tracker)
    if (options?.memoryLimitBytes !== undefined) {
      setMemoryLimit(options.memoryLimitBytes);
      setGPUMemoryLimit(options.memoryLimitBytes);
    }
    // Per spec §0.1 goal #2: memory planning runs ONLY inside compiled regions
    // Store whether it's available, but don't enable it globally
    this.memoryPlanningAvailable = options?.enableMemoryPlanning ?? false;
    this.runtime = new RuntimeEngine(backendName, {
      enableFusion: options?.enableFusion ?? false,
      // Memory planning starts disabled; enabled only during compile()
      enableMemoryPlanning: false,
      // Early release can be enabled globally for memory savings
      enableEarlyRelease: options?.enableEarlyRelease ?? false,
      // Checkpoint segmentation for large model memory savings
      enableCheckpointSegmentation: options?.enableCheckpointSegmentation ?? false,
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
    const compiledFn = this.engine.compile((..._args: unknown[]) => {
      // We don't use engine's TraceTensors directly,
      // instead we enable fusion and run the actual function
      return undefined;
    });

    return (...args: Args): R => {
      // Signal we're in a compile region
      this.inCompileRegion = true;
      const wasFusionEnabled = this.runtime.isFusionEnabled();
      const wasMemoryPlanningEnabled = this.runtime.isMemoryPlanningEnabled();
      const wasAutotuneEnabled = isAutotuneEnabled();

      // Per spec §0.1 goal #2: fusion and memory planning run ONLY inside compiled regions
      this.runtime.setFusionEnabled(true);
      if (this.memoryPlanningAvailable) {
        this.runtime.setMemoryPlanning(true);
      }

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
        this.runtime.setMemoryPlanning(wasMemoryPlanningEnabled);
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

  /** Register a hook that fires before each backward op. Returns unregister function. */
  onBackwardDispatch(hook: (info: { output: Tensor; inputs: Tensor[]; label?: string }) => void): () => void {
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
  // Creation ops — delegated to frontend-creation.ts
  // ============================================================================

  tensorFromArray(
    values: number[] | Float32Array,
    shape: number[],
    options?: TensorCreateOptions,
  ): Tensor {
    return tensorFromArrayImpl(this, values, shape, options);
  }

  rand(shape: number[], options?: TensorCreateOptions): Tensor {
    return randImpl(this, shape, options);
  }

  randn(shape: number[], options?: TensorCreateOptions): Tensor {
    return randnImpl(this, shape, options);
  }

  bernoulli(shape: number[], p = 0.5, options?: TensorCreateOptions): Tensor {
    return bernoulliImpl(this, shape, p, options);
  }

  zeros(shape: number[], options?: TensorCreateOptions): Tensor {
    return zerosImpl(this, shape, options);
  }

  ones(shape: number[], options?: TensorCreateOptions): Tensor {
    return onesImpl(this, shape, options);
  }

  full(shape: number[], fillValue: number, options?: TensorCreateOptions): Tensor {
    return fullImpl(this, shape, fillValue, options);
  }

  arange(end: number, options?: { start?: number; step?: number; device?: DeviceKind; requiresGrad?: boolean }): Tensor {
    return arangeImpl(this, end, options);
  }

  tril(a: Tensor, k = 0): Tensor {
    return trilImpl(this, a, k);
  }

  triu(a: Tensor, k = 0): Tensor {
    return triuImpl(this, a, k);
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

  add(a: Tensor | number, b: Tensor | number): Tensor {
    // Unwrap: numbers go directly to RuntimeEngine as scalars
    const aUnwrap: TensorOrScalar = typeof a === "number" ? a : a._unwrap();
    const bUnwrap: TensorOrScalar = typeof b === "number" ? b : b._unwrap();
    const tensors = [a, b].filter((x): x is Tensor => typeof x !== "number");
    this._assertUsable(...tensors);
    if (typeof a !== "number" && typeof b !== "number") {
      [a, b] = this._applyAutocast("add", [a, b]) as [Tensor, Tensor];
      const inner = this.runtime.add(a._unwrap(), b._unwrap());
      const aShape = a.shape;
      const bShape = b.shape;
      return this._wrapWithGrad(inner, [a, b], (grad, _getSaved) => [
        this._sumToShape(grad, aShape),
        this._sumToShape(grad, bShape),
      ]);
    }
    // At least one operand is a number — no grad needed for the number
    const inner = this.runtime.add(aUnwrap, bUnwrap);
    const tensorInput = typeof a !== "number" ? a : (b as Tensor);
    return this._wrapWithGrad(inner, [tensorInput], (grad, _getSaved) => [
      this._sumToShape(grad, tensorInput.shape),
    ]);
  }

  sub(a: Tensor, b: Tensor, options?: SubOptions): Tensor {
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

  mul(a: Tensor | number, b: Tensor | number): Tensor {
    const tensors = [a, b].filter((x): x is Tensor => typeof x !== "number");
    this._assertUsable(...tensors);
    if (typeof a !== "number" && typeof b !== "number") {
      [a, b] = this._applyAutocast("mul", [a, b]) as [Tensor, Tensor];
      const inner = this.runtime.mul(a._unwrap(), b._unwrap());
      const aShape = a.shape;
      const bShape = b.shape;
      const tensorsToSave =
        a.requiresGrad || b.requiresGrad ? [a, b] : [];
      return this._wrapWithGrad(
        inner,
        [a, b],
        (grad, getSaved) => {
          const savedA = getSaved(0);
          const savedB = getSaved(1);
          const gradA = this._sumToShape(
            this.runtime.mul(grad, savedB._unwrap()),
            aShape,
          );
          const gradB = this._sumToShape(
            this.runtime.mul(grad, savedA._unwrap()),
            bShape,
          );
          return [gradA, gradB];
        },
        tensorsToSave,
      );
    }
    // One operand is a number — simpler backward
    const tensorInput = typeof a !== "number" ? a : (b as Tensor);
    const scalarVal = typeof a === "number" ? a : (b as number);
    const inner = this.runtime.mul(
      typeof a === "number" ? a : a._unwrap(),
      typeof b === "number" ? b : b._unwrap(),
    );
    return this._wrapWithGrad(inner, [tensorInput], (grad, _getSaved) => [
      this._sumToShape(this.runtime.mul(grad, scalarVal), tensorInput.shape),
    ]);
  }

  div(a: Tensor | number, b: Tensor | number): Tensor {
    const tensors = [a, b].filter((x): x is Tensor => typeof x !== "number");
    this._assertUsable(...tensors);
    if (typeof a !== "number" && typeof b !== "number") {
      [a, b] = this._applyAutocast("div", [a, b]) as [Tensor, Tensor];
      const inner = this.runtime.div(a._unwrap(), b._unwrap());
      const aShape = a.shape;
      const bShape = b.shape;
      const tensorsToSave =
        a.requiresGrad || b.requiresGrad ? [a, b] : [];
      return this._wrapWithGrad(
        inner,
        [a, b],
        (grad, getSaved) => {
          const savedA = getSaved(0);
          const savedB = getSaved(1);
          const gradA = this._sumToShape(
            this.runtime.div(grad, savedB._unwrap()),
            aShape,
          );
          const bSquared = this.runtime.mul(savedB._unwrap(), savedB._unwrap());
          const negA = this.runtime.neg(savedA._unwrap());
          const gradB = this._sumToShape(
            this.runtime.mul(grad, this.runtime.div(negA, bSquared)),
            bShape,
          );
          return [gradA, gradB];
        },
        tensorsToSave,
      );
    }
    // One operand is a number — simpler backward
    const tensorInput = typeof a !== "number" ? a : (b as Tensor);
    const scalarVal = typeof a === "number" ? a : (b as number);
    const inner = this.runtime.div(
      typeof a === "number" ? a : a._unwrap(),
      typeof b === "number" ? b : b._unwrap(),
    );
    if (typeof b === "number") {
      // a / scalar → grad_a = grad / scalar
      return this._wrapWithGrad(inner, [tensorInput], (grad, _getSaved) => [
        this._sumToShape(this.runtime.div(grad, scalarVal), tensorInput.shape),
      ]);
    }
    // scalar / b → grad_b = -scalar / b^2 * grad
    return this._wrapWithGrad(inner, [tensorInput], (grad, getSaved) => {
      const savedB = getSaved(0);
      const bSq = this.runtime.mul(savedB._unwrap(), savedB._unwrap());
      return [this._sumToShape(this.runtime.mul(grad, this.runtime.div(-scalarVal, bSq)), tensorInput.shape)];
    }, [tensorInput]);
  }

  matmul(a: Tensor, b: Tensor): Tensor {
    this._assertUsable(a, b);
    // Apply autocast: cast f32 inputs to f16 for compute-bound matmul
    const [castA, castB] = this._applyAutocast("matmul", [a, b]) as [Tensor, Tensor];
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
    const [castInput, castWeight] = this._applyAutocast("matmul", [input, weight]) as [Tensor, Tensor];

    // Forward: Y = input @ weight.T (+ bias)
    const wT = this.runtime.transpose(castWeight._unwrap(), {
      dim0: castWeight.shape.length - 2, dim1: castWeight.shape.length - 1,
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
      if (needsWeightGrad) toSave.push(castInput);  // saved[1 or 0] = input (for dW)
      if (bias) toSave.push(bias);
    }

    return this._wrapWithGrad(inner, allInputs, (grad, getSaved) => {
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
          dim0: grad.shape.length - 2, dim1: grad.shape.length - 1,
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

      return bias ? [resultInput, resultWeight, resultBias!] : [resultInput, resultWeight];
    }, toSave);
  }

  sqrt(a: Tensor): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.sqrt(a._unwrap());
    const tensorsToSave = a.requiresGrad ? [a] : [];
    return this._wrapWithGrad(
      inner,
      [a],
      (grad, getSaved) => {
        // grad_input = grad_output * 0.5 / sqrt(input)
        // Recompute sqrt from saved input for checkpointing support
        const savedA = getSaved(0);
        const sqrtA = this.runtime.sqrt(savedA._unwrap());
        const denom = this.runtime.add(sqrtA, 1e-8);
        const reciprocal = this.runtime.div(0.5, denom);
        const gradInput = this.runtime.mul(grad, reciprocal);
        return [gradInput];
      },
      tensorsToSave,
    );
  }

  relu(a: Tensor): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.relu(a._unwrap());
    const tensorsToSave = a.requiresGrad ? [a] : [];
    return this._wrapWithGrad(
      inner,
      [a],
      (grad, getSaved) => {
        // grad_input = grad_output * (input > 0)
        // Use comparison op to create mask, then multiply
        const savedA = getSaved(0);
        const mask = this.runtime.gt(savedA._unwrap(), 0);
        const gradInput = this.runtime.mul(grad, mask);
        return [gradInput];
      },
      tensorsToSave,
    );
  }

  exp(a: Tensor): Tensor {
    this._assertUsable(a);
    // Autocast: exp is F32-required for numerical stability
    const [castA] = this._applyAutocast("exp", [a]);
    const inner = this.runtime.exp(castA._unwrap());
    const tensorsToSave = a.requiresGrad ? [castA] : [];
    // Gradient of exp(x) is exp(x) = output
    // grad_input = grad_output * exp(input) = grad_output * output
    return this._wrapWithGrad(inner, [a], (grad, getSaved) => {
      // Recompute exp from saved input for checkpointing support
      const savedA = getSaved(0);
      const expA = this.runtime.exp(savedA._unwrap());
      const gradInput = this.runtime.mul(grad, expA);
      return [gradInput];
    }, tensorsToSave);
  }

  log(a: Tensor): Tensor {
    this._assertUsable(a);
    // Autocast: log is F32-required for numerical stability
    const [castA] = this._applyAutocast("log", [a]);
    const inner = this.runtime.log(castA._unwrap());
    const tensorsToSave = a.requiresGrad ? [castA] : [];
    // Gradient of log(x) is 1/x
    // grad_input = grad_output / (input + eps)
    return this._wrapWithGrad(
      inner,
      [a],
      (grad, getSaved) => {
        const savedA = getSaved(0);
        const denominator = this.runtime.add(savedA._unwrap(), 1e-8);
        const gradInput = this.runtime.div(grad, denominator);
        return [gradInput];
      },
      tensorsToSave,
    );
  }

  neg(a: Tensor): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.neg(a._unwrap());
    // Gradient of -x is -1, no tensors need to be saved
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => {
      return [this.runtime.neg(grad)];
    });
  }

  abs(a: Tensor): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.abs(a._unwrap());
    const aShape = a.shape;
    const aDevice = a.device;
    const tensorsToSave = a.requiresGrad ? [a] : [];
    // Gradient of abs(x) is sign(x)
    return this._wrapWithGrad(
      inner,
      [a],
      (grad, getSaved) => {
        const savedA = getSaved(0);
        const inputValues = savedA._unwrap().toArray();
        const gradValues = grad.toArray();
        const outValues = gradValues.map((g, i) =>
          inputValues[i] >= 0 ? g : -g,
        );
        return [this.runtime.tensorFromArray(outValues, aShape, aDevice)];
      },
      tensorsToSave,
    );
  }

  tanh(a: Tensor): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.tanh(a._unwrap());
    const tensorsToSave = a.requiresGrad ? [a] : [];
    // Gradient of tanh(x) is (1 - tanh(x)^2) * grad
    return this._wrapWithGrad(inner, [a], (grad, getSaved) => {
      // Recompute tanh from saved input for checkpointing support
      const savedA = getSaved(0);
      const tanhA = this.runtime.tanh(savedA._unwrap());
      // Use tensor ops instead of toArray for lazy execution
      const tanhSquared = this.runtime.mul(tanhA, tanhA);
      const oneMinusTanhSquared = this.runtime.sub(1, tanhSquared);
      return [this.runtime.mul(oneMinusTanhSquared, grad)];
    }, tensorsToSave);
  }

  sigmoid(a: Tensor): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.sigmoid(a._unwrap());
    const tensorsToSave = a.requiresGrad ? [a] : [];
    // Gradient of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x)) * grad
    return this._wrapWithGrad(inner, [a], (grad, getSaved) => {
      // Recompute sigmoid from saved input for checkpointing support
      const savedA = getSaved(0);
      const sigmoidA = this.runtime.sigmoid(savedA._unwrap());
      // Use tensor ops instead of toArray for lazy execution
      const oneMinusSigmoid = this.runtime.sub(1, sigmoidA);
      const sigmoidGrad = this.runtime.mul(sigmoidA, oneMinusSigmoid);
      return [this.runtime.mul(sigmoidGrad, grad)];
    }, tensorsToSave);
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
            this.runtime.where(
              this.runtime.gt(innerVal, 10),
              10,
              innerVal,
            ),
          );

          const tanhInner = this.runtime.tanh(clampedInner);
          const cdf = this.runtime.mul(0.5, this.runtime.add(1, tanhInner));
          const tanh2 = this.runtime.mul(tanhInner, tanhInner);
          const sech2 = this.runtime.sub(1, tanh2);

          const pdfTerm = this.runtime.add(1, this.runtime.mul(0.134145, x2));
          const pdf = this.runtime.mul(this.runtime.mul(0.7978845608, pdfTerm), sech2);

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

          const z = this.runtime.mul(x, 0.7071067811865476);
          const absZ = this.runtime.abs(z);

          const t = this.runtime.div(1, this.runtime.add(1, this.runtime.mul(0.3275911, absZ)));
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
                this.runtime.add(this.runtime.mul(-1.453152027, t4), this.runtime.mul(1.061405429, t5)),
              ),
            ),
          );

          const z2 = this.runtime.mul(z, z);
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

  silu(a: Tensor): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.silu(a._unwrap());
    const aShape = a.shape;
    const aDevice = a.device;
    const tensorsToSave = a.requiresGrad ? [a] : [];
    return this._wrapWithGrad(
      inner,
      [a],
      (grad, getSaved) => {
        const savedA = getSaved(0);
        const inputValues = savedA._unwrap().toArray();
        const gradValues = grad.toArray();
        const outValues = gradValues.map((g, i) => {
          const x = inputValues[i];
          const sig = 1 / (1 + Math.exp(-x));
          return g * (sig + x * sig * (1 - sig));
        });
        return [this.runtime.tensorFromArray(outValues, aShape, aDevice)];
      },
      tensorsToSave,
    );
  }

  /**
   * Check if values are finite (not NaN and not Inf).
   * Returns 1.0 where finite, 0.0 where NaN or Inf.
   * This op is not differentiable.
   */
  isfinite(a: Tensor): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.isfinite(a._unwrap());
    return this._wrap(inner);
  }

  expand(a: Tensor, shape: number[]): Tensor {
    this._assertUsable(a);
    const aShape = a.shape;
    const inner = this.runtime.expand(a._unwrap(), shape);
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this._sumToShape(grad, aShape),
    ]);
  }

  sum(a: Tensor, options?: SumOptions): Tensor {
    this._assertUsable(a);
    // Autocast: sum is F32-required for numerical stability
    const [castA] = this._applyAutocast("sum", [a]);
    const inner = this.runtime.sum(castA._unwrap(), options);
    const aShape = a.shape;
    const aRank = aShape.length;
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => {
      const dims = this._normalizeDims(options?.dim ?? null, aRank);
      const keepdim = options?.keepdim ?? false;
      const expanded = this._expandGrad(grad, aShape, dims, keepdim);
      return [expanded];
    });
  }

  max(a: Tensor, options?: MaxOptions): number | Tensor {
    this._assertUsable(a);
    const result = this.runtime.max(a._unwrap(), options);
    if (typeof result === "number") {
      return result;
    }
    return this._wrap(result);
  }

  mean(a: Tensor, options?: MeanOptions): number | Tensor {
    this._assertUsable(a);
    // Autocast: mean is F32-required for numerical stability
    const [castA] = this._applyAutocast("mean", [a]);
    const result = this.runtime.mean(castA._unwrap(), options);
    if (typeof result === "number") {
      if (a.requiresGrad) {
        throw new Error("mean with requiresGrad must specify dim");
      }
      return result;
    }
    const aShape = a.shape;
    const dims = this._normalizeDims(options?.dim ?? null, aShape.length);
    const keepdim = options?.keepdim ?? false;
    const reduceCount =
      dims.length === 0 ? 1 : dims.reduce((acc, dim) => acc * aShape[dim], 1);
    return this._wrapWithGrad(result, [a], (grad, _getSaved) => {
      const expanded = this._expandGrad(grad, aShape, dims, keepdim);
      const scaled = this.runtime.mul(expanded, 1 / reduceCount);
      return [scaled];
    });
  }

  argmax(a: Tensor, options: ArgReduceOptions): Tensor {
    this._assertUsable(a);
    const result = this.runtime.argmax(a._unwrap(), options);
    return this._wrap(result);
  }

  argmin(a: Tensor, options: ArgReduceOptions): Tensor {
    this._assertUsable(a);
    const result = this.runtime.argmin(a._unwrap(), options);
    return this._wrap(result);
  }

  // ============================================================================
  // Comparison ops
  // ============================================================================

  gt(a: Tensor, b: Tensor): Tensor {
    this._assertUsable(a, b);
    return this._wrap(this.runtime.gt(a._unwrap(), b._unwrap()));
  }

  lt(a: Tensor, b: Tensor): Tensor {
    this._assertUsable(a, b);
    return this._wrap(this.runtime.lt(a._unwrap(), b._unwrap()));
  }

  ge(a: Tensor, b: Tensor): Tensor {
    this._assertUsable(a, b);
    return this._wrap(this.runtime.ge(a._unwrap(), b._unwrap()));
  }

  le(a: Tensor, b: Tensor): Tensor {
    this._assertUsable(a, b);
    return this._wrap(this.runtime.le(a._unwrap(), b._unwrap()));
  }

  eq(a: Tensor, b: Tensor): Tensor {
    this._assertUsable(a, b);
    return this._wrap(this.runtime.eq(a._unwrap(), b._unwrap()));
  }

  ne(a: Tensor, b: Tensor): Tensor {
    this._assertUsable(a, b);
    return this._wrap(this.runtime.ne(a._unwrap(), b._unwrap()));
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
    q: Tensor, k: Tensor, v: Tensor, scale?: number, isCausal = false,
  ): Tensor {
    return scaledDotProductAttentionImpl(this, q, k, v, scale, isCausal);
  }

  layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps = 1e-5): Tensor {
    return layernormImpl(this, x, weight, bias, eps);
  }

  // ============================================================================
  // Data access
  // ============================================================================

  async cpu(a: Tensor): Promise<number[]> {
    this._assertUsable(a);
    return this._runEntryPoint(async () => {
      this.engine.forceRead(a.baseId);
      return this.runtime.cpu(a._unwrap());
    });
  }

  async item(a: Tensor): Promise<number> {
    this._assertUsable(a);
    return this._runEntryPoint(async () => {
      this.engine.forceRead(a.baseId);
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
      this.engine.forceRead(a.baseId);
      const inner = await this.runtime.transferNow(a._unwrap(), device);
      return this._wrap(inner, a.requiresGrad);
    });
  }

  // ============================================================================
  // In-place operations (§4.3-4.4)
  // ============================================================================

  copy_(dst: Tensor, src: Tensor): Tensor {
    this._assertUsable(dst, src);
    this.runtime.copy_(dst._unwrap(), src._unwrap());
    this._debug_baseCommit(dst.baseId, this.engine.nextMutId());
    return dst;
  }

  add_(dst: Tensor, src: Tensor): Tensor {
    this._assertUsable(dst, src);
    this.runtime.add_(dst._unwrap(), src._unwrap());
    this._debug_baseCommit(dst.baseId, this.engine.nextMutId());
    return dst;
  }

  zero_(dst: Tensor): Tensor {
    this._assertUsable(dst);
    this.runtime.zero_(dst._unwrap());
    this._debug_baseCommit(dst.baseId, this.engine.nextMutId());
    return dst;
  }

  fill_(dst: Tensor, value: number): Tensor {
    this._assertUsable(dst);
    this.runtime.fill_(dst._unwrap(), value);
    this._debug_baseCommit(dst.baseId, this.engine.nextMutId());
    return dst;
  }

  mul_(dst: Tensor, value: number): Tensor {
    this._assertUsable(dst);
    this.runtime.mul_(dst._unwrap(), value);
    this._debug_baseCommit(dst.baseId, this.engine.nextMutId());
    return dst;
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
      return [
        this.runtime.scatterAdd(z, indexInner, grad, options),
      ];
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
      return [this._sumToShape(grad_x, xShape), this._sumToShape(grad_y, yShape)];
    });
  }

  // ============================================================================
  // View/reshape/transpose ops
  // ============================================================================

  view(a: Tensor, shape: number[]): Tensor {
    this._assertUsable(a);
    const aShape = a.shape;
    const inner = this.runtime.view(a._unwrap(), shape);
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this.runtime.reshape(grad, aShape),
    ]);
  }

  reshape(a: Tensor, shape: number[]): Tensor {
    this._assertUsable(a);
    const aShape = a.shape;
    const inner = this.runtime.reshape(a._unwrap(), shape);
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this.runtime.reshape(grad, aShape),
    ]);
  }

  transpose(a: Tensor, options: TransposeOptions): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.transpose(a._unwrap(), options);
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this.runtime.transpose(grad, options),
    ]);
  }

  permute(a: Tensor, dims: number[]): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.permute(a._unwrap(), dims);
    const inverseDims = new Array<number>(dims.length);
    for (let i = 0; i < dims.length; i++) {
      inverseDims[dims[i]] = i;
    }
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this.runtime.permute(grad, inverseDims),
    ]);
  }

  contiguous(a: Tensor): Tensor {
    this._assertUsable(a);
    const inner = this.runtime.contiguous(a._unwrap());
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => [grad]);
  }

  narrow(a: Tensor, dim: number, start: number, length: number): Tensor {
    this._assertUsable(a);
    const originalLength = a.shape[dim];
    const inner = this.runtime.narrow(a._unwrap(), dim, start, length);
    return this._wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this.runtime.narrowBackward(grad, dim, start, originalLength),
    ]);
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
    this.runtime.markEscaped(inner);
    const handle = this.engine.createTensor(inner.baseId);
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
    const requiresGrad = inputs.some((tensor) => tensor.requiresGrad);
    const output = this._wrap(inner, requiresGrad);

    if (requiresGrad) {
      const savedSlots: SavedTensorSlot[] = [];
      const hooks = this._getSavedTensorHooks();

      for (const tensor of tensorsToSave) {
        if (hooks) {
          const packed = hooks.packHook(tensor);
          const record = this.engine._debug_saveForBackward(tensor.baseId);
          savedSlots.push({
            packed,
            unpackHook: hooks.unpackHook,
            record,
          });
        } else {
          this.keep(tensor);
          const record = this.engine._debug_saveForBackward(tensor.baseId);
          savedSlots.push({
            packed: tensor,
            unpackHook: (t) => t as Tensor,
            record,
          });
        }
      }

      if (savedSlots.length > 0) {
        this.engine._debug_publishSave(output.baseId);
      }
      output._setGradNode({ inputs, output, backward, savedSlots, label: this._currentNodeLabel ?? undefined });
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
    return this.engine.runEntryPoint(fn);
  }

  // ============================================================================
  // Lifecycle management
  // ============================================================================

  tidy<T>(fn: () => T): T {
    const tidyMode = new TidyDispatchMode();
    this.runtime.pushDispatchMode(tidyMode);
    let result: T | undefined;
    try {
      this.engine.tidy(() => {
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
      return await this.engine.runWithAsyncScope(async () => {
        const result = await fn();
        for (const et of collectEngineTensors(result)) {
          this.engine.keep(et);
        }
        return result;
      });
    } finally {
      this.runtime.popDispatchMode();
      tidyMode.disposeNonEscaped();
    }
  }

  async runWithAsyncScope<T>(fn: () => Promise<T>): Promise<T> {
    return this.engine.runWithAsyncScope(fn);
  }

  keep(tensor: Tensor): void {
    this._assertUsable(tensor);
    this.engine.keep(tensor._engineTensor());
  }

  dispose(tensor: Tensor): void {
    this.assertSameEngine(tensor);
    const gradNode = tensor._gradNode();
    if (gradNode) {
      gradNode.savedSlots.length = 0;
      gradNode.inputs.length = 0;
      tensor._setGradNode(null);
    }
    this.engine.dispose(tensor._engineTensor());
  }

  async markStep(): Promise<void> {
    // Step 0: If there's a deferred fence from the previous markStep, await it now.
    try {
      await awaitDeferredFence();
    } catch {
      // Safe to ignore if WebGPU backend is not initialized (CPU-only usage).
    }

    // Step 1: Run the engine's markStep (token reset, finalization, etc.)
    await this.engine.markStep();

    // Step 2: Force all pending tensors to materialize
    await this.runtime.forceAllPending();

    // Step 3: GC - destroy all unreachable storages (intermediate buffers)
    storageTracker.destroyUnreachable();

    // Step 3.5: Reset cumulative fusion stats for the next step
    this.runtime.resetCumulativeFusionStats();

    // Step 4: Issue a deferred GPU fence.
    try {
      issueDeferredFence();
    } catch {
      // Safe to ignore if WebGPU backend is not initialized (CPU-only usage).
    }
  }

  _debug_baseCommit(baseId: number, mutId: number): void {
    this.engine._debug_baseCommit(baseId, mutId);
  }

  async beginStep(): Promise<void> {
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

export const torch = new Torchlette();

function collectEngineTensors(value: unknown): EngineTensor[] {
  const out: EngineTensor[] = [];
  const seen = new Set<unknown>();

  const visit = (current: unknown) => {
    if (current === null || current === undefined) {
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
