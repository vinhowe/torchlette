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
  type AMPPolicy,
  type AutocastConfig,
  type AutocastContext,
  createAutocastContext,
  DEFAULT_AMP_POLICY,
  DISABLED_AMP_POLICY,
  F16_ELIGIBLE_OPS,
  F32_REQUIRED_OPS,
  popAutocast,
  pushAutocast,
} from "./engine/amp";
import { OP_DTYPE_RULES, promoteDtype } from "./engine/dtype-rules";
import type { LazyOpCode } from "./engine/lazy";
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
import { storageTracker } from "./engine/lazy";
import type { Tensor as RuntimeTensor } from "./runtime/tensor";

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

export type TensorCreateOptions = {
  requiresGrad?: boolean;
  device?: DeviceKind;
};

/**
 * Accessor function for retrieving saved tensors during backward pass.
 * May trigger recomputation if checkpointing hooks are active.
 */
type GetSavedFn = (idx: number) => Tensor;

/**
 * Backward function that computes gradients with respect to inputs.
 * @param grad - Gradient of the loss with respect to the output
 * @param getSaved - Accessor for saved tensors (may trigger recomputation)
 * @returns Array of gradients for each input (null if input doesn't require grad)
 */
type GradFn = (
  grad: RuntimeTensor,
  getSaved: GetSavedFn,
) => Array<RuntimeTensor | null>;

type AutogradNode = {
  inputs: Tensor[];
  output: Tensor;
  backward: GradFn;
  /** Saved tensor slots with pack/unpack hooks support */
  savedSlots: SavedTensorSlot[];
  /** Opaque annotation captured during forward, available to backward hooks */
  label?: string;
};

// ============================================================================
// Saved Tensor Hooks (for checkpointing, §10)
// ============================================================================

/**
 * Pack hook: transforms a tensor into a packed representation during forward.
 * For checkpointing, this replaces the tensor with a lightweight placeholder.
 */
export type PackHook = (tensor: Tensor) => unknown;

/**
 * Unpack hook: restores a tensor from its packed representation during backward.
 * For checkpointing, this triggers recomputation to reconstruct the tensor.
 */
export type UnpackHook = (packed: unknown) => Tensor;

/**
 * Context for saved tensor hooks.
 */
interface SavedTensorHooksContext {
  packHook: PackHook;
  unpackHook: UnpackHook;
}

/**
 * Slot for a saved tensor with lazy unpacking support.
 * Used internally to defer tensor restoration until backward pass.
 */
interface SavedTensorSlot {
  /** Packed representation (result of packHook, may be placeholder or tensor) */
  packed: unknown;
  /** Hook to restore the tensor */
  unpackHook: UnpackHook;
  /** Metadata for validation */
  record: SavedTensorRecord;
}

export class DisposedTensorError extends Error {
  name = "DisposedTensorError";
}

export class Tensor {
  private readonly engine: Torchlette;
  private readonly inner: RuntimeTensor;
  private readonly engineTensor: EngineTensor;
  private readonly requiresGradValue: boolean;
  private gradNode: AutogradNode | null = null;
  private gradValue: Tensor | null = null;
  private retainGradValue = false;

  constructor(
    engine: Torchlette,
    inner: RuntimeTensor,
    handle: EngineTensor,
    options?: TensorCreateOptions,
  ) {
    this.engine = engine;
    this.inner = inner;
    this.engineTensor = handle;
    this.requiresGradValue = options?.requiresGrad ?? false;
    // Set up disposal callback to free GPU buffers when Engine.dispose() is called
    handle.onDispose = () => {
      inner.dispose();
    };
  }

  get baseId(): number {
    return this.inner.baseId;
  }

  get requiresGrad(): boolean {
    return this.requiresGradValue;
  }

  get grad(): Tensor | null {
    return this.gradValue;
  }

  get shape(): number[] {
    return this.inner.shape.slice();
  }

  get device(): DeviceKind {
    return this.inner.device;
  }

  get dtype(): DType {
    return this.inner.dtype;
  }

  /**
   * Check if this tensor will retain its gradient during backward.
   */
  get isRetainGrad(): boolean {
    return this.retainGradValue;
  }

  /**
   * Request that gradient be retained on this tensor during backward.
   *
   * By default, gradients are only stored on leaf tensors (those without
   * a grad_fn). Calling retainGrad() on a non-leaf tensor will cause its
   * gradient to be stored during backward.
   *
   * @throws Error if tensor does not require gradients
   */
  retainGrad(): void {
    this.ensureNotDisposed();
    if (!this.requiresGradValue) {
      throw new Error(
        "retainGrad() can only be called on tensors that require gradients",
      );
    }
    this.retainGradValue = true;
  }

  zeroGrad(): void {
    this.ensureNotDisposed();
    // Dispose old gradient tensor to free GPU memory
    if (this.gradValue) {
      this.gradValue.dispose();
    }
    this.gradValue = null;
  }

  toArray(): number[] {
    this.ensureNotDisposed();
    return this.inner.toArray();
  }

  async cpu(): Promise<number[]> {
    return this.engine.cpu(this);
  }

  async item(): Promise<number> {
    return this.engine.item(this);
  }

  backward(grad?: Tensor): Promise<void> {
    return this.engine.backward(this, grad);
  }

  keep(): void {
    this.engine.keep(this);
  }

  dispose(): void {
    this.engine.dispose(this);
  }

  [Symbol.toPrimitive](): never {
    this.ensureNotDisposed();
    throw new Error("Tensor cannot be implicitly converted to a primitive");
  }

  valueOf(): never {
    this.ensureNotDisposed();
    throw new Error("Tensor cannot be implicitly converted to a primitive");
  }

  toString(): string {
    this.ensureNotDisposed();
    return `Tensor(shape=[${this.shape.join(", ")}], device=${
      this.device
    }, baseId=${this.baseId})`;
  }

  view(shape: number[]): Tensor {
    return this.engine.view(this, shape);
  }

  reshape(shape: number[]): Tensor {
    return this.engine.reshape(this, shape);
  }

  add(other: Tensor | number): Tensor {
    return this.engine.add(this, other);
  }

  sub(other: Tensor | number, options?: SubOptions): Tensor {
    return this.engine.sub(this, other, options);
  }

  mul(other: Tensor | number): Tensor {
    return this.engine.mul(this, other);
  }

  div(other: Tensor | number): Tensor {
    return this.engine.div(this, other);
  }

  matmul(other: Tensor): Tensor {
    return this.engine.matmul(this, other);
  }

  sqrt(): Tensor {
    return this.engine.sqrt(this);
  }

  relu(): Tensor {
    return this.engine.relu(this);
  }

  exp(): Tensor {
    return this.engine.exp(this);
  }

  log(): Tensor {
    return this.engine.log(this);
  }

  neg(): Tensor {
    return this.engine.neg(this);
  }

  abs(): Tensor {
    return this.engine.abs(this);
  }

  tanh(): Tensor {
    return this.engine.tanh(this);
  }

  sigmoid(): Tensor {
    return this.engine.sigmoid(this);
  }

  gelu(options?: GeluOptions): Tensor {
    return this.engine.gelu(this, options);
  }

  silu(): Tensor {
    return this.engine.silu(this);
  }

  /**
   * Check if values are finite (not NaN and not Inf).
   * Returns 1.0 where finite, 0.0 where NaN or Inf.
   */
  isfinite(): Tensor {
    return this.engine.isfinite(this);
  }

  expand(shape: number[]): Tensor {
    return this.engine.expand(this, shape);
  }

  transpose(options: TransposeOptions): Tensor {
    return this.engine.transpose(this, options);
  }

  /**
   * Permute dimensions according to the given order.
   * Returns a view (no data copy).
   */
  permute(dims: number[]): Tensor {
    return this.engine.permute(this, dims);
  }

  /**
   * Return a contiguous tensor. If already contiguous, returns self.
   * Otherwise materializes to a new contiguous buffer.
   */
  contiguous(): Tensor {
    return this.engine.contiguous(this);
  }

  /**
   * Select a contiguous sub-range along one dimension.
   * Returns a view (zero cost, no data copy).
   */
  narrow(dim: number, start: number, length: number): Tensor {
    return this.engine.narrow(this, dim, start, length);
  }

  /**
   * Cast tensor to a different dtype.
   * Returns a new tensor with the specified dtype.
   */
  toDtype(dtype: DType): Tensor {
    return this.engine.toDtype(this, dtype);
  }

  /**
   * Convenience method to cast to f16 (half precision).
   */
  half(): Tensor {
    return this.toDtype("f16");
  }

  /**
   * Convenience method to cast to f32 (single precision).
   */
  float(): Tensor {
    return this.toDtype("f32");
  }

  /**
   * Convenience method to cast to i32 (signed integer).
   */
  int(): Tensor {
    return this.toDtype("i32");
  }

  /**
   * Transfer to a different device (lazy).
   */
  to(device: DeviceKind): Tensor {
    return this.engine.to(this, device);
  }

  /**
   * Transfer to a different device and force immediately.
   */
  async toNow(device: DeviceKind): Promise<Tensor> {
    return this.engine.toNow(this, device);
  }

  sum(options?: SumOptions): number | Tensor {
    return this.engine.sum(this, options);
  }

  max(options?: MaxOptions): number | Tensor {
    return this.engine.max(this, options);
  }

  mean(options?: MeanOptions): number | Tensor {
    return this.engine.mean(this, options);
  }

  argmax(options: ArgReduceOptions): Tensor {
    return this.engine.argmax(this, options);
  }

  argmin(options: ArgReduceOptions): Tensor {
    return this.engine.argmin(this, options);
  }

  gt(other: Tensor): Tensor {
    return this.engine.gt(this, other);
  }

  lt(other: Tensor): Tensor {
    return this.engine.lt(this, other);
  }

  ge(other: Tensor): Tensor {
    return this.engine.ge(this, other);
  }

  le(other: Tensor): Tensor {
    return this.engine.le(this, other);
  }

  eq(other: Tensor): Tensor {
    return this.engine.eq(this, other);
  }

  ne(other: Tensor): Tensor {
    return this.engine.ne(this, other);
  }

  tril(k = 0): Tensor {
    return this.engine.tril(this, k);
  }

  triu(k = 0): Tensor {
    return this.engine.triu(this, k);
  }

  softmax(dim: number): Tensor {
    return this.engine.softmax(this, dim);
  }

  /**
   * Layer normalization along the last dimension.
   * layernorm(x, weight, bias, eps) = (x - mean) / sqrt(var + eps) * weight + bias
   */
  layernorm(weight: Tensor, bias: Tensor, eps = 1e-5): Tensor {
    return this.engine.layernorm(this, weight, bias, eps);
  }

  gather(index: Tensor, options: GatherOptions): Tensor {
    return this.engine.gather(this, index, options);
  }

  scatterAdd(index: Tensor, src: Tensor, options: ScatterAddOptions): Tensor {
    return this.engine.scatterAdd(this, index, src, options);
  }

  /**
   * Returns a new tensor detached from the computation graph.
   * The result will never require grad and will not participate in autograd.
   * Useful for stopping gradient flow, e.g., in LoRA training where base model
   * outputs should not contribute to the autograd graph.
   */
  detach(): Tensor {
    return this.engine.detach(this);
  }

  // ============================================================================
  // In-place operations (§4.3-4.4)
  // ============================================================================

  /**
   * Copy src values into this tensor in-place.
   * Returns this tensor (same object) with updated values.
   */
  copy_(src: Tensor): Tensor {
    return this.engine.copy_(this, src);
  }

  /**
   * Add src values to this tensor in-place.
   * Returns this tensor (same object) with updated values.
   */
  add_(src: Tensor): Tensor {
    return this.engine.add_(this, src);
  }

  /**
   * Zero out this tensor in-place.
   * Returns this tensor (same object) with all zeros.
   */
  zero_(): Tensor {
    return this.engine.zero_(this);
  }

  /**
   * Fill this tensor with a scalar value in-place.
   * Returns this tensor (same object) with all values set to the given value.
   */
  fill_(value: number): Tensor {
    return this.engine.fill_(this, value);
  }

  /**
   * Multiply this tensor by a scalar in-place.
   * Returns this tensor (same object) with values scaled.
   */
  mul_(value: number): Tensor {
    return this.engine.mul_(this, value);
  }

  _unwrap(): RuntimeTensor {
    return this.inner;
  }

  _engine(): Torchlette {
    return this.engine;
  }

  _engineTensor(): EngineTensor {
    return this.engineTensor;
  }

  _runtimeTensor(): RuntimeTensor {
    return this.inner;
  }

  _gradNode(): AutogradNode | null {
    return this.gradNode;
  }

  _setGradNode(node: AutogradNode | null): void {
    this.gradNode = node;
  }

  _setGrad(value: Tensor | null): void {
    // Dispose old gradient tensor to free GPU memory
    if (this.gradValue && this.gradValue !== value) {
      this.gradValue.dispose();
    }
    this.gradValue = value;
  }

  /**
   * Check if this tensor has been disposed.
   */
  get disposed(): boolean {
    return this.engineTensor.disposed;
  }

  _ensureNotDisposed(): void {
    this.ensureNotDisposed();
  }

  private ensureNotDisposed(): void {
    if (this.engineTensor.disposed) {
      throw new DisposedTensorError("Tensor has been disposed");
    }
  }
}

/**
 * Options for the autocast context.
 */
export type AutocastOptions = {
  /** Whether autocast is enabled (default: true) */
  enabled?: boolean;
  /** The AMP policy to use (default: DEFAULT_AMP_POLICY) */
  policy?: AMPPolicy;
  /** Device type hint for AMP (default: inferred from current backend) */
  deviceType?: "cpu" | "webgpu";
};

export type TorchletteOptions = {
  /** Enable fusion optimizations (§15). Default: false */
  enableFusion?: boolean;
  /** Enable memory planning for buffer reuse. Default: false */
  enableMemoryPlanning?: boolean;
  /** Maximum memory limit in bytes. Default: 10GB */
  memoryLimitBytes?: number;
  /** Enable early buffer release during execution for memory savings. Default: false */
  enableEarlyRelease?: boolean;
  /**
   * Enable segmented execution at checkpoint boundaries.
   * This enables memory savings for large models by executing checkpoint
   * segments separately and flushing buffers between them.
   * Default: false
   */
  enableCheckpointSegmentation?: boolean;
  /**
   * Enable true segmented execution with GPU synchronization.
   * Provides actual memory savings for checkpointed models by waiting for
   * GPU completion between segments before releasing buffers.
   * More effective than enableCheckpointSegmentation but slower.
   * Default: false
   */
  enableTrueSegmentation?: boolean;
};

export class Torchlette {
  private readonly engine: Engine;
  private readonly runtime: RuntimeEngine;
  private readonly autocastContext: AutocastContext;
  private inCompileRegion = false;
  /** Tensors created inside compile regions — eligible for disposal after backward */
  private _compileCreatedTensors = new WeakSet<Tensor>();
  /** Whether memory planning is available for compile regions (§0.1 goal #2) */
  private readonly memoryPlanningAvailable: boolean;
  /** Stack of saved tensor hooks for checkpointing (§10) */
  private savedTensorHooksStack: SavedTensorHooksContext[] = [];
  /** Label to capture on subsequent autograd nodes (for backward attribution) */
  private _currentNodeLabel: string | null = null;
  /** Hooks fired before each backward op */
  private _backwardDispatchHooks: Array<(info: { output: Tensor; inputs: Tensor[]; label?: string }) => void> = [];

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

  setDevice(name: DeviceKind): void {
    this.runtime.setBackend(name);
  }

  /**
   * Execute a function with automatic mixed precision (AMP) enabled.
   *
   * Inside the autocast block, eligible ops (like matmul) will use f16
   * for computation while maintaining f32 for accumulation. This provides
   * a performance boost while preserving numerical stability.
   *
   * Per spec §12, AMP transforms only apply inside compiled regions.
   * The "select-gated commit" mechanism allows the same compiled code
   * to work with or without AMP by using runtime flags.
   *
   * @example
   * ```ts
   * const result = await torch.autocast(async () => {
   *   const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
   *   const b = torch.tensorFromArray([5, 6, 7, 8], [2, 2]);
   *   return torch.matmul(a, b); // Uses f16 compute, f32 accumulate
   * });
   * ```
   */
  autocast<T>(fn: () => T, options?: AutocastOptions): T {
    const deviceType =
      options?.deviceType ?? (this.runtime.currentDefaultDevice === "webgpu" ? "webgpu" : "cpu");

    pushAutocast(this.autocastContext, {
      enabled: options?.enabled ?? true,
      policy: options?.policy ?? DEFAULT_AMP_POLICY,
      deviceType,
    });

    // Set the engine's autocast context for AMP transforms in compile (§12)
    this.engine.setAutocastContext(this.autocastContext);

    try {
      return fn();
    } finally {
      popAutocast(this.autocastContext);
      // Update engine's context to reflect the popped state
      this.engine.setAutocastContext(
        this.autocastContext.configStack.length > 0 ? this.autocastContext : null,
      );
    }
  }

  /**
   * Async version of autocast for async functions.
   *
   * @example
   * ```ts
   * const result = await torch.autocastAsync(async () => {
   *   const a = torch.tensorFromArray([1, 2, 3, 4], [2, 2]);
   *   const b = torch.tensorFromArray([5, 6, 7, 8], [2, 2]);
   *   const c = torch.matmul(a, b);
   *   return await c.cpu();
   * });
   * ```
   */
  async autocastAsync<T>(
    fn: () => Promise<T>,
    options?: AutocastOptions,
  ): Promise<T> {
    const deviceType =
      options?.deviceType ?? (this.runtime.currentDefaultDevice === "webgpu" ? "webgpu" : "cpu");

    pushAutocast(this.autocastContext, {
      enabled: options?.enabled ?? true,
      policy: options?.policy ?? DEFAULT_AMP_POLICY,
      deviceType,
    });

    // Set the engine's autocast context for AMP transforms in compile (§12)
    this.engine.setAutocastContext(this.autocastContext);

    try {
      return await fn();
    } finally {
      popAutocast(this.autocastContext);
      // Update engine's context to reflect the popped state
      this.engine.setAutocastContext(
        this.autocastContext.configStack.length > 0 ? this.autocastContext : null,
      );
    }
  }

  /**
   * Context manager for intercepting tensor save/restore operations.
   * This is the foundation for gradient checkpointing (§10).
   *
   * During forward pass within this context:
   * - When a tensor is saved for backward, `packHook(tensor)` is called
   * - The return value (may be a placeholder) is stored instead of the tensor
   *
   * During backward pass:
   * - When a saved tensor is needed, `unpackHook(packed)` is called
   * - Returns the actual tensor (may trigger recomputation)
   *
   * @example
   * ```ts
   * // Simple passthrough hooks (no-op)
   * const result = torch.saved_tensors_hooks(
   *   (tensor) => tensor,        // pack: keep tensor as-is
   *   (packed) => packed,        // unpack: return tensor as-is
   *   () => torch.mul(a, b)
   * );
   *
   * // Checkpointing hooks (pack to placeholder, unpack triggers recompute)
   * const result = torch.saved_tensors_hooks(
   *   (tensor) => { checkpointIndex: idx, baseId: tensor.baseId },
   *   (packed) => recomputedTensors.get(packed.checkpointIndex),
   *   () => fn(...inputs)
   * );
   * ```
   */
  saved_tensors_hooks<T>(
    packHook: PackHook,
    unpackHook: UnpackHook,
    fn: () => T,
  ): T {
    this.savedTensorHooksStack.push({ packHook, unpackHook });
    try {
      return fn();
    } finally {
      this.savedTensorHooksStack.pop();
    }
  }

  /**
   * Get the current saved tensor hooks context (if any).
   * Returns the topmost hooks on the stack, or null if none.
   */
  _getSavedTensorHooks(): SavedTensorHooksContext | null {
    return this.savedTensorHooksStack.length > 0
      ? this.savedTensorHooksStack[this.savedTensorHooksStack.length - 1]
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

  tensorFromArray(
    values: number[],
    shape: number[],
    options?: TensorCreateOptions,
  ): Tensor {
    return this.wrap(
      this.runtime.tensorFromArray(values, shape, options?.device),
      options?.requiresGrad ?? false,
    );
  }

  /**
   * Create a tensor filled with random values uniformly distributed in [0, 1).
   */
  rand(shape: number[], options?: TensorCreateOptions): Tensor {
    const numElements = shape.reduce((a, b) => a * b, 1);
    const values = new Array(numElements);
    for (let i = 0; i < numElements; i++) {
      values[i] = Math.random();
    }
    return this.tensorFromArray(values, shape, options);
  }

  /**
   * Create a tensor filled with random values from a standard normal distribution.
   * Uses Box-Muller transform.
   */
  randn(shape: number[], options?: TensorCreateOptions): Tensor {
    const numElements = shape.reduce((a, b) => a * b, 1);
    const values = new Array(numElements);
    for (let i = 0; i < numElements; i += 2) {
      // Box-Muller transform
      const u1 = Math.random();
      const u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1 || 1e-10));
      const theta = 2 * Math.PI * u2;
      values[i] = r * Math.cos(theta);
      if (i + 1 < numElements) {
        values[i + 1] = r * Math.sin(theta);
      }
    }
    return this.tensorFromArray(values, shape, options);
  }

  /**
   * Create a tensor with values sampled from a Bernoulli distribution.
   * Each element is 1 with probability p, and 0 with probability (1-p).
   *
   * @param shape - Shape of the output tensor
   * @param p - Probability of 1 (default: 0.5)
   * @param options - Tensor creation options
   */
  bernoulli(shape: number[], p = 0.5, options?: TensorCreateOptions): Tensor {
    if (p < 0 || p > 1) {
      throw new Error(`Bernoulli probability must be between 0 and 1, got ${p}`);
    }
    const numElements = shape.reduce((a, b) => a * b, 1);
    const values = new Array(numElements);
    for (let i = 0; i < numElements; i++) {
      values[i] = Math.random() < p ? 1 : 0;
    }
    return this.tensorFromArray(values, shape, options);
  }

  /**
   * Create a tensor filled with zeros.
   */
  zeros(shape: number[], options?: TensorCreateOptions): Tensor {
    return this.wrap(
      this.runtime.zeros(shape, options?.device),
      options?.requiresGrad ?? false,
    );
  }

  /**
   * Create a tensor filled with ones.
   */
  ones(shape: number[], options?: TensorCreateOptions): Tensor {
    return this.full(shape, 1, options);
  }

  /**
   * Create a tensor filled with a specific value.
   */
  full(shape: number[], fillValue: number, options?: TensorCreateOptions): Tensor {
    return this.wrap(
      this.runtime.full(shape, fillValue, options?.device),
      options?.requiresGrad ?? false,
    );
  }

  /**
   * Create a 1-D tensor of evenly spaced values.
   * Like PyTorch's torch.arange(start, end, step).
   * If only one argument is given, it is treated as `end` (start=0, step=1).
   */
  arange(end: number, options?: { start?: number; step?: number; device?: DeviceKind; requiresGrad?: boolean }): Tensor {
    const start = options?.start ?? 0;
    const step = options?.step ?? 1;
    return this.wrap(
      this.runtime.arange(end, start, step, options?.device),
      options?.requiresGrad ?? false,
    );
  }

  /**
   * Return the lower-triangular part of a matrix, zeroing elements above the k-th diagonal.
   */
  tril(a: Tensor, k = 0): Tensor {
    this.assertUsable(a);
    return this.wrap(this.runtime.tril(a._unwrap(), k), false);
  }

  /**
   * Return the upper-triangular part of a matrix, zeroing elements below the k-th diagonal.
   */
  triu(a: Tensor, k = 0): Tensor {
    this.assertUsable(a);
    return this.wrap(this.runtime.triu(a._unwrap(), k), false);
  }

  // ============================================================================
  // Autocast dispatch helpers (§12)
  // ============================================================================

  /**
   * Cast a tensor for autocast, with differentiable backward through the cast.
   * Backward only upcasts gradients (e.g., f16→f32), never downcasts,
   * to preserve gradient precision during mixed-precision training.
   */
  private _autocastCast(a: Tensor, targetDtype: DType): Tensor {
    if (a.dtype === targetDtype) return a;
    const originalDtype = a.dtype;
    const inner = this.runtime.cast(a._unwrap(), targetDtype);
    return this.wrapWithGrad(inner, [a], (grad, _getSaved) => {
      if (grad.dtype === originalDtype) return [grad];
      // Only upcast: promote to whichever dtype is higher precision
      const target = promoteDtype(grad.dtype, originalDtype);
      if (target === grad.dtype) return [grad]; // originalDtype is lower → keep grad as-is
      return [this.runtime.cast(grad, target)];
    });
  }

  /**
   * Unified autocast dispatch: applies autocast policy AND dtype promotion
   * based on the centralized Op Dtype Registry.
   *
   * - f16_eligible: cast f32 inputs → f16 when autocast is active
   * - f32_required: cast f16 inputs → f32 (always, for numerical stability)
   * - promote_inputs: promote mismatched dtypes to f32 (always active)
   *
   * Uses differentiable casts (_autocastCast) so gradients flow correctly.
   */
  private _applyAutocast(op: string, inputs: Tensor[]): Tensor[] {
    const rule = OP_DTYPE_RULES[op as LazyOpCode];

    // Autocast policy (only when active)
    if (this.autocastContext.current.enabled) {
      const policy = this.autocastContext.current.policy;
      // Check both the registry rule and the supplementary sets
      const isF16Eligible = (rule && rule.category === "f16_eligible") || F16_ELIGIBLE_OPS.has(op);
      const isF32Required = (rule && rule.category === "f32_required") || F32_REQUIRED_OPS.has(op);

      if (isF16Eligible && policy.computeDtype === "f16") {
        inputs = inputs.map(t => t.dtype === "f32" ? this._autocastCast(t, "f16") : t);
      }
      if (isF32Required) {
        inputs = inputs.map(t => t.dtype === "f16" ? this._autocastCast(t, "f32") : t);
      }
    }

    // Binary dtype promotion (always active, with differentiable cast)
    if (rule && rule.category === "promote_inputs" && inputs.length >= 2) {
      const [a, b] = inputs;
      if (a.dtype !== b.dtype) {
        const target = promoteDtype(a.dtype, b.dtype);
        return [
          a.dtype === target ? a : this._autocastCast(a, target),
          b.dtype === target ? b : this._autocastCast(b, target),
          ...inputs.slice(2),
        ];
      }
    }

    return inputs;
  }

  add(a: Tensor | number, b: Tensor | number): Tensor {
    // Unwrap: numbers go directly to RuntimeEngine as scalars
    const aUnwrap: import("./runtime/engine").TensorOrScalar = typeof a === "number" ? a : a._unwrap();
    const bUnwrap: import("./runtime/engine").TensorOrScalar = typeof b === "number" ? b : b._unwrap();
    const tensors = [a, b].filter((x): x is Tensor => typeof x !== "number");
    this.assertUsable(...tensors);
    if (typeof a !== "number" && typeof b !== "number") {
      [a, b] = this._applyAutocast("add", [a, b]) as [Tensor, Tensor];
      const inner = this.runtime.add(a._unwrap(), b._unwrap());
      const aShape = a.shape;
      const bShape = b.shape;
      return this.wrapWithGrad(inner, [a, b], (grad, _getSaved) => [
        this.sumToShape(grad, aShape),
        this.sumToShape(grad, bShape),
      ]);
    }
    // At least one operand is a number — no grad needed for the number
    const inner = this.runtime.add(aUnwrap, bUnwrap);
    const tensorInput = typeof a !== "number" ? a : (b as Tensor);
    return this.wrapWithGrad(inner, [tensorInput], (grad, _getSaved) => [
      this.sumToShape(grad, tensorInput.shape),
    ]);
  }

  sub(a: Tensor, b: Tensor, options?: SubOptions): Tensor {
    this.assertUsable(a, b);
    [a, b] = this._applyAutocast("sub", [a, b]) as [Tensor, Tensor];
    const inner = this.runtime.sub(a._unwrap(), b._unwrap(), options);
    const aShape = a.shape;
    const bShape = b.shape;
    return this.wrapWithGrad(inner, [a, b], (grad, _getSaved) => {
      const alpha = options?.alpha ?? 1;
      const gradA = this.sumToShape(grad, aShape);
      const scaled = this.runtime.mul(grad, -alpha);
      const gradB = this.sumToShape(scaled, bShape);
      return [gradA, gradB];
    });
  }

  mul(a: Tensor | number, b: Tensor | number): Tensor {
    const tensors = [a, b].filter((x): x is Tensor => typeof x !== "number");
    this.assertUsable(...tensors);
    if (typeof a !== "number" && typeof b !== "number") {
      [a, b] = this._applyAutocast("mul", [a, b]) as [Tensor, Tensor];
      const inner = this.runtime.mul(a._unwrap(), b._unwrap());
      const aShape = a.shape;
      const bShape = b.shape;
      const tensorsToSave =
        a.requiresGrad || b.requiresGrad ? [a, b] : [];
      return this.wrapWithGrad(
        inner,
        [a, b],
        (grad, getSaved) => {
          const savedA = getSaved(0);
          const savedB = getSaved(1);
          const gradA = this.sumToShape(
            this.runtime.mul(grad, savedB._unwrap()),
            aShape,
          );
          const gradB = this.sumToShape(
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
    return this.wrapWithGrad(inner, [tensorInput], (grad, _getSaved) => [
      this.sumToShape(this.runtime.mul(grad, scalarVal), tensorInput.shape),
    ]);
  }

  div(a: Tensor | number, b: Tensor | number): Tensor {
    const tensors = [a, b].filter((x): x is Tensor => typeof x !== "number");
    this.assertUsable(...tensors);
    if (typeof a !== "number" && typeof b !== "number") {
      [a, b] = this._applyAutocast("div", [a, b]) as [Tensor, Tensor];
      const inner = this.runtime.div(a._unwrap(), b._unwrap());
      const aShape = a.shape;
      const bShape = b.shape;
      const tensorsToSave =
        a.requiresGrad || b.requiresGrad ? [a, b] : [];
      return this.wrapWithGrad(
        inner,
        [a, b],
        (grad, getSaved) => {
          const savedA = getSaved(0);
          const savedB = getSaved(1);
          const gradA = this.sumToShape(
            this.runtime.div(grad, savedB._unwrap()),
            aShape,
          );
          const bSquared = this.runtime.mul(savedB._unwrap(), savedB._unwrap());
          const negA = this.runtime.neg(savedA._unwrap());
          const gradB = this.sumToShape(
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
      return this.wrapWithGrad(inner, [tensorInput], (grad, _getSaved) => [
        this.sumToShape(this.runtime.div(grad, scalarVal), tensorInput.shape),
      ]);
    }
    // scalar / b → grad_b = -scalar / b^2 * grad
    return this.wrapWithGrad(inner, [tensorInput], (grad, getSaved) => {
      const savedB = getSaved(0);
      const bSq = this.runtime.mul(savedB._unwrap(), savedB._unwrap());
      return [this.sumToShape(this.runtime.mul(grad, this.runtime.div(-scalarVal, bSq)), tensorInput.shape)];
    }, [tensorInput]);
  }

  matmul(a: Tensor, b: Tensor): Tensor {
    this.assertUsable(a, b);
    // Apply autocast: cast f32 inputs to f16 for compute-bound matmul
    const [castA, castB] = this._applyAutocast("matmul", [a, b]) as [Tensor, Tensor];
    const inner = this.runtime.matmul(castA._unwrap(), castB._unwrap());
    // Capture shapes of the ORIGINAL inputs for backward gradient shapes
    const aShape = a.shape;
    const bShape = b.shape;
    // Save cast tensors for backward so backward matmuls also run in the cast dtype
    const tensorsToSave =
      a.requiresGrad || b.requiresGrad ? [castA, castB] : [];
    return this.wrapWithGrad(
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
        const resultA = this.sumToShape(gradA, aShape);
        const resultB = this.sumToShape(gradB, bShape);
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
    this.assertUsable(input, weight);
    if (bias) this.assertUsable(bias);
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

    return this.wrapWithGrad(inner, allInputs, (grad, getSaved) => {
      if (inputShape.length < 2 || weightShape.length < 2)
        throw new Error("linear backward requires rank >= 2");

      let savedIdx = 0;
      // dX = dY @ W  (weight is [out, in], so dY @ W = [..., out] @ [out, in] = [..., in])
      let resultInput: RuntimeTensor | null = null;
      if (needsInputGrad) {
        const savedWeight = getSaved(savedIdx++)._unwrap();
        const gradInput = this.runtime.matmul(grad, savedWeight);
        resultInput = this.sumToShape(gradInput, inputShape);
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
        resultWeight = this.sumToShape(gradWeight, weightShape);
      }

      // dBias = sum(dY, all dims except last)
      let resultBias: RuntimeTensor | null = null;
      if (bias) {
        const biasShape = getSaved(savedIdx).shape;
        resultBias = this.sumToShape(grad, biasShape);
      }

      return bias ? [resultInput, resultWeight, resultBias!] : [resultInput, resultWeight];
    }, toSave);
  }

  sqrt(a: Tensor): Tensor {
    this.assertUsable(a);
    const inner = this.runtime.sqrt(a._unwrap());
    const tensorsToSave = a.requiresGrad ? [a] : [];
    return this.wrapWithGrad(
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
    this.assertUsable(a);
    const inner = this.runtime.relu(a._unwrap());
    const tensorsToSave = a.requiresGrad ? [a] : [];
    return this.wrapWithGrad(
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
    this.assertUsable(a);
    // Autocast: exp is F32-required for numerical stability
    const [castA] = this._applyAutocast("exp", [a]);
    const inner = this.runtime.exp(castA._unwrap());
    const tensorsToSave = a.requiresGrad ? [castA] : [];
    // Gradient of exp(x) is exp(x) = output
    // grad_input = grad_output * exp(input) = grad_output * output
    return this.wrapWithGrad(inner, [a], (grad, getSaved) => {
      // Recompute exp from saved input for checkpointing support
      const savedA = getSaved(0);
      const expA = this.runtime.exp(savedA._unwrap());
      const gradInput = this.runtime.mul(grad, expA);
      return [gradInput];
    }, tensorsToSave);
  }

  log(a: Tensor): Tensor {
    this.assertUsable(a);
    // Autocast: log is F32-required for numerical stability
    const [castA] = this._applyAutocast("log", [a]);
    const inner = this.runtime.log(castA._unwrap());
    const tensorsToSave = a.requiresGrad ? [castA] : [];
    // Gradient of log(x) is 1/x
    // grad_input = grad_output / (input + eps)
    return this.wrapWithGrad(
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
    this.assertUsable(a);
    const inner = this.runtime.neg(a._unwrap());
    // Gradient of -x is -1, no tensors need to be saved
    return this.wrapWithGrad(inner, [a], (grad, _getSaved) => {
      return [this.runtime.neg(grad)];
    });
  }

  abs(a: Tensor): Tensor {
    this.assertUsable(a);
    const inner = this.runtime.abs(a._unwrap());
    const aShape = a.shape;
    const aDevice = a.device;
    const tensorsToSave = a.requiresGrad ? [a] : [];
    // Gradient of abs(x) is sign(x)
    return this.wrapWithGrad(
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
    this.assertUsable(a);
    const inner = this.runtime.tanh(a._unwrap());
    const tensorsToSave = a.requiresGrad ? [a] : [];
    // Gradient of tanh(x) is (1 - tanh(x)^2) * grad
    return this.wrapWithGrad(inner, [a], (grad, getSaved) => {
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
    this.assertUsable(a);
    const inner = this.runtime.sigmoid(a._unwrap());
    const tensorsToSave = a.requiresGrad ? [a] : [];
    // Gradient of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x)) * grad
    return this.wrapWithGrad(inner, [a], (grad, getSaved) => {
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
    this.assertUsable(a);
    const approximate = options?.approximate ?? "tanh";
    const inner = this.runtime.gelu(a._unwrap(), options);
    const tensorsToSave = a.requiresGrad ? [a] : [];

    if (approximate === "tanh") {
      // GELU gradient using tanh approximation:
      // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      // d/dx GELU = cdf + x * pdf * 0.5
      // where cdf = 0.5 * (1 + tanh(inner)), inner = sqrt2OverPi * (x + 0.044715 * x^3)
      // pdf = sqrt2OverPi * (1 + 0.134145 * x^2) * (1 - tanh(inner)^2)
      return this.wrapWithGrad(
        inner,
        [a],
        (grad, getSaved) => {
          const savedA = getSaved(0);
          const x = savedA._unwrap();
          // x^2 and x^3
          const x2 = this.runtime.mul(x, x);
          const x3 = this.runtime.mul(x2, x);

          // inner = sqrt2OverPi * (x + 0.044715 * x^3)
          const term = this.runtime.add(x, this.runtime.mul(0.044715, x3));
          const innerVal = this.runtime.mul(0.7978845608, term);

          // Clamp inner to [-10, 10] to match forward pass and avoid NaN in tanh
          // For WebGPU: tanh(x) returns NaN for x >= 50, but inner can be ~56 for x=11
          // When clamped: tanh(10) ≈ 1.0, sech²(10) ≈ 0, giving correct gradient
          const clampedInner = this.runtime.where(
            this.runtime.lt(innerVal, -10),
            -10,
            this.runtime.where(
              this.runtime.gt(innerVal, 10),
              10,
              innerVal,
            ),
          );

          // tanh(clamped inner)
          const tanhInner = this.runtime.tanh(clampedInner);

          // cdf = 0.5 * (1 + tanh(inner))
          const cdf = this.runtime.mul(0.5, this.runtime.add(1, tanhInner));

          // sech^2(inner) = 1 - tanh^2(inner)
          const tanh2 = this.runtime.mul(tanhInner, tanhInner);
          const sech2 = this.runtime.sub(1, tanh2);

          // pdf = sqrt2OverPi * (1 + 0.134145 * x^2) * sech^2
          const pdfTerm = this.runtime.add(1, this.runtime.mul(0.134145, x2));
          const pdf = this.runtime.mul(this.runtime.mul(0.7978845608, pdfTerm), sech2);

          // grad_input = grad * (cdf + x * pdf * 0.5)
          const xPdfHalf = this.runtime.mul(this.runtime.mul(x, pdf), 0.5);
          const geluGrad = this.runtime.add(cdf, xPdfHalf);
          const gradInput = this.runtime.mul(grad, geluGrad);

          return [gradInput];
        },
        tensorsToSave,
      );
    } else {
      // GELU gradient using exact erf formula:
      // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
      // d/dx GELU = 0.5 * (1 + erf(x / sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi)
      //           = cdf + x * pdf
      // where cdf = 0.5 * (1 + erf(x / sqrt(2))), pdf = exp(-x^2/2) / sqrt(2*pi)
      return this.wrapWithGrad(
        inner,
        [a],
        (grad, getSaved) => {
          const savedA = getSaved(0);
          const x = savedA._unwrap();
          // Compute erf(x / sqrt(2)) using Abramowitz and Stegun approximation 7.1.26
          // erf(z) = 1 - (a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5) * exp(-z^2), where t = 1/(1+p*|z|)

          // z = x / sqrt(2)
          const z = this.runtime.mul(x, 0.7071067811865476);
          const absZ = this.runtime.abs(z);

          // t = 1 / (1 + p * |z|)
          const t = this.runtime.div(1, this.runtime.add(1, this.runtime.mul(0.3275911, absZ)));
          const t2 = this.runtime.mul(t, t);
          const t3 = this.runtime.mul(t2, t);
          const t4 = this.runtime.mul(t3, t);
          const t5 = this.runtime.mul(t4, t);

          // poly = a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
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

          // exp(-z^2)
          const z2 = this.runtime.mul(z, z);
          const negZ2 = this.runtime.mul(-0.5, this.runtime.mul(x, x));
          const expNegZ2 = this.runtime.exp(negZ2);

          // erf(z) = sign(z) * (1 - poly * exp(-z^2))
          const erfAbs = this.runtime.sub(1, this.runtime.mul(poly, expNegZ2));
          // For negative z, erf is negative, but we compute cdf = 0.5 * (1 + erf)
          // which handles the sign correctly through x

          // cdf = 0.5 * (1 + erf(x/sqrt(2)))
          // For x >= 0: cdf = 0.5 * (1 + erfAbs)
          // For x < 0: cdf = 0.5 * (1 - erfAbs)
          // We can use where(x >= 0, 1 + erfAbs, 1 - erfAbs) * 0.5
          const xGe0 = this.runtime.ge(x, 0);
          const erfPos = this.runtime.add(1, erfAbs);
          const erfNeg = this.runtime.sub(1, erfAbs);
          const erfTerm = this.runtime.where(xGe0, erfPos, erfNeg);
          const cdf = this.runtime.mul(0.5, erfTerm);

          // pdf = exp(-x^2/2) / sqrt(2*pi)
          const pdf = this.runtime.mul(expNegZ2, 0.3989422804014327);

          // grad_input = grad * (cdf + x * pdf)
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
    this.assertUsable(a);
    const inner = this.runtime.silu(a._unwrap());
    const aShape = a.shape;
    const aDevice = a.device;
    const tensorsToSave = a.requiresGrad ? [a] : [];
    // SiLU gradient: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //             = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    return this.wrapWithGrad(
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
    this.assertUsable(a);
    const inner = this.runtime.isfinite(a._unwrap());
    // isfinite is not differentiable - just wrap without grad
    return this.wrap(inner);
  }

  expand(a: Tensor, shape: number[]): Tensor {
    this.assertUsable(a);
    const aShape = a.shape;
    const inner = this.runtime.expand(a._unwrap(), shape);
    return this.wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this.sumToShape(grad, aShape),
    ]);
  }

  sum(a: Tensor, options?: SumOptions): Tensor {
    this.assertUsable(a);
    // Autocast: sum is F32-required for numerical stability
    const [castA] = this._applyAutocast("sum", [a]);
    const inner = this.runtime.sum(castA._unwrap(), options);
    const aShape = a.shape;
    const aRank = aShape.length;
    return this.wrapWithGrad(inner, [a], (grad, _getSaved) => {
      const dims = this.normalizeDims(options?.dim ?? null, aRank);
      const keepdim = options?.keepdim ?? false;
      const expanded = this.expandGrad(grad, aShape, dims, keepdim);
      return [expanded];
    });
  }

  max(a: Tensor, options?: MaxOptions): number | Tensor {
    this.assertUsable(a);
    const result = this.runtime.max(a._unwrap(), options);
    if (typeof result === "number") {
      return result;
    }
    // max doesn't support autograd in the standard way - would need argmax tracking
    // For use in softmax (numerical stability), we don't need gradients through max
    return this.wrap(result);
  }

  mean(a: Tensor, options?: MeanOptions): number | Tensor {
    this.assertUsable(a);
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
    const dims = this.normalizeDims(options?.dim ?? null, aShape.length);
    const keepdim = options?.keepdim ?? false;
    const reduceCount =
      dims.length === 0 ? 1 : dims.reduce((acc, dim) => acc * aShape[dim], 1);
    return this.wrapWithGrad(result, [a], (grad, _getSaved) => {
      const expanded = this.expandGrad(grad, aShape, dims, keepdim);
      const scaled = this.runtime.mul(expanded, 1 / reduceCount);
      return [scaled];
    });
  }

  /**
   * Argmax along a dimension.
   * Returns indices of maximum values.
   */
  argmax(a: Tensor, options: ArgReduceOptions): Tensor {
    this.assertUsable(a);
    const result = this.runtime.argmax(a._unwrap(), options);
    // argmax is not differentiable
    return this.wrap(result);
  }

  /**
   * Argmin along a dimension.
   * Returns indices of minimum values.
   */
  argmin(a: Tensor, options: ArgReduceOptions): Tensor {
    this.assertUsable(a);
    const result = this.runtime.argmin(a._unwrap(), options);
    // argmin is not differentiable
    return this.wrap(result);
  }

  /**
   * Greater than comparison.
   * Returns 1.0 where a > b, 0.0 elsewhere.
   */
  gt(a: Tensor, b: Tensor): Tensor {
    this.assertUsable(a, b);
    const result = this.runtime.gt(a._unwrap(), b._unwrap());
    // comparison ops are not differentiable
    return this.wrap(result);
  }

  /**
   * Less than comparison.
   * Returns 1.0 where a < b, 0.0 elsewhere.
   */
  lt(a: Tensor, b: Tensor): Tensor {
    this.assertUsable(a, b);
    const result = this.runtime.lt(a._unwrap(), b._unwrap());
    return this.wrap(result);
  }

  /**
   * Greater than or equal comparison.
   * Returns 1.0 where a >= b, 0.0 elsewhere.
   */
  ge(a: Tensor, b: Tensor): Tensor {
    this.assertUsable(a, b);
    const result = this.runtime.ge(a._unwrap(), b._unwrap());
    return this.wrap(result);
  }

  /**
   * Less than or equal comparison.
   * Returns 1.0 where a <= b, 0.0 elsewhere.
   */
  le(a: Tensor, b: Tensor): Tensor {
    this.assertUsable(a, b);
    const result = this.runtime.le(a._unwrap(), b._unwrap());
    return this.wrap(result);
  }

  /**
   * Equal comparison.
   * Returns 1.0 where a == b, 0.0 elsewhere.
   */
  eq(a: Tensor, b: Tensor): Tensor {
    this.assertUsable(a, b);
    const result = this.runtime.eq(a._unwrap(), b._unwrap());
    return this.wrap(result);
  }

  /**
   * Not equal comparison.
   * Returns 1.0 where a != b, 0.0 elsewhere.
   */
  ne(a: Tensor, b: Tensor): Tensor {
    this.assertUsable(a, b);
    const result = this.runtime.ne(a._unwrap(), b._unwrap());
    return this.wrap(result);
  }

  /**
   * Softmax along a dimension.
   * softmax(x, dim) = exp(x - max(x, dim, keepdim=true)) / sum(exp(...), dim, keepdim=true)
   */
  softmax(a: Tensor, dim: number): Tensor {
    this.assertUsable(a);
    // Autocast: softmax is F32-required for numerical stability
    const [castA] = this._applyAutocast("softmax", [a]);
    const rank = castA.shape.length;
    const normalizedDim = dim < 0 ? dim + rank : dim;
    if (normalizedDim < 0 || normalizedDim >= rank) {
      throw new Error(`softmax: dim ${dim} out of range for tensor of rank ${rank}`);
    }

    // Numerical stability: subtract max
    const maxResult = this.runtime.max(castA._unwrap(), { dim: normalizedDim, keepdim: true });
    if (typeof maxResult === "number") {
      throw new Error("softmax: max with keepdim=true should return tensor");
    }
    const shifted = this.runtime.sub(castA._unwrap(), maxResult);
    const exps = this.runtime.exp(shifted);
    const sumResult = this.runtime.sum(exps, { dim: normalizedDim, keepdim: true });
    if (typeof sumResult === "number") {
      throw new Error("softmax: sum with keepdim=true should return tensor");
    }
    const result = this.runtime.div(exps, sumResult);

    const tensorsToSave = a.requiresGrad ? [castA] : [];
    // Softmax backward: grad_input = softmax * (grad_output - sum(softmax * grad_output, dim, keepdim=true))
    return this.wrapWithGrad(result, [a], (grad, getSaved) => {
      // Recompute softmax from saved input for checkpointing support
      const savedA = getSaved(0);
      const savedMax = this.runtime.max(savedA._unwrap(), { dim: normalizedDim, keepdim: true });
      if (typeof savedMax === "number") {
        throw new Error("softmax backward: max with keepdim=true should return tensor");
      }
      const savedShifted = this.runtime.sub(savedA._unwrap(), savedMax);
      const savedExps = this.runtime.exp(savedShifted);
      const savedSum = this.runtime.sum(savedExps, { dim: normalizedDim, keepdim: true });
      if (typeof savedSum === "number") {
        throw new Error("softmax backward: sum with keepdim=true should return tensor");
      }
      const softmaxResult = this.runtime.div(savedExps, savedSum);

      const softmaxMulGrad = this.runtime.mul(softmaxResult, grad);
      const sumGradResult = this.runtime.sum(softmaxMulGrad, { dim: normalizedDim, keepdim: true });
      if (typeof sumGradResult === "number") {
        throw new Error("softmax backward: sum with keepdim=true should return tensor");
      }
      const gradMinusSum = this.runtime.sub(grad, sumGradResult);
      const gradInput = this.runtime.mul(softmaxResult, gradMinusSum);
      return [gradInput];
    }, tensorsToSave);
  }

  /**
   * Fused cross-entropy forward + backward for WebGPU.
   * logits [B, V] + targets [B] → per-sample loss [B]
   * Backward: fused kernel → grad_logits [B, V]
   */
  _crossEntropyFused(logits: Tensor, targets: Tensor): Tensor {
    this.assertUsable(logits, targets);

    // Upcast f16 logits to f32 for numerical stability (same as softmax)
    let castLogits = logits;
    if (logits.dtype === "f16") {
      castLogits = this._autocastCast(logits, "f32");
    }

    const B = castLogits.shape[0];
    const V = castLogits.shape[1];
    const config = { batchSize: B, vocabSize: V };

    const result = this.runtime.fusedCrossEntropyForward(
      castLogits._unwrap(), targets._unwrap(), config,
    );

    // Save logits for backward (needs recomputation of softmax).
    // Targets are captured via closure — they don't require grad and aren't
    // modified between forward and backward, so keep() is unnecessary.
    const tensorsToSave = logits.requiresGrad ? [castLogits] : [];
    const targetsInner = targets._unwrap();
    return this.wrapWithGrad(result, [logits], (grad, getSaved) => {
      const savedLogits = getSaved(0);
      const gradLogits = this.runtime.fusedCrossEntropyBackward(
        savedLogits._unwrap(), targetsInner, grad, config,
      );
      return [gradLogits];
    }, tensorsToSave);
  }

  /**
   * Scaled dot-product attention with optional causal mask.
   * q, k, v: [batch, heads, seq_len, head_dim]
   * Returns: [batch, heads, seq_len, head_dim]
   *
   * On WebGPU, uses fused FlashAttention kernel (single dispatch).
   * On CPU, falls back to decomposed matmul + softmax + matmul.
   */
  scaledDotProductAttention(
    q: Tensor, k: Tensor, v: Tensor, scale?: number, isCausal = false,
  ): Tensor {
    this.assertUsable(q, k, v);
    const [batch, heads, seq, hd] = q.shape;
    const actualScale = scale ?? (1.0 / Math.sqrt(hd));
    const config = {
      batchSize: batch, numHeads: heads, seqLen: seq, headDim: hd,
      scale: actualScale, isCausal,
    };

    if (q.device === "webgpu") {
      // Fused FlashAttention path
      const fwdResult = this.runtime.fusedAttentionForward(
        q._unwrap(), k._unwrap(), v._unwrap(), config,
      );
      const logsumexp = this.runtime.extractAttentionLogsumexp(fwdResult, config);

      // Save Q, K, V, logsumexp, and output O for backward.
      // IMPORTANT: Create reshape views with independent RuntimeTensors for saved
      // tensors. Without this, wrap(fwdResult) shares the same RuntimeTensor as the
      // wrapWithGrad output. When checkpoint tidy disposes these wrappers, it calls
      // dispose() on the shared RuntimeTensor BEFORE materialization, setting _disposed=true.
      // Later disposal of the real output hits the idempotency guard and skips cleanup,
      // leaking the GPU buffer. Similarly, disposing the logsumexp wrapper unregisters
      // its pending node, preventing extractAttentionLogsumexp from executing and
      // orphaning the side-output buffer. Reshape-to-same-shape creates a new
      // RuntimeTensor that can be safely disposed without poisoning the original.
      const logsumexpRef = this.runtime.reshape(logsumexp, [batch, heads, seq]);
      const logsumexpTensor = this.wrap(logsumexpRef);
      const outputRef = this.runtime.reshape(fwdResult, [batch, heads, seq, hd]);
      const outputTensor = this.wrap(outputRef);

      const tensorsToSave = (q.requiresGrad || k.requiresGrad || v.requiresGrad)
        ? [q, k, v, logsumexpTensor, outputTensor]
        : [];

      return this.wrapWithGrad(fwdResult, [q, k, v], (dO, getSaved) => {
        const sQ = getSaved(0);
        const sK = getSaved(1);
        const sV = getSaved(2);
        const sL = getSaved(3);
        const sO = getSaved(4);

        // O is the 6th input to backward for D precomputation
        const bwdDQ = this.runtime.fusedAttentionBackward(
          sQ._unwrap(), sK._unwrap(), sV._unwrap(), sL._unwrap(), dO, sO._unwrap(), config,
        );
        const dK = this.runtime.extractAttentionDK(bwdDQ, config);
        const dV = this.runtime.extractAttentionDV(bwdDQ, config);
        return [bwdDQ, dK, dV]; // dQ, dK, dV for inputs [q, k, v]
      }, tensorsToSave);
    }

    // CPU fallback: decomposed matmul + softmax + matmul
    const kT = this.runtime.transpose(k._unwrap(), { dim0: 2, dim1: 3 });
    const scores = this.runtime.matmul(q._unwrap(), kT);
    const scaleTensor = this.runtime.full([], actualScale, q.device);
    const scaledScores = this.runtime.mul(scores, scaleTensor);

    let finalScores: import("./runtime/engine").Tensor;
    if (isCausal) {
      // Create causal mask: -inf for positions where j > i
      const maskData: number[] = [];
      for (let i = 0; i < seq; i++) {
        for (let j = 0; j < seq; j++) {
          maskData.push(j > i ? -1e9 : 0);
        }
      }
      const mask = this.runtime.tensorFromArray(maskData, [1, 1, seq, seq], q.device);
      finalScores = this.runtime.add(scaledScores, mask);
    } else {
      finalScores = scaledScores;
    }

    // Softmax along last dim (using the public softmax which handles autograd)
    const softmaxResult = this.softmax(this.wrap(finalScores), -1);
    const output = this.runtime.matmul(softmaxResult._unwrap(), v._unwrap());

    // Wrap with autograd
    const tensorsToSave = (q.requiresGrad || k.requiresGrad || v.requiresGrad)
      ? [q, k, v, softmaxResult]
      : [];

    return this.wrapWithGrad(output, [q, k, v], (dO, getSaved) => {
      const sQ = getSaved(0);
      const sK = getSaved(1);
      const sV = getSaved(2);
      const sSoftmax = getSaved(3);

      // dV = attn_weights^T @ dO
      const attnT = this.runtime.transpose(sSoftmax._unwrap(), { dim0: 2, dim1: 3 });
      const dV = this.runtime.matmul(attnT, dO);

      // dAttn = dO @ V^T
      const vT = this.runtime.transpose(sV._unwrap(), { dim0: 2, dim1: 3 });
      const dAttn = this.runtime.matmul(dO, vT);

      // dScores = softmax_backward(dAttn, softmax_out)
      const dAttnTimesSoftmax = this.runtime.mul(dAttn, sSoftmax._unwrap());
      const sumDAttnSoftmax = this.runtime.sum(dAttnTimesSoftmax, { dim: -1, keepdim: true }) as import("./runtime/engine").Tensor;
      const dScoresSub = this.runtime.sub(dAttn, sumDAttnSoftmax);
      const dScores = this.runtime.mul(sSoftmax._unwrap(), dScoresSub);

      // Scale gradients
      const scaleT = this.runtime.full([], actualScale, sQ.device);
      const dScoresScaled = this.runtime.mul(dScores, scaleT);

      // dQ = dScoresScaled @ K
      const dQ = this.runtime.matmul(dScoresScaled, sK._unwrap());

      // dK = dScoresScaled^T @ Q
      const dScoresT = this.runtime.transpose(dScoresScaled, { dim0: 2, dim1: 3 });
      const dK = this.runtime.matmul(dScoresT, sQ._unwrap());

      return [dQ, dK, dV];
    }, tensorsToSave);
  }

  /**
   * Layer normalization along the last dimension.
   * layernorm(x, weight, bias, eps) = (x - mean) / sqrt(var + eps) * weight + bias
   *
   * Autograd:
   * - grad_input = (grad_output - mean(grad_output) - normalized * mean(grad_output * normalized)) / std
   * - grad_weight = sum(grad_output * normalized, dims except last)
   * - grad_bias = sum(grad_output, dims except last)
   */
  layernorm(x: Tensor, weight: Tensor, bias: Tensor, eps = 1e-5): Tensor {
    this.assertUsable(x, weight, bias);

    // Forward pass: normalize along last dimension
    const xShape = x.shape;
    const rank = xShape.length;
    const dim = -1; // normalize along last dim
    const normalizedDim = dim < 0 ? dim + rank : dim;
    const lastDimSize = xShape[xShape.length - 1];

    // Save inputs for backward
    const tensorsToSave = x.requiresGrad || weight.requiresGrad || bias.requiresGrad
      ? [x, weight, bias]
      : [];

    // Use fused forward kernel on WebGPU
    if (x.device === "webgpu") {
      const numRows = xShape.slice(0, rank - 1).reduce((a, b) => a * b, 1);
      const config = { numRows, featureDim: lastDimSize, eps };
      const result = this.runtime.fusedLayerNormForward(
        x._unwrap(), weight._unwrap(), bias._unwrap(), config,
      );

      // Backward stays decomposed (recomputes forward from saved x, weight, bias)
      return this.wrapWithGrad(result, [x, weight, bias], (grad, getSaved) => {
        return this._layernormBackward(grad, getSaved, normalizedDim, lastDimSize, rank, eps);
      }, tensorsToSave);
    }

    // CPU: decomposed forward
    // mean(x, dim=-1, keepdim=true)
    const meanResult = this.runtime.mean(x._unwrap(), { dim: normalizedDim, keepdim: true });
    if (typeof meanResult === "number") {
      throw new Error("layernorm: mean with keepdim=true should return tensor");
    }

    // centered = x - mean
    const centered = this.runtime.sub(x._unwrap(), meanResult);

    // variance = mean(centered^2, dim=-1, keepdim=true)
    const centeredSq = this.runtime.mul(centered, centered);
    const varianceResult = this.runtime.mean(centeredSq, { dim: normalizedDim, keepdim: true });
    if (typeof varianceResult === "number") {
      throw new Error("layernorm: variance mean with keepdim=true should return tensor");
    }

    // std = sqrt(variance + eps)
    const variancePlusEps = this.runtime.add(varianceResult, eps);
    const std = this.runtime.sqrt(variancePlusEps);

    // normalized = centered / std
    const normalized = this.runtime.div(centered, std);

    // output = normalized * weight + bias
    const scaled = this.runtime.mul(normalized, weight._unwrap());
    const result = this.runtime.add(scaled, bias._unwrap());

    // Autograd - recompute intermediates from saved inputs for checkpointing support
    return this.wrapWithGrad(result, [x, weight, bias], (grad, getSaved) => {
      return this._layernormBackward(grad, getSaved, normalizedDim, lastDimSize, rank, eps);
    }, tensorsToSave);
  }

  /** Shared backward for LayerNorm. Uses fused gradX kernel on WebGPU. */
  private _layernormBackward(
    grad: RuntimeTensor,
    getSaved: (i: number) => Tensor,
    normalizedDim: number,
    lastDimSize: number,
    rank: number,
    eps = 1e-5,
  ): RuntimeTensor[] {
    const savedX = getSaved(0);
    const savedWeight = getSaved(1);

    let gradWeight: RuntimeTensor;
    let gradBias: RuntimeTensor;
    let gradX: RuntimeTensor;

    if (savedX.device === "webgpu") {
      // Fully fused path: 2 dispatches total (gradX + gradWeightBias)
      const numRows = savedX.shape.slice(0, rank - 1).reduce((a: number, b: number) => a * b, 1);
      const config = { numRows, featureDim: lastDimSize, eps };

      gradX = this.runtime.fusedLayerNormBackwardGradX(
        grad, savedX._unwrap(), savedWeight._unwrap(), config,
      );

      const gradWeightTensor = this.runtime.fusedLayerNormBackwardGradWeightBias(
        grad, savedX._unwrap(), config,
      );
      gradWeight = gradWeightTensor;
      gradBias = this.runtime.extractLnBwdGradBias(gradWeightTensor, lastDimSize);
    } else {
      // CPU decomposed path
      const recomputeMean = this.runtime.mean(savedX._unwrap(), { dim: normalizedDim, keepdim: true });
      if (typeof recomputeMean === "number") {
        throw new Error("layernorm backward: mean should return tensor");
      }
      const recomputeCentered = this.runtime.sub(savedX._unwrap(), recomputeMean);
      const recomputeCenteredSq = this.runtime.mul(recomputeCentered, recomputeCentered);
      const recomputeVariance = this.runtime.mean(recomputeCenteredSq, { dim: normalizedDim, keepdim: true });
      if (typeof recomputeVariance === "number") {
        throw new Error("layernorm backward: variance mean should return tensor");
      }
      const recomputeVarPlusEps = this.runtime.add(recomputeVariance, eps);
      const recomputeStd = this.runtime.sqrt(recomputeVarPlusEps);
      const recomputeNormalized = this.runtime.div(recomputeCentered, recomputeStd);

      const sumDims = Array.from({ length: rank - 1 }, (_, i) => i);

      let gradBiasReduced = grad;
      for (let i = sumDims.length - 1; i >= 0; i--) {
        const sumResult = this.runtime.sum(gradBiasReduced, { dim: sumDims[i], keepdim: false });
        if (typeof sumResult === "number") {
          throw new Error("layernorm backward: sum for gradBias should return tensor");
        }
        gradBiasReduced = sumResult;
      }
      gradBias = gradBiasReduced;

      let gradWeightReduced = this.runtime.mul(grad, recomputeNormalized);
      for (let i = sumDims.length - 1; i >= 0; i--) {
        const sumResult = this.runtime.sum(gradWeightReduced, { dim: sumDims[i], keepdim: false });
        if (typeof sumResult === "number") {
          throw new Error("layernorm backward: sum for gradWeight should return tensor");
        }
        gradWeightReduced = sumResult;
      }
      gradWeight = gradWeightReduced;

      // Decomposed gradX for CPU
      const gradNormalized = this.runtime.mul(grad, savedWeight._unwrap());
      const gradCentered = this.runtime.div(gradNormalized, recomputeStd);

      const gradNormCentered = this.runtime.mul(gradNormalized, recomputeCentered);
      const sumGradNormCentered = this.runtime.sum(gradNormCentered, { dim: normalizedDim, keepdim: true });
      if (typeof sumGradNormCentered === "number") {
        throw new Error("layernorm backward: sum should return tensor");
      }
      const varStd = this.runtime.mul(recomputeVarPlusEps, recomputeStd);
      const gradVariance = this.runtime.mul(
        -0.5,
        this.runtime.div(sumGradNormCentered, varStd),
      );

      const gradCenteredFromVar = this.runtime.div(
        this.runtime.mul(
          this.runtime.mul(2, gradVariance),
          recomputeCentered,
        ),
        lastDimSize,
      );

      const totalGradCentered = this.runtime.add(gradCentered, gradCenteredFromVar);
      const sumTotalGradCentered = this.runtime.sum(totalGradCentered, { dim: normalizedDim, keepdim: true });
      if (typeof sumTotalGradCentered === "number") {
        throw new Error("layernorm backward: sum should return tensor");
      }
      const gradMean = this.runtime.neg(this.runtime.div(sumTotalGradCentered, lastDimSize));
      gradX = this.runtime.add(totalGradCentered, gradMean);
    }

    return [gradX, gradWeight, gradBias];
  }

  async cpu(a: Tensor): Promise<number[]> {
    this.assertUsable(a);
    return this.runEntryPoint(async () => {
      this.engine.forceRead(a.baseId);
      return this.runtime.cpu(a._unwrap());
    });
  }

  async item(a: Tensor): Promise<number> {
    this.assertUsable(a);
    return this.runEntryPoint(async () => {
      this.engine.forceRead(a.baseId);
      return this.runtime.item(a._unwrap());
    });
  }

  /**
   * Transfer tensor to a different device.
   * This is lazy - the transfer happens when the tensor is forced.
   */
  to(a: Tensor, device: DeviceKind): Tensor {
    this.assertUsable(a);
    if (a.device === device) {
      return a;
    }
    const inner = this.runtime.transfer(a._unwrap(), device);
    return this.wrap(inner, a.requiresGrad);
  }

  /**
   * Transfer tensor to a different device and force immediately.
   * Use when you need the transfer to complete before continuing.
   */
  async toNow(a: Tensor, device: DeviceKind): Promise<Tensor> {
    this.assertUsable(a);
    return this.runEntryPoint(async () => {
      this.engine.forceRead(a.baseId);
      const inner = await this.runtime.transferNow(a._unwrap(), device);
      return this.wrap(inner, a.requiresGrad);
    });
  }

  // ============================================================================
  // In-place operations (§4.3-4.4)
  // ============================================================================

  /**
   * Copy src values into dst tensor in-place.
   * Returns dst (same tensor object) with updated values.
   *
   * Per spec §4.3, emits base_commit for the in-place mutation.
   */
  copy_(dst: Tensor, src: Tensor): Tensor {
    this.assertUsable(dst, src);
    this.runtime.copy_(dst._unwrap(), src._unwrap());
    // Emit base_commit for mutation tracking (§4.3)
    this._debug_baseCommit(dst.baseId, this.engine.nextMutId());
    return dst;
  }

  /**
   * Add src values to dst tensor in-place.
   * Returns dst (same tensor object) with updated values.
   *
   * Per spec §4.3, emits base_commit for the in-place mutation.
   */
  add_(dst: Tensor, src: Tensor): Tensor {
    this.assertUsable(dst, src);
    this.runtime.add_(dst._unwrap(), src._unwrap());
    // Emit base_commit for mutation tracking (§4.3)
    this._debug_baseCommit(dst.baseId, this.engine.nextMutId());
    return dst;
  }

  /**
   * Zero out a tensor in-place.
   * Returns dst (same tensor object) with all zeros.
   *
   * Per spec §4.3, emits base_commit for the in-place mutation.
   */
  zero_(dst: Tensor): Tensor {
    this.assertUsable(dst);
    this.runtime.zero_(dst._unwrap());
    // Emit base_commit for mutation tracking (§4.3)
    this._debug_baseCommit(dst.baseId, this.engine.nextMutId());
    return dst;
  }

  /**
   * Fill a tensor with a scalar value in-place.
   * Returns dst (same tensor object) with all values set to the given value.
   *
   * Per spec §4.3, emits base_commit for the in-place mutation.
   */
  fill_(dst: Tensor, value: number): Tensor {
    this.assertUsable(dst);
    this.runtime.fill_(dst._unwrap(), value);
    // Emit base_commit for mutation tracking (§4.3)
    this._debug_baseCommit(dst.baseId, this.engine.nextMutId());
    return dst;
  }

  /**
   * Multiply tensor by scalar in-place.
   * Returns dst (same tensor object) with values scaled.
   *
   * Per spec §4.3, emits base_commit for the in-place mutation.
   */
  mul_(dst: Tensor, value: number): Tensor {
    this.assertUsable(dst);
    this.runtime.mul_(dst._unwrap(), value);
    // Emit base_commit for mutation tracking (§4.3)
    this._debug_baseCommit(dst.baseId, this.engine.nextMutId());
    return dst;
  }

  gather(a: Tensor, index: Tensor, options: GatherOptions): Tensor {
    this.assertUsable(a, index);
    const inner = this.runtime.gather(a._unwrap(), index._unwrap(), options);
    const aShape = a.shape;
    // Capture index tensor for backward
    const indexInner = index._unwrap();
    // Backward: scatter the gradient back to the positions we gathered from
    return this.wrapWithGrad(inner, [a], (grad, _getSaved) => {
      // Create zeros with same shape as input, then scatter_add the gradient
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
    this.assertUsable(a, index, src);
    const inner = this.runtime.scatterAdd(
      a._unwrap(),
      index._unwrap(),
      src._unwrap(),
      options,
    );
    // Capture index tensor for backward
    const indexInner = index._unwrap();
    // Backward: gradient for src is gather from output gradient positions
    // gradient for a is just the output gradient (scatter_add adds, so grad passes through)
    return this.wrapWithGrad(inner, [a, src], (grad, _getSaved) => {
      // grad_a = grad (the output gradient passes through to positions not written)
      const grad_a = grad;
      // grad_src = gather from grad at the scattered positions
      const grad_src = this.runtime.gather(grad, indexInner, options);
      return [grad_a, grad_src];
    });
  }

  /**
   * where(condition, x, y): returns x where condition is true (non-zero), else y.
   * Supports broadcasting across all three inputs.
   */
  where(condition: Tensor, x: Tensor, y: Tensor): Tensor {
    this.assertUsable(condition, x, y);
    const inner = this.runtime.where(
      condition._unwrap(),
      x._unwrap(),
      y._unwrap(),
    );
    const xShape = x.shape;
    const yShape = y.shape;
    // Capture condition for backward
    const conditionInner = condition._unwrap();
    // Backward: grad_x = where(condition, grad, 0), grad_y = where(condition, 0, grad)
    return this.wrapWithGrad(inner, [x, y], (grad, _getSaved) => {
      const zerosTensor = this.runtime.zeros(grad.shape, grad.device);
      const grad_x = this.runtime.where(conditionInner, grad, zerosTensor);
      const grad_y = this.runtime.where(conditionInner, zerosTensor, grad);
      return [this.sumToShape(grad_x, xShape), this.sumToShape(grad_y, yShape)];
    });
  }

  view(a: Tensor, shape: number[]): Tensor {
    this.assertUsable(a);
    const aShape = a.shape;
    const inner = this.runtime.view(a._unwrap(), shape);
    return this.wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this.runtime.reshape(grad, aShape),
    ]);
  }

  reshape(a: Tensor, shape: number[]): Tensor {
    this.assertUsable(a);
    const aShape = a.shape;
    const inner = this.runtime.reshape(a._unwrap(), shape);
    return this.wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this.runtime.reshape(grad, aShape),
    ]);
  }

  transpose(a: Tensor, options: TransposeOptions): Tensor {
    this.assertUsable(a);
    const inner = this.runtime.transpose(a._unwrap(), options);
    return this.wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this.runtime.transpose(grad, options),
    ]);
  }

  permute(a: Tensor, dims: number[]): Tensor {
    this.assertUsable(a);
    const inner = this.runtime.permute(a._unwrap(), dims);
    // Compute inverse permutation for backward
    const inverseDims = new Array<number>(dims.length);
    for (let i = 0; i < dims.length; i++) {
      inverseDims[dims[i]] = i;
    }
    return this.wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this.runtime.permute(grad, inverseDims),
    ]);
  }

  contiguous(a: Tensor): Tensor {
    this.assertUsable(a);
    const inner = this.runtime.contiguous(a._unwrap());
    // contiguous is a no-op for grad computation - just pass through
    return this.wrapWithGrad(inner, [a], (grad, _getSaved) => [grad]);
  }

  narrow(a: Tensor, dim: number, start: number, length: number): Tensor {
    this.assertUsable(a);
    const originalLength = a.shape[dim];
    const inner = this.runtime.narrow(a._unwrap(), dim, start, length);
    return this.wrapWithGrad(inner, [a], (grad, _getSaved) => [
      this.runtime.narrowBackward(grad, dim, start, originalLength),
    ]);
  }

  /**
   * Cast tensor to a different dtype.
   * Note: Autograd through dtype casts is not supported (gradients detach).
   */
  toDtype(a: Tensor, dtype: DType): Tensor {
    this.assertUsable(a);
    const inner = this.runtime.cast(a._unwrap(), dtype);
    // Dtype casts are not differentiable - return without autograd tracking
    return this.wrap(inner);
  }

  /**
   * Returns a new tensor detached from the computation graph.
   * The result will never require grad and will not participate in autograd.
   * Useful for stopping gradient flow, e.g., in LoRA training where base model
   * outputs should not contribute to the autograd graph.
   */
  detach(a: Tensor): Tensor {
    this.assertUsable(a);
    // Create a new tensor with the same data but no autograd tracking
    // The underlying tensor data is shared but the frontend wrapper is new
    return this.wrap(a._unwrap());
  }

  async backward(a: Tensor, grad?: Tensor): Promise<void> {
    this.assertUsable(a);
    if (grad) {
      this.assertUsable(grad);
      if (!shapesEqual(grad.shape, a.shape)) {
        throw new Error("backward grad shape mismatch");
      }
    }

    return this.runEntryPoint(async () => {
      // Wrap the entire backward pass in a TidyDispatchMode to auto-dispose
      // unwrapped RuntimeTensors (gradient accumulation intermediates,
      // non-retained grads) at scope exit. Leaf/retained grads survive via
      // wrap() → markEscaped().
      const tidyMode = new TidyDispatchMode();
      this.runtime.pushDispatchMode(tidyMode);
      try {
        return await this.engine.runWithAsyncScope(async () => {
        const seed = grad ? grad._unwrap() : this.seedGrad(a);

        // Use Tensor as key for gradient accumulation
        const gradMap = new Map<Tensor, RuntimeTensor>();
        gradMap.set(a, seed);

        const ordered: AutogradNode[] = [];
        const visited = new Set<AutogradNode>();

        // Graph traversal uses inputs array
        const visit = (node: AutogradNode) => {
          if (visited.has(node)) return;
          for (const input of node.inputs) {
            const inputNode = input._gradNode();
            if (inputNode) {
              visit(inputNode);
            }
          }
          visited.add(node);
          ordered.push(node);
        };

        const rootNode = a._gradNode();
        if (rootNode) visit(rootNode);

        // Force all tensors needed for backward
        // Skip disposed/materialized tensors to avoid redundant plan building
        const tensorsToForce: RuntimeTensor[] = [];
        if (!seed.isMaterialized()) tensorsToForce.push(seed);
        for (const node of ordered) {
          for (const input of node.inputs) {
            const rt = input._unwrap();
            if (!rt.disposed && !rt.isMaterialized()) {
              tensorsToForce.push(rt);
            }
          }
        }
        if (tensorsToForce.length > 0) {
          await this.runtime.forceAll(...tensorsToForce);
        }

        // ========================================================================
        // UNIFIED BACKWARD EXECUTION
        // ========================================================================
        // Phase A: Collect ALL saved tensors from ALL nodes (triggers lazy
        //          recompute graph construction for checkpointed tensors)
        // Phase B: Force ALL in ONE merged plan - this ensures checkpoint
        //          boundaries appear together, enabling proper segmentation
        // Phase C: Run backward functions with materialized tensors
        // ========================================================================

        const allUnpackedTensors = new Map<AutogradNode, Tensor[]>();

        // Phase A: Collect all unpacked tensors
        for (let i = ordered.length - 1; i >= 0; i -= 1) {
          const node = ordered[i];
          for (const slot of node.savedSlots) {
            this.engine._debug_useSavedTensor(slot.record);
          }
          const unpackedTensors: Tensor[] = [];
          for (const slot of node.savedSlots) {
            const tensor = slot.unpackHook(slot.packed);
            unpackedTensors.push(tensor);
          }
          allUnpackedTensors.set(node, unpackedTensors);
        }

        // Phase B: Force all saved tensors in ONE merged plan
        const allPending: RuntimeTensor[] = [];
        for (const tensors of allUnpackedTensors.values()) {
          for (const t of tensors) {
            allPending.push(t._unwrap());
          }
        }
        if (allPending.length > 0) {
          await this.runtime.forceAll(...allPending);
        }

        // Phase C: Run ALL backward functions lazily (no forcing), then force
        // all final gradients in a single forceAll() at the end.
        // Gradient accumulation intermediates (old existing + old gradIn after
        // runtime.add) are tracked by TidyDispatchMode and disposed at scope exit.
        for (let i = ordered.length - 1; i >= 0; i -= 1) {
          const node = ordered[i];
          const gradOutTensor = gradMap.get(node.output);
          if (!gradOutTensor) continue;

          const unpackedTensors = allUnpackedTensors.get(node) || [];
          const getSaved: GetSavedFn = (idx: number): Tensor => {
            if (idx >= unpackedTensors.length) {
              throw new Error(`No saved tensor at index ${idx}`);
            }
            return unpackedTensors[idx];
          };

          // Fire backward dispatch hooks
          for (const hook of this._backwardDispatchHooks) {
            hook({ output: node.output, inputs: node.inputs, label: node.label });
          }

          this.runtime.startIntermediateTracking();
          let gradsIn: Array<RuntimeTensor | null>;
          try {
            gradsIn = node.backward(gradOutTensor, getSaved);
          } catch (_e: any) {
            this.runtime.stopIntermediateTracking();
            await this.runtime.force(gradOutTensor);
            this.runtime.startIntermediateTracking();
            gradsIn = node.backward(gradOutTensor, getSaved);
          }

          const trackedIntermediates = this.runtime.stopIntermediateTracking();

          const keepSet = new Set(gradsIn.filter((g): g is RuntimeTensor => g !== null));
          for (const tensor of trackedIntermediates) {
            if (!keepSet.has(tensor) && !tensor.disposed) {
              tensor.dispose();
            }
          }

          // Track which gradIn RuntimeTensors are actually used (for requiresGrad inputs).
          // We need this because sumToShape may return the SAME RuntimeTensor for multiple
          // inputs when shapes match, so we can't dispose a non-requiresGrad gradIn if
          // it's also used for a requiresGrad input.
          const usedGradIns = new Set<RuntimeTensor>();
          const unusedGradIns: RuntimeTensor[] = [];
          for (let idx = 0; idx < node.inputs.length; idx += 1) {
            const input = node.inputs[idx];
            const gradIn = gradsIn[idx];
            if (!gradIn) continue;
            if (!input.requiresGrad) {
              unusedGradIns.push(gradIn);
              continue;
            }

            usedGradIns.add(gradIn);
            const existing = gradMap.get(input);
            if (existing) {
              const accumulated = this.runtime.add(existing, gradIn);
              gradMap.set(input, accumulated);
              // old existing and gradIn are tracked by TidyDispatchMode (not wrapped/escaped)
              // and will be disposed at backward scope exit.
            } else {
              gradMap.set(input, gradIn);
            }
          }
          // Dispose gradients for non-requiresGrad inputs that aren't shared
          // with any requiresGrad input. Without this, the RuntimeTensor stays
          // in pendingTensorsByNodeId, gets materialized at markStep, and is
          // never marked unreachable — leaking one StorageHandle per step.
          for (const grad of unusedGradIns) {
            if (!grad.disposed && !usedGradIns.has(grad)) {
              grad.dispose();
            }
          }
        }

        // Dispose checkpoint-recomputed tensors (no longer needed after Phase C).
        // Build set of all node inputs (parameters + user tensors) to protect them.
        // During checkpoint recomputation, parameters flow through fn(...inputs)
        // as the SAME Tensor objects and get recaptured — we must not dispose these.
        const protectedTensors = new Set<Tensor>();
        for (const node of ordered) {
          for (const input of node.inputs) {
            protectedTensors.add(input);
          }
        }
        for (const [node, tensors] of allUnpackedTensors.entries()) {
          for (let idx = 0; idx < tensors.length; idx++) {
            const unpacked = tensors[idx];
            const packed = node.savedSlots[idx]?.packed;
            // Non-checkpoint: packed === unpacked (identity hook) -> skip
            // Checkpoint: packed is a placeholder, unpacked is a Tensor
            //   - If unpacked is a parameter/input tensor -> skip (protected)
            //   - If unpacked is a recomputed intermediate -> dispose
            if (unpacked === packed) {
              // identity
            } else if (unpacked.isDisposed) {
              // already disposed (e.g. by tidy in checkpoint recomputation)
            } else if (protectedTensors.has(unpacked)) {
              // protected
            } else {
              unpacked.dispose();
            }
          }
        }
        allUnpackedTensors.clear();

        // Force ALL final gradients in one merged plan.
        const allGrads = [...gradMap.values()].filter(
          (g) => !g.isMaterialized() && !g.disposed
        );
        if (allGrads.length > 0) {
          await this.runtime.forceAll(...allGrads);
        }
        storageTracker.destroyUnreachable();

        // Store final gradients on leaf tensors and retained non-leaf tensors
        // Mark these tensors as "kept" so they survive async scope cleanup
        for (const [tensor, gradTensor] of gradMap) {
          const isLeaf = tensor.requiresGrad && !tensor._gradNode();
          const shouldRetain = tensor.isRetainGrad;

          if (isLeaf || shouldRetain) {
            // Wrap and keep the gradient tensor before async scope exits
            const gradWrapper = this.wrap(gradTensor, false);
            this.keep(gradWrapper);
            tensor._setGrad(gradWrapper);
          }
          // Non-retained grads: not wrapped, so TidyDispatchMode disposes them at scope exit
        }

        // Clean up saved tensors and autograd graph after backward
        // This is critical for memory management - without it, saved tensors
        // accumulate across training steps causing out-of-memory errors.

        // Build set of autograd node outputs - these are forward pass intermediates
        // that are safe to dispose.
        const forwardIntermediates = new Set<Tensor>();
        for (const node of ordered) {
          forwardIntermediates.add(node.output);
        }

        // Build set of tensors that must NOT be disposed:
        // - Parameters (leaf tensors with requiresGrad)
        // - User-held inputs that are not forward intermediates (e.g., x, target)
        const preserved = new Set<Tensor>();
        for (const [tensor, _grad] of gradMap) {
          const isLeaf = tensor.requiresGrad && !tensor._gradNode();
          if (isLeaf) preserved.add(tensor);
        }
        for (const node of ordered) {
          for (const input of node.inputs) {
            if (!forwardIntermediates.has(input) && !this._compileCreatedTensors.has(input)) {
              preserved.add(input);
            }
          }
        }

        // Collect forward intermediates to dispose (excluding preserved tensors)
        const toDispose = new Set<Tensor>();
        for (const node of ordered) {
          if (!preserved.has(node.output)) {
            toDispose.add(node.output);
          }
        }

        for (const node of ordered) {
          // Dispose saved tensors that are internal intermediates (e.g., autocast
          // casts). Only dispose if the saved tensor is NOT a preserved tensor
          // (parameter or user-held input) and is NOT already in toDispose.
          for (const slot of node.savedSlots) {
            const savedTensor = slot.packed as Tensor;
            if (savedTensor && typeof savedTensor.dispose === "function") {
              if (!preserved.has(savedTensor) && !savedTensor.disposed) {
                toDispose.add(savedTensor);
              }
            }
          }
          node.savedSlots.length = 0;
          node.output._setGradNode(null);
          node.inputs.length = 0;
        }

        // Dispose all collected tensors
        for (const tensor of toDispose) {
          if (!tensor.disposed) {
            tensor.dispose();
          }
        }

      });
      } finally {
        this.runtime.popDispatchMode();
        tidyMode.disposeNonEscaped();
      }
    });
  }

  private wrap(inner: RuntimeTensor, requiresGrad = false): Tensor {
    this.runtime.markEscaped(inner);
    const handle = this.engine.createTensor(inner.baseId);
    const tensor = new Tensor(this, inner, handle, { requiresGrad });
    if (this.inCompileRegion) {
      this._compileCreatedTensors.add(tensor);
    }
    return tensor;
  }

  /**
   * Wrap a runtime tensor with autograd support.
   *
   * @param inner - The underlying runtime tensor
   * @param inputs - Input tensors (for gradient flow)
   * @param backward - Backward function that receives (grad, getSaved) accessor
   * @param tensorsToSave - Tensors to save for backward pass (will be packed via hooks)
   */
  private wrapWithGrad(
    inner: RuntimeTensor,
    inputs: Tensor[],
    backward: GradFn,
    tensorsToSave: Tensor[] = [],
  ): Tensor {
    const requiresGrad = inputs.some((tensor) => tensor.requiresGrad);
    const output = this.wrap(inner, requiresGrad);

    if (requiresGrad) {
      // Create saved tensor slots, applying pack hooks if present
      const savedSlots: SavedTensorSlot[] = [];
      const hooks = this._getSavedTensorHooks();

      for (const tensor of tensorsToSave) {
        if (hooks) {
          // Apply pack hook - may return a placeholder instead of the tensor
          const packed = hooks.packHook(tensor);
          const record = this.engine._debug_saveForBackward(tensor.baseId);
          savedSlots.push({
            packed,
            unpackHook: hooks.unpackHook,
            record,
          });
        } else {
          // No hooks - save tensor directly
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

  private seedGrad(output: Tensor): RuntimeTensor {
    const size = sizeOf(output.shape);
    if (size !== 1) {
      throw new Error("backward requires an explicit grad for non-scalars");
    }
    return this.runtime.tensorFromArray([1], [], output.device);
  }

  private saveForBackward(tensor: Tensor): SavedTensorRecord {
    // Mark tensor as escaping any tidy scope so it's not disposed
    // Saved tensors are needed for backward pass
    this.keep(tensor);
    return this.engine._debug_saveForBackward(tensor.baseId);
  }

  private sumToShape(grad: RuntimeTensor, shape: number[]): RuntimeTensor {
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
        reduced = this.runtime.tensorFromArray([summed], [], grad.device);
      } else {
        reduced = summed;
      }
    }
    if (!shapesEqual(reduced.shape, shape)) {
      reduced = this.runtime.reshape(reduced, shape);
    }
    return reduced;
  }

  private expandGrad(
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

  private normalizeDims(dim: number | number[] | null, rank: number): number[] {
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

  /**
   * Async equivalent of tidy(). Tracks RuntimeTensors across await boundaries
   * and auto-disposes unwrapped intermediates at scope exit. Tensors that are
   * wrap()'d (returned from frontend ops) escape and are managed by the engine's
   * async scope instead.
   *
   * Use this for async training loops where intermediate tensors need cleanup
   * across await points.
   *
   * @example
   * ```ts
   * await api.asyncTidy(async () => {
   *   const output = model.forward(input);
   *   const loss = output.crossEntropy(target);
   *   await loss.backward();
   *   optimizer.step();
   *   // intermediate RuntimeTensors auto-disposed
   * });
   * ```
   */
  async asyncTidy<T>(fn: () => Promise<T>): Promise<T> {
    const tidyMode = new TidyDispatchMode();
    this.runtime.pushDispatchMode(tidyMode);
    try {
      return await this.engine.runWithAsyncScope(async () => {
        const result = await fn();
        // Keep returned tensors alive through async scope exit
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

  /**
   * Run an async function with an async scope context that tracks tensors
   * across await boundaries. Tensors created during the async operation
   * (but not within a synchronous tidy scope) are tracked and disposed
   * when the scope exits, unless explicitly kept.
   *
   * This is useful for custom async training loops where you need to
   * ensure intermediate tensors are cleaned up even across await points.
   *
   * Note: backward() automatically uses an async scope internally, so
   * you don't need to wrap backward() calls with this method.
   *
   * @param fn - Async function to run within the scope
   * @returns Result of the function
   *
   * @example
   * ```ts
   * await api.runWithAsyncScope(async () => {
   *   const a = api.randn([100, 100]);
   *   const b = api.randn([100, 100]);
   *   const c = a.matmul(b);
   *   await c.cpu(); // force materialization
   *   // a, b, c will be disposed when scope exits
   * });
   * ```
   */
  async runWithAsyncScope<T>(fn: () => Promise<T>): Promise<T> {
    return this.engine.runWithAsyncScope(fn);
  }

  keep(tensor: Tensor): void {
    this.assertUsable(tensor);
    this.engine.keep(tensor._engineTensor());
  }

  dispose(tensor: Tensor): void {
    this.assertSameEngine(tensor);
    // Clear autograd graph to release saved tensors
    // Saved tensors were marked with keep() so they persist across tidy scopes
    const gradNode = tensor._gradNode();
    if (gradNode) {
      // Clear saved tensor references
      gradNode.savedSlots.length = 0;
      // Clear input references
      gradNode.inputs.length = 0;
      tensor._setGradNode(null);
    }
    // Engine.dispose() will call the onDispose callback which frees GPU buffers
    this.engine.dispose(tensor._engineTensor());
  }

  /**
   * Mark the end of a training step.
   *
   * This is a critical cleanup boundary that:
   * - Forces all pending lazy computations
   * - Finalizes pending location bindings
   * - Drains the finalization queue (GC for tensors)
   * - Resets tokens for the next step
   *
   * Call this at the end of each training iteration to ensure proper
   * memory cleanup and prevent memory leaks.
   *
   * @example
   * ```ts
   * for (let step = 0; step < numSteps; step++) {
   *   const loss = model.forward(input);
   *   await loss.backward();
   *   optimizer.step();
   *   optimizer.zeroGrad();
   *   await api.markStep(); // Clean up intermediate tensors
   * }
   * ```
   */
  async markStep(): Promise<void> {
    // Step 0: If there's a deferred fence from the previous markStep, await it now.
    // This ensures buffers queued for destruction in the previous step are safe
    // to destroy before we issue new work. The fence may have already resolved
    // during the forward pass CPU dispatch, making this a near-instant await.
    try {
      await awaitDeferredFence();
    } catch {
      // Safe to ignore if WebGPU backend is not initialized (CPU-only usage).
    }

    // Step 1: Run the engine's markStep (token reset, finalization, etc.)
    await this.engine.markStep();

    // Step 2: Force all pending tensors to materialize
    // This ensures all pending work is executed before we try to destroy any storages.
    // Without this, we might destroy storages that are still needed by pending operations.
    await this.runtime.forceAllPending();

    // Step 3: GC - destroy all unreachable storages (intermediate buffers)
    storageTracker.destroyUnreachable();

    // Step 3.5: Reset cumulative fusion stats for the next step
    this.runtime.resetCumulativeFusionStats();

    // Step 4: Issue a deferred GPU fence. Instead of blocking here until all GPU
    // work completes (which can take 5-6s), we issue the fence and return immediately.
    // The next markStep will await the fence at Step 0. This allows the next step's
    // forward pass CPU dispatch to overlap with GPU cleanup.
    try {
      issueDeferredFence();
    } catch {
      // Safe to ignore if WebGPU backend is not initialized (CPU-only usage).
    }
  }

  _debug_baseCommit(baseId: number, mutId: number): void {
    this.engine._debug_baseCommit(baseId, mutId);
  }

  /**
   * Begin a step-level shared encoder scope.
   * Keeps the GPU command encoder open across force() boundaries within a training step,
   * reducing the number of GPU submits. Call endStep() when the step is complete.
   */
  async beginStep(): Promise<void> {
    await this.runtime.beginStep();
  }

  /**
   * End the step-level shared encoder scope.
   * Submits all remaining encoded GPU work.
   */
  endStep(): void {
    this.runtime.endStep();
  }

  _runtime(): RuntimeEngine {
    return this.runtime;
  }

  _wrapRuntime(inner: RuntimeTensor, requiresGrad: boolean): Tensor {
    return this.wrap(inner, requiresGrad);
  }

  private assertUsable(...tensors: Tensor[]): void {
    this.assertSameEngine(...tensors);
    this.assertSameDevice(...tensors);
    for (const tensor of tensors) {
      tensor._ensureNotDisposed();
    }
  }

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

  private async runEntryPoint<T>(fn: () => Promise<T>): Promise<T> {
    return this.engine.runEntryPoint(fn);
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

function sizeOf(shape: number[]): number {
  return shape.reduce((acc, dim) => acc * dim, 1);
}

function shapesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}
