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
} from "./backend/types";
import type { EngineTensor } from "./engine/engine";
import type { Torchlette } from "./frontend";
import type { AutogradNode, TensorCreateOptions } from "./frontend-types";
import type { Tensor as RuntimeTensor } from "./runtime/tensor";

export class DisposedTensorError extends Error {
  name = "DisposedTensorError";
}

export class Tensor {
  private readonly engine: Torchlette;
  private readonly inner: RuntimeTensor;
  private readonly engineTensor: EngineTensor;
  private requiresGradValue: boolean;
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

  /**
   * Set whether this tensor requires gradient computation.
   * Like PyTorch's tensor.requires_grad_(bool).
   * Returns this tensor for chaining.
   */
  requires_grad_(value = true): this {
    this.ensureNotDisposed();
    this.requiresGradValue = value;
    return this;
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

  squeeze(dim?: number): Tensor {
    return this.engine.squeeze(this, dim);
  }

  unsqueeze(dim: number): Tensor {
    return this.engine.unsqueeze(this, dim);
  }

  sub(other: Tensor | number, options?: SubOptions): Tensor {
    return this.engine.sub(this, other, options);
  }

  matmul(other: Tensor): Tensor {
    return this.engine.matmul(this, other);
  }
  gelu(options?: GeluOptions): Tensor {
    return this.engine.gelu(this, options);
  }
  softplus(): Tensor {
    return this.engine.softplus(this);
  }
  fmod(other: Tensor): Tensor {
    return this.engine.fmod(this, other);
  }
  clamp(min: number | null, max: number | null): Tensor {
    return this.engine.clamp(this, min, max);
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
   * Split tensor into approximately equal chunks along a dimension.
   * Returns views (zero cost, no data copy). Last chunk may be smaller.
   */
  chunk(chunks: number, dim = 0): Tensor[] {
    return this.engine.chunk(this, chunks, dim);
  }

  /**
   * Split tensor into pieces of given size (or list of sizes) along a dimension.
   * Returns views (zero cost, no data copy).
   */
  split(splitSizeOrSections: number | number[], dim = 0): Tensor[] {
    return this.engine.split(this, splitSizeOrSections, dim);
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

  min(options?: MinOptions): number | Tensor {
    return this.engine.min(this, options);
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

  variance(options?: {
    dim?: number | number[] | null;
    correction?: number;
    keepdim?: boolean;
  }): Tensor {
    return this.engine.variance(this, options);
  }

  std(options?: {
    dim?: number | number[] | null;
    correction?: number;
    keepdim?: boolean;
  }): Tensor {
    return this.engine.std(this, options);
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

  /**
   * RMS normalization along the last dimension.
   * rmsnorm(x, weight, eps) = x / rms(x) * weight
   */
  rmsnorm(weight: Tensor, eps = 1e-5): Tensor {
    return this.engine.rmsnorm(this, weight, eps);
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

  /**
   * Alias for disposed property.
   */
  get isDisposed(): boolean {
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

// ============================================================================
// Loop-generated trivial methods (typed via interface merging)
// ============================================================================

const SIMPLE_UNARY_OPS = [
  "sqrt",
  "relu",
  "exp",
  "log",
  "neg",
  "abs",
  "tanh",
  "sigmoid",
  "silu",
  "sin",
  "cos",
  "rsqrt",
  "floor",
  "ceil",
  "round",
  "sign",
  "isfinite",
] as const;

// eslint-disable-next-line @typescript-eslint/no-empty-object-type
interface Tensor {
  sqrt(): Tensor;
  relu(): Tensor;
  exp(): Tensor;
  log(): Tensor;
  neg(): Tensor;
  abs(): Tensor;
  tanh(): Tensor;
  sigmoid(): Tensor;
  silu(): Tensor;
  sin(): Tensor;
  cos(): Tensor;
  rsqrt(): Tensor;
  floor(): Tensor;
  ceil(): Tensor;
  round(): Tensor;
  sign(): Tensor;
  isfinite(): Tensor;
}

for (const op of SIMPLE_UNARY_OPS) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (Tensor.prototype as any)[op] = function (this: Tensor) {
    return this._engine()[op](this);
  };
}

const SIMPLE_BINARY_OPS = ["add", "mul", "div", "pow"] as const;

interface Tensor {
  add(other: Tensor | number): Tensor;
  mul(other: Tensor | number): Tensor;
  div(other: Tensor | number): Tensor;
  pow(exponent: Tensor | number): Tensor;
}

for (const op of SIMPLE_BINARY_OPS) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (Tensor.prototype as any)[op] = function (
    this: Tensor,
    other: Tensor | number,
  ) {
    return this._engine()[op](this, other);
  };
}

const COMPARISON_OPS = ["gt", "lt", "ge", "le", "eq", "ne"] as const;

interface Tensor {
  gt(other: Tensor): Tensor;
  lt(other: Tensor): Tensor;
  ge(other: Tensor): Tensor;
  le(other: Tensor): Tensor;
  eq(other: Tensor): Tensor;
  ne(other: Tensor): Tensor;
}

for (const op of COMPARISON_OPS) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (Tensor.prototype as any)[op] = function (this: Tensor, other: Tensor) {
    return this._engine()[op](this, other);
  };
}
