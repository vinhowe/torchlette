export type Shape = number[];

export type DeviceKind = "cpu" | "webgpu" | "mock";

export type BackendTensor = {
  shape: Shape;
  toArray(): number[];
  /** Optional cleanup method to release GPU resources */
  destroy?(): void;
};

// ============================================================================
// Strided ViewMeta (§4.2-4.4)
// ============================================================================

/**
 * Metadata describing how a view maps to its base storage.
 * Per spec §2.3.
 */
export type ViewMeta = {
  /** The base tensor this view refers to */
  baseId: number;

  /** Element offset from start of base storage */
  offset: number;

  /** Shape of the view */
  shape: number[];

  /** Elements to skip for each dimension (row-major order) */
  strides: number[];

  /** True if memory layout is contiguous (enables fast paths) */
  isContiguous: boolean;
};

/**
 * Compute strides in elements for a contiguous tensor.
 * Returns row-major (C-style) strides: last dimension is contiguous.
 */
export function computeContiguousStrides(shape: number[]): number[] {
  if (shape.length === 0) return [];
  const strides = new Array<number>(shape.length);
  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/**
 * Check if strides represent a contiguous layout for the given shape.
 * Size-1 dimensions don't affect contiguity since stride doesn't matter.
 */
export function checkContiguous(shape: number[], strides: number[]): boolean {
  if (shape.length !== strides.length) return false;
  const expected = computeContiguousStrides(shape);
  for (let i = 0; i < shape.length; i++) {
    // Size-1 dims don't matter for contiguity
    if (shape[i] <= 1) continue;
    if (strides[i] !== expected[i]) return false;
  }
  return true;
}

/**
 * Compute the total number of elements from a shape.
 */
export function shapeSize(shape: number[]): number {
  if (shape.length === 0) return 1;
  return shape.reduce((a, b) => a * b, 1);
}

/**
 * Compute transposed strides given original strides and dimension swap.
 */
export function transposeStrides(
  strides: number[],
  dim0: number,
  dim1: number,
): number[] {
  const result = strides.slice();
  const tmp = result[dim0];
  result[dim0] = result[dim1];
  result[dim1] = tmp;
  return result;
}

/**
 * Compute transposed shape given original shape and dimension swap.
 */
export function transposeShape(
  shape: number[],
  dim0: number,
  dim1: number,
): number[] {
  const result = shape.slice();
  const tmp = result[dim0];
  result[dim0] = result[dim1];
  result[dim1] = tmp;
  return result;
}

export type DType = "f16" | "f32" | "i32" | "u32" | "bool";

export type SumOptions = {
  dim?: number | number[] | null;
  keepdim?: boolean;
  dtype?: DType | null;
};

export type MeanOptions = SumOptions;

export type MaxOptions = {
  dim?: number | number[] | null;
  keepdim?: boolean;
};

export type ArgReduceOptions = {
  dim: number;
  keepdim?: boolean;
};

export type SubOptions = {
  alpha?: number;
};

export type DivOptions = {
  roundingMode?: "floor" | "trunc" | null;
};

export type GatherOptions = {
  dim: number;
};

export type ScatterAddOptions = {
  dim: number;
};

export type TransposeOptions = {
  dim0: number;
  dim1: number;
};

/**
 * Options for strided scatter operations (§4.4 view mutation lowering).
 * Describes where within a base tensor to write values.
 */
export type StridedScatterOptions = {
  /** Element offset into the base tensor */
  offset: number;
  /** Shape of the view (must match src shape) */
  viewShape: number[];
  /** Element strides for the view */
  viewStrides: number[];
};

/**
 * GELU approximation type matching PyTorch's nn.GELU.
 * - "none": Exact formula using erf: x * 0.5 * (1 + erf(x / sqrt(2)))
 * - "tanh": Tanh approximation (GPT-2 "new GELU"): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
export type GeluApproximate = "none" | "tanh";

export type GeluOptions = {
  approximate?: GeluApproximate;
};

export type FusedAttentionConfig = {
  batchSize: number;
  numHeads: number;
  seqLen: number;
  headDim: number;
  scale: number;
  isCausal: boolean;
};

export type FusedCrossEntropyConfig = {
  batchSize: number;
  vocabSize: number;
};

export type FusedLayerNormConfig = {
  numRows: number;     // product of all dims except last (B*S for [B,S,D])
  featureDim: number;  // last dim size (D)
  eps: number;
};

export type AdamStepConfig = {
  beta1: number;
  beta2: number;
  stepSize: number;
  eps: number;
  weightDecay: number;
  lrTimesWd: number;
  decoupledWd: boolean;
  emitF16?: boolean;
  /** Gradient inverse scale factor for fused unscale (default 1.0 = no unscaling). */
  invScale?: number;
  /** Shared atomic inf-flag buffer for fused unscale inf detection. */
  infFlagBuffer?: unknown;
};

export interface BackendOps {
  tensorFromArray(values: number[], shape: Shape): BackendTensor;
  /** Create a zero-filled tensor. More efficient than tensorFromArray with a zeros array. */
  zeros?(shape: Shape): BackendTensor;
  /** Create a tensor filled with a constant value. */
  full?(shape: Shape, fillValue: number): BackendTensor;
  /** Create a 1-D tensor of evenly spaced values. */
  arange?(end: number, start?: number, step?: number): BackendTensor;
  /** Lower-triangular: zero elements above the k-th diagonal (last 2 dims). */
  tril?(a: BackendTensor, k?: number): BackendTensor;
  /** Upper-triangular: zero elements below the k-th diagonal (last 2 dims). */
  triu?(a: BackendTensor, k?: number): BackendTensor;
  /** Create a tensor filled with uniform random values in [0, 1). */
  rand?(shape: Shape, seed: number): BackendTensor;
  /** Create a tensor filled with standard normal random values. */
  randn?(shape: Shape, seed: number): BackendTensor;
  /** Create a tensor with Bernoulli-distributed values (1 with probability p, else 0). */
  bernoulli?(shape: Shape, p: number, seed: number): BackendTensor;
  add(a: BackendTensor, b: BackendTensor): BackendTensor;
  sub(a: BackendTensor, b: BackendTensor, options?: SubOptions): BackendTensor;
  div(a: BackendTensor, b: BackendTensor, options?: DivOptions): BackendTensor;
  mul(a: BackendTensor, b: BackendTensor): BackendTensor;
  matmul(a: BackendTensor, b: BackendTensor): BackendTensor;
  sqrt(a: BackendTensor): BackendTensor;
  relu(a: BackendTensor): BackendTensor;
  exp?(a: BackendTensor): BackendTensor;
  log?(a: BackendTensor): BackendTensor;
  neg?(a: BackendTensor): BackendTensor;
  abs?(a: BackendTensor): BackendTensor;
  tanh?(a: BackendTensor): BackendTensor;
  sigmoid?(a: BackendTensor): BackendTensor;
  gelu?(a: BackendTensor, options?: GeluOptions): BackendTensor;
  silu?(a: BackendTensor): BackendTensor;
  /** Check if values are finite (not NaN and not Inf). Returns 1.0 where finite, 0.0 elsewhere. */
  isfinite?(a: BackendTensor): BackendTensor;
  expand(a: BackendTensor, shape: Shape): BackendTensor;
  reshape(a: BackendTensor, shape: Shape): BackendTensor;
  transpose(a: BackendTensor, options: TransposeOptions): BackendTensor;
  /** Permute dimensions according to the given order. Returns a view. */
  permute(a: BackendTensor, dims: number[]): BackendTensor;
  /** Select a contiguous sub-range along one dimension. Returns a view (zero cost). */
  narrow?(a: BackendTensor, dim: number, start: number, length: number): BackendTensor;
  /**
   * Backward for narrow: pads gradient back to original shape.
   * Copies grad into [start, start+length) along dim, zeros elsewhere.
   */
  narrowBackward?(grad: BackendTensor, dim: number, start: number, originalLength: number): BackendTensor;
  /** Materialize non-contiguous tensor to contiguous buffer. Returns same tensor if already contiguous. */
  contiguous(a: BackendTensor): BackendTensor;
  /** Cast tensor to a different dtype. Returns same tensor if already target dtype. */
  cast?(a: BackendTensor, dtype: DType): BackendTensor;
  gather(
    a: BackendTensor,
    index: BackendTensor,
    options: GatherOptions,
  ): BackendTensor;
  scatterAdd(
    a: BackendTensor,
    index: BackendTensor,
    src: BackendTensor,
    options: ScatterAddOptions,
  ): BackendTensor;
  /** Sum reduction. Returns 0-d tensor for full reduction (no dim). */
  sum(a: BackendTensor, options?: SumOptions): BackendTensor;
  /** Max reduction. Returns 0-d tensor for full reduction (no dim). */
  max(a: BackendTensor, options?: MaxOptions): BackendTensor;
  /** Mean reduction. Returns 0-d tensor for full reduction (no dim). */
  mean(a: BackendTensor, options?: MeanOptions): BackendTensor;
  /** Argmax reduction. Returns indices of maximum values along dimension. */
  argmax?(a: BackendTensor, options: ArgReduceOptions): BackendTensor;
  /** Argmin reduction. Returns indices of minimum values along dimension. */
  argmin?(a: BackendTensor, options: ArgReduceOptions): BackendTensor;
  /** Greater than comparison. Returns 1.0 where a > b, 0.0 elsewhere. */
  gt?(a: BackendTensor, b: BackendTensor): BackendTensor;
  /** Less than comparison. Returns 1.0 where a < b, 0.0 elsewhere. */
  lt?(a: BackendTensor, b: BackendTensor): BackendTensor;
  /** Greater than or equal comparison. Returns 1.0 where a >= b, 0.0 elsewhere. */
  ge?(a: BackendTensor, b: BackendTensor): BackendTensor;
  /** Less than or equal comparison. Returns 1.0 where a <= b, 0.0 elsewhere. */
  le?(a: BackendTensor, b: BackendTensor): BackendTensor;
  /** Equal comparison. Returns 1.0 where a == b, 0.0 elsewhere. */
  eq?(a: BackendTensor, b: BackendTensor): BackendTensor;
  /** Not equal comparison. Returns 1.0 where a != b, 0.0 elsewhere. */
  ne?(a: BackendTensor, b: BackendTensor): BackendTensor;
  /** Ternary select: where(condition, x, y) -> x if condition else y */
  where(
    condition: BackendTensor,
    x: BackendTensor,
    y: BackendTensor,
  ): BackendTensor;
  /**
   * Copy src values into base tensor at positions defined by view metadata.
   * Returns a new tensor (does not mutate base).
   * Used for view mutation lowering (§4.4).
   */
  stridedScatterCopy(
    base: BackendTensor,
    src: BackendTensor,
    options: StridedScatterOptions,
  ): BackendTensor;
  /**
   * Add src values into base tensor at positions defined by view metadata.
   * Returns a new tensor (does not mutate base).
   * Used for in-place add on views.
   */
  stridedScatterAdd(
    base: BackendTensor,
    src: BackendTensor,
    options: StridedScatterOptions,
  ): BackendTensor;
  /** Fused Adam/AdamW optimizer step. Returns updated param, m, v. */
  adamStep?(
    grad: BackendTensor,
    param: BackendTensor,
    m: BackendTensor,
    v: BackendTensor,
    config: AdamStepConfig,
  ):
    | { param: BackendTensor; m: BackendTensor; v: BackendTensor }
    | Promise<{ param: BackendTensor; m: BackendTensor; v: BackendTensor }>;
  /** Fused unscale + inf-check + zero-mask for GradScaler. */
  unscaleGrad?(
    grad: BackendTensor,
    invScale: number,
    infFlagBuffer: unknown,
  ): BackendTensor;
  /** Fused attention forward: Q,K,V [B,H,N,D] → O [B,H,N,D] + logsumexp [B,H,N]. */
  fusedAttentionForward?(
    q: BackendTensor,
    k: BackendTensor,
    v: BackendTensor,
    config: FusedAttentionConfig,
  ): { output: BackendTensor; logsumexp: BackendTensor };
  /** Fused attention backward: Q,K,V,L,dO,O → dQ,dK,dV [B,H,N,D]. */
  fusedAttentionBackward?(
    q: BackendTensor,
    k: BackendTensor,
    v: BackendTensor,
    logsumexp: BackendTensor,
    dO: BackendTensor,
    output: BackendTensor,
    config: FusedAttentionConfig,
  ): { dQ: BackendTensor; dK: BackendTensor; dV: BackendTensor };
  /** Fused cross-entropy forward: logits [B,V] + targets [B] → per-sample loss [B]. */
  fusedCrossEntropyForward?(
    logits: BackendTensor,
    targets: BackendTensor,
    config: FusedCrossEntropyConfig,
  ): BackendTensor;
  /** Fused cross-entropy backward: logits [B,V] + targets [B] + grad [B] → grad_logits [B,V]. */
  fusedCrossEntropyBackward?(
    logits: BackendTensor,
    targets: BackendTensor,
    gradOutput: BackendTensor,
    config: FusedCrossEntropyConfig,
  ): BackendTensor;
  /** Fused LayerNorm forward: x [N,D] + weight [D] + bias [D] → output [N,D]. */
  fusedLayerNormForward?(
    x: BackendTensor,
    weight: BackendTensor,
    bias: BackendTensor,
    config: FusedLayerNormConfig,
  ): BackendTensor;
  /** Fused LayerNorm backward gradX: grad [N,D] + x [N,D] + weight [D] → gradX [N,D]. */
  fusedLayerNormBackwardGradX?(
    gradOutput: BackendTensor,
    x: BackendTensor,
    weight: BackendTensor,
    config: FusedLayerNormConfig,
  ): BackendTensor;
  /** Fused LayerNorm backward gradWeight+gradBias: grad [N,D] + x [N,D] → {gradWeight [D], gradBias [D]}. */
  fusedLayerNormBackwardGradWeightBias?(
    gradOutput: BackendTensor,
    x: BackendTensor,
    config: FusedLayerNormConfig,
  ): { gradWeight: BackendTensor; gradBias: BackendTensor };
  /** Create a zeroed inf-flag buffer for unscaleGrad. */
  createInfCountBuffer?(): unknown;
  /** Read inf flag (0.0 or 1.0) and destroy buffer. */
  readAndDestroyInfCount?(buffer: unknown): Promise<number>;
  read(a: BackendTensor): Promise<number[]>;
}

export interface Backend {
  name: DeviceKind;
  ops: BackendOps;
  /**
   * Optional pre-tune hook for autotuning matmul shapes before plan execution.
   * Called with matmul dimensions [m, n, k] arrays when autotune is enabled.
   */
  pretuneMatmulShapes?(shapes: Array<[number, number, number]>): Promise<void>;
  /**
   * Optional step-level shared encoder scope.
   * When active, keeps the GPU command encoder open across force() boundaries,
   * reducing the number of GPU submits per training step.
   */
  beginStep?(): void | Promise<void>;
  endStep?(): void;
}
