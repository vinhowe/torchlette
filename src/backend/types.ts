export type Shape = number[];

export type DeviceKind = "cpu" | "webgpu" | "mock";

export type BackendTensor = {
  shape: Shape;
  /** Element data type. Both CPU and WebGPU backends always set this. */
  dtype?: DType;
  /** True if this tensor owns the underlying buffer; false for views/shared tensors. */
  ownsBuffer?: boolean;
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

export { checkContiguous } from "../core/shape";

/**
 * Normalize a possibly-negative dimension index to a non-negative one.
 */
export function normalizeDim(d: number, rank: number): number {
  return d < 0 ? d + rank : d;
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

/** Extract DType from a value, defaulting to "f32". */
export function ensureDType(value: DType | undefined): DType {
  return value ?? "f32";
}

/**
 * Backend-agnostic execution options for output buffer donation.
 * WebGPU backends accept `outBuffer` (a GPUBuffer); CPU backends ignore it.
 */
export type OpExecOptions = { outBuffer?: unknown };

export type ReduceDimOptions = {
  dim?: number | number[] | null;
  keepdim?: boolean;
};

export type SumOptions = ReduceDimOptions & { dtype?: DType | null };
export type MeanOptions = SumOptions;
export type MaxOptions = ReduceDimOptions;
export type MinOptions = ReduceDimOptions;

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

export type CatOptions = {
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
  ignoreIndex?: number;
};

export type FusedRoPEConfig = {
  /** Total element count in qk (and output). */
  total: number;
  seqLen: number;
  headDim: number;
  /** +1 for forward, -1 for backward (inverse rotation). */
  sinScale: number;
};

export type FusedLayerNormConfig = {
  numRows: number; // product of all dims except last (B*S for [B,S,D])
  featureDim: number; // last dim size (D)
  eps: number;
};

export type FusedRMSNormConfig = {
  numRows: number; // product of all dims except last (B*S for [B,S,D])
  featureDim: number; // last dim size (D)
  eps: number;
};

export type AdamStepConfig = {
  beta1: number;
  beta2: number;
  /** ORIGINAL (un-adjusted) epsilon. The kernel derives eps*sqrt(bc2). */
  eps: number;
  weightDecay: number;
  decoupledWd: boolean;
  emitF16?: boolean;
  /** Gradient inverse scale factor for fused unscale (default 1.0 = no unscaling). */
  invScale?: number;
  /** Shared atomic inf-flag buffer for fused unscale inf detection. */
  infFlagBuffer?: unknown;
};
// inc-2a: `stepSize` and `lrTimesWd` were RETIRED from AdamStepConfig. The
// bias-corrected step size and lr*wd are derived IN-KERNEL from the persistent
// on-device `t` (step counter) and `lr` tensor inputs — the config is fully
// static, killing the per-step volatile-uniform repack class.

/** One parameter's inputs for a batched Adam step. */
export type AdamBatchItem = {
  grad: BackendTensor;
  param: BackendTensor;
  m: BackendTensor;
  v: BackendTensor;
  /** Persistent 1-element f32 step counter (shared across the group's params). */
  t: BackendTensor;
  /** Persistent 1-element f32 learning rate (shared across the group's params). */
  lr: BackendTensor;
  config: AdamStepConfig;
};

/** Output tensors for one batched Adam item. */
export type AdamBatchResult = {
  param: BackendTensor;
  m: BackendTensor;
  v: BackendTensor;
};

export interface BackendOps {
  tensorFromArray(
    values: number[] | Float32Array | Int32Array | Uint32Array | Uint16Array,
    shape: Shape,
    dtype?: DType,
  ): BackendTensor;
  /** Create a zero-filled tensor. More efficient than tensorFromArray with a zeros array. */
  zeros?(shape: Shape, dtype?: DType): BackendTensor;
  /** Create a tensor filled with a constant value. */
  full?(shape: Shape, fillValue: number, dtype?: DType): BackendTensor;
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
  add(
    a: BackendTensor,
    b: BackendTensor,
    options?: OpExecOptions,
  ): BackendTensor;
  sub(
    a: BackendTensor,
    b: BackendTensor,
    options?: SubOptions & OpExecOptions,
  ): BackendTensor;
  div(
    a: BackendTensor,
    b: BackendTensor,
    options?: DivOptions & OpExecOptions,
  ): BackendTensor;
  mul(
    a: BackendTensor,
    b: BackendTensor,
    options?: OpExecOptions,
  ): BackendTensor;
  matmul(
    a: BackendTensor,
    b: BackendTensor,
    options?: OpExecOptions,
  ): BackendTensor;
  sqrt(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  relu(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  exp?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  log?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  neg?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  abs?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  tanh?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  sigmoid?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  gelu?(a: BackendTensor, options?: GeluOptions & OpExecOptions): BackendTensor;
  silu?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  sin?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  cos?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  rsqrt?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  floor?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  ceil?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  round?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  sign?(a: BackendTensor, options?: OpExecOptions): BackendTensor;
  clamp?(
    a: BackendTensor,
    min: number | null,
    max: number | null,
    options?: OpExecOptions,
  ): BackendTensor;
  pow?(
    a: BackendTensor,
    b: BackendTensor,
    options?: OpExecOptions,
  ): BackendTensor;
  /** Check if values are finite (not NaN and not Inf). Returns 1.0 where finite, 0.0 elsewhere. */
  isfinite?(a: BackendTensor): BackendTensor;
  expand(a: BackendTensor, shape: Shape): BackendTensor;
  reshape(a: BackendTensor, shape: Shape): BackendTensor;
  transpose(a: BackendTensor, options: TransposeOptions): BackendTensor;
  /** Permute dimensions according to the given order. Returns a view. */
  permute(a: BackendTensor, dims: number[]): BackendTensor;
  /** Select a contiguous sub-range along one dimension. Returns a view (zero cost). */
  narrow?(
    a: BackendTensor,
    dim: number,
    start: number,
    length: number,
  ): BackendTensor;
  /**
   * Backward for narrow: pads gradient back to original shape.
   * Copies grad into [start, start+length) along dim, zeros elsewhere.
   */
  narrowBackward?(
    grad: BackendTensor,
    dim: number,
    start: number,
    originalLength: number,
  ): BackendTensor;
  /** Materialize non-contiguous tensor to contiguous buffer. Returns same tensor if already contiguous. */
  contiguous(a: BackendTensor): BackendTensor;
  /** Cast tensor to a different dtype. Returns same tensor if already target dtype. */
  cast?(a: BackendTensor, dtype: DType): BackendTensor;
  /** 2D convolution. Input: [N,C,H,W], Weight: [Cout,Cin,kH,kW], Bias: [Cout]. */
  conv2d?(
    input: BackendTensor,
    weight: BackendTensor,
    bias: BackendTensor | undefined,
    options?: {
      stride?: number | [number, number];
      padding?: number | [number, number];
      outBuffer?: unknown;
    },
  ): BackendTensor;
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
  /** Concatenate tensors along an existing dimension. */
  cat?(tensors: BackendTensor[], options: CatOptions): BackendTensor;
  /** Sum reduction. Returns 0-d tensor for full reduction (no dim). */
  sum(a: BackendTensor, options?: SumOptions): BackendTensor;
  /** Max reduction. Returns 0-d tensor for full reduction (no dim). */
  max(a: BackendTensor, options?: MaxOptions): BackendTensor;
  /** Min reduction. Returns 0-d tensor for full reduction (no dim). */
  min?(a: BackendTensor, options?: MaxOptions): BackendTensor;
  /** Mean reduction. Returns 0-d tensor for full reduction (no dim). */
  mean(a: BackendTensor, options?: MeanOptions): BackendTensor;
  /** Argmax reduction. Returns indices of maximum values along dimension. */
  argmax?(a: BackendTensor, options: ArgReduceOptions): BackendTensor;
  /** Argmin reduction. Returns indices of minimum values along dimension. */
  argmin?(a: BackendTensor, options: ArgReduceOptions): BackendTensor;
  /** Greater than comparison. Returns 1.0 where a > b, 0.0 elsewhere. */
  gt?(
    a: BackendTensor,
    b: BackendTensor,
    options?: OpExecOptions,
  ): BackendTensor;
  /** Less than comparison. Returns 1.0 where a < b, 0.0 elsewhere. */
  lt?(
    a: BackendTensor,
    b: BackendTensor,
    options?: OpExecOptions,
  ): BackendTensor;
  /** Greater than or equal comparison. Returns 1.0 where a >= b, 0.0 elsewhere. */
  ge?(
    a: BackendTensor,
    b: BackendTensor,
    options?: OpExecOptions,
  ): BackendTensor;
  /** Less than or equal comparison. Returns 1.0 where a <= b, 0.0 elsewhere. */
  le?(
    a: BackendTensor,
    b: BackendTensor,
    options?: OpExecOptions,
  ): BackendTensor;
  /** Equal comparison. Returns 1.0 where a == b, 0.0 elsewhere. */
  eq?(
    a: BackendTensor,
    b: BackendTensor,
    options?: OpExecOptions,
  ): BackendTensor;
  /** Not equal comparison. Returns 1.0 where a != b, 0.0 elsewhere. */
  ne?(
    a: BackendTensor,
    b: BackendTensor,
    options?: OpExecOptions,
  ): BackendTensor;
  /** Ternary select: where(condition, x, y) -> x if condition else y */
  where(
    condition: BackendTensor,
    x: BackendTensor,
    y: BackendTensor,
    options?: OpExecOptions,
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
  /** Batched reduction: N independent same-config reductions in one kernel. */
  batchedReduction?(
    op: string,
    inputs: BackendTensor[],
    dim: number | number[],
    keepdim?: boolean,
  ): BackendTensor[];
  /** Fused Adam/AdamW optimizer step. Returns updated param, m, v. */
  adamStep?(
    grad: BackendTensor,
    param: BackendTensor,
    m: BackendTensor,
    v: BackendTensor,
    t: BackendTensor,
    lr: BackendTensor,
    config: AdamStepConfig,
  ):
    | { param: BackendTensor; m: BackendTensor; v: BackendTensor }
    | Promise<{ param: BackendTensor; m: BackendTensor; v: BackendTensor }>;
  /**
   * Batched Adam step: process N adamStep calls in one backend invocation.
   * Backends are free to fuse same-element-count items into packed kernel
   * dispatches. Returns results in input order.
   */
  adamStepBatch?(items: AdamBatchItem[]): AdamBatchResult[];
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
  /** Fused RoPE: rotate Q/K by position. qk [*,S,D] + cos [S,D/2] + sin [S,D/2]. */
  fusedRoPE?(
    qk: BackendTensor,
    cos: BackendTensor,
    sin: BackendTensor,
    config: FusedRoPEConfig,
  ): BackendTensor;
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
  /** Fused RMSNorm forward: x [N,D] + weight [D] → output [N,D]. */
  fusedRMSNormForward?(
    x: BackendTensor,
    weight: BackendTensor,
    config: FusedRMSNormConfig,
  ): BackendTensor;
  /** Fused RMSNorm backward gradX: grad [N,D] + x [N,D] + weight [D] → gradX [N,D]. */
  fusedRMSNormBackwardGradX?(
    gradOutput: BackendTensor,
    x: BackendTensor,
    weight: BackendTensor,
    config: FusedRMSNormConfig,
  ): BackendTensor;
  /** Fused RMSNorm backward gradWeight: grad [N,D] + x [N,D] + weight [D] → gradWeight [D]. */
  fusedRMSNormBackwardGradWeight?(
    gradOutput: BackendTensor,
    x: BackendTensor,
    weight: BackendTensor,
    config: FusedRMSNormConfig,
  ): BackendTensor;
  /** Create a zeroed inf-flag buffer for unscaleGrad. */
  createInfCountBuffer?(): unknown;
  /** Read inf flag (0.0 or 1.0) and destroy buffer. */
  readAndDestroyInfCount?(buffer: unknown): Promise<number>;
  read(a: BackendTensor): Promise<number[]>;
  /** Start async scalar readback: copy to staging buffer, return finish function. */
  startScalarReadback?(a: BackendTensor): () => Promise<number>;
  /**
   * GPU top-K prefilter readback: top-k (value, index) pairs of a 1-D slice
   * of a forced f32 tensor, sorted by (value desc, index asc). Reads ~2k*4
   * bytes instead of the full tensor — the sampling readback fast path.
   */
  readTopK?(
    a: BackendTensor,
    k: number,
    opts?: { offset?: number; length?: number },
  ): Promise<{ values: Float32Array; indices: Int32Array }>;
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

// ============================================================================
// FusedBackend — extended interface for backends with fused execution support
// ============================================================================

/**
 * Extended backend interface for backends that support fused kernel dispatch.
 *
 * Adds shared command encoder lifecycle, buffer pool management, and arena
 * support that enable the engine to orchestrate optimized execution without
 * importing backend-specific modules directly.
 *
 * Currently implemented by the WebGPU backend. A future Metal or Vulkan
 * backend could implement this interface to get fusion/batching/arena
 * support from the engine layer automatically.
 */
export interface FusedBackend extends Backend {
  /** Begin a shared command encoder scope. Nested calls increment a depth counter. */
  beginSharedEncoder(): void;
  /** End a shared command encoder scope. Submits when depth reaches 0. */
  endSharedEncoder(): void;
  /** Flush the current shared encoder (submit commands) without closing the scope. */
  flushSharedEncoder(): void;
  /** Move pending-release buffers back to the main pool for reuse. */
  flushBufferPool(): void;

  /** Activate a buffer arena for stable buffer identity across steps. */
  setActiveArena(arena: unknown): void;
  /** Deactivate the buffer arena. */
  clearActiveArena(): void;
  /** Set external input buffer mappings for arena resolution. */
  setArenaExternalInputBuffers(buffers: unknown[]): void;
  /** Clear external input buffer mappings. */
  clearArenaExternalInputBuffers(): void;

  /** Begin a batch execution scope (for checkpoint segmentation with GPU sync). */
  beginBatchExecution(): void;
  /** End batch execution and wait for GPU completion. */
  endBatchExecution(): Promise<void>;
  /** Check if a batch execution scope is active. */
  isBatchActive(): boolean;
  /** Abort the current batch execution scope. */
  abortBatch(): void;
}

/** Type guard: check if a backend supports fused execution (shared encoder, arena, etc.). */
export function isFusedBackend(backend: Backend): backend is FusedBackend {
  return "beginSharedEncoder" in backend;
}
