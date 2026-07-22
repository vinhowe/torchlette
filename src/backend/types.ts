export type Shape = number[];

export type DeviceKind = "cpu" | "webgpu" | "mock";

export type BackendTensor = {
  shape: Shape;
  /** Element data type. Both CPU and WebGPU backends always set this. */
  dtype?: DType;
  /** True if this tensor owns the underlying buffer; false for views/shared tensors. */
  ownsBuffer?: boolean;
  /**
   * Optional STORAGE FORMAT (docs/quantization-design.md phase 2). Present only
   * on packed-int weight operands (a matmul B operand). When `packing` is set,
   * the buffer is a packed-int weight and `dtype`/`shape` describe the PACKED
   * layout (u32, [N, K/4]); the companion scales live in `scales`. Consumers
   * that don't understand the packing MUST dequant explicitly — never read the
   * packed bytes as if they were the logical weight. Absent = degenerate
   * (plain dtype) case; the axis is a pure superset. Read ONLY at the backend
   * matmul seam — invisible to planner/tape/profiler.
   */
  format?: StorageFormat;
  /** Companion scales tensor for a packed weight (set iff format.packing). */
  scales?: BackendTensor;
  toArray(): number[];
  /** Optional cleanup method to release GPU resources */
  destroy?(): void;
};

/**
 * A tensor's storage format — the generalization of the dtype arc
 * (docs/quantization-design.md "Phase 2 altitude decision"). Plain dtype is the
 * degenerate `{ elementType }` (no `packing`) case. A packed-int weight carries
 * a `packing` descriptor; the packed buffer + scales companion together
 * reconstruct the logical `[N, K]` weight, but that reconstruction happens ONLY
 * inside the backend matmul kernel (fused dequant) or an explicit dequant op —
 * never as a graph-visible tensor above the backend.
 */
export type StorageFormat = {
  /** The logical element type of the UNPACKED weight (f16/f32). */
  elementType: DType;
  /** Present iff the operand is packed. Absent = plain dtype. */
  packing?: QuantPacking;
};

/** Weight-only quantization packing descriptor (int8 this phase; int4 planned). */
export type QuantPacking = {
  scheme: "int8-grouped";
  /** Group size G along K (weights sharing one scale). Power of two. */
  groupSize: number;
  /** Element type of the per-group scales companion buffer (f16). */
  scalesDtype: DType;
};

/** Named format shorthands the loader/user surface accepts (weightFormat). */
export type WeightFormatName = "int8-64" | "int8-128";

/** Resolve a named weight format to a StorageFormat over an elementType. */
export function resolveWeightFormat(
  name: WeightFormatName,
  elementType: DType = "f16",
): StorageFormat {
  const groupSize = name === "int8-64" ? 64 : 128;
  return {
    elementType,
    packing: { scheme: "int8-grouped", groupSize, scalesDtype: "f16" },
  };
}

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

/**
 * Attention score/mask modifiers (task #64 — FlexAttention-class seams).
 *
 * Modifiers are DATA (tagged specs), not closures: the kernel builder maps
 * each kind to a tile-IR expression at the declared seam points
 * (`attn_score` post-QK-dot pre-softmax; `attn_mask` replacing the active
 * predicate). Data-form keeps them serializable (remote wire), hashable
 * (CSE structural keys + plan fingerprints hash payload content), and
 * interpretable by the CPU decomposed path for cross-path parity.
 *
 * Structural identity (which template) comes from the kinds; numeric params
 * (cap, window) flow as uniforms — see attnModifierKey() in
 * attention-kernel.ts, the single source for the key.
 */
export type AttnScoreModSpec = {
  /** Logit soft-cap: score' = cap * tanh(score / cap) (Gemma-2). */
  kind: "softcap";
  cap: number;
};

export type AttnMaskModSpec =
  | { kind: "causal" }
  | {
      /** Recency bound: active iff kv > q - window (compose with "causal"
       *  for Gemma-2's local attention: q - window < kv <= q). */
      kind: "slidingWindow";
      window: number;
    };

export type AttnModifierSpec = {
  scoreMod?: AttnScoreModSpec;
  /** Composition = AND of all mask mods (and the bounds check). */
  maskMods?: AttnMaskModSpec[];
};

export type FusedAttentionConfig = {
  batchSize: number;
  numHeads: number;
  seqLen: number;
  headDim: number;
  scale: number;
  isCausal: boolean;
  /** Optional score/mask modifiers. Omit the field entirely (do not assign
   *  undefined) when unused so existing payload hashes/keys are byte-stable. */
  modifier?: AttnModifierSpec;
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

/**
 * The GENERIC fused-optimizer step config (derived-optimizer-realizer R5b — the
 * de-naming). Carries NO optimizer name: the executor/backend key every decision
 * on STRUCTURE (`spec` selects the realizer; `stateSlots`/`scalarInputs` drive
 * the in-place dst set, multi-output extras, and the shared-binding pack key).
 * Adam is one client — `spec:"adamw"`, `stateSlots:["m","v"]`,
 * `scalarInputs:["bc","lr"]` — reproducing the prior fused Adam BIT-FOR-BIT.
 */
export type OptStepConfig = {
  /** OPT_STEP_SPECS key selecting the program + role schema the backend folds
   *  into WGSL (`adamw`/`lion`/`sgd`/`sgd_momentum`). Opaque data to the executor
   *  (fingerprint / payload-hash only). */
  spec: string;
  /** Optimizer state slot binding names, in spec order (Adam: `["m","v"]`; SGD+
   *  momentum: `["v"]`; SGD: `[]`). The in-place dst inputs are `[1..1+nState]`,
   *  the multi-output extras are the state outputs, and the rw binding set is
   *  `[param, ...states]` — all derived from THIS, never the op's name. */
  stateSlots: readonly string[];
  /** Shared scalar-DATA input binding names, in spec order (Adam: `["bc","lr"]`).
   *  Each is ONE shared read buffer at input slot `2+nState+i` — the horizontal-
   *  pack key's shared indices. Host-computed live scalars. */
  scalarInputs: readonly string[];
  /** Static f32 hyper uniform VALUES keyed by uniform name (spec.f32Uniforms).
   *  Adam: beta1/beta2/ln_beta1/ln_beta2/eps/weight_decay. */
  hypers: Readonly<Record<string, number>>;
  /** L2 (false) vs decoupled (true) weight-decay branch (the runtime uniform). */
  decoupledWd: boolean;
  emitF16?: boolean;
  /** Gradient inverse scale factor for fused unscale (default 1.0 = no unscaling). */
  invScale?: number;
  /** Shared atomic inf-flag buffer for fused unscale inf detection. */
  infFlagBuffer?: unknown;
};
// inc-2a: per-step scalars (step_size, lr*wd) were RETIRED from the config. The
// bias-corrected step size and lr*wd are derived IN-KERNEL from the persistent
// `bc`/`lr` scalar-DATA inputs — the config is fully static, killing the
// per-step volatile-uniform repack class.

/** One parameter's inputs for a batched fused optimizer step. */
export type OptStepBatchItem = {
  grad: BackendTensor;
  param: BackendTensor;
  /** Optimizer state operands, in spec order (Adam: [m, v]; SGD+mom: [v]; SGD: []). */
  states: BackendTensor[];
  /** Shared scalar-DATA inputs, in spec order (Adam: [bc, lr]). 1-element (lr) or
   *  small ([2] bc) persistent buffers shared across the group's params. */
  scalars: BackendTensor[];
  config: OptStepConfig;
};

/** Output tensors for one batched optimizer item (param + updated states). */
export type OptStepBatchResult = {
  param: BackendTensor;
  states: BackendTensor[];
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
  /** Fused optimizer step (Adam/Lion/SGD, spec-selected). `states` and `scalars`
   *  are variadic per the config's spec; returns updated param + states. */
  optStep?(
    grad: BackendTensor,
    param: BackendTensor,
    states: BackendTensor[],
    scalars: BackendTensor[],
    config: OptStepConfig,
  ):
    | { param: BackendTensor; states: BackendTensor[] }
    | Promise<{ param: BackendTensor; states: BackendTensor[] }>;
  /**
   * Batched optimizer step: process N optStep calls in one backend invocation.
   * Backends are free to fuse same-element-count items into packed kernel
   * dispatches. Returns results in input order.
   */
  optStepBatch?(items: OptStepBatchItem[]): OptStepBatchResult[];
  /** Fused unscale + inf-check + zero-mask for GradScaler.
   *  scaler-as-tensor: `scale` is a persistent 1-element f32 tensor (the
   *  GradScaler's LiveScalar buffer) read LIVE from a storage binding, not a
   *  frozen uniform number (invScale = 1/scale reciprocated in-kernel). */
  unscaleGrad?(
    grad: BackendTensor,
    scale: BackendTensor,
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
  /** Lazy device top-K over one logits row: [.., V] → packed [1,2,k] (row 0 =
   *  values desc, row 1 = token ids as f32). On-device sampling prefilter. */
  deviceTopK?(logits: BackendTensor, config: { k: number }): BackendTensor;
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
  /** [inc-3 ring] Snapshot the shared inf flag into a pool-excluded staging
   *  buffer and re-zero it, in queue order (per-step report isolation under
   *  runahead). Null when the fused unscale path never initialized. */
  snapshotInfFlag?(): unknown | null;
  /** [inc-3 ring] Read (mapAsync self-sync — no shared fence) + destroy a
   *  snapshot from snapshotInfFlag. */
  readInfSnapshot?(snapshot: unknown): Promise<number>;
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
