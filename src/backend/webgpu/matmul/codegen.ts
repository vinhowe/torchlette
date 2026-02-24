/**
 * WGSL code generation for tiled matrix multiplication kernels.
 *
 * Generates optimized shaders with:
 * - Shared memory tiling
 * - Register tiling (multiple outputs per thread)
 * - Transpose variants (NN, NT, TN, TT)
 * - Coalesced memory access patterns
 * - Epilogue fusion using unified op registry
 */

import {
  type DType,
  getWorkgroupSize,
  type MatmulKernelConfig,
  type TransposeMode,
} from "./types";
import { getExpr, isUnaryOp, isBinaryOp } from "../ops/registry";
import { dtypeToWgsl } from "../shape-utils";

/**
 * Epilogue operation to fuse into matmul output.
 * Supports arbitrary chains of elementwise operations from the unified op registry.
 */
type EpilogueOp =
  | { kind: "none" }
  | { kind: "bias"; inputIndex: number } // Special case: broadcast add for bias
  | { kind: "unary"; op: string } // Any unary op from registry (relu, gelu, silu, sigmoid, tanh, etc.)
  | { kind: "binary"; op: string; inputIndex: number } // Any binary op from registry (add, mul, sub, div, etc.)
  | { kind: "cast"; toDtype: DType }
  // Backward compatibility aliases
  | { kind: "relu" }
  | { kind: "gelu" }
  | { kind: "silu" }
  | { kind: "add"; inputIndex: number }
  | { kind: "mul"; inputIndex: number };

export type EpilogueConfig = {
  ops: EpilogueOp[];
  additionalInputCount: number;
  outputDtype: DType;
};

/**
 * Options for shader generation.
 */
export type CodegenOptions = {
  config: MatmulKernelConfig;
  transposeMode: TransposeMode;
  dtype: DType;
  /** dtype for input B (defaults to dtype) */
  dtypeB?: DType;
  epilogue?: EpilogueConfig;
  /** Include batch dimension support */
  batched?: boolean;
  /** Input A buffer is this wider dtype; cast to compute dtype during tile load */
  inputCastA?: DType;
  /** Input B buffer is this wider dtype; cast to compute dtype during tile load */
  inputCastB?: DType;
  /** K-split factor: split K-reduction across this many workgroups (Z dim) */
  kSplit?: number;
};


/**
 * Generate the array element type (always f32 for accumulator).
 */
function accumulatorType(): string {
  return "f32";
}

/**
 * Generate WGSL code for loading A matrix element with transpose handling.
 */
function genLoadA(
  transposeMode: TransposeMode,
  row: string,
  col: string,
  lda: string,
  offset: string = "0u",
): string {
  // For A: transA means we swap row/col indexing
  const isTransposed = transposeMode === "TN" || transposeMode === "TT";
  if (isTransposed) {
    // A is stored as A^T, so A[row, col] = storage[col * lda + row]
    return `a[${offset} + ${col} * ${lda} + ${row}]`;
  }
  // A is stored normally: A[row, col] = storage[row * lda + col]
  return `a[${offset} + ${row} * ${lda} + ${col}]`;
}

/**
 * Generate WGSL code for loading B matrix element with transpose handling.
 */
function genLoadB(
  transposeMode: TransposeMode,
  row: string,
  col: string,
  ldb: string,
  offset: string = "0u",
): string {
  // For B: transB means we swap row/col indexing
  const isTransposed = transposeMode === "NT" || transposeMode === "TT";
  if (isTransposed) {
    // B is stored as B^T, so B[row, col] = storage[col * ldb + row]
    return `b[${offset} + ${col} * ${ldb} + ${row}]`;
  }
  // B is stored normally: B[row, col] = storage[row * ldb + col]
  return `b[${offset} + ${row} * ${ldb} + ${col}]`;
}

/**
 * Generate epilogue code from the epilogue config.
 * Uses the unified op registry for expression generation.
 */
function genEpilogueCode(
  epilogue: EpilogueConfig | undefined,
  accVar: string,
  outIdxExpr: string,
  outputDtype?: DType,
  colIdxExpr?: string,
): string {
  if (!epilogue || epilogue.ops.length === 0) {
    // When output buffer is f16 but accumulator is f32, cast before writing
    if (outputDtype === "f16") {
      return `out[${outIdxExpr}] = f16(${accVar});`;
    }
    return `out[${outIdxExpr}] = ${accVar};`;
  }

  const lines: string[] = [];
  let currentExpr = accVar;
  let tempVarCounter = 0;

  for (const op of epilogue.ops) {
    switch (op.kind) {
      case "none":
        break;

      case "bias": {
        // Bias is indexed by column only, not by flattened output index.
        // Use colIdxExpr (out_col) which is always provided by the caller.
        // Fallback uses params.n (logical column count) — NOT arrayLength(),
        // which returns the physical buffer size and is wrong for pool-rounded buffers.
        const biasIdx = colIdxExpr ?? `${outIdxExpr} % n`;
        lines.push(
          `let biasVal${op.inputIndex} = epilogue_in${op.inputIndex}[${biasIdx}];`,
        );
        currentExpr = getExpr("add", [currentExpr, `biasVal${op.inputIndex}`]);
        break;
      }

      // General unary op from registry
      case "unary": {
        // Complex ops that reference input multiple times need temp variable
        const needsTemp = ["gelu", "gelu_tanh", "gelu_erf", "silu", "softplus"].includes(op.op);
        if (needsTemp) {
          const tempVar = `t${tempVarCounter++}`;
          lines.push(`let ${tempVar} = ${currentExpr};`);
          currentExpr = getExpr(op.op, [tempVar]);
        } else {
          currentExpr = getExpr(op.op, [currentExpr]);
        }
        break;
      }

      // General binary op from registry
      case "binary": {
        lines.push(
          `let binVal${op.inputIndex} = epilogue_in${op.inputIndex}[${outIdxExpr}];`,
        );
        currentExpr = getExpr(op.op, [currentExpr, `binVal${op.inputIndex}`]);
        break;
      }

      // Backward compatibility: specific unary ops
      case "relu":
        currentExpr = getExpr("relu", [currentExpr]);
        break;
      case "gelu": {
        const tempVar = `t${tempVarCounter++}`;
        lines.push(`let ${tempVar} = ${currentExpr};`);
        currentExpr = getExpr("gelu", [tempVar]);
        break;
      }
      case "silu": {
        const tempVar = `t${tempVarCounter++}`;
        lines.push(`let ${tempVar} = ${currentExpr};`);
        currentExpr = getExpr("silu", [tempVar]);
        break;
      }

      // Backward compatibility: specific binary ops
      case "add":
        lines.push(
          `let addVal${op.inputIndex} = epilogue_in${op.inputIndex}[${outIdxExpr}];`,
        );
        currentExpr = getExpr("add", [currentExpr, `addVal${op.inputIndex}`]);
        break;
      case "mul":
        lines.push(
          `let mulVal${op.inputIndex} = epilogue_in${op.inputIndex}[${outIdxExpr}];`,
        );
        currentExpr = getExpr("mul", [currentExpr, `mulVal${op.inputIndex}`]);
        break;

      case "cast":
        if (op.toDtype === "f16") {
          currentExpr = getExpr("cast_f16", [currentExpr]);
        } else if (op.toDtype === "f32") {
          currentExpr = getExpr("cast_f32", [currentExpr]);
        }
        break;
    }
  }

  // Auto-cast to f16 if output dtype is f16 and no explicit cast
  if (
    epilogue.outputDtype === "f16" &&
    !epilogue.ops.some((op) => op.kind === "cast")
  ) {
    currentExpr = getExpr("cast_f16", [currentExpr]);
  }

  lines.push(`out[${outIdxExpr}] = ${currentExpr};`);
  return lines.join("\n      ");
}

/**
 * Generate additional epilogue input bindings.
 */
function genEpilogueBindings(
  epilogue: EpilogueConfig | undefined,
  startBinding: number,
): string {
  if (!epilogue || epilogue.additionalInputCount === 0) {
    return "";
  }

  const bindings: string[] = [];
  for (let i = 0; i < epilogue.additionalInputCount; i++) {
    bindings.push(
      `@group(0) @binding(${startBinding + i}) var<storage, read> epilogue_in${i}: array<f32>;`,
    );
  }
  return bindings.join("\n");
}

/**
 * Generate the complete tiled matmul shader.
 */
export function generateTiledMatmulShader(options: CodegenOptions): string {
  const { config, transposeMode, dtype, dtypeB: dtypeBOpt, epilogue, batched, inputCastA, inputCastB, kSplit } = options;
  const { tileM, tileN, tileK, threadTileM, threadTileN } = config;

  const wgSize = getWorkgroupSize(config);
  const wgSizeX = wgSize.x;
  const wgSizeY = wgSize.y;
  const totalThreads = wgSizeX * wgSizeY;

  const dtypeB = dtypeBOpt ?? dtype;
  // When inputCastA/B is set, the buffer contains the wider dtype (e.g. f32)
  // but the matmul was supposed to compute with the narrower dtype (e.g. f16).
  // Since the accumulator is always f32, we just declare the binding as f32
  // and load directly — the result is numerically equal or better than the
  // roundtrip through f16. This eliminates the separate cast dispatch.
  const wgslDtypeA = inputCastA ? "f32" : dtypeToWgsl(dtype);
  const wgslDtypeB = inputCastB ? "f32" : dtypeToWgsl(dtypeB);
  const accType = accumulatorType();

  // Number of elements each thread loads for cooperative tile loading
  const elementsPerThreadA = Math.ceil((tileM * tileK) / totalThreads);
  const elementsPerThreadB = Math.ceil((tileK * tileN) / totalThreads);

  // Epilogue bindings start after A, B, out, params
  const epilogueBindingStart = 4;
  const epilogueBindings = genEpilogueBindings(epilogue, epilogueBindingStart);

  // Determine if we need f16 support
  // K-split partials are always f32 (accumulated in f32 registers, no epilogue/cast)
  const outputDtype = kSplit ? "f32" as DType : (epilogue?.outputDtype ?? (dtype === "f32" || dtypeB === "f32" ? "f32" : dtype));
  // Need f16 if any actual binding uses f16 or output is f16.
  // When inputCast is active, the overridden binding is f32, so don't count it.
  const needsF16 = wgslDtypeA === "f16" || wgslDtypeB === "f16" || outputDtype === "f16";

  // Generate the shader
  const shader = `
// Tiled Matrix Multiplication Kernel
// Config: TILE_M=${tileM}, TILE_N=${tileN}, TILE_K=${tileK}
// Thread tile: ${threadTileM}x${threadTileN}, Workgroup: ${wgSizeX}x${wgSizeY}
// Transpose mode: ${transposeMode}
// Dtype: ${dtype}, Output: ${outputDtype}
${needsF16 ? "\nenable f16;\n" : ""}
struct Params {
  m: u32,
  n: u32,
  k: u32,
  lda: u32,
  ldb: u32,
  ldc: u32,
  alpha: f32,
  batchSize: u32,
  batchStrideA: u32,
  batchStrideB: u32,
  batchStrideC: u32,
}

@group(0) @binding(0) var<storage, read> a: array<${wgslDtypeA}>;
@group(0) @binding(1) var<storage, read> b: array<${wgslDtypeB}>;
@group(0) @binding(2) var<storage, read_write> out: array<${dtypeToWgsl(epilogue?.outputDtype ?? outputDtype)}>;
@group(0) @binding(3) var<uniform> params: Params;
${epilogueBindings}

// Shared memory tiles
var<workgroup> tileA: array<${accType}, ${tileM * tileK}>;
var<workgroup> tileB: array<${accType}, ${tileK * tileN}>;

const TILE_M: u32 = ${tileM}u;
const TILE_N: u32 = ${tileN}u;
const TILE_K: u32 = ${tileK}u;
const THREAD_TILE_M: u32 = ${threadTileM}u;
const THREAD_TILE_N: u32 = ${threadTileN}u;
const WG_SIZE_X: u32 = ${wgSizeX}u;
const WG_SIZE_Y: u32 = ${wgSizeY}u;
const ELEMS_PER_THREAD_A: u32 = ${elementsPerThreadA}u;
const ELEMS_PER_THREAD_B: u32 = ${elementsPerThreadB}u;

@compute @workgroup_size(${wgSizeX}, ${wgSizeY})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_idx: u32,
) {
  let m = params.m;
  let n = params.n;
  let k = params.k;
  let lda = params.lda;
  let ldb = params.ldb;
  let ldc = params.ldc;
  let alpha = params.alpha;

  // Batch / K-split offset
  ${kSplit ? `// K-split mode: wg_id.z is the K-split index
  let split_idx = wg_id.z;
  let batch_offset_a = 0u;
  let batch_offset_b = 0u;` : `// Using stride-based indexing for proper broadcast support
  // Stride of 0 means broadcast (same data for all batches)
  ${batched ? "let batch_idx = wg_id.z;" : "let batch_idx = 0u;"}
  let batch_offset_a = batch_idx * params.batchStrideA;
  let batch_offset_b = batch_idx * params.batchStrideB;
  let batch_offset_c = batch_idx * params.batchStrideC;`}

  // Workgroup tile position in output matrix
  let wg_row = wg_id.y * TILE_M;
  let wg_col = wg_id.x * TILE_N;

  // Thread position within workgroup
  let thread_row = local_id.y;
  let thread_col = local_id.x;

  // Register accumulator array (THREAD_TILE_M x THREAD_TILE_N)
  var acc: array<${accType}, ${threadTileM * threadTileN}>;
  for (var i = 0u; i < ${threadTileM * threadTileN}u; i = i + 1u) {
    acc[i] = 0.0;
  }

  // K-loop bounds
  ${kSplit ? `// K-split: each Z-slice handles a chunk of K
  let k_per_split = (k + ${kSplit}u - 1u) / ${kSplit}u;
  let k_start = split_idx * k_per_split;
  let k_end = min(k_start + k_per_split, k);
  let num_k_tiles = (k_end - k_start + TILE_K - 1u) / TILE_K;` : `let num_k_tiles = (k + TILE_K - 1u) / TILE_K;
  let k_start = 0u;
  let k_end = k;`}

  // Loop over K tiles
  for (var k_tile = 0u; k_tile < num_k_tiles; k_tile = k_tile + 1u) {
    let k_offset = k_start + k_tile * TILE_K;

    // Cooperative load of A tile into shared memory
    for (var i = 0u; i < ELEMS_PER_THREAD_A; i = i + 1u) {
      let flat_idx = local_idx * ELEMS_PER_THREAD_A + i;
      if (flat_idx < TILE_M * TILE_K) {
        let tile_row = flat_idx / TILE_K;
        let tile_col = flat_idx % TILE_K;
        let global_row = wg_row + tile_row;
        let global_col = k_offset + tile_col;
        if (global_row < m && global_col < k_end) {
          tileA[flat_idx] = ${accType}(${genLoadA(transposeMode, "global_row", "global_col", "lda", "batch_offset_a")});
        } else {
          tileA[flat_idx] = 0.0;
        }
      }
    }

    // Cooperative load of B tile into shared memory
    for (var i = 0u; i < ELEMS_PER_THREAD_B; i = i + 1u) {
      let flat_idx = local_idx * ELEMS_PER_THREAD_B + i;
      if (flat_idx < TILE_K * TILE_N) {
        let tile_row = flat_idx / TILE_N;
        let tile_col = flat_idx % TILE_N;
        let global_row = k_offset + tile_row;
        let global_col = wg_col + tile_col;
        if (global_row < k_end && global_col < n) {
          tileB[flat_idx] = ${accType}(${genLoadB(transposeMode, "global_row", "global_col", "ldb", "batch_offset_b")});
        } else {
          tileB[flat_idx] = 0.0;
        }
      }
    }

    workgroupBarrier();

    // Compute partial products from shared memory
    // Each thread computes a THREAD_TILE_M x THREAD_TILE_N block
    for (var kk = 0u; kk < TILE_K; kk = kk + 1u) {
      // Load thread's column of A values from shared memory
      var a_vals: array<${accType}, ${threadTileM}>;
      for (var tm = 0u; tm < THREAD_TILE_M; tm = tm + 1u) {
        let a_row = thread_row * THREAD_TILE_M + tm;
        a_vals[tm] = tileA[a_row * TILE_K + kk];
      }

      // Load thread's row of B values from shared memory
      var b_vals: array<${accType}, ${threadTileN}>;
      for (var tn = 0u; tn < THREAD_TILE_N; tn = tn + 1u) {
        let b_col = thread_col * THREAD_TILE_N + tn;
        b_vals[tn] = tileB[kk * TILE_N + b_col];
      }

      // Outer product accumulation
      for (var tm = 0u; tm < THREAD_TILE_M; tm = tm + 1u) {
        for (var tn = 0u; tn < THREAD_TILE_N; tn = tn + 1u) {
          acc[tm * THREAD_TILE_N + tn] = acc[tm * THREAD_TILE_N + tn] + a_vals[tm] * b_vals[tn];
        }
      }
    }

    workgroupBarrier();
  }

  // Write results to global memory with bounds checking
  for (var tm = 0u; tm < THREAD_TILE_M; tm = tm + 1u) {
    for (var tn = 0u; tn < THREAD_TILE_N; tn = tn + 1u) {
      let out_row = wg_row + thread_row * THREAD_TILE_M + tm;
      let out_col = wg_col + thread_col * THREAD_TILE_N + tn;
      if (out_row < m && out_col < n) {
        ${kSplit ? `// K-split: write partial sum to temp[split_idx * m * n + ...]
        let out_idx = split_idx * m * n + out_row * ldc + out_col;
        out[out_idx] = acc[tm * THREAD_TILE_N + tn];` : `let out_idx = batch_offset_c + out_row * ldc + out_col;
        let result = acc[tm * THREAD_TILE_N + tn] * alpha;
        ${genEpilogueCode(epilogue, "result", "out_idx", outputDtype, "out_col")}`}
      }
    }
  }
}
`;

  return shader;
}

/**
 * Generate a cache key for a shader configuration.
 */
export function getShaderCacheKey(options: CodegenOptions): string {
  const { config, transposeMode, dtype, dtypeB, epilogue, batched, inputCastA, inputCastB, kSplit } = options;
  const epilogueKey = epilogue
    ? `${epilogue.ops.map((op) => op.kind).join(",")}_${epilogue.outputDtype}`
    : "none";
  return [
    `tiled`,
    `${config.tileM}x${config.tileN}x${config.tileK}`,
    `t${config.threadTileM}x${config.threadTileN}`,
    `v${config.vectorWidth}`,
    config.useSubgroups ? "sg" : "nosg",
    transposeMode,
    dtype,
    dtypeB && dtypeB !== dtype ? `dtypeB_${dtypeB}` : "",
    batched ? "batch" : "nobatch",
    epilogueKey,
    inputCastA ? `castA_${inputCastA}` : "",
    inputCastB ? `castB_${inputCastB}` : "",
    kSplit ? `ksplit${kSplit}` : "",
  ].filter(Boolean).join("_");
}

/**
 * Generate a simple reduction shader that sums K-split partial results.
 * Reads partials[P * M * N] and writes out[M * N] = sum(partials[p * M*N + i] for p in 0..P) * alpha.
 */
export function generateKSplitReductionShader(kSplitCount: number, outputDtype: DType): string {
  const outType = dtypeToWgsl(outputDtype);
  const needsF16 = outputDtype === "f16";
  return `
// K-split reduction: sum ${kSplitCount} partial results
${needsF16 ? "enable f16;\n" : ""}
struct Params {
  totalElements: u32,
  alpha: f32,
}

@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<${outType}>;
@group(0) @binding(2) var<uniform> params: Params;

const K_SPLIT: u32 = ${kSplitCount}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.totalElements) { return; }
  var sum: f32 = 0.0;
  for (var p = 0u; p < K_SPLIT; p = p + 1u) {
    sum = sum + partials[p * params.totalElements + idx];
  }
  out[idx] = ${outputDtype === "f16" ? `f16(sum * params.alpha)` : `sum * params.alpha`};
}
`;
}

/**
 * Generate a cache key for a K-split reduction shader.
 */
export function getKSplitReductionCacheKey(kSplitCount: number, outputDtype: DType): string {
  return `ksplit_reduce_${kSplitCount}_${outputDtype}`;
}
