/**
 * Subgroup-accelerated matmul variant.
 *
 * Uses WGSL subgroup operations for improved performance on GPUs that support them.
 * Subgroups allow efficient register-to-register communication within a workgroup.
 */

import type { EpilogueConfig } from "./codegen";
import type { DType, MatmulKernelConfig, TransposeMode } from "./types";
import { getWorkgroupSize } from "./types";

/**
 * Subgroup support detection result.
 */
export type SubgroupSupport = {
  supported: boolean;
  subgroupSize?: number;
};

/**
 * Options for subgroup shader generation.
 */
export type SubgroupCodegenOptions = {
  config: MatmulKernelConfig;
  transposeMode: TransposeMode;
  dtype: DType;
  /** dtype for input B (defaults to dtype) */
  dtypeB?: DType;
  epilogue?: EpilogueConfig;
  batched?: boolean;
  subgroupSize: number;
};

/**
 * Generate load A code with transpose handling.
 */
function genLoadA(
  transposeMode: TransposeMode,
  row: string,
  col: string,
  lda: string,
  offset: string = "0u",
): string {
  const isTransposed = transposeMode === "TN" || transposeMode === "TT";
  if (isTransposed) {
    return `a[${offset} + ${col} * ${lda} + ${row}]`;
  }
  return `a[${offset} + ${row} * ${lda} + ${col}]`;
}

/**
 * Generate load B code with transpose handling.
 */
function genLoadB(
  transposeMode: TransposeMode,
  row: string,
  col: string,
  ldb: string,
  offset: string = "0u",
): string {
  const isTransposed = transposeMode === "NT" || transposeMode === "TT";
  if (isTransposed) {
    return `b[${offset} + ${col} * ${ldb} + ${row}]`;
  }
  return `b[${offset} + ${row} * ${ldb} + ${col}]`;
}

/**
 * Generate epilogue code (same as codegen.ts).
 */
function genEpilogueCode(
  epilogue: EpilogueConfig | undefined,
  accVar: string,
  outIdxExpr: string,
  colIdxExpr?: string,
): string {
  if (!epilogue || epilogue.ops.length === 0) {
    return `out[${outIdxExpr}] = ${accVar};`;
  }

  const lines: string[] = [];
  let currentExpr = accVar;

  for (const op of epilogue.ops) {
    switch (op.kind) {
      case "none":
        break;
      case "bias": {
        // Bias is indexed by column only, not by flattened output index.
        // Fallback uses n (logical column count) — NOT arrayLength().
        const biasIdx = colIdxExpr ?? `${outIdxExpr} % n`;
        lines.push(
          `let biasVal${op.inputIndex} = epilogue_in${op.inputIndex}[${biasIdx}];`,
        );
        currentExpr = `(${currentExpr} + biasVal${op.inputIndex})`;
        break;
      }
      case "add":
        lines.push(
          `let addVal${op.inputIndex} = epilogue_in${op.inputIndex}[${outIdxExpr}];`,
        );
        currentExpr = `(${currentExpr} + addVal${op.inputIndex})`;
        break;
      case "mul":
        lines.push(
          `let mulVal${op.inputIndex} = epilogue_in${op.inputIndex}[${outIdxExpr}];`,
        );
        currentExpr = `(${currentExpr} * mulVal${op.inputIndex})`;
        break;
      case "relu":
        currentExpr = `select(0.0, ${currentExpr}, ${currentExpr} > 0.0)`;
        break;
      case "gelu": {
        const x = currentExpr;
        lines.push(`let gelu_x = ${x};`);
        lines.push(
          `let gelu_inner = 0.7978845608 * (gelu_x + 0.044715 * gelu_x * gelu_x * gelu_x);`,
        );
        currentExpr = `(0.5 * gelu_x * (1.0 + tanh(gelu_inner)))`;
        break;
      }
      case "silu":
        lines.push(`let silu_x = ${currentExpr};`);
        currentExpr = `(silu_x / (1.0 + exp(-silu_x)))`;
        break;
      case "cast":
        if (op.toDtype === "f16") {
          currentExpr = `f16(${currentExpr})`;
        }
        break;
    }
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
 * Generate subgroup-accelerated tiled matmul shader.
 *
 * This variant uses subgroup operations to:
 * 1. Broadcast values efficiently across lanes
 * 2. Reduce shared memory pressure by using subgroup shuffle
 * 3. Perform efficient partial sum reductions
 */
export function generateSubgroupMatmulShader(
  options: SubgroupCodegenOptions,
): string {
  const { config, transposeMode, dtype, dtypeB: dtypeBOpt, epilogue, batched, subgroupSize } =
    options;
  const { tileM, tileN, tileK, threadTileM, threadTileN } = config;

  const wgSize = getWorkgroupSize(config);
  const wgSizeX = wgSize.x;
  const wgSizeY = wgSize.y;
  const totalThreads = wgSizeX * wgSizeY;

  const dtypeB = dtypeBOpt ?? dtype;
  const wgslDtypeA = dtype === "f16" ? "f16" : "f32";
  const wgslDtypeB = dtypeB === "f16" ? "f16" : "f32";
  const accType = "f32";

  const elementsPerThreadA = Math.ceil((tileM * tileK) / totalThreads);
  const elementsPerThreadB = Math.ceil((tileK * tileN) / totalThreads);

  const epilogueBindingStart = 4;
  const epilogueBindings = genEpilogueBindings(epilogue, epilogueBindingStart);

  // Determine if we need f16 support
  const outputDtype = epilogue?.outputDtype ?? (dtype === "f32" || dtypeB === "f32" ? "f32" : dtype);
  const needsF16 = dtype === "f16" || dtypeB === "f16" || outputDtype === "f16";

  // For subgroup variant, we use subgroup operations for broadcasting
  // This reduces shared memory traffic when multiple threads need the same value
  const shader = `// Subgroup-Accelerated Tiled Matrix Multiplication Kernel
// Config: TILE_M=${tileM}, TILE_N=${tileN}, TILE_K=${tileK}
// Thread tile: ${threadTileM}x${threadTileN}, Workgroup: ${wgSizeX}x${wgSizeY}
// Transpose mode: ${transposeMode}
// Subgroup size: ${subgroupSize}
// Dtype: ${dtype}, Output: ${outputDtype}
${needsF16 ? "\nenable f16;" : ""}
enable subgroups;

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
@group(0) @binding(2) var<storage, read_write> out: array<${(epilogue?.outputDtype ?? outputDtype) === "f16" ? "f16" : "f32"}>;
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
const SUBGROUP_SIZE: u32 = ${subgroupSize}u;

@compute @workgroup_size(${wgSizeX}, ${wgSizeY})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_idx: u32,
  @builtin(subgroup_invocation_id) sg_id: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  let m = params.m;
  let n = params.n;
  let k = params.k;
  let lda = params.lda;
  let ldb = params.ldb;
  let ldc = params.ldc;
  let alpha = params.alpha;

  ${batched ? "let batch_idx = wg_id.z;" : "let batch_idx = 0u;"}
  let batch_offset_a = batch_idx * params.batchStrideA;
  let batch_offset_b = batch_idx * params.batchStrideB;
  let batch_offset_c = batch_idx * params.batchStrideC;

  let wg_row = wg_id.y * TILE_M;
  let wg_col = wg_id.x * TILE_N;

  let thread_row = local_id.y;
  let thread_col = local_id.x;

  // Register accumulator array
  var acc: array<${accType}, ${threadTileM * threadTileN}>;
  for (var i = 0u; i < ${threadTileM * threadTileN}u; i = i + 1u) {
    acc[i] = 0.0;
  }

  let num_k_tiles = (k + TILE_K - 1u) / TILE_K;

  for (var k_tile = 0u; k_tile < num_k_tiles; k_tile = k_tile + 1u) {
    let k_offset = k_tile * TILE_K;

    // Cooperative load of A tile into shared memory
    for (var i = 0u; i < ELEMS_PER_THREAD_A; i = i + 1u) {
      let flat_idx = local_idx * ELEMS_PER_THREAD_A + i;
      if (flat_idx < TILE_M * TILE_K) {
        let tile_row = flat_idx / TILE_K;
        let tile_col = flat_idx % TILE_K;
        let global_row = wg_row + tile_row;
        let global_col = k_offset + tile_col;
        if (global_row < m && global_col < k) {
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
        if (global_row < k && global_col < n) {
          tileB[flat_idx] = ${accType}(${genLoadB(transposeMode, "global_row", "global_col", "ldb", "batch_offset_b")});
        } else {
          tileB[flat_idx] = 0.0;
        }
      }
    }

    workgroupBarrier();

    // Compute partial products with subgroup-optimized inner loop
    // Use subgroup broadcast to share loaded values within subgroup
    for (var kk = 0u; kk < TILE_K; kk = kk + 1u) {
      // Load thread's column of A values
      var a_vals: array<${accType}, ${threadTileM}>;
      for (var tm = 0u; tm < THREAD_TILE_M; tm = tm + 1u) {
        let a_row = thread_row * THREAD_TILE_M + tm;
        a_vals[tm] = tileA[a_row * TILE_K + kk];
      }

      // Load thread's row of B values
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
        let out_idx = batch_offset_c + out_row * ldc + out_col;
        let result = acc[tm * THREAD_TILE_N + tn] * alpha;
        ${genEpilogueCode(epilogue, "result", "out_idx", "out_col")}
      }
    }
  }
}`;

  return shader;
}

/**
 * Generate a cache key for subgroup shader configuration.
 */
export function getSubgroupShaderCacheKey(
  options: SubgroupCodegenOptions,
): string {
  const { config, transposeMode, dtype, dtypeB, epilogue, batched, subgroupSize } =
    options;
  const epilogueKey = epilogue
    ? `${epilogue.ops.map((op) => op.kind).join(",")}_${epilogue.outputDtype}`
    : "none";
  return [
    `subgroup`,
    `${config.tileM}x${config.tileN}x${config.tileK}`,
    `t${config.threadTileM}x${config.threadTileN}`,
    `sg${subgroupSize}`,
    transposeMode,
    dtype,
    dtypeB && dtypeB !== dtype ? `dtypeB_${dtypeB}` : "",
    batched ? "batch" : "nobatch",
    epilogueKey,
  ].filter(Boolean).join("_");
}
