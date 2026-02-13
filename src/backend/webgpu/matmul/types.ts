/**
 * Matmul kernel configuration and types for WebGPU tiled matrix multiplication.
 */

export type DType = "f16" | "f32";

/**
 * Kernel configuration for tiled matrix multiplication.
 * These parameters control tiling, threading, and optimization strategies.
 */
export type MatmulKernelConfig = {
  /** Output tile height per workgroup (e.g., 32, 64, 128) */
  tileM: number;
  /** Output tile width per workgroup (e.g., 32, 64, 128) */
  tileN: number;
  /** K-dimension tile for shared memory (e.g., 8, 16, 32) */
  tileK: number;
  /** Elements per thread in M dimension (e.g., 4, 8) */
  threadTileM: number;
  /** Elements per thread in N dimension (e.g., 4, 8) */
  threadTileN: number;
  /** Whether to use subgroup operations (requires device support) */
  useSubgroups: boolean;
  /** Vector width for loads (1, 2, or 4) */
  vectorWidth: 1 | 2 | 4;
};

/**
 * Transpose flags for matmul operands.
 * - NN: A @ B (no transpose)
 * - NT: A @ B^T
 * - TN: A^T @ B
 * - TT: A^T @ B^T
 */
export type TransposeMode = "NN" | "NT" | "TN" | "TT";

/**
 * Parameters passed to the matmul shader as uniforms.
 */
export type MatmulParams = {
  m: number;
  n: number;
  k: number;
  /** Transpose mode encoded as integer: 0=NN, 1=NT, 2=TN, 3=TT */
  transposeMode: number;
  /** Batch size (total number of batch elements after broadcasting) */
  batchSize: number;
  /** Leading dimension of A (stride between rows) */
  lda: number;
  /** Leading dimension of B (stride between rows) */
  ldb: number;
  /** Leading dimension of output (stride between rows) */
  ldc: number;
  /** Alpha scaling factor (output = alpha * A @ B) */
  alpha: number;
};

/**
 * Shape class for autotuning cache keys.
 * Different shape classes may have different optimal configurations.
 */
export type ShapeClass =
  | "square_small" // M,N,K < 512
  | "square_medium" // 512 <= M,N,K < 2048
  | "square_large" // M,N,K >= 2048
  | "tall_skinny" // M >> K or M >> N
  | "short_wide" // N >> K or N >> M
  | "small_k" // K < 64, large output (e.g. lm_head backward dW)
  | "gemv" // M=1 or N=1
  | "batched_small"; // batch > 1, small matrices

/**
 * Result of autotuning for a specific configuration.
 */
export type TuneResult = {
  config: MatmulKernelConfig;
  gflopsPerSec: number;
  medianMs: number;
  shapeClass: ShapeClass;
  dtype: DType;
};

/**
 * Matmul dispatch options passed by the caller.
 */
export type MatmulOptions = {
  /** Whether A is transposed */
  transA?: boolean;
  /** Whether B is transposed */
  transB?: boolean;
  /** Alpha scaling factor */
  alpha?: number;
  /** Specific kernel config (overrides autotuning) */
  config?: MatmulKernelConfig;
};

/**
 * AMP (Automatic Mixed Precision) configuration.
 */
export type AMPConfig = {
  /** Dtype for computation (f16 for speed, f32 for accuracy) */
  computeDtype: DType;
  /** Dtype for accumulation (always f32 for numerical stability) */
  accumulateDtype: "f32";
  /** Dtype for output */
  outputDtype: DType;
};

/**
 * Default kernel configuration optimized for broad compatibility.
 * Conservative tile sizes (32x32) work across most devices.
 */
export const DEFAULT_CONFIG: MatmulKernelConfig = {
  tileM: 32,
  tileN: 32,
  tileK: 16,
  threadTileM: 4,
  threadTileN: 4,
  useSubgroups: false,
  vectorWidth: 1,
};

/**
 * Tuning space for grid search autotuning.
 */
export const TUNING_SPACE = {
  tileM: [32, 64, 128] as const,
  tileN: [32, 64, 128] as const,
  tileK: [8, 16, 32] as const,
  threadTileM: [4, 8] as const,
  threadTileN: [4, 8] as const,
  vectorWidth: [1, 4] as const,
  useSubgroups: [false, true] as const,
};

/**
 * Convert transpose booleans to TransposeMode.
 */
export function getTransposeMode(
  transA: boolean,
  transB: boolean,
): TransposeMode {
  if (!transA && !transB) return "NN";
  if (!transA && transB) return "NT";
  if (transA && !transB) return "TN";
  return "TT";
}

/**
 * Convert TransposeMode to integer for shader uniforms.
 */
export function transposeModeToInt(mode: TransposeMode): number {
  switch (mode) {
    case "NN":
      return 0;
    case "NT":
      return 1;
    case "TN":
      return 2;
    case "TT":
      return 3;
  }
}

/**
 * Classify matrix dimensions into a shape class for autotuning.
 */
export function classifyShape(
  m: number,
  n: number,
  k: number,
  batchSize: number,
): ShapeClass {
  // GEMV cases
  if (m === 1 || n === 1) {
    return "gemv";
  }

  // Batched cases
  if (batchSize > 1 && m * n < 512 * 512) {
    return "batched_small";
  }

  // Small-K: K is very small but output is large (e.g. lm_head backward dW: K=seq_len)
  if (k < 64 && m * n > 100000) {
    return "small_k";
  }

  // Tall-skinny: M is much larger than K or N
  const tallRatio = m / Math.max(k, n);
  if (tallRatio > 4) {
    return "tall_skinny";
  }

  // Short-wide: N is much larger than K or M
  const wideRatio = n / Math.max(k, m);
  if (wideRatio > 4) {
    return "short_wide";
  }

  // Square cases based on size
  const maxDim = Math.max(m, n, k);
  if (maxDim < 512) {
    return "square_small";
  }
  if (maxDim < 2048) {
    return "square_medium";
  }
  return "square_large";
}

/**
 * Validate that a kernel config is valid (tile sizes are compatible).
 */
export function validateConfig(config: MatmulKernelConfig): void {
  const { tileM, tileN, tileK, threadTileM, threadTileN } = config;

  // Workgroup size must not exceed 256 threads (WebGPU limit)
  const threadsM = tileM / threadTileM;
  const threadsN = tileN / threadTileN;
  const workgroupSize = threadsM * threadsN;

  if (workgroupSize > 256) {
    throw new Error(
      `Invalid config: workgroup size ${workgroupSize} exceeds 256`,
    );
  }

  // Tile dimensions must be divisible by thread tile dimensions
  if (tileM % threadTileM !== 0) {
    throw new Error(
      `tileM (${tileM}) must be divisible by threadTileM (${threadTileM})`,
    );
  }
  if (tileN % threadTileN !== 0) {
    throw new Error(
      `tileN (${tileN}) must be divisible by threadTileN (${threadTileN})`,
    );
  }

  // Shared memory size check (rough estimate)
  // Each tile uses tileM * tileK + tileK * tileN floats
  const sharedMemFloats = tileM * tileK + tileK * tileN;
  const sharedMemBytes = sharedMemFloats * 4;
  // WebGPU limit is typically 16KB for workgroup memory
  if (sharedMemBytes > 16384) {
    throw new Error(
      `Config requires ${sharedMemBytes} bytes of shared memory, exceeds 16KB limit`,
    );
  }
}

/**
 * Compute workgroup dimensions for a given config.
 */
export function getWorkgroupSize(config: MatmulKernelConfig): {
  x: number;
  y: number;
} {
  return {
    x: config.tileN / config.threadTileN,
    y: config.tileM / config.threadTileM,
  };
}

/**
 * Subgroup support detection result.
 */
export type SubgroupSupport = {
  /** Whether subgroups are supported */
  supported: boolean;
  /** Subgroup size (typically 32 on most GPUs, 16 on some mobile) */
  subgroupSize?: number;
};

/**
 * Global subgroup support state (cached after first detection).
 */
let cachedSubgroupSupport: SubgroupSupport | null = null;

/**
 * Set the cached subgroup support result.
 */
export function setSubgroupSupport(support: SubgroupSupport): void {
  cachedSubgroupSupport = support;
}

/**
 * Get the cached subgroup support result.
 */
export function getSubgroupSupport(): SubgroupSupport | null {
  return cachedSubgroupSupport;
}

/**
 * Clear the cached subgroup support result.
 */
export function clearSubgroupSupport(): void {
  cachedSubgroupSupport = null;
}
