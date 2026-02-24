/**
 * Tiled matrix multiplication module for WebGPU.
 *
 * Provides high-performance matmul with:
 * - Shared memory tiling
 * - Transpose variants (NN, NT, TN, TT)
 * - Epilogue fusion (arbitrary elementwise ops)
 * - Autotuning support
 */

export {
  // Autotune functions
  autotune,
  type BenchmarkFn,
  cacheTuningResult,
  clearTuningCache,
  exportTuningCache,
  generateNeighborConfigs,
  generateTuningConfigs,
  getCachedTuningResult,
  getDefaultConfigForShape,
  importTuningCache,
  quickAutotune,
} from "./autotune";

export {
  // Codegen types
  type CodegenOptions,
  type EpilogueConfig,
  // Codegen functions
  generateTiledMatmulShader,
  getShaderCacheKey,
} from "./codegen";

export {
  // Dispatch functions
  clearDispatchTuningCache,
  computeBatchSize,
  computeBatchStrides,
  computeMatmulOutputShape,
  dispatchTiledMatmul,
  isAutotuneEnabled,
  pretuneMatmulShapes,
  setAutotuneEnabled,
} from "./dispatch";

export {
  // Subgroup functions
  generateSubgroupMatmulShader,
  getSubgroupShaderCacheKey,
  // Subgroup types
  type SubgroupCodegenOptions,
} from "./subgroup";
export {
  // Utilities
  classifyShape,
  clearSubgroupSupport,
  // Constants
  DEFAULT_CONFIG,
  // Types
  type DType,
  getSubgroupSupport,
  getTransposeMode,
  getWorkgroupSize,
  type MatmulKernelConfig,
  type ShapeClass,
  type SubgroupSupport,
  setSubgroupSupport,
  type TransposeMode,
  TUNING_SPACE,
  type TuneResult,
  validateConfig,
} from "./types";
