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
  // Autotune types
  type AutotuneOptions,
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
  type EpilogueOp,
  // Codegen functions
  generateTiledMatmulShader,
  getShaderCacheKey,
} from "./codegen";

export {
  // Dispatch functions
  clearDispatchTuningCache,
  clearPerShapeTuningCache,
  clearPipelineCache,
  computeBatchSize,
  computeBatchStrides,
  computeMatmulOutputShape,
  // Dispatch types
  type DispatchMatmulOptions,
  dispatchTiledMatmul,
  getConfigForShape,
  isAutotuneEnabled,
  pretuneMatmulShapes,
  setAutotuneEnabled,
  setTuningResult,
} from "./dispatch";

export {
  // Subgroup functions
  generateSubgroupMatmulShader,
  getSubgroupShaderCacheKey,
  // Subgroup types
  type SubgroupCodegenOptions,
} from "./subgroup";
export {
  type AMPConfig,
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
  type MatmulOptions,
  type MatmulParams,
  type ShapeClass,
  type SubgroupSupport,
  setSubgroupSupport,
  type TransposeMode,
  TUNING_SPACE,
  type TuneResult,
  transposeModeToInt,
  validateConfig,
} from "./types";
