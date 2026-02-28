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
  // Dispatch functions
  clearDispatchTuningCache,
  computeBatchSize,
  computeBatchStrides,
  computeMatmulOutputShape,
  dispatchTiledMatmul,
  isAutotuneEnabled,
  pretuneMatmulShapes,
  resetMatmulState,
  setAutotuneEnabled,
} from "./dispatch";

export {
  // Types
  type CodegenOptions,
  type EpilogueConfig,
  type DType,
  type MatmulKernelConfig,
  type ShapeClass,
  type SubgroupSupport,
  type TransposeMode,
  type TuneResult,
  // Constants
  DEFAULT_CONFIG,
  TUNING_SPACE,
  // Functions
  classifyShape,
  clearSubgroupSupport,
  getShaderCacheKey,
  getSubgroupSupport,
  getTransposeMode,
  getWorkgroupSize,
  setSubgroupSupport,
  validateConfig,
} from "./types";
