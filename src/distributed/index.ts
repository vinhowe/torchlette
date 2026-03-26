/**
 * Distributed training primitives for Streaming DiLoCo.
 *
 * - E3M0 4-bit quantization for pseudo-gradient compression
 * - Nesterov outer optimizer for DiLoCo parameter updates
 * - DiLoCoTrainer for coordinating inner/outer optimization loops
 */

export {
  type DiLoCoConfig,
  DiLoCoTrainer,
  type InnerLoopResult,
} from "./diloco";
export {
  e3m0CompressedSize,
  e3m0Dequantize,
  e3m0Quantize,
} from "./e3m0";

export {
  NesterovOuterOptimizer,
  type OuterOptimizerConfig,
} from "./outer-optimizer";
