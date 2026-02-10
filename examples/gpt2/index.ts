/**
 * GPT-2 Example Module Exports
 *
 * This module exports all components of the GPT-2 finetuning harness.
 */

// Model architecture
export { GPT2, GPT2_SMALL_CONFIG, type GPT2Config } from "./model";
export { CausalSelfAttention, MLP, TransformerBlock } from "./model";

// Weight loading
export {
  loadGPT2Weights,
  loadPretrainedGPT2,
  downloadGPT2,
} from "./loader";

// Data loading
export {
  GPT2Tokenizer,
  FineWebDataLoader,
  createRandomTokens,
  createSyntheticDataLoader,
  downloadTokenizer,
  type FineWebConfig,
} from "./data";

// Training
export {
  GPT2Trainer,
  measureTrainStepTime,
  compareOptimizedVsUnoptimized,
  type GPT2TrainerConfig,
  type TrainResult,
  type TrainCallbacks,
} from "./trainer";

// Benchmarking
export {
  benchmarkForwardPass,
  trackRecompileChurn,
  collectOptimizationStats,
  getDetailedOptimizationStats,
  runPyTorchReference,
  verifyCorrectness,
  runFullBenchmark,
  exportReport,
  type BenchmarkConfig,
  type SpeedComparisonResult,
  type RecompileStats,
  type OptimizationStats,
  type PyTorchResult,
  type CorrectnessResult,
  type FullBenchmarkReport,
} from "./benchmark";
