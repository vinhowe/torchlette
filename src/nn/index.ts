/**
 * Neural network modules and functions.
 * Similar to PyTorch's torch.nn.
 */

// Checkpointing
export {
  type CheckpointOptions,
  checkpoint,
  checkpointSequential,
  createLayerCheckpointer,
  estimateCheckpointSavings,
  type LayerCheckpointPolicy,
} from "./checkpoint";
// Gradient clipping
export { clipGradNorm_, clipGradValue_ } from "./clip-grad";
export { Dropout, type DropoutOptions } from "./dropout";
export { Embedding, type EmbeddingOptions } from "./embedding";
// Functional interface (nn.functional)
export * as functional from "./functional";
export {
  crossEntropy,
  dropout,
  logSoftmax,
  nllLoss,
} from "./functional";
// Weight initialization (nn.init)
export * as init from "./init";
export { LayerNorm, type LayerNormOptions } from "./layernorm";
export { Linear, type LinearOptions } from "./linear";
export { Module } from "./module";
export { ModuleList } from "./modulelist";
export { Sequential } from "./sequential";
