/**
 * Re-export hub for the lazy execution engine.
 *
 * This file was decomposed from a 6,008-line monolith into focused modules:
 *   lazy-types.ts          — Types, interfaces, ref constructors, type guards
 *   node-factory.ts        — Node/storage ID counters, create functions, lazy WebGPU imports
 *   storage-tracker.ts     — StorageTracker class, singleton, early-release helpers
 *   plan-builder.ts        — Plan building, checkpoint segmentation, matmul pretuning
 *   op-dispatch.ts         — Op dispatch switch, getInputStorage, computeContiguousStrides
 *   matmul-epilogue.ts     — Epilogue/prologue detection and execution
 *   reduction-preamble.ts  — Reduction preamble detection and execution
 *   segment-executors.ts   — Fused/sequential segment execution, reclaim interval
 *   executor-sequential.ts — Main sequential executor, checkpoint executors
 *   executor-optimized.ts  — Optimized executor with fusion analysis caching
 *   executor-lowered.ts    — Lowered plan executor (dispatch replay)
 */

// ── Types, interfaces, ref constructors, type guards ──────────────────────────
export type {
  LazyOpCode,
  GeluApproximate,
  StorageHandle,
  LazyIRNode,
  LazyRef,
  ExecutionPlan,
  ExecutePlanOptions,
} from "./lazy-types";
export {
  createPendingRef,
  createMaterializedRef,
  createScalarRef,
  isPending,
  isMaterialized,
} from "./lazy-types";

// ── Node and storage factories ────────────────────────────────────────────────
export {
  resetNodeIdCounter,
  createLazyIRNode,
  resetStorageIdCounter,
  getNextStorageId,
  createStorageHandle,
  _webgpuMatmulImports,
  _webgpuMatmulGeomImports,
  ensureWebGPUMatmulImports,
} from "./node-factory";

// ── Storage tracking ──────────────────────────────────────────────────────────
export { storageTracker, canSafelyRelease, releaseBufferImmediate } from "./storage-tracker";

// ── Plan building ─────────────────────────────────────────────────────────────
export {
  markAsCheckpointBoundary,
  segmentPlanAtCheckpoints,
  buildPlan,
  buildMergedPlan,
  extractPlanMetadata,
  pretunePlanMatmuls,
} from "./plan-builder";

// ── Op dispatch ───────────────────────────────────────────────────────────────
export {
  getInputStorage,
  executeOp,
  executeOpInternal,
  computeContiguousStrides,
} from "./op-dispatch";

// ── Matmul epilogue detection and execution ───────────────────────────────────
export type { MatmulPrologueInfo, MatmulEpiloguePlan } from "./matmul-epilogue";
export {
  detectMatmulEpilogueCore,
  detectMatmulEpilogueFromPlan,
  detectMatmulEpilogue,
  _detectTransposeView,
  executeMatmulWithEpilogue,
  shapesEqual,
} from "./matmul-epilogue";

// ── Reduction preamble detection and execution ────────────────────────────────
export type { ReductionPreamblePlan } from "./reduction-preamble";
export {
  detectReductionPreamble,
  executeReductionWithPreamble,
} from "./reduction-preamble";

// ── Segment executors ─────────────────────────────────────────────────────────
export {
  DEFAULT_RECLAIM_INTERVAL,
  executeFusedSegment,
  executeFusedWebGPU,
  executeSequentialSegment,
  executeSequentialSegmentWithEarlyRelease,
} from "./segment-executors";

// ── Sequential executor ───────────────────────────────────────────────────────
export {
  FILL_IN_OPS,
  executePlan,
  executePlanWithCheckpointSegments,
  executePlanWithTrueSegments,
  findSurvivingOutputs,
} from "./executor-sequential";

// ── Optimized executor ────────────────────────────────────────────────────────
export type {
  OptimizedExecutionOptions,
  OptimizedExecutionStats,
  OptimizedExecutionResult,
  FusionAnalysisTemplate,
  CachedSegmentDesc,
} from "./executor-optimized";
export {
  fusionAnalysisCache,
  getFusionAnalysisTemplate,
  executePlanOptimized,
} from "./executor-optimized";

// ── Lowered plan executor ─────────────────────────────────────────────────────
export { executeLoweredPlan } from "./executor-lowered";
