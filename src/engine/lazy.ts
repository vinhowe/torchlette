/**
 * Re-export hub for the lazy execution engine.
 *
 * Only symbols actually imported via this path are re-exported here.
 * For internal-only symbols, import directly from the source module.
 */

// ── Types, interfaces, ref constructors, type guards ──────────────────────────
export type {
  LazyOpCode,
  StorageHandle,
  LazyIRNode,
  LazyRef,
  ExecutionPlan,
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
  createStorageHandle,
} from "./node-factory";

// ── Storage tracking ──────────────────────────────────────────────────────────
export { storageTracker } from "./storage-tracker";

// ── Plan building ─────────────────────────────────────────────────────────────
export {
  markAsCheckpointBoundary,
  buildPlan,
  buildMergedPlan,
} from "./plan-builder";

// ── Sequential executor ───────────────────────────────────────────────────────
export {
  executePlan,
  executePlanWithCheckpointSegments,
  executePlanWithTrueSegments,
} from "./executor-sequential";

// ── Optimized executor ────────────────────────────────────────────────────────
export type { OptimizedExecutionStats } from "./executor-optimized";
export {
  getFusionAnalysisTemplate,
  executePlanOptimized,
} from "./executor-optimized";

// ── Lowered plan executor ─────────────────────────────────────────────────────
export { executeLoweredPlan } from "./executor-lowered";
