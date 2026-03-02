/**
 * Re-export hub for the lazy execution engine.
 *
 * Only symbols actually imported via this path are re-exported here.
 * For internal-only symbols, import directly from the source module.
 */

// ── Lowered plan executor ─────────────────────────────────────────────────────
export { executeLoweredPlan } from "./executor-lowered";
// ── Optimized executor ────────────────────────────────────────────────────────
export type { OptimizedExecutionStats } from "./executor-optimized";
export {
  executePlanOptimized,
  getFusionAnalysisTemplate,
} from "./executor-optimized";
// ── Sequential executor ───────────────────────────────────────────────────────
export {
  executePlan,
  executePlanSegmented,
} from "./executor-sequential";
// ── Types, interfaces, ref constructors, type guards ──────────────────────────
export type {
  ExecutionPlan,
  LazyIRNode,
  LazyOpCode,
  LazyRef,
  StorageHandle,
} from "./lazy-types";
export {
  createMaterializedRef,
  createPendingRef,
  createScalarRef,
  isMaterialized,
  isPending,
} from "./lazy-types";
// ── Node and storage factories ────────────────────────────────────────────────
export {
  createLazyIRNode,
  createStorageHandle,
  resetNodeIdCounter,
  resetStorageIdCounter,
} from "./node-factory";
// ── Plan building ─────────────────────────────────────────────────────────────
export {
  buildMergedPlan,
  buildPlan,
  markAsCheckpointBoundary,
} from "./plan-builder";
// ── Storage tracking ──────────────────────────────────────────────────────────
export { storageTracker } from "./storage-tracker";
