import type { Token } from "./tokens";

export type LocId = number;
export type BaseId = number;
export type LocRole = "ephemeral" | "persistent";

export interface LocDebugState {
  locLogicalVersion: number;
  locVersion: number;
  role: LocRole;
  hasValue: boolean;
}

export interface BaseDebugState {
  baseCommitVersion: number;
  committedMutations: number[];
}

export interface BaseBindingSnapshot {
  kind: "ssa" | "loc" | "pending_loc";
  locId?: number;
  initTokId?: number;
  initTokKind?: Token["kind"];
}

export interface TraceTensor {
  id: number;
  epoch: number;
  label?: string;
}

export type TensorOrigin =
  | { kind: "global" }
  | { kind: "tidy"; scopeId: number };

export class EngineTensor {
  readonly id: number;
  readonly baseId: BaseId;
  readonly origin: TensorOrigin;
  escapes = false;
  disposed = false;
  /** Optional callback to free backend resources (e.g., GPU buffers) */
  onDispose?: () => void;

  constructor(id: number, baseId: BaseId, origin: TensorOrigin) {
    this.id = id;
    this.baseId = baseId;
    this.origin = origin;
  }
}

export interface RngBasis {
  algorithmId: number;
  seed: number;
}

export interface RngDrawRecord {
  opNonce: number;
  drawNonce: number;
  value: number;
}

export interface RngDrawResult {
  drawNonce: number;
  value: number;
}

export interface SavedTensorRecord {
  id: number;
  baseId: BaseId;
  baseCommitVersionAtSave: number;
}

export interface CheckpointPack {
  id: number;
  reachableBases: BaseId[];
}

export interface BaseState {
  baseCommitVersion: number;
  committed: Set<number>;
}

export interface ExecLock {
  held: boolean;
  ownerId: number;
  depth: number;
}

export interface FinalizeRecord {
  id: number;
}

export interface BaseBinding {
  kind: "ssa" | "loc" | "pending_loc";
  locId?: LocId;
  initTok?: Token;
}

export interface TidyScope {
  id: number;
  tensors: Set<EngineTensor>;
}

/**
 * Async scope context for tracking tensors across await boundaries.
 * Used by async operations like backward() to ensure cleanup of
 * intermediate tensors that would otherwise leak because they're
 * created after synchronous tidy() scopes have exited.
 */
export interface AsyncScope {
  id: number;
  tensors: Set<EngineTensor>;
}

export type TraceTensorStatus = "staging" | "live" | "stale";

export interface TokenSnapshot {
  id: number;
  key: string;
  kind: Token["kind"];
  roots: number[];
}

export interface DebugSnapshot {
  tokGlobal: TokenSnapshot;
  tokLoc: Record<string, TokenSnapshot>;
  locs: Record<string, LocDebugState>;
  bases: Record<string, BaseDebugState>;
  bindings: Record<string, BaseBindingSnapshot>;
}

export interface DebugPlan {
  rootTokenIds: number[];
}

export interface DebugSimulatedState {
  tokGlobalId: number;
  tokLocIds: Record<string, number>;
}

export interface PredictedStateDelta {
  locLogicalVersions: Record<string, number>;
  locVersions: Record<string, number>;
  baseCommitVersions: Record<string, number>;
  baseCommittedMutations: Record<string, number[]>;
  publishSaveCount: number;
}

// ============================================================================
// Engine Visibility Types (Phase 1)
// ============================================================================

/**
 * Memory statistics for engine visibility tooling.
 */
export interface EngineMemoryStats {
  // GPU buffer tracking (from external tracker if available)
  gpuCurrentBytes: number;
  gpuPeakBytes: number;
  gpuLimitBytes: number;

  // Buffer pool state (from external pool if available)
  pooledBuffers: number;
  inUseBuffers: number;
  pendingFenceBuffers: number;

  // Engine state
  activeBases: number;
  totalPinCount: number;
  savedTensorCount: number;
  pendingTensorCount: number;

  // Plan state (from external manager if available)
  activePlans: number;
  completedPlans: number;
}

/**
 * Information about a saved tensor.
 */
export interface SavedTensorInfo {
  id: number;
  baseId: number;
  commitVersionAtSave: number;
  savedAt: number; // timestamp
}

/**
 * Information about a base's state.
 */
export interface BaseStateInfo {
  baseId: number;
  pinCount: number;
  binding: "ssa" | "loc" | "pending_loc";
  locId: number | null;
  hasValue: boolean;
  commitVersion: number;
}

/**
 * Memory snapshot for tracking memory changes over time.
 */
export interface MemorySnapshot {
  label: string;
  timestamp: number;
  stats: EngineMemoryStats;
}

/**
 * Stats provider interface for external memory tracking systems.
 * This allows the engine to query stats from the runtime's buffer pool,
 * GPU memory tracker, and plan manager.
 */
export interface MemoryStatsProvider {
  getGPUStats?: () => {
    currentBytes: number;
    peakBytes: number;
    limitBytes: number;
  };
  getBufferPoolStats?: () => {
    pooledBuffers: number;
    inUseBuffers: number;
    pendingFenceBuffers: number;
  };
  getPlanStats?: () => {
    activePlans: number;
    completedPlans: number;
  };
  getPendingTensorCount?: () => number;
}
