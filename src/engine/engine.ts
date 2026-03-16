import type { AutocastContext } from "./amp";

// Re-export all types, errors, and helpers from extracted modules
export * from "./engine-types";

// ── Engine helpers (merged from engine-helpers.ts) ──────────────────────────
function collectTensorHandles(value: unknown): EngineTensor[] {
  const out: EngineTensor[] = [];
  const seen = new Set<unknown>();
  const visit = (current: unknown) => {
    if (current == null) return;
    if (current instanceof EngineTensor) {
      out.push(current);
      return;
    }
    if (typeof current !== "object") return;
    if (seen.has(current)) return;
    seen.add(current);
    if (Array.isArray(current)) {
      for (const entry of current) visit(entry);
      return;
    }
    for (const entry of Object.values(current as Record<string, unknown>))
      visit(entry);
  };
  visit(value);
  return out;
}

function computeRngValue(
  basis: RngBasis,
  opNonce: number,
  drawNonce: number,
): number {
  const seed = basis.seed >>> 0;
  const algo = basis.algorithmId >>> 0;
  const op = opNonce >>> 0;
  const draw = drawNonce >>> 0;
  let v =
    (seed ^ Math.imul(algo, 0x9e3779b9) ^ op ^ Math.imul(draw, 0x85ebca6b)) >>>
    0;
  v ^= v >>> 16;
  v = Math.imul(v, 0x7feb352d);
  v ^= v >>> 15;
  v = Math.imul(v, 0x846ca68b);
  v ^= v >>> 16;
  return (v >>> 0) / 2 ** 32;
}

// ── Error classes (merged from engine-errors.ts) ────────────────────────────
export class EngineBusyError extends Error {
  name = "EngineBusyError";
}
export class AsyncInCompileError extends Error {
  name = "AsyncInCompileError";
}
export class SavedTensorModifiedError extends Error {
  name = "SavedTensorModifiedError";
}
export class NonReentrantBackwardError extends Error {
  name = "NonReentrantBackwardError";
}
export class PoisonedEngineError extends Error {
  name = "PoisonedEngineError";
}
export class RngReplayExhaustedError extends Error {
  name = "RngReplayExhaustedError";
}
export class RngReplayMismatchError extends Error {
  name = "RngReplayMismatchError";
}

// Local imports from extracted modules (used by Engine class implementation)
import {
  type AsyncScope,
  type BaseId,
  type BaseState,
  type BaseStateInfo,
  type EngineMemoryStats,
  EngineTensor,
  type ExecLock,
  type MemorySnapshot,
  type MemoryStatsProvider,
  type RngBasis,
  type RngDrawRecord,
  type RngDrawResult,
  type SavedTensorInfo,
  type SavedTensorRecord,
  type TensorOrigin,
  type TidyScope,
} from "./engine-types";

export class Engine {
  private readonly baseState = new Map<BaseId, BaseState>();
  private readonly execLock: ExecLock = { held: false, ownerId: 0, depth: 0 };
  private nextOwnerId = 1;
  private poisoned = false;
  private nextSavedTensorId = 1;
  private backwardActive = false;
  private nextTensorId = 1;
  private nextBaseId = 1;
  private nextScopeId = 1;
  private nextMutIdValue = 1;
  private readonly tidyScopes: TidyScope[] = [];
  /** Async scope context that persists across awaits (for backward(), etc.) */
  private asyncScopeContext: AsyncScope | null = null;
  private readonly basePinCount = new Map<BaseId, number>();
  private rngBasis: RngBasis = { algorithmId: 0, seed: 0 };
  private rngDrawNonce = 0;
  private rngCheckpointMode: "record" | "replay" | null = null;
  private rngCheckpointDraws: RngDrawRecord[] = [];
  private rngCheckpointIndex = 0;
  private autocastContext: AutocastContext | null = null;

  // Visibility tracking (Phase 1)
  private readonly savedTensors = new Map<number, SavedTensorInfo>();
  private readonly memorySnapshots: MemorySnapshot[] = [];
  private memoryStatsProvider: MemoryStatsProvider | null = null;

  constructor() {}

  /**
   * Set the autocast context for AMP transforms in compiled regions.
   * This should be called by the frontend when entering an autocast block.
   */
  setAutocastContext(ctx: AutocastContext | null): void {
    this.autocastContext = ctx;
  }

  /**
   * Get the current autocast context.
   */
  getAutocastContext(): AutocastContext | null {
    return this.autocastContext;
  }

  _debug_publishSave(_baseId: BaseId): void {
    // No-op: token algebra removed. Kept for frontend.ts call site.
  }

  _debug_setRngBasis(basis: RngBasis): void {
    this.rngBasis = { ...basis };
    this.rngDrawNonce = 0;
  }

  _debug_getRngDrawNonce(): number {
    return this.rngDrawNonce;
  }

  _debug_startCheckpointRecord(): void {
    if (this.rngCheckpointMode) {
      throw new Error("Checkpoint RNG already active");
    }
    this.rngCheckpointMode = "record";
    this.rngCheckpointDraws = [];
    this.rngCheckpointIndex = 0;
  }

  _debug_finishCheckpointRecord(): RngDrawRecord[] {
    if (this.rngCheckpointMode !== "record") {
      throw new Error("Checkpoint RNG record not active");
    }
    const draws = this.rngCheckpointDraws.slice();
    this.rngCheckpointMode = null;
    this.rngCheckpointDraws = [];
    this.rngCheckpointIndex = 0;
    return draws;
  }

  _debug_startCheckpointReplay(draws: RngDrawRecord[]): void {
    if (this.rngCheckpointMode) {
      throw new Error("Checkpoint RNG already active");
    }
    this.rngCheckpointMode = "replay";
    this.rngCheckpointDraws = draws.slice();
    this.rngCheckpointIndex = 0;
  }

  _debug_finishCheckpointReplay(): void {
    if (this.rngCheckpointMode !== "replay") {
      throw new Error("Checkpoint RNG replay not active");
    }
    this.rngCheckpointMode = null;
    this.rngCheckpointDraws = [];
    this.rngCheckpointIndex = 0;
  }

  _debug_random(opNonce: number, drawNonce?: number): RngDrawResult {
    if (this.rngCheckpointMode === "replay") {
      const record = this.rngCheckpointDraws[this.rngCheckpointIndex];
      if (!record) {
        throw new RngReplayExhaustedError("RNG replay exhausted");
      }
      if (record.opNonce !== opNonce) {
        throw new RngReplayMismatchError("RNG replay opNonce mismatch");
      }
      if (drawNonce !== undefined && drawNonce !== record.drawNonce) {
        throw new RngReplayMismatchError("RNG replay drawNonce mismatch");
      }
      this.rngCheckpointIndex += 1;
      return { drawNonce: record.drawNonce, value: record.value };
    }

    const assignedDrawNonce = drawNonce ?? this.nextRngDrawNonce();
    if (drawNonce !== undefined && drawNonce > this.rngDrawNonce) {
      this.rngDrawNonce = drawNonce;
    }
    const value = computeRngValue(this.rngBasis, opNonce, assignedDrawNonce);

    if (this.rngCheckpointMode === "record") {
      this.rngCheckpointDraws.push({
        opNonce,
        drawNonce: assignedDrawNonce,
        value,
      });
    }

    return { drawNonce: assignedDrawNonce, value };
  }

  _debug_backward<T>(fn: () => T): T {
    if (this.backwardActive) {
      throw new NonReentrantBackwardError("Backward is already running");
    }
    this.backwardActive = true;
    try {
      return fn();
    } finally {
      this.backwardActive = false;
    }
  }

  _debug_saveForBackward(baseId: BaseId): SavedTensorRecord {
    const state = this.getOrCreateBaseState(baseId);
    const id = this.nextSavedTensorId++;
    const record = {
      id,
      baseId,
      baseCommitVersionAtSave: state.baseCommitVersion,
    };

    // Track for visibility
    this.savedTensors.set(id, {
      id,
      baseId,
      commitVersionAtSave: state.baseCommitVersion,
      savedAt: Date.now(),
    });

    return record;
  }

  _debug_useSavedTensor(record: SavedTensorRecord): void {
    const state = this.getOrCreateBaseState(record.baseId);
    if (state.baseCommitVersion !== record.baseCommitVersionAtSave) {
      throw new SavedTensorModifiedError(
        `Saved tensor modified for baseId ${record.baseId}`,
      );
    }
  }

  /**
   * Release a saved tensor from tracking (called after backward completes).
   */
  _debug_releaseSavedTensor(id: number): void {
    this.savedTensors.delete(id);
  }

  /**
   * Clear all saved tensor tracking (for cleanup after backward).
   */
  _debug_clearSavedTensors(): void {
    this.savedTensors.clear();
  }

  compile<Args extends unknown[], R>(
    fn: (...args: Args) => R,
  ): (...args: Args) => R {
    return (...args: Args) => {
      let result: R;
      result = fn(...args);

      if (
        result &&
        typeof (result as unknown as Promise<unknown>).then === "function"
      ) {
        throw new AsyncInCompileError("Async work is not allowed in compile");
      }

      return result;
    };
  }

  createTensor(baseId?: BaseId): EngineTensor {
    const resolvedBaseId = baseId ?? this.nextBaseId++;
    if (resolvedBaseId >= this.nextBaseId) {
      this.nextBaseId = resolvedBaseId + 1;
    }
    const origin: TensorOrigin =
      this.tidyScopes.length > 0
        ? {
            kind: "tidy",
            scopeId: this.tidyScopes[this.tidyScopes.length - 1].id,
          }
        : { kind: "global" };
    const tensor = new EngineTensor(
      this.nextTensorId++,
      resolvedBaseId,
      origin,
    );
    this.registerTensor(tensor);
    return tensor;
  }

  forceRead(_baseId: BaseId): void {
    // No-op: token algebra and loc bindings removed.
    // Callers use this as a barrier; actual read is done by the runtime.
  }

  tidy<T>(fn: () => T): T {
    const scope: TidyScope = { id: this.nextScopeId++, tensors: new Set() };
    this.tidyScopes.push(scope);
    let result!: T;
    let succeeded = false;

    try {
      result = fn();
      succeeded = true;
    } finally {
      const escaped = succeeded ? collectTensorHandles(result) : [];
      for (const tensor of escaped) {
        tensor.escapes = true;
      }

      // Convert to array first since dispose() modifies scope.tensors
      const tensorsToProcess = Array.from(scope.tensors);
      for (const tensor of tensorsToProcess) {
        if (!tensor.escapes) {
          this.dispose(tensor);
        }
      }
      this.tidyScopes.pop();
    }

    return result;
  }

  keep(tensor: EngineTensor): void {
    if (tensor.disposed) {
      return;
    }
    tensor.escapes = true;
  }

  dispose(tensor: EngineTensor): void {
    if (tensor.disposed) {
      return;
    }
    tensor.disposed = true;
    // Call disposal callback to free backend resources (GPU buffers, etc.)
    if (tensor.onDispose) {
      tensor.onDispose();
    }
    const count = this.basePinCount.get(tensor.baseId) ?? 0;
    if (count <= 1) {
      this.basePinCount.delete(tensor.baseId);
    } else {
      this.basePinCount.set(tensor.baseId, count - 1);
    }
    for (const scope of this.tidyScopes) {
      scope.tensors.delete(tensor);
    }
  }

  async markStep(): Promise<void> {
    this._debug_runEntryPoint(() => {
      // No-op: token algebra and loc bindings removed.
      // Runtime handles plan execution directly.
    });
  }

  private acquireExecLock(): void {
    if (this.execLock.held) throw new EngineBusyError("Engine is busy");
    this.execLock.held = true;
    this.execLock.ownerId = this.nextOwnerId++;
    this.execLock.depth = 1;
    this.ensureNotPoisoned();
  }

  private releaseExecLock(): void {
    this.execLock.held = false;
    this.execLock.ownerId = 0;
    this.execLock.depth = 0;
  }

  _debug_runEntryPoint<T>(fn: () => T): T {
    this.acquireExecLock();
    try {
      return fn();
    } finally {
      this.releaseExecLock();
    }
  }

  async runEntryPoint<T>(fn: () => Promise<T>): Promise<T> {
    this.acquireExecLock();
    try {
      return await fn();
    } finally {
      this.releaseExecLock();
    }
  }

  /**
   * Run an async function with an async scope context that tracks tensors
   * across await boundaries. Tensors created during the async operation
   * (but not within a synchronous tidy scope) are tracked and disposed
   * when the scope exits, unless explicitly kept.
   *
   * This solves the memory leak problem where tensors created during
   * async backward() aren't tracked by synchronous tidy() scopes.
   *
   * @param fn - Async function to run within the scope
   * @returns Result of the function
   */
  async runWithAsyncScope<T>(fn: () => Promise<T>): Promise<T> {
    const scope: AsyncScope = {
      id: this.nextScopeId++,
      tensors: new Set(),
    };

    const previous = this.asyncScopeContext;
    this.asyncScopeContext = scope;

    try {
      return await fn();
    } finally {
      this.asyncScopeContext = previous;

      // Dispose all tensors in this scope that weren't kept
      for (const tensor of scope.tensors) {
        if (!tensor.escapes && !tensor.disposed) {
          this.dispose(tensor);
        }
      }
    }
  }

  _debug_poison(): void {
    this.poisoned = true;
  }

  _debug_getBasePinCount(baseId: BaseId): number {
    return this.basePinCount.get(baseId) ?? 0;
  }

  /**
   * Get the next unique mutation ID for in-place operations.
   * Per spec §4.3, each in-place mutation needs a unique mutId.
   */
  nextMutId(): number {
    return this.nextMutIdValue++;
  }

  _debug_baseCommit(baseId: BaseId, mutId: number): void {
    const state = this.getOrCreateBaseState(baseId);
    if (state.committed.has(mutId)) {
      throw new Error(`base_commit already recorded for mutId ${mutId}`);
    }
    state.committed.add(mutId);
    state.baseCommitVersion += 1;
  }

  private ensureNotPoisoned(): void {
    if (this.poisoned) {
      throw new PoisonedEngineError("Engine is poisoned");
    }
  }

  private nextRngDrawNonce(): number {
    this.rngDrawNonce += 1;
    return this.rngDrawNonce;
  }

  private registerTensor(tensor: EngineTensor): void {
    const count = this.basePinCount.get(tensor.baseId) ?? 0;
    this.basePinCount.set(tensor.baseId, count + 1);

    // Register in all active tidy scopes
    for (const scope of this.tidyScopes) {
      scope.tensors.add(tensor);
    }

    // If no tidy scope is active but async scope is, register there
    // This allows tensors created during async operations (after tidy exit)
    // to be tracked and disposed when the async scope exits
    if (this.tidyScopes.length === 0 && this.asyncScopeContext !== null) {
      this.asyncScopeContext.tensors.add(tensor);
    }
  }

  private getOrCreateBaseState(baseId: BaseId): BaseState {
    const existing = this.baseState.get(baseId);
    if (existing) {
      return existing;
    }
    const initial: BaseState = {
      baseCommitVersion: 0,
      committed: new Set<number>(),
    };
    this.baseState.set(baseId, initial);
    return initial;
  }

  // ============================================================================
  // Engine Visibility Methods (Phase 1)
  // ============================================================================

  /**
   * Set the memory stats provider for external memory tracking.
   * This should be called by the runtime during initialization.
   */
  setMemoryStatsProvider(provider: MemoryStatsProvider): void {
    this.memoryStatsProvider = provider;
  }

  /**
   * Get comprehensive memory statistics.
   */
  _debug_getMemoryStats(): EngineMemoryStats {
    const p = this.memoryStatsProvider;
    const gpu = p?.getGPUStats?.() ?? {
      currentBytes: 0,
      peakBytes: 0,
      limitBytes: 0,
    };
    const pool = p?.getBufferPoolStats?.() ?? {
      pooledBuffers: 0,
      inUseBuffers: 0,
      pendingFenceBuffers: 0,
    };
    const plan = p?.getPlanStats?.() ?? { activePlans: 0, completedPlans: 0 };
    let totalPinCount = 0;
    for (const count of this.basePinCount.values()) totalPinCount += count;

    return {
      gpuCurrentBytes: gpu.currentBytes,
      gpuPeakBytes: gpu.peakBytes,
      gpuLimitBytes: gpu.limitBytes,
      pooledBuffers: pool.pooledBuffers,
      inUseBuffers: pool.inUseBuffers,
      pendingFenceBuffers: pool.pendingFenceBuffers,
      activeBases: this.basePinCount.size,
      totalPinCount,
      savedTensorCount: this.savedTensors.size,
      pendingTensorCount: p?.getPendingTensorCount?.() ?? 0,
      activePlans: plan.activePlans,
      completedPlans: plan.completedPlans,
    };
  }

  /**
   * Get information about all currently saved tensors.
   */
  _debug_getSavedTensorsInfo(): SavedTensorInfo[] {
    return Array.from(this.savedTensors.values());
  }

  /**
   * Get information about all base states.
   */
  _debug_getBaseStatesInfo(): BaseStateInfo[] {
    const infos: BaseStateInfo[] = [];

    for (const [baseId, count] of this.basePinCount) {
      infos.push({
        baseId,
        pinCount: count,
        binding: "ssa",
        locId: null,
        hasValue: false,
        commitVersion: this.baseState.get(baseId)?.baseCommitVersion ?? 0,
      });
    }

    return infos;
  }

  /**
   * Take a memory snapshot with a label.
   */
  _debug_takeMemorySnapshot(label: string): void {
    this.memorySnapshots.push({
      label,
      timestamp: Date.now(),
      stats: this._debug_getMemoryStats(),
    });
  }

  /**
   * Get all memory snapshots.
   */
  _debug_getMemorySnapshots(): MemorySnapshot[] {
    return this.memorySnapshots.slice();
  }

  /**
   * Clear all memory snapshots.
   */
  _debug_clearMemorySnapshots(): void {
    this.memorySnapshots.length = 0;
  }
}
