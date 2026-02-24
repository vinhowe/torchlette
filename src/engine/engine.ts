import type { DType } from "../backend/types";
import { type AutocastContext, hashAMPPolicy } from "./amp";
import { applyAMPTransform } from "./amp-ir-transform";
import {
  CompiledCache,
  type CompiledCacheKey,
  generateCacheKey,
} from "./compile-cache";
import {
  collectTensorHandles,
  collectTraceTensorIds,
  computeRngValue,
  isThenable,
  isTraceTensor,
} from "./engine-helpers";
import { buildIRFromTrace, type IRGraph } from "./ir";
import {
  buildPlanLinearOrder,
  type DebugPlanLinearOrder,
  expandSemanticSubeventSchedule,
  type PlanEvent,
  type SemanticSubeventSchedule,
} from "./planner";
import { type Token, TokenStore } from "./tokens";
import { type TraceEvent, TraceRecorder } from "./trace";

// Re-export all types, errors, and helpers from extracted modules
export * from "./engine-types";
export * from "./engine-errors";
export * from "./engine-helpers";

// Local imports from extracted modules (used by Engine class implementation)
import {
  type AsyncScope,
  type BaseBinding,
  type BaseId,
  type BaseState,
  type BaseStateInfo,
  type CheckpointPack,
  type DebugPlan,
  type DebugSimulatedState,
  type DebugSnapshot,
  type EngineMemoryStats,
  EngineTensor,
  type ExecLock,
  type FinalizeRecord,
  type LocDebugState,
  type LocId,
  type MemorySnapshot,
  type MemoryStatsProvider,
  type PredictedStateDelta,
  type RngBasis,
  type RngDrawRecord,
  type SavedTensorInfo,
  type SavedTensorRecord,
  type TensorOrigin,
  type TidyScope,
  type TokenSnapshot,
  type TraceTensor,
  type TraceTensorStatus,
} from "./engine-types";
import {
  AsyncInCompileError,
  CheckpointImpureRegionError,
  EngineBusyError,
  HostReadInCompileError,
  InvalidTraceTensorEscapeError,
  NonReentrantBackwardError,
  PoisonedEngineError,
  RngReplayExhaustedError,
  RngReplayMismatchError,
  SavedTensorModifiedError,
} from "./engine-errors";

export class Engine {
  private readonly tokenStore: TokenStore;
  private tokGlobal: Token;
  private readonly tokLoc = new Map<LocId, Token>();
  private readonly locState = new Map<LocId, LocDebugState>();
  private readonly baseState = new Map<BaseId, BaseState>();
  private readonly baseBindings = new Map<BaseId, BaseBinding>();
  private readonly finalizeQueue: FinalizeRecord[] = [];
  private readonly execLock: ExecLock = { held: false, ownerId: 0, depth: 0 };
  private nextOwnerId = 1;
  private poisoned = false;
  private recomputeMode = false;
  private stagingActive = false;
  private currentEpoch = 0;
  private currentStagingIds = new Set<number>();
  private readonly traceTensorStatus = new Map<number, TraceTensorStatus>();
  private lastCompiledGraph: IRGraph | null = null;
  private lastCacheKey: CompiledCacheKey | null = null;
  private lastCacheHit = false;
  private readonly compiledCache = new CompiledCache();
  private nextTraceTensorId = 1;
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
  private nextCheckpointPackId = 1;
  private checkpointReachableBases: Set<BaseId> | null = null;
  private activeCheckpointPackId: number | null = null;
  private autocastContext: AutocastContext | null = null;

  // Visibility tracking (Phase 1)
  private readonly savedTensors = new Map<number, SavedTensorInfo>();
  private readonly memorySnapshots: MemorySnapshot[] = [];
  private memoryStatsProvider: MemoryStatsProvider | null = null;

  readonly trace: TraceRecorder;

  constructor(trace: TraceRecorder = new TraceRecorder()) {
    this.trace = trace;
    this.tokenStore = new TokenStore();
    this.tokGlobal = this.tokenStore.root;
  }

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

  afterAll(...tokens: Token[]): Token {
    const { token, roots } = this.tokenStore.afterAll(tokens);
    const inputs = Array.from(new Set(tokens.map((tok) => tok.id))).sort(
      (a, b) => a - b,
    );
    this.trace.record({
      type: "after_all",
      inputs,
      output: token.id,
      outputKey: roots.join(","),
    });
    return token;
  }

  orderedAccess(
    locId: LocId,
    op: "load" | "store" | "access" = "access",
    extraTokens: Token[] = [],
  ): Token {
    this.ensureNotPoisoned();
    const tokLoc = this.tokLoc.get(locId);
    const tokIn = this.afterAll(
      this.tokGlobal,
      tokLoc ?? this.tokGlobal,
      ...extraTokens,
    );
    const tokOut = this.tokenStore.createEffectToken();

    this.trace.record({
      type: "effect",
      op: `ordered_${op}`,
      input: tokIn.id,
      output: tokOut.id,
      locId,
    });

    this.tokGlobal = tokOut;
    this.tokLoc.set(locId, tokOut);

    this.trace.record({
      type: "set_token",
      target: "global",
      token: tokOut.id,
    });
    this.trace.record({
      type: "set_token",
      target: "loc",
      locId,
      token: tokOut.id,
    });

    return tokOut;
  }

  emitEffect(op: string): Token {
    this.ensureNotPoisoned();
    return this.emitEffectFrom(this.tokGlobal, op);
  }

  _debug_publishSave(_baseId: BaseId): Token {
    const tokOut = this.emitEffect("publish_save");
    this.trace.record({ type: "publish_save" });
    return tokOut;
  }

  _debug_emitCompiledCall(
    graphInstanceId: number,
    callInstanceId: number,
  ): void {
    this.trace.record({
      type: "compiled_call",
      graphInstanceId,
      callInstanceId,
    });
  }

  _debug_setRngBasis(basis: RngBasis): void {
    this.rngBasis = { ...basis };
    this.rngDrawNonce = 0;
    this.trace.record({
      type: "rng_basis",
      algorithmId: basis.algorithmId,
      seed: basis.seed,
    });
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
    this.trace.record({ type: "rng_checkpoint_record_start" });
  }

  _debug_finishCheckpointRecord(): RngDrawRecord[] {
    if (this.rngCheckpointMode !== "record") {
      throw new Error("Checkpoint RNG record not active");
    }
    const draws = this.rngCheckpointDraws.slice();
    this.rngCheckpointMode = null;
    this.rngCheckpointDraws = [];
    this.rngCheckpointIndex = 0;
    this.trace.record({
      type: "rng_checkpoint_record_finish",
      count: draws.length,
    });
    return draws;
  }

  _debug_startCheckpointReplay(draws: RngDrawRecord[]): void {
    if (this.rngCheckpointMode) {
      throw new Error("Checkpoint RNG already active");
    }
    this.rngCheckpointMode = "replay";
    this.rngCheckpointDraws = draws.slice();
    this.rngCheckpointIndex = 0;
    this.trace.record({
      type: "rng_checkpoint_replay_start",
      count: draws.length,
    });
  }

  _debug_finishCheckpointReplay(): void {
    if (this.rngCheckpointMode !== "replay") {
      throw new Error("Checkpoint RNG replay not active");
    }
    const count = this.rngCheckpointDraws.length;
    this.rngCheckpointMode = null;
    this.rngCheckpointDraws = [];
    this.rngCheckpointIndex = 0;
    this.trace.record({
      type: "rng_checkpoint_replay_finish",
      count,
    });
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
      this.trace.record({
        type: "rng_draw",
        opNonce: record.opNonce,
        drawNonce: record.drawNonce,
        value: record.value,
      });
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

    this.trace.record({
      type: "rng_draw",
      opNonce,
      drawNonce: assignedDrawNonce,
      value,
    });

    return { drawNonce: assignedDrawNonce, value };
  }

  _debug_backward<T>(fn: () => T): T {
    if (this.backwardActive) {
      throw new NonReentrantBackwardError("Backward is already running");
    }
    this.backwardActive = true;
    this.emitEffect("backward_root");
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
      if (this.stagingActive) {
        throw new Error("Compile already active");
      }

      this.stagingActive = true;
      this.currentEpoch += 1;
      this.currentStagingIds = new Set();

      let result: R;
      try {
        result = fn(...args);
      } catch (error) {
        this.finishStaging(undefined);
        throw error;
      }

      if (isThenable(result)) {
        this.finishStaging(undefined);
        throw new AsyncInCompileError("Async work is not allowed in compile");
      }

      this.finishStaging(result);
      return result;
    };
  }

  _debug_hostRead(): void {
    if (this.stagingActive) {
      throw new HostReadInCompileError(
        "Host reads are forbidden during compile",
      );
    }
  }

  _debug_makeTraceTensor(label?: string): TraceTensor {
    if (!this.stagingActive) {
      throw new Error("Trace tensors may only be created during compile");
    }
    const tensor: TraceTensor = {
      id: this.nextTraceTensorId++,
      epoch: this.currentEpoch,
      label,
    };
    this.currentStagingIds.add(tensor.id);
    this.traceTensorStatus.set(tensor.id, "staging");
    return tensor;
  }

  _debug_emitLazyOp(
    op: string,
    options?: {
      inputs?: TraceTensor[];
      shape?: number[];
      dtype?: DType;
      scalarValues?: number[];
    },
  ): TraceTensor {
    const inputs = options?.inputs ?? [];
    for (const input of inputs) {
      if (input.epoch !== this.currentEpoch) {
        throw new Error("Trace tensor belongs to a different epoch");
      }
      this._debug_useTraceTensor(input);
    }
    const tensor = this._debug_makeTraceTensor(op);
    this.trace.record({
      type: "lazy_op",
      op,
      traceId: tensor.id,
      epoch: tensor.epoch,
      inputs: inputs.length > 0 ? inputs.map((input) => input.id) : undefined,
      shape: options?.shape ? options.shape.slice() : undefined,
      dtype: options?.dtype,
      scalarValues: options?.scalarValues ? options.scalarValues.slice() : undefined,
    });
    return tensor;
  }

  _debug_getLastCompiledGraph(): IRGraph | null {
    return this.lastCompiledGraph;
  }

  _debug_getLastCacheKey(): CompiledCacheKey | null {
    return this.lastCacheKey;
  }

  _debug_wasLastCompileCacheHit(): boolean {
    return this.lastCacheHit;
  }

  _debug_getCompiledCacheStats(): {
    size: number;
    entries: { key: string; hitCount: number }[];
  } {
    return this.compiledCache.stats();
  }

  _debug_clearCompiledCache(): void {
    this.compiledCache.clear();
  }

  _debug_useTraceTensor(tensor: TraceTensor): void {
    const status = this.traceTensorStatus.get(tensor.id);
    if (!status) {
      throw new Error(`Unknown trace tensor ${tensor.id}`);
    }
    if (status === "stale") {
      throw new InvalidTraceTensorEscapeError("Trace tensor is stale");
    }
  }

  _debug_bindPendingLoc(baseId: BaseId, locId: LocId): void {
    this.getOrCreateLocState(locId);
    this.baseBindings.set(baseId, { kind: "pending_loc", locId });
  }

  _debug_setLocRole(locId: LocId, role: LocRole): void {
    const state = this.getOrCreateLocState(locId);
    state.role = role;
  }

  _debug_setRecomputeMode(enabled: boolean): void {
    this.recomputeMode = enabled;
    if (!enabled) {
      this.checkpointReachableBases = null;
    }
  }

  _debug_ensureInitialized(
    baseId: BaseId,
    options: { subsumedByStore?: boolean } = {},
  ): Token {
    const binding = this.getBaseBinding(baseId);
    if (binding.kind !== "pending_loc" || binding.locId === undefined) {
      throw new Error("ensureInitialized requires a pending_loc binding");
    }

    if (binding.initTok) {
      return binding.initTok;
    }

    if (this.recomputeMode) {
      throw new CheckpointImpureRegionError(
        "Cannot initialize loc during recompute",
      );
    }

    this.getOrCreateLocState(binding.locId);
    const tokOut = options.subsumedByStore
      ? this.tokenStore.createTokenOnlyToken()
      : this.tokenStore.createEffectToken();
    const op = options.subsumedByStore
      ? "init_loc_token_only"
      : "init_loc_store";
    this.trace.record({
      type: "effect",
      op,
      input: this.tokGlobal.id,
      output: tokOut.id,
      locId: binding.locId,
    });
    this.tokGlobal = tokOut;
    this.trace.record({
      type: "set_token",
      target: "global",
      token: tokOut.id,
    });

    binding.initTok = tokOut;
    return tokOut;
  }

  _debug_orderedAccessBase(
    baseId: BaseId,
    op: "load" | "store" | "access",
  ): Token {
    const binding = this.getBaseBinding(baseId);
    if (binding.kind === "pending_loc" && binding.locId !== undefined) {
      const initTok = this._debug_ensureInitialized(baseId);
      return this.orderedAccess(binding.locId, op, [initTok]);
    }
    if (binding.kind === "loc" && binding.locId !== undefined) {
      return this.orderedAccess(binding.locId, op);
    }
    throw new Error("orderedAccessBase requires a loc-backed binding");
  }

  _debug_recomputeLocStore(locId: LocId): void {
    const state = this.getOrCreateLocState(locId);
    if (this.recomputeMode && state.role === "persistent") {
      throw new CheckpointImpureRegionError(
        "Cannot store to persistent loc during recompute",
      );
    }
    this._debug_commitLocStore(locId);
  }

  _debug_writeSavedState(): void {
    if (this.recomputeMode) {
      throw new CheckpointImpureRegionError(
        "Cannot create saved_state during recompute",
      );
    }
  }

  _debug_recomputeMutateBase(baseId: BaseId, mutId: number): void {
    if (this.recomputeMode && this.checkpointReachableBases?.has(baseId)) {
      throw new CheckpointImpureRegionError(
        `Cannot mutate base ${baseId} during recompute`,
      );
    }
    this._debug_baseCommit(baseId, mutId);
  }

  _debug_checkpointPack(bases: BaseId[]): CheckpointPack {
    const unique = Array.from(new Set(bases)).sort((a, b) => a - b);
    const pack = { id: this.nextCheckpointPackId++, reachableBases: unique };
    this.trace.record({
      type: "checkpoint_pack",
      packId: pack.id,
      reachableBases: pack.reachableBases.slice(),
    });
    return pack;
  }

  _debug_startCheckpointRecompute(pack: CheckpointPack): void {
    this.recomputeMode = true;
    this.checkpointReachableBases = new Set(pack.reachableBases);
    this.activeCheckpointPackId = pack.id;
    this.trace.record({
      type: "checkpoint_recompute_start",
      packId: pack.id,
      reachableBases: pack.reachableBases.slice(),
    });
  }

  _debug_finishCheckpointRecompute(): void {
    this.recomputeMode = false;
    this.checkpointReachableBases = null;
    this.trace.record({
      type: "checkpoint_recompute_finish",
      packId: this.activeCheckpointPackId,
    });
    this.activeCheckpointPackId = null;
  }

  _debug_enqueueFinalize(record: FinalizeRecord): void {
    this.finalizeQueue.push(record);
    this.trace.record({ type: "finalize_enqueue", recordId: record.id });
  }

  _debug_drainFinalizeQueueCleanupOnly(): FinalizeRecord[] {
    const drained = this.finalizeQueue.splice(0);
    this.trace.record({ type: "finalize_drain", count: drained.length });
    return drained;
  }

  createTensor(baseId?: BaseId): EngineTensor {
    const resolvedBaseId = baseId ?? this.nextBaseId++;
    if (resolvedBaseId >= this.nextBaseId) {
      this.nextBaseId = resolvedBaseId + 1;
    }
    const origin =
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

  forceRead(baseId: BaseId): void {
    this._debug_hostRead();
    this.finalizePendingLocBindings();
    const planTokens = this.collectForcePlanTokens(baseId);
    const tokenIds = Array.from(
      new Set(planTokens.map((token) => token.id)),
    ).sort((a, b) => a - b);
    this.trace.record({ type: "force_plan", baseId, tokenIds });
    const token = this.afterAll(planTokens);
    this.emitEffectFrom(token, `host_read:${baseId}`);
  }

  tidy<T>(fn: () => T): T {
    const scope: TidyScope = { id: this.nextScopeId++, tensors: new Set() };
    this.tidyScopes.push(scope);
    let result: T;
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
      this.trace.record({ type: "mark_step_begin" });
      this.emitEffect("mark_step");
      const finalized = this.finalizePendingLocBindings();
      this.trace.record({
        type: "mark_step_finalize_bindings",
        count: finalized,
      });
      this.trace.record({ type: "mark_step_retain" });
      this.trace.record({ type: "mark_step_gc" });
      this._debug_drainFinalizeQueueCleanupOnly();
      this.tokGlobal = this.tokenStore.root;
      this.tokLoc.clear();
      this.trace.record({
        type: "set_token",
        target: "global",
        token: this.tokGlobal.id,
      });
      this.trace.record({ type: "mark_step_end" });
    });
  }

  _debug_runEntryPoint<T>(fn: () => T): T {
    if (this.execLock.held) {
      throw new EngineBusyError("Engine is busy");
    }
    const ownerId = this.nextOwnerId++;
    this.execLock.held = true;
    this.execLock.ownerId = ownerId;
    this.execLock.depth = 1;

    this._debug_drainFinalizeQueueCleanupOnly();

    try {
      this.ensureNotPoisoned();
      return fn();
    } finally {
      this._debug_drainFinalizeQueueCleanupOnly();
      this.execLock.held = false;
      this.execLock.ownerId = 0;
      this.execLock.depth = 0;
    }
  }

  async runEntryPoint<T>(fn: () => Promise<T>): Promise<T> {
    if (this.execLock.held) {
      throw new EngineBusyError("Engine is busy");
    }
    const ownerId = this.nextOwnerId++;
    this.execLock.held = true;
    this.execLock.ownerId = ownerId;
    this.execLock.depth = 1;

    this._debug_drainFinalizeQueueCleanupOnly();

    try {
      this.ensureNotPoisoned();
      return await fn();
    } finally {
      this._debug_drainFinalizeQueueCleanupOnly();
      this.execLock.held = false;
      this.execLock.ownerId = 0;
      this.execLock.depth = 0;
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

  _debugSnapshot(): DebugSnapshot {
    const tokLocEntries = Array.from(this.tokLoc.entries()).sort(
      ([a], [b]) => a - b,
    );
    const tokLoc: Record<string, TokenSnapshot> = {};
    for (const [locId, token] of tokLocEntries) {
      tokLoc[locId.toString()] = this.snapshotToken(token);
    }

    const locEntries = Array.from(this.locState.entries()).sort(
      ([a], [b]) => a - b,
    );
    const locs: Record<string, LocDebugState> = {};
    for (const [locId, state] of locEntries) {
      locs[locId.toString()] = {
        locLogicalVersion: state.locLogicalVersion,
        locVersion: state.locVersion,
        role: state.role,
        hasValue: state.hasValue,
      };
    }

    const baseEntries = Array.from(this.baseState.entries()).sort(
      ([a], [b]) => a - b,
    );
    const bases: Record<string, BaseDebugState> = {};
    for (const [baseId, state] of baseEntries) {
      bases[baseId.toString()] = {
        baseCommitVersion: state.baseCommitVersion,
        committedMutations: Array.from(state.committed).sort((a, b) => a - b),
      };
    }

    const bindingEntries = Array.from(this.baseBindings.entries()).sort(
      ([a], [b]) => a - b,
    );
    const bindings: Record<string, BaseBindingSnapshot> = {};
    for (const [baseId, binding] of bindingEntries) {
      bindings[baseId.toString()] = {
        kind: binding.kind,
        locId: binding.locId,
        initTokId: binding.initTok?.id,
        initTokKind: binding.initTok?.kind,
      };
    }

    return {
      tokGlobal: this.snapshotToken(this.tokGlobal),
      tokLoc,
      locs,
      bases,
      bindings,
    };
  }

  _debugCreateToken(): Token {
    return this.tokenStore.createDebugToken();
  }

  _debug_buildPlan(tokens: Token[]): DebugPlan {
    return { rootTokenIds: tokens.map((token) => token.id) };
  }

  _debug_buildPlanLinearOrder(events: PlanEvent[]): DebugPlanLinearOrder {
    return buildPlanLinearOrder(events);
  }

  _debug_buildPlanFromTrace(
    traceEvents: TraceEvent[] = this.trace.snapshot(),
  ): DebugPlanLinearOrder {
    let opNonce = 0;
    const events: PlanEvent[] = [];

    for (const event of traceEvents) {
      if (event.type === "rng_basis") {
        opNonce += 1;
        events.push({
          name: "rng_basis",
          key: {
            graphInstanceId: 0,
            callInstanceId: 0,
            planInstanceId: 0,
            opNonce,
            drawNonce: 0,
            mutId: 0,
            kind: "rng_basis",
          },
          payload: { algorithmId: event.algorithmId, seed: event.seed },
        });
      }

      if (
        event.type === "rng_checkpoint_record_start" ||
        event.type === "rng_checkpoint_record_finish" ||
        event.type === "rng_checkpoint_replay_start" ||
        event.type === "rng_checkpoint_replay_finish"
      ) {
        opNonce += 1;
        events.push({
          name: event.type,
          key: {
            graphInstanceId: 0,
            callInstanceId: 0,
            planInstanceId: 0,
            opNonce,
            drawNonce: 0,
            mutId: 0,
            kind: event.type,
          },
        });
      }

      if (event.type === "publish_save") {
        opNonce += 1;
        events.push({
          name: "publish_save",
          key: {
            graphInstanceId: 0,
            callInstanceId: 0,
            planInstanceId: 0,
            opNonce,
            drawNonce: 0,
            mutId: 0,
            kind: "publish_save",
          },
        });
      }

      if (event.type === "rng_draw") {
        opNonce = Math.max(opNonce, event.opNonce);
        events.push({
          name: "rng_draw",
          key: {
            graphInstanceId: 0,
            callInstanceId: 0,
            planInstanceId: 0,
            opNonce: event.opNonce,
            drawNonce: event.drawNonce,
            mutId: 0,
            kind: "rng_draw",
          },
          payload: { drawNonce: event.drawNonce, opNonce: event.opNonce },
        });
      }

      if (event.type === "loc_schedule") {
        opNonce += 1;
        events.push({
          name: "loc_schedule",
          key: {
            graphInstanceId: 0,
            callInstanceId: 0,
            planInstanceId: 0,
            opNonce,
            drawNonce: 0,
            mutId: 0,
            kind: "loc_schedule",
          },
          payload: { locId: event.locId },
        });
      }

      if (event.type === "loc_commit") {
        opNonce += 1;
        events.push({
          name: "loc_commit",
          key: {
            graphInstanceId: 0,
            callInstanceId: 0,
            planInstanceId: 0,
            opNonce,
            drawNonce: 0,
            mutId: 0,
            kind: "loc_commit",
          },
          payload: { locId: event.locId },
        });
      }

      if (event.type === "base_commit") {
        opNonce += 1;
        events.push({
          name: "base_commit",
          key: {
            graphInstanceId: 0,
            callInstanceId: 0,
            planInstanceId: 0,
            opNonce,
            drawNonce: 0,
            mutId: event.mutId,
            kind: "base_commit",
          },
          payload: { baseId: event.baseId, mutId: event.mutId },
        });
      }
    }

    return buildPlanLinearOrder(events);
  }

  _debug_buildPlanFromSchedules(
    schedules: SemanticSubeventSchedule[],
    traceEvents: TraceEvent[] = this.trace.snapshot(),
  ): DebugPlanLinearOrder {
    const basePlan = this._debug_buildPlanFromTrace(traceEvents);
    const scheduleEvents = schedules.flatMap((schedule) =>
      expandSemanticSubeventSchedule(schedule),
    );
    return buildPlanLinearOrder([...basePlan.orderedEvents, ...scheduleEvents]);
  }

  _debug_buildPlanWithCompiledCalls(
    schedules: Record<number, SemanticSubeventSchedule>,
    traceEvents: TraceEvent[] = this.trace.snapshot(),
  ): DebugPlanLinearOrder {
    const compiledSchedules: SemanticSubeventSchedule[] = [];
    for (const event of traceEvents) {
      if (event.type === "compiled_call") {
        const schedule = schedules[event.callInstanceId];
        if (!schedule) {
          throw new Error(
            `Missing schedule for compiled call ${event.callInstanceId}`,
          );
        }
        compiledSchedules.push(schedule);
      }
    }
    return this._debug_buildPlanFromSchedules(compiledSchedules, traceEvents);
  }

  _debug_simulateCommitPlan(plan: DebugPlanLinearOrder): PredictedStateDelta {
    const locLogicalVersions: Record<string, number> = {};
    const locVersions: Record<string, number> = {};
    const baseCommitVersions: Record<string, number> = {};
    const baseCommittedMutations: Record<string, number[]> = {};
    let publishSaveCount = 0;

    for (const event of plan.orderedEvents) {
      if (event.key.kind === "loc_schedule") {
        const locId = event.payload?.locId;
        if (locId !== undefined) {
          const key = locId.toString();
          locLogicalVersions[key] = (locLogicalVersions[key] ?? 0) + 1;
        }
      }

      if (event.key.kind === "loc_commit") {
        const locId = event.payload?.locId;
        if (locId !== undefined) {
          const key = locId.toString();
          locVersions[key] = (locVersions[key] ?? 0) + 1;
        }
      }

      if (event.key.kind === "base_commit") {
        const baseId = event.payload?.baseId;
        if (baseId !== undefined) {
          const key = baseId.toString();
          baseCommitVersions[key] = (baseCommitVersions[key] ?? 0) + 1;
          const mutId = event.payload?.mutId ?? event.key.mutId;
          if (!baseCommittedMutations[key]) {
            baseCommittedMutations[key] = [];
          }
          baseCommittedMutations[key].push(mutId);
        }
      }

      if (event.key.kind === "publish_save") {
        publishSaveCount += 1;
      }
    }

    return {
      locLogicalVersions,
      locVersions,
      baseCommitVersions,
      baseCommittedMutations,
      publishSaveCount,
    };
  }

  _debug_scheduleLocAccess(locId: LocId): LocDebugState {
    const state = this.getOrCreateLocState(locId);
    state.locLogicalVersion += 1;
    this.trace.record({
      type: "loc_schedule",
      locId,
      locLogicalVersion: state.locLogicalVersion,
    });
    return { ...state };
  }

  _debug_commitLocStore(locId: LocId): LocDebugState {
    const state = this.getOrCreateLocState(locId);
    state.locVersion += 1;
    state.hasValue = true;
    this.trace.record({
      type: "loc_commit",
      locId,
      locVersion: state.locVersion,
    });
    return { ...state };
  }

  /**
   * Get the next unique mutation ID for in-place operations.
   * Per spec §4.3, each in-place mutation needs a unique mutId.
   */
  nextMutId(): number {
    return this.nextMutIdValue++;
  }

  _debug_baseCommit(baseId: BaseId, mutId: number): BaseDebugState {
    const state = this.getOrCreateBaseState(baseId);
    if (state.committed.has(mutId)) {
      throw new Error(`base_commit already recorded for mutId ${mutId}`);
    }
    state.committed.add(mutId);
    state.baseCommitVersion += 1;
    this.trace.record({
      type: "base_commit",
      baseId,
      mutId,
      baseCommitVersion: state.baseCommitVersion,
    });
    return {
      baseCommitVersion: state.baseCommitVersion,
      committedMutations: Array.from(state.committed).sort((a, b) => a - b),
    };
  }

  _debug_simulateCommit(_plan: DebugPlan): DebugSimulatedState {
    const snapshot = this._debugSnapshot();
    const tokLocIds: Record<string, number> = {};
    for (const [locId, token] of Object.entries(snapshot.tokLoc)) {
      tokLocIds[locId] = token.id;
    }
    return {
      tokGlobalId: snapshot.tokGlobal.id,
      tokLocIds,
    };
  }

  private snapshotToken(token: Token): TokenSnapshot {
    return {
      id: token.id,
      key: token.key,
      kind: token.kind,
      roots: token.roots.slice(),
    };
  }

  private getOrCreateLocState(locId: LocId): LocDebugState {
    const existing = this.locState.get(locId);
    if (existing) {
      return existing;
    }
    const initial: LocDebugState = {
      locLogicalVersion: 0,
      locVersion: 0,
      role: "ephemeral",
      hasValue: false,
    };
    this.locState.set(locId, initial);
    return initial;
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

  private getBaseBinding(baseId: BaseId): BaseBinding {
    const binding = this.baseBindings.get(baseId);
    if (!binding) {
      throw new Error(`No binding for baseId ${baseId}`);
    }
    return binding;
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

  private finishStaging(result: unknown): void {
    const returnedIds = new Set<number>(collectTraceTensorIds(result));
    for (const traceId of this.currentStagingIds) {
      if (returnedIds.has(traceId)) {
        this.traceTensorStatus.set(traceId, "live");
      } else {
        this.traceTensorStatus.set(traceId, "stale");
      }
    }
    let graph = buildIRFromTrace(this.trace.snapshot(), this.currentEpoch);

    // Apply AMP transform if autocast is enabled (§12)
    // This implements the "select-gated commits" pattern
    let ampPolicyHash = "disabled";
    if (this.autocastContext?.current.enabled) {
      const ampResult = applyAMPTransform(graph, this.autocastContext);
      if (ampResult.modified) {
        graph = ampResult.graph;
      }
      ampPolicyHash = hashAMPPolicy(this.autocastContext.current.policy);
    }

    // Cache lookup and storage
    // Include AMP policy hash in the cache key for variant selection
    if (graph.nodes.length > 0) {
      const baseCacheKey = generateCacheKey(graph);
      // Extend cache key with AMP policy hash
      const cacheKey: CompiledCacheKey = {
        ...baseCacheKey,
        irHash: `${baseCacheKey.irHash}:amp=${ampPolicyHash}`,
      };
      const cached = this.compiledCache.get(cacheKey);

      if (cached) {
        // Cache hit - we could reuse the cached graph
        // For now, we just record the hit for statistics
        this.lastCacheHit = true;
        this.lastCacheKey = cacheKey;
        this.lastCompiledGraph = cached.graph;
      } else {
        // Cache miss - store the new graph
        this.compiledCache.set(cacheKey, graph);
        this.lastCacheHit = false;
        this.lastCacheKey = cacheKey;
        this.lastCompiledGraph = graph;
      }
    } else {
      this.lastCompiledGraph = null;
      this.lastCacheKey = null;
      this.lastCacheHit = false;
    }

    this.currentStagingIds.clear();
    this.stagingActive = false;
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

  private finalizePendingLocBindings(): number {
    let finalized = 0;
    for (const [baseId, binding] of this.baseBindings.entries()) {
      if (binding.kind !== "pending_loc" || binding.locId === undefined) {
        continue;
      }
      const locState = this.locState.get(binding.locId);
      if (!locState || !locState.hasValue) {
        continue;
      }
      this.baseBindings.set(baseId, {
        kind: "loc",
        locId: binding.locId,
      });
      finalized += 1;
    }
    return finalized;
  }

  private collectForcePlanTokens(baseId: BaseId): Token[] {
    const tokens: Token[] = [];
    const binding = this.baseBindings.get(baseId);
    if (binding) {
      if (binding.initTok) {
        tokens.push(binding.initTok);
      }
      if (
        (binding.kind === "pending_loc" || binding.kind === "loc") &&
        binding.locId !== undefined
      ) {
        const tokLoc = this.tokLoc.get(binding.locId);
        if (tokLoc) {
          tokens.push(tokLoc);
        }
      }
    }
    if (tokens.length === 0) {
      tokens.push(this.tokGlobal);
    }
    return tokens;
  }

  private emitEffectFrom(input: Token, op: string): Token {
    const tokOut = this.tokenStore.createEffectToken();

    this.trace.record({
      type: "effect",
      op,
      input: input.id,
      output: tokOut.id,
    });

    this.tokGlobal = tokOut;
    this.trace.record({
      type: "set_token",
      target: "global",
      token: tokOut.id,
    });

    return tokOut;
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
    // Get GPU stats from provider if available
    const gpuStats = this.memoryStatsProvider?.getGPUStats?.() ?? {
      currentBytes: 0,
      peakBytes: 0,
      limitBytes: 0,
    };

    // Get buffer pool stats from provider if available
    const poolStats = this.memoryStatsProvider?.getBufferPoolStats?.() ?? {
      pooledBuffers: 0,
      inUseBuffers: 0,
      pendingFenceBuffers: 0,
    };

    // Get plan stats from provider if available
    const planStats = this.memoryStatsProvider?.getPlanStats?.() ?? {
      activePlans: 0,
      completedPlans: 0,
    };

    // Get pending tensor count from provider if available
    const pendingTensorCount =
      this.memoryStatsProvider?.getPendingTensorCount?.() ?? 0;

    // Compute total pin count
    let totalPinCount = 0;
    for (const count of this.basePinCount.values()) {
      totalPinCount += count;
    }

    return {
      gpuCurrentBytes: gpuStats.currentBytes,
      gpuPeakBytes: gpuStats.peakBytes,
      gpuLimitBytes: gpuStats.limitBytes,
      pooledBuffers: poolStats.pooledBuffers,
      inUseBuffers: poolStats.inUseBuffers,
      pendingFenceBuffers: poolStats.pendingFenceBuffers,
      activeBases: this.basePinCount.size,
      totalPinCount,
      savedTensorCount: this.savedTensors.size,
      pendingTensorCount,
      activePlans: planStats.activePlans,
      completedPlans: planStats.completedPlans,
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
      const binding = this.baseBindings.get(baseId);
      const state = this.baseState.get(baseId);

      let locId: number | null = null;
      let hasValue = false;

      if (binding?.locId !== undefined) {
        locId = binding.locId;
        const locState = this.locState.get(binding.locId);
        hasValue = locState?.hasValue ?? false;
      }

      infos.push({
        baseId,
        pinCount: count,
        binding: binding?.kind ?? "ssa",
        locId,
        hasValue,
        commitVersion: state?.baseCommitVersion ?? 0,
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

