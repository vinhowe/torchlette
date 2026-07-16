import { getBackend } from "../backend/registry";
import type {
  AdamStepConfig,
  Backend,
  BackendTensor,
  DType,
} from "../backend/types";
import { isFusedBackend } from "../backend/types";
import {
  arenaLivenessEnabled,
  type BufferArena,
  beginSharedEncoder,
  clearActiveArena,
  clearArenaConflictDetected,
  clearArenaExternalInputBuffers,
  compiledPlannedEnabled,
  destroyArena,
  endSharedEncoder,
  flushBufferPool,
  flushSharedEncoder,
  getArenaConflictDetected,
  setActiveArena,
  setArenaExternalInputBuffers,
  setCurrentOpLabel,
} from "../backend/webgpu";
import { getDispatchSequenceCounters } from "../backend/webgpu/bind-group-cache";
import { bufferPool } from "../backend/webgpu/buffer-pool";
import {
  asGPUTensor,
  type GPUBuffer,
  gpuBuffer,
} from "../backend/webgpu/gpu-types";
import type { EpilogueConfig } from "../backend/webgpu/matmul/types";
import {
  reshapeMeta,
  VIEW_META_OPS,
  viewResultMeta,
} from "../backend/webgpu/ops/view-meta";
import { detectSimpleTranspose } from "../backend/webgpu/ops/views";
import { dtypeBytes } from "../backend/webgpu/shape-utils";
import type { FusionGroup } from "../compiler/fusion-detect";
import {
  buildIdPositionMap,
  computePlanFingerprint,
  type ExecutionSegment,
  groupToRecipe,
  inlinedConstantValue,
  isFusibleOp,
  mergeIslands,
  type PlanFingerprint,
  type PlanPartition,
  reifyPartition,
} from "../compiler/fusion-detect";
import { analyzeGraph } from "../compiler/graph-compiler";
import { runPasses, SIMPLIFICATION_PASSES } from "../compiler/graph-rewrites";
import type { MatmulPrologueInfo } from "../compiler/matmul-epilogue";
import {
  _detectTransposeView,
  executeMatmulWithEpilogue,
} from "../compiler/matmul-epilogue";
import { ENV } from "../core/env";
import {
  checkContiguous,
  contiguousStrides,
  shapesEqual,
  sizeOf,
} from "../core/shape";
import {
  _webgpuMatmulGeomImports,
  _webgpuMatmulImports,
  createStorageHandle,
  ensureWebGPUMatmulImports,
  wrapResultAsStorage,
} from "../graph/node-factory";
import {
  getPlanAnalysisGeneration,
  isProfilingEnabled,
  type PlanAnalysis,
  profileOpBegin,
  profileOpEnd,
  recordPlanAnalysis,
  setProfileModule,
} from "../graph/profiler";
import {
  canSafelyRelease,
  releaseBufferImmediate,
  storageTracker,
} from "../graph/storage-tracker";
import type {
  ExecutionPlan,
  LazyIRNode,
  LazyRef,
  StorageHandle,
} from "../graph/types";
import { diffSegmentsAligned, diffStreams } from "./stream-diff";
import {
  buildDirectOffsetRepack,
  type GeneratedStream,
  generateStream,
} from "./stream-generate";

/** Mark a storage's GPU buffer as liveness-safe for immediate pool reuse. */
function markLivenessSafe(storage: StorageHandle): void {
  const buf = (storage.backendTensor as { buffer?: GPUBuffer }).buffer;
  if (buf) bufferPool.markLivenessSafe(buf);
}

import {
  getLivePendingNodeIds,
  getLivePendingRootNodes,
} from "../runtime/tensor";
import {
  assignSlot,
  buildCompiledPlan,
  buildCompiledPlanFromGenerated,
  type CompiledPlan,
  checkpointBoundaryIndices,
  debugCompiledPlanEntryBytes,
  debugEntryBufferIndex,
  destroyCompiledPlanBuffers,
  executeCompiledPlan,
  isCompilationRecordingActive,
  type NodeResult,
  plannerEntryBytes,
  plannerEntryClaimable,
  recordBarrier,
  recordWrite,
  resetPlannerRegistry,
  setPendingParamsVolatilePack,
  setRecordingNodeIndex,
  startCompilationRecording,
  stopCompilationRecording,
} from "./compiled-plan";
import {
  buildLoweredPlanFromAnalysis,
  getActionNodeIndices,
  type LoweredPlan,
} from "./lowered-plan";
import {
  clearResultEntries,
  getObservedLivenessStats,
  isObservedLivenessEnabled,
  noteInPlaceCommit,
  noteNewTemplate,
  noteTemplateExecuted,
  prunedHarvest,
  registerResultEntries,
  releasableEntryReader,
  releasableLastReader,
  resetObservedLiveness,
  resultEntryFor,
  setObservedLivenessEnabled,
  setTemplateCompiledInvalidator,
  setTemplateIdleRetirer,
  stampResult,
} from "./observed-liveness";

/** [observed-liveness] Node ops that commit in-place mutations to persistent
 *  state. A plan containing any advances the guard's in-place-committed counter
 *  so a pruned-producer recompute against mutated storages fails loudly. */
const IN_PLACE_COMMIT_OPS: ReadonlySet<string> = new Set([
  "adamStep",
  "stridedScatterCopy",
  "stridedScatterAdd",
]);

import {
  crossPlanEdgeHasOtherConsumer,
  crossPlanEdgeKeepSet,
} from "../core/cross-plan-edges";
import {
  STEP_TAPE_RECORD,
  STEP_TAPE_REPLAY,
  stBeginPlan,
  stDeclareBatchCover,
  stEndPlan,
} from "../core/step-tape";
import { currentVariantSelection, variantToken } from "../core/step-variant";
import { TAPE_PROFILE, tpAdd } from "../core/tape-profile";
import {
  assignNodeResult,
  executeOpSync,
  getInputStorage,
  withProfileContext,
} from "./op-dispatch";
import { pretunePlanMatmuls } from "./plan-builder";
import {
  clearActiveScalarTable,
  destroyScalarTable,
  refreshScalarTable,
} from "./scalar-table";
import {
  ensureFusionImports,
  executeFusedSegment,
  executeRowProgram,
} from "./segment-executors";
import { executePlanSequential } from "./sequential";
import { stCaptureCompiledStep } from "./step-tape-replay";

// ============================================================================
// Optimized Execution Types
// ============================================================================

/**
 * Options for optimized plan execution.
 */
interface OptimizedExecutionOptions {
  /** Enable elementwise fusion (default: true for WebGPU) */
  enableFusion?: boolean;
  /** Enable vectorization for fused kernels (default: true) */
  enableVectorization?: boolean;
  /** Enable early buffer release based on lifetime analysis */
  enableEarlyRelease?: boolean;
  /**
   * Run arena-free (pool buffers) so the liveness early-release can actually
   * free intermediates instead of the per-step arena retaining them. Set by
   * the engine for checkpointed training (see RuntimeEngine.setBufferArenaDisabled).
   */
  arenaDisabled?: boolean;
}

/**
 * Statistics from optimized execution.
 */
export interface OptimizedExecutionStats {
  totalNodes: number;
  fusedNodes: number;
  sequentialNodes: number;
  fusionGroups: number;
  fusionEnabled: boolean;
}

/**
 * Result of optimized execution.
 */
export interface OptimizedExecutionResult {
  result: StorageHandle;
  stats: OptimizedExecutionStats;
}

// ============================================================================
// Fusion Analysis Cache
// ============================================================================

/**
 * Cached fusion analysis template. Stores the analysis result as position-based
 * indices that can be applied to any plan with the same structural fingerprint.
 */
export interface FusionAnalysisTemplate {
  /** Maps final plan position → original plan position.
   *  finalPlan[i] = originalPlan[finalPerm[i]] */
  finalPerm: number[];

  /** Segment pattern using positions in the final plan. */
  segments: CachedSegmentDesc[];

  /**
   * [islands I0] The plan's dispatch-partition as first-class data, reified
   * from `segments` (same positions, same emission order). Derived today —
   * one partition per graph; the partition-identity token
   * (`partition.boundaryHash`) is what stage I1 keys the template cache by
   * when a non-default partition is requested. See docs/islands-design.md.
   */
  partition: PlanPartition;

  /** Original plan positions that are epilogue-claimed. */
  epilogueClaimedOrigPoss: number[];
  /** Original plan positions that are prologue-claimed. */
  prologueClaimedOrigPoss: number[];

  /** Matmul epilogue chains: [origPos, [epilogueOrigPoss]]. */
  epilogueChains: Array<[number, number[]]>;

  /** Matmul prologues: [origPos, [{inputIndex, castOrigPos, fromDtype, toDtype}]]. */
  prologueDescs: Array<
    [
      number,
      Array<{
        inputIndex: 0 | 1;
        castOrigPos: number;
        fromDtype: DType;
        toDtype: DType;
      }>,
    ]
  >;

  /** Cached lifetime analysis (position-based). */
  lifetimeTemplate?: Array<{
    firstUse: number;
    lastUse: number;
    isOutput: boolean;
    isInput: boolean;
    bufferSize: number;
  }>;

  /** Cached lowered execution plan (built from graph analysis). */
  loweredPlan?: LoweredPlan;

  /** Per-plan buffer arena: GPUBuffers that persist across steps for bind group cache stability. */
  bufferArena?: unknown;

  /** Cached plan analysis for profiling (stored on first miss, replayed on hits). */
  planAnalysis?: PlanAnalysis;
  /** Generation at which planAnalysis was last replayed (prevents per-step accumulation). */
  replayedGeneration?: number;
  /** Secondary fingerprint hash — used for collision detection on cache hit. */
  fingerprintSecondary: number;

  /** [S3 FUSE LIVE WIRING] When set, this template is an ALIAS of the template
   *  keyed by `aliasOf` (the DEFAULT fingerprint): it SHARES that template's
   *  `loweredPlan` + `bufferArena` (the execution artifact) and differs only in
   *  `partition` (merged) + `fingerprintSecondary`. This makes a grouping-only
   *  partition merge byte-identical in execution to the default (the numerics
   *  null) while re-keying the template + re-witnessing the tape. Its shared
   *  resources are OWNED by the default template — teardown/eviction of an alias
   *  MUST NOT destroy them (guarded below). */
  aliasOf?: number;
}

type CachedSegmentDesc =
  | { kind: "sequential"; finalPoss: number[] }
  | {
      kind: "fused";
      /** All group node positions in final plan. */
      finalPoss: number[];
      /** Output node position in final plan. */
      outputFinalPos: number;
      /** Additional output positions in final plan. */
      additionalOutputFinalPoss: number[];
      /** Needed intermediate positions in final plan. */
      neededIntermediateFinalPoss: number[];
    }
  | {
      kind: "reduction";
      /** All group node positions in final plan. */
      finalPoss: number[];
      /** Reduction node position. */
      reductionFinalPos: number;
      /** Preamble node positions. */
      preambleFinalPoss: number[];
      /** Epilogue node positions. */
      epilogueFinalPoss: number[];
      /** Output node position. */
      outputFinalPos: number;
      /** Serialized preamble ops. */
      preambleOps: Array<{ op: string; arity: number; chainInputPos?: 0 | 1 }>;
      /** Preamble input dtypes. */
      preambleInputDtypes: DType[];
      /** Serialized epilogue ops. */
      epilogueOps: Array<{
        kind: string;
        toDtype?: DType;
        inputIndex?: number;
        op?: string;
      }>;
      /** Output dtype. */
      outputDtype: DType;
      /** Whether this is a mean reduction. */
      isMean: boolean;
    };

/**
 * Module-level cache for fusion analysis results.
 * Keyed by structural fingerprint (FNV-1a hash).
 * Typically holds <10 entries (one per unique plan structure).
 */
const fusionAnalysisCache = new Map<number, FusionAnalysisTemplate>();

// ============================================================================
// [S3 FUSE LIVE WIRING] The editor's accepted-merge REQUEST store.
//
// The `StepEditChannel.applyMerge` hook (docs/step-object-design.md §5.3) binds
// to `applyPartitionMerge` below: an ACCEPTED requestMerge records a directive
// here, keyed by the plan's DEFAULT execution fingerprint (the token-free
// `computePlanFingerprint` primary — a pure, per-step-stable function of the
// static graph). NOTHING here owns membership: the store holds the WHAT (fuse
// islands a,b of the plan whose default fp is F); the DETECTOR'S OWN merge
// (`mergeIslands`, fusion-detect.ts) computes the merged partition when the
// executor next runs plan F. This is the fuse.ts §1.6 discipline lifted live:
// record-request-not-mutate, one owner of membership (the detector).
//
// Effect chain (§5.3): a directive changes the plan's fingerprint (its merged
// boundaryHash mixes in via `computePlanFingerprint`'s I1 `partitionToken`) →
// the template re-fingerprints (cache MISS) → the tape's bucketKey (built from
// the ordered plan fps) changes → the StepObject's BucketMiss fires → the next
// executed steps re-witness a fresh skeleton under the merged partition.
//
// The NON-edited path is BYTE-IDENTICAL: with an empty store, `edit === undefined`
// and the executor computes the default fingerprint exactly as before (no token,
// no re-key, no reification change) — zero behavioral delta by construction.
// ---------------------------------------------------------------------------
interface PartitionMergeRequest {
  /** Island indices (emission order) to fuse — the detector's merge input. */
  readonly aIdx: number;
  readonly bIdx: number;
  /** The merged partition's boundaryHash, cached after the first realization so
   *  the edited fingerprint reproduces byte-stably every subsequent step without
   *  re-deriving the default partition. Undefined until the first edited run. */
  token?: number;
  /** The edited primary fingerprint (the plan's re-keyed template id), stored when
   *  the edited fp first goes active — the read-only handle a probe uses to inspect
   *  the merged template's partition (island count −1). Undefined until active. */
  editedFp?: number;
}

/** default-primary-fp → the accepted merge for that plan (at most one merge per
 *  plan in v1 — binary fuse, fuse.ts §5 ruling 1). Empty ⇒ zero behavior change. */
const editedPartitions = new Map<number, PartitionMergeRequest>();

/**
 * [S3] Read-only accessor: the DETECTOR'S reified partition for the plan whose
 * default fingerprint is `fp` (from the executor's own template cache). The
 * editor binding reads island structure through this — it does NOT own or copy
 * membership; the detector remains the single owner. Undefined until the plan
 * has executed once (been lowered + cached).
 */
export function getCachedPlanPartition(fp: number): PlanPartition | undefined {
  return fusionAnalysisCache.get(fp >>> 0)?.partition;
}

/**
 * [S3] Read-only: the MERGED partition realized for the plan whose default fp is
 * `defaultFp` (the edited template's reified partition — island count one fewer
 * than the default). Undefined until the edit is active + witnessed. Probes/tests
 * use this to assert the merged partition actually runs.
 */
export function getEditedPlanPartition(
  defaultFp: number,
): PlanPartition | undefined {
  const edit = editedPartitions.get(defaultFp >>> 0);
  if (!edit || edit.editedFp === undefined) return undefined;
  return fusionAnalysisCache.get(edit.editedFp)?.partition;
}

/**
 * [S3] The live wiring seam the `StepEditChannel.applyMerge` hook binds to.
 * Records an accepted merge of islands (aIdx, bIdx) for the plan whose DEFAULT
 * fingerprint is `defaultFp`. The DETECTOR'S OWN merge (`mergeIslands`) validates
 * it against the executor's reified partition; a structurally-illegal merge
 * (non-convex / out of range) or a plan not yet cached THROWS a typed refusal —
 * the channel catches the throw and returns `MERGE_REFUSED` with atomic rollback
 * (step-edit-channel.ts). Membership is untouched here; the detector realizes the
 * merge on the next run of plan `defaultFp`.
 */
export function applyPartitionMerge(
  defaultFp: number,
  aIdx: number,
  bIdx: number,
): void {
  const partition = fusionAnalysisCache.get(defaultFp >>> 0)?.partition;
  if (!partition) {
    throw new Error(
      `partition merge refused: no cached partition for plan fp 0x${(defaultFp >>> 0).toString(16)} (run the plan before editing).`,
    );
  }
  const check = mergeIslands(partition, aIdx, bIdx);
  if (!check.ok) {
    throw new Error(
      `partition merge(${aIdx},${bIdx}) is illegal: ${check.code}`,
    );
  }
  // Record the DIRECTIVE (not the membership). Token is cached lazily on the
  // first edited run (the detector re-derives the merged partition there).
  editedPartitions.set(defaultFp >>> 0, { aIdx, bIdx });
}

/**
 * [S3] Rollback: discard the recorded merge for `defaultFp` so the plan
 * re-witnesses under the DEFAULT partition (the inverse `split` back to the
 * original cut — apply∘inverse = identity at this altitude). No-op for an
 * unrecorded fp.
 */
export function rollbackPartitionMerge(defaultFp: number): void {
  editedPartitions.delete(defaultFp >>> 0);
}

/** Test/debug: the count of live partition-merge directives (0 ⇒ null path). */
export function getEditedPartitionCount(): number {
  return editedPartitions.size;
}

/**
 * [S3] Resolve an edited fingerprint for a plan. Given the DEFAULT fingerprint
 * (token-free), consult the merge store and, when a merge with a KNOWN token is
 * recorded, recompute the fingerprint mixing the merged partition's boundaryHash
 * as the I1 `partitionToken` — the plan re-fingerprints so its template re-keys
 * and its tape re-witnesses (§5.3). Returns the (possibly re-computed) fingerprint
 * plus the live request (or undefined). The token is UNKNOWN until the first
 * edited run realizes the default partition (bootstrapped in the miss path), so
 * the first post-accept step runs the default fp and the merge takes effect on
 * the next — exactly "the next executed steps re-witness".
 */
function resolveEditedFingerprint(
  nodes: LazyIRNode[],
  externalNodeIds: Set<number> | undefined,
  defaultFp: PlanFingerprint,
): {
  fingerprint: PlanFingerprint;
  edit: PartitionMergeRequest | undefined;
  active: boolean;
} {
  const edit = editedPartitions.get(defaultFp.primary >>> 0);
  if (!edit) return { fingerprint: defaultFp, edit: undefined, active: false };
  // Bootstrap the merged token from the DETECTOR'S cached default partition (the
  // plan was lowered at least once — its partition is in the cache). This makes
  // the edit take effect on the FIRST step after acceptance, whether that step
  // hits or misses the template cache (the warmed-plan case: it hits, so the
  // token can only come from the cached partition, not a fresh analysis).
  if (edit.token === undefined) {
    const defaultPartition = fusionAnalysisCache.get(
      defaultFp.primary >>> 0,
    )?.partition;
    if (defaultPartition) {
      const m = mergeIslands(defaultPartition, edit.aIdx, edit.bIdx);
      if (m.ok) edit.token = m.partition.boundaryHash;
    }
  }
  if (edit.token !== undefined) {
    const fingerprint = computePlanFingerprint(
      nodes,
      externalNodeIds,
      edit.token,
    );
    edit.editedFp = fingerprint.primary;
    return { fingerprint, edit, active: true };
  }
  return { fingerprint: defaultFp, edit, active: false };
}

// ---------------------------------------------------------------------------
// Payload-thrash detector: structural hash → how many DISTINCT full
// fingerprints were seen for it. A growing count means plans that are
// structurally identical keep missing the template cache because only
// PAYLOAD VALUES differ — a per-step-varying payload defeating caching
// (re-lowering/re-recording every step). The engine reports it; nobody has
// to predict per-op which payloads change. Fix direction in the warning.
// ---------------------------------------------------------------------------
const structuralMissCounts = new Map<number, Set<number>>();
const structuralWarned = new Set<number>();
const PAYLOAD_THRASH_THRESHOLD = 4;

function notePayloadThrash(
  fingerprint: { primary: number; structural: number },
  planNodes: LazyIRNode[],
): void {
  // Count DISTINCT full fingerprints per structural hash: an evicted
  // template re-missing with the SAME full fingerprint is re-lowering but
  // not payload thrash. Set growth = new payload variants.
  let seen = structuralMissCounts.get(fingerprint.structural);
  if (!seen) {
    seen = new Set();
    structuralMissCounts.set(fingerprint.structural, seen);
  }
  if (seen.size <= PAYLOAD_THRASH_THRESHOLD) seen.add(fingerprint.primary);
  const n = seen.size;
  if (
    n < PAYLOAD_THRASH_THRESHOLD ||
    structuralWarned.has(fingerprint.structural)
  ) {
    return;
  }
  structuralWarned.add(fingerprint.structural);
  const payloadOps = [
    ...new Set(planNodes.filter((nd) => nd.payload).map((nd) => nd.op)),
  ].join(", ");
  console.warn(
    `[template] PAYLOAD THRASH: ${n} structurally-identical plans missed the template cache with differing payload values (payload ops: ${payloadOps}). A per-step-varying op option is defeating caching — every step re-lowers and re-records. Fix: pass the value as a tensor/graph scalar (principled path), or lower the option to graph ops at the runtime seam (see runtime.sub alpha), or declare it volatile in PAYLOAD_HASH_EXEMPT WITH a per-execution delivery mechanism.`,
  );
}

/** Test/debug: distinct-payload template misses per structural hash, plus the
 *  observed-liveness guard telemetry (folded here per the telemetry disposition
 *  — no new export). `dirtyMisses` is THE measurement deciding whether stage-3
 *  rematerialization ever needs to serve the build-from-IR pruning path. */
export function getPayloadThrashStats(): {
  structures: number;
  worst: number;
  cleanMisses: number;
  dirtyMisses: number;
  pinnedTemplates: number;
  convergedTemplates: number;
  prunedPairsRemoved: number;
  retiredTemplates: number;
} {
  let worst = 0;
  for (const v of structuralMissCounts.values())
    if (v.size > worst) worst = v.size;
  return {
    structures: structuralMissCounts.size,
    worst,
    ...getObservedLivenessStats(),
  };
}

/** Get a cached fusion analysis template by fingerprint. */
export function getFusionAnalysisTemplate(
  fingerprint: number,
): FusionAnalysisTemplate | undefined {
  return fusionAnalysisCache.get(fingerprint);
}

/** Test/debug: number of DISTINCT lowered-plan templates currently cached
 *  (one per unique plan fingerprint). The task #71 measurement: N distinct
 *  view offsets that fork the template identity show up here as N entries;
 *  after routing the offset through a volatile uniform they collapse to 1. */
export function debugTemplateCount(): number {
  return fusionAnalysisCache.size;
}

/** Test/debug: for every cached template with a valid compiled plan, the SET of
 *  harvested result identities ("nodeIndex:outputIndex"), keyed by fingerprint
 *  hex. The observed-liveness set-parity gate compares this map across the
 *  build-from-IR-pruned run and the recorded-cutover run — the pruned result set
 *  must equal the cutover's live-result survivor set for every recurring
 *  template (single source at the seam, assert agreement). */
export function debugTemplateResultSets(): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const [fp, template] of fusionAnalysisCache) {
    const cp = template.loweredPlan?.compiledPlan;
    if (!cp?.valid) continue;
    out[`0x${fp.toString(16)}`] = cp.results
      .map((r) => `${r.nodeIndex}:${r.outputIndex}`)
      .sort();
  }
  return out;
}

/** Test/debug: per-cached-template compiled-plan memory footprint (planner
 *  registry entry bytes split result/temp) + harvest shape (results kept,
 *  pairs pruned). The stage-1 residual attribution instrument: comparing the
 *  per-template resultMB across pruned/cutover arms names which templates hold
 *  the residual. */
export function debugTemplatePlanMemory(): Record<
  string,
  {
    nodes: number;
    valid: boolean;
    results: number;
    pruned: number;
    resultMB: number;
    tempMB: number;
  }
> {
  const out: Record<
    string,
    {
      nodes: number;
      valid: boolean;
      results: number;
      pruned: number;
      resultMB: number;
      tempMB: number;
    }
  > = {};
  for (const [fp, template] of fusionAnalysisCache) {
    const cp = template.loweredPlan?.compiledPlan;
    const mem = cp
      ? debugCompiledPlanEntryBytes(cp)
      : { resultMB: 0, tempMB: 0, entries: 0 };
    out[`0x${fp.toString(16)}`] = {
      nodes: template.finalPerm.length,
      valid: !!cp?.valid,
      results: cp?.results.length ?? 0,
      pruned: cp?._prunedPairs?.length ?? 0,
      resultMB: mem.resultMB,
      tempMB: mem.tempMB,
    };
  }
  return out;
}

/** [task #99 R1] The declared checkpoint-recompute segment DATA now reaching
 *  the memory-planner input surface, per recompute-bearing template. For each
 *  template whose compiled plan carries a checkpoint boundary (`_hasRecompute`),
 *  reports the stamped `_recomputeSegments` — the checkpointed forward-activation
 *  results (nodeIndex, outputIndex, slot, entryIdx) the planner today pins
 *  whole-step and R2 will liveness-split. Visible/asserted-present surface for
 *  the R1 gate; R1 stamps DATA only (no planning-decision change). Debug-only. */
export function debugRecomputeSegments(): Record<
  string,
  {
    hasRecompute: boolean;
    segments: Array<{
      nodeIndex: number;
      outputIndex: number;
      slot: number;
      entryIdx?: number;
    }>;
  }
> {
  const out: Record<
    string,
    {
      hasRecompute: boolean;
      segments: Array<{
        nodeIndex: number;
        outputIndex: number;
        slot: number;
        entryIdx?: number;
      }>;
    }
  > = {};
  for (const [fp, template] of fusionAnalysisCache) {
    const cp = template.loweredPlan?.compiledPlan;
    if (!cp?._hasRecompute && !cp?._recomputeSegments) continue;
    out[`0x${fp.toString(16)}`] = {
      hasRecompute: !!cp?._hasRecompute,
      segments: (cp?._recomputeSegments ?? []).map((s) => ({
        nodeIndex: s.nodeIndex,
        outputIndex: s.outputIndex,
        slot: s.slot,
        entryIdx: s.entryIdx,
      })),
    };
  }
  return out;
}

/** [stage-3 S3.0 probe] Cross-plan PERSISTENT-slot bindings: for each valid
 *  compiled plan, the registry entries of OTHER plans that its `persistent`
 *  slots bind. Such reads resolve once at build (never per replay), so the
 *  observeConsumed seam is blind to them — a producer pair consumed only this
 *  way looks unconsumed forever. Debug-only. */
export function debugCrossPlanPersistentBindings(): Record<
  string,
  { count: number; MB: number; entries: number[] }
> {
  const bufIndex = debugEntryBufferIndex();
  const out: Record<string, { count: number; MB: number; entries: number[] }> =
    {};
  for (const [fp, template] of fusionAnalysisCache) {
    const cp = template.loweredPlan?.compiledPlan;
    if (!cp?.valid) continue;
    const own = new Set(cp.plannerEntries ?? []);
    let bytes = 0;
    const hit: number[] = [];
    for (const src of cp.slots) {
      if (src.kind !== "persistent") continue;
      const idx = bufIndex.get(src.buffer);
      if (idx !== undefined && !own.has(idx)) {
        hit.push(idx);
        bytes += plannerEntryBytes(idx);
      }
    }
    if (hit.length > 0) {
      out[`0x${fp.toString(16)}`] = {
        count: hit.length,
        MB: +(bytes / 1e6).toFixed(1),
        entries: hit.slice(0, 12),
      };
    }
  }
  return out;
}

/** Stage-2 FLIP (2026-07-08): build-from-IR is THE default build source —
 *  plans compile from IR on their FIRST execution, no recording. RECORDING
 *  survives only for:
 *   - verify modes: TORCHLETTE_STREAM_GENERATE=1 diffs the generated stream
 *     against a recording, so it keeps the recorded build (the stream-diff
 *     gates; the determinism gate pins BUILD_FROM_IR=0 explicitly);
 *   - uncovered plans (census-driven, per-plan): when generateStream cannot
 *     fully cover a plan the build-from-IR block falls through with zero
 *     residue and the plan records on its next execution (recorded cutover);
 *   - the opt-outs: BUILD_FROM_IR=0 (the recorded-replay reference; dies when
 *     the recorded build source is deleted) / COMPILED_PLAN=0 (the lowered
 *     reference), which must disable build-from-IR too or they'd measure
 *     nothing. (The GENERATED_PLAN flag died at stage-2 inc-3c, 2026-07-08:
 *     after inc-3a deleted the cutover swap it was an exact behavioral twin
 *     of BUILD_FROM_IR=0 — the named B5 sunset, executed.)
 *  Single predicate so the observation enable below and the build block can
 *  never disagree. */
function buildFromIRActive(): boolean {
  return (
    ENV.TORCHLETTE_BUILD_FROM_IR !== "0" &&
    ENV.TORCHLETTE_COMPILED_PLAN !== "0" &&
    ENV.TORCHLETTE_STREAM_GENERATE !== "1"
  );
}

// fp → number of times it reached the build-from-IR block. Distinguishes
// recurring templates (reach ≥2 ⇒ recurs, since a covered template
// caches+replays and never re-enters the block) from one-shot warmup graph
// variants (reach 1). Gates constFill coverage so a transient plan's `full`
// never triggers a compile that leaks its plan-owned buffer.
const buildReaches = new Map<number, number>();

// Per-fp count of CONSECUTIVE executions that ran LOWERED with no valid compiled
// plan (the "converged-to-lowered" witness criterion,
// docs/step-data-dependence-design.md §D4 attempt #6). A build-from-IR-UNCOVERED
// recurring producer (the forward+loss plan whose full-reduction sum has no
// generator) can only be compiled — hence STAMPED, hence witnessed — by the
// recorded build; once that build is gone it stays lowered forever and its
// cross-plan outputs are never stamped, so its consumers can never witness its
// edges (the D4 attempt-#5 hole: the recorded build was the STAMPING compiler,
// not merely a witnessing driver). Reproduce its witnessing coverage WITHOUT the
// recorded build: after a template has run lowered for K_w consecutive executions
// with no compiled plan, it has converged to lowered — stamp its harvest outputs
// on the lowered path so its edges are witnessed. The counter RESETS whenever the
// template holds a valid compiled plan, so a warmup-lowered-then-compiled plan
// never reaches the threshold — the mechanism is DORMANT wherever any compiler
// (build-from-IR or the recorded build) exists, and only fires for genuinely
// uncovered plans stranded lowered. Reuses the K_w=2 witness constant.
const loweredWitnessRuns = new Map<number, number>();

// [observed-liveness] The module observes cross-plan liveness only when
// build-from-IR is active (the over-harvest it fixes exists only there). Set
// once at module load; the flag also gates every observation hook to a no-op.
setObservedLivenessEnabled(buildFromIRActive());
// [D2] The step-tape → observed-liveness witnessed-harvest publisher wiring was
// DELETED. The derived crossPlanEdges object (published by the recorder at K_w=2)
// is now the sole harvest owner; observed-liveness reads it directly via
// `crossPlanEdgeKeepSet` (see prunedHarvest). No callback seam to wire.
// The guard invalidates a producer template's compiled plan when a late
// consumer misses its pruned output; route it through the cache here (the
// module can't import this file — circular).
setTemplateCompiledInvalidator((fp: number) => {
  const t = fusionAnalysisCache.get(fp);
  if (t?.loweredPlan?.compiledPlan) {
    destroyCompiledPlanBuffers(t.loweredPlan.compiledPlan);
    t.loweredPlan.compiledPlan = undefined;
  }
  clearResultEntries(fp);
});
// Idle-retire (observed-liveness step boundary): release the compiled plan of
// a template that stopped executing — one-shot warmup templates otherwise pin
// their conservative harvest's registry buffers forever. Gated on every
// storage the last replay's harvest exposed being ALREADY DESTROYED, so no
// live reader dangles; buffer destruction stays fence-gated
// (destroyCompiledPlanBuffers → deferredDestroy). A retired template that
// re-executes rebuilds from IR.
setTemplateIdleRetirer((fp: number) => {
  const t = fusionAnalysisCache.get(fp);
  const cp = t?.loweredPlan?.compiledPlan;
  if (!cp) return "none";
  if (cp._lastHarvestIds?.some((id) => !storageTracker.isDestroyed(id))) {
    return "live";
  }
  destroyCompiledPlanBuffers(cp);
  t!.loweredPlan!.compiledPlan = undefined;
  clearResultEntries(fp);
  return "retired";
});

/**
 * Destroy all buffer arenas across cached templates, freeing GPU memory.
 * The arenas will be rebuilt on the next execution (one slow step, then
 * back to compiled plan speed). Call between training rounds to prevent
 * unbounded GPU memory growth in long-running training.
 *
 * @param force - pass true from cross-session recycle paths to destroy
 *   buffers even if they appear "live" (the liveness state is stale
 *   between sessions and refusing to destroy leaks ~20 MiB per session).
 */
/**
 * Drop EVERY cached template — arenas, compiled plans, scalar tables, the
 * lot. Called by the RuntimeEngine constructor: the template cache is
 * module-global, and a NEW engine instance reusing templates lowered by a
 * previous instance is the cross-instance interference class — fingerprints
 * collide (tests reset node-ID counters; structurally identical models
 * collide regardless), but the cached lowered plan's node rewrites and
 * recorded buffers belong to the dead instance ("Input not ready:
 * transpose" in the second instance of a test file; the differential
 * harness went one-engine-per-process for the same reason). Instance
 * boundaries must be cache boundaries. Warm-start within ONE engine across
 * steps/sessions is unaffected.
 */
export function clearTemplateCacheForNewEngine(): void {
  // [#84] The storage tracker + refcount registry are module-global singletons
  // too. A previous engine's residual storages (its params / optimizer m/v / lr
  // tensors) linger until GC collects the owning tensors — at an unpredictable
  // time DURING the next engine's run — and their buffers are then released to
  // the SHARED pool MID-STEP of that run (the "released-to-pool mid-step"
  // corruption class), so the second-in-process run sporadically diverges. Orphan
  // them HERE, at construction, before the new engine allocates: dropping them
  // from the tracker makes the eventual GC-time destroy() a no-op, so no buffer
  // is re-pooled mid-step of the next run. See disposeAllForNewEngine.
  storageTracker.disposeAllForNewEngine();
  evictAllArenas(false);
  fusionAnalysisCache.clear();
  // Step-scoped shared planner registry is module-global like the template
  // cache: entries co-owned by the dead instance's plans must not seed the
  // new instance's plans. evictAllArenas released plan ownership above (so
  // buffers are already torn down cleanly); the reset bumps the generation
  // as a backstop for any plan object that escaped the cache walk.
  resetPlannerRegistry();
  // [observed-liveness] The per-template observation state keys by fingerprint;
  // a new instance's fingerprints could collide with the dead one's — drop it.
  resetObservedLiveness();
}

export function evictAllArenas(force = false): void {
  // Import is at top of file — destroyArena is already available
  for (const [, template] of fusionAnalysisCache) {
    // [S3] Aliases SHARE the default's arena/loweredPlan — the owner (non-alias)
    // entry tears them down; touching them here would double-destroy.
    if (template.aliasOf !== undefined) continue;
    if (template.bufferArena) {
      destroyArena(template.bufferArena, force);
      template.bufferArena = undefined;
    }
    // Also invalidate compiled plan (it depends on arena buffer identities)
    if (template.loweredPlan?.compiledPlan) {
      destroyCompiledPlanBuffers(template.loweredPlan.compiledPlan);
      template.loweredPlan.compiledPlan = undefined;
    }
    if (template.loweredPlan) destroyScalarTable(template.loweredPlan);
  }
}

/**
 * Invalidate compiled plans without destroying arena buffers.
 * Use after param buffers change (e.g., outer optimizer step) to force
 * recompilation on the next step. Lighter than evictAllArenas() — avoids
 * the transient OOM from arena rebuild.
 */
/** Stage-4 phase 0: snapshot all live compiled streams (for stream diffing). */
export function getCompiledStreams(): Array<{
  label: string;
  commands: import("./compiled-plan").GpuCommand[];
}> {
  const out: Array<{
    label: string;
    commands: import("./compiled-plan").GpuCommand[];
  }> = [];
  for (const [fp, template] of fusionAnalysisCache) {
    if (template.aliasOf !== undefined) continue; // [S3] shared cp — count once
    const cp = template.loweredPlan?.compiledPlan;
    if (cp?.valid) {
      out.push({
        label: `fp=0x${fp.toString(16)}`,
        commands: cp.commands.slice(),
      });
    }
  }
  // Deterministic order for cross-snapshot comparison.
  out.sort((a, b) => (a.label < b.label ? -1 : 1));
  return out;
}

/** Stage-4 phase 0: drop compiled plans but KEEP templates/arenas — the next
 *  execution re-records the same template (the determinism gate's lever). */
export function invalidateCompiledKeepTemplates(): void {
  for (const [, template] of fusionAnalysisCache) {
    if (template.aliasOf !== undefined) continue; // [S3] shared — owner handles it
    if (template.loweredPlan?.compiledPlan) {
      destroyCompiledPlanBuffers(template.loweredPlan.compiledPlan);
      template.loweredPlan.compiledPlan = undefined;
    }
  }
}

export function invalidateCompiledPlans(): void {
  for (const [, template] of fusionAnalysisCache) {
    if (template.aliasOf !== undefined) continue; // [S3] shared — owner handles it
    if (template.loweredPlan?.compiledPlan) {
      destroyCompiledPlanBuffers(template.loweredPlan.compiledPlan);
      template.loweredPlan.compiledPlan = undefined;
    }
    if (template.loweredPlan) destroyScalarTable(template.loweredPlan);
    // Clear arena too — its bind groups reference stale param buffers
    if (template.bufferArena) {
      destroyArena(template.bufferArena);
      template.bufferArena = undefined;
    }
  }
}

/** Collect GPU buffers from all external (materialized/already-resolved) plan inputs. */
/**
 * Check whether any INLINED scalar constant in a fused-action recipe differs
 * from the current step's value. Cheap scan: only fused actions with a built
 * external-input pattern (always true once a compiled plan exists) and only
 * their inlined-constant inputs. Used to invalidate a compiled plan whose
 * recorded kernels baked a value that has since changed.
 */
function inlinedFusionScalarsStale(
  loweredPlan: LoweredPlan,
  planNodes: LazyIRNode[],
): boolean {
  for (const action of loweredPlan.actions) {
    if (action.kind !== "fused") continue;
    const pattern = action.cachedExternalInputPattern;
    if (!pattern) continue;
    const rin = action.recipe.inputs;
    for (let i = 0; i < rin.length && i < pattern.length; i++) {
      const inp = rin[i];
      if (!inp?.isInlinedConstant) continue;
      const p = pattern[i];
      const node = planNodes[action.coveredNodeIndices[p.nodeLocalIdx]];
      const ref = node?.inputs[p.inputIdx];
      if (!ref) continue;
      // Read the CURRENT value payload-first (materialized or not): by this
      // point data-sources may already carry results, and the old
      // isInlinableScalar-based read silently skipped them (frozen recipe).
      const cur = inlinedConstantValue(ref);
      if (cur !== null && cur !== inp.inlinedValue) return true;
    }
  }
  return false;
}

export function collectExternalInputBuffers(
  planNodes: LazyIRNode[],
): GPUBuffer[] {
  const bufs: GPUBuffer[] = [];
  for (const node of planNodes) {
    for (const ref of node.inputs) {
      if (ref.kind === "materialized") {
        const buf = gpuBuffer(ref.storage.backendTensor);
        if (buf) bufs.push(buf);
      } else if (ref.kind === "pending" && ref.node.result) {
        const buf = gpuBuffer(ref.node.result.backendTensor);
        if (buf) bufs.push(buf);
      }
    }
  }
  return bufs;
}

// ============================================================================
// Matmul-epilogue slow-path helpers (extracted for readability)
// ============================================================================

type PlanNodePath = { planNodeIndex: number; inputIndex: number };

/**
 * Walk the matmul-epilogue covered node chain to find external epilogue inputs.
 * For each add/mul in the chain, identifies which input comes from the chain
 * vs. which is an external input (e.g., bias tensor).
 */
function reconstructEpilogueInputs(
  planNodes: LazyIRNode[],
  coveredNodeIndices: number[],
): { epilogueInputRefs: LazyRef[]; epilogueInputPaths: PlanNodePath[] } {
  const epilogueInputRefs: LazyRef[] = [];
  const epilogueInputPaths: PlanNodePath[] = [];
  for (let ci = 1; ci < coveredNodeIndices.length; ci++) {
    const chainNode = planNodes[coveredNodeIndices[ci]];
    if (
      (chainNode.op === "add" || chainNode.op === "mul") &&
      chainNode.inputs.length === 2
    ) {
      const prevChainNodeId = planNodes[coveredNodeIndices[ci - 1]].id;
      const inp0IsChain =
        chainNode.inputs[0].kind === "pending" &&
        chainNode.inputs[0].node.id === prevChainNodeId;
      const externalIdx = inp0IsChain ? 1 : 0;
      epilogueInputRefs.push(chainNode.inputs[externalIdx]);
      epilogueInputPaths.push({
        planNodeIndex: coveredNodeIndices[ci],
        inputIndex: externalIdx,
      });
    }
  }
  return { epilogueInputRefs, epilogueInputPaths };
}

/**
 * Resolve prologue cast decisions: for each prologue, check whether the cast
 * already ran (fused in an earlier group). If not, redirect the matmul input
 * to the pre-cast tensor and tell codegen to cast inline.
 */
function resolvePrologueInputs(
  planNodes: LazyIRNode[],
  matmulNode: LazyIRNode,
  matmulNodeIndex: number,
  actionPrologues: Array<{
    inputIndex: 0 | 1;
    castNodeIndex: number;
    fromDtype: DType;
    toDtype: DType;
  }>,
): {
  prologues: MatmulPrologueInfo[];
  inputCastA: "f16" | "f32" | undefined;
  inputCastB: "f16" | "f32" | undefined;
  resolvedInputRefA: LazyRef;
  resolvedInputRefB: LazyRef;
  inputAPath: PlanNodePath;
  inputBPath: PlanNodePath;
} {
  let inputCastA: "f16" | "f32" | undefined;
  let inputCastB: "f16" | "f32" | undefined;
  let resolvedInputRefA = matmulNode.inputs[0];
  let resolvedInputRefB = matmulNode.inputs[1];
  let inputAPath: PlanNodePath = {
    planNodeIndex: matmulNodeIndex,
    inputIndex: 0,
  };
  let inputBPath: PlanNodePath = {
    planNodeIndex: matmulNodeIndex,
    inputIndex: 1,
  };

  const prologues = actionPrologues.map((p) => ({
    inputIndex: p.inputIndex,
    castNodeId: planNodes[p.castNodeIndex].id,
    originalInputRef: planNodes[p.castNodeIndex].inputs[0],
    fromDtype: p.fromDtype,
    toDtype: p.toDtype,
  }));

  for (const p of actionPrologues) {
    const castRef =
      p.inputIndex === 0 ? matmulNode.inputs[0] : matmulNode.inputs[1];
    const castAlreadyRan =
      castRef.kind === "pending" && castRef.node.result != null;
    if (!castAlreadyRan) {
      if (p.inputIndex === 0) {
        resolvedInputRefA = planNodes[p.castNodeIndex].inputs[0];
        inputCastA = p.toDtype as "f16" | "f32";
        inputAPath = { planNodeIndex: p.castNodeIndex, inputIndex: 0 };
      } else {
        resolvedInputRefB = planNodes[p.castNodeIndex].inputs[0];
        inputCastB = p.toDtype as "f16" | "f32";
        inputBPath = { planNodeIndex: p.castNodeIndex, inputIndex: 0 };
      }
    }
  }

  return {
    prologues,
    inputCastA,
    inputCastB,
    resolvedInputRefA,
    resolvedInputRefB,
    inputAPath,
    inputBPath,
  };
}

/**
 * Compute matmul geometry from resolved inputs and store the dispatch config
 * on the action for fast-path replay on subsequent steps.
 */
async function buildAndCacheDispatchConfig(
  action: {
    cachedDispatchConfig?: unknown;
    epilogueOps: unknown[];
    outputDtype?: DType;
  },
  resolvedInputRefA: LazyRef,
  resolvedInputRefB: LazyRef,
  inputCastA: "f16" | "f32" | undefined,
  inputCastB: "f16" | "f32" | undefined,
  inputAPath: PlanNodePath,
  inputBPath: PlanNodePath,
  epilogueInputRefs: LazyRef[],
  epilogueInputPaths: PlanNodePath[],
): Promise<void> {
  const { computeMatmulOutputShape, computeBatchSize, computeBatchStrides } =
    _webgpuMatmulGeomImports as NonNullable<typeof _webgpuMatmulGeomImports>;
  const { isF16Supported } = await import("../backend/webgpu/index");

  const tensorA = asGPUTensor(getInputStorage(resolvedInputRefA).backendTensor);
  const tensorB = asGPUTensor(getInputStorage(resolvedInputRefB).backendTensor);

  const detA = _detectTransposeView(tensorA);
  const detB = _detectTransposeView(tensorB);
  const transA = detA.transposed;
  const transB = detB.transposed;

  const outShape = computeMatmulOutputShape(
    detA.shape,
    detB.shape,
    transA,
    transB,
  );
  const aRank = detA.shape.length;
  const bRank = detB.shape.length;
  const m = transA ? detA.shape[aRank - 1] : detA.shape[aRank - 2];
  const k = transA ? detA.shape[aRank - 2] : detA.shape[aRank - 1];
  const n = transB ? detB.shape[bRank - 2] : detB.shape[bRank - 1];

  const batchDims = outShape.slice(0, -2);
  const batchSize = computeBatchSize(batchDims);
  const { strideA, strideB, strideC } = computeBatchStrides(
    detA.shape,
    detB.shape,
    batchDims,
    m,
    n,
    k,
  );

  const f16ok = isF16Supported();
  const rawDtypeA =
    tensorA.dtype === "f16" && f16ok ? ("f16" as const) : ("f32" as const);
  const rawDtypeB =
    tensorB.dtype === "f16" && f16ok ? ("f16" as const) : ("f32" as const);
  const dtypeA: "f16" | "f32" =
    inputCastA === "f16" && f16ok ? "f16" : rawDtypeA;
  const dtypeB: "f16" | "f32" =
    inputCastB === "f16" && f16ok ? "f16" : rawDtypeB;
  const promotedDtype =
    dtypeA === "f32" || dtypeB === "f32" ? ("f32" as const) : dtypeA;
  const outputDtype = action.outputDtype ?? promotedDtype;

  action.cachedDispatchConfig = {
    inputAPath,
    inputBPath,
    epilogueInputPaths,
    inputCastA,
    inputCastB,
    m,
    k,
    n,
    transA,
    transB,
    batchSize,
    batchStrideA: strideA,
    batchStrideB: strideB,
    batchStrideC: strideC,
    outShape,
    dtypeA,
    dtypeB: dtypeB !== dtypeA ? dtypeB : undefined,
    outputDtype,
    epilogueConfig: {
      ops: action.epilogueOps,
      additionalInputCount: epilogueInputRefs.length,
      outputDtype: action.outputDtype,
    },
  };
}

/**
 * Execute a lowered plan (cached dispatch sequence).
 *
 * Replaces the full executePlanOptimized() path on cache hits when the
 * lowered plan is available. Skips fusion detection, plan reordering,
 * segmentation, matmul epilogue detection, reduction preamble detection —
 * all of those decisions were recorded during the first execution and are
 * replayed here.
 *
/**
 * Capture the layout metadata the stream generator needs but can't derive
 * post-hoc (liveness frees the inputs before plan-build). SINGLE SOURCE for
 * the five action-cache captures: the lowered main loop calls this with the
 * LIVE backend input tensors; build-without-execution (phase 4.4) calls it
 * with IR-DERIVED metadata stubs (same `{shape,strides,offset,dtype,
 * buffer.size}` surface). Each capture is shape/dtype-pure → template-
 * invariant, so it runs once per template (the `=== undefined` guards).
 *
 * `backendInputs[i]` must expose `.shape/.strides/.offset?/.dtype/.buffer.size`
 * (and `.isContiguous?`); the matmul capture additionally needs whatever
 * planBareMatmul reads, which the IR stub mirrors. `observeResult` enables the
 * reshape-materialization cross-check against the live result (off under
 * build-without-execution — there is no live result to observe).
 */
function captureActionLayouts(
  action: LoweredAction,
  node: LazyIRNode,
  nodeIdx: number,
  backendInputs: unknown[],
  backendName: string,
  observeResult: boolean,
): void {
  // Every layout captured here is consumed ONLY by the webgpu stream generator
  // (stream-generate.ts) — the compiled-plan machinery is webgpu-only. On other
  // backends (CPU) there is nothing to capture and backend tensors carry no
  // `.buffer.size` (they are plain arrays), so touching that field crashes.
  // Bail cleanly for non-webgpu backends: CPU execution never engages the
  // compiled-plan / stream-generator machinery.
  if (backendName !== "webgpu") return;

  // Stage-4 phase-3: capture bare-matmul geometry (transpose detection needs
  // live strides). Geometry is shape/dtype-pure → valid for every step.
  if (
    action.kind === "sequential" &&
    node.op === "matmul" &&
    action.cachedMatmulPlan === undefined &&
    backendName === "webgpu" &&
    backendInputs.length === 2
  ) {
    try {
      action.cachedMatmulPlan = _webgpuMatmulImports!.planBareMatmul(
        asGPUTensor(backendInputs[0]),
        asGPUTensor(backendInputs[1]),
      );
    } catch {
      action.cachedMatmulPlan = "plan-throw";
    }
  }

  // narrowBackward needs its grad's dim size, but that grad is a released
  // multi-output extra at plan-build and nodes carry no per-output shape
  // metadata — cache the input shapes (template-invariant).
  if (
    action.kind === "sequential" &&
    node.op === "narrowBackward" &&
    action.cachedInputShapes === undefined
  ) {
    action.cachedInputShapes = backendInputs.map((t) =>
      (t as { shape: number[] }).shape.slice(),
    );
  }

  // Attention fwd/bwd asContiguous() their inputs inside the op — a non-
  // contiguous input inserts a contiguous-copy prologue. Capture each input's
  // layout (`contiguous` mirrors ensureContiguous EXACTLY) so the generator
  // replays planContiguousDirect for the non-contiguous ones.
  if (
    action.kind === "sequential" &&
    (node.op === "fusedAttentionForward" ||
      node.op === "fusedAttentionBackward") &&
    action.cachedInputContig === undefined
  ) {
    action.cachedInputContig = backendInputs.map((t) => {
      const w = t as {
        isContiguous?: boolean;
        shape: number[];
        strides: number[];
        offset?: number;
        dtype: import("../backend/types").DType;
        buffer: { size: number };
      };
      return {
        // Mirrors ensureContiguous EXACTLY: offset>0 views (even with
        // contiguous strides) are materialized (offset-view class, task #58).
        contiguous: w.isContiguous !== false && (w.offset ?? 0) === 0,
        shape: w.shape.slice(),
        strides: w.strides.slice(),
        offset: w.offset ?? 0,
        dtype: w.dtype,
        bufferSize: w.buffer.size,
      };
    });
  }

  // reshape MATERIALIZES a contiguous copy iff its input is non-contiguous AND
  // the new shape is incompatible with the input strides — DERIVED via the
  // shared reshapeMeta (single source with views.ts). When a live result is
  // present, cross-check the derivation against the observed buffer aliasing.
  if (
    action.kind === "view" &&
    node.op === "reshape" &&
    action.cachedViewInput === undefined &&
    backendInputs.length >= 1 &&
    // A reshape of a SCALAR ref has no backend tensor to derive a layout from
    // (backendInputs[0] is null — e.g. a 0-d graph scalar reshaped to [1,1] in
    // a small-model GradScaler backward). Leave cachedViewInput unset — the
    // non-cached view path handles it; deriving here crashed on `.shape`.
    backendInputs[0] != null
  ) {
    const w = backendInputs[0] as {
      shape: number[];
      strides: number[];
      offset?: number;
      dtype: import("../backend/types").DType;
      buffer: { size: number };
    };
    const derivedMaterialized = reshapeMeta(
      { shape: w.shape, strides: w.strides, offset: w.offset ?? 0 },
      node.shape,
    ).materialized;
    if (observeResult && ENV.TORCHLETTE_DEBUG_COMPILED) {
      const resultBuf = (
        node.result?.backendTensor as { buffer?: object } | undefined
      )?.buffer;
      const observedMaterialized =
        resultBuf !== undefined && resultBuf !== w.buffer;
      if (derivedMaterialized !== observedMaterialized) {
        console.warn(
          `[reshape-mat] node[${nodeIdx}] derived=${derivedMaterialized} observed=${observedMaterialized} shape ${w.shape.join("x")}→${node.shape.join("x")} strides=${w.strides.join(",")}`,
        );
      }
    }
    action.cachedViewInput = {
      contiguous: !derivedMaterialized,
      shape: w.shape.slice(),
      strides: w.strides.slice(),
      offset: w.offset ?? 0,
      dtype: w.dtype,
      bufferSize: w.buffer.size,
    };
  }

  // Sequential elementwise op consuming a STRIDED view (expand / transpose /
  // permute / narrow): that producer is released by plan-build and its
  // stride-bearing layout isn't shape-derivable, so capture each such input's
  // layout (incl. broadcast stride-0). Empty array marks "checked, none".
  if (
    action.kind === "sequential" &&
    action.cachedStridedInputs === undefined
  ) {
    let caps: (import("./lowered-plan").AttnInputContig | null)[] | undefined;
    for (let i = 0; i < node.inputs.length; i++) {
      const ref = node.inputs[i];
      if (
        ref.kind !== "pending" ||
        (ref.node.op !== "expand" &&
          ref.node.op !== "transpose" &&
          ref.node.op !== "permute" &&
          ref.node.op !== "narrow")
      ) {
        continue;
      }
      const w = backendInputs[i] as {
        shape: number[];
        strides: number[];
        offset?: number;
        dtype: import("../backend/types").DType;
        buffer: { size: number };
      };
      (caps ??= new Array(node.inputs.length).fill(null))[i] = {
        contiguous: false,
        shape: w.shape.slice(),
        strides: w.strides.slice(),
        offset: w.offset ?? 0,
        dtype: w.dtype,
        bufferSize: w.buffer.size,
      };
    }
    action.cachedStridedInputs = caps ?? [];
  }
}

// ============================================================================
// Build-without-execution: IR-derived metadata (stage-4 phase 4.4)
// ============================================================================
//
// To build the compiled plan on the FIRST call (no lowered execution), the two
// things that previously needed live tensors — the layout captures and the
// result-metadata harvest — must be derived from the IR. Both bottom out at
// the same recursive metadata derivation here: a node's output {shape, strides,
// offset, dtype, base-buffer-size} is a pure function of its op + its inputs'
// derived metadata, recursing until a leaf/external input whose live storage IS
// available at build time (model params, prior-plan results). Views use the
// shared view-meta transforms (single source with the backend ops); compute ops
// produce contiguous outputs. The base-buffer size feeds only the generator's
// >maxBindingSize guard (planContiguousDirectCore) — it is the one field that
// needn't be byte-exact; the stream-generation differential is the seam check
// that catches any divergence (→ not fully covered → lowered fallback).

interface DerivedMeta {
  shape: number[];
  strides: number[];
  offset: number;
  dtype: DType;
  /** Base storage buffer size in bytes (the >maxBindingSize guard input). */
  bufferSize: number;
  /** Packed-int (quantization) StorageFormat, propagated through transparent
   *  view ops so the build-without-execution matmul capture can recognize a
   *  quantized operand and decline to plan it as a plain tiled matmul (the
   *  seam-only fused-dequant route the stub can't reproduce). */
  format?: import("../backend/types").StorageFormat;
}

/** Output shape of a node's output `oi` (mirrors the harvest's extra-output map). */
function derivedOutputShape(node: LazyIRNode, oi: number): number[] | null {
  if (oi === 0) return node.shape;
  // Multi-output extras present in real plans whose every output has the SAME
  // shape as the primary (= node.shape):
  //  - adamStep oi 1,2 (m, v) — same shape as the param.
  //  - fusedAttentionBackward oi 1,2 (dK, dV) — dQ/dK/dV are all
  //    [batch, heads, seq, headDim] by construction.
  //  - fusedLayerNormBackwardGradWeightBias oi 1 (grad_bias) — grad_weight and
  //    grad_bias are both [featureDim].
  if (
    node.op === "adamStep" ||
    node.op === "fusedAttentionBackward" ||
    node.op === "fusedLayerNormBackwardGradWeightBias"
  ) {
    return node.shape;
  }
  // fusedAttentionForward oi 1 (logsumexp): [batch, heads, seq] from payload.
  if (node.op === "fusedAttentionForward" && oi === 1) {
    const p = node.payload as {
      batchSize: number;
      numHeads: number;
      seqLen: number;
    };
    return [p.batchSize, p.numHeads, p.seqLen];
  }
  return null;
}

/**
 * Recursive IR metadata deriver with memoization. `deriveRefMeta` resolves an
 * input ref's metadata — reading the LIVE storage when it exists (a leaf/
 * external/already-executed base case) and recursing through the producer node
 * otherwise (an intra-plan intermediate not yet materialized under build-
 * without-execution). At the cutover this bottoms out immediately (every intra-
 * plan node is live); under build-without-execution it recurses to the leaves.
 */
function makeMetaDeriver(): {
  deriveRefMeta: (ref: LazyRef) => DerivedMeta | null;
  deriveNodeMeta: (node: LazyIRNode, oi: number) => DerivedMeta | null;
} {
  const memo = new Map<string, DerivedMeta | null>();

  const liveMeta = (st: StorageHandle | undefined): DerivedMeta | null => {
    if (!st) return null;
    const bt = asGPUTensor(st.backendTensor);
    return {
      shape: bt.shape.slice(),
      strides: bt.strides.slice(),
      offset: bt.offset ?? 0,
      dtype: bt.dtype,
      bufferSize: bt.buffer.size,
      // Packed-weight format rides the storage/backendTensor (createQuantizedWeight
      // + the view-op propagation); carry it so a quantized operand stays visible.
      format: st.format ?? bt.format,
    };
  };

  const deriveRefMeta = (ref: LazyRef): DerivedMeta | null => {
    if (!ref || ref.kind === "scalar") return null;
    if (ref.kind === "materialized") return liveMeta(ref.storage);
    const oi = ref.outputIndex ?? 0;
    const st = oi === 0 ? ref.node.result : ref.node.results?.[oi];
    if (st) return liveMeta(st); // live base case (external / already executed)
    return deriveNodeMeta(ref.node, oi); // recurse: intra-plan intermediate
  };

  const deriveNodeMeta = (node: LazyIRNode, oi: number): DerivedMeta | null => {
    const key = `${node.id}:${oi}`;
    const cached = memo.get(key);
    if (cached !== undefined) return cached;
    memo.set(key, null); // cycle guard (DAG → never actually hit)

    let m: DerivedMeta | null = null;
    if (oi === 0 && VIEW_META_OPS.has(node.op)) {
      const inp = deriveRefMeta(node.inputs[0]);
      if (inp) {
        const inpVM = {
          shape: inp.shape,
          strides: inp.strides,
          offset: inp.offset,
        };
        if (node.op === "reshape" || node.op === "view") {
          // reshape may MATERIALIZE (incompatible non-contiguous → fresh
          // contiguous buffer) or stay a free view (shares the base buffer).
          const r = reshapeMeta(inpVM, node.shape);
          m = {
            shape: r.shape,
            strides: r.strides,
            offset: r.offset,
            dtype: node.dtype,
            bufferSize: r.materialized
              ? sizeOf(node.shape) * dtypeBytes(node.dtype)
              : inp.bufferSize,
            // A materializing reshape breaks packing; a free view keeps it.
            format: r.materialized ? undefined : inp.format,
          };
        } else {
          const vm = viewResultMeta(node.op, inpVM, node.shape, node.payload);
          if (vm) {
            m = {
              shape: vm.shape,
              strides: vm.strides,
              offset: vm.offset,
              dtype: node.dtype,
              bufferSize: inp.bufferSize, // metadata-only view → same base buffer
              // Metadata-only view (transpose/permute/…) is format-transparent:
              // the packed buffer + scales are unchanged (mirrors views.ts).
              format: inp.format,
            };
          }
        }
      }
    }
    if (!m) {
      // Compute op (oi 0) or a known multi-output extra: contiguous, offset 0,
      // fresh buffer sized from the output shape.
      const shape = derivedOutputShape(node, oi);
      if (shape) {
        m = {
          shape: shape.slice(),
          strides: contiguousStrides(shape),
          offset: 0,
          dtype: node.dtype,
          bufferSize: sizeOf(shape) * dtypeBytes(node.dtype),
        };
      }
    }
    memo.set(key, m);
    return m;
  };

  return { deriveRefMeta, deriveNodeMeta };
}

/**
 * A WebGPUTensor-shaped metadata stub fed to captureActionLayouts (and through
 * it planBareMatmul) when there is no live tensor. Exposes exactly the surface
 * the captures read: shape/strides/offset/dtype, isContiguous (matching the
 * backend's offset-agnostic definition — see tensor.ts), and a buffer.size for
 * the binding-size guard. planTiledMatmul never touches the buffer at plan time.
 */
function stubTensor(m: DerivedMeta): {
  shape: number[];
  strides: number[];
  offset: number;
  dtype: DType;
  isContiguous: boolean;
  buffer: { size: number };
  format?: import("../backend/types").StorageFormat;
} {
  return {
    shape: m.shape,
    strides: m.strides,
    offset: m.offset,
    dtype: m.dtype,
    isContiguous: checkContiguous(m.shape, m.strides),
    buffer: { size: m.bufferSize },
    format: m.format,
  };
}

/**
 * Populate the action layout caches from IR-derived metadata (build-without-
 * execution). Drives the SAME captureActionLayouts the lowered loop uses, fed
 * metadata stubs instead of live tensors. Returns the deriver (reused by the
 * harvest) and a `reset` thunk that restores every cache this populated to
 * `undefined` — so a fall-through to the lowered path re-captures from live
 * tensors with no residue. Matmul inputs that would force a contiguous copy
 * (non-contiguous AND not a simple transpose — never produced by real plans,
 * but a stub copy would dispatch on a fake buffer) bail to a sentinel, so the
 * generator simply leaves that matmul uncovered (→ lowered fallback).
 */
function populateCapturesFromIR(
  loweredPlan: LoweredPlan,
  planNodes: LazyIRNode[],
  backendName: string,
): {
  deriver: ReturnType<typeof makeMetaDeriver>;
  reset: () => void;
} {
  const deriver = makeMetaDeriver();
  const CACHE_FIELDS = [
    "cachedMatmulPlan",
    "cachedInputShapes",
    "cachedInputContig",
    "cachedViewInput",
    "cachedStridedInputs",
  ] as const;
  const resets: Array<() => void> = [];

  for (const action of loweredPlan.actions) {
    if (action.kind !== "sequential" && action.kind !== "view") continue;
    const nodeIdx = action.nodeIndex;
    const node = planNodes[nodeIdx];
    const stubs = node.inputs.map((ref) => {
      const dm = deriver.deriveRefMeta(ref);
      return dm ? stubTensor(dm) : null;
    });

    // Snapshot which caches are unset, so fall-through can restore them.
    const wasUnset: Record<string, boolean> = {};
    for (const f of CACHE_FIELDS)
      wasUnset[f] = (action as Record<string, unknown>)[f] === undefined;

    // Matmul dispatch guard: skip planBareMatmul if a stub input would copy.
    if (
      action.kind === "sequential" &&
      node.op === "matmul" &&
      action.cachedMatmulPlan === undefined
    ) {
      const a = stubs[0];
      const b = stubs[1];
      // Must mirror resolveMatmulInput: ensureContiguous COPIES when strides
      // are non-contiguous OR the element offset is non-zero (offset-view
      // class, task #58) — planBareMatmul on such a stub would dispatch a
      // real contiguous kernel on a fake buffer.
      const rawBindable = (t: { isContiguous: boolean; offset?: number }) =>
        t.isContiguous && (t.offset ?? 0) === 0;
      const unsafe =
        stubs.length < 2 ||
        !a ||
        !b ||
        (!rawBindable(a) && detectSimpleTranspose(a as never) === null) ||
        (!rawBindable(b) && detectSimpleTranspose(b as never) === null);
      if (unsafe) action.cachedMatmulPlan = "ir-noncontig-matmul";
      // A packed-int (quantized) B operand must route through the seam
      // (matmulQuantizedB) — the generated tiled-matmul stream cannot reproduce
      // the fused-dequant kernel. Bail so this matmul stays uncovered → the plan
      // is not fully covered → the lowered seam path runs (the sole format
      // reader). Mirrors the planBareMatmul guard for the live-capture path.
      else if (b?.format?.packing)
        action.cachedMatmulPlan = "quantized-operand";
    }

    captureActionLayouts(action, node, nodeIdx, stubs, backendName, false);

    for (const f of CACHE_FIELDS) {
      if (wasUnset[f] && (action as Record<string, unknown>)[f] !== undefined) {
        resets.push(() => {
          (action as Record<string, unknown>)[f] = undefined;
        });
      }
    }
  }

  return { deriver, reset: () => resets.forEach((r) => r()) };
}

/**
 * Harvest the NodeResult metadata for the generated stream. The metadata is
 * IR-DERIVED (the build-without-execution source of truth); when a live result
 * is present (the cutover runs post-execution) it's a cross-check that asserts
 * agreement and falls back loudly on a mismatch. The result SET — which
 * (nodeIndex, outputIndex) pairs survive past the plan — is supplied by the
 * caller (`harvestPairs`, in plan order): the cutover passes the live-result
 * survivors, build-without-execution passes the liveness-output set. The slot
 * for each pair comes from the generator's own nodeSlot / nodeSlotExtra map;
 * `genOk` is false iff a pair has no generated slot, or its metadata is neither
 * derivable nor live (→ lowered fallback).
 */
function harvestGenResults(
  planNodes: LazyIRNode[],
  gen: GeneratedStream,
  deriver: ReturnType<typeof makeMetaDeriver>,
  harvestPairs: Array<{ i: number; oi: number }>,
): { genResults: NodeResult[]; genOk: boolean } {
  const genResults: NodeResult[] = [];
  let genOk = true;
  let diverged = 0;

  for (const { i, oi } of harvestPairs) {
    if (!genOk) break;
    const slot =
      oi === 0 ? gen.nodeSlot.get(i) : gen.nodeSlotExtra.get(`${i}:${oi}`);
    if (slot === undefined) {
      genOk = false;
      break;
    }
    const node = planNodes[i];
    const d = deriver.deriveNodeMeta(node, oi);
    const liveSt = oi === 0 ? node.result : node.results?.[oi];
    const bt = liveSt ? asGPUTensor(liveSt.backendTensor) : null;

    let meta = d;
    if (d && bt) {
      const mismatch =
        d.shape.join(",") !== bt.shape.join(",") ||
        d.strides.join(",") !== bt.strides.join(",") ||
        d.offset !== bt.offset ||
        d.dtype !== bt.dtype;
      if (mismatch) {
        diverged++;
        console.warn(
          `[ir-derive] DIVERGE ${node.op} oi=${oi}: derived ${JSON.stringify({ shape: d.shape, strides: d.strides, offset: d.offset, dtype: d.dtype })} != live {shape:[${bt.shape}],strides:[${bt.strides}],offset:${bt.offset},dtype:${bt.dtype}} — using live`,
        );
        meta = null; // fall back to live below
      }
    }
    if (!meta) {
      if (!bt) {
        genOk = false; // neither derivable nor live → can't build w/o execution
        break;
      }
      meta = {
        shape: bt.shape.slice(),
        strides: bt.strides.slice(),
        offset: bt.offset ?? 0,
        dtype: bt.dtype,
        bufferSize: bt.buffer.size,
      };
    }
    genResults.push({
      nodeIndex: i,
      outputIndex: oi,
      slot,
      shape: meta.shape.slice(),
      strides: meta.strides.slice(),
      dtype: meta.dtype,
      offset: meta.offset,
    });
  }

  if (ENV.TORCHLETTE_DEBUG_COMPILED) {
    console.log(
      `[ir-derive] harvested ${genResults.length}/${harvestPairs.length} results, ${diverged} diverged, genOk=${genOk}`,
    );
  }
  return { genResults, genOk };
}

/**
 * The set of node ids whose results must survive past this plan — plan
 * terminal + plan.outputIndices + already-materialized + live pending roots +
 * cross-plan consumers reached from external live roots. Pure function of the
 * plan + the engine's live-pending state (no execution): single source for the
 * liveness-release protection set AND the build-without-execution harvest set.
 */
function computeLivenessOutputIds(
  plan: ExecutionPlan,
  planNodes: LazyIRNode[],
): Set<number> {
  const nodeIdSet = new Set(planNodes.map((n) => n.id));
  const outputIds = new Set<number>();
  outputIds.add(planNodes[planNodes.length - 1].id);
  // Trust plan.outputIndices first — the explicit "caller needs these" contract
  // (without it, multi-output ops like adamStep can have node.result cleared
  // mid-plan even though the param-tied node still needs it for materialization).
  if (plan.outputIndices) {
    for (const idx of plan.outputIndices) {
      outputIds.add(plan.nodes[idx].id);
    }
  }
  // Legacy fallback for callers that don't set outputIndices.
  const livePendingIds = getLivePendingNodeIds();
  for (const node of planNodes) {
    if (node.result) outputIds.add(node.id);
    if (livePendingIds.has(node.id)) outputIds.add(node.id);
  }
  // CROSS-PLAN CONSUMERS: a step's pending graph can split across several plans
  // (e.g. the foreach optimizer's update graph). A node in THIS plan that an
  // external live root's closure references will be read by a LATER plan — walk
  // the pending graph from every live root NOT in this plan and protect any
  // plan node encountered.
  const visited = new Set<number>();
  const stack: LazyIRNode[] = [];
  for (const rootObj of getLivePendingRootNodes()) {
    const root = rootObj as LazyIRNode;
    if (nodeIdSet.has(root.id)) continue; // this plan's own output
    if (!visited.has(root.id)) {
      visited.add(root.id);
      stack.push(root);
    }
  }
  while (stack.length > 0) {
    const n = stack.pop()!;
    if (nodeIdSet.has(n.id)) {
      outputIds.add(n.id); // reached into the plan from outside: protect
      continue;
    }
    if (n.result) continue; // materialized boundary — replays from storage
    for (const ref of n.inputs) {
      if (ref.kind !== "pending") continue;
      const child = ref.node as LazyIRNode;
      if (visited.has(child.id)) continue;
      visited.add(child.id);
      stack.push(child);
    }
  }
  return outputIds;
}

/**
 * Build the harvest pairs for build-without-execution: the ACTION-OUTPUT set —
 * the structural superset of the cutover's live-result survivors.
 *
 * The lowered path calls assignNodeResult for every action's OUTPUT node(s);
 * only nodes ABSORBED inside a fused / epilogue / row-program group (covered
 * but not the group output) never get a node.result. So the action-output set
 * = all plan nodes minus those fused-internal absorbed nodes. The live-result
 * survivors the cutover harvests are a SUBSET of this (some action outputs get
 * released when no live refcount holds them — an execution-/rc-dependent fact
 * we deliberately do NOT model here). Harvesting the full action-output set is
 * therefore conservative: if the generator slotted every one of them the
 * resulting plan is complete (harvestGenResults proceeds); if any lacks a
 * generated slot, genOk is false and we fall through to the lowered path —
 * exactly as the cutover declines when the stream doesn't cover a live result.
 * Multi-output extras (oi>0) come from the generator's nodeSlotExtra.
 */
function actionOutputHarvestPairs(
  loweredPlan: LoweredPlan,
  planNodes: LazyIRNode[],
  gen: GeneratedStream,
): Array<{ i: number; oi: number }> {
  // The PRIMARY (oi 0) result nodes are exactly the action OUTPUT nodes — what
  // the lowered path calls assignNodeResult for. Nodes absorbed inside a group
  // (covered but not the output) get no result; nodes in planNodes but in NO
  // action (CSE'd / bypassed / externally materialized) are not produced by
  // this plan and must not be harvested.
  const isOutput = new Set<number>();
  for (const action of loweredPlan.actions) {
    switch (action.kind) {
      case "sequential":
      case "view":
      case "data-source":
        isOutput.add(action.nodeIndex);
        break;
      case "fused":
        isOutput.add(action.outputNodeIndex);
        for (const ni of action.additionalOutputNodeIndices) isOutput.add(ni);
        break;
      case "matmul-epilogue":
      case "row-program":
        isOutput.add(action.outputNodeIndex);
        break;
      case "adam-batch":
      case "batched-reduction":
        for (const ni of action.nodeIndices) isOutput.add(ni);
        break;
      case "reclaim":
        break;
    }
  }
  const extrasByNode = new Map<number, number[]>();
  for (const k of gen.nodeSlotExtra.keys()) {
    const colon = k.indexOf(":");
    const i = Number(k.slice(0, colon));
    const oi = Number(k.slice(colon + 1));
    let list = extrasByNode.get(i);
    if (!list) extrasByNode.set(i, (list = []));
    list.push(oi);
  }
  const pairs: Array<{ i: number; oi: number }> = [];
  for (let i = 0; i < planNodes.length; i++) {
    if (isOutput.has(i)) pairs.push({ i, oi: 0 });
    const extras = extrasByNode.get(i);
    if (extras) {
      extras.sort((a, b) => a - b);
      for (const oi of extras) pairs.push({ i, oi });
    }
  }
  return pairs;
}

// build-without-execution MUST harvest the full action-output set, NOT just the
// results that outlive the plan. Pruning to "survivors" is FUNDAMENTALLY unsafe
// here, not merely hard — proven, not assumed:
//
// Plans are forced INCREMENTALLY. Two sibling plans can share a node (the DiLoCo
// trainer's 388- and 364-node plans share node 2675, a narrow). The first plan
// to be forced builds + replays; at that instant a COMPLETE walk of the entire
// live pending graph (the maximum information available at build time) reports
// the shared node as NOT cross-plan-consumed and NOT tensor-held — because the
// sibling plan's graph has not been constructed yet. So NO build-time analysis,
// however thorough, can know the node must be harvested. The second plan is
// forced later, reads the node, and finds it gone ("Input not ready").
//
// The lowered/cutover path escapes this only because it materializes every
// result and prunes later by ACTUAL refcount at markStep (it harvests live
// results — node.result is set post-execution). build-from-IR has no such
// after-the-fact signal, so it harvests every action output. The resulting
// over-harvest (every intermediate becomes an exclusive planner entry) is the
// inherent memory cost of building without execution (≈+34% on the 124M plan).
// This was empirically verified with a per-node cross-plan-visibility trace; a
// "survivor" prune deterministically crashes on the shared node.

/**
 * @param plan - The original execution plan
 * @param planNodes - Reordered plan nodes (from template.finalPerm)
 * @param loweredPlan - The cached lowered plan from the template
 * @param backend - The backend to use
 * @param options - Execution options
 * @returns The result and execution stats
 */
export async function executeLoweredPlan(
  plan: ExecutionPlan,
  planNodes: LazyIRNode[],
  loweredPlan: LoweredPlan,
  backend: Backend,
  options: {
    bufferArena?: BufferArena;
    /** [observed-liveness] Structural fingerprint of the template this plan
     *  belongs to. Stamped on any compiled plan built here so the harvest can
     *  tag results with their cross-plan identity. */
    templateFp?: number;
  } = {},
): Promise<OptimizedExecutionResult> {
  // Validate plan node count matches
  if (planNodes.length !== loweredPlan.planNodeCount) {
    throw new Error(
      `Lowered plan node count mismatch: plan has ${planNodes.length}, lowered expects ${loweredPlan.planNodeCount}`,
    );
  }

  // [observed-liveness] Record that this template executed this step, and
  // advance the guard's in-place-committed counter if this plan mutates
  // persistent state (adam / copy_ / add_). Both no-op unless build-from-IR is
  // active. The counter must advance for THIS plan before any LATER consumer
  // plan's bind-time miss reads it, so note here at the single entry point.
  if (options.templateFp !== undefined) {
    noteTemplateExecuted(options.templateFp);
  }
  if (isObservedLivenessEnabled()) {
    if (loweredPlan._hasInPlaceCommit === undefined) {
      loweredPlan._hasInPlaceCommit = planNodes.some((n) =>
        IN_PLACE_COMMIT_OPS.has(n.op),
      );
    }
    if (loweredPlan._hasInPlaceCommit) noteInPlaceCommit();
  }

  // Pre-tune matmul shapes (same as normal path)
  await pretunePlanMatmuls({ nodes: planNodes }, backend);

  // Ensure matmul imports are loaded (cached — only first call does actual import)
  if (backend.name === "webgpu") {
    await ensureWebGPUMatmulImports();
  }

  // Track stats for consistency with executePlanOptimized
  const stats: OptimizedExecutionStats = {
    totalNodes: planNodes.length,
    fusedNodes: 0,
    sequentialNodes: 0,
    fusionGroups: 0,
    fusionEnabled: true,
  };

  const useTopLevelSharedEncoder = backend.name === "webgpu";

  // Refresh per-step scalar values as DATA before ANY execution path —
  // lowered actions and compiled replays both read scalar refs through the
  // table buffers (getInputStorage / recorded dispatch bindings), so this
  // single refresh keeps every value-independent cache honest. See
  // scalar-table.ts.
  const tpS0 = TAPE_PROFILE ? performance.now() : 0;
  refreshScalarTable(loweredPlan, planNodes, backend);
  if (TAPE_PROFILE) tpAdd("scalar-refresh", performance.now() - tpS0);

  // Late-varying inlined scalars: a scalar whose value was constant through
  // the recording executions is INLINED in fused-recipe WGSL (and thus in the
  // recorded compiled plan). If such a value changes later — an LR schedule
  // leaving its warmup plateau — the replay would compute with the baked
  // value and the lowered-path adaptation could never fire (the fast path
  // bypasses the fused actions). Detect it here and drop the compiled plan;
  // the lowered execution below adapts the recipe (demotes the scalar to a
  // runtime input) and re-records. The build-from-IR rebuild must ALSO be
  // skipped for THIS execution: it generates from the un-adapted recipe, so an
  // immediate rebuild re-bakes the stale value forever (the late-LR silent
  // wrongness the stage-2 flip surfaced — invalidate → rebuild-stale →
  // invalidate, never adapting). One lowered execution adapts the recipe on
  // the cached template; the NEXT execution rebuilds from IR with the scalar
  // demoted to a runtime input.
  let scalarAdaptPending = false;
  if (inlinedFusionScalarsStale(loweredPlan, planNodes)) {
    // A cached fused recipe baked an inlined scalar VALUE (scheduled LR, a
    // per-step bias-correction coefficient) that has since changed. Force ONE
    // lowered execution: the fused-segment loop adapts the recipe (demotes the
    // scalar to a runtime input delivered through the scalar table), and the
    // NEXT execution rebuilds from IR adapted.
    //
    // This must fire whether or not a compiled plan currently exists. Under
    // build-from-IR a recurring fused template's FIRST occurrence builds the
    // recipe on the lowered path but records NO compiled plan, so its SECOND
    // occurrence has no compiled plan to trip the old compiledPlan-gated check
    // — build-from-IR would then bake the already-stale recipe and silently
    // corrupt optimizer state for that step (the Adam-elementwise late-LR /
    // frozen per-step bc1/bc2 wrongness the stage-2 flip surfaced: state
    // corrupted at step 2, ~constant trajectory offset thereafter). The
    // compiled-plan invalidation is conditional; scalarAdaptPending gates the
    // build-from-IR block only, so the recorded path (which always holds a
    // compiled plan when stale) is unaffected.
    if (loweredPlan.compiledPlan) {
      destroyCompiledPlanBuffers(loweredPlan.compiledPlan);
      loweredPlan.compiledPlan = undefined;
    }
    scalarAdaptPending = true;
  }

  // =========================================================================
  // FAST PATH: Compiled Plan Execution
  // =========================================================================
  // If we have a valid compiled plan, execute it directly.
  // The compiled plan is a flat sequence of GPU primitives (alloc, dispatch,
  // copy, write, barrier) with abstract slot indices instead of concrete buffers.
  // Compiled plan: enabled by default. Disable with TORCHLETTE_COMPILED_PLAN=0.
  if (
    loweredPlan.compiledPlan?.valid &&
    useTopLevelSharedEncoder &&
    options.bufferArena &&
    ENV.TORCHLETTE_COMPILED_PLAN !== "0" &&
    // Liveness-aware arena reuses one buffer across multiple positions, so
    // compiled replay needs planned mode (the memory planner derives a
    // lifetime-split buffer assignment per plan — see buildCompiledPlan).
    (!arenaLivenessEnabled() || compiledPlannedEnabled())
  ) {
    if (ENV.TORCHLETTE_DEBUG_COMPILED) {
      console.log(
        `[exec] COMPILED nodes=${planNodes.length} cmds=${loweredPlan.compiledPlan.commands.length}`,
      );
    }
    // Stage-4: reclaim the per-position arena's WARMUP buffers. The arena is
    // populated only during the lowered first executions; a planner-backed
    // compiled replay binds registry-materialized buffers (compiled-plan.ts
    // device.createBuffer) and NEVER calls resolveOutputBuffer, so once a plan
    // reaches this path its arena buffers are dead weight (measured ~209MB at
    // steady state, retained forever). destroyArena is canRecycle-gated — a
    // buffer still live (e.g. an external input to a not-yet-compiled consumer)
    // is removed from arena ownership but routed through the pool-release chain
    // (never destroyed-while-live) via fence-gated deferredDestroy; freeable
    // buffers are reclaimed. The arrays re-grow if the plan ever falls back to
    // lowered. Guarded on plannerEntries (the planner path that bypasses the
    // arena) and on non-empty arrays (no-op once reclaimed).
    if (
      loweredPlan.compiledPlan.plannerEntries &&
      (options.bufferArena.resolve.length > 0 ||
        options.bufferArena.alloc.length > 0)
    ) {
      destroyArena(options.bufferArena);
    }
    const tpR0 = TAPE_PROFILE ? performance.now() : 0;
    const externalInputBuffers = collectExternalInputBuffers(planNodes);
    try {
      await executeCompiledPlan(
        loweredPlan.compiledPlan,
        planNodes,
        options.bufferArena,
        backend,
        externalInputBuffers,
      );
    } finally {
      clearActiveScalarTable();
    }
    if (TAPE_PROFILE) tpAdd("replay-total", performance.now() - tpR0);
    return {
      result: planNodes[planNodes.length - 1].result!,
      stats: loweredPlan.cachedStats ?? stats,
    };
  }

  // =========================================================================
  // BUILD-WITHOUT-EXECUTION (stage-4 phase 4.4 → stage-2 DEFAULT, 2026-07-08)
  // =========================================================================
  // Build the compiled plan from the lowered IR on the FIRST call — NO lowered
  // execution, NO recording. Populate the layout captures from IR-derived
  // metadata stubs, generate the stream, harvest the result metadata from the
  // IR (the liveness-output set), build the planner-backed plan, and replay it.
  // THE DEFAULT build source (stage-2 flip; opt-out TORCHLETTE_BUILD_FROM_IR=0
  // restores the record-then-cutover build everywhere — see buildFromIRActive
  // for the recording-survivor cases). Any failure (uncovered stream,
  // underivable result, invalid plan) resets the captures and falls through to
  // the lowered path with zero residue — so opt-out behaviour and the fallback
  // are byte-identical to the unmodified normal path, and an uncovered plan
  // simply records on its next execution (the census-driven fallback).
  if (
    !scalarAdaptPending &&
    buildFromIRActive() &&
    backend.name === "webgpu" &&
    useTopLevelSharedEncoder &&
    options.bufferArena &&
    !loweredPlan.compiledPlan &&
    (!arenaLivenessEnabled() || compiledPlannedEnabled())
  ) {
    await ensureFusionImports();
    const { reset } = populateCapturesFromIR(
      loweredPlan,
      planNodes,
      backend.name,
    );
    let genPlan: CompiledPlan | undefined;
    // constFill gate: a plan-owned FIXED constant buffer (the grad-clip `1.0`
    // ceiling) is only emitted for RECURRING templates (build-block reach ≥2).
    // The build block is re-entered every execution of an uncovered template
    // (a covered one caches + replays), so reach≥2 ⇒ this template recurs;
    // covering a `full` on it is safe. A one-shot warmup graph variant
    // (reach 1) is NOT covered — it would compile once, never recur, and its
    // constFill buffer would sit until idle-retire. Threaded so transient
    // plans stay lowered (docs/stage4 CAPTURE PRIMITIVE).
    let coverConstFill = false;
    if (options.templateFp !== undefined) {
      const r = (buildReaches.get(options.templateFp) ?? 0) + 1;
      buildReaches.set(options.templateFp, r);
      coverConstFill = r >= 2;
    }
    const gen = generateStream(loweredPlan, planNodes, backend, {
      coverConstFill,
      // BUILD path: a stamped 0-d cross-plan external may compile into the
      // fused row-program kernel (the clip chain, residue a) — the bind seam
      // asserts the stamp at every replay.
      fuseStampedScalarExternals: true,
    });
    if (gen?.fullyCovered) {
      // [observed-liveness] The conservative full action-output harvest is the
      // safe default (see the proof above); once observation has converged a
      // recurring template, prune it to the observed needed-set (consumed ∪
      // survived ∪ read-back ∪ guard-grown) ∪ {terminal}.
      //
      // [stage-3 A] DECLARED outputs (plan.outputIndices — the forced roots)
      // are NO LONGER unconditionally kept: they must earn their harvest slot
      // through observation like every other pair. The S3.0 decomposition
      // showed ~700 MB/step of observed-dead mandatory results at 124M-class
      // (grad-accumulation intermediates rooted at force time, read by
      // nothing). Every external reader class is now observed: plan reads
      // (observeConsumed), boundary survival (everSurvived), readbacks
      // (observeReadback — the seam that makes this pruning sound; the loss
      // is the canonical readback-only mandatory pair). A first-ever reader
      // after convergence is the same epistemic boundary as stage-1 pruning:
      // plan reads hit the bind-time guard (clean recovery), readbacks hit
      // the readbackMiss heal (loud once, harvested next step). Only the
      // TERMINAL node stays unconditional (the plan's return value, read by
      // the forcing caller immediately). The EXCLUDED pairs are recorded on
      // the plan so a late reader's miss can name the culprit.
      const actionPairs = actionOutputHarvestPairs(loweredPlan, planNodes, gen);
      const fp = options.templateFp;
      let harvestPairs = actionPairs;
      let prunedPairs: Array<{ ni: number; oi: number }> | undefined;
      let mandatoryKeys: Set<string> | undefined;
      if (fp !== undefined) {
        const outIds = new Set<number>();
        outIds.add(planNodes[planNodes.length - 1].id);
        if (plan.outputIndices) {
          for (const idx of plan.outputIndices) outIds.add(plan.nodes[idx].id);
        }
        const mandatory = new Set<string>();
        for (let i = 0; i < planNodes.length; i++) {
          if (outIds.has(planNodes[i].id)) mandatory.add(`${i}:0`);
        }
        mandatoryKeys = mandatory;
        const keepAlways = new Set<string>([`${planNodes.length - 1}:0`]);
        const pr = prunedHarvest(fp, actionPairs, keepAlways);
        if (pr) {
          harvestPairs = pr.kept;
          prunedPairs = pr.excluded.map((p) => ({ ni: p.i, oi: p.oi }));
        }
      }
      // [stage-3 B] Released-external claims: for each external input whose
      // stamped pair is step-globally releasable WITH THIS TEMPLATE as its
      // K-stable last reader, hand the producer's registry entry to the
      // memory planner as a release event at the slot's final read — this
      // plan's temps overlay the boundary-dead value (grads after the
      // optimizer's reads, forward casts after backward's). Claim-seam
      // assertion: plannerEntryClaimable (the entry must still be the
      // producer's exclusive result entry, same registry generation).
      let externalReleases:
        | Array<{ slot: number; entryIdx: number }>
        | undefined;
      let claimedExternal:
        | Array<{ slot: number; fp: number; ni: number; oi: number }>
        | undefined;
      if (fp !== undefined) {
        for (let si = 0; si < gen.slots.length; si++) {
          const src = gen.slots[si];
          if (src.kind !== "external") continue;
          const ref = planNodes[src.planNodeIndex]?.inputs[src.inputIndex];
          if (!ref || (ref.kind !== "pending" && ref.kind !== "materialized"))
            continue;
          let storage: StorageHandle;
          try {
            storage = getInputStorage(ref, backend);
          } catch {
            continue;
          }
          const stamp = storage.stamp;
          if (!stamp) continue;
          if (releasableLastReader(stamp.fp, stamp.ni, stamp.oi) !== fp)
            continue;
          // [task #97] DERIVED cross-plan liveness governs the overlay-release,
          // not the empirical last-reader alone. The last-reader observation is
          // fed ONLY by the compiled external-slot seam (observeConsumed); a
          // consumer that reads the producer LOWERED — the canonical case is the
          // BACKWARD pass re-reading a saved-for-backward forward activation,
          // resolved through getInputStorage — is structurally invisible to it,
          // so a forward consumer can be misclassified "last reader" while
          // backward still holds a read. graphHeldAt is the single-source derived
          // fact (∃ live _graphRetained clone / G(s)>0, the same axis _derived
          // trusts): a saved-for-backward producer WILL be re-read, so its entry
          // must NOT be overlaid. Declining the claim here is not a memory
          // regression — a graph-held value is live until backward reads it
          // REGARDLESS of the overlay, so it was never a real overlay win; the
          // genuinely-boundary-dead values (grads after the optimizer, casts
          // after their sole read — graphHeld=false) still claim and overlay.
          // This makes the over-prune UNCONSTRUCTIBLE, so the recorded build's
          // compiled-only guardMiss net is no longer load-bearing for soundness.
          if (storageTracker.graphHeldAt(storage.id)) continue;
          // The DERIVED cross-plan edge set SUPERSEDES the graphHeldAt heuristic
          // (docs/step-data-dependence-design.md §5). graphHeldAt only exonerates
          // saved-for-backward values with a LIVE _graphRetained clone; a
          // graphHeld=false activation that a LOWERED consumer still re-reads
          // (the [1,512,768] forwardToForce class) is invisible to it, so the
          // claim would overlay a value with a genuine later reader → the
          // stage-3 B released-read UAF. The witness observed that reader
          // directly (noteWitnessRead fires on the lowered getInputStorage read),
          // so a cross-plan consumer edge to a template OTHER than this claimant
          // means the value is still live: decline the claim. This is the
          // witnessed-cross-plan-reader unblock the #97 STOP named, made
          // principled by DERIVING the consumer edge rather than guessing from a
          // graph-retention flag. Sound only because the witnessing has converged
          // by claim time (this template did not cut over until its edge set
          // reached a fixed point — see the converged-to-lowered witness stamp).
          if (
            crossPlanEdgeHasOtherConsumer(
              stamp.fp,
              `${stamp.ni}:${stamp.oi}`,
              currentVariantSelection(),
              fp,
            )
          )
            continue;
          const entry = resultEntryFor(stamp.fp, stamp.ni, stamp.oi);
          if (!entry || !plannerEntryClaimable(entry.entryIdx, entry.gen))
            continue;
          // Entry-level gate: every pair aliasing this entry must be
          // releasable with THIS template as the reader (one buffer).
          if (releasableEntryReader(stamp.fp, entry.entryIdx) !== fp) continue;
          (externalReleases ??= []).push({
            slot: si,
            entryIdx: entry.entryIdx,
          });
          (claimedExternal ??= []).push({
            slot: si,
            fp: stamp.fp,
            ni: stamp.ni,
            oi: stamp.oi,
          });
        }
      }
      const { genResults, genOk } = harvestGenResults(
        planNodes,
        gen,
        makeMetaDeriver(),
        harvestPairs,
      );
      if (genOk) {
        const built = buildCompiledPlanFromGenerated({
          commands: gen.commands,
          slots: gen.slots,
          nodeResults: genResults,
          externalReleases,
          // [task #99 R1] Same boundary set the recorded path computes — the
          // generated path lacks planNodes in its input, so supply it here for
          // uniform recompute-segment stamping across both build paths.
          recomputeBoundaryIndices: checkpointBoundaryIndices(planNodes),
        });
        if (built.valid) {
          built.templateFp = fp;
          built._prunedPairs = prunedPairs;
          // [stage-3 B] Only entries the planner ACTUALLY overlaid need the
          // loud clear-at-release; released-but-unclaimed values stay intact.
          if (claimedExternal && built._claimedEntries && externalReleases) {
            const claimedSet = new Set(built._claimedEntries);
            const slotToEntry = new Map(
              externalReleases.map((r) => [r.slot, r.entryIdx]),
            );
            const active = claimedExternal.filter((c) =>
              claimedSet.has(slotToEntry.get(c.slot) as number),
            );
            if (active.length > 0) built._claimedExternal = active;
          }
          // [stage-3 S3.0] Register harvested-result → planner-entry mappings
          // for the step-global release observation (which result bytes an
          // overlay could free, and into whose build). Pure observation.
          if (fp !== undefined && built.plannerAssignment) {
            const regEntries: Array<{
              ni: number;
              oi: number;
              entryIdx: number;
              bytes: number;
              mandatory: boolean;
            }> = [];
            for (const r of genResults) {
              const entryIdx = built.plannerAssignment.get(r.slot);
              if (entryIdx !== undefined) {
                regEntries.push({
                  ni: r.nodeIndex,
                  oi: r.outputIndex,
                  entryIdx,
                  bytes: plannerEntryBytes(entryIdx),
                  mandatory:
                    mandatoryKeys?.has(`${r.nodeIndex}:${r.outputIndex}`) ??
                    false,
                  op: `${planNodes[r.nodeIndex]?.op}[${JSON.stringify(planNodes[r.nodeIndex]?.shape)}]`,
                });
              }
            }
            registerResultEntries(fp, built.plannerGen ?? -1, regEntries);
          }
          loweredPlan.compiledPlan = genPlan = built;
          if (ENV.TORCHLETTE_DEBUG_COMPILED) {
            console.log(
              `[compiled] phase-4.4 BUILD-FROM-IR fp=0x${fp?.toString(16)}: ${gen.commands.length} cmds, ${genResults.length} results${prunedPairs ? ` (pruned ${prunedPairs.length})` : ""} — no lowered execution`,
            );
          }
        }
      }
    }
    if (genPlan) {
      const externalInputBuffers = collectExternalInputBuffers(planNodes);
      try {
        // First replay: endCounters is undefined, so executeCompiledPlan lets
        // the dispatch sequence advance naturally (no reset). Capture it AFTER
        // for subsequent FAST-PATH replays (they reset to this absolute value
        // so later plans in the step don't collide on params/output indices).
        await executeCompiledPlan(
          genPlan,
          planNodes,
          options.bufferArena,
          backend,
          externalInputBuffers,
        );
        genPlan.endCounters = getDispatchSequenceCounters();
      } finally {
        clearActiveScalarTable();
      }
      loweredPlan.cachedStats = stats;
      return {
        result: planNodes[planNodes.length - 1].result!,
        stats,
      };
    }
    // Fall through to the lowered path: restore the caches we populated so the
    // normal loop re-captures them from live tensors with no residue.
    reset();
  }

  // =========================================================================
  // NORMAL PATH (with optional compilation recording for compiled plan)
  // =========================================================================

  if (ENV.TORCHLETTE_DEBUG_COMPILED) {
    console.log(
      `[exec] NORMAL nodes=${planNodes.length} hasCompiledPlan=${!!loweredPlan.compiledPlan?.valid} hasArena=${!!options.bufferArena} arenaLen=${options.bufferArena?.resolve?.length ?? "n/a"}`,
    );
  }

  // Shared encoder scope
  if (useTopLevelSharedEncoder) beginSharedEncoder();

  // Activate buffer arena for this plan (stabilizes buffer identities for bind group cache)
  if (options.bufferArena && useTopLevelSharedEncoder) {
    setActiveArena(options.bufferArena);
  }

  // Register external input buffers so the arena can detect conflicts.
  // When the same template is reused (e.g., same transformer block structure
  // across layers), the previous execution's output arena buffer becomes the
  // next execution's external input. Without this, the arena would return the
  // same buffer for both reading (external input) and writing (fused output),
  // causing data corruption.
  if (options.bufferArena && useTopLevelSharedEncoder) {
    setArenaExternalInputBuffers(collectExternalInputBuffers(planNodes));
  }

  // Start compilation recording for compiled plan.
  // Bounded-arena (liveness) mode is supported via PLANNED BUFFERS: slot
  // resolution happens at record time (lifetime splitting in recordAlloc),
  // and buildCompiledPlan pins the recorded pool-buffer assignment to the
  // plan — replays bind the exact lifetime-reusing assignment the recording
  // proved safe, with memory bounded to the recording's working set.
  const shouldCompile =
    useTopLevelSharedEncoder &&
    options.bufferArena &&
    !loweredPlan.compiledPlan &&
    options.bufferArena.resolve.length > 0 && // Arena populated from prior execution
    // Bounded-arena mode: compile only under the experimental planned-buffer
    // flag (see buildCompiledPlan), else stay on the lowered path as before.
    (!arenaLivenessEnabled() || compiledPlannedEnabled()) &&
    // Debug bisection: only compile plans up to N nodes.
    (!ENV.TORCHLETTE_COMPILED_MAX_NODES ||
      planNodes.length <= parseInt(ENV.TORCHLETTE_COMPILED_MAX_NODES, 10)) &&
    // Debug bisection: only compile plans whose node count is in the list.
    (!ENV.TORCHLETTE_COMPILED_ONLY_NODES ||
      ENV.TORCHLETTE_COMPILED_ONLY_NODES.split(",").includes(
        String(planNodes.length),
      ));
  let compilationRecording: ReturnType<
    typeof startCompilationRecording
  > | null = null;
  /** [stage-3 A] Node ids whose results EXISTED before this execution
   *  (skip-executed shared nodes — e.g. attention nodes shared with a
   *  sibling template that already ran this step). A recording claims only
   *  results THIS execution produced: pre-existing ones are excluded from
   *  the result collection instead of tripping the unrecordedNodes
   *  invalidation (consumers re-resolve them per replay as externals via
   *  planNodes[i].inputs, exactly like any prior-plan result). */
  let preExistingResults: Set<number> | null = null;
  if (shouldCompile) {
    if (ENV.TORCHLETTE_DEBUG_COMPILED) {
      console.log(`[exec] RECORDING nodes=${planNodes.length}`);
    }
    compilationRecording = startCompilationRecording();
    preExistingResults = new Set();
    for (const node of planNodes) {
      if (node.result || (node.results && node.results.some((r) => !!r))) {
        preExistingResults.add(node.id);
      }
    }
    // Pre-assign external input slots for inputs that are already materialized
    // (results from prior plan executions). Inputs produced within this plan
    // are NOT yet materialized and will get slots during recording via arena alloc.
    for (let i = 0; i < planNodes.length; i++) {
      const node = planNodes[i];
      for (let j = 0; j < node.inputs.length; j++) {
        const ref = node.inputs[j];
        // Only assign if the input is already materialized
        let storage: StorageHandle | undefined;
        if (ref.kind === "materialized") {
          storage = ref.storage;
        } else if (ref.kind === "scalar") {
          continue; // Scalar constants don't have GPU buffers
        } else {
          const idx = ref.outputIndex ?? 0;
          storage = idx === 0 ? ref.node.result : ref.node.results?.[idx];
        }
        if (!storage) continue;
        const buf = gpuBuffer(storage.backendTensor);
        assignSlot(buf, {
          kind: "external",
          planNodeIndex: i,
          inputIndex: j,
        });
      }
    }
  }

  // Pre-resolve dynamic imports before the action loop so subsequent calls
  // to executeFusedSegment avoid per-call async import overhead.
  await ensureFusionImports();

  // [tape-1a] time the whole lowered (non-replay) execution so any step that
  // falls off the compiled fast path is visible in the G0 table.
  const tpL0 = TAPE_PROFILE ? performance.now() : 0;

  // =========================================================================
  // Liveness analysis: free intermediate buffers as soon as their last
  // consuming action completes. After release, buffers go to the pool's
  // pendingRelease queue and become reusable at the next reclaim point.
  //
  // Key insight: the lowered plan's action order does NOT match plan-node
  // step order (fused groups can cover non-contiguous positions). So we
  // track the last ACTION INDEX that reads each node, not the last step.
  // =========================================================================
  // Liveness-based buffer release: free intermediate buffers mid-plan execution.
  // Enabled by default; opt out with TORCHLETTE_LIVENESS_RELEASE=0.
  const enableLivenessRelease = ENV.TORCHLETTE_LIVENESS_RELEASE !== "0";
  let livenessOutputIds: Set<number> | null = null;
  let livenessReleased: Set<number> | null = null;
  let livenessDeferred: Set<number> | null = null;
  let livenessNodeToStorage: Map<number, StorageHandle> | null = null;
  let livenessNodeIdToIndex: Map<number, number> | null = null;
  // nodeId → last action index that reads from this node
  let livenessLastAction: Map<number, number> | null = null;
  // actionIndex → [nodeIds to release after this action]
  let livenessReleaseSchedule: Map<number, number[]> | null = null;

  if (enableLivenessRelease) {
    const nodeIdSet = new Set(planNodes.map((n) => n.id));

    // Protected nodes: plan terminals + already-materialized + live RuntimeTensors
    // + cross-plan consumers (see computeLivenessOutputIds — single source with
    // the build-without-execution harvest set).
    livenessOutputIds = computeLivenessOutputIds(plan, planNodes);

    // Map nodeId → plan index for O(1) lookup
    livenessNodeIdToIndex = new Map();
    for (let i = 0; i < planNodes.length; i++) {
      livenessNodeIdToIndex.set(planNodes[i].id, i);
    }

    // For each plan node, find the last action that reads it as an input.
    // This operates in ACTION-INDEX space (not plan-step space), which is
    // correct because actions can cover non-contiguous plan positions.
    livenessLastAction = new Map<number, number>();
    for (let ai = 0; ai < loweredPlan.actions.length; ai++) {
      const actionNodeIndices = getActionNodeIndices(loweredPlan.actions[ai]);
      for (const ni of actionNodeIndices) {
        const node = planNodes[ni];
        for (const ref of node.inputs) {
          if (ref.kind === "pending" && nodeIdSet.has(ref.node.id)) {
            const cur = livenessLastAction.get(ref.node.id);
            if (cur === undefined || ai > cur) {
              livenessLastAction.set(ref.node.id, ai);
            }
          }
        }
      }
    }

    // Build release schedule: after action[i] completes, release these nodes
    livenessReleaseSchedule = new Map();
    for (const [nodeId, lastAi] of livenessLastAction) {
      if (livenessOutputIds.has(nodeId)) continue;
      let list = livenessReleaseSchedule.get(lastAi);
      if (!list) {
        list = [];
        livenessReleaseSchedule.set(lastAi, list);
      }
      list.push(nodeId);
    }

    livenessReleased = new Set();
    livenessDeferred = new Set();
    livenessNodeToStorage = new Map();
  }

  try {
    for (
      let actionIndex = 0;
      actionIndex < loweredPlan.actions.length;
      actionIndex++
    ) {
      const action = loweredPlan.actions[actionIndex];
      // Task #71: the per-node offset-repack flag is scoped to ONE direct-
      // elementwise node's dispatch. Clear it at every action boundary so a
      // throw that skipped a node's post-exec clear can't leak its closure into
      // a later action's createParamsBuffer (defensive; a throw normally aborts
      // the plan). No-op when not recording.
      if (compilationRecording) setPendingParamsVolatilePack(null);
      switch (action.kind) {
        case "fused": {
          // Attribute recorded commands to the group's output node — without
          // this they inherit the PREVIOUS sequential node's index, polluting
          // its segment in the stream differential (and misattributing any
          // volatile-uniform recordings).
          if (compilationRecording)
            setRecordingNodeIndex(action.outputNodeIndex);
          // Reconstruct FusionGroup from plan nodes
          const groupNodes = action.coveredNodeIndices.map((i) => planNodes[i]);
          const outputNode = planNodes[action.outputNodeIndex];
          const additionalOutputNodes = action.additionalOutputNodeIndices.map(
            (i) => planNodes[i],
          );
          const neededIntermediates = action.neededIntermediateNodeIndices.map(
            (i) => planNodes[i],
          );

          // Reconstruct external inputs using cached pattern (O(n) instead of O(n²))
          let extInputs: LazyRef[];
          if (action.cachedExternalInputPattern) {
            // Fast path: use pre-computed pattern
            extInputs = action.cachedExternalInputPattern.map(
              (p) => groupNodes[p.nodeLocalIdx].inputs[p.inputIdx],
            );
          } else {
            // First execution: compute pattern with dedup, then cache it
            const groupNodeIds = new Set(groupNodes.map((n) => n.id));
            extInputs = [];
            const pattern: Array<{ nodeLocalIdx: number; inputIdx: number }> =
              [];
            for (let ni = 0; ni < groupNodes.length; ni++) {
              const node = groupNodes[ni];
              for (let ii = 0; ii < node.inputs.length; ii++) {
                const inp = node.inputs[ii];
                if (inp.kind === "pending") {
                  if (
                    !groupNodeIds.has(inp.node.id) &&
                    !extInputs.some(
                      (ei) =>
                        ei.kind === "pending" && ei.node.id === inp.node.id,
                    )
                  ) {
                    extInputs.push(inp);
                    pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                  }
                } else if (inp.kind === "scalar") {
                  // Positional — never value-deduped (must mirror
                  // collectExternalInputs / groupToRecipe; equal-today
                  // values may diverge across steps).
                  extInputs.push(inp);
                  pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                } else {
                  if (
                    !extInputs.some(
                      (ei) =>
                        ei.kind === "materialized" &&
                        ei.storage.id === inp.storage.id,
                    )
                  ) {
                    extInputs.push(inp);
                    pattern.push({ nodeLocalIdx: ni, inputIdx: ii });
                  }
                }
              }
            }
            action.cachedExternalInputPattern = pattern;
          }

          const group: FusionGroup = {
            nodes: groupNodes,
            planIndices: action.coveredNodeIndices,
            externalInputs: extInputs,
            outputNode,
            additionalOutputNodes:
              additionalOutputNodes.length > 0
                ? additionalOutputNodes
                : undefined,
            neededIntermediates:
              neededIntermediates.length > 0 ? neededIntermediates : undefined,
          };

          // Frozen-scalar adaptation: the cached recipe bakes inlined scalar
          // VALUES; the template fingerprint deliberately ignores them. If a
          // scalar's current value differs from the baked one (per-step
          // coefficient, scheduled LR), demote it to a RUNTIME input — its
          // value then flows as data through the scalar table on every
          // execution. The rebuilt recipe replaces the cached one and any
          // compiled plan recorded against the old kernel is invalidated
          // (adaptation fires on the recording execution — the first one
          // where the value can differ — so the re-recording uses the new
          // kernel). Inlining is kept for scalars that never change.
          {
            let stale: Set<number> | null = null;
            const rin = action.recipe.inputs;
            for (let i = 0; i < rin.length; i++) {
              const inp = rin[i];
              if (!inp?.isInlinedConstant) continue;
              const ref = extInputs[i];
              if (!ref) continue;
              // Payload-first current-value read (materialized or not) — see
              // inlinedConstantValue: the old isInlinableScalar read refused
              // already-executed data-sources, silently skipping the check.
              const cur = inlinedConstantValue(ref);
              if (cur !== null && cur !== inp.inlinedValue) {
                if (ENV.TORCHLETTE_DEBUG_SCALARS) {
                  console.log(
                    `[scalar-adapt] input ${i}: baked=${inp.inlinedValue} current=${cur}`,
                  );
                }
                (stale ??= new Set(action.runtimeScalarInputs)).add(i);
              }
            }
            if (stale) {
              if (ENV.TORCHLETTE_DEBUG_SCALARS) {
                console.log(
                  `[scalar-adapt] demoting ${stale.size} inlined scalar(s) on a ${groupNodes.length}-node group`,
                );
              }
              action.runtimeScalarInputs = stale;
              action.recipe = groupToRecipe(group, stale);
              if (loweredPlan.compiledPlan) {
                destroyCompiledPlanBuffers(loweredPlan.compiledPlan);
                loweredPlan.compiledPlan = undefined;
              }
            }
          }

          // BUFFER DONATION eligibility: external inputs whose LAST reader
          // is this action and which are not plan outputs / externally
          // referenced — their buffers may be overwritten in place by the
          // fused kernel's primary output (segment-executors picks one).
          let donatableInputIds: Set<number> | undefined;
          if (livenessLastAction && livenessOutputIds) {
            for (const ref of group.externalInputs) {
              if (!ref || ref.kind !== "pending") continue;
              const nid = ref.node.id;
              if (
                livenessLastAction.get(nid) === actionIndex &&
                !livenessOutputIds.has(nid)
              ) {
                (donatableInputIds ??= new Set()).add(nid);
              }
            }
          }
          await executeFusedSegment(
            group,
            action.recipe,
            backend,
            action.enableVectorization,
            donatableInputIds,
          );
          stats.fusedNodes += groupNodes.length;
          stats.fusionGroups++;
          break;
        }

        case "matmul-epilogue": {
          if (compilationRecording)
            setRecordingNodeIndex(action.matmulNodeIndex);
          const matmulNode = planNodes[action.matmulNodeIndex];
          const outputNode = planNodes[action.outputNodeIndex];

          // Cache the label string (structural, same across steps)
          let epLabel = action.cachedLabel;
          if (!epLabel) {
            const prologueLabel = action.prologues ? "prologue+" : "";
            const epilogueLabel =
              action.epilogueOps.length > 0
                ? "+" + action.epilogueOps.map((o) => o.kind).join("+")
                : "";
            epLabel = `matmul+${prologueLabel}${epilogueLabel}`.replace(
              /\+$/,
              "",
            );
            action.cachedLabel = epLabel;
          }

          // ── FAST PATH: use cached dispatch config (2nd+ lowered plan execution) ──
          if (action.cachedDispatchConfig) {
            const cfg = action.cachedDispatchConfig;
            setCurrentOpLabel(epLabel);
            setProfileModule(matmulNode.module ?? "unknown");
            const _profT0 = profileOpBegin(epLabel);
            try {
              // Resolve GPU buffers from cached paths (plan-node-relative lookups)
              const refA =
                planNodes[cfg.inputAPath.planNodeIndex].inputs[
                  cfg.inputAPath.inputIndex
                ];
              const refB =
                planNodes[cfg.inputBPath.planNodeIndex].inputs[
                  cfg.inputBPath.inputIndex
                ];
              const bufA = gpuBuffer(getInputStorage(refA).backendTensor);
              const bufB = gpuBuffer(getInputStorage(refB).backendTensor);
              const epilogueBuffers: GPUBuffer[] = [];
              for (const path of cfg.epilogueInputPaths) {
                const ref =
                  planNodes[path.planNodeIndex].inputs[path.inputIndex];
                epilogueBuffers.push(
                  gpuBuffer(getInputStorage(ref).backendTensor),
                );
              }

              // Dispatch directly — skips shape computation, transpose detection,
              // contiguous checks, prologue resolution, dynamic imports
              const resultTensor = _webgpuMatmulImports!.dispatchMatmulDirect(
                bufA,
                bufB,
                {
                  m: cfg.m,
                  n: cfg.n,
                  k: cfg.k,
                  transA: cfg.transA,
                  transB: cfg.transB,
                  batchSize: cfg.batchSize,
                  batchStrideA: cfg.batchStrideA,
                  batchStrideB: cfg.batchStrideB,
                  batchStrideC: cfg.batchStrideC,
                  outShape: cfg.outShape,
                  dtypeA: cfg.dtypeA,
                  dtypeB: cfg.dtypeB,
                  outputDtype: cfg.outputDtype,
                  epilogueConfig: cfg.epilogueConfig as
                    | EpilogueConfig
                    | undefined,
                  epilogueBuffers,
                  inputCastA: cfg.inputCastA,
                  inputCastB: cfg.inputCastB,
                },
              );
              // Fix up shape if epilogue absorbed a reshape (e.g. [1,M,N]→[M,N])
              const fastOutShape = outputNode.shape;
              if (!shapesEqual(resultTensor.shape, fastOutShape)) {
                const gpuT = asGPUTensor(resultTensor);
                gpuT.shape = fastOutShape;
                gpuT.strides = contiguousStrides(fastOutShape);
              }
              outputNode.result = createStorageHandle(
                outputNode.device,
                resultTensor,
              );
            } finally {
              profileOpEnd(epLabel, _profT0);
              setCurrentOpLabel(null);
              setProfileModule("unknown");
            }

            break;
          }

          // ── SLOW PATH: first lowered plan execution — full reconstruction ──

          const { epilogueInputRefs, epilogueInputPaths } =
            reconstructEpilogueInputs(planNodes, action.coveredNodeIndices);

          // Reconstruct prologues and resolve prologue decisions
          let prologues: MatmulPrologueInfo[] | undefined;
          let inputCastA: "f16" | "f32" | undefined;
          let inputCastB: "f16" | "f32" | undefined;
          let resolvedInputRefA = matmulNode.inputs[0];
          let resolvedInputRefB = matmulNode.inputs[1];
          let inputAPath: PlanNodePath = {
            planNodeIndex: action.matmulNodeIndex,
            inputIndex: 0,
          };
          let inputBPath: PlanNodePath = {
            planNodeIndex: action.matmulNodeIndex,
            inputIndex: 1,
          };

          if (action.prologues && action.prologues.length > 0) {
            const resolved = resolvePrologueInputs(
              planNodes,
              matmulNode,
              action.matmulNodeIndex,
              action.prologues,
            );
            prologues = resolved.prologues;
            inputCastA = resolved.inputCastA;
            inputCastB = resolved.inputCastB;
            resolvedInputRefA = resolved.resolvedInputRefA;
            resolvedInputRefB = resolved.resolvedInputRefB;
            inputAPath = resolved.inputAPath;
            inputBPath = resolved.inputBPath;
          }

          await withProfileContext(epLabel, matmulNode.module, () =>
            executeMatmulWithEpilogue(matmulNode, {
              consumedCount: action.consumedCount,
              epilogueOps: action.epilogueOps,
              epilogueInputRefs,
              outputDtype: action.outputDtype,
              outputNode,
              prologues,
            }),
          );

          // Cache dispatch config for next step
          await buildAndCacheDispatchConfig(
            action,
            resolvedInputRefA,
            resolvedInputRefB,
            inputCastA,
            inputCastB,
            inputAPath,
            inputBPath,
            epilogueInputRefs,
            epilogueInputPaths,
          );

          break;
        }

        case "adam-batch": {
          const useSharedEncoder = backend.name === "webgpu";
          if (useSharedEncoder) {
            // Flush the shared encoder BEFORE returning pending-release
            // buffers to the pool — otherwise an encoded-but-not-submitted
            // command could still be referencing one. backend.ops.adamStepBatch
            // also flushes internally; this earlier flush exists to make
            // flushBufferPool() safe.
            flushSharedEncoder();
            flushBufferPool();
            recordBarrier();
          }

          const adamBackend =
            getBackend(planNodes[action.nodeIndices[0]].device) ?? backend;
          const adamStepBatch = adamBackend.ops.adamStepBatch;
          if (!adamStepBatch) {
            throw new Error(
              "adamStepBatch not supported by backend; cannot execute adam-batch action",
            );
          }

          // Resolve inputs for every adamStep node, then call the backend op
          // once. The backend handles flushing, packing, and per-item fallback;
          // it returns BackendTensors which we wrap as StorageHandles via the
          // shared multi-output protocol.
          await withProfileContext("adamStep", "optimizer.step", async () => {
            const nodes: LazyIRNode[] = [];
            const inputStorages: StorageHandle[][] = [];
            const items: import("../backend/types").AdamBatchItem[] = [];
            let firstNodeIdx = -1;
            const executedIdx: number[] = [];
            for (const nodeIdx of action.nodeIndices) {
              const node = planNodes[nodeIdx];
              if (node.result) continue;
              if (firstNodeIdx < 0) firstNodeIdx = nodeIdx;
              executedIdx.push(nodeIdx);
              const inputs = node.inputs.map((ref) =>
                getInputStorage(ref, adamBackend),
              );
              const config = node.payload as AdamStepConfig;
              nodes.push(node);
              inputStorages.push(inputs);
              items.push({
                grad: inputs[0].backendTensor,
                param: inputs[1].backendTensor,
                m: inputs[2].backendTensor,
                v: inputs[3].backendTensor,
                t: inputs[4].backendTensor, // inc-2a: persistent step counter
                lr: inputs[5].backendTensor, // inc-2a: persistent learning rate
                config,
              });
            }
            if (items.length === 0) return;
            // Attribute volatile uniform recordings (Adam config) to the first
            // adamStep node — all items in a batch share hyperparameters by
            // construction, so any node is a valid replay-time config source.
            if (compilationRecording) setRecordingNodeIndex(firstNodeIdx);
            // [step-tape 2b G-cover] declare the batch members' payload
            // variance as carried by the representative's TAG_UNIFORM repack.
            // The recorder additionally asserts member↔representative payload
            // AGREEMENT per compared step, so a member whose config diverges
            // (per-group hyperparams) refuses loudly instead of replaying the
            // representative's config silently.
            if (STEP_TAPE_RECORD)
              stDeclareBatchCover(executedIdx, firstNodeIdx);
            const results = adamStepBatch(items);
            for (let i = 0; i < nodes.length; i++) {
              const r = results[i];
              assignNodeResult(
                nodes[i],
                { primary: r.param, extras: [r.m, r.v] },
                inputStorages[i].map((s) => s.backendTensor),
                inputStorages[i],
              );
            }
          });
          break;
        }

        case "batched-reduction": {
          if (compilationRecording)
            setRecordingNodeIndex(action.nodeIndices[0]);
          const batchNodes = action.nodeIndices.map((i) => planNodes[i]);
          const batchBackend = getBackend(batchNodes[0].device) ?? backend;
          const batchInputs = batchNodes.map((node) => {
            const storage = getInputStorage(node.inputs[0], batchBackend);
            return storage.backendTensor;
          });
          const batchedSum = batchBackend.ops.batchedReduction;
          if (batchedSum) {
            const results = batchedSum(
              action.reduceOp,
              batchInputs,
              action.payload.dim,
              action.payload.keepdim,
            );
            for (let bi = 0; bi < batchNodes.length; bi++) {
              batchNodes[bi].result = wrapResultAsStorage(
                batchNodes[bi].device,
                results[bi],
                [batchInputs[bi]],
                [],
              );
            }
            stats.sequentialNodes += batchNodes.length;
          } else {
            // Fallback: dispatch individually
            for (let bi = 0; bi < batchNodes.length; bi++) {
              const node = batchNodes[bi];
              const inputs = node.inputs.map((ref) =>
                getInputStorage(ref, batchBackend),
              );
              const backendInputs = inputs.map((s) => s.backendTensor);
              const result = executeOpSync(node, backendInputs, batchBackend);
              const handlerResult =
                result instanceof Promise ? await result : result;
              assignNodeResult(node, handlerResult, backendInputs, inputs);
              stats.sequentialNodes++;
            }
          }
          break;
        }

        case "sequential":
        case "view":
        case "data-source": {
          const nodeIdx = action.nodeIndex;
          const node = planNodes[nodeIdx];
          if (node.result) break;

          // Tag the node for recording hooks (volatile uniform attribution).
          if (compilationRecording) {
            setRecordingNodeIndex(nodeIdx);
            // Task #71: if this strided-elementwise node's input offset can
            // vary (view chain contains a narrow), flag its direct-dispatch
            // params buffer for a TAG_UNIFORM offset repack so the recorded
            // stream matches the generator's. Set fresh each node (overwriting
            // any prior value — so a throw in a previous node's dispatch, which
            // skips its post-exec clear below, cannot leak that node's closure
            // into this one's createParamsBuffer). Cleared after execution too.
            setPendingParamsVolatilePack(buildDirectOffsetRepack(node));
          }

          setProfileModule(node.module ?? "unknown");
          let inputs;
          try {
            inputs = node.inputs.map((ref) => getInputStorage(ref, backend));
          } catch (e) {
            // Dump context for debugging
            console.error(
              `[lowered-plan] Action #${actionIndex} kind=${action.kind} nodeIdx=${nodeIdx} node=${node.id}:${node.op} shape=${JSON.stringify(node.shape)}`,
            );
            for (let ii = 0; ii < node.inputs.length; ii++) {
              const ref = node.inputs[ii];
              if (ref.kind === "pending") {
                const depPlanIdx = planNodes.findIndex(
                  (n) => n.id === ref.node.id,
                );
                console.error(
                  `[lowered-plan]   input[${ii}]: pending node=${ref.node.id}:${ref.node.op} result=${!!ref.node.result} planIdx=${depPlanIdx}`,
                );
              }
            }
            throw e;
          }
          const backendInputs = inputs.map((s) => s.backendTensor);

          // Use synchronous dispatch to avoid microtask overhead (~5-15µs/await).
          // Only adamStep and transfer are truly async; adamStep never reaches
          // this branch because lowered-plan.ts always emits an adam-batch
          // action for it (see the comment there).
          const resultOrPromise = executeOpSync(node, backendInputs, backend);
          const handlerResult =
            resultOrPromise instanceof Promise
              ? await resultOrPromise
              : resultOrPromise;
          assignNodeResult(node, handlerResult, backendInputs, inputs);
          // Task #71: clear the pending offset-repack flag (consumed by
          // createParamsBuffer during the dispatch above; clear even if the op
          // took a non-direct path and never consumed it).
          if (compilationRecording) setPendingParamsVolatilePack(null);
          stats.sequentialNodes++;

          // Record data-source for compiled plan (re-executed each step)
          if (action.kind === "data-source" && isCompilationRecordingActive()) {
            recordWrite(gpuBuffer(node.result!.backendTensor), nodeIdx);
          }

          // Stage-4 phase-3/4.4: capture the layout metadata the stream
          // generator needs (matmul geometry, attention/reshape/strided-view
          // input layouts) — inputs are live here; plan-build frees them.
          // Single source with the build-without-execution path (see
          // captureActionLayouts). observeResult=true: cross-check reshape
          // materialization against the live result.
          captureActionLayouts(
            action,
            node,
            nodeIdx,
            backendInputs,
            backend.name,
            true,
          );
          break;
        }

        case "prologue-skip": {
          // Prologue-claimed cast nodes are skipped — their work is absorbed
          // into the matmul tile load.
          break;
        }

        case "row-program": {
          if (compilationRecording)
            setRecordingNodeIndex(action.outputNodeIndex);
          // Remap inputRefs to current step's nodes (template reuse makes
          // the original refs stale — they point to cleared previous-step nodes).
          const remappedRefs = action.inputRefs.map((ref, i) => {
            const pos = action.inputRefPositions[i];
            if (pos >= 0 && ref.kind === "pending") {
              const currentNode = planNodes[pos];
              return {
                kind: "pending" as const,
                node: currentNode,
                outputIndex: ref.outputIndex,
              };
            }
            return ref;
          });

          await executeRowProgram(
            action.program,
            remappedRefs,
            planNodes[action.outputNodeIndex],
            action.numRows,
            action.dimSize,
            planNodes.filter((_n, i) => action.coveredNodeIndices.includes(i)),
            backend,
          );
          break;
        }

        case "reclaim": {
          if (useTopLevelSharedEncoder) {
            flushSharedEncoder();
            flushBufferPool();
          }
          recordBarrier();
          break;
        }
      }

      // Liveness-based early release: free dead buffers after each action.
      // Uses the pre-built release schedule (action-index based, not step based).
      // Liveness releases stay ON during recording in planned-buffer mode:
      // the pool reuse they enable IS the planned-buffer assignment the
      // compiled plan captures (recordAlloc splits lifetimes into distinct
      // slots). In default (unbounded-arena) mode, releases are skipped
      // during recording as before — arena identities are already stable.
      if (
        livenessReleaseSchedule &&
        (!compilationRecording || compiledPlannedEnabled())
      ) {
        // Register newly-produced storages for release tracking
        const covered = getActionNodeIndices(action);
        for (const ni of covered) {
          const node = planNodes[ni];
          if (node.result) {
            livenessNodeToStorage!.set(node.id, node.result);
          }
        }

        // Release nodes scheduled for this action index
        const toRelease = livenessReleaseSchedule.get(actionIndex);
        if (toRelease) {
          for (const nodeId of toRelease) {
            if (livenessReleased!.has(nodeId)) continue;
            const storage = livenessNodeToStorage!.get(nodeId);
            if (storage && canSafelyRelease(storage, livenessNodeToStorage!)) {
              markLivenessSafe(storage);
              releaseBufferImmediate(storage);
              livenessNodeToStorage!.delete(nodeId);
              const idx = livenessNodeIdToIndex!.get(nodeId);
              if (idx !== undefined) planNodes[idx].result = undefined;
              livenessReleased!.add(nodeId);
            } else if (storage) {
              livenessDeferred!.add(nodeId);
            }
          }
        }

        // Retry deferred nodes whose rc may have dropped since last attempt.
        if (livenessDeferred!.size > 0) {
          for (const nodeId of livenessDeferred!) {
            if (livenessReleased!.has(nodeId)) {
              livenessDeferred!.delete(nodeId);
              continue;
            }
            const storage = livenessNodeToStorage!.get(nodeId);
            if (storage && canSafelyRelease(storage, livenessNodeToStorage!)) {
              markLivenessSafe(storage);
              releaseBufferImmediate(storage);
              livenessNodeToStorage!.delete(nodeId);
              const idx = livenessNodeIdToIndex!.get(nodeId);
              if (idx !== undefined) planNodes[idx].result = undefined;
              livenessReleased!.add(nodeId);
              livenessDeferred!.delete(nodeId);
            }
          }
        }
      }
    }
    if (livenessReleased?.size && ENV.TORCHLETTE_DEBUG_LIVENESS) {
      const outputCount = livenessOutputIds?.size ?? 0;
      const totalNodes = planNodes.length;
      const reclaimCount = loweredPlan.actions.filter(
        (a) => a.kind === "reclaim",
      ).length;
      console.log(
        `[liveness] released=${livenessReleased.size}/${totalNodes} (${outputCount} protected, ${reclaimCount} reclaims)`,
      );
    }
  } finally {
    // Build compiled plan from recording
    if (compilationRecording) {
      stopCompilationRecording();
      // Collect node results for the compiled plan.
      //
      // The compiled plan is a dispatch-level replay: it can only reproduce results
      // that were produced by GPU dispatches recorded during this execution. If a
      // backend op returned a cached/pre-existing buffer (no dispatch), that buffer
      // won't be in bufferToSlot and the compiled plan can't reproduce it.
      //
      // Backend ops must check isCompilationRecording() (from webgpu-state) and
      // bypass optimization caches during recording. If any result still slips
      // through, we invalidate the compiled plan rather than silently dropping it.
      const nodeResults: NodeResult[] = [];
      const unrecordedNodes: string[] = [];
      for (let i = 0; i < planNodes.length; i++) {
        const node = planNodes[i];
        // [stage-3 A] Results that pre-existed this execution are not this
        // recording's products — skip them (see preExistingResults above).
        if (preExistingResults?.has(node.id)) continue;
        if (node.results && node.results.length > 0) {
          for (let oi = 0; oi < node.results.length; oi++) {
            const sh = node.results[oi];
            if (!sh) continue;
            const bt = asGPUTensor(sh.backendTensor);
            const slot = compilationRecording.bufferToSlot.get(bt.buffer);
            if (slot !== undefined) {
              nodeResults.push({
                nodeIndex: i,
                outputIndex: oi,
                slot,
                shape: bt.shape.slice(),
                strides: bt.strides.slice(),
                dtype: bt.dtype,
                offset: bt.offset,
              });
            } else {
              unrecordedNodes.push(
                `node[${i}].results[${oi}] op=${node.op} shape=${JSON.stringify(node.shape)} dtype=${bt.dtype}`,
              );
            }
          }
        } else if (node.result) {
          const bt = asGPUTensor(node.result.backendTensor);
          const slot = compilationRecording.bufferToSlot.get(bt.buffer);
          if (slot !== undefined) {
            nodeResults.push({
              nodeIndex: i,
              outputIndex: 0,
              slot,
              shape: bt.shape.slice(),
              strides: bt.strides.slice(),
              dtype: bt.dtype,
              offset: bt.offset,
            });
          } else {
            unrecordedNodes.push(
              `node[${i}] op=${node.op} shape=${JSON.stringify(node.shape)} dtype=${bt.dtype}`,
            );
          }
        }
      }
      // [task #99 R2 / D2] The LIVE witness signal for this template (the
      // checkpoint-recompute cross-plan reads observeConsumed is blind to) —
      // sourced from the DERIVED crossPlanEdges keep-set (D2 collapse: the
      // per-producer witnessed-harvest oracle it used to read is gone). The
      // projection equals the retired oracle's set by construction (D1 gate).
      // Empty unless the template has been witnessed (K_w=2 identical reads).
      const witnessedRecomputePairs =
        options.templateFp !== undefined
          ? crossPlanEdgeKeepSet(options.templateFp, currentVariantSelection())
          : undefined;
      const compiled = buildCompiledPlan({
        commandLog: compilationRecording.commandLog,
        arena: options.bufferArena!,
        planNodes,
        bufferToSlot: compilationRecording.bufferToSlot,
        slotSources: compilationRecording.slotSources,
        nodeResults,
        witnessedRecomputePairs,
      });
      if (unrecordedNodes.length > 0) {
        // A backend op returned a cached buffer that wasn't tracked during
        // recording. The op needs to check isCompilationRecording() and bypass
        // its cache. Log details so the developer knows exactly which op to fix.
        console.warn(
          `[compiled-plan] Invalidated: ${unrecordedNodes.length} node(s) have result buffers not tracked during recording.\n` +
            `  This means a backend op returned a cached/pre-existing buffer instead of dispatching.\n` +
            `  Fix: check isCompilationRecording() from webgpu-state and bypass the cache.\n` +
            unrecordedNodes.map((s) => `  - ${s}`).join("\n"),
        );
        compiled.valid = false;
        // [stage-3 A leak fix] The build already assigned planner registry
        // entries (finalizeCompiledPlan → planMemory, owners = this plan).
        // Dropping the plan without releasing them leaks every result entry
        // as a dead-owner resultHolder FOREVER (measured ~390 MB/step when a
        // template re-recorded each step). Route through the one teardown
        // path — same rule as every other invalidation.
        destroyCompiledPlanBuffers(compiled);
      }
      // Arena conflicts no longer block compiled plan storage. The arena
      // hands evicted buffers back to the pool's release chain (see
      // buffer-arena.ts arenaAllocAt + allocateOutputBuffer + resolveOutputBuffer),
      // so orphaned buffers from conflict resolution get destroyed once their
      // owners release. The compiled plan's per-command bind group cache
      // (compiled-plan.ts) handles buffer pointer changes between replays.
      if (compiled.valid) {
        compiled.endCounters = getDispatchSequenceCounters();
        loweredPlan.compiledPlan = compiled;
      }
      // Stage-4: generate the command stream from the lowered plan for the
      // VERIFY mode only (stage-2 B1 deletion, 2026-07-08: the recorded→
      // generated cutover swap is DELETED — build-from-IR builds generated
      // plans directly on first execution, so a plan that reaches this
      // recording path is an uncovered/opt-out plan whose replay source IS the
      // recorded plan just built; swapping it for a generated one was the
      // pre-flip default's job and is dead weight after the flip).
      const wantStreamGen = ENV.TORCHLETTE_STREAM_GENERATE === "1";
      const gen =
        wantStreamGen && compiled.valid
          ? // VERIFY path: constFill applies (a plan reaching the recording
            // path recurs), but the fused row-program for stamped scalars does
            // NOT — the recording being diffed used the sequential fallback
            // for that input class, so the generator must mirror it.
            generateStream(loweredPlan, planNodes, backend, {
              coverConstFill: true,
              fuseStampedScalarExternals: false,
            })
          : undefined;

      // Phase-2 cross-check (TORCHLETTE_STREAM_GENERATE=1): diff the generated
      // stream against the recording at the command level. Divergence = a
      // generator bug or a recording gap, surfaced at build time instead of in
      // a loss curve. Coverage is reported per uncovered op class.
      if (wantStreamGen && gen) {
        const top = [...gen.uncovered.entries()]
          .sort((a, b) => b[1] - a[1])
          .slice(0, 8)
          .map(([k, v]) => `${k}×${v}`)
          .join(" ");
        // Always segment-aligned: gen-vs-record slot numbering matches only
        // up to a bijection (the recording assigns slots in recordAlloc
        // order; the generator in its own order), and the stream is
        // slot-renaming-invariant by design — so raw-slot diffStreams (right
        // for record-vs-record determinism) is too strict here. When fully
        // covered, additionally assert the flat command COUNTS agree, which
        // catches any non-node-attributed command (barriers) the generator
        // failed to emit.
        const seg = diffSegmentsAligned(gen.segments, compiled.commands);
        for (const div of seg.divergences.slice(0, 3)) {
          console.warn(
            `[stream-gen] DIVERGE node[${div.nodeIndex}] op=${planNodes[div.nodeIndex]?.op ?? "?"}: ${div.detail}`,
          );
        }
        if (gen.fullyCovered) {
          // Each constFill slot replaces exactly ONE recorded data-source
          // command (the `full` node's TAG_WRITE): its buffer is pre-filled at
          // build, so the generated stream is shorter by the constFill count.
          // Reconcile before comparing so a covered constFill isn't a false
          // divergence — the VALUE equivalence (pre-fill == write of the same
          // constant) is gated by parity-fullstack + test:gates, not here.
          const constFillCount = gen.slots.filter(
            (s) => s.kind === "constFill",
          ).length;
          const countMatch =
            gen.commands.length + constFillCount === compiled.commands.length;
          if (!countMatch) {
            console.warn(
              `[stream-gen] DIVERGE flat command count: generated ${gen.commands.length} (+${constFillCount} constFill) vs recorded ${compiled.commands.length}`,
            );
          }
          // Params/uniform DATA multiset must agree. diffSegmentsAligned
          // compares DISPATCH by pipeline + workgroups + binding SLOTS but
          // NOT the params bytes — a wrong config (e.g. a shared paramsData
          // array stored by reference and later mutated) is otherwise
          // invisible yet computes garbage. This guard closes that hole.
          const paramsBag = (ss: typeof gen.slots) => {
            const m = new Map<string, number>();
            for (const s of ss)
              if (s.kind === "params") {
                const k = Array.from(s.data).join(",");
                m.set(k, (m.get(k) ?? 0) + 1);
              }
            return m;
          };
          const gpar = paramsBag(gen.slots);
          const rpar = paramsBag(compiled.slots);
          let paramsMatch = gpar.size === rpar.size;
          if (paramsMatch)
            for (const [k, v] of gpar)
              if (rpar.get(k) !== v) {
                paramsMatch = false;
                break;
              }
          if (!paramsMatch) {
            console.warn(
              `[stream-gen] DIVERGE params-data multiset (generated vs recorded uniform bytes differ)`,
            );
          }
          console.log(
            `[stream-gen] FULLY GENERATED ${gen.actionCount} actions: ${seg.verifiedActions}/${gen.segments.length} segments verified, ${seg.divergences.length} diverged, flat ${gen.commands.length}/${compiled.commands.length} cmds${countMatch && paramsMatch ? " ✓" : " ✗"}`,
          );
        } else {
          console.log(
            `[stream-gen] VERIFIED ${seg.verifiedActions}/${gen.segments.length} segments (${seg.verifiedCommands} cmds, ${seg.divergences.length} diverged, ${seg.unmatched} unmatched; covered ${gen.coveredActions}/${gen.actionCount} actions; uncovered: ${top})`,
          );
        }
      }

      compilationRecording = null;
    }

    // Clear arena conflict flag
    if (getArenaConflictDetected()) {
      clearArenaConflictDetected();
    }

    // Deactivate the scalar-ref table for this plan execution
    clearActiveScalarTable();

    // Deactivate buffer arena
    if (options.bufferArena && useTopLevelSharedEncoder) {
      clearActiveArena();
      clearArenaExternalInputBuffers();
    }

    // Scoped result guarantee: only re-execute output nodes that optimization
    // passes absorbed without producing individual results. Uses recursive
    // dependency resolution since absorbed intermediates may depend on each other.
    // Non-output nodes are internal to the plan — their results are consumed
    // by fusion groups or epilogues, so no external caller needs them.
    {
      const outputNodeIds = plan.outputIndices
        ? new Set(plan.outputIndices.map((i) => plan.nodes[i].id))
        : null; // null = all nodes (conservative fallback)
      const { executeNode: execNode } = await import("./sequential");
      const forceWithDeps = async (node: LazyIRNode): Promise<void> => {
        if (node.result) return;
        for (const inp of node.inputs) {
          if (inp.kind === "pending" && !inp.node.result) {
            await forceWithDeps(inp.node);
          }
        }
        await execNode(node, backend);
      };
      for (const node of planNodes) {
        if (!node.result && (!outputNodeIds || outputNodeIds.has(node.id))) {
          await forceWithDeps(node);
        }
      }
    }

    if (useTopLevelSharedEncoder) {
      endSharedEncoder();
    }
  }

  // Get the result from the last original plan node
  const lastNode = plan.nodes[plan.nodes.length - 1];
  if (!lastNode.result) {
    throw new Error("Lowered plan execution failed: no result for last node");
  }

  // Cache stats on the lowered plan so the compiled path can return them
  loweredPlan.cachedStats = stats;

  // Converged-to-lowered WITNESS STAMP (docs/step-data-dependence-design.md §D4
  // attempt #6). A template holding a valid compiled plan resets its counter
  // (any compiler, so DORMANT on the main tree); one that has run lowered for
  // K_w consecutive executions with no compiled plan has converged to lowered —
  // an uncovered recurring producer stranded without the recorded build. Stamp
  // its PRIMARY action outputs (derivable from loweredPlan.actions, the same set
  // actionOutputHarvestPairs uses for oi=0) so its cross-plan edges are witnessed
  // without compilation, reproducing the recorded build's witnessing coverage.
  // The stamp is metadata only; a lowered plan owns no planner registry entry, so
  // its stamped outputs are never claimable/releasable (releasableLastReader
  // requires a converged observed-liveness template + a registry entry) — the
  // stamp purely feeds the cross-plan witness the harvest and the overlay-release
  // now consult.
  if (isObservedLivenessEnabled() && options.templateFp !== undefined) {
    if (loweredPlan.compiledPlan?.valid) {
      loweredWitnessRuns.set(options.templateFp, 0);
    } else {
      const runs = (loweredWitnessRuns.get(options.templateFp) ?? 0) + 1;
      loweredWitnessRuns.set(options.templateFp, runs);
      if (runs >= 2) stampLoweredActionOutputs(loweredPlan, planNodes, options.templateFp);
    }
  }

  if (TAPE_PROFILE) tpAdd("lowered-exec", performance.now() - tpL0);
  return { result: lastNode.result, stats };
}

/** Stamp the PRIMARY (oi=0) action-output results of a lowered plan with its
 *  cross-plan value identity (templateFp, nodeIndex, 0) — the same coordinate
 *  the compiled-replay harvest stamps (compiled-plan.ts stampResult) and the
 *  same output set actionOutputHarvestPairs enumerates for oi=0. Lets an
 *  uncovered lowered-forever producer participate in cross-plan witnessing. */
function stampLoweredActionOutputs(
  loweredPlan: LoweredPlan,
  planNodes: LazyIRNode[],
  templateFp: number,
): void {
  const isOutput = new Set<number>();
  for (const action of loweredPlan.actions) {
    switch (action.kind) {
      case "sequential":
      case "view":
      case "data-source":
        isOutput.add(action.nodeIndex);
        break;
      case "fused":
        isOutput.add(action.outputNodeIndex);
        for (const ni of action.additionalOutputNodeIndices) isOutput.add(ni);
        break;
      case "matmul-epilogue":
      case "row-program":
        isOutput.add(action.outputNodeIndex);
        break;
      case "adam-batch":
      case "batched-reduction":
        for (const ni of action.nodeIndices) isOutput.add(ni);
        break;
      case "reclaim":
        break;
    }
  }
  for (const i of isOutput) {
    const n = planNodes[i];
    if (!n) continue;
    if (n.result) stampResult(n.result, templateFp, i, 0);
    // Multi-output extras (oi>0): the recorded-build harvest stamps them too
    // (SDPA / packed optimizer side outputs), so match its coverage — the D4
    // crux's afe3da58 recovers its full 76-pair set only with the extras.
    if (n.results) {
      for (let oi = 1; oi < n.results.length; oi++) {
        const ro = n.results[oi];
        if (ro) stampResult(ro, templateFp, i, oi);
      }
    }
  }
}

// ============================================================================
// Optimized Plan Execution (entry point)
// ============================================================================

/**
 * Execute a plan with automatic fusion optimization.
 *
 * Pipeline: analyze graph → build lowered plan → execute lowered plan.
 * The lowered plan is built purely from graph analysis (no execution needed)
 * and cached for subsequent steps. executeLoweredPlan() is the sole execution
 * engine for both first-run and replay paths.
 *
 * @param plan - The execution plan
 * @param backend - The backend to use
 * @param options - Optimization options
 */
export async function executePlanOptimized(
  plan: ExecutionPlan,
  backend: Backend & {
    device?: { limits?: { maxStorageBuffersPerShaderStage?: number } };
  },
  options: OptimizedExecutionOptions = {},
): Promise<OptimizedExecutionResult> {
  if (plan.nodes.length === 0) {
    throw new Error("Cannot execute empty plan");
  }

  // Pre-tune matmul shapes if backend supports it
  await pretunePlanMatmuls(plan, backend);

  // Fusion (epilogue/prologue/row-program directives, fused elementwise
  // kernels, and the whole compiled-plan / stream-generator machinery) is
  // webgpu-only — its dispatch paths call into dispatchMatmul/dispatchFusedKernel
  // which require an initialized WebGPU context. The backend's capability, not
  // the caller's requested flag, is the source of truth: a non-fused backend
  // (CPU) can NEVER fuse, even if enableFusion:true is passed. Otherwise the CPU
  // path builds webgpu-only directives and crashes at dispatch. AND the request
  // with the backend capability so CPU cleanly takes the sequential path.
  const enableFusion =
    (options.enableFusion ?? isFusedBackend(backend)) &&
    isFusedBackend(backend);
  const enableVectorization = options.enableVectorization ?? true;

  // Fall back to simple sequential execution when fusion is disabled entirely.
  if (!enableFusion) {
    const result = await executePlanSequential(plan, backend, {
      enableEarlyRelease: options.enableEarlyRelease,
    });
    const stats: OptimizedExecutionStats = {
      totalNodes: plan.nodes.length,
      fusedNodes: 0,
      sequentialNodes: plan.nodes.length,
      fusionGroups: 0,
      fusionEnabled: enableFusion,
    };
    return { result, stats };
  }

  // Derive external node IDs from plan.outputIndices (set by the engine or
  // deserialized from the wire). This replaces the getPendingNodeIds()
  // side-channel — the plan carries its own output contract.
  let externalNodeIds: Set<number> | undefined;
  if (plan.outputIndices) {
    const ids = new Set<number>();
    for (const idx of plan.outputIndices) ids.add(plan.nodes[idx].id);
    if (ids.size > 0) externalNodeIds = ids;
  } else {
    // Fallback for callers that don't set outputIndices (legacy / tests).
    try {
      const { getPendingNodeIds } = await import("../runtime/tensor");
      const pending = getPendingNodeIds();
      if (pending.size > 0) externalNodeIds = pending;
    } catch {
      // runtime/tensor not available — no external node tracking
    }
  }

  // Query device storage buffer limit to constrain fusion group size.
  const maxStorageBuffers: number | undefined =
    backend.device?.limits?.maxStorageBuffersPerShaderStage;

  // Compute 64-bit structural fingerprint (two parallel FNV-1a hashes).
  // Primary is the cache key for O(1) lookup; secondary is checked on hit
  // to detect collisions (effective collision probability: 2^-64).
  // [tape-1a] G0 measurement seams (src/core/tape-profile.ts; sunset: 1c).
  const tpF0 = TAPE_PROFILE ? performance.now() : 0;
  // [D0 variant seam] Mix the current step VARIANT's token into the fingerprint
  // via the SAME token-mixing arg the islands I1 partition uses
  // (docs/step-data-dependence-design.md §3.3/§3.4). The token is delivered from
  // the step's per-step selection (NOT the S3 island-merge resolver, §3.4
  // verdict). SINGLETON == ABSENT: `variantToken()` returns `undefined` for the
  // base `"train"` variant, the exact value `computePlanFingerprint` treats as
  // "no discriminator" — so this is BYTE-IDENTICAL to the pre-variant path on
  // every plan today (null-clean). A future residual variant returns a nonzero
  // token, re-keying its template + re-witnessing its tape through this one seam.
  const defaultFingerprint = computePlanFingerprint(
    plan.nodes,
    externalNodeIds,
    variantToken(),
  );
  // [S3 FUSE LIVE WIRING] Resolve a live partition-merge edit: with an empty
  // store this returns the default fingerprint UNCHANGED (byte-identical null
  // path); with a realized merge it re-fingerprints the plan (the merged
  // boundaryHash mixed in as the I1 token) so the template re-keys + the tape
  // re-witnesses under the merged partition (§5.3).
  const { fingerprint, edit: partitionEdit } = resolveEditedFingerprint(
    plan.nodes,
    externalNodeIds,
    defaultFingerprint,
  );
  if (TAPE_PROFILE) tpAdd("fingerprint", performance.now() - tpF0);
  const cachedTemplate = fusionAnalysisCache.get(fingerprint.primary);

  let planNodes: LazyIRNode[];
  let loweredPlan: LoweredPlan;

  // Validate cache hit via secondary fingerprint.
  let validatedTemplate = cachedTemplate?.loweredPlan
    ? cachedTemplate
    : undefined;
  if (
    validatedTemplate &&
    validatedTemplate.fingerprintSecondary !== fingerprint.secondary
  ) {
    // [S3] An alias shares the default's loweredPlan — never destroy it here; the
    // default (owner) tears it down. Just drop the alias entry (rebuilt on demand).
    if (validatedTemplate.aliasOf === undefined) {
      if (validatedTemplate.loweredPlan?.compiledPlan) {
        destroyCompiledPlanBuffers(validatedTemplate.loweredPlan.compiledPlan);
      }
      if (validatedTemplate.loweredPlan) {
        destroyScalarTable(validatedTemplate.loweredPlan);
      }
    }
    validatedTemplate = undefined;
    fusionAnalysisCache.delete(fingerprint.primary);
  }

  if (!validatedTemplate) {
    notePayloadThrash(fingerprint, plan.nodes);
  }

  if (validatedTemplate) {
    // ── Cache hit: reuse existing lowered plan ──
    // If a cross-session recycle (or arena eviction) cleared the
    // bufferArena, recreate it as an empty arena that the executor
    // will populate during the upcoming plan execution. Without this,
    // the cache-hit path would proceed with options.bufferArena =
    // undefined, the shouldCompile check would fail, and compiled-plan
    // recording would never engage for the rest of the process — every
    // subsequent session would run 3-4× slower than the first because
    // every plan would go through the slow normal path forever.
    if (backend.name === "webgpu" && !validatedTemplate.bufferArena) {
      validatedTemplate.bufferArena = { resolve: [], alloc: [] };
    }
    const tpH0 = TAPE_PROFILE ? performance.now() : 0;
    planNodes = validatedTemplate.finalPerm.map((i) => plan.nodes[i]);
    if (TAPE_PROFILE) tpAdd("template-hit-perm", performance.now() - tpH0);

    // Re-apply graph rewrites to fresh nodes. The lowered plan was built from
    // rewritten refs (identity-cast bypass, sum-reshape fusion, CSE, etc.).
    // Fresh nodes from deserialization or new steps have original refs. Re-running
    // the cheap O(n) passes ensures input refs match the lowered plan's assumptions.
    {
      const tpM0 = TAPE_PROFILE ? performance.now() : 0;
      const consumers = new Map<number, LazyIRNode[]>();
      const consumerCount = new Map<number, number>();
      for (const node of planNodes) {
        for (const inp of node.inputs) {
          if (inp.kind === "pending") {
            consumerCount.set(
              inp.node.id,
              (consumerCount.get(inp.node.id) ?? 0) + 1,
            );
            if (!consumers.has(inp.node.id)) consumers.set(inp.node.id, []);
            consumers.get(inp.node.id)!.push(node);
          }
        }
      }
      if (TAPE_PROFILE) tpAdd("consumer-maps", performance.now() - tpM0);
      runPasses(
        { planNodes, consumers, consumerCount },
        new Set(),
        SIMPLIFICATION_PASSES,
      );
    }

    loweredPlan = validatedTemplate.loweredPlan!;
    // Replay stored plan analysis on cache hits when profiling is enabled.
    // Only replay once per profiler reset (tracked via generation counter).
    if (isProfilingEnabled() && validatedTemplate.planAnalysis) {
      const gen = getPlanAnalysisGeneration();
      if (validatedTemplate.replayedGeneration !== gen) {
        validatedTemplate.replayedGeneration = gen;
        recordPlanAnalysis({ ...validatedTemplate.planAnalysis });
      }
    }
    if (ENV.TORCHLETTE_DEBUG_COMPILED) {
      const extCount = externalNodeIds?.size ?? 0;
      const n0 = plan.nodes[0];
      const n0Info = n0
        ? `n0=${n0.op}(${n0.dtype},${JSON.stringify(n0.shape)})`
        : "";
      console.log(
        `[template] HIT fp=0x${fingerprint.primary.toString(16)} nodes=${plan.nodes.length} ext=${extCount} compiled=${!!loweredPlan.compiledPlan?.valid} ${n0Info}`,
      );
    }
  } else if (
    partitionEdit &&
    fingerprint.primary !== defaultFingerprint.primary &&
    fusionAnalysisCache.get(defaultFingerprint.primary)?.loweredPlan
  ) {
    // ── [S3 FUSE LIVE WIRING] Edited-fp miss: ALIAS the default template ──
    // A grouping-only partition merge changes island BOUNDARIES, never the
    // lowered plan. Rather than build a fresh (numerically-perturbing) template,
    // the edited fp SHARES the default template's execution artifact (loweredPlan
    // + bufferArena) and overrides only the reified partition (merged, island
    // count −1) + the secondary fingerprint. Execution is byte-identical → the
    // trajectory is null; only the template key + recorded partitionHash change,
    // which is exactly what re-fingerprints the plan and re-witnesses the tape.
    const base = fusionAnalysisCache.get(defaultFingerprint.primary)!;
    const dflt = base.partition;
    const m = mergeIslands(dflt, partitionEdit.aIdx, partitionEdit.bIdx);
    const mergedPartition = m.ok ? m.partition : dflt;
    if (partitionEdit.token === undefined && m.ok)
      partitionEdit.token = mergedPartition.boundaryHash;
    const aliasTemplate: FusionAnalysisTemplate = {
      ...base,
      partition: mergedPartition,
      fingerprintSecondary: fingerprint.secondary,
      aliasOf: defaultFingerprint.primary,
    };
    fusionAnalysisCache.set(fingerprint.primary, aliasTemplate);
    planNodes = base.finalPerm.map((i) => plan.nodes[i]);
    // Re-apply the cheap simplification passes (same as the normal HIT path) so
    // fresh nodes' input refs match the shared lowered plan's assumptions.
    {
      const consumers = new Map<number, LazyIRNode[]>();
      const consumerCount = new Map<number, number>();
      for (const node of planNodes) {
        for (const inp of node.inputs) {
          if (inp.kind === "pending") {
            consumerCount.set(
              inp.node.id,
              (consumerCount.get(inp.node.id) ?? 0) + 1,
            );
            if (!consumers.has(inp.node.id)) consumers.set(inp.node.id, []);
            consumers.get(inp.node.id)!.push(node);
          }
        }
      }
      runPasses(
        { planNodes, consumers, consumerCount },
        new Set(),
        SIMPLIFICATION_PASSES,
      );
    }
    loweredPlan = base.loweredPlan!;
  } else {
    // ── Cache miss: run full analysis + build lowered plan ──

    // [observed-liveness] A new template appeared this step — defers the
    // no-new-template-this-step pruning condition (a step that grows the plan
    // set is not yet a stable observation).
    noteNewTemplate();

    const analysis = analyzeGraph(
      plan.nodes,
      externalNodeIds,
      maxStorageBuffers,
    );
    planNodes = analysis.planNodes;

    // Build template and cache it
    const origIdToPos = buildIdPositionMap(plan.nodes);
    const finalPerm = planNodes.map((n) => origIdToPos.get(n.id) as number);
    const finalIdToPos = buildIdPositionMap(planNodes);

    const cachedSegments: CachedSegmentDesc[] = analysis.segments.map((seg) => {
      if (seg.kind === "sequential") {
        return {
          kind: "sequential" as const,
          finalPoss: seg.nodes.map(
            (n: LazyIRNode) => finalIdToPos.get(n.id) as number,
          ),
        };
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      if ((seg as any).kind === "reduction") {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const rg = (seg as any).group;
        return {
          kind: "reduction" as const,
          finalPoss: rg.nodes.map(
            (n: LazyIRNode) => finalIdToPos.get(n.id) as number,
          ),
          reductionFinalPos: finalIdToPos.get(rg.reductionNode.id) as number,
          preambleFinalPoss: rg.preambleNodes.map(
            (n: LazyIRNode) => finalIdToPos.get(n.id) as number,
          ),
          epilogueFinalPoss: rg.epilogueNodes.map(
            (n: LazyIRNode) => finalIdToPos.get(n.id) as number,
          ),
          outputFinalPos: finalIdToPos.get(rg.outputNode.id) as number,
          preambleOps: rg.preambleOps,
          preambleInputDtypes: rg.preambleInputDtypes,
          epilogueOps: rg.epilogueOps,
          outputDtype: rg.outputDtype,
          isMean: rg.isMean,
        };
      }
      return {
        kind: "fused" as const,
        finalPoss: seg.group.nodes.map((n) => finalIdToPos.get(n.id) as number),
        outputFinalPos: finalIdToPos.get(seg.group.outputNode.id) as number,
        additionalOutputFinalPoss: (seg.group.additionalOutputNodes ?? []).map(
          (n) => finalIdToPos.get(n.id) as number,
        ),
        neededIntermediateFinalPoss: (seg.group.neededIntermediates ?? []).map(
          (n) => finalIdToPos.get(n.id) as number,
        ),
      };
    });

    // [islands I0] Reify the partition from the positional segment descriptors —
    // pure derivation, no decision change. All three CachedSegmentDesc kinds
    // carry members as finalPoss. (This is the DEFAULT fp's template; an accepted
    // merge is realized in the alias fast-path above, which SHARES this template's
    // loweredPlan and overrides only the partition — so execution stays byte-
    // identical. Here we only bootstrap the merged token so the NEXT step
    // re-fingerprints to that alias.)
    let reifiedPartition = reifyPartition(cachedSegments);
    if (partitionEdit) {
      const m = mergeIslands(
        reifiedPartition,
        partitionEdit.aIdx,
        partitionEdit.bIdx,
      );
      if (m.ok) {
        if (partitionEdit.token === undefined)
          partitionEdit.token = m.partition.boundaryHash;
        // Fallback: an edited fp reached the normal miss (the default template was
        // evicted after the token bootstrapped, so the alias fast-path above could
        // not fire). Reflect the merge here too, so the reified partition stays a
        // true projection (island count −1) even in this rare eviction race.
        if (fingerprint.primary !== defaultFingerprint.primary)
          reifiedPartition = m.partition;
      }
    }

    const template: FusionAnalysisTemplate = {
      finalPerm,
      segments: cachedSegments,
      partition: reifiedPartition,
      fingerprintSecondary: fingerprint.secondary,
      epilogueClaimedOrigPoss: [...analysis.epilogueClaimedIds].map(
        (id) => origIdToPos.get(id) as number,
      ),
      prologueClaimedOrigPoss: [...analysis.prologueClaimedIds].map(
        (id) => origIdToPos.get(id) as number,
      ),
      epilogueChains: [...analysis.matmulEpilogueChains].map(
        ([mmId, epilogueIds]) =>
          [
            origIdToPos.get(mmId) as number,
            epilogueIds.map((id) => origIdToPos.get(id) as number),
          ] as [number, number[]],
      ),
      prologueDescs: [...analysis.matmulPrologues].map(
        ([mmId, prologues]) =>
          [
            origIdToPos.get(mmId) as number,
            prologues.map((p) => ({
              inputIndex: p.inputIndex,
              castOrigPos: origIdToPos.get(p.castNodeId) as number,
              fromDtype: p.fromDtype,
              toDtype: p.toDtype,
            })),
          ] as [
            number,
            Array<{
              inputIndex: 0 | 1;
              castOrigPos: number;
              fromDtype: DType;
              toDtype: DType;
            }>,
          ],
      ),
    };

    // Build lowered plan from analysis (the sole plan-building path)
    loweredPlan = buildLoweredPlanFromAnalysis({
      segments: analysis.segments,
      planNodes,
      nodeIdToFinalPos: finalIdToPos,
      prologueClaimedIds: analysis.prologueClaimedIds,
      rowProgramMatches: analysis.rowProgramMatches,
      matmulDirectives: analysis.matmulDirectives,
      enableVectorization,
      // Insert reclaim points so released buffers get promoted to the available pool.
      // Matches the enableLivenessRelease default-on behavior.
      reclaimInterval:
        ENV.TORCHLETTE_LIVENESS_RELEASE !== "0" ? 300 : undefined,
    });
    template.loweredPlan = loweredPlan;
    // Create empty arena for WebGPU — populated during first execution,
    // reused on subsequent cache hits for stable buffer identities.
    if (backend.name === "webgpu") {
      template.bufferArena = { resolve: [], alloc: [] };
    }

    fusionAnalysisCache.set(fingerprint.primary, template);

    // Evict old templates if cache grows too large (each has a GPU arena)
    const MAX_CACHED_TEMPLATES = 16;
    if (fusionAnalysisCache.size > MAX_CACHED_TEMPLATES) {
      // Remove oldest entries (first in map insertion order)
      const toRemove = fusionAnalysisCache.size - MAX_CACHED_TEMPLATES;
      let removed = 0;
      // [S3] A default template with a live alias must NOT be evicted (the alias
      // borrows its loweredPlan/arena). Collect fps that are alias targets.
      const aliasTargets = new Set<number>();
      for (const [, t] of fusionAnalysisCache)
        if (t.aliasOf !== undefined) aliasTargets.add(t.aliasOf >>> 0);
      for (const [fp, tmpl] of fusionAnalysisCache) {
        if (removed >= toRemove) break;
        if (fp === fingerprint.primary) continue; // don't evict the one we just added
        if (aliasTargets.has(fp >>> 0)) continue; // owner of a live alias — keep
        if (tmpl.aliasOf === undefined) {
          // Owner (or unaliased) template: it owns its resources — destroy them.
          if (tmpl.bufferArena) {
            destroyArena(tmpl.bufferArena);
          }
          if (tmpl.loweredPlan?.compiledPlan) {
            destroyCompiledPlanBuffers(tmpl.loweredPlan.compiledPlan);
          }
          if (tmpl.loweredPlan) destroyScalarTable(tmpl.loweredPlan);
        }
        // Aliases: drop the map entry only (resources belong to the owner).
        fusionAnalysisCache.delete(fp);
        removed++;
      }
    }

    if (ENV.TORCHLETTE_DEBUG_COMPILED) {
      // Log op histogram for structural comparison
      const opCounts = new Map<string, number>();
      for (const n of plan.nodes)
        opCounts.set(n.op, (opCounts.get(n.op) ?? 0) + 1);
      const opSummary = [...opCounts.entries()]
        .sort((a, b) => b[1] - a[1])
        .map(([op, c]) => `${op}:${c}`)
        .join(" ");
      const extCount = externalNodeIds?.size ?? 0;
      // Check if this is a near-miss with an existing template (same node count ±2)
      let nearMiss = "";
      for (const [
        existingFp,
        existingTemplate,
      ] of fusionAnalysisCache.entries()) {
        if (existingFp !== fingerprint.primary) {
          const nOld = existingTemplate.finalPerm?.length ?? 0;
          const diff = plan.nodes.length - nOld;
          if (Math.abs(diff) <= 2 && diff !== 0) {
            nearMiss += ` [near: fp=0x${(existingFp >>> 0).toString(16)} nodes=${nOld} diff=${diff > 0 ? "+" : ""}${diff}]`;
            // Diff the op sequences to find the extra node
            // Reconstruct old op list from the cached template
            // We can't access old plan.nodes, but we have the current plan.nodes
            // and the old externalNodeIds. Let's find which nodes are new externals.
            if (externalNodeIds && diff === 1) {
              // Find external nodes that could be the extra
              const extNodes = plan.nodes.filter((n) =>
                externalNodeIds.has(n.id),
              );
              const castExts = extNodes.filter((n) => n.op === "cast");
              if (castExts.length > 0) {
                const c = castExts[castExts.length - 1]; // likely the new one
                const cIdx = plan.nodes.indexOf(c);
                const inp = c.inputs[0];
                const inputOp =
                  inp.kind === "pending" ? inp.node.op : "(materialized)";
                const inputDtype =
                  inp.kind === "pending"
                    ? inp.node.dtype
                    : inp.kind === "materialized"
                      ? inp.storage.backendTensor.dtype
                      : "?";
                console.log(
                  `  EXTRA CAST: node[${cIdx}] cast ${inputDtype}->${c.dtype} shape=${JSON.stringify(c.shape)} input=${inputOp} isExt=${externalNodeIds.has(c.id)}`,
                );
                // Also show the consumer of this cast
                const consumers = plan.nodes.filter((n) =>
                  n.inputs.some(
                    (i) => i.kind === "pending" && i.node.id === c.id,
                  ),
                );
                for (const con of consumers) {
                  console.log(
                    `    consumer: ${con.op} shape=${JSON.stringify(con.shape)} dtype=${con.dtype}`,
                  );
                }
              }
            }
          }
        }
      }
      console.log(
        `[template] MISS fp=0x${fingerprint.primary.toString(16)} nodes=${plan.nodes.length} ext=${extCount} actions=${loweredPlan.actions.length} cacheSize=${fusionAnalysisCache.size}${nearMiss}\n  ops: ${opSummary}`,
      );
    }

    // Collect plan analysis for profiling (structural, no execution needed)
    // Store on template so cache hits can replay it when profiling is enabled later.
    const pa = collectProfilingStats(
      analysis.segments,
      analysis.epilogueClaimedIds,
      analysis.prologueClaimedIds,
      analysis.matmulEpilogueChains,
      plan.nodes.length,
    );
    const storedTemplate = fusionAnalysisCache.get(fingerprint.primary);
    if (storedTemplate) storedTemplate.planAnalysis = pa;
  }

  // Execute via the lowered plan — the sole execution engine.
  // The buffer arena (created at template construction) provides stable buffer
  // identities across steps for bind group cache hits.
  // Allow runtime toggling from the browser via globalThis (env vars don't
  // exist in browsers and Vite replaces process.env at compile time).
  const arenaDisabled =
    !!ENV.TORCHLETTE_NO_ARENA ||
    !!(globalThis as { __torchletteNoArena?: boolean }).__torchletteNoArena ||
    // Per-engine opt-out: the trainer disables the arena when checkpointing so
    // the liveness early-release can actually free forward activations (the
    // retained arena would otherwise keep them resident, defeating the point of
    // checkpointing — measured 124M/seq256: steady 6.6GB->2.6GB, peak unchanged
    // 7.8->7.66GB). Trade-off: these plans run lowered (no compiled replay),
    // consistent with checkpointing already trading compute for memory.
    options.arenaDisabled === true;
  const bufferArena = arenaDisabled
    ? undefined
    : ((cachedTemplate ?? fusionAnalysisCache.get(fingerprint.primary))
        ?.bufferArena as BufferArena | undefined);
  if (!STEP_TAPE_RECORD) {
    const r = await executeLoweredPlan(plan, planNodes, loweredPlan, backend, {
      bufferArena,
      templateFp: fingerprint.primary,
    });
    // [observed-liveness] Tag any compiled plan built this call with its
    // template fp so future replays stamp harvested results with their
    // cross-plan identity. No-op unless build-from-IR is active.
    if (loweredPlan.compiledPlan) {
      loweredPlan.compiledPlan.templateFp = fingerprint.primary;
    }
    return r;
  }
  // [step-tape 1b] pure observation: record the plan execution (template id +
  // payload/scalar image for the guard-3 diff), and stamp the template fp on
  // the compiled plan so its invalidation cascades to tapes (guard 4).
  // [step-object phase 6] surface the plan's islands partition-identity token
  // (I1 boundaryHash — already the detector's, `:259`) so the step object's
  // partition facet is a read-only projection (no second owner of membership).
  stBeginPlan(
    fingerprint.primary,
    planNodes,
    cachedTemplate?.partition?.boundaryHash ?? 0,
  );
  try {
    const r = await executeLoweredPlan(plan, planNodes, loweredPlan, backend, {
      bufferArena,
      templateFp: fingerprint.primary,
    });
    if (loweredPlan.compiledPlan) {
      loweredPlan.compiledPlan.tapeFp = fingerprint.primary;
      loweredPlan.compiledPlan.templateFp = fingerprint.primary;
    }
    // [step-tape 1c] capture this NORMAL compiled step as a replay-skeleton
    // candidate (promoted iff the recorder deems it eligible at markStep) and,
    // under TAPE_VERIFY, cross-check the skeleton we would have replayed.
    if (STEP_TAPE_REPLAY && loweredPlan.compiledPlan?.valid && bufferArena) {
      stCaptureCompiledStep(
        fingerprint.primary,
        planNodes,
        loweredPlan,
        bufferArena as BufferArena,
        loweredPlan.compiledPlan.commands,
      );
    }
    return r;
  } finally {
    stEndPlan();
  }
}

// ============================================================================
// Profiling (structural analysis, no execution needed)
// ============================================================================

function collectProfilingStats(
  segments: ExecutionSegment[],
  excludedIds: Set<number>,
  prologueClaimedIds: Set<number>,
  matmulEpilogueChains: Map<number, number[]>,
  totalNodes: number,
): PlanAnalysis {
  let fusedSegCount = 0;
  let seqSegCount = 0;
  let fusedNodeCount = 0;
  let fusionGroupCount = 0;
  const sequentialOps: Record<string, number> = {};
  const unfusedByShape: Record<
    string,
    { count: number; ops: Record<string, number> }
  > = {};

  const recordUnfused = (node: LazyIRNode) => {
    sequentialOps[node.op] = (sequentialOps[node.op] ?? 0) + 1;
    if (isFusibleOp(node.op)) {
      const shapeKey = node.shape.join(",");
      let bucket = unfusedByShape[shapeKey];
      if (!bucket) {
        bucket = { count: 0, ops: {} };
        unfusedByShape[shapeKey] = bucket;
      }
      bucket.count++;
      bucket.ops[node.op] = (bucket.ops[node.op] ?? 0) + 1;
    }
  };

  for (const segment of segments) {
    if (segment.kind === "fused" && segment.group.nodes.length >= 2) {
      fusedSegCount++;
      fusedNodeCount += segment.group.nodes.length;
      fusionGroupCount++;
    } else if (segment.kind === "fused") {
      seqSegCount++;
      for (const node of segment.group.nodes) recordUnfused(node);
    } else if (segment.kind === "sequential") {
      seqSegCount++;
      for (const node of segment.nodes) {
        if (!excludedIds.has(node.id) && !prologueClaimedIds.has(node.id)) {
          recordUnfused(node);
        } else {
          sequentialOps[node.op] = (sequentialOps[node.op] ?? 0) + 1;
        }
      }
    }
  }

  // Count reduction preamble opportunities from sequential segments
  let reductionFusionEstimate = 0;
  for (const segment of segments) {
    if (segment.kind !== "sequential") continue;
    for (let i = 0; i < segment.nodes.length - 1; i++) {
      const cur = segment.nodes[i];
      const next = segment.nodes[i + 1];
      if (
        isFusibleOp(cur.op) &&
        cur.op !== "cast" &&
        (next.op === "sum" || next.op === "mean")
      ) {
        reductionFusionEstimate++;
      }
    }
  }

  const planAnalysisRef: PlanAnalysis = {
    planIndex: 0, // assigned by recordPlanAnalysis
    totalNodes,
    segments: { fused: fusedSegCount, sequential: seqSegCount },
    fusedNodes: fusedNodeCount,
    fusionGroups: fusionGroupCount,
    epilogueFusions: matmulEpilogueChains.size,
    reductionFusions: reductionFusionEstimate,
    sequentialOps,
    unfusedByShape,
  };
  recordPlanAnalysis(planAnalysisRef);
  return planAnalysisRef;
}
