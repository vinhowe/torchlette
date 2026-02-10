export type { Shape } from "../backend/cpu";
export {
  add,
  cpuBackend,
  expand,
  gather,
  matmul,
  mean,
  mul,
  relu,
  reshape,
  scatterAdd,
  sqrt,
  sub,
  sum,
  Tensor,
  tensorFromArray,
  transpose,
} from "../backend/cpu";
export { mockBackend } from "../backend/mock";
export {
  getActiveBackend,
  getBackend,
  ops,
  registerBackend,
  setBackend,
  withBackend,
} from "../backend/registry";
export type { Backend, BackendOps } from "../backend/types";
export type { RuntimeEngineOptions } from "../runtime/engine";
export {
  add as runtimeAdd,
  cpu as runtimeCpu,
  expand as runtimeExpand,
  gather as runtimeGather,
  item as runtimeItem,
  matmul as runtimeMatmul,
  mean as runtimeMean,
  mul as runtimeMul,
  RuntimeEngine,
  relu as runtimeRelu,
  reshape as runtimeReshape,
  scatterAdd as runtimeScatterAdd,
  sqrt as runtimeSqrt,
  sub as runtimeSub,
  sum as runtimeSum,
  tensorFromArray as runtimeTensorFromArray,
  transpose as runtimeTranspose,
  view as runtimeView,
} from "../runtime/engine";
export type { BaseId } from "../runtime/tensor";
export { Tensor as RuntimeTensor } from "../runtime/tensor";
export type {
  CompiledCacheEntry,
  CompiledCacheKey,
  InputSignature,
} from "./compile-cache";
export {
  CompiledCache,
  generateCacheKey,
  hashIRGraph,
  serializeCacheKey,
} from "./compile-cache";
export type {
  AccessTarget,
  AliasGroup,
  ExtendedCompiledCacheKey,
  ExternalizeRequest,
  FunctionalizationResult,
  FunctionalizedMutation,
  NullStateSentinel,
  RegionExitCommit,
  RegionExitPlan,
  SSAWriteback,
  StateAccess,
  StateIfaceSig,
  StateSlotAliasPattern,
} from "./compiled-region";
export {
  aliasGroupsKey,
  aliasPatternKey,
  analyzeExternalizeNeeds,
  analyzeRegionExit,
  buildStateIfaceSig,
  computeAliasGroups,
  computeStateSlotAliasPattern,
  generateExtendedCacheKey,
  getNullStateSentinel,
  isInPlaceMutation,
  resetNullStateSentinels,
  serializeExtendedCacheKey,
  stateIfaceSigKey,
  toOutOfPlaceOp,
} from "./compiled-region";
export type {
  BaseBindingSnapshot,
  BaseDebugState,
  BaseStateInfo,
  CheckpointPack,
  DebugPlan,
  DebugSimulatedState,
  DebugSnapshot,
  EngineMemoryStats,
  FinalizeRecord,
  LocDebugState,
  LocId,
  LocRole,
  MemorySnapshot,
  MemoryStatsProvider,
  PredictedStateDelta,
  RngBasis,
  RngDrawRecord,
  RngDrawResult,
  SavedTensorInfo,
  SavedTensorRecord,
  TensorOrigin,
  TokenSnapshot,
  TraceTensor,
} from "./engine";
export {
  AsyncInCompileError,
  CheckpointImpureRegionError,
  Engine,
  EngineBusyError,
  EngineTensor,
  HostReadInCompileError,
  InvalidTraceTensorEscapeError,
  NonReentrantBackwardError,
  PoisonedEngineError,
  RngReplayExhaustedError,
  RngReplayMismatchError,
  SavedTensorModifiedError,
} from "./engine";
export type { FusionRecipe } from "./fusion";
export { buildFusionRecipes } from "./fusion";
export type { IRFusionGroup, IRGraph, IRNode } from "./ir";
export { buildIRFromTrace } from "./ir";
export type {
  CSEResult,
  DCEResult,
  OptimizeOptions,
  OptimizeResult,
  TokAfterNode,
  TokAfterResult,
} from "./ir-optimize";
export {
  analyzeTokAfterOpportunities,
  generateCSEKey,
  isCSEable,
  isEffectful,
  isOrderedLoad,
  isPureOp,
  isRandomOp,
  optimizeIR,
  performCSE,
  performDCE,
} from "./ir-optimize";
export type {
  ExecutionPlan,
  LazyIRNode,
  LazyOpCode,
  LazyRef,
  StorageHandle,
} from "./lazy";
export {
  buildPlan,
  createLazyIRNode,
  createMaterializedRef,
  createPendingRef,
  createStorageHandle,
  executePlan,
  isMaterialized,
  isPending,
  resetNodeIdCounter,
  resetStorageIdCounter,
} from "./lazy";
export type {
  EventKey,
  EventKind,
  PlanEvent,
  SemanticSubevent,
  SemanticSubeventSchedule,
} from "./planner";
export {
  buildPlanLinearOrder,
  compareEventKey,
  expandSemanticSubeventSchedule,
} from "./planner";
export { CANONICAL_NAN_BITS, canonicalizeF64Bits, encodeF64LE } from "./scalar";
export type { Token, TokenId, TokenKind } from "./tokens";
export type { TraceEvent } from "./trace";
export { TraceRecorder } from "./trace";
export type {
  BufferId,
  BufferInfo,
  DonationDecision,
  InFlightPlan,
  MemoryPlan,
  PlanId,
  SizeClass,
  TensorLifetime,
} from "./memory-planning";
export {
  analyzeLifetimes,
  BufferPool,
  canDonateBuffer,
  computeBufferSize,
  findDeadTensors,
  findDonationOpportunities,
  getSizeClass,
  getSizeForClass,
  InFlightPlanManager,
  MemoryPlanner,
} from "./memory-planning";
export type {
  CrossDeviceAnalysis,
  TransferPath,
  TransferResult,
  TransferStats,
} from "./cross-device";
export {
  analyzeCrossDeviceOps,
  createTransferStats,
  executeTransfer,
  inferOperationDevice,
  needsTransfer,
  recordTransfer,
  resolveTransferPath,
  shouldAutoTransfer,
} from "./cross-device";
export type {
  AMPCastNode,
  AMPPolicy,
  AutocastConfig,
  AutocastContext,
  CompiledRegionAMPState,
  SelectGatedResult,
} from "./amp";
export {
  captureRegionAMPState,
  computeInputCasts,
  computeOutputCast,
  computeSelectGatedDtype,
  createAutocastContext,
  DEFAULT_AMP_POLICY,
  DISABLED_AMP_POLICY,
  F16_ELIGIBLE_OPS,
  F32_REQUIRED_OPS,
  hashAMPPolicy,
  popAutocast,
  pushAutocast,
} from "./amp";
export type { OpDtypeCategory } from "./dtype-rules";
export { OP_DTYPE_RULES, promoteDtype, F16_ELIGIBLE, F32_REQUIRED } from "./dtype-rules";
export type { AMPTransformResult } from "./amp-ir-transform";
export {
  applyAMPTransform,
  getAMPStats,
  isAMPTransformed,
} from "./amp-ir-transform";
export type {
  MemoryPlanningStats,
  MemoryPlannedResult,
} from "./memory-planned-executor";
export {
  createMemoryPlanForExecution,
  executeWithMemoryPlanning,
  getMemoryPlanner,
  getMemoryPlannerStats,
  resetMemoryPlanner,
  setMemoryLimit,
  getMemoryLimit,
  MemoryLimitExceededError,
} from "./memory-planned-executor";
