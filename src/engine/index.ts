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
export { RuntimeEngine } from "../runtime/engine";
export {
  add as runtimeAdd,
  cpu as runtimeCpu,
  expand as runtimeExpand,
  gather as runtimeGather,
  item as runtimeItem,
  matmul as runtimeMatmul,
  mean as runtimeMean,
  mul as runtimeMul,
  relu as runtimeRelu,
  reshape as runtimeReshape,
  scatterAdd as runtimeScatterAdd,
  sqrt as runtimeSqrt,
  sub as runtimeSub,
  sum as runtimeSum,
  tensorFromArray as runtimeTensorFromArray,
  transpose as runtimeTranspose,
} from "../runtime/engine-facade";
export type { BaseId } from "../runtime/tensor";
export { Tensor as RuntimeTensor } from "../runtime/tensor";
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
export type { AMPTransformResult } from "./amp-ir-transform";
export {
  applyAMPTransform,
  getAMPStats,
  isAMPTransformed,
} from "./amp-ir-transform";
export type {
  CompiledCacheEntry,
  CompiledCacheKey,
  InputSignature,
} from "./compile-cache";
export {
  CANONICAL_NAN_BITS,
  CompiledCache,
  canonicalizeF64Bits,
  encodeF64LE,
  generateCacheKey,
  hashIRGraph,
  serializeCacheKey,
} from "./compile-cache";
export type { OpDtypeCategory } from "./dtype-rules";
export {
  F16_ELIGIBLE,
  F32_REQUIRED,
  OP_DTYPE_RULES,
  promoteDtype,
} from "./dtype-rules";
export type {
  BaseBindingSnapshot,
  BaseDebugState,
  BaseStateInfo,
  CheckpointPack,
  DebugPlan,
  DebugSimulatedState,
  DebugSnapshot,
  EngineMemoryStats,
  EventKey,
  EventKind,
  FinalizeRecord,
  LocDebugState,
  LocId,
  LocRole,
  MemorySnapshot,
  MemoryStatsProvider,
  PlanEvent,
  PredictedStateDelta,
  RngBasis,
  RngDrawRecord,
  RngDrawResult,
  SavedTensorInfo,
  SavedTensorRecord,
  SemanticSubevent,
  SemanticSubeventSchedule,
  TensorOrigin,
  TokenSnapshot,
  TraceTensor,
} from "./engine";
export {
  AsyncInCompileError,
  buildPlanLinearOrder,
  CheckpointImpureRegionError,
  compareEventKey,
  Engine,
  EngineBusyError,
  EngineTensor,
  expandSemanticSubeventSchedule,
  HostReadInCompileError,
  InvalidTraceTensorEscapeError,
  NonReentrantBackwardError,
  PoisonedEngineError,
  RngReplayExhaustedError,
  RngReplayMismatchError,
  SavedTensorModifiedError,
} from "./engine";
export type { IRFusionGroup, IRGraph, IRNode } from "./ir";
export { buildIRFromTrace } from "./ir";
export type {
  CSEResult,
  DCEResult,
  OptimizeOptions,
  OptimizeResult,
} from "./ir-optimize";
export {
  generateCSEKey,
  isCSEable,
  isEffectful,
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
export type { TensorLifetime } from "./lifetime-analysis";
export {
  analyzeLifetimes,
  computeBufferSize,
  findDeadTensorsAtStep,
  getSizeClass,
  getSizeForClass,
} from "./lifetime-analysis";
export type { Token, TokenId, TokenKind } from "./tokens";
export type { TraceEvent } from "./trace";
export { TraceRecorder } from "./trace";
