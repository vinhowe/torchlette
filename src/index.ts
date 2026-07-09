// Tile-IR custom kernel API + WebGPU init

// Attention score/mask modifiers (#64 — declarations, as data)
export type {
  AttnMaskModSpec,
  AttnModifierSpec,
  AttnScoreModSpec,
} from "./backend/types";
export {
  type BindingSpec,
  ceilDivGrid,
  compileTileKernel,
  createTileKernelDispatcher,
  type DataType,
  disableProfiling,
  elementwiseGrid,
  enableAllAllocDebug,
  enableProfiling,
  getAndResetFlowCounters,
  getGPUMemoryStats,
  getGpuUncapturedErrorCount,
  getLeakedAllocCount,
  getLeakedAllocCountForStep,
  getProfileJSON,
  getWebGPUDevice,
  getWebGPUInitError,
  type InitWebGPUOptions,
  initGpuTimestamps,
  initWebGPU,
  KernelContext,
  readGpuTimestamps,
  resetProfileStats,
  setAllocStep,
  setGPUMemoryLimit,
  setProfilePhase,
  singleWorkgroup,
  snapshotLeakedAllocs,
  syncWebGPU,
  type TileKernelInstance,
  type TileKernelSpec,
  type UniformType,
  webgpuBackend,
  webgpuDeviceRequirements,
} from "./backend/webgpu";
export { attnModifierKey } from "./backend/webgpu/attention-kernel";
export { STEP_TAPE_RECORD, STEP_TAPE_REPLAY, stStats } from "./core/step-tape";
// Step-tape observability (§6): guard-miss/hit counters for apps.
export { stReplayStats } from "./executor/step-tape-replay";
export {
  type DeviceKind,
  DisposedTensorError,
  type ScopeHandle,
  Tensor as FrontendTensor,
  type TensorCreateOptions,
  Torchlette,
  torch,
} from "./frontend/torchlette";
export { storageTracker } from "./graph/storage-tracker";
// Runtime namespace (works at runtime). NOTE: the dts bundler does not emit a
// usable type for this re-export, so for TYPED access consumers should import
// from the "torchlette/nn" subpath (named exports, fully typed).
export * as nn from "./nn";
export {
  Adam,
  type AdamOptions,
  CosineAnnealingLR,
  ExponentialLR,
  GradScaler,
  type GradScalerOptions,
  type HasLR,
  type LRScheduler,
  PolynomialLR,
  SGD,
  type SGDOptions,
  StepLR,
} from "./optim";
export {
  getDebugLiveTensors,
  getTensorDebugStats,
  resetTensorDebugStats,
  setDebugTracking,
} from "./runtime/tensor";
