// Tile-IR custom kernel API + WebGPU init
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
  getGpuUncapturedErrorCount,
  getGPUMemoryStats,
  getLeakedAllocCount,
  getLeakedAllocCountForStep,
  getProfileJSON,
  getWebGPUDevice,
  getWebGPUInitError,
  initGpuTimestamps,
  initWebGPU,
  type InitWebGPUOptions,
  webgpuDeviceRequirements,
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
} from "./backend/webgpu";
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

// Step-tape observability (§6): guard-miss/hit counters for apps.
export { stReplayStats } from "./executor/step-tape-replay";
export { stStats } from "./core/step-tape";
export { STEP_TAPE_RECORD, STEP_TAPE_REPLAY } from "./core/step-tape";
