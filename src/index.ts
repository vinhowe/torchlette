// Tile-IR custom kernel API + WebGPU init
export {
  type BindingSpec,
  ceilDivGrid,
  compileTileKernel,
  createTileKernelDispatcher,
  type DataType,
  elementwiseGrid,
  enableAllAllocDebug,
  getAndResetFlowCounters,
  getGPUMemoryStats,
  getLeakedAllocCount,
  getLeakedAllocCountForStep,
  getWebGPUDevice,
  getWebGPUInitError,
  initWebGPU,
  KernelContext,
  setAllocStep,
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
  Tensor as FrontendTensor,
  type TensorCreateOptions,
  Torchlette,
  torch,
} from "./frontend/torchlette";
export { storageTracker } from "./graph/storage-tracker";
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
  getTensorDebugStats,
  resetTensorDebugStats,
} from "./runtime/tensor";
