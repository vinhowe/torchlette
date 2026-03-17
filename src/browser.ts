// WebGPU backend exports for browser benchmarking
// Tile-IR custom kernel API
export {
  type BindingSpec,
  compileTileKernel,
  createTileKernelDispatcher,
  type DataType,
  elementwiseGrid,
  getWebGPUDevice,
  getWebGPUInitError,
  initWebGPU,
  KernelContext,
  syncWebGPU,
  type TileKernelInstance,
  type TileKernelSpec,
  type UniformType,
  webgpuBackend,
} from "./backend/webgpu";
export { dispatchTiledMatmul } from "./backend/webgpu/matmul/dispatch";
export {
  DEFAULT_CONFIG,
  getSubgroupSupport,
  type MatmulKernelConfig,
} from "./backend/webgpu/matmul/types";
export {
  type DeviceKind,
  DisposedTensorError,
  Tensor as FrontendTensor,
  type TensorCreateOptions,
  Torchlette,
  torch,
} from "./frontend/torchlette";
export * as nn from "./nn";
export {
  Adam,
  type AdamOptions,
  GradScaler,
  type GradScalerOptions,
  SGD,
  type SGDOptions,
} from "./optim";
