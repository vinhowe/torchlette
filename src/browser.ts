// WebGPU backend exports for browser benchmarking
export {
  dispatchMatmulWithEpilogue,
  getWebGPUDevice,
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
  webgpuBackend,
} from "./backend/webgpu";
export {
  DEFAULT_CONFIG,
  getSubgroupSupport,
  type MatmulKernelConfig,
} from "./backend/webgpu/matmul/types";
export { dispatchTiledMatmul } from "./backend/webgpu/matmul/dispatch";
export * from "./engine";
export {
  type DeviceKind,
  DisposedTensorError,
  Tensor as FrontendTensor,
  type TensorCreateOptions,
  Torchlette,
  torch,
} from "./frontend";
export { Adam, type AdamOptions, SGD, type SGDOptions } from "./optim";

// Tile-IR custom kernel API
export {
  type TileKernelSpec,
  type BindingSpec,
  type DataType,
  type UniformType,
  KernelContext,
  elementwiseGrid,
  compileTileKernel,
  createTileKernelDispatcher,
  type TileKernelInstance,
} from "./backend/webgpu";
