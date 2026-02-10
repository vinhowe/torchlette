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
  dispatchTiledMatmul,
  getSubgroupSupport,
  type MatmulKernelConfig,
} from "./backend/webgpu/matmul";
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
