// Tile-IR custom kernel API
export {
  type BindingSpec,
  ceilDivGrid,
  compileTileKernel,
  createTileKernelDispatcher,
  type DataType,
  elementwiseGrid,
  KernelContext,
  singleWorkgroup,
  type TileKernelInstance,
  type TileKernelSpec,
  type UniformType,
} from "./backend/webgpu";
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
