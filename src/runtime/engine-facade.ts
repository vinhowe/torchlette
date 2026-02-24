import type {
  ArgReduceOptions,
  DeviceKind,
  DivOptions,
  DType,
  GatherOptions,
  GeluOptions,
  MaxOptions,
  MeanOptions,
  ScatterAddOptions,
  SubOptions,
  SumOptions,
  TransposeOptions,
} from "../backend/types";
import type { Tensor } from "./tensor";
import { RuntimeEngine } from "./engine";
import type { TensorOrScalar } from "./engine-types";

const defaultEngine = new RuntimeEngine();

export { defaultEngine };

export function tensorFromArray(
  values: number[],
  shape: number[],
  device?: DeviceKind,
): Tensor {
  return defaultEngine.tensorFromArray(values, shape, device);
}

export function add(a: TensorOrScalar, b: TensorOrScalar): Tensor {
  return defaultEngine.add(a, b);
}

export function sub(a: TensorOrScalar, b: TensorOrScalar, options?: SubOptions): Tensor {
  return defaultEngine.sub(a, b, options);
}

export function div(a: TensorOrScalar, b: TensorOrScalar, options?: DivOptions): Tensor {
  return defaultEngine.div(a, b, options);
}

export function mul(a: TensorOrScalar, b: TensorOrScalar): Tensor {
  return defaultEngine.mul(a, b);
}

export function view(a: Tensor, shape: number[]): Tensor {
  return defaultEngine.view(a, shape);
}

export function reshape(a: Tensor, shape: number[]): Tensor {
  return defaultEngine.reshape(a, shape);
}

export function matmul(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.matmul(a, b);
}

export function sqrt(a: Tensor): Tensor {
  return defaultEngine.sqrt(a);
}

export function relu(a: Tensor): Tensor {
  return defaultEngine.relu(a);
}

export function exp(a: Tensor): Tensor {
  return defaultEngine.exp(a);
}

export function log(a: Tensor): Tensor {
  return defaultEngine.log(a);
}

export function neg(a: Tensor): Tensor {
  return defaultEngine.neg(a);
}

export function abs(a: Tensor): Tensor {
  return defaultEngine.abs(a);
}

export function tanh(a: Tensor): Tensor {
  return defaultEngine.tanh(a);
}

export function sigmoid(a: Tensor): Tensor {
  return defaultEngine.sigmoid(a);
}

export function gelu(a: Tensor, options?: GeluOptions): Tensor {
  return defaultEngine.gelu(a, options);
}

export function silu(a: Tensor): Tensor {
  return defaultEngine.silu(a);
}

export function isfinite(a: Tensor): Tensor {
  return defaultEngine.isfinite(a);
}

export function expand(a: Tensor, shape: number[]): Tensor {
  return defaultEngine.expand(a, shape);
}

export function transpose(a: Tensor, options: TransposeOptions): Tensor {
  return defaultEngine.transpose(a, options);
}

export function permute(a: Tensor, dims: number[]): Tensor {
  return defaultEngine.permute(a, dims);
}

export function contiguous(a: Tensor): Tensor {
  return defaultEngine.contiguous(a);
}

export function cast(a: Tensor, dtype: DType): Tensor {
  return defaultEngine.cast(a, dtype);
}

export function gather(
  a: Tensor,
  index: Tensor,
  options: GatherOptions,
): Tensor {
  return defaultEngine.gather(a, index, options);
}

export function scatterAdd(
  a: Tensor,
  index: Tensor,
  src: Tensor,
  options: ScatterAddOptions,
): Tensor {
  return defaultEngine.scatterAdd(a, index, src, options);
}

export function sum(a: Tensor, options?: SumOptions): number | Tensor {
  return defaultEngine.sum(a, options);
}

export function max(a: Tensor, options?: MaxOptions): number | Tensor {
  return defaultEngine.max(a, options);
}

export function mean(a: Tensor, options?: MeanOptions): number | Tensor {
  return defaultEngine.mean(a, options);
}

export function argmax(a: Tensor, options: ArgReduceOptions): Tensor {
  return defaultEngine.argmax(a, options);
}

export function argmin(a: Tensor, options: ArgReduceOptions): Tensor {
  return defaultEngine.argmin(a, options);
}

export function gt(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.gt(a, b);
}

export function lt(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.lt(a, b);
}

export function ge(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.ge(a, b);
}

export function le(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.le(a, b);
}

export function eq(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.eq(a, b);
}

export function ne(a: Tensor, b: Tensor): Tensor {
  return defaultEngine.ne(a, b);
}

export function where(condition: Tensor, x: Tensor, y: Tensor): Tensor {
  return defaultEngine.where(condition, x, y);
}

export async function cpu(a: Tensor): Promise<number[]> {
  return defaultEngine.cpu(a);
}

export async function item(a: Tensor): Promise<number> {
  return defaultEngine.item(a);
}

// In-place operations

export function copy_(dst: Tensor, src: Tensor): Tensor {
  return defaultEngine.copy_(dst, src);
}

export function add_(dst: Tensor, src: Tensor): Tensor {
  return defaultEngine.add_(dst, src);
}

export function zero_(dst: Tensor): Tensor {
  return defaultEngine.zero_(dst);
}

export function fill_(dst: Tensor, value: number): Tensor {
  return defaultEngine.fill_(dst, value);
}

export function mul_(dst: Tensor, value: number): Tensor {
  return defaultEngine.mul_(dst, value);
}
