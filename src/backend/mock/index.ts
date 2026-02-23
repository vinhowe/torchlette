import {
  expand as cpuExpand,
  gather as cpuGather,
  matmul as cpuMatmul,
  mean as cpuMean,
  relu as cpuRelu,
  reshape as cpuReshape,
  scatterAdd as cpuScatterAdd,
  sqrt as cpuSqrt,
  sum as cpuSum,
  transpose as cpuTranspose,
  type Shape,
  Tensor,
} from "../cpu";
import type {
  Backend,
  DivOptions,
  GatherOptions,
  MeanOptions,
  ScatterAddOptions,
  SubOptions,
  SumOptions,
  TransposeOptions,
} from "../types";

function tensorFromArray(values: number[] | Float32Array, shape: Shape): Tensor {
  return new Tensor(shape, values instanceof Float32Array ? values.slice() : Float32Array.from(values));
}

function add(a: Tensor, _b: Tensor): Tensor {
  return new Tensor(a.shape, Float32Array.from(a.toArray()));
}

function sub(a: Tensor, _b: Tensor, _options?: SubOptions): Tensor {
  return new Tensor(a.shape, Float32Array.from(a.toArray()));
}

function div(a: Tensor, _b: Tensor, _options?: DivOptions): Tensor {
  return new Tensor(a.shape, Float32Array.from(a.toArray()));
}

function mul(a: Tensor, _b: Tensor): Tensor {
  return new Tensor(a.shape, Float32Array.from(a.toArray()));
}

function matmul(a: Tensor, b: Tensor): Tensor {
  return cpuMatmul(a, b);
}

function sqrt(a: Tensor): Tensor {
  return cpuSqrt(a);
}

function relu(a: Tensor): Tensor {
  return cpuRelu(a);
}

function expand(a: Tensor, shape: Shape): Tensor {
  return cpuExpand(a, shape);
}

function reshape(a: Tensor, shape: Shape): Tensor {
  return cpuReshape(a, shape);
}

function transpose(a: Tensor, options: TransposeOptions): Tensor {
  return cpuTranspose(a, options);
}

function gather(a: Tensor, index: Tensor, options: GatherOptions): Tensor {
  return cpuGather(a, index, options);
}

function scatterAdd(
  a: Tensor,
  index: Tensor,
  src: Tensor,
  options: ScatterAddOptions,
): Tensor {
  return cpuScatterAdd(a, index, src, options);
}

function sum(a: Tensor, options?: SumOptions): number | Tensor {
  if (options?.dim != null) {
    return cpuSum(a, options);
  }
  return cpuSum(a);
}

function mean(a: Tensor, options?: MeanOptions): number | Tensor {
  if (options?.dim != null) {
    return cpuMean(a, options);
  }
  return cpuMean(a);
}

function read(a: Tensor): Promise<number[]> {
  return Promise.resolve(a.toArray());
}

export const mockBackend: Backend = {
  name: "mock",
  ops: {
    tensorFromArray,
    add,
    sub,
    div,
    mul,
    matmul,
    sqrt,
    relu,
    expand,
    reshape,
    transpose,
    gather,
    scatterAdd,
    sum,
    mean,
    read,
  },
};
