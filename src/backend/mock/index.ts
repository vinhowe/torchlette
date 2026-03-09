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
  MaxOptions,
  MeanOptions,
  OpExecOptions,
  ScatterAddOptions,
  StridedScatterOptions,
  SubOptions,
  SumOptions,
  TransposeOptions,
} from "../types";

function tensorFromArray(
  values: number[] | Float32Array,
  shape: Shape,
): Tensor {
  return new Tensor(
    shape,
    values instanceof Float32Array ? values.slice() : Float32Array.from(values),
  );
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

function sum(a: Tensor, options?: SumOptions): Tensor {
  return cpuSum(a, options);
}

function mean(a: Tensor, options?: MeanOptions): Tensor {
  return cpuMean(a, options);
}

function read(a: Tensor): Promise<number[]> {
  return Promise.resolve(a.toArray());
}

function permute(a: Tensor, _dims: number[]): Tensor {
  return new Tensor(a.shape, Float32Array.from(a.toArray()));
}

function contiguous(a: Tensor): Tensor {
  return new Tensor(a.shape, Float32Array.from(a.toArray()));
}

function max(a: Tensor, _options?: MaxOptions): Tensor {
  const arr = a.toArray();
  const maxVal = Math.max(...arr);
  return new Tensor([], Float32Array.of(maxVal));
}

function where(
  condition: Tensor,
  x: Tensor,
  y: Tensor,
  _options?: OpExecOptions,
): Tensor {
  const cArr = condition.toArray();
  const xArr = x.toArray();
  const yArr = y.toArray();
  const result = new Float32Array(cArr.length);
  for (let i = 0; i < cArr.length; i++) {
    result[i] = cArr[i] !== 0 ? xArr[i] : yArr[i];
  }
  return new Tensor(condition.shape, result);
}

function stridedScatterCopy(
  base: Tensor,
  _src: Tensor,
  _options: StridedScatterOptions,
): Tensor {
  return new Tensor(base.shape, Float32Array.from(base.toArray()));
}

function stridedScatterAdd(
  base: Tensor,
  _src: Tensor,
  _options: StridedScatterOptions,
): Tensor {
  return new Tensor(base.shape, Float32Array.from(base.toArray()));
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
    permute,
    contiguous,
    max,
    where,
    stridedScatterCopy,
    stridedScatterAdd,
    gather,
    scatterAdd,
    sum,
    mean,
    read,
  },
};
