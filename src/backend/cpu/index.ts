import type { Backend, BackendTensor, DType } from "../types";
import * as numeric from "./numeric";

export * from "./numeric";

function read(tensor: BackendTensor): Promise<number[]> {
  return Promise.resolve((tensor as numeric.Tensor).toArray());
}

/**
 * Cast tensor to a different dtype.
 * CPU backend operates on f64 numbers internally, so this is a no-op.
 */
function cast(a: BackendTensor, _dtype: DType): BackendTensor {
  return a;
}

export const cpuBackend: Backend = {
  name: "cpu",
  ops: {
    tensorFromArray: numeric.tensorFromArray,
    zeros: numeric.zeros,
    full: numeric.full,
    arange: numeric.arange,
    add: numeric.add,
    sub: numeric.sub,
    div: numeric.div,
    mul: numeric.mul,
    matmul: numeric.matmul,
    sqrt: numeric.sqrt,
    relu: numeric.relu,
    exp: numeric.exp,
    log: numeric.log,
    neg: numeric.neg,
    abs: numeric.abs,
    tanh: numeric.tanh,
    sigmoid: numeric.sigmoid,
    gelu: numeric.gelu,
    silu: numeric.silu,
    sin: numeric.sin,
    cos: numeric.cos,
    rsqrt: numeric.rsqrt,
    floor: numeric.floor,
    ceil: numeric.ceil,
    round: numeric.round,
    sign: numeric.sign,
    clamp: numeric.clamp,
    pow: numeric.pow,
    isfinite: numeric.isfinite,
    conv2d: numeric.conv2d,
    gather: numeric.gather,
    scatterAdd: numeric.scatterAdd,
    cat: numeric.cat,
    sum: numeric.sum,
    max: numeric.max,
    mean: numeric.mean,
    min: numeric.min,
    argmax: numeric.argmax,
    argmin: numeric.argmin,
    gt: numeric.gt,
    lt: numeric.lt,
    ge: numeric.ge,
    le: numeric.le,
    eq: numeric.eq,
    ne: numeric.ne,
    expand: numeric.expand,
    reshape: numeric.reshape,
    transpose: numeric.transpose,
    permute: numeric.permute,
    narrow: numeric.narrow,
    narrowBackward: numeric.narrowBackward,
    contiguous: numeric.contiguous,
    cast,
    where: numeric.where,
    tril: numeric.tril,
    triu: numeric.triu,
    stridedScatterCopy: numeric.stridedScatterCopy,
    stridedScatterAdd: numeric.stridedScatterAdd,
    read,
  },
};
