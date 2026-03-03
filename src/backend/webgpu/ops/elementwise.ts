/**
 * Elementwise ops: add, sub, mul, div, sqrt, relu, exp, log, neg, abs, tanh, sigmoid, gelu, silu, isfinite.
 */

import type {
  BackendTensor,
  DivOptions,
  GeluOptions,
  SubOptions,
} from "../../types";
import { dispatchBinary, dispatchUnary } from "../dispatch";
import type { GPUBuffer } from "../gpu-types";
import { asGPUTensor } from "../gpu-types";

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

type UnaryOp = (
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
) => BackendTensor;

type BinaryOp = (
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
) => BackendTensor;

const unary =
  (name: string): UnaryOp =>
  (a, options) =>
    dispatchUnary(name, asGPUTensor(a), options);

const binary =
  (op: string): BinaryOp =>
  (a, b, options) =>
    dispatchBinary(op, asGPUTensor(a), asGPUTensor(b), options);

// ---------------------------------------------------------------------------
// Simple binary ops
// ---------------------------------------------------------------------------

export const add: BinaryOp = binary("+");
export const mul: BinaryOp = binary("*");
export const pow: BinaryOp = binary("pow");

export function sub(
  a: BackendTensor,
  b: BackendTensor,
  options?: SubOptions & { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchBinary("-", asGPUTensor(a), asGPUTensor(b), options);
}

export function div(
  a: BackendTensor,
  b: BackendTensor,
  options?: DivOptions & { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchBinary("/", asGPUTensor(a), asGPUTensor(b), options);
}

// ---------------------------------------------------------------------------
// Simple unary ops
// ---------------------------------------------------------------------------

export const sqrt: UnaryOp = unary("sqrt");
export const relu: UnaryOp = unary("relu");
export const exp: UnaryOp = unary("exp");
export const log: UnaryOp = unary("log");
export const neg: UnaryOp = unary("neg");
export const abs: UnaryOp = unary("abs");
export const tanh: UnaryOp = unary("tanh");
export const sigmoid: UnaryOp = unary("sigmoid");
export const silu: UnaryOp = unary("silu");
export const sin: UnaryOp = unary("sin");
export const cos: UnaryOp = unary("cos");
export const rsqrt: UnaryOp = unary("rsqrt");
export const floor: UnaryOp = unary("floor");
export const ceil: UnaryOp = unary("ceil");
export const round: UnaryOp = unary("round");
export const sign: UnaryOp = unary("sign");

// ---------------------------------------------------------------------------
// Complex ops (not table-driven)
// ---------------------------------------------------------------------------

export function gelu(
  a: BackendTensor,
  options?: GeluOptions & { outBuffer?: GPUBuffer },
): BackendTensor {
  const approximate = options?.approximate ?? "tanh";

  const key = approximate === "tanh" ? "gelu_tanh" : "gelu_erf";
  return dispatchUnary(key, asGPUTensor(a), { outBuffer: options?.outBuffer });
}

export function clamp(
  a: BackendTensor,
  min: number | null,
  max: number | null,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary(`clamp_${min}_${max}`, asGPUTensor(a), options);
}

export const isfinite: UnaryOp = unary("isfinite");
