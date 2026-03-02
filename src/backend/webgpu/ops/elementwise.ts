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
  (name: string, expr: string): UnaryOp =>
  (a, options) =>
    dispatchUnary(name, expr, asGPUTensor(a), options);

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

export const sqrt: UnaryOp = unary("sqrt", "sqrt(x)");
export const relu: UnaryOp = unary("relu", "select(0.0, x, x > 0.0)");
export const exp: UnaryOp = unary("exp", "exp(x)");
export const log: UnaryOp = unary("log", "log(x)");
export const neg: UnaryOp = unary("neg", "-x");
export const abs: UnaryOp = unary("abs", "abs(x)");
export const tanh: UnaryOp = unary("tanh", "tanh(x)");
export const sigmoid: UnaryOp = unary("sigmoid", "(1.0 / (1.0 + exp(-x)))");
export const silu: UnaryOp = unary("silu", "(x / (1.0 + exp(-x)))");
export const sin: UnaryOp = unary("sin", "sin(x)");
export const cos: UnaryOp = unary("cos", "cos(x)");
export const rsqrt: UnaryOp = unary("rsqrt", "inverseSqrt(x)");
export const floor: UnaryOp = unary("floor", "floor(x)");
export const ceil: UnaryOp = unary("ceil", "ceil(x)");
export const round: UnaryOp = unary("round", "round(x)");
export const sign: UnaryOp = unary("sign", "sign(x)");

// ---------------------------------------------------------------------------
// Complex ops (not table-driven)
// ---------------------------------------------------------------------------

export function gelu(
  a: BackendTensor,
  options?: GeluOptions & { outBuffer?: GPUBuffer },
): BackendTensor {
  const approximate = options?.approximate ?? "tanh";

  if (approximate === "tanh") {
    return dispatchUnary(
      "gelu_tanh",
      "(x * 0.5 * (1.0 + tanh(clamp(0.7978845608 * (x + 0.044715 * x * x * x), -10.0, 10.0))))",
      asGPUTensor(a),
      { outBuffer: options?.outBuffer },
    );
  } else {
    return dispatchUnary(
      "gelu_erf",
      "(x * 0.5 * (1.0 + sign(x) * (1.0 - (((((1.061405429 * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + -1.453152027) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + 1.421413741) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + -0.284496736) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + 0.254829592) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) * exp(-x * x * 0.5)))))",
      asGPUTensor(a),
      { outBuffer: options?.outBuffer },
    );
  }
}

export function clamp(
  a: BackendTensor,
  min: number | null,
  max: number | null,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  let expr: string;
  if (min !== null && max !== null) {
    expr = `clamp(x, ${min}, ${max})`;
  } else if (min !== null) {
    expr = `max(x, ${min})`;
  } else if (max !== null) {
    expr = `min(x, ${max})`;
  } else {
    expr = "x";
  }
  return dispatchUnary(`clamp_${min}_${max}`, expr, asGPUTensor(a), options);
}

/**
 * Check if values are finite (not NaN and not Inf).
 * Uses bitcast to check IEEE 754 exponent bits directly.
 */
export const isfinite: UnaryOp = unary(
  "isfinite",
  "select(0.0, 1.0, (bitcast<u32>(x) & 0x7F800000u) != 0x7F800000u)",
);
