/**
 * Elementwise ops: add, sub, mul, div, sqrt, relu, exp, log, neg, abs, tanh, sigmoid, gelu, silu, isfinite.
 * Extracted from index.ts — purely structural refactoring.
 */

import type {
  BackendTensor,
  DivOptions,
  GeluOptions,
  SubOptions,
} from "../../types";
import type { GPUBuffer, WebGPUTensor } from "../gpu-types";
import { dispatchBinary, dispatchUnary } from "../dispatch";

export function add(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchBinary("+", a as WebGPUTensor, b as WebGPUTensor, options);
}

export function sub(
  a: BackendTensor,
  b: BackendTensor,
  options?: SubOptions & { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchBinary("-", a as WebGPUTensor, b as WebGPUTensor, options);
}

export function div(
  a: BackendTensor,
  b: BackendTensor,
  options?: DivOptions & { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchBinary("/", a as WebGPUTensor, b as WebGPUTensor, options);
}

export function mul(
  a: BackendTensor,
  b: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchBinary("*", a as WebGPUTensor, b as WebGPUTensor, options);
}

export function sqrt(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("sqrt", "sqrt(x)", a as WebGPUTensor, options);
}

export function relu(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary(
    "relu",
    "select(0.0, x, x > 0.0)",
    a as WebGPUTensor,
    options,
  );
}

export function exp(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("exp", "exp(x)", a as WebGPUTensor, options);
}

export function log(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("log", "log(x)", a as WebGPUTensor, options);
}

export function neg(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("neg", "-x", a as WebGPUTensor, options);
}

export function abs(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("abs", "abs(x)", a as WebGPUTensor, options);
}

export function tanh(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary("tanh", "tanh(x)", a as WebGPUTensor, options);
}

export function sigmoid(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  return dispatchUnary(
    "sigmoid",
    "(1.0 / (1.0 + exp(-x)))",
    a as WebGPUTensor,
    options,
  );
}

export function gelu(
  a: BackendTensor,
  options?: GeluOptions & { outBuffer?: GPUBuffer },
): BackendTensor {
  const approximate = options?.approximate ?? "tanh";

  if (approximate === "tanh") {
    // Tanh approximation (GPT-2 "new GELU"): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Clamp tanh input to [-10, 10] to avoid overflow (tanh saturates to ±1 beyond this)
    return dispatchUnary(
      "gelu_tanh",
      "(x * 0.5 * (1.0 + tanh(clamp(0.7978845608 * (x + 0.044715 * x * x * x), -10.0, 10.0))))",
      a as WebGPUTensor,
      { outBuffer: options?.outBuffer },
    );
  } else {
    // Exact formula using erf: x * 0.5 * (1 + erf(x / sqrt(2)))
    // WGSL doesn't have erf, so we use a polynomial approximation
    // Abramowitz and Stegun approximation 7.1.26 (max error ~1.5e-7)
    // Single expression with all computations inlined
    return dispatchUnary(
      "gelu_erf",
      "(x * 0.5 * (1.0 + sign(x) * (1.0 - (((((1.061405429 * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + -1.453152027) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + 1.421413741) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + -0.284496736) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) + 0.254829592) * (1.0 / (1.0 + 0.3275911 * abs(x * 0.7071067811865476))) * exp(-x * x * 0.5)))))",
      a as WebGPUTensor,
      { outBuffer: options?.outBuffer },
    );
  }
}

export function silu(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  // SiLU/Swish: x * sigmoid(x) = x / (1 + exp(-x))
  return dispatchUnary(
    "silu",
    "(x / (1.0 + exp(-x)))",
    a as WebGPUTensor,
    options,
  );
}

/**
 * Check if values are finite (not NaN and not Inf).
 * Returns 1.0 where finite, 0.0 where NaN or Inf.
 * Uses arithmetic checks since not all WGSL implementations support isinf/isnan.
 */
export function isfinite(
  a: BackendTensor,
  options?: { outBuffer?: GPUBuffer },
): BackendTensor {
  // Use bitcast to check IEEE 754 exponent bits directly.
  // A f32 is Inf or NaN when all exponent bits (bits 23-30) are set.
  // Exponent mask: 0x7F800000. If (bits & mask) == mask, value is non-finite.
  // This is robust against GPU compiler optimizations that may fold
  // arithmetic checks like x * 0.0 == 0.0 or x - x == 0.0.
  return dispatchUnary(
    "isfinite",
    "select(0.0, 1.0, (bitcast<u32>(x) & 0x7F800000u) != 0x7F800000u)",
    a as WebGPUTensor,
    options,
  );
}
