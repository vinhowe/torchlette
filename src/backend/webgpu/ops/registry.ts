/**
 * Unified Op Registry for WebGPU Fusion
 *
 * This is the SINGLE SOURCE OF TRUTH for all fusible operations.
 * Used by:
 * - Elementwise fusion (fusion-codegen.ts)
 * - Matmul epilogue fusion (matmul/codegen.ts)
 * - Standalone elementwise kernels
 *
 * Each op defines:
 * - WGSL expression generator
 * - Arity (number of inputs)
 * - Fusion compatibility flags
 * - Vectorization support
 */

import type { DType } from "../../types";

// ============================================================================
// Types
// ============================================================================

export type OpArity = 1 | 2 | 3;

export interface OpDef {
  /**
   * WGSL expression generator.
   * For unary ops: (a) => expr
   * For binary ops: (a, b) => expr
   * For ternary ops: (a, b, c) => expr
   *
   * Some ops accept optional vector constants (zero, one) for vectorized code.
   */
  expr: (...inputs: string[]) => string;

  /**
   * Optional WGSL expression generator for vectorized mode.
   * Used when the scalar expr doesn't generalize to vectors (e.g., casts, bitcast).
   * (a, vectorWidth) => expr
   */
  vectorExpr?: (a: string, vectorWidth: number) => string;

  /** Number of inputs */
  arity: OpArity;

  /** Can this op be fused into elementwise kernels? */
  fusible: boolean;

  /** Can this op be vectorized (vec2/vec4)? */
  vectorizable: boolean;

  /**
   * Does this op need vector-compatible zero/one constants?
   * If true, expr may receive additional args: (a, zero, one)
   */
  needsVectorConstants?: boolean;

  /**
   * Output dtype behavior:
   * - 'same': output dtype matches input dtype (default)
   * - 'f32': always outputs f32 (e.g., comparisons return 0.0/1.0)
   * - 'bool': logically boolean (stored as f32 0.0/1.0)
   */
  outputDtype?: "same" | "f32" | "bool";

  /**
   * Category for documentation/organization
   */
  category: "activation" | "math" | "arithmetic" | "comparison" | "ternary" | "cast";
}

// ============================================================================
// Op Registry
// ============================================================================

export const OP_REGISTRY: Record<string, OpDef> = {
  // ---------------------------------------------------------------------------
  // Activation Functions
  // ---------------------------------------------------------------------------
  relu: {
    expr: (a: string, zero = "0.0") => `select(${zero}, ${a}, ${a} > ${zero})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    needsVectorConstants: true,
    category: "activation",
  },
  gelu: {
    // GELU with tanh approximation (GPT-2 "new GELU")
    // Clamp tanh input to [-10, 10] to avoid overflow
    // Uses needsVectorConstants for vector-compatible clamp bounds
    expr: (a: string, _zero = "0.0", one = "1.0") =>
      `(${a} * 0.5 * (1.0 + tanh(clamp(0.7978845608 * (${a} + 0.044715 * ${a} * ${a} * ${a}), -10.0 * ${one}, 10.0 * ${one}))))`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    needsVectorConstants: true,
    category: "activation",
  },
  gelu_tanh: {
    // Alias for gelu (tanh approximation)
    expr: (a: string, _zero = "0.0", one = "1.0") =>
      `(${a} * 0.5 * (1.0 + tanh(clamp(0.7978845608 * (${a} + 0.044715 * ${a} * ${a} * ${a}), -10.0 * ${one}, 10.0 * ${one}))))`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    needsVectorConstants: true,
    category: "activation",
  },
  gelu_erf: {
    // GELU with exact erf formula: x * 0.5 * (1 + erf(x / sqrt(2)))
    // Uses Abramowitz and Stegun polynomial approximation for erf
    expr: (a: string) =>
      `(${a} * 0.5 * (1.0 + sign(${a}) * (1.0 - (((((1.061405429 * (1.0 / (1.0 + 0.3275911 * abs(${a} * 0.7071067811865476))) + -1.453152027) * (1.0 / (1.0 + 0.3275911 * abs(${a} * 0.7071067811865476))) + 1.421413741) * (1.0 / (1.0 + 0.3275911 * abs(${a} * 0.7071067811865476))) + -0.284496736) * (1.0 / (1.0 + 0.3275911 * abs(${a} * 0.7071067811865476))) + 0.254829592) * (1.0 / (1.0 + 0.3275911 * abs(${a} * 0.7071067811865476))) * exp(-${a} * ${a} * 0.5)))))`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
  },
  silu: {
    // SiLU / Swish: x * sigmoid(x)
    expr: (a: string) => `(${a} / (1.0 + exp(-${a})))`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
  },
  sigmoid: {
    expr: (a: string) => `(1.0 / (1.0 + exp(-${a})))`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
  },
  tanh: {
    expr: (a: string) => `tanh(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
  },
  softplus: {
    expr: (a: string) => `log(1.0 + exp(${a}))`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
  },

  // ---------------------------------------------------------------------------
  // Math Functions
  // ---------------------------------------------------------------------------
  neg: {
    expr: (a: string) => `(-${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  abs: {
    expr: (a: string) => `abs(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  exp: {
    expr: (a: string) => `exp(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  log: {
    expr: (a: string) => `log(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  sqrt: {
    expr: (a: string) => `sqrt(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  rsqrt: {
    expr: (a: string) => `inverseSqrt(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  sin: {
    expr: (a: string) => `sin(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  cos: {
    expr: (a: string) => `cos(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  floor: {
    expr: (a: string) => `floor(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  ceil: {
    expr: (a: string) => `ceil(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  round: {
    expr: (a: string) => `round(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  sign: {
    expr: (a: string) => `sign(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
  },
  isfinite: {
    expr: (a: string) =>
      `select(0.0, 1.0, (bitcast<u32>(${a}) & 0x7F800000u) != 0x7F800000u)`,
    vectorExpr: (a: string, w: number) =>
      `select(vec${w}<f32>(0.0), vec${w}<f32>(1.0), (bitcast<vec${w}<u32>>(${a}) & vec${w}<u32>(0x7F800000u)) != vec${w}<u32>(0x7F800000u))`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    outputDtype: "f32",
    category: "math",
  },

  // ---------------------------------------------------------------------------
  // Binary Arithmetic
  // ---------------------------------------------------------------------------
  add: {
    expr: (a: string, b: string) => `(${a} + ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
  },
  sub: {
    expr: (a: string, b: string) => `(${a} - ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
  },
  mul: {
    expr: (a: string, b: string) => `(${a} * ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
  },
  div: {
    expr: (a: string, b: string) => `(${a} / ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
  },
  pow: {
    expr: (a: string, b: string) => `pow(${a}, ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
  },
  min: {
    expr: (a: string, b: string) => `min(${a}, ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
  },
  max: {
    expr: (a: string, b: string) => `max(${a}, ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
  },
  mod: {
    expr: (a: string, b: string) => `(${a} % ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: false, // WGSL % doesn't work on vectors the same way
    category: "arithmetic",
  },

  // ---------------------------------------------------------------------------
  // Comparisons (return f32 0.0 or 1.0 for compatibility)
  // ---------------------------------------------------------------------------
  eq: {
    expr: (a: string, b: string) => `select(0.0, 1.0, ${a} == ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    outputDtype: "f32",
    category: "comparison",
  },
  ne: {
    expr: (a: string, b: string) => `select(0.0, 1.0, ${a} != ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    outputDtype: "f32",
    category: "comparison",
  },
  lt: {
    expr: (a: string, b: string) => `select(0.0, 1.0, ${a} < ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    outputDtype: "f32",
    category: "comparison",
  },
  le: {
    expr: (a: string, b: string) => `select(0.0, 1.0, ${a} <= ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    outputDtype: "f32",
    category: "comparison",
  },
  gt: {
    expr: (a: string, b: string) => `select(0.0, 1.0, ${a} > ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    outputDtype: "f32",
    category: "comparison",
  },
  ge: {
    expr: (a: string, b: string) => `select(0.0, 1.0, ${a} >= ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    outputDtype: "f32",
    category: "comparison",
  },

  // ---------------------------------------------------------------------------
  // Ternary Operations
  // ---------------------------------------------------------------------------
  where: {
    // where(cond, a, b) -> select(b, a, cond > 0)
    // Note: cond is treated as boolean (> 0 means true)
    expr: (cond: string, a: string, b: string) => `select(${b}, ${a}, ${cond} > 0.0)`,
    arity: 3,
    fusible: true,
    vectorizable: true,
    category: "ternary",
  },

  // ---------------------------------------------------------------------------
  // Type Casts
  // ---------------------------------------------------------------------------
  cast_f16: {
    expr: (a: string) => `f16(${a})`,
    vectorExpr: (a: string, w: number) => `vec${w}<f16>(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "cast",
  },
  cast_f32: {
    expr: (a: string) => `f32(${a})`,
    vectorExpr: (a: string, w: number) => `vec${w}<f32>(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "cast",
  },
  cast_i32: {
    expr: (a: string) => `i32(${a})`,
    vectorExpr: (a: string, w: number) => `vec${w}<i32>(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "cast",
  },
  cast_u32: {
    expr: (a: string) => `u32(${a})`,
    vectorExpr: (a: string, w: number) => `vec${w}<u32>(${a})`,
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "cast",
  },
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get the op definition, or null if not found.
 */
export function getOpDef(op: string): OpDef | null {
  return OP_REGISTRY[op] ?? null;
}

/**
 * Check if an op exists in the registry.
 */
export function hasOp(op: string): boolean {
  return op in OP_REGISTRY;
}

/**
 * Get WGSL expression for an op.
 *
 * @param op - Operation name
 * @param inputs - Input expressions
 * @param vectorConstants - Optional vector-compatible zero/one for vectorized code
 * @throws Error if op is not found
 */
export function getExpr(
  op: string,
  inputs: string[],
  vectorConstants?: { zero: string; one: string },
): string {
  const def = OP_REGISTRY[op];
  if (!def) {
    throw new Error(`Unknown op in registry: ${op}`);
  }

  // Validate arity
  if (inputs.length < def.arity) {
    throw new Error(`Op ${op} requires ${def.arity} inputs, got ${inputs.length}`);
  }

  // Handle ops that need vector constants
  if (def.needsVectorConstants && vectorConstants) {
    return def.expr(inputs[0], vectorConstants.zero, vectorConstants.one);
  }

  // Standard expression generation
  return def.expr(...inputs.slice(0, def.arity));
}

/**
 * Check if an op can be fused into elementwise kernels.
 */
export function isFusible(op: string): boolean {
  return OP_REGISTRY[op]?.fusible ?? false;
}

/**
 * Check if an op can be vectorized.
 */
export function canVectorize(op: string): boolean {
  return OP_REGISTRY[op]?.vectorizable ?? false;
}

/**
 * Get the vectorized expression generator for an op, if it has one.
 */
export function getVectorExpr(op: string): ((a: string, w: number) => string) | undefined {
  return OP_REGISTRY[op]?.vectorExpr;
}

/**
 * Get op arity (number of inputs).
 */
export function getArity(op: string): OpArity | null {
  return OP_REGISTRY[op]?.arity ?? null;
}

/**
 * Check if an op is a unary operation.
 */
export function isUnaryOp(op: string): boolean {
  return OP_REGISTRY[op]?.arity === 1;
}

/**
 * Check if an op is a binary operation.
 */
export function isBinaryOp(op: string): boolean {
  return OP_REGISTRY[op]?.arity === 2;
}

/**
 * Check if an op is a ternary operation.
 */
export function isTernaryOp(op: string): boolean {
  return OP_REGISTRY[op]?.arity === 3;
}

/**
 * Get all ops in a category.
 */
export function getOpsByCategory(category: OpDef["category"]): string[] {
  return Object.entries(OP_REGISTRY)
    .filter(([_, def]) => def.category === category)
    .map(([name, _]) => name);
}

/**
 * Get all fusible ops.
 */
export function getAllFusibleOps(): string[] {
  return Object.entries(OP_REGISTRY)
    .filter(([_, def]) => def.fusible)
    .map(([name, _]) => name);
}

/**
 * Get all unary ops.
 */
export function getAllUnaryOps(): string[] {
  return Object.entries(OP_REGISTRY)
    .filter(([_, def]) => def.arity === 1)
    .map(([name, _]) => name);
}

/**
 * Get all binary ops.
 */
export function getAllBinaryOps(): string[] {
  return Object.entries(OP_REGISTRY)
    .filter(([_, def]) => def.arity === 2)
    .map(([name, _]) => name);
}

// ============================================================================
// Backward Compatibility Exports
// ============================================================================

/**
 * UNARY_EXPR - for backward compatibility with fusion-codegen.ts
 * @deprecated Use getExpr() or OP_REGISTRY directly
 */
export const UNARY_EXPR: Record<string, (input: string, zero?: string, one?: string) => string> =
  Object.fromEntries(
    Object.entries(OP_REGISTRY)
      .filter(([_, def]) => def.arity === 1)
      .map(([name, def]) => [name, def.expr as (input: string, zero?: string, one?: string) => string]),
  );

/**
 * BINARY_EXPR - for backward compatibility with fusion-codegen.ts
 * @deprecated Use getExpr() or OP_REGISTRY directly
 */
export const BINARY_EXPR: Record<string, (a: string, b: string) => string> = Object.fromEntries(
  Object.entries(OP_REGISTRY)
    .filter(([_, def]) => def.arity === 2)
    .map(([name, def]) => [name, def.expr as (a: string, b: string) => string]),
);
