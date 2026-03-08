/**
 * Unified Op Registry for WebGPU Fusion
 *
 * This is the SINGLE SOURCE OF TRUTH for all fusible operations.
 * Used by:
 * - Elementwise fusion (fusion-tile-ir.ts)
 * - Matmul epilogue fusion (matmul/tile-matmul.ts)
 * - Standalone elementwise kernels
 *
 * Each op defines:
 * - WGSL expression generator
 * - Arity (number of inputs)
 * - Fusion compatibility flags
 * - Vectorization support
 */

// ============================================================================
// Types
// ============================================================================

type OpArity = 1 | 2 | 3;

interface OpDef {
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
  category:
    | "activation"
    | "math"
    | "arithmetic"
    | "comparison"
    | "ternary"
    | "cast"
    | "bitwise";

  /** WGSL infix operator for tile-compiler binary ops (e.g., "+", "&"). */
  wgslInfix?: string;
  /** WGSL prefix operator for tile-compiler unary ops (e.g., "-" for neg). */
  wgslPrefix?: string;
  /** WGSL function name when it differs from op name (e.g., rsqrt → inverseSqrt). */
  wgslFnName?: string;
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
    wgslPrefix: "-",
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
    wgslFnName: "inverseSqrt",
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
    wgslInfix: "+",
  },
  sub: {
    expr: (a: string, b: string) => `(${a} - ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    wgslInfix: "-",
  },
  mul: {
    expr: (a: string, b: string) => `(${a} * ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    wgslInfix: "*",
  },
  div: {
    expr: (a: string, b: string) => `(${a} / ${b})`,
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    wgslInfix: "/",
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
    vectorizable: false,
    category: "arithmetic",
    wgslInfix: "%",
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
    expr: (cond: string, a: string, b: string) =>
      `select(${b}, ${a}, ${cond} > 0.0)`,
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

// gelu_tanh is an alias for gelu (both use tanh approximation)
OP_REGISTRY.gelu_tanh = OP_REGISTRY.gelu;

// ---------------------------------------------------------------------------
// Bitwise / Shift Ops (used by tile-compiler, not fusible in elementwise)
// ---------------------------------------------------------------------------
const BITWISE_OPS: Record<
  string,
  Pick<OpDef, "expr" | "wgslInfix" | "wgslPrefix">
> = {
  and: { expr: (a: string, b: string) => `(${a} & ${b})`, wgslInfix: "&" },
  or: { expr: (a: string, b: string) => `(${a} | ${b})`, wgslInfix: "|" },
  xor: { expr: (a: string, b: string) => `(${a} ^ ${b})`, wgslInfix: "^" },
  shr: { expr: (a: string, b: string) => `(${a} >> ${b})`, wgslInfix: ">>" },
  shl: { expr: (a: string, b: string) => `(${a} << ${b})`, wgslInfix: "<<" },
};
for (const [name, def] of Object.entries(BITWISE_OPS)) {
  OP_REGISTRY[name] = {
    ...def,
    arity: 2,
    fusible: false,
    vectorizable: false,
    category: "bitwise",
  } as OpDef;
}
OP_REGISTRY["not"] = {
  expr: (a: string) => `(!${a})`,
  arity: 1,
  fusible: false,
  vectorizable: false,
  category: "bitwise",
  wgslPrefix: "!",
};

// ============================================================================
// Helper Functions (only those used in production)
// ============================================================================

/** Check if an op can be vectorized. */
export function canVectorize(op: string): boolean {
  return OP_REGISTRY[op]?.vectorizable ?? false;
}

/** Check if an op is a unary operation. */
export function isUnaryOp(op: string): boolean {
  return OP_REGISTRY[op]?.arity === 1;
}

/** Get WGSL infix operator for a binary op (e.g., "add" → "+"). */
export function getWgslInfix(op: string): string | undefined {
  return OP_REGISTRY[op]?.wgslInfix;
}

/** Get WGSL prefix operator for a unary op (e.g., "neg" → "-"). */
export function getWgslPrefix(op: string): string | undefined {
  return OP_REGISTRY[op]?.wgslPrefix;
}

/** Get WGSL function name, falling back to op name (e.g., "rsqrt" → "inverseSqrt"). */
export function getWgslFnName(op: string): string {
  return OP_REGISTRY[op]?.wgslFnName ?? op;
}
