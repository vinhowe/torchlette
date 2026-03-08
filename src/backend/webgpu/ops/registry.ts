/**
 * Unified Op Registry for WebGPU Fusion
 *
 * Single source of truth for op metadata used by:
 * - Fusion detection (fusion-detect.ts): fusible flag
 * - Op dispatch (op-dispatch.ts): arity, category
 * - Tile compiler (tile-compiler.ts): wgslInfix, wgslPrefix, wgslFnName
 * - Vectorization (fusion-types.ts): vectorizable flag
 *
 * Op behavior (WGSL codegen) lives in fusion-tile-ir.ts via BlockExpr methods.
 */

// ============================================================================
// Types
// ============================================================================

type OpArity = 1 | 2 | 3;

interface OpDef {
  /** Number of inputs */
  arity: OpArity;

  /** Can this op be fused into elementwise kernels? */
  fusible: boolean;

  /** Can this op be vectorized (vec2/vec4)? */
  vectorizable: boolean;

  /** Category for organization and filtering */
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
  relu: { arity: 1, fusible: true, vectorizable: true, category: "activation" },
  gelu: { arity: 1, fusible: true, vectorizable: true, category: "activation" },
  gelu_erf: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
  },
  silu: { arity: 1, fusible: true, vectorizable: true, category: "activation" },
  sigmoid: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
  },
  tanh: { arity: 1, fusible: true, vectorizable: true, category: "activation" },
  softplus: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
  },

  // ---------------------------------------------------------------------------
  // Math Functions
  // ---------------------------------------------------------------------------
  neg: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    wgslPrefix: "-",
  },
  abs: { arity: 1, fusible: true, vectorizable: true, category: "math" },
  exp: { arity: 1, fusible: true, vectorizable: true, category: "math" },
  log: { arity: 1, fusible: true, vectorizable: true, category: "math" },
  sqrt: { arity: 1, fusible: true, vectorizable: true, category: "math" },
  rsqrt: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    wgslFnName: "inverseSqrt",
  },
  sin: { arity: 1, fusible: true, vectorizable: true, category: "math" },
  cos: { arity: 1, fusible: true, vectorizable: true, category: "math" },
  floor: { arity: 1, fusible: true, vectorizable: true, category: "math" },
  ceil: { arity: 1, fusible: true, vectorizable: true, category: "math" },
  round: { arity: 1, fusible: true, vectorizable: true, category: "math" },
  sign: { arity: 1, fusible: true, vectorizable: true, category: "math" },
  isfinite: { arity: 1, fusible: true, vectorizable: true, category: "math" },

  // ---------------------------------------------------------------------------
  // Binary Arithmetic
  // ---------------------------------------------------------------------------
  add: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    wgslInfix: "+",
  },
  sub: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    wgslInfix: "-",
  },
  mul: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    wgslInfix: "*",
  },
  div: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    wgslInfix: "/",
  },
  pow: { arity: 2, fusible: true, vectorizable: true, category: "arithmetic" },
  min: { arity: 2, fusible: true, vectorizable: true, category: "arithmetic" },
  max: { arity: 2, fusible: true, vectorizable: true, category: "arithmetic" },
  mod: {
    arity: 2,
    fusible: true,
    vectorizable: false,
    category: "arithmetic",
    wgslInfix: "%",
  },

  // ---------------------------------------------------------------------------
  // Comparisons (return f32 0.0 or 1.0)
  // ---------------------------------------------------------------------------
  eq: { arity: 2, fusible: true, vectorizable: true, category: "comparison" },
  ne: { arity: 2, fusible: true, vectorizable: true, category: "comparison" },
  lt: { arity: 2, fusible: true, vectorizable: true, category: "comparison" },
  le: { arity: 2, fusible: true, vectorizable: true, category: "comparison" },
  gt: { arity: 2, fusible: true, vectorizable: true, category: "comparison" },
  ge: { arity: 2, fusible: true, vectorizable: true, category: "comparison" },

  // ---------------------------------------------------------------------------
  // Ternary Operations
  // ---------------------------------------------------------------------------
  where: { arity: 3, fusible: true, vectorizable: true, category: "ternary" },

  // ---------------------------------------------------------------------------
  // Type Casts
  // ---------------------------------------------------------------------------
  cast_f16: { arity: 1, fusible: true, vectorizable: true, category: "cast" },
  cast_f32: { arity: 1, fusible: true, vectorizable: true, category: "cast" },
  cast_i32: { arity: 1, fusible: true, vectorizable: true, category: "cast" },
  cast_u32: { arity: 1, fusible: true, vectorizable: true, category: "cast" },
};

// gelu_tanh is an alias for gelu (both use tanh approximation)
OP_REGISTRY.gelu_tanh = OP_REGISTRY.gelu;

// ---------------------------------------------------------------------------
// Bitwise / Shift Ops (used by tile-compiler, not fusible in elementwise)
// ---------------------------------------------------------------------------
for (const [name, infix] of Object.entries({
  and: "&",
  or: "|",
  xor: "^",
  shr: ">>",
  shl: "<<",
} as Record<string, string>)) {
  OP_REGISTRY[name] = {
    arity: 2,
    fusible: false,
    vectorizable: false,
    category: "bitwise",
    wgslInfix: infix,
  };
}
OP_REGISTRY["not"] = {
  arity: 1,
  fusible: false,
  vectorizable: false,
  category: "bitwise",
  wgslPrefix: "!",
};

// ============================================================================
// Helper Functions
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
