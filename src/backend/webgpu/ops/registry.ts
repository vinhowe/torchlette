/**
 * Unified Op Registry for WebGPU Fusion
 *
 * Single source of truth for op metadata used by:
 * - Fusion detection (fusion-detect.ts): fusible flag
 * - Op dispatch (op-dispatch.ts): arity, category
 * - Tile compiler (tile-compiler.ts): wgslInfix, wgslPrefix, wgslFnName
 * - Vectorization (fusion-types.ts): vectorizable flag
 * - Dtype safety (dtype-rules.ts): dtypeRule
 *
 * Op behavior (WGSL codegen) lives in fusion-tile-ir.ts via BlockExpr methods.
 */

// ============================================================================
// Types
// ============================================================================

type OpArity = 1 | 2 | 3;

export type OpDtypeRule =
  | "preserve"
  | "f32_required"
  | "always_f32"
  | "promote_inputs";

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

  /** Dtype behavior rule for runtime dtype safety. */
  dtypeRule?: OpDtypeRule;

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
  // Activation Functions (all preserve dtype)
  // ---------------------------------------------------------------------------
  relu: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
    dtypeRule: "preserve",
  },
  gelu: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
    dtypeRule: "preserve",
  },
  gelu_erf: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
    dtypeRule: "preserve",
  },
  silu: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
    dtypeRule: "preserve",
  },
  sigmoid: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
    dtypeRule: "preserve",
  },
  tanh: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
    dtypeRule: "preserve",
  },
  softplus: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
    dtypeRule: "preserve",
  },

  // ---------------------------------------------------------------------------
  // Math Functions
  // ---------------------------------------------------------------------------
  neg: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
    wgslPrefix: "-",
  },
  abs: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
  },
  exp: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "f32_required",
  },
  log: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "f32_required",
  },
  sqrt: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
  },
  rsqrt: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
    wgslFnName: "inverseSqrt",
  },
  sin: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
  },
  cos: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
  },
  floor: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
  },
  ceil: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
  },
  round: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
  },
  sign: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
  },
  isfinite: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "always_f32",
  },

  // ---------------------------------------------------------------------------
  // Binary Arithmetic
  // ---------------------------------------------------------------------------
  add: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    dtypeRule: "promote_inputs",
    wgslInfix: "+",
  },
  sub: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    dtypeRule: "promote_inputs",
    wgslInfix: "-",
  },
  mul: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    dtypeRule: "promote_inputs",
    wgslInfix: "*",
  },
  div: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    dtypeRule: "promote_inputs",
    wgslInfix: "/",
  },
  pow: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    dtypeRule: "f32_required",
  },
  min: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    dtypeRule: "f32_required",
  },
  max: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    dtypeRule: "f32_required",
  },
  mod: {
    arity: 2,
    fusible: true,
    vectorizable: false,
    category: "arithmetic",
    wgslInfix: "%",
  },

  // ---------------------------------------------------------------------------
  // Comparisons (always produce f32 0.0 or 1.0)
  // ---------------------------------------------------------------------------
  eq: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "comparison",
    dtypeRule: "always_f32",
  },
  ne: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "comparison",
    dtypeRule: "always_f32",
  },
  lt: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "comparison",
    dtypeRule: "always_f32",
  },
  le: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "comparison",
    dtypeRule: "always_f32",
  },
  gt: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "comparison",
    dtypeRule: "always_f32",
  },
  ge: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "comparison",
    dtypeRule: "always_f32",
  },

  // ---------------------------------------------------------------------------
  // Ternary Operations
  // ---------------------------------------------------------------------------
  where: {
    arity: 3,
    fusible: true,
    vectorizable: true,
    category: "ternary",
    dtypeRule: "preserve",
  },

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

/** Get the dtype rule for an op from the registry. */
export function getOpDtypeRule(op: string): OpDtypeRule | undefined {
  return OP_REGISTRY[op]?.dtypeRule;
}
