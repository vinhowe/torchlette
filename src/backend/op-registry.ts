/**
 * Unified Op Registry (backend-agnostic)
 *
 * Single source of truth for op metadata used by:
 * - Fusion detection (engine/fusion-detect.ts): fusible flag
 * - Op dispatch (engine/op-dispatch.ts): arity, category
 * - Tile compiler (webgpu/tile-compiler.ts): wgslInfix, wgslPrefix, wgslFnName
 * - Vectorization (webgpu/fusion-types.ts): vectorizable flag
 * - Dtype safety (engine/dtype-rules.ts): dtypeRule
 * - Autograd (frontend.ts): grad, ttGrad, tsGrad
 *
 * Op behavior (WGSL codegen) lives in fusion-tile-ir.ts via BlockExpr methods.
 */

import type { RuntimeEngine } from "../runtime/engine";
import type { Tensor as RuntimeTensor } from "../runtime/tensor";

// ============================================================================
// Types
// ============================================================================

type OpArity = 1 | 2 | 3;

export type OpDtypeRule =
  | "preserve"
  | "f32_required"
  | "always_f32"
  | "promote_inputs";

/**
 * Gradient function for a unary op.
 * @param rt - RuntimeEngine for tensor ops
 * @param grad - Upstream gradient (RuntimeTensor)
 * @param savedA - Saved input tensor (unwrapped via _unwrap()), undefined if needsSave=false
 */
export type UnaryGradFn = (
  rt: RuntimeEngine,
  grad: RuntimeTensor,
  savedA: RuntimeTensor | undefined,
) => RuntimeTensor;

/**
 * Gradient function for tensor+tensor binary ops.
 * @param rt - RuntimeEngine for tensor ops
 * @param grad - Upstream gradient
 * @param getUnwrapped - Access unwrapped saved tensor at index
 */
export type BinaryTTGradFn = (
  rt: RuntimeEngine,
  grad: RuntimeTensor,
  getUnwrapped: (i: number) => RuntimeTensor,
) => [RuntimeTensor, RuntimeTensor];

/**
 * Gradient function for tensor+scalar binary ops.
 * @param rt - RuntimeEngine for tensor ops
 * @param grad - Upstream gradient
 * @param getUnwrapped - Access unwrapped saved tensor at index
 * @param scalar - The scalar operand value
 * @param scalarIsA - True if the scalar is the left operand
 */
export type BinaryTSGradFn = (
  rt: RuntimeEngine,
  grad: RuntimeTensor,
  getUnwrapped: (i: number) => RuntimeTensor,
  scalar: number,
  scalarIsA: boolean,
) => RuntimeTensor[];

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

  // --- Autograd fields (optional, only for differentiable elementwise ops) ---

  /** Unary gradient function. null = non-differentiable (floor, ceil, etc). */
  grad?: UnaryGradFn | null;
  /** Whether to save input for backward (default true for ops with grad). */
  needsSave?: boolean;
  /** Autocast category name (omit = no autocast). */
  autocast?: string;

  /** Binary tensor+tensor gradient function. */
  ttGrad?: BinaryTTGradFn;
  /** Binary tensor+scalar gradient function (default: pass through). */
  tsGrad?: BinaryTSGradFn;
  /** Whether to save inputs for binary backward. */
  saveBinary?: boolean;
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
    grad: (rt, g, s) => rt.mul(g, rt.gt(s!, 0)),
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
    grad: (rt, g, s) => {
      const sig = rt.sigmoid(s!);
      return rt.mul(g, rt.add(sig, rt.mul(s!, rt.mul(sig, rt.sub(1, sig)))));
    },
  },
  sigmoid: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
    dtypeRule: "preserve",
    grad: (rt, g, s) => {
      const sig = rt.sigmoid(s!);
      return rt.mul(rt.mul(sig, rt.sub(1, sig)), g);
    },
  },
  tanh: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "activation",
    dtypeRule: "preserve",
    grad: (rt, g, s) => {
      const t = rt.tanh(s!);
      return rt.mul(rt.sub(1, rt.mul(t, t)), g);
    },
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
    needsSave: false,
    grad: (rt, g) => rt.neg(g),
  },
  abs: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
    grad: (rt, g, s) => rt.mul(g, rt.sign(s!)),
  },
  exp: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "f32_required",
    autocast: "exp",
    grad: (rt, g, s) => rt.mul(g, rt.exp(s!)),
  },
  log: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "f32_required",
    autocast: "log",
    grad: (rt, g, s) => rt.div(g, rt.add(s!, 1e-8)),
  },
  sqrt: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
    grad: (rt, g, s) => {
      const sqrtA = rt.sqrt(s!);
      return rt.mul(g, rt.div(0.5, rt.add(sqrtA, 1e-8)));
    },
  },
  rsqrt: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
    wgslFnName: "inverseSqrt",
    grad: (rt, g, s) => {
      const r = rt.rsqrt(s!);
      return rt.mul(g, rt.mul(-0.5, rt.mul(r, rt.mul(r, r))));
    },
  },
  sin: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
    grad: (rt, g, s) => rt.mul(g, rt.cos(s!)),
  },
  cos: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
    grad: (rt, g, s) => rt.mul(g, rt.neg(rt.sin(s!))),
  },
  floor: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
    grad: null,
  },
  ceil: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
    grad: null,
  },
  round: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
    grad: null,
  },
  sign: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "preserve",
    grad: null,
  },
  isfinite: {
    arity: 1,
    fusible: true,
    vectorizable: true,
    category: "math",
    dtypeRule: "always_f32",
    grad: null,
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
    ttGrad: (_rt, g) => [g, g],
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
    saveBinary: true,
    ttGrad: (rt, g, gs) => [rt.mul(g, gs(1)), rt.mul(g, gs(0))],
    tsGrad: (rt, g, _gs, s) => [rt.mul(g, s)],
  },
  div: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    dtypeRule: "promote_inputs",
    wgslInfix: "/",
    saveBinary: true,
    ttGrad: (rt, g, gs) => {
      const sA = gs(0),
        sB = gs(1);
      return [rt.div(g, sB), rt.mul(g, rt.div(rt.neg(sA), rt.mul(sB, sB)))];
    },
    tsGrad: (rt, g, gs, scalar, scalarIsA) => {
      if (!scalarIsA) return [rt.div(g, scalar)]; // a / scalar
      // scalar / b → -scalar / b² * grad
      const sB = gs(0);
      return [rt.mul(g, rt.div(-scalar, rt.mul(sB, sB)))];
    },
  },
  pow: {
    arity: 2,
    fusible: true,
    vectorizable: true,
    category: "arithmetic",
    dtypeRule: "f32_required",
    saveBinary: true,
    // d/da pow(a,b) = b * pow(a, b-1), d/db pow(a,b) = pow(a,b) * log(a)
    ttGrad: (rt, g, gs) => {
      const sA = gs(0),
        sB = gs(1);
      return [
        rt.mul(g, rt.mul(sB, rt.pow(sA, rt.sub(sB, 1)))),
        rt.mul(g, rt.mul(rt.pow(sA, sB), rt.log(sA))),
      ];
    },
    tsGrad: (rt, g, gs, scalar, scalarIsA) => {
      const saved = gs(0);
      if (scalarIsA) {
        // d/db a^b = a^b * log(a)
        return [rt.mul(g, rt.mul(rt.pow(scalar, saved), Math.log(scalar)))];
      }
      // d/da a^n = n * a^(n-1)
      return [rt.mul(g, rt.mul(scalar, rt.pow(saved, scalar - 1)))];
    },
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

// ============================================================================
// Autograd helpers
// ============================================================================

/** Op names with unary autograd specs (grad field is defined, even if null). */
export const UNARY_AUTOGRAD_OPS: string[] = Object.keys(OP_REGISTRY).filter(
  (k) => OP_REGISTRY[k].arity === 1 && "grad" in OP_REGISTRY[k],
);

/** Op names with binary autograd specs (ttGrad field is defined). */
export const BINARY_AUTOGRAD_OPS: string[] = Object.keys(OP_REGISTRY).filter(
  (k) => OP_REGISTRY[k].ttGrad != null,
);
