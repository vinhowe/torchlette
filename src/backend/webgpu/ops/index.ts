/**
 * WebGPU Ops Module
 *
 * Unified op registry for all fusion paths.
 */

export {
  // Types
  type OpDef,
  type OpArity,

  // Registry
  OP_REGISTRY,

  // Core functions
  getOpDef,
  hasOp,
  getExpr,
  isFusible,
  canVectorize,
  getArity,

  // Arity checks
  isUnaryOp,
  isBinaryOp,
  isTernaryOp,

  // Listing functions
  getOpsByCategory,
  getAllFusibleOps,
  getAllUnaryOps,
  getAllBinaryOps,

  // Backward compatibility
  UNARY_EXPR,
  BINARY_EXPR,
} from "./registry";
