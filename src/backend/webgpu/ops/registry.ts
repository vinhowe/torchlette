// Re-export from backend-agnostic location (preserves existing import paths)

export type { OpDtypeRule } from "../../../ops/registry";
export {
  canVectorize,
  getOpDtypeRule,
  getWgslFnName,
  getWgslInfix,
  getWgslPrefix,
  isUnaryOp,
  OP_REGISTRY,
} from "../../../ops/registry";
