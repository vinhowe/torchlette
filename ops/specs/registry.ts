import type { OpSpec } from "./types";

export const OP_SPECS: OpSpec[] = [
  {
    name: "add",
    torchOp: "add",
    signature: "add(a, b, options?: { alpha?: number })",
    optionsDefaults: { alpha: 1 },
    status: "implemented",
  },
  {
    name: "mul",
    torchOp: "mul",
    signature: "mul(a, b)",
    status: "implemented",
  },
  {
    name: "sum",
    torchOp: "sum",
    signature:
      "sum(input, options?: { dim?: number | number[] | null; keepdim?: boolean; dtype?: DType })",
    optionsDefaults: { dim: null, keepdim: false, dtype: null },
    status: "implemented",
    note: "dim/keepdim are supported; dtype is not supported yet.",
  },
  {
    name: "sqrt",
    torchOp: "sqrt",
    signature: "sqrt(input)",
    status: "implemented",
  },
  {
    name: "gather",
    torchOp: "gather",
    signature: "gather(input, index, options: { dim: number })",
    status: "implemented",
  },
  {
    name: "scatterAdd",
    torchOp: "scatter_add",
    signature: "scatterAdd(input, index, src, options: { dim: number })",
    status: "implemented",
  },
  {
    name: "sub",
    torchOp: "sub",
    signature: "sub(a, b, options?: { alpha?: number })",
    optionsDefaults: { alpha: 1 },
    status: "implemented",
  },
  {
    name: "div",
    torchOp: "div",
    signature: 'div(a, b, options?: { roundingMode?: "floor" | "trunc" })',
    optionsDefaults: { roundingMode: null },
    status: "planned",
  },
  {
    name: "neg",
    torchOp: "neg",
    signature: "neg(input)",
    status: "planned",
  },
  {
    name: "abs",
    torchOp: "abs",
    signature: "abs(input)",
    status: "planned",
  },
  {
    name: "exp",
    torchOp: "exp",
    signature: "exp(input)",
    status: "planned",
  },
  {
    name: "log",
    torchOp: "log",
    signature: "log(input)",
    status: "planned",
  },
  {
    name: "relu",
    torchOp: "relu",
    signature: "relu(input)",
    status: "implemented",
  },
  {
    name: "matmul",
    torchOp: "matmul",
    signature: "matmul(a, b)",
    status: "implemented",
    note: "Supports PyTorch-style 1D/2D/ND matmul with batch broadcasting.",
  },
  {
    name: "mean",
    torchOp: "mean",
    signature:
      "mean(input, options?: { dim?: number | number[] | null; keepdim?: boolean; dtype?: DType })",
    optionsDefaults: { dim: null, keepdim: false, dtype: null },
    status: "implemented",
    note: "dim/keepdim are supported; dtype is not supported yet.",
  },
  {
    name: "reshape",
    torchOp: "reshape",
    signature: "reshape(input, options: { shape: number[] })",
    status: "implemented",
  },
  {
    name: "transpose",
    torchOp: "transpose",
    signature: "transpose(input, options: { dim0: number; dim1: number })",
    status: "implemented",
  },
  {
    name: "linalg.norm",
    torchOp: "linalg.norm",
    signature:
      "norm(input, options?: { ord?: OrdType | null; dim?: OrdDimType | null; keepdim?: boolean; dtype?: DType })",
    optionsDefaults: { ord: null, dim: null, keepdim: false, dtype: null },
    status: "planned",
    note: "Torchlette will expose this as linalg.norm later.",
  },
];
