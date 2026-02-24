/**
 * General-Purpose Fused Elementwise Kernel Codegen
 *
 * Generates WGSL compute shaders for arbitrary chains of elementwise operations.
 * Uses expression-based SSA code generation where each IR node becomes a WGSL expression.
 *
 * Key design:
 * - Single source of truth for op -> expression mapping (via ops/registry.ts)
 * - Composable with matmul epilogues or standalone
 * - Handles broadcasting for binary ops
 * - Supports scalar specialization
 * - Memory coalescing via vec4/vec2 vectorization (§15.3)
 */

import type { DType } from "../types";
import type { IRNode } from "../../engine/ir";
import { dtypeToWgslStorage as dtypeToWgsl } from "./shape-utils";

// Import from unified op registry - single source of truth
import {
  OP_REGISTRY,
  getExpr,
  isFusible,
  isUnaryOp as registryIsUnaryOp,
  isBinaryOp as registryIsBinaryOp,
  getArity,
  canVectorize as canVectorizeOp,
  getVectorExpr,
  // Re-export for backward compatibility
  UNARY_EXPR,
  BINARY_EXPR,
} from "./ops/registry";

// Re-export UNARY_EXPR and BINARY_EXPR for backward compatibility
export { UNARY_EXPR, BINARY_EXPR };

/**
 * Check if an operation is a unary elementwise op.
 */
export function isUnaryOp(op: string): boolean {
  return registryIsUnaryOp(op);
}

/**
 * Check if an operation is a binary elementwise op.
 */
export function isBinaryOp(op: string): boolean {
  return registryIsBinaryOp(op);
}

/**
 * Check if an operation can be fused (elementwise).
 */
export function isFusibleOp(op: string): boolean {
  return isFusible(op);
}

/**
 * Get expression generator for an operation.
 */
export function getExprGenerator(
  op: string,
): ((inputs: string[]) => string) | null {
  const def = OP_REGISTRY[op];
  if (!def) return null;

  return (inputs: string[]) => getExpr(op, inputs);
}

// ============================================================================
// Fusion Recipe Types
// ============================================================================

/**
 * A node in the fused computation graph.
 */
export interface FusedNode {
  id: number;
  op: string;
  inputs: number[]; // Node IDs or external input indices (negative)
  shape: number[];
  dtype: DType;
  isOutput?: boolean;
}

/**
 * External input to the fused kernel.
 */
export interface FusedInput {
  id: number; // Original node ID
  index: number; // Binding index in the kernel
  shape: number[];
  dtype: DType;
  isScalar?: boolean; // If true, broadcast from single value
  isInlinedConstant?: boolean; // If true, skip storage binding and use literal
  inlinedValue?: number; // The constant value to inline as WGSL literal
}

/**
 * Output descriptor for a fused kernel.
 */
export interface FusedOutput {
  nodeId: number; // Node ID that produces this output
  index: number; // Binding index in the kernel
  shape: number[];
  dtype: DType;
}

/**
 * Recipe for generating a fused kernel.
 * Supports single or multi-output fusion (§15.2).
 */
export interface FusedKernelRecipe {
  /** Unique identifier for this fusion */
  id: string;
  /** Nodes to compute (in topological order) */
  nodes: FusedNode[];
  /** External inputs (tensors from outside the fusion) */
  inputs: FusedInput[];
  /** Outputs (supports multiple for §15.2) */
  outputs: FusedOutput[];
  /** Workgroup size */
  workgroupSize?: number;
}

// ============================================================================
// Vectorization Helpers (§15.3)
// ============================================================================

/**
 * Vector width for memory coalescing.
 */
export type VectorWidth = 1 | 2 | 4;

/**
 * Determine the best vector width for a given shape and dtype.
 *
 * Per §15.3:
 * - Prefer vec4 when innermost dim divisible by 4
 * - Fall back to vec2 when divisible by 2 but not 4
 * - Use scalar when neither applies
 */
export function selectVectorWidth(
  shape: number[],
  dtype: DType,
): VectorWidth {
  // Only f32 and f16 support vectorization currently
  if (dtype !== "f32" && dtype !== "f16") {
    return 1;
  }

  // Need at least one dimension
  if (shape.length === 0) {
    return 1;
  }

  // Check innermost dimension
  const innerDim = shape[shape.length - 1];

  // Minimum size threshold - don't vectorize tiny tensors
  const totalElements = shape.reduce((a, b) => a * b, 1);
  if (totalElements < 16) {
    return 1;
  }

  if (innerDim >= 4 && innerDim % 4 === 0) {
    return 4;
  }
  if (innerDim >= 2 && innerDim % 2 === 0) {
    return 2;
  }
  return 1;
}

/**
 * Check if all inputs are compatible with vectorization at the given width.
 *
 * Inputs are compatible if:
 * - They have the same innermost dimension divisible by vector width, OR
 * - They are scalars (can be splatted to vector)
 */
export function canVectorize(
  outputShape: number[],
  inputs: { shape: number[]; dtype: DType }[],
  vectorWidth: VectorWidth,
): boolean {
  if (vectorWidth === 1) return true;

  // Output must be divisible
  if (outputShape.length === 0) return false;
  const outputInner = outputShape[outputShape.length - 1];
  if (outputInner % vectorWidth !== 0) return false;

  // Check each input
  for (const input of inputs) {
    const inputSize = input.shape.reduce((a, b) => a * b, 1);

    // Scalars are fine (splatted)
    if (inputSize === 1) continue;

    // Check innermost dimension
    if (input.shape.length === 0) return false;
    const inputInner = input.shape[input.shape.length - 1];

    // Input must have same innermost or be broadcastable
    if (inputInner !== outputInner && inputInner !== 1) {
      // Innermost dims don't match and input isn't broadcast - can't vectorize
      return false;
    }

    // If innermost matches output, it must be divisible
    if (inputInner === outputInner && inputInner % vectorWidth !== 0) {
      return false;
    }
  }

  return true;
}

/**
 * Get WGSL vector type name.
 */
export function getVectorType(dtype: DType, width: VectorWidth): string {
  if (width === 1) return dtypeToWgsl(dtype);
  const base = dtypeToWgsl(dtype);
  return `vec${width}<${base}>`;
}

/**
 * Generate a splat expression (scalar to vector).
 */
export function genSplat(expr: string, dtype: DType, width: VectorWidth): string {
  if (width === 1) return expr;
  const vecType = getVectorType(dtype, width);
  return `${vecType}(${expr})`;
}

// ============================================================================
// Broadcasting Helpers
// ============================================================================

/**
 * Generate WGSL code to compute a broadcast index.
 *
 * Given a linear index into the output tensor and the input shape,
 * computes the linear index into the input tensor accounting for broadcasting.
 */
export function genBroadcastIndex(
  outputShape: number[],
  inputShape: number[],
  idxVar: string,
  inputIdxVar: string,
): string {
  // If shapes are identical, no broadcasting needed
  if (
    outputShape.length === inputShape.length &&
    outputShape.every((d, i) => d === inputShape[i])
  ) {
    return `let ${inputIdxVar} = ${idxVar};`;
  }

  // If input is scalar (shape [1] or []), always index 0
  const inputSize = inputShape.reduce((a, b) => a * b, 1);
  if (inputSize === 1) {
    return `let ${inputIdxVar} = 0u;`;
  }

  // General broadcasting: compute multi-dimensional indices, then re-linearize
  const lines: string[] = [];
  const rank = outputShape.length;
  const inputRank = inputShape.length;
  const rankDiff = rank - inputRank;

  // Compute output coordinates
  lines.push(`var _tmp_${inputIdxVar} = ${idxVar};`);
  const outCoords: string[] = [];
  for (let i = rank - 1; i >= 0; i--) {
    const coord = `_c${i}_${inputIdxVar}`;
    const dim = outputShape[i];
    lines.push(`let ${coord} = _tmp_${inputIdxVar} % ${dim}u;`);
    lines.push(`_tmp_${inputIdxVar} = _tmp_${inputIdxVar} / ${dim}u;`);
    outCoords.unshift(coord);
  }

  // Map to input coordinates (with broadcasting)
  const inCoords: string[] = [];
  for (let i = 0; i < inputRank; i++) {
    const outIdx = i + rankDiff;
    const inDim = inputShape[i];
    const outCoord = outCoords[outIdx];
    if (inDim === 1) {
      inCoords.push("0u");
    } else {
      inCoords.push(outCoord);
    }
  }

  // Compute input linear index
  let inIdxExpr = inCoords[0];
  for (let i = 1; i < inputRank; i++) {
    const stride = inputShape.slice(i + 1).reduce((a, b) => a * b, 1);
    inIdxExpr = `(${inIdxExpr} * ${inputShape[i]}u + ${inCoords[i]})`;
  }

  // Simplified case: 1D input
  if (inputRank === 1) {
    inIdxExpr = inCoords[0];
  }

  lines.push(`let ${inputIdxVar} = ${inIdxExpr};`);

  return lines.join("\n  ");
}

/**
 * Check if broadcasting is needed between output and input shapes.
 */
export function needsBroadcast(
  outputShape: number[],
  inputShape: number[],
): boolean {
  if (outputShape.length !== inputShape.length) return true;
  return !outputShape.every((d, i) => d === inputShape[i]);
}

// ============================================================================
// SSA Code Generator
// ============================================================================

/**
 * Result of generating fused expressions.
 */
interface GeneratedExpressions {
  /** Variable declarations and assignments */
  lines: string[];
  /** Map from node ID to variable name */
  varNames: Map<number, string>;
  /** The final output variable name */
  outputVar: string;
}

/**
 * Generate SSA-style WGSL expressions for a fusion recipe.
 * @param recipe - The fusion recipe
 * @param vectorWidth - Vector width (1 for scalar, 2 or 4 for vectorized)
 */
export function generateFusedExpressions(
  recipe: FusedKernelRecipe,
  vectorWidth: VectorWidth = 1,
): GeneratedExpressions {
  const varNames = new Map<number, string>();
  const lines: string[] = [];

  // Generate vector-compatible zero constant
  const zeroExpr = vectorWidth === 1 ? "0.0" : `vec${vectorWidth}<f32>(0.0)`;
  const oneExpr = vectorWidth === 1 ? "1.0" : `vec${vectorWidth}<f32>(1.0)`;

  // Track which nodes are used by later nodes (for determining if we need temp vars)
  const useCount = new Map<number, number>();
  for (const node of recipe.nodes) {
    for (const inputId of node.inputs) {
      useCount.set(inputId, (useCount.get(inputId) ?? 0) + 1);
    }
  }

  // Process nodes in topological order
  for (let nodeIdx = 0; nodeIdx < recipe.nodes.length; nodeIdx++) {
    const node = recipe.nodes[nodeIdx];
    const varName = `t${node.id}`;

    // Get input expressions
    const inputExprs: string[] = [];
    for (const inputId of node.inputs) {
      if (inputId < 0) {
        // External input (negative IDs map to inputs)
        const inputIdx = -inputId - 1;
        inputExprs.push(`v${inputIdx}`);
      } else {
        // Internal node reference
        const refVar = varNames.get(inputId);
        if (!refVar) {
          throw new Error(`Missing variable for node ${inputId}`);
        }
        inputExprs.push(refVar);
      }
    }

    // Generate expression using the unified registry
    let expr: string;
    const vecExprFn = vectorWidth > 1 ? getVectorExpr(node.op) : undefined;
    if (vecExprFn) {
      // Use vector-specific expression (e.g., vec4<f32>(x) instead of f32(x))
      expr = vecExprFn(inputExprs[0], vectorWidth);
    } else if (node.op in UNARY_EXPR) {
      // Pass zero/one expressions for vector-compatible ops like relu
      expr = UNARY_EXPR[node.op](inputExprs[0], zeroExpr, oneExpr);
    } else if (node.op in BINARY_EXPR) {
      expr = BINARY_EXPR[node.op](inputExprs[0], inputExprs[1]);
    } else {
      // Ternary or other registry ops (e.g., where) — use getExpr()
      expr = getExpr(node.op, inputExprs, { zero: zeroExpr, one: oneExpr });
    }

    // If this is the output or used multiple times, create a variable
    const isOutput = node.isOutput ?? nodeIdx === recipe.nodes.length - 1;
    const usedMultiple = (useCount.get(node.id) ?? 0) > 1;

    if (isOutput || usedMultiple) {
      lines.push(`let ${varName} = ${expr};`);
      varNames.set(node.id, varName);
    } else {
      // Inline the expression
      varNames.set(node.id, `(${expr})`);
    }
  }

  // Find output variable
  const outputNode = recipe.nodes[recipe.nodes.length - 1];
  const outputVar = varNames.get(outputNode.id) ?? `t${outputNode.id}`;

  return { lines, varNames, outputVar };
}

// ============================================================================
// Kernel Generation
// ============================================================================

/**
 * Options for kernel generation.
 */
export interface KernelGenOptions {
  /** Workgroup size (default: 256) */
  workgroupSize?: number;
  /** Enable vectorized loads/stores (auto-selects best width if true) */
  vectorize?: boolean;
  /** Force specific vector width (overrides auto-selection) */
  forceVectorWidth?: VectorWidth;
}

/**
 * Generated kernel result.
 */
export interface GeneratedKernel {
  /** WGSL shader source */
  source: string;
  /** Workgroup size used */
  workgroupSize: number;
  /** Number of input bindings */
  inputBindings: number;
  /** Cache key for this kernel */
  cacheKey: string;
  /** Vector width used (1 = scalar, 2 = vec2, 4 = vec4) */
  vectorWidth: VectorWidth;
  /** Number of work items (total elements / vector width) */
  workItems: number;
  /** X dimension of dispatch grid (for 2D dispatch when workgroups > 65535) */
  gridSizeX: number;
}

/**
 * Cheaply compute the cache key and metadata for a fusion recipe WITHOUT
 * generating the full WGSL source. Used by FusionKernelCache to check cache
 * before doing expensive codegen.
 */
// Maximum workgroups per dimension in WebGPU (per spec)
const MAX_WORKGROUPS_PER_DIM = 65535;

export function computeKernelMeta(
  recipe: FusedKernelRecipe,
  options: KernelGenOptions = {},
): { cacheKey: string; vectorWidth: VectorWidth; workItems: number; workgroupSize: number; gridSizeX: number } {
  const workgroupSize = options.workgroupSize ?? recipe.workgroupSize ?? 256;
  const outputShape = recipe.outputs[0].shape;
  const primaryDtype = recipe.outputs[0].dtype;
  const totalElements = outputShape.reduce((a, b) => a * b, 1);

  let vectorWidth: VectorWidth = 1;
  if (options.forceVectorWidth !== undefined) {
    vectorWidth = options.forceVectorWidth;
  } else if (options.vectorize) {
    const candidateWidth = selectVectorWidth(outputShape, primaryDtype);
    if (canVectorize(outputShape, recipe.inputs, candidateWidth)) {
      vectorWidth = candidateWidth;
    }
  }
  if (vectorWidth > 1) {
    if (recipe.nodes.some((n) => !canVectorizeOp(n.op))) {
      vectorWidth = 1;
    }
  }

  const workItems = Math.ceil(totalElements / vectorWidth);
  const totalWorkgroups = Math.ceil(workItems / workgroupSize);
  const gridSizeX = totalWorkgroups <= MAX_WORKGROUPS_PER_DIM
    ? totalWorkgroups
    : MAX_WORKGROUPS_PER_DIM;
  // Include 2D dispatch in cache key so 1D and 2D kernels are cached separately
  const baseCacheKey = generateKernelCacheKey(recipe, vectorWidth);
  const cacheKey = gridSizeX >= MAX_WORKGROUPS_PER_DIM ? baseCacheKey + ":2d" : baseCacheKey;
  return { cacheKey, vectorWidth, workItems, workgroupSize, gridSizeX };
}

/**
 * Generate a complete WGSL compute shader for a fused kernel.
 * Supports single or multi-output fusion (§15.2).
 */
export function generateFusedKernel(
  recipe: FusedKernelRecipe,
  options: KernelGenOptions = {},
): GeneratedKernel {
  // Derive vector width, workgroup size, and cache key from shared logic
  const meta = computeKernelMeta(recipe, options);
  const { vectorWidth, workItems, workgroupSize, gridSizeX } = meta;

  const outputShape = recipe.outputs[0].shape;
  const primaryDtype = recipe.outputs[0].dtype;
  const totalElements = outputShape.reduce((a, b) => a * b, 1);
  const useVec = vectorWidth > 1;
  const vecType = useVec ? getVectorType(primaryDtype, vectorWidth) : dtypeToWgsl(primaryDtype);

  // Build physical binding index map, skipping inlined constants
  const physicalBinding: (number | null)[] = []; // per-input: physical binding or null if inlined
  let nextBinding = 0;
  for (let i = 0; i < recipe.inputs.length; i++) {
    if (recipe.inputs[i].isInlinedConstant) {
      physicalBinding.push(null);
    } else {
      physicalBinding.push(nextBinding++);
    }
  }

  // Generate input bindings (skip inlined constants)
  const inputBindings: string[] = [];
  for (let i = 0; i < recipe.inputs.length; i++) {
    const binding = physicalBinding[i];
    if (binding === null) continue; // inlined constant — no storage binding
    const input = recipe.inputs[i];
    const wgslType = dtypeToWgsl(input.dtype);
    inputBindings.push(
      `@group(0) @binding(${binding}) var<storage, read> in${i}: array<${wgslType}>;`,
    );
  }

  // Output bindings start after non-inlined inputs
  const outputBindings: string[] = [];
  for (let i = 0; i < recipe.outputs.length; i++) {
    const output = recipe.outputs[i];
    const bindingIdx = nextBinding + i;
    const wgslType = dtypeToWgsl(output.dtype);
    outputBindings.push(
      `@group(0) @binding(${bindingIdx}) var<storage, read_write> out${i}: array<${wgslType}>;`,
    );
  }

  // Generate load code for each input
  const loadCode: string[] = [];
  for (let i = 0; i < recipe.inputs.length; i++) {
    const input = recipe.inputs[i];

    // Inlined constant: emit literal value instead of buffer load
    if (input.isInlinedConstant && input.inlinedValue !== undefined) {
      const litVal = formatWgslLiteral(input.inlinedValue, input.dtype);
      if (useVec) {
        loadCode.push(`let v${i} = ${getVectorType(input.dtype, vectorWidth)}(${litVal});`);
      } else {
        loadCode.push(`let v${i} = ${litVal};`);
      }
      continue;
    }

    const inputSize = input.shape.reduce((a, b) => a * b, 1);
    const isScalar = inputSize === 1;

    if (useVec) {
      if (isScalar) {
        // Scalar input: splat to vector using input's own dtype
        loadCode.push(`let v${i} = ${getVectorType(input.dtype, vectorWidth)}(in${i}[0]);`);
      } else if (needsBroadcast(outputShape, input.shape)) {
        // Broadcasting with vectorization - need to handle carefully
        // For now, fall back to element-by-element load with broadcast
        loadCode.push(genVectorizedBroadcastLoad(
          outputShape,
          input.shape,
          i,
          vectorWidth,
          input.dtype,
        ));
      } else {
        // Same shape: vectorized load using input's own dtype
        loadCode.push(genVectorizedLoad(i, vectorWidth, input.dtype));
      }
    } else {
      // Scalar path
      if (needsBroadcast(outputShape, input.shape)) {
        loadCode.push(genBroadcastIndex(outputShape, input.shape, "idx", `idx${i}`));
        loadCode.push(`let v${i} = in${i}[idx${i}];`);
      } else {
        loadCode.push(`let v${i} = in${i}[idx];`);
      }
    }
  }

  // Generate fused expressions with vector-compatible constants
  // For multi-output, we get varNames for all output nodes
  const { lines: exprLines, varNames } = generateFusedExpressions(recipe, vectorWidth);

  // Generate store code for all outputs
  const storeLines: string[] = [];
  for (let i = 0; i < recipe.outputs.length; i++) {
    const output = recipe.outputs[i];
    const outputVar = varNames.get(output.nodeId) ?? `t${output.nodeId}`;
    if (useVec) {
      storeLines.push(...genVectorizedStoreMulti(outputVar, i, vectorWidth));
    } else {
      storeLines.push(`out${i}[idx] = ${outputVar};`);
    }
  }
  const storeCode = storeLines.join("\n  ");

  // Index variable depends on vectorization and 2D dispatch
  const use2D = gridSizeX >= MAX_WORKGROUPS_PER_DIM;
  // For 2D dispatch, compute linear invocation index from 2D workgroup grid
  const gidExpr = use2D
    ? `(gid.x + gid.y * ${gridSizeX}u * ${workgroupSize}u)`
    : "gid.x";
  const idxExpr = useVec ? `${gidExpr} * ${vectorWidth}u` : gidExpr;
  // Bounds check always uses params.total_elements to ensure params binding is used
  // For vectorized: check work items (total_elements / vectorWidth)
  // For scalar: check total elements directly
  const boundsCheck = useVec
    ? `if (${gidExpr} * ${vectorWidth}u >= params.total_elements) { return; }`
    : `if (${gidExpr} >= params.total_elements) { return; }`;

  // Params binding comes after all outputs (using physical binding count)
  const paramsBinding = nextBinding + recipe.outputs.length;

  // Check if any input, output, or node involves f16 — if so, we need `enable f16;`
  const needsF16 =
    recipe.inputs.some(i => i.dtype === "f16") ||
    recipe.outputs.some(o => o.dtype === "f16") ||
    recipe.nodes.some(n => n.dtype === "f16");
  const enableF16 = needsF16 ? "enable f16;\n\n" : "";

  // Combine into full kernel
  const source = `${enableF16}// Fused elementwise kernel: ${recipe.id}
// Ops: ${recipe.nodes.map((n) => n.op).join(" -> ")}
// Outputs: ${recipe.outputs.length}
// Vectorization: ${useVec ? `vec${vectorWidth}` : "scalar"}

${inputBindings.join("\n")}
${outputBindings.join("\n")}

struct Params {
  total_elements: u32,
}
@group(0) @binding(${paramsBinding}) var<uniform> params: Params;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  ${boundsCheck}
  let idx = ${idxExpr};

  // Load inputs${useVec ? " (vectorized)" : ""}
  ${loadCode.join("\n  ")}

  // Fused computation
  ${exprLines.join("\n  ")}

  // Store outputs${useVec ? " (vectorized)" : ""}
  ${storeCode}
}
`.trim();

  return {
    source,
    workgroupSize,
    inputBindings: nextBinding, // physical (non-inlined) input count
    cacheKey: meta.cacheKey,
    vectorWidth,
    workItems,
    gridSizeX,
  };
}

/**
 * Generate vectorized store code for a specific output buffer.
 */
function genVectorizedStoreMulti(
  outputVar: string,
  outputIdx: number,
  vectorWidth: VectorWidth,
): string[] {
  const lines: string[] = [];
  for (let i = 0; i < vectorWidth; i++) {
    const component = vectorWidth === 2
      ? (["x", "y"][i])
      : (["x", "y", "z", "w"][i]);
    lines.push(`out${outputIdx}[idx + ${i}u] = ${outputVar}.${component};`);
  }
  return lines;
}

/**
 * Generate vectorized load code for a non-broadcast input.
 */
function genVectorizedLoad(
  inputIdx: number,
  vectorWidth: VectorWidth,
  dtype: DType,
): string {
  const vecType = getVectorType(dtype, vectorWidth);
  const loads: string[] = [];
  for (let i = 0; i < vectorWidth; i++) {
    loads.push(`in${inputIdx}[idx + ${i}u]`);
  }
  return `let v${inputIdx} = ${vecType}(${loads.join(", ")});`;
}

/**
 * Generate vectorized load with broadcasting.
 * Falls back to computing broadcast index per element.
 */
function genVectorizedBroadcastLoad(
  outputShape: number[],
  inputShape: number[],
  inputIdx: number,
  vectorWidth: VectorWidth,
  dtype: DType,
): string {
  const vecType = getVectorType(dtype, vectorWidth);
  const lines: string[] = [];

  // Generate per-element broadcast loads
  const elemVars: string[] = [];
  for (let i = 0; i < vectorWidth; i++) {
    const elemIdx = `idx + ${i}u`;
    const bcIdxVar = `bc${inputIdx}_${i}`;
    lines.push(genBroadcastIndexSingle(outputShape, inputShape, elemIdx, bcIdxVar));
    elemVars.push(`in${inputIdx}[${bcIdxVar}]`);
  }

  lines.push(`let v${inputIdx} = ${vecType}(${elemVars.join(", ")});`);
  return lines.join("\n  ");
}

/**
 * Generate broadcast index calculation for a single element.
 * Returns just the index assignment without declaring new vars for coords.
 */
function genBroadcastIndexSingle(
  outputShape: number[],
  inputShape: number[],
  idxExpr: string,
  resultVar: string,
): string {
  // If input is scalar, always 0
  const inputSize = inputShape.reduce((a, b) => a * b, 1);
  if (inputSize === 1) {
    return `let ${resultVar} = 0u;`;
  }

  const rank = outputShape.length;
  const inputRank = inputShape.length;
  const rankDiff = rank - inputRank;

  // Compute output coordinates inline
  const lines: string[] = [];
  lines.push(`var _t_${resultVar} = ${idxExpr};`);

  const outCoords: string[] = [];
  for (let i = rank - 1; i >= 0; i--) {
    const coord = `_c${i}_${resultVar}`;
    const dim = outputShape[i];
    lines.push(`let ${coord} = _t_${resultVar} % ${dim}u;`);
    lines.push(`_t_${resultVar} = _t_${resultVar} / ${dim}u;`);
    outCoords.unshift(coord);
  }

  // Map to input coordinates
  const inCoords: string[] = [];
  for (let i = 0; i < inputRank; i++) {
    const outIdx = i + rankDiff;
    const inDim = inputShape[i];
    const outCoord = outCoords[outIdx];
    inCoords.push(inDim === 1 ? "0u" : outCoord);
  }

  // Compute input linear index
  let inIdxExpr = inCoords[0];
  for (let i = 1; i < inputRank; i++) {
    inIdxExpr = `(${inIdxExpr} * ${inputShape[i]}u + ${inCoords[i]})`;
  }
  if (inputRank === 1) {
    inIdxExpr = inCoords[0];
  }

  lines.push(`let ${resultVar} = ${inIdxExpr};`);
  return lines.join("\n  ");
}

/**
 * Generate vectorized store code.
 */
function genVectorizedStore(
  outputVar: string,
  vectorWidth: VectorWidth,
): string {
  const lines: string[] = [];
  for (let i = 0; i < vectorWidth; i++) {
    const component = vectorWidth === 2
      ? (["x", "y"][i])
      : (["x", "y", "z", "w"][i]);
    lines.push(`out[idx + ${i}u] = ${outputVar}.${component};`);
  }
  return lines.join("\n  ");
}

/**
 * Format a numeric value as a WGSL literal for the given dtype.
 * Ensures f32 values have a decimal point, integers are unadorned, etc.
 */
function formatWgslLiteral(value: number, dtype: DType): string {
  switch (dtype) {
    case "f32": {
      // Ensure the literal has a decimal point for WGSL f32
      const s = String(value);
      if (s.includes(".") || s.includes("e") || s.includes("E")) {
        return s;
      }
      return s + ".0";
    }
    case "f16": {
      const s = String(value);
      // f16 literals in WGSL use the 'h' suffix
      if (s.includes(".") || s.includes("e") || s.includes("E")) {
        return s + "h";
      }
      return s + ".0h";
    }
    case "i32":
      return `${Math.trunc(value)}i`;
    case "u32":
      return `${Math.trunc(value)}u`;
    default: {
      const s = String(value);
      if (s.includes(".") || s.includes("e") || s.includes("E")) return s;
      return s + ".0";
    }
  }
}


/**
 * Generate a cache key for a fusion recipe.
 * Supports multi-output fusion (§15.2).
 */
export function generateKernelCacheKey(
  recipe: FusedKernelRecipe,
  vectorWidth: VectorWidth = 1,
): string {
  // Build a STRUCTURAL cache key that is stable across steps.
  // Uses relative node positions (not ephemeral LazyIRNode IDs).
  const parts: string[] = [`vec:${vectorWidth}`];

  // Add output signatures (supports multiple outputs)
  for (const output of recipe.outputs) {
    parts.push(`out${output.index}:${output.shape.join("x")}:${output.dtype}`);
  }

  // Add input signatures (include inlined constant values for cache differentiation)
  for (const input of recipe.inputs) {
    if (input.isInlinedConstant) {
      parts.push(`in${input.index}:const=${input.inlinedValue}`);
    } else {
      parts.push(`in${input.index}:${input.shape.join("x")}:${input.dtype}`);
    }
  }

  // Map absolute node IDs to relative positions for stable keys.
  // Internal refs use position index, external refs keep negative index.
  const idToPos = new Map<number, number>();
  for (let i = 0; i < recipe.nodes.length; i++) {
    idToPos.set(recipe.nodes[i].id, i);
  }

  // Add operation chain with RELATIVE references
  const opParts: string[] = [];
  for (let i = 0; i < recipe.nodes.length; i++) {
    const n = recipe.nodes[i];
    const inputRefs = n.inputs.map(inp => {
      const pos = idToPos.get(inp);
      return pos !== undefined ? `n${pos}` : `e${inp}`; // e = external (negative index)
    }).join(",");
    opParts.push(`${n.op}(${inputRefs})${n.isOutput ? "!" : ""}`);
  }
  parts.push(`ops:[${opParts.join(";")}]`);

  return parts.join("|");
}

// ============================================================================
// Recipe Building from IR
// ============================================================================

/**
 * Build a fusion recipe from IR nodes.
 * Supports multi-output fusion (§15.2).
 *
 * @param nodeIds - Node IDs to fuse (in topological order)
 * @param nodeById - Map of node ID to IRNode
 * @param inputNodeIds - External input node IDs (not in the fusion)
 * @param outputNodeIds - Output node IDs (defaults to last node if not specified)
 */
export function buildRecipeFromIR(
  nodeIds: number[],
  nodeById: Map<number, IRNode>,
  inputNodeIds: number[],
  outputNodeIds?: number[],
): FusedKernelRecipe {
  const nodeSet = new Set(nodeIds);

  // Build input mapping: external node ID -> negative index
  const inputMap = new Map<number, number>();
  const inputs: FusedInput[] = [];
  for (let i = 0; i < inputNodeIds.length; i++) {
    const nodeId = inputNodeIds[i];
    const node = nodeById.get(nodeId);
    inputMap.set(nodeId, -(i + 1)); // Negative indices for external inputs
    inputs.push({
      id: nodeId,
      index: i,
      shape: node?.shape ?? [1],
      dtype: node?.dtype ?? "f32",
    });
  }

  // Determine output nodes
  const outputIds = outputNodeIds ?? [nodeIds[nodeIds.length - 1]];
  const outputSet = new Set(outputIds);

  // Build fused nodes
  const nodes: FusedNode[] = [];
  for (const nodeId of nodeIds) {
    const node = nodeById.get(nodeId);
    if (!node) {
      throw new Error(`Node ${nodeId} not found`);
    }

    // Map inputs to either internal node IDs or external input indices
    const mappedInputs: number[] = [];
    for (const inputId of node.inputs) {
      if (nodeSet.has(inputId)) {
        mappedInputs.push(inputId);
      } else if (inputMap.has(inputId)) {
        mappedInputs.push(inputMap.get(inputId)!);
      } else {
        // This is a new external input
        const newIdx = inputs.length;
        inputMap.set(inputId, -(newIdx + 1));
        const inputNode = nodeById.get(inputId);
        inputs.push({
          id: inputId,
          index: newIdx,
          shape: inputNode?.shape ?? [1],
          dtype: inputNode?.dtype ?? "f32",
        });
        mappedInputs.push(-(newIdx + 1));
      }
    }

    nodes.push({
      id: nodeId,
      op: node.op,
      inputs: mappedInputs,
      shape: node.shape ?? [1],
      dtype: node.dtype ?? "f32",
      isOutput: outputSet.has(nodeId),
    });
  }

  // Build outputs array
  const outputs: FusedOutput[] = [];
  for (let i = 0; i < outputIds.length; i++) {
    const nodeId = outputIds[i];
    const node = nodeById.get(nodeId);
    outputs.push({
      nodeId,
      index: i,
      shape: node?.shape ?? [1],
      dtype: node?.dtype ?? "f32",
    });
  }

  return {
    id: `fused_${nodeIds.join("_")}`,
    nodes,
    inputs,
    outputs,
  };
}
