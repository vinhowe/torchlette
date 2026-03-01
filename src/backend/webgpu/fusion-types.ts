/**
 * Fusion Recipe Types and Utilities
 *
 * Types and helper functions shared between fusion-tile-ir.ts (codegen)
 * and fusion-dispatch.ts (execution). Also used by engine/fusion-detect.ts
 * (recipe construction from IR graphs).
 */

import type { DType } from "../types";
import type { IRNode } from "../../engine/ir";
import { sizeOf } from "../../core/shape";
import { MAX_WORKGROUPS_PER_DIM } from "./shape-utils";
import { canVectorize as canVectorizeOp } from "./ops/registry";

// ============================================================================
// Fusion Recipe Types
// ============================================================================

/** A node in the fused computation graph. */
export interface FusedNode {
  id: number;
  op: string;
  inputs: number[]; // Node IDs or external input indices (negative)
  shape: number[];
  dtype: DType;
  isOutput?: boolean;
}

/** External input to the fused kernel. */
export interface FusedInput {
  id: number; // Original node ID
  index: number; // Binding index in the kernel
  shape: number[];
  dtype: DType;
  isScalar?: boolean; // If true, broadcast from single value
  isInlinedConstant?: boolean; // If true, skip storage binding and use literal
  inlinedValue?: number; // The constant value to inline as WGSL literal
}

/** Output descriptor for a fused kernel. */
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
// Vectorization (§15.3)
// ============================================================================

/** Vector width for memory coalescing. */
export type VectorWidth = 1 | 2 | 4;

/**
 * Determine the best vector width for a given shape and dtype.
 */
export function selectVectorWidth(
  shape: number[],
  dtype: DType,
): VectorWidth {
  if (dtype !== "f32" && dtype !== "f16") return 1;
  if (shape.length === 0) return 1;

  const innerDim = shape[shape.length - 1];
  const totalElements = sizeOf(shape);
  if (totalElements < 16) return 1;

  if (innerDim >= 4 && innerDim % 4 === 0) return 4;
  if (innerDim >= 2 && innerDim % 2 === 0) return 2;
  return 1;
}

/**
 * Check if all inputs are compatible with vectorization at the given width.
 */
export function canVectorize(
  outputShape: number[],
  inputs: { shape: number[]; dtype: DType }[],
  vectorWidth: VectorWidth,
): boolean {
  if (vectorWidth === 1) return true;
  if (outputShape.length === 0) return false;
  const outputInner = outputShape[outputShape.length - 1];
  if (outputInner % vectorWidth !== 0) return false;

  for (const input of inputs) {
    const inputSize = sizeOf(input.shape);
    if (inputSize === 1) continue;
    if (input.shape.length === 0) return false;
    const inputInner = input.shape[input.shape.length - 1];
    if (inputInner !== outputInner && inputInner !== 1) return false;
    if (inputInner === outputInner && inputInner % vectorWidth !== 0) return false;
  }
  return true;
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
// Kernel Generation Options & Result
// ============================================================================

/** Options for kernel generation. */
export interface KernelGenOptions {
  /** Workgroup size (default: 256) */
  workgroupSize?: number;
  /** Enable vectorized loads/stores (auto-selects best width if true) */
  vectorize?: boolean;
  /** Force specific vector width (overrides auto-selection) */
  forceVectorWidth?: VectorWidth;
}

/** Generated kernel result. */
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
 * Generate a cache key for a fusion recipe.
 * Supports multi-output fusion (§15.2).
 */
export function generateKernelCacheKey(
  recipe: FusedKernelRecipe,
  vectorWidth: VectorWidth = 1,
): string {
  const parts: string[] = [`vec:${vectorWidth}`];

  for (const output of recipe.outputs) {
    parts.push(`out${output.index}:${output.shape.join("x")}:${output.dtype}`);
  }

  for (const input of recipe.inputs) {
    if (input.isInlinedConstant) {
      parts.push(`in${input.index}:const=${input.inlinedValue}`);
    } else {
      parts.push(`in${input.index}:${input.shape.join("x")}:${input.dtype}`);
    }
  }

  const idToPos = new Map<number, number>();
  for (let i = 0; i < recipe.nodes.length; i++) {
    idToPos.set(recipe.nodes[i].id, i);
  }

  const opParts: string[] = [];
  for (let i = 0; i < recipe.nodes.length; i++) {
    const n = recipe.nodes[i];
    const inputRefs = n.inputs.map(inp => {
      const pos = idToPos.get(inp);
      return pos !== undefined ? `n${pos}` : `e${inp}`;
    }).join(",");
    opParts.push(`${n.op}(${inputRefs})${n.isOutput ? "!" : ""}`);
  }
  parts.push(`ops:[${opParts.join(";")}]`);

  return parts.join("|");
}

/**
 * Cheaply compute the cache key and metadata for a fusion recipe WITHOUT
 * generating the full WGSL source.
 */
export function computeKernelMeta(
  recipe: FusedKernelRecipe,
  options: KernelGenOptions = {},
): { cacheKey: string; vectorWidth: VectorWidth; workItems: number; workgroupSize: number; gridSizeX: number } {
  const workgroupSize = options.workgroupSize ?? recipe.workgroupSize ?? 256;
  const outputShape = recipe.outputs[0].shape;
  const primaryDtype = recipe.outputs[0].dtype;
  const totalElements = sizeOf(outputShape);

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
  const baseCacheKey = generateKernelCacheKey(recipe, vectorWidth);
  const cacheKey = gridSizeX >= MAX_WORKGROUPS_PER_DIM ? baseCacheKey + ":2d" : baseCacheKey;
  return { cacheKey, vectorWidth, workItems, workgroupSize, gridSizeX };
}

// ============================================================================
// Recipe Building from IR
// ============================================================================

/**
 * Build a fusion recipe from IR nodes.
 * Supports multi-output fusion (§15.2).
 */
export function buildRecipeFromIR(
  nodeIds: number[],
  nodeById: Map<number, IRNode>,
  inputNodeIds: number[],
  outputNodeIds?: number[],
): FusedKernelRecipe {
  const nodeSet = new Set(nodeIds);

  const inputMap = new Map<number, number>();
  const inputs: FusedInput[] = [];
  for (let i = 0; i < inputNodeIds.length; i++) {
    const nodeId = inputNodeIds[i];
    const node = nodeById.get(nodeId);
    inputMap.set(nodeId, -(i + 1));
    inputs.push({
      id: nodeId,
      index: i,
      shape: node?.shape ?? [1],
      dtype: node?.dtype ?? "f32",
    });
  }

  const outputIds = outputNodeIds ?? [nodeIds[nodeIds.length - 1]];
  const outputSet = new Set(outputIds);

  const nodes: FusedNode[] = [];
  for (const nodeId of nodeIds) {
    const node = nodeById.get(nodeId);
    if (!node) {
      throw new Error(`Node ${nodeId} not found`);
    }

    const mappedInputs: number[] = [];
    for (const inputId of node.inputs) {
      if (nodeSet.has(inputId)) {
        mappedInputs.push(inputId);
      } else if (inputMap.has(inputId)) {
        mappedInputs.push(inputMap.get(inputId)!);
      } else {
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
