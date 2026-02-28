import type { DType } from "../backend/types";
import type { IRGraph, IRNode } from "./ir";

/**
 * Output descriptor for a fusion recipe.
 */
type FusionOutput = {
  nodeId: number;
  shape: number[];
  dtype: DType;
};

/**
 * A fusion recipe describes a group of operations that can be fused into a single kernel.
 * Supports single or multi-output fusion (§15.2).
 */
export type FusionRecipe = {
  id: number;
  kind: "elementwise";
  nodeIds: number[];
  inputs: number[];
  /** Output node IDs (for backwards compatibility) */
  outputs: number[];
  /** Detailed output descriptors with shape/dtype per output */
  outputDescriptors: FusionOutput[];
};

function requireNode(nodeById: Map<number, IRNode>, id: number): IRNode {
  const node = nodeById.get(id);
  if (!node) {
    throw new Error(`fusion recipe missing node ${id}`);
  }
  return node;
}

function requireShape(node: IRNode): number[] {
  if (!node.shape) {
    throw new Error(`fusion recipe missing shape for node ${node.id}`);
  }
  return node.shape;
}

function requireDType(node: IRNode): DType {
  if (!node.dtype) {
    throw new Error(`fusion recipe missing dtype for node ${node.id}`);
  }
  return node.dtype;
}

function buildConsumerMap(nodes: IRNode[]): Map<number, Set<number>> {
  const consumers = new Map<number, Set<number>>();
  for (const node of nodes) {
    for (const input of node.inputs) {
      const entry = consumers.get(input);
      if (entry) {
        entry.add(node.id);
      } else {
        consumers.set(input, new Set([node.id]));
      }
    }
  }
  return consumers;
}

export function buildFusionRecipes(graph: IRGraph): FusionRecipe[] {
  const nodeById = new Map<number, IRNode>();
  for (const node of graph.nodes) {
    nodeById.set(node.id, node);
  }
  const consumers = buildConsumerMap(graph.nodes);

  const recipes: FusionRecipe[] = [];
  for (const group of graph.fusionGroups) {
    const nodeSet = new Set(group.nodeIds);
    const inputs: number[] = [];
    const outputs: number[] = [];

    const registerInput = (id: number) => {
      if (!inputs.includes(id)) {
        inputs.push(id);
      }
    };

    for (const nodeId of group.nodeIds) {
      const node = requireNode(nodeById, nodeId);
      if (node.inputs.length === 0) {
        registerInput(nodeId);
        continue;
      }
      for (const input of node.inputs) {
        if (!nodeSet.has(input)) {
          registerInput(input);
        }
      }
    }

    for (const nodeId of group.nodeIds) {
      const userSet = consumers.get(nodeId);
      if (!userSet) {
        outputs.push(nodeId);
        continue;
      }
      let usedInside = false;
      for (const user of userSet) {
        if (nodeSet.has(user)) {
          usedInside = true;
          break;
        }
      }
      if (!usedInside) {
        outputs.push(nodeId);
      }
    }

    // Multi-output fusion (§15.2): allow multiple outputs
    if (outputs.length === 0) {
      throw new Error("fusion recipe has no outputs");
    }

    for (const inputId of inputs) {
      const input = requireNode(nodeById, inputId);
      requireShape(input);
      requireDType(input);
    }

    // Build output descriptors for each output
    const outputDescriptors: FusionOutput[] = [];
    for (const outputId of outputs) {
      const output = requireNode(nodeById, outputId);
      const shape = requireShape(output);
      const dtype = requireDType(output);
      outputDescriptors.push({
        nodeId: outputId,
        shape: shape.slice(),
        dtype,
      });
    }

    // Validate all outputs have the same shape (required for elementwise fusion)
    const firstShape = outputDescriptors[0].shape;
    for (let i = 1; i < outputDescriptors.length; i++) {
      const otherShape = outputDescriptors[i].shape;
      if (
        firstShape.length !== otherShape.length ||
        !firstShape.every((d, j) => d === otherShape[j])
      ) {
        throw new Error(
          `multi-output fusion requires same shape for all outputs, got ${JSON.stringify(firstShape)} and ${JSON.stringify(otherShape)}`,
        );
      }
    }

    recipes.push({
      id: recipes.length,
      kind: "elementwise",
      nodeIds: group.nodeIds.slice(),
      inputs,
      outputs,
      outputDescriptors,
    });
  }

  return recipes;
}
