import type { DType } from "../backend/types";
import type { TraceEvent } from "./trace";

export type IRNode = {
  id: number;
  op: string;
  epoch: number;
  kind: "lazy_op";
  inputs: number[];
  shape?: number[];
  dtype?: DType;
  scalarValues?: number[];  // §8.2.1: scalar constants for this op's scalar inputs
};

export type IRFusionGroup = {
  id: number;
  kind: "elementwise";
  nodeIds: number[];
};

export type IRGraph = {
  epoch: number;
  nodes: IRNode[];
  fusionGroups: IRFusionGroup[];
};

const ELEMENTWISE_OPS = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "neg",
  "abs",
  "exp",
  "log",
  "relu",
  "sqrt",
]);

function isElementwise(op: string): boolean {
  return ELEMENTWISE_OPS.has(op);
}

function broadcastShapes(a: number[], b: number[]): number[] {
  const outRank = Math.max(a.length, b.length);
  const out = new Array<number>(outRank);
  for (let i = 0; i < outRank; i += 1) {
    const aDim = a[a.length - 1 - i] ?? 1;
    const bDim = b[b.length - 1 - i] ?? 1;
    if (aDim !== bDim && aDim !== 1 && bDim !== 1) {
      throw new Error("compile ir shape mismatch: not broadcastable");
    }
    out[outRank - 1 - i] = Math.max(aDim, bDim);
  }
  return out;
}

function inferShape(op: string, inputs: IRNode[]): number[] {
  if (inputs.length === 0) {
    throw new Error(`compile ir cannot infer shape for op ${op}`);
  }
  const first = inputs[0].shape;
  if (!first) {
    throw new Error(`compile ir missing input shape for op ${op}`);
  }
  if (inputs.length === 1) {
    return first.slice();
  }
  let shape = first.slice();
  for (let i = 1; i < inputs.length; i += 1) {
    const next = inputs[i].shape;
    if (!next) {
      throw new Error(`compile ir missing input shape for op ${op}`);
    }
    shape = broadcastShapes(shape, next);
  }
  return shape;
}

function inferDType(op: string, inputs: IRNode[]): DType {
  if (inputs.length === 0) {
    throw new Error(`compile ir cannot infer dtype for op ${op}`);
  }
  const dtype = inputs[0].dtype;
  if (!dtype) {
    throw new Error(`compile ir missing input dtype for op ${op}`);
  }
  for (const input of inputs) {
    if (!input.dtype) {
      throw new Error(`compile ir missing input dtype for op ${op}`);
    }
    if (input.dtype !== dtype) {
      throw new Error(`compile ir dtype mismatch for op ${op}`);
    }
  }
  return dtype;
}

export function buildIRFromTrace(events: TraceEvent[], epoch: number): IRGraph {
  const nodes: IRNode[] = [];
  const nodeById = new Map<number, IRNode>();
  for (const event of events) {
    if (event.type !== "lazy_op" || event.epoch !== epoch) {
      continue;
    }
    const node: IRNode = {
      id: event.traceId,
      op: event.op,
      epoch: event.epoch,
      kind: "lazy_op",
      inputs: event.inputs ? event.inputs.slice() : [],
      shape: event.shape ? event.shape.slice() : undefined,
      dtype: event.dtype,
      scalarValues: event.scalarValues ? event.scalarValues.slice() : undefined,
    };
    nodes.push(node);
    nodeById.set(node.id, node);
  }

  for (const node of nodes) {
    if (!isElementwise(node.op)) {
      continue;
    }
    if (node.inputs.length === 0) {
      continue;
    }
    const inputNodes = node.inputs.map((inputId) => {
      const input = nodeById.get(inputId);
      if (!input) {
        throw new Error(`compile ir missing input node ${inputId}`);
      }
      return input;
    });
    if (!node.shape) {
      node.shape = inferShape(node.op, inputNodes);
    }
    if (!node.dtype) {
      node.dtype = inferDType(node.op, inputNodes);
    }
  }

  const fusionGroups: IRFusionGroup[] = [];
  let current: number[] = [];
  for (const node of nodes) {
    if (isElementwise(node.op)) {
      current.push(node.id);
      continue;
    }
    if (current.length > 1) {
      fusionGroups.push({
        id: fusionGroups.length,
        kind: "elementwise",
        nodeIds: current,
      });
    }
    current = [];
  }
  if (current.length > 1) {
    fusionGroups.push({
      id: fusionGroups.length,
      kind: "elementwise",
      nodeIds: current,
    });
  }

  return {
    epoch,
    nodes,
    fusionGroups,
  };
}
