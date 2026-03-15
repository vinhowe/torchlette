import { beforeEach, describe, expect, it } from "vitest";
import type { LazyIRNode, LazyRef } from "../src/engine/lazy-types";
import { detectRowPrograms } from "../src/engine/row-program-detect";

/** Create a minimal LazyIRNode for testing. */
let nextId = 1;
function makeNode(
  op: string,
  inputs: LazyRef[],
  shape: number[],
  dtype: "f32" | "f16" = "f32",
  payload?: unknown,
): LazyIRNode {
  return {
    id: nextId++,
    op: op as LazyIRNode["op"],
    inputs,
    shape,
    dtype,
    device: "webgpu",
    payload,
  } as LazyIRNode;
}

function pending(node: LazyIRNode): LazyRef {
  return { kind: "pending", node };
}

function buildConsumerMaps(nodes: LazyIRNode[]) {
  const consumers = new Map<number, LazyIRNode[]>();
  const consumerCount = new Map<number, number>();
  for (const node of nodes) {
    for (const ref of node.inputs) {
      if (ref.kind === "pending") {
        const id = ref.node.id;
        consumerCount.set(id, (consumerCount.get(id) ?? 0) + 1);
        if (!consumers.has(id)) consumers.set(id, []);
        consumers.get(id)!.push(node);
      }
    }
  }
  return { consumers, consumerCount };
}

describe("row-program-detect", () => {
  beforeEach(() => {
    nextId = 1;
  });

  it("detects softmax pattern (max → sub → exp → sum → div)", () => {
    // x is an external input (materialized), shape [4, 8]
    // softmax decomposes into: max(x, dim=-1) → sub(x, max) → exp → sum(exp, dim=-1) → div(exp, sum)
    const xNode = makeNode("tensorFromArray", [], [4, 8]);
    xNode.result = { id: 100, device: "webgpu", backendTensor: {} as any };

    const xRef: LazyRef = { kind: "pending", node: xNode };

    const maxNode = makeNode("max", [xRef], [4, 1], "f32", {
      dim: 1,
      keepdim: true,
    });
    const subNode = makeNode("sub", [xRef, pending(maxNode)], [4, 8]);
    const expNode = makeNode("exp", [pending(subNode)], [4, 8]);
    const sumNode = makeNode("sum", [pending(expNode)], [4, 1], "f32", {
      dim: 1,
      keepdim: true,
    });
    const divNode = makeNode(
      "div",
      [pending(expNode), pending(sumNode)],
      [4, 8],
    );

    // The plan: max, sub, exp, sum, div (xNode is already materialized)
    const planNodes = [maxNode, subNode, expNode, sumNode, divNode];
    const { consumers, consumerCount } = buildConsumerMaps(planNodes);

    const matches = detectRowPrograms(
      planNodes,
      consumerCount,
      consumers,
      undefined, // no external node IDs
      undefined, // no already claimed
    );

    expect(matches.length).toBe(1);
    const match = matches[0];
    expect(match.coveredNodeIds).toContain(maxNode.id);
    expect(match.coveredNodeIds).toContain(subNode.id);
    expect(match.coveredNodeIds).toContain(expNode.id);
    expect(match.coveredNodeIds).toContain(sumNode.id);
    expect(match.coveredNodeIds).toContain(divNode.id);
    expect(match.outputNodeId).toBe(divNode.id);
    expect(match.dim).toBe(1);

    // Should have 2 reduce phases (max, sum) + 1 write phase
    const { program } = match;
    expect(program.phases.length).toBe(3);
    expect(program.phases[0].kind).toBe("reduce");
    expect((program.phases[0] as any).reduceOp).toBe("max");
    expect(program.phases[1].kind).toBe("reduce");
    expect((program.phases[1] as any).reduceOp).toBe("sum");
    expect(program.phases[2].kind).toBe("write");

    // Should have 1 input (x)
    expect(program.inputs.length).toBe(1);
    expect(program.output.dtype).toBe("f32");
  });

  it("rejects single-reduction patterns (needs ≥2)", () => {
    const xNode = makeNode("tensorFromArray", [], [4, 8]);
    xNode.result = { id: 100, device: "webgpu", backendTensor: {} as any };
    const xRef: LazyRef = { kind: "pending", node: xNode };

    const sumNode = makeNode("sum", [xRef], [4, 1], "f32", {
      dim: 1,
      keepdim: true,
    });

    const planNodes = [sumNode];
    const { consumers, consumerCount } = buildConsumerMaps(planNodes);

    const matches = detectRowPrograms(planNodes, consumerCount, consumers);
    expect(matches.length).toBe(0);
  });

  it("rejects non-last-dim reductions", () => {
    const xNode = makeNode("tensorFromArray", [], [4, 8]);
    xNode.result = { id: 100, device: "webgpu", backendTensor: {} as any };
    const xRef: LazyRef = { kind: "pending", node: xNode };

    // Reduce along dim=0, not last dim
    const maxNode = makeNode("max", [xRef], [1, 8], "f32", {
      dim: 0,
      keepdim: true,
    });
    const sumNode = makeNode("sum", [pending(maxNode)], [1, 1], "f32", {
      dim: 1,
      keepdim: true,
    });

    const planNodes = [maxNode, sumNode];
    const { consumers, consumerCount } = buildConsumerMaps(planNodes);

    const matches = detectRowPrograms(planNodes, consumerCount, consumers);
    expect(matches.length).toBe(0);
  });

  it("detects a novel pattern: mean → sub → square → mean + rsqrt (inv_std)", () => {
    // This is a pattern compound-patterns.ts CANNOT handle:
    // two reductions feeding into a per-element output.
    // Pattern: mean(x) → sub(x, mean) → square → mean → add(eps) → rsqrt → mul(sub, rsqrt)
    const xNode = makeNode("tensorFromArray", [], [4, 8]);
    xNode.result = { id: 100, device: "webgpu", backendTensor: {} as any };
    const xRef: LazyRef = { kind: "pending", node: xNode };
    const epsRef: LazyRef = { kind: "scalar", value: 1e-5, dtype: "f32" };

    const mean1 = makeNode("mean", [xRef], [4, 1], "f32", {
      dim: 1,
      keepdim: true,
    });
    const sub1 = makeNode("sub", [xRef, pending(mean1)], [4, 8]);
    const sq = makeNode("mul", [pending(sub1), pending(sub1)], [4, 8]);
    const mean2 = makeNode("mean", [pending(sq)], [4, 1], "f32", {
      dim: 1,
      keepdim: true,
    });
    const addEps = makeNode("add", [pending(mean2), epsRef], [4, 1]);
    const rsqrtNode = makeNode("rsqrt", [pending(addEps)], [4, 1]);
    const mulOut = makeNode("mul", [pending(sub1), pending(rsqrtNode)], [4, 8]);

    const planNodes = [mean1, sub1, sq, mean2, addEps, rsqrtNode, mulOut];
    const { consumers, consumerCount } = buildConsumerMaps(planNodes);

    const matches = detectRowPrograms(planNodes, consumerCount, consumers);
    expect(matches.length).toBe(1);

    const { program } = matches[0];
    const reducePhases = program.phases.filter((p) => p.kind === "reduce");
    expect(reducePhases.length).toBe(2);
    expect(program.phases[program.phases.length - 1].kind).toBe("write");
    expect(program.inputs.length).toBe(1);
    expect(matches[0].outputNodeId).toBe(mulOut.id);
  });
});
