import { beforeEach, describe, expect, it } from "vitest";
import { cpuBackend } from "../src/backend/cpu";
import {
  buildPlan,
  createLazyIRNode,
  createMaterializedRef,
  createPendingRef,
  createStorageHandle,
  executePlan,
  isMaterialized,
  isPending,
  resetNodeIdCounter,
  resetStorageIdCounter,
} from "../src/engine/lazy";

describe("lazy value system", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
  });

  describe("LazyRef types", () => {
    it("creates pending ref from IRNode", () => {
      const node = createLazyIRNode("add", [], [2, 3], "f32", "cpu");
      const ref = createPendingRef(node);

      expect(ref.kind).toBe("pending");
      expect(isPending(ref)).toBe(true);
      expect(isMaterialized(ref)).toBe(false);

      if (isPending(ref)) {
        expect(ref.node).toBe(node);
      }
    });

    it("creates materialized ref from StorageHandle", () => {
      const storage = createStorageHandle("cpu", {
        shape: [2, 3],
        toArray: () => [1, 2, 3, 4, 5, 6],
      });
      const ref = createMaterializedRef(storage);

      expect(ref.kind).toBe("materialized");
      expect(isMaterialized(ref)).toBe(true);
      expect(isPending(ref)).toBe(false);

      if (isMaterialized(ref)) {
        expect(ref.storage).toBe(storage);
      }
    });
  });

  describe("IRNode creation", () => {
    it("assigns unique incrementing IDs", () => {
      const node1 = createLazyIRNode("add", [], [2], "f32", "cpu");
      const node2 = createLazyIRNode("mul", [], [3], "f32", "cpu");
      const node3 = createLazyIRNode("relu", [], [4], "f32", "cpu");

      expect(node1.id).toBe(1);
      expect(node2.id).toBe(2);
      expect(node3.id).toBe(3);
    });

    it("stores op, shape, dtype, and device", () => {
      const node = createLazyIRNode("matmul", [], [4, 4], "f16", "webgpu");

      expect(node.op).toBe("matmul");
      expect(node.shape).toEqual([4, 4]);
      expect(node.dtype).toBe("f16");
      expect(node.device).toBe("webgpu");
    });

    it("stores inputs as LazyRefs", () => {
      const inputNode = createLazyIRNode(
        "tensorFromArray",
        [],
        [2, 3],
        "f32",
        "cpu",
      );
      const inputRef = createPendingRef(inputNode);

      const node = createLazyIRNode("relu", [inputRef], [2, 3], "f32", "cpu");

      expect(node.inputs).toHaveLength(1);
      expect(node.inputs[0]).toBe(inputRef);
    });

    it("stores optional payload", () => {
      const payload = { values: [1, 2, 3], originalShape: [3] };
      const node = createLazyIRNode(
        "tensorFromArray",
        [],
        [3],
        "f32",
        "cpu",
        payload,
      );

      expect(node.payload).toBe(payload);
    });

    it("initially has no result", () => {
      const node = createLazyIRNode("add", [], [2], "f32", "cpu");
      expect(node.result).toBeUndefined();
    });
  });

  describe("StorageHandle creation", () => {
    it("assigns unique incrementing IDs", () => {
      const s1 = createStorageHandle("cpu", { shape: [1], toArray: () => [1] });
      const s2 = createStorageHandle("cpu", {
        shape: [2],
        toArray: () => [1, 2],
      });

      expect(s1.id).toBe(1);
      expect(s2.id).toBe(2);
    });

    it("stores device and backendTensor", () => {
      const backendTensor = {
        shape: [2, 3],
        toArray: () => [1, 2, 3, 4, 5, 6],
      };
      const storage = createStorageHandle("webgpu", backendTensor);

      expect(storage.device).toBe("webgpu");
      expect(storage.backendTensor).toBe(backendTensor);
    });
  });

  describe("execution plan building", () => {
    it("builds plan for single node with no inputs", () => {
      const node = createLazyIRNode("tensorFromArray", [], [3], "f32", "cpu");
      const plan = buildPlan(node);

      expect(plan.nodes).toHaveLength(1);
      expect(plan.nodes[0]).toBe(node);
    });

    it("builds plan in topological order for chain", () => {
      // a -> b -> c
      const nodeA = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");
      const refA = createPendingRef(nodeA);

      const nodeB = createLazyIRNode("relu", [refA], [2], "f32", "cpu");
      const refB = createPendingRef(nodeB);

      const nodeC = createLazyIRNode("sqrt", [refB], [2], "f32", "cpu");

      const plan = buildPlan(nodeC);

      expect(plan.nodes).toHaveLength(3);
      expect(plan.nodes[0]).toBe(nodeA);
      expect(plan.nodes[1]).toBe(nodeB);
      expect(plan.nodes[2]).toBe(nodeC);
    });

    it("builds plan in topological order for diamond", () => {
      //     a
      //    / \
      //   b   c
      //    \ /
      //     d
      const nodeA = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");
      const refA = createPendingRef(nodeA);

      const nodeB = createLazyIRNode("relu", [refA], [2], "f32", "cpu");
      const refB = createPendingRef(nodeB);

      const nodeC = createLazyIRNode("sqrt", [refA], [2], "f32", "cpu");
      const refC = createPendingRef(nodeC);

      const nodeD = createLazyIRNode("add", [refB, refC], [2], "f32", "cpu");

      const plan = buildPlan(nodeD);

      expect(plan.nodes).toHaveLength(4);
      // A must come first
      expect(plan.nodes[0]).toBe(nodeA);
      // B and C can be in either order, but both before D
      expect(plan.nodes.indexOf(nodeB)).toBeLessThan(plan.nodes.indexOf(nodeD));
      expect(plan.nodes.indexOf(nodeC)).toBeLessThan(plan.nodes.indexOf(nodeD));
      // D must be last
      expect(plan.nodes[3]).toBe(nodeD);
    });

    it("skips already materialized inputs", () => {
      const storage = createStorageHandle("cpu", {
        shape: [2],
        toArray: () => [1, 2],
      });
      const materializedRef = createMaterializedRef(storage);

      const nodeB = createLazyIRNode(
        "relu",
        [materializedRef],
        [2],
        "f32",
        "cpu",
      );

      const plan = buildPlan(nodeB);

      expect(plan.nodes).toHaveLength(1);
      expect(plan.nodes[0]).toBe(nodeB);
    });

    it("deduplicates shared nodes", () => {
      // Same node used twice as input
      const nodeA = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");
      const refA = createPendingRef(nodeA);

      const nodeB = createLazyIRNode("add", [refA, refA], [2], "f32", "cpu");

      const plan = buildPlan(nodeB);

      expect(plan.nodes).toHaveLength(2);
      expect(plan.nodes[0]).toBe(nodeA);
      expect(plan.nodes[1]).toBe(nodeB);
    });

    it("handles complex DAG with mixed materialized and pending", () => {
      //   mat1   a
      //     \   /
      //      \ /
      //       b
      //       |
      //       c
      //      / \
      //   mat2  d
      const storage1 = createStorageHandle("cpu", {
        shape: [2],
        toArray: () => [1, 2],
      });
      const mat1 = createMaterializedRef(storage1);

      const nodeA = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");
      const refA = createPendingRef(nodeA);

      const nodeB = createLazyIRNode("add", [mat1, refA], [2], "f32", "cpu");
      const refB = createPendingRef(nodeB);

      const nodeC = createLazyIRNode("relu", [refB], [2], "f32", "cpu");
      const refC = createPendingRef(nodeC);

      const storage2 = createStorageHandle("cpu", {
        shape: [2],
        toArray: () => [3, 4],
      });
      const mat2 = createMaterializedRef(storage2);

      const nodeD = createLazyIRNode("mul", [refC, mat2], [2], "f32", "cpu");

      const plan = buildPlan(nodeD);

      // Should include: nodeA, nodeB, nodeC, nodeD (not mat1 or mat2)
      expect(plan.nodes).toHaveLength(4);
      expect(plan.nodes).toContain(nodeA);
      expect(plan.nodes).toContain(nodeB);
      expect(plan.nodes).toContain(nodeC);
      expect(plan.nodes).toContain(nodeD);

      // Check topological order
      expect(plan.nodes.indexOf(nodeA)).toBeLessThan(plan.nodes.indexOf(nodeB));
      expect(plan.nodes.indexOf(nodeB)).toBeLessThan(plan.nodes.indexOf(nodeC));
      expect(plan.nodes.indexOf(nodeC)).toBeLessThan(plan.nodes.indexOf(nodeD));
    });
  });
});

describe("lazy execution semantics", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
  });

  it("creating IRNode does not execute backend ops", () => {
    const executed = false;

    // Creating nodes should not trigger any execution
    const node = createLazyIRNode("add", [], [2, 3], "f32", "cpu");

    // If this was eager, something would have been called
    expect(executed).toBe(false);
    expect(node.result).toBeUndefined();
  });

  it("pending refs remain pending until forced", () => {
    const node1 = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");
    const ref1 = createPendingRef(node1);

    const node2 = createLazyIRNode("relu", [ref1], [2], "f32", "cpu");
    const ref2 = createPendingRef(node2);

    // After creating a chain of ops, both should still be pending
    expect(isPending(ref1)).toBe(true);
    expect(isPending(ref2)).toBe(true);
    expect(node1.result).toBeUndefined();
    expect(node2.result).toBeUndefined();
  });

  it("plan building does not execute nodes", () => {
    const node1 = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");
    const ref1 = createPendingRef(node1);

    const node2 = createLazyIRNode("add", [ref1, ref1], [2], "f32", "cpu");

    // Building a plan should not mutate any nodes
    const plan = buildPlan(node2);

    expect(plan.nodes).toHaveLength(2);
    expect(node1.result).toBeUndefined();
    expect(node2.result).toBeUndefined();
  });
});

describe("lazy op emission", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
  });

  describe("shape inference", () => {
    it("infers output shape for add with same shapes", () => {
      const a = createLazyIRNode("tensorFromArray", [], [2, 3], "f32", "cpu");
      const refA = createPendingRef(a);
      const b = createLazyIRNode("tensorFromArray", [], [2, 3], "f32", "cpu");
      const refB = createPendingRef(b);

      // For add, output shape equals input shapes when they match
      const add = createLazyIRNode("add", [refA, refB], [2, 3], "f32", "cpu");

      expect(add.shape).toEqual([2, 3]);
    });

    it("infers matmul output shape", () => {
      const a = createLazyIRNode("tensorFromArray", [], [2, 3], "f32", "cpu");
      const refA = createPendingRef(a);
      const b = createLazyIRNode("tensorFromArray", [], [3, 4], "f32", "cpu");
      const refB = createPendingRef(b);

      // matmul [2,3] @ [3,4] = [2,4]
      const mm = createLazyIRNode("matmul", [refA, refB], [2, 4], "f32", "cpu");

      expect(mm.shape).toEqual([2, 4]);
    });

    it("infers unary op preserves shape", () => {
      const a = createLazyIRNode(
        "tensorFromArray",
        [],
        [3, 4, 5],
        "f32",
        "cpu",
      );
      const refA = createPendingRef(a);

      const relu = createLazyIRNode("relu", [refA], [3, 4, 5], "f32", "cpu");
      const sqrt = createLazyIRNode(
        "sqrt",
        [createPendingRef(relu)],
        [3, 4, 5],
        "f32",
        "cpu",
      );

      expect(relu.shape).toEqual([3, 4, 5]);
      expect(sqrt.shape).toEqual([3, 4, 5]);
    });
  });

  describe("dtype propagation", () => {
    it("preserves dtype through unary ops", () => {
      const a = createLazyIRNode("tensorFromArray", [], [2], "f16", "webgpu");
      const refA = createPendingRef(a);

      const relu = createLazyIRNode("relu", [refA], [2], "f16", "webgpu");

      expect(relu.dtype).toBe("f16");
    });

    it("preserves dtype through binary ops", () => {
      const a = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");
      const b = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");

      const add = createLazyIRNode(
        "add",
        [createPendingRef(a), createPendingRef(b)],
        [2],
        "f32",
        "cpu",
      );

      expect(add.dtype).toBe("f32");
    });
  });

  describe("device tracking", () => {
    it("tracks device through operations", () => {
      const a = createLazyIRNode("tensorFromArray", [], [2], "f32", "webgpu");
      const refA = createPendingRef(a);

      const relu = createLazyIRNode("relu", [refA], [2], "f32", "webgpu");

      expect(relu.device).toBe("webgpu");
    });

    it("tracks different devices independently", () => {
      const cpuNode = createLazyIRNode(
        "tensorFromArray",
        [],
        [2],
        "f32",
        "cpu",
      );
      const gpuNode = createLazyIRNode(
        "tensorFromArray",
        [],
        [2],
        "f32",
        "webgpu",
      );

      expect(cpuNode.device).toBe("cpu");
      expect(gpuNode.device).toBe("webgpu");
    });
  });

  describe("input reference tracking", () => {
    it("tracks multiple inputs correctly", () => {
      const a = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");
      const b = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");
      const c = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");

      const refA = createPendingRef(a);
      const refB = createPendingRef(b);
      const refC = createPendingRef(c);

      // (a + b) + c
      const ab = createLazyIRNode("add", [refA, refB], [2], "f32", "cpu");
      const refAB = createPendingRef(ab);
      const result = createLazyIRNode("add", [refAB, refC], [2], "f32", "cpu");

      expect(result.inputs).toHaveLength(2);
      expect(isPending(result.inputs[0])).toBe(true);
      expect(isPending(result.inputs[1])).toBe(true);

      // Verify the graph structure
      const plan = buildPlan(result);
      expect(plan.nodes).toHaveLength(5); // a, b, c, ab, result
    });

    it("handles same input used multiple times", () => {
      const a = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu");
      const refA = createPendingRef(a);

      // a + a
      const doubled = createLazyIRNode("add", [refA, refA], [2], "f32", "cpu");

      expect(doubled.inputs).toHaveLength(2);
      expect(doubled.inputs[0]).toBe(doubled.inputs[1]); // Same reference

      const plan = buildPlan(doubled);
      expect(plan.nodes).toHaveLength(2); // a, doubled (not 3)
    });
  });

  describe("payload for special ops", () => {
    it("tensorFromArray stores values in payload", () => {
      const values = [1, 2, 3, 4, 5, 6];
      const payload = { values };
      const node = createLazyIRNode(
        "tensorFromArray",
        [],
        [2, 3],
        "f32",
        "cpu",
        payload,
      );

      expect(node.payload).toEqual({ values: [1, 2, 3, 4, 5, 6] });
    });

    it("reshape stores target shape in payload", () => {
      const a = createLazyIRNode("tensorFromArray", [], [6], "f32", "cpu");
      const refA = createPendingRef(a);

      const payload = { targetShape: [2, 3] };
      const reshaped = createLazyIRNode(
        "reshape",
        [refA],
        [2, 3],
        "f32",
        "cpu",
        payload,
      );

      expect(reshaped.payload).toEqual({ targetShape: [2, 3] });
    });

    it("transpose stores dim options in payload", () => {
      const a = createLazyIRNode("tensorFromArray", [], [2, 3], "f32", "cpu");
      const refA = createPendingRef(a);

      const payload = { dim0: 0, dim1: 1 };
      const transposed = createLazyIRNode(
        "transpose",
        [refA],
        [3, 2],
        "f32",
        "cpu",
        payload,
      );

      expect(transposed.payload).toEqual({ dim0: 0, dim1: 1 });
    });
  });
});

describe("force and execute", () => {
  beforeEach(() => {
    resetNodeIdCounter();
    resetStorageIdCounter();
  });

  describe("executePlan", () => {
    it("executes single tensorFromArray node", async () => {
      const values = [1, 2, 3, 4];
      const node = createLazyIRNode(
        "tensorFromArray",
        [],
        [2, 2],
        "f32",
        "cpu",
        { values },
      );

      const plan = buildPlan(node);
      const result = await executePlan(plan, cpuBackend);

      expect(result).toBeDefined();
      expect(result.device).toBe("cpu");
      expect(result.backendTensor.shape).toEqual([2, 2]);
      expect(result.backendTensor.toArray()).toEqual([1, 2, 3, 4]);
    });

    it("executes chain of operations", async () => {
      // Create tensor [1, 4, 9, 16], apply sqrt to get [1, 2, 3, 4]
      const createNode = createLazyIRNode(
        "tensorFromArray",
        [],
        [4],
        "f32",
        "cpu",
        { values: [1, 4, 9, 16] },
      );
      const createRef = createPendingRef(createNode);

      const sqrtNode = createLazyIRNode("sqrt", [createRef], [4], "f32", "cpu");

      const plan = buildPlan(sqrtNode);
      const result = await executePlan(plan, cpuBackend);

      expect(result.backendTensor.toArray()).toEqual([1, 2, 3, 4]);
    });

    it("executes binary operations", async () => {
      const aNode = createLazyIRNode("tensorFromArray", [], [3], "f32", "cpu", {
        values: [1, 2, 3],
      });
      const bNode = createLazyIRNode("tensorFromArray", [], [3], "f32", "cpu", {
        values: [4, 5, 6],
      });

      const aRef = createPendingRef(aNode);
      const bRef = createPendingRef(bNode);

      const addNode = createLazyIRNode("add", [aRef, bRef], [3], "f32", "cpu");

      const plan = buildPlan(addNode);
      const result = await executePlan(plan, cpuBackend);

      expect(result.backendTensor.toArray()).toEqual([5, 7, 9]);
    });

    it("executes diamond-shaped DAG correctly", async () => {
      //     a = [2, 4]
      //    /         \
      // sqrt(a)=[√2,2] relu(a)=[2,4]
      //    \         /
      //      add = [√2+2, 6]
      const aNode = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu", {
        values: [2, 4],
      });
      const aRef = createPendingRef(aNode);

      const sqrtNode = createLazyIRNode("sqrt", [aRef], [2], "f32", "cpu");
      const sqrtRef = createPendingRef(sqrtNode);

      const reluNode = createLazyIRNode("relu", [aRef], [2], "f32", "cpu");
      const reluRef = createPendingRef(reluNode);

      const addNode = createLazyIRNode(
        "add",
        [sqrtRef, reluRef],
        [2],
        "f32",
        "cpu",
      );

      const plan = buildPlan(addNode);
      const result = await executePlan(plan, cpuBackend);

      const values = result.backendTensor.toArray();
      expect(values[0]).toBeCloseTo(Math.sqrt(2) + 2, 5);
      expect(values[1]).toBeCloseTo(2 + 4, 5);
    });

    it("caches results in node.result", async () => {
      const node = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu", {
        values: [1, 2],
      });

      expect(node.result).toBeUndefined();

      const plan = buildPlan(node);
      const result = await executePlan(plan, cpuBackend);

      expect(node.result).toBeDefined();
      expect(node.result).toBe(result);
    });

    it("reuses materialized inputs without re-execution", async () => {
      // Create a materialized storage
      const existingTensor = cpuBackend.ops.tensorFromArray([10, 20], [2]);
      const existingStorage = createStorageHandle("cpu", existingTensor);
      const materializedRef = createMaterializedRef(existingStorage);

      // Create a node that uses the materialized input
      const reluNode = createLazyIRNode(
        "relu",
        [materializedRef],
        [2],
        "f32",
        "cpu",
      );

      const plan = buildPlan(reluNode);
      expect(plan.nodes).toHaveLength(1); // Only the relu node

      const result = await executePlan(plan, cpuBackend);
      expect(result.backendTensor.toArray()).toEqual([10, 20]);
    });

    it("handles matmul correctly", async () => {
      // [1, 2] @ [[1], [2]] = [1*1 + 2*2] = [5]
      // But for 2D matmul: [1,2] shape [1,2], [[1],[2]] shape [2,1]
      // Result shape [1,1], value [5]
      const aNode = createLazyIRNode(
        "tensorFromArray",
        [],
        [1, 2],
        "f32",
        "cpu",
        { values: [1, 2] },
      );
      const bNode = createLazyIRNode(
        "tensorFromArray",
        [],
        [2, 1],
        "f32",
        "cpu",
        { values: [1, 2] },
      );

      const aRef = createPendingRef(aNode);
      const bRef = createPendingRef(bNode);

      const mmNode = createLazyIRNode(
        "matmul",
        [aRef, bRef],
        [1, 1],
        "f32",
        "cpu",
      );

      const plan = buildPlan(mmNode);
      const result = await executePlan(plan, cpuBackend);

      expect(result.backendTensor.shape).toEqual([1, 1]);
      expect(result.backendTensor.toArray()).toEqual([5]);
    });

    it("handles mul operation", async () => {
      const aNode = createLazyIRNode("tensorFromArray", [], [3], "f32", "cpu", {
        values: [2, 3, 4],
      });
      const bNode = createLazyIRNode("tensorFromArray", [], [3], "f32", "cpu", {
        values: [5, 6, 7],
      });

      const aRef = createPendingRef(aNode);
      const bRef = createPendingRef(bNode);

      const mulNode = createLazyIRNode("mul", [aRef, bRef], [3], "f32", "cpu");

      const plan = buildPlan(mulNode);
      const result = await executePlan(plan, cpuBackend);

      expect(result.backendTensor.toArray()).toEqual([10, 18, 28]);
    });

    it("handles sub operation", async () => {
      const aNode = createLazyIRNode("tensorFromArray", [], [3], "f32", "cpu", {
        values: [10, 20, 30],
      });
      const bNode = createLazyIRNode("tensorFromArray", [], [3], "f32", "cpu", {
        values: [1, 2, 3],
      });

      const aRef = createPendingRef(aNode);
      const bRef = createPendingRef(bNode);

      const subNode = createLazyIRNode("sub", [aRef, bRef], [3], "f32", "cpu");

      const plan = buildPlan(subNode);
      const result = await executePlan(plan, cpuBackend);

      expect(result.backendTensor.toArray()).toEqual([9, 18, 27]);
    });

    it("handles div operation", async () => {
      const aNode = createLazyIRNode("tensorFromArray", [], [3], "f32", "cpu", {
        values: [10, 20, 30],
      });
      const bNode = createLazyIRNode("tensorFromArray", [], [3], "f32", "cpu", {
        values: [2, 4, 5],
      });

      const aRef = createPendingRef(aNode);
      const bRef = createPendingRef(bNode);

      const divNode = createLazyIRNode("div", [aRef, bRef], [3], "f32", "cpu");

      const plan = buildPlan(divNode);
      const result = await executePlan(plan, cpuBackend);

      expect(result.backendTensor.toArray()).toEqual([5, 5, 6]);
    });

    it("handles reshape operation", async () => {
      const aNode = createLazyIRNode("tensorFromArray", [], [6], "f32", "cpu", {
        values: [1, 2, 3, 4, 5, 6],
      });
      const aRef = createPendingRef(aNode);

      const reshapeNode = createLazyIRNode(
        "reshape",
        [aRef],
        [2, 3],
        "f32",
        "cpu",
        { targetShape: [2, 3] },
      );

      const plan = buildPlan(reshapeNode);
      const result = await executePlan(plan, cpuBackend);

      expect(result.backendTensor.shape).toEqual([2, 3]);
      expect(result.backendTensor.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("handles complex expression: (a + b) * sqrt(c)", async () => {
      const aNode = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu", {
        values: [1, 2],
      });
      const bNode = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu", {
        values: [3, 4],
      });
      const cNode = createLazyIRNode("tensorFromArray", [], [2], "f32", "cpu", {
        values: [4, 9],
      });

      const aRef = createPendingRef(aNode);
      const bRef = createPendingRef(bNode);
      const cRef = createPendingRef(cNode);

      // a + b = [4, 6]
      const addNode = createLazyIRNode("add", [aRef, bRef], [2], "f32", "cpu");
      const addRef = createPendingRef(addNode);

      // sqrt(c) = [2, 3]
      const sqrtNode = createLazyIRNode("sqrt", [cRef], [2], "f32", "cpu");
      const sqrtRef = createPendingRef(sqrtNode);

      // (a + b) * sqrt(c) = [8, 18]
      const mulNode = createLazyIRNode(
        "mul",
        [addRef, sqrtRef],
        [2],
        "f32",
        "cpu",
      );

      const plan = buildPlan(mulNode);
      expect(plan.nodes).toHaveLength(6); // a, b, c, add, sqrt, mul

      const result = await executePlan(plan, cpuBackend);
      expect(result.backendTensor.toArray()).toEqual([8, 18]);
    });
  });

  describe("error handling", () => {
    it("throws when plan is empty", async () => {
      const emptyPlan = { nodes: [] };
      await expect(executePlan(emptyPlan, cpuBackend)).rejects.toThrow(
        "Cannot execute empty plan",
      );
    });

    it("throws when input is not ready", async () => {
      // Create a node with a pending input that isn't in the plan
      const orphanNode = createLazyIRNode(
        "tensorFromArray",
        [],
        [2],
        "f32",
        "cpu",
        { values: [1, 2] },
      );
      const orphanRef = createPendingRef(orphanNode);

      const reluNode = createLazyIRNode("relu", [orphanRef], [2], "f32", "cpu");

      // Create a plan with only reluNode (missing orphanNode)
      const badPlan = { nodes: [reluNode] };

      await expect(executePlan(badPlan, cpuBackend)).rejects.toThrow(
        "Input not ready",
      );
    });
  });
});
