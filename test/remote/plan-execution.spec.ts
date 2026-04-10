/**
 * Execution round-trip: deserialized plans execute to bit-exact outputs on
 * the CPU backend. Proves the wire format preserves not only structure but
 * the information the executor needs to run each op correctly.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { cpuBackend } from "../../src/backend/cpu";
import { registerBackend } from "../../src/backend/registry";
import { executePlanSequential } from "../../src/executor/sequential";
import { LazyIRNode, type ExecutionPlan, type LazyRef } from "../../src/graph/types";
import { deserializePlan, serializePlan } from "../../src/remote/serialize";
import type { HandleRef, SerializedPlan } from "../../src/remote/wire";

// ============================================================================
// Setup
// ============================================================================

beforeAll(() => {
  registerBackend(cpuBackend);
});

function makeNodeFactory() {
  let nextId = 1;
  return (
    op: LazyIRNode["op"],
    inputs: LazyRef[],
    shape: number[],
    payload?: unknown,
    dtype: LazyIRNode["dtype"] = "f32",
  ): LazyIRNode => {
    return new LazyIRNode(nextId++, op, inputs, shape, dtype, "cpu", payload);
  };
}

function pending(n: LazyIRNode): LazyRef {
  return { kind: "pending", node: n };
}

const resolveLocal = (id: number): HandleRef => `h:${id}`;

/** Serialize, JSON round-trip, deserialize. */
function cloneViaJson(plan: ExecutionPlan): ExecutionPlan {
  const wire = serializePlan(plan, { resolveHandle: resolveLocal });
  const parsed = JSON.parse(JSON.stringify(wire)) as SerializedPlan;
  return deserializePlan(parsed);
}

async function executeAndRead(plan: ExecutionPlan): Promise<number[]> {
  await executePlanSequential(plan, cpuBackend);
  const last = plan.nodes[plan.nodes.length - 1];
  if (!last.result) throw new Error("no result after execution");
  return cpuBackend.ops.read(last.result.backendTensor);
}

async function assertRoundTripProducesSameOutput(
  buildPlan: () => ExecutionPlan,
): Promise<void> {
  // Build two independent planes from the same construction function.
  const local = buildPlan();
  const cloned = cloneViaJson(buildPlan());

  const [localOut, clonedOut] = await Promise.all([
    executeAndRead(local),
    executeAndRead(cloned),
  ]);

  expect(clonedOut).toEqual(localOut);
}

// ============================================================================
// Cases
// ============================================================================

describe("remote wire format: bit-exact execution round-trip", () => {
  it("elementwise add", async () => {
    await assertRoundTripProducesSameOutput(() => {
      const node = makeNodeFactory();
      const a = node("tensorFromArray", [], [2, 2], {
        values: new Float32Array([1, 2, 3, 4]),
      });
      const b = node("tensorFromArray", [], [2, 2], {
        values: new Float32Array([10, 20, 30, 40]),
      });
      const add = node("add", [pending(a), pending(b)], [2, 2]);
      return { nodes: [a, b, add] };
    });
  });

  it("matmul 3x4 @ 4x2", async () => {
    await assertRoundTripProducesSameOutput(() => {
      const node = makeNodeFactory();
      const a = node("tensorFromArray", [], [3, 4], {
        values: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
      });
      const b = node("tensorFromArray", [], [4, 2], {
        values: new Float32Array([1, 0, 0, 1, 1, 0, 0, 1]),
      });
      const mm = node("matmul", [pending(a), pending(b)], [3, 2]);
      return { nodes: [a, b, mm] };
    });
  });

  it("sum(dim=1)", async () => {
    await assertRoundTripProducesSameOutput(() => {
      const node = makeNodeFactory();
      const a = node("tensorFromArray", [], [2, 3], {
        values: new Float32Array([1, 2, 3, 4, 5, 6]),
      });
      const s = node("sum", [pending(a)], [2], { dim: 1, keepdim: false });
      return { nodes: [a, s] };
    });
  });

  it("full reduction: sum(all)", async () => {
    await assertRoundTripProducesSameOutput(() => {
      const node = makeNodeFactory();
      const a = node("tensorFromArray", [], [2, 3], {
        values: new Float32Array([1, 2, 3, 4, 5, 6]),
      });
      const s = node("sum", [pending(a)], [], { dim: null, keepdim: false });
      return { nodes: [a, s] };
    });
  });

  it("scalar ref: x * 2.5", async () => {
    await assertRoundTripProducesSameOutput(() => {
      const node = makeNodeFactory();
      const a = node("tensorFromArray", [], [4], {
        values: new Float32Array([1, 2, 3, 4]),
      });
      const mul = node(
        "mul",
        [pending(a), { kind: "scalar", value: 2.5, dtype: "f32" }],
        [4],
      );
      return { nodes: [a, mul] };
    });
  });

  it("chain: add → mul → neg", async () => {
    await assertRoundTripProducesSameOutput(() => {
      const node = makeNodeFactory();
      const a = node("tensorFromArray", [], [3], {
        values: new Float32Array([1, 2, 3]),
      });
      const b = node("tensorFromArray", [], [3], {
        values: new Float32Array([10, 20, 30]),
      });
      const add = node("add", [pending(a), pending(b)], [3]);
      const mul = node(
        "mul",
        [pending(add), { kind: "scalar", value: 0.1, dtype: "f32" }],
        [3],
      );
      const n = node("neg", [pending(mul)], [3]);
      return { nodes: [a, b, add, mul, n] };
    });
  });

  it("nonlinear: relu(add(x, -1))", async () => {
    await assertRoundTripProducesSameOutput(() => {
      const node = makeNodeFactory();
      const x = node("tensorFromArray", [], [5], {
        values: new Float32Array([-2, -1, 0, 1, 2]),
      });
      const shifted = node(
        "add",
        [pending(x), { kind: "scalar", value: -1, dtype: "f32" }],
        [5],
      );
      const r = node("relu", [pending(shifted)], [5]);
      return { nodes: [x, shifted, r] };
    });
  });

  it("payload-less creation ops: zeros → add scalar", async () => {
    await assertRoundTripProducesSameOutput(() => {
      const node = makeNodeFactory();
      const z = node("zeros", [], [4]);
      const added = node(
        "add",
        [pending(z), { kind: "scalar", value: 7, dtype: "f32" }],
        [4],
      );
      return { nodes: [z, added] };
    });
  });
});
