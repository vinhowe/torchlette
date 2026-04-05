/**
 * Structural round-trip of SerializedPlan: build a plan, serialize, stringify,
 * parse, deserialize, and assert field-for-field equivalence with the original.
 *
 * Exercises every LazyRef kind (pending/materialized/scalar), Float32Array
 * payload encoding, multi-output outputIndex refs, and representative op
 * payload shapes.
 */

import { describe, expect, it } from "vitest";
import type { ExecutionPlan, LazyIRNode, LazyRef } from "../../src/graph/types";
import { deserializePlan, serializePlan } from "../../src/remote/serialize";
import type { HandleRef, SerializedPlan } from "../../src/remote/wire";

// ============================================================================
// Helpers for building plans by hand
// ============================================================================

function makeNodeFactory() {
  let nextId = 1;
  return (
    op: LazyIRNode["op"],
    inputs: LazyRef[],
    shape: number[],
    dtype: LazyIRNode["dtype"] = "f32",
    payload?: unknown,
  ): LazyIRNode => {
    const n: LazyIRNode = {
      id: nextId++,
      op,
      inputs,
      shape,
      dtype,
      device: "cpu",
    };
    if (payload !== undefined) n.payload = payload;
    return n;
  };
}

function pending(n: LazyIRNode, outputIndex?: number): LazyRef {
  return outputIndex && outputIndex > 0
    ? { kind: "pending", node: n, outputIndex }
    : { kind: "pending", node: n };
}

const resolveLocal = (id: number): HandleRef => `h:${id}`;

const stubStorage = (id: number) => ({
  id,
  device: "cpu" as const,
  // biome-ignore lint/suspicious/noExplicitAny: test stub
  backendTensor: {} as any,
});

// ============================================================================
// Round-trip equivalence checker
// ============================================================================

function assertRoundTripEquivalent(original: ExecutionPlan): void {
  const serialized = serializePlan(original, { resolveHandle: resolveLocal });
  const json = JSON.stringify(serialized);
  const parsed = JSON.parse(json) as SerializedPlan;
  const rebuilt = deserializePlan(parsed, {
    resolveHandle: (h) => stubStorage(Number(h.slice(2))),
  });

  expect(rebuilt.nodes.length).toBe(original.nodes.length);

  for (let i = 0; i < original.nodes.length; i++) {
    const a = original.nodes[i];
    const b = rebuilt.nodes[i];
    expect(b.op).toBe(a.op);
    expect(b.dtype).toBe(a.dtype);
    expect(b.device).toBe(a.device);
    expect(b.shape).toEqual(a.shape);
    expect(b.inputs.length).toBe(a.inputs.length);
    expect(!!b.isCheckpointBoundary).toBe(!!a.isCheckpointBoundary);
    expect(b.module).toBe(a.module);

    for (let j = 0; j < a.inputs.length; j++) {
      const ra = a.inputs[j];
      const rb = b.inputs[j];
      expect(rb.kind).toBe(ra.kind);
      if (ra.kind === "scalar" && rb.kind === "scalar") {
        expect(rb.value).toBe(ra.value);
        expect(rb.dtype).toBe(ra.dtype);
      } else if (ra.kind === "materialized" && rb.kind === "materialized") {
        expect(rb.storage.id).toBe(ra.storage.id);
      } else if (ra.kind === "pending" && rb.kind === "pending") {
        const aIdx = original.nodes.indexOf(ra.node);
        const bIdx = rebuilt.nodes.indexOf(rb.node);
        expect(bIdx).toBe(aIdx);
        expect(rb.outputIndex ?? 0).toBe(ra.outputIndex ?? 0);
      }
    }

    comparePayload(a.payload, b.payload);
  }
}

function comparePayload(a: unknown, b: unknown): void {
  if (a === undefined && b === undefined) return;
  if (a instanceof Float32Array || b instanceof Float32Array) {
    expect(b instanceof Float32Array).toBe(true);
    expect(a instanceof Float32Array).toBe(true);
    const aa = a as Float32Array;
    const bb = b as Float32Array;
    expect(bb.length).toBe(aa.length);
    for (let i = 0; i < aa.length; i++) expect(bb[i]).toBe(aa[i]);
    return;
  }
  expect(JSON.stringify(b)).toBe(JSON.stringify(a));
}

// ============================================================================
// Cases
// ============================================================================

describe("remote wire format: structural round-trip", () => {
  it("elementwise add with Float32Array payloads", () => {
    const node = makeNodeFactory();
    const a = node("tensorFromArray", [], [2, 2], "f32", {
      values: new Float32Array([1, 2, 3, 4]),
    });
    const b = node("tensorFromArray", [], [2, 2], "f32", {
      values: new Float32Array([10, 20, 30, 40]),
    });
    const add = node("add", [pending(a), pending(b)], [2, 2]);
    assertRoundTripEquivalent({ nodes: [a, b, add] });
  });

  it("matmul (no payload)", () => {
    const node = makeNodeFactory();
    const a = node("tensorFromArray", [], [3, 4], "f32", {
      values: new Float32Array(12).fill(1),
    });
    const b = node("tensorFromArray", [], [4, 2], "f32", {
      values: new Float32Array(8).fill(2),
    });
    const mm = node("matmul", [pending(a), pending(b)], [3, 2]);
    assertRoundTripEquivalent({ nodes: [a, b, mm] });
  });

  it("sum reduction with dim/keepdim payload", () => {
    const node = makeNodeFactory();
    const a = node("tensorFromArray", [], [2, 3], "f32", {
      values: new Float32Array([1, 2, 3, 4, 5, 6]),
    });
    const s = node("sum", [pending(a)], [2], "f32", { dim: 1, keepdim: false });
    assertRoundTripEquivalent({ nodes: [a, s] });
  });

  it("transpose with TransposeOptions payload", () => {
    const node = makeNodeFactory();
    const a = node("zeros", [], [4, 8]);
    const t = node("transpose", [pending(a)], [8, 4], "f32", {
      dim0: 0,
      dim1: 1,
    });
    assertRoundTripEquivalent({ nodes: [a, t] });
  });

  it("binary with scalar ref", () => {
    const node = makeNodeFactory();
    const a = node("tensorFromArray", [], [3], "f32", {
      values: new Float32Array([1, 2, 3]),
    });
    const mul = node(
      "mul",
      [pending(a), { kind: "scalar", value: 2.5, dtype: "f32" }],
      [3],
    );
    assertRoundTripEquivalent({ nodes: [a, mul] });
  });

  it("materialized ref (external server-resident tensor)", () => {
    const node = makeNodeFactory();
    const external: LazyRef = {
      kind: "materialized",
      storage: stubStorage(42),
    };
    const cast = node("cast", [external], [10, 10], "f16", { dtype: "f16" });
    assertRoundTripEquivalent({ nodes: [cast] });
  });

  it("multi-output op with outputIndex ref", () => {
    const node = makeNodeFactory();
    const q = node("tensorFromArray", [], [1, 8, 4], "f32", {
      values: new Float32Array(32).fill(1),
    });
    const k = node("tensorFromArray", [], [1, 8, 4], "f32", {
      values: new Float32Array(32).fill(2),
    });
    const v = node("tensorFromArray", [], [1, 8, 4], "f32", {
      values: new Float32Array(32).fill(3),
    });
    const fa = node(
      "fusedAttentionForward",
      [pending(q), pending(k), pending(v)],
      [1, 8, 4],
      "f32",
      {
        batchSize: 1,
        numHeads: 1,
        seqLen: 8,
        headDim: 4,
        scale: 0.5,
        isCausal: true,
      },
    );
    const consumer = node("add", [pending(fa, 1), pending(fa, 1)], [1, 8], "f32");
    assertRoundTripEquivalent({ nodes: [q, k, v, fa, consumer] });
  });

  it("deep chain: mm → add → gelu → cast", () => {
    const node = makeNodeFactory();
    const a = node("tensorFromArray", [], [4, 8], "f32", {
      values: new Float32Array(32).fill(1),
    });
    const b = node("tensorFromArray", [], [8, 16], "f32", {
      values: new Float32Array(128).fill(1),
    });
    const mm = node("matmul", [pending(a), pending(b)], [4, 16], "f32");
    const bias = node("tensorFromArray", [], [16], "f32", {
      values: new Float32Array(16).fill(0.1),
    });
    const bi = node("add", [pending(mm), pending(bias)], [4, 16], "f32");
    const g = node("gelu", [pending(bi)], [4, 16], "f32", { approximate: "tanh" });
    const c = node("cast", [pending(g)], [4, 16], "f16", { dtype: "f16" });
    assertRoundTripEquivalent({ nodes: [a, b, mm, bias, bi, g, c] });
  });

  it("adamStep config payload (without infFlagBuffer)", () => {
    const node = makeNodeFactory();
    const param = node("tensorFromArray", [], [128], "f32", {
      values: new Float32Array(128).fill(0.5),
    });
    const grad = node("tensorFromArray", [], [128], "f32", {
      values: new Float32Array(128).fill(0.01),
    });
    const m = node("zeros", [], [128]);
    const v = node("zeros", [], [128]);
    const adam = node(
      "adamStep",
      [pending(param), pending(grad), pending(m), pending(v)],
      [128],
      "f32",
      {
        beta1: 0.9,
        beta2: 0.999,
        stepSize: 3e-4,
        eps: 1e-8,
        weightDecay: 0.01,
        lrTimesWd: 0.01 * 3e-4,
        decoupledWd: true,
      },
    );
    assertRoundTripEquivalent({ nodes: [param, grad, m, v, adam] });
  });

  it("preserves isCheckpointBoundary and module fields", () => {
    const node = makeNodeFactory();
    const a = node("zeros", [], [4]);
    a.module = "TestModule";
    a.isCheckpointBoundary = true;
    assertRoundTripEquivalent({ nodes: [a] });
  });

  it("rejects unsupported format versions", () => {
    expect(() =>
      deserializePlan(
        // biome-ignore lint/suspicious/noExplicitAny: negative case
        { version: 999, nodes: [], externalHandles: [] } as any,
      ),
    ).toThrow(/unsupported version/);
  });
});
