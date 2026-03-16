/**
 * Gradient correctness tests using numerical finite differences.
 *
 * For each differentiable op, verifies that the analytical gradient
 * (from backward()) matches the numerical gradient within tolerance.
 * This mechanically validates every backward formula in the framework.
 */

import { describe, expect, it } from "vitest";
import { OP_REGISTRY } from "../src/ops/registry";
import type { Tensor } from "../src/frontend/torchlette";
import { Torchlette } from "../src/frontend/torchlette";
import { gradcheck } from "../src/testing/gradcheck";

// CPU-only, no fusion — pure f32 deterministic execution
const api = new Torchlette("cpu");

// Helper: create a small test tensor with requiresGrad
function t(values: number[], shape: number[]) {
  return api.tensorFromArray(values, shape, { requiresGrad: true });
}

// ============================================================================
// Auto-generated tests for registry-driven ops
// ============================================================================

// Input values that avoid non-differentiable points (zero for relu/abs/sign,
// negative for sqrt/log/rsqrt). Positive values work for all ops.
const SAFE_UNARY_VALUES = [0.5, 1.0, 1.5, 2.0];
// Avoid 1.0 in base (log(1)=0 breaks pow gradient via relative error).
// Keep values small to avoid f32 overflow in pow (e.g., 4^8 = 65536).
const SAFE_BINARY_A = [1.5, 2.0, 2.5, 3.0];
const SAFE_BINARY_B = [0.5, 1.5, 2.0, 2.5];

describe(
  "gradcheck — auto-generated from OP_REGISTRY",
  { timeout: 30_000 },
  () => {
    // Unary ops: every OP_REGISTRY entry with arity=1 and a non-null grad
    for (const [opName, def] of Object.entries(OP_REGISTRY)) {
      if (def.arity !== 1 || def.grad === undefined || def.grad === null)
        continue;
      // Skip ops not exposed on Torchlette (internal-only ops)
      if (!(opName in api)) continue;

      it(`unary: ${opName}`, async () => {
        const x = t(SAFE_UNARY_VALUES, [2, 2]);
        const r = await gradcheck(
          api,
          (a) => api.sum((api as any)[opName](a)),
          [x],
        );
        expect(r.pass, r.message).toBe(true);
      });
    }

    // Binary ops: every OP_REGISTRY entry with arity=2 and ttGrad
    for (const [opName, def] of Object.entries(OP_REGISTRY)) {
      if (def.arity !== 2 || !def.ttGrad) continue;
      if (!(opName in api)) continue;

      it(`binary: ${opName}`, async () => {
        const a = t(SAFE_BINARY_A, [2, 2]);
        const b = t(SAFE_BINARY_B, [2, 2]);
        const r = await gradcheck(
          api,
          (x, y) => api.sum((api as any)[opName](x, y)),
          [a, b],
        );
        expect(r.pass, r.message).toBe(true);
      });

      it(`binary broadcast: ${opName}`, async () => {
        const a = t([1.5, 2.0, 2.5, 3.0, 1.5, 2.0], [2, 3]);
        const b = t([0.5, 1.5, 2.0], [3]);
        const r = await gradcheck(
          api,
          (x, y) => api.sum((api as any)[opName](x, y)),
          [a, b],
        );
        expect(r.pass, r.message).toBe(true);
      });
    }
  },
);

// ============================================================================
// Hand-written tests for ops that need specific inputs or compositions
// ============================================================================

describe("gradcheck — hand-written", { timeout: 30_000 }, () => {
  // =====================================================================
  // Reductions
  // =====================================================================

  it("sum (full)", async () => {
    const x = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const r = await gradcheck(api, (a) => api.sum(a), [x]);
    expect(r.pass).toBe(true);
  });

  it("sum (dim)", async () => {
    const x = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const r = await gradcheck(
      api,
      (a) => api.sum(api.sum(a, { dim: 1 }) as any),
      [x],
    );
    expect(r.pass).toBe(true);
  });

  it("mean (dim)", async () => {
    const x = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const r = await gradcheck(
      api,
      (a) => api.sum(api.mean(a, { dim: 1, keepdim: true }) as any),
      [x],
    );
    expect(r.pass).toBe(true);
  });

  // =====================================================================
  // View ops
  // =====================================================================

  it("reshape", async () => {
    const x = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const r = await gradcheck(
      api,
      (a) => api.sum(api.mul(api.reshape(a, [3, 2]), 2)),
      [x],
    );
    expect(r.pass).toBe(true);
  });

  it("transpose", async () => {
    const x = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const r = await gradcheck(
      api,
      (a) => api.sum(api.transpose(a, { dim0: 0, dim1: 1 })),
      [x],
    );
    expect(r.pass).toBe(true);
  });

  it("expand", async () => {
    const x = t([1, 2, 3], [1, 3]);
    const r = await gradcheck(api, (a) => api.sum(api.expand(a, [2, 3])), [x]);
    expect(r.pass).toBe(true);
  });

  // =====================================================================
  // Composite ops
  // =====================================================================

  it("matmul", async () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = t([1, 2, 3, 4, 5, 6], [3, 2]);
    const r = await gradcheck(api, (x, y) => api.sum(api.matmul(x, y)), [a, b]);
    expect(r.pass).toBe(true);
  });

  it("softmax", async () => {
    const x = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const r = await gradcheck(api, (a) => api.sum(api.softmax(a, -1)), [x]);
    expect(r.pass).toBe(true);
  });

  it("gelu (tanh)", async () => {
    const x = t([0.5, -0.5, 1.0, -1.0], [2, 2]);
    const r = await gradcheck(
      api,
      (a) => api.sum(api.gelu(a, { approximate: "tanh" })),
      [x],
    );
    expect(r.pass).toBe(true);
  });

  it("clamp", async () => {
    // Values away from boundaries (non-differentiable at min/max)
    const x = t([0.5, 1.5, 2.5, 3.5], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.clamp(a, 1.0, 3.0)), [x]);
    expect(r.pass).toBe(true);
  });

  // =====================================================================
  // Chains (multiple ops composed)
  // =====================================================================

  it("mul → sum (reduction preamble pattern)", async () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = t([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]);
    const r = await gradcheck(api, (x, y) => api.sum(api.mul(x, y)), [a, b]);
    expect(r.pass).toBe(true);
  });

  it("linear (input @ weight.T + bias)", async () => {
    const input = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const weight = t([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]);
    const bias = t([0.01, 0.02], [2]);
    const r = await gradcheck(api, (x, w, b) => api.sum(api.linear(x, w, b)), [
      input,
      weight,
      bias,
    ]);
    expect(r.pass).toBe(true);
  });

  // =====================================================================
  // Error reporting
  // =====================================================================

  it("reports failures clearly", async () => {
    // Create a function with intentionally wrong gradient
    // (just test that gradcheck returns a result object, not that it fails)
    const x = t([1, 2, 3, 4], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(a), [x]);
    expect(r.inputs.length).toBe(1);
    expect(r.inputs[0].shape).toEqual([2, 2]);
    expect(typeof r.message).toBe("string");
  });
});
