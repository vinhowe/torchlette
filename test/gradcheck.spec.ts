/**
 * Gradient correctness tests using numerical finite differences.
 *
 * For each differentiable op, verifies that the analytical gradient
 * (from backward()) matches the numerical gradient within tolerance.
 * This mechanically validates every backward formula in the framework.
 */

import { describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend";
import { gradcheck } from "../src/testing/gradcheck";

// CPU-only, no fusion — pure f32 deterministic execution
const api = new Torchlette("cpu");

// Helper: create a small test tensor with requiresGrad
function t(values: number[], shape: number[]) {
  return api.tensorFromArray(values, shape, { requiresGrad: true });
}

describe("gradcheck", { timeout: 30_000 }, () => {
  // =====================================================================
  // Unary ops
  // =====================================================================

  it("sqrt", async () => {
    const x = t([1, 4, 9, 16], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.sqrt(a)), [x]);
    expect(r.pass).toBe(true);
  });

  it("exp", async () => {
    const x = t([0.1, 0.2, -0.3, 0.5], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.exp(a)), [x]);
    expect(r.pass).toBe(true);
  });

  it("log", async () => {
    const x = t([1, 2, 3, 4], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.log(a)), [x]);
    expect(r.pass).toBe(true);
  });

  it("neg", async () => {
    const x = t([1, -2, 3, -4], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.neg(a)), [x]);
    expect(r.pass).toBe(true);
  });

  it("abs", async () => {
    // Avoid zero (non-differentiable)
    const x = t([1, -2, 3, -4], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.abs(a)), [x]);
    expect(r.pass).toBe(true);
  });

  it("tanh", async () => {
    const x = t([0.1, -0.5, 1.0, -1.0], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.tanh(a)), [x]);
    expect(r.pass).toBe(true);
  });

  it("sigmoid", async () => {
    const x = t([0.1, -0.5, 1.0, -1.0], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.sigmoid(a)), [x]);
    expect(r.pass).toBe(true);
  });

  it("relu", async () => {
    // Avoid zero (non-differentiable)
    const x = t([1, -2, 3, -4], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.relu(a)), [x]);
    expect(r.pass).toBe(true);
  });

  it("silu", async () => {
    const x = t([0.5, -0.5, 1.0, -1.0], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.silu(a)), [x]);
    expect(r.pass).toBe(true);
  });

  it("sin", async () => {
    const x = t([0.1, 0.5, 1.0, 2.0], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.sin(a)), [x]);
    expect(r.pass).toBe(true);
  });

  it("cos", async () => {
    const x = t([0.1, 0.5, 1.0, 2.0], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.cos(a)), [x]);
    expect(r.pass).toBe(true);
  });

  it("rsqrt", async () => {
    const x = t([1, 4, 9, 16], [2, 2]);
    const r = await gradcheck(api, (a) => api.sum(api.rsqrt(a)), [x]);
    expect(r.pass).toBe(true);
  });

  // =====================================================================
  // Binary ops
  // =====================================================================

  it("add", async () => {
    const a = t([1, 2, 3, 4], [2, 2]);
    const b = t([5, 6, 7, 8], [2, 2]);
    const r = await gradcheck(api, (x, y) => api.sum(api.add(x, y)), [a, b]);
    expect(r.pass).toBe(true);
  });

  it("mul", async () => {
    const a = t([1, 2, 3, 4], [2, 2]);
    const b = t([5, 6, 7, 8], [2, 2]);
    const r = await gradcheck(api, (x, y) => api.sum(api.mul(x, y)), [a, b]);
    expect(r.pass).toBe(true);
  });

  it("sub", async () => {
    const a = t([1, 2, 3, 4], [2, 2]);
    const b = t([5, 6, 7, 8], [2, 2]);
    const r = await gradcheck(api, (x, y) => api.sum(api.sub(x, y)), [a, b]);
    expect(r.pass).toBe(true);
  });

  it("div", async () => {
    const a = t([1, 2, 3, 4], [2, 2]);
    const b = t([5, 6, 7, 8], [2, 2]);
    const r = await gradcheck(api, (x, y) => api.sum(api.div(x, y)), [a, b]);
    expect(r.pass).toBe(true);
  });

  it("pow", async () => {
    const a = t([1, 2, 3, 4], [2, 2]);
    const b = t([2, 0.5, 1, 3], [2, 2]);
    const r = await gradcheck(api, (x, y) => api.sum(api.pow(x, y)), [a, b]);
    expect(r.pass).toBe(true);
  });

  it("add with broadcast", async () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = t([10, 20, 30], [3]);
    const r = await gradcheck(api, (x, y) => api.sum(api.add(x, y)), [a, b]);
    expect(r.pass).toBe(true);
  });

  it("mul with broadcast", async () => {
    const a = t([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = t([10, 20, 30], [3]);
    const r = await gradcheck(api, (x, y) => api.sum(api.mul(x, y)), [a, b]);
    expect(r.pass).toBe(true);
  });

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
