import { describe, it, expect, beforeEach } from "vitest";
import { Torchlette } from "../src/frontend";

describe("arange op", () => {
  let api: Torchlette;

  beforeEach(() => {
    api = new Torchlette();
  });

  it("creates [0, 1, 2, 3, 4] from arange(5)", async () => {
    const t = api.arange(5);
    expect(t.shape).toEqual([5]);
    const data = await t.cpu();
    expect(data).toEqual([0, 1, 2, 3, 4]);
  });

  it("creates [2, 3, 4] from arange(5, {start: 2})", async () => {
    const t = api.arange(5, { start: 2 });
    expect(t.shape).toEqual([3]);
    const data = await t.cpu();
    expect(data).toEqual([2, 3, 4]);
  });

  it("creates [0, 2, 4, 6] from arange(8, {step: 2})", async () => {
    const t = api.arange(8, { step: 2 });
    expect(t.shape).toEqual([4]);
    const data = await t.cpu();
    expect(data).toEqual([0, 2, 4, 6]);
  });

  it("creates [1, 1.5, 2, 2.5] from arange(3, {start: 1, step: 0.5})", async () => {
    const t = api.arange(3, { start: 1, step: 0.5 });
    expect(t.shape).toEqual([4]);
    const data = await t.cpu();
    expect(data[0]).toBeCloseTo(1);
    expect(data[1]).toBeCloseTo(1.5);
    expect(data[2]).toBeCloseTo(2);
    expect(data[3]).toBeCloseTo(2.5);
  });

  it("creates single-element tensor from arange(1)", async () => {
    const t = api.arange(1);
    expect(t.shape).toEqual([1]);
    const data = await t.cpu();
    expect(data).toEqual([0]);
  });

  it("can be used in arithmetic", async () => {
    const indices = api.arange(4);
    const offset = api.full([4], 10);
    const result = api.add(indices, offset);
    const data = await result.cpu();
    expect(data).toEqual([10, 11, 12, 13]);
  });

  it("can be reshaped and broadcast", async () => {
    const rows = api.arange(3).reshape([3, 1]);
    const cols = api.arange(3).reshape([1, 3]);
    // rows + cols = [[0,1,2],[1,2,3],[2,3,4]]
    const sum = api.add(rows, cols);
    expect(sum.shape).toEqual([3, 3]);
    const data = await sum.cpu();
    expect(data).toEqual([0, 1, 2, 1, 2, 3, 2, 3, 4]);
  });

  it("works with comparison ops for causal mask pattern", async () => {
    const n = 4;
    const rows = api.arange(n).reshape([n, 1]);
    const cols = api.arange(n).reshape([1, n]);
    // mask[i][j] = 1.0 if j <= i (causal), 0.0 otherwise
    const mask = cols.le(rows);
    expect(mask.shape).toEqual([n, n]);
    const data = await mask.cpu();
    // Row 0: [1,0,0,0], Row 1: [1,1,0,0], Row 2: [1,1,1,0], Row 3: [1,1,1,1]
    expect(data).toEqual([1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1]);
  });

  it("works with where for causal mask with values", async () => {
    const n = 3;
    const rows = api.arange(n).reshape([n, 1]);
    const cols = api.arange(n).reshape([1, n]);
    const causal = cols.le(rows);
    const zero = api.full([], 0);
    const negInf = api.full([], -1e9);
    const mask = api.where(causal, zero, negInf);
    expect(mask.shape).toEqual([n, n]);
    const data = await mask.cpu();
    // Row 0: [0, -1e9, -1e9], Row 1: [0, 0, -1e9], Row 2: [0, 0, 0]
    expect(data[0]).toBeCloseTo(0);
    expect(data[1]).toBeCloseTo(-1e9);
    expect(data[2]).toBeCloseTo(-1e9);
    expect(data[3]).toBeCloseTo(0);
    expect(data[4]).toBeCloseTo(0);
    expect(data[5]).toBeCloseTo(-1e9);
    expect(data[6]).toBeCloseTo(0);
    expect(data[7]).toBeCloseTo(0);
    expect(data[8]).toBeCloseTo(0);
  });
});
