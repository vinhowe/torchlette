import { describe, it, expect, beforeEach } from "vitest";
import { Torchlette } from "../src/frontend";

describe("full op", () => {
  let api: Torchlette;

  beforeEach(() => {
    api = new Torchlette();
  });

  it("creates a 2D tensor filled with a value", async () => {
    const t = api.full([2, 3], 7);
    expect(t.shape).toEqual([2, 3]);
    const data = await t.cpu();
    expect(data).toEqual([7, 7, 7, 7, 7, 7]);
  });

  it("creates a scalar (0-d) tensor", async () => {
    const t = api.full([], 42);
    expect(t.shape).toEqual([]);
    const val = await t.item();
    expect(val).toBe(42);
  });

  it("creates a 1D tensor", async () => {
    const t = api.full([5], -3);
    expect(t.shape).toEqual([5]);
    const data = await t.cpu();
    expect(data).toEqual([-3, -3, -3, -3, -3]);
  });

  it("handles float fill values", async () => {
    const t = api.full([3], 0.5);
    const data = await t.cpu();
    expect(data[0]).toBeCloseTo(0.5);
    expect(data[1]).toBeCloseTo(0.5);
    expect(data[2]).toBeCloseTo(0.5);
  });

  it("handles zero fill value (like zeros)", async () => {
    const t = api.full([2, 2], 0);
    const data = await t.cpu();
    expect(data).toEqual([0, 0, 0, 0]);
  });

  it("handles fill value of 1 (like ones)", async () => {
    const t = api.full([3], 1);
    const data = await t.cpu();
    expect(data).toEqual([1, 1, 1]);
  });

  it("ones() delegates to full()", async () => {
    const t = api.ones([2, 2]);
    expect(t.shape).toEqual([2, 2]);
    const data = await t.cpu();
    expect(data).toEqual([1, 1, 1, 1]);
  });

  it("full tensor can be used in arithmetic", async () => {
    const a = api.tensorFromArray([1, 2, 3], [3]);
    const b = api.full([3], 10);
    const c = api.add(a, b);
    const data = await c.cpu();
    expect(data).toEqual([11, 12, 13]);
  });

  it("full tensor can be used as scalar broadcast", async () => {
    const a = api.tensorFromArray([2, 4, 6], [3]);
    const scale = api.full([], 0.5);
    const result = api.mul(a, scale);
    const data = await result.cpu();
    expect(data[0]).toBeCloseTo(1);
    expect(data[1]).toBeCloseTo(2);
    expect(data[2]).toBeCloseTo(3);
  });

  it("handles negative float values", async () => {
    const t = api.full([2], -1.5);
    const data = await t.cpu();
    expect(data[0]).toBeCloseTo(-1.5);
    expect(data[1]).toBeCloseTo(-1.5);
  });

  it("handles large fill values", async () => {
    const t = api.full([2], 65536);
    const data = await t.cpu();
    expect(data).toEqual([65536, 65536]);
  });
});
