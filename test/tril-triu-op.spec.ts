import { describe, it, expect, beforeEach } from "vitest";
import { Torchlette } from "../src/frontend";

describe("tril/triu ops", () => {
  let api: Torchlette;

  beforeEach(() => {
    api = new Torchlette();
  });

  describe("tril", () => {
    it("zeros upper triangle of 3x3 matrix", async () => {
      const a = api.full([3, 3], 1);
      const t = api.tril(a);
      const data = await t.cpu();
      expect(data).toEqual([
        1, 0, 0,
        1, 1, 0,
        1, 1, 1,
      ]);
    });

    it("works with k=1 (one above main diagonal)", async () => {
      const a = api.full([3, 3], 1);
      const t = api.tril(a, 1);
      const data = await t.cpu();
      expect(data).toEqual([
        1, 1, 0,
        1, 1, 1,
        1, 1, 1,
      ]);
    });

    it("works with k=-1 (one below main diagonal)", async () => {
      const a = api.full([3, 3], 1);
      const t = api.tril(a, -1);
      const data = await t.cpu();
      expect(data).toEqual([
        0, 0, 0,
        1, 0, 0,
        1, 1, 0,
      ]);
    });

    it("works via tensor instance method", async () => {
      const a = api.full([3, 3], 5);
      const t = a.tril();
      const data = await t.cpu();
      expect(data).toEqual([
        5, 0, 0,
        5, 5, 0,
        5, 5, 5,
      ]);
    });

    it("works with non-square matrix", async () => {
      const a = api.full([2, 4], 1);
      const t = api.tril(a);
      const data = await t.cpu();
      expect(data).toEqual([
        1, 0, 0, 0,
        1, 1, 0, 0,
      ]);
    });

    it("works with batched 3D tensor", async () => {
      // 2 batches of 2x2
      const a = api.full([2, 2, 2], 1);
      const t = api.tril(a);
      const data = await t.cpu();
      expect(data).toEqual([
        1, 0,
        1, 1,
        1, 0,
        1, 1,
      ]);
    });
  });

  describe("triu", () => {
    it("zeros lower triangle of 3x3 matrix", async () => {
      const a = api.full([3, 3], 1);
      const t = api.triu(a);
      const data = await t.cpu();
      expect(data).toEqual([
        1, 1, 1,
        0, 1, 1,
        0, 0, 1,
      ]);
    });

    it("works with k=1 (one above main diagonal)", async () => {
      const a = api.full([3, 3], 1);
      const t = api.triu(a, 1);
      const data = await t.cpu();
      expect(data).toEqual([
        0, 1, 1,
        0, 0, 1,
        0, 0, 0,
      ]);
    });

    it("works with k=-1 (one below main diagonal)", async () => {
      const a = api.full([3, 3], 1);
      const t = api.triu(a, -1);
      const data = await t.cpu();
      expect(data).toEqual([
        1, 1, 1,
        1, 1, 1,
        0, 1, 1,
      ]);
    });

    it("works via tensor instance method", async () => {
      const a = api.full([3, 3], 7);
      const t = a.triu();
      const data = await t.cpu();
      expect(data).toEqual([
        7, 7, 7,
        0, 7, 7,
        0, 0, 7,
      ]);
    });
  });

  describe("causal mask pattern", () => {
    it("triu(full(-1e9), k=1) produces causal mask", async () => {
      const n = 4;
      const mask = api.triu(api.full([n, n], -1e9), 1);
      const data = await mask.cpu();
      // Row 0: [0, -1e9, -1e9, -1e9]
      // Row 1: [0, 0, -1e9, -1e9]
      // Row 2: [0, 0, 0, -1e9]
      // Row 3: [0, 0, 0, 0]
      expect(data[0]).toBeCloseTo(0);
      expect(data[1]).toBeCloseTo(-1e9);
      expect(data[4]).toBeCloseTo(0);
      expect(data[5]).toBeCloseTo(0);
      expect(data[6]).toBeCloseTo(-1e9);
      expect(data[12]).toBeCloseTo(0);
      expect(data[13]).toBeCloseTo(0);
      expect(data[14]).toBeCloseTo(0);
      expect(data[15]).toBeCloseTo(0);
    });

    it("causal mask with broadcast dims [1, 1, n, n]", async () => {
      const n = 3;
      const mask = api.triu(api.full([1, 1, n, n], -1e9), 1);
      expect(mask.shape).toEqual([1, 1, n, n]);
      const data = await mask.cpu();
      // Same pattern as 2D, just with batch dims
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
});
