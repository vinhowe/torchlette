import { describe, expect, it, beforeAll } from "vitest";
import {
  initWebGPU,
  getWebGPUInitError,
  webgpuBackend,
} from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

type WebGPUTensor = ReturnType<typeof webgpuBackend.ops.tensorFromArray> & {
  buffer: unknown;
  strides: number[];
  offset: number;
  isContiguous: boolean;
};

describe.skipIf(cpuOnly)("WebGPU ops", () => {
  beforeAll(async () => {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }
  });

  describe("transpose", () => {
    it("transposes 2D tensor", async () => {
      // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const t = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 1 });

      expect(t.shape).toEqual([3, 2]);
      const values = await webgpuBackend.ops.read(t);
      expect(values).toEqual([1, 4, 2, 5, 3, 6]);
    });

    it("transposes 3D tensor", async () => {
      // Shape [2, 2, 3] -> transpose(0, 2) -> [3, 2, 2]
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = webgpuBackend.ops.tensorFromArray(data, [2, 2, 3]);
      const t = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 2 });

      expect(t.shape).toEqual([3, 2, 2]);
      const values = await webgpuBackend.ops.read(t);
      // Original: [[[0,1,2],[3,4,5]], [[6,7,8],[9,10,11]]]
      // After transpose(0,2): [[[0,6],[3,9]], [[1,7],[4,10]], [[2,8],[5,11]]]
      expect(values).toEqual([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]);
    });
  });

  describe("expand", () => {
    it("broadcasts single element to shape", async () => {
      const a = webgpuBackend.ops.tensorFromArray([5], [1]);
      const e = webgpuBackend.ops.expand(a, [4]);

      expect(e.shape).toEqual([4]);
      const values = await webgpuBackend.ops.read(e);
      expect(values).toEqual([5, 5, 5, 5]);
    });

    it("broadcasts [1, 3] to [2, 3]", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3], [1, 3]);
      const e = webgpuBackend.ops.expand(a, [2, 3]);

      expect(e.shape).toEqual([2, 3]);
      const values = await webgpuBackend.ops.read(e);
      expect(values).toEqual([1, 2, 3, 1, 2, 3]);
    });

    it("broadcasts [3, 1] to [3, 4]", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3], [3, 1]);
      const e = webgpuBackend.ops.expand(a, [3, 4]);

      expect(e.shape).toEqual([3, 4]);
      const values = await webgpuBackend.ops.read(e);
      expect(values).toEqual([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
    });

    it("adds leading dimensions", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2], [2]);
      const e = webgpuBackend.ops.expand(a, [3, 2]);

      expect(e.shape).toEqual([3, 2]);
      const values = await webgpuBackend.ops.read(e);
      expect(values).toEqual([1, 2, 1, 2, 1, 2]);
    });
  });

  describe("sum", () => {
    it("sums along dimension 0", async () => {
      // [[1, 2], [3, 4]] -> sum(dim=0) -> [4, 6]
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const s = webgpuBackend.ops.sum(a, { dim: 0 });

      expect((s as { shape: number[] }).shape).toEqual([2]);
      const values = await webgpuBackend.ops.read(s as { shape: number[] });
      expect(values).toEqual([4, 6]);
    });

    it("sums along dimension 1", async () => {
      // [[1, 2], [3, 4]] -> sum(dim=1) -> [3, 7]
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const s = webgpuBackend.ops.sum(a, { dim: 1 });

      expect((s as { shape: number[] }).shape).toEqual([2]);
      const values = await webgpuBackend.ops.read(s as { shape: number[] });
      expect(values).toEqual([3, 7]);
    });

    it("sums with keepdim=true", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const s = webgpuBackend.ops.sum(a, { dim: 0, keepdim: true });

      expect((s as { shape: number[] }).shape).toEqual([1, 2]);
      const values = await webgpuBackend.ops.read(s as { shape: number[] });
      expect(values).toEqual([4, 6]);
    });

    it("sums 3D tensor along middle dimension", async () => {
      // Shape [2, 3, 2] -> sum(dim=1) -> [2, 2]
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = webgpuBackend.ops.tensorFromArray(data, [2, 3, 2]);
      const s = webgpuBackend.ops.sum(a, { dim: 1 });

      expect((s as { shape: number[] }).shape).toEqual([2, 2]);
      const values = await webgpuBackend.ops.read(s as { shape: number[] });
      // [[0,1], [2,3], [4,5]], [[6,7], [8,9], [10,11]] -> sum(dim=1)
      // [0+2+4, 1+3+5], [6+8+10, 7+9+11] = [6, 9], [24, 27]
      expect(values).toEqual([6, 9, 24, 27]);
    });
  });

  describe("mean", () => {
    it("computes mean along dimension", async () => {
      // [[1, 2], [3, 4]] -> mean(dim=0) -> [2, 3]
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const m = webgpuBackend.ops.mean(a, { dim: 0 });

      expect((m as { shape: number[] }).shape).toEqual([2]);
      const values = await webgpuBackend.ops.read(m as { shape: number[] });
      expect(values[0]).toBeCloseTo(2);
      expect(values[1]).toBeCloseTo(3);
    });

    it("computes mean with keepdim", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const m = webgpuBackend.ops.mean(a, { dim: 1, keepdim: true });

      expect((m as { shape: number[] }).shape).toEqual([2, 1]);
      const values = await webgpuBackend.ops.read(m as { shape: number[] });
      expect(values[0]).toBeCloseTo(1.5);
      expect(values[1]).toBeCloseTo(3.5);
    });
  });

  describe("gather", () => {
    it("gathers along dimension 0", async () => {
      // [[1, 2], [3, 4], [5, 6]] gather with indices [[0], [2]] on dim 0
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4, 5, 6], [3, 2]);
      const idx = webgpuBackend.ops.tensorFromArray([0, 2], [2, 1]);
      const g = webgpuBackend.ops.gather(a, idx, { dim: 0 });

      expect(g.shape).toEqual([2, 1]);
      const values = await webgpuBackend.ops.read(g);
      expect(values).toEqual([1, 5]);
    });

    it("gathers along dimension 1", async () => {
      // [[1, 2, 3], [4, 5, 6]] gather with indices [[0, 2], [1, 0]] on dim 1
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const idx = webgpuBackend.ops.tensorFromArray([0, 2, 1, 0], [2, 2]);
      const g = webgpuBackend.ops.gather(a, idx, { dim: 1 });

      expect(g.shape).toEqual([2, 2]);
      const values = await webgpuBackend.ops.read(g);
      // Row 0: indices [0, 2] -> [1, 3]
      // Row 1: indices [1, 0] -> [5, 4]
      expect(values).toEqual([1, 3, 5, 4]);
    });
  });

  describe("scatterAdd", () => {
    it("scatters and adds along dimension 0", async () => {
      // Start with zeros, scatter add values at specific indices
      const a = webgpuBackend.ops.tensorFromArray([0, 0, 0, 0], [2, 2]);
      const idx = webgpuBackend.ops.tensorFromArray([0, 1, 1, 0], [2, 2]);
      const src = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const s = webgpuBackend.ops.scatterAdd(a, idx, src, { dim: 0 });

      expect(s.shape).toEqual([2, 2]);
      const values = await webgpuBackend.ops.read(s);
      // Index [0,0]: src[0,0]=1 goes to out[0,0], src[1,1]=4 goes to out[0,1]
      // Index [0,1]: src[0,1]=2 goes to out[1,0], src[1,0]=3 goes to out[1,1]
      // But wait, scatterAdd uses idx[i,j] to determine which row of output
      // So: out[idx[0,0], 0] += src[0,0] = out[0,0] += 1
      //     out[idx[0,1], 1] += src[0,1] = out[1,1] += 2
      //     out[idx[1,0], 0] += src[1,0] = out[1,0] += 3
      //     out[idx[1,1], 1] += src[1,1] = out[0,1] += 4
      // Result: [[1, 4], [3, 2]]
      expect(values).toEqual([1, 4, 3, 2]);
    });
  });

  describe("view-based transpose", () => {
    it("returns view sharing same buffer", async () => {
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = webgpuBackend.ops.tensorFromArray(data, [3, 4]) as WebGPUTensor;
      const t = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 1 }) as WebGPUTensor;

      // Same underlying buffer
      expect(t.buffer).toBe(a.buffer);
      // Shape is swapped
      expect(t.shape).toEqual([4, 3]);
    });

    it("has non-contiguous strides after transpose", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]) as WebGPUTensor;

      expect(a.isContiguous).toBe(true);
      expect(a.strides).toEqual([3, 1]);

      const t = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 1 }) as WebGPUTensor;

      expect(t.isContiguous).toBe(false);
      expect(t.strides).toEqual([1, 3]); // Strides swapped
    });

    it("reads correct values from transposed view", async () => {
      // Original: [[0,1,2,3], [4,5,6,7], [8,9,10,11]] shape [3, 4]
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = webgpuBackend.ops.tensorFromArray(data, [3, 4]);
      const t = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 1 });

      // Transposed: [[0,4,8], [1,5,9], [2,6,10], [3,7,11]] shape [4, 3]
      expect(t.shape).toEqual([4, 3]);
      const values = await webgpuBackend.ops.read(t);
      expect(values).toEqual([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
    });

    it("double transpose gives equivalent data", async () => {
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = webgpuBackend.ops.tensorFromArray(data, [3, 4]) as WebGPUTensor;
      const t1 = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 1 }) as WebGPUTensor;
      const t2 = webgpuBackend.ops.transpose(t1, { dim0: 0, dim1: 1 }) as WebGPUTensor;

      // Shape restored
      expect(t2.shape).toEqual([3, 4]);
      // Same buffer throughout
      expect(t2.buffer).toBe(a.buffer);
      // Strides restored to contiguous
      expect(t2.isContiguous).toBe(true);

      const values = await webgpuBackend.ops.read(t2);
      expect(values).toEqual(data);
    });
  });

  describe("contiguous", () => {
    it("returns a non-owning view if already contiguous", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]) as WebGPUTensor;
      const c = webgpuBackend.ops.contiguous(a) as WebGPUTensor;

      // contiguous returns a view sharing the same buffer (ownsBuffer=false)
      // to prevent double-free when the executor creates separate StorageHandles.
      expect(c.buffer).toBe(a.buffer); // Same underlying buffer
      expect(c.ownsBuffer).toBe(false); // View doesn't own the buffer
      expect(c.shape).toEqual(a.shape);
      expect(c.isContiguous).toBe(true);
    });

    it("materializes transposed view to new buffer", async () => {
      const data = Array.from({ length: 12 }, (_, i) => i);
      const a = webgpuBackend.ops.tensorFromArray(data, [3, 4]) as WebGPUTensor;
      const t = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 1 }) as WebGPUTensor;

      expect(t.isContiguous).toBe(false);

      const c = webgpuBackend.ops.contiguous(t) as WebGPUTensor;

      expect(c.isContiguous).toBe(true);
      expect(c.shape).toEqual([4, 3]);
      expect(c.buffer).not.toBe(t.buffer); // New buffer

      // Values match the logical transposed view
      const values = await webgpuBackend.ops.read(c);
      expect(values).toEqual([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
    });

    it("contiguous result has correct strides", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]) as WebGPUTensor;
      const t = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 1 }) as WebGPUTensor;
      const c = webgpuBackend.ops.contiguous(t) as WebGPUTensor;

      // Contiguous strides for [3, 2] should be [2, 1]
      expect(c.strides).toEqual([2, 1]);
    });
  });

  describe("view-based expand", () => {
    it("returns view sharing same buffer", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3], [1, 3]) as WebGPUTensor;
      const e = webgpuBackend.ops.expand(a, [4, 3]) as WebGPUTensor;

      // Same underlying buffer
      expect(e.buffer).toBe(a.buffer);
      expect(e.shape).toEqual([4, 3]);
    });

    it("has stride=0 for broadcast dimensions", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3], [1, 3]) as WebGPUTensor;
      const e = webgpuBackend.ops.expand(a, [4, 3]) as WebGPUTensor;

      // First dimension (broadcast from 1 to 4) should have stride 0
      expect(e.strides[0]).toBe(0);
      // Second dimension (unchanged) should have stride 1
      expect(e.strides[1]).toBe(1);
      expect(e.isContiguous).toBe(false);
    });

    it("reads correct values from expanded view", async () => {
      const a = webgpuBackend.ops.tensorFromArray([10, 20, 30], [1, 3]);
      const e = webgpuBackend.ops.expand(a, [3, 3]);

      const values = await webgpuBackend.ops.read(e);
      // Each row should be the same [10, 20, 30]
      expect(values).toEqual([10, 20, 30, 10, 20, 30, 10, 20, 30]);
    });

    it("handles leading broadcast dimensions", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2], [2]) as WebGPUTensor;
      const e = webgpuBackend.ops.expand(a, [3, 2]) as WebGPUTensor;

      expect(e.buffer).toBe(a.buffer);
      expect(e.strides[0]).toBe(0); // Leading dim, broadcast
      expect(e.strides[1]).toBe(1); // Original dim

      const values = await webgpuBackend.ops.read(e);
      expect(values).toEqual([1, 2, 1, 2, 1, 2]);
    });
  });

  describe("strided elementwise ops", () => {
    it("add works on transposed tensors directly", async () => {
      // Create transposed view and add without calling contiguous()
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const aT = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 1 }); // [3, 2]

      const b = webgpuBackend.ops.tensorFromArray([10, 20, 30, 40, 50, 60], [3, 2]);

      const result = webgpuBackend.ops.add(aT, b);
      const values = await webgpuBackend.ops.read(result);

      // aT = [[1, 4], [2, 5], [3, 6]]
      // b  = [[10, 20], [30, 40], [50, 60]]
      // sum = [[11, 24], [32, 45], [53, 66]]
      expect(values).toEqual([11, 24, 32, 45, 53, 66]);
    });

    it("mul works on expanded tensors directly", async () => {
      // Create expanded view and multiply without calling contiguous()
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3], [1, 3]);
      const aE = webgpuBackend.ops.expand(a, [2, 3]); // Broadcast first dim

      const b = webgpuBackend.ops.tensorFromArray([10, 10, 10, 20, 20, 20], [2, 3]);

      const result = webgpuBackend.ops.mul(aE, b);
      const values = await webgpuBackend.ops.read(result);

      // aE = [[1, 2, 3], [1, 2, 3]]
      // b  = [[10, 10, 10], [20, 20, 20]]
      // mul = [[10, 20, 30], [20, 40, 60]]
      expect(values).toEqual([10, 20, 30, 20, 40, 60]);
    });

    it("sqrt works on transposed tensors directly", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 4, 9, 16], [2, 2]);
      const aT = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 1 }); // [2, 2]

      const result = webgpuBackend.ops.sqrt(aT);
      const values = await webgpuBackend.ops.read(result);

      // aT = [[1, 9], [4, 16]]
      // sqrt = [[1, 3], [2, 4]]
      expect(values[0]).toBeCloseTo(1);
      expect(values[1]).toBeCloseTo(3);
      expect(values[2]).toBeCloseTo(2);
      expect(values[3]).toBeCloseTo(4);
    });

    it("relu works on expanded tensors directly", async () => {
      const a = webgpuBackend.ops.tensorFromArray([-1, 2, -3], [1, 3]);
      const aE = webgpuBackend.ops.expand(a, [2, 3]);

      const result = webgpuBackend.ops.relu(aE);
      const values = await webgpuBackend.ops.read(result);

      // aE = [[-1, 2, -3], [-1, 2, -3]]
      // relu = [[0, 2, 0], [0, 2, 0]]
      expect(values).toEqual([0, 2, 0, 0, 2, 0]);
    });

    it("chained transpose + expand + add works", async () => {
      // Complex chain: transpose, then expand, then add
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const aT = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 1 }); // [2, 2] transposed

      const b = webgpuBackend.ops.tensorFromArray([10, 20], [1, 2]);
      const bE = webgpuBackend.ops.expand(b, [2, 2]); // Broadcast first dim

      const result = webgpuBackend.ops.add(aT, bE);
      const values = await webgpuBackend.ops.read(result);

      // aT = [[1, 3], [2, 4]] (transposed)
      // bE = [[10, 20], [10, 20]] (expanded)
      // sum = [[11, 23], [12, 24]]
      expect(values).toEqual([11, 23, 12, 24]);
    });
  });

  describe("permute", () => {
    it("permutes 3D tensor dimensions", async () => {
      // Shape [2, 3, 4] -> permute [2, 0, 1] -> [4, 2, 3]
      const data = Array.from({ length: 24 }, (_, i) => i);
      const a = webgpuBackend.ops.tensorFromArray(data, [2, 3, 4]);
      const p = webgpuBackend.ops.permute(a, [2, 0, 1]);

      expect(p.shape).toEqual([4, 2, 3]);
      const values = await webgpuBackend.ops.read(p);
      // Verify a few values match expected permutation
      // Original [0,0,0]=0, [0,0,1]=1, [0,0,2]=2, [0,0,3]=3
      // After permute [2,0,1]: [0,0,0]->original[0,0,0]=0, [1,0,0]->original[0,0,1]=1
      expect(values[0]).toBe(0);   // [0,0,0] -> orig[0,0,0]
      expect(values[6]).toBe(1);   // [1,0,0] -> orig[0,0,1]
      expect(values[12]).toBe(2);  // [2,0,0] -> orig[0,0,2]
      expect(values[18]).toBe(3);  // [3,0,0] -> orig[0,0,3]
    });

    it("permute returns view (shares buffer)", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const p = webgpuBackend.ops.permute(a, [1, 0]) as WebGPUTensor;

      expect(p.shape).toEqual([3, 2]);
      expect((a as WebGPUTensor).buffer).toBe(p.buffer);
    });

    it("identity permute returns original shape", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const p = webgpuBackend.ops.permute(a, [0, 1]);

      expect(p.shape).toEqual([2, 3]);
      const values = await webgpuBackend.ops.read(p);
      expect(values).toEqual([1, 2, 3, 4, 5, 6]);
    });

    it("permute is equivalent to transpose for 2D", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const transposed = webgpuBackend.ops.transpose(a, { dim0: 0, dim1: 1 });
      const permuted = webgpuBackend.ops.permute(a, [1, 0]);

      const tVals = await webgpuBackend.ops.read(transposed);
      const pVals = await webgpuBackend.ops.read(permuted);
      expect(pVals).toEqual(tVals);
    });

    it("permute with ops works correctly", async () => {
      // Create [2, 3] tensor, permute to [3, 2], add with broadcast
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3]);
      const p = webgpuBackend.ops.permute(a, [1, 0]); // [3, 2]
      const b = webgpuBackend.ops.tensorFromArray([10, 20], [1, 2]);
      const bE = webgpuBackend.ops.expand(b, [3, 2]);

      const result = webgpuBackend.ops.add(p, bE);
      const values = await webgpuBackend.ops.read(result);
      // p = [[1,4],[2,5],[3,6]], bE = [[10,20],[10,20],[10,20]]
      expect(values).toEqual([11, 24, 12, 25, 13, 26]);
    });

    it("throws for invalid dims length", () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      expect(() => webgpuBackend.ops.permute(a, [0, 1, 2])).toThrow(
        "dims length",
      );
    });

    it("throws for duplicate dims", () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      expect(() => webgpuBackend.ops.permute(a, [0, 0])).toThrow("duplicate");
    });

    it("throws for out of range dims", () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      expect(() => webgpuBackend.ops.permute(a, [0, 3])).toThrow("out of range");
    });
  });

  describe("where", () => {
    it("basic where selects from x or y based on condition", async () => {
      const condition = webgpuBackend.ops.tensorFromArray([1, 0, 1, 0], [4]);
      const x = webgpuBackend.ops.tensorFromArray([10, 20, 30, 40], [4]);
      const y = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [4]);

      const result = webgpuBackend.ops.where(condition, x, y);

      expect(result.shape).toEqual([4]);
      const values = await webgpuBackend.ops.read(result);
      expect(values).toEqual([10, 2, 30, 4]);
    });

    it("where with 2D tensors", async () => {
      const condition = webgpuBackend.ops.tensorFromArray([1, 0, 0, 1], [2, 2]);
      const x = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const y = webgpuBackend.ops.tensorFromArray([10, 20, 30, 40], [2, 2]);

      const result = webgpuBackend.ops.where(condition, x, y);

      expect(result.shape).toEqual([2, 2]);
      const values = await webgpuBackend.ops.read(result);
      expect(values).toEqual([1, 20, 30, 4]);
    });

    it("where with broadcasting condition", async () => {
      // condition [1] broadcasts to [3]
      const condition = webgpuBackend.ops.tensorFromArray([1], [1]);
      const x = webgpuBackend.ops.tensorFromArray([10, 20, 30], [3]);
      const y = webgpuBackend.ops.tensorFromArray([1, 2, 3], [3]);

      const result = webgpuBackend.ops.where(condition, x, y);

      expect(result.shape).toEqual([3]);
      const values = await webgpuBackend.ops.read(result);
      expect(values).toEqual([10, 20, 30]); // All from x since condition is 1
    });

    it("where with broadcasting x and y", async () => {
      const condition = webgpuBackend.ops.tensorFromArray([1, 0, 1], [3]);
      // x [1] broadcasts to [3]
      const x = webgpuBackend.ops.tensorFromArray([100], [1]);
      // y [1] broadcasts to [3]
      const y = webgpuBackend.ops.tensorFromArray([0], [1]);

      const result = webgpuBackend.ops.where(condition, x, y);

      expect(result.shape).toEqual([3]);
      const values = await webgpuBackend.ops.read(result);
      expect(values).toEqual([100, 0, 100]);
    });

    it("where all true selects all from x", async () => {
      const condition = webgpuBackend.ops.tensorFromArray([1, 1, 1, 1], [4]);
      const x = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [4]);
      const y = webgpuBackend.ops.tensorFromArray([10, 20, 30, 40], [4]);

      const result = webgpuBackend.ops.where(condition, x, y);
      const values = await webgpuBackend.ops.read(result);
      expect(values).toEqual([1, 2, 3, 4]);
    });

    it("where all false selects all from y", async () => {
      const condition = webgpuBackend.ops.tensorFromArray([0, 0, 0, 0], [4]);
      const x = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [4]);
      const y = webgpuBackend.ops.tensorFromArray([10, 20, 30, 40], [4]);

      const result = webgpuBackend.ops.where(condition, x, y);
      const values = await webgpuBackend.ops.read(result);
      expect(values).toEqual([10, 20, 30, 40]);
    });

    it("where treats any non-zero as true", async () => {
      const condition = webgpuBackend.ops.tensorFromArray([0.5, -1, 0, 5], [4]);
      const x = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [4]);
      const y = webgpuBackend.ops.tensorFromArray([10, 20, 30, 40], [4]);

      const result = webgpuBackend.ops.where(condition, x, y);
      const values = await webgpuBackend.ops.read(result);
      // 0.5 -> x, -1 -> x, 0 -> y, 5 -> x
      expect(values).toEqual([1, 2, 30, 4]);
    });
  });

  describe("0-d tensors (scalars)", () => {
    it("sum() without dim returns 0-d tensor", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const result = webgpuBackend.ops.sum(a);

      // Full reduction returns shape []
      expect(result.shape).toEqual([]);
      const values = await webgpuBackend.ops.read(result);
      expect(values).toEqual([10]);
    });

    it("mean() without dim returns 0-d tensor", async () => {
      const a = webgpuBackend.ops.tensorFromArray([2, 4, 6, 8], [2, 2]);
      const result = webgpuBackend.ops.mean(a);

      expect(result.shape).toEqual([]);
      const values = await webgpuBackend.ops.read(result);
      expect(values).toEqual([5]);
    });

    it("0-d tensor has size 1", async () => {
      const a = webgpuBackend.ops.tensorFromArray([1, 2, 3], [3]);
      const result = webgpuBackend.ops.sum(a);

      expect(result.shape).toEqual([]);
      expect(result.shape.reduce((a, b) => a * b, 1)).toBe(1); // Product of empty array is 1
    });

    it("0-d tensor can be used in binary ops via broadcast", async () => {
      const scalar = webgpuBackend.ops.sum(
        webgpuBackend.ops.tensorFromArray([5], [1]),
      ); // 0-d tensor with value 5
      const vec = webgpuBackend.ops.tensorFromArray([1, 2, 3], [3]);

      // Expand 0-d to [3] for addition
      const expanded = webgpuBackend.ops.expand(scalar, [3]);
      const result = webgpuBackend.ops.add(vec, expanded);

      const values = await webgpuBackend.ops.read(result);
      expect(values).toEqual([6, 7, 8]);
    });
  });
});
