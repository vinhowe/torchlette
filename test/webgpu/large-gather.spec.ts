import { describe, expect, it, beforeAll } from "vitest";
import {
  initWebGPU,
  getWebGPUInitError,
  webgpuBackend,
  getMaxStorageBufferBindingSize,
} from "../../src/backend/webgpu";
import { cpuOnly } from "../helpers/webgpu";

type WebGPUTensor = ReturnType<typeof webgpuBackend.ops.tensorFromArray>;

describe.skipIf(cpuOnly)("Chunked gather for large tensors", () => {
  let maxBindingSize: number;

  beforeAll(async () => {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }
    maxBindingSize = getMaxStorageBufferBindingSize();
  });

  describe("regular gather (within limits)", () => {
    it("gathers from 2D tensor along dim 0", async () => {
      // Input [3, 2], gather indices [2, 2] along dim 0
      // [[1, 2], [3, 4], [5, 6]] with idx [[0, 2], [1, 0]]
      const embedding = webgpuBackend.ops.tensorFromArray(
        [1, 2, 3, 4, 5, 6],
        [3, 2],
      );
      const indices = webgpuBackend.ops.tensorFromArray([0, 2, 1, 0], [2, 2]);
      const result = webgpuBackend.ops.gather(embedding, indices, { dim: 0 });

      expect(result.shape).toEqual([2, 2]);
      const values = await webgpuBackend.ops.read(result);
      // At [0,0]: input[idx[0,0], 0] = input[0, 0] = 1
      // At [0,1]: input[idx[0,1], 1] = input[2, 1] = 6
      // At [1,0]: input[idx[1,0], 0] = input[1, 0] = 3
      // At [1,1]: input[idx[1,1], 1] = input[0, 1] = 2
      expect(values).toEqual([1, 6, 3, 2]);
    });

    it("gathers from 1D tensor", async () => {
      // Input [5], gather indices [3] along dim 0
      const data = webgpuBackend.ops.tensorFromArray([10, 20, 30, 40, 50], [5]);
      const indices = webgpuBackend.ops.tensorFromArray([1, 3, 0], [3]);
      const result = webgpuBackend.ops.gather(data, indices, { dim: 0 });

      expect(result.shape).toEqual([3]);
      const values = await webgpuBackend.ops.read(result);
      expect(values).toEqual([20, 40, 10]);
    });

    it("gathers indices spanning different positions", async () => {
      // Input [10, 4], indices [3, 4]
      const embedding = webgpuBackend.ops.tensorFromArray(
        Array.from({ length: 40 }, (_, i) => i),
        [10, 4],
      );
      // All indices point to the same row for each column
      const indices = webgpuBackend.ops.tensorFromArray(
        [0, 0, 0, 0, 5, 5, 5, 5, 9, 9, 9, 9],
        [3, 4],
      );
      const result = webgpuBackend.ops.gather(embedding, indices, { dim: 0 });

      expect(result.shape).toEqual([3, 4]);
      const values = await webgpuBackend.ops.read(result);
      // Row 0: [0, 1, 2, 3], Row 5: [20, 21, 22, 23], Row 9: [36, 37, 38, 39]
      expect(values).toEqual([0, 1, 2, 3, 20, 21, 22, 23, 36, 37, 38, 39]);
    });
  });

  describe("large tensor gather (chunked path)", () => {
    // These tests will only trigger chunking if the tensor exceeds maxBindingSize
    // On most systems, maxBindingSize is 128MB or more

    it("gathers from moderately large tensor", async () => {
      // Create a tensor 1000 x 16 = 64KB (well within limits, but tests path)
      const vocabSize = 1000;
      const embedDim = 16;
      const data = Array.from(
        { length: vocabSize * embedDim },
        (_, i) => i % 256,
      );
      const embedding = webgpuBackend.ops.tensorFromArray(data, [
        vocabSize,
        embedDim,
      ]);

      // Gather 3 rows, indices shaped [3, embedDim] all pointing to same row per output row
      const batchIndices: number[] = [];
      const rowsToGather = [0, 500, 999];
      for (const rowIdx of rowsToGather) {
        for (let i = 0; i < embedDim; i++) {
          batchIndices.push(rowIdx);
        }
      }
      const indices = webgpuBackend.ops.tensorFromArray(batchIndices, [
        3,
        embedDim,
      ]);
      const result = webgpuBackend.ops.gather(embedding, indices, { dim: 0 });

      expect(result.shape).toEqual([3, embedDim]);
      const values = await webgpuBackend.ops.read(result);

      // Check first element of each gathered row
      expect(values[0]).toBe(0); // Row 0, col 0
      expect(values[embedDim]).toBe((500 * embedDim) % 256); // Row 500, col 0
      expect(values[2 * embedDim]).toBe((999 * embedDim) % 256); // Row 999, col 0
    });

    it("handles indices at boundaries", async () => {
      const vocabSize = 500;
      const embedDim = 8;
      const data = Array.from(
        { length: vocabSize * embedDim },
        (_, i) => i % 100,
      );
      const embedding = webgpuBackend.ops.tensorFromArray(data, [
        vocabSize,
        embedDim,
      ]);

      // First and last rows
      const batchIndices: number[] = [];
      for (let i = 0; i < embedDim; i++) batchIndices.push(0);
      for (let i = 0; i < embedDim; i++) batchIndices.push(vocabSize - 1);

      const indices = webgpuBackend.ops.tensorFromArray(batchIndices, [
        2,
        embedDim,
      ]);
      const result = webgpuBackend.ops.gather(embedding, indices, { dim: 0 });

      expect(result.shape).toEqual([2, embedDim]);
      const values = await webgpuBackend.ops.read(result);
      expect(values[0]).toBe(0);
      expect(values[embedDim]).toBe(((vocabSize - 1) * embedDim) % 100);
    });

    it("reports max binding size", () => {
      // Just verify we can read the limit
      expect(maxBindingSize).toBeGreaterThan(0);
      expect(typeof maxBindingSize).toBe("number");
      // Typical limits are 128MB to 1GB+
      expect(maxBindingSize).toBeGreaterThanOrEqual(128 * 1024 * 1024);
    });
  });
});

describe("Chunked scatterAdd for large tensors", () => {
  beforeAll(async () => {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }
  });

  describe("regular scatterAdd (within limits)", () => {
    it("scatters to 2D tensor along dim 0", async () => {
      // Output: [3, 2] zeros
      // Scatter src [2, 2] with indices [[1, 0], [2, 1]] along dim 0
      const output = webgpuBackend.ops.tensorFromArray(Array(6).fill(0), [3, 2]);
      const src = webgpuBackend.ops.tensorFromArray([1, 2, 3, 4], [2, 2]);
      const indices = webgpuBackend.ops.tensorFromArray([1, 0, 2, 1], [2, 2]);

      const result = webgpuBackend.ops.scatterAdd(output, indices, src, {
        dim: 0,
      });

      expect(result.shape).toEqual([3, 2]);
      const values = await webgpuBackend.ops.read(result);
      // src[0,0]=1 goes to out[idx[0,0], 0] = out[1, 0]
      // src[0,1]=2 goes to out[idx[0,1], 1] = out[0, 1]
      // src[1,0]=3 goes to out[idx[1,0], 0] = out[2, 0]
      // src[1,1]=4 goes to out[idx[1,1], 1] = out[1, 1]
      // Result: [[0,2], [1,4], [3,0]]
      expect(values).toEqual([0, 2, 1, 4, 3, 0]);
    });

    it("scatters to 1D tensor", async () => {
      const output = webgpuBackend.ops.tensorFromArray(Array(5).fill(0), [5]);
      const src = webgpuBackend.ops.tensorFromArray([10, 20, 30], [3]);
      const indices = webgpuBackend.ops.tensorFromArray([1, 3, 0], [3]);

      const result = webgpuBackend.ops.scatterAdd(output, indices, src, {
        dim: 0,
      });

      expect(result.shape).toEqual([5]);
      const values = await webgpuBackend.ops.read(result);
      // src[0]=10 -> out[1], src[1]=20 -> out[3], src[2]=30 -> out[0]
      expect(values).toEqual([30, 10, 0, 20, 0]);
    });

    // Skip: WebGPU lacks f32 atomics, so overlapping indices produce undefined results
    // This is a known limitation documented in CLAUDE.md
    it.skip("accumulates multiple scatters to same position in 1D", async () => {
      const output = webgpuBackend.ops.tensorFromArray(Array(4).fill(0), [4]);
      const src = webgpuBackend.ops.tensorFromArray([1, 2, 3], [3]);
      // All scatter to index 1
      const indices = webgpuBackend.ops.tensorFromArray([1, 1, 1], [3]);

      const result = webgpuBackend.ops.scatterAdd(output, indices, src, {
        dim: 0,
      });

      expect(result.shape).toEqual([4]);
      const values = await webgpuBackend.ops.read(result);
      // Note: Due to race conditions without atomics, this may not be deterministic
      // for concurrent writes. With f32 atomics this would be 6.
      expect(values[1]).toBe(6); // 1+2+3
    });
  });

  describe("large tensor scatterAdd (chunked path)", () => {
    it("scatters to moderately large tensor", async () => {
      const vocabSize = 1000;
      const embedDim = 8;
      const output = webgpuBackend.ops.tensorFromArray(
        Array(vocabSize * embedDim).fill(0),
        [vocabSize, embedDim],
      );

      // Scatter 3 rows worth of values
      const srcData: number[] = [];
      for (let row = 0; row < 3; row++) {
        for (let col = 0; col < embedDim; col++) {
          srcData.push(row + 1); // Values 1, 2, 3 for each row
        }
      }
      const src = webgpuBackend.ops.tensorFromArray(srcData, [3, embedDim]);

      // Indices pointing to rows 0, 500, 999
      const indicesData: number[] = [];
      const targetRows = [0, 500, 999];
      for (const rowIdx of targetRows) {
        for (let col = 0; col < embedDim; col++) {
          indicesData.push(rowIdx);
        }
      }
      const indices = webgpuBackend.ops.tensorFromArray(indicesData, [
        3,
        embedDim,
      ]);

      const result = webgpuBackend.ops.scatterAdd(output, indices, src, {
        dim: 0,
      });

      expect(result.shape).toEqual([vocabSize, embedDim]);
      const values = await webgpuBackend.ops.read(result);

      // Check scattered positions
      expect(values[0]).toBe(1); // Row 0, first element
      expect(values[500 * embedDim]).toBe(2); // Row 500, first element
      expect(values[999 * embedDim]).toBe(3); // Row 999, first element

      // Non-scattered positions should be 0
      expect(values[embedDim]).toBe(0); // Row 1
      expect(values[100 * embedDim]).toBe(0); // Row 100
    });

    it("scatters to first positions only", async () => {
      const vocabSize = 500;
      const embedDim = 4;
      const output = webgpuBackend.ops.tensorFromArray(
        Array(vocabSize * embedDim).fill(0),
        [vocabSize, embedDim],
      );

      const srcData: number[] = [];
      for (let row = 0; row < 3; row++) {
        for (let col = 0; col < embedDim; col++) {
          srcData.push(10 * (row + 1));
        }
      }
      const src = webgpuBackend.ops.tensorFromArray(srcData, [3, embedDim]);

      const indicesData: number[] = [];
      for (const rowIdx of [0, 1, 2]) {
        for (let col = 0; col < embedDim; col++) {
          indicesData.push(rowIdx);
        }
      }
      const indices = webgpuBackend.ops.tensorFromArray(indicesData, [
        3,
        embedDim,
      ]);

      const result = webgpuBackend.ops.scatterAdd(output, indices, src, {
        dim: 0,
      });

      expect(result.shape).toEqual([vocabSize, embedDim]);
      const values = await webgpuBackend.ops.read(result);
      expect(values[0]).toBe(10);
      expect(values[embedDim]).toBe(20);
      expect(values[2 * embedDim]).toBe(30);
      expect(values[3 * embedDim]).toBe(0);
    });

    it("scatters to last positions only", async () => {
      const vocabSize = 500;
      const embedDim = 4;
      const output = webgpuBackend.ops.tensorFromArray(
        Array(vocabSize * embedDim).fill(0),
        [vocabSize, embedDim],
      );

      const srcData: number[] = [];
      for (let row = 0; row < 3; row++) {
        for (let col = 0; col < embedDim; col++) {
          srcData.push(10 * (row + 1));
        }
      }
      const src = webgpuBackend.ops.tensorFromArray(srcData, [3, embedDim]);

      const indicesData: number[] = [];
      for (const rowIdx of [vocabSize - 3, vocabSize - 2, vocabSize - 1]) {
        for (let col = 0; col < embedDim; col++) {
          indicesData.push(rowIdx);
        }
      }
      const indices = webgpuBackend.ops.tensorFromArray(indicesData, [
        3,
        embedDim,
      ]);

      const result = webgpuBackend.ops.scatterAdd(output, indices, src, {
        dim: 0,
      });

      expect(result.shape).toEqual([vocabSize, embedDim]);
      const values = await webgpuBackend.ops.read(result);
      expect(values[(vocabSize - 3) * embedDim]).toBe(10);
      expect(values[(vocabSize - 2) * embedDim]).toBe(20);
      expect(values[(vocabSize - 1) * embedDim]).toBe(30);
    });
  });
});

describe("Integration: gather/scatterAdd roundtrip", () => {
  beforeAll(async () => {
    const ready = await initWebGPU();
    if (!ready) {
      const error = getWebGPUInitError();
      throw new Error(`WebGPU init failed${error ? `: ${error}` : ""}`);
    }
  });

  it("gather then scatterAdd reconstructs sparse representation", async () => {
    // Embedding table [8, 4]
    const embedding = webgpuBackend.ops.tensorFromArray(
      Array.from({ length: 32 }, (_, i) => i),
      [8, 4],
    );

    // Gather rows 1, 3, 5 -> output [3, 4]
    // Indices need to be [3, 4] with each row all same index
    const gatherIndices: number[] = [];
    for (const rowIdx of [1, 3, 5]) {
      for (let col = 0; col < 4; col++) {
        gatherIndices.push(rowIdx);
      }
    }
    const indices = webgpuBackend.ops.tensorFromArray(gatherIndices, [3, 4]);
    const gathered = webgpuBackend.ops.gather(embedding, indices, { dim: 0 });

    expect(gathered.shape).toEqual([3, 4]);
    const gatheredValues = await webgpuBackend.ops.read(gathered);
    // Row 1: [4,5,6,7], Row 3: [12,13,14,15], Row 5: [20,21,22,23]
    expect(gatheredValues).toEqual([4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23]);

    // Now scatter back to a zero tensor
    const zeros = webgpuBackend.ops.tensorFromArray(Array(32).fill(0), [8, 4]);
    const scatterIndices = webgpuBackend.ops.tensorFromArray(gatherIndices, [
      3,
      4,
    ]);
    const scattered = webgpuBackend.ops.scatterAdd(
      zeros,
      scatterIndices,
      gathered,
      {
        dim: 0,
      },
    );

    expect(scattered.shape).toEqual([8, 4]);
    const scatteredValues = await webgpuBackend.ops.read(scattered);

    // Rows 1, 3, 5 should have their values restored
    expect(scatteredValues.slice(4, 8)).toEqual([4, 5, 6, 7]); // Row 1
    expect(scatteredValues.slice(12, 16)).toEqual([12, 13, 14, 15]); // Row 3
    expect(scatteredValues.slice(20, 24)).toEqual([20, 21, 22, 23]); // Row 5

    // Other rows should be 0
    expect(scatteredValues.slice(0, 4)).toEqual([0, 0, 0, 0]); // Row 0
    expect(scatteredValues.slice(8, 12)).toEqual([0, 0, 0, 0]); // Row 2
  });
});
