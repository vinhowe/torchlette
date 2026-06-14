/**
 * view-meta: the single-source view-metadata transforms (stage-4 phase 4.4).
 *
 * narrow/permute/transpose/expand in the WebGPU backend now COMPUTE their output
 * {shape,strides,offset} via these functions, and build-without-execution's
 * result-metadata derivation uses the same ones — so there's no second source to
 * drift. This spec locks the transforms to hand-computed expectations. (reshape
 * is additionally pinned to the backend by the build-without-execution
 * differential, which ran these against real plans at 0 diffs across every view
 * result; see executor.ts [ir-derive].)
 */
import { describe, expect, it } from "vitest";
import {
  expandMeta,
  narrowMeta,
  permuteMeta,
  reshapeMeta,
  transposeMeta,
} from "../src/backend/webgpu/ops/view-meta";

describe("view-meta transforms", () => {
  it("narrow shifts offset by start·stride[dim], shrinks the dim", () => {
    // [4,6] row-major: strides [6,1]. narrow(dim=1, start=2, length=3).
    const m = narrowMeta({ shape: [4, 6], strides: [6, 1], offset: 0 }, 1, 2, 3);
    expect(m.shape).toEqual([4, 3]);
    expect(m.strides).toEqual([6, 1]);
    expect(m.offset).toBe(2);
    // narrow on a tensor with an existing offset + dim 0.
    const m2 = narrowMeta({ shape: [4, 6], strides: [6, 1], offset: 5 }, 0, 1, 2);
    expect(m2.shape).toEqual([2, 6]);
    expect(m2.offset).toBe(5 + 1 * 6);
  });

  it("permute reorders shape + strides, keeps offset", () => {
    // [2,3,4] strides [12,4,1], permute [2,0,1].
    const m = permuteMeta(
      { shape: [2, 3, 4], strides: [12, 4, 1], offset: 7 },
      [2, 0, 1],
    );
    expect(m.shape).toEqual([4, 2, 3]);
    expect(m.strides).toEqual([1, 12, 4]);
    expect(m.offset).toBe(7);
  });

  it("transpose swaps two dims (= permute)", () => {
    const m = transposeMeta({ shape: [2, 3], strides: [3, 1], offset: 0 }, 0, 1);
    expect(m.shape).toEqual([3, 2]);
    expect(m.strides).toEqual([1, 3]);
  });

  it("expand: size-1 and leading dims get stride 0", () => {
    // [1,3] strides [3,1] → [4,3]: dim0 broadcast (stride 0), dim1 unchanged.
    const m = expandMeta({ shape: [1, 3], strides: [3, 1], offset: 0 }, [4, 3]);
    expect(m.shape).toEqual([4, 3]);
    expect(m.strides).toEqual([0, 1]);
    // leading dim not in input → stride 0.
    const m2 = expandMeta({ shape: [3], strides: [1], offset: 0 }, [2, 3]);
    expect(m2.strides).toEqual([0, 1]);
  });

  it("reshape of contiguous input: contiguous output, offset preserved, not materialized", () => {
    const m = reshapeMeta({ shape: [2, 6], strides: [6, 1], offset: 0 }, [3, 4]);
    expect(m.shape).toEqual([3, 4]);
    expect(m.strides).toEqual([4, 1]);
    expect(m.offset).toBe(0);
    expect(m.materialized).toBe(false);
  });

  it("reshape of incompatible non-contiguous input: materialize → contiguous, offset 0", () => {
    // A transposed [3,2] view (strides [1,3]) reshaped to [6] cannot be a view
    // (rows aren't contiguous in memory) → must materialize.
    const m = reshapeMeta({ shape: [3, 2], strides: [1, 3], offset: 0 }, [6]);
    expect(m.shape).toEqual([6]);
    expect(m.strides).toEqual([1]);
    expect(m.offset).toBe(0);
    expect(m.materialized).toBe(true);
  });
});
