/**
 * Top-K GPU prefilter correctness gate (readTopK, tile-IR kernel #65).
 *
 * The kernel returns the top-k (value, index) pairs of a 1-D f32 slice, sorted
 * (value desc, index asc). Tie-break rule: EQUAL values resolve to the SMALLER
 * index (a CPU linear scan taking the FIRST maximum). These tests pin the known
 * values/indices for odd sizes, ties, k=1, k=n, then differentially check
 * random inputs (including the full Qwen3 vocab size) against a JS reference.
 *
 * WebGPU-only (CPU has no readTopK backend op); auto-skips GPU-less (CI).
 */
import { beforeAll, describe, expect, it } from "vitest";

import { Torchlette } from "../src";
import { initWebGPU } from "../src/backend/webgpu";
import { cpuOnly } from "./helpers/webgpu";

/** JS reference: top-k over the slice, (value desc, index asc) ordering. */
function refTopK(flat: Float32Array, k: number) {
  const idx = Array.from({ length: flat.length }, (_, i) => i);
  idx.sort((a, b) => flat[b]! - flat[a]! || a - b);
  const top = idx.slice(0, k);
  return { values: top.map((i) => flat[i]!), indices: top };
}

describe.skipIf(cpuOnly)("readTopK (tile-IR GPU prefilter)", () => {
  const api = new Torchlette("webgpu");
  beforeAll(async () => {
    if (!(await initWebGPU())) throw new Error("WebGPU init failed");
  });

  async function topk(data: number[], k: number) {
    const t = api.tensorFromArray(data, [data.length]);
    const r = await api.readTopK(t, k);
    t.dispose();
    return r;
  }

  it("odd size, distinct values", async () => {
    const data = [3, 1, 4, 1.5, 9, 2, 6, 5, 3.5]; // n=9
    const r = await topk(data, 3);
    expect(Array.from(r.indices)).toEqual([4, 6, 7]); // 9, 6, 5
    expect(Array.from(r.values)).toEqual([9, 6, 5]);
  });

  it("ties resolve to the smaller index", async () => {
    // values 5 at indices 0,1,2,4,6 -> pick the four smallest indices.
    const r = await topk([5, 5, 5, 2, 5, 1, 5], 4);
    expect(Array.from(r.indices)).toEqual([0, 1, 2, 4]);
    expect(Array.from(r.values)).toEqual([5, 5, 5, 5]);
  });

  it("all-equal values -> ascending indices", async () => {
    const r = await topk([7, 7, 7, 7, 7], 3);
    expect(Array.from(r.indices)).toEqual([0, 1, 2]);
  });

  it("k=1 greedy argmax picks the FIRST maximum on a tie", async () => {
    const r = await topk([2, 9, 3, 9, 1], 1); // max 9 at 1 and 3 -> 1
    expect(Array.from(r.indices)).toEqual([1]);
    expect(Array.from(r.values)).toEqual([9]);
  });

  it("k=n full descending sort", async () => {
    const data = [3, 1, 4, 1, 5, 9, 2, 6];
    const r = await topk(data, data.length);
    const ref = refTopK(new Float32Array(data), data.length);
    expect(Array.from(r.indices)).toEqual(ref.indices);
    expect(Array.from(r.values)).toEqual(ref.values);
  });

  it("differential vs reference across sizes / k / ties (incl. full vocab)", async () => {
    let rng = 987654321;
    const rand = () => {
      rng = (rng * 1103515245 + 12345) & 0x7fffffff;
      return rng / 0x7fffffff;
    };
    for (const n of [7, 63, 256, 257, 1024, 5000, 151936]) {
      for (const k of [1, 8, 64]) {
        if (k > n) continue;
        const data: number[] = [];
        for (let i = 0; i < n; i++) data.push((rand() - 0.5) * 20);
        if (n > 10) data[3] = data[7] = data[n - 2] = 9.5; // exact ties
        const flat = new Float32Array(data);
        const r = await topk(Array.from(flat), k);
        const ref = refTopK(flat, k);
        expect(Array.from(r.indices), `indices n=${n} k=${k}`).toEqual(
          ref.indices,
        );
        expect(Array.from(r.values), `values n=${n} k=${k}`).toEqual(
          ref.values,
        );
      }
    }
  });
});
