/**
 * Arg-reduce over a STRIDED / OFFSET view must return the correct index.
 *
 * The WebGPU arg-reduce kernel derives its addressing from
 * `contiguousStrides(inputShape)` and binds the input buffer flat from element
 * 0. A narrow view carries a non-zero storage offset (and, for a multi-dim
 * narrow, non-contiguous strides), so without a materialization guard the kernel
 * reads the WRONG region and returns the WRONG index.
 *
 * This is the sharp edge named in docs/unrolled-k-decode-design.md §2 Probe 1:
 * argmax over a doubly-narrowed decode logits row returned 40 vs the correct 995.
 * Here it is reproduced deterministically at small scale. The CPU backend indexes
 * views correctly in JS, so the WebGPU arm is the real gate.
 *
 * Fix: argReduceOp forces the input contiguous (offset 0) before dispatch —
 * mirroring the sum/max reduction guard — with a seam assertion that the kernel's
 * contiguous-stride assumption holds.
 */
import { beforeAll, describe, expect, it } from "vitest";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

let hasGPU = false;
beforeAll(async () => {
  hasGPU = await initWebGPU();
});

describe("argmax over strided/offset views (WebGPU)", () => {
  it("argmax over an OFFSET view (last row of a multi-row tensor) is correct", async () => {
    if (!hasGPU) return;
    const api = new Torchlette("webgpu");
    // [1,3,8]: row 0 peaks at idx 2, row 1 at idx 5, row 2 at idx 7.
    // The flat-from-0 (buggy) kernel would read row 0 and return 2.
    const data = [
      0, 0, 9, 0, 0, 0, 0, 0, // row 0 -> argmax 2
      0, 0, 0, 0, 0, 9, 0, 0, // row 1 -> argmax 5
      0, 0, 0, 0, 0, 0, 0, 9, // row 2 -> argmax 7
    ];
    const t = api.tensorFromArray(data, [1, 3, 8]);
    const lastRow = api.narrow(t, 1, 2, 1); // [1,1,8], storage offset = 16
    const idx = api.argmax(lastRow, { dim: -1, keepdim: false });
    const out = Array.from(await api.cpu(idx));
    expect(out[0]).toBe(7);
  });

  it("argmax over a DOUBLY-narrowed view (offset + strided) is correct", async () => {
    if (!hasGPU) return;
    const api = new Torchlette("webgpu");
    // [1,2,6] with a "padded vocab": real vocab is the first V=4 columns.
    // row 0 real-vocab peak at 1; row 1 real-vocab peak at 3 (col 5 is padding).
    const data = [
      0, 9, 0, 0, 5, 5, // row 0: within first 4 -> argmax 1
      0, 0, 0, 9, 8, 8, // row 1: within first 4 -> argmax 3
    ];
    const V = 4;
    const t = api.tensorFromArray(data, [1, 2, 6]);
    const lastRow = api.narrow(t, 1, 1, 1); // [1,1,6] offset view
    const realVocab = api.narrow(lastRow, 2, 0, V); // [1,1,4] strided+offset view
    const idx = api.argmax(realVocab, { dim: -1, keepdim: false });
    const out = Array.from(await api.cpu(idx));
    expect(out[0]).toBe(3);
  });

  it("argmax over a contiguous tensor is unchanged", async () => {
    if (!hasGPU) return;
    const api = new Torchlette("webgpu");
    const t = api.tensorFromArray([1, 7, 3, 2, 5], [5]);
    const idx = api.argmax(t, { dim: -1, keepdim: false });
    const out = Array.from(await api.cpu(idx));
    expect(out[0]).toBe(1);
  });
});
