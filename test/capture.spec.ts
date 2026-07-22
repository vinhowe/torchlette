/**
 * capture() phase-2a API unit spec (docs/staged-execution-phase2a.md).
 *
 * The replay behaviors require TORCHLETTE_STEP_TAPE=1 (read at module load).
 * Under the default suite (flag off) capture() is a transparent pass-through,
 * and only the flag-independent behaviors run; the flag-gated block exercises
 * the derived-coverage core (the G3-class miss is in test/taped-decode-gates
 * via the driver, but the SCALAR-CHANGE-⇒-miss unit is here). Run the full
 * spec with:  TORCHLETTE_STEP_TAPE=1 npx vitest run test/capture.spec.ts
 */

import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

let hasGPU = false;
beforeAll(async () => {
  hasGPU = await initWebGPU();
});

describe("capture() — flag-independent surface", () => {
  it("is a transparent pass-through when the tape flag is off (and correct on)", async () => {
    if (!hasGPU) return;
    const api = new Torchlette("webgpu", { enableFusion: true });
    const w = api.persist(api.tensorFromArray([2, 0, 0, 3], [2, 2]));
    const f = api.capture((x: import("../src/frontend/tensor").Tensor) =>
      api.matmul(x, w),
    );
    const x = api.tensorFromArray([1, 1, 1, 1], [2, 2]);
    const out = (await f(x)) as import("../src/frontend/tensor").Tensor;
    // [[1,1],[1,1]] @ [[2,0],[0,3]] = [[2,3],[2,3]]
    expect(Array.from(await api.cpu(out))).toEqual([2, 3, 2, 3]);
    expect(typeof f.invalidate).toBe("function");
    expect(typeof f.stats).toBe("function");
    expect(f.stats().calls).toBe(1);
  });
});
