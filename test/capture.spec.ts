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
import { STEP_TAPE_REPLAY } from "../src/core/step-tape";

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

// Full replay behaviors (need the flag ON — read at module load).
(STEP_TAPE_REPLAY ? describe : describe.skip)(
  "capture() — derived-coverage replay (TORCHLETTE_STEP_TAPE=1)",
  () => {
    // A small recurring tensor fn: elementwise + matmul, seqLen-1 shaped so it
    // compiles a stable plan and the recorder deems consecutive steps eligible.
    function makeApi() {
      return new Torchlette("webgpu", { enableFusion: true });
    }
    // Drive N calls of a captured fn with fresh per-call upload built INSIDE fn
    // (so it exercises the upload interceptor + skeleton dressing).
    async function drive(
      api: Torchlette,
      f: {
        (x: import("../src/frontend/tensor").Tensor): Promise<
          import("../src/frontend/tensor").Tensor | import("../src/frontend/tensor").Tensor[]
        >;
        stats(): { hits: number; traces: number; calls: number };
      },
      vals: number[][],
    ): Promise<number[][]> {
      const outs: number[][] = [];
      const prev = api.setStepScopedCleanup(true);
      try {
        for (const v of vals) {
          const x = api.tensorFromArray(v, [1, v.length]);
          const out = (await f(x)) as import("../src/frontend/tensor").Tensor;
          outs.push(Array.from(await api.cpu(out)));
          await api.markStep();
        }
      } finally {
        api.setStepScopedCleanup(prev);
      }
      return outs;
    }

    it("replays after warmup; tensor-arg change re-uploads (same tape)", async () => {
      if (!hasGPU) return;
      const api = makeApi();
      const w = api.persist(api.tensorFromArray([1, 0, 0, 1], [2, 2]));
      // fn builds a fresh per-step upload internally (a [1,1] scale — a shape
      // DISTINCT from the [1,2] arg so the shape-keyed skeleton dressing is
      // unambiguous, the phase-1 constraint) then combines it with the arg.
      const f = api.capture((x: import("../src/frontend/tensor").Tensor) => {
        const scale = api.tensorFromArray([1], [1, 1]); // per-step upload slot
        return api.mul(api.matmul(x, w), scale);
      });
      const inputs = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
        [9, 10],
        [11, 12],
        [13, 14],
        [15, 16],
      ];
      const outs = await drive(api, f, inputs);
      expect(outs).toEqual(inputs);
      // Warmup: first ~2 steps trace/record, later steps replay. At least one hit.
      expect(f.stats().hits).toBeGreaterThan(0);
    });

    it("plain-value arg change ⇒ COUNTED cold miss + correct re-record", async () => {
      if (!hasGPU) return;
      const api = makeApi();
      const w = api.persist(api.tensorFromArray([1, 0, 0, 1], [2, 2]));
      // α as a PLAIN-VALUE ARG — the cold knob: its value is hashed into the
      // bucket key, so a change is a counted miss + re-record, never stale.
      const f = api.capture(
        (x: import("../src/frontend/tensor").Tensor, alpha: number) =>
          api.mul(api.matmul(x, w), alpha),
      );
      const prev = api.setStepScopedCleanup(true);
      const run = async (v: number[], alpha: number) => {
        const x = api.tensorFromArray(v, [1, v.length]);
        const out = (await f(x, alpha)) as import("../src/frontend/tensor").Tensor;
        const arr = Array.from(await api.cpu(out));
        await api.markStep();
        return arr;
      };
      try {
        // Warm up at alpha=2 → skeleton ready, hits accrue.
        for (let i = 0; i < 7; i++) await run([1, 1], 2);
        expect(f.stats().hits).toBeGreaterThan(0);
        const cold0 = f.stats().coldMisses;
        // Change alpha WITHOUT invalidate(): new bucket ⇒ counted cold miss,
        // and the output reflects the NEW alpha (re-record), bit-exact.
        expect(await run([1, 1], -5)).toEqual([-5, -5]);
        expect(f.stats().coldMisses).toBeGreaterThan(cold0);
        // Back to 2: its bucket is still warm-able (re-trace or hit, correct).
        expect(await run([1, 1], 2)).toEqual([2, 2]);
      } finally {
        api.setStepScopedCleanup(prev);
      }
    });

    it("tensor arg is a WARM slot: value change replays with fresh data", async () => {
      if (!hasGPU) return;
      const api = makeApi();
      const w = api.persist(api.tensorFromArray([1, 0, 0, 1], [2, 2]));
      // α as a TENSOR ARG — the warm knob: fresh value dressed every call.
      const f = api.capture(
        (
          x: import("../src/frontend/tensor").Tensor,
          alphaT: import("../src/frontend/tensor").Tensor,
        ) => api.mul(api.matmul(x, w), alphaT),
      );
      const prev = api.setStepScopedCleanup(true);
      const run = async (alpha: number) => {
        const x = api.tensorFromArray([1, 1], [1, 2]);
        const a = api.tensorFromArray([alpha], [1, 1]);
        const out = (await f(x, a)) as import("../src/frontend/tensor").Tensor;
        const arr = Array.from(await api.cpu(out));
        await api.markStep();
        return arr;
      };
      try {
        for (let i = 0; i < 7; i++) await run(2);
        const hits0 = f.stats().hits;
        const cold0 = f.stats().coldMisses;
        expect(hits0).toBeGreaterThan(0);
        // α change on the SAME warm tape: fresh slot data, zero cold misses.
        expect(await run(-5)).toEqual([-5, -5]);
        expect(f.stats().hits).toBeGreaterThan(hits0);
        expect(f.stats().coldMisses).toBe(cold0);
      } finally {
        api.setStepScopedCleanup(prev);
      }
    });

    it("closure values are FROZEN at record time (documented contract)", async () => {
      if (!hasGPU) return;
      const api = makeApi();
      const w = api.persist(api.tensorFromArray([1, 0, 0, 1], [2, 2]));
      let alpha = 2; // closure-captured — FROZEN per the arg-boundary contract
      const f = api.capture((x: import("../src/frontend/tensor").Tensor) =>
        api.mul(api.matmul(x, w), alpha),
      );
      await drive(api, f, [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]);
      expect(f.stats().hits).toBeGreaterThan(0);
      // Change the closure WITHOUT informing capture: the documented behavior
      // is FROZEN — replays keep the recorded α=2 (jax.jit semantics; pass
      // varying values as ARGS instead). This asserts the contract, and that
      // the contract is stated in the API surface.
      alpha = -5;
      const out = await drive(api, f, [[1, 1]]);
      expect(out[0]).toEqual([2, 2]); // frozen recorded value, NOT -5
      const src = readFileSync(
        new URL("../src/frontend/capture.ts", import.meta.url),
        "utf8",
      );
      expect(src).toMatch(/CLOSURE-CAPTURED VALUES ARE FROZEN AT RECORD TIME/);
      expect(src).toMatch(/PASS ANYTHING THAT VARIES AS AN ARGUMENT/i);
    });

    it("invalidate() forces a re-record", async () => {
      if (!hasGPU) return;
      const api = makeApi();
      const w = api.persist(api.tensorFromArray([1, 0, 0, 1], [2, 2]));
      const f = api.capture((x: import("../src/frontend/tensor").Tensor) =>
        api.matmul(x, w),
      );
      await drive(api, f, [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]);
      expect(f.stats().hits).toBeGreaterThan(0);
      const tracesBefore = f.stats().traces;
      f.invalidate();
      await drive(api, f, [[2, 2], [2, 2]]);
      // A re-record happened after invalidate (more traces than before).
      expect(f.stats().traces).toBeGreaterThan(tracesBefore);
    });

    it("output ring: reading past the K-window is a LOUD, step-naming error", async () => {
      if (!hasGPU) return;
      const api = makeApi();
      const w = api.persist(api.tensorFromArray([1, 0, 0, 1], [2, 2]));
      const f = api.capture(
        (x: import("../src/frontend/tensor").Tensor) => api.matmul(x, w),
        { ringDepth: 2 },
      );
      const prev = api.setStepScopedCleanup(true);
      const held: import("../src/frontend/tensor").Tensor[] = [];
      try {
        // K+1 = 3 calls without reading → the oldest falls out of the window.
        for (let i = 0; i < 3; i++) {
          const x = api.tensorFromArray([i, i], [1, 2]);
          held.push(
            (await f(x)) as import("../src/frontend/tensor").Tensor,
          );
          await api.markStep();
        }
      } finally {
        api.setStepScopedCleanup(prev);
      }
      // The step-0 output is past the 2-step window → reading it throws loudly.
      await expect(api.cpu(held[0])).rejects.toThrow(/capture.*step 0/);
      // The most-recent output is within the window → still valid.
      expect(Array.from(await api.cpu(held[2]))).toEqual([2, 2]);
    });

    it("stats() counters are sane", async () => {
      if (!hasGPU) return;
      const api = makeApi();
      const w = api.persist(api.tensorFromArray([1, 0, 0, 1], [2, 2]));
      const f = api.capture((x: import("../src/frontend/tensor").Tensor) =>
        api.matmul(x, w),
      );
      await drive(api, f, [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]);
      const s = f.stats();
      expect(s.calls).toBe(7);
      expect(s.traces + s.hits).toBeLessThanOrEqual(s.calls + 1);
      expect(s.hits).toBeGreaterThan(0);
    });
  },
);
