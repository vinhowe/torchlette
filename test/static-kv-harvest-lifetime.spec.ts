/**
 * Regression gate for the static-KV harvest-view reclaimed-read FALSE POSITIVE
 * (task #90 — the Gemma-2 SAE-demo `[lifetime] reading RECLAIMED storage
 * (shape [1,4,256,256])` warning in the static-KV decode path).
 *
 * Reproduces the seam WITHOUT the 5GB model: a persistent KV-style buffer,
 * updated in place every captured step via `copy_(kv, kv.scatterAdd(idx, x))`
 * (the model's kSlot update), then read through a `narrow` view (the attention
 * read). Under TORCHLETTE_STEP_TAPE=1 the compiled-plan harvest re-creates a
 * VIEW handle over the persistent `kv` base each replay (`planOwnedBaseRetain`);
 * the per-replay handle is reaped at markStep (rc 0) so `isDestroyed(viewId)` is
 * true even though its GPU buffer aliases the live base the plan keeps alive.
 * The scatterAdd `dst` external-leaf read then trips the [lifetime] reclaimed
 * guard — pre-fix, STRICT_LIFETIME=1 THROWS here; the read is provably safe
 * (view buffer === live base buffer) so the fix exonerates it.
 *
 * The replay path needs the flag ON (read at module load), so the gate self-
 * skips under the default suite and runs via:
 *   TORCHLETTE_STEP_TAPE=1 npx vitest run test/static-kv-harvest-lifetime.spec.ts
 *
 * Also mirrored as a standalone probe: tools/t-static-kv-harvest-lifetime.ts.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { initWebGPU } from "../src/backend/webgpu";
import { STEP_TAPE_REPLAY } from "../src/core/step-tape";
import type { Tensor } from "../src/frontend/tensor";
import { Torchlette } from "../src/frontend/torchlette";

let hasGPU = false;
beforeAll(async () => {
  hasGPU = await initWebGPU();
});

(STEP_TAPE_REPLAY ? describe : describe.skip)(
  "static-KV harvest-view lifetime (TORCHLETTE_STEP_TAPE=1)",
  () => {
    const H = 4;
    const S = 8;
    const D = 4;
    const NSTEPS = 6;

    async function runTrajectory(
      api: Torchlette,
      captured: boolean,
    ): Promise<{ sums: number[][]; hits: number }> {
      const idxFor = (pos: number): Tensor => {
        const arr = new Float32Array(H * 1 * D).fill(pos);
        return api.tensorFromArray(arr, [1, H, 1, D]);
      };
      const kv = api.persist(api.zeros([1, H, S, D]));
      const body = (x: Tensor): Tensor => {
        // In-place KV update (model's copy_(kSlot, kSlot.scatterAdd(...))).
        kv.copy_(kv.scatterAdd(idxFor(0), x, { dim: 2 }));
        // Read the accumulated KV back through a view (model's narrow read).
        return api.narrow(kv, 2, 0, S).sum([2]); // [1,H,D]
      };
      const decode = captured ? api.capture((x: Tensor) => body(x)) : null;
      const prev = api.setStepScopedCleanup(true);
      const sums: number[][] = [];
      try {
        await api.markStep();
        for (let t = 0; t < NSTEPS; t++) {
          const x = api.tensorFromArray(new Array(H * D).fill(t + 1), [
            1,
            H,
            1,
            D,
          ]);
          const out = decode
            ? ((await decode(x)) as Tensor)
            : api.noGrad(() => body(x));
          sums.push(Array.from(await api.cpu(out)));
          await api.markStep();
        }
      } finally {
        api.setStepScopedCleanup(prev);
      }
      return { sums, hits: decode ? decode.stats().hits : 0 };
    }

    it("captured static-KV decode == golden, with STRICT_LIFETIME (no reclaimed-read throw)", async () => {
      if (!hasGPU) return;
      // The reclaimed-read guard THROWS by default (task #73), so a reclaimed
      // read at the guard site is a hard failure here, not a swallowed warning —
      // no explicit TORCHLETTE_STRICT_LIFETIME=1 needed.
      const api = new Torchlette("webgpu", { enableFusion: true });
      const golden = await runTrajectory(api, false);
      const captured = await runTrajectory(api, true);
      // The harvest/replay path must actually activate, or the gate is vacuous.
      expect(captured.hits).toBeGreaterThan(0);
      // Any stale/reclaimed harvest read shows as a trajectory divergence
      // (and pre-fix the strict guard would have thrown before reaching here).
      for (let p = 0; p < NSTEPS; p++) {
        for (let i = 0; i < golden.sums[p].length; i++) {
          expect(
            Math.abs(captured.sums[p][i] - golden.sums[p][i]),
          ).toBeLessThan(1e-3);
        }
      }
    }, 120_000);
  },
);
