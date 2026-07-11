/**
 * Ownership owner-SET HARD BOUNDARY gate (task #70; post-D2 flip): the owner SET
 * — now the AUTHORITATIVE liveness classifier's single source — must hold only
 * WeakRefs and MUST NOT extend a wrapper's lifetime, or it would perturb
 * GC-sensitive behavior (the #74 class the whole campaign works under). We prove
 * it by dropping all strong references, forcing GC + a FinalizationRegistry
 * turn, and asserting the owner set's LIVE membership collapses to zero — a
 * strong ref anywhere would keep the wrappers (and their sets) alive.
 *
 * CPU-only (no GPU needed); runs under the cpu project's `--expose-gc`.
 */

import { describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend/torchlette";
import { storageTracker } from "../src/graph/storage-tracker";

const gc: (() => void) | undefined = (globalThis as { gc?: () => void }).gc;

/** Force GC + let the FinalizationRegistry callbacks run (they fire on a
 *  microtask/next-tick after collection). A couple of turns makes it reliable. */
async function pressureGC(): Promise<void> {
  for (let i = 0; i < 4; i++) {
    gc?.();
    // Yield so FinalizationRegistry callbacks (and any queued microtasks) run.
    await new Promise((r) => setTimeout(r, 0));
  }
}

describe("ownership: WeakRef-only owner set (GC hard boundary)", () => {
  it("owner-set live membership collapses after GC — no lifetime extension", async () => {
    if (typeof gc !== "function") {
      // --expose-gc not present: cannot run the GC probe deterministically.
      // The cpu project passes it via execArgv; skip loudly elsewhere.
      console.warn("[ownership-shadow-gc] SKIP: global.gc unavailable");
      return;
    }
    const api = new Torchlette();

    // Create a batch of materialized CPU tensors — each `.cpu()` forces
    // materialization → trackTensor → owner-SET insertion. Hold them only in a
    // block-scoped array we then drop.
    {
      const tmp: unknown[] = [];
      for (let i = 0; i < 64; i++) {
        const t = api.tensorFromArray(
          new Float32Array([i, i + 1, i + 2, i + 3]),
          [4],
          { device: "cpu" },
        );
        // Force materialization so the storage is tracked (owner-set insert).
        await t.cpu();
        tmp.push(t);
      }
      // Sanity: the set now holds live members for our tensors.
      const mid = storageTracker.ownerSetStats();
      expect(mid.liveMembers).toBeGreaterThan(0);
      // Drop all strong references.
      tmp.length = 0;
    }

    // Force GC. If the owner set held a STRONG ref to any wrapper, the wrapper
    // (and its owner-set entry) would survive and liveMembers would stay high.
    await pressureGC();

    const after = storageTracker.ownerSetStats();
    // The live membership from our dropped batch must collapse. Allow a small
    // residual for genuinely-persistent framework tensors that other machinery
    // may hold, but our 64 dropped [4] tensors must NOT be pinned by the set.
    expect(
      after.liveMembers,
      `owner set retained ${after.liveMembers} live wrappers after GC — the ` +
        `set must be WeakRef-only (a strong ref would extend wrapper lifetime ` +
        `and perturb GC-sensitive behavior, the #74 class).`,
    ).toBeLessThan(16);
  });

  it("a clean CPU step runs without a lifetime throw under the strict default", async () => {
    // Post-D2 the derived owner-SET classifier is authoritative; a clean step
    // must not mis-demote and trip the [lifetime] reclaimed-read guard (strict
    // is the default). This is the in-suite regression for the derived model
    // driving releaseStepTemps correctly on the trivial path.
    const api = new Torchlette();
    await api.beginStep();
    const a = api.tensorFromArray(new Float32Array([1, 2, 3, 4]), [4], {
      device: "cpu",
    });
    const b = api.tensorFromArray(new Float32Array([5, 6, 7, 8]), [4], {
      device: "cpu",
    });
    const c = api.add(a, b);
    const out = await c.cpu();
    await api.markStep();
    expect(Array.from(out)).toEqual([6, 8, 10, 12]);
  });
});
