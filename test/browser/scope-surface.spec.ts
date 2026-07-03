/**
 * Browser gate for the async scope() surface (docs/scoped-memory-design.md §4).
 *
 * Proves the scope mechanism works IDENTICALLY in a real browser (Playwright /
 * native WebGPU) — i.e. NO Node-only async API (async_hooks / AsyncLocalStorage
 * / AsyncContext) leaked into the implementation. The whole mechanism is a
 * synchronous module-level scope stack, so it must run here unchanged.
 *
 * Exercises create+reclaim, structural escape, and the single-flight
 * overlap-throw.
 */
import { beforeAll, describe, expect, it } from "vitest";
import { initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { storageTracker } from "../../src/graph/storage-tracker";

const api = new Torchlette();

function live(): number {
  return storageTracker.stats().totalStorages;
}
function cleanBaseline(): number {
  storageTracker.destroyUnreachable();
  return storageTracker.stats().totalStorages;
}

describe("scope() surface (browser)", () => {
  let ok = false;
  beforeAll(async () => {
    ok = await initWebGPU();
  });

  it("create + reclaim + escape", async () => {
    if (!ok) return;
    const p = api.tensorFromArray([1, 2, 3, 4], [2, 2], { device: "webgpu" });
    await p.cpu();
    const baseline = cleanBaseline();

    // Nothing returned → all reclaimed.
    await api.scope(async () => {
      const a = api.mul(p, 2);
      const b = api.add(a, a);
      await b.cpu();
    });
    expect(live()).toBe(baseline);

    // Return one tensor → it survives, usable after the scope.
    const out = await api.scope(async () => {
      const keeper = api.mul(p, 3);
      await keeper.cpu();
      return keeper;
    });
    expect(live()).toBe(baseline + 1);
    expect(await out.cpu()).toEqual([3, 6, 9, 12]);
    out.dispose();

    // keep() escapes to root.
    let kept!: typeof p;
    await api.scope(async () => {
      const k = api.add(p, p);
      await k.cpu();
      api.keep(k);
      kept = k;
    });
    expect(await kept.cpu()).toEqual([2, 4, 6, 8]);
    kept.dispose();
  });

  it("overlapping ambient scopes throw (single-flight)", async () => {
    if (!ok) return;
    let release!: () => void;
    const barrier = new Promise<void>((r) => {
      release = r;
    });

    const pA = api.scope(async () => {
      const a = api.tensorFromArray([1], [1], { device: "webgpu" });
      await a.cpu();
      await barrier; // A open across an await
      return null;
    });
    await Promise.resolve();
    await Promise.resolve();

    let threw: unknown = null;
    try {
      await api.scope(async () => {
        const b = api.tensorFromArray([2], [1], { device: "webgpu" });
        await b.cpu();
        return null;
      });
    } catch (err) {
      threw = err;
    }
    release();
    await pA;

    expect(threw).toBeInstanceOf(Error);
    expect((threw as Error).message).toContain("overlapping async scope");
    expect((threw as Error).message).toContain("single-flight");
  });
});
