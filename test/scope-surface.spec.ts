/**
 * Async scope() surface (docs/scoped-memory-design.md §2-4).
 *
 * scope() is a THIN layer over the step path's snapshot/releaseStepTemps
 * reclamation: tensors created in a scope that don't escape are reclaimed at
 * scope exit; returned tensors (and their view bases) survive; keep()/persist()
 * escape to root. Overlap of two concurrent ambient scopes throws (single-flight).
 *
 * Runs on webgpu when available, else cpu — the reclamation path is
 * backend-agnostic (storageTracker rc + snapshot), so both projects exercise it.
 */

import { beforeAll, describe, expect, it } from "vitest";
import { Torchlette } from "../src/frontend/torchlette";
import { storageTracker } from "../src/graph/storage-tracker";
import { canUseWebGPU } from "./helpers/webgpu";

const api = new Torchlette();
let device: "cpu" | "webgpu" = "cpu";

beforeAll(async () => {
  device = (await canUseWebGPU()) ? "webgpu" : "cpu";
});

/** Force materialization so the storage tracker sees the handle. */
async function force(t: { cpu(): Promise<number[]> }): Promise<void> {
  await t.cpu();
}

/** Clean baseline: reclaim any leftover garbage from prior tests so the
 *  scope's own destroyUnreachable doesn't drop cross-test storages and skew
 *  the delta, then count live handles. */
function cleanBaseline(): number {
  storageTracker.destroyUnreachable();
  return storageTracker.stats().totalStorages;
}

function live(): number {
  return storageTracker.stats().totalStorages;
}

describe(`scope() surface`, () => {
  it("(a) reclaims tensors created in a scope and not returned", async () => {
    // Baseline: a persistent tensor created + forced outside the scope.
    const p = api.tensorFromArray([1, 2, 3, 4], [2, 2], { device });
    await force(p);
    const baseline = cleanBaseline();

    await api.scope(async () => {
      const a = api.mul(p, 2);
      const b = api.add(a, a);
      await force(b);
      const c = api.matmul(b, b);
      await force(c);
      // nothing returned → all reclaimed
    });

    expect(live()).toBe(baseline);
    // p still usable
    expect(await p.cpu()).toEqual([1, 2, 3, 4]);
  });

  it("(b) a returned tensor survives and is usable after the scope", async () => {
    const p = api.tensorFromArray([1, 2, 3, 4], [2, 2], { device });
    await force(p);
    const baseline = cleanBaseline();

    const out = await api.scope(async () => {
      const a = api.mul(p, 3);
      const b = api.add(a, a); // 6*p, discarded
      await force(b);
      const keeper = api.mul(a, 1); // survives
      await force(keeper);
      return keeper;
    });

    // Exactly the escapee survives above baseline.
    expect(live()).toBe(baseline + 1);
    const vals = await out.cpu();
    expect(vals).toEqual([3, 6, 9, 12]);

    out.dispose();
  });

  it("(c) a returned view keeps its base alive", async () => {
    const p = api.tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], { device });
    await force(p);
    const baseline = cleanBaseline();

    const view = await api.scope(async () => {
      const big = api.mul(p, 2); // in-scope base, [2,3]
      await force(big);
      const v = big.reshape([3, 2]); // view of big — returned
      return v;
    });

    const vals = await view.cpu();
    expect(vals).toEqual([2, 4, 6, 8, 10, 12]);
    // Base survived (reachability escape) — reading the view didn't hit freed memory.
    expect(live()).toBeGreaterThanOrEqual(baseline + 1);

    view.dispose();
  });

  it("(d) forward-in-scope + backward-OUTSIDE matches no-scope gradients", async () => {
    const mkParams = () => {
      const w = api
        .tensorFromArray([0.5, -0.3, 0.2, 0.8], [2, 2], { device })
        .requires_grad_();
      return w;
    };
    const x = api.tensorFromArray([1, 2, 3, 4], [2, 2], { device });

    // No scope.
    const gradOf = async (w: { grad: { cpu(): Promise<number[]> } | null }) => {
      if (!w.grad) throw new Error("expected a gradient");
      return w.grad.cpu();
    };

    const wA = mkParams();
    const lossA = api.mean(api.mul(api.matmul(x, wA), api.matmul(x, wA)));
    await lossA.backward();
    const gradA = await gradOf(wA);

    // Forward inside scope (materialized there — stresses stage-1 saved-tensor
    // retention), backward outside.
    const wB = mkParams();
    const lossB = await api.scope(async () => {
      const y = api.matmul(x, wB);
      const loss = api.mean(api.mul(y, y));
      await loss.item(); // materialize forward incl. saved activations
      return loss; // escapes
    });
    await lossB.backward();
    const gradB = await gradOf(wB);

    expect(gradB.length).toBe(gradA.length);
    for (let i = 0; i < gradA.length; i++) {
      expect(gradB[i]).toBeCloseTo(gradA[i], 5);
    }
  });

  it("(e) nesting reclaims correctly with NO throw", async () => {
    const p = api.tensorFromArray([1, 2, 3, 4], [2, 2], { device });
    await force(p);
    const baseline = cleanBaseline();

    // scope inside scope (SYNCHRONOUS nesting — child opened during the
    // parent's synchronous prefix), sync tidy inside scope, and a
    // library-style child scope opened via an openScope() handle.
    const out = await api.scope(async () => {
      // Synchronous prefix: do all ambient nesting here, before any await.
      const outer = api.mul(p, 2);

      // sync tidy inside scope
      const tidied = api.tidy(() => {
        const tmp = api.add(outer, outer);
        return api.mul(tmp, 1);
      });

      // synchronous child ambient scope (opened while parent body runs sync)
      const child = api.scope(() => {
        const inner = api.mul(outer, 3);
        return inner; // escapes child → parent interval
      });

      // Now force (awaits allowed after the synchronous nesting is done).
      await force(outer);
      await force(tidied);
      await force(child);

      // library-internal scope via handle (exempt from single-flight)
      const s = api.openScope();
      const libTmp = api.add(child, child);
      await force(libTmp);
      s.close(); // libTmp reclaimed

      return child; // only child escapes the outer scope
    });

    expect(live()).toBe(baseline + 1);
    const vals = await out.cpu();
    expect(vals).toEqual([6, 12, 18, 24]);
    out.dispose();
  });

  it("(f) overlapping ambient scopes throw with the actionable message", async () => {
    let barrierResolve!: () => void;
    const barrier = new Promise<void>((r) => {
      barrierResolve = r;
    });

    // Scope A opens and suspends at an await (barrier), staying open.
    const pA = api.scope(async () => {
      const a = api.tensorFromArray([1], [1], { device });
      await force(a);
      await barrier; // suspend here — A is open across an await
      return null;
    });

    // Give A a tick to reach its await.
    await Promise.resolve();
    await Promise.resolve();

    // Scope B begins while A is suspended → concurrent overlap → throw.
    let threw: unknown = null;
    try {
      await api.scope(async () => {
        const b = api.tensorFromArray([2], [1], { device });
        await force(b);
        return null;
      });
    } catch (err) {
      threw = err;
    }

    barrierResolve();
    await pA;

    expect(threw).toBeInstanceOf(Error);
    expect((threw as Error).message).toContain("overlapping async scope");
    expect((threw as Error).message).toContain("single-flight");
  });

  it("(g) keep()/persist() escape a tensor to root across scopes", async () => {
    const p = api.tensorFromArray([1, 2, 3, 4], [2, 2], { device });
    await force(p);
    const baseline = cleanBaseline();

    let kept!: typeof p;
    let persisted!: typeof p;
    await api.scope(async () => {
      const k = api.mul(p, 2);
      await force(k);
      api.keep(k); // escape to root
      kept = k;

      const q = api.add(p, p);
      await force(q);
      api.persist(q); // escape to root (alias)
      persisted = q;
      // neither returned, but both kept
    });

    // Both survive (kept + persisted) above baseline.
    expect(live()).toBe(baseline + 2);
    expect(await kept.cpu()).toEqual([2, 4, 6, 8]);
    expect(await persisted.cpu()).toEqual([2, 4, 6, 8]);
    kept.dispose();
    persisted.dispose();
  });
});
