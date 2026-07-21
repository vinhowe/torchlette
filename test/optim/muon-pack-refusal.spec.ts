/**
 * Chain-packing P3 — the standing spec of the design's HARD assertion
 * (docs/chain-packing-design.md §6.1): **no packed class may contain an `mm`
 * node.** A `cat` of differently-shaped momentum matrices is NOT one matmul, so
 * silently flat-packing an `mm` would produce a WRONG RESULT (the worst failure
 * mode). The packer's clause-4 gate (`assertFlattenable`) is therefore a hard
 * refusal, not a heuristic.
 *
 * These are pure term-level + declaration-seam assertions (no GPU): they run in
 * the `cpu` project. The byte-exact packed-attempt-vs-per-param TRAJECTORY pin
 * (across the compiled-plan activation threshold) lives in
 * `tools/parity-packed-vs-unpacked.ts` (muon arm) and the end-to-end descent in
 * `test/muon-distil-descent.spec.ts`; both are unaffected by the P3 routing
 * change because the refusal fallback IS the pre-change per-param math.
 *
 * (a) Muon → typed, named refusal (the packer refuses `MUON_PROGRAM`).
 * (b) Refusal → per-param fallback = the exact prior trajectory: the verdict is
 *     PURE (allocates nothing, mutates no state) and cached once-per-class, so
 *     the realization below it is byte-identical to before the routing landed.
 * (c) Adam / Lion / SGD classes still pack — no collateral refusal.
 */

import { describe, expect, it } from "vitest";
import {
  ADAMW_PROGRAM,
  LION_PROGRAM,
  MUON_PROGRAM,
  OPTIMIZER_PROGRAMS,
  SGD_MOMENTUM_PROGRAM,
  SGD_PROGRAM,
} from "../../src/ops/semantic/optimizer";
import {
  assertFlattenable,
  OptimizerPackRefusal,
  optTermHasMatmul,
} from "../../src/optim/pack-optimizer";
import { Torchlette } from "../../src/frontend/torchlette";
import { Muon } from "../../src/optim";

describe("chain-packing P3: no packed class may contain an mm node", () => {
  // ---- (a) Muon → typed, named refusal --------------------------------------
  it("the packer REFUSES the Muon class (contains mm / Newton-Schulz)", () => {
    // The mm node lives in the param update (the orthogonalization), not the
    // momentum state update.
    expect(optTermHasMatmul(MUON_PROGRAM.paramUpdate)).toBe(true);
    expect(
      MUON_PROGRAM.stateUpdates.some((su) => optTermHasMatmul(su.expr)),
    ).toBe(false);

    let caught: unknown;
    try {
      assertFlattenable(MUON_PROGRAM, MUON_PROGRAM.paramUpdate);
    } catch (e) {
      caught = e;
    }
    // TYPED and NAMED (the design requires an inspectable, named refusal).
    expect(caught).toBeInstanceOf(OptimizerPackRefusal);
    expect((caught as Error).name).toBe("OptimizerPackRefusal");
    expect((caught as Error).message).toContain("contraction");
    expect((caught as Error).message).toContain("muon");
  });

  // ---- (c) Adam / Lion / SGD still pack — no collateral refusal --------------
  it("Adam / Lion / SGD classes are flat-packable (accepted, no refusal)", () => {
    for (const program of [
      ADAMW_PROGRAM,
      LION_PROGRAM,
      SGD_PROGRAM,
      SGD_MOMENTUM_PROGRAM,
    ]) {
      for (const su of program.stateUpdates)
        expect(optTermHasMatmul(su.expr)).toBe(false);
      expect(optTermHasMatmul(program.paramUpdate)).toBe(false);
      // The gate must NOT throw for these.
      expect(() =>
        assertFlattenable(program, program.paramUpdate),
      ).not.toThrow();
    }
  });

  // The standing invariant across the WHOLE program catalog: EXACTLY the
  // mm-bearing programs refuse. If a future optimizer adds a contraction, this
  // pins that it too must refuse rather than silently flat-pack.
  it("across the program catalog, exactly the mm-bearing programs refuse", () => {
    const refused: string[] = [];
    for (const program of OPTIMIZER_PROGRAMS) {
      let didRefuse = false;
      try {
        assertFlattenable(program, program.paramUpdate);
      } catch (e) {
        expect(e).toBeInstanceOf(OptimizerPackRefusal);
        didRefuse = true;
      }
      const hasMm =
        optTermHasMatmul(program.paramUpdate) ||
        program.stateUpdates.some((su) => optTermHasMatmul(su.expr));
      // The refusal fires IFF the program contains a contraction (single source).
      expect(didRefuse).toBe(hasMm);
      if (didRefuse) refused.push(program.name);
    }
    expect(refused).toEqual(["muon"]);
  });

  // ---- The realizer seam: Muon.packVerdict() ---------------------------------
  // (a) again at the realizer altitude + (b) the verdict is a PURE, cached,
  // once-per-class decision (no per-step re-throw churn, no side effects), so
  // the per-param fallback below it reproduces the exact prior trajectory.
  it("Muon.packVerdict() returns the typed refusal, cached & side-effect-free", async () => {
    const api = new Torchlette("cpu");
    // Two 2D weight matrices (Muon-routed) + a 1D param (AdamW-routed).
    const w0 = api.tensorFromArray(
      Array.from({ length: 12 }, (_, i) => i * 0.01),
      [3, 4],
      { requiresGrad: true },
    );
    const w1 = api.tensorFromArray(
      Array.from({ length: 8 }, (_, i) => i * 0.02),
      [2, 4],
      { requiresGrad: true },
    );
    const b0 = api.tensorFromArray([0.1, 0.2], [2], { requiresGrad: true });
    const opt = new Muon([w0, w1, b0], { lr: 1e-3, momentum: 0.95 }, api);
    expect(opt.numMuonParams).toBe(2);

    const v1 = opt.packVerdict();
    expect(v1).toBeInstanceOf(OptimizerPackRefusal);
    expect(v1!.name).toBe("OptimizerPackRefusal");

    // Cached: the SAME object every call (decided once — never re-thrown, never
    // re-allocated). This is the "cheap, once-per-class" contract (design §6.1).
    const v2 = opt.packVerdict();
    expect(v2).toBe(v1);

    // Pure: repeated verdict calls mutate no parameter storage (the fallback
    // trajectory is byte-identical to the pre-routing per-param math).
    const read = async (): Promise<number[][]> => {
      const out: number[][] = [];
      for (const p of [w0, w1, b0]) out.push(Array.from(await p.cpu()));
      return out;
    };
    const before = await read();
    for (let i = 0; i < 5; i++) opt.packVerdict();
    const after = await read();
    expect(after).toEqual(before);
  });
});
