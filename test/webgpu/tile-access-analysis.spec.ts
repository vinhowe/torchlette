/**
 * Tile-IR Access Pattern Analysis tests.
 *
 * These tests verify symbolic stride evaluation on tile-IR index expressions
 * to detect coalesced/strided/broadcast patterns and compute safe vec widths.
 *
 * All tests are pure IR analysis (no GPU dispatch needed).
 */

import { describe, expect, it } from "vitest";
import {
  analyzeAccessPatterns,
  computeSafeVecWidth,
  reportAccessPatterns,
} from "../../src/backend/webgpu/tile-access-analysis";
import type { TileKernelSpec } from "../../src/backend/webgpu/tile-ir";

// ============================================================================
// Helper: build a spec with a given kernel
// ============================================================================

function makeSpec(
  name: string,
  kernel: TileKernelSpec["kernel"],
  opts?: {
    workgroupSize?: number | [number, number];
    bindings?: TileKernelSpec["bindings"];
    uniforms?: TileKernelSpec["uniforms"];
  },
): TileKernelSpec {
  return {
    name,
    workgroupSize: opts?.workgroupSize ?? 64,
    bindings: opts?.bindings ?? {
      input: { storage: "read", type: "f32" },
      output: { storage: "read_write", type: "f32" },
    },
    uniforms: opts?.uniforms ?? { N: "u32" },
    grid: (u) => [Math.ceil(u.N / 64)],
    kernel,
  };
}

// ============================================================================
// Tests
// ============================================================================

describe("tile-access-analysis", () => {
  describe("coalesced 1D access", () => {
    it("load(base + globalId.x) → stride 1, coalesced, vec4", () => {
      const spec = makeSpec("coalesced1D", (ctx) => {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      // Should have at least one load and one store
      const loads = patterns.filter((p) => p.accessType === "load");
      const stores = patterns.filter((p) => p.accessType === "store");
      expect(loads.length).toBeGreaterThanOrEqual(1);
      expect(stores.length).toBeGreaterThanOrEqual(1);

      // Both should be coalesced
      for (const p of patterns) {
        expect(p.innerStride).toBe(1);
        expect(p.isCoalesced).toBe(true);
        expect(p.maxVecWidth).toBe(4);
      }
    });
  });

  describe("strided access", () => {
    it("load(globalId.x * N) → stride N, not coalesced", () => {
      const spec = makeSpec("strided", (ctx) => {
        const gid = ctx.globalId(0);
        const stride = ctx.u32(128);
        const idx = gid.mul(stride);
        const val = ctx.load("input", idx);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find(
        (p) => p.accessType === "load" && p.bindingName === "input",
      );
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(128);
      expect(load?.isCoalesced).toBe(false);
      expect(load?.maxVecWidth).toBe(1);
    });
  });

  describe("2D tile access", () => {
    it("load(tid.y * cols + tid.x) → stride 1, coalesced", () => {
      const spec = makeSpec(
        "tiled2D",
        (ctx) => {
          const tidX = ctx.threadIdx(0);
          const tidY = ctx.threadIdx(1);
          const cols = ctx.u32(32);
          const idx = tidY.mul(cols).add(tidX);
          const val = ctx.load("input", idx);
          ctx.emitStore("output", idx, val);
        },
        { workgroupSize: [32, 8] },
      );

      const patterns = analyzeAccessPatterns(spec);
      // All accesses should be coalesced (innerStride = 1)
      for (const p of patterns) {
        expect(p.innerStride).toBe(1);
        expect(p.isCoalesced).toBe(true);
      }
    });

    it("load(tid.x * rows + tid.y) → stride rows, not coalesced", () => {
      const spec = makeSpec(
        "tiled2DColMajor",
        (ctx) => {
          const tidX = ctx.threadIdx(0);
          const tidY = ctx.threadIdx(1);
          const rows = ctx.u32(32);
          const idx = tidX.mul(rows).add(tidY);
          const val = ctx.load("input", idx);
          ctx.emitStore("output", idx, val);
        },
        { workgroupSize: [32, 8] },
      );

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(32);
      expect(load?.isCoalesced).toBe(false);
    });
  });

  describe("broadcast access", () => {
    it("load(programId.x) → stride 0, broadcast", () => {
      const spec = makeSpec("broadcast", (ctx) => {
        const wid = ctx.programId(0);
        const val = ctx.load("input", wid);
        const gid = ctx.globalId(0);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find(
        (p) => p.accessType === "load" && p.bindingName === "input",
      );
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(0);
      expect(load?.isCoalesced).toBe(false);

      // Store should still be coalesced
      const store = patterns.find((p) => p.accessType === "store");
      expect(store).toBeDefined();
      expect(store?.innerStride).toBe(1);
      expect(store?.isCoalesced).toBe(true);
    });
  });

  describe("constant offset access", () => {
    it("load(globalId.x + const) → stride 1, coalesced", () => {
      const spec = makeSpec("constOffset", (ctx) => {
        const gid = ctx.globalId(0);
        const offset = ctx.u32(1024);
        const idx = gid.add(offset);
        const val = ctx.load("input", idx);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load?.innerStride).toBe(1);
      expect(load?.isCoalesced).toBe(true);
      expect(load?.maxVecWidth).toBe(4);
    });
  });

  describe("uniform-scaled access", () => {
    it("load(globalId.x + wid * uniform) → stride 1, coalesced", () => {
      const spec = makeSpec("uniformScaled", (ctx) => {
        const gid = ctx.globalId(0);
        const wid = ctx.programId(0);
        const stride = ctx.uniform("N");
        const base = wid.mul(stride);
        const idx = base.add(gid);
        const val = ctx.load("input", idx);
        ctx.emitStore("output", idx, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load?.innerStride).toBe(1);
      expect(load?.isCoalesced).toBe(true);
    });
  });

  describe("computeSafeVecWidth", () => {
    it("returns 4 for fully coalesced kernel", () => {
      const spec = makeSpec("vec4safe", (ctx) => {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.emitStore("output", gid, val);
      });

      expect(computeSafeVecWidth(spec)).toBe(4);
    });

    it("returns 1 for strided kernel", () => {
      const spec = makeSpec("vec1strided", (ctx) => {
        const gid = ctx.globalId(0);
        const idx = gid.mul(ctx.u32(2));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", idx, val);
      });

      expect(computeSafeVecWidth(spec)).toBe(1);
    });

    it("returns 1 for mixed coalesced + broadcast", () => {
      const spec = makeSpec("mixed", (ctx) => {
        const gid = ctx.globalId(0);
        const wid = ctx.programId(0);
        // Broadcast load (stride=0) → maxVecWidth=1
        const bcast = ctx.load("input", wid);
        ctx.emitStore("output", gid, bcast);
      });

      // Broadcast load has maxVecWidth=1, so overall is 1
      expect(computeSafeVecWidth(spec)).toBe(1);
    });
  });

  describe("reportAccessPatterns", () => {
    it("produces human-readable report", () => {
      const spec = makeSpec("reportTest", (ctx) => {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.emitStore("output", gid, val);
      });

      const report = reportAccessPatterns(spec);
      expect(report).toContain("reportTest");
      expect(report).toContain("coalesced");
      expect(report).toContain("load");
      expect(report).toContain("store");
    });

    it("flags strided access with warning", () => {
      const spec = makeSpec("stridedReport", (ctx) => {
        const gid = ctx.globalId(0);
        const idx = gid.mul(ctx.u32(16));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", idx, val);
      });

      const report = reportAccessPatterns(spec);
      expect(report).toContain("\u26A0"); // ⚠ warning character
      expect(report).toContain("strided");
    });
  });

  describe("guardedStore analysis", () => {
    it("analyzes guarded stores correctly", () => {
      const spec = makeSpec("guarded", (ctx) => {
        const gid = ctx.globalId(0);
        const N = ctx.uniform("N");
        const val = ctx.load("input", gid);
        ctx.guardedStore("output", gid.lt(N), gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const store = patterns.find((p) => p.accessType === "store");
      expect(store).toBeDefined();
      expect(store?.innerStride).toBe(1);
      expect(store?.isCoalesced).toBe(true);
    });
  });

  describe("nested loop access", () => {
    it("finds stores inside forRange", () => {
      const spec = makeSpec("nested", (ctx) => {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.forRange(ctx.u32(0), ctx.u32(4), (_i) => {
          ctx.emitStore("output", gid, val);
        });
      });

      const patterns = analyzeAccessPatterns(spec);
      const stores = patterns.filter((p) => p.accessType === "store");
      expect(stores.length).toBeGreaterThanOrEqual(1);
      expect(stores[0].isCoalesced).toBe(true);
    });
  });

  describe("divisibility tracking", () => {
    it("globalId(0) has divisibility 1 and constantTerm 0", () => {
      const spec = makeSpec("divGlobalId", (ctx) => {
        const gid = ctx.globalId(0);
        const val = ctx.load("input", gid);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.baseDivisibility).toBe(1);
      expect(load?.baseConstantTerm).toBe(0);
    });

    it("globalId(0) * 4 has stride 4 and divisibility 4", () => {
      const spec = makeSpec("divMul4", (ctx) => {
        const gid = ctx.globalId(0);
        const idx = gid.mul(ctx.u32(4));
        ctx.emitStore("output", idx, ctx.load("input", idx));
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(4);
      expect(load?.baseDivisibility).toBe(4);
    });

    it("globalId(0) + const(4) has constantTerm 4", () => {
      const spec = makeSpec("divConst4", (ctx) => {
        const gid = ctx.globalId(0);
        const idx = gid.add(ctx.u32(4));
        ctx.emitStore("output", gid, ctx.load("input", idx));
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.baseConstantTerm).toBe(4);
      expect(load?.maxVecWidth).toBe(4); // constantTerm 4 is 4-aligned
    });

    it("globalId(0) + const(1) limits vec width", () => {
      const spec = makeSpec("divConst1", (ctx) => {
        const gid = ctx.globalId(0);
        const idx = gid.add(ctx.u32(1));
        ctx.emitStore("output", gid, ctx.load("input", idx));
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.baseConstantTerm).toBe(1);
      expect(load?.maxVecWidth).toBe(1); // odd offset → scalar only
    });

    it("globalId(0) + const(2) allows vec2", () => {
      const spec = makeSpec("divConst2", (ctx) => {
        const gid = ctx.globalId(0);
        const idx = gid.add(ctx.u32(2));
        ctx.emitStore("output", gid, ctx.load("input", idx));
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.baseConstantTerm).toBe(2);
      expect(load?.maxVecWidth).toBe(2); // 2-aligned offset → vec2
    });

    it("globalId(0) + uniform has null constantTerm → vec4", () => {
      const spec = makeSpec("divUniform", (ctx) => {
        const gid = ctx.globalId(0);
        const base = ctx.uniform("N");
        const idx = gid.add(base);
        ctx.emitStore("output", gid, ctx.load("input", idx));
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.baseConstantTerm).toBeNull(); // unknown constant
      expect(load?.maxVecWidth).toBe(4); // unknown → conservatively vec4
    });
  });

  describe("integer division analysis", () => {
    it("(globalId.x * 4) / 4 → stride 1, coalesced", () => {
      const spec = makeSpec("divCoalesced", (ctx) => {
        const gid = ctx.globalId(0);
        // gid * 4 → stride 4, then / 4 → stride 1
        const idx = gid.mul(ctx.u32(4)).div(ctx.u32(4));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(1);
      expect(load?.isCoalesced).toBe(true);
    });

    it("(globalId.x * 2) / 4 → stride 0.5 floors to unknown", () => {
      const spec = makeSpec("divNonDivisible", (ctx) => {
        const gid = ctx.globalId(0);
        // gid * 2, then / 4: 2 % 4 != 0 → unknown
        const idx = gid.mul(ctx.u32(2)).div(ctx.u32(4));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe("unknown");
    });

    it("thread-invariant / const → thread-invariant", () => {
      const spec = makeSpec("divThreadInvariant", (ctx) => {
        const gid = ctx.globalId(0);
        const wid = ctx.programId(0);
        // programId / 4 → still thread-invariant (stride 0)
        const idx = wid.div(ctx.u32(4));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find(
        (p) => p.accessType === "load" && p.bindingName === "input",
      );
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(0);
      expect(load?.isCoalesced).toBe(false);
    });
  });

  describe("modular arithmetic analysis", () => {
    it("(globalId.x * 4) % 4 → thread-invariant (stride 0)", () => {
      const spec = makeSpec("modZeroStride", (ctx) => {
        const gid = ctx.globalId(0);
        // gid * 4 → stride 4, then % 4: 4 % 4 == 0 → thread-invariant
        const idx = gid.mul(ctx.u32(4)).mod(ctx.u32(4));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(0);
    });

    it("globalId.x % 128 → stride 1 (coeff < modulus)", () => {
      const spec = makeSpec("modPreservesStride", (ctx) => {
        const gid = ctx.globalId(0);
        // gid has innerCoeff=1, 1 < 128 → stride preserved
        const idx = gid.mod(ctx.u32(128));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(1);
      expect(load?.isCoalesced).toBe(true);
    });

    it("thread-invariant % const → thread-invariant", () => {
      const spec = makeSpec("modThreadInvariant", (ctx) => {
        const gid = ctx.globalId(0);
        const wid = ctx.programId(0);
        const idx = wid.mod(ctx.u32(8));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find(
        (p) => p.accessType === "load" && p.bindingName === "input",
      );
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(0);
    });
  });

  describe("coordinate decomposition (div + mod)", () => {
    it("2D → 1D: row = gid / cols, col = gid % cols → row is stride 0, col is stride 1", () => {
      // Common pattern: linearize 2D index from globalId
      // row * width + col where row = gid / width, col = gid % width
      const spec = makeSpec("coordDecomp", (ctx) => {
        const gid = ctx.globalId(0);
        const width = ctx.u32(32);
        // row = gid / 32 → innerCoeff=1 / 32 fails (1 % 32 != 0) → unknown
        // col = gid % 32 → innerCoeff=1 < 32 → stride 1
        // idx = row * someStride + col → depends on row resolution
        // In practice: col alone is coalesced
        const col = gid.mod(width);
        const val = ctx.load("input", col);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find(
        (p) => p.accessType === "load" && p.bindingName === "input",
      );
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(1);
      expect(load?.isCoalesced).toBe(true);
    });

    it("(gid * 32) / 32 + (gid % 32) * stride → divisible div produces stride 1", () => {
      // Simulates: row = (gid * 32) / 32 = gid → stride 1
      // This is a trivial case but validates div with divisible coeff
      const spec = makeSpec("coordDecompDivisible", (ctx) => {
        const gid = ctx.globalId(0);
        const row = gid.mul(ctx.u32(32)).div(ctx.u32(32));
        const val = ctx.load("input", row);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(1);
      expect(load?.isCoalesced).toBe(true);
    });
  });

  describe("let/var definition tracking", () => {
    it("let offset = globalId.x; load(offset) → coalesced via let resolution", () => {
      const spec = makeSpec("letTracking", (ctx) => {
        const gid = ctx.globalId(0);
        const offset = ctx.emitLet("offset", gid);
        const val = ctx.load("input", offset);
        ctx.emitStore("output", offset, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(1);
      expect(load?.isCoalesced).toBe(true);
    });

    it("let base = programId.x * N; let idx = base + gid → coalesced", () => {
      const spec = makeSpec("letChained", (ctx) => {
        const gid = ctx.globalId(0);
        const wid = ctx.programId(0);
        const base = ctx.emitLet("base", wid.mul(ctx.uniform("N")));
        const idx = ctx.emitLet("idx", base.add(gid));
        const val = ctx.load("input", idx);
        ctx.emitStore("output", gid, val);
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(1);
      expect(load?.isCoalesced).toBe(true);
    });
  });

  describe("loop variable handling", () => {
    it("forRange with constant bounds marks loop var as bounded → stride 0 for loop var", () => {
      const spec = makeSpec("loopVar", (ctx) => {
        const gid = ctx.globalId(0);
        // Store at gid + loopVar — loopVar has constant bounds, so
        // it's thread-invariant (stride 0). Combined: stride 1.
        ctx.forRange(ctx.u32(0), ctx.u32(16), (i) => {
          const idx = gid.add(i);
          ctx.emitStore("output", idx, ctx.load("input", idx));
        });
      });

      const patterns = analyzeAccessPatterns(spec);
      // All loads and stores inside the loop should have stride 1
      // because gid has coeff=1 and loop var i has coeff=0
      for (const p of patterns) {
        expect(p.innerStride).toBe(1);
        expect(p.isCoalesced).toBe(true);
      }
    });

    it("forRange with uniform bounds treats loop var as data-dependent", () => {
      const spec = makeSpec("loopVarUniform", (ctx) => {
        const gid = ctx.globalId(0);
        // Loop bound is a uniform (not a constant), so loop var
        // is not in boundedLoopVars. namedRef → data-dependent.
        ctx.forRange(ctx.u32(0), ctx.uniform("N"), (i) => {
          // i is data-dependent but innerCoeff=0 (thread-invariant)
          const idx = gid.add(i);
          ctx.emitStore("output", idx, ctx.load("input", idx));
        });
      });

      const patterns = analyzeAccessPatterns(spec);
      // Even with data-dependent loop var, gid still provides stride 1
      // because i contributes innerCoeff=0 (thread-invariant)
      for (const p of patterns) {
        expect(p.innerStride).toBe(1);
        expect(p.isCoalesced).toBe(true);
      }
    });

    it("loop var * stride → still thread-invariant component in access", () => {
      const spec = makeSpec("loopVarMul", (ctx) => {
        const gid = ctx.globalId(0);
        ctx.forRange(ctx.u32(0), ctx.u32(8), (i) => {
          // i * 256 → thread-invariant (coeff=0), gid has coeff=1
          // idx = i * 256 + gid → stride 1
          const idx = i.mul(ctx.u32(256)).add(gid);
          const val = ctx.load("input", idx);
          ctx.emitStore("output", idx, val);
        });
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find((p) => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load?.innerStride).toBe(1);
      expect(load?.isCoalesced).toBe(true);
    });
  });
});
