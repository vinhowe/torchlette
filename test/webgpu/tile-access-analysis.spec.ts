/**
 * Tile-IR Access Pattern Analysis tests.
 *
 * These tests verify symbolic stride evaluation on tile-IR index expressions
 * to detect coalesced/strided/broadcast patterns and compute safe vec widths.
 *
 * All tests are pure IR analysis (no GPU dispatch needed).
 */

import { describe, expect, it } from "vitest";
import type { TileKernelSpec } from "../../src/backend/webgpu/tile-ir";
import {
  analyzeAccessPatterns,
  reportAccessPatterns,
  computeSafeVecWidth,
} from "../../src/backend/webgpu/tile-access-analysis";

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
      input:  { storage: "read", type: "f32" },
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
      const loads = patterns.filter(p => p.accessType === "load");
      const stores = patterns.filter(p => p.accessType === "store");
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
      const load = patterns.find(p => p.accessType === "load" && p.bindingName === "input");
      expect(load).toBeDefined();
      expect(load!.innerStride).toBe(128);
      expect(load!.isCoalesced).toBe(false);
      expect(load!.maxVecWidth).toBe(1);
    });
  });

  describe("2D tile access", () => {
    it("load(tid.y * cols + tid.x) → stride 1, coalesced", () => {
      const spec = makeSpec("tiled2D", (ctx) => {
        const tidX = ctx.threadIdx(0);
        const tidY = ctx.threadIdx(1);
        const cols = ctx.u32(32);
        const idx = tidY.mul(cols).add(tidX);
        const val = ctx.load("input", idx);
        ctx.emitStore("output", idx, val);
      }, { workgroupSize: [32, 8] });

      const patterns = analyzeAccessPatterns(spec);
      // All accesses should be coalesced (innerStride = 1)
      for (const p of patterns) {
        expect(p.innerStride).toBe(1);
        expect(p.isCoalesced).toBe(true);
      }
    });

    it("load(tid.x * rows + tid.y) → stride rows, not coalesced", () => {
      const spec = makeSpec("tiled2DColMajor", (ctx) => {
        const tidX = ctx.threadIdx(0);
        const tidY = ctx.threadIdx(1);
        const rows = ctx.u32(32);
        const idx = tidX.mul(rows).add(tidY);
        const val = ctx.load("input", idx);
        ctx.emitStore("output", idx, val);
      }, { workgroupSize: [32, 8] });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find(p => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load!.innerStride).toBe(32);
      expect(load!.isCoalesced).toBe(false);
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
      const load = patterns.find(p => p.accessType === "load" && p.bindingName === "input");
      expect(load).toBeDefined();
      expect(load!.innerStride).toBe(0);
      expect(load!.isCoalesced).toBe(false);

      // Store should still be coalesced
      const store = patterns.find(p => p.accessType === "store");
      expect(store).toBeDefined();
      expect(store!.innerStride).toBe(1);
      expect(store!.isCoalesced).toBe(true);
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
      const load = patterns.find(p => p.accessType === "load");
      expect(load!.innerStride).toBe(1);
      expect(load!.isCoalesced).toBe(true);
      expect(load!.maxVecWidth).toBe(4);
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
      const load = patterns.find(p => p.accessType === "load");
      expect(load!.innerStride).toBe(1);
      expect(load!.isCoalesced).toBe(true);
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
      const store = patterns.find(p => p.accessType === "store");
      expect(store).toBeDefined();
      expect(store!.innerStride).toBe(1);
      expect(store!.isCoalesced).toBe(true);
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
      const stores = patterns.filter(p => p.accessType === "store");
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
      const load = patterns.find(p => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load!.baseDivisibility).toBe(1);
      expect(load!.baseConstantTerm).toBe(0);
    });

    it("globalId(0) * 4 has stride 4 and divisibility 4", () => {
      const spec = makeSpec("divMul4", (ctx) => {
        const gid = ctx.globalId(0);
        const idx = gid.mul(ctx.u32(4));
        ctx.emitStore("output", idx, ctx.load("input", idx));
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find(p => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load!.innerStride).toBe(4);
      expect(load!.baseDivisibility).toBe(4);
    });

    it("globalId(0) + const(4) has constantTerm 4", () => {
      const spec = makeSpec("divConst4", (ctx) => {
        const gid = ctx.globalId(0);
        const idx = gid.add(ctx.u32(4));
        ctx.emitStore("output", gid, ctx.load("input", idx));
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find(p => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load!.baseConstantTerm).toBe(4);
      expect(load!.maxVecWidth).toBe(4); // constantTerm 4 is 4-aligned
    });

    it("globalId(0) + const(1) limits vec width", () => {
      const spec = makeSpec("divConst1", (ctx) => {
        const gid = ctx.globalId(0);
        const idx = gid.add(ctx.u32(1));
        ctx.emitStore("output", gid, ctx.load("input", idx));
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find(p => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load!.baseConstantTerm).toBe(1);
      expect(load!.maxVecWidth).toBe(1); // odd offset → scalar only
    });

    it("globalId(0) + const(2) allows vec2", () => {
      const spec = makeSpec("divConst2", (ctx) => {
        const gid = ctx.globalId(0);
        const idx = gid.add(ctx.u32(2));
        ctx.emitStore("output", gid, ctx.load("input", idx));
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find(p => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load!.baseConstantTerm).toBe(2);
      expect(load!.maxVecWidth).toBe(2); // 2-aligned offset → vec2
    });

    it("globalId(0) + uniform has null constantTerm → vec4", () => {
      const spec = makeSpec("divUniform", (ctx) => {
        const gid = ctx.globalId(0);
        const base = ctx.uniform("N");
        const idx = gid.add(base);
        ctx.emitStore("output", gid, ctx.load("input", idx));
      });

      const patterns = analyzeAccessPatterns(spec);
      const load = patterns.find(p => p.accessType === "load");
      expect(load).toBeDefined();
      expect(load!.baseConstantTerm).toBeNull(); // unknown constant
      expect(load!.maxVecWidth).toBe(4); // unknown → conservatively vec4
    });
  });
});
