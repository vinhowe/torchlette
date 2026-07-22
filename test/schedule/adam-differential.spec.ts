/**
 * P4 byte differential — FUSED ADAM family (§7 LOCAL self-hosting, deliverable 3).
 *
 * (1) AUTHORED (opaque, F3) fused-Adam kernel — CUTOVER-FLIPPED. For each variant
 *     (useVec4 × emitF16 × emitUnscale),
 *
 *       compileTileKernel(applyAdamSchedule(deriveAdamSkeleton(desc), desc))
 *         ==BYTES== generateAdamShaderTileIR(...)   // the live schedule chokepoint
 *
 *     The body now LOWERS FROM the schedule (lowerAdamStepBody); the live dispatch
 *     (getAdamDispatcher) routes through realizeAdamStepSpec. Nowhere in an opaque
 *     skeleton to store a generator (R22).
 *
 * (2) The HORIZONTAL-PACK derivation (the §3 pack move's real tenant): N per-param
 *     elementwise Adam loops pack (concatenate) into ONE pack loop over segments —
 *     the multi-tensor packing the foreach optimizer performs. The packed-dispatch
 *     count is 1 per group (not N).
 *
 * PURE-COMPILATION (no GPU): the tile-IR codegen is a WGSL compiler; both sides
 * compile in ONE process, so byte-identity holds. Lives in the cpu project.
 * Exercises the authored no-second-owner seam assertions (§12 check 3).
 */

import { describe, expect, it } from "vitest";
import { compileTileKernel } from "../../src/backend/webgpu/tile-compiler";
import {
  type AdamDescriptor,
  ADAM_PARAM_SCHEMA,
  adamCacheKey,
  applyAdamSchedule,
  deriveAdamSkeleton,
  deriveHorizontalPackedAdam,
  generateAdamShaderTileIR,
} from "../../src/schedule/adam-skeleton";
import { printScheduleState } from "../../src/schedule/canonical";
import type { ScheduleState, ValueUid } from "../../src/schedule/types";

const v = (s: string): ValueUid => s as unknown as ValueUid;

// The variant corpus (useVec4 requires emitUnscale — the dependent constraint).
const variants: AdamDescriptor[] = [
  { useVec4: false, emitF16: false, emitUnscale: false },
  { useVec4: false, emitF16: true, emitUnscale: false },
  { useVec4: false, emitF16: false, emitUnscale: true },
  { useVec4: true, emitF16: false, emitUnscale: true },
  { useVec4: true, emitF16: true, emitUnscale: true },
];

describe("P4 fused-Adam byte differential (authored / opaque)", () => {
  for (const desc of variants) {
    it(`adamStep [${adamCacheKey(desc)}]`, () => {
      // The LIVE path IS the schedule chokepoint now (the cutover-flip): the
      // dispatcher routes through realizeAdamStepSpec / generateAdamShaderTileIR.
      // R4 (2026-07-22): the authored body is deleted; the schedule chokepoint
      // now lowers the DERIVED body (fork C). This differential proves the two
      // entry points (schedule vs direct generation) agree byte-for-byte —
      // single-source of the derived kernel. The numeric derived==program guard
      // is the fold-parity (optterm-fold).
      const live = generateAdamShaderTileIR(
        desc.useVec4,
        desc.emitF16,
        desc.emitUnscale,
      );
      const derived = compileTileKernel(
        applyAdamSchedule(deriveAdamSkeleton(desc), desc),
      );
      expect(derived).toBe(live);
    });
  }
});

describe("P4 fused-Adam authored-form legality (§6 / F3 / R10)", () => {
  it("the fused Adam kernel wears an OPAQUE skeleton (F3), never derived", () => {
    const sk = deriveAdamSkeleton({
      useVec4: false,
      emitF16: false,
      emitUnscale: false,
    });
    expect(sk.visibility).toBe("opaque");
    if (sk.visibility === "opaque") {
      // Post-cutover-flip: the body was ABSORBED into the schedule module, so
      // the kernelRef names the schedule-owned lowering (single source).
      expect(sk.kernelRef).toContain("adam-skeleton.ts");
      expect(sk.refusalReason).toContain("HORIZONTAL-PACK");
      expect(sk.refusalReason).toContain("LIVE-PATH FLIPPED");
      expect(sk).not.toHaveProperty("schedule");
      expect(sk.params).toBe(ADAM_PARAM_SCHEMA);
    }
  });

  it("the cache-key encoder delegates to the getAdamDispatcher variant key (single source)", () => {
    expect(adamCacheKey({ useVec4: false, emitF16: false, emitUnscale: false })).toBe(
      "adam:false:false:false",
    );
    expect(adamCacheKey({ useVec4: true, emitF16: true, emitUnscale: true })).toBe(
      "adam:true:true:true",
    );
  });

  it("the typed param schema carries the vec4⇒unscale dependent constraint", () => {
    const sk = deriveAdamSkeleton({
      useVec4: false,
      emitF16: false,
      emitUnscale: false,
    });
    if (sk.visibility !== "opaque") throw new Error("expected opaque");
    expect(Object.keys(sk.params.params)).toContain("useVec4");
    expect(sk.params.constraints.length).toBeGreaterThanOrEqual(1);
  });
});

describe("P4 fused-Adam HORIZONTAL-PACK derivation (the pack-move tenant)", () => {
  const segments = [
    { value: v("param0"), numElements: 768 * 768 },
    { value: v("param1"), numElements: 768 },
    { value: v("param2"), numElements: 1024 * 768 },
  ];

  it("packs N per-param loops into ONE pack loop (concatenate)", () => {
    const packed = deriveHorizontalPackedAdam(segments);
    expect(packed.perParamLoopCount).toBe(3);
    expect(packed.packedState.semantic.loopNest.length).toBe(1);
    expect(packed.packMove.move).toBe("pack");
    if (packed.packMove.move === "pack") {
      expect(packed.packMove.kind).toBe("concatenate");
    }
  });

  it("the packed flat has all Σ segment elements", () => {
    const packed = deriveHorizontalPackedAdam(segments);
    const total = 768 * 768 + 768 + 1024 * 768;
    expect(packed.totalElements).toBe(total);
  });

  it("the packed-dispatch count is ONE per group (not N)", () => {
    const packed = deriveHorizontalPackedAdam(segments);
    // 1 packed dispatch vs perParamLoopCount un-packed applications.
    expect(1).toBeLessThan(packed.perParamLoopCount);
  });

  it("the packed derived state lowers via the canonical schema", () => {
    const packed = deriveHorizontalPackedAdam(segments);
    const printed = printScheduleState(packed.packedState as ScheduleState);
    expect(printed).toContain("schedule-state v");
  });

  it("refuses an empty group (nothing to pack)", () => {
    expect(() => deriveHorizontalPackedAdam([])).toThrow(/empty group/);
  });
});
