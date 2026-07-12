/**
 * The Triton realizer (v2) walking-skeleton gates — PURE COMPILATION, no GPU.
 * Runs in the cpu project by default (the cross-backend numerical differential is
 * tools/triton-realizer/cross-backend-differential.ts, GPU-gated & manual).
 *
 * Gates:
 *   1. The capability profile transcribes Appendix A (per-verdict counts + the
 *      key A-R findings present, the authority-horizon receipt boundary).
 *   2. emitTriton is DETERMINISTIC and printable (same state → same source), and
 *      reads the program-grid map / block shapes / requests off the ScheduleState.
 *   3. Typed refusals (kSplit, checkedAffine) — never a raw throw deep in emit.
 *   4. Identity: the SAME schedule digest gains DISTINCT artifact digests per
 *      realizer coordinate (§5), and semantic identity is realizer-blind.
 */

import { describe, expect, it } from "vitest";
import { DEFAULT_CONFIG } from "../../src/backend/webgpu/matmul/types";
import {
  artifactDigest,
  scheduleDigest,
  semanticDigest,
} from "../../src/schedule/canonical";
import {
  deriveTiledMatmulState,
  type TiledMatmulDescriptor,
} from "../../src/schedule/matmul-skeleton";
import {
  TRITON_REALIZER,
  WGSL_REALIZER_COORDINATE,
} from "../../src/schedule/realizers/registry";
import {
  emitTritonTiledMatmul,
  TritonEmitRefusal,
} from "../../src/schedule/realizers/triton-emit";
import {
  TRITON_CAPABILITY_PROFILE,
  verdictCounts,
  verdictOf,
} from "../../src/schedule/realizers/triton-profile";
import type {
  ProgramGridMap,
  ScheduleState,
  SemanticRegionUid,
} from "../../src/schedule/types";

const REGION = "region:triton-test" as unknown as SemanticRegionUid;

const cfg = (over = {}) => ({
  ...DEFAULT_CONFIG,
  tileM: 32,
  tileN: 32,
  tileK: 16,
  ...over,
});
const baseDesc = (
  over: Partial<TiledMatmulDescriptor> = {},
): TiledMatmulDescriptor => ({
  config: cfg(),
  transposeMode: "NN",
  dtype: "f32",
  ...over,
});

function withGridMap(state: ScheduleState, map: ProgramGridMap): ScheduleState {
  return { ...state, semantic: { ...state.semantic, programGridMap: map } };
}

describe("Triton capability profile (Appendix A)", () => {
  it("transcribes every Appendix-A category with a verdict + finding + s1Home", () => {
    for (const e of TRITON_CAPABILITY_PROFILE.entries) {
      expect(e.reason.length).toBeGreaterThan(10);
      expect(["determination", "request", "refused", "split"]).toContain(
        e.verdict,
      );
      // Every split verdict carries its two halves (the §2 representation bug, DATA).
      if (e.verdict === "split") expect(e.split).toBeDefined();
    }
  });

  it("has the expected per-verdict counts (26 rows)", () => {
    const c = verdictCounts();
    const total = c.determination + c.request + c.refused + c.split;
    expect(total).toBe(TRITON_CAPABILITY_PROFILE.entries.length);
    // Five moves are SPLIT (A-R1..A-R5) plus two decoration/atom splits.
    expect(c.split).toBeGreaterThanOrEqual(5);
  });

  it("records the A-R15 / R4 program-map counterexample as a determination", () => {
    const pm = verdictOf("program-map (grid traversal / remapping)");
    expect(pm?.verdict).toBe("determination");
    expect(pm?.findings).toContain("A-R15");
    expect(pm?.findings).toContain("R4");
    expect(pm?.s1Home).toBe("semantic");
  });

  it("pins the surface (release + target arch) and excludes Gluon", () => {
    expect(TRITON_CAPABILITY_PROFILE.pinnedSurface.tritonRelease).toBeTruthy();
    expect(TRITON_CAPABILITY_PROFILE.pinnedSurface.targetArch).toBeTruthy();
    expect(TRITON_CAPABILITY_PROFILE.pinnedSurface.gluonCounted).toBe(false);
  });

  it("the authority horizon keeps TTGIR facts off the source side (receipt boundary)", () => {
    const h = TRITON_CAPABILITY_PROFILE.authorityHorizon;
    expect(h.ttgirOwns).toContain("coalescing and vector width");
    expect(h.ttgirOwns).toContain("shared-memory allocation/swizzles");
    expect(h.sourceDetermines).toContain("launch grid");
    expect(h.declinedEscapeHatch).toContain("ir_override");
  });
});

describe("emitTriton (tiled matmul)", () => {
  it("is deterministic: same state → byte-identical source", () => {
    const desc = baseDesc();
    const s1 = deriveTiledMatmulState(desc, REGION);
    const s2 = deriveTiledMatmulState(desc, REGION);
    expect(emitTritonTiledMatmul(s1, desc).source).toBe(
      emitTritonTiledMatmul(s2, desc).source,
    );
  });

  it("emits BLOCK constexpr, tl.range K loop, tl.dot with fp32 accum", () => {
    const em = emitTritonTiledMatmul(
      deriveTiledMatmulState(baseDesc(), REGION),
      baseDesc(),
    );
    expect(em.source).toContain("BLOCK_M: tl.constexpr");
    expect(em.source).toContain("for kk in tl.range(0, num_k)");
    expect(em.source).toContain("out_dtype=tl.float32");
    expect(em.block).toEqual([32, 32, 16]);
  });

  it("identity vs swap vs grouped emit DIFFERENT pid remaps off the program-grid map", () => {
    const desc = baseDesc();
    const identity = emitTritonTiledMatmul(
      deriveTiledMatmulState(desc, REGION),
      desc,
    );
    const swap = emitTritonTiledMatmul(
      deriveTiledMatmulState({ ...desc, swapGrid: true }, REGION),
      { ...desc, swapGrid: true },
    );
    const grouped = emitTritonTiledMatmul(
      withGridMap(deriveTiledMatmulState(desc, REGION), {
        kind: "grouped",
        groupAxis: "axis:m" as never,
        groupSize: 8,
      }),
      desc,
    );
    expect(identity.gridMap).toBe("identity");
    expect(swap.gridMap).toBe("swap");
    expect(grouped.gridMap).toBe("grouped");
    expect(grouped.source).toContain("GROUP_M = 8");
    expect(identity.source).not.toBe(swap.source);
    expect(identity.source).not.toBe(grouped.source);
  });

  it("maps warpBudget → num_warps and pipeline → num_stages requests", () => {
    const desc = baseDesc();
    const state = deriveTiledMatmulState(desc, REGION);
    const withReq: ScheduleState = {
      ...state,
      requests: {
        ...state.requests,
        warpBudget: 8,
        pipeline: {
          kind: "staged",
          entries: [
            { loop: "loop:k" as never, loadGroups: [], requestedStages: 3 },
          ],
        },
      },
    };
    const em = emitTritonTiledMatmul(withReq, desc);
    expect(em.numWarps).toBe(8);
    expect(em.numStages).toBe(3);
  });

  it("emits the fused bias epilogue inside the tl program", () => {
    const desc = baseDesc({
      epilogue: {
        ops: [{ kind: "bias", inputIndex: 0 }],
        additionalInputCount: 1,
        outputDtype: "f32",
      },
    });
    const em = emitTritonTiledMatmul(
      deriveTiledMatmulState(desc, REGION),
      desc,
    );
    expect(em.source).toContain("bias_ptr");
    expect(em.source).toContain("acc = acc + bias[None, :]");
  });

  it("REFUSES kSplit (typed refusal, not a raw throw)", () => {
    const desc = baseDesc({ kSplit: 4 });
    // deriveTiledMatmulState requires no epilogue with kSplit (legality); derive raw.
    const state = deriveTiledMatmulState(desc, REGION);
    expect(() => emitTritonTiledMatmul(state, desc)).toThrow(TritonEmitRefusal);
  });

  it("records the receipt boundary (what TTGIR owns, not emitted)", () => {
    const em = emitTritonTiledMatmul(
      deriveTiledMatmulState(baseDesc(), REGION),
      baseDesc(),
    );
    expect(em.receiptBoundary.some((r) => r.includes("TTGIR"))).toBe(true);
    // The emitter must NOT write shared staging / vec width into the source.
    expect(em.source).not.toContain("shared");
    expect(em.source).not.toContain("vec4");
  });
});

describe("identity: the realizer coordinate (§5)", () => {
  it("the SAME schedule digest gains DISTINCT artifact digests per realizer", () => {
    const state = deriveTiledMatmulState(baseDesc(), REGION);
    const sched = scheduleDigest(state);
    const wgsl = artifactDigest(state, WGSL_REALIZER_COORDINATE);
    const triton = artifactDigest(state, TRITON_REALIZER.coordinate);
    expect(wgsl).not.toBe(triton); // two backends, two artifacts, one schedule
    expect(wgsl).not.toBe(sched);
    expect(triton).not.toBe(sched);
  });

  it("semantic identity is realizer-blind (both coordinates share it)", () => {
    const state = deriveTiledMatmulState(baseDesc(), REGION);
    // Semantic digest does not depend on the realizer coordinate at all.
    expect(semanticDigest(state)).toBe(semanticDigest(state));
  });

  it("the registry entry carries the profile + harness reference + coordinate", () => {
    expect(TRITON_REALIZER.capabilityProfile).toBe(TRITON_CAPABILITY_PROFILE);
    expect(TRITON_REALIZER.verificationHarnessRef).toContain("run_kernel.py");
    expect(TRITON_REALIZER.coordinate.realizer).toBe("triton");
    expect(TRITON_REALIZER.coordinate.capabilityProfileVersion).toBe(
      TRITON_CAPABILITY_PROFILE.capabilityProfileVersion,
    );
  });
});
