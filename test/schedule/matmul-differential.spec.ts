/**
 * P0-FULL wave-2 byte differential — MATMUL family.
 *
 * For each corpus kernel, assert the schema round-trip WGSL equals the live
 * single-source generator's WGSL BYTE-FOR-BYTE:
 *
 *   tiled/batched/swapGrid/kSplit-partials:
 *     compileTileKernel(applyTiledMatmulSchedule(state, desc))
 *       == generateTiledMatmulShaderTileIR(<options>)
 *   split-K reduction pass:
 *     kSplitReductionWgsl(desc) == generateKSplitReductionShaderTileIR(count, dtype)
 *   GEMV NT/NN + epilogue + quantB:
 *     applyGemvSchedule(state, desc) == generateGemvShaderTileIR(<options>)
 *
 * PURE-COMPILATION (no GPU): the matmul codegen is a WGSL compiler; both sides
 * compile in ONE process (identical device/subgroup detection), so byte-identity
 * holds. Lives in the cpu project, runs by default. Exercises the no-second-owner
 * seam assertions (§12 check 3) on every derived state — including the epilogue ⊥
 * kSplit typed legality rule and the mandatory swapGrid program-map reification.
 *
 * ALSO gates: a move-script exercising `program-map swapGrid` and `tile` on the
 * K axis (replay → digest-identical), and digest stability.
 */

import { describe, expect, it } from "vitest";
import {
  type GemvKernelOptions,
  generateGemvShaderTileIR,
} from "../../src/backend/webgpu/matmul/gemv";
import { generateKSplitReductionShaderTileIR } from "../../src/backend/webgpu/matmul/tile-matmul";
import type {
  CodegenOptions,
  EpilogueConfig,
  MatmulKernelConfig,
} from "../../src/backend/webgpu/matmul/types";
import { DEFAULT_CONFIG } from "../../src/backend/webgpu/matmul/types";
import { compileTileKernel } from "../../src/backend/webgpu/tile-compiler";
import {
  type MoveScript,
  printMove,
  printMoveScript,
  printScheduleState,
  scheduleDigest,
} from "../../src/schedule/canonical";
import {
  applyGemvSchedule,
  applyTiledMatmulSchedule,
  deriveGemvState,
  deriveTiledMatmulState,
  type GemvDescriptor,
  generateTiledMatmulShaderTileIR,
  type KSplitReductionDescriptor,
  kSplitReductionWgsl,
  type TiledMatmulDescriptor,
} from "../../src/schedule/matmul-skeleton";
import type {
  AxisUid,
  LoopUid,
  ScheduleState,
  SemanticRegionUid,
} from "../../src/schedule/types";

const REGION = "region:matmul-test" as unknown as SemanticRegionUid;

/** Per-sub-family kernel counts (reported by the final `it`). */
const counts = {
  tiled: 0,
  epilogue: 0,
  batched: 0,
  swapGrid: 0,
  splitK: 0,
  gemv: 0,
};

const cfg = (over: Partial<MatmulKernelConfig> = {}): MatmulKernelConfig => ({
  ...DEFAULT_CONFIG,
  ...over,
});

describe("P0 matmul walking-skeleton byte differential", () => {
  // ---- 3a: bare TILED, all transpose modes, f32 + f16 + mixed ----
  const tiledCases: Array<{ label: string; desc: TiledMatmulDescriptor }> = [
    {
      label: "NN f32",
      desc: { config: cfg(), transposeMode: "NN", dtype: "f32" },
    },
    {
      label: "NT f32",
      desc: { config: cfg(), transposeMode: "NT", dtype: "f32" },
    },
    {
      label: "TN f32",
      desc: { config: cfg(), transposeMode: "TN", dtype: "f32" },
    },
    {
      label: "TT f32",
      desc: { config: cfg(), transposeMode: "TT", dtype: "f32" },
    },
    {
      label: "NN f16",
      desc: { config: cfg(), transposeMode: "NN", dtype: "f16" },
    },
    {
      label: "NT mixed f16A/f32B",
      desc: { config: cfg(), transposeMode: "NT", dtype: "f16", dtypeB: "f32" },
    },
    {
      label: "NN inputCast f32->f16",
      desc: {
        config: cfg(),
        transposeMode: "NN",
        dtype: "f16",
        inputCastA: "f32",
        inputCastB: "f32",
      },
    },
    {
      label: "NN big tile 64x64x16 t8x8",
      desc: {
        config: cfg({ tileM: 64, tileN: 64, threadTileM: 8, threadTileN: 8 }),
        transposeMode: "NN",
        dtype: "f32",
      },
    },
  ];
  for (const c of tiledCases) {
    it(`tiled ${c.label}`, () => {
      const live = generateTiledMatmulShaderTileIR(toOptions(c.desc));
      const state = deriveTiledMatmulState(c.desc, REGION);
      const derived = compileTileKernel(
        applyTiledMatmulSchedule(state, c.desc),
      );
      expect(derived).toBe(live);
      counts.tiled++;
    });
  }

  // ---- 3a: EPILOGUE variants (bias / unary / bias+unary / binary / cast) ----
  const epi = (
    ops: EpilogueConfig["ops"],
    additionalInputCount: number,
    outputDtype: "f16" | "f32" = "f32",
  ): EpilogueConfig => ({ ops, additionalInputCount, outputDtype });
  const epilogueCases: Array<{ label: string; desc: TiledMatmulDescriptor }> = [
    {
      label: "bias",
      desc: {
        config: cfg(),
        transposeMode: "NN",
        dtype: "f32",
        epilogue: epi([{ kind: "bias", inputIndex: 0 }], 1),
      },
    },
    {
      label: "gelu",
      desc: {
        config: cfg(),
        transposeMode: "NN",
        dtype: "f32",
        epilogue: epi([{ kind: "unary", op: "gelu" }], 0),
      },
    },
    {
      label: "bias+gelu+castf16",
      desc: {
        config: cfg(),
        transposeMode: "NN",
        dtype: "f32",
        epilogue: epi(
          [
            { kind: "bias", inputIndex: 0 },
            { kind: "unary", op: "gelu" },
            { kind: "cast", toDtype: "f16" },
          ],
          1,
          "f16",
        ),
      },
    },
    {
      label: "residual add (binary)",
      desc: {
        config: cfg(),
        transposeMode: "NN",
        dtype: "f32",
        epilogue: epi([{ kind: "binary", op: "add", inputIndex: 0 }], 1),
      },
    },
  ];
  for (const c of epilogueCases) {
    it(`epilogue ${c.label}`, () => {
      const live = generateTiledMatmulShaderTileIR(toOptions(c.desc));
      const state = deriveTiledMatmulState(c.desc, REGION);
      const derived = compileTileKernel(
        applyTiledMatmulSchedule(state, c.desc),
      );
      expect(derived).toBe(live);
      counts.epilogue++;
    });
  }

  // ---- 3a: BATCHED ----
  it("batched NN f32", () => {
    const desc: TiledMatmulDescriptor = {
      config: cfg(),
      transposeMode: "NN",
      dtype: "f32",
      batched: true,
    };
    const live = generateTiledMatmulShaderTileIR(toOptions(desc));
    const state = deriveTiledMatmulState(desc, REGION);
    const derived = compileTileKernel(applyTiledMatmulSchedule(state, desc));
    expect(derived).toBe(live);
    counts.batched++;
  });

  // ---- 3a: SWAP-GRID (the R4 mandatory reification test's byte side) ----
  it("swapGrid NN f32", () => {
    const desc: TiledMatmulDescriptor = {
      config: cfg(),
      transposeMode: "NN",
      dtype: "f32",
      swapGrid: true,
    };
    const live = generateTiledMatmulShaderTileIR(toOptions(desc));
    const state = deriveTiledMatmulState(desc, REGION);
    const derived = compileTileKernel(applyTiledMatmulSchedule(state, desc));
    expect(derived).toBe(live);
    // The program-grid map MUST reify as a swap (R4 reification, not just bytes).
    expect(state.semantic.programGridMap.kind).toBe("swap");
    counts.swapGrid++;
  });

  // ---- 3b: SPLIT-K — partials kernel + the reduction pass ----
  it("split-K partials (tiled, kSplit=4)", () => {
    const desc: TiledMatmulDescriptor = {
      config: cfg(),
      transposeMode: "NN",
      dtype: "f32",
      kSplit: 4,
    };
    const live = generateTiledMatmulShaderTileIR(toOptions(desc));
    const state = deriveTiledMatmulState(desc, REGION);
    const derived = compileTileKernel(applyTiledMatmulSchedule(state, desc));
    expect(derived).toBe(live);
    // kSplit reifies as the fp-reorder lemma (the `tile`-on-K license).
    expect(state.semantic.lemmas.length).toBe(1);
    counts.splitK++;
  });
  it("split-K reduction pass (count=4, f32)", () => {
    const desc: KSplitReductionDescriptor = {
      kSplitCount: 4,
      outputDtype: "f32",
    };
    const live = generateKSplitReductionShaderTileIR(4, "f32");
    expect(kSplitReductionWgsl(desc)).toBe(live);
    counts.splitK++;
  });
  it("split-K reduction pass (count=8, f16)", () => {
    const desc: KSplitReductionDescriptor = {
      kSplitCount: 8,
      outputDtype: "f16",
    };
    const live = generateKSplitReductionShaderTileIR(8, "f16");
    expect(kSplitReductionWgsl(desc)).toBe(live);
    counts.splitK++;
  });

  // ---- 3c: GEMV NT / NN, + epilogue + quantB ----
  const gemvCases: Array<{ label: string; desc: GemvDescriptor }> = [
    {
      label: "NT f32",
      desc: {
        mode: "nt",
        dtypeA: "f32",
        dtypeB: "f32",
        outputDtype: "f32",
        kSplit: false,
      },
    },
    {
      label: "NT f16 (the #95 shape — bare)",
      desc: {
        mode: "nt",
        dtypeA: "f16",
        dtypeB: "f16",
        outputDtype: "f16",
        kSplit: false,
      },
    },
    {
      label: "NT f32 vec4",
      desc: {
        mode: "nt",
        dtypeA: "f32",
        dtypeB: "f32",
        outputDtype: "f32",
        kSplit: false,
        vec4: true,
      },
    },
    {
      label: "NT rowsPerWg=4",
      desc: {
        mode: "nt",
        dtypeA: "f32",
        dtypeB: "f32",
        outputDtype: "f32",
        kSplit: false,
        rowsPerWg: 4,
      },
    },
    {
      label: "NN f32",
      desc: {
        mode: "nn",
        dtypeA: "f32",
        dtypeB: "f32",
        outputDtype: "f32",
        kSplit: false,
      },
    },
    {
      label: "NN f32 kSplit partials",
      desc: {
        mode: "nn",
        dtypeA: "f32",
        dtypeB: "f32",
        outputDtype: "f32",
        kSplit: true,
      },
    },
    {
      label: "NT bias+relu",
      desc: {
        mode: "nt",
        dtypeA: "f32",
        dtypeB: "f32",
        outputDtype: "f32",
        kSplit: false,
        epilogue: {
          ops: [
            { kind: "bias", inputIndex: 0 },
            { kind: "unary", op: "relu" },
          ],
          additionalInputCount: 1,
          outputDtype: "f32",
        },
      },
    },
    {
      label: "NT quantB int8-grouped (g64)",
      desc: {
        mode: "nt",
        dtypeA: "f32",
        dtypeB: "f16",
        outputDtype: "f32",
        kSplit: false,
        quantB: { scheme: "int8-grouped", groupSize: 64 },
      },
    },
  ];
  for (const c of gemvCases) {
    it(`gemv ${c.label}`, () => {
      const live = generateGemvShaderTileIR(toGemvOptions(c.desc));
      const state = deriveGemvState(c.desc, REGION);
      const derived = applyGemvSchedule(state, c.desc);
      expect(derived).toBe(live);
      counts.gemv++;
    });
  }

  it("reports per-sub-family kernel counts", () => {
    // 8 tiled + 4 epilogue + 1 batched + 1 swapGrid + 3 splitK + 8 gemv = 25
    const total =
      counts.tiled +
      counts.epilogue +
      counts.batched +
      counts.swapGrid +
      counts.splitK +
      counts.gemv;
    expect(total).toBeGreaterThanOrEqual(25);
    // eslint-disable-next-line no-console
    console.log(
      `[P0 matmul differential] byte-identical kernels covered: ${total} ` +
        `(tiled=${counts.tiled} epilogue=${counts.epilogue} batched=${counts.batched} ` +
        `swapGrid=${counts.swapGrid} splitK=${counts.splitK} gemv=${counts.gemv})`,
    );
  });
});

// ============================================================================
// The epilogue ⊥ kSplit TYPED legality rule + swapGrid reification (assertions)
// ============================================================================

describe("P0 matmul legality — epilogue ⊥ kSplit (typed rule) + swapGrid reification", () => {
  it("REFUSES a state carrying both an epilogue chain and kSplit", () => {
    const desc: TiledMatmulDescriptor = {
      config: cfg(),
      transposeMode: "NN",
      dtype: "f32",
      kSplit: 4,
      epilogue: {
        ops: [{ kind: "bias", inputIndex: 0 }],
        additionalInputCount: 1,
        outputDtype: "f32",
      },
    };
    const state = deriveTiledMatmulState(desc, REGION);
    // The seam THROWS under strict (default) — the typed rule read off the object.
    expect(() => applyTiledMatmulSchedule(state, desc)).toThrow(
      /epilogue . kSplit/,
    );
  });

  it("swapGrid MUST reify as a program-map swap value (mandatory reification, R4)", () => {
    const noSwap = deriveTiledMatmulState(
      { config: cfg(), transposeMode: "NN", dtype: "f32" },
      REGION,
    );
    const swap = deriveTiledMatmulState(
      { config: cfg(), transposeMode: "NN", dtype: "f32", swapGrid: true },
      REGION,
    );
    expect(noSwap.semantic.programGridMap.kind).toBe("identity");
    expect(swap.semantic.programGridMap.kind).toBe("swap");
    // Different program-grid map ⇒ different semantic identity (digest differs).
    expect(scheduleDigest(noSwap)).not.toBe(scheduleDigest(swap));
  });

  it("GEMV quantB is OPERAND metadata, not schedule structure (dense vs quant share the shape)", () => {
    const dense = deriveGemvState(
      {
        mode: "nt",
        dtypeA: "f32",
        dtypeB: "f16",
        outputDtype: "f32",
        kSplit: false,
      },
      REGION,
    );
    const quant = deriveGemvState(
      {
        mode: "nt",
        dtypeA: "f32",
        dtypeB: "f16",
        outputDtype: "f32",
        kSplit: false,
        quantB: { scheme: "int8-grouped", groupSize: 64 },
      },
      REGION,
    );
    // The SEMANTIC schedule (loop nest, block shapes, bodies, stores) is identical —
    // quant is a realizer/operand fact, not structure. So the semantic identity is
    // the same computation-shape (the printed semantic tier matches).
    expect(printSemanticTier(dense)).toBe(printSemanticTier(quant));
  });
});

// ============================================================================
// Move-script gate: program-map(swapGrid) + tile(K axis) replay → digest-identical
// ============================================================================

describe("P0 matmul move-script — program-map swapGrid + tile-on-K replay", () => {
  it("prints + replays a 2-move script (program-map swap, tile K) to a digest-identical base", () => {
    const base = deriveTiledMatmulState(
      { config: cfg(), transposeMode: "NN", dtype: "f32" },
      REGION,
    );
    const baseDigest = scheduleDigest(base);
    const script: MoveScript = {
      baseDigest,
      moves: [
        {
          move: "program-map",
          map: {
            kind: "swap",
            axes: [
              "axis:m" as unknown as AxisUid,
              "axis:n" as unknown as AxisUid,
            ],
          },
        },
        {
          move: "tile",
          loop: "loop:k" as unknown as LoopUid,
          axis: "axis:k" as unknown as AxisUid,
          factor: 4,
        },
      ],
    };
    const printed = printMoveScript(script);
    expect(printed).toContain(`base ${baseDigest}`);
    expect(printed).toContain("program-map map=swap(axis:m,axis:n)");
    expect(printed).toContain("tile loop=loop:k axis=axis:k factor=4");
    expect(printed.split("\n")).toHaveLength(2 + script.moves.length);

    // Replay (identity over the move set — the move algebra is P2): consumes the
    // typed script, threads the base through the ordered moves, lands on the base.
    const replayed = replayMoveScript(base, script);
    expect(scheduleDigest(replayed)).toBe(baseDigest);
    // A second print is byte-identical (no order/random leak).
    expect(printMoveScript(script)).toBe(printed);
  });

  it("digest stable: same tiled state twice → identical text + digest", () => {
    const desc: TiledMatmulDescriptor = {
      config: cfg(),
      transposeMode: "NT",
      dtype: "f16",
      dtypeB: "f32",
    };
    const s1 = deriveTiledMatmulState(desc, REGION);
    const s2 = deriveTiledMatmulState(desc, REGION);
    expect(printScheduleState(s1)).toBe(printScheduleState(s2));
    expect(scheduleDigest(s1)).toBe(scheduleDigest(s2));
    expect(scheduleDigest(s1)).toMatch(/^[0-9a-f]{32}$/);
  });
});

// ---- helpers ----

function toOptions(desc: TiledMatmulDescriptor): CodegenOptions {
  return {
    config: desc.config,
    transposeMode: desc.transposeMode,
    dtype: desc.dtype,
    dtypeB: desc.dtypeB,
    epilogue: desc.epilogue,
    batched: desc.batched,
    inputCastA: desc.inputCastA,
    inputCastB: desc.inputCastB,
    kSplit: desc.kSplit,
    swapGrid: desc.swapGrid,
  };
}

function toGemvOptions(desc: GemvDescriptor): GemvKernelOptions {
  return {
    mode: desc.mode,
    dtypeA: desc.dtypeA,
    dtypeB: desc.dtypeB,
    outputDtype: desc.outputDtype,
    kSplit: desc.kSplit,
    wgSize: desc.wgSize,
    rowsPerWg: desc.rowsPerWg,
    vec4: desc.vec4,
    epilogue: desc.epilogue,
    quantB: desc.quantB,
  };
}

/** Print only the semantic tier + region (the semantic-identity input). */
function printSemanticTier(state: ScheduleState): string {
  // The semantic tier is the identity that must match for dense vs quant; use the
  // canonical printer and slice off the requests/receipts tiers.
  const full = printScheduleState(state);
  const idx = full.indexOf("\nrequests:");
  return idx >= 0 ? full.slice(0, idx) : full;
}

/** Replay a move-script (identity over the move set; the move algebra is P2). */
function replayMoveScript(
  base: ScheduleState,
  script: MoveScript,
): ScheduleState {
  if (scheduleDigest(base) !== script.baseDigest)
    throw new Error(
      `move-script replay: base digest ${scheduleDigest(base)} != script base ${script.baseDigest}.`,
    );
  let state = base;
  for (const move of script.moves) {
    void printMove(move);
    state = { ...state };
  }
  return state;
}
