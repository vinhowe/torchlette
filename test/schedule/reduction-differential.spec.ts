/**
 * P0-FULL wave-1 byte differential — REDUCTION + ROW-PROGRAM families.
 *
 * For each corpus kernel, assert the schema round-trip
 *
 *   compileTileKernel(applySchedule(deriveScheduleState(k), k))  ==BYTES==  compileTileKernel(<liveSpec>)
 *
 * (reduction: applyReductionSchedule; row-program: applyRowProgramSchedule.)
 *
 * PURE-COMPILATION (no GPU): `compileTileKernel` is a WGSL compiler; both specs
 * compile in ONE process (identical device/subgroup detection), so byte-identity
 * holds. Lives in the cpu project, runs by default. Exercises the no-second-owner
 * seam assertions on every derived state (§12 check 3, family-local).
 *
 * ALSO gates the canonical printer: digest stability (same state twice → identical
 * text + digest) and move-script replay (a synthetic 3-move script replays to a
 * digest-identical state).
 */

import { describe, expect, it } from "vitest";
import { argReduceWGSL } from "../../src/backend/webgpu/ops/ops-tile-ir";
import {
  makeMeanDivSpec,
  makeReductionSpec,
  type PreambleChainKernelOp,
  type ReductionEpilogueOpDesc,
} from "../../src/backend/webgpu/reduction-tile-ir";
import { rowProgramToSpec } from "../../src/backend/webgpu/row-program-codegen";
import { contiguousStrides } from "../../src/backend/webgpu/shape-utils";
import { compileTileKernel } from "../../src/backend/webgpu/tile-compiler";
import type { RowProgram } from "../../src/compiler/row-program-types";
import {
  digestText,
  type MoveScript,
  printMove,
  printMoveScript,
  printScheduleState,
  scheduleDigest,
} from "../../src/schedule/canonical";
import {
  applyArgReduceSchedule,
  applyMeanDivSchedule,
  applyReductionSchedule,
  applyRowProgramSchedule,
  type DimReductionInfo,
  deriveArgReduceState,
  deriveReductionState,
  deriveRowProgramState,
  type ReductionDescriptor,
} from "../../src/schedule/reduction-skeleton";
import type {
  AxisUid,
  LoopUid,
  RoleName,
  ScheduleState,
  SemanticRegionUid,
} from "../../src/schedule/types";

const REGION = "region:reduction-test" as unknown as SemanticRegionUid;

/** Build a DimReductionInfo the way reductions.ts's prepareDimReduction does. */
function dimSetup(
  inputShape: number[],
  normalizedDims: number[],
  keepdim: boolean,
): DimReductionInfo {
  const rank = inputShape.length;
  const outShape: number[] = [];
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) outShape.push(1);
    } else outShape.push(inputShape[i]);
  }
  const inputStrides = contiguousStrides(inputShape);
  const outStrides = contiguousStrides(outShape);
  let reductionSize = 1;
  for (const d of normalizedDims) reductionSize *= inputShape[d];
  const inputToOutDim: number[] = [];
  let outDimIdx = 0;
  for (let i = 0; i < rank; i++) {
    if (normalizedDims.includes(i)) {
      if (keepdim) inputToOutDim.push(outDimIdx++);
      else inputToOutDim.push(-1);
    } else inputToOutDim.push(outDimIdx++);
  }
  return {
    inputShape,
    inputStrides,
    normalizedDims,
    outShape,
    outStrides,
    inputToOutDim,
    parallel: reductionSize > 64,
  };
}

let covered = 0;

describe("P0 reduction + row-program walking-skeleton byte differential", () => {
  // ---- FULL (scalar) reductions ----
  const fullOps: Array<"sum" | "max" | "min"> = ["sum", "max", "min"];
  for (const op of fullOps) {
    it(`full ${op}`, () => {
      const desc: ReductionDescriptor = { reduceOp: op };
      const live = compileTileKernel(makeReductionSpec({ reduceOp: op }));
      const state = deriveReductionState(desc, REGION);
      const derived = compileTileKernel(applyReductionSchedule(state, desc));
      expect(derived).toBe(live);
      covered++;
    });
  }

  // ---- DIM reductions: sequential (small) and parallel (large) ----
  const dimCases: Array<{
    op: "sum" | "max" | "min";
    shape: number[];
    dims: number[];
    keepdim: boolean;
  }> = [
    { op: "sum", shape: [4, 8], dims: [1], keepdim: false }, // seq (8 <= 64)
    { op: "max", shape: [2, 3, 4], dims: [1], keepdim: false }, // seq
    { op: "min", shape: [4, 8], dims: [0], keepdim: true }, // seq keepdim
    { op: "sum", shape: [4, 128], dims: [1], keepdim: false }, // parallel (128 > 64)
    { op: "max", shape: [3, 100], dims: [1], keepdim: true }, // parallel keepdim
  ];
  for (const c of dimCases) {
    it(`dim ${c.op} ${c.shape.join("x")} d${c.dims.join(",")}${c.keepdim ? " keep" : ""}`, () => {
      const dim = dimSetup(c.shape, c.dims, c.keepdim);
      const desc: ReductionDescriptor = { reduceOp: c.op, dim };
      const live = compileTileKernel(
        makeReductionSpec({
          reduceOp: c.op,
          dim: {
            inputShape: dim.inputShape as number[],
            inputStrides: dim.inputStrides as number[],
            normalizedDims: dim.normalizedDims as number[],
            outShape: dim.outShape as number[],
            outStrides: dim.outStrides as number[],
            inputToOutDim: dim.inputToOutDim as number[],
            parallel: dim.parallel,
          },
        }),
      );
      const state = deriveReductionState(desc, REGION);
      const derived = compileTileKernel(applyReductionSchedule(state, desc));
      expect(derived).toBe(live);
      covered++;
    });
  }

  // ---- PREAMBLE chain (sum of a fused elementwise chain — e.g. sum(cast(x)*y)) ----
  it("preamble chain (full sum of mul+cast)", () => {
    const chainOps: PreambleChainKernelOp[] = [
      { op: "mul", arity: 2 },
      { op: "cast_f32", arity: 1 },
    ];
    const desc: ReductionDescriptor = {
      reduceOp: "sum",
      preamble: { chainOps, totalInputs: 2, inputDtypes: ["f32", "f32"] },
    };
    const cfg = {
      reduceOp: "sum" as const,
      preamble: {
        chainOps,
        totalInputs: 2,
        inputDtypes: ["f32", "f32"] as const,
      },
    };
    const live = compileTileKernel(
      makeReductionSpec({
        reduceOp: "sum",
        preamble: { chainOps, totalInputs: 2, inputDtypes: ["f32", "f32"] },
      }),
    );
    void cfg;
    const state = deriveReductionState(desc, REGION);
    const derived = compileTileKernel(applyReductionSchedule(state, desc));
    expect(derived).toBe(live);
    covered++;
  });

  // ---- EPILOGUE chain (dim sum → cast f16) ----
  it("epilogue chain (dim sum -> cast f16)", () => {
    const dim = dimSetup([4, 8], [1], false);
    const epilogueOps: ReductionEpilogueOpDesc[] = [
      { kind: "cast", toDtype: "f16" },
    ];
    const desc: ReductionDescriptor = {
      reduceOp: "sum",
      dim,
      epilogue: { ops: epilogueOps, outputDtype: "f16" },
    };
    const live = compileTileKernel(
      makeReductionSpec({
        reduceOp: "sum",
        dim: {
          inputShape: dim.inputShape as number[],
          inputStrides: dim.inputStrides as number[],
          normalizedDims: dim.normalizedDims as number[],
          outShape: dim.outShape as number[],
          outStrides: dim.outStrides as number[],
          inputToOutDim: dim.inputToOutDim as number[],
          parallel: dim.parallel,
        },
        epilogue: { ops: epilogueOps, outputDtype: "f16" },
      }),
    );
    const state = deriveReductionState(desc, REGION);
    const derived = compileTileKernel(applyReductionSchedule(state, desc));
    expect(derived).toBe(live);
    covered++;
  });

  // ---- MEAN-DIV elementwise (sum -> ÷count) ----
  it("meanDiv", () => {
    const live = compileTileKernel(makeMeanDivSpec());
    const derived = compileTileKernel(applyMeanDivSchedule());
    expect(derived).toBe(live);
    covered++;
  });

  // ---- ROW-PROGRAM: softmax (the FA-adjacent reduction) ----
  const softmax: RowProgram = {
    inputs: [{ dtype: "f32" }],
    output: { dtype: "f32" },
    dim: 1,
    phases: [
      {
        kind: "reduce",
        reduceOp: "max",
        bodyExpr: { kind: "input", bufferIndex: 0 },
      },
      {
        kind: "reduce",
        reduceOp: "sum",
        bodyExpr: {
          op: "exp",
          inputs: [
            {
              op: "sub",
              inputs: [
                { kind: "input", bufferIndex: 0 },
                { kind: "reduceResult", phaseIndex: 0 },
              ],
            },
          ],
        },
      },
      {
        kind: "write",
        bodyExpr: {
          op: "div",
          inputs: [
            {
              op: "exp",
              inputs: [
                {
                  op: "sub",
                  inputs: [
                    { kind: "input", bufferIndex: 0 },
                    { kind: "reduceResult", phaseIndex: 0 },
                  ],
                },
              ],
            },
            { kind: "reduceResult", phaseIndex: 1 },
          ],
        },
      },
    ],
    cacheKey: "test_softmax_3x4",
  };

  it("row-program softmax (FA-adjacent)", () => {
    const live = compileTileKernel(rowProgramToSpec(softmax));
    const state = deriveRowProgramState(softmax, REGION);
    const derived = compileTileKernel(applyRowProgramSchedule(state, softmax));
    expect(derived).toBe(live);
    covered++;
  });

  // ---- ROW-PROGRAM: layernorm inv-std scalar output (mean + rsqrt, scalarOutput) ----
  const layernormInvStd: RowProgram = {
    inputs: [{ dtype: "f32" }],
    output: { dtype: "f32" },
    dim: 1,
    phases: [
      // mean of x^2 (mean reduce → r0)
      {
        kind: "reduce",
        reduceOp: "sum",
        isMean: true,
        bodyExpr: {
          op: "mul",
          inputs: [
            { kind: "input", bufferIndex: 0 },
            { kind: "input", bufferIndex: 0 },
          ],
        },
      },
      // write rsqrt(r0 + eps) — a per-row scalar (reads no per-element input)
      {
        kind: "write",
        scalarOutput: true,
        bodyExpr: {
          op: "rsqrt",
          inputs: [
            {
              op: "add",
              inputs: [
                { kind: "reduceResult", phaseIndex: 0 },
                { kind: "const", value: 1e-5 },
              ],
            },
          ],
        },
      },
    ],
    cacheKey: "test_layernorm_invstd_3x4",
  };

  it("row-program layernorm inv-std (scalarOutput)", () => {
    const live = compileTileKernel(rowProgramToSpec(layernormInvStd));
    const state = deriveRowProgramState(layernormInvStd, REGION);
    const derived = compileTileKernel(
      applyRowProgramSchedule(state, layernormInvStd),
    );
    expect(derived).toBe(live);
    covered++;
  });

  // ---- ARG-REDUCE (argmax / argmin — the wave-1 leftover derivable kernel) ----
  const argCases: Array<{ op: ">" | "<"; shape: number[]; dim: number }> = [
    { op: ">", shape: [4, 8], dim: 1 }, // argmax over last dim
    { op: "<", shape: [3, 5], dim: 1 }, // argmin over last dim
    { op: ">", shape: [4, 8], dim: 0 }, // argmax over first dim
  ];
  for (const c of argCases) {
    it(`arg-reduce ${c.op === ">" ? "argmax" : "argmin"} ${c.shape.join("x")} d${c.dim}`, () => {
      const dim = dimSetup(c.shape, [c.dim], false);
      const desc = {
        compareOp: c.op,
        inputShape: dim.inputShape,
        inputStrides: dim.inputStrides,
        outShape: dim.outShape,
        dim: c.dim,
        inputToOutDim: dim.inputToOutDim,
      };
      const live = argReduceWGSL(
        c.op,
        c.shape,
        dim.inputStrides as number[],
        dim.outShape as number[],
        c.dim,
        dim.inputToOutDim as number[],
      );
      const state = deriveArgReduceState(desc, REGION);
      const derived = applyArgReduceSchedule(state, desc);
      expect(derived).toBe(live);
      covered++;
    });
  }

  it("reports the covered kernel count", () => {
    // 3 full + 5 dim + 1 preamble + 1 epilogue + 1 meanDiv + 2 row-program + 3 arg = 16
    expect(covered).toBeGreaterThanOrEqual(16);
    // eslint-disable-next-line no-console
    console.log(
      `[P0 reduction+row-program differential] byte-identical kernels covered: ${covered}`,
    );
  });
});

// ============================================================================
// Canonical printer: digest stability + move-script replay
// ============================================================================

describe("P0 canonical printer — digest stability + move-script replay", () => {
  const softmax: RowProgram = {
    inputs: [{ dtype: "f32" }],
    output: { dtype: "f32" },
    dim: 1,
    phases: [
      {
        kind: "reduce",
        reduceOp: "max",
        bodyExpr: { kind: "input", bufferIndex: 0 },
      },
      { kind: "write", bodyExpr: { kind: "input", bufferIndex: 0 } },
    ],
    cacheKey: "test_softmax_digest",
  };

  it("same state printed twice -> identical text and digest (no Date/random/order leak)", () => {
    const s1 = deriveRowProgramState(softmax, REGION);
    const s2 = deriveRowProgramState(softmax, REGION);
    const t1 = printScheduleState(s1);
    const t2 = printScheduleState(s2);
    expect(t1).toBe(t2);
    expect(scheduleDigest(s1)).toBe(scheduleDigest(s2));
    // A digest is a 128-bit hex string (strong, not 32-bit FNV).
    expect(scheduleDigest(s1)).toMatch(/^[0-9a-f]{32}$/);
  });

  it("golden test vector: printed form + digest are pinned across process restarts (R27)", () => {
    // A fixed state → a fixed printed text → a fixed digest. Because the digest
    // is a pure function of the printed text (no Date/random/Map-order), this
    // value is stable across process restarts. If the canonical form changes,
    // bump CANONICAL_SCHEMA_VERSION and this vector (R27: test vectors required).
    const s = deriveReductionState(
      { reduceOp: "sum" },
      "region:golden" as unknown as SemanticRegionUid,
    );
    const printed = printScheduleState(s);
    expect(printed).toContain("body reduced = reduce_sum(v(result))");
    expect(printed).toContain("store src=result -> tgt=out:out @loop:out");
    // The golden digests (recomputed identically in a fresh process).
    expect(scheduleDigest(s)).toBe("ddfec844386f1ac5850972312794c0b6");
    // Re-derive the digest by hand from the printed text: digest === digest(print).
    expect(digestText(printed)).toBe("ddfec844386f1ac5850972312794c0b6");
  });

  it("digest changes when the state changes", () => {
    const s1 = deriveReductionState({ reduceOp: "sum" }, REGION);
    const s2 = deriveReductionState({ reduceOp: "max" }, REGION);
    expect(scheduleDigest(s1)).not.toBe(scheduleDigest(s2));
  });

  it("move-script: printed script + a synthetic 3-move program replay to a digest-identical state", () => {
    // The base state.
    const base = deriveReductionState({ reduceOp: "sum" }, REGION);
    const baseDigest = scheduleDigest(base);

    // A synthetic 3-move script over the base (moves are the typed schema; this
    // wave prints + replays them — it does NOT execute the move algebra (P2)).
    const script: MoveScript = {
      baseDigest,
      moves: [
        {
          move: "tile",
          loop: "loop:out" as unknown as LoopUid,
          axis: "axis:out" as unknown as AxisUid,
          factor: 4,
        },
        {
          move: "program-map",
          map: { kind: "identity" },
        },
        {
          move: "role-partition",
          loop: "loop:reduce" as unknown as LoopUid,
          roles: ["cooperative" as unknown as RoleName],
        },
      ],
    };

    const printed = printMoveScript(script);
    // The printed script is stable, references the base by digest, and prints
    // each move on its own numbered line.
    expect(printed).toContain(`base ${baseDigest}`);
    expect(printed.split("\n")).toHaveLength(2 + script.moves.length);
    for (let i = 0; i < script.moves.length; i++)
      expect(printed).toContain(`${i}: ${printMove(script.moves[i])}`);

    // Replay: applying the SAME script's moves to a FRESH copy of the base must
    // reproduce a digest-identical (base) result. This wave's replay is
    // identity over the move set (the move algebra is P2), so the replayed
    // state's digest equals the base digest — proving the replayer consumes the
    // typed script (not text) and lands on the recorded base.
    const replayed = replayMoveScript(base, script);
    expect(scheduleDigest(replayed)).toBe(baseDigest);
    // And a second print of the script is byte-identical (no order/random leak).
    expect(printMoveScript(script)).toBe(printed);
  });
});

/**
 * Replay a move-script: apply each typed move to the base state in order,
 * returning the resulting state. P0-full wave 1 ships the printer + replayer
 * SURFACE; the executable move algebra (partial functions mutating the semantic
 * schedule) is P2. Until then, replay verifies the SCRIPT SHAPE round-trips: it
 * consumes the typed `MoveScript` (not text), threads the base state through the
 * ordered moves, and asserts the base digest the script references matches — so
 * a printed-then-replayed script provably lands on the recorded base state.
 */
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
    // The move algebra is P2; this wave's replay is the identity transform that
    // still WALKS every typed move (so a malformed/out-of-schema move throws at
    // print time via printMove's exhaustive switch).
    void printMove(move);
    state = { ...state };
  }
  return state;
}
