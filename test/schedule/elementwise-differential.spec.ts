/**
 * P0 walking-skeleton byte differential — ELEMENTWISE family (campaign P0).
 *
 * For each op in the elementwise corpus (unary / binary / cast / where /
 * contiguous), assert that the schema round-trip
 *
 *     compileTileKernel(applySchedule(region, deriveScheduleState(desc), desc))
 *
 * is BYTE-IDENTICAL to the live builder's
 *
 *     compileTileKernel(<liveSpec>)
 *
 * This is a PURE-COMPILATION check (no GPU): `compileTileKernel` is a WGSL
 * compiler, so this test lives in the cpu project and runs by default. It also
 * exercises the structural no-second-owner assertion prototype on every derived
 * state (R22 / §7 P0 (d)).
 *
 * Coverage count is printed so the gate reports "N kernels covered".
 */

import { describe, expect, it } from "vitest";
import {
  binaryBroadcastSpec,
  castSpec,
  contiguousSpec,
  unaryStridedSpec,
  whereSpec,
} from "../../src/backend/webgpu/ops/ops-tile-ir";
import { compileTileKernel } from "../../src/backend/webgpu/tile-compiler";
import {
  applySchedule,
  assertNoSecondOwnerElementwise,
  deriveScheduleState,
  type ElementwiseKernelDescriptor,
} from "../../src/schedule/elementwise-skeleton";
import type {
  SemanticBodyNode,
  SemanticRegion,
  SemanticRegionUid,
  StridedAccess,
  ValueDtype,
} from "../../src/schedule/types";

const REGION_UID = "region:elementwise-test" as unknown as SemanticRegionUid;
// The elementwise family is fully summarized by the descriptor; the region is a
// placeholder node list here (a richer family would consume it node-by-node).
const EMPTY_REGION: SemanticRegion = { uid: REGION_UID, nodes: [] };

function access(
  indexShape: number[],
  strides: number[],
  offsetUniform: string,
): StridedAccess {
  return { indexShape, strides, offsetUniform };
}

/** Compile via the schema round-trip and via the live builder; return both. */
function roundTrip(
  desc: ElementwiseKernelDescriptor,
  liveWgsl: string,
): { derived: string; live: string } {
  const state = deriveScheduleState(desc, REGION_UID);
  assertNoSecondOwnerElementwise(state); // structural gate on every state
  const applied = applySchedule(EMPTY_REGION, state, desc);
  return { derived: compileTileKernel(applied), live: liveWgsl };
}

const valueRef = (binding: string): SemanticBodyNode => ({
  kind: "value",
  value: `in:${binding}` as never,
});

let covered = 0;

describe("P0 elementwise walking-skeleton byte differential", () => {
  // ---- UNARY ----
  const unaryCases: Array<[string, ValueDtype, ValueDtype]> = [
    ["relu", "f32", "f32"],
    ["neg", "f32", "f32"],
    ["exp", "f32", "f32"],
    ["sqrt", "f16", "f16"],
    ["tanh", "f32", "f32"],
    ["gelu_tanh", "f32", "f32"],
  ];
  for (const [op, dtype, outDtype] of unaryCases) {
    it(`unary ${op} (${dtype}->${outDtype})`, () => {
      const shape = [4, 8];
      const strides = [8, 1];
      const live = compileTileKernel(
        unaryStridedSpec(op, shape, strides, 0, dtype, outDtype),
      );
      const desc: ElementwiseKernelDescriptor = {
        name: `unary_${op}`,
        enableF16: dtype === "f16" || outDtype === "f16",
        inputs: [
          {
            binding: "a",
            dtype,
            access: access(shape, strides, "base_offset"),
          },
        ],
        output: { binding: "out", dtype: outDtype },
        body: { kind: "apply", catalog: { op }, args: [valueRef("a")] },
      };
      const { derived, live: liveOut } = roundTrip(desc, live);
      expect(derived).toBe(liveOut);
      covered++;
    });
  }

  // ---- BINARY ----
  const binaryCases: Array<[string, string]> = [
    ["+", "add"],
    ["*", "mul"],
    ["-", "sub"],
    ["/", "div"],
    ["max", "maximum"],
    ["min", "minimum"],
  ];
  for (const [wgslOp, fusionOp] of binaryCases) {
    it(`binary ${fusionOp}`, () => {
      const idxShape = [4, 8];
      const aStrides = [8, 1];
      const bStrides = [0, 1]; // broadcast on dim 0
      const dtype: ValueDtype = "f32";
      const live = compileTileKernel(
        binaryBroadcastSpec(wgslOp, idxShape, aStrides, bStrides, 0, 0, dtype),
      );
      const desc: ElementwiseKernelDescriptor = {
        name: `binary_${fusionOp}`,
        enableF16: false,
        inputs: [
          {
            binding: "a",
            dtype,
            access: access(idxShape, aStrides, "a_offset"),
          },
          {
            binding: "b",
            dtype,
            access: access(idxShape, bStrides, "b_offset"),
          },
        ],
        output: { binding: "out", dtype },
        body: {
          kind: "apply",
          catalog: { op: fusionOp },
          args: [valueRef("a"), valueRef("b")],
        },
      };
      const { derived, live: liveOut } = roundTrip(desc, live);
      expect(derived).toBe(liveOut);
      covered++;
    });
  }

  // ---- CAST ----
  const castCases: Array<[ValueDtype, ValueDtype]> = [
    ["f32", "f16"],
    ["f16", "f32"],
    ["f32", "i32"],
    ["i32", "f32"],
  ];
  for (const [src, dst] of castCases) {
    it(`cast ${src}->${dst}`, () => {
      const shape = [4, 8];
      const strides = [8, 1];
      const live = compileTileKernel(castSpec(src, dst, shape, strides, 0));
      const castOp = `cast_${dst}`;
      const desc: ElementwiseKernelDescriptor = {
        name: `cast_${src}_${dst}`,
        enableF16: src === "f16" || dst === "f16",
        inputs: [
          {
            binding: "a",
            dtype: src,
            access: access(shape, strides, "base_offset"),
          },
        ],
        output: { binding: "out", dtype: dst },
        body:
          src === dst
            ? valueRef("a")
            : { kind: "apply", catalog: { op: castOp }, args: [valueRef("a")] },
      };
      const { derived, live: liveOut } = roundTrip(desc, live);
      expect(derived).toBe(liveOut);
      covered++;
    });
  }

  // ---- WHERE (ternary select) ----
  it("where (select)", () => {
    const idxShape = [4, 8];
    const condStrides = [8, 1];
    const xStrides = [8, 1];
    const yStrides = [0, 1];
    const live = compileTileKernel(
      whereSpec(idxShape, condStrides, xStrides, yStrides, 0, 0, 0),
    );
    const dtype: ValueDtype = "f32";
    const desc: ElementwiseKernelDescriptor = {
      name: "where",
      enableF16: false,
      inputs: [
        {
          binding: "cond",
          dtype,
          access: access(idxShape, condStrides, "cond_offset"),
        },
        { binding: "x", dtype, access: access(idxShape, xStrides, "x_offset") },
        { binding: "y", dtype, access: access(idxShape, yStrides, "y_offset") },
      ],
      output: { binding: "out", dtype },
      body: {
        kind: "apply",
        catalog: { op: "select" },
        args: [valueRef("cond"), valueRef("x"), valueRef("y")],
      },
    };
    const { derived, live: liveOut } = roundTrip(desc, live);
    expect(derived).toBe(liveOut);
    covered++;
  });

  // ---- CONTIGUOUS (strided copy) ----
  const contigCases: ValueDtype[] = ["f32", "f16", "i32", "u32"];
  for (const dtype of contigCases) {
    it(`contiguous ${dtype}`, () => {
      const shape = [4, 8];
      const strides = [1, 4]; // transposed view
      const live = compileTileKernel(contiguousSpec(shape, strides, 0, dtype));
      const desc: ElementwiseKernelDescriptor = {
        name: `contiguous_${dtype}`,
        enableF16: dtype === "f16",
        inputs: [
          {
            binding: "input",
            dtype,
            access: access(shape, strides, "base_offset"),
          },
        ],
        output: { binding: "out", dtype },
        body: valueRef("input"), // identity copy
      };
      const { derived, live: liveOut } = roundTrip(desc, live);
      expect(derived).toBe(liveOut);
      covered++;
    });
  }

  it("reports the covered kernel count", () => {
    // 6 unary + 6 binary + 4 cast + 1 where + 4 contiguous = 21
    expect(covered).toBeGreaterThanOrEqual(21);
    // eslint-disable-next-line no-console
    console.log(
      `[P0 elementwise differential] byte-identical kernels covered: ${covered}`,
    );
  });
});
