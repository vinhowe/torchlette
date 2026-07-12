/**
 * P2A Commit C — OFF-MENU lowering proofs (the wave-B readiness gate).
 *
 * The byte-differentials (test/schedule/*-differential.spec.ts) prove the
 * schedule object reproduces every kernel the LIVE builders emit. That alone is
 * consistent with a lowering that only works ON the menu — a replay in disguise.
 * This spec proves the stronger property wave B (moves) needs: `apply*Schedule`
 * lowers a LEGAL ScheduleState that NO existing builder config produces, to
 * WGSL that compiles AND is numerically correct against the UNMOVED state(s)
 * on the same GPU inputs.
 *
 * One proof per LIVE family (P1 matmul, P2A elementwise + reduction):
 *
 *   1. MATMUL   — tile config {tileM:48, tileN:32, tileK:12}: legal per
 *      `validateConfig`, but 48 ∉ TUNING_SPACE.tileM/N and 12 ∉ TUNING_SPACE.tileK,
 *      ≠ DEFAULT_CONFIG, and no shape-class table names it (grep-clean in src/).
 *      Off-menu output == DEFAULT_CONFIG (unmoved) output == CPU reference,
 *      with M/N/K tails exercised (M=90, N=50, K=54).
 *   2. ELEMENTWISE — a depth-2 op-catalog body mul(add(a,b), a). Every live
 *      builder emits a depth-1 body (unary/binary/cast/select/identity); the
 *      nested tree exists in the schema but on no menu. Off-menu single-kernel
 *      output == the two-dispatch composition of unmoved states (add, then mul).
 *   3. REDUCTION — reduceOp "max" WITH a preamble (abs → amax). The only live
 *      preamble site (`sumWithPreambleEpilogue`) is sum-only; max×preamble is a
 *      legal descriptor no live dispatch constructs. Off-menu single-kernel
 *      output == the composition of unmoved states (abs elementwise, then plain
 *      max reduction) == CPU reference.
 *
 * Each proof also pins digest DISTINCTNESS (the off-menu state is a genuinely
 * different point in schedule space, not a re-derivation of the reference).
 */

import { beforeAll, describe, expect, it } from "vitest";
import {
  beginSharedEncoder,
  flushSharedEncoder,
  getWebGPUDevice,
  initWebGPU,
} from "../../src/backend/webgpu";
import type {
  GPUBuffer,
  GPUDevice,
  GPUQueue,
} from "../../src/backend/webgpu/gpu-types";
import { GPUBufferUsage, GPUMapMode } from "../../src/backend/webgpu/gpu-types";
import type { MatmulKernelConfig } from "../../src/backend/webgpu/matmul/types";
import {
  DEFAULT_CONFIG,
  TUNING_SPACE,
  validateConfig,
} from "../../src/backend/webgpu/matmul/types";
import { compileTileKernel } from "../../src/backend/webgpu/tile-compiler";
import { createTileKernelDispatcher } from "../../src/backend/webgpu/tile-dispatch";
import {
  type TileKernelSpec,
  tiledGrid,
} from "../../src/backend/webgpu/tile-ir";
import { scheduleDigest } from "../../src/schedule/canonical";
import {
  applySchedule,
  assertNoSecondOwnerElementwise,
  deriveScheduleState,
  type ElementwiseKernelDescriptor,
  realizeBinaryBroadcastSpec,
  realizeUnaryStridedSpec,
} from "../../src/schedule/elementwise-skeleton";
import {
  applyTiledMatmulSchedule,
  deriveTiledMatmulState,
  type TiledMatmulDescriptor,
} from "../../src/schedule/matmul-skeleton";
import {
  applyReductionSchedule,
  deriveReductionState,
  type ReductionDescriptor,
} from "../../src/schedule/reduction-skeleton";
import type {
  SemanticBodyNode,
  SemanticRegion,
  SemanticRegionUid,
  ValueUid,
} from "../../src/schedule/types";
import { cpuOnly } from "../helpers/webgpu";

const isWebGPUEnabled = !cpuOnly;

const REGION = "region:offmenu-test" as unknown as SemanticRegionUid;
const EMPTY_REGION: SemanticRegion = { uid: REGION, nodes: [] };

// ---------------------------------------------------------------------------
// GPU helpers (the tile-ir-block harness pattern)
// ---------------------------------------------------------------------------

let device: GPUDevice;
let queue: GPUQueue;

function makeF32Buffer(data: Float32Array): GPUBuffer {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(buf.getMappedRange()).set(data);
  buf.unmap();
  return buf;
}

function makeOutputBuffer(numElements: number): GPUBuffer {
  return device.createBuffer({
    size: numElements * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

async function readF32Buffer(
  buf: GPUBuffer,
  count: number,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: count * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, staging, 0, count * 4);
  queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const data = new Float32Array(staging.getMappedRange().slice(0));
  staging.unmap();
  staging.destroy();
  return data;
}

/** Dispatch one spec over named buffers/uniforms and read `count` f32s of `out`. */
async function runSpec(
  spec: TileKernelSpec,
  buffers: Record<string, GPUBuffer>,
  uniforms: Record<string, number>,
  outBuf: GPUBuffer,
  count: number,
): Promise<Float32Array> {
  const kernel = createTileKernelDispatcher(spec);
  beginSharedEncoder();
  kernel.dispatch(buffers, uniforms);
  flushSharedEncoder();
  return readF32Buffer(outBuf, count);
}

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

describe.runIf(isWebGPUEnabled)(
  "P2A Commit C — off-menu lowering proofs (wave-B readiness)",
  () => {
    beforeAll(async () => {
      await initWebGPU();
      const d = getWebGPUDevice();
      if (!d) throw new Error("WebGPU init failed");
      device = d.device;
      queue = d.queue;
    });

    // ========================================================================
    // 1. MATMUL — off-menu tile config {48, 32, 12}
    // ========================================================================
    it("matmul: off-menu {tileM:48,tileN:32,tileK:12} lowers, compiles, and matches the unmoved DEFAULT_CONFIG state + CPU", async () => {
      const offMenuConfig: MatmulKernelConfig = {
        tileM: 48,
        tileN: 32,
        tileK: 12,
        threadTileM: 4,
        threadTileN: 4,
        useSubgroups: false,
        vectorWidth: 1,
      };
      // LEGAL: the config validator accepts it (wg 12×8=96 ≤ 256; shared
      // 48·12+12·32 = 960 f32 ≪ 16KB; tiles divisible by thread tiles).
      expect(() => validateConfig(offMenuConfig)).not.toThrow();
      // OFF-MENU: no builder config produces it — not the default, and outside
      // the autotune tuning space on two axes.
      expect(offMenuConfig).not.toEqual(DEFAULT_CONFIG);
      expect(TUNING_SPACE.tileM).not.toContain(offMenuConfig.tileM);
      expect(TUNING_SPACE.tileK).not.toContain(offMenuConfig.tileK);

      const M = 90,
        N = 50,
        K = 54; // tails on all three axes for BOTH configs
      const uniforms = {
        m: M,
        n: N,
        k: K,
        lda: K,
        ldb: N,
        ldc: N,
        alpha: 1.0,
        batchSize: 1,
        batchStrideA: 0,
        batchStrideB: 0,
        batchStrideC: 0,
      };
      const aData = new Float32Array(M * K);
      const bData = new Float32Array(K * N);
      for (let i = 0; i < M * K; i++) aData[i] = Math.sin(i * 0.07) * 0.3;
      for (let i = 0; i < K * N; i++) bData[i] = Math.cos(i * 0.11) * 0.3;
      const cpuRef = new Float32Array(M * N);
      for (let m = 0; m < M; m++)
        for (let n = 0; n < N; n++) {
          let s = 0;
          for (let k = 0; k < K; k++) s += aData[m * K + k] * bData[k * N + n];
          cpuRef[m * N + n] = s;
        }

      const mkDesc = (config: MatmulKernelConfig): TiledMatmulDescriptor => ({
        config,
        transposeMode: "NN",
        dtype: "f32",
      });
      const offState = deriveTiledMatmulState(mkDesc(offMenuConfig), REGION);
      const refState = deriveTiledMatmulState(mkDesc(DEFAULT_CONFIG), REGION);
      // A genuinely different point in schedule space (blockShapes differ).
      expect(scheduleDigest(offState)).not.toBe(scheduleDigest(refState));

      // Lower both THROUGH the schedule object; give each its real grid
      // (dispatch.ts computes x=ceil(n/tileN), y=ceil(m/tileM) at dispatch time;
      // the spec's placeholder grid is single-workgroup).
      const withGrid = (
        spec: TileKernelSpec,
        cfg: MatmulKernelConfig,
      ): TileKernelSpec => ({
        ...spec,
        grid: tiledGrid({
          x: { uniform: "n", tileSize: cfg.tileN },
          y: { uniform: "m", tileSize: cfg.tileM },
        }),
      });
      const offSpec = withGrid(
        applyTiledMatmulSchedule(offState, mkDesc(offMenuConfig)),
        offMenuConfig,
      );
      const refSpec = withGrid(
        applyTiledMatmulSchedule(refState, mkDesc(DEFAULT_CONFIG)),
        DEFAULT_CONFIG,
      );
      // Valid WGSL: compiles (Dawn pipeline creation happens at dispatch).
      expect(compileTileKernel(offSpec)).toContain("fn main");

      const aBuf = makeF32Buffer(aData);
      const bBuf = makeF32Buffer(bData);
      const offOut = makeOutputBuffer(M * N);
      const refOut = makeOutputBuffer(M * N);
      const offResult = await runSpec(
        offSpec,
        { a: aBuf, b: bBuf, out: offOut },
        uniforms,
        offOut,
        M * N,
      );
      const refResult = await runSpec(
        refSpec,
        { a: aBuf, b: bBuf, out: refOut },
        uniforms,
        refOut,
        M * N,
      );

      expect(maxAbsDiff(offResult, cpuRef)).toBeLessThan(1e-3);
      expect(maxAbsDiff(offResult, refResult)).toBeLessThan(1e-3);

      for (const b of [aBuf, bBuf, offOut, refOut]) b.destroy();
    });

    // ========================================================================
    // 2. ELEMENTWISE — off-menu depth-2 body mul(add(a,b), a)
    // ========================================================================
    it("elementwise: off-menu nested body mul(add(a,b),a) lowers and matches the two-dispatch unmoved composition", async () => {
      const n = 4097; // crosses the 256-wide workgroup boundary (tail exercised)
      const val = (b: string): SemanticBodyNode => ({
        kind: "value",
        value: `in:${b}` as unknown as ValueUid,
      });
      const desc: ElementwiseKernelDescriptor = {
        name: "offmenu_nested_mul_add",
        enableF16: false,
        inputs: [
          {
            binding: "a",
            dtype: "f32",
            access: {
              indexShape: [n],
              strides: [1],
              offsetUniform: "a_offset",
            },
          },
          {
            binding: "b",
            dtype: "f32",
            access: {
              indexShape: [n],
              strides: [1],
              offsetUniform: "b_offset",
            },
          },
        ],
        output: { binding: "out", dtype: "f32" },
        // Depth-2 op-catalog tree: no live builder emits a nested apply.
        body: {
          kind: "apply",
          catalog: { op: "mul" },
          args: [
            {
              kind: "apply",
              catalog: { op: "add" },
              args: [val("a"), val("b")],
            },
            val("a"),
          ],
        },
      };
      const state = deriveScheduleState(desc, REGION);
      // LEGAL: passes the family's structural gate.
      expect(() => assertNoSecondOwnerElementwise(state)).not.toThrow();
      // OFF-MENU: distinct from the on-menu binary states it composes.
      const addState = deriveScheduleState(
        {
          ...desc,
          name: "binary_add",
          body: {
            kind: "apply",
            catalog: { op: "add" },
            args: [val("a"), val("b")],
          },
        },
        REGION,
      );
      expect(scheduleDigest(state)).not.toBe(scheduleDigest(addState));

      const spec = applySchedule(EMPTY_REGION, state, desc);
      expect(compileTileKernel(spec)).toContain("fn main");

      const aData = new Float32Array(n);
      const bData = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        aData[i] = Math.sin(i * 0.13) * 2 - 0.5;
        bData[i] = Math.cos(i * 0.29) * 1.5;
      }
      const aBuf = makeF32Buffer(aData);
      const bBuf = makeF32Buffer(bData);
      const offOut = makeOutputBuffer(n);
      const offResult = await runSpec(
        spec,
        { a: aBuf, b: bBuf, out: offOut },
        { size: n, a_offset: 0, b_offset: 0 },
        offOut,
        n,
      );

      // UNMOVED reference: the same computation as two on-menu dispatches
      // (binary add, then binary mul) through the live realizers.
      const addSpec = realizeBinaryBroadcastSpec("+", [n], [1], [1], "f32");
      const mulSpec = realizeBinaryBroadcastSpec("*", [n], [1], [1], "f32");
      const tmpBuf = makeOutputBuffer(n);
      await runSpec(
        addSpec,
        { a: aBuf, b: bBuf, out: tmpBuf },
        { size: n, a_offset: 0, b_offset: 0 },
        tmpBuf,
        n,
      );
      const refOut = makeOutputBuffer(n);
      const refResult = await runSpec(
        mulSpec,
        { a: tmpBuf, b: aBuf, out: refOut },
        { size: n, a_offset: 0, b_offset: 0 },
        refOut,
        n,
      );

      // Same f32 ops in the same order (fused stays in f32 registers; the
      // composition round-trips f32 global memory) — tolerance is fp-noise only.
      expect(maxAbsDiff(offResult, refResult)).toBeLessThan(1e-6);

      for (const b of [aBuf, bBuf, offOut, tmpBuf, refOut]) b.destroy();
    });

    // ========================================================================
    // 3. REDUCTION — off-menu reduceOp×preamble product: max ∘ abs (amax)
    // ========================================================================
    it("reduction: off-menu max-with-preamble (amax) lowers and matches the unmoved abs→max composition + CPU", async () => {
      const n = 5000;
      const desc: ReductionDescriptor = {
        reduceOp: "max", // the only live preamble site is SUM-only
        preamble: {
          chainOps: [{ op: "abs", arity: 1 }],
          totalInputs: 1,
          inputDtypes: ["f32"],
        },
      };
      const offState = deriveReductionState(desc, REGION);
      const plainMax: ReductionDescriptor = { reduceOp: "max" };
      const plainState = deriveReductionState(plainMax, REGION);
      expect(scheduleDigest(offState)).not.toBe(scheduleDigest(plainState));

      // applyReductionSchedule runs assertReductionSeam (the legality gate).
      const offSpec = applyReductionSchedule(offState, desc);
      expect(compileTileKernel(offSpec)).toContain("fn main");

      const data = new Float32Array(n);
      for (let i = 0; i < n; i++) data[i] = Math.sin(i * 0.37) * (i % 17) - 8;
      let cpuAmax = -Infinity;
      for (let i = 0; i < n; i++)
        cpuAmax = Math.max(cpuAmax, Math.abs(data[i]));

      const inBuf = makeF32Buffer(data);
      const offOut = makeOutputBuffer(1);
      const offResult = await runSpec(
        offSpec,
        { in0: inBuf, out: offOut },
        { size: n },
        offOut,
        1,
      );

      // UNMOVED reference: on-menu abs (elementwise realizer) then on-menu
      // plain max reduction.
      const absSpec = realizeUnaryStridedSpec("abs", [n], [1], "f32", "f32");
      const tmpBuf = makeOutputBuffer(n);
      await runSpec(
        absSpec,
        { a: inBuf, out: tmpBuf },
        { size: n, base_offset: 0 },
        tmpBuf,
        n,
      );
      const refSpec = applyReductionSchedule(plainState, plainMax);
      const refOut = makeOutputBuffer(1);
      const refResult = await runSpec(
        refSpec,
        { input: tmpBuf, out: refOut },
        { size: n },
        refOut,
        1,
      );

      expect(offResult[0]).toBe(refResult[0]); // max is order-insensitive: exact
      expect(Math.abs(offResult[0] - cpuAmax)).toBeLessThan(1e-6);

      for (const b of [inBuf, offOut, tmpBuf, refOut]) b.destroy();
    });
  },
);
