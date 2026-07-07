/**
 * OP CONFORMANCE — WebGPU project.
 *
 * Runs the generated op matrix on the WebGPU backend, comparing every op ×
 * dtype × {contiguous, odd-size, 5-view battery, broadcast, dim sweep} against
 * the CPU backend — the full cross-implementation differential. This is the
 * structural defense against the pow(x<0) class, dtype-oblivious kernels (#59),
 * and offset-view raw-bind reads (#58).
 *
 * Auto-skips GPU-less (CI); matched by the `test/webgpu/*.spec.ts` glob in
 * vitest.config.ts's GPU_TEST_FILES.
 */

import { beforeAll, describe, expect, it } from "vitest";
import {
  getGpuUncapturedErrorCount,
  initWebGPU,
} from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { registerConformance } from "../helpers/op-catalog";
import { cpuOnly } from "../helpers/webgpu";

describe.skipIf(cpuOnly)("op conformance (WebGPU, vs CPU differential)", () => {
  beforeAll(async () => {
    await initWebGPU();
  });

  registerConformance({
    device: "webgpu",
    makeApi: () => new Torchlette("webgpu"),
    makeOracle: () => new Torchlette("cpu"),
    gpuErrorCount: getGpuUncapturedErrorCount,
  });

  // -------------------------------------------------------------------------
  // Catalogued expected-failures (visible debt, not silence).
  // -------------------------------------------------------------------------

  // FINDING #A (surfaced by this harness, #59-class dtype-oblivious): the fused
  // isfinite codegen emits `bitcast<u32>(a[i])` on the input element, valid WGSL
  // only for f32. On an f16 input the shader fails to compile, the enclosing
  // submit is DROPPED, and the readback returns stale buffer data (all-zeros in
  // isolation; coincidentally the correct all-ones when a prior isfinite[f32]
  // left 1.0s in the reused buffer — which is exactly how a naive value-only
  // differential missed it). CPU isfinite[f16] is correct. Not fixed here (the
  // fix is in the fusion codegen's dtype handling, not a one-liner). `it.fails`
  // documents the live bug and flips green→red the moment it is fixed.
  it.fails(
    "isfinite on f16 input emits invalid WGSL → dropped submit (FINDING #A, #59)",
    async () => {
      const gpu = new Torchlette("webgpu");
      const before = getGpuUncapturedErrorCount();
      const x = gpu
        .tensorFromArray([1, 2, -3, 100, 0.5, -0.5], [6], { device: "webgpu" })
        .toDtype("f16");
      const got = Array.from(await gpu.isfinite(x).cpu());
      // Correct is all-ones (every input finite) AND no GPU error was raised.
      expect(getGpuUncapturedErrorCount() - before).toBe(0);
      for (const v of got) expect(Math.abs(v - 1)).toBeLessThan(0.05);
    },
    30_000,
  );

  // FINDING #B (surfaced by this harness, #59-class dtype-oblivious): max/min
  // GPU reductions read an f16 input's storage as f32, returning garbage (e.g.
  // max over dim=-1 of an f16 [4,6] gives [25.59, 0.0027, 0, 0]). sum/mean f16
  // reductions are correct, so this is specific to the max/min reduction kernel.
  // CPU is correct. Not fixed here (reduction-kernel dtype handling, not a
  // one-liner). `it.fails` documents the live bug.
  it.fails(
    "max reduction on f16 input is dtype-oblivious (FINDING #B, #59)",
    async () => {
      const gpu = new Torchlette("webgpu");
      const cpu = new Torchlette("cpu");
      const data = [1.7, -0.3, 2.4, -1.9, 0.8, -2.1, 1.1, 0.05];
      // dim reduction returns a Tensor; readback via .cpu().
      const g = gpu.max(gpu.tensorFromArray(data, [2, 4]).toDtype("f16"), { dim: -1 });
      const c = cpu.max(cpu.tensorFromArray(data, [2, 4]).toDtype("f16"), { dim: -1 });
      const gArr = Array.from(await (g as { cpu(): Promise<Float32Array> }).cpu());
      const cArr = Array.from(await (c as { cpu(): Promise<Float32Array> }).cpu());
      for (let i = 0; i < cArr.length; i++)
        expect(Math.abs(gArr[i] - cArr[i])).toBeLessThan(0.05);
    },
    30_000,
  );

  // #59 dtype-oblivious kernels: gather reads f32 lanes even for an f16 source,
  // returning garbage. The CPU backend is correct (see the assertion), so this
  // is a GPU-only wrongness. `it.fails` documents the live bug and flips to a
  // failure the moment #59 is fixed (prompting removal of this marker).
  it.fails(
    "gather on f16 source is dtype-oblivious (#59)",
    async () => {
      const gpu = new Torchlette("webgpu");
      const src = gpu
        .tensorFromArray([1, 2, 3, 4, 5, 6], [2, 3], { device: "webgpu" })
        .toDtype("f16");
      const idx = gpu.tensorFromArray([0, 2], [2, 1], { device: "webgpu" });
      const got = Array.from(await gpu.gather(src, idx, { dim: 1 }).cpu());
      // Correct result is [1, 6]; GPU currently returns dtype-oblivious garbage.
      expect(Math.abs(got[0] - 1)).toBeLessThan(0.05);
      expect(Math.abs(got[1] - 6)).toBeLessThan(0.05);
    },
    30_000,
  );
});
