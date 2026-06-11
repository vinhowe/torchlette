/**
 * Unit differential for BUFFER DONATION in multi-output fused kernels.
 *
 * Hand-builds a 2-output recipe (out0 = a*b, out1 = a+b) and dispatches it
 * three ways: undonated (reference), donated in0 (a's buffer becomes out0),
 * donated in1. Compares results element-wise. Isolates the donated
 * multi-output kernel/dispatch from all engine bookkeeping — if THIS fails,
 * the defect is in fusion-tile-ir codegen or fusion-dispatch binding math;
 * if it passes, the defect is upstream (liveness/recipe/group bookkeeping).
 */
import { initWebGPU } from "../src/backend/webgpu";

async function main() {
  if (!(await initWebGPU())) {
    console.error("WebGPU init failed");
    process.exit(1);
  }
  const { dispatchFusedKernel } = await import(
    "../src/backend/webgpu/fusion-dispatch"
  );
  const { requireContext } = await import(
    "../src/backend/webgpu/webgpu-state"
  );
  const ctx = requireContext();
  const device = ctx.device;

  const N = 1024 + 7; // non-multiple of vec width to exercise masking
  const shape = [N];
  const aData = Float32Array.from({ length: N }, (_, i) => (i % 13) * 0.5 + 1);
  const bData = Float32Array.from({ length: N }, (_, i) => (i % 7) * 0.25 + 2);

  const mkBuf = (data: Float32Array) => {
    const buf = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    ctx.queue.writeBuffer(buf, 0, data);
    return buf;
  };
  const readBuf = async (buf: unknown, n: number): Promise<Float32Array> => {
    const staging = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(buf as never, 0, staging, 0, n * 4);
    ctx.queue.submit([enc.finish()]);
    await staging.mapAsync(0x0001 /* MAP_READ */);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return out;
  };

  const recipe = {
    id: "donation-unit",
    nodes: [
      { id: 0, op: "mul", inputs: [-1, -2], shape, dtype: "f32" as const, isOutput: true },
      { id: 1, op: "add", inputs: [-1, -2], shape, dtype: "f32" as const, isOutput: true },
    ],
    inputs: [
      { id: 100, index: 0, shape, dtype: "f32" as const },
      { id: 101, index: 1, shape, dtype: "f32" as const },
    ],
    outputs: [
      { nodeId: 0, index: 0, shape, dtype: "f32" as const },
      { nodeId: 1, index: 1, shape, dtype: "f32" as const },
    ],
  };

  const expect0 = Float32Array.from({ length: N }, (_, i) => aData[i] * bData[i]);
  const expect1 = Float32Array.from({ length: N }, (_, i) => aData[i] + bData[i]);

  const check = async (label: string, donatedInput?: number) => {
    const a = mkBuf(aData);
    const b = mkBuf(bData);
    const result = dispatchFusedKernel(
      device,
      recipe as never,
      [
        { buffer: a, shape, dtype: "f32" },
        { buffer: b, shape, dtype: "f32" },
      ] as never,
      { donatedInput } as never,
    );
    const got0 = await readBuf(result.outputs[0].buffer, N);
    const got1 = await readBuf(result.outputs[1].buffer, N);
    let bad0 = 0;
    let bad1 = 0;
    for (let i = 0; i < N; i++) {
      if (Math.abs(got0[i] - expect0[i]) > 1e-5) bad0++;
      if (Math.abs(got1[i] - expect1[i]) > 1e-5) bad1++;
    }
    console.log(
      `${label}: out0 bad=${bad0}/${N} out1 bad=${bad1}/${N} ` +
        `${bad0 + bad1 === 0 ? "PASS" : `FAIL (e.g. got0[0]=${got0[0]} want ${expect0[0]}, got1[0]=${got1[0]} want ${expect1[0]})`}`,
    );
    return bad0 + bad1 === 0;
  };

  const r1 = await check("undonated      ");
  const r2 = await check("donated in0    ", 0);
  const r3 = await check("donated in1    ", 1);
  // Also a 4-output batch like phase 2b singleton batches: independent ops.
  const recipe4 = {
    id: "donation-unit4",
    nodes: [
      { id: 0, op: "mul", inputs: [-1, -2], shape, dtype: "f32" as const, isOutput: true },
      { id: 1, op: "add", inputs: [-1, -2], shape, dtype: "f32" as const, isOutput: true },
      { id: 2, op: "sub", inputs: [-1, -2], shape, dtype: "f32" as const, isOutput: true },
      { id: 3, op: "max", inputs: [-1, -2], shape, dtype: "f32" as const, isOutput: true },
    ],
    inputs: recipe.inputs,
    outputs: [
      { nodeId: 0, index: 0, shape, dtype: "f32" as const },
      { nodeId: 1, index: 1, shape, dtype: "f32" as const },
      { nodeId: 2, index: 2, shape, dtype: "f32" as const },
      { nodeId: 3, index: 3, shape, dtype: "f32" as const },
    ],
  };
  {
    const a = mkBuf(aData);
    const b = mkBuf(bData);
    const result = dispatchFusedKernel(
      device,
      recipe4 as never,
      [
        { buffer: a, shape, dtype: "f32" },
        { buffer: b, shape, dtype: "f32" },
      ] as never,
      { donatedInput: 0 } as never,
    );
    const exps = [expect0, expect1, Float32Array.from({ length: N }, (_, i) => aData[i] - bData[i]), Float32Array.from({ length: N }, (_, i) => Math.max(aData[i], bData[i]))];
    let allBad = 0;
    for (let o = 0; o < 4; o++) {
      const got = await readBuf(result.outputs[o].buffer, N);
      let bad = 0;
      for (let i = 0; i < N; i++) if (Math.abs(got[i] - exps[o][i]) > 1e-5) bad++;
      if (bad) console.log(`4-out donated in0: out${o} bad=${bad}/${N} (got[0]=${got[0]} want ${exps[o][0]})`);
      allBad += bad;
    }
    console.log(`4-out donated in0: ${allBad === 0 ? "PASS" : "FAIL"}`);
  }
  console.log(r1 && r2 && r3 ? "2-out: ALL PASS" : "2-out: FAILURES");
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
