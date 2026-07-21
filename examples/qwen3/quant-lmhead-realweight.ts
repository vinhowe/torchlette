/**
 * Gate 2 + 5 on a REAL model weight: quantize Qwen3-1.7B's tied lm_head /
 * embedding (the largest weight, ~311M params) and compare the int8-grouped
 * GEMV kernel's logits against an f32 reference matmul of the SAME real weight,
 * on a realistic post-RMSNorm hidden state.
 *
 * This isolates weight-quantization error on a REAL weight distribution (gate
 * 2) without the fragile full 28-layer forward — the model-load fence hangs
 * intermittently under Node/Dawn on this box (the vkCreateDevice-class flake
 * the task flags). The hidden vector is synthesized to match a real post-norm
 * decode state: RMSNorm output has unit-RMS per row scaled by the norm weight,
 * so we draw x ~ N(0,1) then RMS-normalize and apply a plausible scale — the
 * absolute logit magnitude tracks a real decode, and the quant DRIFT (what the
 * gate thresholds) is dominated by the real weight's quantization, not x.
 *
 * GATE 2 thresholds (design doc, stated before measuring):
 *   top-1 agreement, top-5 overlap ≥4/5, max-abs drift ≤0.5, mean-abs ≤0.05.
 * GATE 5: bytes resident int8 vs f32/f16, per-call GPU time.
 *
 * Run: CUDA_VISIBLE_DEVICES=0 npx tsx examples/qwen3/quant-lmhead-realweight.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  dispatchMatmul,
  dispatchQuantizedGemvNT,
  getGpuUncapturedErrorCount,
  getWebGPUInitError,
  initWebGPU,
  syncWebGPU,
  webgpuBackend,
} from "../../src/backend/webgpu";
import { requireContext } from "../../src/backend/webgpu/gpu-context";
import { GPUBufferUsage } from "../../src/backend/webgpu/gpu-types";
import { quantizeLinearWeight } from "../../tools/quantize";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const GROUP_SIZE = 64;

// Design-doc gate-2 thresholds (before measuring).
const TOP5_MIN_OVERLAP = 4;
const MAX_ABS_DRIFT = 0.5;
const MEAN_ABS_DRIFT = 0.05;

/** Read a named tensor from the safetensors shards as f32 (bf16 decoded). */
function readWeightF32(
  modelDir: string,
  name: string,
): { data: Float32Array; shape: number[] } {
  const idxPath = path.join(modelDir, "model.safetensors.index.json");
  let shard = "model.safetensors";
  if (fs.existsSync(idxPath)) {
    const idx = JSON.parse(fs.readFileSync(idxPath, "utf-8"));
    shard = (idx.weight_map as Record<string, string>)[name];
  }
  const fd = fs.openSync(path.join(modelDir, shard), "r");
  try {
    const hlBuf = Buffer.alloc(8);
    fs.readSync(fd, hlBuf, 0, 8, 0);
    const headerLen = Number(
      new DataView(hlBuf.buffer, hlBuf.byteOffset, 8).getBigUint64(0, true),
    );
    const hBuf = Buffer.alloc(headerLen);
    fs.readSync(fd, hBuf, 0, headerLen, 8);
    const meta = JSON.parse(new TextDecoder().decode(hBuf));
    const info = meta[name];
    const [start, end] = info.data_offsets as [number, number];
    const raw = Buffer.alloc(end - start);
    fs.readSync(fd, raw, 0, end - start, 8 + headerLen + start);
    const len = (info.shape as number[]).reduce((a, b) => a * b, 1);
    const out = new Float32Array(len);
    if (info.dtype === "BF16") {
      const u16 = new Uint16Array(
        raw.buffer,
        raw.byteOffset,
        (end - start) / 2,
      );
      const b = new ArrayBuffer(4);
      const bu = new Uint32Array(b);
      const bf = new Float32Array(b);
      for (let i = 0; i < u16.length; i++) {
        bu[0] = u16[i] << 16;
        out[i] = bf[0];
      }
    } else if (info.dtype === "F32") {
      out.set(new Float32Array(raw.buffer, raw.byteOffset, (end - start) / 4));
    } else {
      throw new Error(`unexpected dtype ${info.dtype} for ${name}`);
    }
    return { data: out, shape: info.shape };
  } finally {
    fs.closeSync(fd);
  }
}

function uploadU32(data: Uint32Array): GPUBuffer {
  const ctx = requireContext();
  const buf = ctx.device.createBuffer({
    size: data.byteLength,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  ctx.device.queue.writeBuffer(
    buf,
    0,
    data.buffer,
    data.byteOffset,
    data.byteLength,
  );
  return buf;
}

function topK(arr: Float32Array, k: number): number[] {
  const idx = Array.from({ length: arr.length }, (_, i) => i);
  idx.sort((x, y) => arr[y] - arr[x]);
  return idx.slice(0, k);
}

function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 31 - 1;
  };
}

async function main() {
  process.env.TORCHLETTE_COMPILED_PLAN = "0";
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");

  // --- Real lm_head weight [N, K]. ---
  const { data: wf32, shape } = readWeightF32(
    MODEL_DIR,
    "model.embed_tokens.weight",
  );
  const [N, K] = shape;
  console.log(`Real Qwen3-1.7B lm_head: N=${N} K=${K}`);

  // --- Realistic post-RMSNorm hidden state x[1,K]. ---
  // Qwen3 final norm output ≈ RMSNorm(residual)·normWeight. Draw N(0,1),
  // RMS-normalize to unit RMS, then multiply by the REAL model.norm.weight.
  const rng = makeRng(0xbeef);
  const x = new Float32Array(K);
  let ss = 0;
  for (let i = 0; i < K; i++) {
    // crude Box-Muller-free gaussian via sum of uniforms
    const g = rng() + rng() + rng() + rng() + rng() + rng();
    x[i] = g;
    ss += g * g;
  }
  const rms = Math.sqrt(ss / K) + 1e-6;
  const { data: normW } = readWeightF32(MODEL_DIR, "model.norm.weight");
  for (let i = 0; i < K; i++) x[i] = (x[i] / rms) * normW[i];

  // --- Quantize the real weight. ---
  const t0 = performance.now();
  const q = quantizeLinearWeight(wf32, N, K, GROUP_SIZE);
  console.log(
    `Quantized real lm_head in ${(performance.now() - t0).toFixed(0)}ms (G=${GROUP_SIZE})`,
  );
  const int8Bytes = q.packed.byteLength + q.scales.byteLength;
  const f32Bytes = N * K * 4;
  const f16Bytes = N * K * 2;
  console.log(
    `resident: f32=${(f32Bytes / 1e6).toFixed(0)}MB f16=${(f16Bytes / 1e6).toFixed(0)}MB ` +
      `int8+scales=${(int8Bytes / 1e6).toFixed(0)}MB ` +
      `(vs f32 ${(f32Bytes / int8Bytes).toFixed(2)}x, vs f16 ${(f16Bytes / int8Bytes).toFixed(2)}x)`,
  );

  const ctx = requireContext();
  const packedBuf = uploadU32(q.packed);
  const scalesBuf = uploadU32(
    new Uint32Array(q.scales.buffer, q.scales.byteOffset, q.scales.length / 2),
  );

  // --- f32 reference logits: x @ W^T via the standard NT matmul on the REAL
  //     (un-quantized) f32 weight. This is the baseline the quant path must
  //     track; drift = pure weight-quantization error. ---
  const aT = webgpuBackend.ops.tensorFromArray(x, [1, K]);
  const wT = webgpuBackend.ops.tensorFromArray(wf32, [N, K]);
  const cRef = dispatchMatmul(aT, wT, false, true);
  await syncWebGPU();
  const refLogits = new Float32Array(await webgpuBackend.ops.read(cRef));

  // --- Quant kernel logits. ---
  const xbuf = (aT as { buffer: GPUBuffer }).buffer;
  const outBuf = ctx.device.createBuffer({
    size: N * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  dispatchQuantizedGemvNT(
    ctx.device,
    xbuf,
    packedBuf,
    scalesBuf,
    outBuf,
    N,
    K,
    GROUP_SIZE,
    "f32",
  );
  await syncWebGPU();
  const readBuf = ctx.device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  {
    const enc = ctx.device.createCommandEncoder();
    enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, N * 4);
    ctx.device.queue.submit([enc.finish()]);
    await readBuf.mapAsync(1);
  }
  const quantLogits = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  readBuf.destroy();

  // --- Gate 2 ---
  const refTop5 = topK(refLogits, 5);
  const qTop5 = topK(quantLogits, 5);
  const top1 = refTop5[0] === qTop5[0];
  const overlap = qTop5.filter((id) => refTop5.includes(id)).length;
  let maxAbs = 0;
  let sumAbs = 0;
  for (let i = 0; i < N; i++) {
    const d = Math.abs(quantLogits[i] - refLogits[i]);
    if (d > maxAbs) maxAbs = d;
    sumAbs += d;
  }
  const meanAbs = sumAbs / N;
  let lMin = refLogits[0];
  let lMax = refLogits[0];
  for (let i = 1; i < N; i++) {
    if (refLogits[i] < lMin) lMin = refLogits[i];
    if (refLogits[i] > lMax) lMax = refLogits[i];
  }
  const logitRange = lMax - lMin;
  console.log(
    "\n=== GATE 2: real-weight quant parity (int8 vs f32 lm_head) ===",
  );
  console.log(`  ref  top5=${JSON.stringify(refTop5)}`);
  console.log(`  int8 top5=${JSON.stringify(qTop5)}`);
  console.log(
    `  top1=${top1} top5overlap=${overlap}/5 maxAbsDrift=${maxAbs.toFixed(4)} ` +
      `meanAbsDrift=${meanAbs.toExponential(2)} (logit range≈${logitRange.toFixed(1)})`,
  );
  const gate2 =
    top1 &&
    overlap >= TOP5_MIN_OVERLAP &&
    maxAbs <= MAX_ABS_DRIFT &&
    meanAbs <= MEAN_ABS_DRIFT;
  console.log(`  GATE 2: ${gate2 ? "PASS" : "FAIL"}`);

  // --- Gate 5: perf ---
  const ITERS = 100;
  for (let i = 0; i < 10; i++)
    dispatchQuantizedGemvNT(
      ctx.device,
      xbuf,
      packedBuf,
      scalesBuf,
      outBuf,
      N,
      K,
      GROUP_SIZE,
      "f32",
    );
  await syncWebGPU();
  const tS = performance.now();
  for (let i = 0; i < ITERS; i++)
    dispatchQuantizedGemvNT(
      ctx.device,
      xbuf,
      packedBuf,
      scalesBuf,
      outBuf,
      N,
      K,
      GROUP_SIZE,
      "f32",
    );
  await syncWebGPU();
  const perCallQ = (performance.now() - tS) / ITERS;

  // f32 control per-call (same NT matmul path).
  for (let i = 0; i < 10; i++) {
    const c = dispatchMatmul(aT, wT, false, true);
    (c as { destroy?: () => void }).destroy?.();
  }
  await syncWebGPU();
  const tR = performance.now();
  for (let i = 0; i < ITERS; i++) {
    const c = dispatchMatmul(aT, wT, false, true);
    (c as { destroy?: () => void }).destroy?.();
  }
  await syncWebGPU();
  const perCallRef = (performance.now() - tR) / ITERS;
  console.log("\n=== GATE 5: perf (A100) ===");
  console.log(
    `  int8 lm_head: ${perCallQ.toFixed(3)} ms/call   f32 lm_head: ${perCallRef.toFixed(3)} ms/call` +
      `  (int8 ${(int8Bytes / 1e6).toFixed(0)}MB vs f32 ${(f32Bytes / 1e6).toFixed(0)}MB resident)`,
  );
  console.log(
    `  NOTE: A100 (~2TB/s) is not bandwidth-starved; the phase-1 A100 win is ` +
      `RESIDENCY (${(f32Bytes / int8Bytes).toFixed(1)}x vs f32). The tokens/s win ` +
      `is the 16GB Mac's bandwidth term: lm_head bytes/token 1244MB(f32)→311MB(int8).`,
  );

  packedBuf.destroy();
  scalesBuf.destroy();
  outBuf.destroy();
  (cRef as { destroy?: () => void }).destroy?.();
  (aT as { destroy?: () => void }).destroy?.();
  (wT as { destroy?: () => void }).destroy?.();

  const gpuErrs = getGpuUncapturedErrorCount();
  if (gpuErrs > 0) console.log(`WARN ${gpuErrs} uncaptured GPU errors`);
  console.log(
    `\n${gate2 ? "REAL-WEIGHT QUANT PARITY: PASS" : "REAL-WEIGHT QUANT PARITY: FAIL"}`,
  );
  process.exit(gate2 ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
