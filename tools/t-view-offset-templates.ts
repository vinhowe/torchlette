/**
 * Task #71 failing-first probe: N distinct view OFFSETS produce N distinct
 * lowered-plan templates today (offset baked into template identity), and
 * should produce 1 template + N uniform values after the fix.
 *
 * Reproduces the static-KV / RoPE-slice decode pattern in miniature: a
 * persistent base buffer, sliced with a per-step-varying `start` via
 * narrow(dim, start, length), then read through a kernel (add). Each distinct
 * `start` is a new plan fingerprint today, so `debugTemplateCount()` grows by
 * one per offset.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *        npx tsx tools/t-view-offset-templates.ts
 */
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import { debugTemplateCount } from "../src/executor/executor";
import { Torchlette } from "../src/frontend/torchlette";

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  const BASE = 64; // base length along dim 0
  const LEN = 4; // narrow length (stays constant → only `start` varies)
  const OFFSETS = [0, 4, 8, 12, 16, 20]; // N distinct starts
  const N = OFFSETS.length;

  // Persistent base buffer (like the RoPE table / KV cache).
  const baseArr = new Float32Array(BASE * LEN);
  for (let i = 0; i < baseArr.length; i++) baseArr[i] = i;
  const base = api.tensorFromArray(baseArr, [BASE, LEN]);
  const addend = api.tensorFromArray(new Float32Array(LEN * LEN).fill(1), [
    LEN,
    LEN,
  ]);

  const before = debugTemplateCount();

  // Run each offset twice so a template would be a cache HIT the 2nd time —
  // template growth is purely from distinct offsets, not re-execution.
  const results: number[][] = [];
  for (let rep = 0; rep < 2; rep++) {
    for (const start of OFFSETS) {
      const view = base.narrow(0, start, LEN); // [LEN, LEN], offset = start*LEN
      const out = api.add(view.contiguous(), addend);
      const cpu = new Float32Array(await out.cpu());
      if (rep === 0) results.push(Array.from(cpu));
    }
  }

  const after = debugTemplateCount();
  const grew = after - before;

  console.log(`OFFSETS=${JSON.stringify(OFFSETS)} (N=${N})`);
  console.log(`templates before=${before} after=${after} grew=${grew}`);
  // Correctness: each narrow(0,start,LEN)+1 must equal base rows [start,start+LEN)+1.
  let correct = true;
  for (let i = 0; i < N; i++) {
    const start = OFFSETS[i];
    for (let r = 0; r < LEN; r++) {
      for (let c = 0; c < LEN; c++) {
        const expect = baseArr[(start + r) * LEN + c] + 1;
        if (results[i][r * LEN + c] !== expect) {
          correct = false;
          console.log(
            `  MISMATCH offset=${start} [${r},${c}] got=${results[i][r * LEN + c]} expect=${expect}`,
          );
        }
      }
    }
  }
  console.log(`correctness: ${correct ? "PASS" : "FAIL"}`);
  console.log(
    grew >= N
      ? `RESULT: BUG PRESENT — ${grew} templates for ${N} offsets (offset in template identity)`
      : grew <= 1
        ? `RESULT: FIXED — ${grew} template for ${N} offsets (offset is data)`
        : `RESULT: PARTIAL — ${grew} templates for ${N} offsets`,
  );
  process.exit(correct ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
