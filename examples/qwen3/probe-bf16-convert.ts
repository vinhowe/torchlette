/**
 * Gate for the fast loader converters: bf16→f16 raw bits and bf16→f32 must
 * agree with the reference double-conversion path (bf16→f32→f16 via core)
 * across random values + edge cases, including after a GPU round trip.
 */
import { getWebGPUInitError, initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import {
  bf16SliceToF16Bits,
  bf16SliceToF32,
} from "../../packages/qwen3-browser/src/weights-map";

function f32ToBf16Bits(x: number): number {
  const buf = new ArrayBuffer(4);
  new Float32Array(buf)[0] = x;
  return new Uint32Array(buf)[0] >>> 16; // truncate (good enough to generate test inputs)
}

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });

  // Test inputs: typical weight range + edges (zero, ±denormal-ish, big, inf, nan).
  const values = [0, -0, 1, -1, 0.5, -0.0625, 3.14159, -2.71828, 1e-4, -1e-4, 6e-5, 1e-6, 60000, -60000, 1e20, -1e20, Infinity, -Infinity, NaN];
  for (let i = 0; i < 4096 - values.length; i++) {
    values.push((Math.random() - 0.5) * 4); // weight-like
  }
  const n = values.length;
  const bf16 = new Uint16Array(n);
  for (let i = 0; i < n; i++) bf16[i] = f32ToBf16Bits(values[i]);

  // Reference: bf16 → exact f32
  const refF32 = new Float32Array(n);
  const refU32 = new Uint32Array(refF32.buffer);
  for (let i = 0; i < n; i++) refU32[i] = bf16[i] << 16;

  // --- bf16SliceToF32 must equal the reference bit shift
  const gotF32 = new Float32Array(n);
  bf16SliceToF32(bf16, gotF32, 0, n);
  for (let i = 0; i < n; i++) {
    if (!Object.is(gotF32[i], refF32[i]) && !(Number.isNaN(gotF32[i]) && Number.isNaN(refF32[i]))) {
      throw new Error(`bf16→f32 mismatch at ${i}: ${gotF32[i]} vs ${refF32[i]}`);
    }
  }
  console.log("bf16→f32: exact match");

  // --- bf16SliceToF16Bits vs core double-conversion, compared after GPU round trip
  const fastBits = new Uint16Array(n);
  bf16SliceToF16Bits(bf16, fastBits, 0, n);
  const viaFast = api.tensorFromArray(fastBits, [n], { dtype: "f16" });
  const viaCore = api.tensorFromArray(refF32, [n], { dtype: "f16" }); // core f32→f16
  const a = new Float32Array(await viaFast.cpu());
  const b = new Float32Array(await viaCore.cpu());
  let mismatches = 0;
  for (let i = 0; i < n; i++) {
    const bothNaN = Number.isNaN(a[i]) && Number.isNaN(b[i]);
    // Overflow: fast path clamps to ±65504 where core may produce ±inf — accept.
    const overflowClamp = !Number.isFinite(b[i]) && Math.abs(a[i]) === 65504;
    if (a[i] !== b[i] && !bothNaN && !overflowClamp) {
      if (mismatches < 5) console.log(`  mismatch[${i}] in=${values[i]} fast=${a[i]} core=${b[i]}`);
      mismatches++;
    }
  }
  console.log(mismatches === 0 ? "bf16→f16 bits: match (modulo overflow clamp)" : `bf16→f16: ${mismatches} MISMATCHES`);
  process.exit(mismatches === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
