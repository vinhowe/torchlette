/**
 * End-to-end numerical guard for the SDPA-backward multi-output bug.
 *
 * Scaled-dot-product attention's WebGPU backward emits dQ/dK/dV as the three
 * outputs of ONE fused node. When SDPA feeds downstream ops (here: the per-head
 * reshape/permute/contiguous plumbing around a fused qkv projection), the CSE
 * pass once merged the structurally-identical consumers of those distinct
 * outputs, collapsing dQ/dK onto dV — a silent ~0.5-nat gradient corruption
 * (see graph-rewrites.ts structuralKey / tools/sdpa2-diff.ts).
 *
 * The compiler-level guard (graph-rewrites.spec.ts) pins the *mechanism*; this
 * pins the *symptom* end-to-end by differentially checking the WebGPU fused
 * SDPA backward against the CPU decomposed path (the naive reference, which
 * never builds the multi-output node). Per CLAUDE.md "differentially test
 * optimized paths against the naive one."
 */

import { beforeAll, describe, expect, it } from "vitest";
import { Torchlette } from "../src";
import { initWebGPU } from "../src/backend/webgpu";
import { canUseWebGPU } from "./helpers/webgpu";

const B = 2,
  S = 16,
  H = 4,
  hd = 32;
const E = H * hd;
const scale = 1 / Math.sqrt(hd);

// Deterministic, sign-varied inputs (no RNG dependence across runs/backends).
function gen(n: number, seed: number): number[] {
  const a = new Array<number>(n);
  for (let i = 0; i < n; i++) a[i] = Math.sin(i * 0.3 + seed) * 1.5 + Math.cos(i * 0.11 + seed);
  return a;
}

/** Build the qkv→heads→SDPA→merge→loss graph, run fwd+bwd, return d(qkv). */
async function gradQKV(
  api: Torchlette,
  device: "cpu" | "webgpu",
  qkv: number[],
  dout: number[],
): Promise<Float32Array> {
  const x = api.tensorFromArray(qkv, [B, S, 3 * E], { device, requiresGrad: true });
  const dOut = api.tensorFromArray(dout, [B, S, E], { device });
  const [qF, kF, vF] = x.chunk(3, -1);
  const toHeads = (t: typeof qF) =>
    t.reshape([B, S, H, hd]).permute([0, 2, 1, 3]).contiguous();
  const out = api.scaledDotProductAttention(
    toHeads(qF),
    toHeads(kF),
    toHeads(vF),
    scale,
    true,
  );
  const attnFlat = out.permute([0, 2, 1, 3]).reshape([B, S, E]);
  const loss = api.sum(api.mul(attnFlat, dOut));
  await loss.backward();
  const g = x.grad;
  if (!g) throw new Error("no grad on qkv");
  return new Float32Array(await g.cpu());
}

function relErr(a: Float32Array, b: Float32Array): number {
  let d = 0,
    n = 0;
  for (let i = 0; i < a.length; i++) {
    d += (a[i]! - b[i]!) ** 2;
    n += b[i]! ** 2;
  }
  return Math.sqrt(d) / (Math.sqrt(n) + 1e-12);
}

describe("SDPA backward — multi-output gradient parity (WebGPU vs CPU)", () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = await canUseWebGPU();
    if (webgpu) await initWebGPU();
  });

  it("dQ/dK/dV survive the per-head reshape plumbing (no row-0/collapse corruption)", async () => {
    if (!webgpu) return; // needs both backends; skip in CPU-only CI

    const qkv = gen(B * S * 3 * E, 1);
    const dout = gen(B * S * E, 7);

    const cpuApi = new Torchlette("cpu");
    const gpuApi = new Torchlette("webgpu", { enableFusion: true });

    const refGrad = await gradQKV(cpuApi, "cpu", qkv, dout); // decomposed = naive
    const gpuGrad = await gradQKV(gpuApi, "webgpu", qkv, dout); // fused multi-output

    const err = relErr(gpuGrad, refGrad);
    // Correct: WebGPU(flash) vs CPU(decomposed) agree to ~1e-4 in f32. The bug
    // produced rel-err > 1 (dQ/dK collapsed onto dV), so any sane tol catches it.
    expect(err).toBeLessThan(5e-3);
  });
});
