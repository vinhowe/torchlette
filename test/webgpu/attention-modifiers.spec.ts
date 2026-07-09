/**
 * Attention modifiers (#64 iii) — soft-cap + sliding-window parity gates.
 *
 * Gate ladder items 1 (per-modifier parity vs unfused reference) and 2 (the
 * MANDATORY f16 gate — review disposition 3). Every fused-kernel modifier
 * output is diffed against a naive materialized-softmax JS reference
 * computing the same (scoreMod ∘ mask) attention, and against the CPU
 * decomposed path (the cross-path differential — modifiers are interpreted
 * there as tensor ops).
 *
 * The f16 gate matches DEPLOYMENT: the browser runs f16-quantized QKV
 * through the f32 attention kernel (there is no f16 attention kernel; model
 * code casts at the SDPA boundary). Inputs are f16-roundtripped, modifier
 * arithmetic stays f32, tolerance at f16 level.
 */
import { describe, expect, it } from "vitest";
import type { AttnModifierSpec } from "../../src/backend/types";
import { initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/index";
import { cpuOnly } from "../helpers/webgpu";

const B = 1,
  H = 2,
  N = 96, // multiple KV tiles (BC=32), not a multiple of BR=64
  D = 64;
const SCALE = 1.0 / Math.sqrt(D);

function makeData(seed: number, scale = 1.5): number[] {
  const out: number[] = [];
  let s = seed;
  for (let i = 0; i < B * H * N * D; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    out.push((s / 0x7fffffff - 0.5) * scale);
  }
  return out;
}
const qData = makeData(11);
const kData = makeData(22);
const vData = makeData(33);

/** Naive materialized-softmax reference with (softcap ∘ masks). */
function reference(
  q: number[],
  k: number[],
  v: number[],
  opts: { causal: boolean; window?: number; softcap?: number },
): number[] {
  const out = new Float32Array(B * H * N * D);
  for (let b = 0; b < B; b++)
    for (let h = 0; h < H; h++) {
      const base = (b * H + h) * N * D;
      for (let i = 0; i < N; i++) {
        const scores: number[] = [];
        const active: boolean[] = [];
        for (let j = 0; j < N; j++) {
          let ok = true;
          if (opts.causal && j > i) ok = false;
          if (opts.window !== undefined && !(j + opts.window > i)) ok = false;
          active.push(ok);
          if (!ok) {
            scores.push(Number.NEGATIVE_INFINITY);
            continue;
          }
          let dot = 0;
          for (let d = 0; d < D; d++)
            dot += q[base + i * D + d] * k[base + j * D + d];
          let s = dot * SCALE;
          if (opts.softcap !== undefined)
            s = opts.softcap * Math.tanh(s / opts.softcap);
          scores.push(s);
        }
        const m = Math.max(...scores);
        const exps = scores.map((s, j) => (active[j] ? Math.exp(s - m) : 0));
        const denom = exps.reduce((a, c) => a + c, 0);
        for (let d = 0; d < D; d++) {
          let acc = 0;
          for (let j = 0; j < N; j++)
            if (active[j]) acc += (exps[j] / denom) * v[base + j * D + d];
          out[base + i * D + d] = acc;
        }
      }
    }
  return Array.from(out);
}

async function runGPU(
  isCausal: boolean,
  modifier: AttnModifierSpec | undefined,
  data = { q: qData, k: kData, v: vData },
): Promise<number[]> {
  await initWebGPU();
  const gpu = new Torchlette("webgpu");
  const q = gpu.tensorFromArray(data.q, [B, H, N, D]);
  const k = gpu.tensorFromArray(data.k, [B, H, N, D]);
  const v = gpu.tensorFromArray(data.v, [B, H, N, D]);
  const out = await gpu
    .scaledDotProductAttention(q, k, v, SCALE, isCausal, modifier)
    .cpu();
  gpu.markStep();
  return out;
}

async function runCPU(
  isCausal: boolean,
  modifier: AttnModifierSpec | undefined,
): Promise<number[]> {
  const cpu = new Torchlette("cpu");
  const q = cpu.tensorFromArray(qData, [B, H, N, D]);
  const k = cpu.tensorFromArray(kData, [B, H, N, D]);
  const v = cpu.tensorFromArray(vData, [B, H, N, D]);
  const out = await cpu
    .scaledDotProductAttention(q, k, v, SCALE, isCausal, modifier)
    .cpu();
  cpu.markStep();
  return out;
}

function maxAbsDiff(a: number[], b: number[]): number {
  expect(a.length).toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

// Scores at these data magnitudes are ~N(0, 0.2) — Gemma-2's real cap (50)
// would be numerically inert here (tanh(x/50)*50 ≈ x to 1e-5). The gate must
// BITE: cap=0.15 puts scores deep in the tanh saturation regime, so a dropped
// or mis-wired softcap produces a large, unmissable output difference (the
// discrimination test asserts exactly that).
const CAP = 0.15;
const WIN = 24; // < N so the window actually masks

describe.skipIf(cpuOnly)(
  "attention modifiers — softcap/slidingWindow parity (#64 iii)",
  { timeout: 120000 },
  () => {
    it("softcap ∘ causal matches the unfused reference (f32, <1e-4)", async () => {
      const mod: AttnModifierSpec = { scoreMod: { kind: "softcap", cap: CAP } };
      const gpu = await runGPU(true, mod);
      const ref = reference(qData, kData, vData, {
        causal: true,
        softcap: CAP,
      });
      expect(maxAbsDiff(gpu, ref)).toBeLessThan(1e-4);
    });

    it("softcap actually modifies the output (CSE/key discrimination)", async () => {
      // If the modifier were dropped by a cache-key collision, capped and
      // plain would silently agree — assert they DON'T.
      const plain = await runGPU(true, undefined);
      const capped = await runGPU(true, {
        scoreMod: { kind: "softcap", cap: CAP },
      });
      expect(maxAbsDiff(plain, capped)).toBeGreaterThan(1e-3);
    });

    it("causal ∘ slidingWindow matches the unfused reference (f32, <1e-4)", async () => {
      const mod: AttnModifierSpec = {
        maskMods: [{ kind: "causal" }, { kind: "slidingWindow", window: WIN }],
      };
      const gpu = await runGPU(false, mod);
      const ref = reference(qData, kData, vData, { causal: true, window: WIN });
      expect(maxAbsDiff(gpu, ref)).toBeLessThan(1e-4);
    });

    it("softcap ∘ causal ∘ slidingWindow — the Gemma-2 local-layer shape", async () => {
      const mod: AttnModifierSpec = {
        scoreMod: { kind: "softcap", cap: CAP },
        maskMods: [{ kind: "causal" }, { kind: "slidingWindow", window: WIN }],
      };
      const gpu = await runGPU(false, mod);
      const ref = reference(qData, kData, vData, {
        causal: true,
        window: WIN,
        softcap: CAP,
      });
      expect(maxAbsDiff(gpu, ref)).toBeLessThan(1e-4);
      // isCausal=true + window-only modifier must normalize to the same thing
      const folded = await runGPU(true, {
        scoreMod: { kind: "softcap", cap: CAP },
        maskMods: [{ kind: "slidingWindow", window: WIN }],
      });
      expect(maxAbsDiff(gpu, folded)).toBe(0);
    });

    it("cross-path: CPU decomposed interpretation agrees with the fused kernel", async () => {
      const mod: AttnModifierSpec = {
        scoreMod: { kind: "softcap", cap: CAP },
        maskMods: [{ kind: "causal" }, { kind: "slidingWindow", window: WIN }],
      };
      const gpu = await runGPU(false, mod);
      const cpu = await runCPU(false, mod);
      expect(maxAbsDiff(gpu, cpu)).toBeLessThan(1e-3);
    });

    it("f16 gate (MANDATORY): f16-roundtripped QKV, f32 modifier math, <1e-2", async () => {
      await initWebGPU();
      const gpu = new Torchlette("webgpu");
      // Deployment shape: f16-quantized values into the f32 kernel.
      const rt = (d: number[]) =>
        gpu.tensorFromArray(d, [B, H, N, D]).half().float();
      const mod: AttnModifierSpec = {
        scoreMod: { kind: "softcap", cap: CAP },
        maskMods: [{ kind: "causal" }, { kind: "slidingWindow", window: WIN }],
      };
      const out = await gpu
        .scaledDotProductAttention(
          rt(qData),
          rt(kData),
          rt(vData),
          SCALE,
          false,
          mod,
        )
        .cpu();
      gpu.markStep();
      const ref = reference(qData, kData, vData, {
        causal: true,
        window: WIN,
        softcap: CAP,
      });
      expect(maxAbsDiff(out, ref)).toBeLessThan(1e-2);
    });

    it("window larger than N ≡ causal only (boundary)", async () => {
      const windowed = await runGPU(false, {
        maskMods: [
          { kind: "causal" },
          { kind: "slidingWindow", window: N + 8 },
        ],
      });
      const causal = await runGPU(true, undefined);
      expect(maxAbsDiff(windowed, causal)).toBeLessThan(1e-6);
    });

    it("maskMods backward is supported; scoreMod backward throws at forward", async () => {
      await initWebGPU();
      const gpu = new Torchlette("webgpu");
      const q = gpu.tensorFromArray(qData, [B, H, N, D], {
        requiresGrad: true,
      });
      const k = gpu.tensorFromArray(kData, [B, H, N, D], {
        requiresGrad: true,
      });
      const v = gpu.tensorFromArray(vData, [B, H, N, D], {
        requiresGrad: true,
      });
      // window+causal mask: grads flow (masks are constant structure)
      const out = gpu.scaledDotProductAttention(q, k, v, SCALE, true, {
        maskMods: [{ kind: "slidingWindow", window: WIN }],
      });
      await out.sum().backward();
      const dq = await q.grad?.cpu();
      expect(dq?.some((x: number) => x !== 0)).toBe(true);
      gpu.markStep();
      // scoreMod + grads: forward-time throw (inference-first)
      expect(() =>
        gpu.scaledDotProductAttention(q, k, v, SCALE, true, {
          scoreMod: { kind: "softcap", cap: CAP },
        }),
      ).toThrow(/not implemented|inference-first/);
      gpu.markStep();
    });
  },
);
