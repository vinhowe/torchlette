/**
 * Attention modifier seams (#64) — commit-A differential gates.
 *
 * The parity gate that must exist BEFORE the legacy causal branch is deleted
 * (commit B): the causal MODIFIER path (structural mask via the "attn_mask"
 * seam) must be BIT-IDENTICAL to the legacy isCausal-uniform path, forward
 * and backward, because commit B routes all causal attention through it.
 *
 * Also gates: null-modifier canonicalization ({} ≡ undefined — a `{}`-keyed
 * template under the same "" cache key would be the WGSL-cache collision
 * class), isCausal-fold normalization, and the inference-first scoreMod
 * guard (fails at forward, not deep in backward).
 */
import { describe, expect, it } from "vitest";
import type { AttnModifierSpec } from "../../src/backend/types";
import { initWebGPU } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/index";
import { cpuOnly } from "../helpers/webgpu";

const B = 2,
  H = 3,
  N = 100, // deliberately not a multiple of BR=64 / BC=32 — exercises guards
  D = 64;
const SCALE = 1.0 / Math.sqrt(D);

function makeData(seed: number): number[] {
  const out: number[] = [];
  let s = seed;
  for (let i = 0; i < B * H * N * D; i++) {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    out.push((s / 0x7fffffff - 0.5) * 1.5);
  }
  return out;
}
const qData = makeData(1);
const kData = makeData(2);
const vData = makeData(3);

/** Run SDPA fwd(+bwd) on GPU; returns output and (optionally) grads. */
async function run(
  isCausal: boolean,
  modifier: AttnModifierSpec | undefined,
  withBackward: boolean,
): Promise<{ out: number[]; dQ?: number[]; dK?: number[]; dV?: number[] }> {
  await initWebGPU();
  const gpu = new Torchlette("webgpu");
  const q = gpu.tensorFromArray(qData, [B, H, N, D], {
    requiresGrad: withBackward,
  });
  const k = gpu.tensorFromArray(kData, [B, H, N, D], {
    requiresGrad: withBackward,
  });
  const v = gpu.tensorFromArray(vData, [B, H, N, D], {
    requiresGrad: withBackward,
  });
  const result = gpu.scaledDotProductAttention(
    q,
    k,
    v,
    SCALE,
    isCausal,
    modifier,
  );
  const out = await result.cpu();
  if (!withBackward) {
    gpu.markStep();
    return { out };
  }
  await result.sum().backward();
  const dQ = await q.grad?.cpu();
  const dK = await k.grad?.cpu();
  const dV = await v.grad?.cpu();
  gpu.markStep();
  return { out, dQ, dK, dV };
}

function expectBitIdentical(a: number[], b: number[], label: string): void {
  expect(a.length).toBe(b.length);
  let mismatches = 0;
  let firstIdx = -1;
  for (let i = 0; i < a.length; i++) {
    if (!Object.is(a[i], b[i])) {
      mismatches++;
      if (firstIdx < 0) firstIdx = i;
    }
  }
  if (mismatches > 0) {
    // Loud context before the assertion fires
    console.error(
      `${label}: ${mismatches}/${a.length} mismatches; first at ${firstIdx}: ` +
        `${a[firstIdx]} vs ${b[firstIdx]}`,
    );
  }
  expect(mismatches).toBe(0);
}

describe.skipIf(cpuOnly)(
  "attention modifier seams — causal-as-modifier bit-parity (#64 A)",
  { timeout: 120000 },
  () => {
    it("forward: causal modifier ≡ legacy isCausal (bit-identical)", async () => {
      const legacy = await run(true, undefined, false);
      const modded = await run(
        false,
        { maskMods: [{ kind: "causal" }] },
        false,
      );
      expectBitIdentical(legacy.out, modded.out, "fwd causal");
    });

    it("backward: causal modifier ≡ legacy isCausal (dQ/dK/dV bit-identical)", async () => {
      const legacy = await run(true, undefined, true);
      const modded = await run(false, { maskMods: [{ kind: "causal" }] }, true);
      expectBitIdentical(legacy.out, modded.out, "bwd-run fwd");
      expectBitIdentical(legacy.dQ as number[], modded.dQ as number[], "dQ");
      expectBitIdentical(legacy.dK as number[], modded.dK as number[], "dK");
      expectBitIdentical(legacy.dV as number[], modded.dV as number[], "dV");
    });

    it("null modifier canonicalization: {} ≡ undefined (non-causal)", async () => {
      const bare = await run(false, undefined, false);
      const empty = await run(false, {}, false);
      expectBitIdentical(bare.out, empty.out, "null-mod fwd");
    });

    it("normalization folds isCausal=true INTO a present modifier", async () => {
      // isCausal + modifier-without-causal must still be causal — dropping
      // the fold would silently lose causality on the modifier path.
      const legacy = await run(true, undefined, false);
      const folded = await run(true, {}, false); // {} canonicalizes → legacy
      expectBitIdentical(legacy.out, folded.out, "isCausal fold (empty mod)");
    });

    it("inference-first: scoreMod + requiresGrad fails AT FORWARD", async () => {
      await initWebGPU();
      const gpu = new Torchlette("webgpu");
      const q = gpu.tensorFromArray(qData, [B, H, N, D], {
        requiresGrad: true,
      });
      const k = gpu.tensorFromArray(kData, [B, H, N, D]);
      const v = gpu.tensorFromArray(vData, [B, H, N, D]);
      expect(() =>
        gpu.scaledDotProductAttention(q, k, v, SCALE, true, {
          scoreMod: { kind: "softcap", cap: 50 },
        }),
      ).toThrow(/scoreMod.*not implemented|inference-first/);
      gpu.markStep();
    });
  },
);
