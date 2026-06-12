/**
 * Checkpoint x autocast x fusion parity gate.
 *
 * Until 2026-06-12, the full-finetuning suite carried four sites of
 * "autocast disabled due to known reshape issue" / "fusion disabled due to
 * reshape issues" — a known-broken combination routed around in comments,
 * never reproduced, tracked, or re-validated. The combination is every
 * transformer a user actually trains (checkpointing for memory + f16 for
 * speed). The historical bug was fixed by this cycle's autocast+checkpoint
 * narrowBackward and multi-output lifetime work; this spec converts the
 * folklore into coverage:
 *  - checkpoint alone must be EXACT vs the plain reference;
 *  - checkpoint must add ZERO error on top of autocast (consistent f16
 *    recomputation — checkpoint+ac must match ac within tight tolerance);
 *  - all params get finite grads in every combination, fusion included.
 */
import { beforeAll, describe, expect, it } from "vitest";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { Torchlette } from "../src/frontend/torchlette";
import { canUseWebGPU } from "./helpers/webgpu";

const CFG: GPT2Config = {
  vocabSize: 500,
  blockSize: 64,
  numLayers: 4,
  numHeads: 4,
  embedDim: 128,
  dropoutRate: 0.0,
};

interface RunResult {
  loss: number;
  nullGrads: number;
  nonfiniteParams: number;
  gradSample: number[];
}

async function run(
  ckpt: boolean,
  ac: boolean,
  fusion: boolean,
): Promise<RunResult> {
  const api = new Torchlette("webgpu", {
    enableFusion: fusion,
    enableMemoryPlanning: true,
  });
  const model = new GPT2(api, CFG, { device: "webgpu" });
  model.train();
  const B = 2;
  const S = 16;
  let x = 1234;
  const tok = () => {
    x = (x * 1103515245 + 12345) % 2147483648;
    return x % CFG.vocabSize;
  };
  const input = api.tensorFromArray(Array.from({ length: B * S }, tok), [B, S]);
  const target = api.tensorFromArray(Array.from({ length: B * S }, tok), [B, S]);
  const fwd = () => model.forwardWithLoss(input, target, { useCheckpoint: ckpt });
  const { loss } = ac ? api.autocast(fwd) : fwd();
  if (!loss) throw new Error("loss is null");
  const lv = await loss.item();
  await loss.backward();
  let nullGrads = 0;
  let nonfiniteParams = 0;
  const gradSample: number[] = [];
  for (const p of model.parameters()) {
    const g = (p as unknown as { grad: { cpu(): Promise<Float32Array> } | null })
      .grad;
    if (!g) {
      nullGrads++;
      continue;
    }
    const arr = await g.cpu();
    for (let i = 0; i < Math.min(4, arr.length); i++) gradSample.push(arr[i]);
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i])) {
        nonfiniteParams++;
        break;
      }
    }
  }
  return { loss: lv, nullGrads, nonfiniteParams, gradSample };
}

function maxSampleDiff(a: RunResult, b: RunResult): number {
  let worst = 0;
  const n = Math.min(a.gradSample.length, b.gradSample.length);
  for (let i = 0; i < n; i++) {
    worst = Math.max(worst, Math.abs(a.gradSample[i] - b.gradSample[i]));
  }
  return worst;
}

describe("checkpoint x autocast x fusion parity", { timeout: 600_000 }, () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = await canUseWebGPU();
  });

  it("checkpoint(+fusion) is exact; autocast band is consistent with and without checkpoint", async () => {
    if (!webgpu) return;
    const ref = await run(false, false, false);
    expect(ref.nullGrads).toBe(0);
    expect(ref.nonfiniteParams).toBe(0);

    // Checkpoint must be a pure memory optimization: exact recomputation.
    const ckpt = await run(true, false, false);
    expect(Math.abs(ckpt.loss - ref.loss)).toBeLessThan(1e-4);
    expect(maxSampleDiff(ckpt, ref)).toBeLessThan(1e-5);
    expect(ckpt.nullGrads).toBe(0);
    expect(ckpt.nonfiniteParams).toBe(0);

    const ckptFused = await run(true, false, true);
    expect(Math.abs(ckptFused.loss - ref.loss)).toBeLessThan(1e-3);
    expect(ckptFused.nullGrads).toBe(0);
    expect(ckptFused.nonfiniteParams).toBe(0);

    // Autocast moves loss by the f16 band; checkpoint must add ~nothing on
    // top of it (recompute-consistency: same casts both passes).
    const ac = await run(false, true, false);
    const ckptAc = await run(true, true, false);
    expect(Math.abs(ac.loss - ref.loss)).toBeLessThan(0.5); // f16 band
    expect(Math.abs(ckptAc.loss - ac.loss)).toBeLessThan(1e-3);
    expect(ckptAc.nullGrads).toBe(0);
    expect(ckptAc.nonfiniteParams).toBe(0);

    const full = await run(true, true, true);
    expect(Math.abs(full.loss - ac.loss)).toBeLessThan(1e-2);
    expect(full.nullGrads).toBe(0);
    expect(full.nonfiniteParams).toBe(0);
  });
});
