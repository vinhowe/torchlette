/**
 * Second-engine-in-process reclaimed-read probe (task #70 D4 — indictment row 8).
 *
 * The gpt2-memorization overfit FP is NOT a parallel-run flake and NOT a
 * gen-perturbation (the D1/D2 attributions were both wrong). It is the SECOND-
 * ENGINE-IN-PROCESS class (#84): before the D4 fix, `disposeAllForNewEngine`
 * did not clear the storage tracker's `_stepStartTensors` snapshot, so a second
 * Torchlette engine in one process inherited the dead first engine's stale
 * non-null snapshot. Its very first implied-boundary `releaseStepTemps` then ran
 * against that stale snapshot and reaped the second engine's live step-0 forward
 * activation (owners=1, snap=false) → a [lifetime] reclaimed-read throw under the
 * strict default. The FIRST engine never hit it (its snapshot was still null on
 * its first boundary).
 *
 * This probe builds N GPT-2 engines back-to-back in ONE process, each running the
 * stepAsync + implied-boundary overfit loop under the STRICT lifetime default.
 * Pre-fix: engine #0 passes, engine #1+ throw — DETERMINISTICALLY, no parallelism.
 * Post-fix: every engine converges to ~0 with no throw.
 *
 *   LD_LIBRARY_PATH=tools/vk-shim VULKAN_DEVICE_INDEX=N npx tsx \
 *     tools/t-second-engine-overfit-probe.ts
 *
 * Env: RUNS (default 3) — number of engines to build in the process.
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim";

class CharTok {
  private m = new Map<string, number>();
  readonly vocabSize: number;
  constructor(chars: string) {
    const u = [...new Set(chars.split(""))].sort();
    this.m.set("<PAD>", 0);
    u.forEach((c, i) => this.m.set(c, i + 1));
    this.vocabSize = u.length + 1;
  }
  encode(t: string): number[] {
    return t.split("").map((c) => this.m.get(c) ?? 0);
  }
}

async function overfitOnce(): Promise<number> {
  const sequence = "Hello World!";
  const tok = new CharTok(sequence);
  const cfg: GPT2Config = {
    vocabSize: tok.vocabSize,
    blockSize: sequence.length + 1,
    numLayers: 2,
    numHeads: 4,
    embedDim: 128,
    dropoutRate: 0,
  };
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: true,
  });
  const model = new GPT2(api, cfg, { device: "webgpu" });
  model.train();
  const opt = new Adam(model.parameters(), { lr: 0.01 }, api);
  const e = tok.encode(sequence);
  const inp = api.tensorFromArray(e.slice(0, -1), [1, e.length - 1], {
    device: "webgpu",
  });
  const tgt = api.tensorFromArray(e.slice(1), [1, e.length - 1], {
    device: "webgpu",
  });
  let final = Number.POSITIVE_INFINITY;
  for (let step = 0; step < 1000; step++) {
    const { loss } = model.forwardWithLoss(inp, tgt);
    if (!loss) throw new Error("no loss");
    const lv = await loss.item();
    await loss.backward();
    await opt.stepAsync();
    opt.zeroGrad();
    loss.dispose();
    final = lv;
    if (lv < 0.0001) break;
  }
  return final;
}

async function main(): Promise<void> {
  await initWebGPU();
  const runs = Number.parseInt(process.env.RUNS ?? "3", 10);
  let failures = 0;
  for (let r = 0; r < runs; r++) {
    try {
      const loss = await overfitOnce();
      const ok = loss < 0.01;
      if (!ok) failures++;
      console.log(
        `[2nd-engine] engine #${r} final loss=${loss.toFixed(4)} ${ok ? "OK" : "NOT-CONVERGED"}`,
      );
    } catch (err) {
      failures++;
      console.error(
        `[2nd-engine] engine #${r} THREW: ${(err as Error).message.split("\n")[0]}`,
      );
    }
  }
  console.log(
    failures === 0
      ? `[2nd-engine] PASS — all ${runs} in-process engines converged, no reclaimed-read throw`
      : `[2nd-engine] FAIL — ${failures}/${runs} engines threw or did not converge`,
  );
  destroyWebGPU();
  process.exit(failures === 0 ? 0 : 1);
}
main();
