/**
 * shape=[] GradScaler grad-seed lifetime gate (recorded-build sunset, task #43).
 *
 * THE CLASS. Under selective checkpointing, `backward()` force-materializes the
 * grad seed `full([],1.0)` in a SEPARATE "forward tensors" plan (autograd.ts,
 * `forwardToForce`). That makes the seed a CROSS-PLAN value: produced in that
 * plan, consumed by the main backward plan. With a GradScaler the consumer is the
 * extra `mul(seed, scale)` backward node (the derivative of `scale(loss)`), which
 * runs in a LATER-forced segment.
 *
 * On the RECORDED build the harvest rc-pins every produced result, so the seed
 * survives to that later read. When the recorded build is retired and the compiled
 * plan is built from the generated stream, the observed-liveness harvest cannot
 * WITNESS that later cross-plan read (it is data-dependent — the GradScaler
 * inf-skip re-fingerprints the plans), so it prunes the seed's harvested `full`
 * result; its rc then hits 0 and `destroyUnreachable` reaps it mid-backward,
 * before the consumer reads it — `[lifetime] reading RECLAIMED storage (shape=[])`.
 *
 * DETERMINISTIC REPRO (deleted tree): apply the harvest-deletion diff
 * (`.claude/harvest-deletion-43a-reconciled.diff`) and run
 *   VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *     TORCHLETTE_STEP_TAPE=record CELL=scaler-inf npx tsx tools/t-witness-harvest-matrix.ts
 * PRE-FIX it throws `[lifetime] reading RECLAIMED storage id=… (shape=[])` at
 * step ~6; POST-FIX the shape=[] class is gone (see stage4 §Task #43 for the
 * remaining broader `forwardToForce` cross-plan class that still blocks the full
 * deletion).
 *
 * THE FIX (autograd.ts): the grad seed is a LEAF CONSTANT — do not force it in the
 * separate forward-tensors plan; leaving it lazy materializes it INSIDE the main
 * backward plan alongside its consumer (intra-plan), so it is never a prunable
 * cross-plan harvested result. Null on the recorded build (the harvest pins the
 * seed either way — this spec passes on both trees). This gate is the regression
 * guard: it drives the exact checkpoint + GradScaler backward and asserts the seed
 * chain produces finite grads with no lifetime throw, matching the checkpoint-only
 * reference trajectory.
 */
import { beforeAll, describe, expect, it } from "vitest";
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler } from "../src/optim/index";
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
}

async function run(useScaler: boolean): Promise<RunResult> {
  const api = new Torchlette("webgpu", { enableMemoryPlanning: true });
  const model = new GPT2(api, CFG, { device: "webgpu" });
  model.train();
  const params = [...model.parameters()];
  const opt = new Adam(params, { lr: 1e-3 }, api);
  // A large initial scale so `scale(loss)` shifts the seed→scale→loss backward
  // chain (the cross-plan `mul(seed, scale)` node the class needs).
  const scaler = new GradScaler(api, { initScale: 4096 });

  const B = 2;
  const S = 16;
  let x = 4321;
  const tok = () => {
    x = (x * 1103515245 + 12345) % 2147483648;
    return x % CFG.vocabSize;
  };
  const input = api.tensorFromArray(Array.from({ length: B * S }, tok), [B, S]);
  const target = api.tensorFromArray(Array.from({ length: B * S }, tok), [
    B,
    S,
  ]);

  const { loss } = model.forwardWithLoss(input, target, {
    useCheckpoint: true,
  });
  if (!loss) throw new Error("loss is null");
  const lv = await loss.item();

  // WITH scaler: backward the SCALED loss — this is the config that surfaces the
  // shape=[] seed as a cross-plan value under checkpointing.
  const backTgt = useScaler ? scaler.scale(loss) : loss;
  await backTgt.backward();

  let nullGrads = 0;
  let nonfiniteParams = 0;
  for (const p of params) {
    const g = (
      p as unknown as { grad: { cpu(): Promise<Float32Array> } | null }
    ).grad;
    if (!g) {
      nullGrads++;
      continue;
    }
    const arr = await g.cpu();
    for (let i = 0; i < arr.length; i++) {
      if (!Number.isFinite(arr[i])) {
        nonfiniteParams++;
        break;
      }
    }
  }
  opt.zeroGrad();
  return { loss: lv, nullGrads, nonfiniteParams };
}

describe(
  "shape=[] grad-seed lifetime under checkpoint + GradScaler",
  { timeout: 600_000 },
  () => {
    let webgpu = false;
    beforeAll(async () => {
      webgpu = await canUseWebGPU();
    });

    it("checkpoint + GradScaler backward: finite grads, no lifetime throw, seed materializes intra-plan", async () => {
      if (!webgpu) return;
      // The reference: checkpoint backward WITHOUT the scaler (backward the raw
      // scalar loss). The seed still exists, but no cross-plan `mul(seed, scale)`.
      const ref = await run(false);
      expect(ref.nullGrads).toBe(0);
      expect(ref.nonfiniteParams).toBe(0);

      // WITH the scaler: the seed→scale backward chain. Must complete under STRICT
      // lifetime (default) with finite grads — the shape=[] reclaim must not fire.
      const scaled = await run(true);
      expect(scaled.nullGrads).toBe(0);
      expect(scaled.nonfiniteParams).toBe(0);
      // The scaled loss value equals the reference (scale is applied to the grad
      // chain, not the reported loss the harness reads pre-scale).
      expect(Math.abs(scaled.loss - ref.loss)).toBeLessThan(1e-4);
    });
  },
);
