/**
 * Multi-engine GPU-memory reclaim gate (task #94, item 2).
 *
 * Building many engines in one process VkOOM'd probe harnesses after ~8 engines:
 * the implicit new-engine path ORPHANS the previous engine's GPU buffers (a #84
 * safety), leaking device residency for the process lifetime (measured ~7 MB per
 * distil-class engine). The explicit api.destroy() reclaims fully (device
 * teardown + module-global pool/arena/memory-tracker reset), so sequential
 * engines don't accumulate.
 *
 * SCOPE OF THIS GATE: it asserts the LEAK characterization — orphan-only engine
 * construction grows tracked GPU residency — WITHOUT tearing down the real
 * device. api.destroy() itself calls destroyWebGPU(); doing that inside the
 * shared, single-fork webgpu project poisons every subsequent spec (Dawn cannot
 * recreate a device reliably in the shared CI/container Vulkan loader —
 * "vkCreateDevice: Failed to create device chain"). The full end-to-end
 * api.destroy() reclaim (currentBytes → ~0 across N engines, plus correctness of
 * a fresh engine on the reclaimed device) is validated by the standalone
 * tools/t-multiengine-reclaim-probe.ts, which owns its own process and can cycle
 * real devices. Numbers there: DESTROY=0 grows +63 MB / 10 engines (~7 MB each);
 * DESTROY=1 reclaims to 0 MB.
 *
 * WebGPU-only; auto-skips GPU-less (CI).
 */

import { describe, expect, it } from "vitest";
import { getGPUMemoryStats } from "../../src/backend/webgpu";
import { Torchlette } from "../../src/frontend/torchlette";
import { GPT2, type GPT2Config } from "../../examples/gpt2/model";
import { Adam } from "../../src/optim/index";
import { canUseWebGPU } from "../helpers/webgpu";

const TIMEOUT = 120_000;

const CONFIG: GPT2Config = {
  vocabSize: 128,
  blockSize: 32,
  numLayers: 2,
  numHeads: 2,
  embedDim: 32,
  dropoutRate: 0,
};

// Train one orphan-path engine (NO destroy) — mirrors the fp-spike loop shape.
async function trainOrphanEngine(): Promise<void> {
  const BATCH = 2;
  const SEQ = 16;
  const STEPS = 2;
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = new GPT2(api, CONFIG, { device: "webgpu" });
  model.train(true);
  const params = model.parameters();
  const opt = new Adam(params, { lr: 1e-3 }, api);
  const V = CONFIG.vocabSize;
  let seed = 7;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff);
  for (let stp = 0; stp < STEPS; stp++) {
    const inp: number[] = [];
    const tgt: number[] = [];
    for (let i = 0; i < BATCH * SEQ; i++) {
      inp.push(rnd() % V);
      tgt.push(rnd() % V);
    }
    const input = api.tensorFromArray(inp, [BATCH, SEQ], { device: "webgpu" });
    const target = api.tensorFromArray(tgt, [BATCH, SEQ], { device: "webgpu" });
    const loss = api.tidy(() => {
      const l = model.forwardWithLoss(input, target).loss!;
      api.keep(l);
      return l;
    });
    // Detached read BEFORE backward (backward clears the autograd graph and
    // disposes the loss handle — mirror tools/t-fp-spike-probe.ts).
    const lossOut = api.noGrad(() => api.mul(loss, 1));
    await loss.backward();
    opt.step();
    opt.zeroGrad();
    await lossOut.item();
    lossOut.dispose();
    input.dispose();
    target.dispose();
  }
  // No api.destroy() → the engine's buffers are ORPHANED (the leak this gate
  // characterizes; api.destroy() is validated by the standalone probe).
}

describe("multi-engine GPU-memory reclaim (task #94)", () => {
  it(
    "orphan-only sequential engines accumulate tracked GPU residency (the leak api.destroy() reclaims)",
    async () => {
      if (!(await canUseWebGPU())) return;
      const base = getGPUMemoryStats().currentBytes;
      const N = 5;
      let prev = base;
      let grew = false;
      for (let e = 0; e < N; e++) {
        await trainOrphanEngine();
        const cur = getGPUMemoryStats().currentBytes;
        if (cur > prev) grew = true;
        prev = cur;
      }
      const leaked = getGPUMemoryStats().currentBytes;
      // Orphan path: each engine leaves residency behind → residency GREW and the
      // last observation exceeds the starting point. (api.destroy() drives this
      // back to ~0 — see the standalone probe.)
      expect(grew).toBe(true);
      expect(leaked).toBeGreaterThan(base);
    },
    TIMEOUT,
  );
});
