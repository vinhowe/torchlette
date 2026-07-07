/**
 * Local-only correctness test for the toy-compartmentalization model.
 * Reproduces the loss-diverges-after-N-steps issue the user is hitting in
 * the browser without remote training, so we can isolate whether it's a
 * numerical regression in the local execution path.
 */
import { nn, Adam } from "../src/index";
import { Torchlette } from "../src/frontend/torchlette";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import {
  generateBatch,
  setTransitionMatrices,
  VOCAB_SIZE_DATA,
} from "../examples/toy-compartmentalization/src/lib/data";
import { initWebGPU, webgpuBackend } from "../src/backend/webgpu";
import { registerBackend } from "../src/backend/registry";

async function main() {
  const STEPS = parseInt(process.env.STEPS ?? "30", 10);
  const LR = parseFloat(process.env.LR ?? "1e-4");

  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA + 1, S = 10, B = 64;

  const useWebGPU = process.env.BACKEND !== "cpu";
  if (useWebGPU) {
    const ok = await initWebGPU();
    if (!ok) { console.error("WebGPU init failed"); process.exit(1); }
    registerBackend(webgpuBackend);
  } else {
    const { cpuBackend } = await import("../src/backend/cpu");
    registerBackend(cpuBackend);
  }
  const api = new Torchlette(useWebGPU ? "webgpu" : "cpu");
  api.manualSeed(42);

  const m = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });
  const o = new Adam(m.parameters(), { lr: LR });

  console.log(`STEPS=${STEPS} LR=${LR} batch=${B} seq=${S} vocab=${V}`);

  for (let step = 0; step < STEPS; step++) {
    const t0 = performance.now();
    await api.beginStep();
    const batch = generateBatch({ seqLen: S, batchSize: B });
    const tok = api.tensorFromArray(batch.tokens, [B, S], { dtype: "i32" });
    const tgt = api.tensorFromArray(batch.targets as number[], [B * (S - 1)], { dtype: "i32" });

    const loss = api.tidy(() => {
      const fwd = m.forward(tok);
      const logits = fwd.logits.narrow(1, 0, S - 1).contiguous().reshape([B * (S - 1), V]);
      const l = crossEntropy(api, logits, tgt);
      api.keep(l);
      return l;
    });
    tok.dispose(); tgt.dispose();

    const v = await loss.item();
    await loss.backward(); loss.dispose();
    o.step(); o.zeroGrad();
    await api.endStep();
    const elapsed = performance.now() - t0;
    console.log(`step ${step}: loss=${v?.toFixed?.(4) ?? v} (${elapsed.toFixed(0)}ms)`);

    if (!Number.isFinite(v)) {
      console.error(`!!! Loss diverged at step ${step} !!!`);
      process.exit(1);
    }
  }

  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
