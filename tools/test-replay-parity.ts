/**
 * Loss parity test for dispatch replay.
 * MODE=forward-only: no backward/optimizer, tests pure replay determinism
 * MODE=training (default): full training loop
 */
import * as path from "node:path";
import { Torchlette, type Tensor } from "../src/frontend";
import { initWebGPU, destroyWebGPU } from "../src/backend/webgpu";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { Adam } from "../src/optim";
import { crossEntropy } from "../src/nn";

async function main() {
  await initWebGPU();

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });

  const mode = process.env.MODE || "training";
  const optimizer = mode === "training"
    ? new Adam(model.parameters(), { lr: 1e-4, weightDecay: 0.01 }, api)
    : null;

  const seqLen = 31;
  const vocabSize = 50257;
  const inputData = Array.from({ length: seqLen }, (_, i) => i % vocabSize);
  const targetData = Array.from({ length: seqLen }, (_, i) => (i + 1) % vocabSize);

  const compiledForward = api.compile((inp: Tensor, tgt: Tensor) => {
    const logits = model.forward(inp);
    const flatLogits = logits.reshape([seqLen, vocabSize]);
    const flatTargets = tgt.reshape([seqLen]);
    return crossEntropy(api, flatLogits, flatTargets);
  });

  const NUM_STEPS = parseInt(process.env.NUM_STEPS || "8", 10);
  const losses: number[] = [];

  // Get first param for gradient dump
  const params = model.parameters();
  const firstParam = params.length > 0 ? params[0] : null;

  for (let step = 0; step < NUM_STEPS; step++) {
    const input = api.tensorFromArray(inputData, [1, seqLen], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, seqLen], { device: "webgpu" });

    const t0 = performance.now();
    const loss = compiledForward(input, target);
    const lossValue = await loss.item();

    if (mode === "training" || mode === "backward-only") {
      await loss.backward();
    }

    // Dump gradient of first param (wte embedding) at specific steps
    if (process.env.DUMP_GRADS === "1" && firstParam && (firstParam as any).grad && step >= 3) {
      const grad = (firstParam as any).grad as Tensor;
      const gradData = await grad.cpu();
      const vals = gradData.slice(0, 8);
      console.log(`  grad[0:8] @ step ${step}: [${vals.map((v: number) => v.toFixed(12)).join(", ")}]`);
    }

    if (mode === "training") {
      optimizer!.step();
      optimizer!.zeroGrad();
    }

    loss.dispose();
    input.dispose();
    target.dispose();
    await api.markStep();
    const t4 = performance.now();

    losses.push(lossValue);
    console.log(`Step ${step}: loss=${lossValue.toFixed(8)} | total=${(t4-t0).toFixed(0)}ms`);
  }

  const replay = process.env.TORCHLETTE_DISPATCH_REPLAY === "0" ? "OFF" : "ON";
  console.log(`\nMode: ${mode} | Replay: ${replay}`);
  console.log(`Losses: [${losses.map(l => l.toFixed(8)).join(", ")}]`);

  destroyWebGPU();
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
