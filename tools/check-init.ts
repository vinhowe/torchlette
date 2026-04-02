/**
 * Check init weights for determinism across platforms.
 * Run on V100: DAWN_GPU=15 npx tsx tools/check-init.ts
 * Compare output with browser console.
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu/index.js";
import { Torchlette } from "../src/frontend/torchlette.js";
import { normal_ } from "../src/nn/init.js";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(42);

  const { GPT2WithLoRA } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora"
  );
  const model = new GPT2WithLoRA(
    api,
    {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: 12,
      numHeads: 12,
      embedDim: 768,
      dropoutRate: 0,
    },
    { rank: 1, alpha: 1 },
    "webgpu",
  );

  const params = model.getAllParameters();
  for (const p of params) {
    if (p.shape.length >= 2) normal_(api, p, 0, 0.02);
  }
  await api._runtime().forceAllPending();

  // Print first 4 values of first few params
  for (let i = 0; i < 5; i++) {
    const data = await params[i].cpu();
    const vals = Array.from(data.slice(0, 4)).map((v) => v.toFixed(8));
    console.log(`param[${i}] shape=${params[i].shape} first4=[${vals}]`);
  }
  console.log(`Total params: ${params.length}`);

  await destroyWebGPU();
  process.exit(0);
}

main();
