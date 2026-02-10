/**
 * Quick check: logit values for specific tokens.
 */
import * as path from "node:path";
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { loadPretrainedGPT2 } from "./loader";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: true });
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.eval();

  const input = api.tensorFromArray([15496], [1, 1], { device: "webgpu" });
  const { logits } = model.forwardWithLoss(input);

  // Gather specific logits
  const tokenIds = [383, 13, 11, 0, 1];
  const idxTensor = api.tensorFromArray(tokenIds, [1, 1, tokenIds.length], { device: "webgpu" });
  const selectedLogits = logits.gather(idxTensor, { dim: 2 });
  const selData = Array.from(await selectedLogits.cpu());
  console.log("Selected logits:");
  for (let i = 0; i < tokenIds.length; i++) {
    console.log(`  logits[${tokenIds[i]}] = ${selData[i]?.toFixed(6)}`);
  }
  console.log("(oracle: 383=-29.42, 13=-30.23, 11=-30.29)");
  process.exit(0);
}

main().catch(e => { console.error("ERROR:", e.message, e.stack); process.exit(1); });
