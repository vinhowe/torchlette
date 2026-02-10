/**
 * Minimal debug - just load model and check embedding.
 */
import * as path from "node:path";
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { loadPretrainedGPT2 } from "./loader";

async function main() {
  console.log("Step 1: initWebGPU");
  await initWebGPU();
  console.log("Step 2: create API");
  const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: true });
  console.log("Step 3: load model");
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.eval();
  console.log("Step 4: create input");
  const input = api.tensorFromArray([15496], [1, 1], { device: "webgpu" });
  console.log("Step 5: embedding lookup");
  const emb = model.wte.forward(input);
  console.log(`Step 6: embedding shape: ${emb.shape}`);
  console.log("Step 7: read back");
  const data = await emb.cpu();
  const arr = Array.from(data);
  console.log(`Step 8: first 5 = [${arr.slice(0, 5).map(v => v.toFixed(6))}]`);
  console.log("Step 9: done");
}

main().catch(e => { console.error("ERROR:", e); process.exit(1); });
