/**
 * Debug: call attn.forward() as one lazy graph (no intermediate forcing).
 * Also test full forward pass.
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

  // Test 1: Single attention layer
  console.log("=== Test 1: Single attention layer ===");
  const input = api.tensorFromArray([15496], [1, 1], { device: "webgpu" });
  const pos = api.tensorFromArray([0], [1, 1]);
  const tokEmb = model.wte.forward(input);
  const posEmb = model.wpe.forward(pos);
  let x = api.add(tokEmb, posEmb).reshape([1, 1, 768]);
  await x.cpu(); // force

  const attnOut = model.h[0].attn.forward(model.h[0].ln1.forward(x));
  const attnData = await attnOut.cpu();
  const attnArr = Array.from(attnData);
  const attnMean = attnArr.reduce((a,b)=>a+b,0)/attnArr.length;
  console.log(`attn mean=${attnMean.toFixed(6)}, first 5: [${attnArr.slice(0,5).map(v=>v.toFixed(4))}]`);
  console.log(`(step-by-step oracle: mean=0.003247, first 5: [-0.3959,-0.6008,0.1411,0.0321,0.0778])`);

  // Test 2: Full forward pass (single token)
  console.log("\n=== Test 2: Full forward pass ===");
  const { logits } = model.forwardWithLoss(input);
  const logitsData = Array.from(await logits.cpu());
  const logitsMean = logitsData.reduce((a, b) => a + b, 0) / logitsData.length;
  const logitsStd = Math.sqrt(logitsData.reduce((a, b) => a + (b - logitsMean) ** 2, 0) / logitsData.length);
  console.log(`logits mean=${logitsMean.toFixed(4)}, std=${logitsStd.toFixed(4)}`);
  console.log(`(oracle: mean=-41.0766, std=2.7435)`);

  // Top 5
  const indexed = logitsData.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => b.v - a.v);
  console.log("Top 5:");
  for (let i = 0; i < 5; i++) {
    console.log(`  ${i+1}. token_id=${indexed[i].i}, logit=${indexed[i].v.toFixed(4)}`);
  }
  console.log("(oracle top: 383=-29.42, 13=-30.23, 11=-30.29)");

  console.log("\n=== Done ===");
}

main().catch(e => { console.error("ERROR:", e.message, e.stack); process.exit(1); });
