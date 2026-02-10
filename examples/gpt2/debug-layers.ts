/**
 * Compare hidden states at each transformer layer against PyTorch oracle.
 */
import * as path from "node:path";
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { loadPretrainedGPT2 } from "./loader";

function stats(arr: number[]) {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const std = Math.sqrt(arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length);
  return { mean, std };
}

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: true });
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.eval();

  const input = api.tensorFromArray([15496], [1, 1], { device: "webgpu" });

  // PyTorch oracle hidden states (from oracle_gpt2.py):
  // Layer 0: mean=-0.005223, std=0.136276
  // Layer 1: mean=0.280448, std=9.644617
  // Layer 2: mean=0.170069, std=9.648085
  // Layer 3: mean=0.221858, std=9.696393
  // Layer 4: mean=0.190048, std=9.692048
  // Layer 5: mean=0.155413, std=9.676968
  // Layer 6: mean=0.142023, std=9.638908

  // Reproduce forward pass step by step
  const posData = [0];
  const pos = api.tensorFromArray(posData, [1, 1]);
  const tokEmb = model.wte.forward(input);
  const posEmb = model.wpe.forward(pos);
  let x = api.add(tokEmb, posEmb);
  x = model.drop.forward(x);

  // Layer 0 = input embedding
  let xData = Array.from(await x.cpu());
  let s = stats(xData);
  console.log(`Layer 0 (embed): mean=${s.mean.toFixed(6)}, std=${s.std.toFixed(6)} (oracle: mean=-0.005223, std=0.136276)`);

  // Pass through each block
  for (let i = 0; i < model.h.length; i++) {
    x = x.reshape([1, 1, 768]); // ensure proper shape
    x = model.h[i].forward(x);
    xData = Array.from(await x.cpu());
    s = stats(xData);
    console.log(`Layer ${i+1} (block ${i}): mean=${s.mean.toFixed(6)}, std=${s.std.toFixed(6)}`);
  }

  // Final layer norm
  x = model.lnF.forward(x);
  xData = Array.from(await x.cpu());
  s = stats(xData);
  console.log(`After lnF: mean=${s.mean.toFixed(6)}, std=${s.std.toFixed(6)}`);

  // Logits
  const logits = x.matmul(model.wte.weight.transpose({ dim0: 0, dim1: 1 }).contiguous());
  const logitsData = Array.from(await logits.cpu());
  s = stats(logitsData);
  console.log(`Logits: mean=${s.mean.toFixed(4)}, std=${s.std.toFixed(4)} (oracle: mean=-41.0766, std=2.7435)`);

  // Top 5
  const indexed = logitsData.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => b.v - a.v);
  console.log("Top 5:");
  for (let i = 0; i < 5; i++) {
    console.log(`  ${i+1}. token_id=${indexed[i].i}, logit=${indexed[i].v.toFixed(4)}`);
  }
  console.log("(oracle top: 383=-29.42, 13=-30.23, 11=-30.29, 317=-30.63, 464=-30.69)");

  console.log("\n=== Done ===");
}

main().catch(e => { console.error("ERROR:", e.message, e.stack); process.exit(1); });
