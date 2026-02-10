/**
 * Compare CPU dot product vs GPU matmul for logit computation.
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

  // Get final hidden state
  const input = api.tensorFromArray([15496], [1, 1], { device: "webgpu" });
  const pos = api.tensorFromArray([0], [1, 1]);
  let x = api.add(model.wte.forward(input), model.wpe.forward(pos));
  x = model.drop.forward(x);
  for (const block of model.h) x = block.forward(x);
  x = model.lnF.forward(x);
  const hs = Array.from(await x.cpu());
  console.log(`hs mean=${(hs.reduce((a,b)=>a+b,0)/768).toFixed(6)}, first 3: [${hs.slice(0,3).map(v=>v.toFixed(6))}]`);

  // Get embedding for token 383
  const emb383 = Array.from(await model.wte.forward(api.tensorFromArray([383], [1, 1], { device: "webgpu" })).cpu());
  console.log(`emb383 first 3: [${emb383.slice(0,3).map(v=>v.toFixed(6))}]`);

  // CPU dot product
  let cpuDot = 0;
  for (let j = 0; j < 768; j++) cpuDot += hs[j] * emb383[j];
  console.log(`CPU dot(hs, wte[383]) = ${cpuDot.toFixed(6)}`);
  console.log(`(oracle logits[383] = -29.416668)`);

  // Also check dot product for token 0
  const emb0 = Array.from(await model.wte.forward(api.tensorFromArray([0], [1, 1], { device: "webgpu" })).cpu());
  let cpuDot0 = 0;
  for (let j = 0; j < 768; j++) cpuDot0 += hs[j] * emb0[j];
  console.log(`CPU dot(hs, wte[0]) = ${cpuDot0.toFixed(6)}`);

  // GPU matmul: small test with just 2 rows
  const smallW = api.tensorFromArray([...emb383, ...emb0], [2, 768], { device: "webgpu" });
  const smallLogits = x.reshape([1, 768]).matmul(smallW.transpose({ dim0: 0, dim1: 1 }).contiguous());
  const smallData = Array.from(await smallLogits.cpu());
  console.log(`\nGPU matmul(hs, [wte383, wte0]^T):`);
  console.log(`  [0] (token 383) = ${smallData[0]?.toFixed(6)}`);
  console.log(`  [1] (token 0) = ${smallData[1]?.toFixed(6)}`);

  process.exit(0);
}

main().catch(e => { console.error("ERROR:", e.message, e.stack); process.exit(1); });
