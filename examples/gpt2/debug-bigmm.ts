/**
 * Test: is the large matmul [1,768] @ [768, 50257] broken?
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

  // Compute final hidden state
  const input = api.tensorFromArray([15496], [1, 1], { device: "webgpu" });
  const pos = api.tensorFromArray([0], [1, 1]);
  let x = api.add(model.wte.forward(input), model.wpe.forward(pos));
  x = model.drop.forward(x);
  for (const block of model.h) x = block.forward(x);
  x = model.lnF.forward(x);
  // Force materialization
  const hsData = await x.cpu();
  console.log(`hs first 3: [${Array.from(hsData).slice(0,3).map(v=>v.toFixed(6))}]`);

  // Test 1: matmul with full wte.weight^T
  console.log("\nTest 1: full matmul x @ wte^T");
  const wteT = model.wte.weight.transpose({ dim0: 0, dim1: 1 }).contiguous();
  // Materialize wte^T first
  console.log(`wte^T shape: [${wteT.shape}]`);

  // Do the matmul
  const logitsFull = x.reshape([1, 768]).matmul(wteT);
  // Gather just token 383
  const idx383 = api.tensorFromArray([383], [1, 1], { device: "webgpu" });
  const logit383 = logitsFull.gather(idx383, { dim: 1 });
  const logit383Data = Array.from(await logit383.cpu());
  console.log(`logits[383] from full matmul = ${logit383Data[0]?.toFixed(6)}`);
  console.log(`(oracle: -29.416668)`);

  // Test 2: same but x is [1, 1, 768] (batched)
  console.log("\nTest 2: batched matmul x[1,1,768] @ wte^T[768,50257]");
  const logitsBatch = x.matmul(wteT);
  const idx383b = api.tensorFromArray([383], [1, 1, 1], { device: "webgpu" });
  const logit383b = logitsBatch.gather(idx383b, { dim: 2 });
  const logit383bData = Array.from(await logit383b.cpu());
  console.log(`logits[383] from batched matmul = ${logit383bData[0]?.toFixed(6)}`);

  // Test 3: model.forwardWithLoss directly
  console.log("\nTest 3: model.forwardWithLoss");
  const { logits: fwdLogits } = model.forwardWithLoss(input);
  const fwdIdx = api.tensorFromArray([383], [1, 1, 1], { device: "webgpu" });
  const fwdLogit383 = fwdLogits.gather(fwdIdx, { dim: 2 });
  const fwdLogit383Data = Array.from(await fwdLogit383.cpu());
  console.log(`logits[383] from forwardWithLoss = ${fwdLogit383Data[0]?.toFixed(6)}`);

  process.exit(0);
}

main().catch(e => { console.error("ERROR:", e.message, e.stack); process.exit(1); });
