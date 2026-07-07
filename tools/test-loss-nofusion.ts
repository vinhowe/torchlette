import * as path from "node:path";
import { initWebGPU, destroyWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { Adam } from "../src/optim";

async function main() {
  await initWebGPU();
  // Disable ALL optimizations
  const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: false, enableCheckpointSegmentation: false });
  
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 });
  model.train();
  
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);

  const BASE_TOKENS = [2484, 439, 314, 8996, 20218, 284, 257, 3931, 338, 1110, 30, 198];
  const SEQ_LEN = 64;
  const tokens: number[] = [];
  for (let i = 0; i < SEQ_LEN + 1; i++) tokens.push(BASE_TOKENS[i % BASE_TOKENS.length]);
  const inputTokens = tokens.slice(0, -1);
  const targetTokens = tokens.slice(1);
  
  for (let step = 0; step < 5; step++) {
    const input = api.tensorFromArray(inputTokens, [1, SEQ_LEN], { device: "webgpu" });
    const target = api.tensorFromArray(targetTokens, [1, SEQ_LEN], { device: "webgpu" });
    
    const result = model.forwardWithLoss(input, target);
    if (!result.loss) throw new Error("No loss");
    const lossValue = await result.loss.item();
    
    await result.loss.backward();
    optimizer.step();
    optimizer.zeroGrad();
    
    result.loss.dispose();
    input.dispose();
    target.dispose();
    await api.markStep();
    
    console.log(`Step ${step}: loss=${lossValue.toFixed(4)}`);
  }
  
  destroyWebGPU();
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
