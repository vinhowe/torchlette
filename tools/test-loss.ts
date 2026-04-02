import * as path from "node:path";
import { initWebGPU, destroyWebGPU, isF16Supported } from "../src/backend/webgpu";
import { Torchlette, type Tensor } from "../src/frontend/torchlette";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { Adam, GradScaler } from "../src/optim";
import { crossEntropy } from "../src/nn";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: true, enableMemoryPlanning: true, enableCheckpointSegmentation: true });
  
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 });
  model.train();
  
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
  const useAMP = isF16Supported();
  const scaler = useAMP ? new GradScaler(api, { initScale: 1024.0 }) : null;

  const BASE_TOKENS = [2484, 439, 314, 8996, 20218, 284, 257, 3931, 338, 1110, 30, 198, 1986, 280, 1242, 517, 8855, 290, 517, 29815, 13, 198, 49, 619, 9985, 466, 13508, 262, 38482, 31007, 286, 1737];
  const SEQ_LEN = 512;
  const tokens: number[] = [];
  for (let i = 0; i < SEQ_LEN + 1; i++) tokens.push(BASE_TOKENS[i % BASE_TOKENS.length]);
  const inputTokens = tokens.slice(0, -1);
  const targetTokens = tokens.slice(1);
  
  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    return api.autocast(() => {
      const result = model.forwardWithLoss(input, target, { useCheckpoint: true });
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });
  
  for (let step = 0; step < 5; step++) {
    if (scaler) await scaler.resolveDeferred();
    await api.beginStep();
    const input = api.tensorFromArray(inputTokens, [1, SEQ_LEN], { device: "webgpu" });
    const target = api.tensorFromArray(targetTokens, [1, SEQ_LEN], { device: "webgpu" });
    
    const loss = compiledForward(input, target);
    const lossValue = await loss.item();
    
    if (scaler) {
      const scaledLoss = scaler.scale(loss);
      await scaledLoss.backward();
      scaler.unscale_(optimizer);
      scaler.step(optimizer);
      scaler.update();
    } else {
      await loss.backward();
      optimizer.step();
    }
    optimizer.zeroGrad();
    
    loss.dispose();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
    
    console.log(`Step ${step}: loss=${lossValue.toFixed(4)}, scale=${scaler?.getScale().toFixed(0) ?? "N/A"}`);
  }
  
  destroyWebGPU();
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
