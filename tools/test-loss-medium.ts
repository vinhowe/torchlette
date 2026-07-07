import * as path from "node:path";
import { initWebGPU, destroyWebGPU, isF16Supported } from "../src/backend/webgpu";
import { Torchlette, type Tensor } from "../src/frontend";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { GPT2_MEDIUM_CONFIG } from "../examples/gpt2/model";
import { Adam, GradScaler } from "../src/optim";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: true, enableMemoryPlanning: true, enableCheckpointSegmentation: true });
  
  const modelDir = path.join(process.cwd(), "models", "gpt2-medium");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 });
  model.train();
  
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
  const scaler = isF16Supported() ? new GradScaler(api, { initScale: 1024.0 }) : null;

  const BASE_TOKENS = [2484, 439, 314, 8996, 20218, 284, 257, 3931, 338, 1110, 30, 198];
  const SEQ_LEN = 64;  // short for speed
  const tokens: number[] = [];
  for (let i = 0; i < SEQ_LEN + 1; i++) tokens.push(BASE_TOKENS[i % BASE_TOKENS.length]);
  
  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    return api.autocast(() => {
      const result = model.forwardWithLoss(input, target, { useCheckpoint: true });
      if (!result.loss) throw new Error("Loss is null");
      return result.loss;
    });
  });
  
  for (let step = 0; step < 4; step++) {
    if (scaler) await scaler.resolveDeferred();
    await api.beginStep();
    const input = api.tensorFromArray(tokens.slice(0, -1), [1, SEQ_LEN], { device: "webgpu" });
    const target = api.tensorFromArray(tokens.slice(1), [1, SEQ_LEN], { device: "webgpu" });
    
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
