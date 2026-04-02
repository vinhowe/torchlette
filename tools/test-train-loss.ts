/**
 * Minimal training test to verify loss convergence without profiler overhead.
 */
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { crossEntropy } from "../src/nn";
import { Adam } from "../src/optim";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });
  const model = await loadPretrainedGPT2(
    api,
    "./models/distilgpt2",
    { dropoutRate: 0 },
    { device: "webgpu" },
  );
  const optimizer = new Adam(model.parameters(), { lr: 1e-4 });

  // Hardcoded GPT-2 token IDs for "To be or not to be, that is the question."
  const tokens = [2514, 307, 393, 407, 284, 307, 11, 326, 318, 262, 1808, 13];
  const inputTokens = tokens.slice(0, -1);
  const targetTokens = tokens.slice(1);
  const input = api.tensorFromArray(inputTokens, [1, inputTokens.length]);
  const target = api.tensorFromArray(targetTokens, [1, targetTokens.length]);

  for (let step = 0; step < 5; step++) {
    optimizer.zeroGrad();
    const logits = model.forward(input);
    const [B, S, V] = logits.shape;
    const loss = crossEntropy(
      logits.reshape([B * S, V]),
      target.reshape([B * S]),
    );
    const lossVal = await loss.item();
    console.log(`Step ${step}: loss=${lossVal.toFixed(4)}`);
    loss.backward();
    optimizer.step();
    api.markStep();
  }

  process.exit(0);
}
main();
