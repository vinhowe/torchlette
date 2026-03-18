/**
 * Debug script: instruments GPUBuffer.destroy() to find which calls happen
 * during the shared encoder scope.
 */

import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { initWebGPU } from "../src/backend/webgpu";
import { getWebGPUDevice } from "../src/backend/webgpu/gpu-context";
import { sharedEncoderActive } from "../src/backend/webgpu/webgpu-state";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim";

async function main() {
  await initWebGPU();

  // Monkey-patch GPUBuffer.destroy to detect calls during shared encoder scope
  const gpu = getWebGPUDevice()!;
  const dummyBuf = gpu.device.createBuffer({ size: 16, usage: 0x80 }); // STORAGE
  const proto = Object.getPrototypeOf(dummyBuf);
  const origDestroy = proto.destroy;
  let destroyCount = 0;
  proto.destroy = function (this: any) {
    if (sharedEncoderActive) {
      destroyCount++;
      console.log(`\n!!! GPUBuffer.destroy() called during shared encoder !!!`);
      console.log(`  Buffer size: ${this.size}`);
      console.log(
        `  Stack:\n${new Error().stack?.split("\n").slice(1, 8).join("\n")}`,
      );
    }
    return origDestroy.call(this);
  };
  dummyBuf.destroy();

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

  // Use synthetic random tokens
  const { createRandomTokens } = await import("../examples/gpt2/data");
  const seqLen = 32;
  const tokens = createRandomTokens(seqLen + 1, 50257);
  const inputData = Array.from(tokens.slice(0, seqLen));
  const targetData = Array.from(tokens.slice(1, seqLen + 1));
  const input = api.tensorFromArray(inputData, [1, seqLen]);
  const target = api.tensorFromArray(targetData, [1, seqLen]);

  for (let step = 0; step < 3; step++) {
    destroyCount = 0;
    optimizer.zeroGrad();
    const { loss } = model.forwardWithLoss(input, target);
    if (!loss) throw new Error("No loss");
    const lossVal = await loss.item();
    console.log(
      `Step ${step}: loss=${lossVal.toFixed(4)}, destroys_during_enc=${destroyCount}`,
    );
    await loss.backward();
    optimizer.step();
    await api.markStep();
  }

  process.exit(0);
}
main();
