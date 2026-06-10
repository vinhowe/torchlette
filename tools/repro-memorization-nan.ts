/**
 * Minimal repro of the gpt2-memorization NaN: tiny GPT-2 trained with
 * Adam.stepAsync() (per-param elementwise path, NO beginStep/markStep),
 * mirroring test/gpt2-memorization.spec.ts. NaN by ~step 50 when broken.
 *
 * Env: STEPS (default 80), STEPMODE=step|stepAsync (default stepAsync)
 */
import { GPT2, type GPT2Config } from "../examples/gpt2/model";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim";

const STEPS = parseInt(process.env.STEPS ?? "80", 10);
const STEPMODE = process.env.STEPMODE ?? "stepAsync";

async function main() {
  if (!(await initWebGPU())) {
    console.error("WebGPU init failed");
    process.exit(1);
  }
  const config: GPT2Config = {
    vocabSize: 30,
    blockSize: 32,
    numLayers: 2,
    numHeads: 4,
    embedDim: 128,
    dropoutRate: 0.0,
  };
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: true,
  });
  const model = new GPT2(api, config, { device: "webgpu" });
  model.train();
  const optimizer = new Adam(model.parameters(), { lr: 0.01 }, api);

  const batch = 5;
  const seqLen = 30;
  // Deterministic pseudo-random tokens in [1, vocabSize)
  let x = 1234;
  const tok = () => {
    x = (x * 1103515245 + 12345) % 2147483648;
    return 1 + (x % (config.vocabSize - 1));
  };
  const inputData = Array.from({ length: batch * seqLen }, tok);
  const targetData = Array.from({ length: batch * seqLen }, tok);
  const inputTensor = api.tensorFromArray(inputData, [batch, seqLen], {
    device: "webgpu",
  });
  const targetTensor = api.tensorFromArray(targetData, [batch, seqLen], {
    device: "webgpu",
  });

  let sawNaN = -1;
  for (let step = 0; step < STEPS; step++) {
    const { loss } = model.forwardWithLoss(inputTensor, targetTensor);
    if (!loss) throw new Error("Loss is null");
    const lossValue = await loss.item();
    await loss.backward();
    if (STEPMODE === "stepAsync") {
      await optimizer.stepAsync();
    } else {
      optimizer.step();
    }
    optimizer.zeroGrad();
    loss.dispose();
    if ((step + 1) % 10 === 0 || step === 0) {
      console.log(`Step ${step + 1}: loss = ${lossValue.toFixed(4)}`);
    }
    if (Number.isNaN(lossValue) && sawNaN < 0) {
      sawNaN = step + 1;
      console.log(`FIRST NaN at step ${sawNaN}`);
      break;
    }
  }
  console.log(sawNaN < 0 ? "RESULT: CLEAN" : `RESULT: NaN at step ${sawNaN}`);
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
