/**
 * Sanity: does the forward pass give different logits for different inputs?
 * And does training actually update wte?
 */
import { Torchlette, initWebGPU, nn as torchNN, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel } from "../examples/toy-compartmentalization/src/lib/model";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu");

  const vocabSize = 10;
  const model = createModel(api, torchNN, {
    vocabSize, seqLen: 4, embedDim: 16, numHeads: 2, numLayers: 1, mlpDim: 32,
  });

  // Train to predict: always output token 7 regardless of input
  const optimizer = new Adam(model.parameters(), { lr: 1e-3 });
  for (let step = 0; step < 200; step++) {
    await api.beginStep();
    const inputs = api.tensorFromArray([1, 2, 3, 4], [1, 4], { dtype: "i32" });
    const targets = api.tensorFromArray([7, 7, 7], [3]);
    const fwd = model.forward(inputs);
    const logits = fwd.logits.narrow(1, 0, 3).contiguous().reshape([3, vocabSize]);
    const loss = crossEntropy(api, logits, targets);
    const lv = await loss.item();
    await loss.backward();
    optimizer.step();
    optimizer.zeroGrad();
    api.endStep();
    if (step % 25 === 0) console.log(`step ${step}: loss=${lv.toFixed(4)}`);
  }

  // Now check: forward gives correct predictions?
  await api.beginStep();
  const inputs = api.tensorFromArray([1, 2, 3, 4], [1, 4], { dtype: "i32" });
  const fwd = api.noGrad(() => model.forward(inputs));
  const logitsData = await fwd.logits.cpu();
  for (let pos = 0; pos < 4; pos++) {
    let mx = -Infinity, arg = 0;
    for (let v = 0; v < vocabSize; v++) {
      const l = logitsData[pos * vocabSize + v];
      if (l > mx) { mx = l; arg = v; }
    }
    console.log(`pos ${pos}: argmax=${arg}, max_logit=${mx.toFixed(3)}`);
  }
  api.endStep();
  process.exit(0);
}
main().catch(e => { console.error(e); process.exit(1); });
