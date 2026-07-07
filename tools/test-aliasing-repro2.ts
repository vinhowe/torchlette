/**
 * Repro: optimizer.step() outside beginStep/endStep, then next step.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const nC = 2, vocabSize = VOCAB_SIZE_DATA * nC + 1, seqLen = 10, batchSize = 32;
  const batch = generateBatchWithCompartments({ seqLen, batchSize }, nC);

  for (const fusion of [false, true]) {
    console.log(`\n=== fusion=${fusion} ===`);
    const api = new Torchlette("webgpu", { enableFusion: fusion });
    api.manualSeed(42);
    const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });

    // Warmup
    await api.beginStep();
    const wt = api.tensorFromArray(batch.tokens, [batchSize, seqLen], { dtype: "i32" });
    await model.forward(wt).logits.cpu(); wt.dispose();
    api.endStep();

    // Step 1: normal
    await api.beginStep();
    const tok1 = api.tensorFromArray(batch.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt1 = api.tensorFromArray(batch.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss1 = api.tidy(() => {
      const fwd = model.forward(tok1);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt1); api.keep(l); return l;
    });
    tok1.dispose(); tgt1.dispose();
    const v1 = await loss1.item();
    await loss1.backward(); loss1.dispose();
    api.endStep();
    console.log(`  step 1: loss=${v1.toFixed(4)}`);

    // Optimizer OUTSIDE step scope (like the parity test)
    const opt = new Adam(model.parameters(), { lr: 1e-3 });
    opt.step(); opt.zeroGrad();

    // Step 2: this is where aliasing might happen
    await api.beginStep();
    const tok2 = api.tensorFromArray(batch.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt2 = api.tensorFromArray(batch.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss2 = api.tidy(() => {
      const fwd = model.forward(tok2);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt2); api.keep(l); return l;
    });
    tok2.dispose(); tgt2.dispose();
    const v2 = await loss2.item();
    await loss2.backward(); loss2.dispose();
    opt.step(); opt.zeroGrad();
    api.endStep();
    console.log(`  step 2: loss=${v2.toFixed(4)} ${Number.isFinite(v2) ? "OK" : "BAD"}`);
  }
  process.exit(0);
}
main().catch((e) => { console.error(e.message); process.exit(1); });
