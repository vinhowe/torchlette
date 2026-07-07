/**
 * Minimal repro of the buffer aliasing bug in merged backward plans.
 * Enables Dawn validation to catch the exact conflicting buffer.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const nC = 2, vocabSize = VOCAB_SIZE_DATA * nC + 1, seqLen = 10, batchSize = 32;

  // Test matrix: which features trigger the aliasing?
  const configs = [
    { fusion: false, label: "no fusion" },
    { fusion: true, label: "with fusion" },
  ];

  for (const cfg of configs) {
    console.log(`\n=== ${cfg.label} ===`);
    const api = new Torchlette("webgpu", { enableFusion: cfg.fusion });
    api.manualSeed(42);
    const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });

    // Warmup
    await api.beginStep();
    const wt = api.tensorFromArray(generateBatchWithCompartments({ seqLen, batchSize: 2 }, nC).tokens, [2, seqLen], { dtype: "i32" });
    await model.forward(wt).logits.cpu(); wt.dispose();
    api.endStep();

    // 3 training steps
    const opt = new Adam(model.parameters(), { lr: 1e-3 });
    for (let step = 0; step < 3; step++) {
      try {
        await api.beginStep();
        const b = generateBatchWithCompartments({ seqLen, batchSize }, nC);
        const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
        const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
        const loss = api.tidy(() => {
          const fwd = model.forward(tok);
          const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
          const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
        });
        tok.dispose(); tgt.dispose();
        const v = await loss.item();
        await loss.backward(); loss.dispose();
        opt.step(); opt.zeroGrad();
        api.endStep();
        console.log(`  step ${step}: loss=${v.toFixed(4)} OK`);
      } catch (e: any) {
        console.log(`  step ${step}: ERROR ${e.message.split("\n")[0]}`);
        break;
      }
    }
  }

  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
