/**
 * Minimal repro of the narrow "Input not ready" bug.
 * Test C from bisection: Adam created + warmup + merged backward.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const vocabSize = VOCAB_SIZE_DATA * 2 + 1, seqLen = 10, batchSize = 64;
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
  const optimizer = new Adam(model.parameters(), { lr: 1e-2 });

  // Warmup (forward only)
  const bw = generateBatchWithCompartments({ seqLen, batchSize: 2 }, 2);
  const tw = api.tensorFromArray(bw.tokens, [2, seqLen], { dtype: "i32" });
  await model.forward(tw).logits.cpu(); tw.dispose();

  // Merged forward+backward (no pre-force in autograd.ts)
  const b = generateBatchWithCompartments({ seqLen, batchSize }, 2);
  const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
  const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
  const loss = api.tidy(() => {
    const fwd = model.forward(tok);
    const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
    const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
  });
  tok.dispose(); tgt.dispose();
  try {
    await loss.backward();
    console.log("PASS");
  } catch(e: any) {
    console.log("FAIL:", e.message);
  }
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
