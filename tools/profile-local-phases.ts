/**
 * Phase breakdown for the merged backward path.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: true });
  setTransitionMatrices(0.765);
  const nC = 2, vocabSize = VOCAB_SIZE_DATA * nC + 1, seqLen = 10, batchSize = 64;
  const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
  const optimizer = new Adam(model.parameters(), { lr: 1e-2 });

  // Warmup
  for (let i = 0; i < 3; i++) {
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
    await loss.item();
    await loss.backward(); loss.dispose();
    optimizer.step(); optimizer.zeroGrad();
    api.endStep();
  }
  console.log("Warmup done.\n");

  const N = 30;
  const phases: Record<string, number> = { begin: 0, data: 0, graph: 0, item: 0, bwd: 0, opt: 0, end: 0 };
  for (let step = 0; step < N; step++) {
    const t0 = performance.now();
    await api.beginStep();
    const t1 = performance.now();
    const b = generateBatchWithCompartments({ seqLen, batchSize }, nC);
    const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const t2 = performance.now();
    const loss = api.tidy(() => {
      const fwd = model.forward(tok);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
    });
    tok.dispose(); tgt.dispose();
    const t3 = performance.now();
    if (step % 10 === 0) await loss.item();
    const t4 = performance.now();
    await loss.backward(); loss.dispose();
    const t5 = performance.now();
    optimizer.step(); optimizer.zeroGrad();
    const t6 = performance.now();
    api.endStep();
    const t7 = performance.now();
    phases.begin += t1 - t0;
    phases.data += t2 - t1;
    phases.graph += t3 - t2;
    phases.item += t4 - t3;
    phases.bwd += t5 - t4;
    phases.opt += t6 - t5;
    phases.end += t7 - t6;
  }
  const total = Object.values(phases).reduce((a, b) => a + b, 0);
  console.log(`${N} steps in ${(total / 1000).toFixed(2)}s (${(total / N).toFixed(1)}ms/step)`);
  for (const [k, v] of Object.entries(phases)) {
    console.log(`  ${k.padEnd(6)} ${(v / N).toFixed(1)}ms  (${((v / total) * 100).toFixed(0)}%)`);
  }
  process.exit(0);
}
main().catch((e) => { console.error(e); process.exit(1); });
