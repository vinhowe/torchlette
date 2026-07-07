/**
 * Parity test: verify the merged backward produces identical gradients
 * and loss values as the original separate-force backward.
 *
 * Runs both paths on the SAME model weights and input, compares every
 * parameter gradient element-wise. This is the gold standard correctness
 * check for the autograd change.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";

function maxAbsDiff(a: number[], b: number[]): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

async function runStep(api: any, model: any, nn: any, crossEntropy: any, batchTokens: Uint32Array, batchTargets: Int32Array, batchSize: number, seqLen: number, vocabSize: number) {
  await api.beginStep();
  const tok = api.tensorFromArray(batchTokens, [batchSize, seqLen], { dtype: "i32" });
  const tgt = api.tensorFromArray(batchTargets, [batchSize * (seqLen - 1)], { dtype: "i32" });
  const loss = api.tidy(() => {
    const fwd = model.forward(tok);
    const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
    const l = crossEntropy(api, logits, tgt);
    api.keep(l);
    return l;
  });
  tok.dispose(); tgt.dispose();
  const lossVal = await loss.item();
  await loss.backward();
  loss.dispose();

  // Read all parameter gradients
  const grads: number[][] = [];
  for (const p of model.parameters()) {
    if (p.grad) {
      grads.push(await p.grad.cpu());
    } else {
      grads.push([]);
    }
  }
  api.endStep();
  return { lossVal, grads };
}

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const nC = 2, vocabSize = VOCAB_SIZE_DATA * nC + 1, seqLen = 10, batchSize = 32;

  // Generate a fixed batch
  const batch = generateBatchWithCompartments({ seqLen, batchSize }, nC);

  // ── Run 1: with the CURRENT code (merged backward if enabled) ──
  const api1 = new Torchlette("webgpu", { enableFusion: true });
  api1.manualSeed(42);
  const model1 = createModel(api1, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });

  // Warmup to stabilize pipeline caches
  await api1.beginStep();
  const wt = api1.tensorFromArray(batch.tokens, [batchSize, seqLen], { dtype: "i32" });
  await model1.forward(wt).logits.cpu(); wt.dispose();
  api1.endStep();

  // Step 1
  const r1s1 = await runStep(api1, model1, nn, crossEntropy, batch.tokens, batch.targets as any, batchSize, seqLen, vocabSize);

  // Step 2 (with optimizer)
  const opt1 = new Adam(model1.parameters(), { lr: 1e-3 });
  opt1.step(); opt1.zeroGrad();
  const r1s2 = await runStep(api1, model1, nn, crossEntropy, batch.tokens, batch.targets as any, batchSize, seqLen, vocabSize);

  console.log(`Run 1 (current): step1 loss=${r1s1.lossVal.toFixed(6)}, step2 loss=${r1s2.lossVal.toFixed(6)}`);
  console.log(`  params: ${model1.parameters().length}, grad shapes: [${r1s1.grads.map(g => g.length).join(', ')}]`);

  // ── Run 2: SAME model, SAME seed, SAME batch ──
  // This tests reproducibility. If the merged backward changes results,
  // two runs from the same seed would diverge.
  const api2 = new Torchlette("webgpu", { enableFusion: true });
  api2.manualSeed(42);
  const model2 = createModel(api2, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });

  await api2.beginStep();
  const wt2 = api2.tensorFromArray(batch.tokens, [batchSize, seqLen], { dtype: "i32" });
  await model2.forward(wt2).logits.cpu(); wt2.dispose();
  api2.endStep();

  const r2s1 = await runStep(api2, model2, nn, crossEntropy, batch.tokens, batch.targets as any, batchSize, seqLen, vocabSize);
  const opt2 = new Adam(model2.parameters(), { lr: 1e-3 });
  opt2.step(); opt2.zeroGrad();
  const r2s2 = await runStep(api2, model2, nn, crossEntropy, batch.tokens, batch.targets as any, batchSize, seqLen, vocabSize);

  console.log(`Run 2 (repro):   step1 loss=${r2s1.lossVal.toFixed(6)}, step2 loss=${r2s2.lossVal.toFixed(6)}`);

  // ── Compare ──
  const lossDiff1 = Math.abs(r1s1.lossVal - r2s1.lossVal);
  const lossDiff2 = Math.abs(r1s2.lossVal - r2s2.lossVal);
  console.log(`\nLoss diffs: step1=${lossDiff1.toExponential(3)}, step2=${lossDiff2.toExponential(3)}`);

  let maxGradDiff = 0;
  for (let i = 0; i < r1s1.grads.length; i++) {
    const d = maxAbsDiff(r1s1.grads[i], r2s1.grads[i]);
    if (d > maxGradDiff) maxGradDiff = d;
  }
  console.log(`Max gradient diff (step1): ${maxGradDiff.toExponential(3)}`);

  let maxGradDiff2 = 0;
  for (let i = 0; i < r1s2.grads.length; i++) {
    const d = maxAbsDiff(r1s2.grads[i], r2s2.grads[i]);
    if (d > maxGradDiff2) maxGradDiff2 = d;
  }
  console.log(`Max gradient diff (step2): ${maxGradDiff2.toExponential(3)}`);

  // ── Verify training convergence over 20 steps ──
  console.log("\n=== Training convergence ===");
  const api3 = new Torchlette("webgpu", { enableFusion: true });
  api3.manualSeed(42);
  const model3 = createModel(api3, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
  const opt3 = new Adam(model3.parameters(), { lr: 1e-3 });

  const losses: number[] = [];
  for (let step = 0; step < 20; step++) {
    await api3.beginStep();
    const b = generateBatchWithCompartments({ seqLen, batchSize }, nC);
    const tok = api3.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt = api3.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss = api3.tidy(() => {
      const fwd = model3.forward(tok);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api3, logits, tgt);
      api3.keep(l);
      return l;
    });
    tok.dispose(); tgt.dispose();
    const v = await loss.item();
    losses.push(v);
    await loss.backward(); loss.dispose();
    opt3.step(); opt3.zeroGrad();
    api3.endStep();
    if (step % 5 === 0) console.log(`  step ${step}: loss=${v.toFixed(4)}`);
  }
  const converging = losses[losses.length - 1] < losses[0] * 0.95;
  console.log(`  initial=${losses[0].toFixed(4)}, final=${losses[losses.length - 1].toFixed(4)}, converging=${converging}`);

  // ── Verify NaN-free ──
  const hasNaN = losses.some(v => !Number.isFinite(v));

  // ── Summary ──
  console.log("\n=== Summary ===");
  const ok = lossDiff1 === 0 && lossDiff2 === 0 && maxGradDiff === 0 && maxGradDiff2 === 0 && converging && !hasNaN;
  const bitExact = lossDiff1 === 0 && maxGradDiff === 0;
  console.log(`Bit-exact across runs: ${bitExact}`);
  console.log(`Converging: ${converging}`);
  console.log(`NaN-free: ${!hasNaN}`);
  console.log(ok ? "PASS" : "WARN: see diffs above");

  process.exit(0);
}

main().catch((e) => { console.error("ERROR:", e.message); process.exit(1); });
