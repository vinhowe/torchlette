/**
 * Exact reproduction of the profiler path that fails.
 * Binary search: add one thing at a time until it breaks.
 */
import { Torchlette, initWebGPU, nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import {
  generateBatchWithCompartments,
  setTransitionMatrices, VOCAB_SIZE_DATA,
} from "../examples/toy-compartmentalization/src/lib/data";

async function test(label: string, fn: () => Promise<void>) {
  try {
    await fn();
    console.log(`  ${label}: PASS`);
  } catch (e: any) {
    console.log(`  ${label}: FAIL — ${e.message.split("\n")[0]}`);
  }
}

async function main() {
  await initWebGPU();
  setTransitionMatrices(0.765);
  const vocabSize = VOCAB_SIZE_DATA * 2 + 1;
  const seqLen = 10, batchSize = 64;

  // Test A: no warmup, no Adam, no step management
  await test("A: bare forward+backward", async () => {
    const api = new Torchlette("webgpu", { enableFusion: true });
    const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
    const b = generateBatchWithCompartments({ seqLen, batchSize }, 2);
    const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss = api.tidy(() => {
      const fwd = model.forward(tok);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
    });
    tok.dispose(); tgt.dispose();
    await loss.backward();
  });

  // Test B: with warmup (forward-only force before the merged plan)
  await test("B: warmup then merged", async () => {
    const api = new Torchlette("webgpu", { enableFusion: true });
    const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
    // Warmup: force model weights
    const bw = generateBatchWithCompartments({ seqLen, batchSize: 2 }, 2);
    const tw = api.tensorFromArray(bw.tokens, [2, seqLen], { dtype: "i32" });
    await model.forward(tw).logits.cpu(); tw.dispose();
    // Merged step
    const b = generateBatchWithCompartments({ seqLen, batchSize }, 2);
    const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss = api.tidy(() => {
      const fwd = model.forward(tok);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
    });
    tok.dispose(); tgt.dispose();
    await loss.backward();
  });

  // Test C: with Adam created (but not stepped)
  await test("C: Adam created, warmup, merged", async () => {
    const api = new Torchlette("webgpu", { enableFusion: true });
    const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
    const optimizer = new Adam(model.parameters(), { lr: 1e-2 });
    const bw = generateBatchWithCompartments({ seqLen, batchSize: 2 }, 2);
    const tw = api.tensorFromArray(bw.tokens, [2, seqLen], { dtype: "i32" });
    await model.forward(tw).logits.cpu(); tw.dispose();
    const b = generateBatchWithCompartments({ seqLen, batchSize }, 2);
    const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss = api.tidy(() => {
      const fwd = model.forward(tok);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
    });
    tok.dispose(); tgt.dispose();
    await loss.backward();
  });

  // Test D: with beginStep/endStep in warmup
  await test("D: step-managed warmup, merged", async () => {
    const api = new Torchlette("webgpu", { enableFusion: true });
    const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
    await api.beginStep();
    const bw = generateBatchWithCompartments({ seqLen, batchSize: 2 }, 2);
    const tw = api.tensorFromArray(bw.tokens, [2, seqLen], { dtype: "i32" });
    await model.forward(tw).logits.cpu(); tw.dispose();
    api.endStep();
    // Merged step (no beginStep)
    const b = generateBatchWithCompartments({ seqLen, batchSize }, 2);
    const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss = api.tidy(() => {
      const fwd = model.forward(tok);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
    });
    tok.dispose(); tgt.dispose();
    await loss.backward();
  });

  // Test E: full warmup with loss.item + backward + Adam step
  await test("E: full warmup cycle, then merged", async () => {
    const api = new Torchlette("webgpu", { enableFusion: true });
    const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
    const optimizer = new Adam(model.parameters(), { lr: 1e-2 });
    // Full warmup step
    await api.beginStep();
    const bw = generateBatchWithCompartments({ seqLen, batchSize }, 2);
    const tw = api.tensorFromArray(bw.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgtw = api.tensorFromArray(bw.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const lw = api.tidy(() => {
      const fwd = model.forward(tw);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgtw); api.keep(l); return l;
    });
    tw.dispose(); tgtw.dispose();
    await lw.item();
    await lw.backward(); lw.dispose();
    optimizer.step(); optimizer.zeroGrad();
    api.endStep();
    // Merged step
    const b = generateBatchWithCompartments({ seqLen, batchSize }, 2);
    const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss = api.tidy(() => {
      const fwd = model.forward(tok);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
    });
    tok.dispose(); tgt.dispose();
    await loss.backward();
  });

  // Test F: 3 full warmup cycles, then merged
  await test("F: 3 warmup cycles, then merged", async () => {
    const api = new Torchlette("webgpu", { enableFusion: true });
    const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
    const optimizer = new Adam(model.parameters(), { lr: 1e-2 });
    for (let i = 0; i < 3; i++) {
      await api.beginStep();
      const bw = generateBatchWithCompartments({ seqLen, batchSize }, 2);
      const tw = api.tensorFromArray(bw.tokens, [batchSize, seqLen], { dtype: "i32" });
      const tgtw = api.tensorFromArray(bw.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
      const lw = api.tidy(() => {
        const fwd = model.forward(tw);
        const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
        const l = crossEntropy(api, logits, tgtw); api.keep(l); return l;
      });
      tw.dispose(); tgtw.dispose();
      await lw.item();
      await lw.backward(); lw.dispose();
      optimizer.step(); optimizer.zeroGrad();
      api.endStep();
    }
    // Merged step
    const b = generateBatchWithCompartments({ seqLen, batchSize }, 2);
    const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss = api.tidy(() => {
      const fwd = model.forward(tok);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
    });
    tok.dispose(); tgt.dispose();
    await loss.backward();
  });

  // Test G: same as C but with rewritePlan/DSL disabled via env
  await test("G: Adam, warmup, NO rewrite rules", async () => {
    process.env.TORCHLETTE_NO_REWRITE = "1";
    const api = new Torchlette("webgpu", { enableFusion: true });
    const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
    const optimizer = new Adam(model.parameters(), { lr: 1e-2 });
    const bw = generateBatchWithCompartments({ seqLen, batchSize: 2 }, 2);
    const tw = api.tensorFromArray(bw.tokens, [2, seqLen], { dtype: "i32" });
    await model.forward(tw).logits.cpu(); tw.dispose();
    const b = generateBatchWithCompartments({ seqLen, batchSize }, 2);
    const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss = api.tidy(() => {
      const fwd = model.forward(tok);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
    });
    tok.dispose(); tgt.dispose();
    await loss.backward();
    delete process.env.TORCHLETTE_NO_REWRITE;
  });

  // Test H: same as C but with enableFusion: false
  await test("H: Adam, warmup, NO fusion", async () => {
    const api = new Torchlette("webgpu", { enableFusion: false });
    const model = createModel(api, nn, { ...MESS3_CONFIG, seqLen, vocabSize, posEncoding: "rope" });
    const optimizer = new Adam(model.parameters(), { lr: 1e-2 });
    const bw = generateBatchWithCompartments({ seqLen, batchSize: 2 }, 2);
    const tw = api.tensorFromArray(bw.tokens, [2, seqLen], { dtype: "i32" });
    await model.forward(tw).logits.cpu(); tw.dispose();
    const b = generateBatchWithCompartments({ seqLen, batchSize }, 2);
    const tok = api.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt = api.tensorFromArray(b.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss = api.tidy(() => {
      const fwd = model.forward(tok);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous().reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api, logits, tgt); api.keep(l); return l;
    });
    tok.dispose(); tgt.dispose();
    await loss.backward();
  });

  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
