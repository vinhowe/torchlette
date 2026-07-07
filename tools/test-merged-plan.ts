/**
 * Minimal test: forward + backward as ONE merged plan (no pre-force).
 * Tests whether executePlanOptimized handles the merged plan correctly.
 */
import { Torchlette, initWebGPU, nn } from "../src/index";
import { crossEntropy } from "../src/nn/functional";

async function main() {
  await initWebGPU();

  // Test 1: simple MLP (no RoPE, no attention) — should work
  console.log("=== Test 1: Simple MLP ===");
  try {
    const api = new Torchlette("webgpu", { enableFusion: true });
    const W = api.tensorFromArray([0.1, -0.2, 0.3, 0.05, 0.1, -0.1], [2, 3]).requires_grad_(true);
    const b = api.tensorFromArray([0, 0, 0], [3]).requires_grad_(true);
    const X = api.tensorFromArray([1, 0, 0, 1], [2, 2]);
    const T = api.tensorFromArray([0, 1], [2], { dtype: "i32" });

    // Build forward + backward as one lazy graph (no intermediate force)
    const logits = api.add(api.matmul(X, W), b);
    const loss = crossEntropy(api, logits, T, { reduction: "mean" });
    await loss.backward(); // forces EVERYTHING in one plan
    console.log("  MLP: PASS (backward completed)");
    const grad = await W.grad!.cpu();
    console.log("  grad:", grad.slice(0, 4));
  } catch (e: any) {
    console.log("  MLP: FAIL:", e.message);
  }

  // Test 2: transformer model with RoPE (the actual failure case)
  console.log("\n=== Test 2: MESS3 Transformer with RoPE ===");
  try {
    const { createModel, MESS3_CONFIG } = await import("../examples/toy-compartmentalization/src/lib/model");
    const { generateBatch, generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } = await import("../examples/toy-compartmentalization/src/lib/data");

    const api2 = new Torchlette("webgpu", { enableFusion: true });
    setTransitionMatrices(0.765);

    const vocabSize = VOCAB_SIZE_DATA * 2 + 1;
    const seqLen = 10, batchSize = 64;
    const model = createModel(api2, nn, {
      ...MESS3_CONFIG,
      seqLen, vocabSize,
      posEncoding: "rope",
    });

    // Force model creation (warmup)
    await api2.beginStep();
    const b1 = generateBatch({ seqLen, batchSize });
    const tok1 = api2.tensorFromArray(b1.tokens, [batchSize, seqLen], { dtype: "i32" });
    const fwd1 = model.forward(tok1);
    await fwd1.logits.cpu();
    api2.endStep();
    console.log("  Warmup done");

    // Step 1: merged forward + backward (no pre-force)
    await api2.beginStep();
    const b2 = generateBatchWithCompartments({ seqLen, batchSize }, 2);
    const tok2 = api2.tensorFromArray(b2.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt2 = api2.tensorFromArray(b2.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });

    const loss2 = api2.tidy(() => {
      const fwd = model.forward(tok2);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous()
        .reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api2, logits, tgt2);
      api2.keep(l);
      return l;
    });

    // This is where the pre-force normally happens. We skip it.
    // backward() forces the merged fwd+bwd plan.
    tok2.dispose(); tgt2.dispose();
    await loss2.backward();
    console.log("  Transformer: PASS (backward completed)");

    api2.endStep();
  } catch (e: any) {
    console.log("  Transformer: FAIL:", e.message);
    console.log("  Stack:", e.stack?.split("\n").slice(0, 5).join("\n"));
  }

  // Test 3: transformer WITHOUT RoPE (learned pos encoding)
  console.log("\n=== Test 3: Transformer without RoPE ===");
  try {
    const { createModel, MESS3_CONFIG } = await import("../examples/toy-compartmentalization/src/lib/model");
    const { generateBatch, generateBatchWithCompartments, setTransitionMatrices, VOCAB_SIZE_DATA } = await import("../examples/toy-compartmentalization/src/lib/data");

    const api3 = new Torchlette("webgpu", { enableFusion: true });
    setTransitionMatrices(0.765);
    const vocabSize = VOCAB_SIZE_DATA * 2 + 1;
    const seqLen = 10, batchSize = 4;
    const model = createModel(api3, nn, {
      ...MESS3_CONFIG,
      seqLen, vocabSize,
      posEncoding: "learned",
    });

    await api3.beginStep();
    const b = generateBatch({ seqLen, batchSize });
    const tok = api3.tensorFromArray(b.tokens, [batchSize, seqLen], { dtype: "i32" });
    const fwd = model.forward(tok);
    await fwd.logits.cpu();
    api3.endStep();
    console.log("  Warmup done");

    await api3.beginStep();
    const b2 = generateBatch({ seqLen, batchSize });
    const tok2 = api3.tensorFromArray(b2.tokens, [batchSize, seqLen], { dtype: "i32" });
    const tgt2 = api3.tensorFromArray(b2.targets, [batchSize * (seqLen - 1)], { dtype: "i32" });
    const loss = api3.tidy(() => {
      const fwd = model.forward(tok2);
      const logits = fwd.logits.narrow(1, 0, seqLen - 1).contiguous()
        .reshape([batchSize * (seqLen - 1), vocabSize]);
      const l = crossEntropy(api3, logits, tgt2);
      api3.keep(l);
      return l;
    });
    await loss.backward();
    console.log("  Transformer (learned): PASS");
    api3.endStep();
  } catch (e: any) {
    console.log("  Transformer (learned): FAIL:", e.message);
  }

  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
