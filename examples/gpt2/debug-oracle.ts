/**
 * Compare torchlette GPT-2 outputs against PyTorch oracle values.
 */
import * as path from "node:path";
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { loadPretrainedGPT2 } from "./loader";

function arrClose(name: string, got: number[], expected: number[], atol = 0.01) {
  const maxDiff = Math.max(...got.map((v, i) => Math.abs(v - expected[i])));
  const match = maxDiff < atol;
  console.log(`  ${match ? "PASS" : "FAIL"} ${name}: maxDiff=${maxDiff.toFixed(6)}${!match ? ` (atol=${atol})` : ""}`);
  if (!match) {
    console.log(`    got:      [${got.map(v => v.toFixed(6))}]`);
    console.log(`    expected: [${expected.map(v => v.toFixed(6))}]`);
  }
}

async function main() {
  console.log("=== GPT-2 Oracle Comparison ===\n");
  await initWebGPU();

  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: true });
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.eval();

  // Oracle values from PyTorch (distilgpt2)
  const oracle = {
    hello_token: 15496,
    hello_tok_emb_first5: [-0.09042633324861526, -0.15380056202411652, 0.031470343470573425, -0.16417577862739563, -0.12917229533195496],
    hello_pos_emb_first5: [-0.01882071979343891, -0.19741860032081604, 0.004026724956929684, 0.01134685892611742, 0.0638241171836853],
    hello_combined_first5: [-0.10924705117940903, -0.35121917724609375, 0.035497069358825684, -0.15282891690731049, -0.06534817814826965],
    hello_ln1_first5: [-0.05224231258034706, -0.1534513682126999, 0.00012769662134815007, -0.07599286735057831, -0.034560833126306534],
    hello_qkv_first5: null as number[] | null, // will fill from oracle file
  };

  // Test 1: Token Embedding
  console.log("--- Test 1: Token Embedding ---");
  const inputTensor = api.tensorFromArray([oracle.hello_token], [1, 1], { device: "webgpu" });
  const tokEmb = model.wte.forward(inputTensor);
  const tokEmbData = Array.from(await tokEmb.cpu());
  arrClose("tok_emb first 5", tokEmbData.slice(0, 5), oracle.hello_tok_emb_first5);

  const tokMean = tokEmbData.reduce((a, b) => a + b, 0) / tokEmbData.length;
  const tokStd = Math.sqrt(tokEmbData.reduce((a, b) => a + (b - tokMean) ** 2, 0) / tokEmbData.length);
  console.log(`  tok_emb shape=[${tokEmb.shape}], mean=${tokMean.toFixed(6)}, std=${tokStd.toFixed(6)} (oracle: mean=-0.005223, std=0.136276)`);

  // Test 2: Position Embedding
  console.log("\n--- Test 2: Position Embedding ---");
  const posTensor = api.tensorFromArray([0], [1, 1]);
  const posEmb = model.wpe.forward(posTensor);
  const posEmbData = Array.from(await posEmb.cpu());
  arrClose("pos_emb first 5", posEmbData.slice(0, 5), oracle.hello_pos_emb_first5);

  // Test 3: Combined Embedding
  console.log("\n--- Test 3: Combined Embedding ---");
  const combined = api.add(tokEmb, posEmb);
  const combinedData = Array.from(await combined.cpu());
  arrClose("combined first 5", combinedData.slice(0, 5), oracle.hello_combined_first5);

  // Test 4: Layer Norm 1
  console.log("\n--- Test 4: LayerNorm 1 ---");
  // Input to LN1 is [1, 1, 768] (after dropout which is identity in eval mode)
  const x = combined.reshape([1, 1, 768]);
  const ln1Out = model.h[0].ln1.forward(x);
  const ln1Data = Array.from(await ln1Out.cpu());
  arrClose("ln1 first 5", ln1Data.slice(0, 5), oracle.hello_ln1_first5);

  // Test 5: c_attn (QKV projection)
  console.log("\n--- Test 5: c_attn (QKV projection) ---");
  const qkvOut = model.h[0].attn.cAttn.forward(ln1Out);
  const qkvData = Array.from(await qkvOut.cpu());
  console.log(`  qkv shape: [${qkvOut.shape}]`);
  console.log(`  qkv first 5: [${qkvData.slice(0, 5).map(v => v.toFixed(6))}]`);
  console.log(`  qkv mean=${(qkvData.reduce((a,b)=>a+b,0)/qkvData.length).toFixed(6)}`);
  console.log(`  (oracle: mean=0.011560, std=0.699651)`);

  // Test 6: Full attention (single token, seq_len=1 should be simple)
  console.log("\n--- Test 6: Full attention block ---");
  const attnOut = model.h[0].attn.forward(ln1Out);
  const attnData = Array.from(await attnOut.cpu());
  console.log(`  attn output shape: [${attnOut.shape}]`);
  const attnMean = attnData.reduce((a, b) => a + b, 0) / attnData.length;
  const attnStd = Math.sqrt(attnData.reduce((a, b) => a + (b - attnMean) ** 2, 0) / attnData.length);
  console.log(`  attn mean=${attnMean.toFixed(6)}, std=${attnStd.toFixed(6)}`);
  console.log(`  first 5: [${attnData.slice(0, 5).map(v => v.toFixed(6))}]`);

  // Test 7: Full first block
  console.log("\n--- Test 7: Full transformer block 0 ---");
  const block0Out = model.h[0].forward(x);
  const block0Data = Array.from(await block0Out.cpu());
  const block0Mean = block0Data.reduce((a, b) => a + b, 0) / block0Data.length;
  const block0Std = Math.sqrt(block0Data.reduce((a, b) => a + (b - block0Mean) ** 2, 0) / block0Data.length);
  console.log(`  block0 shape: [${block0Out.shape}], mean=${block0Mean.toFixed(6)}, std=${block0Std.toFixed(6)}`);
  console.log(`  (oracle layer 1: mean=0.280448, std=9.644617)`);

  // Test 8: Full forward pass
  console.log("\n--- Test 8: Full forward ---");
  const { logits } = model.forwardWithLoss(inputTensor);
  const logitsData = Array.from(await logits.cpu());
  const logitsMean = logitsData.reduce((a, b) => a + b, 0) / logitsData.length;
  const logitsStd = Math.sqrt(logitsData.reduce((a, b) => a + (b - logitsMean) ** 2, 0) / logitsData.length);
  console.log(`  logits shape: [${logits.shape}], mean=${logitsMean.toFixed(4)}, std=${logitsStd.toFixed(4)}`);
  console.log(`  (oracle: mean=-41.0766, std=2.7435)`);

  // Top 10
  const indexed = logitsData.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => b.v - a.v);
  console.log("  Top 10:");
  for (let i = 0; i < 10; i++) {
    console.log(`    ${i+1}. token_id=${indexed[i].i}, logit=${indexed[i].v.toFixed(4)}`);
  }
  console.log("  (oracle top: 383=-29.42, 13=-30.23, 11=-30.29)");

  console.log("\n=== Done ===");
}

main().catch(console.error);
