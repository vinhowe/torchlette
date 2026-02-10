/**
 * Debug script to isolate GPT-2 forward pass issues.
 * Tests each component to find where the output goes wrong.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { loadPretrainedGPT2 } from "./loader";

async function main() {
  console.log("=== GPT-2 Forward Pass Debug ===\n");
  await initWebGPU();

  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const api = new Torchlette("webgpu", {
    enableFusion: false,
    enableMemoryPlanning: true,
  });

  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.eval();

  // Simple input: "Hello" -> token IDs [15496]
  // Or just use a simple sequence
  const inputIds = [15496]; // "Hello"
  const seqLen = inputIds.length;

  console.log(`\nInput tokens: [${inputIds}], seqLen=${seqLen}`);

  // Step 1: Check embeddings
  console.log("\n--- Step 1: Embeddings ---");
  const inputTensor = api.tensorFromArray(inputIds, [1, seqLen], { device: "webgpu" });
  const posData = Array.from({ length: seqLen }, (_, i) => i);
  const posTensor = api.tensorFromArray(posData, [1, seqLen]);

  const tokEmb = model.wte.forward(inputTensor);
  const posEmb = model.wpe.forward(posTensor);

  const tokEmbData = await tokEmb.cpu();
  const posEmbData = await posEmb.cpu();

  const tokArr = Array.from(tokEmbData);
  const posArr = Array.from(posEmbData);

  console.log(`  tokEmb shape: ${tokEmb.shape}, first 5: [${tokArr.slice(0, 5).map(v => v.toFixed(4))}]`);
  console.log(`  tokEmb range: [${Math.min(...tokArr).toFixed(4)}, ${Math.max(...tokArr).toFixed(4)}]`);
  console.log(`  posEmb shape: ${posEmb.shape}, first 5: [${posArr.slice(0, 5).map(v => v.toFixed(4))}]`);

  // Check if embeddings look reasonable (not all zeros, not huge)
  const tokMean = tokArr.reduce((a, b) => a + b, 0) / tokArr.length;
  const tokStd = Math.sqrt(tokArr.reduce((a, b) => a + (b - tokMean) ** 2, 0) / tokArr.length);
  console.log(`  tokEmb mean=${tokMean.toFixed(6)}, std=${tokStd.toFixed(6)}`);

  // Step 2: Combined embeddings
  console.log("\n--- Step 2: Combined embeddings ---");
  const combined = api.add(tokEmb, posEmb);
  const combData = await combined.cpu();
  const combArr = Array.from(combData);
  const combMean = combArr.reduce((a, b) => a + b, 0) / combArr.length;
  const combStd = Math.sqrt(combArr.reduce((a, b) => a + (b - combMean) ** 2, 0) / combArr.length);
  console.log(`  combined shape: ${combined.shape}, mean=${combMean.toFixed(6)}, std=${combStd.toFixed(6)}`);

  // Step 3: Full forward pass
  console.log("\n--- Step 3: Full forward pass ---");
  const { logits } = model.forwardWithLoss(inputTensor);

  const logitsData = await logits.cpu();
  const logitsArr = Array.from(logitsData);

  console.log(`  logits shape: ${logits.shape}, total elements: ${logitsArr.length}`);

  // Check logits for last (only) position
  const vocabSize = 50257;
  const lastLogits = logitsArr.slice(0, vocabSize);  // Only 1 position

  const logitsMean = lastLogits.reduce((a, b) => a + b, 0) / lastLogits.length;
  const logitsStd = Math.sqrt(lastLogits.reduce((a, b) => a + (b - logitsMean) ** 2, 0) / lastLogits.length);
  const logitsMin = Math.min(...lastLogits);
  const logitsMax = Math.max(...lastLogits);

  console.log(`  logits mean=${logitsMean.toFixed(4)}, std=${logitsStd.toFixed(4)}, range=[${logitsMin.toFixed(4)}, ${logitsMax.toFixed(4)}]`);

  // Check for NaN/Inf
  const nanCount = lastLogits.filter(v => isNaN(v)).length;
  const infCount = lastLogits.filter(v => !isFinite(v)).length;
  console.log(`  NaN count: ${nanCount}, Inf count: ${infCount}`);

  // Top 10 predictions
  const indexed = lastLogits.map((v, i) => ({ v, i }));
  indexed.sort((a, b) => b.v - a.v);
  console.log("\n  Top 10 predicted tokens:");
  for (let i = 0; i < 10; i++) {
    console.log(`    ${i + 1}. token_id=${indexed[i].i}, logit=${indexed[i].v.toFixed(4)}`);
  }

  // Step 4: Test with longer sequence
  console.log("\n--- Step 4: Longer sequence ---");
  // "The" = 464
  const inputIds2 = [464]; // "The"
  const inputTensor2 = api.tensorFromArray(inputIds2, [1, 1], { device: "webgpu" });
  const { logits: logits2 } = model.forwardWithLoss(inputTensor2);
  const logitsData2 = await logits2.cpu();
  const logitsArr2 = Array.from(logitsData2);
  const lastLogits2 = logitsArr2.slice(0, vocabSize);

  const indexed2 = lastLogits2.map((v, i) => ({ v, i }));
  indexed2.sort((a, b) => b.v - a.v);
  console.log(`  Input: "The" (token 464)`);
  console.log("  Top 10 predictions:");
  for (let i = 0; i < 10; i++) {
    console.log(`    ${i + 1}. token_id=${indexed2[i].i}, logit=${indexed2[i].v.toFixed(4)}`);
  }

  // Step 5: Check attention QKV split with a simple test
  console.log("\n--- Step 5: QKV Split Test ---");
  // Create a simple [1, 1, 3*768] input and test the split
  const testSize = 12; // Small for debugging: 3 * 4 (embedDim=4)
  const testData = Array.from({ length: testSize }, (_, i) => i);
  const testTensor = api.tensorFromArray(testData, [1, 1, testSize], { device: "webgpu" });

  // Reshape to [1, 1, 3, 4]
  const split = testTensor.reshape([1, 1, 3, 4]);
  const splitData = await split.cpu();
  console.log(`  Original: [${testData}]`);
  console.log(`  Reshaped [1,1,3,4]: [${Array.from(splitData)}]`);

  // Permute to [3, 1, 1, 4]
  const perm = split.permute([2, 0, 1, 3]);
  const permData = await perm.contiguous().cpu();
  console.log(`  Permuted [3,1,1,4]: [${Array.from(permData)}]`);

  // Flatten to [3, 4]
  const flat = perm.contiguous().reshape([3, 4]);
  const flatData = await flat.cpu();
  console.log(`  Flat [3,4]: [${Array.from(flatData)}]`);

  // Gather row 0 (Q), row 1 (K), row 2 (V)
  const gIdx0 = api.tensorFromArray([0], [1, 1]).expand([1, 4]);
  const gIdx1 = api.tensorFromArray([1], [1, 1]).expand([1, 4]);
  const gIdx2 = api.tensorFromArray([2], [1, 1]).expand([1, 4]);

  // Check if expand is contiguous
  console.log(`  idx0 expand contiguous check - creating contiguous version...`);
  const gIdx0c = gIdx0.contiguous();
  const gIdx1c = gIdx1.contiguous();
  const gIdx2c = gIdx2.contiguous();

  const idx0Data = await gIdx0.cpu();
  const idx0cData = await gIdx0c.cpu();
  console.log(`  idx0 (expanded): [${Array.from(idx0Data)}]`);
  console.log(`  idx0 (contiguous): [${Array.from(idx0cData)}]`);

  // Gather with expanded (non-contiguous) indices
  const q0 = flat.gather(gIdx0, { dim: 0 });
  const k0 = flat.gather(gIdx1, { dim: 0 });
  const v0 = flat.gather(gIdx2, { dim: 0 });

  const q0Data = await q0.cpu();
  const k0Data = await k0.cpu();
  const v0Data = await v0.cpu();

  console.log(`  Q (gather expanded idx0): [${Array.from(q0Data)}] (expected: [0,1,2,3])`);
  console.log(`  K (gather expanded idx1): [${Array.from(k0Data)}] (expected: [4,5,6,7])`);
  console.log(`  V (gather expanded idx2): [${Array.from(v0Data)}] (expected: [8,9,10,11])`);

  // Gather with contiguous indices
  const q1 = flat.gather(gIdx0c, { dim: 0 });
  const k1 = flat.gather(gIdx1c, { dim: 0 });
  const v1 = flat.gather(gIdx2c, { dim: 0 });

  const q1Data = await q1.cpu();
  const k1Data = await k1.cpu();
  const v1Data = await v1.cpu();

  console.log(`  Q (gather contiguous idx0): [${Array.from(q1Data)}] (expected: [0,1,2,3])`);
  console.log(`  K (gather contiguous idx1): [${Array.from(k1Data)}] (expected: [4,5,6,7])`);
  console.log(`  V (gather contiguous idx2): [${Array.from(v1Data)}] (expected: [8,9,10,11])`);

  // Also check: device of intermediate tensors
  console.log("\n--- Step 6: Device Check ---");
  const cpuTensor = api.tensorFromArray([1, 2, 3], [3]);
  const gpuTensor = api.tensorFromArray([1, 2, 3], [3], { device: "webgpu" });
  console.log(`  CPU tensor device: ${(cpuTensor as any)._device ?? "unknown"}`);
  console.log(`  GPU tensor device: ${(gpuTensor as any)._device ?? "unknown"}`);

  // In attention, mask and scale are created without device:
  const mask = api.tensorFromArray([0, -1e9, 0, 0], [1, 1, 2, 2]);
  console.log(`  Mask tensor device: ${(mask as any)._device ?? "unknown"}`);

  console.log("\n=== Debug complete ===");
}

main().catch(console.error);
