/**
 * Debug attention forward pass step-by-step, forcing each op.
 */
import * as path from "node:path";
import { Torchlette } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { loadPretrainedGPT2 } from "./loader";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: true });
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.eval();

  const input = api.tensorFromArray([15496], [1, 1], { device: "webgpu" });
  const pos = api.tensorFromArray([0], [1, 1]);
  const tokEmb = model.wte.forward(input);
  const posEmb = model.wpe.forward(pos);
  let x = api.add(tokEmb, posEmb);

  // Make sure embedding is materialized
  const xData = await x.cpu();
  console.log(`Embedding OK, first 3: [${Array.from(xData).slice(0,3).map(v=>v.toFixed(4))}]`);

  // Reshape for block input
  x = x.reshape([1, 1, 768]);

  // LN1
  const ln1Out = model.h[0].ln1.forward(x);
  const ln1Data = await ln1Out.cpu();
  console.log(`LN1 OK, first 3: [${Array.from(ln1Data).slice(0,3).map(v=>v.toFixed(4))}]`);

  // Now reproduce attention forward step-by-step
  const attn = model.h[0].attn;
  const batch = 1, seqLen = 1, embedDim = 768, numHeads = 12, headDim = 64;

  // Step 1: QKV projection
  console.log("Step 1: QKV projection...");
  const qkv = attn.cAttn.forward(ln1Out);
  const qkvData = await qkv.cpu();
  console.log(`  QKV OK, shape=${qkv.shape}`);

  // Step 2: reshape to [batch, seqLen, 3, embedDim]
  console.log("Step 2: QKV reshape...");
  const qkvSplit3 = qkv.reshape([batch, seqLen, 3, embedDim]);
  console.log(`  shape=${qkvSplit3.shape}`);

  // Step 3: permute to [3, batch, seqLen, embedDim]
  console.log("Step 3: QKV permute...");
  const qkvTransposed = qkvSplit3.permute([2, 0, 1, 3]);
  console.log(`  shape=${qkvTransposed.shape}`);

  // Step 4: contiguous + flatten
  console.log("Step 4: contiguous + flatten...");
  const flatTotal = batch * seqLen * embedDim;
  const qkvFlattened = qkvTransposed.contiguous().reshape([3, flatTotal]);
  const flatData = await qkvFlattened.cpu();
  console.log(`  Flattened OK, shape=${qkvFlattened.shape}`);

  // Step 5: create indices
  console.log("Step 5: gather indices...");
  const idx0 = api.tensorFromArray([0], [1, 1]).expand([1, flatTotal]);
  const idx1 = api.tensorFromArray([1], [1, 1]).expand([1, flatTotal]);
  const idx2 = api.tensorFromArray([2], [1, 1]).expand([1, flatTotal]);

  // Make indices contiguous (matching what Embedding does)
  console.log("Step 5b: make indices contiguous...");
  const idx0c = idx0.contiguous();
  const idx1c = idx1.contiguous();
  const idx2c = idx2.contiguous();
  const idx0Data = await idx0c.cpu();
  console.log(`  idx0 OK, first 3: [${Array.from(idx0Data).slice(0,3)}]`);

  // Step 6: gather Q, K, V
  console.log("Step 6a: gather Q...");
  const qFlat = qkvFlattened.gather(idx0c, { dim: 0 });
  const qFlatData = await qFlat.cpu();
  console.log(`  Q OK, first 3: [${Array.from(qFlatData).slice(0,3).map(v=>v.toFixed(4))}]`);

  console.log("Step 6b: gather K...");
  const kFlat = qkvFlattened.gather(idx1c, { dim: 0 });
  const kFlatData = await kFlat.cpu();
  console.log(`  K OK, first 3: [${Array.from(kFlatData).slice(0,3).map(v=>v.toFixed(4))}]`);

  console.log("Step 6c: gather V...");
  const vFlat = qkvFlattened.gather(idx2c, { dim: 0 });
  const vFlatData = await vFlat.cpu();
  console.log(`  V OK, first 3: [${Array.from(vFlatData).slice(0,3).map(v=>v.toFixed(4))}]`);

  // Step 7: reshape Q to multi-head
  console.log("Step 7: reshape Q to heads...");
  const q = qFlat.contiguous().reshape([batch, seqLen, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
  const k = kFlat.contiguous().reshape([batch, seqLen, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
  const v = vFlat.contiguous().reshape([batch, seqLen, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
  const qData = await q.cpu();
  console.log(`  Q heads OK, shape=${q.shape}`);

  // Step 8: k transpose
  console.log("Step 8: k transpose...");
  const kT = k.transpose({ dim0: 2, dim1: 3 }).contiguous();
  const kTData = await kT.cpu();
  console.log(`  kT OK, shape=${kT.shape}`);

  // Step 9: q @ kT
  console.log("Step 9: q @ kT...");
  const scores = q.matmul(kT);
  const scoresData = await scores.cpu();
  console.log(`  scores OK, shape=${scores.shape}, val=${Array.from(scoresData).slice(0,3).map(v=>v.toFixed(4))}`);

  // Step 10: scale
  console.log("Step 10: scale...");
  const scale = 1.0 / Math.sqrt(headDim);
  const scaleTensor = api.tensorFromArray([scale], []);
  const scaledScores = api.mul(scores, scaleTensor);
  const scaledData = await scaledScores.cpu();
  console.log(`  scaled OK, shape=${scaledScores.shape}, val=${Array.from(scaledData).slice(0,3).map(v=>v.toFixed(4))}`);

  // Step 11: mask
  console.log("Step 11: causal mask...");
  const maskData = new Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      maskData[i * seqLen + j] = j <= i ? 0 : -1e9;
    }
  }
  const mask = api.tensorFromArray(maskData, [1, 1, seqLen, seqLen]);
  const maskedScores = api.add(scaledScores, mask);
  const maskedData = await maskedScores.cpu();
  console.log(`  masked OK, shape=${maskedScores.shape}, val=${Array.from(maskedData).slice(0,3).map(v=>v.toFixed(4))}`);

  // Step 12: softmax
  console.log("Step 12: softmax...");
  const attnWeights = maskedScores.softmax(-1);
  const attnWData = await attnWeights.cpu();
  console.log(`  softmax OK, shape=${attnWeights.shape}, val=${Array.from(attnWData).slice(0,3).map(v=>v.toFixed(4))}`);

  // Step 13: attn @ v
  console.log("Step 13: attn @ v...");
  const attnOutput = attnWeights.matmul(v);
  const attnOutData = await attnOutput.cpu();
  console.log(`  attn@v OK, shape=${attnOutput.shape}, first 3: [${Array.from(attnOutData).slice(0,3).map(v=>v.toFixed(4))}]`);

  // Step 14: concat heads
  console.log("Step 14: concat heads...");
  const attnConcat = attnOutput.permute([0, 2, 1, 3]).contiguous().reshape([batch, seqLen, embedDim]);
  const concatData = await attnConcat.cpu();
  console.log(`  concat OK, shape=${attnConcat.shape}`);

  // Step 15: output projection
  console.log("Step 15: output projection...");
  const output = attn.cProj.forward(attnConcat);
  const outputData = await output.cpu();
  const outputArr = Array.from(outputData);
  const outputMean = outputArr.reduce((a,b)=>a+b,0)/outputArr.length;
  console.log(`  output OK, shape=${output.shape}, mean=${outputMean.toFixed(6)}`);
  console.log(`  first 5: [${outputArr.slice(0,5).map(v=>v.toFixed(4))}]`);

  console.log("\n=== Attention forward pass completed successfully ===");
}

main().catch(e => { console.error("ERROR:", e.message, e.stack); process.exit(1); });
