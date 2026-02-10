/**
 * Step-by-step debug of forward pass.
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

  console.log("1. Embeddings...");
  const tokEmb = model.wte.forward(input);
  const posEmb = model.wpe.forward(pos);
  let x = api.add(tokEmb, posEmb);
  console.log(`   x shape: ${x.shape}`);
  const xData = await x.cpu();
  console.log(`   x first 5: [${Array.from(xData).slice(0,5).map(v=>v.toFixed(4))}]`);

  // Reshape for block input
  x = x.reshape([1, 1, 768]);

  console.log("2. LN1...");
  const ln1Out = model.h[0].ln1.forward(x);
  const ln1Data = await ln1Out.cpu();
  console.log(`   ln1 first 5: [${Array.from(ln1Data).slice(0,5).map(v=>v.toFixed(4))}]`);
  console.log(`   (oracle: [-0.0522, -0.1535, 0.0001, -0.0760, -0.0346])`);

  console.log("3. c_attn projection...");
  const qkv = model.h[0].attn.cAttn.forward(ln1Out);
  const qkvData = await qkv.cpu();
  const qkvArr = Array.from(qkvData);
  console.log(`   qkv shape: ${qkv.shape}, first 5: [${qkvArr.slice(0,5).map(v=>v.toFixed(4))}]`);
  const qkvMean = qkvArr.reduce((a,b)=>a+b,0)/qkvArr.length;
  console.log(`   qkv mean=${qkvMean.toFixed(6)} (oracle: 0.011560)`);

  console.log("4. QKV split (gather approach)...");
  const batch = 1, seqLen = 1, embedDim = 768;
  const qkvSplit3 = qkv.reshape([batch, seqLen, 3, embedDim]);
  const qkvTransposed = qkvSplit3.permute([2, 0, 1, 3]);
  console.log("   permuted shape:", qkvTransposed.shape);
  const flatTotal = batch * seqLen * embedDim;
  const qkvFlattened = qkvTransposed.contiguous().reshape([3, flatTotal]);
  console.log("   flattened shape:", qkvFlattened.shape);

  // Check: read the flattened data - Q should be first embedDim values, K next, V last
  const flatData = await qkvFlattened.cpu();
  const flatArr = Array.from(flatData);
  console.log(`   Q (from flat) first 5: [${flatArr.slice(0, 5).map(v=>v.toFixed(4))}]`);
  console.log(`   K (from flat) first 5: [${flatArr.slice(768, 773).map(v=>v.toFixed(4))}]`);
  console.log(`   V (from flat) first 5: [${flatArr.slice(1536, 1541).map(v=>v.toFixed(4))}]`);
  console.log(`   Q mean=${(flatArr.slice(0,768).reduce((a,b)=>a+b,0)/768).toFixed(6)} (oracle: -0.015929)`);
  console.log(`   K mean=${(flatArr.slice(768,1536).reduce((a,b)=>a+b,0)/768).toFixed(6)} (oracle: 0.053499)`);
  console.log(`   V mean=${(flatArr.slice(1536,2304).reduce((a,b)=>a+b,0)/768).toFixed(6)} (oracle: -0.002889)`);

  console.log("5. Gather Q, K, V...");
  // This is the potentially buggy part
  const idx0 = api.tensorFromArray([0], [1, 1]).expand([1, flatTotal]);
  const idx1 = api.tensorFromArray([1], [1, 1]).expand([1, flatTotal]);
  const idx2 = api.tensorFromArray([2], [1, 1]).expand([1, flatTotal]);

  console.log("   idx0 shape:", idx0.shape);
  // Try reading idx0 to check if expand works
  const idx0Data = await idx0.cpu();
  console.log(`   idx0 first 5: [${Array.from(idx0Data).slice(0,5)}], last 5: [${Array.from(idx0Data).slice(-5)}]`);

  const qFlat = qkvFlattened.gather(idx0, { dim: 0 });
  console.log("   qFlat gathered, shape:", qFlat.shape);
  const qFlatData = await qFlat.cpu();
  console.log(`   Q (gathered) first 5: [${Array.from(qFlatData).slice(0,5).map(v=>v.toFixed(4))}]`);

  console.log("6. Reshape Q to heads and compute attention...");
  const numHeads = 12, headDim = 64;
  const q = qFlat.contiguous().reshape([batch, seqLen, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
  console.log(`   q shape: ${q.shape}`);

  // For seqLen=1, attention is trivial: softmax([score]) = [1.0], output = v
  // So attention output should just be V reshaped
  console.log("7. Full attention forward...");
  const attnOut = model.h[0].attn.forward(x.reshape([1, 1, 768]));
  const attnData = await attnOut.cpu();
  const attnArr = Array.from(attnData);
  const attnMean = attnArr.reduce((a,b)=>a+b,0)/attnArr.length;
  console.log(`   attn shape: ${attnOut.shape}, mean=${attnMean.toFixed(6)}`);
  console.log(`   first 5: [${attnArr.slice(0,5).map(v=>v.toFixed(4))}]`);

  console.log("\n=== Done ===");
}

main().catch(e => { console.error("ERROR:", e.message); process.exit(1); });
