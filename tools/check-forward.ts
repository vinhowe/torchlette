/**
 * Diagnostic: compare fusion vs non-fusion per-layer for GPT-2 Large.
 * Dumps first 10 float values after each layer to detect bit-level divergence.
 */
import { Torchlette } from "../src/frontend";
import { initWebGPU } from "../src/backend/webgpu";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { GPT2Tokenizer } from "../examples/gpt2/data";

async function main() {
  await initWebGPU();
  const FUSION = process.env.FUSION !== "0";
  const MAX_LAYERS = parseInt(process.env.MAX_LAYERS ?? "6", 10);
  console.log(`Config: enableFusion=${FUSION}, maxLayers=${MAX_LAYERS}`);
  const api = new Torchlette("webgpu", { enableFusion: FUSION, enableMemoryPlanning: true });

  const modelDir = `models/${process.env.MODEL ?? "gpt2-large"}`;
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.001 }, { device: "webgpu" });
  model.eval();

  const tokenizer = new GPT2Tokenizer();
  await tokenizer.load(modelDir);

  const tokens = tokenizer.encode("The capital of France is");
  console.log("tokens:", tokens);

  const inputTensor = api.tensorFromArray(tokens, [1, tokens.length], { device: "webgpu" });
  const pos = model.posIndices.narrow(1, 0, tokens.length);
  const tokEmb = model.wte.forward(inputTensor);
  const posEmb = model.wpe.forward(pos);
  let x = api.add(tokEmb, posEmb);

  // Dump embedding values
  const embData = await x.cpu();
  const embArr = Array.from(embData);
  console.log(`EMB [0:10]:`, embArr.slice(0, 10).map(v => v.toFixed(8)));

  const numLayers = Math.min(model.h.length, MAX_LAYERS);
  for (let i = 0; i < numLayers; i++) {
    x = model.h[i].forward(x);
    const layerData = await x.cpu();
    const layerArr = Array.from(layerData);
    const lMean = layerArr.reduce((a, b) => a + b, 0) / layerArr.length;
    // Print first 10 values at high precision
    console.log(`L${i} [0:10]:`, layerArr.slice(0, 10).map(v => v.toFixed(8)));
    // Also print values at offset embedDim (start of position 1)
    console.log(`L${i} [${model.config.embedDim}:${model.config.embedDim+5}]:`, layerArr.slice(model.config.embedDim, model.config.embedDim + 5).map(v => v.toFixed(8)));
    console.log(`L${i} mean=${lMean.toFixed(8)}`);
  }

  // Final LN + logits
  x = model.lnF.forward(x);
  const logits = api.linear(x, model.wte.weight, null);
  const logitsData = await logits.cpu();
  const allLogits = Array.from(logitsData);
  const stride = model.paddedVocabSize;
  const lastIdx = (tokens.length - 1) * stride;
  const lastLogits = allLogits.slice(lastIdx, lastIdx + tokenizer.vocabSize);
  const maxIdx = lastLogits.indexOf(Math.max(...lastLogits));
  console.log(`\nArgmax: ${maxIdx} "${tokenizer.decode([maxIdx])}" logit=${lastLogits[maxIdx].toFixed(6)}`);

  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
