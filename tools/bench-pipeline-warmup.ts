/**
 * Benchmark: Pipeline Warmup via createComputePipelineAsync
 *
 * Measures step 0 time and compares sync vs async pipeline compilation.
 *
 * Usage:
 *   npx tsx tools/bench-pipeline-warmup.ts
 */

import * as path from "node:path";
import { Torchlette, type Tensor } from "../src/frontend";
import {
  initWebGPU,
  destroyWebGPU,
  getWebGPUDevice,
  isF16Supported,
  startPipelineRecording,
  stopPipelineRecording,
  warmupPipelines,
  clearWarmupCache,
} from "../src/backend/webgpu";
import { GPT2, DISTILGPT2_CONFIG } from "../examples/gpt2/model";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { Adam, GradScaler } from "../src/optim";
import { crossEntropy, checkpoint } from "../src/nn";

const SEQ_LEN = parseInt(process.env.TORCHLETTE_SEQ_LEN ?? "512", 10);
const BASE_TOKENS = [
  2484, 439, 314, 8996, 20218, 284, 257, 3931, 338, 1110, 30,
  198, 1986, 280, 1242, 517, 8855, 290, 517, 29815, 13,
  198, 49, 619, 9985, 466, 13508, 262, 38482, 31007, 286, 1737,
];
const FIXED_TOKENS: number[] = [];
for (let i = 0; i < SEQ_LEN + 1; i++) {
  FIXED_TOKENS.push(BASE_TOKENS[i % BASE_TOKENS.length]);
}

function forwardWithLoss(model: GPT2, api: Torchlette, idx: Tensor, targets: Tensor): Tensor {
  const [_batch, seqLen] = idx.shape;
  const tokEmb = model.wte.forward(idx);
  const pos = api.arange(seqLen).reshape([1, seqLen]);
  const posEmb = model.wpe.forward(pos);
  let x = api.add(tokEmb, posEmb);

  for (let i = 0; i < model.h.length; i++) {
    const block = model.h[i];
    const ln1Out = block.ln1.forward(x);
    const qkv = block.attn.cAttn.forward(ln1Out);
    const [batch, sl, _ed] = ln1Out.shape;
    const numHeads = block.attn["numHeads"] as number;
    const headDim = block.attn["headDim"] as number;
    const embedDim = block.attn["embedDim"] as number;
    const qkvFor3 = qkv.reshape([batch, sl, 3, embedDim]);
    const q = qkvFor3.narrow(2, 0, 1).reshape([batch, sl, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const k = qkvFor3.narrow(2, 1, 1).reshape([batch, sl, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const v = qkvFor3.narrow(2, 2, 1).reshape([batch, sl, numHeads, headDim]).permute([0, 2, 1, 3]).contiguous();
    const scale = 1.0 / Math.sqrt(headDim);
    const attnOutput = api.scaledDotProductAttention(q, k, v, scale, true);
    const attnConcat = attnOutput.permute([0, 2, 1, 3]).contiguous().reshape([batch, sl, embedDim]);
    const attnProjOut = block.attn.cProj.forward(attnConcat);
    const h = api.add(x, attnProjOut);
    x = checkpoint(api, (input: Tensor) => {
      const ln2Out = block.ln2.forward(input);
      let mlpH = block.mlp.cFc.forward(ln2Out);
      mlpH = mlpH.gelu();
      mlpH = block.mlp.cProj.forward(mlpH);
      return api.add(input, mlpH);
    }, [h]);
  }

  x = model.lnF.forward(x);
  const logits = api.linear(x, model.wte.weight, null);
  const [batch2, seqLenT] = targets.shape;
  const flatLogits = logits.reshape([batch2 * seqLenT, model.paddedVocabSize]);
  const realLogits = model.paddedVocabSize > model.config.vocabSize
    ? flatLogits.narrow(1, 0, model.config.vocabSize)
    : flatLogits;
  const flatTargets = targets.reshape([batch2 * seqLenT]);
  return crossEntropy(api, realLogits, flatTargets);
}

async function runStep(api: Torchlette, compiledForward: (...args: Tensor[]) => Tensor, optimizer: Adam, scaler: GradScaler | null) {
  if (scaler) await scaler.resolveDeferred();
  await api.beginStep();
  const input = api.tensorFromArray(FIXED_TOKENS.slice(0, -1), [1, SEQ_LEN], { device: "webgpu" });
  const target = api.tensorFromArray(FIXED_TOKENS.slice(1), [1, SEQ_LEN], { device: "webgpu" });

  const loss = compiledForward(input, target);
  const lossValue = await loss.item();

  if (scaler) {
    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();
    scaler.unscale_(optimizer);
    scaler.step(optimizer);
    scaler.update();
    scaledLoss.dispose();
  } else {
    await loss.backward();
    optimizer.step();
  }
  optimizer.zeroGrad();

  loss.dispose();
  input.dispose();
  target.dispose();
  api.endStep();
  await api.markStep();
  return lossValue;
}

async function main() {
  console.log("=== Pipeline Warmup Benchmark ===\n");

  const gpuOk = await initWebGPU();
  if (!gpuOk) {
    console.error("WebGPU init failed");
    process.exit(1);
  }

  const devInfo = getWebGPUDevice()!;
  const hasAsync = !!devInfo.device.createComputePipelineAsync;
  console.log(`createComputePipelineAsync: ${hasAsync ? "available" : "NOT available (will use sync fallback)"}`);

  const api = new Torchlette("webgpu", {
    enableFusion: true,
    enableMemoryPlanning: true,
    enableCheckpointSegmentation: true,
  });

  console.log("Loading model (distilgpt2)...");
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
  model.train();

  const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
  const useAMP = isF16Supported();
  const scaler = useAMP ? new GradScaler(api, { initScale: 1024.0 }) : null;
  console.log(`AMP: ${useAMP ? "enabled" : "disabled"}, Seq len: ${SEQ_LEN}\n`);

  const compiledForward = api.compile((input: Tensor, target: Tensor) => {
    if (useAMP) {
      return api.autocast(() => forwardWithLoss(model, api, input, target));
    }
    return forwardWithLoss(model, api, input, target);
  });

  // =========================================================================
  // Step 0: record all pipelines (this is the slow step with sync compilation)
  // =========================================================================
  console.log("--- Step 0 (baseline, sync compilation) ---");
  startPipelineRecording();
  const t0 = performance.now();
  const loss0 = await runStep(api, compiledForward, optimizer, scaler);
  const step0Time = performance.now() - t0;
  const registry = stopPipelineRecording();
  console.log(`  Time: ${step0Time.toFixed(0)}ms`);
  console.log(`  Loss: ${loss0.toFixed(4)}`);
  console.log(`  Pipelines compiled: ${registry.length}`);

  // Run steady-state steps
  for (let i = 1; i <= 4; i++) {
    const ts = performance.now();
    const l = await runStep(api, compiledForward, optimizer, scaler);
    console.log(`  Step ${i}: ${(performance.now() - ts).toFixed(0)}ms (loss=${l.toFixed(4)})`);
  }

  // =========================================================================
  // Warmup benchmark: compile the same shaders in parallel (isolated timing)
  // =========================================================================
  console.log("\n--- Warmup Compilation Benchmark ---");
  console.log(`  Shaders to compile: ${registry.length}`);

  // Clear warmup cache to force recompilation
  clearWarmupCache();

  const tw = performance.now();
  const warmupResult = await warmupPipelines(devInfo.device, registry);
  const warmupTime = performance.now() - tw;

  console.log(`  Parallel (createComputePipelineAsync): ${warmupTime.toFixed(0)}ms (${warmupResult.compiled} compiled, ${warmupResult.skipped} skipped)`);

  // Also measure sync compilation of the same shaders for fair comparison
  clearWarmupCache();
  const ts = performance.now();
  for (const entry of registry) {
    const module = devInfo.device.createShaderModule({ code: entry.wgsl });
    devInfo.device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }
  const syncTime = performance.now() - ts;
  console.log(`  Sequential (createComputePipeline):    ${syncTime.toFixed(0)}ms`);

  // =========================================================================
  // Summary
  // =========================================================================
  console.log(`\n${"=".repeat(60)}`);
  console.log("SUMMARY");
  console.log("=".repeat(60));
  console.log(`  Step 0 wall clock (includes all overhead): ${step0Time.toFixed(0)}ms`);
  console.log(`  Pipeline compilation only:`);
  console.log(`    Sequential:  ${syncTime.toFixed(0)}ms (${registry.length} shaders)`);
  console.log(`    Parallel:    ${warmupTime.toFixed(0)}ms (${registry.length} shaders)`);
  console.log(`    Speedup:     ${(syncTime / warmupTime).toFixed(2)}×`);
  console.log(`  Projected step 0 with warmup:`);
  const nonCompileTime = step0Time - syncTime;
  console.log(`    Non-compilation overhead:  ${nonCompileTime.toFixed(0)}ms`);
  console.log(`    With parallel warmup:      ${(nonCompileTime + warmupTime).toFixed(0)}ms`);
  console.log(`    vs baseline:               ${step0Time.toFixed(0)}ms`);
  console.log(`    Step 0 speedup:            ${(step0Time / (nonCompileTime + warmupTime)).toFixed(2)}×`);

  destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
