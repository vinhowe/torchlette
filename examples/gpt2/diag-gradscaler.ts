/**
 * Diagnostic: trace where inf/nan gradients come from in the finetuning pipeline.
 * Tests:
 *   1. Forward/backward without AMP or GradScaler (baseline)
 *   2. Forward/backward with AMP autocast but no scaler
 *   3. Forward/backward with AMP + scaler at scale=1
 *   4. Forward/backward with AMP + scaler at scale=1024
 */
import * as fs from "node:fs";
import * as path from "node:path";
import { Torchlette, type Tensor } from "../../src/frontend";
import { initWebGPU } from "../../src/backend/webgpu";
import { GPT2, DISTILGPT2_CONFIG } from "../../examples/gpt2/model";
import { loadPretrainedGPT2 } from "../../examples/gpt2/loader";
import { Adam, GradScaler } from "../../src/optim";

function paramNames(model: GPT2): string[] {
  const names: string[] = [];
  names.push("wte.weight");
  names.push("wpe.weight");
  for (let i = 0; i < model.h.length; i++) {
    names.push(`h.${i}.ln1.weight`, `h.${i}.ln1.bias`);
    names.push(`h.${i}.attn.cAttn.weight`, `h.${i}.attn.cAttn.bias`);
    names.push(`h.${i}.attn.cProj.weight`, `h.${i}.attn.cProj.bias`);
    names.push(`h.${i}.ln2.weight`, `h.${i}.ln2.bias`);
    names.push(`h.${i}.mlp.cFc.weight`, `h.${i}.mlp.cFc.bias`);
    names.push(`h.${i}.mlp.cProj.weight`, `h.${i}.mlp.cProj.bias`);
  }
  names.push("lnF.weight", "lnF.bias");
  return names;
}

async function checkGrads(api: Torchlette, model: GPT2, label: string) {
  const params = model.parameters();
  const names = paramNames(model);
  let totalParams = 0;
  let nullGrads = 0;
  let infGrads = 0;
  let nanGrads = 0;
  let zeroGrads = 0;
  let maxMag = 0;
  const zeroNames: string[] = [];

  for (let pi = 0; pi < params.length; pi++) {
    const p = params[pi];
    totalParams++;
    const grad = p.grad;
    if (!grad) { nullGrads++; continue; }
    const data = await grad.cpu();
    let hasInf = false, hasNan = false, allZero = true;
    for (const v of data) {
      if (!Number.isFinite(v)) { if (Number.isNaN(v)) hasNan = true; else hasInf = true; }
      if (v !== 0) allZero = false;
      const abs = Math.abs(v);
      if (Number.isFinite(abs) && abs > maxMag) maxMag = abs;
    }
    if (hasInf) infGrads++;
    if (hasNan) nanGrads++;
    if (allZero) { zeroGrads++; zeroNames.push(names[pi] ?? `param[${pi}]`); }
  }
  console.log(`[${label}] params=${totalParams}, null=${nullGrads}, inf=${infGrads}, nan=${nanGrads}, zero=${zeroGrads}, maxMag=${maxMag.toExponential(3)}`);
  if (zeroNames.length > 0) {
    console.log(`  zero-grad params: ${zeroNames.join(", ")}`);
  }
}

async function main() {
  const success = await initWebGPU();
  if (!success) { console.error("No WebGPU"); process.exit(1); }

  const modelDir = path.join(process.cwd(), "models", "distilgpt2");

  // Small input
  const inputData = [15496, 11, 314, 716, 2993, 284, 910, 326, 262]; // "Hello, I just wanted to say that the"
  const targetData = [11, 314, 716, 2993, 284, 910, 326, 262, 995]; // shifted by 1

  // ── Test 1: No AMP, no scaler ──
  {
    console.log("\n=== Test 1: No AMP, no GradScaler ===");
    const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: false });
    const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
    model.train();

    const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });

    const result = model.forwardWithLoss(input, target);
    const lossVal = await result.loss!.item();
    console.log(`  loss = ${lossVal.toFixed(4)}`);

    await result.loss!.backward();
    await checkGrads(api, model, "no-AMP");
    await api.markStep();
  }

  // ── Test 2: AMP autocast, no scaler ──
  {
    console.log("\n=== Test 2: AMP autocast, no GradScaler ===");
    const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: false });
    const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
    model.train();

    const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });

    const loss = api.autocast(() => {
      const result = model.forwardWithLoss(input, target);
      if (!result.loss) throw new Error("no loss");
      return result.loss;
    });
    const lossVal = await loss.item();
    console.log(`  loss = ${lossVal.toFixed(4)}`);

    await loss.backward();
    await checkGrads(api, model, "AMP-no-scaler");
    await api.markStep();
  }

  // ── Test 3: AMP + scaler at scale=1 ──
  {
    console.log("\n=== Test 3: AMP + GradScaler(scale=1) ===");
    const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: false });
    const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
    model.train();
    const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
    const scaler = new GradScaler(api, { initScale: 1.0 });

    const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });

    const loss = api.autocast(() => {
      const result = model.forwardWithLoss(input, target);
      if (!result.loss) throw new Error("no loss");
      return result.loss;
    });
    const lossVal = await loss.item();
    console.log(`  loss = ${lossVal.toFixed(4)}`);

    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();

    scaler.unscale_(optimizer);
    console.log(`  foundInf = ${scaler.foundInf}`);
    await checkGrads(api, model, "AMP-scale1");
    await api.markStep();
  }

  // ── Test 4: AMP + scaler at scale=1024 ──
  {
    console.log("\n=== Test 4: AMP + GradScaler(scale=1024) ===");
    const api = new Torchlette("webgpu", { enableFusion: false, enableMemoryPlanning: false });
    const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
    model.train();
    const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
    const scaler = new GradScaler(api, { initScale: 1024.0 });

    const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });

    const loss = api.autocast(() => {
      const result = model.forwardWithLoss(input, target);
      if (!result.loss) throw new Error("no loss");
      return result.loss;
    });
    const lossVal = await loss.item();
    console.log(`  loss = ${lossVal.toFixed(4)}`);

    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();

    scaler.unscale_(optimizer);
    console.log(`  foundInf = ${scaler.foundInf}`);
    await checkGrads(api, model, "AMP-scale1024");
    await api.markStep();
  }

  // ── Test 5: AMP + compile + checkpoint + scaler at scale=1 ──
  {
    console.log("\n=== Test 5: AMP + compile + checkpoint + GradScaler(scale=1) ===");
    const api = new Torchlette("webgpu", { enableFusion: true, enableMemoryPlanning: true });
    const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 }, { device: "webgpu" });
    model.train();
    const optimizer = new Adam(model.parameters(), { lr: 1e-4 }, api);
    const scaler = new GradScaler(api, { initScale: 1.0 });

    const compiledForward = api.compile((input: Tensor, target: Tensor) => {
      return api.autocast(() => {
        const result = model.forwardWithLoss(input, target, { useCheckpoint: true });
        if (!result.loss) throw new Error("no loss");
        return result.loss;
      });
    });

    const input = api.tensorFromArray(inputData, [1, inputData.length], { device: "webgpu" });
    const target = api.tensorFromArray(targetData, [1, targetData.length], { device: "webgpu" });

    const loss = compiledForward(input, target);
    const lossVal = await loss.item();
    console.log(`  loss = ${lossVal.toFixed(4)}`);

    const scaledLoss = scaler.scale(loss);
    await scaledLoss.backward();

    scaler.unscale_(optimizer);
    console.log(`  foundInf = ${scaler.foundInf}`);
    await checkGrads(api, model, "full-pipeline-scale1");
    await api.markStep();
  }
}

main().catch(e => { console.error(e); process.exit(1); });
