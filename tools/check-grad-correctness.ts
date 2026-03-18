/**
 * Gradient correctness check: load two independent models, run forward+backward
 * on each, compare gradients. Both use the same GPU device but separate
 * RuntimeEngine instances, so no shared state.
 */
import * as path from "node:path";
import { loadPretrainedGPT2 } from "../examples/gpt2/loader";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

const FIXED_TOKENS: number[] = [
  2484, 439, 314, 8996, 20218, 284, 257, 3931, 338, 1110, 30, 198, 1986, 280,
  1242, 517, 8855, 290, 517, 29815, 13, 198, 49, 619, 9985, 466, 13508, 262,
  38482, 31007, 286, 1737,
];

async function run(label: string, enableFusion: boolean) {
  const api = new Torchlette("webgpu", { enableFusion });
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(
    api,
    modelDir,
    { dropoutRate: 0 },
    { device: "webgpu" },
  );
  model.train();

  await api.beginStep();
  const inputData = FIXED_TOKENS.slice(0, -1);
  const targetData = FIXED_TOKENS.slice(1);
  const input = api.tensorFromArray(inputData, [1, inputData.length], {
    device: "webgpu",
  });
  const target = api.tensorFromArray(targetData, [1, targetData.length], {
    device: "webgpu",
  });

  const result = model.forwardWithLoss(input, target, {});
  const loss = result.loss!;
  const lossVal = await loss.item();
  await loss.backward();

  const grads: Float32Array[] = [];
  const params = model.parameters();
  for (let i = 0; i < Math.min(params.length, 8); i++) {
    const g = params[i].grad;
    grads.push(g ? new Float32Array(await g.cpu()) : new Float32Array(0));
  }

  console.log(`${label}: loss=${lossVal.toFixed(6)}, params=${params.length}`);

  loss.dispose();
  input.dispose();
  target.dispose();
  api.endStep();
  await api.markStep();

  return { lossVal, grads, shapes: params.slice(0, 8).map((p) => p.shape) };
}

async function main() {
  await initWebGPU();

  // Same model, same runtime, same config — two backward passes
  // If THESE differ, there's non-determinism or state corruption
  const api = new Torchlette("webgpu", { enableFusion: true });
  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const model = await loadPretrainedGPT2(
    api,
    modelDir,
    { dropoutRate: 0 },
    { device: "webgpu" },
  );
  const inputData = FIXED_TOKENS.slice(0, -1);
  const targetData = FIXED_TOKENS.slice(1);

  // Run 1
  model.train();
  await api.beginStep();
  const input1 = api.tensorFromArray(inputData, [1, inputData.length], {
    device: "webgpu",
  });
  const target1 = api.tensorFromArray(targetData, [1, targetData.length], {
    device: "webgpu",
  });
  const loss1 = model.forwardWithLoss(input1, target1, {}).loss!;
  const lv1 = await loss1.item();
  await loss1.backward();
  const grads1: Float32Array[] = [];
  for (let i = 0; i < Math.min(model.parameters().length, 8); i++) {
    const g = model.parameters()[i].grad;
    grads1.push(g ? new Float32Array(await g.cpu()) : new Float32Array(0));
  }
  loss1.dispose();
  input1.dispose();
  target1.dispose();
  api.endStep();
  await api.markStep();

  // Zero grads
  const { Adam } = await import("../src/optim");
  const opt = new Adam(model.parameters(), { lr: 1e-10 }, api);
  opt.zeroGrad();

  // Run 2 (same model, same input, grads should be identical)
  model.train();
  await api.beginStep();
  const input2 = api.tensorFromArray(inputData, [1, inputData.length], {
    device: "webgpu",
  });
  const target2 = api.tensorFromArray(targetData, [1, targetData.length], {
    device: "webgpu",
  });
  const loss2 = model.forwardWithLoss(input2, target2, {}).loss!;
  const lv2 = await loss2.item();
  await loss2.backward();
  const grads2: Float32Array[] = [];
  for (let i = 0; i < Math.min(model.parameters().length, 8); i++) {
    const g = model.parameters()[i].grad;
    grads2.push(g ? new Float32Array(await g.cpu()) : new Float32Array(0));
  }
  loss2.dispose();
  input2.dispose();
  target2.dispose();
  api.endStep();
  await api.markStep();

  console.log(`Run 1 loss: ${lv1.toFixed(6)}, Run 2 loss: ${lv2.toFixed(6)}`);
  const r1 = {
    lossVal: lv1,
    grads: grads1,
    shapes: model
      .parameters()
      .slice(0, 8)
      .map((p) => p.shape),
  };
  const r2 = { lossVal: lv2, grads: grads2, shapes: r1.shapes };

  console.log(
    `\nLoss diff: ${Math.abs(r1.lossVal - r2.lossVal).toExponential(4)}`,
  );

  let maxRelErr = 0;
  let totalMismatch = 0;
  let totalParams = 0;
  for (let i = 0; i < r1.grads.length; i++) {
    const g1 = r1.grads[i];
    const g2 = r2.grads[i];
    if (g1.length === 0 || g2.length === 0) {
      console.log(`param[${i}]: no grad`);
      continue;
    }
    let pMax = 0;
    let pMismatch = 0;
    for (let j = 0; j < g1.length; j++) {
      const denom = Math.max(Math.abs(g1[j]), 1e-8);
      const re = Math.abs(g2[j] - g1[j]) / denom;
      if (re > pMax) pMax = re;
      if (re > 0.01) pMismatch++;
    }
    totalParams += g1.length;
    totalMismatch += pMismatch;
    if (pMax > maxRelErr) maxRelErr = pMax;
    console.log(
      `param[${i}] ${JSON.stringify(r1.shapes[i])}: maxRelErr=${(pMax * 100).toFixed(4)}% mismatch(>1%)=${pMismatch}/${g1.length}`,
    );
    if (pMismatch > 0) {
      for (let j = 0; j < g1.length; j++) {
        const denom = Math.max(Math.abs(g1[j]), 1e-8);
        if (Math.abs(g2[j] - g1[j]) / denom > 0.01) {
          console.log(
            `  [${j}] seq=${g1[j].toFixed(10)} fused=${g2[j].toFixed(10)}`,
          );
          break;
        }
      }
    }
  }

  console.log(
    `\nOverall: maxRelErr=${(maxRelErr * 100).toFixed(4)}% mismatches(>1%)=${totalMismatch}/${totalParams}`,
  );
  if (maxRelErr < 0.001) console.log("\nEXCELLENT: gradients match (<0.1%)");
  else if (maxRelErr < 0.01) console.log("\nGOOD: minor differences (<1%)");
  else if (maxRelErr < 0.05) console.log("\nOK: within AMP tolerance (<5%)");
  else console.log("\nBUG: significant gradient differences");

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
