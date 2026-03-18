/**
 * Compare gradients: checkpoint+fusion vs reference (no checkpoint, no fusion).
 * If gradients differ, backward is broken and training diverges.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import {
  type GPT2Config,
  GPT2WithLoRA,
} from "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora";
import { GPT2Tokenizer } from "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim";

const CONFIG: GPT2Config = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 6,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0,
};

function loadWeights() {
  const d = path.join(process.cwd(), "models", "distilgpt2");
  const buf = fs.readFileSync(path.join(d, "model.safetensors"));
  const hl = Number(
    new DataView(buf.buffer, buf.byteOffset, 8).getBigUint64(0, true),
  );
  const hdr = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + hl)));
  const w = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const [n, m] of Object.entries(hdr) as [string, any][]) {
    if (n === "__metadata__" || m.dtype !== "F32") continue;
    const r = buf.subarray(
      8 + hl + m.data_offsets[0],
      8 + hl + m.data_offsets[1],
    );
    w.set(n.replace(/^transformer\./, ""), {
      data: new Float32Array(new Uint8Array(r).slice().buffer),
      shape: m.shape,
    });
  }
  return w;
}

async function getGradients(
  fusion: boolean,
  ckpt: boolean,
  tokens: number[],
): Promise<{ loss: number; grads: Float32Array[] }> {
  const api = new Torchlette("webgpu", { enableFusion: fusion });
  const d = path.join(process.cwd(), "models", "distilgpt2");
  const tok = new GPT2Tokenizer();
  tok.load(
    JSON.parse(fs.readFileSync(path.join(d, "vocab.json"), "utf-8")),
    fs
      .readFileSync(path.join(d, "merges.txt"), "utf-8")
      .split("\n")
      .filter((l) => l && !l.startsWith("#")),
  );
  const model = new GPT2WithLoRA(
    api,
    CONFIG,
    { rank: 16, alpha: 16 },
    "webgpu",
  );
  model.loadBaseWeights(loadWeights());
  await api.markStep();

  if (ckpt) model.enableCheckpointing(true);
  model.train(true);

  const loraParams = model.getLoRAParameters();

  await api.beginStep();
  const seqLen = 32;
  const input = api.tensorFromArray(tokens.slice(0, seqLen), [1, seqLen], {
    device: "webgpu",
  });
  const target = api.tensorFromArray(tokens.slice(1, seqLen + 1), [1, seqLen], {
    device: "webgpu",
  });
  const { loss } = model.forwardWithLoss(input, target);
  const lossVal = await loss.item();
  await loss.backward();

  const grads: Float32Array[] = [];
  for (const p of loraParams) {
    if (p.grad) {
      grads.push(new Float32Array(await p.grad.cpu()));
    } else {
      grads.push(new Float32Array(p.shape.reduce((a, b) => a * b, 1)));
    }
  }

  api.endStep();
  await api.markStep();
  return { loss: lossVal, grads };
}

async function main() {
  await initWebGPU();
  const d = path.join(process.cwd(), "models", "distilgpt2");
  const tok = new GPT2Tokenizer();
  tok.load(
    JSON.parse(fs.readFileSync(path.join(d, "vocab.json"), "utf-8")),
    fs
      .readFileSync(path.join(d, "merges.txt"), "utf-8")
      .split("\n")
      .filter((l) => l && !l.startsWith("#")),
  );
  const tokens = tok.encode(
    "The quick brown fox jumps over the lazy dog and then runs across the wide open field. " +
      "Meanwhile, in a galaxy far far away, there lived a brave knight who fought many battles.",
  );

  console.log("=== Gradient Correctness Test ===\n");

  const ref = await getGradients(false, false, tokens);
  console.log(`Reference (no fusion, no ckpt): loss=${ref.loss.toFixed(6)}`);

  const fusionOnly = await getGradients(true, false, tokens);
  console.log(
    `Fusion only:                    loss=${fusionOnly.loss.toFixed(6)}`,
  );

  const ckptOnly = await getGradients(false, true, tokens);
  console.log(
    `Checkpoint only:                loss=${ckptOnly.loss.toFixed(6)}`,
  );

  const both = await getGradients(true, true, tokens);
  console.log(`Fusion + Checkpoint:            loss=${both.loss.toFixed(6)}`);

  // Compare gradients
  console.log("\n--- Gradient comparison ---");
  for (const [label, test] of [
    ["fusion-only", fusionOnly],
    ["ckpt-only", ckptOnly],
    ["fusion+ckpt", both],
  ] as [string, typeof ref][]) {
    let maxDiff = 0;
    let totalDiff = 0;
    let count = 0;
    for (let p = 0; p < ref.grads.length; p++) {
      for (let i = 0; i < ref.grads[p].length; i++) {
        const d = Math.abs(ref.grads[p][i] - test.grads[p][i]);
        maxDiff = Math.max(maxDiff, d);
        totalDiff += d;
        count++;
      }
    }
    const avgDiff = totalDiff / count;
    const lossDiff = Math.abs(ref.loss - test.loss);
    console.log(
      `[${label}] lossDiff=${lossDiff.toExponential(2)} maxGradDiff=${maxDiff.toExponential(2)} avgGradDiff=${avgDiff.toExponential(2)} ${maxDiff < 1e-3 ? "OK" : "WRONG"}`,
    );
  }

  // Also test: does 5 steps of training converge with fusion+ckpt?
  console.log(
    "\n--- 5-step training test (fusion+ckpt, same data each step) ---",
  );
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = new GPT2WithLoRA(
    api,
    CONFIG,
    { rank: 16, alpha: 16 },
    "webgpu",
  );
  model.loadBaseWeights(loadWeights());
  await api.markStep();
  model.enableCheckpointing(true);
  model.train(true);
  const loraParams = model.getLoRAParameters();
  const optimizer = new Adam(loraParams, { lr: 2e-3 }, api);

  for (let step = 0; step < 10; step++) {
    await api.beginStep();
    const input = api.tensorFromArray(tokens.slice(0, 32), [1, 32], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(tokens.slice(1, 33), [1, 32], {
      device: "webgpu",
    });
    const { loss } = model.forwardWithLoss(input, target);
    const lossVal = await loss.item();
    await loss.backward();
    optimizer.step();
    optimizer.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
    console.log(`  Step ${step}: loss=${lossVal.toFixed(4)}`);
  }

  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
