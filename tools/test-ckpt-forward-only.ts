/**
 * Isolated test: does checkpoint change the forward pass loss with the real GPT-2 model?
 * Tests all 4 combinations of fusion × checkpoint.
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

async function forwardLoss(fusion: boolean, ckpt: boolean): Promise<number> {
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
  const model = new GPT2WithLoRA(api, CONFIG, { rank: 4, alpha: 4 }, "webgpu");
  model.loadBaseWeights(loadWeights());
  await api.markStep();

  if (ckpt) model.enableCheckpointing(true);
  model.train(true);

  await api.beginStep();
  const tokens = tok.encode(
    "The quick brown fox jumps over the lazy dog and runs",
  );
  const seqLen = Math.min(32, tokens.length - 1);
  const input = api.tensorFromArray(tokens.slice(0, seqLen), [1, seqLen], {
    device: "webgpu",
  });
  const target = api.tensorFromArray(tokens.slice(1, seqLen + 1), [1, seqLen], {
    device: "webgpu",
  });
  const { loss } = model.forwardWithLoss(input, target);
  const lossVal = await loss.item();
  api.endStep();
  await api.markStep();
  return lossVal;
}

async function main() {
  await initWebGPU();

  for (const [label, fus, ck] of [
    ["no-fusion no-ckpt", false, false],
    ["no-fusion ckpt   ", false, true],
    ["fusion    no-ckpt", true, false],
    ["fusion    ckpt   ", true, true],
  ] as [string, boolean, boolean][]) {
    const l = await forwardLoss(fus, ck);
    console.log(`[${label}] loss = ${l.toFixed(6)}`);
  }

  // Extra test: fusion + checkpoint but only 1 layer
  console.log("\n--- Varying layer count with fusion+ckpt ---");
  for (const nLayers of [1, 2, 3, 6]) {
    const api = new Torchlette("webgpu", { enableFusion: true });
    const d = path.join(process.cwd(), "models", "distilgpt2");
    const tok = new GPT2Tokenizer();
    tok.load(
      JSON.parse(fs.readFileSync(path.join(d, "vocab.json"), "utf-8")),
      fs
        .readFileSync(path.join(d, "merges.txt"), "utf-8")
        .split("\n")
        .filter((l) => l && !l.startsWith("#")),
    );
    const cfg = { ...CONFIG, numLayers: nLayers };
    const model = new GPT2WithLoRA(api, cfg, { rank: 4, alpha: 4 }, "webgpu");
    model.loadBaseWeights(loadWeights());
    await api.markStep();
    model.enableCheckpointing(true);
    model.train(true);
    await api.beginStep();
    const tokens = tok.encode(
      "The quick brown fox jumps over the lazy dog and runs",
    );
    const seqLen = Math.min(32, tokens.length - 1);
    const input = api.tensorFromArray(tokens.slice(0, seqLen), [1, seqLen], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(
      tokens.slice(1, seqLen + 1),
      [1, seqLen],
      { device: "webgpu" },
    );
    const { loss } = model.forwardWithLoss(input, target);
    const lossVal = await loss.item();
    api.endStep();
    await api.markStep();

    // Reference: same config without checkpoint
    const api2 = new Torchlette("webgpu", { enableFusion: true });
    const model2 = new GPT2WithLoRA(api2, cfg, { rank: 4, alpha: 4 }, "webgpu");
    model2.loadBaseWeights(loadWeights());
    await api2.markStep();
    model2.train(true);
    await api2.beginStep();
    const input2 = api2.tensorFromArray(tokens.slice(0, seqLen), [1, seqLen], {
      device: "webgpu",
    });
    const target2 = api2.tensorFromArray(
      tokens.slice(1, seqLen + 1),
      [1, seqLen],
      { device: "webgpu" },
    );
    const { loss: loss2 } = model2.forwardWithLoss(input2, target2);
    const refVal = await loss2.item();
    api2.endStep();
    await api2.markStep();

    const diff = Math.abs(lossVal - refVal);
    console.log(
      `layers=${nLayers}: ckpt=${lossVal.toFixed(6)} ref=${refVal.toFixed(6)} diff=${diff.toExponential(2)} ${diff < 0.001 ? "OK" : "WRONG"}`,
    );
  }

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
