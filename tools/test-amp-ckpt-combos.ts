/**
 * Compare all 4 combinations of AMP × Checkpoint to isolate oscillation source.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import {
  type GPT2Config,
  GPT2WithLoRA,
} from "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora";
import { GPT2Tokenizer } from "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer";
import { LoRATrainer } from "../examples/gpt2-lora-trainer/src/lib/torchlette/trainer";
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

// Repetitive text so each batch has similar difficulty
const base =
  "First Citizen: Before we proceed any further, hear me speak. " +
  "All: Speak, speak. " +
  "First Citizen: You are all resolved rather to die than to famish? " +
  "All: Resolved. resolved. " +
  "First Citizen: First, you know Caius Marcius is chief enemy to the people. " +
  "All: We know't, we know't. " +
  "First Citizen: Let us kill him, and we'll have corn at our own price. ";
const text = base.repeat(80);

async function run(label: string, amp: boolean, ckpt: boolean) {
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
  const model = new GPT2WithLoRA(
    api,
    CONFIG,
    { rank: 16, alpha: 16 },
    "webgpu",
  );
  model.loadBaseWeights(loadWeights());
  await api.markStep();

  const trainer = new LoRATrainer(api, model, tok);
  const losses: number[] = [];
  try {
    await trainer.train(
      text,
      {
        maxSteps: 30,
        batchSize: 4,
        seqLength: 64,
        learningRate: 2e-3,
        useAMP: amp,
        useCheckpointing: ckpt,
      },
      { onStepEnd: (_, l) => losses.push(l) },
    );
  } catch (e: any) {
    console.log(`[${label}] FAILED after ${losses.length} steps: ${e.message}`);
    return;
  }

  let inc = 0;
  for (let i = 1; i < losses.length; i++) {
    if (losses[i] > losses[i - 1] + 0.01) inc++;
  }
  console.log(`[${label}] ${losses.map((l) => l.toFixed(3)).join(" ")}`);
  console.log(
    `  ${losses[0].toFixed(3)} → ${losses[losses.length - 1].toFixed(3)}  increases=${inc}/${losses.length - 1}`,
  );
}

async function main() {
  await initWebGPU();
  await run("no-amp no-ckpt", false, false);
  await run("no-amp ckpt   ", false, true);
  // AMP may fail on Dawn without shader-f16
  await run("amp    no-ckpt", true, false);
  await run("amp    ckpt   ", true, true);
  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
