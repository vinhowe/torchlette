/**
 * Verify checkpoint + fusion produces identical gradients to non-checkpoint.
 * Also benchmarks different training configs to find stable defaults.
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

function loadWeights(tokDir: string) {
  const buf = fs.readFileSync(path.join(tokDir, "model.safetensors"));
  const hl = Number(
    new DataView(buf.buffer, buf.byteOffset, 8).getBigUint64(0, true),
  );
  const hdr = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + hl)));
  const w = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const [n, m] of Object.entries(hdr) as [string, any][]) {
    if (n === "__metadata__") continue;
    if (m.dtype !== "F32") continue;
    const raw = buf.subarray(
      8 + hl + m.data_offsets[0],
      8 + hl + m.data_offsets[1],
    );
    w.set(n.replace(/^transformer\./, ""), {
      data: new Float32Array(new Uint8Array(raw).slice().buffer),
      shape: m.shape,
    });
  }
  return w;
}

async function trainNStepsWithFusion(
  label: string,
  enableFusion: boolean,
  opts: {
    checkpoint: boolean;
    steps: number;
    batchSize: number;
    seqLen: number;
    lr: number;
  },
) {
  const api = new Torchlette("webgpu", { enableFusion });
  return trainNStepsImpl(label, api, opts);
}

async function trainNSteps(
  label: string,
  opts: {
    checkpoint: boolean;
    steps: number;
    batchSize: number;
    seqLen: number;
    lr: number;
  },
) {
  const api = new Torchlette("webgpu", { enableFusion: true });
  return trainNStepsImpl(label, api, opts);
}

async function trainNStepsImpl(
  label: string,
  api: Torchlette,
  opts: {
    checkpoint: boolean;
    steps: number;
    batchSize: number;
    seqLen: number;
    lr: number;
  },
) {
  const tokDir = path.join(process.cwd(), "models", "distilgpt2");
  const tokenizer = new GPT2Tokenizer();
  tokenizer.load(
    JSON.parse(fs.readFileSync(path.join(tokDir, "vocab.json"), "utf-8")),
    fs
      .readFileSync(path.join(tokDir, "merges.txt"), "utf-8")
      .split("\n")
      .filter((l) => l && !l.startsWith("#")),
  );
  const model = new GPT2WithLoRA(api, CONFIG, { rank: 4, alpha: 4 }, "webgpu");
  model.loadBaseWeights(loadWeights(tokDir));
  await api.markStep();

  const text =
    "The quick brown fox jumps over the lazy dog. " +
    "In a galaxy far far away, there lived a brave knight. " +
    "The rain in Spain falls mainly on the plain. " +
    "To be or not to be, that is the question. " +
    "All that glitters is not gold, but it sure looks shiny. " +
    "A stitch in time saves nine, or so they say around here.";

  const trainer = new LoRATrainer(api, model, tokenizer);
  const losses: number[] = [];
  const result = await trainer.train(
    text,
    {
      maxSteps: opts.steps,
      batchSize: opts.batchSize,
      seqLength: opts.seqLen,
      learningRate: opts.lr,
      useAMP: false,
      useCheckpointing: opts.checkpoint,
    },
    {
      onStepEnd: (_s, l) => losses.push(l),
    },
  );

  console.log(
    `[${label}] losses: ${losses.map((l) => l.toFixed(4)).join(" → ")}  final=${result.finalLoss.toFixed(4)}`,
  );
  return { losses, finalLoss: result.finalLoss };
}

async function main() {
  await initWebGPU();

  // Test 1: Gradient parity (checkpoint vs non-checkpoint)
  console.log("=== Test 1: Gradient Parity ===");
  const ref = await trainNSteps("no-ckpt ", {
    checkpoint: false,
    steps: 5,
    batchSize: 1,
    seqLen: 32,
    lr: 1e-3,
  });
  const ckpt = await trainNSteps("ckpt   ", {
    checkpoint: true,
    steps: 5,
    batchSize: 1,
    seqLen: 32,
    lr: 1e-3,
  });

  const lossDiffs = ref.losses.map((l, i) => Math.abs(l - ckpt.losses[i]));
  const maxLossDiff = Math.max(...lossDiffs);
  console.log(`Max loss diff across steps: ${maxLossDiff.toExponential(3)}`);
  console.log(`Parity: ${maxLossDiff < 0.01 ? "PASS" : "FAIL"}\n`);

  // Test 2: Isolate — is the bug fusion-specific or general?
  console.log("=== Test 2: Checkpoint with fusion OFF ===");
  const ckptNoFusion = await trainNStepsWithFusion("ckpt-nofusion", false, {
    checkpoint: true,
    steps: 5,
    batchSize: 1,
    seqLen: 32,
    lr: 1e-3,
  });
  console.log(
    `Checkpoint without fusion: ${ckptNoFusion.losses[0].toFixed(4)} (should be ~5.1)\n`,
  );

  // Test 3: First-step loss only (no optimizer, just forward+backward)
  console.log("=== Test 3: Single forward loss comparison ===");
  for (const [label, fus, ck] of [
    ["no-fusion no-ckpt", false, false],
    ["no-fusion ckpt   ", false, true],
    ["fusion    no-ckpt", true, false],
    ["fusion    ckpt   ", true, true],
  ] as [string, boolean, boolean][]) {
    const r = await trainNStepsWithFusion(label, fus, {
      checkpoint: ck,
      steps: 1,
      batchSize: 1,
      seqLen: 32,
      lr: 0,
    });
    console.log(`[${label}] step-0 loss = ${r.losses[0].toFixed(6)}`);
  }

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
