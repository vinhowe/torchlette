/**
 * Test all 4 combinations: {AMP, no-AMP} × {checkpoint, no-checkpoint}
 * GPT-2 small, rank=64, lr=1e-3, 200 steps on 1393 Shakespeare tokens.
 * The no-AMP no-checkpoint combo converges (3.85→0.58). Do the others?
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
  numLayers: 12,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0,
};

function loadWeights() {
  const d = path.join(process.cwd(), "models", "gpt2");
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

const shakespeare = fs
  .readFileSync("node_modules/.cache/tinyshakespeare.txt", "utf-8")
  .slice(0, 5000);

async function run(label: string, amp: boolean, ckpt: boolean) {
  const api = new Torchlette("webgpu", { enableFusion: true });
  const d = path.join(process.cwd(), "models", "gpt2");
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
    { rank: 64, alpha: 64 },
    "webgpu",
  );
  model.loadBaseWeights(loadWeights());
  await api.markStep();

  const trainer = new LoRATrainer(api, model, tok);
  const losses: number[] = [];
  try {
    await trainer.train(
      shakespeare,
      {
        maxSteps: 100,
        batchSize: 1,
        seqLength: 128,
        learningRate: 1e-3,
        useAMP: amp,
        useCheckpointing: ckpt,
      },
      { onStepEnd: (_, l) => losses.push(l) },
    );
  } catch (e: any) {
    console.log(
      `[${label}] FAILED at step ${losses.length}: ${e.message.slice(0, 100)}`,
    );
    return;
  }

  const first = losses[0];
  const last = losses[losses.length - 1];
  const min = Math.min(...losses);
  const converged = last < first;
  console.log(
    `[${label}] ${first.toFixed(2)} → ${last.toFixed(2)} (min=${min.toFixed(2)}) ${converged ? "OK" : "DIVERGED"}`,
  );
}

async function main() {
  await initWebGPU();
  await run("no-amp no-ckpt", false, false);
  await run("no-amp ckpt   ", false, true);
  await run("amp    no-ckpt", true, false);
  await run("amp    ckpt   ", true, true);
  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
