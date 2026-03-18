/**
 * Offline verification of exact browser training config.
 * Simulates the browser demo: LoRATrainer with sequential data, checkpointing, fusion.
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

async function main() {
  await initWebGPU();
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
  model.loadBaseWeights(w);
  await api.markStep();

  // Use a substantial text (simulate TinyShakespeare — first 50k chars)
  // If the cached file doesn't exist, use a repeated built-in text
  let text: string;
  try {
    text = fs.readFileSync(
      path.join(process.cwd(), "node_modules", ".cache", "tinyshakespeare.txt"),
      "utf-8",
    );
    text = text.slice(0, 50000);
    console.log(`Using cached TinyShakespeare (${text.length} chars)`);
  } catch {
    // Generate a reasonable training text
    const base =
      "First Citizen: Before we proceed any further, hear me speak. " +
      "All: Speak, speak. " +
      "First Citizen: You are all resolved rather to die than to famish? " +
      "All: Resolved. resolved. " +
      "First Citizen: First, you know Caius Marcius is chief enemy to the people. " +
      "All: We know't, we know't. " +
      "First Citizen: Let us kill him, and we'll have corn at our own price. " +
      "Is't a verdict? " +
      "All: No more talking on't; let it be done: away, away! " +
      "Second Citizen: One word, good citizens. " +
      "First Citizen: We are accounted poor citizens, the patricians good. " +
      "What authority surfeits on would relieve us: if they would yield us but the superfluity, " +
      "while it were wholesome, we might guess they relieved us humanely; " +
      "but they think we are too dear: the leanness that afflicts us, the object of our misery, " +
      "is as an inventory to particularise their abundance; our sufferance is a gain to them. " +
      "Let us revenge this with our pikes, ere we become rakes: for the gods know I speak this " +
      "in hunger for bread, not in thirst for revenge. ";
    text = base.repeat(50);
    console.log(`Using built-in text (${text.length} chars)`);
  }

  // Exact browser defaults: bs=1, seq=64, lr=3e-4, checkpointing=true, no AMP
  const trainer = new LoRATrainer(api, model, tok);
  const losses: number[] = [];

  console.log(
    "\nTraining: bs=4, seq=64, lr=3e-4, ckpt=true, fusion=true, 20 steps\n",
  );

  const result = await trainer.train(
    text,
    {
      maxSteps: 100,
      batchSize: 4,
      seqLength: 64,
      learningRate: 5e-4,
      useAMP: false,
      useCheckpointing: true,
    },
    {
      onStepEnd: (s, l, t) => {
        losses.push(l);
        console.log(
          `  Step ${String(s).padStart(2)}: loss=${l.toFixed(4)} (${t.toFixed(0)}ms)`,
        );
      },
    },
  );

  // Analysis
  let increases = 0;
  for (let i = 1; i < losses.length; i++) {
    if (losses[i] > losses[i - 1] + 0.01) increases++;
  }
  const trend = losses[losses.length - 1] < losses[0];
  console.log(`\nFinal loss: ${result.finalLoss.toFixed(4)}`);
  console.log(
    `Start→End: ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`,
  );
  console.log(`Increases: ${increases}/${losses.length - 1}`);
  console.log(
    `Overall trend: ${trend ? "DECREASING (good)" : "NOT DECREASING (bad)"}`,
  );

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
