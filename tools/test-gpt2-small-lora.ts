/**
 * GPT-2 small (12 layers) LoRA training + generation test.
 * Matches PyTorch baseline: rank=64, Adam lr=1e-3, 200 steps.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import {
  type GPT2Config,
  GPT2WithLoRA,
} from "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora";
import { generateTokens } from "../examples/gpt2-lora-trainer/src/lib/torchlette/inference";
import { GPT2Tokenizer } from "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim";

const CONFIG: GPT2Config = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 12,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0,
};

async function main() {
  await initWebGPU();
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

  const text = fs
    .readFileSync("node_modules/.cache/tinyshakespeare.txt", "utf-8")
    .slice(0, 5000);
  const tokens = tok.encode(text);
  console.log(`GPT-2 small (12L), rank=64, ${tokens.length} tokens\n`);

  // Generate BEFORE
  console.log("=== Before Training ===");
  model.train(false);
  for (const prompt of ["First Citizen:", "We are"]) {
    let gen = "";
    for await (const t of generateTokens(api, model, tok, prompt, {
      maxNewTokens: 40,
      temperature: 0.7,
      topK: 40,
    }))
      gen += t;
    console.log(`"${prompt}${gen.replace(/\n/g, "\\n")}"\n`);
  }

  // Train
  model.train(true);
  const loraParams = model.getLoRAParameters();
  const optimizer = new Adam(loraParams, { lr: 1e-3 }, api);
  const sl = 128;
  let di = 0;

  console.log("=== Training 200 steps ===");
  for (let step = 0; step < 200; step++) {
    if (di + sl + 1 > tokens.length) di = 0;
    await api.beginStep();
    const input = api.tensorFromArray(tokens.slice(di, di + sl), [1, sl], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(
      tokens.slice(di + 1, di + sl + 1),
      [1, sl],
      {
        device: "webgpu",
      },
    );
    di += sl;
    const { loss } = model.forwardWithLoss(input, target);
    const lv = await loss.item();
    await loss.backward();
    optimizer.step();
    optimizer.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
    if (step % 50 === 0 || step === 199)
      console.log(`  Step ${step}: loss=${lv.toFixed(4)}`);
  }

  // Generate AFTER
  console.log("\n=== After Training ===");
  model.train(false);
  for (const prompt of ["First Citizen:", "We are", "MENENIUS:", "All:"]) {
    let gen = "";
    for await (const t of generateTokens(api, model, tok, prompt, {
      maxNewTokens: 50,
      temperature: 0.7,
      topK: 40,
    }))
      gen += t;
    console.log(`"${prompt}${gen}"\n`);
  }

  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
