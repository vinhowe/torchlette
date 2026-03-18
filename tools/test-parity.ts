/**
 * Parity test: compare torchlette vs PyTorch for LoRA training step-by-step.
 * Uses deterministic LoRA init (ones*0.01 for A, zeros for B).
 * Prints loss + gradient norms for direct comparison.
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
  numLayers: 12,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0,
};

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: false }); // No fusion for parity
  const d = path.join(process.cwd(), "models", "gpt2");
  const tok = new GPT2Tokenizer();
  tok.load(
    JSON.parse(fs.readFileSync(path.join(d, "vocab.json"), "utf-8")),
    fs
      .readFileSync(path.join(d, "merges.txt"), "utf-8")
      .split("\n")
      .filter((l) => l && !l.startsWith("#")),
  );

  // Create model with rank=64 but we'll override LoRA init
  const model = new GPT2WithLoRA(
    api,
    CONFIG,
    { rank: 64, alpha: 64 },
    "webgpu",
  );

  // Load base weights
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

  // Override LoRA A to deterministic values (ones * 0.01)
  // Override LoRA B to zeros (already default)
  const loraParams = model.getLoRAParameters();
  const runtime = api._runtime();
  for (let i = 0; i < loraParams.length; i += 2) {
    const loraA = loraParams[i];
    const loraB = loraParams[i + 1];
    // Set A = ones * 0.01
    const aSize = loraA.shape.reduce((a: number, b: number) => a * b, 1);
    const aData = new Float32Array(aSize).fill(0.01);
    const aTensor = api.tensorFromArray(aData, loraA.shape, {
      device: "webgpu",
    });
    runtime.copy_(loraA._unwrap(), aTensor._unwrap());
    // B is already zeros from init
  }
  await api.markStep();

  const text = fs
    .readFileSync("node_modules/.cache/tinyshakespeare.txt", "utf-8")
    .slice(0, 5000);
  const tokens = tok.encode(text);
  console.log(`Tokens: ${tokens.length}`);

  model.train(true);
  const optimizer = new Adam(loraParams, { lr: 1e-3 }, api);
  const sl = 128;
  let di = 0;

  for (let step = 0; step < 5; step++) {
    if (di + sl + 1 > tokens.length) di = 0;
    await api.beginStep();
    const input = api.tensorFromArray(tokens.slice(di, di + sl), [1, sl], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(
      tokens.slice(di + 1, di + sl + 1),
      [1, sl],
      { device: "webgpu" },
    );
    di += sl;
    const { loss } = model.forwardWithLoss(input, target);
    const lv = await loss.item();
    await loss.backward();

    // Read gradient norms
    const aGrad = loraParams[0].grad;
    const bGrad = loraParams[1].grad;
    const aGradData = aGrad ? await aGrad.cpu() : new Float32Array(0);
    const bGradData = bGrad ? await bGrad.cpu() : new Float32Array(0);
    const aNorm = Math.sqrt(
      Array.from(aGradData).reduce((s, v) => s + v * v, 0),
    );
    const bNorm = Math.sqrt(
      Array.from(bGradData).reduce((s, v) => s + v * v, 0),
    );

    console.log(
      `Step ${step}: loss=${lv.toFixed(6)} B0.A_grad_norm=${aNorm.toFixed(6)} B0.B_grad_norm=${bNorm.toFixed(6)}`,
    );
    if (step === 0) {
      console.log(
        `  B0.B grad[:5]: ${Array.from(bGradData.slice(0, 5)).map((v) => v.toExponential(6))}`,
      );
    }

    optimizer.step();
    optimizer.zeroGrad();

    if (step === 0) {
      // Read weights after first step
      const bData = await loraParams[1].cpu();
      const aData = await loraParams[0].cpu();
      console.log(
        `  B0.B weight[:5] after: ${Array.from(bData.slice(0, 5)).map((v) => v.toExponential(6))}`,
      );
      console.log(
        `  B0.A weight[:5] after: ${Array.from(aData.slice(0, 5)).map((v) => v.toExponential(6))}`,
      );
    }

    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
  }

  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
