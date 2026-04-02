/**
 * Compare LoRA B weights after 2 Adam steps: fusion on vs off.
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

async function run(
  fusion: boolean,
): Promise<{ weights: Float32Array[]; losses: number[] }> {
  const api = new Torchlette("webgpu", { enableFusion: fusion });
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
  const loraParams = model.getLoRAParameters();
  const runtime = api._runtime();
  for (let i = 0; i < loraParams.length; i += 2) {
    const a = loraParams[i];
    const sz = a.shape.reduce((x: number, y: number) => x * y, 1);
    const t = api.tensorFromArray(new Float32Array(sz).fill(0.01), a.shape, {
      device: "webgpu",
    });
    runtime.copy_(a._unwrap(), t._unwrap());
  }
  await api.markStep();

  const text = fs
    .readFileSync("node_modules/.cache/tinyshakespeare.txt", "utf-8")
    .slice(0, 5000);
  const tokens = tok.encode(text);
  model.train(true);
  const optimizer = new Adam(loraParams, { lr: 1e-3 }, api);
  const sl = 128;
  let di = 0;
  const losses: number[] = [];

  for (let step = 0; step < 2; step++) {
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
    losses.push(await loss.item());
    await loss.backward();
    optimizer.step();
    optimizer.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
  }

  // Read B0 and A0 weights after 2 steps
  const weights: Float32Array[] = [];
  for (let i = 0; i < 2; i++) {
    weights.push(new Float32Array(await loraParams[i].cpu()));
  }
  return { weights, losses };
}

async function main() {
  await initWebGPU();
  const nf = await run(false);
  const f = await run(true);

  console.log(
    `Step 0 loss: nf=${nf.losses[0].toFixed(6)} f=${f.losses[0].toFixed(6)}`,
  );
  console.log(
    `Step 1 loss: nf=${nf.losses[1].toFixed(6)} f=${f.losses[1].toFixed(6)}`,
  );

  for (let i = 0; i < 2; i++) {
    const label = i === 0 ? "A0" : "B0";
    let maxDiff = 0;
    for (let j = 0; j < nf.weights[i].length; j++) {
      maxDiff = Math.max(maxDiff, Math.abs(nf.weights[i][j] - f.weights[i][j]));
    }
    const nfNorm = Math.sqrt(
      Array.from(nf.weights[i]).reduce((s, v) => s + v * v, 0),
    );
    const fNorm = Math.sqrt(
      Array.from(f.weights[i]).reduce((s, v) => s + v * v, 0),
    );
    console.log(
      `${label} after 2 steps: maxDiff=${maxDiff.toExponential(2)} nfNorm=${nfNorm.toFixed(4)} fNorm=${fNorm.toFixed(4)}`,
    );
    console.log(
      `  nf[:5]=${Array.from(nf.weights[i].slice(0, 5)).map((v) => v.toExponential(4))}`,
    );
    console.log(
      `  f [:5]=${Array.from(f.weights[i].slice(0, 5)).map((v) => v.toExponential(4))}`,
    );
  }

  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
