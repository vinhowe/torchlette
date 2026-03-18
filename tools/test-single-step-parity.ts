/**
 * Single training step comparison: torchlette vs PyTorch reference values.
 * 1-layer GPT-2 on 4 tokens, rank=4, deterministic LoRA init.
 */
import * as fs from "node:fs";
import * as path from "node:path";
import {
  type GPT2Config,
  GPT2WithLoRA,
} from "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam } from "../src/optim";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: false });
  const CONFIG: GPT2Config = {
    vocabSize: 50257,
    blockSize: 1024,
    numLayers: 12,
    numHeads: 12,
    embedDim: 768,
    dropoutRate: 0,
  };

  const d = path.join(process.cwd(), "models", "gpt2");
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

  // Set deterministic LoRA init
  const loraParams = model.getLoRAParameters();
  const runtime = api._runtime();
  for (let i = 0; i < loraParams.length; i += 2) {
    const loraA = loraParams[i];
    const aSize = loraA.shape.reduce((a: number, b: number) => a * b, 1);
    const aTensor = api.tensorFromArray(
      new Float32Array(aSize).fill(0.01),
      loraA.shape,
      { device: "webgpu" },
    );
    runtime.copy_(loraA._unwrap(), aTensor._unwrap());
  }
  await api.markStep();

  model.train(true);
  const optimizer = new Adam(loraParams, { lr: 1e-3 }, api);

  // Simple tokens: "Hello, world is"
  const tokens = [15496, 11, 995, 318, 257, 1049, 1295, 284]; // 8 tokens
  const sl = 7;

  // Step 0
  await api.beginStep();
  const input = api.tensorFromArray(tokens.slice(0, sl), [1, sl], {
    device: "webgpu",
  });
  const target = api.tensorFromArray(tokens.slice(1, sl + 1), [1, sl], {
    device: "webgpu",
  });
  const { loss } = model.forwardWithLoss(input, target);
  const lv = await loss.item();
  await loss.backward();

  // Read B0 gradient
  const b0Grad = loraParams[1].grad ? await loraParams[1].grad.cpu() : null;
  const b0Norm = b0Grad
    ? Math.sqrt(Array.from(b0Grad).reduce((s, v) => s + v * v, 0))
    : 0;

  console.log(`Step 0: loss=${lv.toFixed(6)}`);
  console.log(`B0 grad norm: ${b0Norm.toFixed(6)}`);
  if (b0Grad) {
    console.log(
      `B0 grad[:5]: ${Array.from(b0Grad.slice(0, 5)).map((v) => v.toExponential(4))}`,
    );
  }

  optimizer.step();
  optimizer.zeroGrad();
  input.dispose();
  target.dispose();
  api.endStep();
  await api.markStep();

  // Steps 1-4
  let di = sl;
  for (let step = 1; step < 5; step++) {
    if (di + sl + 1 > tokens.length) di = 0;
    await api.beginStep();
    const inp = api.tensorFromArray(tokens.slice(di, di + sl), [1, sl], {
      device: "webgpu",
    });
    const tgt = api.tensorFromArray(
      tokens.slice(di + 1, di + sl + 1),
      [1, sl],
      { device: "webgpu" },
    );
    di += sl;
    const res = model.forwardWithLoss(inp, tgt);
    const l = await res.loss.item();
    await res.loss.backward();
    optimizer.step();
    optimizer.zeroGrad();
    inp.dispose();
    tgt.dispose();
    api.endStep();
    await api.markStep();
    console.log(`Step ${step}: loss=${l.toFixed(6)}`);
  }

  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
