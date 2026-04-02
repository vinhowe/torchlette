/**
 * Compare gradients: fusion on vs off, after step 0.
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

async function getGrads(fusion: boolean) {
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

  await api.beginStep();
  const input = api.tensorFromArray(tokens.slice(0, 128), [1, 128], {
    device: "webgpu",
  });
  const target = api.tensorFromArray(tokens.slice(1, 129), [1, 128], {
    device: "webgpu",
  });
  const { loss } = model.forwardWithLoss(input, target);
  const lv = await loss.item();
  await loss.backward();

  const grads: Record<string, { norm: number; first5: number[] }> = {};
  for (let i = 0; i < 12; i++) {
    const B = loraParams[i * 2 + 1];
    const bg = B.grad ? await B.grad.cpu() : null;
    if (bg) {
      grads[`block${i}_B`] = {
        norm: Math.sqrt(Array.from(bg).reduce((s, v) => s + v * v, 0)),
        first5: Array.from(bg.slice(0, 5)),
      };
    }
  }

  api.endStep();
  await api.markStep();
  return { loss: lv, grads };
}

async function main() {
  await initWebGPU();
  const nf = await getGrads(false);
  const f = await getGrads(true);

  console.log(
    `Loss: no-fusion=${nf.loss.toFixed(6)} fusion=${f.loss.toFixed(6)} diff=${Math.abs(nf.loss - f.loss).toExponential(2)}`,
  );
  console.log();
  for (let i = 0; i < 12; i++) {
    const k = `block${i}_B`;
    const a = nf.grads[k],
      b = f.grads[k];
    if (!a || !b) continue;
    const normDiff = Math.abs(a.norm - b.norm);
    const relDiff = normDiff / Math.max(a.norm, 1e-10);
    const valDiffs = a.first5.map((v, j) => Math.abs(v - b.first5[j]));
    const maxValDiff = Math.max(...valDiffs);
    const status = relDiff > 0.01 ? "*** MISMATCH" : "ok";
    console.log(
      `block${i} B_grad: normDiff=${normDiff.toExponential(2)} relDiff=${relDiff.toExponential(2)} maxValDiff=${maxValDiff.toExponential(2)} ${status}`,
    );
  }

  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
