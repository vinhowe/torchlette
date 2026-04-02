/**
 * Run 5 training attempts with different random LoRA inits.
 * Check if NaN is random-seed dependent.
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

const C: GPT2Config = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 12,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0,
};

function loadW() {
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

async function main() {
  await initWebGPU();
  const d = path.join(process.cwd(), "models", "gpt2");
  const tok = new GPT2Tokenizer();
  tok.load(
    JSON.parse(fs.readFileSync(path.join(d, "vocab.json"), "utf-8")),
    fs
      .readFileSync(path.join(d, "merges.txt"), "utf-8")
      .split("\n")
      .filter((l) => l && !l.startsWith("#")),
  );
  const text = fs
    .readFileSync("node_modules/.cache/tinyshakespeare.txt", "utf-8")
    .slice(0, 5000);

  for (let run = 0; run < 5; run++) {
    const api = new Torchlette("webgpu", { enableFusion: true });
    const model = new GPT2WithLoRA(api, C, { rank: 64, alpha: 64 }, "webgpu");
    model.loadBaseWeights(loadW());
    await api.markStep();
    const tokens = tok.encode(text);
    model.train(true);
    const loraParams = model.getLoRAParameters();
    const optimizer = new Adam(loraParams, { lr: 1e-3 }, api);
    let di = 0;
    let nanStep = -1;
    for (let step = 0; step < 20; step++) {
      if (di + 129 > tokens.length) di = 0;
      await api.beginStep();
      const inp = api.tensorFromArray(tokens.slice(di, di + 128), [1, 128], {
        device: "webgpu",
      });
      const tgt = api.tensorFromArray(
        tokens.slice(di + 1, di + 129),
        [1, 128],
        { device: "webgpu" },
      );
      di += 128;
      const { loss } = model.forwardWithLoss(inp, tgt);
      const lv = await loss.item();
      if (isNaN(lv)) {
        nanStep = step;
        break;
      }
      await loss.backward();
      optimizer.step();
      optimizer.zeroGrad();
      inp.dispose();
      tgt.dispose();
      api.endStep();
      await api.markStep();
    }
    if (nanStep >= 0) {
      console.log(`Run ${run}: NaN at step ${nanStep}`);
    } else {
      console.log(`Run ${run}: OK 20 steps`);
    }
  }
  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
