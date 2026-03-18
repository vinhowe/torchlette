import * as fs from "node:fs";
import * as path from "node:path";
import { GPT2WithLoRA } from "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora";
import { GPT2Tokenizer } from "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer";
import { LoRATrainer } from "../examples/gpt2-lora-trainer/src/lib/torchlette/trainer";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
const C = {vocabSize:50257,blockSize:1024,numLayers:12,numHeads:12,embedDim:768,dropoutRate:0};
function loadW() {
  const d = path.join(process.cwd(), "models", "gpt2");
  const buf = fs.readFileSync(path.join(d, "model.safetensors"));
  const hl = Number(new DataView(buf.buffer, buf.byteOffset, 8).getBigUint64(0, true));
  const hdr = JSON.parse(new TextDecoder().decode(buf.subarray(8, 8 + hl)));
  const w = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const [n, m] of Object.entries(hdr) as [string, any][]) {
    if (n === "__metadata__" || m.dtype !== "F32") continue;
    const r = buf.subarray(8 + hl + m.data_offsets[0], 8 + hl + m.data_offsets[1]);
    w.set(n.replace(/^transformer\./, ""), { data: new Float32Array(new Uint8Array(r).slice().buffer), shape: m.shape });
  }
  return w;
}
async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: true });
  const d = path.join(process.cwd(), "models", "gpt2");
  const tok = new GPT2Tokenizer();
  tok.load(JSON.parse(fs.readFileSync(path.join(d, "vocab.json"), "utf-8")),
    fs.readFileSync(path.join(d, "merges.txt"), "utf-8").split("\n").filter(l => l && !l.startsWith("#")));
  const model = new GPT2WithLoRA(api, C, { rank: 64, alpha: 64 }, "webgpu");
  model.loadBaseWeights(loadW());
  await api.markStep();
  const text = fs.readFileSync("node_modules/.cache/tinyshakespeare.txt", "utf-8").slice(0, 5000);
  const trainer = new LoRATrainer(api, model, tok);
  await trainer.train(text, {
    maxSteps: 10, batchSize: 1, seqLength: 128, learningRate: 1e-3,
    useAMP: false, useCheckpointing: false,
  }, { onStepEnd: (s, l) => console.log("step " + s + ": " + (isNaN(l) ? "NaN!" : l.toFixed(4))) });
  await destroyWebGPU(); process.exit(0);
}
main().catch(e => { console.error("ERR:", e.message?.slice(0, 200) ?? e); process.exit(1); });
