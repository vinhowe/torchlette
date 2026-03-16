import * as fs from "fs";
import * as path from "path";
import {
  type GPT2Config,
  GPT2WithLoRA,
} from "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora";
import { initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";

async function main() {
  await initWebGPU();
  const api = new Torchlette("webgpu", { enableFusion: false });

  const config: GPT2Config = {
    vocabSize: 50257,
    blockSize: 1024,
    numLayers: 6,
    numHeads: 12,
    embedDim: 768,
    dropoutRate: 0,
  };
  const model = new GPT2WithLoRA(api, config, { rank: 4, alpha: 4 }, "webgpu");

  // Load weights
  const modelPath = "models/distilgpt2/model.safetensors";
  const buf = fs.readFileSync(modelPath);
  const headerLen = Number(
    new DataView(buf.buffer, buf.byteOffset, 8).getBigUint64(0, true),
  );
  const header = JSON.parse(
    new TextDecoder().decode(buf.subarray(8, 8 + headerLen)),
  );
  const dataOffset = 8 + headerLen;

  const weights = new Map<string, { data: Float32Array; shape: number[] }>();
  for (const [name, meta] of Object.entries(header) as [string, any][]) {
    if (name === "__metadata__") continue;
    const { dtype, shape, data_offsets } = meta;
    const [start, end] = data_offsets;
    const raw = buf.subarray(dataOffset + start, dataOffset + end);
    const copy = new Uint8Array(raw).slice();
    if (dtype !== "F32") {
      console.log("skip", name, dtype);
      continue;
    }
    weights.set(name, { data: new Float32Array(copy.buffer), shape });
  }

  // Check what weight names are available vs what model expects
  console.log("\nWeight names in safetensors:");
  for (const name of weights.keys()) {
    console.log(" ", name);
  }

  model.loadBaseWeights(weights);

  // Check embedding after load
  const embData = await model.wte.weight.cpu();
  console.log(
    "\nEmbedding weight first 10:",
    Array.from(embData.slice(0, 10)).map((v) => v.toFixed(4)),
  );
  console.log(
    "All zeros?",
    embData.every((v) => v === 0),
  );

  // Check first layer ln1 after load
  const ln1Data = await (model as any).h[0].ln1.weight.cpu();
  console.log(
    "\nLN1 weight first 10:",
    Array.from(ln1Data.slice(0, 10)).map((v) => v.toFixed(4)),
  );

  process.exit(0);
}
main().catch((e) => {
  console.error(e);
  process.exit(1);
});
