/**
 * Sample text from a DiLoCo checkpoint.
 *
 * Usage:
 *   CHECKPOINT_PATH=ckpts/x.bin PROMPT='Once upon a time' \
 *     NUM_LAYERS=12 NUM_HEADS=12 EMBED_DIM=768 \
 *     npx tsx tools/diloco-sample.ts
 *
 * The model config (layers/heads/embed) must match what the checkpoint
 * was trained with. We don't embed config in the file — fragile to
 * roundtrip, easier to assume the caller knows the shape.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import { loadCheckpoint } from "./diloco-checkpoint";

const CKPT = process.env.CHECKPOINT_PATH;
if (!CKPT) {
  console.error("CHECKPOINT_PATH required");
  process.exit(1);
}
const PROMPT = process.env.PROMPT ?? "Once upon a time";
const MAX_NEW = parseInt(process.env.MAX_NEW_TOKENS ?? "200", 10);
const TEMPERATURE = parseFloat(process.env.TEMPERATURE ?? "0.7");
const TOP_K = parseInt(process.env.TOP_K ?? "40", 10);

const NUM_LAYERS = parseInt(process.env.NUM_LAYERS ?? "12", 10);
const NUM_HEADS = parseInt(process.env.NUM_HEADS ?? "12", 10);
const EMBED_DIM = parseInt(process.env.EMBED_DIM ?? "768", 10);
const MODEL_DIR = process.env.MODEL ?? "gpt2";

const log = (m: string) => console.error(`[sample] ${m}`);

async function main() {
  if (!(await initWebGPU())) {
    log("WebGPU not available");
    process.exit(1);
  }
  const api = new Torchlette("webgpu", { enableFusion: true });

  const { GPT2Tokenizer } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer"
  );
  const vocabPath = path.join(process.cwd(), "models", MODEL_DIR, "vocab.json");
  const mergesPath = path.join(
    process.cwd(),
    "models",
    MODEL_DIR,
    "merges.txt",
  );
  const tokenizer = new GPT2Tokenizer();
  tokenizer.load(
    JSON.parse(fs.readFileSync(vocabPath, "utf-8")),
    fs
      .readFileSync(mergesPath, "utf-8")
      .split("\n")
      .filter((l: string) => l && !l.startsWith("#")),
  );

  const { GPT2WithLoRA } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora"
  );
  const model = new GPT2WithLoRA(
    api,
    {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: NUM_LAYERS,
      numHeads: NUM_HEADS,
      embedDim: EMBED_DIM,
      dropoutRate: 0,
    },
    { rank: 1, alpha: 1 },
    "webgpu",
  );
  model.setFullFinetuning(true);
  // biome-ignore lint/suspicious/noExplicitAny: model param shape isn't typed
  const params: any[] = model.getAllParameters();

  log(`Loading checkpoint from ${CKPT}`);
  const { shapes, tensors } = loadCheckpoint(CKPT);
  if (tensors.length !== params.length) {
    throw new Error(
      `Checkpoint has ${tensors.length} tensors but model has ${params.length}`,
    );
  }
  for (let i = 0; i < params.length; i++) {
    const expected = params[i].shape;
    if (
      shapes[i].length !== expected.length ||
      shapes[i].some((s, j) => s !== expected[j])
    ) {
      throw new Error(
        `Tensor ${i} shape mismatch: ckpt=${JSON.stringify(shapes[i])} model=${JSON.stringify(expected)}`,
      );
    }
  }
  await api.beginStep();
  for (let i = 0; i < params.length; i++) {
    api.copy_(
      params[i],
      api.tensorFromArray(tensors[i], shapes[i], { device: "webgpu" }),
    );
  }
  api.endStep();
  await api.markStep();
  log("Checkpoint loaded");

  const { generateTokens } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/inference"
  );

  process.stdout.write(PROMPT);
  for await (const tok of generateTokens(api, model, tokenizer, PROMPT, {
    maxNewTokens: MAX_NEW,
    temperature: TEMPERATURE,
    topK: TOP_K,
    stopSequences: [],
  })) {
    process.stdout.write(tok);
  }
  process.stdout.write("\n");

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
