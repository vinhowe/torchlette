/**
 * Gemma-2 generation smoke (M2): greedy static-KV decode via the generalized
 * stack (prefill + taped per-token decode). Reports the generated ids/text,
 * tape hit count, and tokens/sec.
 *
 * The tokenizer is loaded from the HF snapshot via @huggingface/transformers.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *      TORCHLETTE_STEP_TAPE=1 npx tsx examples/gemma2/generate-smoke.ts "The capital of France is" 24
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  getWebGPUInitError,
  initWebGPU,
  setGPUMemoryLimit,
} from "../../src/backend/webgpu";
import { Torchlette, type Tensor } from "../../src/frontend/torchlette";
import { kvBucketLen } from "../../packages/gemma2-browser/src/model";
import { loadPretrainedGemma2 } from "./loader";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/gemma-2-2b");

async function main() {
  const promptText = process.argv[2] ?? "The capital of France is";
  const numNew = Number(process.argv[3] ?? 24);
  const outFile = process.argv[4];

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  setGPUMemoryLimit(30 * 1024 * 1024 * 1024);
  const api = new Torchlette("webgpu", { enableFusion: true });

  const weightDtype = (process.env.GEMMA2_DTYPE === "f32" ? "f32" : "f16") as
    | "f32"
    | "f16";
  const model = await loadPretrainedGemma2(api, MODEL_DIR, {
    maxSeqLen: 256,
    weightDtype,
  });
  const vocab = model.config.vocabSize;

  // Tokenizer via @huggingface/transformers (Node). Falls back to a raw
  // <bos>-prefixed id list from argv if the tokenizer package isn't linked at
  // the workspace root (it's a per-example dep). Pass a JSON id array as argv[2]
  // to bypass tokenization: e.g. '[2,651,6037,576,6081,603]'.
  let tokenizer:
    | { encode(t: string): number[]; decode(ids: number[], o?: unknown): string }
    | null = null;
  let enc: number[];
  try {
    const { AutoTokenizer } = await import("@huggingface/transformers");
    // biome-ignore lint: dynamic
    tokenizer = (await AutoTokenizer.from_pretrained(MODEL_DIR)) as never;
  } catch {
    console.warn("(@huggingface/transformers not linked at root — id-list mode)");
  }
  if (promptText.trim().startsWith("[")) {
    enc = JSON.parse(promptText);
  } else if (tokenizer) {
    enc = tokenizer.encode(promptText);
  } else {
    throw new Error(
      "No tokenizer available; pass a JSON token-id array as the prompt arg",
    );
  }
  console.log(`prompt -> ${enc.length} tokens: ${JSON.stringify(enc.slice(0, 12))}...`);

  const staticKV = model.allocStaticKV(256);
  const prevScope = api.setStepScopedCleanup(true);
  const genIds: number[] = [];
  const argmaxFrom = async (logits: Tensor, pos: number): Promise<number> => {
    const top = await api.readTopK(logits, 1, {
      offset: pos * vocab,
      length: vocab,
    });
    logits.dispose();
    return top.indices[0];
  };

  try {
    await api.markStep();
    // Prefill.
    const t0 = Date.now();
    let nextTok: number;
    {
      const idx = api.tensorFromArray(enc, [1, enc.length]);
      const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
      nextTok = await argmaxFrom(logits, enc.length - 1);
      await api.markStep();
    }
    const prefillMs = Date.now() - t0;

    // Taped decode.
    const decode = api.capture(
      (idx: Tensor) =>
        api.noGrad(() => model.forward(idx, { staticKV }).logits),
      {
        key: () =>
          `kv:bkt${kvBucketLen(staticKV.len + 1, 256)}:mod${model.attnModKey}`,
      },
    );
    const tDecode0 = Date.now();
    let count = 0;
    while (count < numNew && nextTok !== 1 /* <eos> */ && nextTok !== 107 /* <end_of_turn> */) {
      genIds.push(nextTok);
      count++;
      const logits = (await decode(
        api.tensorFromArray([nextTok], [1, 1]),
      )) as Tensor;
      nextTok = await argmaxFrom(logits, 0);
      await api.markStep();
    }
    const decodeSec = (Date.now() - tDecode0) / 1000;

    const text = tokenizer
      ? tokenizer.decode(genIds, { skip_special_tokens: true })
      : "(no tokenizer — ids only)";
    const stats = decode.stats();
    console.log(`\ngenerated ids: ${JSON.stringify(genIds)}`);
    console.log(`generated text: ${JSON.stringify(text)}`);
    console.log(
      `prefill=${prefillMs}ms  decode=${count} toks in ${decodeSec.toFixed(2)}s  ` +
        `= ${(count / Math.max(decodeSec, 0.001)).toFixed(1)} tok/s`,
    );
    console.log(
      `tape: hits=${stats.hits} calls=${stats.calls} traces=${stats.traces} ready=${stats.ready}`,
    );
    if (outFile) {
      const fs = await import("node:fs");
      fs.writeFileSync(
        outFile,
        JSON.stringify({ genIds, text, tokPerSec: count / decodeSec, tape: stats }),
      );
    }
  } finally {
    api.setStepScopedCleanup(prevScope);
  }
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
