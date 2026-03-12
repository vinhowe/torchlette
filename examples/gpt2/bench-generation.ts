/**
 * Benchmark: KV-cached vs uncached generation
 *
 * Run with: TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2/bench-generation.ts
 */

import * as path from "node:path";
import { initWebGPU } from "../../src/backend/webgpu";
import { type Tensor, Torchlette } from "../../src/frontend";
import { GPT2Tokenizer } from "./data";
import { loadPretrainedGPT2 } from "./loader";
import { DISTILGPT2_CONFIG, type GPT2, type KVCache } from "./model";

const PROMPT = "To be or not to be";
const MAX_TOKENS = 50;

async function generateUncached(
  api: Torchlette,
  model: GPT2,
  tokens: number[],
): Promise<number> {
  const generated = [...tokens];
  const stride = model.paddedVocabSize;
  const t0 = performance.now();

  for (let i = 0; i < MAX_TOKENS; i++) {
    const contextTokens = generated.slice(-DISTILGPT2_CONFIG.blockSize);

    const logits = api.noGrad(() =>
      api.tidy(() => {
        const input = api.tensorFromArray(contextTokens, [
          1,
          contextTokens.length,
        ]);
        return model.forward(input);
      }),
    );

    const data = await logits.cpu();
    const seqLen = contextTokens.length;
    const startIdx = (seqLen - 1) * stride;
    // Greedy pick for determinism
    let maxVal = -Infinity;
    let nextToken = 0;
    for (let j = 0; j < 50257; j++) {
      if (data[startIdx + j] > maxVal) {
        maxVal = data[startIdx + j];
        nextToken = j;
      }
    }
    generated.push(nextToken);
    logits.dispose();
    await api.markStep();
  }

  return performance.now() - t0;
}

async function generateCached(
  api: Torchlette,
  model: GPT2,
  tokens: number[],
): Promise<number> {
  const generated = [...tokens];
  const stride = model.paddedVocabSize;
  const t0 = performance.now();

  // Prefill
  let kvCache: KVCache[] | undefined;
  {
    const { logits, presentKVs } = api.noGrad(() => {
      const input = api.tensorFromArray(tokens, [1, tokens.length]);
      return model.forwardCached(input);
    });
    kvCache = presentKVs;
    const data = await logits.cpu();
    logits.dispose();

    const startIdx = (tokens.length - 1) * stride;
    let maxVal = -Infinity;
    let nextToken = 0;
    for (let j = 0; j < 50257; j++) {
      if (data[startIdx + j] > maxVal) {
        maxVal = data[startIdx + j];
        nextToken = j;
      }
    }
    generated.push(nextToken);
    await api.markStep();
  }

  // Decode
  for (let i = 1; i < MAX_TOKENS; i++) {
    const lastToken = generated[generated.length - 1];
    const posOffset = generated.length - 1;

    const { logits, presentKVs } = api.noGrad(() => {
      const input = api.tensorFromArray([lastToken], [1, 1]);
      return model.forwardCached(input, kvCache, posOffset);
    });
    kvCache = presentKVs;

    const data = await logits.cpu();
    logits.dispose();

    let maxVal = -Infinity;
    let nextToken = 0;
    for (let j = 0; j < 50257; j++) {
      if (data[j] > maxVal) {
        maxVal = data[j];
        nextToken = j;
      }
    }
    generated.push(nextToken);
    await api.markStep();
  }

  return performance.now() - t0;
}

async function main() {
  const success = await initWebGPU();
  if (!success) {
    console.error("WebGPU not available!");
    process.exit(1);
  }

  const modelDir = path.join(process.cwd(), "models", "distilgpt2");
  const tokenizer = new GPT2Tokenizer();
  await tokenizer.load(modelDir);

  const api = new Torchlette("webgpu");
  const model = await loadPretrainedGPT2(api, modelDir, { dropoutRate: 0.0 });
  model.eval();

  const tokens = tokenizer.encode(PROMPT);
  console.log(
    `Prompt: "${PROMPT}" (${tokens.length} tokens), generating ${MAX_TOKENS} tokens\n`,
  );

  // Warmup
  console.log("Warmup (uncached)...");
  await generateUncached(api, model, tokens);
  console.log("Warmup (cached)...");
  await generateCached(api, model, tokens);

  // Benchmark
  const RUNS = 3;

  console.log(`\nBenchmarking ${RUNS} runs each...\n`);

  const uncachedTimes: number[] = [];
  for (let i = 0; i < RUNS; i++) {
    const ms = await generateUncached(api, model, tokens);
    uncachedTimes.push(ms);
    console.log(`  Uncached run ${i + 1}: ${ms.toFixed(0)}ms`);
  }

  const cachedTimes: number[] = [];
  for (let i = 0; i < RUNS; i++) {
    const ms = await generateCached(api, model, tokens);
    cachedTimes.push(ms);
    console.log(`  Cached run ${i + 1}: ${ms.toFixed(0)}ms`);
  }

  const avgUncached =
    uncachedTimes.reduce((a, b) => a + b) / uncachedTimes.length;
  const avgCached = cachedTimes.reduce((a, b) => a + b) / cachedTimes.length;
  const speedup = avgUncached / avgCached;

  console.log(
    `\nResults (${MAX_TOKENS} tokens from ${tokens.length}-token prompt):`,
  );
  console.log(`  Uncached: ${avgUncached.toFixed(0)}ms avg`);
  console.log(`  Cached:   ${avgCached.toFixed(0)}ms avg`);
  console.log(`  Speedup:  ${speedup.toFixed(2)}x`);
  console.log(
    `  Per-token: ${(avgUncached / MAX_TOKENS).toFixed(1)}ms uncached, ${(avgCached / MAX_TOKENS).toFixed(1)}ms cached`,
  );

  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
