/**
 * Task #43 4.4-COVERAGE census — measure build-from-IR bail-class frequency on
 * the three real workloads BEFORE (and after) extending the stream generator.
 *
 * Runs, under TORCHLETTE_COVERAGE_CENSUS=1, each of:
 *   1. distilgpt2 training (from-scratch WebGPUGPT2Trainer, small config —
 *      exercises the same forward/backward/optimizer plan classes as
 *      tools/profile-training.ts without an HF weight download).
 *   2. gpt2 decode (KV-cache decode with per-position narrow offsets —
 *      mirrors tools/t-gpt2-decode-template-count.ts).
 *   3. the 124M DiLoCo regression class (WebGPUGPT2Trainer at embed768 L12 —
 *      the chunked-buffer / large-vocab plan class).
 *
 * Each workload prints its own coverage-census table (reset between workloads),
 * then a combined verdict. This is the measured-frequency oracle: cover the
 * classes that actually fire; skip classes that fire zero times on all three.
 *
 * Run: VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *        npx tsx tools/t-coverage-census.ts
 */

import * as fs from "node:fs";
import type { KVCache } from "../examples/gpt2/model";
import { DISTILGPT2_CONFIG, GPT2 } from "../examples/gpt2/model";
import { getWebGPUInitError, initWebGPU } from "../src/backend/webgpu";
import {
  type TokenSource,
  WebGPUGPT2Trainer,
} from "../src/distributed/protocol/webgpu-gpt2-trainer";
import {
  dumpCoverageCensus,
  resetCoverageCensus,
} from "../src/executor/executor";
import { Torchlette } from "../src/frontend/torchlette";

const TOKENS_PATH =
  process.env.LOCAL_TOKENS ??
  "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin";

class FullCacheTokenSource implements TokenSource {
  private cached: Uint16Array | null = null;
  constructor(private readonly path: string) {}
  load(): Uint16Array {
    if (this.cached) return this.cached;
    const buf = fs.readFileSync(this.path);
    this.cached = new Uint16Array(
      buf.buffer,
      buf.byteOffset,
      Math.floor(buf.byteLength / 2),
    );
    return this.cached;
  }
  async fetch(_minTokens: number): Promise<ArrayLike<number>> {
    return this.load();
  }
}

async function trainCensus(
  label: string,
  cfg: { numLayers: number; numHeads: number; embedDim: number },
  batchSize: number,
  seqLen: number,
  rounds: number,
): Promise<string> {
  resetCoverageCensus();
  const api = new Torchlette("webgpu", { enableFusion: true });
  const tokenSource = new FullCacheTokenSource(TOKENS_PATH);
  const trainer = new WebGPUGPT2Trainer({
    api,
    modelConfig: {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: cfg.numLayers,
      numHeads: cfg.numHeads,
      embedDim: cfg.embedDim,
      dropoutRate: 0,
    },
    tokenSource,
    innerLr: 5e-4,
    outerLr: 1.0,
    outerMu: 0.0,
    innerSteps: 20,
    batchSize,
    seqLen,
    accumSteps: 1,
    weightDecay: 0.01,
    checkpointing: process.env.CHECKPOINTING !== "0",
    useAutocast: process.env.USE_AUTOCAST !== "0",
    gradClipNorm: process.env.GRAD_CLIP === "0" ? undefined : 1.0,
    log: () => {},
  });
  await trainer.initialize();
  await trainer.setAnchor();
  for (let r = 0; r < rounds; r++) {
    const loss = await trainer.innerSteps(r);
    console.error(`[census:${label}] round ${r} loss=${loss.toFixed(3)}`);
  }
  console.log(`\n===== CENSUS: ${label} =====`);
  return dumpCoverageCensus();
}

async function decodeCensus(steps: number): Promise<string> {
  resetCoverageCensus();
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = new GPT2(api, { ...DISTILGPT2_CONFIG });
  const PROMPT = [40, 716, 257];
  let pastKVs: KVCache[] | undefined;

  const firstLogits = api.noGrad(() => {
    const idx = api.tensorFromArray(PROMPT, [1, PROMPT.length]);
    const { logits, presentKVs } = model.forwardCached(idx, undefined, 0);
    pastKVs = presentKVs;
    return logits;
  });
  await firstLogits.cpu();
  firstLogits.dispose();
  let posOffset = PROMPT.length;
  await api.markStep();

  let tok = 1;
  for (let i = 0; i < steps; i++) {
    const logits = api.noGrad(() => {
      const idx = api.tensorFromArray([tok], [1, 1]);
      const { logits: lg, presentKVs } = model.forwardCached(
        idx,
        pastKVs,
        posOffset,
      );
      pastKVs = presentKVs;
      return lg;
    });
    const data = (await logits.cpu()) as Float32Array;
    logits.dispose();
    let best = 0;
    for (let v = 1; v < model.config.vocabSize; v++)
      if (data[v] > data[best]) best = v;
    tok = best;
    posOffset += 1;
    await api.markStep();
  }
  console.log(`\n===== CENSUS: gpt2-decode =====`);
  return dumpCoverageCensus();
}

async function main() {
  if (!(await initWebGPU()))
    throw new Error(getWebGPUInitError() || "WebGPU init failed");

  const results: Record<string, string> = {};

  // NOTE: the production DiLoCo trainer runs checkpointing=true, which ties the
  // buffer arena OFF (commit b66ead7) → those plans run LOWERED (no compiled
  // replay, no build-from-IR, no recorded build). The coverage question is only
  // live in the arena-enabled / compiled-replay configuration, so the training
  // workloads are measured with CHECKPOINTING=0 (the config that actually
  // exercises build-from-IR and the recorded fallback). Set CENSUS_CKPT=1 to
  // additionally confirm the checkpointing=true path is fully lowered.
  process.env.CHECKPOINTING = process.env.CENSUS_CKPT === "1" ? "1" : "0";

  // Workload 1: distilgpt2-class training (small, no HF download).
  results["distil-train"] = await trainCensus(
    "distil-train",
    { numLayers: 6, numHeads: 12, embedDim: 768 },
    1,
    256,
    2,
  );

  // Workload 2: gpt2 decode (per-position narrow offsets).
  results["gpt2-decode"] = await decodeCensus(8);

  // Workload 3: 124M DiLoCo regression class (large vocab / chunked buffers).
  results["124M-diloco"] = await trainCensus(
    "124M-diloco",
    { numLayers: 12, numHeads: 12, embedDim: 768 },
    2,
    256,
    2,
  );

  console.log("\n\n========== COMBINED SUMMARY ==========");
  for (const [k, v] of Object.entries(results)) {
    console.log(`\n--- ${k} ---\n${v}`);
  }
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
