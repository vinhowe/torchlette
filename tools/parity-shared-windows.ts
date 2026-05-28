/**
 * Cross-framework parity driver.
 *
 * Runs the production WebGPUGPT2Trainer with window starts injected from a
 * shared offsets file (tools/diloco-pytorch/gen_offsets.py), so torchlette
 * and the PyTorch baseline train on bit-identical token windows. Prints
 * STATS lines matching the PyTorch script so the two curves diff directly.
 *
 * Any persisting loss gap after sharing data is attributable to numerics
 * (kernel differences / init), not data ordering.
 *
 *   VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim \
 *   WINDOW_OFFSETS=ckpts/window-offsets.i32 SEED=42 \
 *     npx tsx tools/parity-shared-windows.ts
 */

import * as fs from "node:fs";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import {
  type TokenSource,
  WebGPUGPT2Trainer,
} from "../src/distributed/protocol/webgpu-gpt2-trainer";

const SEED = parseInt(process.env.SEED ?? "42", 10);
const ROUNDS = parseInt(process.env.ROUNDS ?? "30", 10);
const STEPS = parseInt(process.env.STEPS ?? "20", 10);
const BATCH = parseInt(process.env.BATCH_SIZE ?? "8", 10);
const SEQ = parseInt(process.env.SEQ_LEN ?? "256", 10);
const TOKENS_PATH =
  process.env.LOCAL_TOKENS ??
  "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin";
const OFFSETS_PATH =
  process.env.WINDOW_OFFSETS ??
  "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/window-offsets.i32";

const log = (m: string) => console.error(`[parity] ${m}`);

class FullCacheTokenSource implements TokenSource {
  private cached: Uint16Array | null = null;
  constructor(private readonly path: string) {}
  load(): Uint16Array {
    if (this.cached) return this.cached;
    const buf = fs.readFileSync(this.path);
    this.cached = new Uint16Array(
      buf.buffer,
      buf.byteOffset,
      buf.byteLength / 2,
    );
    return this.cached;
  }
  async fetch(): Promise<ArrayLike<number>> {
    return this.load();
  }
}

async function main() {
  if (!(await initWebGPU())) {
    log("FATAL: WebGPU init failed");
    process.exit(1);
  }
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "4000";

  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(SEED);

  const tokenSource = new FullCacheTokenSource(TOKENS_PATH);
  tokenSource.load();

  // Shared window offsets, int32. Indexed by global inner-step:
  //   base = stepIndex * batchSize;  starts = offsets[base : base+batchSize]
  const ob = fs.readFileSync(OFFSETS_PATH);
  const offsets = new Int32Array(ob.buffer, ob.byteOffset, ob.byteLength / 4);
  log(`shared window offsets: ${offsets.length} from ${OFFSETS_PATH}`);

  const trainer = new WebGPUGPT2Trainer({
    api,
    modelConfig: {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: 8,
      numHeads: 4,
      embedDim: 128,
      dropoutRate: 0,
    },
    tokenSource,
    innerLr: 5e-4,
    outerLr: 1.0,
    outerMu: 0.0,
    innerSteps: STEPS,
    batchSize: BATCH,
    seqLen: SEQ,
    accumSteps: 1,
    weightDecay: 0.01,
    checkpointing: true,
    useAutocast: process.env.USE_AUTOCAST !== "0",
    gradClipNorm: 1.0,
    sampleWindowStarts: (stepIndex, batchSize, maxStart) => {
      const base = stepIndex * batchSize;
      const out: number[] = [];
      for (let b = 0; b < batchSize; b++) {
        // Clamp defensively; offsets were generated against the same
        // maxStart so this should be a no-op.
        out.push(Math.min(offsets[base + b]!, maxStart - 1));
      }
      return out;
    },
    log: (m: string) => log(`trainer: ${m}`),
  });
  await trainer.initialize();
  await trainer.setAnchor();

  for (let r = 0; r < ROUNDS; r++) {
    const t0 = Date.now();
    const loss = await trainer.innerSteps(r);
    console.error(
      `STATS ${JSON.stringify({
        round: r,
        loss: +loss.toFixed(4),
        round_s: +((Date.now() - t0) / 1000).toFixed(2),
      })}`,
    );
  }

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
