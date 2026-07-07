/**
 * Localize the 124M memory blowup. Runs the exact WebGPUGPT2Trainer the DiLoCo
 * agent uses (no relay, no f16 exchange) at 124M, reporting peak/current GPU
 * memory at each phase: after init, after setAnchor, and after each inner round.
 * Tells us whether the ~29GB is the trainer base (model/adam/anchor/activations)
 * vs DiLoCo exchange, and whether it grows per round (leak).
 *
 * Env: NUM_LAYERS NUM_HEADS EMBED_DIM BATCH_SIZE SEQ_LEN STEPS USE_AUTOCAST
 *      CHECKPOINTING GPU_LIMIT_GB
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import {
  type TokenSource,
  WebGPUGPT2Trainer,
} from "../src/distributed/protocol/webgpu-gpt2-trainer";
import {
  getGPUMemoryStats,
  setGPUMemoryLimit,
} from "../src/backend/webgpu/memory-tracker";

const L = parseInt(process.env.NUM_LAYERS ?? "12", 10);
const H = parseInt(process.env.NUM_HEADS ?? "12", 10);
const E = parseInt(process.env.EMBED_DIM ?? "768", 10);
const BATCH = parseInt(process.env.BATCH_SIZE ?? "4", 10);
const SEQ = parseInt(process.env.SEQ_LEN ?? "256", 10);
const STEPS = parseInt(process.env.STEPS ?? "20", 10);
const ROUNDS = parseInt(process.env.ROUNDS ?? "4", 10);
const log = (m: string) => console.error(`[mem] ${m}`);
const GB = 1 / (1024 * 1024 * 1024);
function mem(tag: string) {
  const s = getGPUMemoryStats();
  log(
    `${tag.padEnd(18)} cur=${(s.currentBytes * GB).toFixed(2)}GB peak=${(s.peakBytes * GB).toFixed(2)}GB`,
  );
}

class StubTokens implements TokenSource {
  private data: Uint16Array;
  constructor(n: number) {
    this.data = new Uint16Array(n);
    for (let i = 0; i < n; i++) this.data[i] = (i * 131 + 7) % 50000;
  }
  load(): Uint16Array {
    return this.data;
  }
  async fetch(_min: number): Promise<ArrayLike<number>> {
    return this.data;
  }
}

async function main() {
  if (!(await initWebGPU())) process.exit(1);
  setGPUMemoryLimit(parseFloat(process.env.GPU_LIMIT_GB ?? "31.5") * 1024 ** 3);
  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(42);

  const trainer = new WebGPUGPT2Trainer({
    api,
    modelConfig: {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: L,
      numHeads: H,
      embedDim: E,
      dropoutRate: 0,
    },
    tokenSource: new StubTokens(BATCH * SEQ * STEPS * 4 + 100000),
    innerLr: 5e-4,
    outerLr: 0.7,
    outerMu: 0.9,
    innerSteps: STEPS,
    batchSize: BATCH,
    seqLen: SEQ,
    accumSteps: 1,
    weightDecay: 0.01,
    checkpointing: process.env.CHECKPOINTING !== "0",
    useAutocast: process.env.USE_AUTOCAST !== "0",
    gradClipNorm: 1.0,
    log: () => {},
  });

  log(
    `config: ${L}L/${E}E/${H}H batch=${BATCH} seq=${SEQ} autocast=${process.env.USE_AUTOCAST !== "0"} checkpoint=${process.env.CHECKPOINTING !== "0"}`,
  );
  await trainer.initialize();
  mem("after init");
  await trainer.setAnchor();
  mem("after setAnchor");
  for (let r = 0; r < ROUNDS; r++) {
    const loss = await trainer.innerSteps(r);
    mem(`after round ${r} (loss ${loss.toFixed(3)})`);
  }
  await destroyWebGPU();
  process.exit(0);
}
main().catch((e) => {
  log(`FATAL: ${e?.stack || e}`);
  process.exit(1);
});
