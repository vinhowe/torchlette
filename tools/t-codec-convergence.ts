/**
 * Codec convergence gate (task #46): does a wire codec preserve the DiLoCo
 * trajectory when it round-trips the PSEUDO-GRADIENTS each outer step?
 *
 * The E3M0 spec only checks quantize/dequantize round-trip error (≤50% relRMSE
 * by its own bound!) — never training convergence. f16 (wire-codec) round-trips
 * at ~1e-3. This harness answers the real question the task poses: run two
 * in-process peers of the PRODUCTION WebGPUGPT2Trainer (shared init + shared
 * token stream → lockstep), and each round average pseudo-grads AFTER routing
 * each peer's grad through the codec. Compare the loss trajectory across
 * {none (f32 identity), f16, e3m0}. A codec that harms convergence shows up as
 * a higher loss curve vs the f32 control.
 *
 * Run: VULKAN_DEVICE_INDEX=0 LD_LIBRARY_PATH=tools/vk-shim \
 *        CODEC=none|f16|e3m0 npx tsx tools/t-codec-convergence.ts
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { Torchlette } from "../src/frontend/torchlette";
import {
  type TokenSource,
  WebGPUGPT2Trainer,
} from "../src/distributed/protocol/webgpu-gpt2-trainer";
import { encodeTensors, decodeTensors } from "../src/distributed/wire-codec";
import { e3m0Quantize, e3m0Dequantize } from "../src/distributed/e3m0";

const ROUNDS = parseInt(process.env.ROUNDS ?? "10", 10);
const CODEC = (process.env.CODEC ?? "none") as "none" | "f16" | "e3m0";
const SEED = 42;
const TOKENS_PATH =
  process.env.LOCAL_TOKENS ??
  "/mnt/pccfs2/backed_up/vin/dev/torchlette/ckpts/tinystories-tokens.bin";
const log = (m: string) => console.error(`[codec-conv ${CODEC}] ${m}`);

/** Round-trip one tensor through the selected codec (mirrors the wire path). */
function roundTrip(arr: Float32Array): Float32Array {
  if (CODEC === "none") return arr;
  if (CODEC === "f16") {
    const enc = encodeTensors([arr], "f16");
    return decodeTensors(enc, [[arr.length]], "f16")[0];
  }
  // e3m0: pad to multiple of 8 (the browser's compressPseudoGrads contract).
  const padded = new Float32Array(Math.ceil(arr.length / 8) * 8);
  padded.set(arr);
  const { codes, scales } = e3m0Quantize(padded);
  return e3m0Dequantize(codes, scales, padded.length).slice(0, arr.length);
}

import * as fs from "node:fs";
function loadTokens(): Uint16Array {
  const buf = fs.readFileSync(TOKENS_PATH);
  return new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
}

class FixedTokens implements TokenSource {
  constructor(private data: Uint16Array) {}
  load(): Uint16Array {
    return this.data;
  }
  async fetch(): Promise<ArrayLike<number>> {
    return this.data;
  }
}

function makeTrainer(api: Torchlette, tokens: Uint16Array): WebGPUGPT2Trainer {
  return new WebGPUGPT2Trainer({
    api,
    modelConfig: {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: 8,
      numHeads: 4,
      embedDim: 128,
      dropoutRate: 0,
    },
    tokenSource: new FixedTokens(tokens),
    innerLr: 5e-4,
    outerLr: 1.0,
    outerMu: 0.0,
    innerSteps: 20,
    batchSize: 8,
    seqLen: 256,
    accumSteps: 1,
    weightDecay: 0.01,
    checkpointing: true,
    useAutocast: true,
    gradClipNorm: 1.0,
    log: () => {},
  });
}

async function main() {
  if (!(await initWebGPU())) {
    log("FATAL: WebGPU init failed");
    process.exit(1);
  }
  const tokens = loadTokens();
  log(`tokens=${tokens.length.toLocaleString()} rounds=${ROUNDS}`);

  // Two peers, shared init (same seed) + shared token stream → lockstep, so any
  // trajectory divergence from the f32 control is the codec's doing.
  const apiA = new Torchlette("webgpu", { enableFusion: true });
  apiA.manualSeed(SEED);
  const apiB = new Torchlette("webgpu", { enableFusion: true });
  apiB.manualSeed(SEED);
  const A = makeTrainer(apiA, tokens);
  const B = makeTrainer(apiB, tokens);
  await A.initialize();
  await B.initialize();
  await A.setAnchor();
  await B.setAnchor();

  const losses: number[] = [];
  for (let r = 0; r < ROUNDS; r++) {
    const lossA = await A.innerSteps(r);
    const lossB = await B.innerSteps(r);
    const pgA = await A.pseudograd();
    const pgB = await B.pseudograd();
    // Wire round-trip + average (what a 2-peer all-reduce computes).
    const avg: Float32Array[] = pgA.map((ta, i) => {
      const da = roundTrip(ta);
      const db = roundTrip(pgB[i]);
      const out = new Float32Array(da.length);
      for (let j = 0; j < out.length; j++) out[j] = (da[j] + db[j]) / 2;
      return out;
    });
    await A.applyOuterStep(avg);
    await B.applyOuterStep(avg);
    const loss = (lossA + lossB) / 2;
    losses.push(loss);
    log(`round ${r}: loss=${loss.toFixed(4)}`);
  }

  console.log(JSON.stringify({ codec: CODEC, losses }));
  destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
