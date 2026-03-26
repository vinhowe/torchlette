/**
 * DiLoCo Node Agent
 *
 * A headless GPT-2 124M training worker for distributed pretraining.
 * Communicates pseudo-gradients via temp files (coordinator reads/writes).
 * Uses E3M0 4-bit compression for gradient exchange.
 *
 * Usage:
 *   npx tsx tools/diloco-agent.ts
 *
 * Environment:
 *   SEED=42        Global RNG seed (all agents must match)
 *   STEPS=100      Inner training steps between syncs
 *   ROUNDS=10      Number of DiLoCo sync rounds
 *   AGENT_ID=0     Agent index (assigned by coordinator)
 *   SYNC_DIR=/tmp/diloco  Directory for gradient exchange files
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { e3m0Dequantize, e3m0Quantize } from "../src/distributed";
import { Torchlette } from "../src/frontend/torchlette";
import { clipGradNorm_ } from "../src/nn";
import { Adam } from "../src/optim";

// ============================================================================
// Config
// ============================================================================

const SEED = parseInt(process.env.SEED ?? "42", 10);
const INNER_STEPS = parseInt(process.env.STEPS ?? "100", 10);
const OUTER_ROUNDS = parseInt(process.env.ROUNDS ?? "10", 10);
const LR = parseFloat(process.env.LR ?? "4e-4");
const SEQ_LEN = parseInt(process.env.SEQ_LEN ?? "128", 10);
const MODEL_DIR = process.env.MODEL ?? "gpt2";
const AGENT_ID = parseInt(process.env.AGENT_ID ?? "0", 10);
const NUM_AGENTS = parseInt(process.env.NUM_AGENTS ?? "2", 10);
const SYNC_DIR = process.env.SYNC_DIR ?? "/tmp/diloco";

const GPT2_CONFIG = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 12,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0,
};

const log = (msg: string) => console.error(`[agent-${AGENT_ID}] ${msg}`);

// ============================================================================
// File-based sync protocol
// ============================================================================

/**
 * Write compressed pseudo-gradients to a file.
 * Format: numParams (u32) + per-param: numValues (u32) + codes + scales
 */
function writePseudoGrads(pseudoGrads: Float32Array[], round: number): void {
  const parts: Buffer[] = [];

  // Header: numParams
  const header = Buffer.alloc(4);
  header.writeUInt32LE(pseudoGrads.length, 0);
  parts.push(header);

  for (const pg of pseudoGrads) {
    const padded = new Float32Array(Math.ceil(pg.length / 8) * 8);
    padded.set(pg);
    const { codes, scales } = e3m0Quantize(padded);

    // Per-param header: numValues (u32) + codesLen (u32) + scalesLen (u32)
    const paramHeader = Buffer.alloc(12);
    paramHeader.writeUInt32LE(pg.length, 0);
    paramHeader.writeUInt32LE(codes.byteLength, 4);
    paramHeader.writeUInt32LE(scales.byteLength, 8);
    parts.push(paramHeader);

    parts.push(Buffer.from(codes.buffer, codes.byteOffset, codes.byteLength));
    parts.push(Buffer.from(scales));
  }

  const filePath = path.join(SYNC_DIR, `grad-${AGENT_ID}-${round}.bin`);
  fs.writeFileSync(filePath, Buffer.concat(parts));
}

/** Read and decompress pseudo-gradients from a file. */
function readPseudoGrads(filePath: string): Float32Array[] {
  const data = fs.readFileSync(filePath);
  let offset = 0;

  const numParams = data.readUInt32LE(offset);
  offset += 4;
  const result: Float32Array[] = [];

  for (let i = 0; i < numParams; i++) {
    const numValues = data.readUInt32LE(offset);
    offset += 4;
    const codesLen = data.readUInt32LE(offset);
    offset += 4;
    const scalesLen = data.readUInt32LE(offset);
    offset += 4;

    const codes = new Uint32Array(
      data.buffer.slice(
        data.byteOffset + offset,
        data.byteOffset + offset + codesLen,
      ),
    );
    offset += codesLen;

    const scales = new Uint8Array(
      data.buffer.slice(
        data.byteOffset + offset,
        data.byteOffset + offset + scalesLen,
      ),
    );
    offset += scalesLen;

    const padLen = Math.ceil(numValues / 8) * 8;
    const restored = e3m0Dequantize(codes, scales, padLen);
    result.push(restored.slice(0, numValues));
  }

  return result;
}

/** Wait for a file to appear (polling). */
function waitForFile(filePath: string, timeoutMs = 300000): Promise<void> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    const check = () => {
      if (fs.existsSync(filePath)) {
        resolve();
        return;
      }
      if (Date.now() - start > timeoutMs) {
        reject(new Error(`Timeout waiting for ${filePath}`));
        return;
      }
      setTimeout(check, 100);
    };
    check();
  });
}

// ============================================================================
// Data + Model Loading (reused from previous version)
// ============================================================================

async function loadTokenizer(modelDir: string) {
  const { GPT2Tokenizer } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer"
  );
  const vocabPath = path.join(process.cwd(), "models", modelDir, "vocab.json");
  const mergesPath = path.join(process.cwd(), "models", modelDir, "merges.txt");
  const vocabJson = JSON.parse(fs.readFileSync(vocabPath, "utf-8"));
  const mergesText = fs.readFileSync(mergesPath, "utf-8");
  const tokenizer = new GPT2Tokenizer();
  tokenizer.load(
    vocabJson,
    mergesText.split("\n").filter((l: string) => l && !l.startsWith("#")),
  );
  return tokenizer;
}

function loadTrainingTokens(tokenizer: any): number[] {
  const textFiles = [
    "examples/gpt2-lora-trainer/static/datasets/austen.txt",
    "examples/gpt2-lora-trainer/static/datasets/lovecraft.txt",
    "examples/gpt2-lora-trainer/static/datasets/aurelius.txt",
  ];
  const filePath = path.join(
    process.cwd(),
    textFiles[AGENT_ID % textFiles.length],
  );
  if (!fs.existsSync(filePath)) {
    log(`Data not found: ${filePath}, using synthetic`);
    return tokenizer.encode(
      "The quick brown fox jumps over the lazy dog. ".repeat(500),
    );
  }
  const text = fs.readFileSync(filePath, "utf-8");
  log(
    `Loaded ${(text.length / 1024).toFixed(0)}KB from ${path.basename(filePath)}`,
  );
  return tokenizer.encode(text);
}

async function loadModel(api: Torchlette, modelDir: string) {
  const { GPT2WithLoRA } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora"
  );
  const model = new GPT2WithLoRA(
    api,
    GPT2_CONFIG,
    { rank: 1, alpha: 1 },
    "webgpu",
  );

  const modelPath = path.join(
    process.cwd(),
    "models",
    modelDir,
    "model.safetensors",
  );
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
    let f32: Float32Array;
    if (dtype === "F32") {
      f32 = new Float32Array(new Uint8Array(raw).slice().buffer);
    } else if (dtype === "F16") {
      const u16 = new Uint16Array(
        raw.buffer,
        raw.byteOffset,
        raw.byteLength / 2,
      );
      f32 = new Float32Array(u16.length);
      for (let i = 0; i < u16.length; i++) {
        const h = u16[i];
        const s = (h >> 15) & 1,
          e = (h >> 10) & 0x1f,
          m = h & 0x3ff;
        if (e === 0) f32[i] = (s ? -1 : 1) * (m / 1024) * 2 ** -14;
        else if (e === 31) f32[i] = m === 0 ? (s ? -Infinity : Infinity) : NaN;
        else f32[i] = (s ? -1 : 1) * (1 + m / 1024) * 2 ** (e - 15);
      }
    } else continue;
    weights.set(name.replace(/^transformer\./, ""), {
      data: new Float32Array(f32),
      shape,
    });
  }
  model.loadBaseWeights(weights);
  return model;
}

// ============================================================================
// Training
// ============================================================================

async function trainStep(
  api: Torchlette,
  model: any,
  optimizer: Adam,
  params: any[],
  tokens: number[],
  offset: number,
): Promise<number> {
  const start = offset % Math.max(1, tokens.length - SEQ_LEN - 1);
  const input = api.tensorFromArray(
    tokens.slice(start, start + SEQ_LEN),
    [1, SEQ_LEN],
    { device: "webgpu" },
  );
  const target = api.tensorFromArray(
    tokens.slice(start + 1, start + SEQ_LEN + 1),
    [1, SEQ_LEN],
    { device: "webgpu" },
  );

  await api.beginStep();
  const loss = api.tidy(() => {
    const { loss: l } = model.forwardWithLoss(input, target);
    api.keep(l);
    return l;
  });
  const lossVal = await loss.item();
  await loss.backward();
  clipGradNorm_(api, params, 1.0);
  optimizer.step();
  optimizer.zeroGrad();
  input.dispose();
  target.dispose();
  api.endStep();
  await api.markStep();
  return lossVal;
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  fs.mkdirSync(SYNC_DIR, { recursive: true });

  const ok = await initWebGPU();
  if (!ok) {
    log("WebGPU not available");
    process.exit(1);
  }

  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(SEED);

  log(`Loading model (${MODEL_DIR})...`);
  const model = await loadModel(api, MODEL_DIR);
  model.train(true);
  model.enableCheckpointing(true);
  model.setFullFinetuning(true);
  const params = model.getAllParameters();
  const totalParams = params.reduce(
    (s, p) => s + p.shape.reduce((a, b) => a * b, 1),
    0,
  );

  log("Loading tokenizer...");
  const tokenizer = await loadTokenizer(MODEL_DIR);
  log("Loading training data...");
  const tokens = loadTrainingTokens(tokenizer);
  log(`${tokens.length} tokens, ${totalParams.toLocaleString()} params`);

  // Signal ready
  fs.writeFileSync(path.join(SYNC_DIR, `ready-${AGENT_ID}`), "");

  // Wait for all agents to be ready
  for (let i = 0; i < NUM_AGENTS; i++) {
    await waitForFile(path.join(SYNC_DIR, `ready-${i}`));
  }
  log(
    `All ${NUM_AGENTS} agents ready. Training: ${OUTER_ROUNDS} rounds × ${INNER_STEPS} steps`,
  );

  for (let round = 0; round < OUTER_ROUNDS; round++) {
    const roundStart = performance.now();

    // Snapshot global params
    const globalSnapshot: Float32Array[] = [];
    for (const p of params)
      globalSnapshot.push(new Float32Array(await p.cpu()));

    // Inner training
    const innerOpt = new Adam(params, { lr: LR, weightDecay: 0.1 }, api);
    const losses: number[] = [];
    for (let step = 0; step < INNER_STEPS; step++) {
      const l = await trainStep(
        api,
        model,
        innerOpt,
        params,
        tokens,
        (round * INNER_STEPS + step) * SEQ_LEN,
      );
      losses.push(l);
      if (step % 25 === 0) {
        const avg =
          losses.slice(-10).reduce((a, b) => a + b, 0) /
          Math.min(10, losses.length);
        log(`  step ${step}: loss=${avg.toFixed(4)}`);
      }
    }

    // Compute + write compressed pseudo-gradients
    const pseudoGrads: Float32Array[] = [];
    for (let i = 0; i < params.length; i++) {
      const local = await params[i].cpu();
      const delta = new Float32Array(local.length);
      for (let j = 0; j < delta.length; j++)
        delta[j] = local[j] - globalSnapshot[i][j];
      pseudoGrads.push(delta);
    }
    writePseudoGrads(pseudoGrads, round);
    log(`Wrote pseudo-grads (round ${round})`);

    // Wait for all agents to write their gradients
    for (let i = 0; i < NUM_AGENTS; i++) {
      await waitForFile(path.join(SYNC_DIR, `grad-${i}-${round}.bin`));
    }

    // Read and average all agents' pseudo-gradients
    const avgGrads: Float32Array[] = pseudoGrads.map(
      (pg) => new Float32Array(pg.length),
    );
    for (let a = 0; a < NUM_AGENTS; a++) {
      const grads = readPseudoGrads(
        path.join(SYNC_DIR, `grad-${a}-${round}.bin`),
      );
      for (let p = 0; p < grads.length; p++) {
        for (let j = 0; j < grads[p].length; j++) {
          avgGrads[p][j] += grads[p][j] / NUM_AGENTS;
        }
      }
    }

    // Apply outer update: snapshot + averaged pseudo-gradient
    await api.beginStep();
    for (let i = 0; i < params.length; i++) {
      const updated = new Float32Array(globalSnapshot[i].length);
      for (let j = 0; j < updated.length; j++) {
        updated[j] = globalSnapshot[i][j] + avgGrads[i][j];
      }
      api.copy_(
        params[i],
        api.tensorFromArray(Array.from(updated), params[i].shape, {
          device: "webgpu",
        }),
      );
    }
    api.endStep();
    await api.markStep();

    const elapsed = ((performance.now() - roundStart) / 1000).toFixed(1);
    const avgLoss = losses.reduce((a, b) => a + b, 0) / losses.length;
    const gradSize = fs.statSync(
      path.join(SYNC_DIR, `grad-${AGENT_ID}-${round}.bin`),
    ).size;
    log(
      `round ${round}: loss=${avgLoss.toFixed(4)}, ${elapsed}s, grad=${(gradSize / 1024 / 1024).toFixed(1)}MB`,
    );
  }

  log("Done");
  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
