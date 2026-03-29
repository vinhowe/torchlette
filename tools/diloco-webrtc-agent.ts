/**
 * DiLoCo WebRTC Agent
 *
 * Trains GPT-2 124M and exchanges E3M0-compressed pseudo-gradients with
 * browser peers via a WebSocket bridge on the host (sivri). The bridge
 * translates WebSocket↔WebRTC so browsers connect over the public internet.
 *
 * Usage:
 *   BRIDGE_URL=ws://172.17.0.1:9876 npx tsx tools/diloco-webrtc-agent.ts
 */

import * as fs from "node:fs";
import * as path from "node:path";
import WebSocket from "ws";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { e3m0Dequantize, e3m0Quantize } from "../src/distributed/e3m0";
import { NesterovOuterOptimizer } from "../src/distributed/outer-optimizer";
import { Torchlette } from "../src/frontend/torchlette";
import { clipGradNorm_ } from "../src/nn";
import { normal_ } from "../src/nn/init";
import { Adam } from "../src/optim";

// ============================================================================
// Config
// ============================================================================

const SEED = parseInt(process.env.SEED ?? "42", 10);
const INNER_STEPS = parseInt(process.env.STEPS ?? "20", 10);
const OUTER_ROUNDS = parseInt(process.env.ROUNDS ?? "100", 10);
const LR = parseFloat(process.env.LR ?? "1e-4");
const OUTER_LR = parseFloat(process.env.OUTER_LR ?? "0.7");
const OUTER_MU = parseFloat(process.env.OUTER_MU ?? "0.9");
const SEQ_LEN = parseInt(process.env.SEQ_LEN ?? "512", 10);
const BATCH_SIZE = parseInt(process.env.BATCH_SIZE ?? "6", 10);
const MODEL_DIR = process.env.MODEL ?? "gpt2";
const SERVER_URL = process.env.SERVER_URL ?? "ws://5.78.181.14:443";

const GPT2_CONFIG = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 12,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0,
};

const log = (msg: string) => console.error(`[diloco-webrtc] ${msg}`);

// ============================================================================
// E3M0 Binary Serialization (same format as file-based agent)
// ============================================================================

function serializePseudoGrads(pseudoGrads: Float32Array[]): Buffer {
  const parts: Buffer[] = [];
  const header = Buffer.alloc(4);
  header.writeUInt32LE(pseudoGrads.length, 0);
  parts.push(header);

  for (const pg of pseudoGrads) {
    const padded = new Float32Array(Math.ceil(pg.length / 8) * 8);
    padded.set(pg);
    const { codes, scales } = e3m0Quantize(padded);
    const paramHeader = Buffer.alloc(12);
    paramHeader.writeUInt32LE(pg.length, 0);
    paramHeader.writeUInt32LE(codes.byteLength, 4);
    paramHeader.writeUInt32LE(scales.byteLength, 8);
    parts.push(paramHeader);
    parts.push(Buffer.from(codes.buffer, codes.byteOffset, codes.byteLength));
    parts.push(Buffer.from(scales));
  }
  return Buffer.concat(parts);
}

function deserializePseudoGrads(data: Buffer): Float32Array[] {
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
    result.push(e3m0Dequantize(codes, scales, padLen).slice(0, numValues));
  }
  return result;
}

// ============================================================================
// Data + Model (same as diloco-agent.ts)
// ============================================================================

async function loadTokenizer(modelDir: string) {
  const { GPT2Tokenizer } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer"
  );
  const vocabPath = path.join(process.cwd(), "models", modelDir, "vocab.json");
  const mergesPath = path.join(process.cwd(), "models", modelDir, "merges.txt");
  const tokenizer = new GPT2Tokenizer();
  tokenizer.load(
    JSON.parse(fs.readFileSync(vocabPath, "utf-8")),
    fs
      .readFileSync(mergesPath, "utf-8")
      .split("\n")
      .filter((l: string) => l && !l.startsWith("#")),
  );
  return tokenizer;
}

function loadTrainingTokens(tokenizer: any): number[] {
  const finewebPath = "/tmp/fineweb-shards/shard-0.txt";
  if (fs.existsSync(finewebPath)) {
    const text = fs.readFileSync(finewebPath, "utf-8");
    log(`Loaded FineWeb shard (${(text.length / 1024).toFixed(0)}KB)`);
    return tokenizer.encode(text);
  }
  const fallback = "examples/gpt2-lora-trainer/static/datasets/austen.txt";
  const filePath = path.join(process.cwd(), fallback);
  const text = fs.readFileSync(filePath, "utf-8");
  log(
    `Loaded ${(text.length / 1024).toFixed(0)}KB from ${path.basename(filePath)}`,
  );
  return tokenizer.encode(text);
}

async function createModel(api: Torchlette, modelDir: string) {
  const { GPT2WithLoRA } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora"
  );
  const model = new GPT2WithLoRA(
    api,
    GPT2_CONFIG,
    { rank: 1, alpha: 1 },
    "webgpu",
  );
  const params = model.getAllParameters();
  for (const p of params) {
    if (p.shape.length >= 2) normal_(api, p, 0, 0.02);
  }
  log("Initialized GPT-2 124M from scratch");
  return model;
}

// ============================================================================
// Training Step
// ============================================================================

async function trainStep(
  api: Torchlette,
  model: any,
  optimizer: Adam,
  params: any[],
  tokens: number[],
  offset: number,
): Promise<number> {
  const maxStart = Math.max(1, tokens.length - SEQ_LEN - 1);
  const inputData: number[] = [];
  const targetData: number[] = [];
  for (let b = 0; b < BATCH_SIZE; b++) {
    const start = (offset + b * SEQ_LEN) % maxStart;
    for (let i = 0; i < SEQ_LEN; i++) {
      inputData.push(tokens[start + i]);
      targetData.push(tokens[start + i + 1]);
    }
  }

  await api.beginStep();
  const input = api.tensorFromArray(inputData, [BATCH_SIZE, SEQ_LEN], {
    device: "webgpu",
  });
  const target = api.tensorFromArray(targetData, [BATCH_SIZE, SEQ_LEN], {
    device: "webgpu",
  });
  const loss = api.tidy(() => {
    const l = api.autocast(() => model.forwardWithLoss(input, target).loss);
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
  const ok = await initWebGPU();
  if (!ok) {
    log("WebGPU not available");
    process.exit(1);
  }

  const { setGPUMemoryLimit } = await import(
    "../src/backend/webgpu/memory-tracker"
  );
  setGPUMemoryLimit(30 * 1024 * 1024 * 1024);

  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(SEED);

  const model = await createModel(api, MODEL_DIR);
  model.train(true);
  model.enableCheckpointing(true);
  model.fullCheckpoint = true;
  model.setFullFinetuning(true);
  const params = model.getAllParameters();
  const totalParams = params.reduce(
    (s, p) => s + p.shape.reduce((a, b) => a * b, 1),
    0,
  );

  const tokenizer = await loadTokenizer(MODEL_DIR);
  const tokens = loadTrainingTokens(tokenizer);
  log(`${tokens.length} tokens, ${totalParams.toLocaleString()} params`);

  const innerOpt = new Adam(params, { lr: LR, weightDecay: 0.1 }, api);
  const outerOpt = new NesterovOuterOptimizer(api, {
    lr: OUTER_LR,
    momentum: OUTER_MU,
  });

  // ── Connect to coordination server ──
  const ws = new WebSocket(SERVER_URL);
  let peerGrads: Float32Array[] | null = null;
  let peerCount = 0;
  let myPeerId = "";

  await new Promise<void>((resolve, reject) => {
    ws.on("open", () => {
      ws.send(
        JSON.stringify({
          type: "register",
          peerId: "v100-" + Date.now(),
          model: "gpt2-124m",
        }),
      );
      log(`Connected to ${SERVER_URL}`);
    });
    ws.on("message", (data: Buffer) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.type === "registered") {
          myPeerId = msg.peerId;
          peerCount = msg.peers - 1;
          log(`Registered as ${myPeerId} (${msg.peers} peers total)`);
          resolve();
        }
      } catch {}
    });
    ws.on("error", (e) => reject(e));
    setTimeout(() => reject(new Error("Server timeout")), 10000);
  });

  ws.on("message", async (data: Buffer) => {
    try {
      const msg = JSON.parse(data.toString());
      if (msg.type === "peer-joined") {
        peerCount = msg.peers - 1;
        log(`Peer joined: ${msg.peerId} (${msg.peers} total)`);
      } else if (msg.type === "peer-left") {
        peerCount = msg.peers;
        log(`Peer left: ${msg.peerId} (${msg.peers} total)`);
      } else if (msg.type === "send-weights") {
        // Send last checkpoint file (no GPU memory cost)
        const ckptPath = "/tmp/diloco-webrtc-checkpoint.bin";
        if (fs.existsSync(ckptPath)) {
          const payload = fs.readFileSync(ckptPath);
          const targetBuf = Buffer.from(msg.target, "utf8");
          const header = Buffer.alloc(6 + targetBuf.length);
          header.write("WGHT", 0);
          header.writeUInt16LE(targetBuf.length, 4);
          targetBuf.copy(header, 6);
          ws.send(Buffer.concat([header, payload]));
          log(
            `Sent ${(payload.length / 1024 / 1024).toFixed(1)}MB checkpoint to ${msg.target}`,
          );
        } else {
          log(`No checkpoint yet, ${msg.target} will use random init`);
        }
      }
      return;
    } catch {
      // Binary = gradient blob from neighbor
      log(
        `Received ${(data.length / 1024 / 1024).toFixed(1)}MB gradient from neighbor`,
      );
      peerGrads = deserializePseudoGrads(Buffer.from(data));
    }
  });

  const tokensPerStep = BATCH_SIZE * SEQ_LEN;
  log(
    `Training: ${OUTER_ROUNDS} rounds × ${INNER_STEPS} steps, batch=${BATCH_SIZE}, seq=${SEQ_LEN}, ${tokensPerStep} tok/step`,
  );

  // ── Training loop ──
  for (let round = 0; round < OUTER_ROUNDS; round++) {
    const roundStart = performance.now();

    // Only snapshot if we have peers (outer update needs snapshot to restore)
    const hasPeers = peerCount > 0;
    const globalSnapshot: Float32Array[] = [];
    if (hasPeers) {
      for (const p of params)
        globalSnapshot.push(new Float32Array(await p.cpu()));
    }

    const losses: number[] = [];
    for (let step = 0; step < INNER_STEPS; step++) {
      const l = await trainStep(
        api,
        model,
        innerOpt,
        params,
        tokens,
        (round * INNER_STEPS + step) * tokensPerStep,
      );
      losses.push(l);
      if (step % 10 === 0) log(`  step ${step}: loss=${l.toFixed(4)}`);
    }

    // Free GPU memory before any CPU readback (checkpoint, pseudo-grads)
    await api.evictArenas();
    await api.flushStep();

    // Skip gradient exchange entirely when solo — just keep training
    if (!hasPeers && !peerGrads) {
      log(`  solo (no peers) — continuing training`);
    } else {
      // Compute pseudo-gradients
      const pseudoGrads: Float32Array[] = [];
      for (let i = 0; i < params.length; i++) {
        const local = await params[i].cpu();
        const delta = new Float32Array(local.length);
        for (let j = 0; j < delta.length; j++)
          delta[j] = local[j] - globalSnapshot[i][j];
        pseudoGrads.push(delta);
      }

      // Send gradients to neighbors
      ws.send(JSON.stringify({ type: "request-neighbors", round }));
      const compressed = serializePseudoGrads(pseudoGrads);
      ws.send(compressed);
      log(
        `Sent ${(compressed.length / 1024 / 1024).toFixed(1)}MB gradients (round ${round})`,
      );

      // Wait briefly for peer grads
      await new Promise((r) => setTimeout(r, 3000));

      // Average with received peer grads
      let numContributors = 1;
      const avgGrads = pseudoGrads.map((pg) => new Float32Array(pg.length));
      for (let p = 0; p < pseudoGrads.length; p++)
        for (let j = 0; j < pseudoGrads[p].length; j++)
          avgGrads[p][j] += pseudoGrads[p][j];

      if (peerGrads && peerGrads.length === params.length) {
        for (let p = 0; p < peerGrads.length; p++)
          for (let j = 0; j < peerGrads[p].length; j++)
            avgGrads[p][j] += peerGrads[p][j];
        numContributors++;
        peerGrads = null;
      }

      if (numContributors > 1) {
        for (let p = 0; p < avgGrads.length; p++)
          for (let j = 0; j < avgGrads[p].length; j++)
            avgGrads[p][j] /= numContributors;

        // Outer Nesterov update
        await api.beginStep();
        for (let i = 0; i < params.length; i++) {
          api.copy_(
            params[i],
            api.tensorFromArray(globalSnapshot[i], params[i].shape, {
              device: "webgpu",
            }),
          );
        }
        const avgTensors = avgGrads.map((pg, i) =>
          api.tensorFromArray(pg, params[i].shape, { device: "webgpu" }),
        );
        await api._runtime().forceAllPending();
        await outerOpt.step(params, avgTensors);
        api.endStep();
        await api.markStep();
        log(`  averaged with ${numContributors} contributors`);
      } else {
        log("  solo round — keeping local params");
      }
    }

    // Save checkpoint only when we have snapshot data (i.e., peers are connected)
    if (globalSnapshot.length > 0) {
      const ckptParts: Buffer[] = [];
      const ckptHeader = Buffer.alloc(4);
      ckptHeader.writeUInt32LE(params.length, 0);
      ckptParts.push(ckptHeader);
      for (let i = 0; i < params.length; i++) {
        const shapeBuf = Buffer.alloc(4 + params[i].shape.length * 4);
        shapeBuf.writeUInt32LE(params[i].shape.length, 0);
        for (let d = 0; d < params[i].shape.length; d++)
          shapeBuf.writeUInt32LE(params[i].shape[d], 4 + d * 4);
        ckptParts.push(shapeBuf);
        ckptParts.push(Buffer.from(globalSnapshot[i].buffer));
      }
      fs.writeFileSync(
        "/tmp/diloco-webrtc-checkpoint.bin",
        Buffer.concat(ckptParts),
      );
    }

    const elapsed = ((performance.now() - roundStart) / 1000).toFixed(1);
    const avgLoss = losses.reduce((a, b) => a + b, 0) / losses.length;
    log(
      `round ${round}: loss=${avgLoss.toFixed(4)}, ${elapsed}s, peers=${peerCount + 1}`,
    );
  }

  log("Training complete.");
  await new Promise((r) => setTimeout(r, 5000));
  ws.close();
  outerOpt.dispose();
  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
