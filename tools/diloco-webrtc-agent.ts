/**
 * DiLoCo V100 Agent
 *
 * GPT-2 124M distributed pretraining. Connects to coordination server,
 * exchanges E3M0-compressed pseudo-gradients with peers, sends f16
 * checkpoint weights to late-joining browsers.
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

// ── Config ──
const SEED = parseInt(process.env.SEED ?? "42", 10);
const INNER_STEPS = parseInt(process.env.STEPS ?? "20", 10);
const OUTER_ROUNDS = parseInt(process.env.ROUNDS ?? "100", 10);
const LR = parseFloat(process.env.LR ?? "1e-4");
const OUTER_LR = parseFloat(process.env.OUTER_LR ?? "0.7");
const OUTER_MU = parseFloat(process.env.OUTER_MU ?? "0.9");
const SEQ_LEN = parseInt(process.env.SEQ_LEN ?? "512", 10);
const BATCH_SIZE = parseInt(process.env.BATCH_SIZE ?? "4", 10);
const ACCUM_STEPS = parseInt(process.env.ACCUM_STEPS ?? "1", 10);
const MODEL_DIR = process.env.MODEL ?? "gpt2";
const SERVER_URL = process.env.SERVER_URL ?? "ws://5.78.181.14:443";
const EFFECTIVE_BATCH = BATCH_SIZE * ACCUM_STEPS;

const NUM_LAYERS = parseInt(process.env.NUM_LAYERS ?? "12", 10);
const EMBED_DIM = parseInt(process.env.EMBED_DIM ?? "768", 10);
const NUM_HEADS = parseInt(process.env.NUM_HEADS ?? "12", 10);
const HF_DATASET = process.env.HF_DATASET ?? "HuggingFaceFW/fineweb-edu";
const HF_CONFIG = process.env.HF_CONFIG ?? "sample-10BT";

const GPT2_CONFIG = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: NUM_LAYERS,
  numHeads: NUM_HEADS,
  embedDim: EMBED_DIM,
  dropoutRate: 0,
};

const log = (msg: string) => console.error(`[diloco-webrtc] ${msg}`);

// ── E3M0 ──
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
    const nv = data.readUInt32LE(offset);
    offset += 4;
    const cl = data.readUInt32LE(offset);
    offset += 4;
    const sl = data.readUInt32LE(offset);
    offset += 4;
    const codes = new Uint32Array(
      data.buffer.slice(
        data.byteOffset + offset,
        data.byteOffset + offset + cl,
      ),
    );
    offset += cl;
    const scales = new Uint8Array(
      data.buffer.slice(
        data.byteOffset + offset,
        data.byteOffset + offset + sl,
      ),
    );
    offset += sl;
    result.push(
      e3m0Dequantize(codes, scales, Math.ceil(nv / 8) * 8).slice(0, nv),
    );
  }
  return result;
}

// ── Tokenizer ──
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

// ── Data: fetch fresh from HuggingFace each round ──
const HF_TOTAL_ROWS = parseInt(process.env.HF_ROWS ?? "9672101", 10);
const HF_FETCH_ROWS = 100;

async function fetchFreshTokens(
  tokenizer: any,
  minTokens?: number,
): Promise<number[]> {
  const target = minTokens ?? INNER_STEPS * EFFECTIVE_BATCH * SEQ_LEN;
  let allText = "";
  let totalFetched = 0;
  let consecutiveFailures = 0;
  // Fetch enough rows to cover one round of training
  while (true) {
    const offset = Math.floor(Math.random() * (HF_TOTAL_ROWS - HF_FETCH_ROWS));
    const url = `https://datasets-server.huggingface.co/rows?dataset=${HF_DATASET}&config=${HF_CONFIG}&split=train&offset=${offset}&length=${HF_FETCH_ROWS}`;
    try {
      const resp = await fetch(url, { signal: AbortSignal.timeout(30000) });
      const ct = resp.headers.get("content-type") ?? "";
      if (!resp.ok || !ct.includes("application/json")) {
        throw new Error(`HF ${resp.status} ${ct.split(";")[0]}`);
      }
      const data = await resp.json();
      if (!data.rows) throw new Error("HF response missing rows");
      const text = data.rows.map((r: any) => r.row.text).join("\n\n");
      allText += (allText ? "\n\n" : "") + text;
      totalFetched += HF_FETCH_ROWS;
      consecutiveFailures = 0;
      const tokens = tokenizer.encode(allText);
      if (tokens.length >= target || totalFetched >= 1000) {
        log(
          `Fetched ${tokens.length} tokens (${totalFetched} rows, target ${target})`,
        );
        return tokens;
      }
    } catch (e) {
      consecutiveFailures++;
      const wait = Math.min(60000, 2000 * 2 ** Math.min(consecutiveFailures, 5));
      log(
        `HF fetch failed (${consecutiveFailures}): ${(e as Error).message} — retry in ${wait / 1000}s`,
      );
      await new Promise((r) => setTimeout(r, wait));
    }
  }
}

// ── Model ──
async function createModel(api: Torchlette, _modelDir: string) {
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

  // Always init from seed first (deterministic across all peers)
  for (const p of params) {
    if (p.shape.length >= 2) normal_(api, p, 0, 0.02);
  }
  await api._runtime().forceAllPending();
  log("Initialized GPT-2 124M from seed");

  // Resume from checkpoint if available.
  const ckptPath = "/tmp/diloco-webrtc-checkpoint.bin";
  if (fs.existsSync(ckptPath)) {
    const buf = fs.readFileSync(ckptPath);
    let off = 0;
    const np = buf.readUInt32LE(off);
    off += 4;
    if (np === params.length) {
      await api.beginStep();
      for (let i = 0; i < np; i++) {
        const rank = buf.readUInt32LE(off);
        off += 4;
        let nel = 1;
        for (let d = 0; d < rank; d++) {
          nel *= buf.readUInt32LE(off);
          off += 4;
        }
        const f32 = new Float32Array(buf.buffer, buf.byteOffset + off, nel);
        off += nel * 4;
        api.copy_(
          params[i],
          api.tensorFromArray(f32, params[i].shape, { device: "webgpu" }),
        );
      }
      api.endStep();
      await api.markStep();
      log(
        `Resumed from checkpoint (${(buf.length / 1024 / 1024).toFixed(0)}MB)`,
      );
    }
  }

  return model;
}

// ── f16 weight encoding for checkpoint sync ──
// Frame body (after server strips routing prefix and re-prepends "F16W"):
//   [F16W (4)][round (4 LE)][numParams (4 LE)][per-param: nel(4) + f16 data]
function encodeCheckpointF16(ckptPath: string, round: number): Buffer {
  const buf = fs.readFileSync(ckptPath);
  let off = 0;
  const np = buf.readUInt32LE(off);
  off += 4;
  const parts: Buffer[] = [];
  const roundBuf = Buffer.alloc(4);
  roundBuf.writeUInt32LE(round, 0);
  parts.push(roundBuf);
  const npBuf = Buffer.alloc(4);
  npBuf.writeUInt32LE(np, 0);
  parts.push(npBuf);

  const tmpF32 = new Float32Array(1);
  const tmpU16 = new Uint16Array(tmpF32.buffer);

  for (let i = 0; i < np; i++) {
    const rank = buf.readUInt32LE(off);
    off += 4;
    let nel = 1;
    for (let d = 0; d < rank; d++) {
      nel *= buf.readUInt32LE(off);
      off += 4;
    }
    const current = new Float32Array(buf.buffer, buf.byteOffset + off, nel);
    off += nel * 4;

    const f16 = new Uint16Array(nel);
    for (let j = 0; j < nel; j++) {
      tmpF32[0] = current[j];
      const bits = (tmpU16[1] << 16) | tmpU16[0];
      const sign = (bits >> 31) & 1;
      const exp = (bits >> 23) & 0xff;
      const mant = bits & 0x7fffff;
      let h: number;
      if (exp === 0) {
        h = sign << 15;
      } else if (exp === 0xff) {
        h = (sign << 15) | 0x7c00 | (mant ? 0x200 : 0);
      } else {
        const newExp = exp - 127 + 15;
        if (newExp >= 31) {
          h = (sign << 15) | 0x7c00;
        } else if (newExp <= 0) {
          h = sign << 15;
        } else {
          h = (sign << 15) | (newExp << 10) | (mant >> 13);
        }
      }
      f16[j] = h;
    }
    const nelBuf = Buffer.alloc(4);
    nelBuf.writeUInt32LE(nel, 0);
    parts.push(nelBuf);
    parts.push(Buffer.from(f16.buffer, f16.byteOffset, f16.byteLength));
  }
  return Buffer.concat(parts);
}

// Decode the body that follows the [F16W][round] header into the local params.
// Body layout (after caller strips F16W prefix + round):
//   [numParams(4)][per-param: nel(4) + f16 data].
async function applyF16WeightsToParams(
  api: Torchlette,
  params: import("../src/frontend/tensor").Tensor[],
  body: Buffer,
): Promise<void> {
  let off = 0;
  const np = body.readUInt32LE(off);
  off += 4;
  if (np !== params.length) {
    throw new Error(
      `F16W param count mismatch: got ${np}, expected ${params.length}`,
    );
  }
  await api.beginStep();
  for (let i = 0; i < np; i++) {
    const nel = body.readUInt32LE(off);
    off += 4;
    const f16 = new Uint16Array(
      body.buffer.slice(
        body.byteOffset + off,
        body.byteOffset + off + nel * 2,
      ),
    );
    off += nel * 2;
    const f32 = new Float32Array(nel);
    for (let j = 0; j < nel; j++) {
      const h = f16[j];
      const sign = (h >> 15) & 1;
      const exp = (h >> 10) & 0x1f;
      const mant = h & 0x3ff;
      let val: number;
      if (exp === 0) {
        val = (mant / 1024) * 2 ** -14;
      } else if (exp === 31) {
        val = mant ? Number.NaN : Number.POSITIVE_INFINITY;
      } else {
        val = 2 ** (exp - 15) * (1 + mant / 1024);
      }
      f32[j] = sign ? -val : val;
    }
    api.copy_(
      params[i],
      api.tensorFromArray(f32, params[i].shape, { device: "webgpu" }),
    );
  }
  api.endStep();
  await api.markStep();
}

// ── Training Step ──
async function trainStep(
  api: Torchlette,
  model: any,
  optimizer: Adam,
  params: any[],
  tokens: number[],
  offset: number,
  accumGrads: import("../src/frontend/tensor").Tensor[],
): Promise<number> {
  const maxStart = Math.max(1, tokens.length - SEQ_LEN - 1);
  let totalLoss = 0;
  for (const ag of accumGrads) api.zero_(ag);

  for (let acc = 0; acc < ACCUM_STEPS; acc++) {
    const microOffset = offset + acc * BATCH_SIZE * SEQ_LEN;
    const inputData: number[] = [];
    const targetData: number[] = [];
    for (let b = 0; b < BATCH_SIZE; b++) {
      const start = (microOffset + b * SEQ_LEN) % maxStart;
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
    totalLoss += await loss.item();
    await loss.backward();
    for (let i = 0; i < params.length; i++) {
      if (params[i].grad)
        api.add_(accumGrads[i], api.mul(params[i].grad, 1 / ACCUM_STEPS));
    }
    await api._runtime().forceAllPending();
    optimizer.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
  }

  await api.beginStep();
  for (let i = 0; i < params.length; i++)
    params[i]._setGrad(api.mul(accumGrads[i], 1));
  clipGradNorm_(api, params, 1.0);
  optimizer.step();
  optimizer.zeroGrad();
  api.endStep();
  await api.markStep();

  trainStep._stepCount = (trainStep._stepCount ?? 0) + 1;
  if (trainStep._stepCount % 5 === 0) {
    const { getGPUMemoryStats } = await import(
      "../src/backend/webgpu/memory-tracker"
    );
    const { bufferPool: bp } = await import(
      "../src/backend/webgpu/buffer-pool"
    );
    const ms = getGPUMemoryStats();
    const ps = bp.stats();
    log(
      `    mem: tracked=${Math.round(ms.currentBytes / 1e6)}MB peak=${Math.round(ms.peakBytes / 1e6)}MB pool=${Math.round(ps.pooledBytes / 1e6)}MB`,
    );
  }
  return totalLoss / ACCUM_STEPS;
}

// ── Main ──
async function main() {
  process.env.TORCHLETTE_POOL_BUDGET_MB ??= "2000";

  const ok = await initWebGPU();
  if (!ok) {
    log("WebGPU not available");
    process.exit(1);
  }

  const { setGPUMemoryLimit } = await import(
    "../src/backend/webgpu/memory-tracker"
  );
  setGPUMemoryLimit(31.5 * 1024 * 1024 * 1024);

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
  // biome-ignore lint/style/useConst: reassigned each round
  let tokens = await fetchFreshTokens(tokenizer);
  log(`${totalParams.toLocaleString()} params`);

  const innerOpt = new Adam(params, { lr: LR, weightDecay: 0.1 }, api);
  const outerOpt = new NesterovOuterOptimizer(api, {
    lr: OUTER_LR,
    momentum: OUTER_MU,
  });

  // ── Connect ──
  // maxPayload must accommodate full-size GPT-2 f16 weights (~238MB) plus headers.
  const WS_MAX_PAYLOAD = 500 * 1024 * 1024;
  const ws = new WebSocket(SERVER_URL, { maxPayload: WS_MAX_PAYLOAD });
  let peerGrads: Float32Array[] | null = null;
  let peerTokenCount = 0;
  let peerRound = -1;
  let peerCount = 0;
  let myPeerId = "";
  let sendingWeights = false; // serialize weight sends
  let needsSync = false;
  let currentRound = 0;
  // Late-join sync: store the F16W blob in the handler, parse + load it from
  // main() context. Mirrors the browser's pattern — handler is buffer-only,
  // engine work happens in the main coroutine.
  let pendingSyncBlob: { body: Buffer; sourceRound: number } | null = null;
  let initialSyncResolver: (() => void) | null = null;
  const initialSyncPromise = new Promise<void>((r) => {
    initialSyncResolver = r;
  });

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
          needsSync = !!msg.needsSync;
          log(
            `Registered as ${myPeerId} (${msg.peers} peers total, needsSync=${needsSync})`,
          );
          resolve();
        }
      } catch {}
    });
    ws.on("error", (e) => reject(e));
    setTimeout(() => reject(new Error("Server timeout")), 10000);
  });

  const messageHandler = async (data: Buffer) => {
    try {
      const msg = JSON.parse(data.toString());
      if (msg.type === "peer-joined") {
        peerCount = msg.peers - 1;
        log(`Peer joined: ${msg.peerId} (${msg.peers} total)`);
      } else if (msg.type === "peer-left") {
        peerCount = msg.peers;
        log(`Peer left: ${msg.peerId} (${msg.peers} total)`);
      } else if (msg.type === "send-weights") {
        // Serialize: one weight send at a time
        if (sendingWeights) {
          log(`Already sending weights, skipping request from ${msg.target}`);
          return;
        }
        const ckptPath = "/tmp/diloco-webrtc-checkpoint.bin";
        if (fs.existsSync(ckptPath)) {
          sendingWeights = true;
          const payload = encodeCheckpointF16(ckptPath, currentRound);
          const targetBuf = Buffer.from(msg.target, "utf8");
          const header = Buffer.alloc(6 + targetBuf.length);
          header.write("F16W", 0);
          header.writeUInt16LE(targetBuf.length, 4);
          targetBuf.copy(header, 6);
          conn.send(Buffer.concat([header, payload]));
          log(
            `Sent ${(payload.length / 1024 / 1024).toFixed(1)}MB weights (f16, round ${currentRound}) to ${msg.target}`,
          );
          sendingWeights = false;
        } else {
          log(`No checkpoint yet, ${msg.target} will use random init`);
        }
      }
      return;
    } catch {
      const raw = Buffer.from(data);
      if (raw.length > 8 && raw.toString("utf8", 0, 4) === "F16W") {
        // Late-join weight sync: [F16W][round (4)][numParams (4)][per-param: nel + f16 data]
        // Handler is BUFFER-ONLY: store a COPY of the blob (raw.slice/subarray
        // return views — the underlying ws buffer can be recycled before main
        // processes it). main() applies the weights in its own coroutine.
        if (!initialSyncResolver) {
          log(
            `Ignoring F16W blob (${(raw.length / 1024 / 1024).toFixed(1)}MB): initial sync already complete`,
          );
        } else {
          const sourceRound = raw.readUInt32LE(4);
          // Buffer.from(buf) copies the bytes into a fresh allocation.
          pendingSyncBlob = { body: Buffer.from(raw.subarray(8)), sourceRound };
          log(
            `Buffered F16W blob (${(raw.length / 1024 / 1024).toFixed(1)}MB, source round ${sourceRound}) — main will apply`,
          );
          const resolver = initialSyncResolver;
          initialSyncResolver = null;
          resolver();
        }
      } else if (raw.length > 16 && raw.toString("utf8", 0, 4) === "GRAD") {
        peerTokenCount = raw.readUInt32LE(4);
        peerRound = raw.readUInt32LE(8);
        const peerLoss = raw.readFloatLE(12);
        peerGrads = deserializePseudoGrads(raw.slice(16));
        log(
          `Received ${(raw.length / 1024 / 1024).toFixed(1)}MB gradient (${peerTokenCount} tok, r${peerRound}, loss ${peerLoss.toFixed(2)})`,
        );
      } else {
        peerGrads = deserializePseudoGrads(raw);
        log(`Received ${(raw.length / 1024 / 1024).toFixed(1)}MB gradient`);
      }
    }
  };
  ws.on("message", messageHandler);

  // Keepalive + reconnect wrapper
  const conn = {
    ws,
    send(data: Parameters<WebSocket["send"]>[0]) {
      if (this.ws.readyState === WebSocket.OPEN) this.ws.send(data);
    },
  };
  const pingInterval = setInterval(() => {
    if (conn.ws.readyState === WebSocket.OPEN)
      conn.send(JSON.stringify({ type: "ping" }));
  }, 30000);

  function setupReconnect(socket: WebSocket) {
    socket.on("close", () => {
      log("WebSocket disconnected — reconnecting in 5s...");
      setTimeout(() => {
        try {
          const newWs = new WebSocket(SERVER_URL, { maxPayload: WS_MAX_PAYLOAD });
          newWs.on("error", (e) => {
            log(
              `Reconnect failed: ${(e as Error).message} — retrying in 10s...`,
            );
            setTimeout(() => setupReconnect(newWs), 10000);
          });
          newWs.on("open", () => {
            newWs.send(
              JSON.stringify({
                type: "register",
                peerId: myPeerId,
                model: "gpt2-124m",
              }),
            );
            log("Reconnected to server");
          });
          newWs.on("message", messageHandler);
          conn.ws = newWs;
          setupReconnect(newWs);
        } catch (e) {
          log(`Reconnect error: ${e} — retrying in 10s...`);
          setTimeout(() => setupReconnect(socket), 10000);
        }
      }, 5000);
    });
  }
  setupReconnect(ws);

  // ── Late-join sync: pull weights from existing peer if needed ──
  // Server auto-fires `send-weights` to an existing peer on our register when
  // peers.size > 0. But that peer may not have a checkpoint written yet
  // (its first round is still in progress). Retry periodically until F16W
  // arrives or we hit the overall timeout.
  if (needsSync) {
    log("Awaiting weight sync from existing peer (server-triggered)...");
    const SYNC_TIMEOUT_MS = 300_000; // 5 min total
    const RETRY_INTERVAL_MS = 20_000;
    let timeoutHandle: ReturnType<typeof setTimeout> | null = null;
    const timeoutPromise = new Promise<"timeout">((r) => {
      timeoutHandle = setTimeout(() => r("timeout"), SYNC_TIMEOUT_MS);
    });
    const retryHandle = setInterval(() => {
      if (initialSyncResolver) {
        log("Sync still pending — sending request-weights retry");
        conn.send(JSON.stringify({ type: "request-weights" }));
      }
    }, RETRY_INTERVAL_MS);
    const result = await Promise.race([
      initialSyncPromise.then(() => "synced" as const),
      timeoutPromise,
    ]);
    clearInterval(retryHandle);
    if (timeoutHandle) clearTimeout(timeoutHandle);
    if (result === "timeout") {
      log(
        `Sync timed out (${SYNC_TIMEOUT_MS / 1000}s) — proceeding from local init at round 0`,
      );
    } else if (pendingSyncBlob) {
      // Apply weights now, in main coroutine context.
      const { body, sourceRound } = pendingSyncBlob;
      pendingSyncBlob = null;
      await applyF16WeightsToParams(api, params, body);
      currentRound = sourceRound;
      log(
        `Synced weights into params (jumped to round ${sourceRound})`,
      );
    }
  }

  const tokensPerStep = EFFECTIVE_BATCH * SEQ_LEN;
  log(
    `Training: ${OUTER_ROUNDS} rounds × ${INNER_STEPS} steps, batch=${BATCH_SIZE}×${ACCUM_STEPS}accum=${EFFECTIVE_BATCH}, seq=${SEQ_LEN}, ${tokensPerStep} tok/step (starting at round ${currentRound})`,
  );

  // Persistent gradient accumulators (before any beginStep)
  const accumGrads = params.map((p) =>
    api.zeros(p.shape, { device: "webgpu" }),
  );
  await api._runtime().forceAllPending();

  // ── Training loop ──
  for (; currentRound < OUTER_ROUNDS; currentRound++) {
    const round = currentRound; // local alias for log/header use
    const roundStart = performance.now();
    tokens = await fetchFreshTokens(tokenizer);

    const hasPeers = peerCount > 0;
    const globalSnapshot: Float32Array[] = [];
    for (const p of params)
      globalSnapshot.push(new Float32Array(await p.cpu()));

    const losses: number[] = [];
    for (let step = 0; step < INNER_STEPS; step++) {
      const l = await trainStep(
        api,
        model,
        innerOpt,
        params,
        tokens,
        (round * INNER_STEPS + step) * tokensPerStep,
        accumGrads,
      );
      losses.push(l);
      if (step % 10 === 0 || step < 3)
        log(`  step ${step}: loss=${l.toFixed(4)}`);
    }

    await api.flushStep();

    if (!hasPeers && !peerGrads) {
      log(`  solo (no peers) — continuing training`);
      await api._runtime().forceAllPending();
    } else {
      const pseudoGrads: Float32Array[] = [];
      for (let i = 0; i < params.length; i++) {
        const local = await params[i].cpu();
        const delta = new Float32Array(local.length);
        for (let j = 0; j < delta.length; j++)
          delta[j] = local[j] - globalSnapshot[i][j];
        pseudoGrads.push(delta);
      }

      // Send GRAD with 16-byte header: "GRAD" + u32 tokens + u32 round + f32 loss
      const avgLoss = losses.reduce((a, b) => a + b, 0) / losses.length;
      conn.send(JSON.stringify({ type: "request-neighbors", round }));
      const compressed = serializePseudoGrads(pseudoGrads);
      const gradHeader = Buffer.alloc(16);
      gradHeader.write("GRAD", 0);
      gradHeader.writeUInt32LE(INNER_STEPS * tokensPerStep, 4);
      gradHeader.writeUInt32LE(round, 8);
      gradHeader.writeFloatLE(avgLoss, 12);
      conn.send(Buffer.concat([gradHeader, compressed]));
      log(
        `Sent ${(compressed.length / 1024 / 1024).toFixed(1)}MB gradients (round ${round})`,
      );

      await new Promise((r) => setTimeout(r, 3000));

      // Token-weighted + staleness-discounted averaging
      const myTokens = INNER_STEPS * tokensPerStep;
      let totalTokens = myTokens;
      const avgGrads = pseudoGrads.map((pg) => new Float32Array(pg.length));
      for (let p = 0; p < pseudoGrads.length; p++)
        for (let j = 0; j < pseudoGrads[p].length; j++)
          avgGrads[p][j] += pseudoGrads[p][j] * myTokens;

      let numContributors = 1;
      if (peerGrads && peerGrads.length === params.length) {
        const roundLag = peerRound >= 0 ? Math.abs(round - peerRound) : 0;
        if (roundLag > 10) {
          log(`  discarding stale gradient (${roundLag} rounds behind)`);
          peerGrads = null;
        } else {
          const peerToks = peerTokenCount || myTokens;
          const stalenessWeight = 0.7 ** roundLag;
          const effectiveToks = peerToks * stalenessWeight;
          totalTokens += effectiveToks;
          for (let p = 0; p < peerGrads.length; p++)
            for (let j = 0; j < peerGrads[p].length; j++)
              avgGrads[p][j] += peerGrads[p][j] * effectiveToks;
          numContributors++;
          if (roundLag > 0)
            log(
              `  peer gradient from round ${peerRound} (lag=${roundLag}, weight=${stalenessWeight.toFixed(2)})`,
            );
          peerGrads = null;
        }
      }

      if (numContributors > 1) {
        for (let p = 0; p < avgGrads.length; p++)
          for (let j = 0; j < avgGrads[p].length; j++)
            avgGrads[p][j] /= totalTokens;

        // Compute outer step on CPU. stepFromCpu now manages its own step
        // boundaries (begin/markStep cycles) so each batch of 20 uploads is
        // fully fenced before the next batch — prevents Dawn "Buffer used in
        // submit while destroyed" warnings caused by step-scoped temps being
        // destroyed before their copy_ commands have actually been submitted.
        await outerOpt.stepFromCpu(params, globalSnapshot, avgGrads);
        log(`  averaged with ${numContributors} contributors`);
      } else {
        log("  solo round — keeping local params");
      }
    }

    // Save checkpoint
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
  clearInterval(pingInterval);
  conn.ws.close();
  outerOpt.dispose();
  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
