/**
 * DiLoCo Agent v2.
 *
 * Wires together:
 *   - WebSocketRelayTransport against server/diloco-server-v2.cjs
 *   - WebGPUGPT2Trainer (real GPT-2 + Adam + Nesterov)
 *   - HierarchicalBarrierStateMachine (sync barrier, cluster-aware)
 *
 * The existing diloco-webrtc-agent.ts stays in place — this is a parallel
 * agent that talks to the v2 relay (different port). Both can run during
 * transition.
 *
 * Env knobs:
 *   SERVER_URL                ws://… of the v2 relay (default :8443 on localhost)
 *   ROUNDS, STEPS, BATCH_SIZE, SEQ_LEN, ACCUM_STEPS, LR
 *   OUTER_LR, OUTER_MU
 *   NUM_LAYERS, NUM_HEADS, EMBED_DIM, MODEL_DIR
 *   HF_DATASET, HF_CONFIG, HF_ROWS
 *   SEED
 *   QUORUM_MIN, QUORUM_TARGET_FRAC
 *   PEER_ID
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { HierarchicalBarrierStateMachine } from "../src/distributed/protocol/hierarchical-state-machine";
import {
  type TokenSource,
  WebGPUGPT2Trainer,
} from "../src/distributed/protocol/webgpu-gpt2-trainer";
import { WebSocketRelayTransport } from "../src/distributed/transports/websocket-relay";
import { Torchlette } from "../src/frontend/torchlette";
import { saveCheckpoint } from "./diloco-checkpoint";

// ── Config ──
const SEED = parseInt(process.env.SEED ?? "42", 10);
const ROUNDS = parseInt(process.env.ROUNDS ?? "100", 10);
const INNER_STEPS = parseInt(process.env.STEPS ?? "20", 10);
const BATCH_SIZE = parseInt(process.env.BATCH_SIZE ?? "4", 10);
const SEQ_LEN = parseInt(process.env.SEQ_LEN ?? "512", 10);
const ACCUM_STEPS = parseInt(process.env.ACCUM_STEPS ?? "1", 10);
const LR = parseFloat(process.env.LR ?? "1e-4");
const OUTER_LR = parseFloat(process.env.OUTER_LR ?? "0.7");
const OUTER_MU = parseFloat(process.env.OUTER_MU ?? "0.9");

const NUM_LAYERS = parseInt(process.env.NUM_LAYERS ?? "12", 10);
const NUM_HEADS = parseInt(process.env.NUM_HEADS ?? "12", 10);
const EMBED_DIM = parseInt(process.env.EMBED_DIM ?? "768", 10);
const MODEL_DIR = process.env.MODEL ?? "gpt2";

const SERVER_URL = process.env.SERVER_URL ?? "ws://127.0.0.1:8443";
const PEER_ID = process.env.PEER_ID ?? `v2-${Date.now()}`;
const QUORUM_MIN = parseInt(process.env.QUORUM_MIN ?? "2", 10);
const QUORUM_TARGET_FRAC = parseFloat(
  process.env.QUORUM_TARGET_FRAC ?? "1.0",
);

const HF_DATASET = process.env.HF_DATASET ?? "HuggingFaceFW/fineweb-edu";
const HF_CONFIG = process.env.HF_CONFIG ?? "sample-10BT";
const HF_TOTAL_ROWS = parseInt(process.env.HF_ROWS ?? "9672101", 10);
const HF_FETCH_ROWS = 100;

const log = (msg: string) => console.error(`[v2] ${msg}`);

// ── Tokenizer + HF data source ──
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

/**
 * Reads a pre-tokenized TinyStories blob (uint16 token ids) from disk and
 * returns deterministic random windows. Lets torchlette training match the
 * PyTorch baseline's data pipeline (which reads the same .bin file), so
 * convergence comparisons aren't muddied by HF datasets-server variance.
 */
class LocalTokenSource implements TokenSource {
  private cached: Uint16Array | null = null;
  constructor(
    private readonly path: string,
    private readonly seed: number,
  ) {}
  private load(): Uint16Array {
    if (this.cached) return this.cached;
    const buf = fs.readFileSync(this.path);
    this.cached = new Uint16Array(
      buf.buffer,
      buf.byteOffset,
      buf.byteLength / 2,
    );
    log(`Local tokens: ${this.cached.length.toLocaleString()} from ${this.path}`);
    return this.cached;
  }
  async fetch(_minTokens: number): Promise<ArrayLike<number>> {
    // Return the entire Uint16Array as ArrayLike<number>. The trainer
    // indexes into it with `(round * inner_steps + step) * tokens_per_step`
    // % maxStart — with a ~474M-token cache that spreads inner-step
    // windows across diverse stories, matching PyTorch's random-window
    // behavior closely enough for convergence comparison. No copy.
    return this.load();
  }
}

class HFTokenSource implements TokenSource {
  // biome-ignore lint/suspicious/noExplicitAny: tokenizer type comes from a dynamic import
  constructor(private readonly tokenizer: any) {}
  async fetch(minTokens: number): Promise<number[]> {
    let allText = "";
    let consecutiveFailures = 0;
    while (true) {
      const offset = Math.floor(
        Math.random() * (HF_TOTAL_ROWS - HF_FETCH_ROWS),
      );
      const url = `https://datasets-server.huggingface.co/rows?dataset=${HF_DATASET}&config=${HF_CONFIG}&split=train&offset=${offset}&length=${HF_FETCH_ROWS}`;
      try {
        const resp = await fetch(url, { signal: AbortSignal.timeout(30_000) });
        const ct = resp.headers.get("content-type") ?? "";
        if (!resp.ok || !ct.includes("application/json")) {
          throw new Error(`HF ${resp.status} ${ct.split(";")[0]}`);
        }
        const data = await resp.json();
        if (!data.rows) throw new Error("HF response missing rows");
        // biome-ignore lint/suspicious/noExplicitAny: HF row shape isn't typed
        const text = data.rows.map((r: any) => r.row.text).join("\n\n");
        allText += (allText ? "\n\n" : "") + text;
        consecutiveFailures = 0;
        const tokens: number[] = this.tokenizer.encode(allText);
        if (tokens.length >= minTokens) {
          return tokens;
        }
      } catch (e) {
        consecutiveFailures++;
        const wait = Math.min(
          60_000,
          2_000 * 2 ** Math.min(consecutiveFailures, 5),
        );
        log(
          `HF fetch failed (${consecutiveFailures}): ${(e as Error).message} — retry in ${wait / 1000}s`,
        );
        await new Promise((r) => setTimeout(r, wait));
      }
    }
  }
}

// ── Main ──
async function main(): Promise<void> {
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

  const localTokensPath = process.env.LOCAL_TOKENS;
  let tokenSource: TokenSource;
  if (localTokensPath) {
    tokenSource = new LocalTokenSource(localTokensPath, SEED);
  } else {
    const tokenizer = await loadTokenizer(MODEL_DIR);
    tokenSource = new HFTokenSource(tokenizer);
  }

  const trainer = new WebGPUGPT2Trainer({
    api,
    modelConfig: {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: NUM_LAYERS,
      numHeads: NUM_HEADS,
      embedDim: EMBED_DIM,
      dropoutRate: 0,
    },
    tokenSource,
    innerLr: LR,
    outerLr: OUTER_LR,
    outerMu: OUTER_MU,
    innerSteps: INNER_STEPS,
    batchSize: BATCH_SIZE,
    seqLen: SEQ_LEN,
    accumSteps: ACCUM_STEPS,
    weightDecay: parseFloat(process.env.WEIGHT_DECAY ?? "0.1"),
    checkpointing: process.env.CHECKPOINTING !== "0",
    useAutocast: process.env.USE_AUTOCAST !== "0",
    gradClipNorm: parseFloat(process.env.GRAD_CLIP ?? "1.0"),
    log,
  });
  await trainer.initialize();

  log(`Connecting to ${SERVER_URL} as ${PEER_ID}`);
  const transport = await WebSocketRelayTransport.create({
    serverUrl: SERVER_URL,
    peerId: PEER_ID,
    model: "gpt2-124m",
    log,
  });

  const dlMs = parseInt(process.env.DEADLINE_MS ?? "60000", 10);
  const sm = new HierarchicalBarrierStateMachine(transport, trainer, {
    quorumMin: QUORUM_MIN,
    quorumTargetFrac: QUORUM_TARGET_FRAC,
    intraDeadlineMs: dlMs,
    interDeadlineMs: dlMs,
    globalDeadlineMs: dlMs,
    f16wDebounceMs: 10_000,
  });

  await sm.awaitJoined();
  const self = sm.getSelf();
  log(
    `Joined: peerId=${self?.peerId} cluster=${self?.clusterId}${self?.isHead ? "/head" : ""}`,
  );

  const { getGPUMemoryStats: streamMem } = await import(
    "../src/backend/webgpu/memory-tracker"
  );
  const { bufferPool: streamPool } = await import(
    "../src/backend/webgpu/buffer-pool"
  );
  const ckptPath = process.env.CHECKPOINT_PATH;
  const ckptEvery = parseInt(process.env.CHECKPOINT_EVERY ?? "10", 10);
  const writeCkpt = async (tag: string) => {
    if (!ckptPath) return;
    const params = await trainer.snapshotAnchor();
    const shapes = trainer.paramShapes();
    fs.mkdirSync(path.dirname(ckptPath), { recursive: true });
    saveCheckpoint(ckptPath, shapes, params);
    log(`Checkpoint saved (${tag}): ${ckptPath} (${params.length} tensors)`);
  };
  let ckptInFlight = false;
  sm.onReport = (r) => {
    const mem = streamMem();
    const pool = streamPool.stats();
    const rss = process.memoryUsage().rss;
    console.error(
      `STATS ${JSON.stringify({
        t: new Date().toISOString(),
        round: r.round,
        anchor_round: r.anchorAfter,
        outer_step: r.outerStepTaken,
        contributors: r.contributors,
        clusters: r.clustersContributed,
        f16w_applied: r.f16wApplied,
        loss: Number.isFinite(r.innerLoss) ? +r.innerLoss.toFixed(4) : null,
        gpu_mb: Math.round(mem.currentBytes / 1e6),
        peak_mb: Math.round(mem.peakBytes / 1e6),
        pool_mb: Math.round(pool.pooledBytes / 1e6),
        cpu_rss_mb: Math.round(rss / 1e6),
      })}`,
    );
    if (
      ckptPath &&
      r.outerStepTaken &&
      r.anchorAfter % ckptEvery === 0 &&
      !ckptInFlight
    ) {
      ckptInFlight = true;
      writeCkpt(`anchor=${r.anchorAfter}`).finally(() => {
        ckptInFlight = false;
      });
    }
  };
  let shuttingDown = false;
  const shutdown = async () => {
    if (shuttingDown) return;
    shuttingDown = true;
    log("SIGTERM/SIGINT received — saving checkpoint and exiting");
    try {
      await writeCkpt("shutdown");
    } catch (e) {
      log(`shutdown checkpoint failed: ${e}`);
    }
    transport.close();
    await destroyWebGPU();
    process.exit(0);
  };
  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
  const reports = await sm.run(ROUNDS);
  log(
    `Training complete: ${reports.length} rounds, anchor=${sm.getAnchorRound()}`,
  );

  // Per-round STATS streamed via sm.onReport; periodic checkpoint saves
  // there too. Final save at end-of-run.
  log(`Reports collected: ${reports.length}`);
  await writeCkpt("final");

  transport.close();
  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  log(`FATAL: ${e}`);
  process.exit(1);
});
