/**
 * End-to-end remote training test: Node.js client → WebSocket → real server.
 *
 * Uses createRemoteEngine (same as browser demo) with a Node.js WebSocket.
 *
 * Usage:
 *   1. Start server: npx tsx examples/remote-training-demo/server.ts --port 9882
 *   2. Run this:     npx tsx tools/test-remote-e2e.ts
 */
import WebSocket from "ws";
import { nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import {
  generateBatch,
  setTransitionMatrices,
  VOCAB_SIZE_DATA,
} from "../examples/toy-compartmentalization/src/lib/data";
import type {
  ExecuteParams,
  ExecuteResult,
  DownloadParams,
  DownloadResult,
  ReadScalarParams,
  ReadScalarResult,
  UploadParams,
  UploadResult,
  ReleaseParams,
  ReleaseResult,
  RpcResponse,
  HelloResult,
} from "../src/remote/rpc";
import type { Transport } from "../src/remote/client-engine";
import { createRemoteEngine } from "../src/remote/client-engine";

// ============================================================================
// Node.js WebSocket Transport (mirrors remote-transport.ts for browser)
// ============================================================================

class NodeTransport implements Transport {
  private ws!: WebSocket;
  private nextId = 1;
  private pending = new Map<number, { resolve: (r: any) => void; reject: (e: Error) => void }>();
  sessionId = "";

  async connect(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(url);
      let helloed = false;
      this.ws.on("open", () => console.log("[rpc] connected"));
      this.ws.on("message", (data: Buffer | string, isBinary: boolean) => {
        if (!isBinary) {
          const msg = JSON.parse(data.toString());
          if (!helloed && msg.id === 0) {
            this.sessionId = msg.result.sessionId;
            console.log(`[rpc] session ${this.sessionId}`);
            helloed = true;
            resolve();
            return;
          }
          const p = this.pending.get(msg.id);
          if (p) {
            this.pending.delete(msg.id);
            if (msg.error) p.reject(new Error(msg.error.message));
            else p.resolve(msg.result);
          }
        }
      });
      this.ws.on("error", reject);
    });
  }

  private rpc<T>(method: string, params: unknown): Promise<T> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  execute(params: ExecuteParams): Promise<ExecuteResult> { return this.rpc("execute", params); }
  upload(params: UploadParams): Promise<UploadResult> { return this.rpc("upload", params); }
  download(params: DownloadParams): Promise<DownloadResult> { return this.rpc("download", params); }
  readScalar(params: ReadScalarParams): Promise<ReadScalarResult> { return this.rpc("readScalar", params); }
  release(params: ReleaseParams): Promise<ReleaseResult> { return this.rpc("release", params); }

  close() { this.ws.close(); }
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const SERVER_URL = process.env.SERVER_URL ?? "ws://localhost:9882/ws";
  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA * 1 + 1, S = 10, B = 128;

  const transport = new NodeTransport();
  await transport.connect(SERVER_URL);

  const engine = createRemoteEngine(transport);
  const api = engine.torch;
  api.manualSeed(42);
  const m = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });
  const o = new Adam(m.parameters(), { lr: 1e-3 });

  // Pre-upload weights before endStep so they're still pending
  const uploaded = await engine.preUpload(m.parameters());
  console.log(`[preUpload] ${uploaded} param tensors`);

  const STEPS = 8;
  const LOG_INTERVAL = 10;
  for (let step = 0; step < STEPS; step++) {
    const t0 = performance.now();
    await api.beginStep();
    const batch = generateBatch({ seqLen: S, batchSize: B });
    const tok = api.tensorFromArray(batch.tokens, [B, S], { dtype: "i32" });
    const tgt = api.tensorFromArray(batch.targets as number[], [B * (S - 1)], { dtype: "i32" });

    const loss = api.tidy(() => {
      const fwd = m.forward(tok);
      const logits = fwd.logits.narrow(1, 0, S - 1).contiguous().reshape([B * (S - 1), V]);
      const l = crossEntropy(api, logits, tgt);
      api.keep(l);
      return l;
    });
    tok.dispose(); tgt.dispose();

    // Skip item() on most steps (like the browser's LOG_INTERVAL=10)
    if (step % LOG_INTERVAL === 0) {
      try {
        const v = await loss.item();
        console.log(`step ${step}: loss=${typeof v === "number" ? v.toFixed(4) : v}`);
      } catch (e) {
        console.log(`step ${step}: item() failed: ${(e as Error).message}`);
      }
    }

    await loss.backward(); loss.dispose();
    o.step(); o.zeroGrad();

    await api.endStep();
    // Debug: check if storage 55 exists
    console.log(`  handle 55: ${engine.handles.getHandle(55) ?? 'MISSING'}`);
    const elapsed = performance.now() - t0;
    console.log(`step ${step}: ${elapsed.toFixed(0)}ms, handles=${engine.handles.size()}`);
  }

  console.log(`\nFinal stats: executes=${engine.stats.executes} handles=${engine.handles.size()} released=${engine.stats.handlesReleased}`);
  transport.close();
  process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
