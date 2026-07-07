/**
 * Long-running e2e test that mirrors the toy-compartmentalization browser
 * demo's training+viz pattern. The demo runs an extra forward pass every
 * VIZ_INTERVAL=50 steps that does its own .cpu() readback. The user reports
 * "Input not ready: adamStep[0]" around step 250 in the browser; this test
 * reproduces that pattern in Node so we can debug locally.
 *
 * Usage:
 *   npx tsx examples/remote-training-demo/server.ts --port 9882 &
 *   STEPS=300 npx tsx tools/test-remote-viz-pattern.ts
 */
import WebSocket from "ws";
import { nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import {
  generateBatch,
  generateBatchForComp,
  setTransitionMatrices,
  VOCAB_SIZE_DATA,
} from "../examples/toy-compartmentalization/src/lib/data";
import type { Transport } from "../src/remote/client-engine";
import { createRemoteEngine } from "../src/remote/client-engine";
import {
  type BinaryFrame,
  decodeBinaryFrame,
  encodeBinaryFrame,
  valuesToTypedArray,
} from "../src/remote/binary-frame";
import type {
  DownloadParams,
  DownloadResult,
  ExecuteParams,
  ExecuteResult,
  ReadScalarParams,
  ReadScalarResult,
  ReleaseParams,
  ReleaseResult,
  UploadParams,
  UploadResult,
} from "../src/remote/rpc";

// Mirrors examples/toy-compartmentalization/src/lib/remote-transport.ts
class NodeTransport implements Transport {
  private ws!: WebSocket;
  private nextId = 1;
  private pendingText = new Map<number, (r: any) => void>();
  private pendingBinary = new Map<number, (f: BinaryFrame) => void>();
  sessionId = "";

  async connect(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(url);
      let helloed = false;
      this.ws.on("message", (data: Buffer | string, isBinary: boolean) => {
        if (!isBinary) {
          const msg = JSON.parse(data.toString());
          if (!helloed && msg.id === 0) {
            this.sessionId = msg.result.sessionId;
            helloed = true;
            resolve();
            return;
          }
          const tr = this.pendingText.get(msg.id);
          if (tr) {
            this.pendingText.delete(msg.id);
            tr(msg);
            return;
          }
          const br = this.pendingBinary.get(msg.id);
          if (br && msg.error) {
            this.pendingBinary.delete(msg.id);
            throw new Error(`[rpc] ${msg.error.message}`);
          }
          return;
        }
        // Binary frame — download response
        const buf = data as Buffer;
        const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
        const frame = decodeBinaryFrame(ab);
        const r = this.pendingBinary.get(frame.id);
        if (r) {
          this.pendingBinary.delete(frame.id);
          r(frame);
        }
      });
      this.ws.on("error", reject);
    });
  }
  private rpcText<T>(method: string, params: unknown): Promise<T> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pendingText.set(id, (msg) => {
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  execute(p: ExecuteParams): Promise<ExecuteResult> { return this.rpcText("execute", p); }
  readScalar(p: ReadScalarParams): Promise<ReadScalarResult> { return this.rpcText("readScalar", p); }
  release(p: ReleaseParams): Promise<ReleaseResult> { return this.rpcText("release", p); }
  async upload(p: UploadParams): Promise<UploadResult> {
    const id = this.nextId++;
    const values = valuesToTypedArray(p.values, p.dtype as any);
    const frame = encodeBinaryFrame({ id, dtype: p.dtype as any, shape: p.shape, values });
    const promise = new Promise<UploadResult>((resolve, reject) => {
      this.pendingText.set(id, (msg) => {
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result);
      });
    });
    this.ws.send(Buffer.from(frame));
    return promise;
  }
  async download(p: DownloadParams): Promise<DownloadResult> {
    const id = this.nextId++;
    const promise = new Promise<BinaryFrame>((resolve) => {
      this.pendingBinary.set(id, resolve);
    });
    this.ws.send(JSON.stringify({ id, method: "download", params: p }));
    const frame = await promise;
    return { values: Array.from(frame.values) };
  }
  close() { this.ws.close(); }
}

process.on("unhandledRejection", (reason) => {
  console.error("\n!!! UNHANDLED REJECTION !!!");
  console.error(reason);
  process.exit(1);
});
process.on("uncaughtException", (err) => {
  console.error("\n!!! UNCAUGHT EXCEPTION !!!");
  console.error(err);
  process.exit(1);
});

async function main() {
  const SERVER_URL = process.env.SERVER_URL ?? "ws://localhost:9882/ws";
  const STEPS = parseInt(process.env.STEPS ?? "300", 10);
  const VIZ_INTERVAL = 50;
  // Match toy-compartmentalization demo defaults exactly
  const PROBE_BATCH = 256;

  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA + 1, S = 10, B = 64;

  const transport = new NodeTransport();
  await transport.connect(SERVER_URL);

  const engine = createRemoteEngine(transport);
  const api = engine.torch;
  api.manualSeed(42);

  const m = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });
  const o = new Adam(m.parameters(), { lr: 1e-2 });
  await engine.preUpload(m.parameters());

  // Mirrors extractActivationsAndBeliefs() from the demo: separate forward pass,
  // grabs the last residual, reads it back to CPU.
  async function extractResidual(): Promise<void> {
    const batch = generateBatchForComp({ seqLen: S, batchSize: PROBE_BATCH }, 0);
    const tok = api.tensorFromArray(batch.tokens, [PROBE_BATCH, S], { dtype: "i32" });
    const residual = api.tidy(() => {
      const fwd = api.noGrad(() => m.forward(tok));
      const lastRes = fwd.residuals[fwd.residuals.length - 1];
      api.keep(lastRes);
      return lastRes;
    });
    tok.dispose();
    await residual.cpu();
    residual.dispose();
  }

  const losses: { step: number; loss: number }[] = [];
  const steptimes: number[] = [];

  for (let step = 0; step < STEPS; step++) {
    const t0 = performance.now();
    try {
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

      // Loss readback every 10 steps (matches demo's LOG_INTERVAL)
      if (step % 10 === 0) {
        try {
          const v = await loss.item();
          losses.push({ step, loss: v });
        } catch (e) {
          losses.push({ step, loss: NaN });
        }
      }

      await loss.backward(); loss.dispose();
      o.step(); o.zeroGrad();

      // Viz update — runs BEFORE endStep, mirroring the demo
      if (step > 0 && step % VIZ_INTERVAL === 0) {
        process.stderr.write(`  [step ${step}] viz update start\n`);
        const tViz0 = performance.now();
        await extractResidual();
        process.stderr.write(`  [step ${step}] viz update done in ${(performance.now() - tViz0).toFixed(0)}ms\n`);
      }

      await api.endStep();
      await engine.markStep([...o.getAllKeepTensors(), ...m.persistentTensors()]);
      steptimes.push(performance.now() - t0);

      // Log every step so we see exactly where it hangs
      if (step % 10 === 0 || step >= 240) {
        process.stderr.write(`step ${step}: ${steptimes[steptimes.length - 1].toFixed(0)}ms handles=${engine.handles.size()}\n`);
      }
    } catch (e) {
      const err = e as Error;
      console.error(`\n!!! FAILURE at step ${step} !!!`);
      console.error(err.message);
      console.error(err.stack);
      console.error(`\nHandles at failure: ${engine.handles.size()}`);
      console.error(`Stats: ${JSON.stringify(engine.stats, null, 2)}`);
      transport.close();
      process.exit(1);
    }
  }

  const lastN = 20;
  const steady = steptimes.slice(-lastN).reduce((a, b) => a + b, 0) / lastN;
  console.log(`\nCompleted ${STEPS} steps`);
  console.log(`Steady-state (last ${lastN}): ${steady.toFixed(1)}ms/step`);
  for (const { step, loss } of losses) {
    console.log(`  step ${step}: loss=${loss?.toFixed?.(4) ?? loss}`);
  }

  transport.close();
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
