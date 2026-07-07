/**
 * Reproduce + diagnose the arena buffer leak.
 *
 * Runs a long-ish remote training session and dumps GPU memory + buffer pool
 * state every step to find where buffers accumulate.
 */
import WebSocket from "ws";
import { nn, Adam } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import { generateBatch, setTransitionMatrices, VOCAB_SIZE_DATA } from "../examples/toy-compartmentalization/src/lib/data";
import type { Transport } from "../src/remote/client-engine";
import { createRemoteEngine } from "../src/remote/client-engine";
import { decodeBinaryFrame } from "../src/remote/binary-frame";

class T implements Transport {
  private ws!: WebSocket;
  private nextId = 1;
  private pendingText = new Map<number, (r: any) => void>();
  private pendingBinary = new Map<number, (f: any) => void>();
  sessionId = "";
  async connect(url: string) {
    return new Promise<void>((resolve, reject) => {
      this.ws = new WebSocket(url);
      let helloed = false;
      this.ws.on("message", (data: Buffer | string, isBinary: boolean) => {
        if (!isBinary) {
          const msg = JSON.parse(data.toString());
          if (!helloed && msg.id === 0) { this.sessionId = msg.result.sessionId; helloed = true; resolve(); return; }
          const tr = this.pendingText.get(msg.id);
          if (tr) { this.pendingText.delete(msg.id); tr(msg); return; }
        } else {
          const buf = data as Buffer;
          const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
          const f = decodeBinaryFrame(ab);
          const r = this.pendingBinary.get(f.id);
          if (r) { this.pendingBinary.delete(f.id); r(f); }
        }
      });
      this.ws.on("error", reject);
    });
  }
  rpc<T>(method: string, params: unknown): Promise<T> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pendingText.set(id, (msg) => { if (msg.error) reject(new Error(msg.error.message)); else resolve(msg.result); });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  execute(p: any) { return this.rpc("execute", p); }
  upload(p: any) { return this.rpc("upload", p); }
  download(p: any) {
    const id = this.nextId++;
    const promise = new Promise<any>((resolve) => { this.pendingBinary.set(id, resolve); });
    this.ws.send(JSON.stringify({ id, method: "download", params: p }));
    return promise.then((f) => ({ values: Array.from(f.values) }));
  }
  readScalar(p: any) { return this.rpc("readScalar", p); }
  release(p: any) { return this.rpc("release", p); }
  stats() { return this.rpc("stats", {}); }
}

async function main() {
  const SERVER_URL = process.env.SERVER_URL ?? "ws://localhost:9882/ws";
  const STEPS = parseInt(process.env.STEPS ?? "50", 10);

  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA + 1, S = 10, B = 64;

  const transport = new T();
  await transport.connect(SERVER_URL);

  const engine = createRemoteEngine(transport);
  const api = engine.torch;
  api.manualSeed(42);

  const m = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });
  const o = new Adam(m.parameters(), { lr: 1e-2 });
  await engine.preUpload(m.parameters());

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

    if (step % 5 === 0) await loss.item();
    await loss.backward(); loss.dispose();
    o.step(); o.zeroGrad();
    await api.endStep();
    await engine.markStep([...o.getAllKeepTensors(), ...m.persistentTensors()]);

    const elapsed = performance.now() - t0;
    if (step % 5 === 0 || step >= STEPS - 5) {
      try {
        const s = await transport.stats() as any;
        console.log(
          `step ${step}: ${elapsed.toFixed(0)}ms | handles=${s.handles} gpu=${s.gpu?.currentMB ?? "?"}MB peak=${s.gpu?.peakMB ?? "?"}MB | clientHandles=${engine.handles.size()}`
        );
      } catch {
        console.log(`step ${step}: ${elapsed.toFixed(0)}ms (no stats)`);
      }
    }
  }
  process.exit(0);
}
main().catch((e) => { console.error(e); process.exit(1); });
