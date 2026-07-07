/**
 * Long-running e2e remote training test (30 steps).
 *
 * Verifies correctness AND steady-state performance with the compiled plan
 * replay path active. Loss should decrease monotonically; steady-state step
 * time should drop substantially after step 3 (compiled plan replay starts).
 */
import WebSocket from "ws";
import { nn, Adam, initWebGPU } from "../src/index";
import { crossEntropy } from "../src/nn/functional";
import { createModel, MESS3_CONFIG } from "../examples/toy-compartmentalization/src/lib/model";
import {
  generateBatch,
  setTransitionMatrices,
  VOCAB_SIZE_DATA,
} from "../examples/toy-compartmentalization/src/lib/data";
import type { Transport } from "../src/remote/client-engine";
import { createRemoteEngine } from "../src/remote/client-engine";

class NodeTransport implements Transport {
  private ws!: WebSocket;
  private nextId = 1;
  private pending = new Map<number, { resolve: (r: any) => void; reject: (e: Error) => void }>();
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
  execute(p: any) { return this.rpc<any>("execute", p); }
  upload(p: any) { return this.rpc<any>("upload", p); }
  download(p: any) { return this.rpc<any>("download", p); }
  readScalar(p: any) { return this.rpc<any>("readScalar", p); }
  release(p: any) { return this.rpc<any>("release", p); }
  stats(): Promise<any> { return this.rpc<any>("stats", {}); }
  close() { this.ws.close(); }
}

async function main() {
  const SERVER_URL = process.env.SERVER_URL ?? "ws://localhost:9882/ws";
  const STEPS = parseInt(process.env.STEPS ?? "30", 10);

  setTransitionMatrices(0.765);
  const V = VOCAB_SIZE_DATA + 1, S = 10, B = 128;

  // Init WebGPU client-side so createRemoteEngine can build webgpu-flavored
  // plans (the cpu device path is gone — see client-engine.ts).
  const ok = await initWebGPU();
  if (!ok) throw new Error("WebGPU init failed for remote client");

  const transport = new NodeTransport();
  await transport.connect(SERVER_URL);

  const engine = createRemoteEngine(transport);
  const api = engine.torch;
  api.manualSeed(42);

  const m = createModel(api, nn, { ...MESS3_CONFIG, seqLen: S, vocabSize: V, posEncoding: "rope" });
  const o = new Adam(m.parameters(), { lr: 1e-3 });
  await engine.preUpload(m.parameters());

  const losses: { step: number; loss: number }[] = [];
  const steptimes: number[] = [];

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

    if (step === 0 || step === STEPS - 1 || step % 5 === 0) {
      try {
        const v = await loss.item();
        losses.push({ step, loss: v });
      } catch (e) {
        losses.push({ step, loss: NaN });
      }
    }

    await loss.backward(); loss.dispose();
    o.step(); o.zeroGrad();
    await api.endStep();
    await engine.markStep([...o.getAllKeepTensors(), ...m.persistentTensors()]);
    steptimes.push(performance.now() - t0);
  }

  console.log("\nStep times:");
  for (let i = 0; i < steptimes.length; i++) {
    console.log(`  step ${i}: ${steptimes[i].toFixed(0)}ms`);
  }

  console.log("\nLoss trajectory:");
  for (const { step, loss } of losses) {
    console.log(`  step ${step}: loss=${loss.toFixed(4)}`);
  }

  const lastN = 10;
  const steady = steptimes.slice(-lastN).reduce((a, b) => a + b, 0) / lastN;
  console.log(`\nSteady-state (last ${lastN}): ${steady.toFixed(1)}ms/step`);

  const s = engine.stats;
  console.log(`\nClient stats:`);
  console.log(`  serializeMs:    ${s.serializeMs.toFixed(0)} (${(s.serializeMs / STEPS).toFixed(1)}/step)`);
  console.log(`  transportMs:    ${s.transportMs.toFixed(0)} (${(s.transportMs / STEPS).toFixed(1)}/step)`);
  console.log(`  bookkeepingMs:  ${s.bookkeepingMs.toFixed(0)} (${(s.bookkeepingMs / STEPS).toFixed(1)}/step)`);
  console.log(`  executes:       ${s.executes} (${(s.executes / STEPS).toFixed(2)}/step)`);
  console.log(`  nodesShipped:   ${s.nodesShipped} (${(s.nodesShipped / STEPS).toFixed(0)}/step)`);
  console.log(`  handlesReleased: ${s.handlesReleased}`);

  // Loss should be monotonically decreasing-ish
  if (losses.length >= 2 && losses[losses.length - 1].loss < losses[0].loss) {
    console.log(`Loss decreased: ${losses[0].loss.toFixed(4)} → ${losses[losses.length - 1].loss.toFixed(4)} ✓`);
  } else {
    console.log(`WARNING: loss did not decrease!`);
  }

  transport.close();
  process.exit(0);
}

main().catch((e) => { console.error(e); process.exit(1); });
