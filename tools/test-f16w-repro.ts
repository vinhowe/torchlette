/**
 * Reproducer: connect to Hetzner, await real F16W from agent #1,
 * apply it, train. Watch for buffer warnings.
 */
import WebSocket from "ws";
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { NesterovOuterOptimizer } from "../src/distributed/outer-optimizer";
import { Torchlette } from "../src/frontend/torchlette";
import { normal_ } from "../src/nn/init";
import { Adam } from "../src/optim";

async function applyF16WeightsToParams(
  api: Torchlette,
  // biome-ignore lint/suspicious/noExplicitAny: framework
  params: any[],
  body: Buffer,
): Promise<void> {
  let off = 0;
  const np = body.readUInt32LE(off);
  off += 4;
  if (np !== params.length) {
    throw new Error(`F16W param count mismatch: ${np} vs ${params.length}`);
  }
  await api.beginStep();
  // biome-ignore lint/suspicious/noExplicitAny: framework
  let batch: any[] = [];
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
      if (exp === 0) val = (mant / 1024) * 2 ** -14;
      else if (exp === 31) val = mant ? Number.NaN : Number.POSITIVE_INFINITY;
      else val = 2 ** (exp - 15) * (1 + mant / 1024);
      f32[j] = sign ? -val : val;
    }
    const t = api.tensorFromArray(f32, params[i].shape, { device: "webgpu" });
    api.copy_(params[i], t);
    batch.push(t);
    if (i % 20 === 0) {
      await api._runtime().forceAllPending();
      await api.markStep();
      for (const tt of batch) tt.dispose();
      batch = [];
      await api.beginStep();
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  await api._runtime().forceAllPending();
  api.endStep();
  await api.markStep();
  for (const tt of batch) tt.dispose();
}

async function main() {
  const ok = await initWebGPU();
  if (!ok) throw new Error("WebGPU not available");

  const { setGPUMemoryLimit } = await import(
    "../src/backend/webgpu/memory-tracker"
  );
  setGPUMemoryLimit(31.5 * 1024 * 1024 * 1024);

  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(42);

  const { GPT2WithLoRA } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora"
  );
  const model = new GPT2WithLoRA(
    api,
    {
      vocabSize: 50257,
      blockSize: 1024,
      numLayers: 12,
      numHeads: 12,
      embedDim: 768,
      dropoutRate: 0,
    },
    { rank: 1, alpha: 1 },
    "webgpu",
  );
  const params = model.getAllParameters();
  for (const p of params) {
    if (p.shape.length >= 2) normal_(api, p, 0, 0.02);
  }
  await api._runtime().forceAllPending();
  console.error(`[repro] model: ${params.length} params`);

  model.train(true);
  model.enableCheckpointing(true);
  // biome-ignore lint/suspicious/noExplicitAny: framework field
  (model as any).fullCheckpoint = true;
  // biome-ignore lint/suspicious/noExplicitAny: framework field
  (model as any).setFullFinetuning(true);
  const innerOpt = new Adam(params, { lr: 1e-4, weightDecay: 0.1 }, api);

  // Connect, register
  const ws = new WebSocket("ws://5.78.181.14:443", {
    maxPayload: 500 * 1024 * 1024,
  });
  let pendingBlob: Buffer | null = null;
  let resolveSync: (() => void) | null = null;
  const syncPromise = new Promise<void>((r) => {
    resolveSync = r;
  });

  await new Promise<void>((resolve, reject) => {
    ws.on("open", () => {
      ws.send(
        JSON.stringify({
          type: "register",
          peerId: "repro-" + Date.now(),
          model: "gpt2-124m",
        }),
      );
    });
    ws.on("message", (data: Buffer) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.type === "registered") {
          console.error(
            `[repro] WS registered (peers=${msg.peers}, needsSync=${msg.needsSync})`,
          );
          resolve();
        }
      } catch {}
    });
    ws.on("error", reject);
    setTimeout(() => reject(new Error("WS timeout")), 10000);
  });

  // Match the agent: subscribe to the request-neighbors flow so we receive
  // GRAD blobs from agent#1 when it ends its rounds.
  ws.on("message", async (data: Buffer) => {
    try {
      JSON.parse(data.toString());
    } catch {
      const raw = Buffer.from(data);
      if (raw.length > 8 && raw.toString("utf8", 0, 4) === "F16W") {
        if (!resolveSync) {
          console.error(
            `[repro] Ignoring F16W (${(raw.length / 1024 / 1024).toFixed(1)}MB)`,
          );
          return;
        }
        const srcRound = raw.readUInt32LE(4);
        pendingBlob = Buffer.from(raw.subarray(8));
        console.error(
          `[repro] Buffered F16W (${(raw.length / 1024 / 1024).toFixed(1)}MB, source round ${srcRound})`,
        );
        const r = resolveSync;
        resolveSync = null;
        r();
      } else if (raw.length > 16 && raw.toString("utf8", 0, 4) === "GRAD") {
        // Mimic agent's GRAD handler: deserialize 124M params worth of E3M0
        // pseudo-grad data, allocating a giant Float32Array. This is the
        // memory-pressure event suspected of triggering GC mid-train.
        const tokens = raw.readUInt32LE(4);
        const round = raw.readUInt32LE(8);
        const loss = raw.readFloatLE(12);
        const { e3m0Dequantize } = await import(
          "../src/distributed/e3m0"
        );
        const body = raw.subarray(16);
        let off = 0;
        const numParams = body.readUInt32LE(off);
        off += 4;
        const result: Float32Array[] = [];
        for (let i = 0; i < numParams; i++) {
          const nv = body.readUInt32LE(off);
          off += 4;
          const cl = body.readUInt32LE(off);
          off += 4;
          const sl = body.readUInt32LE(off);
          off += 4;
          const codes = new Uint32Array(
            body.buffer.slice(
              body.byteOffset + off,
              body.byteOffset + off + cl,
            ),
          );
          off += cl;
          const scales = new Uint8Array(
            body.buffer.slice(
              body.byteOffset + off,
              body.byteOffset + off + sl,
            ),
          );
          off += sl;
          result.push(
            e3m0Dequantize(codes, scales, Math.ceil(nv / 8) * 8).slice(0, nv),
          );
        }
        const totalBytes = result.reduce((s, a) => s + a.byteLength, 0);
        console.error(
          `[repro] GRAD recv: ${(raw.length / 1024 / 1024).toFixed(1)}MB compressed → ${(totalBytes / 1024 / 1024).toFixed(1)}MB f32 arrays (round ${round}, loss ${loss.toFixed(2)})`,
        );
      }
    }
  });

  console.error(`[repro] Awaiting F16W blob...`);
  const interval = setInterval(() => {
    if (resolveSync) {
      console.error(`[repro] Re-requesting weights`);
      ws.send(JSON.stringify({ type: "request-weights" }));
    }
  }, 20000);
  ws.send(JSON.stringify({ type: "request-weights" }));
  await Promise.race([
    syncPromise,
    new Promise((_, rej) =>
      setTimeout(() => rej(new Error("sync timeout")), 300000),
    ),
  ]);
  clearInterval(interval);

  console.error(`[repro] Applying F16W to params...`);
  if (!pendingBlob) throw new Error("no blob");
  // pendingBlob already has F16W+round stripped; starts at numParams.
  await applyF16WeightsToParams(api, params, pendingBlob);
  console.error(`[repro] Applied`);

  const outerOpt = new NesterovOuterOptimizer(api, { lr: 0.7, momentum: 0.9 });

  // Mimic agent: create accumGrads AFTER F16W apply.
  console.error(`[repro] creating accumGrads`);
  const accumGrads = params.map((p) =>
    api.zeros(p.shape, { device: "webgpu" }),
  );
  await api._runtime().forceAllPending();

  // Many ROUNDS, each with 20 inner steps + outer step. Bug appeared at round 3 in agent.
  console.error(`[repro] full round phase begin`);
  const BATCH = 1;
  const SEQ = 256;
  const INNER_STEPS = 20;
  const ROUNDS = parseInt(process.env.ROUNDS ?? "5", 10);
  for (let round = 0; round < ROUNDS; round++) {
    // snapshot
    const snap: Float32Array[] = [];
    for (const p of params) snap.push(new Float32Array(await p.cpu()));
    // inner loop (use trainStep-like pattern)
    for (let step = 0; step < INNER_STEPS; step++) {
    for (const ag of accumGrads) api.zero_(ag);
    const inputData: number[] = [];
    const targetData: number[] = [];
    for (let i = 0; i < BATCH * SEQ; i++) {
      inputData.push(Math.floor(Math.random() * 50257));
      targetData.push(Math.floor(Math.random() * 50257));
    }
    await api.beginStep();
    const input = api.tensorFromArray(inputData, [BATCH, SEQ], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(targetData, [BATCH, SEQ], {
      device: "webgpu",
    });
    const loss = api.tidy(() => {
      const l = api.autocast(
        // biome-ignore lint/suspicious/noExplicitAny: framework
        () => (model as any).forwardWithLoss(input, target).loss,
      );
      api.keep(l);
      return l;
    });
    const lossVal = await loss.item();
    await loss.backward();
    for (let i = 0; i < params.length; i++) {
      // biome-ignore lint/suspicious/noExplicitAny: framework
      if ((params[i] as any).grad)
        // biome-ignore lint/suspicious/noExplicitAny: framework
        api.add_(accumGrads[i], (params[i] as any).grad);
    }
    await api._runtime().forceAllPending();
    innerOpt.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();

    await api.beginStep();
    for (let i = 0; i < params.length; i++)
      // biome-ignore lint/suspicious/noExplicitAny: framework
      (params[i] as any)._setGrad(api.mul(accumGrads[i], 1));
    innerOpt.step();
    innerOpt.zeroGrad();
    api.endStep();
    await api.markStep();

      console.error(`[repro] r${round} step ${step}: loss=${lossVal.toFixed(4)}`);
    }
    // outer step using stepFromCpu (matches agent's path)
    const fakeAvgRound = params.map((p) => {
      const sz = p.shape.reduce((a, b) => a * b, 1);
      const a = new Float32Array(sz);
      for (let i = 0; i < sz; i++) a[i] = (Math.random() - 0.5) * 0.02;
      return a;
    });
    await api.beginStep();
    await outerOpt.stepFromCpu(params, snap, fakeAvgRound);
    api.endStep();
    await api.markStep();
    console.error(`[repro] r${round} OUTER STEP applied`);
  }
  console.error(`[repro] full round phase end`);

  ws.close();
  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(`[repro] FATAL: ${e}`);
  process.exit(1);
});
