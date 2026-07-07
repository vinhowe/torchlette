/**
 * Minimum reproducer for the "Buffer used in submit while destroyed" warning
 * that fires during the agent's inner training loop. Strips the agent down to
 * the smallest possible loop on the same primitives:
 *   - GPT2WithLoRA model (full-finetuning mode, like the agent)
 *   - one beginStep/markStep cycle per training step
 *   - forward → backward → Adam.step
 *
 * Run:
 *   VULKAN_DEVICE_INDEX=N \
 *   LD_LIBRARY_PATH=/mnt/pccfs2/backed_up/vin/dev/torchlette/tools/vk-shim:$LD_LIBRARY_PATH \
 *   npx tsx tools/repro-train-warning.ts
 *
 * Default config: 6-layer model (≈30M params), batch=1, seq=128, 8 steps.
 * Counts "used in submit while destroyed" lines on stderr.
 */
import { destroyWebGPU, initWebGPU } from "../src/backend/webgpu";
import { NesterovOuterOptimizer } from "../src/distributed/outer-optimizer";
import { Torchlette } from "../src/frontend/torchlette";
import { clipGradNorm_ } from "../src/nn/clip-grad";
import { normal_ } from "../src/nn/init";
import { Adam } from "../src/optim";

const NUM_LAYERS = parseInt(process.env.NUM_LAYERS ?? "6", 10);
const NUM_HEADS = parseInt(process.env.NUM_HEADS ?? "6", 10);
const EMBED_DIM = parseInt(process.env.EMBED_DIM ?? "384", 10);
const VOCAB_SIZE = parseInt(process.env.VOCAB ?? "50257", 10);
const BLOCK_SIZE = parseInt(process.env.BLOCK ?? "1024", 10);
const BATCH_SIZE = parseInt(process.env.BATCH ?? "1", 10);
const SEQ_LEN = parseInt(process.env.SEQ ?? "128", 10);
const NUM_STEPS = parseInt(process.env.STEPS ?? "8", 10);
const LR = parseFloat(process.env.LR ?? "1e-4");

// Capture Dawn validation messages — they go to stderr but Dawn writes them
// directly via fprintf, so a process-level wrapper around process.stderr.write
// catches every line.
let warnCount = 0;
const origWrite = process.stderr.write.bind(process.stderr);
// biome-ignore lint/suspicious/noExplicitAny: stderr.write override
(process.stderr as any).write = (chunk: any, ...args: any[]): boolean => {
  if (typeof chunk === "string" && chunk.includes("used in submit while destroyed")) {
    warnCount++;
  } else if (chunk instanceof Buffer || chunk instanceof Uint8Array) {
    if (Buffer.from(chunk).toString("utf8").includes("used in submit while destroyed")) {
      warnCount++;
    }
  }
  return origWrite(chunk, ...args);
};

async function main() {
  const ok = await initWebGPU();
  if (!ok) {
    console.error("WebGPU init failed");
    process.exit(1);
  }

  const { setGPUMemoryLimit } = await import(
    "../src/backend/webgpu/memory-tracker"
  );
  setGPUMemoryLimit(31 * 1024 * 1024 * 1024);

  const api = new Torchlette("webgpu", { enableFusion: true });
  api.manualSeed(42);

  const { GPT2WithLoRA } = await import(
    "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora"
  );
  const config = {
    vocabSize: VOCAB_SIZE,
    blockSize: BLOCK_SIZE,
    numLayers: NUM_LAYERS,
    numHeads: NUM_HEADS,
    embedDim: EMBED_DIM,
    dropoutRate: 0,
  };
  const model = new GPT2WithLoRA(api, config, { rank: 1, alpha: 1 }, "webgpu");
  const params = model.getAllParameters();
  for (const p of params) {
    if (p.shape.length >= 2) normal_(api, p, 0, 0.02);
  }
  await api._runtime().forceAllPending();

  model.train(true);
  model.enableCheckpointing(true);
  // biome-ignore lint/suspicious/noExplicitAny: framework field
  (model as any).fullCheckpoint = true;
  // biome-ignore lint/suspicious/noExplicitAny: framework field
  (model as any).setFullFinetuning(true);

  const optimizer = new Adam(params, { lr: LR, weightDecay: 0.1 }, api);
  const outerOpt = new NesterovOuterOptimizer(api, { lr: 0.7, momentum: 0.9 });
  const accumGrads = params.map((p) => api.zeros(p.shape, { device: "webgpu" }));
  await api._runtime().forceAllPending();
  const ROUND_LEN = parseInt(process.env.ROUND_LEN ?? "10", 10);

  console.error(
    `[repro] model: ${params.length} params, ${NUM_LAYERS}L/${EMBED_DIM}d/${NUM_HEADS}h, batch=${BATCH_SIZE}, seq=${SEQ_LEN}, ckpt+fullFT`,
  );
  console.error(`[repro] training ${NUM_STEPS} steps`);

  // Mimic F16W apply: overwrite all params with fresh data (the natural
  // pattern, like the agent does on join). This points params to new
  // storages, which is the configuration training sees in real life.
  if (process.env.SKIP_F16W !== "1") {
    await api.beginStep();
    for (let i = 0; i < params.length; i++) {
      const data = new Float32Array(params[i].shape.reduce((a, b) => a * b, 1));
      for (let j = 0; j < data.length; j++) data[j] = (Math.random() - 0.5) * 0.04;
      api.copy_(
        params[i],
        api.tensorFromArray(data, params[i].shape, { device: "webgpu" }),
      );
    }
    api.endStep();
    await api.markStep();
    console.error(`[repro] F16W-style upload done`);
  }

  // Build globalSnapshot like the agent does at the start of each round.
  let globalSnapshot: Float32Array[] = [];
  for (const p of params) globalSnapshot.push(new Float32Array(await p.cpu()));

  for (let step = 0; step < NUM_STEPS; step++) {
    const inputData: number[] = [];
    const targetData: number[] = [];
    for (let i = 0; i < BATCH_SIZE * SEQ_LEN; i++) {
      inputData.push(Math.floor(Math.random() * VOCAB_SIZE));
      targetData.push(Math.floor(Math.random() * VOCAB_SIZE));
    }

    // Phase 1: forward + backward, accumulate grad into accumGrads
    for (const ag of accumGrads) api.zero_(ag);
    await api.beginStep();
    const input = api.tensorFromArray(inputData, [BATCH_SIZE, SEQ_LEN], {
      device: "webgpu",
    });
    const target = api.tensorFromArray(targetData, [BATCH_SIZE, SEQ_LEN], {
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
    optimizer.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();

    // Phase 2: copy accumGrads to params.grad, clip, step
    await api.beginStep();
    for (let i = 0; i < params.length; i++) {
      // biome-ignore lint/suspicious/noExplicitAny: framework
      (params[i] as any)._setGrad(api.mul(accumGrads[i], 1));
    }
    clipGradNorm_(api, params, 1.0);
    optimizer.step();
    optimizer.zeroGrad();
    api.endStep();
    await api.markStep();

    console.error(
      `[repro] step ${step}: loss=${lossVal.toFixed(4)} warnings_total=${warnCount}`,
    );

    // End-of-round: simulate the agent's outer-step + snapshot rebuild + checkpoint.
    if ((step + 1) % ROUND_LEN === 0) {
      await api.flushStep();
      // pseudoGrads via cpu reads on every param.
      const pseudoGrads: Float32Array[] = [];
      for (let i = 0; i < params.length; i++) {
        const local = await params[i].cpu();
        const delta = new Float32Array(local.length);
        for (let j = 0; j < delta.length; j++)
          delta[j] = local[j] - globalSnapshot[i][j];
        pseudoGrads.push(delta);
      }
      // outer-step using the same pseudoGrads as both delta and avg.
      await outerOpt.stepFromCpu(params, globalSnapshot, pseudoGrads);
      // Snapshot for the next round (more cpu reads).
      globalSnapshot = [];
      for (const p of params)
        globalSnapshot.push(new Float32Array(await p.cpu()));
      // Simulate checkpoint write (allocates a large buffer on CPU heap).
      const ckptParts: Buffer[] = [];
      for (let i = 0; i < params.length; i++) {
        const arr = new Float32Array(globalSnapshot[i]);
        ckptParts.push(Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
      }
      const ckptTotal = ckptParts.reduce((s, b) => s + b.length, 0);
      console.error(
        `[repro]   --- end of round at step ${step}, ckpt=${(ckptTotal / 1e6).toFixed(0)}MB, warnings_total=${warnCount} ---`,
      );
    }
  }

  console.error(`[repro] DONE — total warnings: ${warnCount}`);
  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error(`[repro] FATAL: ${e}\n${(e as Error).stack}`);
  process.exit(1);
});
