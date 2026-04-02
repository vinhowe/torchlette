/**
 * Diagnose per-step GPU buffer allocation leaks.
 *
 * Runs LoRA training with alloc debug enabled and reports which allocation
 * call sites produce buffers that are never deallocated.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import {
  type GPT2Config,
  GPT2WithLoRA,
} from "../examples/gpt2-lora-trainer/src/lib/torchlette/gpt2-lora";
import { GPT2Tokenizer } from "../examples/gpt2-lora-trainer/src/lib/torchlette/tokenizer";
import { nn, storageTracker } from "../src";
import {
  destroyWebGPU,
  enableAllAllocDebug,
  getAndResetFlowCounters,
  getGPUMemoryStats,
  getLeakedAllocCount,
  getLeakedAllocCountForStep,
  initWebGPU,
  setAllocStep,
  snapshotLeakedAllocs,
} from "../src/backend/webgpu";
import { getTrackedBuffers } from "../src/backend/webgpu/memory-tracker";
import { Torchlette } from "../src/frontend/torchlette";
import { Adam, GradScaler } from "../src/optim";

const DISTILGPT2_CONFIG: GPT2Config = {
  vocabSize: 50257,
  blockSize: 1024,
  numLayers: 6,
  numHeads: 12,
  embedDim: 768,
  dropoutRate: 0.0,
};

async function loadModel(api: Torchlette) {
  const tokenizerDir = path.join(process.cwd(), "models", "distilgpt2");
  const vocabJson = JSON.parse(
    fs.readFileSync(path.join(tokenizerDir, "vocab.json"), "utf-8"),
  );
  const mergesText = fs.readFileSync(
    path.join(tokenizerDir, "merges.txt"),
    "utf-8",
  );
  const tokenizer = new GPT2Tokenizer();
  tokenizer.load(
    vocabJson,
    mergesText.split("\n").filter((l) => l && !l.startsWith("#")),
  );

  const loraConfig = { rank: 8, alpha: 8 };
  const model = new GPT2WithLoRA(api, DISTILGPT2_CONFIG, loraConfig, "webgpu");

  // Load weights
  const modelPath = path.join(tokenizerDir, "model.safetensors");
  const buf = fs.readFileSync(modelPath);
  const headerLen = Number(
    new DataView(buf.buffer, buf.byteOffset, 8).getBigUint64(0, true),
  );
  const headerJson = new TextDecoder().decode(buf.subarray(8, 8 + headerLen));
  const header = JSON.parse(headerJson);
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
        const sign = (h >> 15) & 1;
        const exp = (h >> 10) & 0x1f;
        const mant = h & 0x3ff;
        if (exp === 0) f32[i] = (sign ? -1 : 1) * (mant / 1024) * 2 ** -14;
        else if (exp === 31)
          f32[i] = mant === 0 ? (sign ? -Infinity : Infinity) : NaN;
        else f32[i] = (sign ? -1 : 1) * (1 + mant / 1024) * 2 ** (exp - 15);
      }
    } else continue;
    weights.set(name.replace(/^transformer\./, ""), { data: f32, shape });
  }
  model.loadBaseWeights(weights);
  return { model, tokenizer };
}

async function main() {
  console.log("=== GPU Allocation Leak Diagnostic ===\n");

  const ok = await initWebGPU();
  if (!ok) {
    console.error("WebGPU not available");
    process.exit(1);
  }

  const useAMP = process.argv.includes("--amp");
  const useCheckpointing = process.argv.includes("--ckpt");
  console.log(`Config: AMP=${useAMP}, Checkpointing=${useCheckpointing}`);

  const api = new Torchlette("webgpu", { enableFusion: true });
  const { model, tokenizer } = await loadModel(api);

  // Enable alloc debug tracking
  enableAllAllocDebug();

  const trainingText =
    "The quick brown fox jumps over the lazy dog. " +
    "In a galaxy far far away, there lived a brave knight. " +
    "The rain in Spain falls mainly on the plain. " +
    "To be or not to be, that is the question. " +
    "All that glitters is not gold, but it sure looks shiny. " +
    "Once upon a time in a land of enchantment, the rivers flowed with silver.";
  const tokens = tokenizer.encode(trainingText);

  const loraParams = model.getLoRAParameters();
  const optimizer = new Adam(loraParams, { lr: 1e-4 }, api);
  let gradScaler: GradScaler | null = null;
  if (useAMP) {
    gradScaler = new GradScaler(api, {
      initScale: 1.0,
      growthFactor: 2.0,
      backoffFactor: 0.5,
      growthInterval: 1000,
    });
  }

  if (useCheckpointing) model.enableCheckpointing(true);
  model.train(true);

  const seqLen = 32;
  const WARMUP_STEPS = 3;
  const MEASURE_STEPS = 5;

  // Warmup (arena allocation, JIT compilation, etc.)
  console.log(`\nWarmup: ${WARMUP_STEPS} steps...`);
  for (let step = 0; step < WARMUP_STEPS; step++) {
    await api.beginStep();
    const start = Math.floor(Math.random() * (tokens.length - seqLen - 1));
    const input = api.tensorFromArray(
      tokens.slice(start, start + seqLen),
      [1, seqLen],
      { device: "webgpu" },
    );
    const target = api.tensorFromArray(
      tokens.slice(start + 1, start + seqLen + 1),
      [1, seqLen],
      { device: "webgpu" },
    );

    const loss = api.tidy(() => {
      if (useAMP) {
        let l = api.autocast(() => model.forwardWithLoss(input, target).loss, {
          deviceType: "webgpu",
        });
        if (gradScaler) l = gradScaler.scale(l);
        api.keep(l);
        return l;
      }
      const r = model.forwardWithLoss(input, target);
      api.keep(r.loss);
      return r.loss;
    });

    await loss.backward();
    await nn.clipGradNorm_(api, loraParams, 1.0);

    if (gradScaler) {
      await gradScaler.unscale_(optimizer);
      await gradScaler.step(optimizer);
      gradScaler.update();
    } else {
      optimizer.step();
    }
    optimizer.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
  }

  // Reset counters after warmup
  getAndResetFlowCounters();

  // Measurement phase
  console.log(`\nMeasurement: ${MEASURE_STEPS} steps...\n`);
  const memBefore = getGPUMemoryStats().currentBytes;
  const leakedBefore = getLeakedAllocCount();
  let prevReachableIds = storageTracker.getReachableIds();
  // biome-ignore lint/style/useLet: mutated later
  let step0NewReachableIds: number[] | null = null;

  for (let step = 0; step < MEASURE_STEPS; step++) {
    const stepId = 100 + step; // Use distinct step IDs
    setAllocStep(stepId);
    if (step === 1) api._debugPermanentLog = true;

    await api.beginStep();
    const start = Math.floor(Math.random() * (tokens.length - seqLen - 1));
    const input = api.tensorFromArray(
      tokens.slice(start, start + seqLen),
      [1, seqLen],
      { device: "webgpu" },
    );
    const target = api.tensorFromArray(
      tokens.slice(start + 1, start + seqLen + 1),
      [1, seqLen],
      { device: "webgpu" },
    );

    const loss = api.tidy(() => {
      if (useAMP) {
        let l = api.autocast(() => model.forwardWithLoss(input, target).loss, {
          deviceType: "webgpu",
        });
        if (gradScaler) l = gradScaler.scale(l);
        api.keep(l);
        return l;
      }
      const r = model.forwardWithLoss(input, target);
      api.keep(r.loss);
      return r.loss;
    });

    const lossVal = await loss.item();
    await loss.backward();
    await nn.clipGradNorm_(api, loraParams, 1.0);

    if (gradScaler) {
      await gradScaler.unscale_(optimizer);
      await gradScaler.step(optimizer);
      gradScaler.update();
    } else {
      optimizer.step();
    }
    optimizer.zeroGrad();
    input.dispose();
    target.dispose();
    api.endStep();
    await api.markStep();
    if (step === 1) api._debugPermanentLog = false;

    const mem = getGPUMemoryStats();
    const leakedThisStep = getLeakedAllocCountForStep(stepId);
    const flow = getAndResetFlowCounters();
    const st = storageTracker.stats();
    const newReachable = storageTracker.getNewReachableSince(prevReachableIds);
    prevReachableIds = storageTracker.getReachableIds();
    console.log(
      `Step ${step}: loss=${lossVal.toFixed(4)} mem=${(mem.currentBytes / 1e6).toFixed(1)}MB ` +
        `net_allocs=${flow.allocs - flow.deallocs} ` +
        `storages=${st.totalStorages}(reachable=${st.reachableStorages}) ` +
        `new_reachable=${newReachable.length}`,
    );
    // Capture step 0 new reachable IDs for persistence check
    if (step === 0) {
      step0NewReachableIds = newReachable.map((e) => e.id);
    }
    // Show new reachable storages on first step
    if (step === 0 && newReachable.length > 0) {
      console.log("  New reachable storages:");
      const byShape = new Map<string, number>();
      for (const entry of newReachable) {
        const key = entry.debugInfo
          ? `${entry.debugInfo.type}:${entry.debugInfo.shape?.join("x") ?? "?"}:${entry.debugInfo.dtype ?? "?"}`
          : "unknown";
        byShape.set(key, (byShape.get(key) ?? 0) + 1);
      }
      for (const [key, count] of [...byShape.entries()].sort(
        (a, b) => b[1] - a[1],
      )) {
        console.log(`    ${count}x ${key}`);
      }
    }
    // On step 2: check which storages from step 0 are STILL reachable (=persistent leak)
    if (step === 2 && step0NewReachableIds) {
      const currentReachable = storageTracker.getReachableIds();
      let persisted = 0;
      let liveRef = 0;
      let deadRef = 0;
      const byShape = new Map<
        string,
        { count: number; live: number; dead: number }
      >();
      for (const id of step0NewReachableIds) {
        if (currentReachable.has(id)) {
          persisted++;
          const hasLive = storageTracker.hasLiveTensorRef(id);
          if (hasLive) liveRef++;
          else deadRef++;
          const info = storageTracker.getTensorRefDebugInfo(id);
          const key = info
            ? `${info.shape?.join("x") ?? "scalar"}:${info.dtype ?? "?"}${info.disposed ? "(DISPOSED)" : ""}`
            : "no-ref";
          const e = byShape.get(key) ?? { count: 0, live: 0, dead: 0 };
          e.count++;
          if (hasLive) e.live++;
          else e.dead++;
          byShape.set(key, e);
        }
      }
      console.log(
        `  Step 0 storages persisting to step 2: ${persisted}/${step0NewReachableIds.length} (live=${liveRef} dead=${deadRef})`,
      );
      for (const [key, e] of [...byShape.entries()].sort(
        (a, b) => b[1].count - a[1].count,
      )) {
        console.log(`    ${e.count}x ${key} (live=${e.live} dead=${e.dead})`);
      }
    }
  }

  const memAfter = getGPUMemoryStats().currentBytes;
  const leakedAfter = getLeakedAllocCount();
  const memDelta = memAfter - memBefore;
  const leakedDelta = leakedAfter - leakedBefore;

  console.log(`\n=== Summary ===`);
  console.log(
    `Memory: ${(memBefore / 1e6).toFixed(1)}MB → ${(memAfter / 1e6).toFixed(1)}MB (${memDelta > 0 ? "+" : ""}${(memDelta / 1e6).toFixed(1)}MB)`,
  );
  console.log(
    `Leaked allocs: ${leakedBefore} → ${leakedAfter} (${leakedDelta > 0 ? "+" : ""}${leakedDelta})`,
  );

  // Snapshot leaked allocs from measurement steps
  console.log(`\n=== Leaked Allocations by Call Site ===`);
  for (let step = 0; step < MEASURE_STEPS; step++) {
    const stepId = 100 + step;
    const leaked = snapshotLeakedAllocs(stepId);
    if (leaked.size > 0) {
      console.log(`\n--- Step ${step} (id=${stepId}) ---`);
      const sorted = [...leaked.entries()].sort(
        (a, b) => b[1].totalBytes - a[1].totalBytes,
      );
      for (const [site, info] of sorted) {
        console.log(
          `  ${info.count} buffers, ${(info.totalBytes / 1024).toFixed(1)}KB total, avg ${(info.totalBytes / info.count / 1024).toFixed(1)}KB`,
        );
        console.log(
          `  Site: ${site
            .split("\n")
            .map((l) => l.trim())
            .join(" → ")}`,
        );
        console.log(
          `  Full stack:\n${info.exampleStack
            .split("\n")
            .slice(0, 10)
            .map((l) => "    " + l.trim())
            .join("\n")}`,
        );
      }
    }
  }

  // Also snapshot ALL leaked allocs regardless of step
  console.log(`\n=== ALL Leaked Allocations (across all steps) ===`);
  const allLeaked = snapshotLeakedAllocs();
  const allSorted = [...allLeaked.entries()].sort(
    (a, b) => b[1].totalBytes - a[1].totalBytes,
  );
  for (const [site, info] of allSorted.slice(0, 10)) {
    console.log(
      `  ${info.count} buffers, ${(info.totalBytes / 1024).toFixed(1)}KB total, avg ${(info.totalBytes / info.count / 1024).toFixed(1)}KB`,
    );
    console.log(
      `  Site: ${site
        .split("\n")
        .map((l) => l.trim())
        .join(" → ")}`,
    );
  }

  await destroyWebGPU();
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
