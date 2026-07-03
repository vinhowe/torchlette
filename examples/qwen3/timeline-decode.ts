/**
 * Stage-1 timeline instrumentation for a steady-state Qwen3 decode step.
 *
 * Produces, for ONE post-warmup decode step:
 *   - phase timeline (tensorFromArray / graph build / force (plan replay +
 *     encode + submits) / logits readback / argmax / markStep sub-phases)
 *   - per-submit records: wall time of the submit call, CPU time inside it,
 *     number of CBs, the call stack (WHY the submit boundary exists), and the
 *     GPU completion time (queue.onSubmittedWorkDone) for idle-gap analysis
 *   - createBindGroup / createBuffer counts + CPU time per phase
 *
 * Run (repo root, GPU otherwise quiet):
 *   npx tsx examples/qwen3/timeline-decode.ts [numSteps=12] [dtype=f32]
 */

import * as path from "node:path";
import { fileURLToPath } from "node:url";
import {
  getWebGPUDevice,
  getWebGPUInitError,
  initWebGPU,
} from "../../src/backend/webgpu";
import {
  awaitDeferredFence,
  issueDeferredFence,
} from "../../src/backend/webgpu/buffer-pool";
import { Torchlette } from "../../src/frontend/torchlette";
import { storageTracker } from "../../src/graph/storage-tracker";
import { loadPretrainedQwen3 } from "./loader";
import type { StaticKV } from "./model";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MODEL_DIR = path.join(__dirname, "../../ckpts/qwen3-1.7b");
const PROMPT = [785, 6722, 315, 9625, 374]; // "The capital of France is"

// ---------------------------------------------------------------------------
// Instrumentation state
// ---------------------------------------------------------------------------

type SubmitRec = {
  idx: number;
  tCall: number; // ms since step start
  cpuMs: number; // CPU time inside queue.submit
  numCBs: number;
  stack: string[];
  tDone?: number; // ms since step start when onSubmittedWorkDone resolved
};

let stepT0 = 0;
let recording = false;
let submits: SubmitRec[] = [];
let bindGroupCount = 0;
let bindGroupMs = 0;
let bufferCount = 0;
let bufferMs = 0;
let pendingDone: Promise<void>[] = [];

function now(): number {
  return performance.now();
}

function cleanStack(stack: string): string[] {
  return (stack ?? "")
    .split("\n")
    .slice(1)
    .map((l) => l.trim())
    .filter(
      (l) =>
        l.includes("/src/") &&
        !l.includes("timeline-decode") &&
        !l.includes("node_modules"),
    )
    .slice(0, 6)
    .map((l) =>
      l
        .replace(/^at /, "")
        .replace(/\(.*\/src\//, "(src/")
        .replace(/file:.*\/src\//, "src/"),
    );
}

function installHooks() {
  const dev = getWebGPUDevice();
  if (!dev) throw new Error("no device");
  const queue = dev.queue as unknown as {
    submit: (cbs: unknown[]) => void;
    onSubmittedWorkDone?: () => Promise<void>;
  };
  const device = dev.device as unknown as {
    createBindGroup: (d: unknown) => unknown;
    createBuffer: (d: unknown) => unknown;
  };

  const origSubmit = queue.submit.bind(queue);
  queue.submit = (cbs: unknown[]) => {
    if (!recording) return origSubmit(cbs);
    const t0 = now();
    origSubmit(cbs);
    const t1 = now();
    const rec: SubmitRec = {
      idx: submits.length,
      tCall: t0 - stepT0,
      cpuMs: t1 - t0,
      numCBs: Array.isArray(cbs) ? cbs.length : 1,
      stack: cleanStack(new Error().stack ?? ""),
    };
    submits.push(rec);
    if (typeof queue.onSubmittedWorkDone === "function") {
      // biome-ignore lint/suspicious/noExplicitAny: harness
      (globalThis as any).__markHarnessDone?.(true);
      const p = queue.onSubmittedWorkDone().then(() => {
        rec.tDone = now() - stepT0;
      });
      // biome-ignore lint/suspicious/noExplicitAny: harness
      (globalThis as any).__markHarnessDone?.(false);
      pendingDone.push(p);
    }
  };

  const origCBG = device.createBindGroup.bind(device);
  device.createBindGroup = (d: unknown) => {
    if (!recording) return origCBG(d);
    const t0 = now();
    const r = origCBG(d);
    bindGroupMs += now() - t0;
    bindGroupCount++;
    return r;
  };
  const origCB = device.createBuffer.bind(device);
  device.createBuffer = (d: unknown) => {
    if (!recording) return origCB(d);
    const t0 = now();
    // biome-ignore lint/suspicious/noExplicitAny: harness
    const r = origCB(d) as any;
    bufferMs += now() - t0;
    bufferCount++;
    // Wrap mapAsync to time readback waits
    if (typeof r?.mapAsync === "function") {
      const origMap = r.mapAsync.bind(r);
      r.mapAsync = (mode: number) => {
        const m0 = now();
        const p = origMap(mode);
        p.then(() => {
          if (recording)
            console.log(
              `  [mapAsync] size=${r.size} wait=${(now() - m0).toFixed(2)}ms @${(m0 - stepT0).toFixed(2)}ms`,
            );
        });
        return p;
      };
    }
    return r;
  };

  // Time onSubmittedWorkDone waits from src (read()'s fence)
  if (typeof queue.onSubmittedWorkDone === "function") {
    const origDone = queue.onSubmittedWorkDone.bind(queue);
    let fromHarness = false;
    // biome-ignore lint/suspicious/noExplicitAny: harness
    (globalThis as any).__markHarnessDone = (v: boolean) => {
      fromHarness = v;
    };
    queue.onSubmittedWorkDone = () => {
      const isHarness = fromHarness;
      const d0 = now();
      const p = origDone();
      if (recording && !isHarness) {
        p.then(() => {
          console.log(
            `  [workDone] wait=${(now() - d0).toFixed(2)}ms @${(d0 - stepT0).toFixed(2)}ms`,
          );
        });
      }
      return p;
    };
  }
}

// ---------------------------------------------------------------------------

async function main() {
  const numSteps = Number(process.argv[2] ?? 12);
  const dtype = (process.argv[3] === "f16" ? "f16" : "f32") as "f32" | "f16";

  const ok = await initWebGPU();
  if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");
  const api = new Torchlette("webgpu", { enableFusion: true });
  const model = await loadPretrainedQwen3(api, MODEL_DIR, {
    maxSeqLen: 512,
    weightDtype: dtype,
  });
  const vocab = model.config.vocabSize;
  installHooks();

  // biome-ignore lint/suspicious/noExplicitAny: instrumentation harness
  const rt = (api as any).runtime;

  const argmaxOfFlat = (flat: number[] | Float32Array, off: number) => {
    let best = 0;
    for (let v = 1; v < vocab; v++)
      if (flat[off + v] > flat[off + best]) best = v;
    return best;
  };

  // --- Prefill (uninstrumented)
  const tokens = [...PROMPT];
  const staticKV: StaticKV = model.allocStaticKV(512);
  {
    const idx = api.tensorFromArray(tokens, [1, tokens.length]);
    const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
    const flat = await api.cpu(logits);
    tokens.push(argmaxOfFlat(flat, (tokens.length - 1) * vocab));
    logits.dispose();
    await api.markStep();
  }

  const MEASURE = new Set([numSteps - 3, numSteps - 2]); // two steady-state steps
  for (let i = 0; i < numSteps; i++) {
    const measured = MEASURE.has(i);
    const marks: [string, number][] = [];
    const mark = (name: string) => marks.push([name, now() - stepT0]);

    stepT0 = now();
    submits = [];
    pendingDone = [];
    bindGroupCount = 0;
    bindGroupMs = 0;
    bufferCount = 0;
    bufferMs = 0;
    recording = measured;

    await api.beginStep();
    mark("beginStep");
    const last = tokens[tokens.length - 1];
    const idx = api.tensorFromArray([last], [1, 1]);
    mark("tensorFromArray(idx)");
    const { logits } = api.noGrad(() => model.forward(idx, { staticKV }));
    mark("forward graph build");

    // Split readTopK into force (plan replay + encode + submit) and readback.
    rt.forceRead(logits.baseId);
    // biome-ignore lint/suspicious/noExplicitAny: harness
    const inner = (logits as any)._unwrap();
    await rt.force(inner);
    mark("force (replay+encode+submit)");
    const backend = rt.getBackend("webgpu");
    const top = await backend.ops.readTopK(inner.backendTensor, 64, {
      length: vocab,
    });
    mark("readTopK (2 dispatches+flush+mapAsync)");
    tokens.push(top.indices[0]);
    mark("argmax (indices[0])");
    logits.dispose();

    // markStep, decomposed (mirrors Torchlette.markStep):
    api.endStep();
    mark("endStep");
    // biome-ignore lint/suspicious/noExplicitAny: harness
    (api as any)._pendingStepBoundary = null;
    await awaitDeferredFence();
    mark("markStep: await prior fence");
    await rt.markStep();
    mark("markStep: runtime.markStep");
    await rt.forceAllPending();
    mark("markStep: forceAllPending");
    const swStats = storageTracker.stats();
    const sw0 = now();
    const d1 = storageTracker.destroyUnreachable();
    const sw1 = now();
    const rel = storageTracker.releaseStepTemps();
    const sw2 = now();
    const d2 = storageTracker.destroyUnreachable();
    const sw3 = now();
    mark("markStep: storage sweeps");
    if (measured) {
      console.log(
        `  [sweep] storages=${swStats.totalStorages} (reach=${swStats.reachableStorages}) destroy1=${d1} (${(sw1 - sw0).toFixed(2)}ms) release=${rel} (${(sw2 - sw1).toFixed(2)}ms) destroy2=${d2} (${(sw3 - sw2).toFixed(2)}ms)`,
      );
    }
    rt.resetCumulativeFusionStats();
    issueDeferredFence();
    await awaitDeferredFence();
    mark("markStep: fence");

    recording = false;
    const wall = now() - stepT0;
    console.log(`decode step ${i}: ${wall.toFixed(1)}ms`);

    if (measured) {
      await Promise.all(pendingDone);
      console.log(
        `\n===== TIMELINE step ${i} (wall ${wall.toFixed(2)}ms) =====`,
      );
      let prev = 0;
      for (const [name, t] of marks) {
        console.log(
          `  ${(t - prev).toFixed(2).padStart(8)}ms  [${t.toFixed(2).padStart(8)}]  ${name}`,
        );
        prev = t;
      }
      console.log(
        `  createBindGroup: ${bindGroupCount}x ${bindGroupMs.toFixed(2)}ms | createBuffer: ${bufferCount}x ${bufferMs.toFixed(2)}ms`,
      );
      console.log(`  --- submits (${submits.length}) ---`);
      for (const s of submits) {
        console.log(
          `  #${s.idx} @${s.tCall.toFixed(2)}ms cpu=${s.cpuMs.toFixed(2)}ms cbs=${s.numCBs} gpuDone=${s.tDone?.toFixed(2) ?? "?"}ms`,
        );
        for (const l of s.stack) console.log(`        ${l}`);
      }
      console.log("");
    }
  }
  console.log(`tokens: ${tokens.join(",")}`);
  process.exit(0);
}

main().catch((e) => {
  console.error("FAILED:", e);
  process.exit(1);
});
