/**
 * Browser entry point for the remote-training demo.
 *
 * Connects to the WS server, builds a tiny transformer, drives training.
 * Every plan goes over the wire — the browser only builds the autograd graph.
 */

import { initWebGPU } from "../../../src/backend/webgpu/index.ts";
import {
  createRemoteEngine,
  type RemoteEngine,
} from "../../../src/remote/client-engine.ts";
import { buildCharDataset, createModel, forward, parameters } from "./model.ts";
import { modelConfigSmall, type TrainConfig, trainStep } from "./train.ts";
import { RpcClient } from "./transport.ts";

// ============================================================================
// UI state
// ============================================================================

const $ = <T extends HTMLElement = HTMLElement>(sel: string): T => {
  const el = document.querySelector<T>(sel);
  if (!el) throw new Error(`not found: ${sel}`);
  return el;
};

const logEl = $<HTMLDivElement>("#log");
const statusEl = $("#status");
const btnTrain = $<HTMLButtonElement>("#btn-train");
const btnStop = $<HTMLButtonElement>("#btn-stop");
const btnSample = $<HTMLButtonElement>("#btn-sample");
const statStep = $("#stat-step");
const statLoss = $("#stat-loss");
const statTps = $("#stat-tps");
const statPlans = $("#stat-plans");
const sampleEl = $<HTMLDivElement>("#sample");
const canvas = $<HTMLCanvasElement>("#chart");

function log(msg: string, cls = ""): void {
  const line = document.createElement("div");
  line.className = `log-line ${cls}`;
  line.textContent = msg;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(connected: boolean, text: string): void {
  statusEl.className = `status ${connected ? "connected" : "disconnected"}`;
  statusEl.textContent = text;
}

// ============================================================================
// Loss chart
// ============================================================================

const losses: number[] = [];

function drawChart(): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  if (canvas.width !== cssW * dpr) {
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.scale(dpr, dpr);
  }
  ctx.clearRect(0, 0, cssW, cssH);

  if (losses.length < 2) {
    ctx.fillStyle = "#666";
    ctx.font = "12px ui-monospace, monospace";
    ctx.fillText("(no data yet)", 12, 20);
    return;
  }

  const padL = 48;
  const padT = 12;
  const padR = 12;
  const padB = 24;
  const w = cssW - padL - padR;
  const h = cssH - padT - padB;
  const lo = Math.min(...losses);
  const hi = Math.max(...losses);
  const yRange = Math.max(hi - lo, 1e-6);

  // Axes
  ctx.strokeStyle = "#23272e";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + h);
  ctx.lineTo(padL + w, padT + h);
  ctx.stroke();

  // Gridlines + labels
  ctx.fillStyle = "#666";
  ctx.font = "10px ui-monospace, monospace";
  for (let i = 0; i <= 4; i++) {
    const y = padT + (h * i) / 4;
    const val = hi - (yRange * i) / 4;
    ctx.strokeStyle = "#1a1d21";
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + w, y);
    ctx.stroke();
    ctx.fillText(val.toFixed(3), 4, y + 3);
  }

  // Line
  ctx.strokeStyle = "#3a8fff";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < losses.length; i++) {
    const x = padL + (w * i) / Math.max(losses.length - 1, 1);
    const y = padT + h - ((losses[i] - lo) / yRange) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // X-axis label
  ctx.fillStyle = "#666";
  ctx.fillText(`step ${losses.length}`, padL + w - 60, cssH - 8);
}

// ============================================================================
// Training state
// ============================================================================

interface TrainingSession {
  remote: RemoteEngine;
  // biome-ignore lint/suspicious/noExplicitAny: convenience
  model: any;
  // biome-ignore lint/suspicious/noExplicitAny: convenience
  ds: any;
  cfg: TrainConfig;
  // biome-ignore lint/suspicious/noExplicitAny: convenience
  params: any[];
  rng: () => number;
}

let session: TrainingSession | null = null;
let stopRequested = false;
let training = false;

// ============================================================================
// Training text — a short public-domain sample
// ============================================================================

const TRAIN_TEXT = `the quick brown fox jumps over the lazy dog. how vexingly quick daft zebras jump! pack my box with five dozen liquor jugs. the five boxing wizards jump quickly. sphinx of black quartz, judge my vow. glib jocks quiz nymph to vex dwarf. `
  .repeat(6)
  .trim();

// ============================================================================
// Setup + event handlers
// ============================================================================

async function connect(): Promise<RemoteEngine> {
  // Init WebGPU client-side: createRemoteEngine builds a webgpu-flavored
  // client Torchlette so the lazy-graph dtype/op decisions match what the
  // server's webgpu executor will run. No actual GPU dispatch happens
  // here — only the registry needs the webgpu backend entry.
  const ok = await initWebGPU();
  if (!ok) throw new Error("WebGPU init failed for remote client");

  const wsUrl = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
  log(`connecting to ${wsUrl}...`);
  const rpc = new RpcClient({ url: wsUrl, onLog: (m) => log(m) });
  rpc.onClose(() => setStatus(false, "disconnected"));
  await rpc.connect();
  setStatus(true, `session ${rpc.sessionId.slice(0, 8)}`);
  return createRemoteEngine(rpc);
}

async function initSession(): Promise<void> {
  const remote = await connect();

  const ds = buildCharDataset(TRAIN_TEXT);
  log(`dataset: ${ds.text.length} chars, vocab=${ds.vocabSize}`);

  const cfg = modelConfigSmall(ds.vocabSize);
  const model = createModel(remote.torch, cfg, 42);
  const params = parameters(model);
  log(
    `model: ${cfg.numLayers}L D=${cfg.embedDim} H=${cfg.numHeads} ` +
      `T=${cfg.blockSize} — ${params.length} param tensors`,
  );

  const trainCfg: TrainConfig = {
    lr: 0.03,
    batchSize: 4,
    seqLen: cfg.blockSize,
    seed: 1,
  };

  let s = trainCfg.seed >>> 0 || 1;
  const rng = (): number => {
    s = (Math.imul(s, 1103515245) + 12345) >>> 0;
    return ((s >>> 0) / 0x100000000) * 2 - 1;
  };

  session = { remote, model, ds, cfg: trainCfg, params, rng };
  btnSample.disabled = false;
}

async function doTraining(): Promise<void> {
  if (!session) await initSession();
  if (!session) return;
  if (training) return;

  training = true;
  stopRequested = false;
  btnTrain.disabled = true;
  btnStop.disabled = false;

  const t0 = performance.now();
  let tokensSeen = 0;
  const { remote, model, ds, cfg, params, rng } = session;

  try {
    while (!stopRequested) {
      const stepStart = performance.now();
      const lossVal = await trainStep(
        remote.torch,
        model,
        ds,
        params,
        cfg,
        rng,
      );
      await remote.markStep(params);
      const stepMs = performance.now() - stepStart;
      tokensSeen += cfg.batchSize * cfg.seqLen;

      losses.push(lossVal);
      const elapsed = performance.now() - t0;
      const tps = (tokensSeen / elapsed) * 1000;
      const plansPerStep = remote.stats.executes / (losses.length || 1);

      statStep.textContent = String(losses.length);
      statLoss.textContent = lossVal.toFixed(4);
      statTps.textContent = tps.toFixed(1);
      statPlans.textContent = plansPerStep.toFixed(1);
      drawChart();
      void stepMs;

      // Yield to the event loop so the UI paints.
      await new Promise((r) => setTimeout(r, 0));
    }
  } catch (e) {
    log(`training error: ${(e as Error).message}`, "err");
  } finally {
    training = false;
    btnTrain.disabled = false;
    btnStop.disabled = true;
  }
}

async function doSample(): Promise<void> {
  if (!session) {
    log("need to train (or at least init) first", "err");
    return;
  }
  const { remote, model, ds, cfg } = session;
  const prompt = "the ";
  const promptTokens = ds.encode(prompt);
  const maxNew = 80;

  const ids: number[] = [...promptTokens];
  sampleEl.textContent = prompt;

  for (let i = 0; i < maxNew; i++) {
    // Take last cfg.seqLen tokens as context, pad with 0 if needed.
    const ctx: number[] = new Array(cfg.seqLen).fill(0);
    const start = Math.max(0, ids.length - cfg.seqLen);
    for (let t = 0; t < cfg.seqLen; t++) {
      ctx[t] = t < ids.length - start ? ids[start + t] : 0;
    }
    // Take the position of the last real token.
    const lastPos = Math.min(ids.length - 1, cfg.seqLen - 1);

    const inputTensor = remote.torch.tensorFromArray(
      ctx,
      [1, cfg.seqLen],
      { device: "cpu", dtype: "i32" },
    );
    const logits = remote.torch.noGrad(() => forward(remote.torch, model, inputTensor));
    // logits: [1, T, V]. Take [0, lastPos, :] via narrow.
    const lastLogits = remote.torch.narrow(logits, 1, lastPos, 1);
    const flat = remote.torch.reshape(lastLogits, [ds.vocabSize]);
    const values = await flat.cpu();

    // Greedy argmax.
    let bestIdx = 0;
    let best = values[0];
    for (let v = 1; v < values.length; v++) {
      if (values[v] > best) {
        best = values[v];
        bestIdx = v;
      }
    }
    ids.push(bestIdx);
    sampleEl.textContent = ds.decode(ids);

    // Release intermediate handles after each token.
    await remote.markStep(session.params);
  }
  log(`sampled ${maxNew} tokens`);
}

btnTrain.addEventListener("click", () => {
  void doTraining();
});
btnStop.addEventListener("click", () => {
  stopRequested = true;
});
btnSample.addEventListener("click", () => {
  void doSample();
});

window.addEventListener("resize", () => drawChart());
drawChart();
log("ready — click Start training");
