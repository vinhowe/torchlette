/**
 * Training store -- config, data, training loop, LoRA export, DiLoCo peer sync.
 */

import {
  LoRATrainer,
  type StepPhaseTimings,
  type TrainingConfig,
} from "$lib/torchlette/trainer";
import { serializeLoRAToSafetensors } from "$lib/torchlette/weights";
import { modelStore } from "./model.svelte";

// ============================================================================
// DiLoCo peer state
// ============================================================================

let peerConnected = $state(false);
let peerCount = $state(0);
let peerId = $state("");
// biome-ignore lint/style/useConst: Svelte $state rune
let peerServerId = $state("ws://5.78.181.14:443");
let peerStatus = $state("");

// biome-ignore lint/style/useConst: mutable
let _ws: WebSocket | null = null;
let _peerGrads: Uint8Array | null = null; // raw E3M0 compressed blob from peers

const DILOCO_INNER_STEPS = 20; // steps per round

async function connectToPeer(): Promise<void> {
  if (_ws) return;
  peerStatus = "Connecting...";
  try {
    const ws = new WebSocket(peerServerId);
    ws.binaryType = "arraybuffer";
    _ws = ws;

    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => {
        const myId = "browser-" + Date.now();
        ws.send(
          JSON.stringify({
            type: "register",
            peerId: myId,
            model: "gpt2-124m",
          }),
        );
      };
      ws.onmessage = (e) => {
        if (typeof e.data === "string") {
          const msg = JSON.parse(e.data);
          if (msg.type === "registered") {
            peerId = msg.peerId;
            peerCount = msg.peers - 1;
            peerConnected = true;
            peerStatus = `Connected (${msg.peers} peers)`;
            resolve();
          }
        }
      };
      ws.onerror = () => reject(new Error("WebSocket error"));
      setTimeout(() => reject(new Error("Timeout")), 10000);
    });

    // Set up ongoing message handler
    _ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === "peer-joined") {
            peerCount = msg.peers - 1;
            peerStatus = `${msg.peers} peers`;
          } else if (msg.type === "peer-left") {
            peerCount = msg.peers;
            peerStatus = `${msg.peers} peers`;
          }
        } catch {}
      } else if (e.data instanceof ArrayBuffer) {
        // Binary gradient blob from a neighbor
        _peerGrads = new Uint8Array(e.data);
        peerStatus = `Got ${(e.data.byteLength / 1024 / 1024).toFixed(1)}MB grads`;
      }
    };
    _ws.onclose = () => {
      peerConnected = false;
      peerCount = 0;
      peerStatus = "Disconnected";
      _ws = null;
    };
  } catch (e) {
    peerStatus = `Failed: ${e instanceof Error ? e.message : e}`;
    _ws = null;
  }
}

function disconnectPeer(): void {
  _ws?.close();
  _ws = null;
  peerConnected = false;
  peerCount = 0;
  peerId = "";
  peerStatus = "";
}

// ============================================================================
// E3M0 compression for browser (uses the same format as the Node agent)
// ============================================================================

async function compressPseudoGrads(
  pseudoGrads: Float32Array[],
): Promise<Uint8Array> {
  const { e3m0Quantize } = await import("../../../../../src/distributed/e3m0");
  const parts: Uint8Array[] = [];

  // Header: numParams (u32)
  const header = new Uint8Array(4);
  new DataView(header.buffer).setUint32(0, pseudoGrads.length, true);
  parts.push(header);

  for (const pg of pseudoGrads) {
    const padded = new Float32Array(Math.ceil(pg.length / 8) * 8);
    padded.set(pg);
    const { codes, scales } = e3m0Quantize(padded);
    const paramHeader = new Uint8Array(12);
    const pv = new DataView(paramHeader.buffer);
    pv.setUint32(0, pg.length, true);
    pv.setUint32(4, codes.byteLength, true);
    pv.setUint32(8, scales.byteLength, true);
    parts.push(paramHeader);
    parts.push(
      new Uint8Array(codes.buffer, codes.byteOffset, codes.byteLength),
    );
    parts.push(scales);
  }

  // Concatenate
  const total = parts.reduce((s, p) => s + p.length, 0);
  const result = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    result.set(p, off);
    off += p.length;
  }
  return result;
}

async function decompressPseudoGrads(
  data: Uint8Array,
): Promise<Float32Array[]> {
  const { e3m0Dequantize } = await import(
    "../../../../../src/distributed/e3m0"
  );
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let offset = 0;
  const numParams = view.getUint32(offset, true);
  offset += 4;
  const result: Float32Array[] = [];
  for (let i = 0; i < numParams; i++) {
    const numValues = view.getUint32(offset, true);
    offset += 4;
    const codesLen = view.getUint32(offset, true);
    offset += 4;
    const scalesLen = view.getUint32(offset, true);
    offset += 4;
    const codes = new Uint32Array(
      data.buffer.slice(
        data.byteOffset + offset,
        data.byteOffset + offset + codesLen,
      ),
    );
    offset += codesLen;
    const scales = new Uint8Array(
      data.buffer.slice(
        data.byteOffset + offset,
        data.byteOffset + offset + scalesLen,
      ),
    );
    offset += scalesLen;
    const padLen = Math.ceil(numValues / 8) * 8;
    result.push(e3m0Dequantize(codes, scales, padLen).slice(0, numValues));
  }
  return result;
}

// ============================================================================
// Datasets
// ============================================================================

const TINY_SHAKESPEARE_URL =
  "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

export const DATASETS = [
  {
    id: "shakespeare",
    label: "Shakespeare",
    size: "1.1 MB",
    url: TINY_SHAKESPEARE_URL,
  },
  {
    id: "austen",
    label: "Jane Austen",
    size: "755 KB",
    url: "/datasets/austen.txt",
  },
  {
    id: "lovecraft",
    label: "Lovecraft",
    size: "91 KB",
    url: "/datasets/lovecraft.txt",
  },
  {
    id: "aurelius",
    label: "Marcus Aurelius",
    size: "416 KB",
    url: "/datasets/aurelius.txt",
  },
] as const;

// ============================================================================
// Config + State
// ============================================================================

let rank = $state(8);
let alpha = $state(8);
let maxSteps = $state(1000);
let batchSize = $state(1);
let seqLen = $state(128);
let lr = $state(5e-4);
let useAMP = $state(true);
let useCheckpointing = $state(true);
// biome-ignore lint/style/useConst: Svelte $state runes require reassignment
let fullFinetune = $state(false);

let dataSource = $state("");
let dataText = $state("");
let dataTokenCount = $state(0);
let dataLoading = $state(false);

let running = $state(false);
let step = $state(0);
let loss = $state(0);
let lossHistory = $state<number[]>([]);
// biome-ignore lint/style/useConst: Svelte $state runes require reassignment
let memoryHistory = $state<number[]>([]);
// biome-ignore lint/style/useConst: Svelte $state runes require reassignment
let memoryMB = $state(0);
let tokPerSec = $state(0);
// biome-ignore lint/style/useConst: Svelte $state runes require reassignment
let lastPhases = $state<StepPhaseTimings | null>(null);
let error = $state("");

let loraBlob = $state<ArrayBuffer | null>(null);
let trainer = $state<LoRATrainer | null>(null);

let shouldStop = false;

const canTrain = $derived(
  dataTokenCount >= batchSize * (seqLen + 1) && !running && modelStore.isReady,
);

// ============================================================================
// Data loading
// ============================================================================

async function fetchShakespeare(): Promise<void> {
  if (dataText) return;
  await loadDataset("shakespeare");
}

async function loadDataset(id: string): Promise<void> {
  const ds = DATASETS.find((d) => d.id === id);
  if (!ds) return;
  dataLoading = true;
  try {
    const resp = await fetch(ds.url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const text = await resp.text();
    setData(text, `${ds.label} (${ds.size})`);
  } catch (e) {
    error = `Failed to fetch ${ds.label}: ${e instanceof Error ? e.message : e}`;
  } finally {
    dataLoading = false;
  }
}

function setData(text: string, source: string): void {
  dataText = text;
  dataSource = source;
  if (modelStore.tokenizer) {
    dataTokenCount = modelStore.tokenizer.encode(text).length;
  } else {
    dataTokenCount = Math.floor(text.length / 4);
  }
}

function handleFileDrop(file: File): void {
  const reader = new FileReader();
  reader.onload = () => {
    const text = reader.result as string;
    setData(text, `${file.name} (${(file.size / 1024).toFixed(0)} KB)`);
  };
  reader.readAsText(file);
}

// ============================================================================
// Training (with optional DiLoCo)
// ============================================================================

async function startTraining(): Promise<void> {
  if (
    !canTrain ||
    !modelStore.api ||
    !modelStore.model ||
    !modelStore.tokenizer
  )
    return;

  running = true;
  shouldStop = false;
  step = 0;
  loss = 0;
  lossHistory = [];
  memoryHistory = [];
  memoryMB = 0;
  tokPerSec = 0;
  error = "";
  loraBlob = null;

  try {
    trainer = new LoRATrainer(
      modelStore.api,
      modelStore.model,
      modelStore.tokenizer,
    );

    const config: TrainingConfig = {
      maxSteps: peerConnected ? DILOCO_INNER_STEPS : maxSteps,
      batchSize,
      seqLength: seqLen,
      learningRate: lr,
      useAMP,
      useCheckpointing,
      fullFinetune,
    };

    const api = modelStore.api;
    const model = modelStore.model;

    if (peerConnected && _ws) {
      // ── DiLoCo training: run in rounds ──
      const params = fullFinetune
        ? model.getAllParameters()
        : model.getLoRAParameters();
      const totalRounds = Math.ceil(maxSteps / DILOCO_INNER_STEPS);
      let globalStep = 0;

      for (let round = 0; round < totalRounds && !shouldStop; round++) {
        // Snapshot parameters
        const snapshot: Float32Array[] = [];
        for (const p of params) {
          snapshot.push(new Float32Array(await p.cpu()));
        }

        // Train inner loop
        config.maxSteps = Math.min(DILOCO_INNER_STEPS, maxSteps - globalStep);

        await trainer.train(dataText, config, {
          onStepStart: (s) => {
            step = globalStep + s;
          },
          onStepEnd: (s, l, timeMs, mem, phases) => {
            step = globalStep + s + 1;
            loss = l;
            lossHistory = [...lossHistory, l];
            if (mem !== undefined) {
              memoryMB = mem;
              memoryHistory = [...memoryHistory, mem];
            }
            tokPerSec = (batchSize * seqLen) / (timeMs / 1000);
            if (phases) lastPhases = phases;
          },
          shouldStop: () => shouldStop,
        });
        globalStep += config.maxSteps;

        if (shouldStop) break;

        // Compute pseudo-gradients
        peerStatus = "Computing pseudo-gradients...";
        const pseudoGrads: Float32Array[] = [];
        for (let i = 0; i < params.length; i++) {
          const local = await params[i].cpu();
          const delta = new Float32Array(local.length);
          for (let j = 0; j < delta.length; j++)
            delta[j] = local[j] - snapshot[i][j];
          pseudoGrads.push(delta);
        }

        // Send E3M0 compressed
        peerStatus = "Compressing & sending gradients...";
        const compressed = await compressPseudoGrads(pseudoGrads);
        _ws.send(JSON.stringify({ type: "request-neighbors", round }));
        _ws.send(compressed);
        peerStatus = `Sent ${(compressed.length / 1024 / 1024).toFixed(1)}MB (round ${round})`;

        // Wait for peer grads (up to 5s)
        _peerGrads = null;
        await new Promise((r) => setTimeout(r, 5000));

        // Average if we got peer grads
        if (_peerGrads) {
          peerStatus = "Averaging with peer gradients...";
          const peerPseudoGrads = await decompressPseudoGrads(_peerGrads);
          _peerGrads = null;

          if (peerPseudoGrads.length === params.length) {
            // Average: (ours + theirs) / 2
            for (let i = 0; i < params.length; i++) {
              for (let j = 0; j < pseudoGrads[i].length; j++) {
                pseudoGrads[i][j] =
                  (pseudoGrads[i][j] + peerPseudoGrads[i][j]) / 2;
              }
            }

            // Restore snapshot + apply averaged pseudo-gradient
            await api.beginStep();
            for (let i = 0; i < params.length; i++) {
              api.copy_(
                params[i],
                api.tensorFromArray(snapshot[i], params[i].shape, {
                  device: "webgpu",
                }),
              );
              // Simple outer SGD: params += lr * avg_pseudo_grad
              const update = api.tensorFromArray(
                pseudoGrads[i],
                params[i].shape,
                { device: "webgpu" },
              );
              api.add_(params[i], update);
            }
            api.endStep();
            await api.markStep();
            peerStatus = `Round ${round}: averaged with peer`;
          }
        } else {
          peerStatus = `Round ${round}: solo (no peer grads received)`;
        }
      }
    } else {
      // ── Normal training (no DiLoCo) ──
      await trainer.train(dataText, config, {
        onStepStart: (s) => {
          step = s;
        },
        onStepEnd: (s, l, timeMs, mem, phases) => {
          step = s + 1;
          loss = l;
          lossHistory = [...lossHistory, l];
          if (mem !== undefined) {
            memoryMB = mem;
            memoryHistory = [...memoryHistory, mem];
          }
          tokPerSec = (batchSize * seqLen) / (timeMs / 1000);
          if (phases) lastPhases = phases;
        },
        shouldStop: () => shouldStop,
      });
    }

    // Export weights
    const weights = await trainer.exportLoRAWeights();
    loraBlob = serializeLoRAToSafetensors(weights, {
      rank: String(rank),
      alpha: String(alpha),
      base_model: "openai-community/gpt2",
      target_modules: "c_attn",
    });
  } catch (e) {
    error = e instanceof Error ? e.message : "Training failed";
  } finally {
    running = false;
  }
}

function stopTraining(): void {
  shouldStop = true;
}

function downloadLoRA(): void {
  if (!loraBlob) return;
  const blob = new Blob([loraBlob], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "gpt2-lora.safetensors";
  a.click();
  URL.revokeObjectURL(url);
}

// ============================================================================
// Store export
// ============================================================================

export const trainingStore = {
  // Config
  get rank() {
    return rank;
  },
  set rank(v: number) {
    rank = v;
  },
  get alpha() {
    return alpha;
  },
  set alpha(v: number) {
    alpha = v;
  },
  get maxSteps() {
    return maxSteps;
  },
  set maxSteps(v: number) {
    maxSteps = v;
  },
  get batchSize() {
    return batchSize;
  },
  set batchSize(v: number) {
    batchSize = v;
  },
  get seqLen() {
    return seqLen;
  },
  set seqLen(v: number) {
    seqLen = v;
  },
  get lr() {
    return lr;
  },
  set lr(v: number) {
    lr = v;
  },
  get useAMP() {
    return useAMP;
  },
  set useAMP(v: boolean) {
    useAMP = v;
  },
  get useCheckpointing() {
    return useCheckpointing;
  },
  set useCheckpointing(v: boolean) {
    useCheckpointing = v;
  },
  get fullFinetune() {
    return fullFinetune;
  },
  set fullFinetune(v: boolean) {
    fullFinetune = v;
  },

  // Data
  get dataSource() {
    return dataSource;
  },
  get dataText() {
    return dataText;
  },
  get dataTokenCount() {
    return dataTokenCount;
  },
  get dataLoading() {
    return dataLoading;
  },

  // Progress
  get running() {
    return running;
  },
  get step() {
    return step;
  },
  get loss() {
    return loss;
  },
  get lossHistory() {
    return lossHistory;
  },
  get memoryHistory() {
    return memoryHistory;
  },
  get memoryMB() {
    return memoryMB;
  },
  get tokPerSec() {
    return tokPerSec;
  },
  get phases() {
    return lastPhases;
  },
  get error() {
    return error;
  },
  get canTrain() {
    return canTrain;
  },

  // Export
  get loraBlob() {
    return loraBlob;
  },

  // DiLoCo peer
  get peerConnected() {
    return peerConnected;
  },
  get peerCount() {
    return peerCount;
  },
  get peerId() {
    return peerId;
  },
  get peerServerId() {
    return peerServerId;
  },
  set peerServerId(v: string) {
    peerServerId = v;
  },
  get peerStatus() {
    return peerStatus;
  },

  // Actions
  fetchShakespeare,
  loadDataset,
  setData,
  handleFileDrop,
  startTraining,
  stopTraining,
  downloadLoRA,
  connectToPeer,
  disconnectPeer,
};
