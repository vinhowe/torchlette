<script lang="ts">
import { onMount } from "svelte";
import { modelStore } from "$lib/stores/model.svelte";

// ── State ──
let status = $state("Initializing...");
let connected = $state(false);
let peerCount = $state(0);
let myId = $state("");
let round = $state(0);
let innerStep = $state(0);
let loss = $state(0);
let lossHistory = $state<number[]>([]);
let tokensProcessed = $state(0);
let tokPerSec = $state(0);
let currentChunk = $state("");
let gradStatus = $state("");
let contributors = $state(0);
let running = $state(false);
let generating = $state(false);
let genOutput = $state("");
// biome-ignore lint/style/useConst: Svelte $state
let genPrompt = $state("The");

const serverUrl = "ws://5.78.181.14:443";
let ws: WebSocket | null = null;
let peerGrads: Uint8Array | null = null;
// biome-ignore lint/style/useConst: mutable
let receivedWeights: Uint8Array | null = null;
// biome-ignore lint/style/useConst: Svelte $state
let needsSync = $state(false);

// ── Chart ──
function lossPath(history: number[]): string {
  if (history.length < 2) return "";
  const min = Math.min(...history);
  const max = Math.max(...history);
  const range = max - min || 1;
  return history
    .map((v, i) => {
      const x = (i / (history.length - 1)) * 100;
      const y = 100 - ((v - min) / range) * 100;
      return `${i === 0 ? "M" : "L"}${x},${y}`;
    })
    .join(" ");
}

// ── Connect ──
async function connect() {
  status = "Connecting to mesh...";
  ws = new WebSocket(serverUrl);
  ws.binaryType = "arraybuffer";

  await new Promise<void>((resolve, reject) => {
    ws!.onopen = () => {
      ws!.send(
        JSON.stringify({
          type: "register",
          peerId: "browser-" + Date.now(),
          model: "gpt2-124m",
        }),
      );
    };
    ws!.onmessage = (e) => {
      if (typeof e.data === "string") {
        const msg = JSON.parse(e.data);
        if (msg.type === "registered") {
          myId = msg.peerId;
          peerCount = msg.peers - 1;
          connected = true;
          needsSync = !!msg.needsSync;
          status = needsSync
            ? `Waiting for weights from peer...`
            : `In mesh (${msg.peers} peers)`;
          resolve();
        }
      }
    };
    ws!.onerror = () => reject(new Error("Connection failed"));
    setTimeout(() => reject(new Error("Timeout")), 10000);
  });

  ws!.onmessage = (e) => {
    if (typeof e.data === "string") {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "peer-joined") {
          peerCount = msg.peers - 1;
          status = `In mesh (${msg.peers} peers)`;
        }
        if (msg.type === "peer-left") {
          peerCount = msg.peers;
          status = `In mesh (${msg.peers + 1} peers)`;
        }
      } catch {}
    } else if (e.data instanceof ArrayBuffer) {
      const bytes = new Uint8Array(e.data);
      // Check for "WGHT" prefix = weight sync
      if (
        bytes.length > 4 &&
        bytes[0] === 87 &&
        bytes[1] === 71 &&
        bytes[2] === 72 &&
        bytes[3] === 84
      ) {
        receivedWeights = bytes.slice(4); // strip "WGHT" prefix
        needsSync = false;
        status = `Received ${(receivedWeights.length / 1024 / 1024).toFixed(1)}MB weights from peer`;
      } else {
        peerGrads = bytes;
        gradStatus = `Received ${(e.data.byteLength / 1024 / 1024).toFixed(1)}MB`;
      }
    }
  };
  ws!.onclose = () => {
    connected = false;
    status = "Disconnected";
  };
}

// ── E3M0 ──
async function compress(grads: Float32Array[]): Promise<Uint8Array> {
  const { e3m0Quantize } = await import("../../../../../src/distributed/e3m0");
  const parts: Uint8Array[] = [];
  const h = new Uint8Array(4);
  new DataView(h.buffer).setUint32(0, grads.length, true);
  parts.push(h);
  for (const pg of grads) {
    const padded = new Float32Array(Math.ceil(pg.length / 8) * 8);
    padded.set(pg);
    const { codes, scales } = e3m0Quantize(padded);
    const ph = new Uint8Array(12);
    const pv = new DataView(ph.buffer);
    pv.setUint32(0, pg.length, true);
    pv.setUint32(4, codes.byteLength, true);
    pv.setUint32(8, scales.byteLength, true);
    parts.push(ph);
    parts.push(
      new Uint8Array(codes.buffer, codes.byteOffset, codes.byteLength),
    );
    parts.push(scales);
  }
  const total = parts.reduce((s, p) => s + p.length, 0);
  const result = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    result.set(p, off);
    off += p.length;
  }
  return result;
}

async function decompress(data: Uint8Array): Promise<Float32Array[]> {
  const { e3m0Dequantize } = await import(
    "../../../../../src/distributed/e3m0"
  );
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let offset = 0;
  const n = view.getUint32(offset, true);
  offset += 4;
  const result: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    const nv = view.getUint32(offset, true);
    offset += 4;
    const cl = view.getUint32(offset, true);
    offset += 4;
    const sl = view.getUint32(offset, true);
    offset += 4;
    const codes = new Uint32Array(
      data.buffer.slice(
        data.byteOffset + offset,
        data.byteOffset + offset + cl,
      ),
    );
    offset += cl;
    const scales = new Uint8Array(
      data.buffer.slice(
        data.byteOffset + offset,
        data.byteOffset + offset + sl,
      ),
    );
    offset += sl;
    result.push(
      e3m0Dequantize(codes, scales, Math.ceil(nv / 8) * 8).slice(0, nv),
    );
  }
  return result;
}

// ── Train ──
async function startPretraining() {
  if (!modelStore.api || !modelStore.model || !modelStore.tokenizer) return;
  running = true;
  const api = modelStore.api;
  const model = modelStore.model;
  const tokenizer = modelStore.tokenizer;

  // If we received weights from a peer (late join), load raw f32 weights
  if (receivedWeights) {
    status = "Loading weights from peer...";
    const wv = new DataView(
      receivedWeights.buffer,
      receivedWeights.byteOffset,
      receivedWeights.byteLength,
    );
    let woff = 0;
    const np = wv.getUint32(woff, true);
    woff += 4;
    const allP = model.getAllParameters();
    if (np === allP.length) {
      await api.beginStep();
      for (let i = 0; i < np; i++) {
        const rank = wv.getUint32(woff, true);
        woff += 4;
        const shape: number[] = [];
        for (let d = 0; d < rank; d++) {
          shape.push(wv.getUint32(woff, true));
          woff += 4;
        }
        const nel = shape.reduce((a, b) => a * b, 1);
        const f32 = new Float32Array(
          receivedWeights.buffer,
          receivedWeights.byteOffset + woff,
          nel,
        );
        woff += nel * 4;
        api.copy_(
          allP[i],
          api.tensorFromArray(f32, shape, { device: "webgpu" }),
        );
      }
      api.endStep();
      await api.markStep();
      status = `Loaded ${(receivedWeights.length / 1024 / 1024).toFixed(0)}MB weights from peer`;
    }
    receivedWeights = null;
  }

  model.train(true);
  model.enableCheckpointing(true);
  const params = model.getAllParameters();

  const { Adam } = await import("../../../../../src/optim");
  const { clipGradNorm_ } = await import("../../../../../src/nn");
  const { NesterovOuterOptimizer } = await import(
    "../../../../../src/distributed/outer-optimizer"
  );
  const optimizer = new Adam(params, { lr: 1e-4, weightDecay: 0.1 }, api);
  const outerOpt = new NesterovOuterOptimizer(api, { lr: 0.7, momentum: 0.9 });

  // Load dataset
  const dataResp = await fetch("/datasets/austen.txt");
  const dataText = await dataResp.text();
  const tokens = tokenizer.encode(dataText);
  status = `Training on ${tokens.length.toLocaleString()} tokens`;

  const BATCH = 1;
  const SEQ = 128;
  const INNER = 20;
  const maxStart = Math.max(1, tokens.length - SEQ - 1);
  let totalTokens = 0;

  for (let r = 0; ; r++) {
    if (!running) break;
    round = r;

    // Snapshot
    const snapshot: Float32Array[] = [];
    for (const p of params) snapshot.push(new Float32Array(await p.cpu()));

    // Inner loop
    for (let s = 0; s < INNER && running; s++) {
      innerStep = s;
      const offset = ((r * INNER + s) * BATCH * SEQ) % maxStart;

      // Show current training text
      const chunkTokens = tokens.slice(offset, offset + SEQ);
      currentChunk = tokenizer.decode(chunkTokens);

      const inputData: number[] = [];
      const targetData: number[] = [];
      for (let b = 0; b < BATCH; b++) {
        const start = (offset + b * SEQ) % maxStart;
        for (let i = 0; i < SEQ; i++) {
          inputData.push(tokens[start + i]);
          targetData.push(tokens[start + i + 1]);
        }
      }

      const t0 = performance.now();
      await api.beginStep();
      const input = api.tensorFromArray(inputData, [BATCH, SEQ], {
        device: "webgpu",
      });
      const target = api.tensorFromArray(targetData, [BATCH, SEQ], {
        device: "webgpu",
      });
      const l = api.tidy(() => {
        const lo = api.autocast(
          () => model.forwardWithLoss(input, target).loss,
        );
        api.keep(lo);
        return lo;
      });
      const lossVal = await l.item();
      await l.backward();
      clipGradNorm_(api, params, 1.0);
      optimizer.step();
      optimizer.zeroGrad();
      input.dispose();
      target.dispose();
      api.endStep();
      await api.markStep();

      loss = lossVal;
      lossHistory = [...lossHistory, lossVal];
      totalTokens += BATCH * SEQ;
      tokensProcessed = totalTokens;
      tokPerSec = (BATCH * SEQ) / ((performance.now() - t0) / 1000);
    }

    if (!running) break;

    // ── DiLoCo gradient exchange ──
    if (ws && connected) {
      gradStatus = "Computing pseudo-gradients...";
      const pseudoGrads: Float32Array[] = [];
      for (let i = 0; i < params.length; i++) {
        const local = await params[i].cpu();
        const delta = new Float32Array(local.length);
        for (let j = 0; j < delta.length; j++)
          delta[j] = local[j] - snapshot[i][j];
        pseudoGrads.push(delta);
      }

      gradStatus = "Compressing & sending...";
      const compressed = await compress(pseudoGrads);
      ws.send(JSON.stringify({ type: "request-neighbors", round: r }));
      ws.send(compressed);
      gradStatus = `Sent ${(compressed.length / 1024 / 1024).toFixed(1)}MB`;

      // Use peer grads if any have arrived (from this or previous rounds).
      // Don't block — the V100 peer may be on a different round cadence.
      if (peerGrads) {
        gradStatus = "Averaging with peer...";
        const pg = await decompress(peerGrads);
        peerGrads = null;
        contributors = 2;

        if (pg.length === params.length) {
          // Average pseudo-gradients
          const avgGrads = pseudoGrads.map((g) => new Float32Array(g.length));
          for (let i = 0; i < params.length; i++) {
            for (let j = 0; j < avgGrads[i].length; j++)
              avgGrads[i][j] = (pseudoGrads[i][j] + pg[i][j]) / 2;
          }

          // Restore snapshot + Nesterov outer update (matches V100)
          await api.beginStep();
          for (let i = 0; i < params.length; i++) {
            api.copy_(
              params[i],
              api.tensorFromArray(snapshot[i], params[i].shape, {
                device: "webgpu",
              }),
            );
          }
          const avgTensors = avgGrads.map((g, i) =>
            api.tensorFromArray(g, params[i].shape, { device: "webgpu" }),
          );
          await outerOpt.step(params, avgTensors);
          api.endStep();
          await api.markStep();
          gradStatus = `Round ${r}: averaged with peer`;
        }
      } else {
        contributors = 1;
        gradStatus = `Round ${r}: solo`;
      }
    }
  }
  running = false;
}

// ── Generate ──
async function generate() {
  if (
    !modelStore.api ||
    !modelStore.model ||
    !modelStore.tokenizer ||
    generating
  )
    return;
  generating = true;
  genOutput = "";
  const { generateTokens } = await import("$lib/torchlette/inference");
  const model = modelStore.model;
  model.train(false);
  for await (const token of generateTokens(
    modelStore.api,
    model,
    modelStore.tokenizer,
    genPrompt,
    { maxTokens: 60, temperature: 0.8, topK: 40 },
  )) {
    genOutput += token;
  }
  model.train(true);
  generating = false;
}

onMount(() => {
  modelStore.loadForPretraining();
});
</script>

<div class="text-slate-200 font-sans">
  <!-- Header -->
  <header class="border-b border-slate-800 px-2 p-2 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <h1 class="text-sm font-bold tracking-tight">GPT-2 DiLoCo Pretraining</h1>
      <span class="text-xs text-slate-500 font-mono">124M params &middot; from scratch</span>
    </div>
    <div class="flex items-center gap-2">
      {#if connected}
        <span class="w-2 h-2 rounded-full bg-green-500 inline-block"></span>
        <span class="text-xs text-green-400 font-mono">{peerCount + 1} peers</span>
      {:else}
        <span class="w-2 h-2 rounded-full bg-slate-600 inline-block"></span>
        <span class="text-xs text-slate-500">offline</span>
      {/if}
    </div>
  </header>

  <div class="max-w-4xl mx-auto p-2 flex flex-col gap-2">
    <!-- Status bar -->
    <div class="flex items-center gap-4 text-xs font-mono">
      <span class="text-slate-500">{status}</span>
      {#if running}
        <span>round <span class="text-blue-400">{round}</span></span>
        <span>step <span class="text-blue-400">{innerStep}</span></span>
        <span>loss <span class="text-amber-400">{loss.toFixed(4)}</span></span>
        <span><span class="text-green-400">{tokPerSec.toFixed(0)}</span> tok/s</span>
        <span class="text-slate-600">{tokensProcessed.toLocaleString()} tokens total</span>
      {/if}
    </div>

    {#if !modelStore.isReady}
      <!-- Loading -->
      <div class="border border-slate-800 p-2 text-center">
        <p class="text-sm text-slate-400">{modelStore.status}</p>
        {#if modelStore.progress > 0}
          <div class="mt-2 h-1 bg-slate-800 rounded">
            <div class="h-full bg-blue-600 rounded transition-all" style="width:{modelStore.progress}%"></div>
          </div>
        {/if}
      </div>
    {:else}
      <!-- Controls -->
      <div class="flex gap-2">
        {#if !connected}
          <button onclick={connect} class="px-2 py-0.5 text-xs bg-blue-700 hover:bg-blue-600 text-white">Join Mesh</button>
        {:else}
          <span class="px-2 py-0.5 text-xs bg-green-900 text-green-300">In Mesh</span>
        {/if}

        {#if !running}
          <button onclick={startPretraining} class="px-2 py-0.5 text-xs bg-green-700 hover:bg-green-600 text-white">Start Pretraining</button>
        {:else}
          <button onclick={() => { running = false; }} class="px-2 py-0.5 text-xs bg-red-700 hover:bg-red-600 text-white">Stop</button>
        {/if}

        <button onclick={generate} disabled={generating} class="px-2 py-0.5 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300 disabled:opacity-30">
          {generating ? "Generating..." : "Test Generate"}
        </button>
      </div>

      <!-- Main content: 2 columns -->
      <div class="grid grid-cols-2 gap-4">
        <!-- Left: Training data + loss -->
        <div class="flex flex-col gap-3">
          <!-- Current training chunk -->
          <div class="border border-slate-800 p-2">
            <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">Training on</div>
            <div class="font-mono text-xs text-slate-400 h-24 overflow-hidden leading-relaxed">
              {#if currentChunk}
                {currentChunk}
              {:else}
                <span class="text-slate-600 italic">Waiting to start...</span>
              {/if}
            </div>
          </div>

          <!-- Loss chart -->
          {#if lossHistory.length >= 2}
            <div class="border border-slate-800 p-2">
              <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">Loss</div>
              <svg viewBox="-2 -2 104 54" class="w-full h-24" preserveAspectRatio="none">
                <path d={lossPath(lossHistory)} fill="none" stroke="#3b82f6" stroke-width="1" vector-effect="non-scaling-stroke" />
              </svg>
              <div class="flex justify-between text-xs font-mono text-slate-600">
                <span>{Math.min(...lossHistory).toFixed(2)}</span>
                <span>{lossHistory.length} steps</span>
                <span>{Math.max(...lossHistory).toFixed(2)}</span>
              </div>
            </div>
          {/if}
        </div>

        <!-- Right: Mesh + Generation -->
        <div class="flex flex-col gap-3">
          <!-- Gradient exchange status -->
          <div class="border border-slate-800 p-2">
            <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">Gradient Exchange</div>
            <div class="font-mono text-xs space-y-1">
              <div class="text-slate-400">{gradStatus || "No exchange yet"}</div>
              {#if contributors > 0}
                <div class="text-slate-500">Contributors: <span class="text-slate-300">{contributors}</span></div>
              {/if}
              <div class="text-slate-500">Peers: <span class="text-slate-300">{peerCount + (connected ? 1 : 0)}</span></div>
            </div>
          </div>

          <!-- Generation output -->
          <div class="border border-slate-800 p-2">
            <div class="text-xs uppercase tracking-wider text-slate-500 mb-1">Generation Test</div>
            <div class="flex gap-1 mb-2">
              <input type="text" bind:value={genPrompt} class="flex-1 bg-slate-900 border border-slate-700 text-xs font-mono px-1 py-0.5 text-slate-300" placeholder="Prompt..." />
              <button onclick={generate} disabled={generating} class="px-2 py-0.5 text-xs bg-slate-700 text-slate-300 disabled:opacity-30">Go</button>
            </div>
            <div class="font-mono text-xs text-slate-400 h-20 overflow-y-auto whitespace-pre-wrap">
              {#if genOutput}
                <span class="text-slate-500">{genPrompt}</span>{genOutput}
              {:else}
                <span class="text-slate-600 italic">Generate to test model quality...</span>
              {/if}
            </div>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>
