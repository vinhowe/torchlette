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
let peerTokenCount = 0;
let peerRound = -1;
// biome-ignore lint/style/useConst: mutable
let receivedWeights: Uint8Array | null = null;
// biome-ignore lint/style/useConst: Svelte $state
let needsSync = $state(false);
// biome-ignore lint/style/useConst: Svelte $state
let downloadProgress = $state(0); // 0-100, for weight/data downloads
// biome-ignore lint/style/useConst: Svelte $state
let downloadLabel = $state("");

// ── Chart ──
function lossPath(history: number[]): string {
  if (history.length < 2) return "";
  const min = Math.min(...history);
  const max = Math.max(...history);
  const range = max - min || 1;
  return history
    .map((v, i) => {
      const x = (i / (history.length - 1)) * 200;
      const y = 50 - ((v - min) / range) * 50;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

// ── Global stats (from coordinator) ──
// biome-ignore lint/style/useConst: Svelte $state
let globalTokens = $state(0);
// biome-ignore lint/style/useConst: Svelte $state
let globalRound = $state(0);
// biome-ignore lint/style/useConst: Svelte $state
let globalExchanges = $state(0);
// biome-ignore lint/style/useConst: Svelte $state
let globalPeersEver = $state(0);
// biome-ignore lint/style/useConst: Svelte $state
let globalLossHistory = $state<Array<{ round: number; loss: number }>>([]);

function globalLossPath(
  history: Array<{ round: number; loss: number }>,
): string {
  if (history.length < 2) return "";
  const losses = history.map((h) => h.loss);
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const range = max - min || 1;
  return history
    .map((h, i) => {
      const x = (i / (history.length - 1)) * 200;
      const y = 50 - ((h.loss - min) / range) * 50;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

// ── WebRTC P2P ──
const ICE_CONFIG = {
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    {
      urls: "turn:5.78.181.14:3478",
      username: "torchlette",
      credential: "torchlette123",
    },
  ],
};
const rtcPeers = new Map<
  string,
  { pc: RTCPeerConnection; dc: RTCDataChannel | null; ready: boolean }
>();

function setupPeerConnection(peerId: string, initiator: boolean) {
  if (rtcPeers.has(peerId)) return;
  const pc = new RTCPeerConnection(ICE_CONFIG);
  const entry = { pc, dc: null as RTCDataChannel | null, ready: false };
  rtcPeers.set(peerId, entry);

  pc.onicecandidate = (e) => {
    if (e.candidate && ws?.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "signal",
          target: peerId,
          signal: { type: "candidate", candidate: e.candidate },
        }),
      );
    }
  };

  pc.ondatachannel = (e) => {
    entry.dc = e.channel;
    setupDataChannel(entry, peerId);
  };

  if (initiator) {
    const dc = pc.createDataChannel("grads", { ordered: true });
    entry.dc = dc;
    setupDataChannel(entry, peerId);
    pc.createOffer().then((offer) => {
      pc.setLocalDescription(offer);
      ws?.send(
        JSON.stringify({
          type: "signal",
          target: peerId,
          signal: { type: "offer", sdp: offer.sdp },
        }),
      );
    });
  }
}

function setupDataChannel(
  entry: { dc: RTCDataChannel | null; ready: boolean },
  peerId: string,
) {
  const dc = entry.dc!;
  dc.binaryType = "arraybuffer";
  dc.onopen = () => {
    entry.ready = true;
    console.log(`[webrtc] data channel open to ${peerId}`);
  };
  dc.onclose = () => {
    entry.ready = false;
    console.log(`[webrtc] data channel closed to ${peerId}`);
  };
  dc.onmessage = (e) => {
    handleP2PMessage(peerId, e.data);
  };
}

// Chunked send: WebRTC data channels have ~256KB message limit
const CHUNK_SIZE = 200 * 1024; // 200KB chunks

async function sendOverDataChannel(dc: RTCDataChannel, data: Uint8Array) {
  const BUFFER_THRESHOLD = 1024 * 1024; // 1MB
  // Header: 8 bytes — "CHNK" + u32 totalSize
  const header = new Uint8Array(8);
  header[0] = 67;
  header[1] = 72;
  header[2] = 78;
  header[3] = 75;
  new DataView(header.buffer).setUint32(4, data.length, true);
  dc.send(header);
  for (let i = 0; i < data.length; i += CHUNK_SIZE) {
    // Wait for send buffer to drain before sending more
    while (dc.bufferedAmount > BUFFER_THRESHOLD) {
      await new Promise((r) => setTimeout(r, 10));
    }
    dc.send(data.slice(i, i + CHUNK_SIZE));
  }
}

// Reassembly buffer per peer
const reassembly = new Map<
  string,
  { totalSize: number; received: Uint8Array[]; got: number }
>();

function handleP2PMessage(peerId: string, data: ArrayBuffer) {
  const bytes = new Uint8Array(data);
  // Check for CHNK header
  if (
    bytes.length === 8 &&
    bytes[0] === 67 &&
    bytes[1] === 72 &&
    bytes[2] === 78 &&
    bytes[3] === 75
  ) {
    const totalSize = new DataView(bytes.buffer).getUint32(4, true);
    reassembly.set(peerId, { totalSize, received: [], got: 0 });
    return;
  }
  const buf = reassembly.get(peerId);
  if (buf) {
    buf.received.push(bytes);
    buf.got += bytes.length;
    if (buf.got >= buf.totalSize) {
      // Reassemble
      const full = new Uint8Array(buf.totalSize);
      let off = 0;
      for (const chunk of buf.received) {
        full.set(chunk, off);
        off += chunk.length;
      }
      reassembly.delete(peerId);
      // Process as gradient (same format as WebSocket binary)
      if (
        full.length > 16 &&
        full[0] === 71 &&
        full[1] === 82 &&
        full[2] === 65 &&
        full[3] === 68
      ) {
        const dv = new DataView(full.buffer, full.byteOffset, full.byteLength);
        peerTokenCount = dv.getUint32(4, true);
        peerRound = dv.getUint32(8, true);
        peerGrads = full.slice(16);
        gradStatus = `P2P: ${(full.length / 1024 / 1024).toFixed(1)}MB (${peerTokenCount} tok, r${peerRound})`;
      }
    }
  }
}

async function handleSignal(from: string, signal: any) {
  if (signal.type === "offer") {
    setupPeerConnection(from, false);
    const entry = rtcPeers.get(from)!;
    await entry.pc.setRemoteDescription({ type: "offer", sdp: signal.sdp });
    const answer = await entry.pc.createAnswer();
    await entry.pc.setLocalDescription(answer);
    ws?.send(
      JSON.stringify({
        type: "signal",
        target: from,
        signal: { type: "answer", sdp: answer.sdp },
      }),
    );
  } else if (signal.type === "answer") {
    const entry = rtcPeers.get(from);
    if (entry)
      await entry.pc.setRemoteDescription({ type: "answer", sdp: signal.sdp });
  } else if (signal.type === "candidate") {
    const entry = rtcPeers.get(from);
    if (entry) await entry.pc.addIceCandidate(signal.candidate);
  }
}

// ── Sync log ──
// biome-ignore lint/style/useConst: Svelte $state
let syncLog = $state<
  Array<{
    round: number;
    lag: number;
    weight: string;
    tokens: string;
    time: string;
  }>
>([]);

// ── Connect ──
async function connect() {
  status = "Connecting to mesh...";
  ws = new WebSocket(serverUrl);
  ws.binaryType = "arraybuffer";

  // Single unified handler from the start — avoids race where binary DLTA/WGHT
  // arrives between the registration response and handler swap
  let resolveRegistration: (() => void) | null = null;

  ws.onopen = () => {
    ws!.send(
      JSON.stringify({
        type: "register",
        peerId: "browser-" + Date.now(),
        model: "gpt2-124m",
      }),
    );
  };

  ws.onmessage = (e) => {
    if (typeof e.data === "string") {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "registered") {
          myId = msg.peerId;
          peerCount = msg.peers - 1;
          connected = true;
          needsSync = !!msg.needsSync;
          status = needsSync
            ? `Waiting for weights from peer...`
            : `In mesh (${msg.peers} peers)`;
          resolveRegistration?.();
        } else if (msg.type === "peer-joined") {
          peerCount = msg.peers - 1;
          status = `In mesh (${msg.peers} peers)`;
        } else if (msg.type === "peer-left") {
          peerCount = msg.peers - 1; // msg.peers is total remaining (includes us)
          status = `In mesh (${msg.peers} peers)`;
        } else if (msg.type === "pong" && msg.stats) {
          globalTokens = msg.stats.totalTokens;
          globalRound = msg.stats.maxRound;
          globalExchanges = msg.stats.totalExchanges;
          globalPeersEver = msg.stats.totalPeersEver;
          if (msg.stats.lossHistory) globalLossHistory = msg.stats.lossHistory;
        } else if (msg.type === "signal") {
          handleSignal(msg.from, msg.signal);
        } else if (msg.type === "neighbors") {
          // Establish WebRTC connections to assigned neighbors
          for (const nid of msg.neighbors) {
            if (!rtcPeers.has(nid)) {
              console.log(`[webrtc] initiating connection to ${nid}`);
              setupPeerConnection(nid, true);
            }
          }
        }
      } catch {}
    } else if (e.data instanceof ArrayBuffer) {
      const bytes = new Uint8Array(e.data);
      // Check for weight sync prefixes (4 bytes)
      const prefix =
        bytes.length > 4
          ? String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3])
          : "";
      if (prefix === "DLTA" || prefix === "WGHT" || prefix === "F16W") {
        receivedWeights = bytes;
        needsSync = false;
        status = `Received ${((bytes.length - 4) / 1024 / 1024).toFixed(1)}MB weights (${prefix}) from peer`;
      } else {
        // Gradient blob — GRAD header (16 bytes: "GRAD" + u32 tokens + u32 round + f32 loss)
        if (
          bytes.length > 16 &&
          bytes[0] === 71 &&
          bytes[1] === 82 &&
          bytes[2] === 65 &&
          bytes[3] === 68
        ) {
          const dv = new DataView(
            bytes.buffer,
            bytes.byteOffset,
            bytes.byteLength,
          );
          peerTokenCount = dv.getUint32(4, true);
          peerRound = dv.getUint32(8, true);
          peerGrads = bytes.slice(16);
          gradStatus = `Received ${(bytes.length / 1024 / 1024).toFixed(1)}MB (${peerTokenCount} tok, r${peerRound})`;
        } else {
          peerTokenCount = 0;
          peerRound = -1;
          peerGrads = bytes;
          gradStatus = `Received ${(e.data.byteLength / 1024 / 1024).toFixed(1)}MB`;
        }
      }
    }
  };
  ws.onclose = () => {
    connected = false;
    status = "Disconnected — reconnecting...";
    setTimeout(() => {
      if (!connected) connect().catch(() => {});
    }, 3000);
  };

  ws.onerror = () => {
    resolveRegistration = null;
  };

  // Wait for registration to complete
  await new Promise<void>((resolve, reject) => {
    resolveRegistration = resolve;
    setTimeout(() => reject(new Error("Server timeout")), 10000);
  });

  // Keepalive ping every 30s to prevent idle timeout
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ping" }));
    }
  }, 30000);

  // Immediately request weights from peers and load them
  if (modelStore.api && modelStore.model && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "request-weights" }));
    downloadLabel = "Requesting checkpoint (~238MB f16)...";
    downloadProgress = 0;

    // Wait up to 120s for weights
    await new Promise<void>((resolve) => {
      let elapsed = 0;
      const check = setInterval(() => {
        if (receivedWeights) {
          clearInterval(check);
          downloadLabel = `Received ${(receivedWeights.length / 1024 / 1024).toFixed(1)}MB checkpoint`;
          downloadProgress = 50;
          resolve();
          return;
        }
        elapsed += 500;
        downloadProgress = Math.min(45, Math.round((elapsed / 120000) * 45));
        downloadLabel = `Downloading checkpoint (~238MB)... ${(elapsed / 1000).toFixed(0)}s`;
      }, 500);
      setTimeout(() => {
        clearInterval(check);
        resolve();
      }, 120000);
    });

    // Load weights if received
    if (receivedWeights) {
      const prefix = String.fromCharCode(
        receivedWeights[0],
        receivedWeights[1],
        receivedWeights[2],
        receivedWeights[3],
      );
      const payload = receivedWeights.slice(4);
      const allP = modelStore.model.getAllParameters();
      const api = modelStore.api;
      const isF16 = prefix === "F16W" || prefix === "DLTA";
      const isDelta = prefix === "DLTA";

      if (isF16) {
        const dv = new DataView(
          payload.buffer,
          payload.byteOffset,
          payload.byteLength,
        );
        let doff = 0;
        const np = dv.getUint32(doff, true);
        doff += 4;
        const totalMB = (payload.length / 1024 / 1024).toFixed(1);
        if (np === allP.length) {
          downloadLabel = `Loading ${np} params to GPU (0/${totalMB}MB)...`;
          await api.beginStep();
          for (let i = 0; i < np; i++) {
            const nel = dv.getUint32(doff, true);
            doff += 4;
            const f16 = new Uint16Array(
              payload.buffer,
              payload.byteOffset + doff,
              nel,
            );
            doff += nel * 2;
            const f32 = new Float32Array(nel);
            for (let j = 0; j < nel; j++) {
              const h = f16[j];
              const sign = (h >> 15) & 1;
              const exp = (h >> 10) & 0x1f;
              const mant = h & 0x3ff;
              let val: number;
              if (exp === 0) {
                val = (mant / 1024) * 2 ** -14;
              } else if (exp === 31) {
                val = mant ? NaN : Infinity;
              } else {
                val = 2 ** (exp - 15) * (1 + mant / 1024);
              }
              f32[j] = sign ? -val : val;
            }
            const t = api.tensorFromArray(f32, allP[i].shape, {
              device: "webgpu",
            });
            if (isDelta) {
              api.add_(allP[i], t);
            } else {
              api.copy_(allP[i], t);
            }
            const loadedMB = (doff / 1024 / 1024).toFixed(1);
            downloadProgress = 50 + Math.round(((i + 1) / np) * 50);
            downloadLabel = `Loading params to GPU (${loadedMB}/${totalMB}MB)...`;
            // Flush every 20 params to avoid OOM on 8GB GPUs
            if (i % 20 === 0) {
              await api._runtime().forceAllPending();
              await api.markStep();
              await api.beginStep();
              await new Promise((r) => setTimeout(r, 0));
            }
          }
          await api._runtime().forceAllPending();
          api.endStep();
          await api.markStep();
          status = `Loaded ${(payload.length / 1024 / 1024).toFixed(0)}MB weights — ready to train or generate`;
          console.log("[weight-sync] Loaded weights in connect()");
        }
      }
      receivedWeights = null;
    }
    downloadLabel = "";
    downloadProgress = 0;
  }
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

  // Weights are loaded in connect() — no need to load here

  model.train(true);
  model.enableCheckpointing(true);
  model.fullCheckpoint = true;
  const params = model.getAllParameters();

  const { Adam } = await import("../../../../../src/optim");
  const { clipGradNorm_ } = await import("../../../../../src/nn");
  const { NesterovOuterOptimizer } = await import(
    "../../../../../src/distributed/outer-optimizer"
  );
  const optimizer = new Adam(params, { lr: 1e-4, weightDecay: 0.1 }, api);
  const outerOpt = new NesterovOuterOptimizer(api, { lr: 0.7, momentum: 0.9 });

  // Match V100 agent: half-size model, TinyStories, seq=256
  const BATCH = 1;
  const SEQ = 256;
  const INNER = 20;
  const HF_TOTAL_ROWS = 2_119_719;
  const HF_FETCH_ROWS = 100;
  let totalTokens = 0;

  async function fetchFreshData(): Promise<number[]> {
    const offset = Math.floor(Math.random() * (HF_TOTAL_ROWS - HF_FETCH_ROWS));
    const url = `https://datasets-server.huggingface.co/rows?dataset=roneneldan/TinyStories&config=default&split=train&offset=${offset}&length=${HF_FETCH_ROWS}`;
    const resp = await fetch(url);
    const data = await resp.json();
    const text = data.rows.map((r: any) => r.row.text).join("\n\n");
    const toks = tokenizer.encode(text);
    status = `Round ${round}: ${toks.length.toLocaleString()} tokens (offset ${offset})`;
    return toks;
  }

  for (let r = 0; ; r++) {
    if (!running) break;
    round = r;

    // Fetch fresh training data each round
    const tokens = await fetchFreshData();
    const maxStart = Math.max(1, tokens.length - SEQ - 1);

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
      // Prepend GRAD header: "GRAD" + u32 tokens + u32 round + f32 loss (16 bytes)
      const myTokens = INNER * BATCH * SEQ;
      const gradHeader = new Uint8Array(16);
      const ghDv = new DataView(gradHeader.buffer);
      gradHeader[0] = 71;
      gradHeader[1] = 82;
      gradHeader[2] = 65;
      gradHeader[3] = 68;
      ghDv.setUint32(4, myTokens, true);
      ghDv.setUint32(8, r, true);
      ghDv.setFloat32(12, loss, true);
      const gradPayload = new Uint8Array(16 + compressed.length);
      gradPayload.set(gradHeader);
      gradPayload.set(compressed, 16);
      // Send gradients: prefer P2P data channels, fall back to relay
      ws.send(JSON.stringify({ type: "request-neighbors", round: r }));
      let sentP2P = 0;
      for (const [pid, entry] of rtcPeers) {
        if (entry.ready && entry.dc?.readyState === "open") {
          await sendOverDataChannel(entry.dc, gradPayload);
          sentP2P++;
        }
      }
      if (sentP2P === 0) {
        // No P2P connections — fall back to relay
        ws.send(gradPayload);
        gradStatus = `Sent ${(gradPayload.length / 1024 / 1024).toFixed(1)}MB relay (${myTokens} tok)`;
      } else {
        gradStatus = `Sent ${(gradPayload.length / 1024 / 1024).toFixed(1)}MB P2P to ${sentP2P} peers (${myTokens} tok)`;
      }

      // Use peer grads if any have arrived. Apply staleness penalty.
      if (peerGrads) {
        const roundLag = peerRound >= 0 ? Math.abs(r - peerRound) : 0;
        if (roundLag > 10) {
          // Too stale — discard and re-download checkpoint
          console.log(
            `[diloco] Discarding stale gradient (${roundLag} rounds behind) — re-requesting weights`,
          );
          peerGrads = null;
          contributors = 1;
          gradStatus = `Stale gradient (lag=${roundLag}), requesting fresh weights...`;
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "request-weights" }));
          }
        } else {
          gradStatus = "Averaging with peer...";
          const pg = await decompress(peerGrads);
          peerGrads = null;
          contributors = 2;

          if (pg.length === params.length) {
            // Weighted average with staleness discount: weight = tokens * 0.7^lag
            const myToks = INNER * BATCH * SEQ;
            const peerToks = peerTokenCount || myToks;
            const stalenessWeight = 0.7 ** roundLag;
            const effectivePeerToks = peerToks * stalenessWeight;
            const totalToks = myToks + effectivePeerToks;
            const avgGrads = pseudoGrads.map((g) => new Float32Array(g.length));
            for (let i = 0; i < params.length; i++) {
              for (let j = 0; j < avgGrads[i].length; j++)
                avgGrads[i][j] =
                  (pseudoGrads[i][j] * myToks + pg[i][j] * effectivePeerToks) /
                  totalToks;
            }
            gradStatus =
              roundLag > 0
                ? `Averaged: ${myToks} local + ${peerToks} peer tokens (lag=${roundLag}, weight=${stalenessWeight.toFixed(2)})`
                : `Averaged: ${myToks} local + ${peerToks} peer tokens`;

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
            await api._runtime().forceAllPending();
            await outerOpt.step(params, avgTensors);
            api.endStep();
            await api.markStep();
            gradStatus = `Round ${r}: averaged with peer`;
            syncLog = [
              ...syncLog.slice(-19),
              {
                round: r,
                lag: roundLag,
                weight: stalenessWeight.toFixed(2),
                tokens: `${myToks}+${peerToks}`,
                time: new Date().toLocaleTimeString(),
              },
            ];
          }
        }
      } else {
        contributors = 1;
        gradStatus = `Round ${r}: solo`;
        syncLog = [
          ...syncLog.slice(-19),
          {
            round: r,
            lag: -1,
            weight: "solo",
            tokens: `${INNER * BATCH * SEQ}`,
            time: new Date().toLocaleTimeString(),
          },
        ];
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

<div class="min-h-screen bg-[#0a0a0a] text-[#999] font-mono text-[11px] leading-tight selection:bg-[#333]">
  <div class="border-b border-[#222] px-3 py-1.5 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <span class="text-[#eee] font-bold text-xs">diloco</span>
      <span class="text-[#555]">gpt2-124m</span>
      <span class="text-[#555]">{status}</span>
    </div>
    <div class="flex items-center gap-3">
      {#if running}
        <span class="text-[#5f5]">TRAIN</span>
      {/if}
      {#if connected}
        <span class="text-[#5f5]">{peerCount + 1}p</span>
      {:else}
        <span class="text-[#555]">offline</span>
      {/if}
    </div>
  </div>

  {#if downloadLabel}
    <div class="border-b border-[#222] px-3 py-1">
      <span class="text-[#888]">{downloadLabel}</span>
      <span class="text-[#5af] ml-2">{downloadProgress}%</span>
      <div class="mt-1 h-px bg-[#222] relative">
        <div class="absolute top-0 left-0 h-px bg-[#5af] transition-all" style="width:{downloadProgress}%"></div>
      </div>
    </div>
  {/if}

  {#if !modelStore.isReady}
    <div class="px-3 py-8 text-center">
      <div class="text-[#666]">{modelStore.status}</div>
      {#if modelStore.progress > 0}
        <div class="mt-2 max-w-xs mx-auto h-px bg-[#222] relative">
          <div class="absolute top-0 left-0 h-px bg-[#5af]" style="width:{modelStore.progress}%"></div>
        </div>
      {/if}
    </div>
  {:else}
    <div class="flex gap-0">
      <!-- left: main area -->
      <div class="flex-1 min-w-0">
        <!-- controls -->
        <div class="border-b border-[#222] px-3 py-1.5 flex items-center gap-2">
          {#if !connected}
            <button onclick={connect} class="px-2 py-0.5 bg-[#1a1a2e] border border-[#333] text-[#5af] hover:bg-[#222] active:bg-[#111]">join</button>
          {:else}
            <span class="px-2 py-0.5 border border-[#1a2a1a] text-[#5a5]">mesh</span>
          {/if}
          {#if !running}
            <button onclick={startPretraining} class="px-2 py-0.5 bg-[#1a2a1a] border border-[#333] text-[#5f5] hover:bg-[#222] active:bg-[#111]">train</button>
          {:else}
            <button onclick={() => { running = false; }} class="px-2 py-0.5 bg-[#2a1a1a] border border-[#333] text-[#f55] hover:bg-[#222]">stop</button>
          {/if}
          <button onclick={generate} disabled={generating} class="px-2 py-0.5 bg-[#1a1a1a] border border-[#333] text-[#aaa] hover:bg-[#222] disabled:opacity-30">gen</button>
          <div class="flex-1"></div>
          {#if running}
            <span class="text-[#555]">r{round} s{innerStep}/20</span>
          {/if}
        </div>

        <!-- data stream -->
        <div class="border-b border-[#222] px-3 py-2 min-h-[80px] max-h-[120px] overflow-hidden">
          {#if currentChunk}
            <div class="text-[#666] leading-relaxed break-all">{@html currentChunk.split(' ').map((w, i) =>
              i < 5 ? `<span class="text-[#aaa]">${w}</span>` :
              i < 15 ? `<span class="text-[#777]">${w}</span>` : w
            ).join(' ')}</div>
          {:else}
            <span class="text-[#333]">waiting for data</span>
          {/if}
        </div>

        <!-- loss chart -->
        <div class="border-b border-[#222]">
          {#if lossHistory.length >= 2}
            <div class="px-3 py-1 flex items-center gap-4 border-b border-[#1a1a1a]">
              <span class="text-[#555]">loss</span>
              <span class="text-[#e94]">{loss.toFixed(4)}</span>
              <span class="text-[#555]">min</span>
              <span class="text-[#5a5]">{Math.min(...lossHistory).toFixed(4)}</span>
              <span class="text-[#555]">max</span>
              <span class="text-[#a55]">{Math.max(...lossHistory).toFixed(4)}</span>
              <span class="text-[#555]">{lossHistory.length} steps</span>
            </div>
            <div class="px-3 py-2">
              <svg viewBox="-1 -1 202 52" class="w-full h-32" preserveAspectRatio="none">
                <line x1="0" y1="0" x2="200" y2="0" stroke="#1a1a1a" stroke-width="0.5" vector-effect="non-scaling-stroke" />
                <line x1="0" y1="25" x2="200" y2="25" stroke="#1a1a1a" stroke-width="0.5" vector-effect="non-scaling-stroke" />
                <line x1="0" y1="50" x2="200" y2="50" stroke="#1a1a1a" stroke-width="0.5" vector-effect="non-scaling-stroke" />
                <path d={lossPath(lossHistory)} fill="none" stroke="#e94" stroke-width="1" vector-effect="non-scaling-stroke" />
              </svg>
            </div>
          {:else}
            <div class="px-3 py-6 text-[#333] text-center">no loss data yet</div>
          {/if}
        </div>

        <!-- global loss (from coordinator) -->
        <div class="border-b border-[#222]">
          {#if globalLossHistory.length >= 2}
            <div class="px-3 py-1 flex items-center gap-4 border-b border-[#1a1a1a]">
              <span class="text-[#555]">global loss</span>
              <span class="text-[#5af]">{globalLossHistory[globalLossHistory.length - 1]?.loss.toFixed(4)}</span>
              <span class="text-[#555]">{globalLossHistory.length} reports</span>
            </div>
            <div class="px-3 py-2">
              <svg viewBox="-1 -1 202 52" class="w-full h-20" preserveAspectRatio="none">
                <line x1="0" y1="0" x2="200" y2="0" stroke="#1a1a1a" stroke-width="0.5" vector-effect="non-scaling-stroke" />
                <line x1="0" y1="25" x2="200" y2="25" stroke="#1a1a1a" stroke-width="0.5" vector-effect="non-scaling-stroke" />
                <line x1="0" y1="50" x2="200" y2="50" stroke="#1a1a1a" stroke-width="0.5" vector-effect="non-scaling-stroke" />
                <path d={globalLossPath(globalLossHistory)} fill="none" stroke="#5af" stroke-width="1" vector-effect="non-scaling-stroke" />
              </svg>
            </div>
          {:else}
            <div class="px-3 py-3 text-[#333]">global loss: waiting for coordinator data</div>
          {/if}
        </div>

        <!-- generation -->
        <div class="px-3 py-2">
          <div class="flex gap-1 mb-1">
            <span class="text-[#555]">&gt;</span>
            <input type="text" bind:value={genPrompt} class="flex-1 bg-transparent border-none outline-none text-[#ccc] caret-[#5af]" placeholder="prompt..." />
          </div>
          {#if genOutput}
            <div class="text-[#888] whitespace-pre-wrap mt-1"><span class="text-[#5af]">{genPrompt}</span>{genOutput}{#if generating}<span class="inline-block w-[6px] h-[11px] bg-[#5af] ml-px animate-pulse"></span>{/if}</div>
          {/if}
        </div>
      </div>

      <!-- right sidebar: stats -->
      <div class="w-48 border-l border-[#222] flex-shrink-0">
        <div class="border-b border-[#222] px-3 py-1 text-[#555]">stats</div>
        <div class="divide-y divide-[#1a1a1a]">
          <div class="px-3 py-1.5 flex justify-between">
            <span class="text-[#555]">loss</span>
            <span class="text-[#e94]">{loss ? loss.toFixed(4) : '-'}</span>
          </div>
          <div class="px-3 py-1.5 flex justify-between">
            <span class="text-[#555]">tok/s</span>
            <span class="text-[#5a5]">{tokPerSec ? tokPerSec.toFixed(0) : '-'}</span>
          </div>
          <div class="px-3 py-1.5 flex justify-between">
            <span class="text-[#555]">tokens</span>
            <span>{tokensProcessed ? tokensProcessed.toLocaleString() : '-'}</span>
          </div>
          <div class="px-3 py-1.5 flex justify-between">
            <span class="text-[#555]">round</span>
            <span>{round}</span>
          </div>
          <div class="px-3 py-1.5 flex justify-between">
            <span class="text-[#555]">step</span>
            <span>{innerStep}/20</span>
          </div>
          <div class="px-3 py-1.5 flex justify-between">
            <span class="text-[#555]">peers</span>
            <span class="{connected ? 'text-[#5a5]' : ''}">{peerCount + (connected ? 1 : 0)}</span>
          </div>
          <div class="px-3 py-1.5 flex justify-between">
            <span class="text-[#555]">contrib</span>
            <span class="{contributors > 1 ? 'text-[#5a5]' : ''}">{contributors > 1 ? 'merged' : contributors === 1 ? 'solo' : '-'}</span>
          </div>
        </div>
        <div class="border-t border-[#222] px-3 py-1 text-[#555]">global</div>
        <div class="divide-y divide-[#1a1a1a]">
          <div class="px-3 py-1.5 flex justify-between">
            <span class="text-[#555]">tokens</span>
            <span class="text-[#aaa]">{globalTokens >= 1e6 ? (globalTokens / 1e6).toFixed(1) + 'M' : globalTokens >= 1e3 ? (globalTokens / 1e3).toFixed(0) + 'K' : globalTokens}</span>
          </div>
          <div class="px-3 py-1.5 flex justify-between">
            <span class="text-[#555]">round</span>
            <span>{globalRound}</span>
          </div>
          <div class="px-3 py-1.5 flex justify-between">
            <span class="text-[#555]">syncs</span>
            <span>{globalExchanges}</span>
          </div>
          <div class="px-3 py-1.5 flex justify-between">
            <span class="text-[#555]">peers seen</span>
            <span>{globalPeersEver}</span>
          </div>
        </div>
        <div class="border-t border-[#222] px-3 py-1 text-[#555]">gradient</div>
        <div class="px-3 py-1.5 text-[#666] break-all">{gradStatus || 'idle'}</div>
        <div class="border-t border-[#222] px-3 py-1 text-[#555]">sync log</div>
        <div class="max-h-[200px] overflow-y-auto">
          {#each [...syncLog].reverse() as entry}
            <div class="px-3 py-0.5 border-b border-[#111] flex flex-col">
              <div class="flex justify-between">
                <span class="text-[#555]">r{entry.round}</span>
                <span class="text-[#444]">{entry.time}</span>
              </div>
              <div class="flex justify-between">
                {#if entry.lag === -1}
                  <span class="text-[#665]">solo</span>
                {:else}
                  <span class="text-[#5a5]">lag={entry.lag} w={entry.weight}</span>
                {/if}
                <span class="text-[#555]">{entry.tokens}tok</span>
              </div>
            </div>
          {/each}
          {#if syncLog.length === 0}
            <div class="px-3 py-1 text-[#333]">no syncs yet</div>
          {/if}
        </div>
      </div>
    </div>
  {/if}
</div>
