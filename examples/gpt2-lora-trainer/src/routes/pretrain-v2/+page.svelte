<script lang="ts">
// Browser DiLoCo peer on the VALIDATED v2 stack (task #46 unification):
// WebGPUGPT2Trainer + HierarchicalBarrierStateMachine + WebSocketBrowserTransport
// + f16 wire codec — the exact stack tools/diloco-agent-v2.ts runs, just with the
// browser transport + a browser HF token source. No hand-rolled WS / E3M0 fork.
import { onMount } from "svelte";
import {
  getWebGPUInitError,
  initWebGPU,
  setGPUMemoryLimit,
  Torchlette,
} from "torchlette";
import { GPT2Tokenizer } from "gpt2-browser";
import { fetchTokenizer } from "gpt2-browser";

// v2 stack (browser-safe modules)
import { WebGPUGPT2Trainer } from "../../../../../src/distributed/protocol/webgpu-gpt2-trainer";
import { HierarchicalBarrierStateMachine } from "../../../../../src/distributed/protocol/hierarchical-state-machine";
import { WebSocketBrowserTransport } from "../../../../../src/distributed/transports/websocket-browser";

const SERVER =
  (typeof window !== "undefined" &&
    new URLSearchParams(window.location.search).get("server")) ||
  "ws://localhost:9882";

// Model dims are URL-configurable so the same page can run a 124M peer (default,
// matches the node agent) OR a smaller model that fits a memory-tight machine —
// e.g. /pretrain-v2?layers=6&embed=512&heads=8. The node agent must use the
// matching NUM_LAYERS/EMBED_DIM/NUM_HEADS env so all peers share one model.
const qp = (k: string, d: number) =>
  typeof window === "undefined"
    ? d
    : parseInt(new URLSearchParams(window.location.search).get(k) || `${d}`, 10);
const NUM_LAYERS = qp("layers", 12);
const EMBED_DIM = qp("embed", 768);
const NUM_HEADS = qp("heads", 12);

let status = $state("idle");
let round = $state(-1);
let loss = $state(NaN);
let contributors = $state(0);
let peers = $state(0);
let running = $state(false);
const logLines: string[] = [];
let logText = $state("");
function log(s: string) {
  logLines.push(s);
  logText = logLines.slice(-14).join("\n");
  console.log("[pretrain-v2]", s);
}

// HuggingFace FineWeb-Edu token source (browser fetch + GPT-2 BPE).
function makeHFTokenSource(tok: GPT2Tokenizer) {
  const TOTAL = 9_672_101;
  return {
    async fetch(minTokens: number): Promise<number[]> {
      const toks: number[] = [];
      while (toks.length < minTokens) {
        const offset = Math.floor(Math.random() * (TOTAL - 100));
        const url = `https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT&split=train&offset=${offset}&length=64`;
        try {
          const resp = await fetch(url, { signal: AbortSignal.timeout(30000) });
          const data = await resp.json();
          const text = (data.rows ?? [])
            .map((r: any) => r.row.text)
            .join("\n\n");
          for (const t of tok.encode(text)) toks.push(t);
        } catch (e) {
          await new Promise((r) => setTimeout(r, 2000));
        }
      }
      return toks;
    },
  };
}

async function start() {
  if (running) return;
  running = true;
  try {
    status = "WebGPU init...";
    if (!(await initWebGPU()))
      throw new Error(getWebGPUInitError() || "WebGPU init failed");
    // 124M training peaks ~6.8GB. The late-joiner f16w apply is now streamed
    // param-by-param (WebGPUGPT2Trainer.applyF16W), so it no longer spikes the
    // whole ~500MB upload at once — it fits under an 8GB budget, which is safe
    // alongside Chrome + OS on a 16GB Mac.
    setGPUMemoryLimit(8 * 1024 * 1024 * 1024);
    const api = new Torchlette("webgpu", { enableFusion: true });
    api.manualSeed(42);

    status = "loading tokenizer...";
    const tokData = await fetchTokenizer(() => {});
    const tokenizer = new GPT2Tokenizer();
    tokenizer.load(tokData.vocab, tokData.merges);

    status = "building 124M trainer...";
    const trainer = new WebGPUGPT2Trainer({
      api,
      modelConfig: {
        vocabSize: 50257,
        blockSize: 1024,
        numLayers: NUM_LAYERS,
        numHeads: NUM_HEADS,
        embedDim: EMBED_DIM,
        dropoutRate: 0,
      },
      tokenSource: makeHFTokenSource(tokenizer),
      innerLr: 1e-4,
      outerLr: 0.7,
      outerMu: 0.9,
      innerSteps: 10,
      batchSize: 1,
      seqLen: 256,
      accumSteps: 1,
      weightDecay: 0.1,
      log,
    });
    await trainer.initialize();
    log("trainer ready (124M)");

    const peerId = "browser-" + Date.now();
    status = `connecting to ${SERVER}...`;
    const transport = await WebSocketBrowserTransport.create({
      serverUrl: SERVER,
      peerId,
      model: "gpt2-124m",
      log,
    });
    const sm = new HierarchicalBarrierStateMachine(transport, trainer, {
      quorumMin: 2,
      quorumTargetFrac: 0.5,
      // Generous deadlines: a 124M browser round (inner steps + ~250MB chunked
      // grad exchange over the internet) is slow, and the node must wait at the
      // barrier long enough for the late-joining browser to sync + catch up.
      intraDeadlineMs: 300000,
      interDeadlineMs: 300000,
      globalDeadlineMs: 300000,
      f16wDebounceMs: 10000,
    });
    await sm.awaitJoined();
    const self = sm.getSelf();
    log(`joined: ${self?.peerId} cluster=${self?.clusterId}${self?.isHead ? "/head" : ""}`);
    status = "in mesh — training";
    sm.onReport = (r) => {
      round = r.round;
      loss = r.innerLoss;
      contributors = r.contributors;
      log(
        `round ${r.round}: loss=${r.innerLoss.toFixed(3)} outer=${r.outerStepTaken} contributors=${r.contributors}${r.f16wApplied ? " [synced]" : ""}`,
      );
      (window as any).__V2 = { round, loss, contributors };
    };
    await sm.run(1000);
  } catch (e: any) {
    log("ERROR: " + (e?.stack || String(e)));
    status = "error";
  } finally {
    running = false;
  }
}
</script>

<div style="font:13px monospace;padding:16px;max-width:640px">
  <h2>DiLoCo browser peer — v2 (validated stack)</h2>
  <p>server: {SERVER}</p>
  <p>
    status: <b>{status}</b> · round: {round} · loss:
    {Number.isNaN(loss) ? "—" : loss.toFixed(3)} · contributors: {contributors}
  </p>
  <button onclick={start} disabled={running} style="padding:6px 14px;margin:8px 0">
    {running ? "running…" : "join + train"}
  </button>
  <pre style="background:#111;color:#5f5;padding:10px;white-space:pre-wrap">{logText}</pre>
</div>
