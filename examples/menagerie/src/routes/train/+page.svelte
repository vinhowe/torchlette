<script lang="ts">
import { onMount } from "svelte";
import { page } from "$app/state";
import { Torchlette, initWebGPU, getWebGPUInitError } from "torchlette";
import { session } from "$lib/hf/session.svelte";
import { fetchModelConfig } from "$lib/hf/repo";
import type { ModelConfig, Checkpoint } from "$lib/hf/types";
import { fetchTokenizer, createTokenizer, type GPT2Tokenizer, type GPT2WithLoRA } from "gpt2-browser";
import { createModel, loadWeightsFromRepo, serializeModel } from "$lib/model/engine";
import { trainModel, type TrainResult } from "$lib/train";
import { DATASETS, loadDatasetSlice } from "$lib/data";
import type { HfDatasetDiet } from "$lib/hf/types";
import { snapshotModel } from "$lib/snapshot";

type Phase = "loading" | "ready" | "training" | "trained" | "snapshotting" | "done" | "error";

let repo = $state<string | null>(null);
let phase = $state<Phase>("loading");
let log = $state<string[]>([]);
let errorMsg = $state<string | null>(null);

let config = $state<ModelConfig | null>(null);
let loadPct = $state(0); // 0..1 during weight download + parse
let loadStatus = $state("");
let api: Torchlette | null = null;
let model: GPT2WithLoRA | null = null;
let tokenizer: GPT2Tokenizer | null = null;

// training controls
let datasetIdx = $state(0);
let offset = $state(0);
let rows = $state(50);
let steps = $state(20);
let batchSize = $state(1);
let seqLength = $state(128);
let lr = $state(3e-4);
let weightDecay = $state(0.1);

let lossHistory = $state<number[]>([]);
let curStep = $state(0);
let diet = $state<HfDatasetDiet | null>(null);
let result = $state<TrainResult | null>(null);
let stopRequested = false;

function say(s: string) {
  log = [...log, s];
}

onMount(async () => {
  repo = page.url.searchParams.get("repo");
  if (!repo) {
    phase = "error";
    errorMsg = "No ?repo= specified.";
    return;
  }
  try {
    say("Initializing WebGPU…");
    const ok = await initWebGPU();
    if (!ok) throw new Error(getWebGPUInitError() || "WebGPU init failed");

    say(`Loading config for ${repo}…`);
    config = await fetchModelConfig(repo);
    if (!config) throw new Error("Repo has no config.json — not a Menagerie model.");
    seqLength = Math.min(seqLength, config.blockSize);

    api = new Torchlette("webgpu", { enableFusion: true });
    api.manualSeed(42);

    say("Fetching tokenizer (gpt2)…");
    const { vocab, merges } = await fetchTokenizer();
    tokenizer = createTokenizer(vocab, merges);

    say(`Downloading + parsing weights (this can take a while for large models)…`);
    const weights = await loadWeightsFromRepo(repo, "main", (loaded, total, status) => {
      loadPct = total ? loaded / total : 0;
      loadStatus = status;
    });
    loadStatus = "";
    model = createModel(api, config, weights);

    say("Ready to train.");
    phase = "ready";
  } catch (e) {
    phase = "error";
    errorMsg = (e as Error).message;
    say(`Error: ${errorMsg}`);
  }
});

async function run() {
  if (!api || !model || !tokenizer || !config) return;
  stopRequested = false;
  lossHistory = [];
  curStep = 0;
  result = null;
  phase = "training";
  try {
    const choice = DATASETS[datasetIdx];
    say(`Loading ${rows} rows of ${choice.id} @${offset}…`);
    const slice = await loadDatasetSlice(choice, offset, rows, (t) => tokenizer!.encode(t));
    diet = slice.diet;
    say(`Tokenized ${slice.tokens.length.toLocaleString()} tokens (${slice.charCount.toLocaleString()} chars).`);

    say(`Training ${steps} steps (batch ${batchSize} × seq ${seqLength}, lr ${lr})…`);
    result = await trainModel({
      api,
      model,
      tokens: slice.tokens,
      config: { steps, batchSize, seqLength, lr, weightDecay },
      shouldStop: () => stopRequested,
      onStep: ({ step, loss }) => {
        curStep = step + 1;
        lossHistory = [...lossHistory, loss];
      },
    });
    say(`Done. final loss ${result.finalLoss.toFixed(4)}, ${(result.wallclockMs / 1000).toFixed(1)}s.`);
    phase = "trained";
  } catch (e) {
    phase = "error";
    errorMsg = (e as Error).message;
    say(`Error: ${errorMsg}`);
  }
}

async function snapshot() {
  if (!model || !repo || !result || !diet || !config) return;
  if (!session.loggedIn || !session.username || !session.sub || !session.accessToken) {
    errorMsg = "Log in to snapshot.";
    return;
  }
  phase = "snapshotting";
  try {
    say("Serializing weights…");
    const weights = await serializeModel(model);
    say(`Committing snapshot (${(weights.size / 1e6).toFixed(1)} MB) + lineage…`);
    const checkpoint: Checkpoint = {
      op: "train",
      created_by: session.username,
      created_by_sub: session.sub,
      created_at: new Date().toISOString(),
      steps: result.steps,
      tokens: result.tokensSeen,
      wallclock_ms: Math.round(result.wallclockMs),
      dtype: config.dtype,
      hparams: { lr, weightDecay, batch: batchSize, seq: seqLength, optimizer: "adamw" },
      diet: [diet],
      client: { ua: navigator.userAgent, tier_used: config.arch_preset },
    };
    await snapshotModel({ repo, accessToken: session.accessToken, weights, checkpoint });
    say("Snapshot committed.");
    phase = "done";
  } catch (e) {
    phase = "error";
    errorMsg = (e as Error).message;
    say(`Error: ${errorMsg}`);
  }
}

// sparkline path for the loss history
const spark = $derived.by(() => {
  const h = lossHistory;
  if (h.length < 2) return "";
  const min = Math.min(...h), max = Math.max(...h), range = max - min || 1;
  const W = 320, H = 60;
  return h
    .map((v, i) => {
      const x = (i / (h.length - 1)) * W;
      const y = H - ((v - min) / range) * H;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
});
</script>

<main class="mx-auto max-w-3xl px-6 py-10">
  <a href={`/model?repo=${encodeURIComponent(repo ?? "")}`} class="text-sm text-slate-500 hover:text-slate-300">← model</a>
  <h1 class="mt-2 font-mono text-2xl font-semibold">Train {repo}</h1>

  {#if config}
    <p class="mt-1 text-sm text-slate-400">
      {config.arch_preset} · {config.numLayers}L · {config.embedDim}d · ctx {config.blockSize}
    </p>
  {/if}

  {#if phase === "loading"}
    <section class="mt-6">
      <div class="mb-1 flex justify-between text-sm text-slate-400">
        <span>{loadStatus || "Loading…"}</span>
        <span class="font-mono">{Math.round(loadPct * 100)}%</span>
      </div>
      <div class="h-2 w-full overflow-hidden rounded bg-slate-800">
        <div
          class="h-full bg-emerald-500 transition-all"
          style={`width:${Math.round(loadPct * 100)}%`}
        ></div>
      </div>
    </section>
  {/if}

  {#if phase !== "loading" && phase !== "error"}
    <section class="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3">
      <label class="text-sm">Dataset
        <select bind:value={datasetIdx} disabled={phase === "training"} class="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1">
          {#each DATASETS as d, i}<option value={i}>{d.label}</option>{/each}
        </select>
      </label>
      <label class="text-sm">Rows
        <input type="number" bind:value={rows} min="1" disabled={phase === "training"} class="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1" />
      </label>
      <label class="text-sm">Row offset
        <input type="number" bind:value={offset} min="0" disabled={phase === "training"} class="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1" />
      </label>
      <label class="text-sm">Steps
        <input type="number" bind:value={steps} min="1" disabled={phase === "training"} class="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1" />
      </label>
      <label class="text-sm">Batch
        <input type="number" bind:value={batchSize} min="1" disabled={phase === "training"} class="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1" />
      </label>
      <label class="text-sm">Seq len
        <input type="number" bind:value={seqLength} min="1" max={config?.blockSize ?? 1024} disabled={phase === "training"} class="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1" />
      </label>
      <label class="text-sm">LR
        <input type="number" bind:value={lr} step="0.0001" disabled={phase === "training"} class="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1" />
      </label>
      <label class="text-sm">Weight decay
        <input type="number" bind:value={weightDecay} step="0.01" disabled={phase === "training"} class="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1" />
      </label>
    </section>

    <div class="mt-4 flex gap-3">
      {#if phase === "training"}
        <button class="rounded-lg bg-rose-600 px-4 py-2 text-sm font-medium" onclick={() => (stopRequested = true)}>Stop</button>
      {:else}
        <button class="rounded-lg bg-emerald-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-emerald-400 disabled:opacity-50" onclick={run} disabled={phase === "snapshotting"}>Train</button>
      {/if}
      {#if (phase === "trained" || phase === "done")}
        <button class="rounded-lg bg-amber-500 px-4 py-2 text-sm font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50" onclick={snapshot} disabled={!session.loggedIn}>
          Snapshot to HF
        </button>
      {/if}
      {#if phase === "done"}
        <a class="rounded-lg border border-slate-700 px-4 py-2 text-sm hover:bg-slate-800" href={`/model?repo=${encodeURIComponent(repo ?? "")}`}>View model →</a>
      {/if}
    </div>
  {/if}

  {#if lossHistory.length}
    <section class="mt-6">
      <div class="flex items-baseline justify-between text-sm">
        <span class="text-slate-400">loss</span>
        <span class="font-mono">step {curStep}/{steps} · {lossHistory[lossHistory.length - 1].toFixed(4)}</span>
      </div>
      <svg viewBox="0 0 320 60" class="mt-1 h-16 w-full">
        <path d={spark} fill="none" stroke="#34d399" stroke-width="1.5" />
      </svg>
    </section>
  {/if}

  <section class="mt-6 rounded-lg border border-slate-800 bg-slate-900/40 p-3">
    <pre class="max-h-60 overflow-auto whitespace-pre-wrap text-xs text-slate-400">{log.join("\n")}</pre>
  </section>

  {#if errorMsg}
    <p class="mt-3 text-sm text-rose-400">{errorMsg}</p>
  {/if}
</main>
