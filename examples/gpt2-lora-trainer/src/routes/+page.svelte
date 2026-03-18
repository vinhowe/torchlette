<script lang="ts">
import { onMount } from "svelte";
import { generationStore } from "$lib/stores/generation.svelte";
import { modelStore } from "$lib/stores/model.svelte";
import { trainingStore } from "$lib/stores/training.svelte";

// Fetch Shakespeare data on mount
onMount(() => {
  trainingStore.fetchShakespeare();
});

// --- Helpers ---
function fmtLoss(v: number): string {
  return v === 0 ? "--" : v.toFixed(3);
}
function fmtTokSec(v: number): string {
  if (v === 0) return "--";
  return v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0);
}
function fmtLr(v: number): string {
  if (v >= 0.01) return v.toFixed(3);
  return v.toExponential(0);
}
function fmtTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`;
  return String(n);
}

// --- Loss chart ---
function lossChartPath(history: number[]): string {
  if (history.length < 2) return "";
  const maxLoss = Math.max(...history);
  const minLoss = Math.min(...history);
  const range = maxLoss - minLoss || 1;
  const w = 100;
  const h = 100;
  const pts = history.map((l, i) => {
    const x = (i / (history.length - 1)) * w;
    const y = h - ((l - minLoss) / range) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return `M${pts.join(" L")}`;
}

// --- File drop ---
let dragging = $state(false);

function onDragOver(e: DragEvent) {
  e.preventDefault();
  dragging = true;
}
function onDragLeave() {
  dragging = false;
}
function onDrop(e: DragEvent) {
  e.preventDefault();
  dragging = false;
  const file = e.dataTransfer?.files?.[0];
  if (file && file.name.endsWith(".txt")) {
    trainingStore.handleFileDrop(file);
  }
}

// --- Slider bind helpers (range inputs give strings) ---
function setRank(e: Event) {
  trainingStore.rank = +(e.target as HTMLInputElement).value;
}
function setAlpha(e: Event) {
  trainingStore.alpha = +(e.target as HTMLInputElement).value;
}
function setLr(e: Event) {
  const v = +(e.target as HTMLInputElement).value;
  // Map 0-100 to log scale: 1e-5 to 1e-2
  trainingStore.lr = 10 ** (-5 + (v / 100) * 3);
}
function lrToSlider(lr: number): number {
  return ((Math.log10(lr) + 5) / 3) * 100;
}
function setSteps(e: Event) {
  trainingStore.maxSteps = +(e.target as HTMLInputElement).value;
}
function setBatch(e: Event) {
  trainingStore.batchSize = +(e.target as HTMLInputElement).value;
}
function setSeqLen(e: Event) {
  trainingStore.seqLen = +(e.target as HTMLInputElement).value;
}
function setTemp(e: Event) {
  generationStore.temperature = +(e.target as HTMLInputElement).value / 100;
}
function setTopK(e: Event) {
  generationStore.topK = +(e.target as HTMLInputElement).value;
}

function handleLoadModel() {
  modelStore.load(trainingStore.rank, trainingStore.alpha);
}
</script>

<div class="max-w-5xl mx-auto p-2 font-sans">
  <!-- HEADER -->
  <header class="flex items-center gap-3 px-2 py-1.5 border-b border-slate-800">
    <span class="text-sm font-semibold text-slate-200 tracking-tight">GPT-2 LoRA</span>

    {#if modelStore.status === 'ready'}
      <span class="flex items-center gap-1 text-xs text-green-400">
        <span class="inline-block w-1.5 h-1.5 bg-green-400 rounded-full"></span>
        Ready
      </span>
    {:else if modelStore.status === 'loading'}
      <span class="flex items-center gap-1 text-xs text-amber-400">
        <span class="inline-block w-1.5 h-1.5 bg-amber-400 rounded-full animate-pulse"></span>
        {modelStore.progressText}
      </span>
    {:else if modelStore.status === 'error'}
      <span class="text-xs text-red-400">{modelStore.error}</span>
    {:else}
      <span class="text-xs text-slate-500">Not loaded</span>
    {/if}

    <div class="flex-1"></div>

    {#if modelStore.status === 'idle' || modelStore.status === 'error'}
      <button
        onclick={handleLoadModel}
        class="px-2 py-0.5 text-xs bg-blue-600 hover:bg-blue-500 text-white"
      >Load Model</button>
    {/if}

    {#if modelStore.status === 'loading'}
      <div class="w-32 h-1 bg-slate-800">
        <div class="h-full bg-blue-500 transition-all" style="width:{modelStore.progress}%"></div>
      </div>
    {/if}

    <span class="text-xs text-slate-600 font-mono">distilgpt2</span>
  </header>

  <!-- MAIN GRID -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-0 border-b border-slate-800">
    <!-- LEFT: TRAINING -->
    <div class="border-r border-slate-800 p-2 flex flex-col gap-2">
      <div class="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Training</div>

      <!-- Config sliders -->
      <div class="grid grid-cols-[auto_1fr_auto] gap-x-2 gap-y-1 items-center text-xs">
        <label for="sl-rank" class="text-slate-400">rank</label>
        <input id="sl-rank" type="range" min="1" max="64" value={trainingStore.rank} oninput={setRank}
               class="w-full" disabled={trainingStore.running} />
        <span class="font-mono text-slate-300 w-8 text-right">{trainingStore.rank}</span>

        <label for="sl-alpha" class="text-slate-400">alpha</label>
        <input id="sl-alpha" type="range" min="1" max="64" value={trainingStore.alpha} oninput={setAlpha}
               class="w-full" disabled={trainingStore.running} />
        <span class="font-mono text-slate-300 w-8 text-right">{trainingStore.alpha}</span>

        <label for="sl-lr" class="text-slate-400">lr</label>
        <input id="sl-lr" type="range" min="0" max="100" value={lrToSlider(trainingStore.lr)} oninput={setLr}
               class="w-full" disabled={trainingStore.running} />
        <span class="font-mono text-slate-300 w-12 text-right">{fmtLr(trainingStore.lr)}</span>

        <label for="sl-steps" class="text-slate-400">steps</label>
        <input id="sl-steps" type="range" min="10" max="1000" step="10" value={trainingStore.maxSteps} oninput={setSteps}
               class="w-full" disabled={trainingStore.running} />
        <span class="font-mono text-slate-300 w-8 text-right">{trainingStore.maxSteps}</span>

        <label for="sl-batch" class="text-slate-400">batch</label>
        <input id="sl-batch" type="range" min="1" max="16" value={trainingStore.batchSize} oninput={setBatch}
               class="w-full" disabled={trainingStore.running} />
        <span class="font-mono text-slate-300 w-8 text-right">{trainingStore.batchSize}</span>

        <label for="sl-seqlen" class="text-slate-400">seqlen</label>
        <input id="sl-seqlen" type="range" min="16" max="256" step="16" value={trainingStore.seqLen} oninput={setSeqLen}
               class="w-full" disabled={trainingStore.running} />
        <span class="font-mono text-slate-300 w-8 text-right">{trainingStore.seqLen}</span>
      </div>

      <!-- Checkboxes -->
      <div class="flex gap-4 text-xs text-slate-400">
        <label class="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={trainingStore.useAMP}
                 onchange={() => { trainingStore.useAMP = !trainingStore.useAMP; }}
                 disabled={trainingStore.running}
                 class="accent-blue-500" />
          AMP
        </label>
        <label class="flex items-center gap-1 cursor-pointer">
          <input type="checkbox" checked={trainingStore.useCheckpointing}
                 onchange={() => { trainingStore.useCheckpointing = !trainingStore.useCheckpointing; }}
                 disabled={trainingStore.running}
                 class="accent-blue-500" />
          Ckpt
        </label>
      </div>

      <!-- Train / Stop buttons -->
      <div class="flex gap-2 items-center">
        {#if !trainingStore.running}
          <button
            onclick={() => trainingStore.startTraining()}
            disabled={!trainingStore.canTrain}
            class="px-2 py-0.5 text-xs bg-green-700 hover:bg-green-600 text-white disabled:opacity-30 disabled:cursor-not-allowed"
          >&#9654; Train</button>
        {:else}
          <button
            onclick={() => trainingStore.stopTraining()}
            class="px-2 py-0.5 text-xs bg-red-700 hover:bg-red-600 text-white"
          >&#9632; Stop</button>
        {/if}

        <!-- Step / Loss / Tok/s -->
        <div class="font-mono text-xs text-slate-400 flex gap-3">
          {#if trainingStore.running || trainingStore.lossHistory.length > 0}
            <span>step <span class="text-slate-200">{trainingStore.step}</span>/{trainingStore.maxSteps}</span>
            <span>loss <span class="text-slate-200">{fmtLoss(trainingStore.loss)}</span></span>
            <span><span class="text-slate-200">{fmtTokSec(trainingStore.tokPerSec)}</span> tok/s</span>
          {/if}
        </div>
      </div>

      <!-- Error -->
      {#if trainingStore.error}
        <div class="text-xs text-red-400 font-mono">{trainingStore.error}</div>
      {/if}

      <!-- Loss chart -->
      {#if trainingStore.lossHistory.length >= 2}
        <div class="border border-slate-800 p-1">
          <svg viewBox="-2 -2 104 104" class="w-full h-24" preserveAspectRatio="none">
            <path d={lossChartPath(trainingStore.lossHistory)}
                  fill="none" stroke="#3b82f6" stroke-width="1.5" vector-effect="non-scaling-stroke" />
          </svg>
          <div class="flex justify-between text-[9px] font-mono text-slate-600 px-0.5">
            <span>0</span>
            <span>loss</span>
            <span>{trainingStore.maxSteps}</span>
          </div>
        </div>
      {/if}

      <!-- Data info + drop zone -->
      <div
        class="border border-dashed text-xs px-2 py-1.5 {dragging ? 'border-blue-500 bg-blue-950/30' : 'border-slate-700'}"
        role="region"
        aria-label="File drop zone"
        ondragover={onDragOver}
        ondragleave={onDragLeave}
        ondrop={onDrop}
      >
        {#if trainingStore.dataLoading}
          <span class="text-amber-400">Fetching TinyShakespeare...</span>
        {:else if trainingStore.dataSource}
          <span class="text-slate-400">data:</span>
          <span class="text-slate-300 font-mono">{trainingStore.dataSource}</span>
          <span class="text-slate-500 font-mono ml-1">{fmtTokens(trainingStore.dataTokenCount)} tok</span>
          <span class="text-slate-600 ml-2">drop .txt to replace</span>
        {:else}
          <span class="text-slate-500">Drop a .txt file here for training data</span>
        {/if}
      </div>
    </div>

    <!-- RIGHT: GENERATE -->
    <div class="p-2 flex flex-col gap-2">
      <div class="text-[10px] uppercase tracking-widest text-slate-500 font-semibold">Generate</div>

      <!-- Prompt -->
      <textarea
        class="w-full bg-slate-900 border border-slate-700 text-sm text-slate-200 font-mono p-1.5 resize-none focus:outline-none focus:border-blue-600"
        rows="2"
        placeholder="Enter prompt..."
        value={generationStore.prompt}
        oninput={(e) => { generationStore.prompt = (e.target as HTMLTextAreaElement).value; }}
        disabled={generationStore.isGenerating}
      ></textarea>

      <!-- Generate button + sampling params -->
      <div class="flex items-center gap-3">
        {#if generationStore.isGenerating}
          <button
            onclick={() => generationStore.stopGenerate()}
            class="px-2 py-0.5 text-xs bg-red-600 hover:bg-red-500 text-white"
          >■ Stop</button>
        {:else}
          <button
            onclick={() => generationStore.generate()}
            disabled={!generationStore.canGenerate}
            class="px-2 py-0.5 text-xs bg-blue-600 hover:bg-blue-500 text-white disabled:opacity-30 disabled:cursor-not-allowed"
          >Generate</button>
        {/if}

        <div class="flex items-center gap-2 text-xs text-slate-500">
          <label class="flex items-center gap-1">
            temp
            <input type="range" min="1" max="200" value={Math.round(generationStore.temperature * 100)}
                   oninput={setTemp} class="w-16" />
            <span class="font-mono text-slate-400 w-6">{generationStore.temperature.toFixed(1)}</span>
          </label>
          <label class="flex items-center gap-1">
            k
            <input type="range" min="1" max="100" value={generationStore.topK}
                   oninput={setTopK} class="w-12" />
            <span class="font-mono text-slate-400 w-5">{generationStore.topK}</span>
          </label>
        </div>
      </div>

      <!-- Output -->
      <div class="flex-1 min-h-[200px] border border-slate-800 p-2 font-mono text-xs text-slate-300 overflow-y-auto whitespace-pre-wrap">
        {#if generationStore.isGenerating}
          <span class="text-slate-400">{generationStore.prompt}</span>{generationStore.output}<span class="animate-pulse text-blue-400">|</span>
        {:else if generationStore.output}
          <span class="text-slate-400">{generationStore.prompt}</span>{generationStore.output}
        {:else}
          <span class="text-slate-600 italic">Output will appear here...</span>
        {/if}
      </div>

      <!-- Error -->
      {#if generationStore.error}
        <div class="text-xs text-red-400 font-mono">{generationStore.error}</div>
      {/if}

      <!-- Download LoRA -->
      {#if trainingStore.loraBlob}
        <button
          onclick={() => trainingStore.downloadLoRA()}
          class="self-end px-2 py-0.5 text-xs bg-slate-700 hover:bg-slate-600 text-slate-300"
        >&#8595; Download LoRA</button>
      {/if}
    </div>
  </div>
</div>
