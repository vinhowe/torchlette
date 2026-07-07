<script lang="ts">
  import ActionButton from "$lib/components/buttons/ActionButton.svelte";
  import CapacityBar from "$lib/components/data/CapacityBar.svelte";
  import FormLabel from "$lib/components/controls/FormLabel.svelte";
  import Note from "$lib/components/feedback/Note.svelte";
  import Panel from "$lib/components/primitives/Panel.svelte";
  import Statistic from "$lib/components/feedback/Statistic.svelte";
  import ThemeToggle from "$lib/components/theme/ThemeToggle.svelte";
  import NetworkViz from "$lib/NetworkViz.svelte";
  import {
    createSteeringEngine,
    LOCAL_MODELS,
    type GenEvent,
    type GenStats,
    type ModelInfo,
    type SteeringEngine,
    type TensorLoadEvent,
    type VectorInfo,
  } from "$lib/local-engine";
  import { STEERING_PRESETS } from "$lib/steering";

  // ---- Model load state ----
  let modelId = $state(LOCAL_MODELS[1].id); // 1.7B — the Golden Gate defaults are tuned for it
  let engine = $state<SteeringEngine | null>(null);
  let info = $state<ModelInfo | null>(null);
  let loading = $state(false);
  let loadPct = $state(0);
  let loadStatus = $state("");
  let loadBytes = $state(0);
  let loadTotal = $state(0);
  let error = $state<string | null>(null);

  let netTensors = $state<
    { name: string; shape: number[]; elems: number; dtype: string; skipped: boolean }[]
  >([]);
  let netFill = $state<Record<string, number>>({});
  function onTensorEvent(ev: TensorLoadEvent) {
    if (ev.type === "manifest") netTensors.push(...ev.tensors);
    else if (ev.type === "start") netFill[ev.name] = 0;
    else if (ev.type === "progress") netFill[ev.name] = ev.fraction;
    else if (ev.type === "done") netFill[ev.name] = 1;
  }

  async function loadModel() {
    loading = true;
    error = null;
    netTensors = [];
    netFill = {};
    try {
      engine = await createSteeringEngine(
        modelId,
        (loaded, total, status) => {
          loadBytes = loaded;
          loadTotal = total;
          loadPct = total > 0 ? Math.min(100, (loaded / total) * 100) : 0;
          loadStatus = status;
        },
        onTensorEvent,
        (e) => (error = e),
      );
      info = engine.info;
      // Default the layer slider to the middle of the stack.
      // ~0.57 depth (layer 16 of 28): the coherent steering band on Qwen3 —
      // earlier layers break, later ones barely steer. Tuned via layer sweep.
      layer = Math.round(info.numLayers * 0.57);
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  // ---- Steering-vector state ----
  let posPrompt = $state(STEERING_PRESETS[0].pos);
  let negPrompt = $state(STEERING_PRESETS[0].neg);
  let layer = $state(6);
  let computing = $state(false);
  let vector = $state<VectorInfo | null>(null);

  function applyPreset(p: { pos: string; neg: string }) {
    posPrompt = p.pos;
    negPrompt = p.neg;
  }

  async function computeVector() {
    if (!engine || computing) return;
    computing = true;
    error = null;
    try {
      vector = await engine.computeVector(posPrompt.trim(), negPrompt.trim(), layer);
    } catch (e) {
      error = String(e);
    } finally {
      computing = false;
    }
  }

  // ---- Generation state ----
  let prompt = $state("What should I do this weekend?");
  let alpha = $state(3);
  let maxNewTokens = $state(80);
  let busy = $state(false);

  type Run = {
    label: string;
    alpha: number;
    text: string;
    stats: GenStats | null;
  };
  let steeredRun = $state<Run | null>(null);
  let baselineRun = $state<Run | null>(null);

  async function runOne(target: "steered" | "baseline", a: number) {
    const run: Run = {
      label: target === "baseline" ? "baseline (α = 0)" : `steered (α = ${a})`,
      alpha: a,
      text: "",
      stats: null,
    };
    if (target === "baseline") baselineRun = run;
    else steeredRun = run;
    await engine!.generate(prompt, a, maxNewTokens, (e: GenEvent) => {
      if ("delta" in e) run.text += e.delta;
      else if ("replace" in e) run.text = e.replace;
      else if ("error" in e) error = e.error;
      else if ("done" in e) run.stats = e.stats;
      // Reassign to trigger Svelte 5 reactivity on the nested object.
      if (target === "baseline") baselineRun = { ...run };
      else steeredRun = { ...run };
    });
  }

  async function generateSteered() {
    if (!engine || busy) return;
    busy = true;
    error = null;
    try {
      await runOne("steered", alpha);
    } catch (e) {
      error = String(e);
    } finally {
      busy = false;
    }
  }

  async function generateBoth() {
    if (!engine || busy) return;
    busy = true;
    error = null;
    try {
      await runOne("baseline", 0);
      await runOne("steered", alpha);
    } catch (e) {
      error = String(e);
    } finally {
      busy = false;
    }
  }

  const alphaTone = $derived(alpha > 0 ? "text-primary" : alpha < 0 ? "text-destructive" : "text-muted-foreground");
</script>

<div class="fixed inset-0 flex flex-col overflow-hidden bg-background text-foreground">
  <!-- Purple project-header motif -->
  <header
    class="flex shrink-0 items-center justify-between border-b border-purple-300 bg-purple-200 px-2 py-1 text-purple-900 dark:border-purple-900 dark:bg-purple-950 dark:text-purple-200"
  >
    <div class="font-mono text-xs uppercase tracking-wider">
      qwen3-steering · contrastive activation steering · torchlette
    </div>
    <div class="flex items-center gap-2">
      {#if info}
        <span class="type-value text-success">{info.modelId} · {info.weightDtype} · resident</span>
      {/if}
      <ThemeToggle />
    </div>
  </header>

  <main class="min-h-0 flex-1 overflow-y-auto overscroll-none px-2 py-2">
    <div class="mx-auto flex max-w-5xl flex-col stack-section">
      <!-- ============ MODEL LOAD ============ -->
      {#if !engine}
        <Panel title="Model">
          <p class="type-caption text-muted-foreground">
            Steer Qwen3's generation in your browser by adding a contrastively-derived
            activation vector to the residual stream. Everything runs on your GPU via WebGPU —
            weights stream from Hugging Face into this tab (cached in IndexedDB for next time).
          </p>
          <div class="flex flex-wrap items-end gap-2">
            <div class="stack-tight">
              <FormLabel value="Model" />
              <div class="inline-flex items-stretch border border-border bg-card">
                {#each LOCAL_MODELS as mo, i}
                  <button
                    class="{i > 0 ? 'border-l border-border ' : ''}px-1.5 py-0.5 type-button {modelId === mo.id ? 'bg-primary/12 text-primary' : 'text-muted-foreground hover:text-foreground'}"
                    disabled={loading}
                    onclick={() => (modelId = mo.id)}>{mo.label}</button
                  >
                {/each}
              </div>
            </div>
            <ActionButton color="green" disabled={loading} onclick={loadModel}>
              {loading ? "Loading…" : "Load model"}
            </ActionButton>
          </div>

          {#if loading}
            <CapacityBar
              id="model-download"
              label="Weights"
              unit="MB"
              max={Math.max(1, Math.round(loadTotal / 1e6))}
              segments={[{ label: "downloaded", value: Math.round(loadBytes / 1e6) }]}
            />
            <Note label="Status" type="info">{loadStatus} ({loadPct.toFixed(0)}%)</Note>
          {/if}
        </Panel>
        {#if netTensors.length > 0}
          <NetworkViz tensors={netTensors} fill={netFill} />
        {/if}
      {/if}

      {#if engine && info}
        <!-- ============ STEERING VECTOR ============ -->
        <Panel title="Steering vector">
          <p class="type-caption text-muted-foreground">
            Contrast a positive and negative concept. The direction is
            <span class="type-code">normalize(mean(pos) − mean(neg))</span> of the
            residual stream at layer L, mean-pooled over positions.
          </p>

          <div class="stack-tight">
            <FormLabel value="Presets" />
            <div class="flex flex-wrap gap-1">
              {#each STEERING_PRESETS as p}
                <button
                  class="border border-border bg-card px-1.5 py-0.5 type-button text-muted-foreground hover:bg-muted hover:text-foreground"
                  onclick={() => applyPreset(p)}>{p.name}</button
                >
              {/each}
            </div>
          </div>

          <div class="grid gap-2 md:grid-cols-2">
            <div class="stack-tight">
              <FormLabel value="Positive concept" />
              <textarea
                bind:value={posPrompt}
                rows="2"
                class="w-full resize-none border border-border bg-card p-1 type-body text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring"
              ></textarea>
            </div>
            <div class="stack-tight">
              <FormLabel value="Negative concept" />
              <textarea
                bind:value={negPrompt}
                rows="2"
                class="w-full resize-none border border-border bg-card p-1 type-body text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring"
              ></textarea>
            </div>
          </div>

          <div class="flex flex-wrap items-end gap-3">
            <div class="stack-tight grow">
              <FormLabel value={`Layer L — ${layer} / ${info.numLayers - 1}`} />
              <input
                type="range"
                min="0"
                max={info.numLayers - 1}
                step="1"
                bind:value={layer}
                class="w-full accent-[var(--primary)]"
              />
            </div>
            <ActionButton color="purple" disabled={computing || busy} onclick={computeVector}>
              {computing ? "Computing…" : "Compute vector"}
            </ActionButton>
          </div>

          {#if vector}
            <Note label="Vector ready" type="info">
              [{vector.hiddenSize}] contrast direction @ layer {vector.layer}.
              Positive α steers toward “{vector.posPrompt.split("\n")[0].slice(0, 36)}…”, negative toward
              “{vector.negPrompt.split("\n")[0].slice(0, 36)}…”.
            </Note>
          {/if}
        </Panel>

        <!-- ============ GENERATION ============ -->
        <Panel title="Generate">
          <div class="stack-tight">
            <FormLabel value="Prompt" />
            <textarea
              bind:value={prompt}
              rows="2"
              class="w-full resize-none border border-border bg-card p-1 type-body text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring"
            ></textarea>
          </div>

          <div class="flex flex-wrap items-end gap-3">
            <div class="stack-tight grow">
              <FormLabel value="Steering strength α" />
              <div class="flex items-center gap-2">
                <input
                  type="range"
                  min="-5"
                  max="5"
                  step="0.25"
                  bind:value={alpha}
                  class="w-full accent-[var(--primary)]"
                />
                <span class="w-10 shrink-0 text-right type-value {alphaTone}">{alpha}</span>
              </div>
            </div>
            <div class="stack-tight">
              <FormLabel value="Max tokens" />
              <input
                type="number"
                min="8"
                max="256"
                bind:value={maxNewTokens}
                class="w-20 border border-border bg-card px-1 py-0.5 type-value text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
            </div>
          </div>

          <div class="flex flex-wrap gap-2">
            <ActionButton
              color="blue"
              disabled={busy || !vector || !prompt.trim()}
              onclick={generateSteered}
            >
              {busy ? "Generating…" : "Generate (steered)"}
            </ActionButton>
            <ActionButton
              color="gray"
              disabled={busy || !vector || !prompt.trim()}
              onclick={generateBoth}
            >
              Compare baseline vs steered
            </ActionButton>
            {#if !vector}
              <span class="self-center type-caption text-muted-foreground">
                Compute a steering vector first.
              </span>
            {/if}
          </div>

          <div class="grid gap-2 md:grid-cols-2">
            {#if baselineRun}
              {@render runCard(baselineRun)}
            {/if}
            {#if steeredRun}
              {@render runCard(steeredRun)}
            {/if}
          </div>
        </Panel>
      {/if}

      {#if error}
        <Note label="Error" type="error">{error}</Note>
      {/if}
    </div>
  </main>
</div>

{#snippet runCard(run: Run)}
  <div class="border border-border bg-card">
    <div class="flex items-center justify-between border-b border-border bg-panel px-1 py-0.5">
      <span class="type-label {run.alpha > 0 ? 'text-primary' : run.alpha < 0 ? 'text-destructive' : 'text-subtle-foreground'}">{run.label}</span>
      {#if run.stats}
        <span class="type-value text-muted-foreground">{run.stats.tokPerSec} tok/s</span>
      {/if}
    </div>
    <p class="whitespace-pre-wrap p-1.5 type-body">
      {run.text}{#if busy && !run.stats}<span class="text-primary">▌</span>{/if}
    </p>
    {#if run.stats}
      <div class="flex flex-wrap gap-2 border-t border-border p-1.5">
        <Statistic label="tokens">{run.stats.promptTokens}+{run.stats.newTokens}</Statistic>
        <Statistic label="prefill">{run.stats.prefillMs}ms</Statistic>
        <Statistic label="steered">{run.stats.steered ? "yes" : "no"}</Statistic>
        {#if run.stats.tape}
          <Statistic label="tape">{run.stats.tape.hits}/{run.stats.tape.replays} hits</Statistic>
        {/if}
        {#if run.stats.decodeBreakdown}
          {@const d = run.stats.decodeBreakdown}
          <Statistic label="ms/tok build·lower·fence·sample·step"
            >{d.buildMs}·{d.lowerMs}·{d.fenceMs}·{d.sampleMs}·{d.stepMs}</Statistic>
        {/if}
      </div>
    {/if}
  </div>
{/snippet}
