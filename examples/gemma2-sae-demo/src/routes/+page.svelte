<script lang="ts">
  import ActionButton from "$lib/components/buttons/ActionButton.svelte";
  import FormLabel from "$lib/components/controls/FormLabel.svelte";
  import Note from "$lib/components/feedback/Note.svelte";
  import Panel from "$lib/components/primitives/Panel.svelte";
  import Statistic from "$lib/components/feedback/Statistic.svelte";
  import ThemeToggle from "$lib/components/theme/ThemeToggle.svelte";
  import NetworkViz from "$lib/NetworkViz.svelte";
  import {
    createSAEEngine,
    MODEL,
    type FeatureReport,
    type GenEvent,
    type GenStats,
    type ModelInfo,
    type SAEEngine,
    type SteerSpec,
    type TensorLoadEvent,
  } from "$lib/local-engine";
  import { PRESETS, DEFAULT_PROMPT, DEFAULT_ALPHA_RANGE } from "$lib/presets";

  // The SAE .bin files are served as static assets (see static/sae/).
  const SAE_BASE_URL = `${import.meta.env.BASE_URL}sae`;

  // Optional quantization opt-in (default OFF this wave): ?weightFormat=int8-64.
  // A URL/config flag, NOT a TORCHLETTE_* env. When set, projection weights load
  // as packed-int operands (docs/quantization-design.md phase 2).
  const weightFormatParam =
    typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("weightFormat")
      : null;
  const WEIGHT_FORMAT =
    weightFormatParam === "int8-64" || weightFormatParam === "int8-128"
      ? (weightFormatParam as "int8-64" | "int8-128")
      : undefined;

  // ---- Model load state ----
  let engine = $state<SAEEngine | null>(null);
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
      engine = await createSAEEngine(
        MODEL.id,
        SAE_BASE_URL,
        (loaded, total, status) => {
          loadBytes = loaded;
          loadTotal = total;
          loadPct = total > 0 ? Math.min(100, (loaded / total) * 100) : 0;
          loadStatus = status;
        },
        onTensorEvent,
        (e) => (error = e),
        WEIGHT_FORMAT,
      );
      info = engine.info;
    } catch (e) {
      error = String(e);
    } finally {
      loading = false;
    }
  }

  // ---- Feature inspector state ----
  let inspectPrompt = $state("The Golden Gate Bridge stands over the bay.");
  let inspecting = $state(false);
  let report = $state<FeatureReport | null>(null);
  let inspectMode = $state<"agg" | "last">("agg");
  const shownHits = $derived(
    report ? (inspectMode === "agg" ? report.topAgg : report.topLast) : [],
  );

  async function inspect() {
    if (!engine || inspecting) return;
    inspecting = true;
    error = null;
    try {
      report = await engine.inspect(inspectPrompt.trim(), 24);
    } catch (e) {
      error = String(e);
    } finally {
      inspecting = false;
    }
  }

  // ---- Steering state ----
  // Active steering features: feature index + α. Multiple simultaneous.
  let steer = $state<SteerSpec[]>([]);
  let newFeature = $state<number>(0);

  function addFeature(feature: number, alpha = 120) {
    if (steer.some((s) => s.feature === feature)) return;
    steer = [...steer, { feature, alpha }];
  }
  function removeFeature(feature: number) {
    steer = steer.filter((s) => s.feature !== feature);
  }
  function setAlpha(feature: number, alpha: number) {
    steer = steer.map((s) => (s.feature === feature ? { ...s, alpha } : s));
  }
  function applyPreset(p: (typeof PRESETS)[number]) {
    steer = [{ feature: p.feature, alpha: p.alpha }];
    prompt = p.testPrompt;
  }
  function neuronpedia(feature: number) {
    return info
      ? `https://www.neuronpedia.org/gemma-2-2b/${info.neuronpediaSaeId}/${feature}`
      : "#";
  }

  // ---- Generation state ----
  let prompt = $state(DEFAULT_PROMPT);
  let maxNewTokens = $state(80);
  let temperature = $state(0.7);
  let busy = $state(false);

  type Run = { label: string; steered: boolean; text: string; stats: GenStats | null };
  let steeredRun = $state<Run | null>(null);
  let baselineRun = $state<Run | null>(null);

  async function runOne(target: "steered" | "baseline", specs: SteerSpec[]) {
    const run: Run = {
      label: target === "baseline" ? "baseline (no steering)" : "steered",
      steered: specs.length > 0,
      text: "",
      stats: null,
    };
    if (target === "baseline") baselineRun = run;
    else steeredRun = run;
    await engine!.generate(prompt, specs, maxNewTokens, temperature, (e: GenEvent) => {
      if ("delta" in e) run.text += e.delta;
      else if ("replace" in e) run.text = e.replace;
      else if ("error" in e) error = e.error;
      else if ("done" in e) run.stats = e.stats;
      if (target === "baseline") baselineRun = { ...run };
      else steeredRun = { ...run };
    });
  }

  async function generateSteered() {
    if (!engine || busy) return;
    busy = true;
    error = null;
    try {
      await runOne("steered", steer);
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
      await runOne("baseline", []);
      await runOne("steered", steer);
    } catch (e) {
      error = String(e);
    } finally {
      busy = false;
    }
  }
</script>

<div class="fixed inset-0 flex flex-col overflow-hidden bg-background text-foreground">
  <header
    class="flex shrink-0 items-center justify-between border-b border-purple-300 bg-purple-200 px-2 py-1 text-purple-900 dark:border-purple-900 dark:bg-purple-950 dark:text-purple-200"
  >
    <div class="font-mono text-xs uppercase tracking-wider">
      gemma-scope SAE steering · gemma-2-2b · layer 20 · torchlette
    </div>
    <div class="flex items-center gap-2">
      {#if info}
        <span class="type-value text-success">{info.modelId} · {info.weightDtype} · resident</span>
        <span class="type-value text-muted-foreground">{info.numFeatures} feats</span>
        <span class="type-value {info.tapeOn ? 'text-success' : 'text-destructive'}"
          >tape {info.tapeOn ? "on" : info.tapeFlagSet ? "flag set, const OFF" : "flag missing"}</span>
      {/if}
      <ThemeToggle />
    </div>
  </header>

  <main class="min-h-0 flex-1 overflow-y-auto overscroll-none px-2 py-2">
    <div class="mx-auto flex max-w-5xl flex-col stack-section">
      <!-- ============ MODEL LOAD ============ -->
      {#if !engine}
        <Panel title="Gemma-2-2B + Gemma Scope SAE">
          <p class="type-caption text-muted-foreground">
            Inspect and STEER Gemma-2-2B's internal features in your browser, using the
            Gemma Scope layer-20 residual-stream sparse autoencoder (16,384 JumpReLU
            features). Everything runs on your GPU via WebGPU — Gemma-2 weights (~5GB f16)
            stream from Hugging Face and cache in IndexedDB; the SAE (~0.3GB) loads from
            this app. Comfortable on a 16GB Mac.
          </p>
          <div class="flex flex-wrap items-end gap-2">
            <ActionButton color="green" disabled={loading} onclick={loadModel}>
              {loading ? "Loading…" : `Load ${MODEL.label} (~${MODEL.approxGB}GB)`}
            </ActionButton>
          </div>

          {#if loading}
            <div class="h-2 w-full overflow-hidden border border-border bg-card">
              <div class="h-full bg-primary transition-all" style="width: {loadPct}%"></div>
            </div>
            <Note label="Status" type="info">{loadStatus} ({loadPct.toFixed(0)}%)</Note>
          {/if}
        </Panel>
        {#if netTensors.length > 0}
          <NetworkViz tensors={netTensors} fill={netFill} />
        {/if}
      {/if}

      {#if engine && info}
        <!-- ============ FEATURE INSPECTOR ============ -->
        <Panel title="Feature inspector">
          <p class="type-caption text-muted-foreground">
            Run a prompt and see which SAE features fire at layer {info.saeLayer}. Each
            feature is a monosemantic direction; click a feature to steer with it, or open
            its Neuronpedia page for the human-readable interpretation.
          </p>
          <div class="stack-tight">
            <FormLabel value="Prompt" />
            <textarea
              bind:value={inspectPrompt}
              rows="2"
              class="w-full resize-none border border-border bg-card p-1 type-body text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring"
            ></textarea>
          </div>
          <div class="flex flex-wrap items-end gap-2">
            <ActionButton color="purple" disabled={inspecting || busy} onclick={inspect}>
              {inspecting ? "Encoding…" : "Inspect features"}
            </ActionButton>
            {#if report}
              <div class="inline-flex items-stretch border border-border bg-card">
                <button
                  class="px-1.5 py-0.5 type-button {inspectMode === 'agg' ? 'bg-primary/12 text-primary' : 'text-muted-foreground hover:text-foreground'}"
                  onclick={() => (inspectMode = "agg")}>max over seq</button>
                <button
                  class="border-l border-border px-1.5 py-0.5 type-button {inspectMode === 'last' ? 'bg-primary/12 text-primary' : 'text-muted-foreground hover:text-foreground'}"
                  onclick={() => (inspectMode = "last")}>last token</button>
              </div>
              <span class="self-center type-caption text-muted-foreground">
                {report.nActive} features active at last token (L0)
              </span>
            {/if}
          </div>

          {#if report}
            <div class="grid gap-1 sm:grid-cols-2 lg:grid-cols-3">
              {#each shownHits as hit}
                <div class="flex items-center justify-between gap-1 border border-border bg-card px-1.5 py-0.5">
                  <div class="flex items-center gap-1.5 overflow-hidden">
                    <button
                      class="type-value text-primary hover:underline"
                      title="steer with feature {hit.feature}"
                      onclick={() => addFeature(hit.feature)}>#{hit.feature}</button>
                    <a
                      href={hit.neuronpediaUrl}
                      target="_blank"
                      rel="noopener"
                      class="type-caption text-muted-foreground hover:text-foreground hover:underline"
                      >neuronpedia ↗</a>
                  </div>
                  <span class="type-value text-subtle-foreground">{hit.activation.toFixed(1)}</span>
                </div>
              {/each}
            </div>
          {/if}
        </Panel>

        <!-- ============ STEERING ============ -->
        <Panel title="Steering">
          <p class="type-caption text-muted-foreground">
            Add <span class="type-code">Σ α·W_dec[feature]</span> to the residual stream at
            layer {info.saeLayer}. Positive α amplifies a feature; negative suppresses it.
            W_dec rows are unit-norm, so α is in residual-norm units.
          </p>

          {#if PRESETS.length > 0}
            <div class="stack-tight">
              <FormLabel value="Presets — the Golden Gate moment" />
              <div class="flex flex-wrap gap-1">
                {#each PRESETS as p}
                  <button
                    class="border border-border bg-card px-1.5 py-0.5 type-button text-muted-foreground hover:bg-muted hover:text-foreground"
                    title={p.note}
                    onclick={() => applyPreset(p)}>{p.name}</button>
                {/each}
              </div>
            </div>
          {/if}

          <div class="flex flex-wrap items-end gap-2">
            <div class="stack-tight">
              <FormLabel value="Add feature by index" />
              <input
                type="number"
                min="0"
                max={info.numFeatures - 1}
                bind:value={newFeature}
                class="w-24 border border-border bg-card px-1 py-0.5 type-value text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring"
              />
            </div>
            <ActionButton color="gray" onclick={() => addFeature(newFeature)}>Add</ActionButton>
          </div>

          {#if steer.length === 0}
            <Note label="No features" type="info">
              Add a feature from the inspector or by index, or pick a preset.
            </Note>
          {:else}
            <div class="stack-field">
              {#each steer as s (s.feature)}
                <div class="flex items-center gap-2 border border-border bg-card px-1.5 py-1">
                  <a href={neuronpedia(s.feature)} target="_blank" rel="noopener"
                    class="w-20 shrink-0 type-value text-primary hover:underline">#{s.feature} ↗</a>
                  <input
                    type="range"
                    min={DEFAULT_ALPHA_RANGE.min}
                    max={DEFAULT_ALPHA_RANGE.max}
                    step={DEFAULT_ALPHA_RANGE.step}
                    value={s.alpha}
                    oninput={(e) => setAlpha(s.feature, Number(e.currentTarget.value))}
                    class="w-full accent-[var(--primary)]"
                  />
                  <span class="w-12 shrink-0 text-right type-value {s.alpha > 0 ? 'text-primary' : s.alpha < 0 ? 'text-destructive' : 'text-muted-foreground'}">{s.alpha}</span>
                  <button class="shrink-0 type-button text-muted-foreground hover:text-destructive"
                    onclick={() => removeFeature(s.feature)}>✕</button>
                </div>
              {/each}
            </div>
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
            <div class="stack-tight">
              <FormLabel value="Max tokens" />
              <input type="number" min="8" max="200" bind:value={maxNewTokens}
                class="w-20 border border-border bg-card px-1 py-0.5 type-value text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring" />
            </div>
            <div class="stack-tight">
              <FormLabel value="Temperature" />
              <input type="number" min="0" max="1.5" step="0.1" bind:value={temperature}
                class="w-20 border border-border bg-card px-1 py-0.5 type-value text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring" />
            </div>
          </div>

          <div class="flex flex-wrap gap-2">
            <ActionButton color="blue" disabled={busy || steer.length === 0 || !prompt.trim()} onclick={generateSteered}>
              {busy ? "Generating…" : "Generate (steered)"}
            </ActionButton>
            <ActionButton color="gray" disabled={busy || !prompt.trim()} onclick={generateBoth}>
              Compare baseline vs steered
            </ActionButton>
            {#if steer.length === 0}
              <span class="self-center type-caption text-muted-foreground">Add a steering feature first.</span>
            {/if}
          </div>

          <div class="grid gap-2 md:grid-cols-2">
            {#if baselineRun}{@render runCard(baselineRun)}{/if}
            {#if steeredRun}{@render runCard(steeredRun)}{/if}
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
      <span class="type-label {run.steered ? 'text-primary' : 'text-subtle-foreground'}">{run.label}</span>
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
          <Statistic label="tape">{run.stats.tape.hits}h/{run.stats.tape.traces}t of {run.stats.tape.calls} · {run.stats.tape.ready ? "ready" : "cold"}</Statistic>
        {/if}
        {#if run.stats.decodeBreakdown}
          {@const d = run.stats.decodeBreakdown}
          <Statistic label="ms/tok b·l·f·s·st">{d.buildMs}·{d.lowerMs}·{d.fenceMs}·{d.sampleMs}·{d.stepMs}</Statistic>
        {/if}
      </div>
    {/if}
  </div>
{/snippet}
