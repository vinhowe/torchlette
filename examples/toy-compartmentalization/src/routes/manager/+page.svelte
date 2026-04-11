<script lang="ts">
  /**
   * Experiment manager — list view + create form.
   *
   * Talks to the compartmentalization-server WebSocket via the singleton
   * ExperimentClient. Shows every experiment the server knows about with
   * its live status, lets you launch new ones from any registered script,
   * and links to per-experiment detail pages.
   *
   * The `client.experiments` map is reactive ($state inside the client),
   * so as the server pushes `metric` / `status` / `created` / `deleted`
   * events the table updates automatically without any local refresh
   * loop.
   */
  import "../../app.css";
  import { onMount, onDestroy } from "svelte";
  import {
    ActionButton,
    BorderedGroup,
    Slider,
    NumberInput,
    TextInput,
    CheckboxInput,
    SelectInput,
  } from "piston-controls";
  import { getExperimentClient, type ExperimentClient } from "$lib/experiment-client-singleton.svelte";
  import type { ExperimentRecord, ScriptInfo, ParamSpec } from "$lib/experiment-client.svelte";

  let client: ExperimentClient = $state(getExperimentClient());

  // The experiments map is a $state on the client. Reading it through a
  // $derived getter makes Svelte track the dependency cleanly even though
  // we're indirecting through a class instance.
  let experiments = $derived(Object.values(client.experiments));
  let scripts = $derived(client.scripts);
  let connected = $derived(client.connected);
  let lastError = $derived(client.lastError);

  // ── create form state ──
  //
  // Persisted to localStorage so a page reload (or closing and reopening
  // the browser) doesn't lose what the user was about to launch. The
  // initial values below are populated from localStorage if present —
  // we do this synchronously at module init (the route has ssr=false
  // so this only runs in the browser, and localStorage is available
  // before any reactive effect fires).
  type FormValue = number | boolean | string;
  type PersistedForm = {
    selectedScript?: string;
    totalSteps?: number;
    description?: string;
    formParams?: Record<string, Record<string, FormValue>>;
  };
  const STORAGE_KEY = "experiment-manager-form";
  function loadPersistedForm(): PersistedForm {
    if (typeof window === "undefined") return {};
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      if (typeof parsed !== "object" || parsed === null) return {};
      return parsed as PersistedForm;
    } catch {
      return {};
    }
  }
  const persisted = loadPersistedForm();

  let selectedScript = $state<string>(persisted.selectedScript ?? "");
  let totalSteps = $state<number>(persisted.totalSteps ?? 5000);
  let description = $state<string>(persisted.description ?? "");
  // Per-script param values: keyed by script name so switching scripts
  // and back preserves what you typed. Values can be numbers (Slider),
  // booleans (CheckboxInput), or strings (SelectInput) depending on the
  // script's ParamSpec.type. ensureFormParams populates defaults for
  // every known type.
  let formParams = $state<Record<string, Record<string, FormValue>>>(
    persisted.formParams ?? {},
  );
  let creatingError = $state<string | null>(null);
  let creating = $state(false);

  // Persist on every change. localStorage writes are fast (<1ms) so we
  // don't bother debouncing even though this fires on every slider
  // drag. Wrapped in try/catch because localStorage can throw on
  // quota-exceeded / denied / private-mode.
  $effect(() => {
    if (typeof window === "undefined") return;
    const snapshot: PersistedForm = {
      selectedScript,
      totalSteps,
      description,
      formParams,
    };
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
    } catch {
      // storage full / denied — silent fail, form still works in-memory
    }
  });

  let selectedScriptInfo = $derived<ScriptInfo | undefined>(
    scripts.find((s) => s.name === selectedScript),
  );

  /** Populate formParams[name] with every param's default. Handles all
   *  three types (number / boolean / select) so the template can bind
   *  Slider / CheckboxInput / SelectInput to the same record slot
   *  without worrying about undefined intermediaries. Called synchronously
   *  from the pre-render effect below — a post-render $effect would
   *  race against the first template pass on script change. */
  function ensureFormParams(name: string) {
    if (!name || formParams[name]) return;
    const info = scripts.find((s) => s.name === name);
    if (!info) return;
    const init: Record<string, FormValue> = {};
    for (const [key, spec] of Object.entries(info.params)) {
      const d = spec.default;
      if (typeof d === "number" || typeof d === "boolean" || typeof d === "string") {
        init[key] = d;
      }
    }
    formParams = { ...formParams, [name]: init };
  }

  // Run BEFORE the DOM updates so the ensuring happens in the same microtask
  // as the selectedScript / scripts change. A plain $effect would fire after
  // the template re-runs, and bindings like `formParams[selectedScript][key]`
  // would crash on the intermediate render where selectedScript is set but
  // formParams[selectedScript] is still undefined.
  $effect.pre(() => {
    ensureFormParams(selectedScript);
  });

  // Auto-pick the first script when scripts arrive. Also initializes the
  // form params for that script so there's no one-render gap.
  $effect(() => {
    if (!selectedScript && scripts.length > 0) {
      selectedScript = scripts[0].name;
      ensureFormParams(scripts[0].name);
    }
  });

  onMount(() => {
    // The singleton is connected on first import; just make sure we have
    // fresh data on mount in case the page navigated here after a long
    // tab idle.
    void client.list().catch(() => {});
    void client.listScripts().catch(() => {});
  });

  // We don't close the client on unmount — other manager pages may still
  // be using it. The singleton outlives any individual route.
  onDestroy(() => {});

  async function handleCreate() {
    if (!selectedScript) return;
    creatingError = null;
    creating = true;
    try {
      const params = formParams[selectedScript] ?? {};
      const id = await client.create({
        script: selectedScript,
        params,
        total_steps: totalSteps,
        description: description || undefined,
      });
      // Redirect to the detail page so the user lands on something useful.
      if (typeof window !== "undefined") {
        window.location.href = `/manager/${id}`;
      }
    } catch (e: any) {
      creatingError = e?.message ?? String(e);
    } finally {
      creating = false;
    }
  }

  async function handleStop(id: string) {
    try {
      await client.stop(id);
    } catch (e: any) {
      console.error("stop failed", e);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm(`Delete experiment ${id}? This wipes its checkpoint and metrics from disk.`)) {
      return;
    }
    try {
      await client.deleteExperiment(id);
    } catch (e: any) {
      alert(`delete failed: ${e?.message ?? e}`);
    }
  }

  /** Copy an existing experiment's settings into the create form so the
   *  user can launch a new run with the same (or tweaked) parameters.
   *  Copies script, params, total_steps, description. Any non-number /
   *  non-boolean / non-string values are skipped (shouldn't happen in
   *  practice — all ParamSpec-typed values are one of the three). */
  function cloneSettings(rec: ExperimentRecord) {
    selectedScript = rec.script;
    description = rec.description ?? "";
    totalSteps = rec.total_steps;
    const copy: Record<string, FormValue> = {};
    for (const [k, v] of Object.entries(rec.params)) {
      if (typeof v === "number" || typeof v === "boolean" || typeof v === "string") {
        copy[k] = v;
      }
    }
    formParams = { ...formParams, [rec.script]: copy };
    if (typeof window !== "undefined") {
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  }

  function statusBadgeClasses(status: string): string {
    const base = "inline-block px-[6px] py-[1px] font-mono text-[10px] uppercase tracking-[0.06em] rounded";
    switch (status) {
      case "running":
        return `${base} bg-[rgba(46,204,113,0.15)] text-[#1e8449]`;
      case "stopped":
        return `${base} bg-[rgba(0,0,0,0.06)] text-[rgba(0,0,0,0.54)]`;
      case "stopping":
        return `${base} bg-[rgba(255,196,0,0.18)] text-[#a06000]`;
      case "failed":
        return `${base} bg-[rgba(214,39,40,0.15)] text-[#c0392b]`;
      case "paused":
        return `${base} bg-[rgba(74,154,222,0.18)] text-[#1f618d]`;
      default:
        return `${base} bg-[rgba(0,0,0,0.06)] text-[rgba(0,0,0,0.54)]`;
    }
  }

  function fmtLoss(rec: ExperimentRecord): string {
    const loss = rec.latest_metrics?.loss;
    return typeof loss === "number" ? loss.toFixed(4) : "—";
  }

  function fmtProgress(rec: ExperimentRecord): string {
    return `${rec.step_count} / ${rec.total_steps}`;
  }

  function paramSpecRange(spec: ParamSpec): { min: number; max: number; step: number; useLog: boolean } {
    const min = typeof spec.min === "number" ? spec.min : 0;
    const max = typeof spec.max === "number" ? spec.max : 1;
    const useLog = spec.scale === "log";
    const span = max - min;
    let step = span / 100;
    if (Number.isInteger(spec.default) && Number.isInteger(min) && Number.isInteger(max) && span >= 16) {
      step = 1;
    } else if (useLog) {
      step = (max - min) / 1000;
    } else if (span <= 1) {
      step = 0.01;
    }
    return { min, max, step, useLog };
  }

  /** Classify a param spec by control type. Dispatches Slider/Checkbox/Select.
   *  A spec that declares `type: "select"` is treated as a select regardless
   *  of the default's JS type (the default is typically one of the choices).
   */
  function controlKind(spec: ParamSpec): "number" | "boolean" | "select" | "unknown" {
    if (spec.type === "select" || Array.isArray(spec.choices)) return "select";
    if (spec.type === "boolean" || typeof spec.default === "boolean") return "boolean";
    if (spec.type === "number" || typeof spec.default === "number") return "number";
    return "unknown";
  }
</script>

<svelte:head>
  <title>Experiments — compartmentalization</title>
</svelte:head>

<div class="mx-auto max-w-[1100px] px-6 pt-10 pb-24 text-[rgba(0,0,0,0.84)]">
  <header class="mb-7 flex items-baseline justify-between">
    <h1 class="text-[28px] font-bold tracking-[-0.02em]">Experiments</h1>
    <div class="flex items-center gap-3 font-mono text-[11px] text-[rgba(0,0,0,0.54)]">
      <span class={connected ? "text-[#1e8449]" : "text-[#c0392b]"}>
        ● {connected ? "connected" : "disconnected"}
      </span>
      <span>{experiments.length} experiments</span>
    </div>
  </header>

  {#if lastError}
    <div class="mb-5 border-l-[3px] border-l-[#d62728] bg-[rgba(214,39,40,0.05)] px-[14px] py-2 text-[12px]">
      {lastError}
    </div>
  {/if}

  <!-- ── create form ── -->
  <section class="mb-10">
    <h2 class="mb-3 text-[15px] font-semibold tracking-[-0.01em]">New experiment</h2>
    {#if scripts.length === 0}
      <div class="text-[13px] text-[rgba(0,0,0,0.54)]">No scripts loaded yet. Is the server running?</div>
    {:else}
      <div class="mb-3 flex flex-wrap gap-2 font-mono text-[11px]">
        {#each scripts as script}
          <button
            type="button"
            onclick={() => (selectedScript = script.name)}
            class="border px-[10px] py-[3px] uppercase tracking-[0.06em] rounded {selectedScript === script.name
              ? 'border-[rgba(0,0,0,0.84)] bg-[rgba(0,0,0,0.04)]'
              : 'border-[rgba(0,0,0,0.18)] text-[rgba(0,0,0,0.54)] hover:text-[rgba(0,0,0,0.84)]'}"
          >
            {script.name}
          </button>
        {/each}
      </div>
      {#if selectedScriptInfo?.description}
        <p class="mb-3 max-w-[680px] text-[12px] text-[rgba(0,0,0,0.54)]">
          {selectedScriptInfo.description}
        </p>
      {/if}
      <div class="grid grid-cols-[repeat(auto-fit,minmax(240px,1fr))] gap-3">
        <BorderedGroup title="Run" id="grp-run" contentClass="p-2 space-y-2">
          <NumberInput id="total-steps" label="Total steps" min={1} step={100}
                       bind:value={totalSteps} />
          <TextInput id="description" label="Description (optional)"
                     bind:value={description} />
        </BorderedGroup>

        {#if selectedScriptInfo && formParams[selectedScript]}
          {@const liveParams = Object.entries(selectedScriptInfo.params).filter(([, s]) => s.live)}
          {@const structParams = Object.entries(selectedScriptInfo.params).filter(([, s]) => !s.live)}
          {#if liveParams.length > 0}
            <BorderedGroup title="Live params (editable mid-run)" id="grp-live" contentClass="p-2 space-y-2">
              {#each liveParams as [key, spec]}
                {@const kind = controlKind(spec)}
                {#if kind === "number"}
                  {@const range = paramSpecRange(spec)}
                  <Slider
                    id={`live-${key}`}
                    label={spec.description || key}
                    min={range.min}
                    max={range.max}
                    step={range.step}
                    useLog={range.useLog}
                    bind:value={formParams[selectedScript][key] as number}
                  />
                {:else if kind === "boolean"}
                  <CheckboxInput
                    id={`live-${key}`}
                    label={spec.description || key}
                    bind:checked={formParams[selectedScript][key] as boolean}
                  />
                {:else if kind === "select"}
                  <SelectInput
                    id={`live-${key}`}
                    label={spec.description || key}
                    options={(spec.choices ?? []).map((c) => ({ value: c }))}
                    bind:value={formParams[selectedScript][key] as string}
                  />
                {/if}
              {/each}
            </BorderedGroup>
          {/if}
          {#if structParams.length > 0}
            <BorderedGroup title="Structural (fixed at creation)" id="grp-struct" contentClass="p-2 space-y-2">
              {#each structParams as [key, spec]}
                {@const kind = controlKind(spec)}
                {#if kind === "number"}
                  {@const range = paramSpecRange(spec)}
                  <Slider
                    id={`struct-${key}`}
                    label={spec.description || key}
                    min={range.min}
                    max={range.max}
                    step={range.step}
                    useLog={range.useLog}
                    bind:value={formParams[selectedScript][key] as number}
                  />
                {:else if kind === "boolean"}
                  <CheckboxInput
                    id={`struct-${key}`}
                    label={spec.description || key}
                    bind:checked={formParams[selectedScript][key] as boolean}
                  />
                {:else if kind === "select"}
                  <SelectInput
                    id={`struct-${key}`}
                    label={spec.description || key}
                    options={(spec.choices ?? []).map((c) => ({ value: c }))}
                    bind:value={formParams[selectedScript][key] as string}
                  />
                {/if}
              {/each}
            </BorderedGroup>
          {/if}
        {/if}
      </div>

      <div class="mt-3 flex items-center gap-3">
        <ActionButton
          color="blue"
          onclick={handleCreate}
          disabled={creating || !connected}
        >
          {creating ? "creating…" : "kick off"}
        </ActionButton>
        {#if creatingError}
          <span class="text-[12px] text-[#c0392b]">{creatingError}</span>
        {/if}
      </div>
    {/if}
  </section>

  <!-- ── list ── -->
  <section>
    <h2 class="mb-3 text-[15px] font-semibold tracking-[-0.01em]">All experiments</h2>
    {#if experiments.length === 0}
      <div class="text-[13px] text-[rgba(0,0,0,0.54)]">No experiments yet. Create one above.</div>
    {:else}
      <table class="w-full border-collapse font-mono text-[12px]">
        <thead>
          <tr class="border-b border-[rgba(0,0,0,0.18)] text-[10.5px] uppercase tracking-[0.06em] text-[rgba(0,0,0,0.54)]">
            <th class="py-[6px] pr-3 text-left">id</th>
            <th class="py-[6px] pr-3 text-left">script</th>
            <th class="py-[6px] pr-3 text-left">status</th>
            <th class="py-[6px] pr-3 text-right">gpu</th>
            <th class="py-[6px] pr-3 text-right">step / total</th>
            <th class="py-[6px] pr-3 text-right">latest loss</th>
            <th class="py-[6px] pr-3 text-right">checkpt</th>
            <th class="py-[6px] pr-3 text-right">actions</th>
          </tr>
        </thead>
        <tbody>
          {#each experiments as rec (rec.id)}
            <tr class="border-b border-[rgba(0,0,0,0.06)] hover:bg-[rgba(0,0,0,0.02)]">
              <td class="py-[6px] pr-3">
                <a href={`/manager/${rec.id}`} class="underline decoration-[rgba(0,0,0,0.18)] hover:decoration-[rgba(0,0,0,0.84)]">
                  {rec.id.slice(0, 24)}{rec.id.length > 24 ? "…" : ""}
                </a>
              </td>
              <td class="py-[6px] pr-3">{rec.script}</td>
              <td class="py-[6px] pr-3">
                <span class={statusBadgeClasses(rec.status)}>{rec.status}</span>
              </td>
              <td class="py-[6px] pr-3 text-right">{rec.gpu ?? "—"}</td>
              <td class="py-[6px] pr-3 text-right">{fmtProgress(rec)}</td>
              <td class="py-[6px] pr-3 text-right">{fmtLoss(rec)}</td>
              <td class="py-[6px] pr-3 text-right">{rec.last_checkpoint_step}</td>
              <td class="py-[6px] pr-3 text-right space-x-3">
                <button
                  type="button"
                  onclick={() => cloneSettings(rec)}
                  title="Copy settings into the create form above"
                  class="text-[rgba(0,0,0,0.54)] underline hover:text-[rgba(0,0,0,0.84)]"
                >clone</button>
                {#if rec.status === "running" || rec.status === "stopping"}
                  <button type="button" onclick={() => handleStop(rec.id)} class="text-[rgba(0,0,0,0.54)] underline hover:text-[#c0392b]">stop</button>
                {:else}
                  <button type="button" onclick={() => handleDelete(rec.id)} class="text-[rgba(0,0,0,0.54)] underline hover:text-[#c0392b]">delete</button>
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </section>
</div>
