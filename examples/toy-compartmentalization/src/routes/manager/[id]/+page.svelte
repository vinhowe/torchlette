<script lang="ts">
  /**
   * Experiment detail view — live loss chart, live param editing, controls.
   *
   * Subscribes to one experiment id (from the route params) on mount,
   * unsubscribes on destroy, and reads everything off the singleton
   * client's reactive map. The loss chart is fed straight from
   * `rec.history` which is the merge of (server-pushed history backfill
   * at subscribe time) + (live metric stream).
   *
   * Live param editing: any param spec'd as `live: true` gets a slider
   * that pushes set_param RPCs on change, with a small debounce so
   * dragging the slider doesn't fire 200 RPCs.
   */
  import "../../../app.css";
  import { onMount, onDestroy } from "svelte";
  import { page } from "$app/stores";
  import {
    ActionButton,
    BorderedGroup,
    Slider,
    CheckboxInput,
    SelectInput,
  } from "piston-controls";
  import { LineChart } from "$lib/components";
  import { baseChartOpt, chartAxes, lineSeries, legendBlock } from "$lib/chart-helpers";
  import { THEME, SERIES_PALETTE } from "$lib/theme";
  import { getExperimentClient, type ExperimentClient } from "$lib/experiment-client-singleton.svelte";
  import type { ExperimentRecord, MetricEntry, ParamSpec, ScriptInfo } from "$lib/experiment-client.svelte";

  type FormValue = number | boolean | string;

  let client: ExperimentClient = $state(getExperimentClient());

  // Route param. Svelte's `$page` store is reactive; the id will update
  // if the user navigates between detail pages without a full reload.
  let id = $derived($page.params.id);

  let rec: ExperimentRecord | undefined = $derived(client.experiments[id]);
  let scriptInfo: ScriptInfo | undefined = $derived(
    rec ? client.scripts.find((s) => s.name === rec.script) : undefined,
  );
  let connected = $derived(client.connected);
  let lastError = $derived(client.lastError);

  // Subscribe / unsubscribe lifecycle. Re-subscribe whenever the id changes
  // (the user navigated to a different experiment in the same tab).
  let lastSubscribedId: string | null = null;
  $effect(() => {
    const targetId = id;
    if (!targetId || targetId === lastSubscribedId) return;
    if (lastSubscribedId) {
      void client.unsubscribe(lastSubscribedId).catch(() => {});
    }
    lastSubscribedId = targetId;
    void client.subscribe(targetId).catch((e) => {
      console.error("subscribe failed", e);
    });
  });

  onMount(() => {
    // Make sure scripts are loaded so we can render the param schema.
    if (client.scripts.length === 0) {
      void client.listScripts().catch(() => {});
    }
  });

  onDestroy(() => {
    if (lastSubscribedId) {
      void client.unsubscribe(lastSubscribedId).catch(() => {});
    }
  });

  // ── derived chart options ──
  //
  // The history is a flat list of {step, metrics} events. Different keys
  // appear at different cadences: `loss` lands on every step while
  // `probe_r2` / `cos_sim` / `acc_cN` / `translation_loss` only land on
  // eval steps (every N steps). We render ONE chart per metric "family"
  // (top-level key, ignoring any `_cN` compartment suffix), each family
  // holding one series per compartment it has data for.
  //
  // This means: the three mess3 keys `probe_r2`, `probe_r2_c1`,
  // `probe_r2_c2` collapse into a single "probe_r2" chart with 3 series.
  // The four bio keys `acc_c0`, `acc_c1`, `acc_c2`, `acc_c3` collapse
  // into one "acc" chart with 4 series. `loss` / `cos_sim` /
  // `translation_loss` are each their own chart with one series.
  //
  // Family extraction strips a trailing `_c<digits>` from the key if
  // present. Keys without that suffix pass through unchanged.

  type ChartFamily = {
    key: string; // family name: "loss", "probe_r2", "acc", ...
    series: { name: string; data: [number, number][] }[];
    yType: "value" | "log";
    yMin?: number;
    yMax?: number;
  };

  function familyOf(metricKey: string): { family: string; compIdx: number | null } {
    const m = metricKey.match(/^(.*)_c(\d+)$/);
    if (m) {
      return { family: m[1], compIdx: Number(m[2]) };
    }
    return { family: metricKey, compIdx: null };
  }

  // Per-family axis hints: loss is log-scale, probe_r2 is bounded [-0.2, 1]
  // (negative happens early when the probe is way off), cos_sim is
  // bounded [-1, 1], acc is bounded [0, 1]. Anything else gets auto.
  function axisHint(family: string): { yType: "value" | "log"; yMin?: number; yMax?: number } {
    if (family === "loss" || family === "translation_loss") return { yType: "log" };
    if (family === "probe_r2") return { yType: "value", yMin: -0.2, yMax: 1 };
    if (family === "cos_sim") return { yType: "value", yMin: -1, yMax: 1 };
    if (family === "acc") return { yType: "value", yMin: 0, yMax: 1 };
    return { yType: "value" };
  }

  let chartFamilies = $derived.by((): ChartFamily[] => {
    if (!rec || rec.history.length === 0) return [];
    // family → compIdx|null → series data
    // compIdx=null collapses to a single unnamed series in the family
    const grouped = new Map<string, Map<number | null, [number, number][]>>();
    // Downsample: cap each series at ~4000 points. Echarts handles more
    // but interaction gets laggy.
    const maxPoints = 4000;
    const stride = Math.max(1, Math.ceil(rec.history.length / maxPoints));
    for (let i = 0; i < rec.history.length; i += stride) {
      const entry = rec.history[i] as MetricEntry;
      if (!entry.metrics) continue;
      for (const [k, v] of Object.entries(entry.metrics)) {
        if (typeof v !== "number" || !Number.isFinite(v)) continue;
        const { family, compIdx } = familyOf(k);
        if (!grouped.has(family)) grouped.set(family, new Map());
        const byComp = grouped.get(family)!;
        if (!byComp.has(compIdx)) byComp.set(compIdx, []);
        byComp.get(compIdx)!.push([entry.step, v]);
      }
    }
    // Sort families in a stable order: loss first, then alphabetical.
    const families: ChartFamily[] = [];
    const familyOrder = (name: string): number => {
      if (name === "loss") return 0;
      if (name === "probe_r2") return 1;
      if (name === "cos_sim") return 2;
      if (name === "acc") return 3;
      if (name === "translation_loss") return 4;
      return 100;
    };
    const sortedFams = [...grouped.keys()].sort(
      (a, b) => familyOrder(a) - familyOrder(b) || a.localeCompare(b),
    );
    for (const fam of sortedFams) {
      const byComp = grouped.get(fam)!;
      // Sort compartments numerically; null (plain key) first.
      const compKeys = [...byComp.keys()].sort((a, b) => {
        if (a === null) return -1;
        if (b === null) return 1;
        return (a as number) - (b as number);
      });
      const series = compKeys.map((c) => ({
        name: c === null ? fam : `c${c}`,
        data: byComp.get(c)!,
      }));
      families.push({ key: fam, series, ...axisHint(fam) });
    }
    return families;
  });

  function chartOptionFor(fam: ChartFamily) {
    const multi = fam.series.length > 1;
    return {
      ...baseChartOpt(),
      grid: { top: multi ? 32 : 28, right: 12, bottom: 24, left: 44 },
      ...(multi ? { legend: { ...legendBlock(), top: 4 } } : {}),
      ...chartAxes({ yType: fam.yType, yMin: fam.yMin, yMax: fam.yMax }),
      series: fam.series.map((s, i) => ({
        type: "line" as const,
        name: s.name,
        data: s.data,
        showSymbol: false,
        lineStyle: { width: 1.5, color: SERIES_PALETTE[i % SERIES_PALETTE.length] },
        itemStyle: { color: SERIES_PALETTE[i % SERIES_PALETTE.length] },
      })),
    };
  }

  function titleFor(family: string): string {
    switch (family) {
      case "loss": return "Training loss";
      case "probe_r2": return "Belief probe R²";
      case "cos_sim": return "Cross-compartment cosine similarity";
      case "acc": return "QA accuracy per compartment";
      case "translation_loss": return "Translation loss";
      default: return family;
    }
  }

  // ── live param editing ──
  //
  // The Slider component only supports bind:value (two-way). We can't
  // bind directly to `rec.params[key]` because:
  //   1. rec is $derived from the reactive client map, not mutable here
  //   2. Every slider drag would have to round-trip through the server
  //      before the UI updated, which feels terrible
  //
  // Instead we keep a local `liveValues` state that the sliders bind to,
  // and a debounced $effect that pushes any change back to the server.
  // Server-pushed updates (e.g. from another tab, or a resume-loaded
  // experiment) are synced in via a separate $effect that copies
  // rec.params → liveValues when the user hasn't recently touched a key.
  let liveValues = $state<Record<string, FormValue>>({});
  // Track which key is currently being edited (i.e. has a debounce timer
  // running). While dirty, we ignore incoming server pushes for that key
  // so the slider doesn't jump back under the user's thumb.
  const dirtyKeys = new Set<string>();
  const debounceMs = 80;
  const debounceTimers: Record<string, ReturnType<typeof setTimeout>> = {};
  // Last value we've actually sent to the server — used as a diff base
  // so the push-effect doesn't re-send the same value.
  const lastPushed: Record<string, FormValue> = {};

  // Initialize + sync liveValues from rec.params. Runs whenever rec
  // changes; skips keys that are currently being edited. Handles all
  // three live-param types: number / boolean / string (select).
  $effect(() => {
    if (!rec || !scriptInfo) return;
    for (const [key, spec] of Object.entries(scriptInfo.params)) {
      if (!spec.live) continue;
      if (dirtyKeys.has(key)) continue;
      const raw = rec.params[key];
      let coerced: FormValue | undefined;
      if (typeof raw === "number" || typeof raw === "boolean" || typeof raw === "string") {
        coerced = raw;
      } else if (spec.default !== undefined) {
        coerced = spec.default as FormValue;
      }
      if (coerced !== undefined && liveValues[key] !== coerced) {
        liveValues[key] = coerced;
        lastPushed[key] = coerced;
      }
    }
  });

  // Debounced push: whenever liveValues changes, schedule a set_param
  // for each key whose value drifted from what we last sent.
  $effect(() => {
    if (!rec) return;
    const expId = rec.id;
    for (const key of Object.keys(liveValues)) {
      const value = liveValues[key];
      if (lastPushed[key] === value) continue;
      dirtyKeys.add(key);
      if (debounceTimers[key]) clearTimeout(debounceTimers[key]);
      debounceTimers[key] = setTimeout(() => {
        delete debounceTimers[key];
        lastPushed[key] = value;
        void client
          .setParam(expId, key, value)
          .catch((e) => console.error("set_param failed", e))
          .finally(() => {
            // Release the dirty flag shortly after the RPC completes so
            // incoming server pushes for this key can resume syncing.
            // Using a short delay (vs immediate) absorbs the echo from
            // the server's `updated` global broadcast that will arrive
            // just after our setParam ack.
            setTimeout(() => dirtyKeys.delete(key), 200);
          });
      }, debounceMs);
    }
  });

  // ── controls ──
  async function handleStop() {
    if (!rec) return;
    try {
      await client.stop(rec.id);
    } catch (e: any) {
      console.error("stop failed", e);
    }
  }

  async function handleDelete() {
    if (!rec) return;
    if (!confirm(`Delete experiment ${rec.id}? This wipes its checkpoint and metrics.`)) {
      return;
    }
    try {
      await client.deleteExperiment(rec.id);
      if (typeof window !== "undefined") {
        window.location.href = "/manager";
      }
    } catch (e: any) {
      alert(`delete failed: ${e?.message ?? e}`);
    }
  }

  // ── helpers ──
  function statusBadgeClasses(status: string): string {
    const base = "inline-block px-[8px] py-[2px] font-mono text-[11px] uppercase tracking-[0.06em] rounded";
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

  function controlKind(spec: ParamSpec): "number" | "boolean" | "select" | "unknown" {
    if (spec.type === "select" || Array.isArray(spec.choices)) return "select";
    if (spec.type === "boolean" || typeof spec.default === "boolean") return "boolean";
    if (spec.type === "number" || typeof spec.default === "number") return "number";
    return "unknown";
  }

  function fmtLoss(rec: ExperimentRecord | undefined): string {
    const loss = rec?.latest_metrics?.loss;
    return typeof loss === "number" ? loss.toFixed(4) : "—";
  }
</script>

<svelte:head>
  <title>{rec?.id ?? id} — experiment</title>
</svelte:head>

<div class="mx-auto max-w-[1100px] px-6 pt-10 pb-24 text-[rgba(0,0,0,0.84)]">
  <nav class="mb-6 font-mono text-[11px]">
    <a href="/manager" class="text-[rgba(0,0,0,0.54)] underline decoration-[rgba(0,0,0,0.18)] hover:text-[rgba(0,0,0,0.84)]">
      ← experiments
    </a>
  </nav>

  {#if !rec}
    <div class="text-[13px] text-[rgba(0,0,0,0.54)]">
      {#if !connected}Connecting…{:else}Loading experiment {id}…{/if}
    </div>
  {:else}
    <header class="mb-6 flex items-baseline justify-between">
      <div>
        <h1 class="mb-1 text-[24px] font-bold tracking-[-0.02em]">{rec.script}</h1>
        <div class="font-mono text-[11px] text-[rgba(0,0,0,0.54)]">{rec.id}</div>
      </div>
      <div class="flex items-center gap-3">
        <span class={statusBadgeClasses(rec.status)}>{rec.status}</span>
        {#if rec.status === "running" || rec.status === "stopping"}
          <ActionButton color="red" onclick={handleStop}>stop</ActionButton>
        {:else}
          <ActionButton color="gray" onclick={handleDelete}>delete</ActionButton>
        {/if}
      </div>
    </header>

    {#if lastError}
      <div class="mb-5 border-l-[3px] border-l-[#d62728] bg-[rgba(214,39,40,0.05)] px-[14px] py-2 text-[12px]">
        {lastError}
      </div>
    {/if}

    <div class="mb-6 flex flex-wrap items-center gap-x-[18px] gap-y-1 border-y border-[rgba(0,0,0,0.08)] py-3 font-mono text-[12px] text-[rgba(0,0,0,0.54)]">
      <span>step <b class="font-medium text-[rgba(0,0,0,0.84)]">{rec.step_count} / {rec.total_steps}</b></span>
      <span>loss <b class="font-medium text-[rgba(0,0,0,0.84)]">{fmtLoss(rec)}</b></span>
      <span>gpu <b class="font-medium text-[rgba(0,0,0,0.84)]">{rec.gpu ?? "—"}</b></span>
      <span>checkpoint <b class="font-medium text-[rgba(0,0,0,0.84)]">{rec.last_checkpoint_step}</b></span>
      <span>created <b class="font-medium text-[rgba(0,0,0,0.84)]">{rec.created_at}</b></span>
    </div>

    <section class="mb-9 space-y-8">
      {#each chartFamilies as fam (fam.key)}
        <div>
          <h2 class="mb-2 text-[15px] font-semibold tracking-[-0.01em]">
            {titleFor(fam.key)}
          </h2>
          <LineChart option={chartOptionFor(fam)} height={260} />
        </div>
      {/each}
      {#if chartFamilies.length === 0}
        <div class="text-[13px] text-[rgba(0,0,0,0.54)]">No metrics yet.</div>
      {/if}
    </section>

    {#if scriptInfo}
      {@const liveParams = Object.entries(scriptInfo.params).filter(([, s]) => s.live)}
      {@const structParams = Object.entries(scriptInfo.params).filter(([, s]) => !s.live)}
      <section class="mb-9">
        <h2 class="mb-3 text-[15px] font-semibold tracking-[-0.01em]">Parameters</h2>
        <div class="grid grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-3">
          {#if liveParams.length > 0}
            <BorderedGroup title="Live (edits push to running worker)" id="grp-live" contentClass="p-2 space-y-2">
              {#each liveParams as [key, spec]}
                {@const kind = controlKind(spec)}
                {#if liveValues[key] !== undefined}
                  {#if kind === "number"}
                    {@const range = paramSpecRange(spec)}
                    <Slider
                      id={`live-${key}`}
                      label={spec.description || key}
                      min={range.min}
                      max={range.max}
                      step={range.step}
                      useLog={range.useLog}
                      bind:value={liveValues[key] as number}
                    />
                  {:else if kind === "boolean"}
                    <CheckboxInput
                      id={`live-${key}`}
                      label={spec.description || key}
                      bind:checked={liveValues[key] as boolean}
                    />
                  {:else if kind === "select"}
                    <SelectInput
                      id={`live-${key}`}
                      label={spec.description || key}
                      options={(spec.choices ?? []).map((c) => ({ value: c }))}
                      bind:value={liveValues[key] as string}
                    />
                  {/if}
                {/if}
              {/each}
            </BorderedGroup>
          {/if}
          {#if structParams.length > 0}
            <BorderedGroup title="Structural (fixed at creation)" id="grp-struct" contentClass="p-2 space-y-2">
              {#each structParams as [key]}
                <div class="flex items-baseline justify-between font-mono text-[11px]">
                  <span class="text-[rgba(0,0,0,0.54)]">{key}</span>
                  <span class="text-[rgba(0,0,0,0.84)]">{String(rec.params[key] ?? "—")}</span>
                </div>
              {/each}
            </BorderedGroup>
          {/if}
        </div>
      </section>
    {/if}
  {/if}
</div>
