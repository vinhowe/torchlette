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
  import { BorderedGroup, Slider } from "piston-controls";
  import { LineChart } from "$lib/components";
  import { baseChartOpt, chartAxes, lineSeries, legendBlock } from "$lib/chart-helpers";
  import { THEME } from "$lib/theme";
  import { getExperimentClient, type ExperimentClient } from "$lib/experiment-client-singleton.svelte";
  import type { ExperimentRecord, ParamSpec, ScriptInfo } from "$lib/experiment-client.svelte";

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

  // ── derived chart option ──
  // Down-sample the loss history if it gets long. Echarts handles 10k
  // points fine but 100k starts to lag — we just stride for v0.1.
  let lossSeries = $derived.by((): [number, number][] => {
    if (!rec || rec.history.length === 0) return [];
    const maxPoints = 4000;
    const stride = Math.max(1, Math.ceil(rec.history.length / maxPoints));
    const out: [number, number][] = [];
    for (let i = 0; i < rec.history.length; i += stride) {
      const e = rec.history[i];
      const loss = e.metrics?.loss;
      if (typeof loss === "number") out.push([e.step, loss]);
    }
    return out;
  });

  let chartOption = $derived({
    ...baseChartOpt(),
    ...chartAxes({ yType: "value" }),
    legend: legendBlock(),
    series: [lineSeries(lossSeries, { color: THEME.accent, name: "loss" })],
  });

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
  let liveValues = $state<Record<string, number>>({});
  // Track which key is currently being edited (i.e. has a debounce timer
  // running). While dirty, we ignore incoming server pushes for that key
  // so the slider doesn't jump back under the user's thumb.
  const dirtyKeys = new Set<string>();
  const debounceMs = 80;
  const debounceTimers: Record<string, ReturnType<typeof setTimeout>> = {};
  // Last value we've actually sent to the server — used as a diff base
  // so the push-effect doesn't re-send the same value.
  const lastPushed: Record<string, number> = {};

  // Initialize + sync liveValues from rec.params. Runs whenever rec
  // changes; skips keys that are currently being edited.
  $effect(() => {
    if (!rec || !scriptInfo) return;
    for (const [key, spec] of Object.entries(scriptInfo.params)) {
      if (!spec.live) continue;
      if (dirtyKeys.has(key)) continue;
      const raw = rec.params[key];
      const numeric = typeof raw === "number" ? raw : Number(raw ?? spec.default ?? 0);
      if (Number.isFinite(numeric) && liveValues[key] !== numeric) {
        liveValues[key] = numeric;
        lastPushed[key] = numeric;
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
          <button type="button" onclick={handleStop}
            class="bg-[rgba(0,0,0,0.84)] px-[12px] py-[5px] font-mono text-[11px] uppercase tracking-[0.06em] text-white hover:bg-[rgba(0,0,0,1)]">
            stop
          </button>
        {:else}
          <button type="button" onclick={handleDelete}
            class="border border-[rgba(0,0,0,0.18)] bg-white px-[12px] py-[5px] font-mono text-[11px] uppercase tracking-[0.06em] text-[rgba(0,0,0,0.54)] hover:border-[#c0392b] hover:text-[#c0392b]">
            delete
          </button>
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

    <section class="mb-9">
      <h2 class="mb-2 text-[15px] font-semibold tracking-[-0.01em]">Loss</h2>
      <LineChart option={chartOption} height={320} />
    </section>

    {#if scriptInfo}
      {@const liveParams = Object.entries(scriptInfo.params).filter(([, s]) => s.live)}
      {@const structParams = Object.entries(scriptInfo.params).filter(([, s]) => !s.live)}
      <section class="mb-9">
        <h2 class="mb-3 text-[15px] font-semibold tracking-[-0.01em]">Parameters</h2>
        <div class="grid grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-3">
          {#if liveParams.length > 0}
            <BorderedGroup title="Live (sliders push to running worker)" id="grp-live" contentClass="p-2 space-y-2">
              {#each liveParams as [key, spec]}
                {@const range = paramSpecRange(spec)}
                {#if liveValues[key] !== undefined}
                  <Slider
                    id={`live-${key}`}
                    label={spec.description || key}
                    min={range.min}
                    max={range.max}
                    step={range.step}
                    useLog={range.useLog}
                    bind:value={liveValues[key]}
                  />
                {/if}
              {/each}
            </BorderedGroup>
          {/if}
          {#if structParams.length > 0}
            <BorderedGroup title="Structural (fixed at creation)" id="grp-struct" contentClass="p-2 space-y-2">
              {#each structParams as [key, spec]}
                <div class="flex items-baseline justify-between font-mono text-[11px]">
                  <span class="text-[rgba(0,0,0,0.54)]">{key}</span>
                  <span class="text-[rgba(0,0,0,0.84)]">{rec.params[key] ?? "—"}</span>
                </div>
              {/each}
            </BorderedGroup>
          {/if}
        </div>
      </section>
    {/if}
  {/if}
</div>
