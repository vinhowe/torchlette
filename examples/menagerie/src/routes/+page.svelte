<script lang="ts">
import { onMount } from "svelte";
import {
  PRESETS,
  paramCount,
  formatParams,
  probeGpu,
  type GpuProbe,
} from "$lib/tiers";
import { isOAuthConfigured } from "$lib/hf/config";

let probe = $state<GpuProbe | null>(null);
let probing = $state(true);

onMount(async () => {
  try {
    probe = await probeGpu();
  } catch (e) {
    probe = { supported: false, error: (e as Error).message };
  } finally {
    probing = false;
  }
});

function fmtBytes(n?: number): string {
  if (!n) return "—";
  const GB = 1024 ** 3;
  const MB = 1024 ** 2;
  if (n >= GB) return `${(n / GB).toFixed(1)} GB`;
  return `${Math.round(n / MB)} MB`;
}
</script>

<main class="mx-auto max-w-5xl px-6 py-12">
  <header class="mb-10">
    <h1 class="text-4xl font-semibold tracking-tight">Menagerie</h1>
    <p class="mt-3 max-w-2xl text-slate-300">
      An open-ended, genealogical ecology of models you train, fork, merge, and
      evaluate — entirely in your browser. Every model is a Hugging Face repo;
      its git history is the model's life. Pick something, fork it, train it,
      and pass it on.
    </p>
    <a
      href="/explore"
      class="mt-4 inline-block rounded-lg border border-slate-700 px-4 py-2 text-sm hover:bg-slate-800"
      >Explore the ecology →</a
    >
  </header>

  {#if !isOAuthConfigured()}
    <section
      class="mb-10 rounded-lg border border-amber-900/50 bg-amber-950/30 px-4 py-3 text-sm text-amber-200"
    >
      Login is disabled until <span class="font-mono">VITE_HF_OAUTH_CLIENT_ID</span>
      is set. Register a Developer Application at
      <a
        class="underline"
        href="https://huggingface.co/settings/connected-applications"
        target="_blank"
        rel="noreferrer">huggingface.co/settings/connected-applications</a
      >
      (scopes: openid, profile, read-repos, write-repos) and copy
      <span class="font-mono">.env.example</span> → <span class="font-mono">.env.local</span>.
    </section>
  {/if}

  <!-- WebGPU capability probe -->
  <section class="mb-10 rounded-xl border border-slate-800 bg-slate-900/50 p-5">
    <h2 class="mb-3 text-lg font-medium">Your hardware</h2>
    {#if probing}
      <p class="text-slate-400">Probing WebGPU…</p>
    {:else if probe && !probe.supported}
      <p class="text-rose-400">WebGPU unavailable.</p>
      <p class="mt-1 text-sm text-slate-400">{probe.error}</p>
    {:else if probe}
      <div class="grid grid-cols-1 gap-x-8 gap-y-1 text-sm sm:grid-cols-2">
        <div class="flex justify-between border-b border-slate-800 py-1">
          <span class="text-slate-400">Adapter</span>
          <span class="font-mono"
            >{probe.description || probe.vendor || "unknown"}{probe.architecture
              ? ` (${probe.architecture})`
              : ""}</span
          >
        </div>
        <div class="flex justify-between border-b border-slate-800 py-1">
          <span class="text-slate-400">max buffer</span>
          <span class="font-mono">{fmtBytes(probe.maxBufferSize)}</span>
        </div>
        <div class="flex justify-between border-b border-slate-800 py-1">
          <span class="text-slate-400">max storage binding</span>
          <span class="font-mono">{fmtBytes(probe.maxStorageBufferBindingSize)}</span>
        </div>
        <div class="flex justify-between border-b border-slate-800 py-1">
          <span class="text-slate-400">recommended tier</span>
          <span class="font-mono text-emerald-400">{probe.recommended}</span>
        </div>
      </div>
      <p class="mt-3 text-xs text-slate-500">
        Browsers don't expose total VRAM, so this is a coarse estimate from
        adapter limits — you can always pick a bigger tier and we'll downshift if
        it won't fit.
      </p>
    {/if}
  </section>

  <!-- Preset tiers -->
  <section>
    <h2 class="mb-3 text-lg font-medium">Choose your size</h2>
    <div class="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {#each PRESETS as p (p.id)}
        {@const recommended = probe?.recommended === p.id}
        <div
          class="rounded-xl border p-4 transition-colors {recommended
            ? 'border-emerald-500/60 bg-emerald-500/5'
            : 'border-slate-800 bg-slate-900/40'}"
        >
          <div class="flex items-baseline justify-between">
            <h3 class="font-medium">{p.label}</h3>
            <span class="font-mono text-sm text-slate-300"
              >{formatParams(paramCount(p))}</span
            >
          </div>
          <p class="mt-1 text-sm text-slate-400">{p.blurb}</p>
          <p class="mt-2 text-xs text-slate-500">
            {p.numLayers}L · {p.embedDim}d · {p.numHeads}h · ctx {p.blockSize}
          </p>
          <p class="mt-1 text-xs text-slate-600">{p.target}</p>
          {#if recommended}
            <span
              class="mt-3 inline-block rounded bg-emerald-500/15 px-2 py-0.5 text-xs font-medium text-emerald-400"
              >recommended for you</span
            >
          {/if}
        </div>
      {/each}
    </div>
  </section>

  <footer class="mt-12 text-xs text-slate-600">
    Phase 0 scaffold · design: <span class="font-mono">docs/menagerie-design.md</span>
  </footer>
</main>
