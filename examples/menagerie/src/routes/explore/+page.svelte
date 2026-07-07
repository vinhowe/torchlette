<script lang="ts">
import { onMount } from "svelte";
import { listEcologyModels } from "$lib/hf/repo";
import type { EcologyModel } from "$lib/hf/types";

let models = $state<EcologyModel[]>([]);
let loading = $state(true);
let error = $state<string | null>(null);

onMount(async () => {
  try {
    models = await listEcologyModels();
  } catch (e) {
    error = (e as Error).message;
  } finally {
    loading = false;
  }
});
</script>

<main class="mx-auto max-w-5xl px-6 py-10">
  <div class="mb-1 flex items-center justify-between">
    <h1 class="text-2xl font-semibold">Explore the ecology</h1>
    <a
      href="/create"
      class="rounded-lg bg-emerald-500 px-3 py-1.5 text-sm font-medium text-slate-900 hover:bg-emerald-400"
      >+ Create</a
    >
  </div>
  <p class="mb-6 text-sm text-slate-400">
    Every model tagged <span class="font-mono">menagerie</span> on the Hub. Fork
    one to add a branch to the tree.
  </p>

  {#if loading}
    <p class="text-slate-400">Loading from the Hub…</p>
  {:else if error}
    <p class="text-rose-400">{error}</p>
  {:else if models.length === 0}
    <div class="rounded-xl border border-slate-800 bg-slate-900/40 p-6 text-slate-400">
      <p>No models in the ecology yet — it's brand new.</p>
      <p class="mt-2 text-sm text-slate-500">
        Forking (phase 2) seeds the first ones. Until then this is an honest
        empty tree.
      </p>
    </div>
  {:else}
    <div class="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {#each models as m (m.repo)}
        <a
          href={`/model?repo=${encodeURIComponent(m.repo)}`}
          class="rounded-xl border border-slate-800 bg-slate-900/40 p-4 hover:border-slate-600"
        >
          <div class="font-mono text-sm text-slate-200">{m.repo}</div>
          <div class="mt-2 flex gap-4 text-xs text-slate-500">
            {#if m.likes}<span>♥ {m.likes}</span>{/if}
            {#if m.downloads}<span>↓ {m.downloads}</span>{/if}
          </div>
        </a>
      {/each}
    </div>
  {/if}
</main>
