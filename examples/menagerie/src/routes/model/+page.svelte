<script lang="ts">
import { onMount } from "svelte";
import { page } from "$app/state";
import { fetchModelConfig, fetchLineage, fetchCommits } from "$lib/hf/repo";
import type { ModelConfig, Lineage } from "$lib/hf/types";

let repo = $state<string | null>(null);
let config = $state<ModelConfig | null>(null);
let lineage = $state<Lineage | null>(null);
let commits = $state<Array<{ id: string; title: string; date: string }>>([]);
let loading = $state(true);
let error = $state<string | null>(null);

onMount(async () => {
  repo = page.url.searchParams.get("repo");
  if (!repo) {
    error = "No ?repo= specified.";
    loading = false;
    return;
  }
  try {
    [config, lineage, commits] = await Promise.all([
      fetchModelConfig(repo),
      fetchLineage(repo),
      fetchCommits(repo).catch(() => []),
    ]);
  } catch (e) {
    error = (e as Error).message;
  } finally {
    loading = false;
  }
});
</script>

<main class="mx-auto max-w-4xl px-6 py-10">
  <a href="/explore" class="text-sm text-slate-500 hover:text-slate-300">← explore</a>

  {#if loading}
    <p class="mt-4 text-slate-400">Loading {repo}…</p>
  {:else if error}
    <p class="mt-4 text-rose-400">{error}</p>
  {:else}
    <div class="mt-2 flex items-center justify-between gap-4">
      <h1 class="font-mono text-2xl font-semibold">{repo}</h1>
      {#if config}
        <div class="flex shrink-0 gap-2">
          <a
            href={`/train?repo=${encodeURIComponent(repo!)}`}
            class="rounded-lg border border-emerald-600 px-3 py-1.5 text-sm font-medium text-emerald-400 hover:bg-emerald-950/40"
            >Train</a
          >
          <a
            href={`/create?fork=${encodeURIComponent(repo!)}`}
            class="rounded-lg bg-emerald-500 px-3 py-1.5 text-sm font-medium text-slate-900 hover:bg-emerald-400"
            >Fork</a
          >
        </div>
      {/if}
    </div>
    {#if !config && !lineage}
      <p class="mt-4 text-amber-300">
        This repo has no <span class="font-mono">config.json</span> /
        <span class="font-mono">lineage.json</span> — it isn't a Menagerie model
        (or hasn't been initialized yet).
      </p>
    {/if}

    {#if config}
      <section class="mt-6">
        <h2 class="mb-2 text-lg font-medium">Architecture</h2>
        <div class="grid grid-cols-2 gap-x-8 gap-y-1 text-sm sm:grid-cols-3">
          <div class="flex justify-between border-b border-slate-800 py-1">
            <span class="text-slate-400">preset</span><span class="font-mono">{config.arch_preset}</span>
          </div>
          <div class="flex justify-between border-b border-slate-800 py-1">
            <span class="text-slate-400">layers</span><span class="font-mono">{config.numLayers}</span>
          </div>
          <div class="flex justify-between border-b border-slate-800 py-1">
            <span class="text-slate-400">embed</span><span class="font-mono">{config.embedDim}</span>
          </div>
          <div class="flex justify-between border-b border-slate-800 py-1">
            <span class="text-slate-400">heads</span><span class="font-mono">{config.numHeads}</span>
          </div>
          <div class="flex justify-between border-b border-slate-800 py-1">
            <span class="text-slate-400">ctx</span><span class="font-mono">{config.blockSize}</span>
          </div>
          <div class="flex justify-between border-b border-slate-800 py-1">
            <span class="text-slate-400">dtype</span><span class="font-mono">{config.dtype}</span>
          </div>
        </div>
      </section>
    {/if}

    {#if lineage}
      <section class="mt-6">
        <h2 class="mb-2 text-lg font-medium">Lineage</h2>
        <div class="text-sm text-slate-300">
          <div>
            root: <a class="font-mono underline" href={`/model?repo=${encodeURIComponent(lineage.root_repo)}`}>{lineage.root_repo}</a>
          </div>
          <div class="mt-1">
            parent:
            {#if lineage.parent}
              <a class="font-mono underline" href={`/model?repo=${encodeURIComponent(lineage.parent.repo)}`}>{lineage.parent.repo}</a>
              <span class="text-slate-500">@ {lineage.parent.commit_sha.slice(0, 8)}</span>
            {:else}
              <span class="text-slate-500">none (from-scratch root)</span>
            {/if}
          </div>
        </div>

        <h3 class="mb-2 mt-4 text-sm font-medium text-slate-400">History</h3>
        <ol class="space-y-2">
          {#each lineage.checkpoints.slice().reverse() as ck}
            <li class="rounded-lg border border-slate-800 bg-slate-900/40 p-3 text-sm">
              <div class="flex items-center justify-between">
                <span class="rounded bg-slate-800 px-2 py-0.5 font-mono text-xs">{ck.op}</span>
                <span class="text-slate-500">{ck.created_by} · {new Date(ck.created_at).toLocaleString()}</span>
              </div>
              <div class="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400">
                {#if ck.steps != null}<span>{ck.steps} steps</span>{/if}
                {#if ck.tokens != null}<span>{ck.tokens.toLocaleString()} tokens</span>{/if}
                {#if ck.wallclock_ms != null}<span>{(ck.wallclock_ms / 1000).toFixed(0)}s</span>{/if}
                {#if ck.dtype}<span>{ck.dtype}</span>{/if}
              </div>
              {#if ck.diet?.length}
                <div class="mt-1 text-xs text-slate-500">
                  diet: {ck.diet.map((d) => (d.kind === "hf-dataset" ? d.dataset : d.path)).join(", ")}
                </div>
              {/if}
            </li>
          {/each}
        </ol>
      </section>
    {/if}

    {#if commits.length}
      <section class="mt-6">
        <h2 class="mb-2 text-lg font-medium">Commits</h2>
        <ol class="space-y-1 text-sm">
          {#each commits.slice(0, 12) as c}
            <li class="flex gap-3">
              <span class="font-mono text-slate-500">{c.id.slice(0, 8)}</span>
              <span class="text-slate-300">{c.title}</span>
            </li>
          {/each}
        </ol>
      </section>
    {/if}
  {/if}
</main>
