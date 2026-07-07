<script lang="ts">
import { onMount } from "svelte";
import { goto } from "$app/navigation";
import { page } from "$app/state";
import { session } from "$lib/hf/session.svelte";
import { ADOPTABLE, adoptCheckpoint, forkModel } from "$lib/ecology";
import { fetchCommits } from "$lib/hf/repo";

type Mode = "adopt" | "fork";

let mode = $state<Mode>("adopt");
let forkParent = $state<string | null>(null);
let forkSha = $state<string | null>(null);
let resolvingSha = $state(false);

let sourceId = $state(ADOPTABLE[0].id);
let name = $state("");
let busy = $state(false);
let error = $state<string | null>(null);

function suggestName(base: string): string {
  const slug = base.split("/").pop()!.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
  return `${slug}-${Math.random().toString(36).slice(2, 6)}`;
}

onMount(async () => {
  const fork = page.url.searchParams.get("fork");
  if (fork) {
    mode = "fork";
    forkParent = fork;
    name = suggestName(fork);
    resolvingSha = true;
    try {
      const commits = await fetchCommits(fork);
      forkSha = commits[0]?.id ?? null;
      if (!forkSha) error = "Could not resolve the parent's latest commit.";
    } catch (e) {
      error = `Could not resolve parent commit: ${(e as Error).message}`;
    } finally {
      resolvingSha = false;
    }
  } else {
    name = suggestName(sourceId);
  }
});

async function submit() {
  error = null;
  if (!session.loggedIn || !session.username || !session.sub || !session.accessToken) {
    error = "Please log in first.";
    return;
  }
  if (!name.trim()) {
    error = "Pick a name.";
    return;
  }
  busy = true;
  try {
    let repo: string;
    if (mode === "fork") {
      if (!forkParent || !forkSha) throw new Error("No parent/commit to fork.");
      repo = await forkModel({
        parentRepo: forkParent,
        parentCommitSha: forkSha,
        owner: session.username,
        newName: name.trim(),
        username: session.username,
        sub: session.sub,
        accessToken: session.accessToken,
      });
    } else {
      const source = ADOPTABLE.find((s) => s.id === sourceId)!;
      repo = await adoptCheckpoint({
        source,
        owner: session.username,
        newName: name.trim(),
        username: session.username,
        sub: session.sub,
        accessToken: session.accessToken,
      });
    }
    await goto(`/model?repo=${encodeURIComponent(repo)}`);
  } catch (e) {
    error = (e as Error).message;
  } finally {
    busy = false;
  }
}
</script>

<main class="mx-auto max-w-xl px-6 py-10">
  <h1 class="mb-1 text-2xl font-semibold">
    {mode === "fork" ? "Fork a model" : "Adopt a root"}
  </h1>
  <p class="mb-6 text-sm text-slate-400">
    {#if mode === "fork"}
      Creates a new repo in your namespace, seeded from the parent's pinned
      snapshot. Provenance records the parent + commit.
    {:else}
      Bootstrap a new lineage from a known checkpoint. It becomes a root of the
      ecology under your identity. (From-scratch random init arrives with the
      trainer.)
    {/if}
  </p>

  {#if !session.loggedIn}
    <div class="rounded-lg border border-amber-900/50 bg-amber-950/30 px-4 py-3 text-sm text-amber-200">
      Log in with Hugging Face (top right) to create models in your namespace.
    </div>
  {:else}
    {#if mode === "fork"}
      <div class="mb-4 rounded-lg border border-slate-800 bg-slate-900/40 p-3 text-sm">
        <div>parent: <span class="font-mono">{forkParent}</span></div>
        <div class="mt-1 text-slate-400">
          commit:
          {#if resolvingSha}resolving…{:else if forkSha}<span class="font-mono">{forkSha.slice(0, 12)}</span>{:else}<span class="text-rose-400">unresolved</span>{/if}
        </div>
      </div>
    {:else}
      <label class="mb-4 block">
        <span class="mb-1 block text-sm text-slate-400">Source checkpoint</span>
        <select
          bind:value={sourceId}
          class="w-full rounded-lg border border-slate-700 bg-slate-900 px-3 py-2"
        >
          {#each ADOPTABLE as s (s.id)}
            <option value={s.id}>{s.label} — {s.id}</option>
          {/each}
        </select>
      </label>
    {/if}

    <label class="mb-4 block">
      <span class="mb-1 block text-sm text-slate-400">New repo name</span>
      <div class="flex items-center gap-2">
        <span class="font-mono text-slate-500">{session.username}/</span>
        <input
          bind:value={name}
          class="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 font-mono"
          placeholder="my-creature"
        />
      </div>
    </label>

    {#if error}
      <p class="mb-3 text-sm text-rose-400">{error}</p>
    {/if}

    <button
      class="rounded-lg bg-emerald-500 px-4 py-2 font-medium text-slate-900 hover:bg-emerald-400 disabled:opacity-50"
      onclick={submit}
      disabled={busy || (mode === "fork" && !forkSha)}
    >
      {busy ? "Working…" : mode === "fork" ? "Fork" : "Adopt"}
    </button>
    <p class="mt-3 text-xs text-slate-500">
      This commits a public repo to your Hugging Face account (weights become
      public under your identity).
    </p>
  {/if}
</main>
