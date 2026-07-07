<script lang="ts">
import { session } from "$lib/hf/session.svelte";
import { isOAuthConfigured } from "$lib/hf/config";

let busy = $state(false);

async function login() {
  busy = true;
  try {
    await session.login();
  } finally {
    // If login() didn't redirect (misconfig), re-enable the button.
    busy = false;
  }
}
</script>

<header
  class="flex items-center justify-between border-b border-slate-800 px-6 py-3"
>
  <div class="flex items-center gap-5">
    <a href="/" class="text-lg font-semibold tracking-tight">Menagerie</a>
    <a href="/explore" class="text-sm text-slate-400 hover:text-slate-200">Explore</a>
    <a href="/create" class="text-sm text-slate-400 hover:text-slate-200">Create</a>
  </div>

  <div class="flex items-center gap-3 text-sm">
    {#if !session.ready}
      <span class="text-slate-500">…</span>
    {:else if session.loggedIn && session.user}
      <a
        href={session.user.profile}
        target="_blank"
        rel="noreferrer"
        class="flex items-center gap-2 hover:opacity-80"
      >
        <img
          src={session.user.picture}
          alt={session.username}
          class="h-7 w-7 rounded-full border border-slate-700"
        />
        <span class="font-mono">{session.username}</span>
      </a>
      <button
        class="rounded border border-slate-700 px-2 py-1 text-slate-400 hover:bg-slate-800"
        onclick={() => session.logout()}
      >
        logout
      </button>
    {:else}
      <button
        class="rounded-lg bg-amber-500 px-3 py-1.5 font-medium text-slate-900 hover:bg-amber-400 disabled:opacity-50"
        onclick={login}
        disabled={busy || !isOAuthConfigured()}
        title={isOAuthConfigured()
          ? "Sign in with Hugging Face"
          : "Set VITE_HF_OAUTH_CLIENT_ID to enable login"}
      >
        Login with Hugging Face
      </button>
    {/if}
  </div>
</header>

{#if session.error}
  <div class="border-b border-rose-900/50 bg-rose-950/40 px-6 py-2 text-sm text-rose-300">
    {session.error}
  </div>
{/if}
