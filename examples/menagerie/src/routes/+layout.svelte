<script lang="ts">
import { onMount } from "svelte";
import "../app.css";
import Header from "$lib/components/Header.svelte";
import { session } from "$lib/hf/session.svelte";

const { children } = $props();

onMount(() => {
  // Consume an OAuth redirect (?code=...) or restore a persisted session.
  void session.init();
});
</script>

<div class="min-h-screen bg-slate-950 text-slate-100">
  <Header />
  {@render children()}
</div>
