<script lang="ts">
  import ActionButton from "./lib/components/buttons/ActionButton.svelte";
  import CapacityBar from "./lib/components/data/CapacityBar.svelte";
  import Note from "./lib/components/feedback/Note.svelte";
  import Statistic from "./lib/components/feedback/Statistic.svelte";
  import Panel from "./lib/components/primitives/Panel.svelte";
  import ThemeProvider from "./lib/components/theme/ThemeProvider.svelte";
  import ThemeToggle from "./lib/components/theme/ThemeToggle.svelte";
  import {
    createLocalEngine,
    LOCAL_MODELS,
    type EngineEvent,
    type LocalEngine,
    type TensorLoadEvent,
  } from "./lib/local-engine";
  import NetworkViz from "./lib/NetworkViz.svelte";

  type Message = { role: "user" | "assistant"; content: string };

  let engineMode = $state<"server" | "local">("server");
  let localModelId = $state(LOCAL_MODELS[1].id);
  let localEngine = $state<LocalEngine | null>(null);
  let loading = $state(false);
  let loadPct = $state(0);
  let loadStatus = $state("");
  let loadBytes = $state(0);
  let loadTotal = $state(0);
  let netTensors = $state<{ name: string; shape: number[]; elems: number; dtype: string; skipped: boolean }[]>([]);
  let netFill = $state<Record<string, number>>({});
  function onTensorEvent(ev: TensorLoadEvent) {
    if (ev.type === "manifest") netTensors.push(...ev.tensors);
    else if (ev.type === "start") netFill[ev.name] = 0;
    else if (ev.type === "progress") netFill[ev.name] = ev.fraction;
    else if (ev.type === "done") netFill[ev.name] = 1;
  }

  let messages = $state<Message[]>([]);
  let input = $state("");
  let busy = $state(false);
  let stats = $state<{
    tokPerSec: number;
    prefillMs: number;
    promptTokens: number;
    newTokens: number;
    decodeBreakdown?: { buildMs: number; lowerMs: number; fenceMs: number; sampleMs: number; stepMs: number };
  } | null>(null);
  let error = $state<string | null>(null);
  let listEl: HTMLElement | undefined;

  const scrollDown = () => {
    queueMicrotask(() => listEl?.scrollTo({ top: listEl.scrollHeight }));
  };

  async function loadLocal() {
    loading = true;
    error = null;
    netTensors = [];
    netFill = {};
    try {
      localEngine = await createLocalEngine(
        localModelId,
        (loaded, total, status) => {
          loadBytes = loaded;
          loadTotal = total;
          loadPct = total > 0 ? Math.min(100, (loaded / total) * 100) : 0;
          loadStatus = status;
        },
        onTensorEvent,
        (e) => (error = e),
      );
    } catch (e) {
      error = String(e);
      engineMode = "server";
    } finally {
      loading = false;
    }
  }

  function handleEvent(e: EngineEvent) {
    const last = messages[messages.length - 1];
    if ("delta" in e) last.content += e.delta;
    else if ("replace" in e) last.content = e.replace;
    else if ("error" in e) error = e.error;
    else if ("done" in e) stats = e.stats;
    scrollDown();
  }

  async function sendServer(history: Message[]) {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ messages: history }),
    });
    const reader = res.body!.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const events = buf.split("\n\n");
      buf = events.pop()!;
      for (const ev of events) {
        if (ev.startsWith("data: ")) handleEvent(JSON.parse(ev.slice(6)));
      }
    }
  }

  async function send() {
    const content = input.trim();
    if (!content || busy || loading) return;
    if (engineMode === "local" && !localEngine) return;
    input = "";
    error = null;
    messages.push({ role: "user", content });
    const history = messages.map((m) => ({ ...m }));
    messages.push({ role: "assistant", content: "" });
    busy = true;
    scrollDown();
    try {
      if (engineMode === "local") {
        await localEngine!.generate(history, handleEvent);
      } else {
        await sendServer(history);
      }
    } catch (e) {
      error = String(e);
    } finally {
      busy = false;
      if (messages[messages.length - 1]?.content === "") messages.pop();
    }
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  const engineReady = $derived(engineMode === "server" || localEngine !== null);
</script>

<ThemeProvider>
  <div class="fixed inset-0 flex flex-col overflow-hidden bg-background text-foreground">
    <header
      class="flex shrink-0 items-center justify-between border-b border-purple-300 bg-purple-200 px-2 py-1 text-purple-900 dark:border-purple-900 dark:bg-purple-950 dark:text-purple-200"
    >
      <div class="font-mono text-xs uppercase tracking-wider">qwen3-chat · torchlette</div>
      <ThemeToggle />
    </header>

    <div class="flex shrink-0 flex-wrap items-center gap-2 border-b border-border bg-background px-2 py-1">
      <div class="stack-tight">
        <span class="type-label text-subtle-foreground">Engine</span>
        <div class="inline-flex items-stretch border border-border bg-card">
          <button
            class="px-1.5 py-0.5 type-button {engineMode === 'server' ? 'bg-primary/12 text-primary' : 'text-muted-foreground hover:text-foreground'}"
            onclick={() => (engineMode = "server")}>Server</button
          >
          <button
            class="border-l border-border px-1.5 py-0.5 type-button {engineMode === 'local' ? 'bg-primary/12 text-primary' : 'text-muted-foreground hover:text-foreground'}"
            onclick={() => (engineMode = "local")}>In-browser</button
          >
        </div>
      </div>

      {#if engineMode === "local" && !localEngine}
        <div class="stack-tight">
          <span class="type-label text-subtle-foreground">Model</span>
          <div class="inline-flex items-stretch border border-border bg-card">
            {#each LOCAL_MODELS as m, i}
              <button
                class="{i > 0 ? 'border-l border-border ' : ''}px-1.5 py-0.5 type-button {localModelId === m.id ? 'bg-primary/12 text-primary' : 'text-muted-foreground hover:text-foreground'}"
                disabled={loading}
                onclick={() => (localModelId = m.id)}>{m.label}</button
              >
            {/each}
          </div>
        </div>
        <div class="self-end">
          <ActionButton color="green" disabled={loading} onclick={loadLocal}>
            {loading ? "Loading…" : "Load model"}
          </ActionButton>
        </div>
      {/if}

      {#if engineMode === "local" && localEngine}
        <span class="self-end pb-0.5 type-value text-success">{localEngine.modelId} · resident</span>
      {/if}

      <div class="ml-auto flex items-stretch gap-2">
        {#if stats}
          <Statistic label="tok/s">{stats.tokPerSec}</Statistic>
          <Statistic label="prefill">{stats.prefillMs}ms</Statistic>
          <Statistic label="tokens">{stats.promptTokens}+{stats.newTokens}</Statistic>
          {#if stats.decodeBreakdown}
            <!-- per-token ms: build graph · lower/encode · GPU fence · sample · markStep -->
            <Statistic label="ms/tok b·l·g·s·m">
              {stats.decodeBreakdown.buildMs}·{stats.decodeBreakdown.lowerMs}·{stats.decodeBreakdown.fenceMs}·{stats.decodeBreakdown.sampleMs}·{stats.decodeBreakdown.stepMs}
            </Statistic>
          {/if}
        {/if}
      </div>
    </div>

    {#if loading || netTensors.length > 0}
      <div class="shrink-0 border-b border-border px-2 py-1.5 stack-tight">
        {#if loading}
          <Panel title="Model download">
            <CapacityBar
              id="model-download"
              label="Weights"
              unit="MB"
              max={Math.max(1, Math.round(loadTotal / 1e6))}
              segments={[{ label: "downloaded", value: Math.round(loadBytes / 1e6) }]}
            />
            <Note label="Status">{loadStatus} ({loadPct.toFixed(0)}%)</Note>
          </Panel>
        {/if}
        {#if netTensors.length > 0}
          <NetworkViz tensors={netTensors} fill={netFill} />
        {/if}
      </div>
    {/if}

    <main bind:this={listEl} class="min-h-0 flex-1 overflow-y-auto overscroll-none px-2 py-1.5">
      <div class="mx-auto flex max-w-3xl flex-col stack-group">
        {#if messages.length === 0}
          <p class="type-caption text-muted-foreground">
            {engineMode === "local"
              ? localEngine
                ? "Model resident in this tab. Say something."
                : "Pick a model and load it — weights stream from Hugging Face into your GPU."
              : "Server mode: inference runs on the V100 box via SSE. Switch to In-browser to run the model in this tab."}
          </p>
        {/if}
        {#each messages as m, i}
          <div class="stack-tight">
            <span class="type-label {m.role === 'user' ? 'text-primary' : 'text-subtle-foreground'}">{m.role}</span>
            <p class="whitespace-pre-wrap type-body">{m.content}{#if busy && i === messages.length - 1 && m.role === "assistant"}<span class="text-primary">▌</span>{/if}</p>
          </div>
        {/each}
        {#if error}
          <Note label="Error" type="error">{error}</Note>
        {/if}
      </div>
    </main>

    <footer class="shrink-0 border-t border-border px-2 py-1.5">
      <div class="mx-auto flex max-w-3xl items-stretch gap-1">
        <div class="flex flex-1 items-stretch border border-border bg-card">
          <textarea
            bind:value={input}
            onkeydown={onKeydown}
            placeholder={busy ? "generating…" : engineReady ? "Message (Enter to send)" : "Load a model first"}
            disabled={busy || !engineReady}
            rows="2"
            class="flex-1 resize-none border-0 bg-card p-1 type-body outline-none placeholder:text-muted-foreground"
          ></textarea>
        </div>
        <ActionButton color="blue" disabled={busy || !input.trim() || !engineReady} onclick={send}>
          Send
        </ActionButton>
      </div>
    </footer>
  </div>
</ThemeProvider>
