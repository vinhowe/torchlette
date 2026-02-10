<script lang="ts">
  import { chatStore } from '$lib/stores/chat.svelte';
  import { modelStore } from '$lib/stores/model.svelte';

  let inputText = $state('');
  let messagesContainer: HTMLDivElement;

  async function handleSubmit(event: Event) {
    event.preventDefault();
    if (!inputText.trim() || !chatStore.canChat) return;

    const message = inputText;
    inputText = '';
    await chatStore.sendMessage(message);

    // Scroll to bottom after response
    setTimeout(() => {
      messagesContainer?.scrollTo({ top: messagesContainer.scrollHeight, behavior: 'smooth' });
    }, 100);
  }

  function handleKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  }
</script>

<div class="bg-slate-800 rounded-lg p-4 flex flex-col h-full">
  <div class="flex items-center justify-between mb-4">
    <h3 class="text-lg font-semibold text-white">Chat</h3>
    {#if chatStore.messages.length > 0}
      <button
        class="text-sm text-slate-400 hover:text-slate-200 transition-colors"
        onclick={() => chatStore.clearChat()}
      >
        Clear
      </button>
    {/if}
  </div>

  <!-- Generation config -->
  <div class="grid grid-cols-3 gap-2 mb-4">
    <div>
      <label class="block text-xs text-slate-400 mb-1">Temp: {chatStore.temperature.toFixed(1)}</label>
      <input
        type="range"
        min="0.1"
        max="2"
        step="0.1"
        bind:value={chatStore.temperature}
        class="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
      />
    </div>
    <div>
      <label class="block text-xs text-slate-400 mb-1">Tokens: {chatStore.maxTokens}</label>
      <input
        type="range"
        min="10"
        max="200"
        step="10"
        bind:value={chatStore.maxTokens}
        class="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
      />
    </div>
    <div>
      <label class="block text-xs text-slate-400 mb-1">Top-K: {chatStore.topK}</label>
      <input
        type="range"
        min="1"
        max="100"
        step="1"
        bind:value={chatStore.topK}
        class="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer"
      />
    </div>
  </div>

  <!-- Messages -->
  <div
    bind:this={messagesContainer}
    class="flex-1 overflow-y-auto space-y-3 min-h-[200px] max-h-[400px] mb-4"
  >
    {#if chatStore.messages.length === 0}
      <div class="flex items-center justify-center h-full">
        <p class="text-slate-500 text-sm">
          {#if modelStore.isLoaded}
            Start a conversation
          {:else}
            Load the model to chat
          {/if}
        </p>
      </div>
    {:else}
      {#each chatStore.messages as message}
        <div
          class="p-3 rounded-lg {message.role === 'user'
            ? 'bg-blue-900/50 ml-8'
            : 'bg-slate-700 mr-8'}"
        >
          <p class="text-xs text-slate-400 mb-1">
            {message.role === 'user' ? 'You' : 'GPT-2'}
          </p>
          <p class="text-slate-200 text-sm whitespace-pre-wrap">{message.content}</p>
        </div>
      {/each}

      <!-- Current streaming response -->
      {#if chatStore.isGenerating && chatStore.currentResponse}
        <div class="p-3 rounded-lg bg-slate-700 mr-8">
          <p class="text-xs text-slate-400 mb-1">GPT-2</p>
          <p class="text-slate-200 text-sm whitespace-pre-wrap">
            {chatStore.currentResponse}<span class="animate-pulse">▌</span>
          </p>
        </div>
      {/if}
    {/if}
  </div>

  <!-- Error display -->
  {#if chatStore.chatError}
    <div class="mb-4 p-2 bg-red-900/50 rounded-md border border-red-700">
      <p class="text-red-400 text-xs">{chatStore.chatError}</p>
    </div>
  {/if}

  <!-- Input -->
  <form onsubmit={handleSubmit} class="flex gap-2">
    <input
      type="text"
      bind:value={inputText}
      onkeydown={handleKeydown}
      placeholder={modelStore.isLoaded ? 'Type a message...' : 'Load model to chat'}
      disabled={!chatStore.canChat}
      class="flex-1 bg-slate-700 text-white rounded-md px-3 py-2 text-sm placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
    />
    <button
      type="submit"
      disabled={!chatStore.canChat || !inputText.trim()}
      class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {#if chatStore.isGenerating}
        <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
          <path
            class="opacity-75"
            fill="currentColor"
            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
          />
        </svg>
      {:else}
        Send
      {/if}
    </button>
  </form>
</div>
