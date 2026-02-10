<script lang="ts">
  import { modelStore } from '$lib/stores/model.svelte';
  import { getCacheInfo, clearCache } from '$lib/torchlette/weights';
  import { onMount } from 'svelte';

  let cacheInfo = $state<{ hasWeights: boolean; hasTokenizer: boolean; weightsSize: number } | null>(null);

  onMount(() => {
    modelStore.checkWebGPU();
    checkCache();
  });

  async function checkCache() {
    cacheInfo = await getCacheInfo();
  }

  async function handleClearCache() {
    await clearCache();
    cacheInfo = await getCacheInfo();
  }

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }
</script>

<div class="bg-slate-800 rounded-lg p-4 mb-6">
  <div class="flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div
        class="w-3 h-3 rounded-full {modelStore.isLoaded
          ? 'bg-green-500'
          : modelStore.isLoading
            ? 'bg-yellow-500 animate-pulse'
            : 'bg-slate-500'}"
      ></div>
      <span class="text-slate-200 font-medium">
        {#if modelStore.isLoaded}
          Model Ready
        {:else if modelStore.isLoading}
          Loading Model...
        {:else}
          Model Not Loaded
        {/if}
      </span>
      {#if cacheInfo?.hasWeights && !modelStore.isLoaded && !modelStore.isLoading}
        <span class="text-xs text-green-400 bg-green-900/30 px-2 py-0.5 rounded">Cached</span>
      {/if}
    </div>

    <div class="flex items-center gap-2">
      {#if cacheInfo?.hasWeights && !modelStore.isLoading}
        <button
          class="px-3 py-1.5 text-slate-400 hover:text-slate-200 text-xs transition-colors"
          onclick={handleClearCache}
          title="Clear cached model ({formatBytes(cacheInfo.weightsSize)})"
        >
          Clear Cache
        </button>
      {/if}
      {#if !modelStore.isLoaded && !modelStore.isLoading}
        <button
          class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          onclick={() => modelStore.loadModel()}
          disabled={modelStore.webgpuSupported === false}
        >
          {cacheInfo?.hasWeights ? 'Load from Cache' : 'Download Model'}
        </button>
      {/if}
    </div>
  </div>

  {#if modelStore.isLoading}
    <div class="mt-4">
      <div class="flex justify-between text-sm text-slate-400 mb-1">
        <span>{modelStore.loadStatus}</span>
        <span>{modelStore.loadProgress}%</span>
      </div>
      <div class="w-full bg-slate-700 rounded-full h-2">
        <div
          class="bg-blue-500 h-2 rounded-full transition-all duration-300"
          style="width: {modelStore.loadProgress}%"
        ></div>
      </div>
    </div>
  {/if}

  {#if modelStore.error}
    <div class="mt-3 p-3 bg-red-900/50 rounded-md border border-red-700">
      <p class="text-red-400 text-sm">{modelStore.error}</p>
    </div>
  {/if}

  {#if modelStore.webgpuSupported === false}
    <div class="mt-3 p-3 bg-yellow-900/50 rounded-md border border-yellow-700">
      <p class="text-yellow-400 text-sm">
        WebGPU is not supported in this browser. Please use Chrome 113+, Edge 113+, or another WebGPU-enabled browser.
      </p>
    </div>
  {/if}
</div>
