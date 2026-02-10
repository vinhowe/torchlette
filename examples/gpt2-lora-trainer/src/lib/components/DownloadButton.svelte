<script lang="ts">
  import { trainingStore } from '$lib/stores/training.svelte';
</script>

{#if trainingStore.loraWeights}
  <div class="bg-slate-800 rounded-lg p-4">
    <div class="flex items-center gap-3">
      <div class="flex-shrink-0 w-10 h-10 bg-green-600 rounded-full flex items-center justify-center">
        <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
      </div>
      <div class="flex-1">
        <p class="text-white font-medium">Training Complete!</p>
        <p class="text-slate-400 text-sm">
          Final loss: {trainingStore.currentLoss.toFixed(4)}
        </p>
      </div>
      <button
        class="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm font-medium transition-colors flex items-center gap-2"
        onclick={() => trainingStore.downloadLoRA()}
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
          />
        </svg>
        Download LoRA
      </button>
    </div>
  </div>
{/if}
