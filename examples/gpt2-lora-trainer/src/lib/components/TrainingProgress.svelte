<script lang="ts">
  import { trainingStore } from '$lib/stores/training.svelte';

  // Simple loss chart using SVG
  const chartWidth = 300;
  const chartHeight = 100;

  const chartPath = $derived(() => {
    const history = trainingStore.lossHistory;
    if (history.length < 2) return '';

    const maxLoss = Math.max(...history, 0.1);
    const minLoss = Math.min(...history, 0);
    const range = maxLoss - minLoss || 1;

    const points = history.map((loss, i) => {
      const x = (i / (history.length - 1)) * chartWidth;
      const y = chartHeight - ((loss - minLoss) / range) * chartHeight;
      return `${x},${y}`;
    });

    return `M ${points.join(' L ')}`;
  });

  const progressPercent = $derived(
    trainingStore.maxSteps > 0 ? (trainingStore.currentStep / trainingStore.maxSteps) * 100 : 0
  );
</script>

<div class="bg-slate-800 rounded-lg p-4">
  <div class="flex items-center justify-between mb-4">
    <h3 class="text-lg font-semibold text-white">Training Progress</h3>
    {#if trainingStore.isTraining}
      <button
        class="px-3 py-1 bg-red-600 hover:bg-red-700 text-white rounded-md text-sm font-medium transition-colors"
        onclick={() => trainingStore.stopTraining()}
      >
        Stop
      </button>
    {:else if trainingStore.canStartTraining}
      <button
        class="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded-md text-sm font-medium transition-colors"
        onclick={() => trainingStore.startTraining()}
      >
        Start Training
      </button>
    {/if}
  </div>

  <!-- Progress bar -->
  <div class="mb-4">
    <div class="flex justify-between text-sm text-slate-400 mb-1">
      <span>Step {trainingStore.currentStep} / {trainingStore.maxSteps}</span>
      <span>{progressPercent.toFixed(0)}%</span>
    </div>
    <div class="w-full bg-slate-700 rounded-full h-2">
      <div
        class="bg-green-500 h-2 rounded-full transition-all duration-300"
        style="width: {progressPercent}%"
      ></div>
    </div>
  </div>

  <!-- Stats -->
  <div class="grid grid-cols-2 gap-4 mb-4">
    <div class="bg-slate-700 rounded-md p-3">
      <p class="text-slate-400 text-xs uppercase mb-1">Loss</p>
      <p class="text-white text-lg font-mono">
        {trainingStore.currentLoss > 0 ? trainingStore.currentLoss.toFixed(4) : '—'}
      </p>
    </div>
    <div class="bg-slate-700 rounded-md p-3">
      <p class="text-slate-400 text-xs uppercase mb-1">Tokens/sec</p>
      <p class="text-white text-lg font-mono">
        {trainingStore.tokensPerSecond > 0 ? trainingStore.tokensPerSecond.toFixed(0) : '—'}
      </p>
    </div>
  </div>

  <!-- Loss chart -->
  {#if trainingStore.lossHistory.length > 1}
    <div class="bg-slate-700 rounded-md p-3">
      <p class="text-slate-400 text-xs uppercase mb-2">Loss History</p>
      <svg
        viewBox="0 0 {chartWidth} {chartHeight}"
        class="w-full h-24"
        preserveAspectRatio="none"
      >
        <path d={chartPath()} fill="none" stroke="#22c55e" stroke-width="2" />
      </svg>
    </div>
  {/if}

  <!-- Error display -->
  {#if trainingStore.trainingError}
    <div class="mt-4 p-3 bg-red-900/50 rounded-md border border-red-700">
      <p class="text-red-400 text-sm">{trainingStore.trainingError}</p>
    </div>
  {/if}

  <!-- Not ready message -->
  {#if !trainingStore.canStartTraining && !trainingStore.isTraining}
    <div class="mt-4 p-3 bg-slate-700 rounded-md">
      <p class="text-slate-400 text-sm">
        {#if trainingStore.files.length === 0}
          Add training files to begin
        {:else if trainingStore.totalTokens < trainingStore.batchSize * (trainingStore.seqLength + 1)}
          Need at least {trainingStore.batchSize * (trainingStore.seqLength + 1)} tokens
          (have {trainingStore.totalTokens})
        {:else}
          Load the model first
        {/if}
      </p>
    </div>
  {/if}
</div>
