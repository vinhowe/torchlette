<script lang="ts">
  import { modelStore } from '$lib/stores/model.svelte';
  import { trainingStore } from '$lib/stores/training.svelte';
</script>

<div class="bg-slate-800 rounded-lg p-4">
  <h3 class="text-lg font-semibold text-white mb-4">Training Configuration</h3>

  <div class="space-y-4">
    <!-- LoRA Rank -->
    <div>
      <label class="block text-sm text-slate-300 mb-1">
        LoRA Rank: {modelStore.loraRank}
      </label>
      <input
        type="range"
        min="2"
        max="32"
        step="2"
        bind:value={modelStore.loraRank}
        class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
        disabled={modelStore.isLoaded || trainingStore.isTraining}
      />
      <p class="text-xs text-slate-500 mt-1">Lower = smaller adapter, higher = more capacity</p>
    </div>

    <!-- LoRA Alpha -->
    <div>
      <label class="block text-sm text-slate-300 mb-1">
        LoRA Alpha: {modelStore.loraAlpha}
      </label>
      <input
        type="range"
        min="4"
        max="64"
        step="4"
        bind:value={modelStore.loraAlpha}
        class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
        disabled={modelStore.isLoaded || trainingStore.isTraining}
      />
      <p class="text-xs text-slate-500 mt-1">Scaling factor (typically equal to rank)</p>
    </div>

    <hr class="border-slate-700" />

    <!-- Training Steps -->
    <div>
      <label class="block text-sm text-slate-300 mb-1">
        Training Steps: {trainingStore.maxSteps}
      </label>
      <input
        type="range"
        min="10"
        max="500"
        step="10"
        bind:value={trainingStore.maxSteps}
        class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
        disabled={trainingStore.isTraining}
      />
    </div>

    <!-- Batch Size -->
    <div>
      <label class="block text-sm text-slate-300 mb-1">
        Batch Size: {trainingStore.batchSize}
      </label>
      <input
        type="range"
        min="1"
        max="4"
        step="1"
        bind:value={trainingStore.batchSize}
        class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
        disabled={trainingStore.isTraining}
      />
      <p class="text-xs text-slate-500 mt-1">Keep low to reduce GPU memory usage</p>
    </div>

    <!-- Sequence Length -->
    <div>
      <label class="block text-sm text-slate-300 mb-1">
        Sequence Length: {trainingStore.seqLength}
      </label>
      <input
        type="range"
        min="16"
        max="128"
        step="16"
        bind:value={trainingStore.seqLength}
        class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
        disabled={trainingStore.isTraining}
      />
      <p class="text-xs text-slate-500 mt-1">Shorter = less memory, longer = better context</p>
    </div>

    <!-- Learning Rate -->
    <div>
      <label class="block text-sm text-slate-300 mb-1">
        Learning Rate: {trainingStore.learningRate.toExponential(0)}
      </label>
      <input
        type="range"
        min="-5"
        max="-3"
        step="0.5"
        value={Math.log10(trainingStore.learningRate)}
        oninput={(e) => (trainingStore.learningRate = Math.pow(10, parseFloat((e.target as HTMLInputElement).value)))}
        class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
        disabled={trainingStore.isTraining}
      />
    </div>

    <hr class="border-slate-700" />

    <!-- Memory Optimization Section -->
    <div class="space-y-3">
      <h4 class="text-sm font-medium text-slate-400">Memory Optimization</h4>

      <!-- AMP Toggle -->
      <div class="flex items-center justify-between">
        <div>
          <span class="text-sm text-slate-300">Mixed Precision (AMP)</span>
          <p class="text-xs text-slate-500">Use f16 for compute, reduces memory ~50%</p>
        </div>
        <button
          class="relative inline-flex h-6 w-11 items-center rounded-full transition-colors {trainingStore.useAMP ? 'bg-blue-600' : 'bg-slate-600'}"
          onclick={() => (trainingStore.useAMP = !trainingStore.useAMP)}
          disabled={trainingStore.isTraining}
        >
          <span
            class="inline-block h-4 w-4 transform rounded-full bg-white transition-transform {trainingStore.useAMP ? 'translate-x-6' : 'translate-x-1'}"
          ></span>
        </button>
      </div>

      <!-- Checkpointing Toggle -->
      <div class="flex items-center justify-between">
        <div>
          <span class="text-sm text-slate-300">Gradient Checkpointing</span>
          <p class="text-xs text-slate-500">Trade compute for memory, ~2x slower</p>
        </div>
        <button
          class="relative inline-flex h-6 w-11 items-center rounded-full transition-colors {trainingStore.useCheckpointing ? 'bg-blue-600' : 'bg-slate-600'}"
          onclick={() => (trainingStore.useCheckpointing = !trainingStore.useCheckpointing)}
          disabled={trainingStore.isTraining}
        >
          <span
            class="inline-block h-4 w-4 transform rounded-full bg-white transition-transform {trainingStore.useCheckpointing ? 'translate-x-6' : 'translate-x-1'}"
          ></span>
        </button>
      </div>
    </div>
  </div>
</div>
