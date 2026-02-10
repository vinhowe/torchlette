<script lang="ts">
  import { trainingStore } from '$lib/stores/training.svelte';
  import { modelStore } from '$lib/stores/model.svelte';

  let isDragging = $state(false);
  let fileInput: HTMLInputElement;

  async function handleFiles(files: FileList | null) {
    if (!files || !modelStore.tokenizer) return;

    for (const file of Array.from(files)) {
      if (file.name.endsWith('.txt') || file.type === 'text/plain') {
        const content = await file.text();
        trainingStore.addTrainingData(content, file.name);
      }
    }
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    isDragging = false;
    handleFiles(event.dataTransfer?.files ?? null);
  }

  function handleDragOver(event: DragEvent) {
    event.preventDefault();
    isDragging = true;
  }

  function handleDragLeave() {
    isDragging = false;
  }

  function handleClick() {
    fileInput?.click();
  }

  function handleFileInput(event: Event) {
    const input = event.target as HTMLInputElement;
    handleFiles(input.files);
    input.value = '';
  }
</script>

<div class="bg-slate-800 rounded-lg p-4">
  <h3 class="text-lg font-semibold text-white mb-3">Training Data</h3>

  <!-- Drop zone -->
  <button
    class="w-full border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer {isDragging
      ? 'border-blue-500 bg-blue-500/10'
      : 'border-slate-600 hover:border-slate-500'}"
    ondrop={handleDrop}
    ondragover={handleDragOver}
    ondragleave={handleDragLeave}
    onclick={handleClick}
    disabled={!modelStore.isLoaded}
  >
    <input
      bind:this={fileInput}
      type="file"
      accept=".txt,text/plain"
      multiple
      class="hidden"
      onchange={handleFileInput}
    />

    <svg
      class="w-10 h-10 mx-auto mb-3 text-slate-400"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        stroke-linecap="round"
        stroke-linejoin="round"
        stroke-width="2"
        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
      />
    </svg>

    {#if modelStore.isLoaded}
      <p class="text-slate-300">Drop .txt files here or click to browse</p>
      <p class="text-slate-500 text-sm mt-1">Files will be used to train your LoRA</p>
    {:else}
      <p class="text-slate-500">Load the model first to add training data</p>
    {/if}
  </button>

  <!-- File list -->
  {#if trainingStore.files.length > 0}
    <div class="mt-4 space-y-2">
      {#each trainingStore.files as file, index}
        <div class="flex items-center justify-between bg-slate-700 rounded-md px-3 py-2">
          <div class="flex items-center gap-2">
            <svg class="w-4 h-4 text-slate-400" fill="currentColor" viewBox="0 0 20 20">
              <path
                fill-rule="evenodd"
                d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"
                clip-rule="evenodd"
              />
            </svg>
            <span class="text-slate-200 text-sm">{file.name}</span>
            <span class="text-slate-500 text-xs">({file.tokens.toLocaleString()} tokens)</span>
          </div>
          <button
            class="text-slate-400 hover:text-red-400 transition-colors"
            onclick={() => trainingStore.removeTrainingData(index)}
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      {/each}
    </div>

    <div class="mt-3 flex justify-between items-center">
      <span class="text-slate-400 text-sm">
        Total: {trainingStore.totalTokens.toLocaleString()} tokens
      </span>
      <button
        class="text-sm text-slate-400 hover:text-slate-200 transition-colors"
        onclick={() => trainingStore.clearTrainingData()}
      >
        Clear all
      </button>
    </div>
  {/if}
</div>
