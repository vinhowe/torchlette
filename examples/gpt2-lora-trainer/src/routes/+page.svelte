<script lang="ts">
  import FileDropZone from '$lib/components/FileDropZone.svelte';
  import TrainingProgress from '$lib/components/TrainingProgress.svelte';
  import LoRAConfig from '$lib/components/LoRAConfig.svelte';
  import ChatInterface from '$lib/components/ChatInterface.svelte';
  import ModelStatus from '$lib/components/ModelStatus.svelte';
  import DownloadButton from '$lib/components/DownloadButton.svelte';
</script>

<div class="container mx-auto p-6 max-w-7xl">
  <header class="mb-8">
    <h1 class="text-4xl font-bold text-white mb-2">GPT-2 LoRA Trainer</h1>
    <p class="text-slate-400">Train a LoRA adapter on your text data, entirely in the browser using WebGPU</p>
  </header>

  <ModelStatus />

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
    <!-- Left Column: Training -->
    <div class="space-y-6">
      <LoRAConfig />
      <FileDropZone />
      <TrainingProgress />
      <DownloadButton />
    </div>

    <!-- Right Column: Chat -->
    <div>
      <ChatInterface />
    </div>
  </div>

  <footer class="mt-12 text-center text-slate-500 text-sm">
    <p>Powered by <a href="https://github.com/anthropics/torchlette" class="text-blue-400 hover:underline">Torchlette</a> - WebGPU Tensor Library</p>
  </footer>
</div>
