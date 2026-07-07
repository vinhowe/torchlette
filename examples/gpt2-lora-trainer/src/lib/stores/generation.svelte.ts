/**
 * Generation store -- prompt, streaming output, sampling config.
 */

import {
  type GenerateOptions,
  generateTokens,
} from "gpt2-browser";
import { modelStore } from "./model.svelte";

let prompt = $state("Once upon a time");
let output = $state("");
let temperature = $state(0.8);
let topK = $state(40);
let maxTokens = $state(200);
let isGenerating = $state(false);
const stopSignal = { value: false };
let error = $state("");

const canGenerate = $derived(modelStore.isReady && !isGenerating);

async function generate(): Promise<void> {
  if (
    !canGenerate ||
    !modelStore.api ||
    !modelStore.model ||
    !modelStore.tokenizer
  )
    return;
  if (!prompt.trim()) return;

  isGenerating = true;
  stopSignal.value = false;
  output = "";
  error = "";

  try {
    const options: GenerateOptions = {
      maxNewTokens: maxTokens,
      temperature,
      topK,
    };

    for await (const token of generateTokens(
      modelStore.api,
      modelStore.model,
      modelStore.tokenizer,
      prompt,
      options,
    )) {
      if (stopSignal.value) break;
      output += token;
    }
  } catch (e) {
    error = e instanceof Error ? e.message : "Generation failed";
  } finally {
    isGenerating = false;
  }
}

function stopGenerate(): void {
  stopSignal.value = true;
}

export const generationStore = {
  get prompt() {
    return prompt;
  },
  set prompt(v: string) {
    prompt = v;
  },
  get output() {
    return output;
  },
  get temperature() {
    return temperature;
  },
  set temperature(v: number) {
    temperature = v;
  },
  get topK() {
    return topK;
  },
  set topK(v: number) {
    topK = v;
  },
  get maxTokens() {
    return maxTokens;
  },
  set maxTokens(v: number) {
    maxTokens = v;
  },
  get isGenerating() {
    return isGenerating;
  },
  get error() {
    return error;
  },
  get canGenerate() {
    return canGenerate;
  },

  generate,
  stopGenerate,
};
