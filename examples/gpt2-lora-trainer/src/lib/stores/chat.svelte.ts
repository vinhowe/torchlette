/**
 * Chat state store using Svelte 5 runes.
 */

import { generateTokens, type GenerateOptions } from '$lib/torchlette/inference';
import { modelStore } from './model.svelte';

export type Message = {
  role: 'user' | 'assistant';
  content: string;
};

// Chat state
let messages = $state<Message[]>([]);
let isGenerating = $state(false);
let currentResponse = $state('');
let chatError = $state<string | null>(null);

// Generation config
let temperature = $state(0.7);
let maxTokens = $state(100);
let topK = $state(50);

// Derived
const canChat = $derived(modelStore.isLoaded && !isGenerating);

/**
 * Add a message to the chat.
 */
function addMessage(role: 'user' | 'assistant', content: string): void {
  messages = [...messages, { role, content }];
}

/**
 * Clear chat history.
 */
function clearChat(): void {
  messages = [];
  currentResponse = '';
  chatError = null;
}

/**
 * Send a message and generate a response.
 */
async function sendMessage(userMessage: string): Promise<void> {
  if (!canChat || !userMessage.trim()) return;
  if (!modelStore.api || !modelStore.model || !modelStore.tokenizer) return;

  // Add user message
  addMessage('user', userMessage);

  isGenerating = true;
  currentResponse = '';
  chatError = null;

  try {
    // Build prompt from conversation history
    const prompt = buildPrompt(userMessage);

    const options: GenerateOptions = {
      maxNewTokens: maxTokens,
      temperature,
      topK,
      stopSequences: ['\n\nUser:', '\n\nHuman:'],
    };

    // Generate response token by token
    for await (const token of generateTokens(
      modelStore.api,
      modelStore.model,
      modelStore.tokenizer,
      prompt,
      options
    )) {
      currentResponse += token;
    }

    // Add assistant message
    addMessage('assistant', currentResponse.trim());
    currentResponse = '';
  } catch (e) {
    chatError = e instanceof Error ? e.message : 'Generation failed';
  } finally {
    isGenerating = false;
  }
}

/**
 * Build prompt from conversation history.
 */
function buildPrompt(userMessage: string): string {
  // Simple prompt format
  let prompt = '';

  // Include recent messages for context (last 4 exchanges)
  const recentMessages = messages.slice(-8);
  for (const msg of recentMessages) {
    if (msg.role === 'user') {
      prompt += `User: ${msg.content}\n`;
    } else {
      prompt += `Assistant: ${msg.content}\n`;
    }
  }

  // Add current message
  prompt += `User: ${userMessage}\nAssistant:`;

  return prompt;
}

// Export store
export const chatStore = {
  // Getters
  get messages() { return messages; },
  get isGenerating() { return isGenerating; },
  get currentResponse() { return currentResponse; },
  get chatError() { return chatError; },
  get temperature() { return temperature; },
  get maxTokens() { return maxTokens; },
  get topK() { return topK; },
  get canChat() { return canChat; },

  // Setters
  set temperature(v: number) { temperature = v; },
  set maxTokens(v: number) { maxTokens = v; },
  set topK(v: number) { topK = v; },

  // Actions
  addMessage,
  clearChat,
  sendMessage,
};
