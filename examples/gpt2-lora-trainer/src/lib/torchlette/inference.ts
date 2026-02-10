/**
 * Text generation/inference for GPT-2 with LoRA.
 */

import type { FrontendTensor as Tensor, Torchlette } from 'torchlette';
import type { GPT2WithLoRA } from './gpt2-lora';
import type { GPT2Tokenizer } from './tokenizer';

export type GenerateOptions = {
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  stopSequences?: string[];
};

/**
 * Generate text token by token.
 *
 * Yields each generated token as a string for streaming output.
 */
export async function* generateTokens(
  api: Torchlette,
  model: GPT2WithLoRA,
  tokenizer: GPT2Tokenizer,
  prompt: string,
  options: GenerateOptions = {}
): AsyncGenerator<string, void, unknown> {
  const {
    maxNewTokens = 100,
    temperature = 0.7,
    topK = 50,
    stopSequences = ['\n\n'],
  } = options;

  // Tokenize prompt
  let tokens = tokenizer.encode(prompt);
  const maxLen = model.config.blockSize;

  // Set model to eval mode
  model.eval();

  let generatedText = '';

  for (let i = 0; i < maxNewTokens; i++) {
    // Truncate to max context length
    const inputTokens = tokens.slice(-maxLen);

    // Create input tensor
    const input = api.tensorFromArray(inputTokens, [1, inputTokens.length], {
      device: 'webgpu',
    });

    // Forward pass
    const logits = model.forward(input);

    // Get logits for last position: [1, seqLen, vocabSize] -> [vocabSize]
    const lastLogits = await getLastLogits(api, logits, inputTokens.length - 1);

    // Sample next token
    const nextToken = await sampleToken(lastLogits, temperature, topK);

    // Decode single token
    const tokenStr = tokenizer.decode([nextToken]);
    generatedText += tokenStr;

    yield tokenStr;

    // Add to context
    tokens.push(nextToken);

    // Check stop sequences
    if (stopSequences.some((seq) => generatedText.endsWith(seq))) {
      break;
    }

    // Check for EOS
    if (nextToken === tokenizer.eosToken) {
      break;
    }

    // Cleanup
    await api.markStep();
  }
}

/**
 * Generate complete text (non-streaming).
 */
export async function generateText(
  api: Torchlette,
  model: GPT2WithLoRA,
  tokenizer: GPT2Tokenizer,
  prompt: string,
  options: GenerateOptions = {}
): Promise<string> {
  let result = '';

  for await (const token of generateTokens(api, model, tokenizer, prompt, options)) {
    result += token;
  }

  return result;
}

/**
 * Get logits for the last position in the sequence.
 */
async function getLastLogits(
  api: Torchlette,
  logits: Tensor,
  lastIdx: number
): Promise<Float32Array> {
  // logits shape: [1, seqLen, vocabSize]
  const [_batch, seqLen, vocabSize] = logits.shape;

  // Read all logits and extract last position
  const allLogits = await logits.cpu();

  // Extract last position: offset = lastIdx * vocabSize
  const offset = lastIdx * vocabSize;
  return new Float32Array(allLogits.slice(offset, offset + vocabSize));
}

/**
 * Sample a token from logits using temperature and top-k.
 */
async function sampleToken(
  logits: Float32Array,
  temperature: number,
  topK: number
): Promise<number> {
  // Apply temperature
  const scaledLogits = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    scaledLogits[i] = logits[i] / temperature;
  }

  // Softmax
  const maxLogit = Math.max(...scaledLogits);
  const expLogits = scaledLogits.map((l) => Math.exp(l - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  const probs = expLogits.map((e) => e / sumExp);

  // Top-K filtering
  const indexed = Array.from(probs)
    .map((p, i) => ({ prob: p, idx: i }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, topK);

  // Renormalize
  const topKSum = indexed.reduce((s, x) => s + x.prob, 0);
  const normalizedProbs = indexed.map((x) => ({
    ...x,
    prob: x.prob / topKSum,
  }));

  // Sample
  let r = Math.random();
  for (const { prob, idx } of normalizedProbs) {
    r -= prob;
    if (r <= 0) {
      return idx;
    }
  }

  // Fallback to most likely
  return normalizedProbs[0].idx;
}
