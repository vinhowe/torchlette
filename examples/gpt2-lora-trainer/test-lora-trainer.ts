#!/usr/bin/env npx tsx
/**
 * Node.js test for LoRA trainer debugging.
 *
 * This test loads the GPT-2 model and tokenizer from cached files
 * and runs the training loop to debug issues.
 *
 * Usage:
 *   TORCHLETTE_WEBGPU=1 npx tsx examples/gpt2-lora-trainer/test-lora-trainer.ts
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { Torchlette, type FrontendTensor as Tensor, Adam, GradScaler } from '../../src';
import { initWebGPU } from '../../src/backend/webgpu';
import { GPT2WithLoRA, GPT2_SMALL_CONFIG } from './src/lib/torchlette/gpt2-lora';
import { createLoRAConfig } from './src/lib/torchlette/lora';
import { GPT2Tokenizer } from './src/lib/torchlette/tokenizer';

// Cache directory for weights
const CACHE_DIR = path.join(process.cwd(), '.cache', 'gpt2-lora-test');
const WEIGHTS_CACHE = path.join(CACHE_DIR, 'model.safetensors');
const TOKENIZER_CACHE = path.join(CACHE_DIR, 'tokenizer.json');

// HuggingFace URL
const HF_BASE_URL = 'https://huggingface.co/openai-community/gpt2/resolve/main';

// ============================================================================
// Weight Loading (with file caching)
// ============================================================================

type WeightData = {
  data: number[];
  shape: number[];
};

/**
 * Load weights from cache or fetch from HuggingFace.
 */
async function loadWeights(): Promise<Map<string, { data: Float32Array; shape: number[] }>> {
  // Check cache
  if (fs.existsSync(WEIGHTS_CACHE)) {
    console.log('Loading weights from cache...');
    const buffer = fs.readFileSync(WEIGHTS_CACHE);
    const weights = parseSafetensors(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
    console.log(`Loaded ${weights.size} tensors from cache`);
    return weights;
  }

  console.log('Downloading weights from HuggingFace...');
  const response = await fetch(`${HF_BASE_URL}/model.safetensors`);
  if (!response.ok) {
    throw new Error(`Failed to fetch weights: ${response.status}`);
  }

  const buffer = await response.arrayBuffer();
  console.log(`Downloaded ${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB`);

  // Cache the raw safetensors file
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  fs.writeFileSync(WEIGHTS_CACHE, Buffer.from(buffer));
  console.log(`Cached weights to ${WEIGHTS_CACHE}`);

  // Parse safetensors
  const weights = parseSafetensors(buffer);
  return weights;
}

/**
 * Load tokenizer from cache or fetch from HuggingFace.
 */
async function loadTokenizer(): Promise<GPT2Tokenizer> {
  // Check cache
  if (fs.existsSync(TOKENIZER_CACHE)) {
    console.log('Loading tokenizer from cache...');
    const cached = JSON.parse(fs.readFileSync(TOKENIZER_CACHE, 'utf-8'));
    const tokenizer = new GPT2Tokenizer();
    tokenizer.load(cached.vocab, cached.merges);
    return tokenizer;
  }

  console.log('Downloading tokenizer from HuggingFace...');
  const [vocabRes, mergesRes] = await Promise.all([
    fetch(`${HF_BASE_URL}/vocab.json`),
    fetch(`${HF_BASE_URL}/merges.txt`),
  ]);

  if (!vocabRes.ok || !mergesRes.ok) {
    throw new Error('Failed to fetch tokenizer');
  }

  const vocab = await vocabRes.json();
  const mergesText = await mergesRes.text();
  const merges = mergesText.split('\n').slice(1).filter(line => line.trim());

  // Cache
  fs.mkdirSync(CACHE_DIR, { recursive: true });
  fs.writeFileSync(TOKENIZER_CACHE, JSON.stringify({ vocab, merges }));
  console.log(`Cached tokenizer to ${TOKENIZER_CACHE}`);

  const tokenizer = new GPT2Tokenizer();
  tokenizer.load(vocab, merges);
  return tokenizer;
}

/**
 * Parse safetensors format.
 */
function parseSafetensors(buffer: ArrayBuffer): Map<string, { data: Float32Array; shape: number[] }> {
  const view = new DataView(buffer);
  const headerLength = Number(view.getBigUint64(0, true));
  const headerBytes = new Uint8Array(buffer, 8, headerLength);
  const headerText = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerText);

  const dataOffset = 8 + headerLength;
  const weights = new Map<string, { data: Float32Array; shape: number[] }>();

  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;

    const tensorInfo = info as { dtype: string; shape: number[]; data_offsets: [number, number] };
    const [startOffset, endOffset] = tensorInfo.data_offsets;
    const tensorData = new Uint8Array(buffer, dataOffset + startOffset, endOffset - startOffset);

    let float32Data: Float32Array;
    switch (tensorInfo.dtype) {
      case 'F32': {
        const alignedBuffer = new ArrayBuffer(tensorData.length);
        new Uint8Array(alignedBuffer).set(tensorData);
        float32Data = new Float32Array(alignedBuffer);
        break;
      }
      case 'F16':
        float32Data = convertFloat16ToFloat32(tensorData);
        break;
      case 'BF16':
        float32Data = convertBFloat16ToFloat32(tensorData);
        break;
      default:
        console.warn(`Unsupported dtype: ${tensorInfo.dtype} for ${name}`);
        continue;
    }

    weights.set(name, { data: float32Data, shape: tensorInfo.shape });
  }

  return weights;
}

function convertFloat16ToFloat32(data: Uint8Array): Float32Array {
  const alignedBuffer = new ArrayBuffer(data.length);
  new Uint8Array(alignedBuffer).set(data);
  const float16View = new Uint16Array(alignedBuffer);
  const float32 = new Float32Array(float16View.length);

  for (let i = 0; i < float16View.length; i++) {
    const h = float16View[i];
    const sign = (h & 0x8000) >> 15;
    const exponent = (h & 0x7c00) >> 10;
    const fraction = h & 0x03ff;

    if (exponent === 0) {
      float32[i] = fraction === 0 ? (sign ? -0 : 0) : (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
    } else if (exponent === 0x1f) {
      float32[i] = fraction === 0 ? (sign ? -Infinity : Infinity) : NaN;
    } else {
      float32[i] = (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
    }
  }

  return float32;
}

function convertBFloat16ToFloat32(data: Uint8Array): Float32Array {
  const alignedBuffer = new ArrayBuffer(data.length);
  new Uint8Array(alignedBuffer).set(data);
  const bf16View = new Uint16Array(alignedBuffer);
  const float32 = new Float32Array(bf16View.length);

  for (let i = 0; i < bf16View.length; i++) {
    const asUint32 = bf16View[i] << 16;
    const float32View = new Float32Array(1);
    new Uint32Array(float32View.buffer)[0] = asUint32;
    float32[i] = float32View[0];
  }

  return float32;
}

// ============================================================================
// Simple Data Loader
// ============================================================================

class SimpleDataLoader {
  private tokens: Uint32Array;
  private currentIdx = 0;
  private fixedBatchInput: number[] | null = null;
  private fixedBatchTarget: number[] | null = null;

  constructor(
    private api: Torchlette,
    tokens: number[],
    private batchSize: number,
    private seqLength: number,
    private device: 'cpu' | 'webgpu' = 'webgpu',
    private useFixedBatch: boolean = true // Use same batch every time for debugging
  ) {
    this.tokens = new Uint32Array(tokens);
  }

  nextBatch(): { input: Tensor; target: Tensor } {
    // Return cached batch for debugging loss decrease on same data
    if (this.useFixedBatch && this.fixedBatchInput && this.fixedBatchTarget) {
      return {
        input: this.api.tensorFromArray(this.fixedBatchInput, [this.batchSize, this.seqLength], { device: this.device }),
        target: this.api.tensorFromArray(this.fixedBatchTarget, [this.batchSize, this.seqLength], { device: this.device }),
      };
    }

    const inputData: number[] = [];
    const targetData: number[] = [];

    for (let b = 0; b < this.batchSize; b++) {
      if (this.currentIdx + this.seqLength + 1 > this.tokens.length) {
        this.currentIdx = Math.floor(Math.random() * (this.tokens.length - this.seqLength - 1));
      }

      for (let i = 0; i < this.seqLength; i++) {
        inputData.push(this.tokens[this.currentIdx + i]);
        targetData.push(this.tokens[this.currentIdx + i + 1]);
      }

      this.currentIdx = Math.floor(Math.random() * (this.tokens.length - this.seqLength - 1));
    }

    // Cache for fixed batch mode
    if (this.useFixedBatch) {
      this.fixedBatchInput = inputData;
      this.fixedBatchTarget = targetData;
    }

    return {
      input: this.api.tensorFromArray(inputData, [this.batchSize, this.seqLength], { device: this.device }),
      target: this.api.tensorFromArray(targetData, [this.batchSize, this.seqLength], { device: this.device }),
    };
  }
}

// ============================================================================
// Training Config
// ============================================================================

type TrainingConfig = {
  maxSteps: number;
  batchSize: number;
  seqLength: number;
  learningRate: number;
  useAMP: boolean;
  useCheckpointing: boolean;
};

// ============================================================================
// Main Test
// ============================================================================

async function main(): Promise<void> {
  console.log('='.repeat(60));
  console.log('GPT-2 LoRA Trainer Test');
  console.log('='.repeat(60));

  // Initialize WebGPU
  console.log('\nInitializing WebGPU...');
  const ok = await initWebGPU();
  if (!ok) {
    throw new Error('Failed to initialize WebGPU');
  }
  console.log('WebGPU initialized');

  // Try CPU to see if NaN is WebGPU-specific
  const useCPU = process.env.USE_CPU === '1';

  // Create API with optimizations
  console.log('\nCreating Torchlette API...');
  const api = new Torchlette(useCPU ? 'cpu' : 'webgpu', {
    enableFusion: false,  // Keep disabled for debugging
    enableMemoryPlanning: true,  // Re-enabled after matmul strided input fix
  });
  console.log(`API created (${useCPU ? 'CPU' : 'WebGPU'} backend)`);

  // Load tokenizer
  console.log('\nLoading tokenizer...');
  const tokenizer = await loadTokenizer();
  console.log('Tokenizer loaded');

  // Load weights
  console.log('\nLoading weights...');
  const weights = await loadWeights();
  console.log(`Loaded ${weights.size} weight tensors`);

  // Create model with LoRA
  console.log('\nCreating GPT-2 model with LoRA...');
  const loraConfig = createLoRAConfig(8, 16);
  const device = useCPU ? 'cpu' : 'webgpu';
  const model = new GPT2WithLoRA(api, GPT2_SMALL_CONFIG, loraConfig, device);
  console.log(`Model created with LoRA rank=${loraConfig.rank}, alpha=${loraConfig.alpha}`);

  // Load base weights
  console.log('\nLoading base weights into model...');
  model.loadBaseWeights(weights);
  console.log('Base weights loaded');

  // Training config - reduced for testing
  const config: TrainingConfig = {
    maxSteps: 5,
    batchSize: 1,
    seqLength: 16, // Reduced to help with memory
    learningRate: 1e-3, // Lowered LR
    useAMP: process.env.USE_AMP === '1', // Enable via env var
    useCheckpointing: process.env.USE_CHECKPOINT === '1', // Enable via env var
  };

  console.log('\nTraining config:');
  console.log(`  maxSteps: ${config.maxSteps}`);
  console.log(`  batchSize: ${config.batchSize}`);
  console.log(`  seqLength: ${config.seqLength}`);
  console.log(`  learningRate: ${config.learningRate}`);
  console.log(`  useAMP: ${config.useAMP}`);
  console.log(`  useCheckpointing: ${config.useCheckpointing}`);

  // Create training data
  const trainingText = `
    The quick brown fox jumps over the lazy dog. This is a sample text for testing
    the GPT-2 LoRA training pipeline. We need enough text to create a reasonable
    number of tokens for training. The model will learn to predict the next token
    based on the context. LoRA allows us to efficiently fine-tune the model by
    only training low-rank adapter matrices instead of all parameters.
  `.repeat(20); // Repeat to get enough tokens

  const tokens = tokenizer.encode(trainingText);
  console.log(`\nTraining data: ${tokens.length} tokens`);

  if (tokens.length < config.batchSize * (config.seqLength + 1)) {
    throw new Error(`Not enough tokens: need ${config.batchSize * (config.seqLength + 1)}, got ${tokens.length}`);
  }

  // Create data loader
  const dataLoader = new SimpleDataLoader(api, tokens, config.batchSize, config.seqLength, device);

  // Create optimizer for LoRA parameters only
  const loraParams = model.getLoRAParameters();
  console.log(`\nLoRA parameters: ${loraParams.length} tensors`);
  const optimizer = new Adam(loraParams, { lr: config.learningRate }, api);

  // Create gradient scaler for AMP
  let gradScaler: GradScaler | null = null;
  if (config.useAMP) {
    gradScaler = new GradScaler(api, {
      initScale: 65536.0,
      growthFactor: 2.0,
      backoffFactor: 0.5,
      growthInterval: 2000,
    });
    console.log('GradScaler created for AMP');
  }

  // Enable checkpointing if requested
  if (config.useCheckpointing) {
    model.enableCheckpointing(true);
    console.log('Gradient checkpointing enabled');
  }

  // Set model to training mode
  model.train(true);

  // Create compiled forward function for AMP
  console.log('\nCompiling forward pass...');
  const compiledForwardWithLoss = config.useAMP
    ? api.compile((batchInput: Tensor, batchTarget: Tensor) => {
        return api.autocast(() => {
          const { loss } = model.forwardWithLoss(batchInput, batchTarget);
          return loss;
        }, { deviceType: device });
      })
    : null;
  console.log('Forward pass compiled');

  // Training loop
  console.log('\n' + '='.repeat(60));
  console.log('Starting training...');
  console.log('='.repeat(60));

  const startTime = performance.now();

  for (let step = 0; step < config.maxSteps; step++) {
    const stepStart = performance.now();
    console.log(`\n--- Step ${step + 1}/${config.maxSteps} ---`);

    // Get batch
    console.log('  Getting batch...');
    const { input, target } = dataLoader.nextBatch();
    console.log(`  Input shape: [${input.shape.join(', ')}]`);
    console.log(`  Target shape: [${target.shape.join(', ')}]`);

    // Forward pass
    console.log('  Forward pass...');
    let loss: Tensor;
    try {
      if (config.useAMP && compiledForwardWithLoss) {
        loss = compiledForwardWithLoss(input, target);
        console.log('  Scaling loss...');
        loss = gradScaler!.scale(loss);
      } else {
        const result = model.forwardWithLoss(input, target);
        loss = result.loss;

        // Debug: check logits for extreme values
        const logitsSum = await result.logits.sum().item();
        console.log(`  Logits sum: ${logitsSum.toExponential(4)}`);
        if (Number.isNaN(logitsSum)) {
          console.log('  ERROR: Logits sum is NaN!');
        }
      }
      console.log(`  Loss tensor created, shape: [${loss.shape.join(', ')}]`);
      // Read loss value BEFORE backward (to check if NaN starts in forward pass)
      const preBackwardLoss = await loss.item();
      console.log(`  Pre-backward loss value: ${preBackwardLoss.toFixed(4)}`);
    } catch (e) {
      console.error('  ERROR in forward pass:', e);
      throw e;
    }

    // Backward pass
    console.log('  Backward pass...');
    try {
      await loss.backward();
      console.log('  Backward complete');

      // Check ALL LoRA parameter gradients for NaN and print first few valid ones
      console.log('  Checking gradients...');
      let firstNaNIdx = -1;
      let firstValidIdx = -1;
      for (let i = 0; i < loraParams.length; i++) {
        const param = loraParams[i];
        if (param.grad) {
          const gradSum = param.grad.sum();
          const gradSumVal = await gradSum.item();
          if (Number.isNaN(gradSumVal)) {
            if (firstNaNIdx === -1) firstNaNIdx = i;
            console.log(`  param[${i}] shape=[${param.shape}] grad sum = NaN <<<`);
          } else {
            if (firstValidIdx === -1) {
              firstValidIdx = i;
              console.log(`  param[${i}] shape=[${param.shape}] grad sum = ${gradSumVal.toExponential(4)} (first valid)`);
            }
          }
        } else {
          console.log(`  param[${i}] has no grad`);
        }
      }

      // Print more details about first NaN param
      if (firstNaNIdx >= 0) {
        const param = loraParams[firstNaNIdx];
        console.log(`\n  First NaN param details (idx=${firstNaNIdx}):`);
        console.log(`    shape: [${param.shape}]`);

        // Check param values themselves
        const paramSum = param.sum();
        const paramSumVal = await paramSum.item();
        console.log(`    param sum: ${paramSumVal}`);

        // Check loraA (idx 0) and loraB (idx 1) for first layer
        if (firstNaNIdx === 0 || firstNaNIdx === 1) {
          // param[0] is loraA, param[1] is loraB
          const loraA = loraParams[0];
          const loraB = loraParams[1];

          const loraASum = await loraA.sum().item();
          const loraBSum = await loraB.sum().item();
          console.log(`    loraA sum: ${loraASum}`);
          console.log(`    loraB sum: ${loraBSum} (should be ~0 if initialized to zeros)`);

          // Also check layer 1 loraB (param[3]) to compare
          const layer1LoraB = loraParams[3];
          const layer1LoraBSum = await layer1LoraB.sum().item();
          console.log(`    layer1 loraB sum: ${layer1LoraBSum}`);
        }
      }
    } catch (e) {
      console.error('  ERROR in backward pass:', e);
      throw e;
    }

    // Optimizer step
    console.log('  Optimizer step...');
    try {
      if (config.useAMP && gradScaler) {
        await gradScaler.unscale_(optimizer);
        const stepped = await gradScaler.step(optimizer);
        gradScaler.update();
        if (!stepped) {
          console.log(`  Skipped due to NaN/Inf, scale=${gradScaler.getScale()}`);
        }
      } else {
        optimizer.step();
      }
      optimizer.zeroGrad();
      console.log('  Optimizer step complete');

      // Check first LoRA param for NaN after optimizer step
      const param0 = loraParams[0];
      const param0Sum = param0.sum();
      const param0SumVal = await param0Sum.item();
      console.log(`  After optimizer: param0 sum = ${param0SumVal}`);
    } catch (e) {
      console.error('  ERROR in optimizer step:', e);
      throw e;
    }

    // Get loss value
    console.log('  Getting loss value...');
    try {
      let lossValue = await loss.item();
      if (config.useAMP && gradScaler) {
        lossValue = lossValue / gradScaler.getScale();
      }
      console.log(`  Loss: ${lossValue.toFixed(4)}`);
    } catch (e) {
      console.error('  ERROR getting loss value:', e);
      throw e;
    }

    // Cleanup
    console.log('  Disposing batch tensors...');
    try {
      input.dispose();
      target.dispose();
      console.log('  Disposed');
    } catch (e) {
      console.error('  ERROR disposing:', e);
      throw e;
    }

    // Mark step
    console.log('  Marking step...');
    try {
      await api.markStep();
      console.log('  Step marked');
    } catch (e) {
      console.error('  ERROR marking step:', e);
      throw e;
    }

    const stepTime = performance.now() - stepStart;
    console.log(`  Step time: ${stepTime.toFixed(0)}ms`);
  }

  const totalTime = performance.now() - startTime;

  // Cleanup
  model.train(false);
  if (config.useCheckpointing) {
    model.enableCheckpointing(false);
  }

  console.log('\n' + '='.repeat(60));
  console.log('Training Complete!');
  console.log('='.repeat(60));
  console.log(`Total steps: ${config.maxSteps}`);
  console.log(`Total time: ${(totalTime / 1000).toFixed(2)}s`);
  console.log(`Avg time/step: ${(totalTime / config.maxSteps).toFixed(0)}ms`);
}

main().catch((e) => {
  console.error('\nFATAL ERROR:', e);
  process.exit(1);
});
