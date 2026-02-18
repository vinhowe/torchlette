/**
 * GPT-2 Weight Loading
 *
 * Load pretrained GPT-2 weights from HuggingFace format into torchlette model.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { Tensor, Torchlette, DeviceKind } from "../../src/frontend";
import { GPT2, GPT2_SMALL_CONFIG, type GPT2Config } from "./model";

// ============================================================================
// Types
// ============================================================================

export type LoaderOptions = {
  device?: DeviceKind;
};

type SafetensorsMetadata = {
  [key: string]: {
    dtype: string;
    shape: number[];
    data_offsets: [number, number];
  };
};

// ============================================================================
// Weight Name Mapping
// ============================================================================

/**
 * Map HuggingFace GPT-2 weight names to our model's structure.
 */
function mapWeightName(hfName: string): {
  path: string[];
  needsTranspose: boolean;
} | null {
  // Strip common prefixes (distilgpt2 uses "transformer." prefix)
  let name = hfName;
  if (name.startsWith("transformer.")) {
    name = name.slice("transformer.".length);
  }

  // Skip attention bias (causal mask) - we generate this at runtime
  if (name.endsWith(".attn.bias")) {
    return null;
  }

  // Token embeddings
  if (name === "wte.weight") {
    return { path: ["wte", "weight"], needsTranspose: false };
  }

  // Position embeddings
  if (name === "wpe.weight") {
    return { path: ["wpe", "weight"], needsTranspose: false };
  }

  // Final layer norm
  if (name === "ln_f.weight") {
    return { path: ["lnF", "weight"], needsTranspose: false };
  }
  if (name === "ln_f.bias") {
    return { path: ["lnF", "bias"], needsTranspose: false };
  }

  // Transformer blocks: h.{i}.{component}
  const blockMatch = name.match(/^h\.(\d+)\.(.+)$/);
  if (blockMatch) {
    const blockIdx = blockMatch[1];
    const component = blockMatch[2];

    // Layer norm 1
    if (component === "ln_1.weight") {
      return { path: ["h", blockIdx, "ln1", "weight"], needsTranspose: false };
    }
    if (component === "ln_1.bias") {
      return { path: ["h", blockIdx, "ln1", "bias"], needsTranspose: false };
    }

    // Attention
    if (component === "attn.c_attn.weight") {
      // GPT-2 uses Conv1D which stores as [in, out], we need [out, in]
      return { path: ["h", blockIdx, "attn", "cAttn", "weight"], needsTranspose: true };
    }
    if (component === "attn.c_attn.bias") {
      return { path: ["h", blockIdx, "attn", "cAttn", "bias"], needsTranspose: false };
    }
    if (component === "attn.c_proj.weight") {
      return { path: ["h", blockIdx, "attn", "cProj", "weight"], needsTranspose: true };
    }
    if (component === "attn.c_proj.bias") {
      return { path: ["h", blockIdx, "attn", "cProj", "bias"], needsTranspose: false };
    }

    // Layer norm 2
    if (component === "ln_2.weight") {
      return { path: ["h", blockIdx, "ln2", "weight"], needsTranspose: false };
    }
    if (component === "ln_2.bias") {
      return { path: ["h", blockIdx, "ln2", "bias"], needsTranspose: false };
    }

    // MLP
    if (component === "mlp.c_fc.weight") {
      return { path: ["h", blockIdx, "mlp", "cFc", "weight"], needsTranspose: true };
    }
    if (component === "mlp.c_fc.bias") {
      return { path: ["h", blockIdx, "mlp", "cFc", "bias"], needsTranspose: false };
    }
    if (component === "mlp.c_proj.weight") {
      return { path: ["h", blockIdx, "mlp", "cProj", "weight"], needsTranspose: true };
    }
    if (component === "mlp.c_proj.bias") {
      return { path: ["h", blockIdx, "mlp", "cProj", "bias"], needsTranspose: false };
    }
  }

  // Unknown weight
  return null;
}

// ============================================================================
// Safetensors Parser
// ============================================================================

/**
 * Parse a safetensors file and extract weight tensors.
 */
async function parseSafetensors(
  filePath: string,
): Promise<Map<string, { dtype: string; shape: number[]; data: Float32Array }>> {
  const buffer = await fs.promises.readFile(filePath);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  // Read header length (first 8 bytes, little-endian uint64)
  const headerLength = Number(view.getBigUint64(0, true));

  // Read header JSON
  const headerBytes = buffer.subarray(8, 8 + headerLength);
  const headerJson = new TextDecoder().decode(headerBytes);
  const metadata: SafetensorsMetadata = JSON.parse(headerJson);

  // Data starts after header
  const dataStart = 8 + headerLength;

  const weights = new Map<string, { dtype: string; shape: number[]; data: Float32Array }>();

  for (const [name, info] of Object.entries(metadata)) {
    if (name === "__metadata__") continue;

    const [startOffset, endOffset] = info.data_offsets;
    const byteLength = endOffset - startOffset;

    // Extract raw bytes
    const rawData = buffer.subarray(
      dataStart + startOffset,
      dataStart + endOffset,
    );

    // Convert to Float32Array based on dtype
    let floatData: Float32Array;

    if (info.dtype === "F32") {
      // Already float32 - copy to aligned buffer
      const alignedBuffer = new ArrayBuffer(byteLength);
      new Uint8Array(alignedBuffer).set(rawData);
      floatData = new Float32Array(alignedBuffer);
    } else if (info.dtype === "F16") {
      // Convert float16 to float32 - copy to aligned buffer first
      const alignedBuffer = new ArrayBuffer(byteLength);
      new Uint8Array(alignedBuffer).set(rawData);
      const uint16View = new Uint16Array(alignedBuffer);
      floatData = new Float32Array(uint16View.length);
      for (let i = 0; i < uint16View.length; i++) {
        floatData[i] = float16ToFloat32(uint16View[i]);
      }
    } else if (info.dtype === "BF16") {
      // Convert bfloat16 to float32 - copy to aligned buffer first
      const alignedBuffer = new ArrayBuffer(byteLength);
      new Uint8Array(alignedBuffer).set(rawData);
      const uint16View = new Uint16Array(alignedBuffer);
      floatData = new Float32Array(uint16View.length);
      for (let i = 0; i < uint16View.length; i++) {
        floatData[i] = bfloat16ToFloat32(uint16View[i]);
      }
    } else {
      console.warn(`Unsupported dtype ${info.dtype} for weight ${name}, skipping`);
      continue;
    }

    // Make a copy to avoid memory issues
    weights.set(name, {
      dtype: info.dtype,
      shape: info.shape,
      data: Float32Array.from(floatData),
    });
  }

  return weights;
}

/**
 * Convert float16 bits to float32.
 */
function float16ToFloat32(bits: number): number {
  const sign = (bits >> 15) & 0x1;
  const exp = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;

  if (exp === 0) {
    // Denormalized or zero
    if (frac === 0) {
      return sign ? -0 : 0;
    }
    // Denormalized
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  } else if (exp === 31) {
    // Infinity or NaN
    if (frac === 0) {
      return sign ? -Infinity : Infinity;
    }
    return NaN;
  }

  // Normalized
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

/**
 * Convert bfloat16 bits to float32.
 */
function bfloat16ToFloat32(bits: number): number {
  // BF16 is just the top 16 bits of a float32
  const float32Bits = bits << 16;
  const buffer = new ArrayBuffer(4);
  new Uint32Array(buffer)[0] = float32Bits;
  return new Float32Array(buffer)[0];
}

// ============================================================================
// Transpose Helper
// ============================================================================

/**
 * Transpose a 2D array stored as flat Float32Array.
 */
function transpose2D(
  data: Float32Array,
  shape: number[],
): { data: Float32Array; shape: number[] } {
  if (shape.length !== 2) {
    throw new Error(`transpose2D requires 2D tensor, got ${shape.length}D`);
  }

  const [rows, cols] = shape;
  const transposed = new Float32Array(data.length);

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      transposed[j * rows + i] = data[i * cols + j];
    }
  }

  return { data: transposed, shape: [cols, rows] };
}

// ============================================================================
// Model Loading
// ============================================================================

/**
 * Load GPT-2 weights from a safetensors file.
 */
export async function loadGPT2Weights(
  api: Torchlette,
  modelPath: string,
  options?: LoaderOptions & { paddedVocabSize?: number },
): Promise<Map<string, Tensor>> {
  const device = options?.device;

  // Find safetensors file
  let safetensorsPath: string;
  if (modelPath.endsWith(".safetensors")) {
    safetensorsPath = modelPath;
  } else {
    // Assume it's a directory
    safetensorsPath = path.join(modelPath, "model.safetensors");
    if (!fs.existsSync(safetensorsPath)) {
      throw new Error(
        `Could not find model.safetensors in ${modelPath}. ` +
        `Please download the model first.`,
      );
    }
  }

  console.log(`Loading weights from ${safetensorsPath}...`);
  const rawWeights = await parseSafetensors(safetensorsPath);

  const weights = new Map<string, Tensor>();

  for (const [name, { shape, data }] of rawWeights) {
    const mapping = mapWeightName(name);
    if (!mapping) {
      console.warn(`Unknown weight: ${name}, skipping`);
      continue;
    }

    let finalData = data;
    let finalShape = shape;

    // Transpose if needed (Conv1D weights)
    if (mapping.needsTranspose && shape.length === 2) {
      const transposed = transpose2D(data, shape);
      finalData = transposed.data;
      finalShape = transposed.shape;
    }

    // Pad wte weight to paddedVocabSize if specified (zero-padded rows for matmul alignment)
    const key = mapping.path.join(".");
    if (key === "wte.weight" && options?.paddedVocabSize && options.paddedVocabSize > finalShape[0]) {
      const paddedRows = options.paddedVocabSize;
      const cols = finalShape[1];
      const paddedData = new Float32Array(paddedRows * cols); // zeros by default
      paddedData.set(finalData); // copy vocabSize rows, rest stays zero
      finalData = paddedData;
      finalShape = [paddedRows, cols];
    }

    // Create tensor
    const tensor = api.tensorFromArray(
      Array.from(finalData),
      finalShape,
      { requiresGrad: true, device },
    );

    weights.set(key, tensor);
  }

  console.log(`Loaded ${weights.size} weight tensors`);
  return weights;
}

/**
 * Apply loaded weights to a GPT2 model.
 */
function applyWeights(model: GPT2, weights: Map<string, Tensor>): void {
  // Token embeddings (already padded to paddedVocabSize in loadGPT2Weights)
  const wteWeight = weights.get("wte.weight");
  if (wteWeight) {
    copyWeight(model.wte.weight, wteWeight);
  }

  // Position embeddings
  const wpeWeight = weights.get("wpe.weight");
  if (wpeWeight) {
    copyWeight(model.wpe.weight, wpeWeight);
  }

  // Final layer norm
  const lnFWeight = weights.get("lnF.weight");
  const lnFBias = weights.get("lnF.bias");
  if (lnFWeight && model.lnF.weight) {
    copyWeight(model.lnF.weight, lnFWeight);
  }
  if (lnFBias && model.lnF.bias) {
    copyWeight(model.lnF.bias, lnFBias);
  }

  // Transformer blocks
  for (let i = 0; i < model.h.length; i++) {
    const block = model.h[i];

    // Layer norm 1
    const ln1Weight = weights.get(`h.${i}.ln1.weight`);
    const ln1Bias = weights.get(`h.${i}.ln1.bias`);
    if (ln1Weight && block.ln1.weight) copyWeight(block.ln1.weight, ln1Weight);
    if (ln1Bias && block.ln1.bias) copyWeight(block.ln1.bias, ln1Bias);

    // Attention
    const cAttnWeight = weights.get(`h.${i}.attn.cAttn.weight`);
    const cAttnBias = weights.get(`h.${i}.attn.cAttn.bias`);
    if (cAttnWeight) copyWeight(block.attn.cAttn.weight, cAttnWeight);
    if (cAttnBias && block.attn.cAttn.bias) copyWeight(block.attn.cAttn.bias, cAttnBias);

    const cProjWeight = weights.get(`h.${i}.attn.cProj.weight`);
    const cProjBias = weights.get(`h.${i}.attn.cProj.bias`);
    if (cProjWeight) copyWeight(block.attn.cProj.weight, cProjWeight);
    if (cProjBias && block.attn.cProj.bias) copyWeight(block.attn.cProj.bias, cProjBias);

    // Layer norm 2
    const ln2Weight = weights.get(`h.${i}.ln2.weight`);
    const ln2Bias = weights.get(`h.${i}.ln2.bias`);
    if (ln2Weight && block.ln2.weight) copyWeight(block.ln2.weight, ln2Weight);
    if (ln2Bias && block.ln2.bias) copyWeight(block.ln2.bias, ln2Bias);

    // MLP
    const cFcWeight = weights.get(`h.${i}.mlp.cFc.weight`);
    const cFcBias = weights.get(`h.${i}.mlp.cFc.bias`);
    if (cFcWeight) copyWeight(block.mlp.cFc.weight, cFcWeight);
    if (cFcBias && block.mlp.cFc.bias) copyWeight(block.mlp.cFc.bias, cFcBias);

    const cProjWeight2 = weights.get(`h.${i}.mlp.cProj.weight`);
    const cProjBias2 = weights.get(`h.${i}.mlp.cProj.bias`);
    if (cProjWeight2) copyWeight(block.mlp.cProj.weight, cProjWeight2);
    if (cProjBias2 && block.mlp.cProj.bias) copyWeight(block.mlp.cProj.bias, cProjBias2);
  }
}

/**
 * Copy weight values from source to destination tensor using copy_.
 */
function copyWeight(dest: Tensor, src: Tensor): void {
  // Verify shapes match
  if (JSON.stringify(dest.shape) !== JSON.stringify(src.shape)) {
    throw new Error(
      `Shape mismatch: destination ${dest.shape} vs source ${src.shape}`,
    );
  }
  dest.copy_(src);
}

/**
 * Load a pretrained GPT-2 model from a directory.
 */
export async function loadPretrainedGPT2(
  api: Torchlette,
  modelPath: string,
  config?: Partial<GPT2Config>,
  options?: LoaderOptions,
): Promise<GPT2> {
  // Determine config from model path or use default
  let fullConfig: GPT2Config;

  // Check if there's a config.json
  const configPath = path.join(modelPath, "config.json");
  if (fs.existsSync(configPath)) {
    const configJson = JSON.parse(
      await fs.promises.readFile(configPath, "utf-8"),
    );
    fullConfig = {
      vocabSize: configJson.vocab_size ?? GPT2_SMALL_CONFIG.vocabSize,
      blockSize: configJson.n_positions ?? GPT2_SMALL_CONFIG.blockSize,
      numLayers: configJson.n_layer ?? GPT2_SMALL_CONFIG.numLayers,
      numHeads: configJson.n_head ?? GPT2_SMALL_CONFIG.numHeads,
      embedDim: configJson.n_embd ?? GPT2_SMALL_CONFIG.embedDim,
      dropoutRate: configJson.attn_pdrop ?? GPT2_SMALL_CONFIG.dropoutRate,
      ...config,
    };
  } else {
    fullConfig = { ...GPT2_SMALL_CONFIG, ...config };
  }

  console.log(`Creating GPT-2 model with config:`, fullConfig);

  // Create model
  const model = new GPT2(api, fullConfig, options);

  // Load weights (pass paddedVocabSize so wte data is zero-padded before tensor creation)
  const weights = await loadGPT2Weights(api, modelPath, {
    ...options,
    paddedVocabSize: model.paddedVocabSize,
  });

  // Apply weights to model
  applyWeights(model, weights);

  console.log(`Model loaded successfully`);
  return model;
}

/**
 * Download GPT-2 weights from HuggingFace Hub.
 * Requires huggingface-cli to be installed and authenticated.
 */
export async function downloadGPT2(
  modelId = "gpt2",
  outputPath?: string,
): Promise<string> {
  const { execSync } = await import("node:child_process");

  const destPath = outputPath ?? path.join(process.cwd(), "models", modelId);

  // Create directory
  await fs.promises.mkdir(destPath, { recursive: true });

  console.log(`Downloading ${modelId} to ${destPath}...`);

  try {
    // Download using huggingface-cli
    execSync(
      `huggingface-cli download ${modelId} --local-dir ${destPath} --include "*.safetensors" "config.json"`,
      { stdio: "inherit" },
    );
  } catch (e) {
    // Fallback: try using curl for specific files
    console.log("huggingface-cli not available, trying direct download...");

    const baseUrl = `https://huggingface.co/${modelId}/resolve/main`;

    // Download model.safetensors
    execSync(
      `curl -L -o "${path.join(destPath, "model.safetensors")}" "${baseUrl}/model.safetensors"`,
      { stdio: "inherit" },
    );

    // Download config.json
    execSync(
      `curl -L -o "${path.join(destPath, "config.json")}" "${baseUrl}/config.json"`,
      { stdio: "inherit" },
    );
  }

  console.log(`Download complete: ${destPath}`);
  return destPath;
}
