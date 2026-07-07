/**
 * Architecture presets ("choose-your-own-adventure by hardware") and the
 * client-side hardware-tier recommendation.
 *
 * WebGPU does NOT expose total VRAM (fingerprinting defense). We recommend a
 * tier from `adapter.limits` (a coarse signal) and let the user override; the
 * real safety net is downshift-on-OOM when a model is actually allocated
 * (phase 3). See docs/menagerie-design.md.
 *
 * All presets reuse the GPT-2 BPE vocab (50257) with tied input/output
 * embeddings. Note: at small sizes the token-embedding table dominates the
 * parameter count, so "nano"/"micro" are embedding-heavy rather than truly tiny
 * transformers. A small-vocab tokenizer for the mobile tier is a later option.
 */

export type TierId = "nano" | "micro" | "mini-distil" | "base-124m" | "big-350m";

export interface ArchPreset {
  id: TierId;
  label: string;
  blurb: string;
  /** Rough device class this is meant for. */
  target: string;
  vocabSize: number;
  blockSize: number; // context length
  numLayers: number;
  numHeads: number;
  embedDim: number;
}

const VOCAB = 50257;

export const PRESETS: ArchPreset[] = [
  {
    id: "nano",
    label: "Nano",
    blurb: "Runs on a phone. Tiny transformer, embedding-dominated.",
    target: "iPhone / mobile Safari",
    vocabSize: VOCAB,
    blockSize: 256,
    numLayers: 4,
    numHeads: 4,
    embedDim: 128,
  },
  {
    id: "micro",
    label: "Micro",
    blurb: "Low-end laptop or older GPU.",
    target: "low-end laptop",
    vocabSize: VOCAB,
    blockSize: 512,
    numLayers: 6,
    numHeads: 8,
    embedDim: 256,
  },
  {
    id: "mini-distil",
    label: "Mini (distilgpt2)",
    blurb: "DistilGPT-2 shape. A typical laptop can train this.",
    target: "typical laptop",
    vocabSize: VOCAB,
    blockSize: 1024,
    numLayers: 6,
    numHeads: 12,
    embedDim: 768,
  },
  {
    id: "base-124m",
    label: "Base (GPT-2 124M)",
    blurb: "Full GPT-2 small. Wants a real desktop GPU.",
    target: "desktop / good GPU",
    vocabSize: VOCAB,
    blockSize: 1024,
    numLayers: 12,
    numHeads: 12,
    embedDim: 768,
  },
  {
    id: "big-350m",
    label: "Big (350M)",
    blurb: "Only a few clients can afford to train this from scratch.",
    target: "high-end GPU",
    vocabSize: VOCAB,
    blockSize: 1024,
    numLayers: 24,
    numHeads: 16,
    embedDim: 1024,
  },
];

export function presetById(id: TierId): ArchPreset {
  const p = PRESETS.find((x) => x.id === id);
  if (!p) throw new Error(`unknown preset ${id}`);
  return p;
}

/** Honest parameter count for a preset (tied embeddings counted once). */
export function paramCount(p: ArchPreset): number {
  const E = p.embedDim;
  const tokEmb = p.vocabSize * E; // tied with lm_head → counted once
  const posEmb = p.blockSize * E;
  const perLayer =
    // two layernorms (weight + bias)
    2 * (2 * E) +
    // attention: qkv proj (w+b) + output proj (w+b)
    (E * 3 * E + 3 * E) +
    (E * E + E) +
    // mlp: fc (w+b) + proj (w+b)
    (E * 4 * E + 4 * E) +
    (4 * E * E + E);
  const finalLn = 2 * E;
  return tokEmb + posEmb + p.numLayers * perLayer + finalLn;
}

export function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return `${n}`;
}

export interface GpuProbe {
  supported: boolean;
  error?: string;
  vendor?: string;
  architecture?: string;
  description?: string;
  maxBufferSize?: number;
  maxStorageBufferBindingSize?: number;
  recommended?: TierId;
}

/**
 * Probe WebGPU and recommend a tier. Conservative and read-only — no large
 * allocations here (that happens at model-load time, with OOM downshift).
 */
export async function probeGpu(): Promise<GpuProbe> {
  const nav = navigator as Navigator & { gpu?: GPU };
  if (!nav.gpu) {
    return { supported: false, error: "navigator.gpu is unavailable (no WebGPU in this browser)." };
  }
  let adapter: GPUAdapter | null = null;
  try {
    adapter = await nav.gpu.requestAdapter({ powerPreference: "high-performance" });
  } catch (e) {
    return { supported: false, error: `requestAdapter failed: ${(e as Error).message}` };
  }
  if (!adapter) {
    return { supported: false, error: "No WebGPU adapter (GPU may be blocklisted or disabled)." };
  }

  const limits = adapter.limits;
  const maxBufferSize = Number(limits.maxBufferSize ?? 0);
  const maxStorage = Number(limits.maxStorageBufferBindingSize ?? 0);

  // adapter.info is the modern API; requestAdapterInfo() the older one.
  let info: GPUAdapterInfo | undefined = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  if (!info && typeof (adapter as any).requestAdapterInfo === "function") {
    try {
      info = await (adapter as any).requestAdapterInfo();
    } catch {
      /* ignore */
    }
  }

  return {
    supported: true,
    vendor: info?.vendor || undefined,
    architecture: info?.architecture || undefined,
    description: info?.description || undefined,
    maxBufferSize,
    maxStorageBufferBindingSize: maxStorage,
    recommended: recommendTier(maxBufferSize),
  };
}

/**
 * Map maxBufferSize to a recommended tier. This is a coarse signal — desktop
 * GPUs report multi-GB maxBufferSize, mobile typically caps far lower. The user
 * can always override, and model-load OOM downshifts.
 */
export function recommendTier(maxBufferSize: number): TierId {
  const GB = 1024 * 1024 * 1024;
  if (maxBufferSize >= 4 * GB) return "base-124m";
  if (maxBufferSize >= 2 * GB) return "mini-distil";
  if (maxBufferSize >= 1 * GB) return "micro";
  return "nano";
}
