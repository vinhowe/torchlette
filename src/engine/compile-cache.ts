import type { DType } from "../backend/types";
import type { IRGraph } from "./ir";

/**
 * Input signature for cache key generation.
 * Captures the abstract properties of inputs that affect compilation.
 */
export type InputSignature = {
  shape: number[];
  dtype: DType;
  isInput: boolean; // true if this is an external input, false if computed
};

/**
 * Cache key components for a compiled region.
 * Per spec §8.2: normalized IR + inputSigAbstract + policies + device caps
 */
export type CompiledCacheKey = {
  irHash: string;
  inputSignatures: InputSignature[];
};

/**
 * Cached compiled entry.
 */
export type CompiledCacheEntry = {
  key: CompiledCacheKey;
  graph: IRGraph;
  createdAt: number;
  hitCount: number;
};

/**
 * Compute a deterministic hash of an IR graph structure.
 * This implements "normalized IR structural hash" from spec §8.2.
 *
 * The hash captures:
 * - Operation types and their order
 * - Data flow (which nodes connect to which)
 * - Shapes and dtypes
 *
 * It does NOT include:
 * - Node IDs (which are runtime-specific)
 * - Epoch numbers (which vary per call)
 */
export function hashIRGraph(graph: IRGraph): string {
  // Build a normalized representation
  // Map original node IDs to canonical indices (0, 1, 2, ...)
  const idToIndex = new Map<number, number>();
  for (let i = 0; i < graph.nodes.length; i++) {
    idToIndex.set(graph.nodes[i].id, i);
  }

  const parts: string[] = [];

  for (const node of graph.nodes) {
    // Normalize input references to use canonical indices
    const normalizedInputs = node.inputs.map((id) => {
      const idx = idToIndex.get(id);
      // Input might be external (not in this graph)
      return idx !== undefined ? `n${idx}` : `ext${id}`;
    });

    // Build node signature: op[inputs](shape,dtype)
    const shapePart = node.shape ? node.shape.join("x") : "?";
    const dtypePart = node.dtype ?? "?";
    const nodeSig = `${node.op}[${normalizedInputs.join(",")}](${shapePart},${dtypePart})`;
    parts.push(nodeSig);
  }

  // Add fusion group info
  for (const group of graph.fusionGroups) {
    const normalizedIds = group.nodeIds.map((id) => idToIndex.get(id) ?? -1);
    parts.push(`fuse:${group.kind}[${normalizedIds.join(",")}]`);
  }

  // Simple hash: join all parts and compute a string hash
  const normalized = parts.join("|");
  return simpleHash(normalized);
}

/**
 * Simple string hash function.
 * Uses djb2 algorithm for reasonable distribution.
 */
function simpleHash(str: string): string {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = (hash * 33) ^ str.charCodeAt(i);
  }
  // Convert to hex string, handle negative numbers
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/**
 * Extract input signatures from an IR graph.
 * Identifies which nodes are external inputs vs computed nodes.
 */
export function extractInputSignatures(graph: IRGraph): InputSignature[] {
  const nodeIds = new Set(graph.nodes.map((n) => n.id));
  const signatures: InputSignature[] = [];

  for (const node of graph.nodes) {
    // Check if any input is external (not in this graph)
    const hasExternalInput = node.inputs.some((id) => !nodeIds.has(id));

    signatures.push({
      shape: node.shape?.slice() ?? [],
      dtype: node.dtype ?? "f32",
      isInput: hasExternalInput || node.inputs.length === 0,
    });
  }

  return signatures;
}

/**
 * Generate a full cache key for a compiled graph.
 */
export function generateCacheKey(graph: IRGraph): CompiledCacheKey {
  return {
    irHash: hashIRGraph(graph),
    inputSignatures: extractInputSignatures(graph),
  };
}

/**
 * Serialize a cache key to a string for use as a map key.
 */
export function serializeCacheKey(key: CompiledCacheKey): string {
  const sigParts = key.inputSignatures.map(
    (sig) => `${sig.shape.join("x")}:${sig.dtype}:${sig.isInput ? "i" : "c"}`,
  );
  return `${key.irHash}|${sigParts.join(";")}`;
}

/**
 * LRU cache for compiled regions.
 */
export class CompiledCache {
  private cache = new Map<string, CompiledCacheEntry>();
  private maxSize: number;

  constructor(maxSize = 64) {
    this.maxSize = maxSize;
  }

  /**
   * Look up a cached entry by key.
   * Returns undefined if not found.
   */
  get(key: CompiledCacheKey): CompiledCacheEntry | undefined {
    const keyStr = serializeCacheKey(key);
    const entry = this.cache.get(keyStr);
    if (entry) {
      entry.hitCount++;
      // Move to end for LRU (delete and re-add)
      this.cache.delete(keyStr);
      this.cache.set(keyStr, entry);
    }
    return entry;
  }

  /**
   * Store a compiled graph in the cache.
   */
  set(key: CompiledCacheKey, graph: IRGraph): CompiledCacheEntry {
    const keyStr = serializeCacheKey(key);

    // Evict oldest entries if at capacity
    while (this.cache.size >= this.maxSize) {
      const oldest = this.cache.keys().next().value;
      if (oldest) {
        this.cache.delete(oldest);
      }
    }

    const entry: CompiledCacheEntry = {
      key,
      graph,
      createdAt: Date.now(),
      hitCount: 0,
    };
    this.cache.set(keyStr, entry);
    return entry;
  }

  /**
   * Check if a key exists in the cache.
   */
  has(key: CompiledCacheKey): boolean {
    return this.cache.has(serializeCacheKey(key));
  }

  /**
   * Get current cache size.
   */
  get size(): number {
    return this.cache.size;
  }

  /**
   * Clear all cached entries.
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics.
   */
  stats(): { size: number; entries: { key: string; hitCount: number }[] } {
    const entries = Array.from(this.cache.entries()).map(([key, entry]) => ({
      key,
      hitCount: entry.hitCount,
    }));
    return { size: this.cache.size, entries };
  }
}
