/**
 * Training data = a deterministic, fingerprinted slice of a public HF dataset,
 * resolved live via the datasets-server `/rows` API. Pinning a contiguous
 * [offset, offset+length) slice (rather than random sampling) makes the diet
 * reproducible — which is what lets the eval layer's contamination check and
 * the provenance story actually mean something.
 */
import type { HfDatasetDiet } from "$lib/hf/types";

export interface DatasetChoice {
  id: string;
  config: string;
  split: string;
  label: string;
}

export const DATASETS: DatasetChoice[] = [
  {
    id: "HuggingFaceFW/fineweb-edu",
    config: "sample-10BT",
    split: "train",
    label: "FineWeb-Edu (sample-10BT)",
  },
  { id: "roneneldan/TinyStories", config: "default", split: "train", label: "TinyStories" },
];

export async function sha256Hex(s: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const ROWS_API = "https://datasets-server.huggingface.co/rows";

/** Fetch `length` rows of `.row.text` starting at `offset` (paginated; API caps 100/req). */
export async function fetchRowsText(
  dataset: string,
  config: string,
  split: string,
  offset: number,
  length: number,
): Promise<string[]> {
  const texts: string[] = [];
  let fetched = 0;
  while (fetched < length) {
    const batch = Math.min(100, length - fetched);
    const url = `${ROWS_API}?dataset=${encodeURIComponent(dataset)}&config=${encodeURIComponent(
      config,
    )}&split=${encodeURIComponent(split)}&offset=${offset + fetched}&length=${batch}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`rows fetch failed: ${res.status} ${res.statusText}`);
    const data = (await res.json()) as { rows: Array<{ row: { text?: string } }> };
    for (const r of data.rows) texts.push(r.row.text ?? "");
    fetched += data.rows.length;
    if (data.rows.length < batch) break; // reached end of split
  }
  return texts;
}

export interface LoadedSlice {
  tokens: number[];
  charCount: number;
  diet: HfDatasetDiet;
}

/**
 * Resolve a pinned dataset slice into a flat token stream + its diet-provenance
 * entry (with a content fingerprint over the exact text consumed).
 */
export async function loadDatasetSlice(
  choice: DatasetChoice,
  offset: number,
  length: number,
  encode: (text: string) => number[],
): Promise<LoadedSlice> {
  const texts = await fetchRowsText(choice.id, choice.config, choice.split, offset, length);
  const joined = texts.join("\n\n");
  const fp = await sha256Hex(joined);
  const tokens = encode(joined);
  return {
    tokens,
    charCount: joined.length,
    diet: {
      kind: "hf-dataset",
      dataset: choice.id,
      config: choice.config,
      split: choice.split,
      rows: { offset, length: texts.length },
      fingerprint: `sha256:${fp}`,
    },
  };
}
