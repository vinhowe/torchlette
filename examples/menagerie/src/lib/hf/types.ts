/**
 * Provenance / genealogy data model. These types describe the JSON files we
 * store in each model repo (see docs/menagerie-design.md). Phase 1 only reads
 * them; phases 2-4 write them.
 */
import type { TierId } from "$lib/tiers";

/** Marks a model repo as part of the ecology (HF model-card tag + search filter). */
export const MENAGERIE_TAG = "menagerie";

export type GenealogyOp = "init" | "fork" | "train" | "merge" | "lora-absorb";

/** A reference to a pinned, fingerprinted slice of a public HF dataset. */
export interface HfDatasetDiet {
  kind: "hf-dataset";
  dataset: string;
  config: string;
  split: string;
  rows: { offset: number; length: number };
  fingerprint: string; // sha256 of resolved content
}

/** Private text bundled into the repo at this checkpoint (becomes public). */
export interface BundledDiet {
  kind: "bundled";
  path: string; // e.g. diet/<sha>.txt
  fingerprint: string;
  tokens: number;
}

export type DietEntry = HfDatasetDiet | BundledDiet;

export interface MergeParent {
  repo: string;
  commit_sha: string;
  weight: number;
}

export interface Checkpoint {
  commit_sha?: string; // filled in after the commit lands
  op: GenealogyOp;
  created_by: string; // HF username (renameable — for display + namespace)
  created_by_sub?: string; // stable HF user id (survives renames) — identity anchor
  created_at: string; // ISO 8601
  tokens?: number;
  steps?: number;
  wallclock_ms?: number;
  dtype?: string;
  hparams?: Record<string, number | string>;
  client?: {
    ua?: string;
    webgpu_adapter?: string;
    tier_recommended?: TierId;
    tier_used?: TierId;
  };
  diet?: DietEntry[];
  merge_parents?: MergeParent[];
}

export interface Lineage {
  schema: 1;
  root_repo: string;
  parent: { repo: string; commit_sha: string } | null;
  arch_preset: TierId;
  checkpoints: Checkpoint[];
}

/** A model's `config.json` (architecture + tokenizer). */
export interface ModelConfig {
  arch_preset: TierId;
  vocabSize: number;
  blockSize: number;
  numLayers: number;
  numHeads: number;
  embedDim: number;
  dtype: string;
  tokenizer: string; // e.g. "gpt2"
}

/** Summary row for the discovery grid. */
export interface EcologyModel {
  repo: string; // owner/name
  owner: string;
  name: string;
  lastModified?: string;
  likes?: number;
  downloads?: number;
}
