/**
 * Genealogy operations that create new ecology repos: fork (seed from an
 * existing ecology model at a pinned commit) and adopt (bootstrap a root from a
 * known external HF checkpoint). Both reuse the HF write layer; neither needs
 * the training engine — weights are seeded by handing the parent's pinned
 * `model.safetensors` resolve URL to the uploader.
 *
 * "Create from scratch" (random init) is intentionally NOT here: it requires
 * instantiating + serializing a model, which lands with the trainer (phase 3).
 */
import type { TierId } from "$lib/tiers";
import { fetchModelConfig, fetchLineage, resolveUrl } from "./hf/repo";
import {
  buildModelCard,
  createMenagerieRepo,
  uploadModelWeights,
} from "./hf/write";
import type { Checkpoint, Lineage, ModelConfig } from "./hf/types";

/** Known external checkpoints we can bootstrap a root from (dims hardcoded). */
export interface AdoptableSource {
  id: string; // HF repo id
  label: string;
  preset: TierId;
  vocabSize: number;
  blockSize: number;
  numLayers: number;
  numHeads: number;
  embedDim: number;
}

export const ADOPTABLE: AdoptableSource[] = [
  {
    id: "distilbert/distilgpt2",
    label: "DistilGPT-2 (~82M)",
    preset: "mini-distil",
    vocabSize: 50257,
    blockSize: 1024,
    numLayers: 6,
    numHeads: 12,
    embedDim: 768,
  },
  {
    id: "openai-community/gpt2",
    label: "GPT-2 (124M)",
    preset: "base-124m",
    vocabSize: 50257,
    blockSize: 1024,
    numLayers: 12,
    numHeads: 12,
    embedDim: 768,
  },
];

function nowIso(): string {
  return new Date().toISOString();
}

function clientInfo(): Checkpoint["client"] {
  return {
    ua: typeof navigator !== "undefined" ? navigator.userAgent : undefined,
  };
}

/** Bootstrap a new ecology root by adopting a known external checkpoint. */
export async function adoptCheckpoint(params: {
  source: AdoptableSource;
  owner: string;
  newName: string;
  username: string;
  sub: string;
  accessToken: string;
}): Promise<string> {
  const { source, owner, newName, username, sub, accessToken } = params;
  const repo = `${owner}/${newName}`;

  const config: ModelConfig = {
    arch_preset: source.preset,
    vocabSize: source.vocabSize,
    blockSize: source.blockSize,
    numLayers: source.numLayers,
    numHeads: source.numHeads,
    embedDim: source.embedDim,
    dtype: "f32",
    tokenizer: "gpt2",
  };

  const lineage: Lineage = {
    schema: 1,
    root_repo: repo,
    parent: null,
    arch_preset: source.preset,
    checkpoints: [
      {
        op: "init",
        created_by: username,
        created_by_sub: sub,
        created_at: nowIso(),
        dtype: "f32",
        client: clientInfo(),
      },
    ],
  };

  await createMenagerieRepo({
    owner,
    name: newName,
    accessToken,
    config,
    lineage,
    readme: buildModelCard({ name: newName, preset: source.preset, seededFrom: source.id }),
  });

  await uploadModelWeights({
    repo,
    accessToken,
    source: new URL(resolveUrl(source.id, "model.safetensors")),
    commitTitle: `Seed weights from ${source.id}`,
  });

  return repo;
}

/** Fork an existing ecology model at a pinned commit into the user's namespace. */
export async function forkModel(params: {
  parentRepo: string;
  parentCommitSha: string;
  owner: string;
  newName: string;
  username: string;
  sub: string;
  accessToken: string;
}): Promise<string> {
  const { parentRepo, parentCommitSha, owner, newName, username, sub, accessToken } = params;
  const repo = `${owner}/${newName}`;

  const parentConfig = await fetchModelConfig(parentRepo, parentCommitSha);
  if (!parentConfig) {
    throw new Error(`${parentRepo}@${parentCommitSha} has no config.json — not a Menagerie model.`);
  }
  const parentLineage = await fetchLineage(parentRepo, parentCommitSha);

  const lineage: Lineage = {
    schema: 1,
    root_repo: parentLineage?.root_repo ?? parentRepo,
    parent: { repo: parentRepo, commit_sha: parentCommitSha },
    arch_preset: parentConfig.arch_preset,
    checkpoints: [
      {
        op: "fork",
        created_by: username,
        created_by_sub: sub,
        created_at: nowIso(),
        dtype: parentConfig.dtype,
        client: clientInfo(),
      },
    ],
  };

  await createMenagerieRepo({
    owner,
    name: newName,
    accessToken,
    config: parentConfig,
    lineage,
    readme: buildModelCard({ name: newName, preset: parentConfig.arch_preset, parentRepo }),
  });

  await uploadModelWeights({
    repo,
    accessToken,
    source: new URL(resolveUrl(parentRepo, "model.safetensors", parentCommitSha)),
    commitTitle: `Fork from ${parentRepo}@${parentCommitSha.slice(0, 8)}`,
  });

  return repo;
}
