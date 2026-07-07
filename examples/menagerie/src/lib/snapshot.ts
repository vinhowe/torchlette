/**
 * Snapshot a trained model back to its HF repo: one atomic commit containing the
 * new `model.safetensors` and an updated `lineage.json` (with a `train`
 * checkpoint appended). One snapshot per explicit save — HF is a milestone
 * store, not a per-batch store (see docs/menagerie-design.md).
 */
import { fetchLineage } from "$lib/hf/repo";
import { commitFiles, jsonBlob } from "$lib/hf/write";
import type { Checkpoint } from "$lib/hf/types";

export async function snapshotModel(params: {
  repo: string;
  accessToken: string;
  weights: Blob;
  checkpoint: Checkpoint;
  title?: string;
}): Promise<void> {
  const lineage = await fetchLineage(params.repo);
  if (!lineage) {
    throw new Error(`${params.repo} has no lineage.json — not a Menagerie model.`);
  }
  lineage.checkpoints.push(params.checkpoint);

  await commitFiles({
    repo: params.repo,
    accessToken: params.accessToken,
    files: [
      { path: "model.safetensors", content: params.weights },
      { path: "lineage.json", content: jsonBlob(lineage) },
    ],
    title:
      params.title ??
      `Train: +${params.checkpoint.steps ?? 0} steps, ${params.checkpoint.tokens ?? 0} tokens`,
  });
}
