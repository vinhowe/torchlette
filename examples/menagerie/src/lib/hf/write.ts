/**
 * Write-side HF Hub access (requires the authenticated session token):
 * creating ecology repos and committing files. Used by fork/adopt now and by
 * snapshot/eval-submit later.
 */
import { createRepo, uploadFiles, type RepoDesignation } from "@huggingface/hub";
import { MENAGERIE_TAG, type Lineage, type ModelConfig } from "./types";

export function jsonBlob(obj: unknown): Blob {
  return new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
}
export function textBlob(s: string): Blob {
  return new Blob([s], { type: "text/plain" });
}

/** A model-card README whose frontmatter carries the `menagerie` tag. */
export function buildModelCard(opts: {
  name: string;
  preset: string;
  parentRepo?: string | null;
  seededFrom?: string | null;
}): string {
  const { name, preset, parentRepo, seededFrom } = opts;
  const lines = [
    "---",
    "tags:",
    `- ${MENAGERIE_TAG}`,
    "library_name: torchlette",
    "license: mit",
    "---",
    "",
    `# ${name}`,
    "",
    "A model in the [Menagerie](https://github.com/) ecology — an open-ended,",
    "genealogical population of models trained, forked, and evaluated in the browser.",
    "",
    `- **Architecture preset:** \`${preset}\``,
  ];
  if (parentRepo) lines.push(`- **Forked from:** [${parentRepo}](https://huggingface.co/${parentRepo})`);
  if (seededFrom) lines.push(`- **Seeded from:** [${seededFrom}](https://huggingface.co/${seededFrom})`);
  lines.push("", "Provenance and lineage live in `lineage.json`.", "");
  return lines.join("\n");
}

/**
 * Create a new ecology model repo in the user's namespace and seed it with the
 * lightweight provenance files. Returns the repo id (`owner/name`).
 *
 * The (potentially large) `model.safetensors` is uploaded separately via
 * {@link uploadModelWeights} — createRepo only accepts lightweight files.
 */
export async function createMenagerieRepo(params: {
  owner: string;
  name: string;
  accessToken: string;
  config: ModelConfig;
  lineage: Lineage;
  readme: string;
  visibility?: "public" | "private";
}): Promise<string> {
  const repo = `${params.owner}/${params.name}`;
  await createRepo({
    repo: repo as RepoDesignation,
    accessToken: params.accessToken,
    visibility: params.visibility ?? "public",
    files: [
      { path: "README.md", content: textBlob(params.readme) },
      { path: "config.json", content: jsonBlob(params.config) },
      { path: "lineage.json", content: jsonBlob(params.lineage) },
    ],
  });
  return repo;
}

/**
 * Upload model weights to a repo. `source` may be a Blob (we hold the bytes,
 * e.g. freshly serialized) or a URL (the SDK fetches it — used to seed a fork
 * from a parent's pinned `model.safetensors` without us materializing it).
 */
export async function uploadModelWeights(params: {
  repo: string;
  accessToken: string;
  source: Blob | URL;
  path?: string;
  commitTitle?: string;
}): Promise<void> {
  await uploadFiles({
    repo: params.repo as RepoDesignation,
    accessToken: params.accessToken,
    files: [{ path: params.path ?? "model.safetensors", content: params.source }],
    commitTitle: params.commitTitle ?? "Add model weights",
  });
}

/** Commit one or more files (JSON/text/blob) to an existing repo. */
export async function commitFiles(params: {
  repo: string;
  accessToken: string;
  files: Array<{ path: string; content: Blob | URL }>;
  title: string;
}): Promise<void> {
  await uploadFiles({
    repo: params.repo as RepoDesignation,
    accessToken: params.accessToken,
    files: params.files,
    commitTitle: params.title,
  });
}
