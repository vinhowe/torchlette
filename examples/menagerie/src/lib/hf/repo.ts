/**
 * Read-side HF Hub access for the ecology. Discovery + provenance fetch only;
 * writes (fork / snapshot / eval submit) come in later phases and live with the
 * authenticated session.
 *
 * We hit the public Hub HTTP API directly (no auth needed for public repos):
 *   - list:    https://huggingface.co/api/models?filter=<tag>
 *   - resolve: https://huggingface.co/<repo>/resolve/<rev>/<path>
 */
import { MENAGERIE_TAG, type EcologyModel, type Lineage, type ModelConfig } from "./types";

const HUB = "https://huggingface.co";

/** List models that have opted into the ecology (the `menagerie` tag). */
export async function listEcologyModels(limit = 100): Promise<EcologyModel[]> {
  const url = `${HUB}/api/models?filter=${encodeURIComponent(
    MENAGERIE_TAG,
  )}&sort=lastModified&direction=-1&limit=${limit}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Hub list failed: ${res.status} ${res.statusText}`);
  const raw = (await res.json()) as Array<{
    id: string;
    lastModified?: string;
    likes?: number;
    downloads?: number;
  }>;
  return raw.map((m) => {
    const [owner, ...rest] = m.id.split("/");
    return {
      repo: m.id,
      owner,
      name: rest.join("/"),
      lastModified: m.lastModified,
      likes: m.likes,
      downloads: m.downloads,
    };
  });
}

/** Build a resolve URL for a file in a repo at a given revision (branch or SHA). */
export function resolveUrl(repo: string, path: string, rev = "main"): string {
  return `${HUB}/${repo}/resolve/${rev}/${path}`;
}

async function fetchJsonIfPresent<T>(url: string): Promise<T | null> {
  const res = await fetch(url);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status}`);
  return (await res.json()) as T;
}

export function fetchModelConfig(repo: string, rev = "main"): Promise<ModelConfig | null> {
  return fetchJsonIfPresent<ModelConfig>(resolveUrl(repo, "config.json", rev));
}

export function fetchLineage(repo: string, rev = "main"): Promise<Lineage | null> {
  return fetchJsonIfPresent<Lineage>(resolveUrl(repo, "lineage.json", rev));
}

/** List a repo's commit history (newest first). Public, no auth. */
export async function fetchCommits(
  repo: string,
  rev = "main",
): Promise<Array<{ id: string; title: string; date: string }>> {
  const url = `${HUB}/api/models/${repo}/commits/${rev}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`commits fetch failed: ${res.status}`);
  const raw = (await res.json()) as Array<{ id: string; title: string; date: string }>;
  return raw;
}
