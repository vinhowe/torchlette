import { base } from "$app/paths";

/**
 * HF OAuth config. The client ID comes from a Developer Application registered
 * at https://huggingface.co/settings/connected-applications (NOT a Space).
 * Set it at build/dev time via `VITE_HF_OAUTH_CLIENT_ID` (see .env.example).
 *
 * The app's allowed redirect URIs (configured on the HF app page) must include
 * exactly the value returned by `redirectUrl()` for each origin you serve from
 * (e.g. http://localhost:5173/ during dev, and your production origin).
 */
export const HF_OAUTH_CLIENT_ID: string | undefined =
  import.meta.env.VITE_HF_OAUTH_CLIENT_ID;

/**
 * Scopes: profile (username/avatar) + repo access. Note the distinction:
 *   - write-repos  → commit to EXISTING repos (snapshots, eval results)
 *   - manage-repos → CREATE / delete repos (fork, adopt) — required, else
 *                    createRepo 403s with "don't have the rights to create…".
 * See https://huggingface.co/docs/hub/oauth#scopes.
 */
export const HF_OAUTH_SCOPES =
  "openid profile read-repos write-repos manage-repos";

/** The redirect URI we hand to HF; must be allow-listed on the HF app. */
export function redirectUrl(): string {
  return `${window.location.origin}${base}/`;
}

export const isOAuthConfigured = (): boolean => !!HF_OAUTH_CLIENT_ID;
