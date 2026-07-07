import {
  oauthLoginUrl,
  oauthHandleRedirectIfPresent,
  type OAuthResult,
  type UserInfo,
} from "@huggingface/hub";
import {
  HF_OAUTH_CLIENT_ID,
  HF_OAUTH_SCOPES,
  isOAuthConfigured,
  redirectUrl,
} from "./config";

const STORE_KEY = "menagerie:session";

interface StoredSession {
  accessToken: string;
  expiresAt: number; // epoch ms
  userInfo: UserInfo;
}

/**
 * Browser-side HF auth session. The `@huggingface/hub` SDK handles the PKCE
 * dance (nonce + code_verifier in localStorage) and the code→token exchange;
 * we own persisting the resulting token + user info so reloads stay logged in
 * until the token expires.
 */
class HfSession {
  user = $state<UserInfo | null>(null);
  accessToken = $state<string | null>(null);
  expiresAt = $state<number | null>(null);
  /** true once init() has finished (redirect handled or storage restored). */
  ready = $state(false);
  error = $state<string | null>(null);

  get loggedIn(): boolean {
    return !!this.accessToken && (this.expiresAt ?? 0) > Date.now();
  }

  /** Renameable display name + repo namespace. Use {@link sub} for identity. */
  get username(): string | null {
    return this.user?.preferred_username ?? null;
  }

  /** Stable user identifier (survives renames). The canonical identity anchor. */
  get sub(): string | null {
    return this.user?.sub ?? null;
  }

  /**
   * Run once on app load. First consumes an OAuth redirect if present
   * (`?code=...`), otherwise restores a persisted session.
   */
  async init(): Promise<void> {
    if (this.ready) return;
    try {
      let result: OAuthResult | false = false;
      try {
        result = await oauthHandleRedirectIfPresent();
      } catch (e) {
        // Bad/expired code, state mismatch, user-denied, etc. Clear and move on.
        this.error = `OAuth redirect failed: ${(e as Error).message}`;
        this.clearStored();
      }
      if (result) {
        this.adopt(result);
        // Strip the oauth query params from the URL so a refresh doesn't re-handle.
        const url = new URL(window.location.href);
        url.search = "";
        window.history.replaceState({}, "", url.toString());
      } else {
        this.restore();
      }
    } finally {
      this.ready = true;
    }
  }

  /** Begin the login flow (redirects the browser to HF). */
  async login(): Promise<void> {
    if (!isOAuthConfigured()) {
      this.error =
        "HF OAuth client ID is not set. Define VITE_HF_OAUTH_CLIENT_ID (see .env.example).";
      return;
    }
    const url = await oauthLoginUrl({
      clientId: HF_OAUTH_CLIENT_ID,
      scopes: HF_OAUTH_SCOPES,
      redirectUrl: redirectUrl(),
    });
    window.location.href = url;
  }

  logout(): void {
    this.user = null;
    this.accessToken = null;
    this.expiresAt = null;
    this.clearStored();
  }

  private adopt(result: OAuthResult): void {
    this.accessToken = result.accessToken;
    this.expiresAt = result.accessTokenExpiresAt.getTime();
    this.user = result.userInfo;
    const stored: StoredSession = {
      accessToken: result.accessToken,
      expiresAt: this.expiresAt,
      userInfo: result.userInfo,
    };
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(stored));
    } catch {
      /* storage may be unavailable; session still works for this page load */
    }
  }

  private restore(): void {
    let raw: string | null = null;
    try {
      raw = localStorage.getItem(STORE_KEY);
    } catch {
      return;
    }
    if (!raw) return;
    try {
      const s = JSON.parse(raw) as StoredSession;
      if (s.expiresAt > Date.now()) {
        this.accessToken = s.accessToken;
        this.expiresAt = s.expiresAt;
        this.user = s.userInfo;
      } else {
        this.clearStored();
      }
    } catch {
      this.clearStored();
    }
  }

  private clearStored(): void {
    try {
      localStorage.removeItem(STORE_KEY);
    } catch {
      /* ignore */
    }
  }
}

export const session = new HfSession();
