export type TokenId = number;
export type TokenKind = "root" | "join" | "effect" | "token_only";

export interface Token {
  id: TokenId;
  kind: TokenKind;
  roots: number[];
  key: string;
}

interface TokenJoinResult {
  token: Token;
  roots: number[];
}

function normalizeRoots(roots: number[]): number[] {
  const unique = Array.from(new Set(roots));
  unique.sort((a, b) => a - b);
  return unique;
}

function rootsKey(roots: number[]): string {
  return roots.join(",");
}

export class TokenStore {
  private nextId = 1;
  private readonly joinCache = new Map<string, Token>();
  readonly root: Token;

  constructor() {
    this.root = this.makeToken("root", [0], 0);
    this.nextId = 1;
  }

  createEffectToken(): Token {
    const id = this.nextId++;
    return this.makeToken("effect", [id], id);
  }

  createTokenOnlyToken(): Token {
    const id = this.nextId++;
    return this.makeToken("token_only", [id], id);
  }

  createDebugToken(): Token {
    return this.createEffectToken();
  }

  afterAll(tokens: Token[]): TokenJoinResult {
    if (tokens.length === 0) {
      throw new Error("afterAll requires at least one token");
    }

    const roots = normalizeRoots(tokens.flatMap((token) => token.roots));
    const key = rootsKey(roots);

    for (const token of tokens) {
      if (token.key === key) {
        return { token, roots };
      }
    }

    const cached = this.joinCache.get(key);
    if (cached) {
      return { token: cached, roots };
    }

    const join = this.makeToken("join", roots, this.nextId++);
    this.joinCache.set(key, join);
    return { token: join, roots };
  }

  private makeToken(kind: TokenKind, roots: number[], id: number): Token {
    const normalized = normalizeRoots(roots);
    return {
      id,
      kind,
      roots: normalized,
      key: rootsKey(normalized),
    };
  }
}
