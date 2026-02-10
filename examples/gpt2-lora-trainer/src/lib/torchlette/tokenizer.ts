/**
 * GPT-2 BPE Tokenizer for browser.
 *
 * Implements byte-pair encoding (BPE) tokenization compatible with
 * the original GPT-2 tokenizer.
 */

export class GPT2Tokenizer {
  private encoder: Map<string, number> = new Map();
  private decoder: Map<number, string> = new Map();
  private bpeRanks: Map<string, number> = new Map();
  private cache: Map<string, string> = new Map();
  private byteEncoder: Map<number, string> = new Map();
  private byteDecoder: Map<string, number> = new Map();

  readonly eosToken = 50256;
  readonly vocabSize = 50257;

  /**
   * Load tokenizer from vocab and merges data.
   */
  load(vocab: Record<string, number>, merges: string[]): void {
    // Build encoder/decoder
    this.encoder = new Map(Object.entries(vocab));
    for (const [token, id] of this.encoder) {
      this.decoder.set(id, token);
    }

    // Build BPE ranks
    for (let i = 0; i < merges.length; i++) {
      this.bpeRanks.set(merges[i], i);
    }

    // Build byte encoder/decoder
    this.buildByteEncoder();
  }

  /**
   * Build the byte encoder mapping.
   * GPT-2 uses a specific mapping of bytes to unicode characters.
   */
  private buildByteEncoder(): void {
    // Printable ASCII range
    const bs: number[] = [];
    for (let i = 33; i <= 126; i++) bs.push(i); // '!' to '~'
    for (let i = 161; i <= 172; i++) bs.push(i); // '¡' to '¬'
    for (let i = 174; i <= 255; i++) bs.push(i); // '®' to 'ÿ'

    const cs = [...bs];
    let n = 0;

    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }

    for (let i = 0; i < bs.length; i++) {
      this.byteEncoder.set(bs[i], String.fromCharCode(cs[i]));
      this.byteDecoder.set(String.fromCharCode(cs[i]), bs[i]);
    }
  }

  /**
   * Get pairs of consecutive characters in a word.
   */
  private getPairs(word: string[]): Set<string> {
    const pairs = new Set<string>();
    for (let i = 0; i < word.length - 1; i++) {
      pairs.add(`${word[i]} ${word[i + 1]}`);
    }
    return pairs;
  }

  /**
   * Apply BPE to a token.
   */
  private bpe(token: string): string {
    if (this.cache.has(token)) {
      return this.cache.get(token)!;
    }

    let word = token.split('');

    if (word.length === 0) {
      return token;
    }

    let pairs = this.getPairs(word);

    while (pairs.size > 0) {
      // Find the pair with the lowest rank
      let minPair: string | null = null;
      let minRank = Infinity;

      for (const pair of pairs) {
        const rank = this.bpeRanks.get(pair);
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          minPair = pair;
        }
      }

      if (minPair === null) {
        break;
      }

      const [first, second] = minPair.split(' ');
      const newWord: string[] = [];
      let i = 0;

      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          newWord.push(...word.slice(i));
          break;
        }

        newWord.push(...word.slice(i, j));

        if (j < word.length - 1 && word[j] === first && word[j + 1] === second) {
          newWord.push(first + second);
          i = j + 2;
        } else {
          newWord.push(word[j]);
          i = j + 1;
        }
      }

      word = newWord;

      if (word.length === 1) {
        break;
      }

      pairs = this.getPairs(word);
    }

    const result = word.join(' ');
    this.cache.set(token, result);
    return result;
  }

  /**
   * Encode text to token IDs.
   */
  encode(text: string): number[] {
    const tokens: number[] = [];

    // Convert text to bytes and then to GPT-2's byte encoding
    const textEncoder = new TextEncoder();
    const bytes = textEncoder.encode(text);

    // Split on whitespace and punctuation patterns (simplified)
    // GPT-2 uses a more complex regex, but this works for basic cases
    const pattern = /('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)/gu;
    const matches = text.match(pattern) || [text];

    for (const match of matches) {
      // Convert to byte-encoded string
      const matchBytes = textEncoder.encode(match);
      let byteStr = '';
      for (const b of matchBytes) {
        byteStr += this.byteEncoder.get(b) ?? String.fromCharCode(b);
      }

      // Apply BPE
      const bpeTokens = this.bpe(byteStr).split(' ');

      for (const bpeToken of bpeTokens) {
        const id = this.encoder.get(bpeToken);
        if (id !== undefined) {
          tokens.push(id);
        }
      }
    }

    return tokens;
  }

  /**
   * Decode token IDs to text.
   */
  decode(tokens: number[]): string {
    const textDecoder = new TextDecoder('utf-8', { fatal: false });
    const bytes: number[] = [];

    for (const token of tokens) {
      const tokenStr = this.decoder.get(token);
      if (tokenStr === undefined) continue;

      for (const char of tokenStr) {
        const byte = this.byteDecoder.get(char);
        if (byte !== undefined) {
          bytes.push(byte);
        }
      }
    }

    return textDecoder.decode(new Uint8Array(bytes));
  }

  /**
   * Get the EOS (end of sequence) token ID.
   */
  getEosToken(): number {
    return this.eosToken;
  }
}

/**
 * Create and load a tokenizer from fetched data.
 */
export function createTokenizer(
  vocab: Record<string, number>,
  merges: string[]
): GPT2Tokenizer {
  const tokenizer = new GPT2Tokenizer();
  tokenizer.load(vocab, merges);
  return tokenizer;
}
