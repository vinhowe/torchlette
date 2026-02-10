/**
 * GPT-2 Data Loading
 *
 * GPT-2 BPE Tokenizer and FineWeb data loader.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { Tensor, Torchlette, DeviceKind } from "../../src/frontend";

// ============================================================================
// GPT-2 BPE Tokenizer
// ============================================================================

/**
 * GPT-2 BPE Tokenizer.
 *
 * Loads vocab.json and merges.txt from HuggingFace tokenizer files.
 */
export class GPT2Tokenizer {
  private encoder: Map<string, number> = new Map();
  private decoder: Map<number, string> = new Map();
  private bpeRanks: Map<string, number> = new Map();
  private cache: Map<string, string> = new Map();
  private byteEncoder: Map<number, string> = new Map();
  private byteDecoder: Map<string, number> = new Map();

  readonly vocabSize = 50257;
  readonly eosToken = 50256;
  readonly padToken = 50256; // GPT-2 uses EOS as pad

  private loaded = false;

  /**
   * Load tokenizer files.
   */
  async load(tokenizerPath: string): Promise<void> {
    const vocabPath = path.join(tokenizerPath, "vocab.json");
    const mergesPath = path.join(tokenizerPath, "merges.txt");

    // Check if files exist
    if (!fs.existsSync(vocabPath) || !fs.existsSync(mergesPath)) {
      throw new Error(
        `Tokenizer files not found at ${tokenizerPath}. ` +
        `Expected vocab.json and merges.txt`,
      );
    }

    // Load vocabulary
    const vocabJson = JSON.parse(
      await fs.promises.readFile(vocabPath, "utf-8"),
    );
    for (const [token, id] of Object.entries(vocabJson)) {
      this.encoder.set(token, id as number);
      this.decoder.set(id as number, token);
    }

    // Load merges
    const mergesText = await fs.promises.readFile(mergesPath, "utf-8");
    const mergeLines = mergesText.split("\n").slice(1); // Skip header
    for (let i = 0; i < mergeLines.length; i++) {
      const line = mergeLines[i].trim();
      if (line) {
        this.bpeRanks.set(line, i);
      }
    }

    // Build byte encoder/decoder
    this.buildByteEncoder();

    this.loaded = true;
    console.log(`Tokenizer loaded: ${this.encoder.size} tokens`);
  }

  /**
   * Build byte-to-unicode mapping (GPT-2 specific encoding).
   */
  private buildByteEncoder(): void {
    // GPT-2 uses a specific byte-to-unicode mapping
    const bs: number[] = [];
    const cs: number[] = [];

    // Printable ASCII + extended Latin
    for (let i = 33; i <= 126; i++) bs.push(i);
    for (let i = 161; i <= 172; i++) bs.push(i);
    for (let i = 174; i <= 255; i++) bs.push(i);

    // Copy to cs
    for (const b of bs) cs.push(b);

    // Fill remaining bytes
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }

    for (let i = 0; i < bs.length; i++) {
      this.byteEncoder.set(bs[i], String.fromCodePoint(cs[i]));
      this.byteDecoder.set(String.fromCodePoint(cs[i]), bs[i]);
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

    let word = token.split("");
    let pairs = this.getPairs(word);

    if (pairs.size === 0) {
      return token;
    }

    while (true) {
      // Find the pair with lowest rank
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

      const [first, second] = minPair.split(" ");

      // Merge the pair in the word
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

    const result = word.join(" ");
    this.cache.set(token, result);
    return result;
  }

  /**
   * Encode text to token IDs.
   */
  encode(text: string): number[] {
    if (!this.loaded) {
      throw new Error("Tokenizer not loaded. Call load() first.");
    }

    const tokens: number[] = [];

    // GPT-2 tokenization: split by whitespace and punctuation
    // Simplified regex pattern (actual GPT-2 uses a more complex pattern)
    const pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+/g;
    const matches = text.match(pattern) || [];

    for (const match of matches) {
      // Convert to byte representation
      const bytes = new TextEncoder().encode(match);
      let tokenStr = "";
      for (const byte of bytes) {
        tokenStr += this.byteEncoder.get(byte) ?? String.fromCharCode(byte);
      }

      // Apply BPE
      const bpeTokens = this.bpe(tokenStr).split(" ");
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
    if (!this.loaded) {
      throw new Error("Tokenizer not loaded. Call load() first.");
    }

    const textParts: string[] = [];
    for (const token of tokens) {
      const str = this.decoder.get(token);
      if (str !== undefined) {
        textParts.push(str);
      }
    }

    const text = textParts.join("");

    // Convert byte representation back to actual bytes
    const bytes: number[] = [];
    for (const char of text) {
      const byte = this.byteDecoder.get(char);
      if (byte !== undefined) {
        bytes.push(byte);
      } else {
        bytes.push(char.charCodeAt(0));
      }
    }

    return new TextDecoder().decode(new Uint8Array(bytes));
  }
}

// ============================================================================
// FineWeb Data Loader
// ============================================================================

export type FineWebConfig = {
  seqLength: number;
  batchSize: number;
  dataPath?: string;
  shuffle?: boolean;
  seed?: number;
};

/**
 * Data loader for FineWeb or similar text datasets.
 *
 * For simplicity, this loads pre-tokenized data from a binary file,
 * or tokenizes text files on the fly.
 */
export class FineWebDataLoader {
  private readonly api: Torchlette;
  private readonly config: FineWebConfig;
  private readonly device?: DeviceKind;

  private tokenizer: GPT2Tokenizer | null = null;
  private tokens: Uint32Array | null = null;
  private currentIdx = 0;

  constructor(
    api: Torchlette,
    config: FineWebConfig,
    options?: { device?: DeviceKind },
  ) {
    this.api = api;
    this.config = config;
    this.device = options?.device;
  }

  /**
   * Initialize the data loader.
   */
  async init(tokenizerPath?: string): Promise<void> {
    // Load tokenizer if path provided
    if (tokenizerPath) {
      this.tokenizer = new GPT2Tokenizer();
      await this.tokenizer.load(tokenizerPath);
    }

    // Try to load pre-tokenized data
    const dataPath = this.config.dataPath;
    if (dataPath) {
      if (dataPath.endsWith(".bin")) {
        // Pre-tokenized binary file (uint32 tokens)
        const buffer = await fs.promises.readFile(dataPath);
        this.tokens = new Uint32Array(
          buffer.buffer,
          buffer.byteOffset,
          buffer.byteLength / 4,
        );
        console.log(`Loaded ${this.tokens.length} pre-tokenized tokens`);
      } else if (dataPath.endsWith(".txt")) {
        // Text file - tokenize on load
        if (!this.tokenizer) {
          throw new Error("Tokenizer required to load text files");
        }
        const text = await fs.promises.readFile(dataPath, "utf-8");
        const tokens = this.tokenizer.encode(text);
        this.tokens = new Uint32Array(tokens);
        console.log(`Tokenized ${this.tokens.length} tokens from text`);
      } else {
        throw new Error(`Unsupported data format: ${dataPath}`);
      }
    }

    // Shuffle if requested
    if (this.config.shuffle && this.tokens) {
      this.shuffleTokens();
    }
  }

  /**
   * Shuffle tokens (simple Fisher-Yates on sequence boundaries).
   */
  private shuffleTokens(): void {
    if (!this.tokens) return;

    // We shuffle at sequence-level granularity
    const seqLen = this.config.seqLength + 1; // +1 for target
    const numSeqs = Math.floor(this.tokens.length / seqLen);

    // Create sequence indices
    const indices = Array.from({ length: numSeqs }, (_, i) => i);

    // Fisher-Yates shuffle
    const seed = this.config.seed ?? Date.now();
    let rng = seed;
    const random = () => {
      rng = (rng * 1103515245 + 12345) & 0x7fffffff;
      return rng / 0x7fffffff;
    };

    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    // Reorder tokens based on shuffled indices
    const shuffled = new Uint32Array(numSeqs * seqLen);
    for (let i = 0; i < numSeqs; i++) {
      const srcOffset = indices[i] * seqLen;
      const dstOffset = i * seqLen;
      for (let j = 0; j < seqLen; j++) {
        shuffled[dstOffset + j] = this.tokens![srcOffset + j];
      }
    }

    this.tokens = shuffled;
  }

  /**
   * Get the next batch of sequences.
   */
  async nextBatch(): Promise<{ input: Tensor; target: Tensor }> {
    if (!this.tokens) {
      throw new Error("Data not loaded. Call init() first.");
    }

    const batchSize = this.config.batchSize;
    const seqLen = this.config.seqLength;
    const seqLenPlusOne = seqLen + 1; // Need +1 for target shift

    // Collect batch data
    const inputData: number[] = [];
    const targetData: number[] = [];

    for (let b = 0; b < batchSize; b++) {
      // Check if we have enough tokens
      if (this.currentIdx + seqLenPlusOne > this.tokens.length) {
        // Wrap around
        this.currentIdx = 0;
      }

      // Get sequence of seqLen + 1 tokens
      for (let i = 0; i < seqLen; i++) {
        inputData.push(this.tokens[this.currentIdx + i]);
        targetData.push(this.tokens[this.currentIdx + i + 1]);
      }

      this.currentIdx += seqLen;
    }

    // Create tensors
    const input = this.api.tensorFromArray(inputData, [batchSize, seqLen], {
      device: this.device,
    });
    const target = this.api.tensorFromArray(targetData, [batchSize, seqLen], {
      device: this.device,
    });

    return { input, target };
  }

  /**
   * Reset to the beginning of the dataset.
   */
  reset(): void {
    this.currentIdx = 0;
  }

  /**
   * Get total number of batches available.
   */
  get numBatches(): number {
    if (!this.tokens) return 0;
    const seqLenPlusOne = this.config.seqLength + 1;
    const tokensPerBatch = this.config.batchSize * seqLenPlusOne;
    return Math.floor(this.tokens.length / tokensPerBatch);
  }

  /**
   * Get the tokenizer (if loaded).
   */
  getTokenizer(): GPT2Tokenizer | null {
    return this.tokenizer;
  }
}

// ============================================================================
// Helpers for creating synthetic data
// ============================================================================

/**
 * Create random token data for testing/benchmarking.
 */
export function createRandomTokens(
  numTokens: number,
  vocabSize = 50257,
  seed = 42,
): Uint32Array {
  const tokens = new Uint32Array(numTokens);
  let rng = seed;

  const random = () => {
    rng = (rng * 1103515245 + 12345) & 0x7fffffff;
    return rng / 0x7fffffff;
  };

  for (let i = 0; i < numTokens; i++) {
    tokens[i] = Math.floor(random() * vocabSize);
  }

  return tokens;
}

/**
 * Create a simple in-memory data loader with random tokens.
 */
export function createSyntheticDataLoader(
  api: Torchlette,
  config: FineWebConfig,
  numTokens?: number,
  options?: { device?: DeviceKind },
): FineWebDataLoader {
  const loader = new FineWebDataLoader(api, config, options);

  // Create synthetic tokens
  const totalTokens = numTokens ?? config.batchSize * (config.seqLength + 1) * 1000;
  const tokens = createRandomTokens(totalTokens);

  // Inject tokens directly
  (loader as any).tokens = tokens;

  return loader;
}

/**
 * Download tokenizer files from HuggingFace.
 */
export async function downloadTokenizer(
  modelId = "gpt2",
  outputPath?: string,
): Promise<string> {
  const { execSync } = await import("node:child_process");

  const destPath = outputPath ?? path.join(process.cwd(), "models", modelId);
  await fs.promises.mkdir(destPath, { recursive: true });

  console.log(`Downloading tokenizer for ${modelId} to ${destPath}...`);

  const baseUrl = `https://huggingface.co/${modelId}/resolve/main`;

  try {
    // Download vocab.json
    execSync(
      `curl -L -o "${path.join(destPath, "vocab.json")}" "${baseUrl}/vocab.json"`,
      { stdio: "inherit" },
    );

    // Download merges.txt
    execSync(
      `curl -L -o "${path.join(destPath, "merges.txt")}" "${baseUrl}/merges.txt"`,
      { stdio: "inherit" },
    );
  } catch (e) {
    throw new Error(`Failed to download tokenizer: ${e}`);
  }

  console.log(`Tokenizer downloaded to ${destPath}`);
  return destPath;
}
