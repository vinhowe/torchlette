/**
 * FineWeb-Edu Data Loader for Distributed Pretraining
 *
 * Streams training data from HuggingFace's FineWeb-Edu dataset directly
 * in the browser. Uses hyparquet to read parquet row groups via HTTP
 * range requests — no full file download needed.
 *
 * Each peer picks a random shard (0-2409) and iterates row groups within
 * it. Different peers get different shards = different data = exactly
 * what DiLoCo needs.
 *
 * Pipeline: fetch row group → tokenize → buffer → train
 * The next row group is prefetched while training runs on the current one.
 */

/**
 * HuggingFace API URL pattern for FineWeb-Edu converted parquet shards.
 * The API returns a redirect to the CDN URL with range request support.
 */
const SHARD_COUNT = 2410;

function shardUrl(index: number): string {
  return `https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/parquet/default/train/${index}.parquet`;
}

/** A tokenizer that converts text to token IDs. */
export interface Tokenizer {
  encode(text: string): number[];
}

/** Configuration for the data loader. */
export interface FineWebLoaderConfig {
  /** GPT-2 tokenizer instance */
  tokenizer: Tokenizer;
  /** Sequence length for training (default: 128) */
  seqLength?: number;
  /** Which shard to load (0-2409). Random if not specified. */
  shardIndex?: number;
  /** Starting row group within the shard (default: 0) */
  startRowGroup?: number;
}

/**
 * Pipelined FineWeb-Edu data loader.
 *
 * Fetches parquet row groups, tokenizes text, and maintains a token
 * buffer. Call `nextBatch()` to get training batches. The loader
 * automatically prefetches the next row group when the buffer runs low.
 */
export class FineWebLoader {
  private readonly tokenizer: Tokenizer;
  private readonly seqLength: number;
  private readonly shardIndex: number;

  /** Circular token buffer */
  private tokens: number[] = [];
  private readPos = 0;

  /** Prefetch state */
  private nextRowGroup: number;
  private prefetchPromise: Promise<number[]> | null = null;
  private totalRowGroups = 0;
  private parquetMetadata: any = null;

  constructor(config: FineWebLoaderConfig) {
    this.tokenizer = config.tokenizer;
    this.seqLength = config.seqLength ?? 128;
    this.shardIndex =
      config.shardIndex ?? Math.floor(Math.random() * SHARD_COUNT);
    this.nextRowGroup = config.startRowGroup ?? 0;
  }

  /** Initialize: read parquet metadata for the shard. */
  async init(): Promise<void> {
    const { parquetMetadataAsync } = await import("hyparquet");

    // Resolve the API URL to get the actual CDN URL
    const apiUrl = shardUrl(this.shardIndex);
    const resp = await fetch(apiUrl, { redirect: "follow" });
    const cdnUrl = resp.url;

    // Read parquet metadata via range request on the CDN URL
    this.parquetMetadata = await parquetMetadataAsync({
      file: { url: cdnUrl },
    });
    this.totalRowGroups = this.parquetMetadata.row_groups.length;

    // Start prefetching the first row group
    this.prefetchPromise = this.fetchAndTokenize(this.nextRowGroup);
  }

  /** Get the shard index this loader is using. */
  getShard(): number {
    return this.shardIndex;
  }

  /** Get how many tokens are buffered. */
  bufferedTokens(): number {
    return this.tokens.length - this.readPos;
  }

  /**
   * Get the next training batch: { input, target } token arrays.
   * Automatically prefetches more data when the buffer runs low.
   * Returns null if waiting for data (shouldn't happen with pipelining).
   */
  async nextBatch(): Promise<{
    input: number[];
    target: number[];
  } | null> {
    const needed = this.seqLength + 1; // +1 for target offset

    // If buffer is low, wait for prefetch and refill
    if (this.bufferedTokens() < needed) {
      if (this.prefetchPromise) {
        const newTokens = await this.prefetchPromise;
        this.appendTokens(newTokens);
        this.prefetchPromise = null;
      }
      // Start prefetching the next row group
      this.nextRowGroup = (this.nextRowGroup + 1) % this.totalRowGroups;
      this.prefetchPromise = this.fetchAndTokenize(this.nextRowGroup);
    }

    // Still not enough? (shouldn't happen with proper pipelining)
    if (this.bufferedTokens() < needed) {
      return null;
    }

    // Extract batch
    const start = this.readPos;
    const input = this.tokens.slice(start, start + this.seqLength);
    const target = this.tokens.slice(start + 1, start + this.seqLength + 1);
    this.readPos += this.seqLength;

    // Compact buffer when we've consumed most of it
    if (this.readPos > this.tokens.length / 2) {
      this.tokens = this.tokens.slice(this.readPos);
      this.readPos = 0;
    }

    // Start prefetch if buffer is getting low and we don't have one running
    if (this.bufferedTokens() < needed * 10 && !this.prefetchPromise) {
      this.nextRowGroup = (this.nextRowGroup + 1) % this.totalRowGroups;
      this.prefetchPromise = this.fetchAndTokenize(this.nextRowGroup);
    }

    return { input, target };
  }

  /** Fetch a row group and tokenize its text column. */
  private async fetchAndTokenize(rowGroupIndex: number): Promise<number[]> {
    const { parquetRead } = await import("hyparquet");

    const apiUrl = shardUrl(this.shardIndex);
    const resp = await fetch(apiUrl, { redirect: "follow" });
    const cdnUrl = resp.url;

    const rows: any[] = [];
    await parquetRead({
      file: { url: cdnUrl },
      metadata: this.parquetMetadata,
      rowStart: this.parquetMetadata.row_groups
        .slice(0, rowGroupIndex)
        .reduce((sum: number, rg: any) => sum + Number(rg.num_rows), 0),
      rowEnd: this.parquetMetadata.row_groups
        .slice(0, rowGroupIndex + 1)
        .reduce((sum: number, rg: any) => sum + Number(rg.num_rows), 0),
      columns: ["text"],
      onComplete: (data: any[]) => rows.push(...data),
    });

    // Tokenize all rows
    const tokens: number[] = [];
    for (const row of rows) {
      const text = row.text ?? row[0];
      if (typeof text === "string" && text.length > 0) {
        tokens.push(...this.tokenizer.encode(text));
      }
    }

    return tokens;
  }

  private appendTokens(newTokens: number[]): void {
    this.tokens.push(...newTokens);
  }
}
