import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type { DType } from "../../ops/specs/types";

export type OracleTensor = {
  values: number[];
  shape: number[];
  dtype?: DType;
};

export type OracleCase = {
  op: string;
  caseName: string;
  inputs: OracleTensor[];
  options?: Record<string, unknown>;
};

export type OracleOutput = {
  values: number[];
  shape: number[];
};

type OracleResult = {
  ok: boolean;
  output?: OracleOutput;
  grads?: OracleOutput[];
  error?: string;
  caseName?: string;
};

const python = (() => {
  if (process.env.TORCH_ORACLE_PYTHON) {
    return process.env.TORCH_ORACLE_PYTHON;
  }

  const root = process.cwd();
  const candidates = [
    path.resolve(root, ".venv", "bin", "python"),
    path.resolve(root, ".venv", "Scripts", "python.exe"),
  ];

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return "python3";
})();

const oracleScript = (() => {
  const filename = fileURLToPath(import.meta.url);
  const dir = path.dirname(filename);
  return path.resolve(dir, "../../tools/torch_oracle/torch_oracle.py");
})();

// ---------------------------------------------------------------------------
// Persistent Oracle Server
// ---------------------------------------------------------------------------

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
};

class OracleServer {
  private proc: ChildProcess | null = null;
  private ready = false;
  private dead = false;
  private startPromise: Promise<void> | null = null;
  private pending: PendingRequest[] = [];
  private buffer = "";

  async ensureStarted(): Promise<void> {
    if (this.ready && !this.dead) return;
    if (this.startPromise && !this.dead) return this.startPromise;

    this.startPromise = new Promise<void>((resolve, reject) => {
      this.dead = false;
      this.ready = false;
      this.buffer = "";

      const proc = spawn(python, [oracleScript, "--server"], {
        stdio: ["pipe", "pipe", "pipe"],
      });
      this.proc = proc;

      let startupStderr = "";

      const onStartupError = (err: Error) => {
        this.dead = true;
        reject(
          new Error(`Failed to start torch oracle server: ${err.message}`),
        );
      };

      proc.on("error", onStartupError);

      proc.stderr!.on("data", (chunk: Buffer) => {
        startupStderr += chunk.toString();
      });

      // Wait for the {"ready":true} line
      const onData = (chunk: Buffer) => {
        this.buffer += chunk.toString();
        const nlIdx = this.buffer.indexOf("\n");
        if (nlIdx === -1) return;

        const line = this.buffer.slice(0, nlIdx);
        this.buffer = this.buffer.slice(nlIdx + 1);

        try {
          const msg = JSON.parse(line);
          if (msg.ready) {
            this.ready = true;
            // Switch to normal data handler
            proc.stdout!.removeListener("data", onData);
            proc.removeListener("error", onStartupError);
            proc.stdout!.on("data", this.onData);
            proc.on("error", this.onError);
            proc.on("close", this.onClose);
            resolve();
            // Drain any remaining data in buffer
            this.drainBuffer();
            return;
          }
        } catch {
          // Not valid JSON yet, keep buffering
        }

        // If we got a line but it wasn't the ready signal, something is wrong
        this.dead = true;
        reject(
          new Error(
            `Torch oracle server sent unexpected startup message: ${line}. stderr: ${startupStderr}`,
          ),
        );
      };

      proc.stdout!.on("data", onData);

      // Handle early close during startup
      proc.on("close", (code) => {
        if (!this.ready) {
          this.dead = true;
          reject(
            new Error(
              `Torch oracle server exited during startup (code ${code}). ${startupStderr}`,
            ),
          );
        }
      });
    });

    return this.startPromise;
  }

  private onData = (chunk: Buffer) => {
    this.buffer += chunk.toString();
    this.drainBuffer();
  };

  private drainBuffer() {
    while (true) {
      const nlIdx = this.buffer.indexOf("\n");
      if (nlIdx === -1) break;

      const line = this.buffer.slice(0, nlIdx);
      this.buffer = this.buffer.slice(nlIdx + 1);

      if (!line.trim()) continue;

      const req = this.pending.shift();
      if (req) {
        try {
          req.resolve(JSON.parse(line));
        } catch (err) {
          req.reject(
            new Error(
              `Torch oracle returned invalid JSON: ${err instanceof Error ? err.message : String(err)}\n${line}`,
            ),
          );
        }
      }
    }
  }

  private onError = (err: Error) => {
    this.dead = true;
    this.rejectAll(
      new Error(`Torch oracle server error: ${err.message}`),
    );
  };

  private onClose = (_code: number | null) => {
    this.dead = true;
    this.ready = false;
    this.startPromise = null;
    this.rejectAll(new Error("Torch oracle server closed unexpectedly"));
  };

  private rejectAll(err: Error) {
    const pending = this.pending.splice(0);
    for (const req of pending) {
      req.reject(err);
    }
  }

  async request(payload: unknown): Promise<unknown> {
    await this.ensureStarted();
    return new Promise((resolve, reject) => {
      this.pending.push({ resolve, reject });
      const line = JSON.stringify(payload) + "\n";
      this.proc!.stdin!.write(line, (err) => {
        if (err) {
          // Remove from pending and reject
          const idx = this.pending.findIndex((p) => p.resolve === resolve);
          if (idx !== -1) this.pending.splice(idx, 1);
          reject(new Error(`Failed to write to oracle server: ${err.message}`));
        }
      });
    });
  }

  shutdown() {
    if (this.proc) {
      this.proc.stdin!.end();
      this.proc.kill();
      this.proc = null;
    }
    this.ready = false;
    this.dead = true;
    this.startPromise = null;
    this.rejectAll(new Error("Oracle server shut down"));
  }
}

// Module-level singleton
let serverInstance: OracleServer | null = null;

function getServer(): OracleServer {
  if (!serverInstance) {
    serverInstance = new OracleServer();

    const cleanup = () => {
      if (serverInstance) {
        serverInstance.shutdown();
        serverInstance = null;
      }
    };

    process.on("exit", cleanup);
    process.on("SIGTERM", cleanup);
  }
  return serverInstance;
}

// ---------------------------------------------------------------------------
// Spawn-per-call fallback (original behavior)
// ---------------------------------------------------------------------------

function spawnOracleOnce(payload: string): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const proc = spawn(python, [oracleScript], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    proc.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    proc.on("error", (error) => {
      reject(new Error(`Failed to start torch oracle: ${error.message}`));
    });
    proc.on("close", (code) => {
      if (code !== 0) {
        const details = stderr.trim() || stdout.trim();
        reject(
          new Error(
            `Torch oracle failed (exit ${code}). ${details || "No output."}`,
          ),
        );
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Unknown parse error";
        reject(
          new Error(
            `Torch oracle returned invalid JSON. ${message}\n${stdout}`,
          ),
        );
      }
    });

    proc.stdin.write(payload);
    proc.stdin.end();
  });
}

// ---------------------------------------------------------------------------
// Unified request function
// ---------------------------------------------------------------------------

async function oracleRawRequest(
  payload: Record<string, unknown>,
): Promise<unknown> {
  if (process.env.TORCH_ORACLE_SPAWN_PER_CALL === "1") {
    return spawnOracleOnce(JSON.stringify(payload));
  }

  const server = getServer();
  try {
    return await server.request(payload);
  } catch {
    // Server may have died — retry once with a fresh server
    serverInstance?.shutdown();
    serverInstance = null;
    const freshServer = getServer();
    return freshServer.request(payload);
  }
}

// ---------------------------------------------------------------------------
// Response parsing helpers
// ---------------------------------------------------------------------------

function parseOracleResults(raw: unknown): OracleResult[] {
  const parsed = raw as { results?: OracleResult[] };
  const results = parsed.results ?? [];
  const failures = results.filter((r) => !r.ok);

  if (failures.length > 0) {
    const messages = failures.map(
      (r) => `${r.caseName ?? "case"}: ${r.error ?? "Unknown error"}`,
    );
    throw new Error(`Torch oracle case failures:\n${messages.join("\n")}`);
  }

  return results;
}

// ---------------------------------------------------------------------------
// Exported functions (signatures unchanged)
// ---------------------------------------------------------------------------

export type ExtendedOracleResult = {
  output: OracleOutput;
  grads: OracleOutput[];
  activations?: Record<string, OracleOutput>;
};

// Generic oracle result that includes all possible fields
export type FullOracleResult = {
  ok: boolean;
  output?: OracleOutput;
  grads?: (OracleOutput | null)[];
  activations?: Record<string, unknown>;
  error?: string;
  caseName?: string;
  // Checkpoint/AMP specific fields
  memorySnapshots?: Array<{
    label: string;
    allocatedBytes: number;
    reservedBytes: number;
  }>;
  scale?: number;
  scaledOutput?: OracleOutput;
  foundInf?: boolean;
  // Memory comparison fields
  withoutCheckpoint?: {
    peakMemory: number;
    grads: OracleOutput[];
  };
  withCheckpoint?: {
    peakMemory: number;
    grads: OracleOutput[];
  };
  gradsMatch?: boolean;
};

export async function runTorchOracleBatch(
  cases: OracleCase[],
): Promise<OracleOutput[]> {
  if (cases.length === 0) return [];

  const raw = await oracleRawRequest({ cases });
  const results = parseOracleResults(raw);

  return results.map((result) => {
    if (!result.output) {
      throw new Error("Torch oracle returned an empty output.");
    }
    return result.output;
  });
}

export async function runTorchOracleBackwardBatch(
  cases: OracleCase[],
): Promise<{ output: OracleOutput; grads: OracleOutput[] }[]> {
  if (cases.length === 0) return [];

  const raw = await oracleRawRequest({ cases });
  const results = parseOracleResults(raw);

  return results.map((result) => {
    if (!result.output || !result.grads) {
      throw new Error("Torch oracle returned an empty output.");
    }
    return { output: result.output, grads: result.grads };
  });
}

export async function runTorchOracleExtendedBatch(
  cases: OracleCase[],
): Promise<ExtendedOracleResult[]> {
  if (cases.length === 0) return [];

  const raw = await oracleRawRequest({ cases });
  type ExtendedResult = OracleResult & {
    activations?: Record<string, OracleOutput>;
  };
  const parsed = raw as { results?: ExtendedResult[] };
  const results = parsed.results ?? [];
  const failures = results.filter((r) => !r.ok);

  if (failures.length > 0) {
    const messages = failures.map(
      (r) => `${r.caseName ?? "case"}: ${r.error ?? "Unknown error"}`,
    );
    throw new Error(`Torch oracle case failures:\n${messages.join("\n")}`);
  }

  return results.map((result) => {
    if (!result.output || !result.grads) {
      throw new Error("Torch oracle returned an empty output.");
    }
    return {
      output: result.output,
      grads: result.grads,
      activations: result.activations,
    };
  });
}

export async function runTorchOracleFullBatch(
  cases: OracleCase[],
): Promise<FullOracleResult[]> {
  if (cases.length === 0) return [];

  const raw = await oracleRawRequest({ cases });
  const results = parseOracleResults(raw) as FullOracleResult[];
  return results;
}
