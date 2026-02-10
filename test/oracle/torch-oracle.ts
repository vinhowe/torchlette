import { spawn } from "node:child_process";
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

export async function runTorchOracleBatch(
  cases: OracleCase[],
): Promise<OracleOutput[]> {
  if (cases.length === 0) {
    return [];
  }

  const payload = JSON.stringify({ cases });

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
        const parsed = JSON.parse(stdout) as { results: OracleResult[] };
        const results = parsed.results ?? [];
        const failures = results.filter((result) => !result.ok);

        if (failures.length > 0) {
          const messages = failures.map(
            (result) =>
              `${result.caseName ?? "case"}: ${result.error ?? "Unknown error"}`,
          );
          reject(
            new Error(`Torch oracle case failures:\n${messages.join("\n")}`),
          );
          return;
        }

        resolve(
          results.map((result) => {
            if (!result.output) {
              throw new Error("Torch oracle returned an empty output.");
            }
            return result.output;
          }),
        );
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

export async function runTorchOracleBackwardBatch(
  cases: OracleCase[],
): Promise<{ output: OracleOutput; grads: OracleOutput[] }[]> {
  if (cases.length === 0) {
    return [];
  }

  const payload = JSON.stringify({ cases });

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
        const parsed = JSON.parse(stdout) as { results: OracleResult[] };
        const results = parsed.results ?? [];
        const failures = results.filter((result) => !result.ok);

        if (failures.length > 0) {
          const messages = failures.map(
            (result) =>
              `${result.caseName ?? "case"}: ${result.error ?? "Unknown error"}`,
          );
          reject(
            new Error(`Torch oracle case failures:\n${messages.join("\n")}`),
          );
          return;
        }

        resolve(
          results.map((result) => {
            if (!result.output || !result.grads) {
              throw new Error("Torch oracle returned an empty output.");
            }
            return { output: result.output, grads: result.grads };
          }),
        );
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

export async function runTorchOracleExtendedBatch(
  cases: OracleCase[],
): Promise<ExtendedOracleResult[]> {
  if (cases.length === 0) {
    return [];
  }

  const payload = JSON.stringify({ cases });

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
        type ExtendedResult = OracleResult & {
          activations?: Record<string, OracleOutput>;
        };
        const parsed = JSON.parse(stdout) as { results: ExtendedResult[] };
        const results = parsed.results ?? [];
        const failures = results.filter((result) => !result.ok);

        if (failures.length > 0) {
          const messages = failures.map(
            (result) =>
              `${result.caseName ?? "case"}: ${result.error ?? "Unknown error"}`,
          );
          reject(
            new Error(`Torch oracle case failures:\n${messages.join("\n")}`),
          );
          return;
        }

        resolve(
          results.map((result) => {
            if (!result.output || !result.grads) {
              throw new Error("Torch oracle returned an empty output.");
            }
            return {
              output: result.output,
              grads: result.grads,
              activations: result.activations,
            };
          }),
        );
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

/**
 * Run oracle and return full results without validation.
 * Use this for checkpoint/AMP tests that return extra fields.
 */
export async function runTorchOracleFullBatch(
  cases: OracleCase[],
): Promise<FullOracleResult[]> {
  if (cases.length === 0) {
    return [];
  }

  const payload = JSON.stringify({ cases });

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
        const parsed = JSON.parse(stdout) as { results: FullOracleResult[] };
        const results = parsed.results ?? [];
        const failures = results.filter((result) => !result.ok);

        if (failures.length > 0) {
          const messages = failures.map(
            (result) =>
              `${result.caseName ?? "case"}: ${result.error ?? "Unknown error"}`,
          );
          reject(
            new Error(`Torch oracle case failures:\n${messages.join("\n")}`),
          );
          return;
        }

        resolve(results);
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
