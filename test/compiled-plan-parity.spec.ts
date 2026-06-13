/**
 * Compiled-plan correctness gates, promoted from loose tools/ scripts into
 * the suite so they FAIL THE BUILD instead of rotting (they were manual-only;
 * see task #44 / the 2026-06-13 project audit). Three load-bearing checks,
 * each subprocess-isolated (the engine's module-global caches make multiple
 * Torchlette instances per process interfere — same one-engine-per-process
 * methodology as fused-vs-elementwise.spec.ts):
 *
 *  1. compiled == lowered trajectory — the frozen-step_size / clip-divergence
 *     class. Runs the full production inner step (autocast+checkpoint+
 *     scaler+clip+AdamW) on a tiny real GPT-2 twice (default vs
 *     TORCHLETTE_COMPILED_PLAN=0); per-step losses must agree to fp noise.
 *  2. stream generation: no divergence — the stage-4 generated stream must
 *     match the recording (TORCHLETTE_STREAM_GENERATE=1 → segment diff).
 *  3. stream determinism — recording one template twice is byte-identical
 *     (everything downstream assumes it).
 *
 * These exercise the compiled plan (the tiny GPT-2 is multi-layer → arena
 * populated → compiled replay on the 2nd+ step). No PyTorch fixtures: init
 * is deterministic so two processes match (tools/t-compiled-parity-probe.ts).
 */
import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import { beforeAll, describe, expect, it } from "vitest";
import { canUseWebGPU } from "./helpers/webgpu";

const execFileP = promisify(execFile);
const TOOLS = path.join(__dirname, "..", "tools");
const PARITY_PROBE = path.join(TOOLS, "t-compiled-parity-probe.ts");
const DETERMINISM = path.join(TOOLS, "t-stream-determinism.ts");
const TIMEOUT = 300_000;

/** Spawn a tsx tool with retry/backoff (Dawn child spawn is flaky under
 *  suite load — same wrapper rationale as fused-vs-elementwise.spec.ts). */
async function runTool(
  script: string,
  env: Record<string, string>,
): Promise<{ stdout: string; stderr: string; code: number }> {
  for (let attempt = 0; ; attempt++) {
    try {
      const { stdout, stderr } = await execFileP("npx", ["tsx", script], {
        env: { ...process.env, ...env },
        timeout: 240_000,
        maxBuffer: 64 * 1024 * 1024,
      });
      return { stdout, stderr, code: 0 };
    } catch (e) {
      const err = e as { code?: number; stdout?: string; stderr?: string };
      // Non-zero exit with output is a real result (the tool ran, signalled
      // failure) — return it; only retry true spawn/transient failures.
      if (typeof err.code === "number" && (err.stdout || err.stderr)) {
        return { stdout: err.stdout ?? "", stderr: err.stderr ?? "", code: err.code };
      }
      if (attempt >= 4) throw e;
      await new Promise((r) => setTimeout(r, 2500 * (attempt + 1)));
    }
  }
}

function parseLosses(stdout: string): number[] {
  const line = stdout
    .trim()
    .split("\n")
    .reverse()
    .find((l) => l.trim().startsWith('{"losses"'));
  if (!line) throw new Error(`no losses JSON in probe output:\n${stdout.slice(-500)}`);
  return (JSON.parse(line) as { losses: number[] }).losses;
}

describe("compiled-plan correctness gates", () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = await canUseWebGPU();
  });

  it(
    "compiled-plan trajectory == lowered trajectory (full inner step)",
    async () => {
      if (!webgpu) return;
      const def = parseLosses((await runTool(PARITY_PROBE, { STEPS: "12" })).stdout);
      const lowered = parseLosses(
        (await runTool(PARITY_PROBE, { STEPS: "12", TORCHLETTE_COMPILED_PLAN: "0" }))
          .stdout,
      );
      expect(def.length).toBe(12);
      expect(lowered.length).toBe(12);
      let maxDiff = 0;
      for (let i = 0; i < def.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(def[i] - lowered[i]));
      }
      // Compiled replay and the lowered path must agree to fp noise; a frozen
      // per-step uniform (step_size/inv_scale/LR) would diverge over steps.
      expect(maxDiff, `compiled vs lowered max |Δloss|=${maxDiff}`).toBeLessThan(1e-3);
    },
    TIMEOUT,
  );

  it(
    "stream generation matches the recording (no divergence)",
    async () => {
      if (!webgpu) return;
      const { stdout, stderr } = await runTool(PARITY_PROBE, {
        STEPS: "4",
        TORCHLETTE_STREAM_GENERATE: "1",
      });
      const out = `${stdout}\n${stderr}`;
      const lines = out.split("\n").filter((l) => l.includes("[stream-gen]"));
      const diverged = lines.filter((l) => l.includes("DIVERGE"));
      const verified = lines.filter(
        (l) => l.includes("VERIFIED") || l.includes("FULLY GENERATED"),
      );
      expect(diverged, `divergences:\n${diverged.join("\n")}`).toHaveLength(0);
      // Must actually have generated+verified something (else the gate is vacuous).
      expect(verified.length).toBeGreaterThan(0);
    },
    TIMEOUT,
  );

  it(
    "stream recording is deterministic (record twice → identical)",
    async () => {
      if (!webgpu) return;
      const { stdout, code } = await runTool(DETERMINISM, {});
      expect(stdout, stdout.slice(-400)).toContain("DETERMINISM: PASS");
      expect(code).toBe(0);
    },
    TIMEOUT,
  );
});
