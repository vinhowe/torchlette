/**
 * Task #96 — the clip-chain stale-external gate (failing-first).
 *
 * The class: clipGradNorm_ produces `clipCoef = minimum(div(maxNorm,
 * norm+eps), full([],1.0))` in ONE plan and consumes it CROSS-PLAN in the
 * next (`mul(g, clipCoef)` fused into a reduction row-program). Two
 * build-from-IR bail classes guarded the seam:
 *   - data-source:full            (the `1.0` ceiling — a host constant)
 *   - row-program[scalar-steptemp-input]  (the 0-d cross-plan clipCoef)
 * Pre-fix, both RECUR on every training workload, so the clip chain never
 * compiled — and a NAIVE lift of the row-program bail (without resolving the
 * action's lowering-time inputRefs snapshot to the CURRENT step's refs) would
 * silently feed a stale clip coefficient into every gradient (the #93/#87
 * silent-corruption class).
 *
 * The gate binds the WHOLE mechanism, not just value parity:
 *   1. the compiled trajectory must equal the lowered trajectory to fp noise
 *      over enough steps to cross template convergence rebuilds (~step 13),
 *      with the clip chain COMPILED — and
 *   2. the clip-chain training plans must actually COMPILE (build-from-IR) →
 *      the fused row-program + constFill execute inside the compiled plan;
 *      without this the parity check passes vacuously via the lowered fallback.
 *      (Re-based off the deleted coverage census onto the executor's
 *      BUILD-FROM-IR debug log — task #43 recorded-build sunset.) And
 *   3. no [lifetime] guard fires (a stale/swept bind is loud, never silent).
 *
 * Failing-first record (2026-07-13, this worktree):
 *   - pre-task tree: (2) fails — RECURRING-BAIL row-program[scalar-steptemp-
 *     input] ×5 reaches + data-source:full on every training workload.
 *   - constFill + naive bail-lift (no fresh-ref provenance): (2) still fails —
 *     the row-program action's frozen inputRefs snapshot (same storage id on
 *     every reach, stamp=none) keeps the bail firing; and the one ordering
 *     variant where it DID compile bound through the stale snapshot.
 *   - both halves + provenance: all three assertions pass.
 */
import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import { beforeAll, describe, expect, it } from "vitest";
import { canUseWebGPU } from "./helpers/webgpu";

const execFileP = promisify(execFile);
const TOOLS = path.join(__dirname, "..", "tools");
const PARITY_PROBE = path.join(TOOLS, "t-compiled-parity-probe.ts");
const TIMEOUT = 300_000;

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
      if (typeof err.code === "number" && (err.stdout || err.stderr)) {
        return {
          stdout: err.stdout ?? "",
          stderr: err.stderr ?? "",
          code: err.code,
        };
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
  if (!line)
    throw new Error(`no losses JSON in probe output:\n${stdout.slice(-500)}`);
  return (JSON.parse(line) as { losses: number[] }).losses;
}

describe("stale-external rebind gate (task #96 clip chain)", () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = await canUseWebGPU();
  });

  it(
    "clip chain compiles (zero recurring bails) AND compiled == lowered across rebuilds",
    async () => {
      if (!webgpu) return;
      // 20 steps: crosses compile (2nd exec), observation convergence (K=3)
      // and the convergence-pruning rebuilds (~step 13) where a stale
      // snapshot would surface as divergence.
      const compiled = await runTool(PARITY_PROBE, {
        STEPS: "20",
        TORCHLETTE_DEBUG_COMPILED: "1",
      });
      const lowered = await runTool(PARITY_PROBE, {
        STEPS: "20",
        TORCHLETTE_COMPILED_PLAN: "0",
      });
      expect(compiled.code, compiled.stderr.slice(-500)).toBe(0);
      expect(lowered.code, lowered.stderr.slice(-500)).toBe(0);

      // (3) A stale/swept external bind is LOUD — none may fire.
      const allOut = `${compiled.stdout}\n${compiled.stderr}`;
      expect(allOut).not.toMatch(/\[lifetime\]/);

      // (2) The clip-chain training plans must actually COMPILE — otherwise (1)
      // passes vacuously via the lowered-without-record fallback (the recorded
      // build is gone since the task #43 sunset, so an uncovered plan runs
      // lowered and never records). RE-BASED off the coverage census (deleted
      // with the recorded build): under TORCHLETTE_DEBUG_COMPILED the executor
      // logs `[compiled] ...BUILD-FROM-IR fp=...` for every plan built from IR.
      // The two clip plans (the `minimum` ceiling producing clipCoef, and the
      // fused row-program `mul(g,clipCoef)` consumer) were the ONLY recurring
      // training bails pre-fix — so plans building from IR proves the clip chain
      // is on the compiled path, not the bail's lowered fallback.
      const builtFromIR = allOut
        .split("\n")
        .filter((l) => l.includes("BUILD-FROM-IR fp="));
      expect(
        builtFromIR.length,
        `expected the clip-chain training plans to build from IR (compiled), not fall to lowered:\n${allOut.slice(-800)}`,
      ).toBeGreaterThan(0);

      // (1) Value truth: per-step losses agree with the lowered path.
      const a = parseLosses(compiled.stdout);
      const b = parseLosses(lowered.stdout);
      expect(a.length).toBe(20);
      expect(b.length).toBe(20);
      let maxDiff = 0;
      for (let i = 0; i < a.length; i++)
        maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
      // Same tolerance rationale as the compiled-plan parity gate: a frozen
      // or stale clip coefficient diverges the trajectory across steps.
      expect(maxDiff, `compiled vs lowered max |Δloss|=${maxDiff}`).toBeLessThan(
        1e-3,
      );
    },
    TIMEOUT,
  );
});
