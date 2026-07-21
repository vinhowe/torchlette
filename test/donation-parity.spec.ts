/**
 * Buffer-donation P2 parity gate (docs/buffer-donation-design.md §2.5,
 * design-named `test/donation-parity.spec.ts`).
 *
 * Donation changes only BUFFER ASSIGNMENT, never values. This gate asserts the
 * opt-in `TORCHLETTE_PLANNER_DONATION=1` flag is BIT-EXACT against OFF across the
 * compiled-plan activation threshold, on the packed-optimizer path (Adam/Lion/SGD
 * arms — the donation edge's target). Any drift is a real aliasing bug (the §7.2
 * "later data leaks into an earlier step's results" class), NEVER a tolerance to
 * widen.
 *
 * STATUS (§2.5): the P2 donation edge is currently INERT on this (and every real
 * training) workload — the packed plans are `fullyCovered=false` (they run on the
 * recorded build, not the generated stream where the edge lives) and the packed
 * buffers are oversized → chunked. So the ON arm is bit-identical to OFF BY
 * CONSTRUCTION (donation never fires). This gate is therefore the standing SAFETY
 * proof that the opt-in flag can never perturb a trajectory; it upgrades to a live
 * differential the moment the coverage prerequisite (§2.5 (A)+(B)) lands.
 *
 * Subprocess-isolated (engine module-global caches interfere across in-process
 * Torchlette instances — same methodology as compiled-plan-parity.spec.ts). The
 * probe itself runs ON-vs-OFF from byte-identical initial weights and diffs.
 */
import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import { beforeAll, describe, expect, it } from "vitest";
import { canUseWebGPU } from "./helpers/webgpu";

const execFileP = promisify(execFile);
const PROBE = path.join(__dirname, "..", "tools", "t-donation-parity-probe.ts");
const TIMEOUT = 300_000;

async function runProbe(opt: string): Promise<string> {
  const { stderr } = await execFileP(
    "npx",
    ["tsx", PROBE],
    {
      env: { ...process.env, OPT: opt, STEPS: "12", TOL: "0" },
      timeout: TIMEOUT,
      maxBuffer: 32 * 1024 * 1024,
    },
  );
  return stderr;
}

describe("buffer-donation P2 parity (TORCHLETTE_PLANNER_DONATION)", () => {
  let webgpu = false;
  beforeAll(async () => {
    webgpu = await canUseWebGPU();
  });

  for (const opt of ["adam", "lion", "sgd"]) {
    it(
      `donation ON == OFF, bit-exact across the compiled-plan threshold (${opt})`,
      async () => {
        if (!webgpu) {
          // GPU-less CI: the whole webgpu project auto-skips (CLAUDE.md).
          return;
        }
        const out = await runProbe(opt);
        expect(out).toContain("RESULT: PASS");
        expect(out).not.toContain("RESULT: FAIL");
        // Assert the observed diff is exactly zero (not merely under tol).
        expect(out).toMatch(/maxDiff=0\.000e\+0/);
      },
      TIMEOUT,
    );
  }
});
