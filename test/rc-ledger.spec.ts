/**
 * RC RETAIN/RELEASE LEDGER regression gate (commit 9d9f757a —
 * `_retainedInputIds` in src/graph/node-factory.ts).
 *
 * The ledger's invariant: releaseNodeInputRefs releases EXACTLY the storage ids
 * retainPlanInputRefs recorded — never a set re-derived from `node.inputs`,
 * which mid-force graph rewrites (redirectConsumers: CSE / mul-by-1 / identity-
 * cast bypass, re-applied on template hits) mutate BETWEEN retain and release.
 * A re-derived release was the phantom-release bug: it freed a storage never
 * retained (destroying an rc=1 persistent scalar under its reader).
 *
 * This gate drives a full-stack inner step (autocast + checkpoint + GradScaler +
 * clip + AdamW) under TORCHLETTE_RC_TRACE=verbose and asserts, from the probe's
 * self-tallied per-storage plan-site counts:
 *   - ZERO plan-site imbalance: every storage retained @plan.input is released
 *     @plan.inputConsumed exactly as many times (no leak, no over-release);
 *   - ZERO plan.inputConsumed DOUBLE-RELEASE (rc never driven negative by the
 *     ledger release path);
 *   - flat late-window TOTAL storage count (a ledger leak would pin storages
 *     forever → monotone climb).
 *
 * Adversarial-review provenance: branch rc-ledger-review. The probe
 * (tools/t-ledger-attack-probe.ts) exercises CSE / template-hit rewrites,
 * multi-plan steps, and the GradScaler live-scale tensor — the exact ledger
 * attack surface. GPU-only: auto-skips without WebGPU (CI runs GPU-less).
 */
import { execFile } from "node:child_process";
import * as path from "node:path";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import { canUseWebGPU } from "./helpers/webgpu";

const execFileP = promisify(execFile);
const PROBE = path.join(__dirname, "..", "tools", "t-ledger-attack-probe.ts");

interface Verdict {
  planSiteStorages: number;
  imbalanced: number;
  planDoubleRelease: number;
  balanceChecked: boolean;
  reachDrift: number;
  totalDrift: number;
  lateReachable: [number, number];
  lateTotal: [number, number];
  steps: number;
}

async function runProbe(): Promise<Verdict> {
  for (let attempt = 0; ; attempt++) {
    try {
      const { stdout } = await execFileP("npx", ["tsx", PROBE], {
        env: {
          ...process.env,
          STEPS: "16",
          TORCHLETTE_RC_TRACE: "verbose",
        },
        timeout: 300_000,
        maxBuffer: 512 * 1024 * 1024, // verbose rc trace is large
      });
      const line = stdout
        .trim()
        .split("\n")
        .reverse()
        .find((l) => l.trim().startsWith('{"rcLedger"'));
      if (!line) throw new Error(`no probe JSON:\n${stdout.slice(-500)}`);
      return JSON.parse(line).rcLedger as Verdict;
    } catch (e) {
      if (attempt >= 2) throw e;
      await new Promise((r) => setTimeout(r, 2500 * (attempt + 1)));
    }
  }
}

describe("rc retain/release ledger — balance invariant (commit 9d9f757a)", () => {
  it(
    "releases exactly what it retained: zero plan-site imbalance, zero over-release, flat memory",
    async () => {
      if (!(await canUseWebGPU())) return;
      const v = await runProbe();

      // The balance check must actually have run (verbose trace populated the
      // per-storage tallies), else the assertions below are vacuous.
      expect(v.balanceChecked).toBe(true);
      expect(v.planSiteStorages).toBeGreaterThan(100);

      // THE INVARIANT: retain@plan.input == release@plan.inputConsumed per
      // storage. Any imbalance is a ledger leak (retain>release) or the
      // phantom over-release the ledger was built to kill (release>retain).
      expect(v.imbalanced).toBe(0);

      // The ledger release path never drives any rc negative.
      expect(v.planDoubleRelease).toBe(0);

      // No ledger leak: late-window TOTAL storage count is flat (a leak pins
      // storages forever → climbing total). reachable can dip momentarily on a
      // markStep boundary, so total (not reachable) is the leak signal.
      expect(v.lateTotal[1] - v.lateTotal[0]).toBeLessThanOrEqual(2);
    },
    600_000,
  );
});
