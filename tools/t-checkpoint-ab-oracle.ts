/**
 * CHECKPOINT TWO-GATE A/B ORACLE — task #99 phase R0 gate 2, the R2 acceptance
 * oracle (docs/arena-recompute-design.md §5 Phase R0 item 2 + §6 Phase 3 STOP).
 *
 * === D3 RE-FRAME + STOP (2026-07-16) ===
 * The memory side re-frames from steady-CURRENT-parity to PEAK-parity per Vin's D3
 * ruling (arena-ON peak ≤ arena-free peak + 5%; current reported-not-gated). Stage-1
 * fresh like-for-like measurement (device VULKAN 0, commit 112c9fa4, 3 repeats each,
 * perfectly reproducible) FALSIFIED the ruling's precondition:
 *   distil steadyPeak: arena-ON 4278.5MB vs arena-free 3933.5MB → +8.8%  FAIL
 *   distil globalPeak: arena-ON 5824.6MB vs arena-free 4789.9MB → +21.6% FAIL
 *   medium steadyPeak: arena-ON 11306.7MB vs arena-free 12789.6MB → −11.6% PASS
 *   medium globalPeak: arena-ON 18011.4MB vs arena-free 15690.8MB → +14.8% FAIL
 * The ruling's cited inversion ("arena-ON 4278.5 < arena-free 4790") compared arena-ON
 * STEADY peak against arena-free GLOBAL peak — a methodology MISMATCH. Like-for-like the
 * inversion is NOT robust: distil FAILS under steady peak; BOTH configs FAIL under the
 * FIT-honest global peak. Per Stage-1's own STOP clause the campaign HALTED here — the
 * checkpoint bypass is RETAINED (deleting it would REGRESS distil peak +8.8%). This
 * oracle now gates the memory side on like-for-like steadyPeak and returns FAIL.
 *
 * The step-object campaign STOPPED on a hard two-gate conflict: no arena config
 * gives compiled + low-memory for checkpointed steps. This tool reproduces that
 * conflict deterministically as ONE committed gate with TWO sides, and records
 * its CURRENT (pre-R2) FAIL. It is the oracle R2 must flip to a two-sided PASS.
 *
 * The two contradictory gates (both on distil@512 + selective checkpointing):
 *
 *   MEMORY side  — t-planner-pin-attribution: arena-ON steady current is +155%
 *                  over arena-free (the planner registry pins the checkpointed
 *                  forward activations whole-step). The memory gate (peak/current
 *                  ≤ arena-free +5%) is met ONLY arena-free.
 *   WITNESS side — t-witness-harvest-matrix CELL=checkpoint: the observed-liveness
 *                  witness/prune mechanism only engages on the COMPILED path.
 *                  Arena-free runs LOWERED, so witnessedTemplates 4→0 = FAIL(B).
 *                  (inputNotReady=0 throughout — arena-free is SOUND, just inert.)
 *
 * TODAY (pre-R2): the two sides cannot both pass —
 *   - arena-ON  : WITNESS passes (witnessed=4, pruned=22), MEMORY fails (~4585 MB).
 *   - arena-free: MEMORY passes (~1798 MB), WITNESS fails(B) (witnessed=0).
 * VERDICT = FAIL (the two-gate conflict is live). R2 flips it: arena-ON compiled
 * becomes BOTH low-memory (registry liveness split) AND witness-engaged.
 *
 * === MEASURED THIS COMMIT (A100 dw-2-1 device 10, distil@512 + selective ckpt) ===
 *   arena-ON  : witnessedTemplates=4 prunedPairsRemoved=22 inputNotReady=0 | current=4584.7MB
 *   arena-free: witnessedTemplates=0 prunedPairsRemoved=0  inputNotReady=0 | current=1798.3MB
 *   VERDICT: FAIL — two-gate conflict LIVE (memory needs arena-free, witness needs arena-ON).
 *
 * This tool is an ORCHESTRATOR: it shells the two underlying probes sequentially
 * (one GPU process at a time — Dawn device-chain contention) and asserts the
 * two-gate conflict. It uses only existing flags: TORCHLETTE_NO_ARENA=1 (the
 * ambient arena-disable already honored at executor.ts:3628) drives the
 * witness-matrix arena-free side; no new env flag.
 *
 * Run (solo GPU):
 *   VULKAN_DEVICE_INDEX=10 LD_LIBRARY_PATH=tools/vk-shim \
 *     npx tsx tools/t-checkpoint-ab-oracle.ts
 * Exit 0 ⇒ two-sided PASS (R2 landed); exit 1 ⇒ the two-gate conflict is live
 * (the EXPECTED pre-R2 result — this gate is failing-first by construction).
 */
import { execFileSync } from "node:child_process";

const ROOT = process.cwd();
const log = (m: string) => console.error(`[ckpt-ab] ${m}`);

function run(
  env: Record<string, string>,
  args: string[],
  script: string,
): string {
  // The underlying probes EXIT 1 on their own FAIL verdict (by construction — this
  // is a failing-first gate). Capture stdout regardless of exit code; the JSON
  // verdict line is what we assert on, not the child exit status.
  try {
    return execFileSync("npx", ["tsx", script, ...args], {
      cwd: ROOT,
      env: { ...process.env, ...env },
      encoding: "utf8",
      stdio: ["ignore", "pipe", "inherit"],
      maxBuffer: 64 * 1024 * 1024,
    });
  } catch (e) {
    const out = (e as { stdout?: string }).stdout;
    if (typeof out === "string" && out.trim().length > 0) return out;
    throw e;
  }
}

function lastJsonLine(out: string): Record<string, unknown> {
  const lines = out
    .trim()
    .split("\n")
    .filter((l) => l.trim().startsWith("{"));
  if (lines.length === 0) throw new Error("no JSON verdict line in output");
  return JSON.parse(lines[lines.length - 1]);
}

function main() {
  const gpuEnv = {
    VULKAN_DEVICE_INDEX: process.env.VULKAN_DEVICE_INDEX ?? "10",
    LD_LIBRARY_PATH: process.env.LD_LIBRARY_PATH ?? "tools/vk-shim",
  };

  // --- WITNESS side, arena-ON (compiled): expect PASS (witnessed=4) ---
  log("WITNESS side, arena-ON (compiled)...");
  const witArenaOn = lastJsonLine(
    run(
      { ...gpuEnv, TORCHLETTE_STEP_TAPE: "record", CELL: "checkpoint" },
      [],
      "tools/t-witness-harvest-matrix.ts",
    ),
  );
  // --- WITNESS side, arena-free (lowered): expect FAIL(B) (witnessed=0) ---
  log("WITNESS side, arena-free (lowered)...");
  const witArenaFree = lastJsonLine(
    run(
      {
        ...gpuEnv,
        TORCHLETTE_STEP_TAPE: "record",
        TORCHLETTE_NO_ARENA: "1",
        CELL: "checkpoint",
      },
      [],
      "tools/t-witness-harvest-matrix.ts",
    ),
  );
  // --- MEMORY side, both modes (attribution) ---
  log("MEMORY side, arena-ON...");
  const memArenaOn = lastJsonLine(
    run(
      { ...gpuEnv, TORCHLETTE_STEP_TAPE: "record" },
      ["arena"],
      "tools/t-planner-pin-attribution.ts",
    ),
  );
  log("MEMORY side, arena-free...");
  const memArenaFree = lastJsonLine(
    run({ ...gpuEnv }, ["free"], "tools/t-planner-pin-attribution.ts"),
  );

  const witOnPass =
    witArenaOn.pass === true && witArenaOn.witnessedTemplates !== 0;
  const witFreePass =
    witArenaFree.pass === true && witArenaFree.witnessedTemplates !== 0;
  // D3 (2026-07-16): the memory side re-frames from steady-CURRENT-parity to
  // PEAK-parity (arena-ON peak ≤ arena-free peak + 5%), current reported-not-gated.
  // The comparison MUST be like-for-like: arena-ON steadyPeak vs arena-free steadyPeak
  // (the durable, reproducible watermark — global peak is warmup-arena noise). The
  // D3 ruling's cited inversion (arena-ON 4278.5 < arena-free 4790) was a methodology
  // MISMATCH — 4278.5 is arena-ON steadyPeak, 4790 is arena-free GLOBAL peak. Measured
  // like-for-like this commit (distil@512), the inversion does NOT hold.
  const arenaOnPeak = memArenaOn.peakMB as number;
  const arenaFreePeak = memArenaFree.peakMB as number;
  const arenaOnGlobalPeak = memArenaOn.globalPeakMB as number;
  const arenaFreeGlobalPeak = memArenaFree.globalPeakMB as number;
  const arenaOnCur = memArenaOn.currentMB as number;
  const arenaFreeCur = memArenaFree.currentMB as number;
  const memOnWithinBudget = arenaOnPeak <= arenaFreePeak * 1.05;

  log("=== TWO-GATE VERDICT ===");
  log(
    `WITNESS  arena-ON: templates=${witArenaOn.witnessedTemplates} pruned=${witArenaOn.prunedPairsRemoved} inputNotReady=${witArenaOn.inputNotReady} → ${witOnPass ? "PASS" : "FAIL"}`,
  );
  log(
    `WITNESS  arena-free: templates=${witArenaFree.witnessedTemplates} pruned=${witArenaFree.prunedPairsRemoved} inputNotReady=${witArenaFree.inputNotReady} → ${witFreePass ? "PASS" : "FAIL(B)"}`,
  );
  log(
    `MEMORY   arena-ON steadyPeak=${arenaOnPeak}MB vs arena-free ${arenaFreePeak}MB (+5% budget=${(arenaFreePeak * 1.05).toFixed(1)}MB) → ${memOnWithinBudget ? "PASS" : "FAIL"}`,
  );
  log(
    `MEMORY   (reported, not gated) current arena-ON=${arenaOnCur}MB free=${arenaFreeCur}MB | globalPeak arena-ON=${arenaOnGlobalPeak}MB free=${arenaFreeGlobalPeak}MB`,
  );

  // R2/D3 acceptance: arena-ON (compiled) must satisfy BOTH the witness gate AND the
  // peak-memory gate simultaneously — witness engaged AND peak within arena-free+5%.
  const twoSidedPass = witOnPass && memOnWithinBudget;
  if (twoSidedPass) {
    log(
      "VERDICT: PASS — arena-ON compiled is BOTH witness-engaged AND low-memory. R2 landed; the two-gate conflict is resolved.",
    );
    process.exit(0);
  }
  log(
    "VERDICT: FAIL — the peak-memory gate is NOT met: " +
      (memOnWithinBudget
        ? ""
        : `arena-ON steadyPeak ${arenaOnPeak}MB exceeds arena-free+5% (${(arenaFreePeak * 1.05).toFixed(1)}MB); `) +
      (witOnPass ? "" : "witness disengaged arena-ON; ") +
      (witFreePass
        ? ""
        : "witness disengaged arena-free (FAIL(B), sound-but-inert); ") +
      "arena-ON does NOT beat arena-free on peak like-for-like — the D3 bypass-deletion precondition (arena-ON peak ≤ arena-free peak + 5%) is unmet. The checkpoint bypass is RETAINED.",
  );
  process.exit(1);
}

try {
  main();
} catch (e) {
  console.error(`[ckpt-ab] FATAL: ${(e as Error)?.stack ?? e}`);
  process.exit(1);
}
