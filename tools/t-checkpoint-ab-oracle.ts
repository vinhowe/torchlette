/**
 * CHECKPOINT TWO-GATE A/B ORACLE — task #99 phase R0 gate 2, the R2 acceptance
 * oracle (docs/arena-recompute-design.md §5 Phase R0 item 2 + §6 Phase 3 STOP).
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
  const arenaOnCur = memArenaOn.currentMB as number;
  const arenaFreeCur = memArenaFree.currentMB as number;
  const memOnWithinBudget = arenaOnCur <= arenaFreeCur * 1.05;

  log("=== TWO-GATE VERDICT ===");
  log(
    `WITNESS  arena-ON: templates=${witArenaOn.witnessedTemplates} pruned=${witArenaOn.prunedPairsRemoved} inputNotReady=${witArenaOn.inputNotReady} → ${witOnPass ? "PASS" : "FAIL"}`,
  );
  log(
    `WITNESS  arena-free: templates=${witArenaFree.witnessedTemplates} pruned=${witArenaFree.prunedPairsRemoved} inputNotReady=${witArenaFree.inputNotReady} → ${witFreePass ? "PASS" : "FAIL(B)"}`,
  );
  log(
    `MEMORY   arena-ON current=${arenaOnCur}MB vs arena-free ${arenaFreeCur}MB (+5% budget=${(arenaFreeCur * 1.05).toFixed(1)}MB) → ${memOnWithinBudget ? "PASS" : "FAIL"}`,
  );

  // R2 acceptance: arena-ON (compiled) must satisfy BOTH the witness gate AND the
  // memory gate simultaneously — witness engaged AND current within arena-free+5%.
  const twoSidedPass = witOnPass && memOnWithinBudget;
  if (twoSidedPass) {
    log(
      "VERDICT: PASS — arena-ON compiled is BOTH witness-engaged AND low-memory. R2 landed; the two-gate conflict is resolved.",
    );
    process.exit(0);
  }
  log(
    "VERDICT: FAIL (EXPECTED PRE-R2) — the two-gate conflict is LIVE: " +
      (memOnWithinBudget
        ? ""
        : `arena-ON memory ${arenaOnCur}MB exceeds arena-free+5% (${(arenaFreeCur * 1.05).toFixed(1)}MB); `) +
      (witOnPass ? "" : "witness disengaged arena-ON; ") +
      (witFreePass
        ? ""
        : "witness disengaged arena-free (FAIL(B), sound-but-inert); ") +
      "no config gives compiled + low-memory for checkpointed steps. R2 multi-segment liveness must flip this.",
  );
  process.exit(1);
}

try {
  main();
} catch (e) {
  console.error(`[ckpt-ab] FATAL: ${(e as Error)?.stack ?? e}`);
  process.exit(1);
}
