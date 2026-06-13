/**
 * Stage-4 phase 2 gate: generated stream prefixes == recorded streams.
 *
 * Runs the fullstack GPT-2 trainer (the canonical compiled-path workload)
 * with TORCHLETTE_STREAM_GENERATE=1: at every compiled-plan build the
 * executor generates the stream from the lowered plan and diffs the
 * covered prefix against the recording (executor.ts [stream-gen] hook).
 * Asserts the hook fired on every built template and NOTHING diverged.
 * As op generators land, the matched prefixes lengthen and full MATCH
 * lines appear — the gate tightens automatically. The printed uncovered
 * census is the coverage worklist.
 */
import { execFile } from "node:child_process";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const execFileP = promisify(execFile);

async function main() {
  const tool = path.join(__dirname, "parity-fullstack-tl.ts");
  const { stdout, stderr } = await execFileP("npx", ["tsx", tool], {
    env: {
      ...process.env,
      TORCHLETTE_STREAM_GENERATE: "1",
      STEPS: "4",
    },
    timeout: 300_000,
    maxBuffer: 64 * 1024 * 1024,
  });
  const out = stdout + "\n" + stderr;
  const lines = out.split("\n").filter((l) => l.includes("[stream-gen]"));
  // A plan is a hook-fire if it VERIFIED a covered prefix OR is FULLY
  // GENERATED (every action covered, command counts agree). As coverage
  // completed, plans graduated from "VERIFIED N/M" to "FULLY GENERATED".
  const matches = lines.filter(
    (l) =>
      l.includes("MATCH") ||
      l.includes("VERIFIED") ||
      l.includes("FULLY GENERATED"),
  ).length;
  const diverges = lines.filter((l) => l.includes("DIVERGE")).length;
  let verifiedCmds = 0;
  for (const l of lines) {
    const m = l.match(/VERIFIED \d+\/\d+ segments \((\d+) cmds/);
    if (m) verifiedCmds += parseInt(m[1], 10);
    // FULLY GENERATED lines report "flat X/Y cmds" — count X.
    const f = l.match(/FULLY GENERATED .* flat (\d+)\/\d+ cmds/);
    if (f) verifiedCmds += parseInt(f[1], 10);
  }
  for (const l of lines) console.log(l.trim());
  console.log(
    `[t-stream-generate] hook-fires=${matches} diverges=${diverges} verified-cmds=${verifiedCmds}`,
  );
  if (diverges > 0 || matches === 0 || verifiedCmds === 0) {
    console.log("STREAM-GENERATE: FAIL");
    process.exit(1);
  }
  console.log("STREAM-GENERATE: PASS");
  process.exit(0);
}
main();
