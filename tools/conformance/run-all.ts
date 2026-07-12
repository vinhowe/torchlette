/**
 * THE CONFORMANCE CORPUS RUNNER — the standing completeness gate.
 *
 * Runs EVERY conformance-corpus entry and exits 0 iff all entries pass. This is
 * the P4-adjacent standing gate (docs/schedule-state-design.md §7 R25/R4): the
 * grammar-completeness claim is FALSE while any registered corpus entry is
 * unrepresented (a failed derivation) — this runner is the check.
 *
 *   TORCHLETTE_CPU_ONLY=1 npx tsx tools/conformance/run-all.ts   # oracles only (CI)
 *   VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH \
 *     npx tsx tools/conformance/run-all.ts                        # + any GPU parity
 *
 * Adding an entry: create tools/conformance/<id>.ts exporting `module:
 * ConformanceModule` (see harness.ts + any existing entry as the template), then
 * add it to CORPUS below. The doc (docs/design-corpus/conformance.md) records the
 * entry's technique, citation, and outcome.
 */

import { module as chunkedReduction } from "./chunked-reduction"; // exercise 2

// The registered corpus. Order is the ladder order (rung-ascending).
import { module as coalescedTranspose } from "./coalesced-transpose"; // exercise 4
import { module as groupedMatmul } from "./grouped-matmul"; // R4 / ex 5-7
import { type ConformanceModule, runModule } from "./harness";
import { module as ksplitEpilogueRefusal } from "./ksplit-epilogue-refusal"; // exercise 7 (refusal)
import { module as layernormWelford } from "./layernorm-welford"; // exercise 3
import { module as onlineSoftmax } from "./online-softmax"; // exercise 8

const CORPUS: readonly ConformanceModule[] = [
  chunkedReduction,
  layernormWelford,
  coalescedTranspose,
  groupedMatmul,
  ksplitEpilogueRefusal,
  onlineSoftmax,
];

async function main(): Promise<void> {
  // eslint-disable-next-line no-console
  console.log(
    "=== EXTERNAL CONFORMANCE CORPUS — the standing grammar-completeness gate ===\n",
  );
  // eslint-disable-next-line no-console
  console.log(
    `  ${CORPUS.length} entries; CPU-only=${process.env.TORCHLETTE_CPU_ONLY === "1"}\n`,
  );

  const results: Array<{ id: string; failures: number; oracleCount: number }> =
    [];
  for (const mod of CORPUS) {
    // eslint-disable-next-line no-await-in-loop
    results.push(await runModule(mod, { print: true }));
  }

  // The per-entry outcome table.
  // eslint-disable-next-line no-console
  console.log("=== CORPUS SUMMARY ===\n");
  // eslint-disable-next-line no-console
  console.log("  entry                        status   oracles  failures");
  let totalFailures = 0;
  let totalOracles = 0;
  for (const r of results) {
    totalFailures += r.failures;
    totalOracles += r.oracleCount;
    // eslint-disable-next-line no-console
    console.log(
      `  ${r.id.padEnd(28)} ${(r.failures === 0 ? "PASS" : "FAIL").padEnd(8)} ${String(r.oracleCount).padStart(7)} ${String(r.failures).padStart(9)}`,
    );
  }
  // eslint-disable-next-line no-console
  console.log(
    `\n  TOTAL: ${results.length} entries, ${totalOracles} oracles, ${totalFailures} failures`,
  );

  const failedEntries = results.filter((r) => r.failures > 0).map((r) => r.id);
  // eslint-disable-next-line no-console
  console.log(
    `\n=== CORPUS ${totalFailures === 0 ? "GATE PASSED — every entry's derivation succeeds and its oracles hold" : `GATE FAILED — unrepresented entries: ${failedEntries.join(", ")}`} ===`,
  );

  process.exit(totalFailures === 0 ? 0 : 1);
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
