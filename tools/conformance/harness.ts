/**
 * THE EXTERNAL CONFORMANCE CORPUS — shared entry harness.
 *
 * Track-2 of the schedule-state campaign (docs/schedule-state-design.md §7 P4,
 * R25): grammar-completeness is a SEPARATE standing claim, gated on an EXTERNAL
 * published-kernel corpus. Each corpus entry is a runnable script under
 * tools/conformance/ that EXIT-0 asserts a published technique is reachable
 * (or provably refused) inside the move grammar — the corpus IS a test suite.
 *
 * This module is the shared spine every entry uses, so an entry file is just its
 * schema + its oracles (no per-file harness boilerplate). An ENTRY declares:
 *   - technique + citation      (the published kernel technique it conforms to)
 *   - base                       (the semantic base ScheduleState, named)
 *   - script                     (the move / lemma applications it performs)
 *   - outcome                    (byte-target where an in-repo kernel is the
 *                                 endpoint; numeric+cost-class otherwise;
 *                                 OR a documented BOUNDARY / typed REFUSAL)
 *   - the runnable gate          (the oracle assertions, run in `run()`)
 *
 * See docs/design-corpus/conformance.md for what "in-closure" means, what the
 * corpus covers, and what it provably does NOT (rungs 8-10, decoupled-lookback).
 *
 *   TORCHLETTE_CPU_ONLY=1 npx tsx tools/conformance/<entry>.ts   # oracles only
 *   VULKAN_DEVICE_INDEX=N LD_LIBRARY_PATH=tools/vk-shim:$LD_LIBRARY_PATH \
 *     npx tsx tools/conformance/<entry>.ts                        # + GPU parity
 */

/** One conformance-corpus entry's declarative header (printed at run start). */
export interface ConformanceEntry {
  /** Stable short id, used by the runner and the doc table (e.g. "grouped-matmul"). */
  readonly id: string;
  /** The published technique this entry conforms the grammar to. */
  readonly technique: string;
  /** The citation for the published technique (author/title/venue or URL). */
  readonly citation: string;
  /** The named semantic base state the derivation starts from. */
  readonly baseState: string;
  /** The move-script (+ lemma applications) as a human-readable one-liner. */
  readonly moveScript: string;
  /**
   * The expected outcome. `byte-target` = the derivation reaches an in-repo
   * kernel exactly. `numeric+cost` = numeric parity vs a reference + a stated
   * cost-class effect. `boundary` = a documented OUT-of-closure fact the entry
   * records honestly. `typed-refusal` = the corpus asserting NEGATIVE knowledge
   * (an illegal composition is refused by a typed rule).
   */
  readonly outcomeKind:
    | "byte-target"
    | "numeric+cost"
    | "boundary"
    | "typed-refusal";
  /** One-line statement of the expected outcome (the claim the gate checks). */
  readonly outcome: string;
  /** Which ladder exercise / in-closure claim this entry realizes. */
  readonly ladderRef: string;
}

/**
 * The run context handed to an entry's `run`: a scoped oracle asserter that
 * tracks failures, and CPU-only detection. The entry returns nothing; it calls
 * `oracle(...)` for every checked claim. A failing oracle → non-zero exit.
 */
export interface EntryContext {
  /** Assert one oracle. `cond` false → recorded failure (non-zero exit). */
  oracle(cond: boolean, msg: string): void;
  /** Print a boundary / honesty statement (not an assertion — always shown). */
  note(msg: string): void;
  /** True under TORCHLETTE_CPU_ONLY=1 (GPU parity blocks are skipped). */
  readonly cpuOnly: boolean;
}

export interface ConformanceModule {
  readonly entry: ConformanceEntry;
  /** The runnable gate. May be async (GPU parity). Calls ctx.oracle for each claim. */
  run(ctx: EntryContext): void | Promise<void>;
}

/**
 * Run one conformance module standalone: print its header, execute its gate,
 * print the oracle summary, and process.exit(0/1). Every entry file ends with
 * `runEntry(module)` so it is directly runnable (`npx tsx <entry>.ts`).
 *
 * The runner (run-all.ts) imports the module and calls `runModule` instead, so
 * a single Dawn teardown / process governs the whole corpus.
 */
export async function runEntry(mod: ConformanceModule): Promise<void> {
  const { failures } = await runModule(mod, { print: true });
  // Standalone: Dawn (if any entry touched it) is torn down inside runModule's
  // GPU blocks; here we just exit on the aggregate.
  process.exit(failures === 0 ? 0 : 1);
}

/**
 * Execute a module's gate and return its failure count. Shared by the standalone
 * `runEntry` and the corpus `run-all` runner. `print` controls whether the
 * per-entry header + summary are echoed (the runner prints its own framing).
 */
export async function runModule(
  mod: ConformanceModule,
  opts: { print: boolean },
): Promise<{ id: string; failures: number; oracleCount: number }> {
  const { entry } = mod;
  let failures = 0;
  let oracleCount = 0;
  const log = (s: string): void => {
    if (opts.print) {
      // eslint-disable-next-line no-console
      console.log(s);
    }
  };
  const ctx: EntryContext = {
    cpuOnly: process.env.TORCHLETTE_CPU_ONLY === "1",
    oracle(cond: boolean, msg: string): void {
      oracleCount++;
      if (cond) {
        log(`  ✓ ORACLE: ${msg}`);
      } else {
        failures++;
        log(`  ✗ ORACLE FAILED: ${msg}`);
      }
    },
    note(msg: string): void {
      log(`  · ${msg}`);
    },
  };

  log(`=== CONFORMANCE ENTRY [${entry.id}] ===`);
  log(`  technique : ${entry.technique}`);
  log(`  citation  : ${entry.citation}`);
  log(`  base      : ${entry.baseState}`);
  log(`  script    : ${entry.moveScript}`);
  log(`  outcome   : (${entry.outcomeKind}) ${entry.outcome}`);
  log(`  ladder    : ${entry.ladderRef}`);
  log("");

  try {
    await mod.run(ctx);
  } catch (err) {
    failures++;
    log(`  ✗ ENTRY THREW: ${err instanceof Error ? err.message : String(err)}`);
  }

  log("");
  log(
    `  [${entry.id}] ${failures === 0 ? "PASS" : "FAIL"} — ${oracleCount} oracle(s), ${failures} failure(s)`,
  );
  log("");
  return { id: entry.id, failures, oracleCount };
}
