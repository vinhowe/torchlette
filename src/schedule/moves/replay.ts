/**
 * THE MOVE-SCRIPT REPLAYER (docs/p2-moves-design.md deliverable 5;
 * canonical.ts §"There is NO PARSER — printer + replayer only").
 *
 * A schedule PROGRAM is a base-state ref + an ordered list of typed moves. The
 * replayer applies a script's moves to a base state via the move algebra
 * (moves.ts), reproducing the state — that is the replay gate: applying a printed
 * script reproduces the state, digest-verified. This does NOT parse the printed
 * TEXT (there is no parser, by ruling); it consumes the in-memory `MoveScript`.
 *
 * Lives here (not in canonical.ts) to avoid an import cycle: the replayer needs
 * the move algebra, which transitively imports canonical.ts (via streamability →
 * attention-skeleton → canonical). canonical.ts stays dependency-light.
 */

import { type MoveScript, scheduleDigest } from "../canonical";
import type { ScheduleState } from "../types";
import { applyMove } from "./moves";

export type ReplayOutcome =
  | { kind: "ok"; state: ScheduleState }
  | { kind: "refused"; index: number; code: string; reason: string };

/**
 * Replay a move-script from `base`, applying each move in order via the move
 * algebra. Asserts the base digest matches the script's `baseDigest` (the
 * canonical form's base ref). Returns the final state, or the first refusal with
 * its index. A refusal here is a script that does not apply to its base — the
 * "jam" surfaced at the replay seam, never silently dropped.
 */
export function replayMoveScript(
  base: ScheduleState,
  script: MoveScript,
): ReplayOutcome {
  if (scheduleDigest(base) !== script.baseDigest)
    return {
      kind: "refused",
      index: -1,
      code: "BASE_DIGEST_MISMATCH",
      reason:
        `the base state's digest (${scheduleDigest(base)}) does not match the ` +
        `script's baseDigest (${script.baseDigest}).`,
    };
  let state = base;
  for (let i = 0; i < script.moves.length; i++) {
    const outcome = applyMove(state, script.moves[i]);
    if (outcome.kind === "refused")
      return {
        kind: "refused",
        index: i,
        code: outcome.refusal.code,
        reason: outcome.refusal.reason,
      };
    state = outcome.state;
  }
  return { kind: "ok", state };
}

/**
 * Round-trip a script: replay it and return the final digest. The FA derivation
 * uses this to prove the printed script replays to a digest-identical final state
 * (the deliverable-5 gate).
 */
export function replayDigest(base: ScheduleState, script: MoveScript): string {
  const outcome = replayMoveScript(base, script);
  if (outcome.kind === "refused")
    throw new Error(
      `replayDigest: move ${outcome.index} refused (${outcome.code}): ${outcome.reason}`,
    );
  return scheduleDigest(outcome.state);
}
