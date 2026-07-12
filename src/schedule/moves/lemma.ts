/**
 * LEMMA APPLICATION as a move-adjacent operation (docs/p2-moves-design.md
 * deliverable 2; schedule-state-design.md §3.4 F27/F28).
 *
 * ------------------------------------------------------------------------
 * WHY THIS IS NOT A MOVE
 * ------------------------------------------------------------------------
 * Moves rearrange WHEN/WHERE the same arithmetic happens; their legality is
 * structural. The online-softmax identity computes DIFFERENT intermediates for
 * the same function (accumulate softmax·V block-by-block, rescaling the partial
 * output by exp(m_old − m_new) whenever the running max rises). That is an
 * algebra fact about exp — an ADMITTED LEMMA, not a free structural rewrite. So
 * `applyLemma` is a move-adjacent operation: it rewrites the affected box's body
 * (the softmax `div(exp(x−m), sum(exp(x−m)))` root → `online_softmax(...)`
 * carrying the (m,ℓ,o) recurrence) and records the LemmaApplication in the
 * schedule's `lemmas` set (F27 first-class carried state, F28 obligation ID).
 *
 * ------------------------------------------------------------------------
 * WHAT DISCHARGING THE OBLIGATION MEANS
 * ------------------------------------------------------------------------
 * The `stream(softmax)` refusal (streamability.ts) NAMES the obligation
 * `ONLINE_SOFTMAX_OBLIGATION` its discharge would admit. `applyLemma` with the
 * matching lemma performs the BoxRewrite that supplies the (m,ℓ,o) recomposition
 * — after which the SAME streamability predicate ADMITS the value (F17
 * refusal-first → discharge → re-admit). The lemma library binding is by
 * ObligationId, NEVER by parsing refusal text.
 *
 * The lemma carries ITS OWN differential gate (the lemma's proof obligation is
 * its correctness): `onlineSoftmaxDifferential` evaluates the naive softmax body
 * and the online-softmax recurrence on the same inputs and asserts they agree to
 * the training-tolerance envelope. That is the numeric half the design's "each
 * lemma has its own differential gate" (§3.4) requires.
 */

import {
  ONLINE_SOFTMAX_LEMMA,
  ONLINE_SOFTMAX_OBLIGATION,
  onlineSoftmaxLemma,
} from "../attention-skeleton";
import type {
  LemmaApplication,
  ObligationId,
  ScheduleState,
  SemanticBody,
  SemanticBodyNode,
  SemanticSchedule,
  ValueUid,
} from "../types";
import { classifyBody } from "./streamability";

// ============================================================================
// The lemma library — obligation ID → lemma (F28 binding, not refusal text)
// ============================================================================

/** A rewrite the lemma performs on a matching body, plus its carried state. */
export interface AdmittedLemma {
  readonly obligation: ObligationId;
  /** Produce the LemmaApplication record (first-class carried state, F27). */
  readonly application: () => LemmaApplication;
  /** Whether this lemma's BoxRewrite matches a given body root (the pattern). */
  readonly matches: (body: SemanticBodyNode) => boolean;
  /** The BoxRewrite: rewrite the matched body into its post-lemma form. */
  readonly rewrite: (body: SemanticBodyNode) => SemanticBodyNode;
}

/**
 * The v1 admitted-lemma library, keyed by the obligation each discharges (F28).
 * v1 admits exactly the one entry the P2 acceptance needs: online softmax.
 * (The Welford lemma — variance streaming — is P4 territory; mean streams
 * directly as the (sum,count) monoid, no lemma. See §5 ruling 3.)
 */
export const LEMMA_LIBRARY: Record<string, AdmittedLemma> = {
  [ONLINE_SOFTMAX_OBLIGATION]: {
    obligation: ONLINE_SOFTMAX_OBLIGATION,
    application: onlineSoftmaxLemma,
    matches: isSoftmaxShaped,
    rewrite: rewriteSoftmaxToOnline,
  },
};

/** Look up the lemma that discharges an obligation (the F28 seam). */
export function lemmaFor(obligation: ObligationId): AdmittedLemma | undefined {
  return LEMMA_LIBRARY[obligation as unknown as string];
}

// ============================================================================
// applyLemma — the move-adjacent operation
// ============================================================================

export type LemmaOutcome =
  | { kind: "applied"; state: ScheduleState; application: LemmaApplication }
  | { kind: "refused"; reason: string };

/**
 * Apply the lemma that discharges `obligation` to the body producing `value` in
 * `state`. Rewrites the box's body (the BoxRewrite) and records the
 * LemmaApplication in the schedule's `lemmas` set. Refused (at the seam) if the
 * lemma's pattern does not match the value's body — a lemma cannot be applied
 * where its obligation does not arise.
 */
export function applyLemma(
  state: ScheduleState,
  value: ValueUid,
  obligation: ObligationId,
): LemmaOutcome {
  const lemma = lemmaFor(obligation);
  if (!lemma)
    return {
      kind: "refused",
      reason: `no admitted lemma discharges obligation ${obligation}.`,
    };
  const s = state.semantic;
  const idx = s.bodies.findIndex((b) => b.result === value);
  if (idx < 0)
    return {
      kind: "refused",
      reason: `value ${value} has no body in the schedule; nothing to rewrite.`,
    };
  const body = s.bodies[idx];
  if (!lemma.matches(body.expr))
    return {
      kind: "refused",
      reason:
        `the ${obligation} lemma's BoxRewrite does not match value ${value}'s body ` +
        `root — the obligation does not arise here.`,
    };

  const rewritten: SemanticBody = {
    result: body.result,
    expr: lemma.rewrite(body.expr),
  };
  const bodies = s.bodies.map((b, i) => (i === idx ? rewritten : b));
  const application = lemma.application();
  const lemmas = [...s.lemmas, application];
  const semantic: SemanticSchedule = { ...s, bodies, lemmas };
  return { kind: "applied", state: { ...state, semantic }, application };
}

// ============================================================================
// The online-softmax BoxRewrite (§3.4 NCD F7)
// ============================================================================

/**
 * Recognize the numerically-stable softmax body
 *   div( exp(sub(x, max(...))), sum(exp(sub(x, max(...)))) ).
 * (Same discriminator as streamability's `isSoftmaxShaped`, owned here for the
 * rewrite; the two agree by construction — the classify seam re-checks.)
 */
function isSoftmaxShaped(expr: SemanticBodyNode): boolean {
  if (expr.kind !== "apply" || expr.catalog.op !== "div") return false;
  if (expr.args.length !== 2) return false;
  return containsReduction(expr.args[1]);
}

function containsReduction(node: SemanticBodyNode): boolean {
  if (node.kind !== "apply") return false;
  if (node.catalog.op === "sum" || node.catalog.op === "reduce_sum")
    return true;
  return node.args.some(containsReduction);
}

/**
 * The BoxRewrite: replace the div-by-full-denominator softmax root with
 * `online_softmax(scores)` — the marker apply the streamability predicate reads
 * as the post-lemma streamable body (it carries the (m,ℓ,o) recurrence in the
 * lemma's carried state, not in the body tree). The rewrite preserves the
 * SCORES argument (the value the running max/normalizer fold over) so the box's
 * data dependency is unchanged.
 */
function rewriteSoftmaxToOnline(expr: SemanticBodyNode): SemanticBodyNode {
  const scores = extractScores(expr);
  return {
    kind: "apply",
    catalog: { op: "online_softmax" },
    args: [scores],
  };
}

/** Pull the `scores` value out of the softmax body's exp(sub(x, max)) numerator. */
function extractScores(expr: SemanticBodyNode): SemanticBodyNode {
  // expr = div(exp(sub(x, max(x))), sum(...)). Walk to the innermost `x` operand
  // of the sub. Falls back to the numerator subtree if the shape is unexpected.
  if (expr.kind === "apply" && expr.catalog.op === "div") {
    const num = expr.args[0];
    const x = findSubLeft(num);
    if (x) return x;
    return num;
  }
  return expr;
}

function findSubLeft(node: SemanticBodyNode): SemanticBodyNode | null {
  if (node.kind !== "apply") return null;
  if (node.catalog.op === "sub" && node.args.length === 2) return node.args[0];
  for (const a of node.args) {
    const found = findSubLeft(a);
    if (found) return found;
  }
  return null;
}

// ============================================================================
// The lemma's OWN differential gate (§3.4 — each lemma has its own gate)
// ============================================================================

/**
 * The online-softmax lemma's numeric differential: evaluate the NAIVE softmax
 * (full-row div-by-denominator) and the ONLINE recurrence ((m,ℓ,o) with the
 * exp(m_old−m_new) rescale) over the SAME scores/values, and return the max
 * absolute difference. The lemma's proof obligation is exactly that these agree
 * — this is its executable witness (F33). The FA derivation runs it as the gate
 * that the discharge is numerically sound before streaming the post-lemma body.
 *
 * scores: [S] row of attention logits (one KV row); values: [S][D] the V rows.
 * Returns the softmax·V output both ways and their max abs diff.
 */
export function onlineSoftmaxDifferential(
  scores: readonly number[],
  values: readonly (readonly number[])[],
  blockSize = 4,
): { naive: number[]; online: number[]; maxAbsDiff: number } {
  const S = scores.length;
  const D = values[0]?.length ?? 0;

  // ---- Naive: full-row softmax, then P·V. ----
  let m = -Infinity;
  for (const x of scores) m = Math.max(m, x);
  let l = 0;
  const p = new Array<number>(S);
  for (let j = 0; j < S; j++) {
    p[j] = Math.exp(scores[j] - m);
    l += p[j];
  }
  const naive = new Array<number>(D).fill(0);
  for (let j = 0; j < S; j++)
    for (let d = 0; d < D; d++) naive[d] += (p[j] / l) * values[j][d];

  // ---- Online: block-by-block (m,ℓ,o) recurrence with exp(m_old−m_new) rescale. ----
  let mRun = -Infinity;
  let lRun = 0;
  const o = new Array<number>(D).fill(0);
  for (let start = 0; start < S; start += blockSize) {
    const end = Math.min(start + blockSize, S);
    // block max
    let mBlock = -Infinity;
    for (let j = start; j < end; j++) mBlock = Math.max(mBlock, scores[j]);
    const mNew = Math.max(mRun, mBlock);
    // rescale the running accumulators by exp(m_old − m_new)
    const rescale = mRun === -Infinity ? 0 : Math.exp(mRun - mNew);
    lRun *= rescale;
    for (let d = 0; d < D; d++) o[d] *= rescale;
    // fold the block
    for (let j = start; j < end; j++) {
      const pj = Math.exp(scores[j] - mNew);
      lRun += pj;
      for (let d = 0; d < D; d++) o[d] += pj * values[j][d];
    }
    mRun = mNew;
  }
  const online = o.map((x) => x / lRun);

  let maxAbsDiff = 0;
  for (let d = 0; d < D; d++)
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(naive[d] - online[d]));
  return { naive, online, maxAbsDiff };
}

// ============================================================================
// The discharge round-trip helper (F17 read as an assertion)
// ============================================================================

/**
 * Assert the F17 discharge round-trip on a value's body: the pre-lemma body
 * REFUSES streaming and names `obligation`; the post-lemma body (after the
 * matching BoxRewrite) is ADMITTED. Returns the obligation named + the two
 * verdicts so the FA script can assert each oracle. Throws if the pre-body is
 * not the softmax shape the lemma expects.
 */
export function assertDischargeRoundTrip(body: SemanticBodyNode): {
  obligationNamed: ObligationId;
  postAdmitted: boolean;
} {
  const before = classifyBody(body);
  if (before.streamable)
    throw new Error(
      "assertDischargeRoundTrip: the pre-lemma body is already streamable — " +
        "the refusal-first boundary does not arise.",
    );
  const obligationNamed = before.refusal.dischargedBy as ObligationId;
  const lemma = lemmaFor(obligationNamed);
  if (!lemma)
    throw new Error(
      `assertDischargeRoundTrip: no lemma discharges the named obligation ${obligationNamed}.`,
    );
  const after = classifyBody(lemma.rewrite(body));
  return { obligationNamed, postAdmitted: after.streamable };
}

/** Re-export the wave-3 engine objects the FA script binds against. */
export { ONLINE_SOFTMAX_LEMMA, ONLINE_SOFTMAX_OBLIGATION };
