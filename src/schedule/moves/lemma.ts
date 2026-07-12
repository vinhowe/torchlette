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
  D_PRECOMPUTE_LEMMA,
  D_PRECOMPUTE_OBLIGATION,
  dPrecomputeLemma,
  ONLINE_SOFTMAX_LEMMA,
  ONLINE_SOFTMAX_OBLIGATION,
  onlineSoftmaxLemma,
  RECOMPUTE_P_LEMMA,
  RECOMPUTE_P_OBLIGATION,
  recomputePLemma,
} from "../attention-skeleton";
import {
  WELFORD_LEMMA,
  WELFORD_OBLIGATION,
  welfordLemma,
} from "../reduction-skeleton";
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
 * The admitted-lemma library, keyed by the obligation each discharges (F28).
 * P2 admitted online softmax (the FA forward's license). P4 (§7 local
 * self-hosting) grows it by THREE — the attention-BACKWARD + variance lemmas that
 * license re-deriving the framework's own hand-crafted backward/variance kernels:
 *   - RECOMPUTE_P: recompute the [S,S] probabilities from the saved logsumexp L
 *     instead of materializing P (recompute-from-saved-statistic).
 *   - D_PRECOMPUTE: the rowsum(dO∘O) refactor — the per-(i,j) inner sum precomputed
 *     once per row as the carried statistic D.
 *   - WELFORD: the variance pair-merge (the teaching lemma, now engine-real;
 *     layernorm's inv-std path consumes the (count,mean,M2) triple).
 * Each carries its BoxRewrite + carried state + its own numerical differential.
 */
export const LEMMA_LIBRARY: Record<string, AdmittedLemma> = {
  [ONLINE_SOFTMAX_OBLIGATION]: {
    obligation: ONLINE_SOFTMAX_OBLIGATION,
    application: onlineSoftmaxLemma,
    matches: isSoftmaxShaped,
    rewrite: rewriteSoftmaxToOnline,
  },
  [RECOMPUTE_P_OBLIGATION]: {
    obligation: RECOMPUTE_P_OBLIGATION,
    application: recomputePLemma,
    matches: isMaterializedPShaped,
    rewrite: rewritePToRecompute,
  },
  [D_PRECOMPUTE_OBLIGATION]: {
    obligation: D_PRECOMPUTE_OBLIGATION,
    application: dPrecomputeLemma,
    matches: isInlineDInnerSumShaped,
    rewrite: rewriteDToPrecompute,
  },
  [WELFORD_OBLIGATION]: {
    obligation: WELFORD_OBLIGATION,
    application: welfordLemma,
    matches: isNaiveVarianceShaped,
    rewrite: rewriteVarianceToWelford,
  },
};

// keep the LemmaUid exports reachable (each lemma's identity is part of the seam).
void RECOMPUTE_P_LEMMA;
void D_PRECOMPUTE_LEMMA;
void WELFORD_LEMMA;

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
// P4 LEMMA 1 — RECOMPUTATION identity (attention backward, §7 P4)
// ============================================================================

/**
 * Recognize the MATERIALIZED-P backward body: the box reads a stored [S,S]
 * probability matrix (marker `materialized_P(scores)`), which the backward
 * schedule cannot fold block-locally without holding the whole O(S²) intermediate
 * — the refusal-first boundary the recomputation lemma discharges. We match the
 * marker apply so the pattern is a structural fact, not a label.
 */
function isMaterializedPShaped(expr: SemanticBodyNode): boolean {
  return isApplyOp(expr, "materialized_P");
}

/**
 * The BoxRewrite: replace `materialized_P(scores)` with
 * `recompute_P(scores, L)` — the marker the streamability predicate reads as the
 * post-lemma admitted body. It carries the saved logsumexp statistic `L` in the
 * lemma's carried state; P is recomputed as exp((Q·K)·scale − L) at the consume
 * site. Preserves the SCORES argument and adds the carried L value leaf so the
 * box's data dependency reflects the saved statistic (not the [S,S] store).
 */
function rewritePToRecompute(expr: SemanticBodyNode): SemanticBodyNode {
  const scores = firstArg(expr) ?? expr;
  return {
    kind: "apply",
    catalog: { op: "recompute_P" },
    args: [scores, { kind: "value", value: "L" as unknown as ValueUid }],
  };
}

/**
 * The recomputation lemma's OWN differential (§3.4): recompute P from the saved
 * logsumexp L and assert it equals the forward softmax row P. The identity is
 * `exp(s − L) == exp(s − m)/ℓ` where `L = m + log(ℓ)`. Evaluates both over a row
 * of raw scores and returns the max abs diff between the recomputed P and the
 * directly-normalized softmax P — the executable witness of the proof obligation
 * `recomputed P == forward P`.
 */
export function recomputePDifferential(scores: readonly number[]): {
  forwardP: number[];
  recomputedP: number[];
  maxAbsDiff: number;
} {
  const S = scores.length;
  // Forward: m, ℓ, then P[j] = exp(s[j] − m)/ℓ and the saved L = m + log(ℓ).
  let m = -Infinity;
  for (const x of scores) m = Math.max(m, x);
  let l = 0;
  for (const x of scores) l += Math.exp(x - m);
  const L = m + Math.log(l);
  const forwardP = scores.map((s) => Math.exp(s - m) / l);
  // Backward RECOMPUTE from the saved statistic: P[j] = exp(s[j] − L).
  const recomputedP = scores.map((s) => Math.exp(s - L));
  let maxAbsDiff = 0;
  for (let j = 0; j < S; j++)
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(forwardP[j] - recomputedP[j]));
  return { forwardP, recomputedP, maxAbsDiff };
}

// ============================================================================
// P4 LEMMA 2 — D-PRECOMPUTE (the rowsum(dO∘O) refactor, §7 P4)
// ============================================================================

/**
 * Recognize the INLINE per-(i,j) inner-sum backward body: the box computes
 * `Σ_k P[i,k]·(dO_i·V_k)` afresh for every (i,j) — marker
 * `inline_softmax_grad_innersum(...)`. This recomputed inner sum is what the
 * D-precompute refactor carries out of the loop as one saved per-row statistic.
 */
function isInlineDInnerSumShaped(expr: SemanticBodyNode): boolean {
  return isApplyOp(expr, "inline_softmax_grad_innersum");
}

/**
 * The BoxRewrite: replace the inline inner-sum marker with
 * `precomputed_D(dO, O)` — the marker carrying the saved per-row statistic
 * `D = rowsum(dO ∘ O)` (the dedicated D-precompute kernel's output). The
 * downstream `ds = P·(dO·V − D)` reads the carried D instead of recomputing the
 * inner sum per (i,j).
 */
function rewriteDToPrecompute(expr: SemanticBodyNode): SemanticBodyNode {
  void expr;
  return {
    kind: "apply",
    catalog: { op: "precomputed_D" },
    args: [
      { kind: "value", value: "dO" as unknown as ValueUid },
      { kind: "value", value: "O" as unknown as ValueUid },
    ],
  };
}

/**
 * The D-precompute lemma's OWN differential (§3.4): assert
 * `rowsum(dO ∘ O) == Σ_k P[i,k]·(dO·V_k)` where `O = Σ_k P[i,k] V_k`. Builds a
 * softmax row P from scores, forms O = P·V, and checks the precomputed statistic
 * D = Σ_d dO[d]·O[d] equals the inline sum Σ_k P[k]·(Σ_d dO[d]·V[k,d]).
 */
export function dPrecomputeDifferential(
  scores: readonly number[],
  values: readonly (readonly number[])[],
  dO: readonly number[],
): { precomputed: number; inline: number; absDiff: number } {
  const S = scores.length;
  const D = values[0]?.length ?? 0;
  // Softmax row P.
  let m = -Infinity;
  for (const x of scores) m = Math.max(m, x);
  let l = 0;
  for (const x of scores) l += Math.exp(x - m);
  const P = scores.map((s) => Math.exp(s - m) / l);
  // O = Σ_k P[k] V[k].
  const O = new Array<number>(D).fill(0);
  for (let k = 0; k < S; k++)
    for (let d = 0; d < D; d++) O[d] += P[k] * values[k][d];
  // Precomputed: D = Σ_d dO[d]·O[d].
  let precomputed = 0;
  for (let d = 0; d < D; d++) precomputed += dO[d] * O[d];
  // Inline: Σ_k P[k]·(Σ_d dO[d]·V[k,d]).
  let inline = 0;
  for (let k = 0; k < S; k++) {
    let dov = 0;
    for (let d = 0; d < D; d++) dov += dO[d] * values[k][d];
    inline += P[k] * dov;
  }
  return { precomputed, inline, absDiff: Math.abs(precomputed - inline) };
}

// ============================================================================
// P4 LEMMA 3 — WELFORD variance pair-merge (§7 P4; layernorm inv-std consumer)
// ============================================================================

/**
 * Recognize the NAIVE variance body:  `sub(reduce_mean(mul(x,x)), mul(mean, mean))`
 * — the `E[x²] − E[x]²` form (cancellation-prone, no stable block recomposition).
 * We match the top-level `sub` whose left arg is a mean-of-squares and whose right
 * arg is a square-of-mean (both structural). This is the value the refusal fires
 * on until the Welford lemma supplies the (count,mean,M2) pair-merge.
 */
function isNaiveVarianceShaped(expr: SemanticBodyNode): boolean {
  if (expr.kind !== "apply" || expr.catalog.op !== "sub") return false;
  if (expr.args.length !== 2) return false;
  const [lhs, rhs] = expr.args;
  // lhs = reduce_mean(mul(x,x))  (E[x²])
  const lhsIsMeanSq =
    lhs.kind === "apply" && isReduceMean(lhs) && isApplyOp(lhs.args[0], "mul");
  // rhs = mul(mean, mean) or square(mean)  (E[x]²)
  const rhsIsMeanSq =
    isApplyOp(rhs, "mul") || isApplyOp(rhs, "square") || isReduceMean(rhs);
  return Boolean(lhsIsMeanSq && rhsIsMeanSq);
}

function isReduceMean(node: SemanticBodyNode): boolean {
  return isApplyOp(node, "reduce_mean") || isApplyOp(node, "mean");
}

/**
 * The BoxRewrite: replace the naive `E[x²] − E[x]²` variance with
 * `welford_variance(x)` — the marker the streamability predicate reads as the
 * post-lemma admitted (streamable) body. It carries the (count, mean, M2)
 * pair-merge triple in the lemma's carried state; `var = M2/count`. Preserves the
 * input `x` leaf so the box's data dependency is unchanged.
 */
function rewriteVarianceToWelford(expr: SemanticBodyNode): SemanticBodyNode {
  const x = findVarianceInput(expr) ?? expr;
  return { kind: "apply", catalog: { op: "welford_variance" }, args: [x] };
}

/** Pull the `x` leaf out of the naive variance body (the mul(x,x) inside E[x²]). */
function findVarianceInput(expr: SemanticBodyNode): SemanticBodyNode | null {
  if (expr.kind !== "apply") return null;
  if (expr.catalog.op === "mul" && expr.args.length === 2) {
    // mul(x,x) → x
    return expr.args[0];
  }
  for (const a of expr.args) {
    const found = findVarianceInput(a);
    if (found) return found;
  }
  return null;
}

/**
 * The Welford lemma's OWN differential (§3.4): the (count,mean,M2) pair-merge over
 * ANY block partition equals the single-pass variance. Returns the single-pass
 * variance, the pair-merged variance over `blockSize` blocks, and their abs diff —
 * the executable witness that the pair-merge is numerically sound (the teaching
 * lemma's proof obligation).
 */
export function welfordDifferential(
  xs: readonly number[],
  blockSize = 4,
): { singlePass: number; welford: number; absDiff: number } {
  const N = xs.length;
  // Single pass (reference): var = mean((x − mean)²).
  let mean0 = 0;
  for (const x of xs) mean0 += x;
  mean0 /= N;
  let s0 = 0;
  for (const x of xs) s0 += (x - mean0) * (x - mean0);
  const singlePass = s0 / N;

  // Welford pair-merge over blocks: each block → (count, mean, M2); merge blocks.
  type Acc = { count: number; mean: number; M2: number };
  const merge = (a: Acc, b: Acc): Acc => {
    if (a.count === 0) return b;
    if (b.count === 0) return a;
    const count = a.count + b.count;
    const delta = b.mean - a.mean;
    const mean = a.mean + (delta * b.count) / count;
    const M2 = a.M2 + b.M2 + (delta * delta * a.count * b.count) / count;
    return { count, mean, M2 };
  };
  const blockAcc = (start: number, end: number): Acc => {
    let acc: Acc = { count: 0, mean: 0, M2: 0 };
    for (let i = start; i < end; i++) {
      const c = acc.count + 1;
      const d = xs[i] - acc.mean;
      const mean = acc.mean + d / c;
      const M2 = acc.M2 + d * (xs[i] - mean);
      acc = { count: c, mean, M2 };
    }
    return acc;
  };
  let total: Acc = { count: 0, mean: 0, M2: 0 };
  for (let start = 0; start < N; start += blockSize)
    total = merge(total, blockAcc(start, Math.min(start + blockSize, N)));
  const welford = total.M2 / total.count;

  return { singlePass, welford, absDiff: Math.abs(singlePass - welford) };
}

// ============================================================================
// Small tree helpers (shared by the P4 BoxRewrites)
// ============================================================================

function isApplyOp(node: SemanticBodyNode, op: string): boolean {
  return node.kind === "apply" && node.catalog.op === op;
}

function firstArg(node: SemanticBodyNode): SemanticBodyNode | undefined {
  return node.kind === "apply" ? node.args[0] : undefined;
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
