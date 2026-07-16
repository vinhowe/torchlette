/**
 * STEP VARIANTS — the declared residual structural forks of a recurring step
 * (docs/step-data-dependence-design.md §2, §3.3; campaign phase D0).
 *
 * WHAT THIS IS. Element A of the step's data-dependence: after value-level
 * variation is ROUTED AS DATA (the frozen-scalar discipline generalized — see
 * the GradScaler's unconditional scale write, §3.2), the RESIDUAL variation that
 * genuinely forks the plan graph is a small DECLARED enum. Each variant is a
 * structurally-distinct program a recurring step may run; each witnesses on its
 * OWN two-consecutive-within-variant stream (the `regime`-fence generalization,
 * §3.5). The per-step SELECTION is a RECEIPT (`StepReceipts.variant`), read at a
 * readback — it hashes into NEITHER identity; only the variant's structural
 * TOKEN does (§3.3 point 2).
 *
 * V1 IS A SINGLETON (`"train"`). The evidence (E-2/E-3) dissolved the motivating
 * "inf-skip variant": the GradScaler window is a route-as-data case, not a
 * structural fork (open-Q1 verdict, D0). So the declared set is a singleton and
 * this whole facet is NULL-CLEAN by construction:
 *
 *   SINGLETON == ABSENT ENCODING. `variantToken("train")` returns `undefined`,
 *   the exact value `computePlanFingerprint`'s `partitionToken` arg already
 *   treats as "no discriminator" (`fusion-detect.ts:1737` guard). So threading
 *   the variant token into the fingerprint seam (executor) is BYTE-IDENTICAL to
 *   the pre-variant path on every existing plan — the reification adds the seam
 *   without perturbing a single fingerprint. A future residual variant (eval, a
 *   structural milestone) returns a stable nonzero discriminator instead, and
 *   the SAME seam re-keys its template + re-witnesses its tape.
 *
 * SINGLE SOURCE (ruling 1: no second whole-step mechanism). The variant token
 * rides the EXISTING token-mixing primitive (`computePlanFingerprint`'s third
 * arg, the islands I1 seam), delivered from the step object's per-step selection
 * — NOT through the S3 `resolveEditedFingerprint` island-merge resolver (§3.4
 * verdict: that resolver is structure-keyed + plan-scoped + persistent; a variant
 * is step-scoped + receipt-selected). This module owns ONLY the declaration + the
 * current selection; the fingerprint math stays where it is.
 */

/** The declared residual structural variants of a recurring training step.
 *  V1: a singleton. Extended (never widened silently) when a genuine structural
 *  fork is declared — an unwitnessed selection is a typed refusal, never a stale
 *  replay (§6.4 UnwitnessedVariant). */
export type StepVariant = "train";

/** The declared variant set (§2: `variants: VariantSet`). A small enum, DATA on
 *  the existing step object — not a second owner. */
export type VariantSet = readonly StepVariant[];

/** The declared set — singleton in v1 (open-Q1 verdict: route-as-data, not
 *  declare-variant, so the scaler window does NOT add a variant). */
export const DECLARED_VARIANTS: VariantSet = ["train"];

/** The per-step runtime selection (the `receipts.variant` source). Resolved at a
 *  readback/flag; defaults to the base variant. Module-scoped so the fingerprint
 *  seam (executor) and the step-object derivation read ONE source. */
let currentVariant: StepVariant = "train";

/** Select the active variant for subsequent steps (the selector binding, §3.3
 *  point 1). A no-op in v1 (singleton), present so the seam is complete. */
export function selectVariant(v: StepVariant): void {
  currentVariant = v;
}

/** The per-step selector RECEIPT value (§3.3 point 2 — hashes into neither
 *  identity; recorded on `StepReceipts.variant`). */
export function currentVariantSelection(): StepVariant {
  return currentVariant;
}

/** Tiny FNV-1a over the variant name — the stable discriminator a FUTURE
 *  residual variant mixes into identity. Inlined (no `step-tape` import) to keep
 *  this module cycle-free (step-tape reads the selection from here). */
function variantHash(name: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * The variant TOKEN mixed into `computePlanFingerprint`'s token-mixing arg
 * (§3.3 point 3 / §3.4). SINGLETON == ABSENT: the base `"train"` variant returns
 * `undefined` (byte-identical to the no-token path); a residual variant returns
 * its stable nonzero hash so two variants of one graph key two templates.
 */
export function variantToken(
  v: StepVariant = currentVariant,
): number | undefined {
  return v === "train" ? undefined : variantHash(v);
}
