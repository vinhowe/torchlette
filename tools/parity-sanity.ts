/**
 * Absolute-value sanity for parity/probe REFERENCE arms — the device-2 lesson.
 *
 * Parity gates diff two arms; they are BLIND to MUTUAL corruption. A silent
 * submit-drop (a tainted/OOM'd device that drops GPU submits without erroring)
 * makes BOTH arms read the same wrong value — often 0 — and the parity delta is
 * 0.0, so the gate PASSES while nothing trained. inc-3's first model-scale
 * measurement hit exactly this: every arm read loss=0 from step 1 on device 2
 * (a foreign 25.6GB idle allocation dropped submits), and the parity check was
 * happy about it.
 *
 * The defense is an ABSOLUTE assertion on the REFERENCE arm: the initial loss of
 * next-token cross-entropy over a vocab of size V, on uniform-random tokens (or
 * a fresh/lightly-trained model), sits near ln(V). A dropped-submit arm reads ~0
 * (or a NaN/frozen value), which is nowhere near ln(V). We assert loss[0] lands
 * in a plausible band DERIVED from V and fail LOUD with device-taint guidance.
 *
 * This is a floor, not a tight check — it exists to catch "everything is zero",
 * not to validate training quality. Keep the band wide enough that a legitimate
 * (if lightly-trained or checkpoint-loaded) initial loss never trips it.
 */

/** The plausible initial-loss band for next-token CE over a vocab of size V.
 *  Centered on ln(V); wide enough for random-init through lightly-trained. */
export function initialLossBand(vocabSize: number): [number, number] {
  const base = Math.log(vocabSize); // ln(50257) ≈ 10.82
  // Lower: a partially-trained / checkpoint-loaded model can start well below
  // ln(V); allow down to ~0.55·ln(V) (≈6.0 for GPT-2 vocab). A dropped-submit
  // arm reads ~0 — far below this. Upper: fresh init overshoots ln(V) modestly.
  return [base * 0.55, base + 2.0];
}

/**
 * Assert the reference arm's initial loss is in the ln(V)-derived band. Throws
 * with pointed device-taint guidance otherwise. `label` names the harness/arm
 * in the message. Also rejects non-finite loss[0] (NaN/Inf = a different
 * corruption, equally worth failing loud on).
 */
export function assertInitialLossSane(
  loss0: number,
  vocabSize: number,
  label: string,
): void {
  const [lo, hi] = initialLossBand(vocabSize);
  if (!Number.isFinite(loss0) || loss0 < lo || loss0 > hi) {
    throw new Error(
      `[sanity] ${label}: reference initial loss ${loss0} is OUTSIDE the ` +
        `plausible band [${lo.toFixed(2)}, ${hi.toFixed(2)}] for vocab=${vocabSize} ` +
        `(ln(V)=${Math.log(vocabSize).toFixed(2)}). This usually means a SILENT ` +
        `SUBMIT-DROP / DEVICE TAINT (a tainted or OOM'd GPU dropping submits so ` +
        `BOTH arms read ~0 and the parity delta is a false 0.0). Check nvidia-smi ` +
        `for a foreign allocation on this device; pin to a FREE device via ` +
        `VULKAN_DEVICE_INDEX + tools/vk-shim. A parity gate CANNOT catch mutual ` +
        `corruption — this absolute check is the only guard.`,
    );
  }
}

/**
 * Non-vocab reference arm (a toy MSE probe): assert `loss0` is finite and above
 * a small floor. A silent submit-drop reads ~0; the toy's real initial loss is
 * O(1). `label` names the arm; `floor` is the minimum plausible initial loss.
 */
export function assertReferenceLossNonzero(
  loss0: number,
  label: string,
  floor = 1e-3,
): void {
  if (!Number.isFinite(loss0) || loss0 < floor) {
    throw new Error(
      `[sanity] ${label}: reference initial loss ${loss0} is non-finite or below ` +
        `${floor} — the SILENT SUBMIT-DROP / DEVICE TAINT signature (a parity gate ` +
        `diffs two arms and CANNOT catch mutual corruption where both read ~0). ` +
        `Check nvidia-smi for a foreign allocation; pin to a FREE device via ` +
        `VULKAN_DEVICE_INDEX + tools/vk-shim.`,
    );
  }
}

/**
 * Token-level analog (kv-differential / decode): a dropped-submit arm reads
 * all-zero logits, so the per-position argmax variance collapses. Assert the
 * reference logits have real spread. `label` names the arm.
 */
export function assertLogitsSane(
  logits: Float32Array | number[],
  label: string,
): void {
  let mn = Infinity;
  let mx = -Infinity;
  let anyNaN = false;
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (!Number.isFinite(v)) {
      anyNaN = true;
      break;
    }
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const spread = mx - mn;
  if (anyNaN || spread < 1e-3) {
    throw new Error(
      `[sanity] ${label}: reference logits have ${anyNaN ? "NON-FINITE values" : `near-zero spread (max-min=${spread.toExponential(2)})`}. ` +
        `All-zero / flat logits are the SILENT SUBMIT-DROP / DEVICE TAINT ` +
        `signature (a token-level parity gate diffs two arms and CANNOT catch ` +
        `mutual corruption). Check nvidia-smi for a foreign allocation; pin to a ` +
        `FREE device via VULKAN_DEVICE_INDEX + tools/vk-shim.`,
    );
  }
}
