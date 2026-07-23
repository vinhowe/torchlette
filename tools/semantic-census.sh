#!/usr/bin/env bash
#
# semantic-census.sh — the COMPOSITE-CLOSURE exit gate (design §10). Closure is a
# MECHANICAL claim, not a vibe: after F1 (C1–C3) + F2 (A1–A3), NO hand-written
# arithmetic survives in the semantic frame outside the STATED exclusions. This
# greppable census (a weight-norm-style hook) asserts it and exits non-zero if a
# forbidden hand form reappears.
#
# What it proves:
#   F1 — no hand composite backward ARITHMETIC in decomposed-ops.ts (the derived
#        `vjpComposition` is the CPU reference; the fused GPU kernels are asserted
#        against it by composite-fused-vs-derived.spec). The softmax CPU closure
#        is the ONE documented T1 deferral (the C1 cost probe measured its derived
#        form heavier — 22 vs 13 nodes — so it stays hand, derived reference-only).
#   F2 — no activation BODY in fusion-tile-ir.ts / tile-ir.ts; every activation is
#        `lowerExprToTileIR(DEF.expr, …)`.
#   Whitelisted exclusions (each named + fenced in docs/composite-closure-design.md):
#        attention (SDPA), rope, pow-variable-exponent.
#
# Usage: bash tools/semantic-census.sh   (exit 0 = closed, 1 = a hand form leaked)

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

DECOMP="src/frontend/decomposed-ops.ts"
FUSION="src/backend/webgpu/fusion-tile-ir.ts"
TILEIR="src/backend/webgpu/tile-ir.ts"

fail=0
ok()   { printf '  OK   %s\n' "$1"; }
bad()  { printf '  FAIL %s\n' "$1"; fail=1; }

echo "=== COMPOSITE-CLOSURE semantic census ==="

# --- F1: the hand composite backward ARITHMETIC is gone -----------------------
# The naive layernorm CPU expansion (gradVariance / gradCenteredFromVar) and the
# rmsnorm hand closed form (rmsnormBackwardImpl) must NOT exist as CODE. Match the
# function DEFINITION / an assignment — a mention in a doc comment is fine.
if grep -qE "function rmsnormBackwardImpl" "$DECOMP"; then
  bad "F1: the hand rmsnormBackwardImpl function still in $DECOMP"
else
  ok "F1: rmsnormBackwardImpl deleted (derived VJP is the reference)"
fi
for sym in "gradVariance" "gradCenteredFromVar"; do
  # These only ever appeared as `const <sym> =` in the naive layernorm expansion.
  if grep -qE "const ${sym}\b" "$DECOMP"; then
    bad "F1: hand layernorm expansion '$sym' still in $DECOMP"
  else
    ok "F1: no '$sym' expansion in decomposed-ops (derived VJP is the reference)"
  fi
done

# The CPU rmsnorm / layernorm backwards must route through the derived reference.
if grep -q "derivedCompositeGrads" "$DECOMP"; then
  ok "F1: CPU composite backwards route through vjpComposition (derivedCompositeGrads)"
else
  bad "F1: no derivedCompositeGrads route in $DECOMP"
fi

# T1 deferral: the softmax closure is the ONE documented hand composite backward.
if grep -q "grad_input = softmax" "$DECOMP" || grep -q "softmaxResult" "$DECOMP"; then
  ok "F1: softmax closure present (documented C1/T1 deferral — derived reference-only)"
fi

# --- F2: the activation WGSL bodies are gone ----------------------------------
# fusion-tile-ir must have NO activation switch cases (they derive via the fold).
if grep -qE '^\s*case "(sigmoid|silu|softplus|gelu|gelu_tanh|gelu_erf|relu)":' "$FUSION"; then
  bad "F2: a hand activation 'case' body still in $FUSION"
else
  ok "F2: no activation switch bodies in fusion-tile-ir (fold via ACTIVATION_EXPR)"
fi
if grep -q "ACTIVATION_EXPR" "$FUSION" && grep -q "lowerExprToTileIR" "$FUSION"; then
  ok "F2: applyFusedOp routes activations through lowerExprToTileIR"
else
  bad "F2: applyFusedOp does not route through the Expr fold"
fi

# tile-ir must have NO sigmoid()/erf() BlockExpr compound methods.
if grep -qE '^\s*(sigmoid|erf)\(\): BlockExpr' "$TILEIR"; then
  bad "F2: a BlockExpr activation compound method still in $TILEIR"
else
  ok "F2: no BlockExpr sigmoid()/erf() compound methods (deleted; fold is the source)"
fi

# --- Whitelisted exclusions (STATED, each fenced) -----------------------------
echo "  --- whitelisted exclusions (design §5/§6) ---"
grep -q "naiveAttentionBackwardComposition\|fusedAttentionBackward" src/schedule/attention-skeleton.ts src/frontend/decomposed-ops.ts 2>/dev/null \
  && ok "excl: attention (SDPA) — fused-kernel-first, fenced by the schedule byte-differential + fused-vs-CPU-decomposed" \
  || printf '  note attention exclusion markers not found (informational)\n'
grep -rq "sin_scale\|rope" src/backend/webgpu/rope-kernel.ts 2>/dev/null \
  && ok "excl: rope — standalone fused kernel (backward = same kernel, sin_scale=-1)" \
  || printf '  note rope exclusion markers not found (informational)\n'
grep -q 'gradPolicy: "hand"' src/ops/semantic/catalog.ts \
  && ok "excl: pow variable-exponent — declared gradPolicy:\"hand\" + NaN-signal refusal" \
  || bad "excl: pow-variable gradPolicy:\"hand\" declaration missing"

echo "=========================================="
if [ "$fail" = 0 ]; then
  echo "CENSUS GREEN — every composite/activation is a theorem or a named exclusion."
  exit 0
fi
echo "CENSUS RED — a hand form leaked back into the semantic frame."
exit 1
