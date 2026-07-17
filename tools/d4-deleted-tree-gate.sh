#!/usr/bin/env bash
# D4 attempt #6 acceptance gate — the WITNESS-TO-CONVERGENCE deleted-tree oracle
# (docs/step-data-dependence-design.md §D4 "Failing-first").
#
# Proves the mechanism (converged-to-lowered witness stamping + the overlay-
# release consulting the derived edge set, §5) reproduces the recorded build's
# cross-plan witnessing coverage WITHOUT the recorded build: it exercises the
# DELETED-TREE condition honestly by applying the preserved deletion diff
# (.claude/D4-deletion-attempt5-STOPPED.diff) in a throwaway scratch worktree at
# HEAD, building it, and asserting the crux cells the STOP named:
#
#   scaler-inf : >= 6 producers AND >= 943 pairs*edges-class witnessing, threw=false
#                (main: 6 producers / 459 pairs / 943 edges; the deleted tree
#                 restores 6/459 with a ~2-edge consumer-multiplicity residual).
#   checkpoint : threw=false (the distilgpt2@512 selective-ckpt memory-stability
#                twin — the class-#2 Input-not-ready contiguous[32,128] symptom).
#
# It does NOT commit the deletion (that is attempt #6, a separate pass once this
# gate is green). The two import reconciliations below are the SAME hand-resolves
# the STOP note records for the 3-way apply (the deletion removed the
# crossPlanEdgeKeepSet / currentVariantSelection executor imports the mechanism
# now re-needs) — kept explicit so the gate is honest and reproducible.
#
# Usage:  DEVICE=0 bash tools/d4-deleted-tree-gate.sh
# GPU-only (like test:gates); needs the vk-shim + XDG_RUNTIME_DIR the repo uses.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && git rev-parse --show-toplevel)"
DIFF="$REPO/.claude/D4-deletion-attempt5-STOPPED.diff"
DEVICE="${DEVICE:-0}"
XDG="${XDG_RUNTIME_DIR:-/run/user/0}"
HEAD_SHA="$(cd "$REPO" && git rev-parse HEAD)"
WT="$(mktemp -d)/d4-del"

cleanup() { (cd "$REPO" && git worktree remove --force "$WT" 2>/dev/null) || true; rm -rf "$(dirname "$WT")" 2>/dev/null || true; }
trap cleanup EXIT

echo "[d4-gate] scratch worktree at $HEAD_SHA -> $WT"
(cd "$REPO" && git worktree add --detach "$WT" "$HEAD_SHA" >/dev/null 2>&1)
ln -sfn "$REPO/node_modules" "$WT/node_modules"

echo "[d4-gate] applying the deletion diff (3-way) + the recorded import reconciliations"
(cd "$WT" && git apply --3way "$DIFF" 2>/dev/null) || true
# Resolve the two import conflicts the deletion introduces vs the mechanism.
python3 - "$WT/src/executor/executor.ts" <<'PY'
import sys, re
f = sys.argv[1]; s = open(f).read()
# 1) collapse any conflict block around the cross-plan-edges import to the one
#    symbol the mechanism still needs on the deleted tree.
s = re.sub(
    r"<<<<<<< ours\nimport \{[^}]*\} from \"\.\./core/cross-plan-edges\";\n=======\n>>>>>>> theirs\n",
    'import { crossPlanEdgeHasOtherConsumer } from "../core/cross-plan-edges";\n',
    s)
# 2) the deletion drops currentVariantSelection from the step-variant import; the
#    overlay-release consultation re-needs it.
if "currentVariantSelection" not in s.split("../core/step-variant")[0].rsplit("import",1)[-1]:
    s = s.replace(
        'import { variantToken } from "../core/step-variant";',
        'import { currentVariantSelection, variantToken } from "../core/step-variant";')
open(f, "w").write(s)
PY
if grep -q "<<<<<<< ours" "$WT/src/executor/executor.ts"; then
  echo "[d4-gate] FAIL: unresolved conflict markers remain"; exit 1
fi

echo "[d4-gate] build"
(cd "$WT" && npm run build >/dev/null 2>&1)

run_cell() {
  local cell="$1"
  (cd "$WT" && XDG_RUNTIME_DIR="$XDG" VULKAN_DEVICE_INDEX="$DEVICE" \
    LD_LIBRARY_PATH=tools/vk-shim TORCHLETTE_STEP_TAPE=record CELL="$cell" \
    npx tsx tools/t-witness-harvest-matrix.ts 2>/dev/null | grep -E '^\{' | tail -1)
}

echo "[d4-gate] scaler-inf (the crux)"
SI="$(run_cell scaler-inf)"; echo "  $SI"
echo "[d4-gate] checkpoint (the memory-stability twin)"
CK="$(run_cell checkpoint)"; echo "  $CK"

node -e '
const si = JSON.parse(process.argv[1]); const ck = JSON.parse(process.argv[2]);
let ok = true;
const req = (c, cond, msg) => { if (!cond) { console.error(`  FAIL[${c}]: ${msg}`); ok = false; } };
req("scaler-inf", si.edgeProducers >= 6, `producers ${si.edgeProducers} < 6`);
req("scaler-inf", si.threw === false, "threw");
req("scaler-inf", si.shadowDerivedMissing === 0, `derived-missing ${si.shadowDerivedMissing}`);
req("checkpoint", ck.threw === false, "threw");
req("checkpoint", ck.inputNotReady === 0, `inputNotReady ${ck.inputNotReady}`);
if (ok) console.log(`[d4-gate] witnessing crux PASS — scaler-inf ${si.edgeProducers} producers/${si.edgePairs} pairs/${si.edges} edges threw=${si.threw}; checkpoint threw=${ck.threw}`);
else { console.error("[d4-gate] FAIL (witnessing crux)"); process.exit(1); }
' "$SI" "$CK"

# --- attempt #7 cells (the SIXTH class: WITNESSING != COMPILATION) ------------
# Failing-first as of 2026-07-17: build-from-IR does not COVER the fullstack
# optimizer plan / the embedding-grad backward plan the recorded build compiled,
# so on the deleted tree they run lowered forever. Consequences gated here:
#   tape      : t-train-tape-matrix fused+cosine must FORM a tape (today:
#               eligiblePairs=0 loweredPairs=6 tapeCount=0 -> exit 1).
#   profiler  : profile-training distil@512 must survive the build-from-IR
#               cutover at step 5 (today: Input not ready contiguous[512,768]
#               feeding the embedding-grad scatterAdd -> exit 139).
#   distil-ft : the in-suite Memory-Stability twin, single-file (today:
#               Input not ready contiguous[32,128], retry x2 -> exit 1).
# All three PASS on main under identical conditions. Run whole-node exclusive:
# device-chain contention on this node fabricates failures wholesale.
FAIL7=0
echo "[d4-gate] #7 tape (t-train-tape-matrix fused+cosine — a tape must form)"
(cd "$WT" && XDG_RUNTIME_DIR="$XDG" VULKAN_DEVICE_INDEX="$DEVICE" LD_LIBRARY_PATH=tools/vk-shim \
  TORCHLETTE_STEP_TAPE=record FUSED=1 SCHED=1 npx tsx tools/t-train-tape-matrix.ts >/dev/null 2>&1) \
  && echo "  PASS" || { echo "  FAIL (no eligible tape without the recorded build)"; FAIL7=1; }
echo "[d4-gate] #7 profiler (distil@512, 8 steps — the fall-through materialization)"
(cd "$WT" && XDG_RUNTIME_DIR="$XDG" VULKAN_DEVICE_INDEX="$DEVICE" LD_LIBRARY_PATH=tools/vk-shim \
  TORCHLETTE_PROFILE=1 TORCHLETTE_MODEL=distilgpt2 TORCHLETTE_SEQ_LEN=512 NUM_STEPS=8 \
  npx tsx tools/profile-training.ts >/dev/null 2>&1) \
  && echo "  PASS" || { echo "  FAIL (Input-not-ready at the build-from-IR cutover)"; FAIL7=1; }
echo "[d4-gate] #7 distil-ft single-file (the in-suite Memory-Stability twin)"
(cd "$WT" && XDG_RUNTIME_DIR="$XDG" VULKAN_DEVICE_INDEX="$DEVICE" LD_LIBRARY_PATH=tools/vk-shim \
  npx vitest run --project webgpu test/distilgpt2-full-finetuning.spec.ts >/dev/null 2>&1) \
  && echo "  PASS" || { echo "  FAIL (Memory-Stability throws Input-not-ready contiguous[32,128])"; FAIL7=1; }
if [ "$FAIL7" -ne 0 ]; then
  echo "[d4-gate] FAIL — the #7 compilation-coverage cells are red (the sixth class stands)"; exit 1
fi
echo "[d4-gate] PASS — witnessing crux AND #7 compilation-coverage cells green"
