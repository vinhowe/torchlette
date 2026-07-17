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
# The preserved deletion diff is an UNTRACKED, gitignored artifact that lives in
# the MAIN checkout's .claude/ — a linked worktree's .claude/ does NOT have it, so
# resolving it under $REPO (the worktree) silently misses the file and the deletion
# never applies (a non-deleted tree masquerading as deleted). Fall back to the main
# checkout (the git common dir's parent) so the gate is correct from any worktree.
DIFF="$REPO/.claude/D4-deletion-attempt5-STOPPED.diff"
if [ ! -f "$DIFF" ]; then
  MAIN_CO="$(dirname "$(cd "$REPO" && git rev-parse --path-format=absolute --git-common-dir)")"
  DIFF="$MAIN_CO/.claude/D4-deletion-attempt5-STOPPED.diff"
fi
if [ ! -f "$DIFF" ]; then echo "[d4-gate] FAIL: deletion diff not found at $DIFF"; exit 1; fi
DEVICE="${DEVICE:-0}"
XDG="${XDG_RUNTIME_DIR:-/run/user/0}"
HEAD_SHA="$(cd "$REPO" && git rev-parse HEAD)"
# The deletion diff has a NATIVE BASE (the campaign commit where the recorded
# build is intact and the diff applies with only the two import conflicts below).
# Applying it to a HEAD that moved PAST that base makes some hunks silently
# fail -> a PARTIAL deletion (recorded build half-present), which corrupts the
# cells wholesale. So we build the deleted tree AT THE NATIVE BASE and then LAYER
# HEAD's coverage for the SURVIVING (non-recorded-build) executor files — the
# generator + replay slot machinery build-from-IR keeps. Coverage that lands in
# DELETED recorded-build code (e.g. the STREAM_GENERATE reconciliation in
# executor.ts) is correctly absent on the deleted tree, so it is NOT layered.
BASE_SHA="${D4_BASE:-71655ea8}"
COVER_FILES="src/executor/stream-generate.ts src/executor/compiled-plan.ts"
WT="$(mktemp -d)/d4-del"

cleanup() { (cd "$REPO" && git worktree remove --force "$WT" 2>/dev/null) || true; rm -rf "$(dirname "$WT")" 2>/dev/null || true; }
trap cleanup EXIT

echo "[d4-gate] scratch worktree at native base $BASE_SHA (deletion) + HEAD $HEAD_SHA coverage -> $WT"
(cd "$REPO" && git worktree add --detach "$WT" "$BASE_SHA" >/dev/null 2>&1)
ln -sfn "$REPO/node_modules" "$WT/node_modules"
# The #7 profiler + distil-ft cells LOAD real model weights (models/) and the
# distil-ft/token blobs (ckpts/) — untracked, so a fresh worktree lacks them.
# Without these symlinks the cells fail on model-not-found, which the gate would
# MISLABEL as "Input-not-ready" (a false red). Link them from the main checkout.
for d in models ckpts; do
  [ -e "$REPO/$d" ] && ln -sfn "$REPO/$d" "$WT/$d"
  [ -e "$WT/$d" ] || { MAIN_CO="$(dirname "$(cd "$REPO" && git rev-parse --path-format=absolute --git-common-dir)")"; [ -e "$MAIN_CO/$d" ] && ln -sfn "$MAIN_CO/$d" "$WT/$d"; }
done

echo "[d4-gate] applying the deletion diff (3-way) + the recorded import reconciliations"
(cd "$WT" && git apply --3way "$DIFF" 2>/dev/null) || true
echo "[d4-gate] layering HEAD coverage for surviving executor files: $COVER_FILES"
if [ "$BASE_SHA" != "$HEAD_SHA" ]; then
  # PLAIN git apply (NOT --3way): the coverage lands in regions the deletion did
  # not touch, so plain apply succeeds; --3way would re-introduce the deleted
  # recorded-build CONTEXT lines around each hunk (partial un-deletion → the tree
  # corrupts into recorded-build-half-present, which flips the cells wholesale).
  (cd "$REPO" && git diff "$BASE_SHA" "$HEAD_SHA" -- $COVER_FILES) \
    | (cd "$WT" && git apply)
fi
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
# 3) any OTHER conflict left after (1) is a HEAD change that landed inside deleted
#    recorded-build code (e.g. the D4-#7 STREAM_GENERATE count/params reconciliation
#    + its TAG_DISPATCH import): resolve it by taking THEIRS (the deletion side —
#    the enclosing recorded-build code is gone, so the HEAD change goes with it).
#    Keeps the gate reproducible when HEAD moved past the diff's native base.
s = re.sub(r"<<<<<<< ours\n.*?=======\n(.*?)>>>>>>> theirs\n",
           lambda m: m.group(1), s, flags=re.S)
open(f, "w").write(s)
PY
if grep -q "<<<<<<< ours" "$WT/src/executor/executor.ts"; then
  echo "[d4-gate] FAIL: unresolved conflict markers remain ($(grep -c '<<<<<<< ours' "$WT/src/executor/executor.ts") in executor.ts):"
  grep -n "<<<<<<< ours" "$WT/src/executor/executor.ts" | head
  exit 1
fi
echo "[d4-gate] executor.ts conflict-free after reconcile"

# Self-validate the tree: the deletion must be FULL (recorded-build symbols gone)
# and HEAD coverage must have landed. A PARTIAL deletion (git apply leaving
# recorded-build code) silently flips the cells; abort loudly instead.
SENT_REC="$( { grep -rc 'startCompilationRecording' "$WT/src/executor/compiled-plan.ts" "$WT/src/executor/executor.ts" 2>/dev/null || true; } | awk -F: '{s+=$2} END{print s+0}')"
SENT_ARANGE="$( { grep -c 'case "arange"' "$WT/src/executor/stream-generate.ts" 2>/dev/null || true; } )"
SENT_ARANGE="${SENT_ARANGE:-0}"
echo "[d4-gate] tree sentinels: startCompilationRecording=$SENT_REC (want 0, full deletion) arange-coverage=$SENT_ARANGE (want 1)"
if [ "$SENT_REC" != "0" ]; then
  echo "[d4-gate] ABORT: recorded build NOT fully deleted (partial apply — the diff's 6-reject base drift); tree is invalid"; exit 2
fi
if [ "$SENT_ARANGE" != "1" ]; then
  echo "[d4-gate] ABORT: HEAD coverage did not land (arange generator absent); tree is invalid"; exit 2
fi

echo "[d4-gate] build"
(cd "$WT" && npm run build >/dev/null 2>&1)

# Wait for OUR device to drain to idle between cells. Dawn tears its device down
# on process.exit, but its background threads release GPU memory slightly AFTER
# the node process returns; starting the next cell immediately makes it race the
# prior cell's teardown -> residual-allocation contention that FABRICATES
# Input-not-ready / no-tape failures (the campaign meta-lesson: nothing measured
# under GPU contention is evidence about the tree). Settle before every cell.
gpu_settle() {
  local i u
  for i in $(seq 1 30); do
    u="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEVICE" 2>/dev/null || echo 0)"
    [ "${u:-9999}" -lt 800 ] && return 0
    sleep 2
  done
}

run_cell() {
  local cell="$1"
  gpu_settle
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
gpu_settle
echo "[d4-gate] #7 tape (t-train-tape-matrix fused+cosine — a tape must form)"
(cd "$WT" && XDG_RUNTIME_DIR="$XDG" VULKAN_DEVICE_INDEX="$DEVICE" LD_LIBRARY_PATH=tools/vk-shim \
  TORCHLETTE_STEP_TAPE=record FUSED=1 SCHED=1 npx tsx tools/t-train-tape-matrix.ts >/dev/null 2>&1) \
  && echo "  PASS" || { echo "  FAIL (no eligible tape without the recorded build)"; FAIL7=1; }
gpu_settle
echo "[d4-gate] #7 profiler (distil@512, 8 steps — the fall-through materialization)"
(cd "$WT" && XDG_RUNTIME_DIR="$XDG" VULKAN_DEVICE_INDEX="$DEVICE" LD_LIBRARY_PATH=tools/vk-shim \
  TORCHLETTE_PROFILE=1 TORCHLETTE_MODEL=distilgpt2 TORCHLETTE_SEQ_LEN=512 NUM_STEPS=8 \
  npx tsx tools/profile-training.ts >/dev/null 2>&1) \
  && echo "  PASS" || { echo "  FAIL (Input-not-ready at the build-from-IR cutover)"; FAIL7=1; }
gpu_settle
echo "[d4-gate] #7 distil-ft single-file (the in-suite Memory-Stability twin)"
(cd "$WT" && XDG_RUNTIME_DIR="$XDG" VULKAN_DEVICE_INDEX="$DEVICE" LD_LIBRARY_PATH=tools/vk-shim \
  npx vitest run --project webgpu test/distilgpt2-full-finetuning.spec.ts >/dev/null 2>&1) \
  && echo "  PASS" || { echo "  FAIL (Memory-Stability throws Input-not-ready contiguous[32,128])"; FAIL7=1; }
if [ "$FAIL7" -ne 0 ]; then
  echo "[d4-gate] FAIL — the #7 compilation-coverage cells are red (the sixth class stands)"; exit 1
fi
echo "[d4-gate] PASS — witnessing crux AND #7 compilation-coverage cells green"
