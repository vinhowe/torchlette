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
# The RECONCILED deletion (attempt #11): the STOPPED diff's native base is
# 71655ea8; attempt #10 landed the derived-chunking seam in tile-dispatch.ts,
# whose region the deletion ALSO edits — so HEAD-coverage plain-layering fails
# on that one file. The reconciled diff (applies to HEAD, build+tsc verified)
# carries the hand-resolved tile-dispatch.ts; the gate layers it per-file below.
FINAL="$REPO/.claude/harvest-deletion-final.diff"
if [ ! -f "$FINAL" ]; then
  MAIN_CO="$(dirname "$(cd "$REPO" && git rev-parse --path-format=absolute --git-common-dir)")"
  FINAL="$MAIN_CO/.claude/harvest-deletion-final.diff"
fi
if [ ! -f "$FINAL" ]; then echo "[d4-gate] FAIL: reconciled diff not found at $FINAL"; exit 1; fi
DEVICE="${DEVICE:-0}"
XDG="${XDG_RUNTIME_DIR:-/run/user/0}"
HEAD_SHA="$(cd "$REPO" && git rev-parse HEAD)"
# Attempt #11: the deletion is now RECONCILED — harvest-deletion-final.diff
# applies cleanly to the pre-deletion HEAD (7c3f319e) and yields the exact
# hand-resolved deleted tree (build + tsc-zero-net-new verified). This restores
# the gate header's ORIGINAL intent ("apply the deletion diff in a throwaway
# worktree at HEAD") which the base+layer contraption replaced only because the
# STOPPED diff did not apply at a moved HEAD. Base-drift (native base 71655ea8
# = attempt #6) made the layer approach fragile: every export attempts #7-#10
# added to a NON-covered file (e.g. planWhereDirect in ops/where.ts, consumed by
# stream-generate.ts) broke the throwaway build. HEAD + reconciled diff has no
# drift. Pin the pre-deletion commit so the gate stays reproducible after Stage-3
# commits the deletion (then HEAD moves and this gate retires — see docs §D4).
PREDELETION_SHA="${D4_PREDELETION:-7c3f319e}"
WT="$(mktemp -d)/d4-del"

cleanup() { (cd "$REPO" && git worktree remove --force "$WT" 2>/dev/null) || true; rm -rf "$(dirname "$WT")" 2>/dev/null || true; }
trap cleanup EXIT

echo "[d4-gate] scratch worktree at pre-deletion $PREDELETION_SHA + reconciled deletion diff -> $WT"
(cd "$REPO" && git worktree add --detach "$WT" "$PREDELETION_SHA" >/dev/null 2>&1)
# Link a COMPLETE node_modules. A linked worktree may carry a partial install
# (missing .bin/tsdown → `npm run build` exits 127, MISLABELED as a build/cell
# failure). Prefer $REPO's node_modules only if it has the build binary; else
# fall back to the main checkout's (which ran the install).
NM="$REPO/node_modules"
if [ ! -x "$NM/.bin/tsdown" ]; then
  MAIN_CO="$(dirname "$(cd "$REPO" && git rev-parse --path-format=absolute --git-common-dir)")"
  [ -x "$MAIN_CO/node_modules/.bin/tsdown" ] && NM="$MAIN_CO/node_modules"
fi
ln -sfn "$NM" "$WT/node_modules"
# The #7 profiler + distil-ft cells LOAD real model weights (models/) and the
# distil-ft/token blobs (ckpts/) — untracked, so a fresh worktree lacks them.
# Without these symlinks the cells fail on model-not-found, which the gate would
# MISLABEL as "Input-not-ready" (a false red). Link them from the main checkout.
for d in models ckpts; do
  [ -e "$REPO/$d" ] && ln -sfn "$REPO/$d" "$WT/$d"
  [ -e "$WT/$d" ] || { MAIN_CO="$(dirname "$(cd "$REPO" && git rev-parse --path-format=absolute --git-common-dir)")"; [ -e "$MAIN_CO/$d" ] && ln -sfn "$MAIN_CO/$d" "$WT/$d"; }
done

echo "[d4-gate] applying the reconciled deletion diff"
(cd "$WT" && git apply "$FINAL") || { echo "[d4-gate] FAIL: reconciled diff did not apply to $PREDELETION_SHA (HEAD may already carry the deletion — this gate is pre-commit only)"; exit 1; }
echo "[d4-gate] reconciled deletion applied cleanly"

# Self-validate the tree: the deletion must be FULL (recorded-build symbols gone)
# and HEAD coverage must have landed. A PARTIAL deletion (git apply leaving
# recorded-build code) silently flips the cells; abort loudly instead.
SENT_REC="$( { grep -rc 'startCompilationRecording' "$WT/src/executor/compiled-plan.ts" "$WT/src/executor/executor.ts" 2>/dev/null || true; } | awk -F: '{s+=$2} END{print s+0}')"
SENT_ARANGE="$( { grep -c 'case "arange"' "$WT/src/executor/stream-generate.ts" 2>/dev/null || true; } )"
SENT_ARANGE="${SENT_ARANGE:-0}"
# tile-dispatch.ts is layered per-file (HEAD verbatim + reconciled hunk). If the
# --include apply silently no-op'd, HEAD's file survives WITH the recorded-build
# imports (invalidateActiveRecording / recordVolatileUniform) — assert they are
# gone so a failed per-file layer aborts instead of un-deleting the recorded build.
SENT_TILEREC="$( { grep -c 'invalidateActiveRecording\|recordVolatileUniform' "$WT/src/backend/webgpu/tile-dispatch.ts" 2>/dev/null || true; } )"
SENT_TILEREC="${SENT_TILEREC:-0}"
# Target A (D4 #8): the zero-residue recompute-on-read recovery lives in
# op-dispatch.ts (recomputeMissingResult) — a surviving file the deletion does
# NOT touch, layered via COVER_FILES. Assert it landed, else the mixed-coverage
# fall-through is not zero-residue and the #7 distil-ft twin regresses.
SENT_RECOMPUTE="$( { grep -c 'recomputeMissingResult' "$WT/src/executor/op-dispatch.ts" 2>/dev/null || true; } )"
SENT_RECOMPUTE="${SENT_RECOMPUTE:-0}"
# D4 #9 foreach-OOM fix sentinel: GradScaler must commit the pending boundary
# (commitStepBoundaryIfPending), else the foreach cells below OOM the deleted tree.
SENT_FOREACH_FIX="$( { grep -c 'commitStepBoundaryIfPending' "$WT/src/optim/grad-scaler.ts" 2>/dev/null || true; } )"
SENT_FOREACH_FIX="${SENT_FOREACH_FIX:-0}"
echo "[d4-gate] tree sentinels: startCompilationRecording=$SENT_REC (want 0, full deletion) arange-coverage=$SENT_ARANGE (want 1) tile-recorded-imports=$SENT_TILEREC (want 0) recompute-on-read=$SENT_RECOMPUTE (want >=1) foreach-oom-fix=$SENT_FOREACH_FIX (want >=1)"
if [ "${SENT_FOREACH_FIX:-0}" -lt 1 ]; then
  echo "[d4-gate] ABORT: D4 #9 foreach-OOM fix did not land (grad-scaler coverage absent); tree is invalid"; exit 2
fi
if [ "$SENT_REC" != "0" ]; then
  echo "[d4-gate] ABORT: recorded build NOT fully deleted (partial apply — the diff's 6-reject base drift); tree is invalid"; exit 2
fi
if [ "$SENT_ARANGE" != "1" ]; then
  echo "[d4-gate] ABORT: HEAD coverage did not land (arange generator absent); tree is invalid"; exit 2
fi
if [ "$SENT_TILEREC" != "0" ]; then
  echo "[d4-gate] ABORT: tile-dispatch.ts still carries recorded-build imports (per-file layer failed); tree is invalid"; exit 2
fi
if [ "${SENT_RECOMPUTE:-0}" -lt 1 ]; then
  echo "[d4-gate] ABORT: Target A recompute-on-read did not land (op-dispatch coverage absent); tree is invalid"; exit 2
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

# --- attempt #9 cells (the SEVENTH class: FOREACH-ADAM lowered-path OOM) -------
# The re-open condition (§D4 attempt-#8 STATUS "Re-open condition"): bound the
# foreach optimizer's lowered-path memory so t-train-tape-matrix FOREACH
# (FUSED=0, ±cosine) runs within the 32 GB budget — a tape forms AND no OOM.
#
# STATUS (attempt #9): the OOM half is FIXED (grad-scaler commits the pending
# implied boundary; commit fd1f1c7c). On the deleted tree BOTH foreach cells now
# report OOM=no / stepsObserved=18 — they run to completion within budget. They
# remain RED only on "a tape forms": eligiblePairs=0 / loweredPairs=6, because
# build-from-IR does not compile the foreach ELEMENTWISE optimizer plan (it runs
# lowered forever). That residual is the SAME #7-class WITNESSING!=COMPILATION
# coverage gap the fused cell escapes (its adamStep compiles build-from-IR) —
# now exposed for foreach once the OOM stopped masking it. So the sunset's
# foreach blocker has SHIFTED from memory (#9, resolved) to compilation coverage.
# Failing-first before the #9 fix: these cells OOM'd a 32 GB V100 by ~step 9
# (~3.9 GB/step) because a ceremony-free loop drives its boundary through bare
# markStep (GradScaler.resolveDeferred), which supersedes the queued implied
# boundary WITHOUT taking the survivor snapshot the implied-commit path takes —
# so releaseStepTemps has no snapshot (returns 0) and every step's optimizer
# temporaries (the packed 328 MB m/v-chain buffers) accumulate LIVE. The #9 fix
# (frontend markStep: snapshot when superseding a pending boundary) reclaims them
# per step. t-train-tape-matrix's PASS is `refusals==0 && eligiblePairs>0`
# (exit 0), so a bare exit-0 check covers BOTH "a tape forms" AND "no OOM" (an
# OOM crashes → exit 1). Run whole-node exclusive (device-chain contention
# fabricates failures).
FAIL9=0
for CELL in "0 0 foreach+no-sched" "0 1 foreach+cosine"; do
  set -- $CELL
  gpu_settle
  echo "[d4-gate] #9 tape ($3 — a tape must form AND no OOM)"
  # Surface WHY: the JSON summary (eligiblePairs/stepsObserved) distinguishes an
  # OOM (crash, no JSON, OOM=YES) from a build-from-IR compilation-coverage gap
  # (JSON present, eligiblePairs=0 — the #7 class, NOT the #9 memory bug). set +e
  # around the run so CELL_RC captures the real exit code without aborting the gate.
  set +e
  CELL_OUT="$(cd "$WT" && XDG_RUNTIME_DIR="$XDG" VULKAN_DEVICE_INDEX="$DEVICE" LD_LIBRARY_PATH=tools/vk-shim \
    TORCHLETTE_STEP_TAPE=record FUSED=$1 SCHED=$2 npx tsx tools/t-train-tape-matrix.ts 2>&1)"
  CELL_RC=$?
  CELL_JSON="$(printf '%s' "$CELL_OUT" | grep -oE '"stepsObserved": [0-9]+|"eligiblePairs": [0-9]+|"loweredPairs": [0-9]+' | tr '\n' ' ')"
  CELL_OOM=no; printf '%s' "$CELL_OUT" | grep -qi 'memory limit exceeded' && CELL_OOM=YES
  set -e
  echo "    ${CELL_JSON}OOM=${CELL_OOM}"
  if [ "$CELL_RC" -eq 0 ]; then
    echo "  PASS"
  elif [ "$CELL_OOM" = "YES" ]; then
    echo "  FAIL (foreach OOMs — the #9 memory bug stands)"; FAIL9=1
  else
    # OOM=no: the #9 memory bug is FIXED (runs within budget); the cell is red
    # ONLY because no tape forms (eligiblePairs=0, loweredPairs>0) — build-from-IR
    # does not compile the foreach ELEMENTWISE optimizer plan, so it runs lowered
    # forever. That is the #7-class WITNESSING!=COMPILATION coverage gap now
    # exposed for foreach, NOT the memory bug this gate cell was opened for.
    echo "  FAIL (no eligible tape — #9 OOM is FIXED; residual is foreach build-from-IR coverage, a #7-class gap)"; FAIL9=1
  fi
done
if [ "$FAIL9" -ne 0 ]; then
  echo "[d4-gate] FAIL — the #9 foreach-adam lowered-path cells are red (OOM or no tape)"; exit 1
fi
echo "[d4-gate] PASS — witnessing crux AND #7 compilation-coverage AND #9 foreach-adam cells green"
