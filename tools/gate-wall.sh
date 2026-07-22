#!/usr/bin/env bash
#
# gate-wall.sh — the standing correctness wall as ONE command.
#
# Runs the gate battery SERIALLY (GPU work is serial-exclusive — see
# docs/agent-ops.md), captures a bare exit code per gate, and — on any GPU-gate
# failure — does one AUTO-ISOLATED RERUN (the documented vkCreateDevice/OOM
# flake protocol: a device-contention failure clears on a clean re-run in its
# own process). The final table classifies each gate:
#
#     PASS  first run green
#     ENV   first run failed, isolated rerun green  → environmental flake
#     REAL  failed both times                       → a real regression
#
# Profiles (cumulative):
#   quick     build, test:gates, whole-step differential (both compiled modes)
#   training  quick + fullstack parity, tape-matrix (4 cells), step-object-null,
#             step-edit-null, ring-probe, ledger (default), 124M regression,
#             whole-step-checkpoint refusal spec, checkpoint-segmentation spec
#   full      training + witness-harvest matrix (5 cells), ledger (48 steps),
#             distil+medium profiles, FULL vitest suite (cpu+webgpu)
#
# Usage:
#   tools/gate-wall.sh --profile quick
#   tools/gate-wall.sh --profile training
#   tools/gate-wall.sh --profile full --no-pick     # reuse env VULKAN_DEVICE_INDEX
#
# Env:
#   GATE_TIMEOUT   per-gate timeout seconds (default 3600)
#   MODEL          model for whole-step/tape gates (default distilgpt2)
#   VULKAN_DEVICE_INDEX  if set and --no-pick given, reused instead of picking
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || { echo "gate-wall: cannot cd to $ROOT" >&2; exit 1; }
PROFILE="quick"
DO_PICK=1
GATE_TIMEOUT="${GATE_TIMEOUT:-3600}"
MODEL="${MODEL:-distilgpt2}"

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    --no-pick) DO_PICK=0; shift;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "gate-wall: unknown arg $1" >&2; exit 2;;
  esac
done
case "$PROFILE" in quick|training|full) ;; *) echo "bad --profile: $PROFILE" >&2; exit 2;; esac

# --- worktree pre-flight: refuse to run in an unset-up tree ------------------
# Enforcement beats instruction. A missing models/.venv symlink (the item-1 gap)
# silently fails half the gates AND makes the torch oracle look "unavailable".
# The wall will not run until agent-worktree-setup.sh --verify is green.
if ! bash "$ROOT/tools/agent-worktree-setup.sh" --verify >/dev/null 2>&1; then
  echo "[gate-wall] REFUSING to run: worktree is not set up." >&2
  echo "[gate-wall] --- agent-worktree-setup.sh --verify ---" >&2
  bash "$ROOT/tools/agent-worktree-setup.sh" --verify >&2 || true
  echo "[gate-wall] fix: bash tools/agent-worktree-setup.sh   (then re-run the wall)" >&2
  exit 3
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
LOGDIR="${TMPDIR:-/tmp}/gate-wall-$STAMP"
mkdir -p "$LOGDIR"
RESULTS="$LOGDIR/results.tsv"
: >"$RESULTS"
log() { echo "[gate-wall] $*" >&2; }

# --- GPU reservation ---------------------------------------------------------
OWNER="$$"
released=0
release_gpu() {
  [ "$released" = 1 ] && return
  released=1
  if [ "$DO_PICK" = 1 ]; then
    TORCHLETTE_GPU_OWNER_PID="$OWNER" bash "$ROOT/tools/pick-gpu.sh" --release >/dev/null 2>&1 || true
  fi
}
trap release_gpu EXIT INT TERM

if [ "$DO_PICK" = 1 ]; then
  log "reserving a free GPU via pick-gpu.sh ..."
  PICK="$(TORCHLETTE_GPU_OWNER_PID="$OWNER" bash "$ROOT/tools/pick-gpu.sh")" || {
    log "FATAL: could not reserve a GPU"; exit 1; }
  eval "$PICK"
  log "using VULKAN_DEVICE_INDEX=$VULKAN_DEVICE_INDEX (physical GPU ${TORCHLETTE_PICKED_PHYS:-?})"
else
  if [ -z "${VULKAN_DEVICE_INDEX:-}" ]; then
    log "WARN: --no-pick but VULKAN_DEVICE_INDEX unset; Dawn will pick device 0"
  fi
  export LD_LIBRARY_PATH="$ROOT/tools/vk-shim:${LD_LIBRARY_PATH:-}"
fi

# --- gate registry -----------------------------------------------------------
# Each gate: NAME <tab> KIND(build|vitest|tsx|suite) <tab> COMMAND
GATE_NAMES=(); GATE_KINDS=(); GATE_CMDS=()
add_gate() { GATE_NAMES+=("$1"); GATE_KINDS+=("$2"); GATE_CMDS+=("$3"); }

register_quick() {
  add_gate "build"            build  "npm run build"
  add_gate "test:gates"       vitest "npm run test:gates"
  add_gate "whole-step-diff"  tsx    "MODEL=$MODEL npx tsx tools/t-whole-step-diff.ts"
}
register_training() {
  add_gate "parity-fullstack"       tsx    "npx tsx tools/parity-fullstack-tl.ts"
  # Step-tape gates (tape:*, step-object-null, step-edit-null, ring-probe) were
  # RETIRED with the step-tape deletion (P4b-R): they exercised the deleted
  # recorder/replay/ring mechanism, not surviving behaviour.
  add_gate "ledger-default"         tsx    "npx tsx tools/t-ledger-attack-probe.ts"
  add_gate "124M-regression"        tsx    "npx tsx tools/diloco-regression-check.ts"
  add_gate "refusal-spec"           vitest "npx vitest run --project webgpu test/whole-step-checkpoint-refusal.spec.ts"
  add_gate "checkpoint-seg-spec"    vitest "npx vitest run --project webgpu test/checkpoint-segmentation.spec.ts"
}
register_full() {
  # The witness-harvest matrix gates were RETIRED with the step-tape deletion
  # (P4b-R): they drove the deleted tape witness-harvest mechanism.
  add_gate "ledger-48"              tsx    "STEPS=48 npx tsx tools/t-ledger-attack-probe.ts"
  add_gate "profile-distil"         tsx    "TORCHLETTE_PROFILE=1 TORCHLETTE_MODEL=distilgpt2 TORCHLETTE_SEQ_LEN=512 NUM_STEPS=18 npx tsx tools/profile-training.ts"
  add_gate "profile-medium"         tsx    "TORCHLETTE_PROFILE=1 TORCHLETTE_MODEL=gpt2-medium TORCHLETTE_SEQ_LEN=512 NUM_STEPS=18 npx tsx tools/profile-training.ts"
  add_gate "full-suite"             suite  "npm run test"
}

register_quick
[ "$PROFILE" = "training" ] || [ "$PROFILE" = "full" ] && register_training
[ "$PROFILE" = "full" ] && register_full

# --- runner ------------------------------------------------------------------
run_once() {
  # $1 = logfile, rest = command string
  local logfile="$1"; shift
  timeout "$GATE_TIMEOUT" bash -c "$*" >"$logfile" 2>&1
  return $?
}

log "profile=$PROFILE, ${#GATE_NAMES[@]} gates, logs in $LOGDIR"
i=0
while [ "$i" -lt "${#GATE_NAMES[@]}" ]; do
  name="${GATE_NAMES[$i]}"; cmd="${GATE_CMDS[$i]}"
  slug="$(echo "$name" | tr '/ :,' '____')"
  log1="$LOGDIR/${i}-${slug}.log"
  log "[$((i+1))/${#GATE_NAMES[@]}] $name ..."
  run_once "$log1" "$cmd"; rc1=$?
  rerun="-"; verdict=""
  if [ "$rc1" = 0 ]; then
    verdict="PASS"
    log "  -> PASS"
  else
    log "  -> first run FAILED (rc=$rc1); isolated rerun ..."
    # Isolated rerun in a fresh process (the vkCreateDevice/OOM flake protocol).
    # Our vitest gates are single-spec (already isolated); tsx gates are single
    # process. A short cooldown lets a contended device settle.
    sleep 3
    log2="$LOGDIR/${i}-${slug}.rerun.log"
    run_once "$log2" "$cmd"; rc2=$?
    if [ "$rc2" = 0 ]; then rerun="pass"; verdict="ENV"; log "  -> rerun PASS (ENV flake)";
    else rerun="FAIL($rc2)"; verdict="REAL"; log "  -> rerun FAILED (REAL, rc=$rc2)"; fi
  fi
  printf '%s\t%s\t%s\t%s\n' "$name" "$([ "$rc1" = 0 ] && echo pass || echo "FAIL($rc1)")" "$rerun" "$verdict" >>"$RESULTS"
  i=$((i+1))
done

# --- final table -------------------------------------------------------------
echo
echo "================= GATE WALL: $PROFILE ================="
printf '%-24s %-12s %-12s %-8s\n' "gate" "first-run" "isolated" "verdict"
printf '%-24s %-12s %-12s %-8s\n' "------------------------" "-----------" "-----------" "-------"
real=0; env=0; pass=0
while IFS=$'\t' read -r n f r v; do
  printf '%-24s %-12s %-12s %-8s\n' "$n" "$f" "$r" "$v"
  case "$v" in REAL) real=$((real+1));; ENV) env=$((env+1));; PASS) pass=$((pass+1));; esac
done <"$RESULTS"
echo "------------------------------------------------------"
echo "PASS=$pass  ENV=$env  REAL=$real   logs: $LOGDIR"
echo "======================================================"

release_gpu
[ "$real" = 0 ] && exit 0 || exit 1
