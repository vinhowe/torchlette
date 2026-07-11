#!/usr/bin/env bash
# t-fp-spike-sweep.sh — drive tools/t-fp-spike-probe.ts across many runs in
# BATCHED CHILD PROCESSES (each child does BATCH runs then exits, releasing all
# GPU memory). Avoids the multi-run-per-process VkOOM that accumulates when >~10
# fresh engines/models are built in one Node process. Cycles across free GPUs.
#
# Env: TOTAL (default 200) SEQ (512) STEPS (24) BATCH (8) SEED_MODE (vary)
#      GPUS (space list, default "0 1 10 11") OUTDIR (required)
set -euo pipefail
cd "$(dirname "$0")/.."
TOTAL=${TOTAL:-200}; STEPS=${STEPS:-24}; SEQ=${SEQ:-512}
BATCH=${BATCH:-8}; SEED_MODE=${SEED_MODE:-vary}
GPUS=(${GPUS:-0 1 10 11})
OUTDIR=${OUTDIR:?set OUTDIR}
mkdir -p "$OUTDIR"
NG=${#GPUS[@]}
# Number of batches; each batch covers BATCH seeds starting at SEED_START.
nbatch=$(( (TOTAL + BATCH - 1) / BATCH ))
echo "[sweep] TOTAL=$TOTAL batch=$BATCH nbatch=$nbatch gpus=${GPUS[*]} seedMode=$SEED_MODE"
pids=(); slot=0
for ((b=0; b<nbatch; b++)); do
  g=${GPUS[$(( b % NG ))]}
  start=$(( b * BATCH ))
  # SEED_STRIDE below matches the probe's vary stride (7919); SEED_BASE0 offsets
  # each batch so children cover disjoint seed ranges.
  env VULKAN_DEVICE_INDEX=$g LD_LIBRARY_PATH=tools/vk-shim \
      RUNS=$BATCH STEPS=$STEPS SEQ_LEN=$SEQ SEED_MODE=$SEED_MODE \
      SEED_BASE0=$start \
      OUT="$OUTDIR/b${b}_g${g}.jsonl" \
      npx tsx tools/t-fp-spike-probe.ts > "$OUTDIR/b${b}_g${g}.summary.json" 2> "$OUTDIR/b${b}_g${g}.log" &
  pids+=($!)
  # keep at most NG children in flight
  if (( ${#pids[@]} >= NG )); then wait "${pids[0]}" || echo "[sweep] batch failed (see logs)"; pids=("${pids[@]:1}"); fi
done
for p in "${pids[@]}"; do wait "$p" || echo "[sweep] batch failed (see logs)"; done
echo "[sweep] DONE"
