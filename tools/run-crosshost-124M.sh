#!/bin/bash
# CROSS-HOST 2-peer DiLoCo: peer-a local (sivri container GPU 4), peer-b on
# candlelight via SSH. The relay runs in this container; candlelight reaches
# it through an SSH REVERSE TUNNEL (-R), so no docker port mapping or
# firewall assumptions. 124M config matching run-4peer-124M.sh (the
# validated single-host soak), LOCAL_TOKENS per the data-starvation lesson.
# Env overrides: ROUNDS (default 10), CL_GPU (candlelight GPU, default 0),
# TAG (log suffix, default crosshost).
set -u
cd /mnt/pccfs2/backed_up/vin/dev/torchlette
export LD_LIBRARY_PATH="tools/vk-shim:${LD_LIBRARY_PATH:-}"
TAG=${TAG:-crosshost}
CL_GPU=${CL_GPU:-0}

export TORCHLETTE_POOL_BUDGET_MB=8000
export KEEPALIVE_MS=600000
export NUM_LAYERS=12 NUM_HEADS=12 EMBED_DIM=768
export LR=5e-4 STEPS=20 BATCH_SIZE=4 SEQ_LEN=256 ACCUM_STEPS=1
export WEIGHT_DECAY=0.01 USE_AUTOCAST=1 GRAD_CLIP=1.0
export OUTER_LR=0.7 OUTER_MU=0.9 ROUNDS=${ROUNDS:-10} SEED=42
export QUORUM_MIN=2 QUORUM_TARGET_FRAC=1.0
export LOCAL_TOKENS=ckpts/tinystories-tokens.bin
export SERVER_URL=ws://127.0.0.1:8443

echo "[xh] relay up :8443"
node server/diloco-server-v2.cjs > /tmp/xh-relay-$TAG.log 2>&1 &
RELAY=$!
sleep 3

echo "[xh] local peer-a (GPU 4)"
VULKAN_DEVICE_INDEX=4 PEER_ID=peer-a npx tsx tools/diloco-agent-v2.ts > /tmp/xh-a-$TAG.log 2>&1 &
A=$!

echo "[xh] candlelight peer-b (GPU $CL_GPU) via ssh -R tunnel"
ssh -F /dev/null -o BatchMode=yes -o ServerAliveInterval=60 \
  -R 18443:127.0.0.1:8443 remote@candlelight \
  "\
   NUM_LAYERS=$NUM_LAYERS NUM_HEADS=$NUM_HEADS EMBED_DIM=$EMBED_DIM \
   LR=$LR STEPS=$STEPS BATCH_SIZE=$BATCH_SIZE SEQ_LEN=$SEQ_LEN ACCUM_STEPS=$ACCUM_STEPS \
   WEIGHT_DECAY=$WEIGHT_DECAY USE_AUTOCAST=$USE_AUTOCAST GRAD_CLIP=$GRAD_CLIP \
   OUTER_LR=$OUTER_LR OUTER_MU=$OUTER_MU ROUNDS=$ROUNDS SEED=$SEED \
   QUORUM_MIN=$QUORUM_MIN QUORUM_TARGET_FRAC=$QUORUM_TARGET_FRAC \
   TORCHLETTE_POOL_BUDGET_MB=$TORCHLETTE_POOL_BUDGET_MB KEEPALIVE_MS=$KEEPALIVE_MS \
   LOCAL_TOKENS=$LOCAL_TOKENS SERVER_URL=ws://127.0.0.1:18443 \
   VULKAN_DEVICE_INDEX=$CL_GPU PEER_ID=peer-b \
   bash /mnt/pccfs2/backed_up/vin/dev/torchlette/tools/launch-agent-v2-candlelight.sh" > /tmp/xh-b-$TAG.log 2>&1 &
B=$!

echo "[xh] relay=$RELAY local=$A ssh=$B; waiting"
wait $A $B
kill -TERM $RELAY 2>/dev/null
FINAL=$(grep -oE '"round":[0-9]+.*"loss":[0-9.]+' /tmp/xh-a-$TAG.log | tail -1)
echo "[xh] DONE — $FINAL"
bash ~/.claude/notify.sh "Cross-host DiLoCo ($TAG) done. $FINAL" 2>/dev/null || true
