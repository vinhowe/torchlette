#!/bin/bash
# Single-peer 124M DiLoCo A/B for the planned-mode round-spike hunt: same
# config as run-4peer-124M.sh but ONE peer (quorum 1) on its own relay port,
# so it can run alongside other experiments. MODE=planned|lowered picks the
# compiled-planned setting; GPU picks the device; ROUNDS/STEPS overridable.
set -u
cd /mnt/pccfs2/backed_up/vin/dev/torchlette
export LD_LIBRARY_PATH="tools/vk-shim:${LD_LIBRARY_PATH:-}"
MODE=${MODE:-planned}
GPU=${GPU:-0}
PORT=${PORT:-8444}
TAG=${TAG:-$MODE}

export TORCHLETTE_ARENA_LIVENESS=1
export TORCHLETTE_POOL_BUDGET_MB=8000
export KEEPALIVE_MS=600000
if [ "$MODE" = "lowered" ]; then
  export TORCHLETTE_COMPILED_PLANNED=0
fi

export NUM_LAYERS=12 NUM_HEADS=12 EMBED_DIM=768
export LR=5e-4 STEPS=20 BATCH_SIZE=4 SEQ_LEN=256 ACCUM_STEPS=1
export WEIGHT_DECAY=0.01 USE_AUTOCAST=1 GRAD_CLIP=1.0
export OUTER_LR=0.7 OUTER_MU=0.9 ROUNDS=${ROUNDS:-10} SEED=42
export QUORUM_MIN=1 QUORUM_TARGET_FRAC=1.0
export LOCAL_TOKENS=ckpts/tinystories-tokens.bin
export SERVER_URL=ws://127.0.0.1:$PORT
export V2_PORT=$PORT

node server/diloco-server-v2.cjs > /tmp/1p-relay-$TAG.log 2>&1 &
RELAY=$!
sleep 3
VULKAN_DEVICE_INDEX=$GPU PEER_ID=peer-a npx tsx tools/diloco-agent-v2.ts > /tmp/1p-$TAG.log 2>&1
kill -TERM $RELAY 2>/dev/null
echo "[1p-$TAG] done: $(grep -oE '"round":[0-9]+[^}]*"loss":[0-9.]+' /tmp/1p-$TAG.log | tail -1)"
