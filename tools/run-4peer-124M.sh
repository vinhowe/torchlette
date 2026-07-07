#!/bin/bash
# GPT-2-small (124M, 12L/768/12H) x 4-peer DiLoCo, LOCAL TinyStories tokens,
# current post-fix code. Scale test: real model size + >2 peers on the now-
# validated path. Inner config = the regression-matched known-good one; outer =
# DiLoCo Nesterov (0.7/0.9). QUORUM_MIN=3 = safety valve (proceed if one peer
# dies) while targeting all 4.
set -u
cd /mnt/pccfs2/backed_up/vin/dev/torchlette
export LD_LIBRARY_PATH="tools/vk-shim:${LD_LIBRARY_PATH:-}"
# Bounded-arena mode: spill the large (>2MB) activation buffers to the budgeted,
# fence-safe pool so the arena doesn't balloon to ~27GB and OOM. Peak 26.9->10.4GB,
# correct (regression converges, no leak). Slower (compiled plan off) so bump the
# relay keepalive to avoid the connection-drop stall on long rounds.
export TORCHLETTE_ARENA_LIVENESS=1
export TORCHLETTE_POOL_BUDGET_MB=8000
export KEEPALIVE_MS=600000

export NUM_LAYERS=12 NUM_HEADS=12 EMBED_DIM=768
# batch 4 (not 8): 124M @ batch8/seq256 OOM'd the 31.5GB V100 on the first
# forward — activations for 12L/768 are ~4x the 8M model's. batch 4 fits ~18GB.
export LR=5e-4 STEPS=20 BATCH_SIZE=4 SEQ_LEN=256 ACCUM_STEPS=1
export WEIGHT_DECAY=0.01 USE_AUTOCAST=1 GRAD_CLIP=1.0
export OUTER_LR=0.7 OUTER_MU=0.9 ROUNDS=25 SEED=42
export QUORUM_MIN=3 QUORUM_TARGET_FRAC=1.0
export LOCAL_TOKENS=ckpts/tinystories-tokens.bin
export SERVER_URL=ws://127.0.0.1:8443

echo "[run] relay up :8443"
node server/diloco-server-v2.cjs > /tmp/4p-relay.log 2>&1 &
RELAY=$!
sleep 4

echo "[run] launching 4 agents on GPUs 4,5,6,7"
VULKAN_DEVICE_INDEX=4 PEER_ID=peer-a npx tsx tools/diloco-agent-v2.ts > /tmp/4p-a.log 2>&1 &
A=$!
VULKAN_DEVICE_INDEX=5 PEER_ID=peer-b npx tsx tools/diloco-agent-v2.ts > /tmp/4p-b.log 2>&1 &
B=$!
VULKAN_DEVICE_INDEX=6 PEER_ID=peer-c npx tsx tools/diloco-agent-v2.ts > /tmp/4p-c.log 2>&1 &
C=$!
VULKAN_DEVICE_INDEX=7 PEER_ID=peer-d npx tsx tools/diloco-agent-v2.ts > /tmp/4p-d.log 2>&1 &
D=$!

echo "[run] relay=$RELAY agents=$A,$B,$C,$D; waiting"
wait $A $B $C $D
kill -TERM $RELAY 2>/dev/null
FINAL=$(grep -oE '"round":[0-9]+.*"loss":[0-9.]+' /tmp/4p-a.log | tail -1)
echo "[run] DONE — $FINAL"
bash ~/.claude/notify.sh "DiLoCo 124M x4-peer done. $FINAL (full trajectory in /tmp/4p-a.log)" 2>/dev/null || true
