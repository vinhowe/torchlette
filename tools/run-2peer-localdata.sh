#!/bin/bash
# 2-peer DiLoCo, LOCAL TinyStories tokens (no HF), current (post-fix) code.
# Re-runs the conv-b plateau scenario with the data-starvation confound removed.
# Inner config matches the regression harness (reaches ~4.9 SOLO); outer is the
# DiLoCo-paper Nesterov (0.7/0.9) as in conv-b. Question: does 2-peer reach ~5.0
# (=> plateau was data starvation) or stall ~6.5 (=> 2-peer aggregation issue)?
set -u
cd /mnt/pccfs2/backed_up/vin/dev/torchlette
export LD_LIBRARY_PATH="tools/vk-shim:${LD_LIBRARY_PATH:-}"

# shared config
export NUM_LAYERS=8 NUM_HEADS=4 EMBED_DIM=128
export LR=5e-4 STEPS=20 BATCH_SIZE=8 SEQ_LEN=256 ACCUM_STEPS=1
export WEIGHT_DECAY=0.01 USE_AUTOCAST=1 GRAD_CLIP=1.0
export OUTER_LR=0.7 OUTER_MU=0.9 ROUNDS=30 SEED=42 QUORUM_MIN=2
export LOCAL_TOKENS=ckpts/tinystories-tokens.bin
export SERVER_URL=ws://127.0.0.1:8443

echo "[run] starting relay (server/diloco-server-v2.cjs :8443)"
node server/diloco-server-v2.cjs > /tmp/2peer-relay.log 2>&1 &
RELAY=$!
sleep 3

echo "[run] starting agent A (GPU 6) + agent B (GPU 7)"
VULKAN_DEVICE_INDEX=6 PEER_ID=peer-a npx tsx tools/diloco-agent-v2.ts > /tmp/2peer-a.log 2>&1 &
A=$!
VULKAN_DEVICE_INDEX=7 PEER_ID=peer-b npx tsx tools/diloco-agent-v2.ts > /tmp/2peer-b.log 2>&1 &
B=$!

echo "[run] relay=$RELAY agentA=$A agentB=$B; waiting for agents to finish"
wait $A $B
echo "[run] agents done; stopping relay"
kill -TERM $RELAY 2>/dev/null
echo "[run] DONE"
