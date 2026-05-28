#!/bin/bash
# Launch wrapper for the v2 agent on candlelight (BYU CS lab box).
# Node is installed user-space at ~/node20 (no sudo); pccfs2 is mounted
# at the usual path. Caller sets the rest of the env (SERVER_URL,
# PEER_ID, VULKAN_DEVICE_INDEX, training hyperparams).
cd /mnt/pccfs2/backed_up/vin/dev/torchlette
export PATH="$HOME/node20/bin:$PATH"
exec "$HOME/node20/bin/npx" tsx tools/diloco-agent-v2.ts
