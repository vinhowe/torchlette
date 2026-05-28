#!/bin/bash
# Convenience wrapper: cd into the project root so process.cwd() resolves
# the tokenizer files + relative module paths correctly. Env vars are
# expected to be set by the caller (`ssh remote@dw-2-1 'env ... ./launch-agent-v2-dw21.sh'`).
cd /mnt/pccfs2/backed_up/vin/dev/torchlette
exec /usr/bin/npx tsx tools/diloco-agent-v2.ts
