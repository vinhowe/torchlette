#!/usr/bin/env bash
# The framework's size vector — the "weight norm" (CLAUDE.md: Complexity budget).
# Append a snapshot to docs/weight-norm.history at campaign-end commits:
#   bash tools/weight-norm.sh --log
set -euo pipefail
cd "$(dirname "$0")/.."

loc=$(find src -name '*.ts' -not -name '*.spec.ts' -print0 | xargs -0 cat | wc -l)
files=$(find src -name '*.ts' -not -name '*.spec.ts' | wc -l)
exports=$(( $(grep -cE '^\s*export' src/index.ts) + $(grep -cE '^\s*export' src/browser.ts) ))
flags=$(grep -rhoE 'TORCHLETTE_[A-Z_]+' src | sort -u | wc -l)
testloc=$(find test -name '*.ts' -print0 2>/dev/null | xargs -0 cat | wc -l)

line="$(date +%F) $(git rev-parse --short HEAD) srcLOC=$loc files=$files exports=$exports envFlags=$flags testLOC=$testloc"
echo "$line"
if [[ "${1:-}" == "--log" ]]; then
  echo "$line" >> docs/weight-norm.history
  echo "(appended to docs/weight-norm.history)"
fi
