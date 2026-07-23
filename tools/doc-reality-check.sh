#!/usr/bin/env bash
#
# doc-reality-check.sh — mechanical self-description drift guard
# (cleanliness audit 2026-07, fix-ladder rung 4). Catches the drift classes the
# coherence audit found — stale doc paths, ledger↔src flag skew, dead vitest
# includes — mechanically, forever. Exits nonzero and lists EVERY violation.
#
# The four bijections asserted:
#   A. every ACTIVE row of the env-flag-ledger "Opt-outs of a shipped default"
#      table (first cell is a non-struck `TORCHLETTE_*`) names a flag that is
#      grep-findable in src/. (Struck `~~...~~` / DELETED rows are skipped.)
#   B. every `ENV.TORCHLETTE_*` read in src/ is documented in the ledger — by its
#      full name, or by the `TORCHLETTE_`-stripped abbreviation the DEBUG family
#      is catalogued under (e.g. src `TORCHLETTE_DEBUG_FUSION` ↔ ledger
#      `DEBUG_FUSION`).
#   C. every `src/...` path referenced in torchlette/CLAUDE.md resolves (file,
#      dir, or — for `*` globs — matches ≥1 entry).
#   D. every `test/...` glob in vitest.config.ts matches ≥1 real file (no dead
#      include pointing at a deleted spec).
#
# WIRING (the tool exists; do NOT edit .claude/settings.json — it is untracked /
# orchestrator-owned). To make drift self-reporting on every commit, add a line
# next to the existing weight-norm self-report in the PostToolUse hook:
#     bash tools/doc-reality-check.sh || true
# or run it as a CI step (it is GPU-less and fast — pure grep, no build).
#
# Run manually:  bash tools/doc-reality-check.sh   (exit 0 = clean, 1 = drift)

set -uo pipefail
shopt -s globstar nullglob
cd "$(dirname "$0")/.."

LEDGER=docs/env-flag-ledger.md
CLAUDEMD=CLAUDE.md
VITEST=vitest.config.ts

violations=0
fail() {
  echo "VIOLATION: $*"
  violations=$((violations + 1))
}

# ── A. ledger active Opt-out row → src flag ──────────────────────────────────
# Scoped to the "Opt-outs of a shipped default" table ONLY (awk block); the
# "DELETED this sweep" table also has `TORCHLETTE_*` rows but those correctly
# name already-removed flags. Active rows start with "| `TORCHLETTE_" (a backtick
# immediately after "| "); struck rows ("| ~~`TORCHLETTE_") are excluded by the
# anchor.
while IFS= read -r flag; do
  [ -z "$flag" ] && continue
  grep -rqF "$flag" src/ ||
    fail "A: ledger Opt-out row names '$flag' but it is not grep-findable in src/"
done < <(awk '/^## Opt-outs of a shipped default/{f=1;next} /^## /{f=0} f' "$LEDGER" |
  grep -E '^\| `TORCHLETTE_[A-Z0-9_]+`' |
  sed -E 's/^\| `(TORCHLETTE_[A-Z0-9_]+)`.*/\1/')

# ── B. src ENV read → ledger mention ─────────────────────────────────────────
while IFS= read -r flag; do
  [ -z "$flag" ] && continue
  if grep -qF "$flag" "$LEDGER"; then continue; fi
  grep -qF "${flag#TORCHLETTE_}" "$LEDGER" && continue
  fail "B: src reads '$flag' but the ledger has no row/mention for it"
done < <(grep -rhoE "ENV\.TORCHLETTE_[A-Z0-9_]+" src/ | sed 's/ENV\.//' | sort -u)

# ── C. CLAUDE.md src/ path → resolves ────────────────────────────────────────
while IFS= read -r p; do
  [ -z "$p" ] && continue
  if [[ "$p" == *"*"* ]]; then
    compgen -G "$p" >/dev/null ||
      fail "C: CLAUDE.md references glob '$p' — matches no file"
  elif [ ! -e "$p" ]; then
    fail "C: CLAUDE.md references '$p' — does not resolve"
  fi
done < <(grep -oE "src/[A-Za-z0-9_./*-]+" "$CLAUDEMD" | sed 's/[.,):]*$//' | sort -u)

# ── D. vitest.config test/ glob → matches ≥1 file ────────────────────────────
while IFS= read -r g; do
  [ -z "$g" ] && continue
  if [[ "$g" == *[*?[]* ]]; then
    # Glob entry — direct expansion (globstar+nullglob set above; compgen -G does
    # not honour the `**` the browser include uses).
    matches=($g)
    [ "${#matches[@]}" -gt 0 ] ||
      fail "D: vitest.config.ts include '$g' matches no file (dead include)"
  else
    # Literal path — nullglob leaves it verbatim, so test existence directly.
    [ -e "$g" ] ||
      fail "D: vitest.config.ts include '$g' does not resolve (dead include)"
  fi
done < <(grep -oE '"test/[^"]+"' "$VITEST" | tr -d '"' | sort -u)

# ── verdict ──────────────────────────────────────────────────────────────────
if [ "$violations" -eq 0 ]; then
  echo "doc-reality-check: clean (ledger↔src flags, CLAUDE.md paths, vitest includes all consistent)"
  exit 0
fi
echo "doc-reality-check: $violations violation(s) above"
exit 1
