#!/usr/bin/env bash
#
# agent-worktree-setup.sh — make a fresh git worktree usable for GPU work.
#
# A linked worktree starts EMPTY of the big, gitignored shared assets
# (node_modules, .venv, models, ckpts, data). Rather than reinstall/redownload
# gigabytes per worktree, we symlink them from the primary checkout. This script
# does that safely and refuses to leave a half-set-up tree:
#
#   1. symlinks node_modules/.venv/models/ckpts/data from the primary checkout;
#   2. adds those names to THIS worktree's git exclude (via the git-path idiom,
#      NOT a hand-typed .git path — the common-dir trap);
#   3. exports/echoes TORCH_ORACLE_PYTHON;
#   4. verifies sentinels: models/distilgpt2 exists; ckpts/tinystories-tokens.bin
#      is exactly 947984472 bytes; the venv python imports torch (hard) and
#      numpy (warn — a missing numpy silently burned the oracle suite once);
#   5. installs a pre-commit hook that REJECTS symlinks (mode 120000) and any
#      staged ckpts/ data/ models/ path — the clobber-disaster guard;
#   6. refuses to proceed on any hard failure.
#
# Run from the worktree root (or anywhere inside it):
#   bash tools/agent-worktree-setup.sh
#
# Env:
#   TORCHLETTE_PRIMARY  override the primary checkout (default: derived from the
#                       git common dir).
set -euo pipefail

fail() { echo "[wt-setup] FAIL: $*" >&2; exit 1; }
warn() { echo "[wt-setup] WARN: $*" >&2; }
info() { echo "[wt-setup] $*" >&2; }

# --- locate worktree + primary ----------------------------------------------
WT="$(git rev-parse --show-toplevel 2>/dev/null)" || fail "not inside a git worktree"
COMMON_GITDIR="$(realpath "$(git rev-parse --git-common-dir)")" || fail "no git common dir"
PRIMARY="${TORCHLETTE_PRIMARY:-$(dirname "$COMMON_GITDIR")}"
[ -d "$PRIMARY" ] || fail "primary checkout not found: $PRIMARY"

if [ "$(realpath "$WT")" = "$(realpath "$PRIMARY")" ]; then
  fail "this IS the primary checkout ($PRIMARY) — run from a linked worktree"
fi
info "worktree: $WT"
info "primary : $PRIMARY"

# --- 1. symlink shared assets -----------------------------------------------
ASSETS=(node_modules .venv models ckpts data)
for a in "${ASSETS[@]}"; do
  src="$PRIMARY/$a"
  dst="$WT/$a"
  if [ ! -e "$src" ]; then
    warn "primary has no '$a' — skipping (nothing to link)"
    continue
  fi
  if [ -L "$dst" ]; then
    cur="$(readlink "$dst")"
    if [ "$cur" = "$src" ]; then info "symlink ok: $a -> $src"; continue; fi
    rm -f "$dst"
  elif [ -e "$dst" ]; then
    fail "$dst already exists and is NOT a symlink — refusing to clobber"
  fi
  ln -s "$src" "$dst" || fail "could not symlink $a"
  info "linked: $a -> $src"
done

# --- 2. exclude the symlinks (worktree-aware path) --------------------------
EXCLUDE="$(git rev-parse --git-path info/exclude)"
mkdir -p "$(dirname "$EXCLUDE")"
touch "$EXCLUDE"
for a in "${ASSETS[@]}"; do
  if ! grep -qxF "/$a" "$EXCLUDE" 2>/dev/null; then
    echo "/$a" >>"$EXCLUDE"
    info "excluded: /$a"
  fi
done

# --- 3. TORCH_ORACLE_PYTHON --------------------------------------------------
ORACLE_PY="$WT/.venv/bin/python"
if [ ! -x "$ORACLE_PY" ]; then
  ORACLE_PY="$PRIMARY/.venv/bin/python"
fi

# --- 4. sentinels ------------------------------------------------------------
[ -d "$WT/models/distilgpt2" ] || fail "sentinel missing: models/distilgpt2 (bad models symlink?)"
info "sentinel ok: models/distilgpt2"

TOK="$WT/ckpts/tinystories-tokens.bin"
if [ -e "$TOK" ]; then
  sz="$(stat -c '%s' "$TOK" 2>/dev/null || echo -1)"
  [ "$sz" = "947984472" ] || fail "sentinel byte-count mismatch: $TOK is $sz, expected 947984472"
  info "sentinel ok: tinystories-tokens.bin (947984472 bytes)"
else
  warn "ckpts/tinystories-tokens.bin absent — token-blob tests will not run"
fi

if [ -x "$ORACLE_PY" ]; then
  if "$ORACLE_PY" -c "import torch" >/dev/null 2>&1; then
    info "sentinel ok: venv python imports torch"
  else
    fail "venv python cannot import torch: $ORACLE_PY"
  fi
  if "$ORACLE_PY" -c "import numpy" >/dev/null 2>&1; then
    info "sentinel ok: venv python imports numpy"
  else
    warn "venv python CANNOT import numpy — the PyTorch oracle suite will break. Fix: '$ORACLE_PY -m pip install numpy'"
  fi
else
  warn "no venv python found ($ORACLE_PY) — oracle tests unavailable"
fi

# --- 5. pre-commit guard hook -----------------------------------------------
HOOK="$(git rev-parse --git-path hooks/pre-commit)"
MARKER="torchlette-agent-guard"
mkdir -p "$(dirname "$HOOK")"
if [ -f "$HOOK" ] && ! grep -q "$MARKER" "$HOOK" 2>/dev/null; then
  cp "$HOOK" "$HOOK.pre-agent-guard.bak"
  warn "backed up existing pre-commit hook to $HOOK.pre-agent-guard.bak"
fi
cat >"$HOOK" <<'HOOK_EOF'
#!/usr/bin/env bash
# torchlette-agent-guard — reject the two clobber-disaster classes.
set -euo pipefail

# (a) staged symlinks (git mode 120000). Never commit node_modules/.venv/
#     models/ckpts/data symlinks — they poison every other checkout.
syms="$(git diff --cached --raw --no-renames | awk '$2=="120000"{print $NF}')"
if [ -n "$syms" ]; then
  echo "pre-commit BLOCKED: staged symlink(s) (git mode 120000):" >&2
  echo "$syms" | sed 's/^/    /' >&2
  echo "  symlinks to shared assets must never be committed." >&2
  exit 1
fi

# (b) staged content under ckpts/ data/ models/ — huge shared assets that
#     must stay out of history.
paths="$(git diff --cached --name-only --diff-filter=AM | grep -E '^(ckpts|data|models)/' || true)"
if [ -n "$paths" ]; then
  echo "pre-commit BLOCKED: staged path(s) under ckpts/ data/ models/:" >&2
  echo "$paths" | sed 's/^/    /' >&2
  echo "  these are shared assets, not repo content." >&2
  exit 1
fi
exit 0
HOOK_EOF
chmod +x "$HOOK"
info "installed pre-commit guard: $HOOK"

# --- done: echo eval-able env -----------------------------------------------
info "setup complete."
echo "export TORCH_ORACLE_PYTHON=$ORACLE_PY"
