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
#   4. verifies sentinels: the asset symlinks resolve; models/distilgpt2 exists;
#      ckpts/tinystories-tokens.bin is exactly 947984472 bytes; the venv python
#      imports torch AND numpy (a missing numpy silently burned the oracle suite
#      once, and a missing .venv symlink made P0-P2 report the oracle
#      "unavailable" while it worked fine — that discrepancy WAS this gap);
#   5. installs a pre-commit hook that REJECTS symlinks (mode 120000) and any
#      staged ckpts/ data/ models/ path — the clobber-disaster guard;
#   6. refuses to proceed on any hard failure — and reports EVERY missing
#      sentinel at once (fail-loud), not just the first.
#
# Idempotent: re-running is a no-op on an already-set-up tree.
#
# Modes:
#   (default)   set up THIS worktree (create symlinks + exclude + hook + verify).
#   --verify    CHECK ONLY — no writes. Asserts the tree is set up (symlinks
#               resolve, sentinels pass, python imports torch AND numpy), prints
#               the export line, and exits nonzero LISTING every problem if not.
#               This is the gate-wall pre-flight: enforcement beats instruction.
#
# Run from the worktree root (or anywhere inside it):
#   bash tools/agent-worktree-setup.sh
#   bash tools/agent-worktree-setup.sh --verify
#
# Env:
#   TORCHLETTE_PRIMARY  override the primary checkout (default: derived from the
#                       git common dir).
set -uo pipefail

MODE="setup"
case "${1:-}" in
  --verify) MODE="verify" ;;
  "" ) ;;
  -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) echo "[wt-setup] unknown arg: $1 (use --verify or no arg)" >&2; exit 2 ;;
esac

warn() { echo "[wt-setup] WARN: $*" >&2; }
info() { echo "[wt-setup] $*" >&2; }

# fail-loud: collect problems, report them ALL at the end, then exit 1.
PROBLEMS=()
problem() { PROBLEMS+=("$*"); echo "[wt-setup] PROBLEM: $*" >&2; }

# A hard error that must stop immediately (bad invocation environment, not a
# recoverable sentinel). Setup-only; --verify prefers to accumulate.
die() { echo "[wt-setup] FATAL: $*" >&2; exit 1; }

ASSETS=(node_modules .venv models ckpts data)

# --- locate worktree + primary ----------------------------------------------
WT="$(git rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git worktree"
COMMON_GITDIR="$(realpath "$(git rev-parse --git-common-dir)")" || die "no git common dir"
PRIMARY="${TORCHLETTE_PRIMARY:-$(dirname "$COMMON_GITDIR")}"
[ -d "$PRIMARY" ] || die "primary checkout not found: $PRIMARY"

if [ "$(realpath "$WT")" = "$(realpath "$PRIMARY")" ]; then
  die "this IS the primary checkout ($PRIMARY) — run from a linked worktree"
fi
info "mode: $MODE"
info "worktree: $WT"
info "primary : $PRIMARY"

# --- 1. symlink shared assets (setup only) ----------------------------------
if [ "$MODE" = "setup" ]; then
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
      die "$dst already exists and is NOT a symlink — refusing to clobber"
    fi
    ln -s "$src" "$dst" || die "could not symlink $a"
    info "linked: $a -> $src"
  done

  # --- 2. exclude the symlinks (worktree-aware path) ------------------------
  EXCLUDE="$(git rev-parse --git-path info/exclude)"
  mkdir -p "$(dirname "$EXCLUDE")"
  touch "$EXCLUDE"
  for a in "${ASSETS[@]}"; do
    if ! grep -qxF "/$a" "$EXCLUDE" 2>/dev/null; then
      echo "/$a" >>"$EXCLUDE"
      info "excluded: /$a"
    fi
  done
fi

# --- verify: the asset symlinks must actually resolve -----------------------
# In --verify mode this is the item-1 gap detector: a missing/dangling .venv or
# models symlink is exactly what made agents report assets "absent on arrival"
# and the oracle "unavailable". Only assets the primary actually has are required
# (data is optional — the primary may not have it).
for a in "${ASSETS[@]}"; do
  src="$PRIMARY/$a"
  dst="$WT/$a"
  [ -e "$src" ] || continue # primary lacks it → nothing to require
  if [ ! -e "$dst" ]; then
    problem "asset '$a' is MISSING in this worktree (expected symlink -> $src). Run: bash tools/agent-worktree-setup.sh"
  elif [ -L "$dst" ] && [ ! -e "$(readlink "$dst")" ]; then
    problem "asset '$a' symlink is DANGLING (-> $(readlink "$dst"))"
  fi
done

# --- 3. resolve TORCH_ORACLE_PYTHON -----------------------------------------
ORACLE_PY="$WT/.venv/bin/python"
if [ ! -x "$ORACLE_PY" ]; then
  ORACLE_PY="$PRIMARY/.venv/bin/python"
fi

# --- 4. sentinels ------------------------------------------------------------
if [ -d "$WT/models/distilgpt2" ]; then
  info "sentinel ok: models/distilgpt2"
else
  problem "sentinel missing: models/distilgpt2 (bad models symlink?)"
fi

TOK="$WT/ckpts/tinystories-tokens.bin"
if [ -e "$TOK" ]; then
  sz="$(stat -c '%s' "$TOK" 2>/dev/null || echo -1)"
  if [ "$sz" = "947984472" ]; then
    info "sentinel ok: tinystories-tokens.bin (947984472 bytes)"
  else
    problem "sentinel byte-count mismatch: $TOK is $sz, expected 947984472"
  fi
else
  warn "ckpts/tinystories-tokens.bin absent — token-blob tests will not run"
fi

# The oracle: torch AND numpy are BOTH required. --verify asserts both hard
# (a missing numpy is not a warning here — it silently breaks the oracle suite).
if [ -x "$ORACLE_PY" ]; then
  if "$ORACLE_PY" -c "import torch" >/dev/null 2>&1; then
    info "sentinel ok: venv python imports torch"
  else
    problem "venv python cannot import torch: $ORACLE_PY"
  fi
  if "$ORACLE_PY" -c "import numpy" >/dev/null 2>&1; then
    info "sentinel ok: venv python imports numpy"
  else
    problem "venv python CANNOT import numpy: $ORACLE_PY (oracle suite will break). Fix: '$ORACLE_PY -m pip install numpy'"
  fi
else
  problem "no venv python found ($ORACLE_PY) — the torch oracle is unavailable (bad .venv symlink?)"
fi

# --- 5. pre-commit guard hook (setup only) ----------------------------------
if [ "$MODE" = "setup" ]; then
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
fi

# --- report: fail-loud if anything is wrong ---------------------------------
if [ "${#PROBLEMS[@]}" -gt 0 ]; then
  echo "[wt-setup] FAIL: ${#PROBLEMS[@]} problem(s) — this worktree is NOT set up:" >&2
  for p in "${PROBLEMS[@]}"; do echo "  - $p" >&2; done
  if [ "$MODE" = "verify" ]; then
    echo "[wt-setup] fix: bash tools/agent-worktree-setup.sh   (then re-run --verify)" >&2
  fi
  exit 1
fi

# --- done: echo eval-able env (BOTH modes) ----------------------------------
info "$([ "$MODE" = verify ] && echo "verify OK — worktree is set up." || echo "setup complete.")"
echo "export TORCH_ORACLE_PYTHON=$ORACLE_PY"
