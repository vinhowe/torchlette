#!/usr/bin/env bash
#
# pick-gpu.sh — pick a FREE physical GPU, reserve it, and print the
# VULKAN_DEVICE_INDEX that lands on it. Kills the device-collision class:
# VULKAN_DEVICE_INDEX is a *Vulkan enumeration* index, not the nvidia-smi
# physical index, and the mapping is dynamic. Two agents both assuming
# index==index is what fabricated ~300 phantom failures.
#
# What it does:
#   1. reaps stale locks (dead holder, or older than the TTL);
#   2. probes candidate VULKAN_DEVICE_INDEX values (allocate-and-watch, via
#      tools/gpu-probe-map.ts) to learn which physical GPU each lands on;
#   3. for a candidate that maps to a physically-free device (<FREE_MIB used),
#      atomically acquires an flock-guarded reservation
#      (/tmp/torchlette-gpu-locks/<phys>.lock, pid+timestamp);
#   4. prints, on stdout, exactly:  export VULKAN_DEVICE_INDEX=N
#      (also exports LD_LIBRARY_PATH prepend for the vk-shim.)
#
# Usage:
#   eval "$(tools/pick-gpu.sh)"        # reserve + set env for this shell
#   tools/pick-gpu.sh --release        # release everything this owner holds
#   tools/pick-gpu.sh --release 7      # release physical GPU 7
#   tools/pick-gpu.sh --list           # show current reservations
#
# Env knobs:
#   TORCHLETTE_GPU_LOCK_DIR   (default /tmp/torchlette-gpu-locks)
#   TORCHLETTE_GPU_FREE_MIB   (default 500)   physical "free" threshold
#   TORCHLETTE_GPU_LOCK_TTL   (default 21600) stale-lock age in seconds (6h)
#   TORCHLETTE_GPU_OWNER_PID  (default $PPID)  reservation owner (liveness key)
#   PROBE_MIB                 (default 256)   probe allocation size (MiB)
#
# NOTE: stdout carries ONLY the `export ...` lines so `eval "$(...)"` is safe;
# all diagnostics go to stderr.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_DIR="${TORCHLETTE_GPU_LOCK_DIR:-/tmp/torchlette-gpu-locks}"
FREE_MIB="${TORCHLETTE_GPU_FREE_MIB:-500}"
STALE_SEC="${TORCHLETTE_GPU_LOCK_TTL:-21600}"
OWNER_PID="${TORCHLETTE_GPU_OWNER_PID:-$PPID}"
VK_SHIM="${ROOT}/tools/vk-shim"
GUARD="${LOCK_DIR}/.reserve.flock"

log() { echo "[pick-gpu] $*" >&2; }

ensure_lock_dir() {
  mkdir -p "$LOCK_DIR" 2>/dev/null || true
  # Best-effort world-writable so co-tenant agents share the same registry.
  chmod 1777 "$LOCK_DIR" 2>/dev/null || true
  : >"$GUARD" 2>/dev/null || true
}

now() { date +%s; }

# --- nvidia-smi helpers ------------------------------------------------------
# echoes "idx used_mib" per line
nvidia_used() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
}

phys_used_mib() {
  local target="$1"
  nvidia_used | awk -v t="$target" '$1==t {print $2; found=1} END{if(!found) print -1}'
}

gpu_count() { nvidia_used | wc -l | tr -d ' '; }

# --- lock helpers ------------------------------------------------------------
lock_file() { echo "${LOCK_DIR}/${1}.lock"; }

# read "pid ts owner_pid vk host" from a lock file (empty if absent)
read_lock() { cat "$(lock_file "$1")" 2>/dev/null || true; }

pid_alive() { kill -0 "$1" 2>/dev/null; }

# a lock is HELD iff its file exists, the recorded pid is alive, and it is not
# older than the TTL.
lock_held() {
  local phys="$1" content pid ts
  content="$(read_lock "$phys")"
  [ -n "$content" ] || return 1
  pid="$(echo "$content" | awk '{print $1}')"
  ts="$(echo "$content" | awk '{print $2}')"
  [ -n "$pid" ] || return 1
  if ! pid_alive "$pid"; then return 1; fi
  if [ -n "$ts" ] && [ "$(( $(now) - ts ))" -ge "$STALE_SEC" ]; then return 1; fi
  return 0
}

reap_stale() {
  ensure_lock_dir
  shopt -s nullglob
  for f in "$LOCK_DIR"/*.lock; do
    local phys content pid ts reason=""
    phys="$(basename "$f" .lock)"
    content="$(cat "$f" 2>/dev/null || true)"
    pid="$(echo "$content" | awk '{print $1}')"
    ts="$(echo "$content" | awk '{print $2}')"
    if [ -z "$pid" ]; then reason="empty";
    elif ! pid_alive "$pid"; then reason="dead-holder(pid $pid)";
    elif [ -n "$ts" ] && [ "$(( $(now) - ts ))" -ge "$STALE_SEC" ]; then reason="stale(>${STALE_SEC}s)";
    fi
    if [ -n "$reason" ]; then
      rm -f "$f" 2>/dev/null && log "reaped $f — $reason"
    fi
  done
  shopt -u nullglob
}

# Atomically try to reserve a physical device. Returns 0 on success.
try_reserve() {
  local phys="$1" vk="$2"
  ensure_lock_dir
  (
    flock -w 10 9 || { log "could not take reserve guard"; exit 1; }
    # Re-check under the guard.
    if lock_held "$phys"; then exit 2; fi
    local used
    used="$(phys_used_mib "$phys")"
    if [ "$used" -lt 0 ] || [ "$used" -ge "$FREE_MIB" ]; then exit 3; fi
    printf '%s %s %s %s %s\n' "$OWNER_PID" "$(now)" "$OWNER_PID" "$vk" "$(hostname)" \
      >"$(lock_file "$phys")"
    exit 0
  ) 9>"$GUARD"
}

probe_phys_for_vk() {
  local vk="$1" out phys
  out="$(cd "$ROOT" && VULKAN_DEVICE_INDEX="$vk" \
    LD_LIBRARY_PATH="${VK_SHIM}:${LD_LIBRARY_PATH:-}" PROBE_MIB="${PROBE_MIB:-256}" \
    npx tsx tools/gpu-probe-map.ts 2>/dev/null | grep -oE '^PHYS=-?[0-9]+' | head -1)"
  phys="${out#PHYS=}"
  echo "${phys:--1}"
}

# --- subcommands -------------------------------------------------------------
do_list() {
  ensure_lock_dir
  reap_stale
  log "reservations in $LOCK_DIR:"
  shopt -s nullglob
  local any=0
  for f in "$LOCK_DIR"/*.lock; do
    any=1
    log "  phys $(basename "$f" .lock): $(cat "$f")"
  done
  [ "$any" = 0 ] && log "  (none)"
  shopt -u nullglob
}

do_release() {
  ensure_lock_dir
  local target="${1:-}"
  shopt -s nullglob
  if [ -n "$target" ]; then
    rm -f "$(lock_file "$target")" && log "released physical GPU $target"
  else
    # Release everything owned by OWNER_PID.
    local released=0
    for f in "$LOCK_DIR"/*.lock; do
      local content owner
      content="$(cat "$f" 2>/dev/null || true)"
      owner="$(echo "$content" | awk '{print $3}')"
      if [ "$owner" = "$OWNER_PID" ]; then
        rm -f "$f" && { released=1; log "released $(basename "$f" .lock) (owner $OWNER_PID)"; }
      fi
    done
    [ "$released" = 0 ] && log "no reservations owned by pid $OWNER_PID"
  fi
  shopt -u nullglob
}

do_pick() {
  ensure_lock_dir
  reap_stale
  local n; n="$(gpu_count)"
  if [ "$n" -le 0 ]; then log "no GPUs visible to nvidia-smi"; exit 1; fi

  # Randomize candidate order to spread concurrent pickers across devices and
  # reduce head-of-line probe collisions.
  local candidates
  candidates="$(seq 0 $((n - 1)) | shuf)"

  log "probing up to $n Vulkan indices for a free physical GPU (free<${FREE_MIB}MiB)"
  local vk phys used
  for vk in $candidates; do
    phys="$(probe_phys_for_vk "$vk")"
    if [ "$phys" -lt 0 ]; then
      log "  VK $vk -> (inconclusive/occupied), skip"
      continue
    fi
    used="$(phys_used_mib "$phys")"
    if lock_held "$phys"; then
      log "  VK $vk -> phys $phys — already reserved, skip"
      continue
    fi
    if [ "$used" -lt 0 ] || [ "$used" -ge "$FREE_MIB" ]; then
      log "  VK $vk -> phys $phys — busy (${used}MiB), skip"
      continue
    fi
    if try_reserve "$phys" "$vk"; then
      log "RESERVED physical GPU $phys via VULKAN_DEVICE_INDEX=$vk (owner pid $OWNER_PID)"
      # stdout: the eval-able env. LD_LIBRARY_PATH prepends the vk-shim so the
      # index actually takes effect.
      echo "export VULKAN_DEVICE_INDEX=$vk"
      echo "export LD_LIBRARY_PATH=${VK_SHIM}:\${LD_LIBRARY_PATH:-}"
      echo "export TORCHLETTE_PICKED_PHYS=$phys"
      exit 0
    fi
    log "  VK $vk -> phys $phys — lost reservation race, retrying"
  done
  log "FAILED: no free, reservable GPU found across $n Vulkan indices"
  exit 1
}

# --- arg parse ---------------------------------------------------------------
case "${1:-}" in
  --release) shift; do_release "${1:-}";;
  --list)    do_list;;
  -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2;;
  "")        do_pick;;
  *)         log "unknown arg: $1 (see --help)"; exit 2;;
esac
