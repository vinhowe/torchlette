# Agent Ops — the litany

Single-sourced operational rules for agents doing GPU work in this repo. Cite it
in one line from a brief ("follow docs/agent-ops.md"). Imperative, one screen.

## GPU device selection — never trust index == index
- **Pick a GPU with `tools/pick-gpu.sh`, don't hardcode `VULKAN_DEVICE_INDEX`.**
  `VULKAN_DEVICE_INDEX` is a *Vulkan enumeration* index; the map to the nvidia-smi
  physical index is DYNAMIC and does not match. Two agents both assuming
  `index==index` is the collision class that fabricated ~300 phantom failures.
  - `eval "$(tools/pick-gpu.sh)"` reserves a free device + sets the env.
  - `tools/pick-gpu.sh --release` frees your reservations; `--list` shows them.
  - Reservations are advisory flocks under `/tmp/torchlette-gpu-locks/` keyed by
    physical index, holder pid + timestamp; stale (dead pid / >6h) locks are reaped.
- **GPU work is SERIAL-EXCLUSIVE.** Run one GPU job at a time per device; the
  webgpu vitest project must not race another Dawn process on the same device
  (device-chain contention). Reserve, run, release.

## Process hygiene
- **`process.exit(0)` at the end of every standalone Dawn/WebGPU script.** Dawn
  holds background threads; Node will not exit on its own.
- **SIGTERM, never SIGKILL, for GPU processes.** SIGKILL skips userspace teardown
  → the Dawn/Vulkan device handle leaks to the kernel allocator and that memory is
  unreclaimable without a container restart / `nvidia-smi --gpu-reset`. A live
  process tears down cleanly on SIGTERM and the memory actually returns.
- **`nvidia-smi` inside the container reports HOST pids.** They won't match `ps`.
  Kill GPU jobs by name: `pgrep -af <pattern>` → `kill -TERM <container-pid>`.
- **`[Not Found]` in `nvidia-smi` is NOT a zombie** — it's any pid outside our
  namespace (a neighbor's live job looks identical to a leak). Resolve from the
  HOST before declaring anything leaked; only a pid dead ON THE HOST is a real leak.

## Poll, don't park
- **No foreground `sleep` to wait on a condition.** Poll with an until-loop:
  ```bash
  until [ -f "$done_marker" ] || ! kill -0 "$pid" 2>/dev/null; do sleep 5; done
  ```
  Run long jobs in the background and re-check; don't block a whole turn parked.

## Worktree discipline (no-push)
- **Set a fresh worktree up with `tools/agent-worktree-setup.sh`** — symlinks
  node_modules/.venv/models/ckpts/data from the primary, excludes them, installs
  the commit guard, and verifies sentinels. Stay in your worktree.
- **NEVER `git push`** from an agent worktree. Stage and commit only.
- **NEVER `git stash`.** It is global across worktrees and silently swallows the
  other agent's in-flight work. To undo, `git checkout -- <path>` or use a scratch
  worktree; never stash.
- **Never commit symlinks (git mode 120000) or `ckpts/` `data/` `models/` paths.**
  The pre-commit guard rejects both — do not `--no-verify` past it. Committing a
  shared-asset symlink poisons every other checkout (the clobber disaster).
- **`/tmp` hygiene:** write scratch to your scratchpad dir, not `/tmp`; clean up
  probe/log files you create.

## Measurement scope — don't compare across regimes
- **Memory is NOT comparable across hardware or arena eras.** V100-32GB numbers sit
  below A100-80GB numbers for reasons that are hardware, not regressions. Compare
  only within one hardware+era row (see torchlette/CLAUDE.md Performance Baselines).
- **Read LATE (steady-state) profiler steps**, not the warmup-inflated global
  average; pool reuse must settle (~step 10+) before a per-step number is honest.

## Triage — environmental first
- **A GPU failure is environmental until proven real.** vkCreateDevice / VkOOM /
  dropped-submit failures clear on a clean isolated rerun. `tools/gate-wall.sh`
  auto-reruns a failed gate in its own process and labels it ENV (flake) vs REAL
  (failed twice). Run the wall before believing a red gate:
  `tools/gate-wall.sh --profile quick|training|full`.
- **Reproduce before theorizing.** Prior agents' failure attributions are
  hypotheses, not facts; demand a deterministic repro before writing a fix.
