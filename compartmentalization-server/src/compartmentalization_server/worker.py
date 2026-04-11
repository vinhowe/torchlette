"""
Worker subprocess entry point.

Spawned by the coordinator with:
  - CUDA_VISIBLE_DEVICES=N in the env (so cuda:0 is GPU N)
  - argv: --id <id> --script <name> --storage <root> [--resume]
  - stdin: control messages (ipc.Control)
  - stdout: events (ipc.Event)
  - stderr: free-form text → worker.log

Lifecycle:
  1. Parse args, init torch on cuda:0, load the experiment class.
  2. Read metadata.json for params + total_steps + step_count.
  3. Construct the experiment, call setup(), if --resume load checkpoint.pt.
  4. Emit `ready`.
  5. Loop: drain stdin control queue, call step(), emit `metric`,
     checkpoint every CHECKPOINT_INTERVAL steps, repeat until total_steps
     or stop signal.
  6. On any clean exit (done, stop, SIGTERM): one final checkpoint, emit
     `done` (or `status: stopped`), exit 0.
  7. On exception: emit `error` with traceback, save what we can, exit 1.

The "save on shutdown" guarantee is implemented two ways at once:
  - A SIGTERM handler that sets `_stop_requested = True`. The loop checks
    this between steps, exits the loop, falls through to the final
    checkpoint code path. We do NOT save inside the signal handler — the
    signal handler runs in some random place inside torch, and torch.save
    would deadlock or crash. The flag-and-check pattern is the only safe
    one.
  - The loop's normal exit path always runs the final checkpoint code,
    regardless of whether it exited because of stop signal or completion.

Why no asyncio in the worker: the inner loop is hot, spending most of its
time inside torch dispatches. asyncio would add complication for no real
benefit — the only "concurrent" thing the worker does is "occasionally
read stdin between steps", which a non-blocking O_NONBLOCK fd handles
fine inside a synchronous loop.
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import signal
import sys
import traceback
from pathlib import Path
from typing import Any

# We import these lazily inside main() so import errors land in the
# worker.log file (after stderr is set up) instead of dying silently before
# the coordinator can see what happened.

# Checkpoint cadence. Hardcoded for v0.1; later this can be a per-script
# class attribute or a server config.
CHECKPOINT_INTERVAL = 100

# How often (in steps) to flush our stdout buffer if Python is buffering it.
# We set stdout to line-buffered explicitly below so this rarely matters,
# but it's a safety net for environments where line buffering doesn't take.
STDOUT_FLUSH_INTERVAL = 1


_stop_requested = False


def _install_signal_handlers() -> None:
    """SIGTERM/SIGINT → set the stop flag and return.

    The loop checks `_stop_requested` between steps and exits cleanly,
    triggering the final checkpoint. Doing the save inside the handler
    itself is unsafe (torch state isn't reentrant).
    """

    def handler(signum: int, frame) -> None:  # noqa: ARG001
        global _stop_requested
        _stop_requested = True

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _set_unbuffered_stdout() -> None:
    """Force stdout to flush after every write so the coordinator sees
    events promptly. Without this, events can sit in a 4 KB buffer for
    seconds at low metric rates.
    """
    # Reopen stdout in line-buffered mode (buffering=1) on the same fd.
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1, encoding="utf-8")


def _emit(event: dict[str, Any]) -> None:
    from .ipc import encode

    sys.stdout.write(encode(event))
    sys.stdout.flush()


def _drain_control_messages() -> list[dict[str, Any]]:
    """Read any control messages currently available on stdin without blocking.

    Uses select() with a 0 timeout. Returns a list of decoded messages
    (possibly empty). Each step we drain whatever has arrived and apply it
    before running the next training step.
    """
    from .ipc import decode

    messages: list[dict[str, Any]] = []
    while True:
        # POSIX-only, fine for our target environment.
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            break
        line = sys.stdin.readline()
        if not line:
            break
        msg = decode(line)
        if msg is not None:
            messages.append(msg)
    return messages


def _save_checkpoint(
    experiment: Any,
    checkpoint_path: Path,
    step: int,
) -> int:
    """Atomically save the experiment's state_dict to checkpoint_path.

    Bundles the script's own state_dict() with framework-level fields
    (step, params snapshot, RNG state) so on resume we can fully restore
    without going through metadata for the step count.

    Returns the byte count of the written file (for the EvtCheckpoint
    event).
    """
    import torch

    blob = {
        "framework_version": 1,
        "step": step,
        "params": dict(experiment.params),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "experiment": experiment.state_dict(),
    }
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, tmp)
    # We don't fsync the torch.save output explicitly: torch writes via a
    # ZipFile under the hood and the Python GC + close should flush before
    # rename. If this turns out to be insufficient under power-loss
    # scenarios we can re-open + fsync the file before rename.
    os.replace(tmp, checkpoint_path)
    return checkpoint_path.stat().st_size


def _load_checkpoint(experiment: Any, checkpoint_path: Path) -> int:
    """Restore experiment from checkpoint_path. Returns the saved step.

    We load WITHOUT map_location because some entries in the blob (notably
    the RNG state ByteTensors) MUST be CPU tensors and would be silently
    moved to GPU otherwise — torch.set_rng_state then rejects them with
    "RNG state must be a torch.ByteTensor". The script's load_state_dict
    is responsible for placing model weights on its own device.
    """
    import torch

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if blob.get("framework_version", 0) != 1:
        raise ValueError(f"unknown checkpoint framework_version: {blob.get('framework_version')}")
    # We don't restore params from the checkpoint — coordinator-driven param
    # values are authoritative and may have been edited since the last save.
    # The experiment script reads self.params on each step.
    torch.set_rng_state(blob["torch_rng"])
    if blob.get("torch_cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(blob["torch_cuda_rng"])
    experiment.load_state_dict(blob["experiment"])
    return int(blob["step"])


def main() -> int:
    parser = argparse.ArgumentParser(description="compartmentalization-server worker")
    parser.add_argument("--id", required=True, help="experiment id")
    parser.add_argument("--script", required=True, help="registered experiment name")
    parser.add_argument("--storage", required=True, type=Path, help="storage root directory")
    parser.add_argument("--resume", action="store_true", help="load checkpoint.pt before starting")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Configure logging to stderr — captured by coordinator into worker.log.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    logger = logging.getLogger("worker")

    _set_unbuffered_stdout()
    _install_signal_handlers()

    # Heavy imports happen here, after stderr is set up. If torch fails to
    # load (e.g. CUDA driver mismatch), the traceback lands in worker.log.
    try:
        import torch

        from .api import Experiment, get_experiment_class
        from .registry import discover_experiments
        from .storage import (
            StoragePaths,
            append_metric,
            read_metadata,
        )
    except Exception as e:
        # We don't have _emit yet (no stdout), but stderr is up.
        logger.exception("worker failed during imports: %s", e)
        return 2

    # Discover experiments so the registry is populated. We rely on the
    # coordinator passing the same project root, so registry.experiments_dir()
    # resolves to the same location both processes know about.
    discover_experiments()

    paths = StoragePaths(args.storage, args.id)
    meta = read_metadata(paths)
    if meta is None:
        logger.error("no metadata at %s — aborting", paths.metadata)
        return 2

    try:
        experiment_cls = get_experiment_class(args.script)
    except KeyError as e:
        logger.error("%s", e)
        return 2

    if not torch.cuda.is_available():
        logger.warning("CUDA not available — running on CPU (slow!)")
        device = torch.device("cpu")
    else:
        # Because the coordinator launched us with CUDA_VISIBLE_DEVICES=N,
        # cuda:0 is the only device we can see and it's the one we want.
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # Set seeds before constructing the experiment so init randomness
    # is reproducible across resume cycles.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        experiment: Experiment = experiment_cls(
            params=meta.get("params", {}),
            device=device,
            seed=args.seed,
        )
        experiment.setup()
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("setup() failed: %s\n%s", e, tb)
        _emit({"type": "error", "message": str(e), "traceback": tb})
        return 1

    start_step = 0
    if args.resume and paths.checkpoint.exists():
        try:
            start_step = _load_checkpoint(experiment, paths.checkpoint)
            logger.info("resumed from step %d", start_step)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("checkpoint load failed: %s\n%s", e, tb)
            _emit({"type": "error", "message": f"resume failed: {e}", "traceback": tb})
            return 1

    experiment.step_count = start_step
    total_steps = int(meta.get("total_steps", 0))

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    _emit({"type": "ready", "step": start_step, "gpu_name": gpu_name})

    global _stop_requested
    last_checkpoint_step = start_step
    exit_reason = "completed"
    try:
        while experiment.step_count < total_steps:
            if _stop_requested:
                exit_reason = "stop_requested"
                break

            # Apply any pending live param edits before stepping. Each
            # message mutates self.params, which the script re-reads.
            for ctl in _drain_control_messages():
                ctl_type = ctl.get("type")
                if ctl_type == "set_param":
                    experiment.params[ctl["key"]] = ctl["value"]
                elif ctl_type == "stop":
                    _stop_requested = True
                # pause/resume not implemented in v0.1

            try:
                metrics = experiment.step()
            except Exception as e:
                tb = traceback.format_exc()
                logger.error("step %d failed: %s\n%s", experiment.step_count, e, tb)
                _emit({"type": "error", "message": str(e), "traceback": tb})
                exit_reason = "step_error"
                break

            experiment.step_count += 1
            _emit(
                {
                    "type": "metric",
                    "step": experiment.step_count,
                    "metrics": metrics,
                }
            )

            # Periodic checkpoint. Note: this runs AFTER the step is
            # logged, so a crash between metric emit and checkpoint write
            # leaves us with the metric in metrics.jsonl but no
            # checkpoint covering it. The next resume will simply re-run
            # that step. Idempotent if step() is deterministic given
            # params + RNG state.
            if (
                experiment.step_count - last_checkpoint_step >= CHECKPOINT_INTERVAL
                or experiment.step_count == total_steps
            ):
                try:
                    size = _save_checkpoint(experiment, paths.checkpoint, experiment.step_count)
                    last_checkpoint_step = experiment.step_count
                    _emit(
                        {
                            "type": "checkpoint",
                            "step": experiment.step_count,
                            "bytes": size,
                        }
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    logger.error("checkpoint write failed: %s\n%s", e, tb)
                    _emit({"type": "error", "message": f"checkpoint failed: {e}", "traceback": tb})
                    exit_reason = "checkpoint_error"
                    break
    finally:
        # Final checkpoint regardless of why we're exiting. This is the
        # "save on shutdown" half of the resume guarantee. Skip only if
        # we already saved at this exact step number (avoid wasted I/O).
        if last_checkpoint_step != experiment.step_count:
            try:
                _save_checkpoint(experiment, paths.checkpoint, experiment.step_count)
                logger.info("final checkpoint at step %d", experiment.step_count)
            except Exception as e:
                logger.exception("final checkpoint failed: %s", e)
        try:
            experiment.teardown()
        except Exception:  # noqa: BLE001
            logger.exception("teardown raised, ignoring")

    # We do NOT emit a "status: stopped" event for stop_requested. The
    # coordinator can't distinguish "user told this experiment to stop"
    # from "the whole server is shutting down and will resume me later"
    # — both look like SIGTERM to us. The pump task in the manager
    # decides on the right terminal status from the exit code + the
    # current `meta.status` (which the user-stop path sets to "stopping"
    # before sending the signal).
    if exit_reason == "completed":
        _emit({"type": "done", "step": experiment.step_count})
        return 0
    if exit_reason == "stop_requested":
        return 0
    return 1


def cli() -> None:
    """Entry point for `compartmentalization-worker` CLI."""
    raise SystemExit(main())


if __name__ == "__main__":
    cli()


# Keep `append_metric` available so other modules importing this file get
# the same name resolution path. Not used directly inside the worker.
__all__ = ["main", "cli"]
