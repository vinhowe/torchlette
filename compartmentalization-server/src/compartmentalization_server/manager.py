"""
ExperimentManager: in-process registry of experiments + their worker subprocesses.

Owns:
  - the in-memory map of experiment id → ExperimentRecord
  - the asyncio.subprocess.Process for each running experiment
  - the GPU pool (round-robin across GPUS_AVAILABLE)
  - the subscriber map (id → set of asyncio queues for the WS handlers)
  - the disk lifecycle: create dirs, write metadata, atomic checkpoint reads

The big invariant: a single asyncio task (`_pump_worker_events`) per running
worker reads stdout in a loop. Everything that wants to react to a worker
event — persistence to disk, broadcast to subscribers, status transitions —
goes through that pump. There's no other code path that touches a worker's
stdout, so we don't need a lock around it.

The ExperimentManager is itself owned by exactly one event loop (FastAPI's),
so all of its mutations are single-threaded. Async only — we never call
`subprocess.run` or block the loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import signal
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import storage
from .api import default_params, get_experiment_class, list_experiments
from .ipc import decode, encode
from .registry import discover_experiments

logger = logging.getLogger(__name__)

# Hardcoded for v0.1 — we want GPUs 8..15 per the user's spec. The first 8
# are reserved for other workloads on this host. Eventually this should be
# a server config / env override.
GPUS_AVAILABLE: tuple[int, ...] = tuple(range(8, 16))

# How long we wait for a worker to checkpoint and exit gracefully after
# SIGTERM before escalating to SIGKILL. Generous enough for sizable model
# saves; tighten later if it becomes painful.
GRACEFUL_SHUTDOWN_SECONDS = 30.0


# ──────────────────────────────────────────────────────────────────────────
# In-memory record per experiment
# ──────────────────────────────────────────────────────────────────────────


class ExperimentRecord:
    """Mutable in-memory state for one experiment.

    `meta` is the latest metadata snapshot — kept in sync with what's on
    disk. `process` is None when the worker is not running. `subscribers`
    is the set of asyncio queues to broadcast events to.
    """

    def __init__(self, meta: storage.Metadata, paths: storage.StoragePaths) -> None:
        self.meta = meta
        self.paths = paths
        self.process: asyncio.subprocess.Process | None = None
        self.pump_task: asyncio.Task[None] | None = None
        self.subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        # Latest known step + last metric — used to short-circuit "give me
        # the current state" RPCs without re-reading metrics.jsonl.
        self.latest_step: int = int(meta.get("step_count", 0))
        self.latest_metrics: dict[str, float] = {}

    @property
    def id(self) -> str:
        return self.meta["id"]

    def public_view(self) -> dict[str, Any]:
        """Return a JSON-serializable summary for the browser's list view."""
        return {
            "id": self.id,
            "script": self.meta.get("script"),
            "description": self.meta.get("description", ""),
            "params": self.meta.get("params", {}),
            "total_steps": self.meta.get("total_steps", 0),
            "step_count": self.latest_step,
            "status": self.meta.get("status", "created"),
            "gpu": self.meta.get("gpu"),
            "pid": self.meta.get("pid"),
            "created_at": self.meta.get("created_at"),
            "updated_at": self.meta.get("updated_at"),
            "last_checkpoint_step": self.meta.get("last_checkpoint_step", 0),
            "latest_metrics": self.latest_metrics,
        }


# ──────────────────────────────────────────────────────────────────────────
# Manager
# ──────────────────────────────────────────────────────────────────────────


class ExperimentManager:
    def __init__(self, storage_root: Path) -> None:
        self.storage_root = storage_root
        self.records: dict[str, ExperimentRecord] = {}
        # Server-wide subscribers: receive lifecycle events for ALL
        # experiments (created, deleted, status). The browser uses this
        # for the list view auto-refresh.
        self.global_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        # Set during shutdown(). Used by _handle_worker_exit to decide
        # whether a worker exit means "leave it as running so the next
        # startup picks it back up" (shutdown drain) or "transition to
        # stopped/failed" (normal exit). Without this flag, every server
        # restart loses the running state.
        self._shutting_down = False

    # ── startup / shutdown ──

    async def startup(self) -> None:
        """Discover experiment scripts, scan storage, restart any that
        were 'running' when the server last shut down.
        """
        discovered = discover_experiments()
        logger.info("loaded %d experiment scripts: %s", len(discovered), ", ".join(discovered))

        on_disk = storage.discover_experiments(self.storage_root)
        logger.info("found %d experiments in storage", len(on_disk))

        for meta in on_disk:
            paths = storage.StoragePaths(self.storage_root, meta["id"])
            rec = ExperimentRecord(meta, paths)
            self.records[meta["id"]] = rec

            # Crash recovery: anything left "running" or "stopping" by a
            # previous server is now stale (no live process). Reset to
            # a status we can transition out of cleanly.
            if meta.get("status") in ("running", "stopping"):
                logger.info("relaunching %s (was %s)", meta["id"], meta.get("status"))
                # The metadata will be updated to "running" with new pid+gpu
                # by _spawn_worker.
                try:
                    await self._spawn_worker(rec, resume=True)
                except Exception:
                    logger.exception("failed to relaunch %s", meta["id"])
                    self._update_meta(rec, status="failed")

    async def shutdown(self) -> None:
        """Send SIGTERM to all running workers and wait for them to drain.

        Each worker's SIGTERM handler triggers a final checkpoint then
        exits clean. We escalate to SIGKILL after GRACEFUL_SHUTDOWN_SECONDS
        if a worker is wedged. The metadata stays as "running" on disk so
        the next server startup picks them up and resumes.
        """
        # Set the flag BEFORE sending any signals so the pump tasks see it
        # by the time they reach _handle_worker_exit. Without this the
        # exit handler races us and demotes the experiments to "stopped".
        self._shutting_down = True

        running = [r for r in self.records.values() if r.process is not None]
        if not running:
            return

        logger.info("shutdown: signaling %d running workers", len(running))
        for rec in running:
            assert rec.process is not None
            try:
                rec.process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

        # Wait for each worker to finish, with a per-worker timeout. We
        # use asyncio.wait_for instead of a single global deadline so a
        # slow worker doesn't make us SIGKILL the fast ones.
        async def _drain(rec: ExperimentRecord) -> None:
            assert rec.process is not None
            try:
                await asyncio.wait_for(rec.process.wait(), timeout=GRACEFUL_SHUTDOWN_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("worker for %s did not exit in time, sending SIGKILL", rec.id)
                try:
                    rec.process.kill()
                    await rec.process.wait()
                except ProcessLookupError:
                    pass

        await asyncio.gather(*(_drain(r) for r in running), return_exceptions=True)

        # Cancel any leftover pump tasks (they should have exited on EOF
        # naturally, but be defensive).
        for rec in running:
            if rec.pump_task and not rec.pump_task.done():
                rec.pump_task.cancel()
                try:
                    await rec.pump_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

        logger.info("shutdown drain complete")

    # ── public RPC surface ──

    def list_experiments(self) -> list[dict[str, Any]]:
        return [r.public_view() for r in self.records.values()]

    def list_scripts(self) -> list[dict[str, Any]]:
        return list_experiments()

    async def create_experiment(
        self,
        script: str,
        params: dict[str, Any] | None,
        total_steps: int,
        description: str = "",
    ) -> str:
        """Create a new experiment, persist its metadata, spawn the worker."""
        cls = get_experiment_class(script)  # raises KeyError if unknown
        # Merge supplied params over defaults so the user only has to
        # pass the ones they want to override.
        merged: dict[str, Any] = default_params(script)
        merged.update(params or {})

        gpu = self._claim_gpu()
        if gpu is None:
            raise RuntimeError(
                f"no available GPUs; all of {GPUS_AVAILABLE} are in use. "
                f"Stop an existing experiment first."
            )

        experiment_id = self._mint_id(script)
        paths = storage.StoragePaths(self.storage_root, experiment_id)
        paths.ensure_dir()

        meta: storage.Metadata = {
            "id": experiment_id,
            "script": script,
            "description": description or cls.description,
            "params": merged,
            "total_steps": total_steps,
            "step_count": 0,
            "status": "created",
            "gpu": gpu,
            "pid": None,
            "created_at": storage.now_iso(),
            "updated_at": storage.now_iso(),
            "last_checkpoint_step": 0,
        }
        storage.write_metadata(paths, meta)

        rec = ExperimentRecord(meta, paths)
        self.records[experiment_id] = rec

        await self._spawn_worker(rec, resume=False)
        self._broadcast_global({"type": "created", "experiment": rec.public_view()})
        return experiment_id

    async def stop_experiment(self, experiment_id: str) -> None:
        rec = self._require(experiment_id)
        if rec.process is None:
            return
        self._update_meta(rec, status="stopping")
        try:
            rec.process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
        # The pump task will see the process exit and finalize the record.

    async def delete_experiment(self, experiment_id: str) -> None:
        """Delete a STOPPED experiment from disk and the registry.

        Refuses if the experiment is still running — caller must stop it
        first. Removes the storage/<id>/ directory entirely.
        """
        rec = self._require(experiment_id)
        if rec.process is not None:
            raise RuntimeError(f"experiment {experiment_id} is still running; stop it first")
        # Drop subscribers (they should already be gone if the WS dropped).
        rec.subscribers.clear()
        # Remove from disk.
        if rec.paths.dir.exists():
            for child in rec.paths.dir.iterdir():
                try:
                    child.unlink()
                except OSError as e:
                    logger.warning("could not unlink %s: %s", child, e)
            try:
                rec.paths.dir.rmdir()
            except OSError as e:
                logger.warning("could not rmdir %s: %s", rec.paths.dir, e)
        del self.records[experiment_id]
        self._broadcast_global({"type": "deleted", "id": experiment_id})

    def set_description(self, experiment_id: str, description: str) -> None:
        """Update an experiment's description.

        Description is metadata only — not part of the worker state — so
        this never has to touch the running subprocess. We just rewrite
        metadata.json and broadcast `updated` so every subscribed browser
        sees the new value immediately. Works regardless of whether the
        experiment is running, stopped, or still in "created" state.
        """
        rec = self._require(experiment_id)
        self._update_meta(rec, description=description)

    def set_param(self, experiment_id: str, key: str, value: Any) -> None:
        """Forward a live param edit to the worker subprocess.

        We don't validate live-vs-structural here in v0.1 — the worker
        just shoves it into self.params and the script reads it on the
        next iteration. A bad value (out of range, wrong type) will
        manifest as either a worker crash (we'll see in worker.log) or a
        NaN loss (we'll see in the metrics chart). v0.2 should validate
        against the script's param spec.
        """
        rec = self._require(experiment_id)
        if rec.process is None or rec.process.stdin is None:
            raise RuntimeError(f"experiment {experiment_id} has no live worker")
        # Persist the change so a future restart picks it up.
        params = dict(rec.meta.get("params", {}))
        params[key] = value
        self._update_meta(rec, params=params)
        # Send to the live worker. asyncio's StreamWriter.write is sync;
        # the drain is async but we don't need to await every send.
        line = encode({"type": "set_param", "key": key, "value": value})
        try:
            rec.process.stdin.write(line.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("worker for %s closed stdin during set_param", experiment_id)

    # ── subscriptions ──

    def subscribe(self, experiment_id: str) -> asyncio.Queue[dict[str, Any]]:
        """Create a queue that will receive future events for one experiment.

        The caller (a WebSocket handler) reads from the queue in a loop and
        forwards to the browser. When the WS disconnects, the handler is
        responsible for calling unsubscribe() with the same queue.
        """
        rec = self._require(experiment_id)
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
        rec.subscribers.add(q)
        return q

    def unsubscribe(self, experiment_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        rec = self.records.get(experiment_id)
        if rec is None:
            return
        rec.subscribers.discard(q)

    def subscribe_global(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1024)
        self.global_subscribers.add(q)
        return q

    def unsubscribe_global(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self.global_subscribers.discard(q)

    # ── helpers ──

    def get_metrics_history(self, experiment_id: str) -> list[dict[str, Any]]:
        rec = self._require(experiment_id)
        return storage.read_metrics(rec.paths)

    def _require(self, experiment_id: str) -> ExperimentRecord:
        rec = self.records.get(experiment_id)
        if rec is None:
            raise KeyError(f"unknown experiment id: {experiment_id}")
        return rec

    def _claim_gpu(self) -> int | None:
        """Return the first GPU index from GPUS_AVAILABLE that has no
        running experiment, or None if all are occupied.

        v0.1 policy: one experiment per GPU. We pick GPUs in order so
        the lowest unused index is always assigned, which makes
        nvidia-smi readouts predictable.
        """
        in_use = {
            r.meta.get("gpu")
            for r in self.records.values()
            if r.process is not None and r.meta.get("gpu") is not None
        }
        for gpu in GPUS_AVAILABLE:
            if gpu not in in_use:
                return gpu
        return None

    def _mint_id(self, script: str) -> str:
        """Generate a unique experiment id.

        Format: <script>_<utc-timestamp>_<random>. The timestamp makes ids
        sortable, the random suffix prevents collisions if two creates land
        in the same second, and the script prefix makes the storage
        directory tree easy to skim by hand.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        suffix = secrets.token_hex(3)
        return f"{script}_{ts}_{suffix}"

    def _update_meta(self, rec: ExperimentRecord, **fields: Any) -> None:
        """Apply field updates to the in-memory record AND persist to disk."""
        rec.meta.update(fields)  # type: ignore[typeddict-item]
        storage.write_metadata(rec.paths, rec.meta)
        self._broadcast_global({"type": "updated", "experiment": rec.public_view()})

    def _broadcast(self, rec: ExperimentRecord, event: dict[str, Any]) -> None:
        for q in list(rec.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("subscriber queue full for %s, dropping event", rec.id)

    def _broadcast_global(self, event: dict[str, Any]) -> None:
        for q in list(self.global_subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("global subscriber queue full, dropping event")

    # ── worker subprocess management ──

    async def _spawn_worker(self, rec: ExperimentRecord, *, resume: bool) -> None:
        """Launch the worker subprocess for an experiment.

        Sets CUDA_VISIBLE_DEVICES so the worker only sees its assigned
        physical GPU as cuda:0. Captures stdout via PIPE for the IPC
        channel and stderr to a file (worker.log) for arbitrary debug
        output. Starts the pump task that drains stdout into events.
        """
        gpu = rec.meta.get("gpu")
        if gpu is None:
            # Re-claim a GPU on resume — the previous assignment is stale.
            gpu = self._claim_gpu()
            if gpu is None:
                raise RuntimeError("no GPU available to resume experiment")
            self._update_meta(rec, gpu=gpu)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Force unbuffered Python so the child's stdout reaches us promptly
        # in addition to the worker's own line-buffering.
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [
            sys.executable,
            "-m",
            "compartmentalization_server.worker",
            "--id",
            rec.id,
            "--script",
            rec.meta["script"],
            "--storage",
            str(self.storage_root),
        ]
        if resume:
            cmd.append("--resume")

        # Open the worker.log file for stderr capture. We append so a
        # restart preserves the prior log; users can rotate manually.
        log_fd = open(rec.paths.worker_log, "ab")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=log_fd,
            env=env,
        )
        # We can close our copy of the fd now that the child has it.
        log_fd.close()

        rec.process = proc
        self._update_meta(rec, status="running", pid=proc.pid)
        rec.pump_task = asyncio.create_task(
            self._pump_worker_events(rec),
            name=f"pump-{rec.id}",
        )
        logger.info("spawned worker for %s on GPU %d (pid %d)", rec.id, gpu, proc.pid)

    async def _pump_worker_events(self, rec: ExperimentRecord) -> None:
        """Read worker stdout, decode events, dispatch them.

        This is the only place that touches `rec.process.stdout`, so
        there's no race with other readers. It's also the only place
        that broadcasts metric/checkpoint events to subscribers and
        persists metrics to disk.

        On worker exit (EOF on stdout), it transitions the record to
        the appropriate terminal status and clears `rec.process`.
        """
        assert rec.process is not None
        assert rec.process.stdout is not None
        proc = rec.process

        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                event = decode(line.decode("utf-8", errors="replace"))
                if event is None:
                    continue
                await self._handle_event(rec, event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("pump task for %s crashed", rec.id)
        finally:
            await self._handle_worker_exit(rec)

    async def _handle_event(self, rec: ExperimentRecord, event: dict[str, Any]) -> None:
        et = event.get("type")
        if et == "ready":
            self._update_meta(rec, status="running")
            self._broadcast(rec, event)
        elif et == "metric":
            step = int(event.get("step", 0))
            metrics = event.get("metrics", {})
            rec.latest_step = step
            rec.latest_metrics = metrics
            storage.append_metric(rec.paths, step, metrics)
            # Persist step_count to metadata so a server crash mid-stream
            # doesn't lose the count entirely. We don't write metadata
            # on every metric — that would be a write per step. Instead
            # rely on the periodic checkpoint event to persist
            # last_checkpoint_step + step_count.
            self._broadcast(rec, event)
        elif et == "checkpoint":
            step = int(event.get("step", 0))
            self._update_meta(
                rec,
                step_count=step,
                last_checkpoint_step=step,
            )
            self._broadcast(rec, event)
        elif et == "status":
            new_status = event.get("status", "")
            if new_status:
                self._update_meta(rec, status=new_status)
            self._broadcast(rec, event)
        elif et == "log":
            self._broadcast(rec, event)
        elif et == "error":
            logger.error(
                "worker %s reported error: %s\n%s",
                rec.id,
                event.get("message"),
                event.get("traceback", ""),
            )
            self._broadcast(rec, event)
        elif et == "done":
            step = int(event.get("step", 0))
            self._update_meta(rec, step_count=step, status="stopped")
            self._broadcast(rec, event)
        else:
            logger.warning("unknown event type from worker %s: %s", rec.id, et)

    async def _handle_worker_exit(self, rec: ExperimentRecord) -> None:
        """Worker process has died. Decide the right terminal status.

        Decision matrix (current `meta.status` × exit code × shutdown flag):

          shutdown? rc current     -> new
          --------- -- ----------  ----
          yes       *  *           -> RUNNING (leave as-is so resume picks up)
          no        0  stopping    -> stopped (user-initiated stop completed)
          no        0  running     -> stopped (worker hit total_steps and emitted `done`,
                                       which already wrote stopped — this branch is
                                       defensive for the rare case the done event
                                       arrives after EOF)
          no       !=0 *           -> failed
        """
        proc = rec.process
        if proc is None:
            return
        try:
            await proc.wait()
        except Exception:  # noqa: BLE001
            pass
        rc = proc.returncode

        rec.process = None
        current = rec.meta.get("status", "")

        if self._shutting_down:
            # Whole server is going down. Leave status as-is (almost
            # certainly "running"); the next server startup will see it
            # and relaunch the worker with --resume.
            self._update_meta(rec, pid=None)
        elif current in ("running", "stopping"):
            if rc == 0:
                self._update_meta(rec, status="stopped", pid=None)
            else:
                self._update_meta(rec, status="failed", pid=None)
        else:
            # Status was already terminal (stopped/failed) — don't clobber.
            self._update_meta(rec, pid=None)

        logger.info("worker for %s exited (rc=%s, status=%s)", rec.id, rc, rec.meta.get("status"))


# ──────────────────────────────────────────────────────────────────────────
# Convenience: drain queue cleanly
# ──────────────────────────────────────────────────────────────────────────


async def consume_events(q: asyncio.Queue[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Generator yielding events from a queue. Used by WS handlers.

    NOTE: this is a sync generator over an async source, which doesn't
    quite work in Python. Kept here as a placeholder; the WS handler
    actually uses `await q.get()` directly.
    """
    raise NotImplementedError("use `await q.get()` directly in WS handlers")
