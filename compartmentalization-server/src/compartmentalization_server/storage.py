"""
On-disk layout for experiment state.

Each experiment lives in a directory under storage/:

    storage/<id>/
        metadata.json    fixed schema, atomically rewritten on every change
        checkpoint.pt    torch.save() blob, atomically rewritten on every save
        metrics.jsonl    append-only one-line-per-event metrics log
        worker.log       captured worker stderr (raw text)
        control.fifo     (transient) named pipe used by coordinator → worker IPC

The atomic-rewrite pattern (write to .tmp, fsync, rename) is what makes the
"kill the server, resume right where we left off" guarantee actually hold:
even an OOM-killed checkpoint write leaves the prior good checkpoint intact,
and metadata.json is never seen in a half-written state by the server's
startup scan.

This module is deliberately small and free of any framework imports — both
the coordinator and the worker subprocesses use it.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Metadata schema
# ──────────────────────────────────────────────────────────────────────────

ExperimentStatus = Literal[
    "created",  # registered, no worker yet
    "running",  # worker subprocess is alive and stepping
    "paused",  # worker is alive but waiting (not implemented in v0.1)
    "stopping",  # SIGTERM sent, awaiting clean exit
    "stopped",  # worker exited cleanly (user requested or completed)
    "failed",  # worker crashed; see worker.log
]


class Metadata(TypedDict, total=False):
    """The on-disk metadata.json schema. Plain dict so json.dump just works.

    `id` is also encoded in the directory name; we duplicate it in the file
    so a metadata.json moved out of context is still self-describing.

    `script` is the registered name (e.g. "mess3"), not a file path. The
    same script may be loaded from different files in the future without
    breaking existing experiments — the registry handles the mapping.

    `total_steps` is the target step count. The script's loop runs until
    `step_count >= total_steps` or until told to stop.
    """

    id: str
    script: str
    description: str
    params: dict[str, Any]
    total_steps: int
    step_count: int
    status: ExperimentStatus
    gpu: int | None  # physical GPU index (8-15), None when not running
    pid: int | None  # OS pid of the worker subprocess, None when not running
    created_at: str  # ISO-8601 UTC
    updated_at: str  # ISO-8601 UTC
    last_checkpoint_step: int  # latest step at which a checkpoint was successfully written


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ──────────────────────────────────────────────────────────────────────────
# Path layout
# ──────────────────────────────────────────────────────────────────────────


class StoragePaths:
    """Resolves the file paths for one experiment under a root storage dir."""

    def __init__(self, root: Path, experiment_id: str) -> None:
        self.root = root
        self.id = experiment_id
        self.dir = root / experiment_id
        self.metadata = self.dir / "metadata.json"
        self.checkpoint = self.dir / "checkpoint.pt"
        self.metrics = self.dir / "metrics.jsonl"
        self.worker_log = self.dir / "worker.log"

    def ensure_dir(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────
# Atomic write helpers
# ──────────────────────────────────────────────────────────────────────────


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` atomically.

    Strategy: write to a sibling .tmp file, fsync it, rename over the target.
    On POSIX, rename is atomic — readers either see the old file or the new
    one, never a half-written one. The fsync ensures the new contents are
    durable before the rename publishes them; without it a power loss could
    expose a renamed-but-empty file.

    On Linux, os.replace() implements POSIX rename(2) semantics. We use it
    instead of os.rename() because os.rename refuses to overwrite an
    existing file on Windows (irrelevant here, but consistency is cheap).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    data = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, data)


# ──────────────────────────────────────────────────────────────────────────
# Metadata read/write
# ──────────────────────────────────────────────────────────────────────────


def read_metadata(paths: StoragePaths) -> Metadata | None:
    """Load metadata.json or return None if it doesn't exist or is corrupt.

    A corrupt metadata file logs a warning but doesn't raise — the server
    startup scan should be able to skip a broken experiment instead of
    failing to start entirely. The corrupt file is left in place for
    debugging.
    """
    if not paths.metadata.exists():
        return None
    try:
        with open(paths.metadata) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("could not read metadata at %s: %s", paths.metadata, e)
        return None


def write_metadata(paths: StoragePaths, meta: Metadata) -> None:
    paths.ensure_dir()
    meta["updated_at"] = now_iso()
    atomic_write_json(paths.metadata, meta)


def update_metadata(paths: StoragePaths, **fields: Any) -> Metadata:
    """Read-modify-write metadata atomically.

    NOT safe for concurrent updates from multiple processes — only the
    coordinator should call this for a given experiment, never the worker.
    """
    meta = read_metadata(paths) or {}
    meta.update(fields)
    write_metadata(paths, meta)  # type: ignore[arg-type]
    return meta  # type: ignore[return-value]


# ──────────────────────────────────────────────────────────────────────────
# Metrics: append-only JSONL
# ──────────────────────────────────────────────────────────────────────────


def append_metric(
    paths: StoragePaths,
    step: int,
    metrics: dict[str, float],
    ts: str | None = None,
) -> None:
    """Append a single metric record to metrics.jsonl.

    JSONL is the simplest sane format for an append-only log: line-oriented,
    streaming-readable, no parser state. We open in "a" mode each call;
    Python's text mode buffer is line-buffered when isatty() is true (it
    isn't here) but we explicitly flush so a coordinator crash doesn't
    swallow recent metrics.
    """
    paths.ensure_dir()
    record = {"step": step, "metrics": metrics, "ts": ts or now_iso()}
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with open(paths.metrics, "a") as f:
        f.write(line)
        f.flush()


def read_metrics(paths: StoragePaths) -> list[dict[str, Any]]:
    """Load the entire metrics history. Used to seed late subscribers.

    For long-running experiments this can grow large (10k steps × 1
    metric ≈ 80 KB; 100k steps ≈ 800 KB). v0.1 just reads it all; if it
    becomes painful we can add a `since: step` filter to subscribe.
    """
    if not paths.metrics.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(paths.metrics) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Tolerate a partial trailing line — can happen if the
                # writer was killed mid-write before flush.
                logger.warning("skipping malformed metrics line in %s", paths.metrics)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Discovery
# ──────────────────────────────────────────────────────────────────────────


def discover_experiments(root: Path) -> list[Metadata]:
    """Scan storage/ on server startup and return all valid experiment metadata.

    Skips directories that don't have a metadata.json. Returns metadata in
    creation order (so list views are stable).
    """
    if not root.exists():
        return []
    out: list[Metadata] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        paths = StoragePaths(root, child.name)
        meta = read_metadata(paths)
        if meta is None:
            continue
        out.append(meta)
    out.sort(key=lambda m: m.get("created_at", ""))
    return out
