"""
JSON-line IPC protocol between the coordinator and a worker subprocess.

The worker's stdin receives control messages, the worker's stdout emits
event messages. Stderr is captured but not parsed — it goes straight to
worker.log for debugging.

We use newline-delimited JSON because:
  1. It's trivially parseable from any language (the browser will speak
     a similar protocol when we add the UI).
  2. Python's stdlib `json` + line-buffered text I/O give us a working
     framing layer in ~5 lines without dragging in a serialization library.
  3. It survives partial reads cleanly: a half-line at the end of a buffer
     just waits for the next read to complete it.

There's no length-prefixed binary frame here because the worker only emits
small text records (metrics, status, log lines). Tensor handles, weight
blobs, etc. live in the on-disk checkpoint, not on the IPC channel.
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict, Union


# ──────────────────────────────────────────────────────────────────────────
# Coordinator → Worker (sent on worker stdin)
# ──────────────────────────────────────────────────────────────────────────


class CtlSetParam(TypedDict):
    type: Literal["set_param"]
    key: str
    value: float | int | bool | str


class CtlStop(TypedDict):
    type: Literal["stop"]


class CtlPause(TypedDict):
    type: Literal["pause"]


class CtlResume(TypedDict):
    type: Literal["resume"]


Control = Union[CtlSetParam, CtlStop, CtlPause, CtlResume]


# ──────────────────────────────────────────────────────────────────────────
# Worker → Coordinator (sent on worker stdout)
# ──────────────────────────────────────────────────────────────────────────


class EvtReady(TypedDict):
    """Sent once after setup() completes successfully and the loop is about to start."""

    type: Literal["ready"]
    step: int  # current step (nonzero on resume)
    gpu_name: str  # torch.cuda.get_device_name(0)


class EvtMetric(TypedDict):
    """Single training-step metric record. Coordinator forwards to subscribers
    AND persists to metrics.jsonl. Step is the just-completed step number.
    """

    type: Literal["metric"]
    step: int
    metrics: dict[str, float]


class EvtCheckpoint(TypedDict):
    """Emitted right after a successful checkpoint write. Coordinator updates
    metadata.last_checkpoint_step. Lets the UI badge "saved at step N".
    """

    type: Literal["checkpoint"]
    step: int
    bytes: int


class EvtStatus(TypedDict):
    """Worker is changing its own perceived status (e.g. running → paused).
    Coordinator records this in metadata and broadcasts.
    """

    type: Literal["status"]
    status: str  # one of storage.ExperimentStatus values


class EvtLog(TypedDict):
    """Free-form log line from the script. Coordinator broadcasts but does
    NOT persist (worker.log is the durable copy via stderr).
    """

    type: Literal["log"]
    level: Literal["debug", "info", "warning", "error"]
    message: str


class EvtError(TypedDict):
    """Worker is about to die because something went wrong. Includes the
    exception message + traceback. Coordinator marks the experiment failed.
    """

    type: Literal["error"]
    message: str
    traceback: str


class EvtDone(TypedDict):
    """Loop completed normally — step_count reached total_steps."""

    type: Literal["done"]
    step: int


Event = Union[EvtReady, EvtMetric, EvtCheckpoint, EvtStatus, EvtLog, EvtError, EvtDone]


# ──────────────────────────────────────────────────────────────────────────
# Encode / decode
# ──────────────────────────────────────────────────────────────────────────


def encode(message: dict[str, Any]) -> str:
    """Serialize a single message as a JSON line (no embedded newlines)."""
    return json.dumps(message, separators=(",", ":")) + "\n"


def decode(line: str) -> dict[str, Any] | None:
    """Parse a single JSON line. Returns None on empty line or parse error.

    Returning None instead of raising lets the reader loop tolerate noise
    on the channel — e.g. a worker that accidentally writes a print() to
    stdout instead of stderr won't crash the coordinator's parser.
    """
    line = line.strip()
    if not line:
        return None
    try:
        result = json.loads(line)
        if isinstance(result, dict) and "type" in result:
            return result
        return None
    except json.JSONDecodeError:
        return None
