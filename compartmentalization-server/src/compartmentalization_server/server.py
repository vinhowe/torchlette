"""
FastAPI application + WebSocket protocol.

The server exposes two HTTP endpoints (a health check and a script catalog)
and one WebSocket endpoint at /ws that speaks a small JSON-line RPC. The WS
is the only thing the browser actually needs.

Lifecycle is wired through FastAPI's lifespan context manager so the
ExperimentManager's startup() (which restores running experiments from disk)
runs before any request is handled, and its shutdown() (which SIGTERMs all
workers and waits for clean exit) runs before the process exits.

Why a single WS instead of REST + SSE: the same connection that polls for
list/create needs to push live metric events out, and the cardinality is
low (one open WS per browser tab). REST + SSE works but doubles the
endpoint count for no real benefit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .manager import ExperimentManager

logger = logging.getLogger(__name__)

# Default storage root: <project>/storage. Override with COMPSERV_STORAGE.
DEFAULT_STORAGE = Path(__file__).resolve().parent.parent.parent / "storage"

# Global handle, set in lifespan and used by route handlers. FastAPI's
# dependency injection would be cleaner but adds boilerplate for one
# manager instance.
_manager: ExperimentManager | None = None


def _get_manager() -> ExperimentManager:
    if _manager is None:
        raise RuntimeError("server not initialized")
    return _manager


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    global _manager
    storage_root = Path(os.environ.get("COMPSERV_STORAGE", str(DEFAULT_STORAGE)))
    storage_root.mkdir(parents=True, exist_ok=True)
    _manager = ExperimentManager(storage_root)
    logger.info("starting up; storage at %s", storage_root)
    await _manager.startup()
    try:
        yield
    finally:
        logger.info("shutting down; draining workers")
        await _manager.shutdown()
        _manager = None


app = FastAPI(title="compartmentalization-server", lifespan=lifespan)

# CORS open for v0.1 — the browser app may run on a different port (e.g.
# the SvelteKit dev server on 5173). Tighten before any deployment that's
# more than one researcher on a trusted network.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────
# HTTP routes
# ──────────────────────────────────────────────────────────────────────────


@app.get("/")
async def root() -> dict[str, Any]:
    mgr = _get_manager()
    return {
        "service": "compartmentalization-server",
        "version": "0.1.0",
        "experiments": len(mgr.records),
        "scripts": len(mgr.list_scripts()),
    }


@app.get("/scripts")
async def http_list_scripts() -> dict[str, Any]:
    return {"scripts": _get_manager().list_scripts()}


@app.get("/experiments")
async def http_list_experiments() -> dict[str, Any]:
    return {"experiments": _get_manager().list_experiments()}


# ──────────────────────────────────────────────────────────────────────────
# WebSocket protocol
# ──────────────────────────────────────────────────────────────────────────


class _Subscription:
    """Track one per-experiment subscription so we can clean it up on disconnect.

    Holding both the queue (so the manager can `unsubscribe` it from the
    record's subscribers set) and the task (so we can cancel the forwarder
    coroutine) — losing track of either leaves a leak that surfaces as
    "subscriber queue full" warnings as soon as the WS goes away.
    """

    __slots__ = ("queue", "task")

    def __init__(self, queue: asyncio.Queue[dict[str, Any]], task: asyncio.Task[None]) -> None:
        self.queue = queue
        self.task = task


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    mgr = _get_manager()

    # Map of experiment id → subscription record. We need both the task
    # (to cancel) and the queue (to unsubscribe from the manager) on
    # cleanup; tracking only the task leaks the queue.
    subs: dict[str, _Subscription] = {}
    global_q: asyncio.Queue[dict[str, Any]] | None = None
    global_task: asyncio.Task[None] | None = None

    async def forward_subscription(experiment_id: str, q: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            while True:
                event = await q.get()
                await ws.send_json({"channel": experiment_id, **event})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("forwarder for %s crashed", experiment_id)

    async def forward_global(q: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            while True:
                event = await q.get()
                await ws.send_json({"channel": "_global", **event})
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("global forwarder crashed")

    try:
        while True:
            try:
                msg = await ws.receive_json()
            except WebSocketDisconnect:
                break

            try:
                await _handle_ws_message(
                    mgr,
                    ws,
                    msg,
                    subs,
                    forward_subscription,
                )
            except Exception as e:
                logger.exception("WS message handler raised: %s", e)
                await ws.send_json(
                    {
                        "type": "error",
                        "request": msg.get("type"),
                        "message": str(e),
                    }
                )

            # Lazily set up the global subscription on first message so a
            # browser that just connects without doing anything doesn't
            # leak a queue. We attach the global stream once any RPC has
            # arrived since that's enough to know the client is real.
            if global_q is None:
                global_q = mgr.subscribe_global()
                global_task = asyncio.create_task(forward_global(global_q))

    finally:
        # Tear down all per-experiment subscriptions: cancel the forwarder
        # task AND unsubscribe the queue from the manager. Forgetting the
        # second step leaks the queue, which the manager keeps trying to
        # push events to until it fills up and starts dropping events.
        for exp_id, sub in subs.items():
            sub.task.cancel()
            mgr.unsubscribe(exp_id, sub.queue)
        for sub in subs.values():
            try:
                await sub.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if global_task is not None:
            global_task.cancel()
            try:
                await global_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if global_q is not None:
            mgr.unsubscribe_global(global_q)


async def _handle_ws_message(
    mgr: ExperimentManager,
    ws: WebSocket,
    msg: dict[str, Any],
    subs: dict[str, _Subscription],
    forward_subscription,
) -> None:
    msg_type = msg.get("type")
    request_id = msg.get("request_id")  # optional, echoed in reply for client correlation

    def reply(payload: dict[str, Any]) -> dict[str, Any]:
        out = {"type": f"{msg_type}_result", **payload}
        if request_id is not None:
            out["request_id"] = request_id
        return out

    if msg_type == "list":
        await ws.send_json(reply({"experiments": mgr.list_experiments()}))

    elif msg_type == "list_scripts":
        await ws.send_json(reply({"scripts": mgr.list_scripts()}))

    elif msg_type == "create":
        script = msg["script"]
        params = msg.get("params") or {}
        total_steps = int(msg.get("total_steps", 1000))
        description = msg.get("description", "")
        try:
            exp_id = await mgr.create_experiment(
                script=script,
                params=params,
                total_steps=total_steps,
                description=description,
            )
        except (KeyError, RuntimeError) as e:
            await ws.send_json(reply({"error": str(e)}))
            return
        await ws.send_json(reply({"id": exp_id}))

    elif msg_type == "subscribe":
        exp_id = msg["id"]
        # Idempotent: if already subscribed on this WS, replace.
        if exp_id in subs:
            old = subs.pop(exp_id)
            old.task.cancel()
            mgr.unsubscribe(exp_id, old.queue)
            try:
                await old.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

        try:
            q = mgr.subscribe(exp_id)
        except KeyError as e:
            await ws.send_json(reply({"error": str(e)}))
            return

        # Send the metric history first so the client can backfill its chart.
        # Downsample to at most `max_entries` entries on the server — stops
        # long experiments from shipping multi-MB JSON payloads that freeze
        # the browser on the reactive cascade. Default 2000; client can
        # request more (or all, via null) with a `max_history` field.
        raw_max = msg.get("max_history")
        if raw_max is None:
            max_entries: int | None = 2000
        elif raw_max == 0 or raw_max == "all":
            max_entries = None
        else:
            max_entries = int(raw_max)
        history = mgr.get_metrics_history(exp_id, max_entries=max_entries)
        rec = mgr.records[exp_id]
        await ws.send_json(
            reply(
                {
                    "experiment": rec.public_view(),
                    "history": history,
                }
            )
        )
        # Then start forwarding live events.
        task = asyncio.create_task(forward_subscription(exp_id, q))
        subs[exp_id] = _Subscription(q, task)

    elif msg_type == "unsubscribe":
        exp_id = msg["id"]
        sub = subs.pop(exp_id, None)
        if sub is not None:
            sub.task.cancel()
            mgr.unsubscribe(exp_id, sub.queue)
            try:
                await sub.task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await ws.send_json(reply({}))

    elif msg_type == "stop":
        exp_id = msg["id"]
        try:
            await mgr.stop_experiment(exp_id)
        except KeyError as e:
            await ws.send_json(reply({"error": str(e)}))
            return
        await ws.send_json(reply({}))

    elif msg_type == "delete":
        exp_id = msg["id"]
        try:
            await mgr.delete_experiment(exp_id)
        except (KeyError, RuntimeError) as e:
            await ws.send_json(reply({"error": str(e)}))
            return
        await ws.send_json(reply({}))

    elif msg_type == "set_param":
        exp_id = msg["id"]
        try:
            mgr.set_param(exp_id, msg["key"], msg["value"])
        except (KeyError, RuntimeError) as e:
            await ws.send_json(reply({"error": str(e)}))
            return
        await ws.send_json(reply({}))

    elif msg_type == "set_description":
        exp_id = msg["id"]
        try:
            mgr.set_description(exp_id, str(msg.get("description", "")))
        except KeyError as e:
            await ws.send_json(reply({"error": str(e)}))
            return
        await ws.send_json(reply({}))

    else:
        await ws.send_json({"type": "error", "message": f"unknown message type: {msg_type}"})


# ──────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────


def cli() -> None:
    """Entry point for `compartmentalization-server` CLI."""
    logging.basicConfig(
        level=os.environ.get("COMPSERV_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    host = os.environ.get("COMPSERV_HOST", "0.0.0.0")  # noqa: S104  intentional per spec
    port = int(os.environ.get("COMPSERV_PORT", "9883"))

    # Uvicorn handles SIGINT/SIGTERM and triggers the lifespan context's
    # shutdown phase, which is where ExperimentManager.shutdown runs.
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=os.environ.get("COMPSERV_LOG_LEVEL", "info").lower(),
        access_log=False,
        # Important: with multiple workers, each would get its own
        # ExperimentManager instance and you'd end up with N copies of
        # every experiment. Force single worker.
        workers=1,
        # Long-lived WS connections — disable any keepalive timeout that
        # would kick connected clients off after idle.
        ws_ping_interval=20.0,
        ws_ping_timeout=60.0,
    )
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    cli()
