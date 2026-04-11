# compartmentalization-server

Long-lived experiment manager for compartmentalization research. A single FastAPI/WebSocket server orchestrates training subprocesses pinned to GPUs 8–15, persists checkpoints, and resumes everything on restart so you can edit code without losing in-flight runs.

## Why

- **Interactive experimentation.** Frob hyperparameters, kick experiments off, walk away. Come back, look at the loss curve, tweak something else.
- **Edit-restart-resume cycle.** Stop the server (`Ctrl+C` / `SIGTERM`), workers checkpoint and exit cleanly, restart the server, every running experiment picks up exactly where it left off. The whole cycle takes seconds.
- **Multi-GPU.** Each experiment is its own subprocess, pinned to a specific GPU via `CUDA_VISIBLE_DEVICES`. Round-robins across GPUs 8–15.
- **Crash safety.** Periodic checkpoints every 100 steps mean a process crash loses at most 100 steps of work.

## Architecture

```
browser (web UI)
   │  WebSocket
   ▼
server.py  (FastAPI, single process, 0.0.0.0:9883)
   │
   ├─ ExperimentManager — registry, persistence, lifecycle
   │
   ├─ Worker subprocess  ─ pinned to GPU 8 ─ running mess3
   ├─ Worker subprocess  ─ pinned to GPU 9 ─ running mess3 (different params)
   └─ Worker subprocess  ─ pinned to GPU 10 ─ running bio
       │
       │  stdin/stdout JSON-line IPC
       │  stderr → worker.log
       │
       └─ checkpoints to storage/<id>/checkpoint.pt every 100 steps
                                   + on SIGTERM
```

## Run

```
cd compartmentalization-server
uv sync
uv run compartmentalization-server
```

Server binds to `0.0.0.0:9883`. WebSocket endpoint at `ws://<host>:9883/ws`.

## Add an experiment

Create a file in `experiments/`:

```python
# experiments/my_experiment.py
import torch
from compartmentalization_server.api import Experiment, register

@register(
    "my_experiment",
    description="A new experiment",
    params={
        "lr":         {"default": 1e-3, "min": 1e-5, "max": 1, "live": True, "scale": "log"},
        "batch_size": {"default": 64,   "min": 1,    "max": 1024, "live": True},
    },
)
class MyExperiment(Experiment):
    def setup(self) -> None:
        self.model = ...
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"])

    def step(self) -> dict[str, float]:
        # one training step; return metrics
        loss = ...
        return {"loss": loss.item()}

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": torch.get_rng_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
```

The framework handles: the loop, periodic checkpointing, signal handling, metrics streaming, GPU pinning. Your script supplies model + step + state ser/de.

## Storage layout

```
storage/
├── exp_2026-04-11T22-33-44-abc12/
│   ├── metadata.json    # script, params, status, gpu, created_at, ...
│   ├── checkpoint.pt    # latest torch.save dump (atomic write)
│   ├── metrics.jsonl    # append-only {step, metrics, ts} per line
│   └── worker.log       # captured stderr from the subprocess
```

The `storage/` directory is gitignored. Each experiment is self-contained.
