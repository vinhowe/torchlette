"""
Experiment base class and the @register decorator that scripts use to publish
themselves to the framework.

An experiment script lives at compartmentalization-server/experiments/<name>.py
and looks like::

    @register(
        "mess3",
        description="...",
        params={
            "lr": {"default": 1e-2, "min": 1e-5, "max": 1, "live": True, "scale": "log"},
            ...
        },
    )
    class Mess3(Experiment):
        def setup(self) -> None: ...
        def step(self) -> dict[str, float]: ...
        def state_dict(self) -> dict: ...
        def load_state_dict(self, state: dict) -> None: ...

The framework supplies the `params` dict and a `device` (a torch.device) at
construction. Scripts shouldn't read os.environ for the GPU index — they get
exactly one CUDA device thanks to CUDA_VISIBLE_DEVICES, addressed as cuda:0.

Distinction between live params (editable mid-run) and structural params
(model dimensions, vocab size — fixed at construction) lets the UI decide
which knobs to render as live sliders vs locked-once-created fields.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

import torch


class ParamSpec(TypedDict, total=False):
    """Schema for a single tunable parameter as declared on an experiment class.

    `default` is required. Numeric params should also supply min/max so the UI
    can render a bounded slider. `live=True` means the value can be edited
    while the experiment is running; `live=False` means it's fixed at creation
    (typically because it affects model architecture or batch shape).
    """

    type: Literal["number", "boolean", "select"]
    default: float | int | bool | str
    min: float | int
    max: float | int
    scale: Literal["linear", "log"]
    live: bool
    choices: list[str]
    description: str


class Experiment:
    """Base class for experiment scripts.

    Subclasses override setup/step/state_dict/load_state_dict. The framework
    constructs the instance, calls setup() once, then drives step() in a loop
    while periodically asking for state_dict() to checkpoint.

    Live parameter edits arrive via `self.params[key] = value` from the
    runtime — subclasses should re-read `self.params` on every iteration of
    step() rather than caching at setup time.
    """

    #: Subclasses set this via @register, NOT by overriding directly.
    name: str = ""
    description: str = ""
    param_specs: dict[str, ParamSpec] = {}

    def __init__(
        self,
        params: dict[str, Any],
        device: torch.device,
        seed: int = 0,
    ) -> None:
        # Use a plain dict so external mutation (live param edits) is visible
        # without any reactive plumbing — subclasses just re-read self.params.
        self.params: dict[str, Any] = dict(params)
        self.device = device
        self.seed = seed
        # Step counter is owned by the framework, not the script. The script
        # reads it for logging if it wants but should not mutate it.
        self.step_count: int = 0

    # ── overridable hooks ──

    def setup(self) -> None:
        """Build model, optimizer, dataset, anything that lives across steps.

        Called once before the first step(). Reads structural params from
        self.params. The default implementation is empty.
        """

    def step(self) -> dict[str, float]:
        """Run exactly one training step. Return a dict of scalar metrics.

        The framework increments self.step_count AFTER step() returns. Live
        param values may have changed since the previous call — re-read
        self.params here, do not cache.
        """
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        """Serialize everything needed to resume from this exact step.

        Typically: model.state_dict(), optimizer.state_dict(), and
        torch.get_rng_state() / torch.cuda.get_rng_state(). The framework
        will torch.save() the result and add a few framework-level fields
        (step_count, params snapshot) before writing.
        """
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore from a previous state_dict() snapshot.

        Called by the framework after setup() if a checkpoint exists on disk.
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """Optional: release any resources before the worker exits.

        Called on graceful shutdown after the final checkpoint is written.
        Default is a no-op.
        """


# ──────────────────────────────────────────────────────────────────────────
# Registration
# ──────────────────────────────────────────────────────────────────────────

# Module-level registry. Populated as @register decorators run during the
# initial import of the experiments package.
_REGISTRY: dict[str, type[Experiment]] = {}


def register(
    name: str,
    *,
    description: str = "",
    params: dict[str, ParamSpec] | None = None,
):
    """Decorator that registers an Experiment subclass under a stable name.

    The framework discovers experiments by importing every .py module in the
    experiments/ directory; each module's @register call adds an entry to
    the global registry. Names must be unique across all modules.
    """

    def decorator(cls: type[Experiment]) -> type[Experiment]:
        if not issubclass(cls, Experiment):
            raise TypeError(f"@register target {cls.__name__} must subclass Experiment")
        if name in _REGISTRY:
            existing = _REGISTRY[name]
            raise ValueError(
                f"Experiment name {name!r} is already registered "
                f"by {existing.__module__}.{existing.__qualname__}"
            )
        cls.name = name
        cls.description = description
        cls.param_specs = dict(params or {})
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_experiment_class(name: str) -> type[Experiment]:
    """Look up a registered experiment class by name."""
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"unknown experiment {name!r}; registered: {known}")
    return _REGISTRY[name]


def list_experiments() -> list[dict[str, Any]]:
    """Return a JSON-serializable list of all registered experiments and
    their parameter schemas. Used by the server to populate the UI's
    "create experiment" form.
    """
    return [
        {
            "name": cls.name,
            "description": cls.description,
            "params": cls.param_specs,
        }
        for cls in _REGISTRY.values()
    ]


def default_params(name: str) -> dict[str, Any]:
    """Return a dict of default values for an experiment's params, suitable
    as the starting point for create-experiment forms.
    """
    cls = get_experiment_class(name)
    return {key: spec["default"] for key, spec in cls.param_specs.items() if "default" in spec}
