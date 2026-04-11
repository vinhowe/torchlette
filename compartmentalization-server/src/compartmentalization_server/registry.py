"""
Discover experiment scripts at startup.

The server's `experiments/` directory holds one Python module per experiment
script. Each module decorates an Experiment subclass with @register, which
populates the module-level registry in `api.py`. This module loads all of
those modules so the registry is populated before the server starts handling
RPCs.

Why explicit discovery instead of `from experiments import *`: we want to
control the search path (so the worker subprocess can use the same logic
to find a script by name without needing the experiments package on its
PYTHONPATH), and we want to give nice errors when a script fails to import.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def experiments_dir() -> Path:
    """Resolve the experiments/ directory relative to the project root.

    The package lives at src/compartmentalization_server/, so the project
    root is two levels up. Experiments live next to src/.
    """
    return Path(__file__).resolve().parent.parent.parent / "experiments"


def discover_experiments(directory: Path | None = None) -> list[str]:
    """Import every .py file in `directory` (default: experiments/).

    Each successful import registers any decorated Experiment subclass
    with `api._REGISTRY`. Returns the list of module names that were
    successfully loaded. Failures are logged but don't abort discovery —
    one broken script shouldn't prevent the rest from being available.
    """
    target = directory or experiments_dir()
    if not target.exists():
        logger.warning("experiments directory does not exist: %s", target)
        return []

    loaded: list[str] = []
    for py_file in sorted(target.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"compartmentalization_experiments.{py_file.stem}"
        try:
            _import_file(py_file, module_name)
            loaded.append(py_file.stem)
        except Exception as e:
            logger.exception("failed to load experiment script %s: %s", py_file.name, e)
    return loaded


def _import_file(path: Path, module_name: str) -> None:
    """Import a single .py file as a uniquely-named module.

    We assign each script a synthetic module name under
    `compartmentalization_experiments.*` so they don't collide with anything
    else on sys.modules and so the worker can re-import the same script
    by file path without depending on package layout.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
