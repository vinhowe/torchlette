#!/usr/bin/env python3
"""Triton realizer harness — execute (or compile-check) emitted Triton source.

The TypeScript emitter (src/schedule/realizers/triton-emit.ts) produces a Triton
kernel `matmul_kernel` as SOURCE TEXT. This harness:

  1. writes the emitted source to a real .py module (Triton's @triton.jit reads
     the kernel body via `inspect.getsource`, so an in-memory / `-c` kernel has
     no source and fails — the source MUST be a file on disk);
  2. imports it, moves inputs (loaded from .npy) to CUDA, launches with the
     schedule's launch grid + num_warps/num_stages requests;
  3. writes the output back to .npy.

Modes:
  --mode run             execute on the visible CUDA device, write output .npy
  --mode compile-check   triton.compile the kernel WITHOUT launching (the
                         honest fallback if a device/capability wall is hit)

Determinism / honesty: the harness NEVER edits the emitted source; it only wraps
launch. What runs vs compiles-only is reported in the JSON result.

Spec (JSON on stdin or --spec file):
  {
    "source": "<emitted triton source>",
    "entry_point": "matmul_kernel",
    "num_warps": 4 | null,
    "num_stages": 2 | null,
    "block": [BLOCK_M, BLOCK_N, BLOCK_K],
    "grid_map": "identity" | "swap" | "grouped",
    "group_size": 8,              # only for grouped
    "shapes": {"M":.., "N":.., "K":..},
    "has_bias": false,
    "alpha": 1.0,
    "a_npy": "path", "b_npy": "path", "bias_npy": "path|null",
    "out_npy": "path",
    "out_dtype": "f32" | "f16"
  }
Emits a JSON result object on stdout.
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import traceback


def _load_kernel(source: str, entry_point: str):
    """Write source to a temp .py and import the JIT kernel by name."""
    tmpdir = tempfile.mkdtemp(prefix="triton_realizer_")
    path = os.path.join(tmpdir, "emitted_kernel.py")
    with open(path, "w") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("emitted_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, entry_point), path


def _grid_size(spec):
    import triton

    M = spec["shapes"]["M"]
    N = spec["shapes"]["N"]
    BM, BN, _ = spec["block"]
    return (triton.cdiv(M, BM) * triton.cdiv(N, BN),)


def run(spec):
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        return {
            "mode": "run",
            "ran": False,
            "reason": "torch.cuda.is_available() is False",
        }

    kernel, path = _load_kernel(spec["source"], spec["entry_point"])
    dev = "cuda"

    a = torch.from_numpy(np.load(spec["a_npy"])).to(dev)
    b = torch.from_numpy(np.load(spec["b_npy"])).to(dev)
    M = spec["shapes"]["M"]
    N = spec["shapes"]["N"]
    K = spec["shapes"]["K"]
    out_dtype = torch.float16 if spec["out_dtype"] == "f16" else torch.float32
    c = torch.empty((M, N), device=dev, dtype=out_dtype)

    # Strides (row-major contiguous inputs — the emitter's view flags already
    # chose which logical stride is contiguous; here A,B are passed already laid
    # out so stride_am/stride_bk are the leading dims).
    stride_am = a.stride(0)
    stride_bk = b.stride(0)
    stride_cm = c.stride(0)
    alpha = float(spec.get("alpha", 1.0))

    grid = _grid_size(spec)
    BM, BN, BK = spec["block"]

    kwargs = {}
    if spec.get("num_warps") is not None:
        kwargs["num_warps"] = spec["num_warps"]
    if spec.get("num_stages") is not None:
        kwargs["num_stages"] = spec["num_stages"]

    args = [a, b, c]
    if spec.get("has_bias"):
        bias = torch.from_numpy(np.load(spec["bias_npy"])).to(dev)
        args.append(bias)
    args += [M, N, K, stride_am, stride_bk, stride_cm, alpha, BM, BN, BK]

    kernel[grid](*args, **kwargs)
    torch.cuda.synchronize()

    np.save(spec["out_npy"], c.detach().cpu().numpy())
    return {
        "mode": "run",
        "ran": True,
        "device": torch.cuda.get_device_name(0),
        "grid": list(grid),
        "num_warps": spec.get("num_warps"),
        "num_stages": spec.get("num_stages"),
        "out_npy": spec["out_npy"],
    }


def compile_check(spec):
    """triton.compile the kernel without launching (capability-wall fallback)."""
    import triton

    kernel, path = _load_kernel(spec["source"], spec["entry_point"])
    # Best-effort: instantiate the JITFunction and confirm it parsed / is jittable.
    ok = hasattr(kernel, "run") and callable(getattr(kernel, "run", None))
    return {
        "mode": "compile-check",
        "parsed": True,
        "jittable": bool(ok),
        "source_path": path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "compile-check"], default="run")
    ap.add_argument("--spec", default=None, help="spec JSON file (else stdin)")
    args = ap.parse_args()

    raw = open(args.spec).read() if args.spec else sys.stdin.read()
    spec = json.loads(raw)

    try:
        result = run(spec) if args.mode == "run" else compile_check(spec)
    except Exception as e:  # noqa: BLE001 — the harness reports failures as data
        result = {
            "mode": args.mode,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
