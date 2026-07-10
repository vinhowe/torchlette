"""Convert a Gemma Scope params.npz to flat little-endian f32 .bin files + a
JSON manifest, so the browser/node SAE loader needs no npz parser.

Emits, into OUT_DIR:
  W_enc.bin     [dModel, numFeatures] f32
  b_enc.bin     [numFeatures]         f32
  W_dec.bin     [numFeatures, dModel] f32
  b_dec.bin     [dModel]              f32
  threshold.bin [numFeatures]         f32
  sae.json      { dModel, numFeatures, layer, width, l0, files }

Run:
  uv run --with numpy python3 packages/gemma-scope-sae/src/convert-npz.py \
    ckpts/gemma-scope-2b-pt-res/layer_20/width_16k/average_l0_71/params.npz \
    ckpts/gemma-scope-2b-pt-res/sae-layer20-16k \
    --layer 20 --width 16384 --l0 71
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("outdir")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--l0", type=int, required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    d = np.load(args.npz)
    W_enc = d["W_enc"].astype("<f4")  # [dModel, numFeatures]
    b_enc = d["b_enc"].astype("<f4")
    W_dec = d["W_dec"].astype("<f4")  # [numFeatures, dModel]
    b_dec = d["b_dec"].astype("<f4")
    threshold = d["threshold"].astype("<f4")

    dModel, numFeatures = W_enc.shape
    assert W_dec.shape == (numFeatures, dModel), W_dec.shape
    assert b_enc.shape == (numFeatures,)
    assert b_dec.shape == (dModel,)
    assert threshold.shape == (numFeatures,)

    for name, arr in [
        ("W_enc", W_enc),
        ("b_enc", b_enc),
        ("W_dec", W_dec),
        ("b_dec", b_dec),
        ("threshold", threshold),
    ]:
        arr.tofile(os.path.join(args.outdir, f"{name}.bin"))

    manifest = {
        "model": "gemma-2-2b",
        "sae": "gemma-scope-2b-pt-res",
        "layer": args.layer,
        "width": args.width,
        "l0": args.l0,
        "dModel": int(dModel),
        "numFeatures": int(numFeatures),
        "neuronpediaSaeId": f"{args.layer}-gemmascope-res-{args.width // 1024}k",
        "files": {
            "W_enc": "W_enc.bin",
            "b_enc": "b_enc.bin",
            "W_dec": "W_dec.bin",
            "b_dec": "b_dec.bin",
            "threshold": "threshold.bin",
        },
        "dtype": "float32",
    }
    with open(os.path.join(args.outdir, "sae.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.outdir}: dModel={dModel} numFeatures={numFeatures}")


if __name__ == "__main__":
    main()
