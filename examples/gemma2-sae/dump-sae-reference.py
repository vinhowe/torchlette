"""SAE parity reference: run Gemma-2-2B, capture the layer-20 residual
(hidden_states[21] = post-block-20, pre-final-norm), and encode it with a numpy
JumpReLU reference on the SAME residual. Emits, under
ckpts/gemma-scope-2b-pt-res/sae-parity/:

  manifest.json          prompts, token ids, shapes, per-prompt residual/acts files, top-K
  resid_{i}.bin          [seq, dModel]     f32  (the layer-20 residual — the TS SAE gate
                                                 encodes THIS exact dump)
  acts_{i}.bin           [seq, numFeatures] f32 (numpy JumpReLU reference activations)

The JumpReLU convention (Gemma Scope, arXiv 2408.05147 App. A + tutorial):
  pre  = x @ W_enc + b_enc          (no b_dec pre-centering)
  acts = relu(pre) * (pre > threshold)

Run:
  CUDA_VISIBLE_DEVICES=0 uv run --with transformers --with torch --with numpy \
    python3 examples/gemma2-sae/dump-sae-reference.py
"""

import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "../../ckpts/gemma-2-2b")
SAE_DIR = os.path.join(
    HERE, "../../ckpts/gemma-scope-2b-pt-res/sae-layer20-16k"
)
OUT_DIR = os.path.join(
    HERE, "../../ckpts/gemma-scope-2b-pt-res/sae-parity"
)
LAYER = 20  # residual SAE hookpoint: blocks.20.hook_resid_post

PROMPTS = [
    "The capital of France is",
    "The Golden Gate Bridge is a famous landmark in",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n",
]
TOPK = 20


def load_sae_bins():
    with open(os.path.join(SAE_DIR, "sae.json")) as f:
        m = json.load(f)
    d, F = m["dModel"], m["numFeatures"]

    def rd(name, shape):
        a = np.fromfile(os.path.join(SAE_DIR, m["files"][name]), dtype="<f4")
        return a.reshape(shape)

    return (
        m,
        rd("W_enc", (d, F)),
        rd("b_enc", (F,)),
        rd("threshold", (F,)),
        rd("W_dec", (F, d)),
        rd("b_dec", (d,)),
    )


def jumprelu_encode(x, W_enc, b_enc, threshold):
    # x: [seq, dModel] -> pre: [seq, F]
    pre = x @ W_enc + b_enc
    return np.where(pre > threshold, np.maximum(pre, 0.0), 0.0).astype("<f4")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m, W_enc, b_enc, threshold, W_dec, b_dec = load_sae_bins()

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model = model.cuda().eval()
    cfg = model.config

    manifest = {
        "model": "unsloth/gemma-2-2b",
        "sae_layer": LAYER,
        "dModel": m["dModel"],
        "numFeatures": m["numFeatures"],
        "neuronpediaSaeId": m["neuronpediaSaeId"],
        "topk": TOPK,
        "prompts": [],
    }

    for i, prompt in enumerate(PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        # hidden_states[0] = embeddings; hidden_states[k] = output of layer k-1.
        # Layer-20 residual (post-block-20) = hidden_states[21].
        resid = out.hidden_states[LAYER + 1][0].float().cpu().numpy().astype("<f4")
        seq = resid.shape[0]

        acts = jumprelu_encode(resid, W_enc, b_enc, threshold)  # [seq, F]

        # Aggregate (max over sequence positions) → per-prompt top-K features.
        agg = acts.max(axis=0)  # [F]
        top_idx = np.argsort(-agg)[:TOPK].tolist()
        top_val = [round(float(agg[j]), 5) for j in top_idx]
        # Last-position top-K too (what steering-on-the-last-token would see).
        last = acts[-1]
        last_top_idx = np.argsort(-last)[:TOPK].tolist()

        resid.tofile(os.path.join(OUT_DIR, f"resid_{i}.bin"))
        acts.tofile(os.path.join(OUT_DIR, f"acts_{i}.bin"))

        manifest["prompts"].append(
            {
                "text": prompt,
                "token_ids": ids[0].cpu().tolist(),
                "seq_len": int(seq),
                "resid_file": f"resid_{i}.bin",
                "acts_file": f"acts_{i}.bin",
                "resid_shape": [int(seq), m["dModel"]],
                "acts_shape": [int(seq), m["numFeatures"]],
                "agg_topk_idx": top_idx,
                "agg_topk_val": top_val,
                "last_topk_idx": last_top_idx,
                "n_active_last": int((last > 0).sum()),
            }
        )
        print(
            f"prompt {i}: seq={seq} n_active_last={(last>0).sum()} "
            f"agg_top5={top_idx[:5]}"
        )

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
