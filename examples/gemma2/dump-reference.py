"""Dump HF-transformers fp32 reference activations for Gemma-2 parity testing.

Produces, under ckpts/gemma-2-2b/reference/:
  manifest.json                  prompts, token ids, shapes, file names
  logits_{i}.bin                 [seq, vocab] f32 raw
  hidden_{i}.bin                 [numLayers+1, seq, hidden] f32 raw (embeddings + each layer output)

Run: CUDA_VISIBLE_DEVICES=0 uv run --with transformers --with torch examples/gemma2/dump-reference.py
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../ckpts/gemma-2-2b")
OUT_DIR = os.path.join(MODEL_DIR, "reference")

# Short prompts for logit/hidden parity. Gemma vocab is 256000 so logits bins
# stay modest at these lengths.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model = model.cuda().eval()
    cfg = model.config

    manifest = {
        "model": "unsloth/gemma-2-2b",
        "dtype": "float32",
        "num_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "prompts": [],
    }

    for i, prompt in enumerate(PROMPTS):
        # add_special_tokens=True prepends <bos> (id 2) — Gemma expects it.
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        logits = out.logits[0].float().cpu()  # [seq, vocab]
        hidden = torch.stack([h[0].float().cpu() for h in out.hidden_states])

        logits_file = f"logits_{i}.bin"
        hidden_file = f"hidden_{i}.bin"
        logits.numpy().tofile(os.path.join(OUT_DIR, logits_file))
        hidden.numpy().tofile(os.path.join(OUT_DIR, hidden_file))

        top5 = logits[-1].topk(5)
        manifest["prompts"].append(
            {
                "text": prompt,
                "token_ids": ids[0].cpu().tolist(),
                "seq_len": ids.shape[1],
                "logits_file": logits_file,
                "logits_shape": list(logits.shape),
                "hidden_file": hidden_file,
                "hidden_shape": list(hidden.shape),
                "top5_last_ids": top5.indices.tolist(),
                "top5_last_logits": [round(v, 4) for v in top5.values.tolist()],
            }
        )
        print(f"prompt {i}: seq={ids.shape[1]} ids={ids[0].cpu().tolist()} top5={top5.indices.tolist()}")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
