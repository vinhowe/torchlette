"""
Generate from a PyTorch checkpoint trained by train_solo.py / agent.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from model import GPT, GPTConfig  # noqa: E402
from transformers import GPT2TokenizerFast  # noqa: E402


def main() -> None:
    ckpt = os.environ["CHECKPOINT_PATH"]
    prompt = os.environ.get("PROMPT", "Once upon a time, there was a little girl named")
    max_new = int(os.environ.get("MAX_NEW_TOKENS", "150"))
    temperature = float(os.environ.get("TEMPERATURE", "0.7"))
    top_k = int(os.environ.get("TOP_K", "40"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = GPTConfig(**blob["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new, temperature=temperature, top_k=top_k)
    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
