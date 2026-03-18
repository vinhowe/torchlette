"""
PyTorch oracle: dump intermediate tensors and gradients for comparison with torchlette.

Runs a single forward+backward step and saves:
1. Every intermediate activation (after each layer/op)
2. Every gradient (for all LoRA parameters)
3. Updated weights after one Adam step

Usage: python3 tools/oracle-compare.py > /tmp/pytorch_oracle.json
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import struct
import tiktoken
import sys

device = "cuda"
enc = tiktoken.get_encoding("gpt2")


def load_st(path):
    with open(path, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hl))
        ds = 8 + hl
        w = {}
        for n, info in hdr.items():
            if n == "__metadata__":
                continue
            dt = {"F32": torch.float32}.get(info["dtype"])
            if not dt:
                continue
            s, e = info["data_offsets"]
            f.seek(ds + s)
            w[n.replace("transformer.", "")] = torch.frombuffer(
                bytearray(f.read(e - s)), dtype=dt
            ).reshape(info["shape"])
    return w


class LoRALinear(nn.Module):
    def __init__(self, w, b, r=64):
        super().__init__()
        o, i = w.shape
        self.w = nn.Parameter(w, requires_grad=False)
        self.b = nn.Parameter(b, requires_grad=False) if b is not None else None
        # Deterministic init for reproducibility
        
        self.A = nn.Parameter(torch.ones(r, i) * 0.01)
        self.B = nn.Parameter(torch.zeros(o, r))
        self.s = 1.0

    def forward(self, x):
        base = F.linear(x, self.w, self.b)
        lora = F.linear(F.linear(x, self.A), self.B) * self.s
        # Match torchlette: detach base
        return base.detach() + lora


class Block(nn.Module):
    def __init__(self, W, p, r=64):
        super().__init__()
        self.l1w = nn.Parameter(W[f"{p}.ln_1.weight"], requires_grad=False)
        self.l1b = nn.Parameter(W[f"{p}.ln_1.bias"], requires_grad=False)
        self.l2w = nn.Parameter(W[f"{p}.ln_2.weight"], requires_grad=False)
        self.l2b = nn.Parameter(W[f"{p}.ln_2.bias"], requires_grad=False)
        self.attn = LoRALinear(
            W[f"{p}.attn.c_attn.weight"].T, W[f"{p}.attn.c_attn.bias"], r
        )
        self.pw = nn.Parameter(
            W[f"{p}.attn.c_proj.weight"].T, requires_grad=False
        )
        self.pb = nn.Parameter(W[f"{p}.attn.c_proj.bias"], requires_grad=False)
        self.fw = nn.Parameter(W[f"{p}.mlp.c_fc.weight"].T, requires_grad=False)
        self.fb = nn.Parameter(W[f"{p}.mlp.c_fc.bias"], requires_grad=False)
        self.pw2 = nn.Parameter(
            W[f"{p}.mlp.c_proj.weight"].T, requires_grad=False
        )
        self.pb2 = nn.Parameter(W[f"{p}.mlp.c_proj.bias"], requires_grad=False)

    def forward(self, x):
        B, T, C = x.shape
        h = F.layer_norm(x, (C,), self.l1w, self.l1b)
        qkv = self.attn(h)
        q, k, v = qkv.split(C, -1)
        q = q.view(B, T, 12, 64).transpose(1, 2)
        k = k.view(B, T, 12, 64).transpose(1, 2)
        v = v.view(B, T, 12, 64).transpose(1, 2)
        a = (q @ k.transpose(-2, -1)) / 8.0
        a = a.masked_fill(
            torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float("-inf")
        )
        y = (F.softmax(a, -1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + F.linear(y, self.pw, self.pb)
        h = F.layer_norm(x, (C,), self.l2w, self.l2b)
        return x + F.linear(
            F.gelu(F.linear(h, self.fw, self.fb), approximate="tanh"),
            self.pw2,
            self.pb2,
        )


class GPT2(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.wte = nn.Parameter(W["wte.weight"], requires_grad=False)
        self.wpe = nn.Parameter(W["wpe.weight"], requires_grad=False)
        self.blocks = nn.ModuleList([Block(W, f"h.{i}") for i in range(12)])
        self.lnw = nn.Parameter(W["ln_f.weight"], requires_grad=False)
        self.lnb = nn.Parameter(W["ln_f.bias"], requires_grad=False)

    def forward(self, idx):
        B, T = idx.shape
        x = F.embedding(idx, self.wte) + self.wpe[:T]
        intermediates = {"emb": x.detach().cpu().flatten()[:20].tolist()}
        for i, b in enumerate(self.blocks):
            x = b(x)
            intermediates[f"block{i}"] = x.detach().cpu().flatten()[:20].tolist()
        x = F.layer_norm(x, (768,), self.lnw, self.lnb)
        intermediates["ln_f"] = x.detach().cpu().flatten()[:20].tolist()
        logits = x @ self.wte.T
        intermediates["logits"] = logits.detach().cpu().flatten()[:20].tolist()
        return logits, intermediates


W = load_st("models/gpt2/model.safetensors")
model = GPT2(W).to(device)

text = open("node_modules/.cache/tinyshakespeare.txt").read()[:5000]
tokens = enc.encode(text)
tid = torch.tensor(tokens, dtype=torch.long)

# Forward
x = tid[:128].unsqueeze(0).to(device)
y = tid[1:129].unsqueeze(0).to(device)
logits, intermediates = model(x)
loss = F.cross_entropy(logits.view(-1, 50257), y.view(-1))
intermediates["loss"] = loss.item()

# Backward
loss.backward()

# Collect gradients
grads = {}
for i in range(12):
    b = model.blocks[i].attn
    if b.A.grad is not None:
        grads[f"block{i}_A_grad_norm"] = b.A.grad.norm().item()
        grads[f"block{i}_A_grad_first5"] = b.A.grad.flatten()[:5].tolist()
    if b.B.grad is not None:
        grads[f"block{i}_B_grad_norm"] = b.B.grad.norm().item()
        grads[f"block{i}_B_grad_first5"] = b.B.grad.flatten()[:5].tolist()

result = {**intermediates, **grads}
print(json.dumps(result, indent=2))
