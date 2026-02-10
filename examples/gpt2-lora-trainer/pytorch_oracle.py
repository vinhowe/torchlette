#!/usr/bin/env python3
"""
PyTorch oracle for LoRA training comparison.
Outputs activations and gradients as JSON for comparison with Torchlette.
"""

import json
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# Configuration - cache is in project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache" / "gpt2-lora-test"
WEIGHTS_FILE = CACHE_DIR / "model.safetensors"

# Global counter for deterministic LoRA initialization
_lora_init_counter = 0


class LoRALinear(nn.Module):
    """LoRA adapter for linear layers.

    HuggingFace GPT-2 stores weights as [in_features, out_features].
    PyTorch F.linear expects [out_features, in_features].
    So we receive HF weights and transpose them.
    """

    def __init__(self, hf_weight: torch.Tensor, hf_bias: torch.Tensor | None,
                 rank: int, alpha: float):
        super().__init__()
        # HF weight is [in_features, out_features], transpose to [out_features, in_features]
        base_weight = hf_weight.T
        in_features = base_weight.shape[1]
        out_features = base_weight.shape[0]

        self.register_buffer('base_weight', base_weight)
        if hf_bias is not None:
            self.register_buffer('base_bias', hf_bias)
        else:
            self.base_bias = None

        self.rank = rank
        self.scale = alpha / rank

        # LoRA matrices - trainable
        # A: [rank, in_features], B: [out_features, rank]
        # Use deterministic initialization for comparison
        global _lora_init_counter
        torch.manual_seed(42 + _lora_init_counter)
        _lora_init_counter += 1
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward: x @ base_weight.T (F.linear does this internally)
        out = F.linear(x, self.base_weight, self.base_bias)
        # LoRA forward: x @ A.T @ B.T * scale
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return out + lora_out


class LayerNorm(nn.Module):
    """Layer normalization matching GPT-2."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self.eps)


class CausalSelfAttention(nn.Module):
    """GPT-2 attention with LoRA on c_attn."""

    def __init__(self, c_attn_weight: torch.Tensor, c_attn_bias: torch.Tensor,
                 c_proj_weight: torch.Tensor, c_proj_bias: torch.Tensor,
                 n_head: int, lora_rank: int, lora_alpha: float):
        super().__init__()
        self.n_head = n_head
        # GPT-2 c_attn weight is [768, 2304] (input_dim, 3*output_dim)
        # embed_dim = 2304 / 3 = 768
        embed_dim = c_attn_weight.shape[1] // 3
        self.head_dim = embed_dim // n_head
        self.embed_dim = embed_dim

        # c_attn with LoRA - pass HF weight directly, LoRALinear handles transpose
        self.c_attn = LoRALinear(c_attn_weight, c_attn_bias, lora_rank, lora_alpha)

        # c_proj without LoRA (frozen) - HF is [768, 768], needs transpose for F.linear
        self.register_buffer('c_proj_weight', c_proj_weight.T)
        self.register_buffer('c_proj_bias', c_proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # QKV projection - output is [B, T, 3*embed_dim]
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=-1)

        # Reshape for attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)

        # Output projection (frozen)
        out = F.linear(out, self.c_proj_weight, self.c_proj_bias)
        return out


class MLP(nn.Module):
    """GPT-2 MLP (frozen).

    HuggingFace weights:
    - c_fc: [768, 3072]
    - c_proj: [3072, 768]
    """

    def __init__(self, c_fc_weight: torch.Tensor, c_fc_bias: torch.Tensor,
                 c_proj_weight: torch.Tensor, c_proj_bias: torch.Tensor):
        super().__init__()
        # HF c_fc is [768, 3072], transpose to [3072, 768] for F.linear
        self.register_buffer('c_fc_weight', c_fc_weight.T)
        self.register_buffer('c_fc_bias', c_fc_bias)
        # HF c_proj is [3072, 768], transpose to [768, 3072] for F.linear
        self.register_buffer('c_proj_weight', c_proj_weight.T)
        self.register_buffer('c_proj_bias', c_proj_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.linear(x, self.c_fc_weight, self.c_fc_bias)
        h = F.gelu(h, approximate='tanh')
        h = F.linear(h, self.c_proj_weight, self.c_proj_bias)
        return h


class Block(nn.Module):
    """GPT-2 transformer block."""

    def __init__(self, weights: dict, block_idx: int, n_head: int,
                 lora_rank: int, lora_alpha: float):
        super().__init__()
        prefix = f"h.{block_idx}."

        self.ln_1 = LayerNorm(
            weights[prefix + "ln_1.weight"],
            weights[prefix + "ln_1.bias"]
        )
        self.attn = CausalSelfAttention(
            weights[prefix + "attn.c_attn.weight"],
            weights[prefix + "attn.c_attn.bias"],
            weights[prefix + "attn.c_proj.weight"],
            weights[prefix + "attn.c_proj.bias"],
            n_head, lora_rank, lora_alpha
        )
        self.ln_2 = LayerNorm(
            weights[prefix + "ln_2.weight"],
            weights[prefix + "ln_2.bias"]
        )
        self.mlp = MLP(
            weights[prefix + "mlp.c_fc.weight"],
            weights[prefix + "mlp.c_fc.bias"],
            weights[prefix + "mlp.c_proj.weight"],
            weights[prefix + "mlp.c_proj.bias"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2LoRA(nn.Module):
    """GPT-2 with LoRA adapters."""

    def __init__(self, weights: dict, n_layer: int = 12, n_head: int = 12,
                 lora_rank: int = 8, lora_alpha: float = 16.0):
        super().__init__()

        # Embeddings (frozen)
        self.register_buffer('wte', weights['wte.weight'])
        self.register_buffer('wpe', weights['wpe.weight'])

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(weights, i, n_head, lora_rank, lora_alpha)
            for i in range(n_layer)
        ])

        # Final layer norm
        self.ln_f = LayerNorm(weights['ln_f.weight'], weights['ln_f.bias'])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape

        # Token + position embeddings
        tok_emb = F.embedding(input_ids, self.wte)
        pos_emb = F.embedding(torch.arange(T, device=input_ids.device), self.wpe)
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # LM head (tied weights)
        logits = F.linear(x, self.wte)
        return logits

    def forward_with_loss(self, input_ids: torch.Tensor,
                          targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    def get_lora_parameters(self) -> list[nn.Parameter]:
        params = []
        for block in self.blocks:
            params.append(block.attn.c_attn.lora_A)
            params.append(block.attn.c_attn.lora_B)
        return params


def load_weights() -> dict[str, torch.Tensor]:
    """Load weights from safetensors file."""
    weights = {}
    with safe_open(str(WEIGHTS_FILE), framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def tensor_to_list(t: torch.Tensor) -> list:
    """Convert tensor to nested list for JSON serialization."""
    return t.detach().cpu().numpy().tolist()


def reset_lora_counter():
    """Reset the LoRA initialization counter for reproducible comparisons."""
    global _lora_init_counter
    _lora_init_counter = 0


def run_comparison(input_ids: list, targets: list, lora_rank: int = 8,
                   lora_alpha: float = 16.0, lr: float = 0.01,
                   use_amp: bool = False, use_checkpointing: bool = False) -> dict:
    """
    Run forward and backward pass, return activations and gradients.
    """
    device = torch.device("cpu")  # Use CPU for exact comparison

    # Reset LoRA counter for reproducible initialization
    reset_lora_counter()

    # Load weights and create model
    weights = load_weights()
    model = GPT2LoRA(weights, n_layer=12, n_head=12,
                     lora_rank=lora_rank, lora_alpha=lora_alpha)
    model = model.to(device)
    model.train()

    # Convert inputs
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
    target_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    # Get LoRA parameters
    lora_params = model.get_lora_parameters()

    results = {
        "forward": {},
        "gradients": {},
        "losses": [],
        "lora_init": {}
    }

    # Store initial LoRA values for comparison
    for i, param in enumerate(lora_params):
        name = f"lora_param_{i}"
        results["lora_init"][name] = {
            "sum": param.sum().item(),
            "mean": param.mean().item(),
            "shape": list(param.shape)
        }

    # Initialize optimizer
    optimizer = torch.optim.Adam(lora_params, lr=lr)

    # Run 3 training steps
    for step in range(3):
        optimizer.zero_grad()

        # AMP context
        if use_amp:
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                logits, loss = model.forward_with_loss(input_tensor, target_tensor)
        else:
            logits, loss = model.forward_with_loss(input_tensor, target_tensor)

        results["losses"].append(loss.item())

        # Store forward pass outputs (only first step)
        if step == 0:
            results["forward"]["logits_sum"] = logits.sum().item()
            results["forward"]["logits_mean"] = logits.mean().item()
            results["forward"]["loss"] = loss.item()

        # Backward
        loss.backward()

        # Store gradients (only first step)
        if step == 0:
            for i, param in enumerate(lora_params):
                if param.grad is not None:
                    name = f"lora_param_{i}"
                    results["gradients"][name] = {
                        "sum": param.grad.sum().item(),
                        "mean": param.grad.mean().item(),
                        "abs_max": param.grad.abs().max().item(),
                        "shape": list(param.grad.shape)
                    }

        # Optimizer step
        optimizer.step()

    return results


def main():
    """Main entry point - reads config from stdin, outputs JSON."""
    # Read configuration from command line args or stdin
    if len(sys.argv) > 1:
        config = json.loads(sys.argv[1])
    else:
        config = json.load(sys.stdin)

    input_ids = config.get("input_ids", [[0] * 32])
    targets = config.get("targets", [[1] * 32])
    lora_rank = config.get("lora_rank", 8)
    lora_alpha = config.get("lora_alpha", 16.0)
    lr = config.get("lr", 0.01)
    use_amp = config.get("use_amp", False)
    use_checkpointing = config.get("use_checkpointing", False)

    try:
        results = run_comparison(
            input_ids, targets, lora_rank, lora_alpha, lr, use_amp, use_checkpointing
        )
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
