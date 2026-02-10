#!/usr/bin/env python3
"""
Minimal LoRA test - single layer, fixed weights.
Output format for comparison with Torchlette.
"""

import json
import torch
import torch.nn.functional as F

def main():
    torch.manual_seed(42)

    # Fixed dimensions
    batch = 1
    seq = 4
    in_features = 8
    out_features = 8
    rank = 2
    alpha = 4.0
    scale = alpha / rank

    # Fixed input
    x = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    ], requires_grad=False).view(batch, seq, in_features)

    # Fixed base weight [out, in]
    base_weight = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
        [0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1],
        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
        [0.3, 0.0, 0.3, 0.0, 0.3, 0.0, 0.3, 0.0],
        [0.0, 0.3, 0.0, 0.3, 0.0, 0.3, 0.0, 0.3],
    ], requires_grad=False)

    # LoRA matrices - A: [rank, in], B: [out, rank]
    lora_A = torch.tensor([
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
        [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07, -0.08],
    ], requires_grad=True)

    lora_B = torch.zeros(out_features, rank, requires_grad=True)

    # Forward pass
    # base: x @ W^T
    base_out = F.linear(x, base_weight)

    # LoRA: x @ A^T @ B^T * scale
    lora_out = F.linear(F.linear(x, lora_A), lora_B) * scale

    # Combined output
    out = base_out + lora_out

    # Loss = sum of output
    loss = out.sum()

    # Backward
    loss.backward()

    result = {
        "input": x.tolist(),
        "base_weight": base_weight.tolist(),
        "lora_A": lora_A.tolist(),
        "lora_B_init": [[0.0] * rank for _ in range(out_features)],
        "scale": scale,
        "base_out_sum": base_out.sum().item(),
        "lora_out_sum": lora_out.sum().item(),
        "out_sum": out.sum().item(),
        "loss": loss.item(),
        "lora_A_grad": lora_A.grad.tolist() if lora_A.grad is not None else None,
        "lora_B_grad": lora_B.grad.tolist() if lora_B.grad is not None else None,
        "lora_A_grad_sum": lora_A.grad.sum().item() if lora_A.grad is not None else 0,
        "lora_B_grad_sum": lora_B.grad.sum().item() if lora_B.grad is not None else 0,
    }

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
