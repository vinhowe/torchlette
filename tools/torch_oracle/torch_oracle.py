import json
import sys
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def resolve_dtype(name):
    if name is None:
        return torch.float32
    mapping = {
        "f16": torch.float16,
        "f32": torch.float32,
        "i32": torch.int32,
        "u32": torch.uint32 if hasattr(torch, "uint32") else None,
        "bool": torch.bool,
    }
    if name not in mapping or mapping[name] is None:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def build_tensor(payload):
    dtype = resolve_dtype(payload.get("dtype"))
    values = payload.get("values")
    shape = payload.get("shape")
    tensor = torch.tensor(values, dtype=dtype, device="cpu")
    if shape is not None:
        tensor = tensor.reshape(shape)
    return tensor


def sanitize_float(x):
    """Convert NaN/Inf to JSON-safe null."""
    import math
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def tensor_payload(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        shape = list(tensor.shape)
        raw_values = tensor.reshape(-1).tolist()
        # Sanitize NaN/Inf values for JSON
        values = [sanitize_float(v) if isinstance(v, float) else v for v in raw_values]
        return {"shape": shape, "values": values}
    val = float(value)
    return {"shape": [], "values": [sanitize_float(val)]}


def option(options, key, default=None):
    if options is None:
        return default
    return options.get(key, default)


def maybe_dtype(options):
    value = option(options, "dtype")
    if value is None:
        return None
    return resolve_dtype(value)


def apply_op(op, inputs, options):
    if op == "add":
        alpha = option(options, "alpha", 1)
        return torch.add(inputs[0], inputs[1], alpha=alpha)
    if op == "mul":
        return torch.mul(inputs[0], inputs[1])
    if op == "sum":
        kwargs = {}
        dim = option(options, "dim")
        if dim is not None:
            kwargs["dim"] = dim
        keepdim = option(options, "keepdim")
        if keepdim is not None:
            kwargs["keepdim"] = keepdim
        dtype = maybe_dtype(options)
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.sum(inputs[0], **kwargs)
    if op == "sqrt":
        return torch.sqrt(inputs[0])
    if op == "gather":
        dim = option(options, "dim")
        if dim is None:
            raise ValueError("gather requires options.dim")
        index = inputs[1].to(torch.int64)
        return torch.gather(inputs[0], dim, index)
    if op == "scatter_add":
        dim = option(options, "dim")
        if dim is None:
            raise ValueError("scatter_add requires options.dim")
        index = inputs[1].to(torch.int64)
        return torch.scatter_add(inputs[0], dim, index, inputs[2])
    if op == "sub":
        alpha = option(options, "alpha", 1)
        return torch.sub(inputs[0], inputs[1], alpha=alpha)
    if op == "div":
        kwargs = {}
        rounding = option(options, "roundingMode")
        if rounding is not None:
            kwargs["rounding_mode"] = rounding
        return torch.div(inputs[0], inputs[1], **kwargs)
    if op == "neg":
        return torch.neg(inputs[0])
    if op == "abs":
        return torch.abs(inputs[0])
    if op == "exp":
        return torch.exp(inputs[0])
    if op == "log":
        return torch.log(inputs[0])
    if op == "relu":
        return torch.relu(inputs[0])
    if op == "matmul":
        return torch.matmul(inputs[0], inputs[1])
    if op == "mean":
        kwargs = {}
        dim = option(options, "dim")
        if dim is not None:
            kwargs["dim"] = dim
        keepdim = option(options, "keepdim")
        if keepdim is not None:
            kwargs["keepdim"] = keepdim
        dtype = maybe_dtype(options)
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.mean(inputs[0], **kwargs)
    if op == "reshape":
        shape = option(options, "shape")
        if shape is None:
            raise ValueError("reshape requires options.shape")
        return torch.reshape(inputs[0], shape)
    if op == "transpose":
        dim0 = option(options, "dim0")
        dim1 = option(options, "dim1")
        if dim0 is None or dim1 is None:
            raise ValueError("transpose requires options.dim0 and options.dim1")
        return torch.transpose(inputs[0], dim0, dim1)
    if op == "linalg.norm":
        kwargs = {}
        ord_value = option(options, "ord")
        if ord_value is not None:
            kwargs["ord"] = ord_value
        dim = option(options, "dim")
        if dim is not None:
            kwargs["dim"] = dim
        keepdim = option(options, "keepdim")
        if keepdim is not None:
            kwargs["keepdim"] = keepdim
        dtype = maybe_dtype(options)
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.linalg.norm(inputs[0], **kwargs)
    raise ValueError(f"Unsupported op: {op}")


def run_case(case):
    op = case.get("op")
    inputs = [build_tensor(tensor) for tensor in case.get("inputs", [])]
    options = case.get("options")

    # Generic backward test for any op
    if op == "backward":
        inner_op = option(options, "op")
        inner_options = option(options, "opOptions")
        requires_grad_mask = option(options, "requiresGrad", [True] * len(inputs))

        # Enable gradients on specified inputs
        grad_inputs = []
        for i, tensor in enumerate(inputs):
            if requires_grad_mask[i]:
                tensor = tensor.float().requires_grad_(True)
            grad_inputs.append(tensor)

        output = apply_op(inner_op, grad_inputs, inner_options)

        # Reduce to scalar for backward
        loss = output.sum()
        loss.backward()

        grads = []
        for i, tensor in enumerate(grad_inputs):
            if requires_grad_mask[i] and tensor.grad is not None:
                grads.append(tensor_payload(tensor.grad))
            else:
                grads.append(None)

        return {
            "output": tensor_payload(output),
            "grads": grads,
        }

    # LayerNorm forward + backward
    if op == "layer_norm_backward":
        x = inputs[0].float().requires_grad_(True)
        weight = inputs[1].float().requires_grad_(True) if len(inputs) > 1 else None
        bias = inputs[2].float().requires_grad_(True) if len(inputs) > 2 else None
        normalized_shape = option(options, "normalizedShape", [x.shape[-1]])
        eps = option(options, "eps", 1e-5)

        output = F.layer_norm(x, normalized_shape, weight, bias, eps)
        loss = output.sum()
        loss.backward()

        grads = [tensor_payload(x.grad)]
        if weight is not None:
            grads.append(tensor_payload(weight.grad))
        if bias is not None:
            grads.append(tensor_payload(bias.grad))

        return {
            "output": tensor_payload(output),
            "grads": grads,
        }

    # GELU forward + backward
    if op == "gelu_backward":
        x = inputs[0].float().requires_grad_(True)
        # Use tanh approximation to match torchlette's "new GELU" (GPT-2 style)
        approximate = option(options, "approximate", "tanh")
        output = F.gelu(x, approximate=approximate)
        loss = output.sum()
        loss.backward()

        return {
            "output": tensor_payload(output),
            "grads": [tensor_payload(x.grad)],
        }

    # Softmax forward + backward
    if op == "softmax_backward":
        x = inputs[0].float().requires_grad_(True)
        dim = option(options, "dim", -1)
        output = F.softmax(x, dim=dim)
        loss = output.sum()
        loss.backward()

        return {
            "output": tensor_payload(output),
            "grads": [tensor_payload(x.grad)],
        }

    # Embedding forward only (for step-by-step debugging)
    if op == "embedding_forward":
        input_tokens = inputs[0].to(torch.long)  # [batch, seq]
        wte = inputs[1].float()  # [vocab, embed]
        wpe = inputs[2].float()  # [blockSize, embed]
        seq_len = option(options, "seqLen", input_tokens.shape[-1])

        # Token embeddings via gather
        tok_emb = F.embedding(input_tokens, wte)

        # Position embeddings
        pos = torch.arange(seq_len, device=input_tokens.device).unsqueeze(0)
        pos_emb = F.embedding(pos, wpe)

        # Combine
        output = tok_emb + pos_emb
        return {"output": tensor_payload(output)}

    # Layer norm forward only
    if op == "layer_norm_forward":
        x = inputs[0].float()
        weight = inputs[1].float() if len(inputs) > 1 else None
        bias = inputs[2].float() if len(inputs) > 2 else None
        normalized_shape = option(options, "normalized_shape", [x.shape[-1]])
        eps = option(options, "eps", 1e-5)

        output = F.layer_norm(x, normalized_shape, weight, bias, eps)
        return {"output": tensor_payload(output)}

    # Linear forward only (x @ W^T + b)
    if op == "linear_forward":
        x = inputs[0].float()
        weight = inputs[1].float()  # [out, in]
        bias = inputs[2].float() if len(inputs) > 2 else None

        output = x @ weight.T
        if bias is not None:
            output = output + bias
        return {"output": tensor_payload(output)}

    # Attention forward only (for debugging)
    if op == "attention_forward":
        x = inputs[0].float()  # [batch, seq, embed]
        c_attn_weight = inputs[1].float()  # [3*embed, embed]
        c_attn_bias = inputs[2].float()  # [3*embed]
        c_proj_weight = inputs[3].float()  # [embed, embed]
        c_proj_bias = inputs[4].float()  # [embed]

        embed_dim = option(options, "embedDim", 64)
        num_heads = option(options, "numHeads", 2)
        head_dim = embed_dim // num_heads

        batch, seq_len, _ = x.shape

        # QKV projection
        qkv = x @ c_attn_weight.T + c_attn_bias  # [batch, seq, 3*embed]
        qkv = qkv.reshape(batch, seq_len, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scale = 1.0 / (head_dim ** 0.5)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

        # Softmax and weighted sum
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        # Concatenate heads and project
        out = out.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, embed_dim)
        out = out @ c_proj_weight.T + c_proj_bias

        return {"output": tensor_payload(out)}

    # Attention debug - return intermediate values
    if op == "attention_debug":
        x = inputs[0].float()  # [batch, seq, embed]
        c_attn_weight = inputs[1].float()  # [3*embed, embed]
        c_attn_bias = inputs[2].float()  # [3*embed]

        embed_dim = option(options, "embedDim", 64)
        num_heads = option(options, "numHeads", 2)
        head_dim = embed_dim // num_heads

        batch, seq_len, _ = x.shape

        # QKV projection
        qkv = x @ c_attn_weight.T + c_attn_bias  # [batch, seq, 3*embed]

        # Return qkv raw before any reshaping
        return {"output": tensor_payload(qkv)}

    # GPT-2 forward only (no backward)
    if op == "gpt2_forward":
        config = options or {}
        vocab_size = config.get("vocabSize", 1000)
        block_size = config.get("blockSize", 128)
        embed_dim = config.get("embedDim", 64)
        num_layers = config.get("numLayers", 2)
        num_heads = config.get("numHeads", 2)

        input_tokens = inputs[0].to(torch.long)

        model = SimpleGPT2(vocab_size, block_size, embed_dim, num_layers, num_heads)

        # Load weights from inputs
        if len(inputs) > 1:
            state_dict = model.state_dict()
            param_names = list(state_dict.keys())
            for i, name in enumerate(param_names):
                if i + 1 < len(inputs):
                    state_dict[name] = inputs[i + 1].reshape(state_dict[name].shape)
            model.load_state_dict(state_dict)

        model.eval()
        with torch.no_grad():
            logits = model(input_tokens)

        return {"output": tensor_payload(logits)}

    # Cross-entropy forward + backward
    if op == "cross_entropy_backward":
        logits = inputs[0].float().requires_grad_(True)
        targets = inputs[1].to(torch.long)
        output = F.cross_entropy(logits, targets)
        output.backward()

        return {
            "output": tensor_payload(output),
            "grads": [tensor_payload(logits.grad)],
        }

    # Embedding (gather with gradient)
    if op == "embedding_backward":
        weight = inputs[0].float().requires_grad_(True)  # [vocab, embed]
        indices = inputs[1].to(torch.long)  # [batch, seq]

        # Use embedding module for proper gradient
        vocab_size, embed_dim = weight.shape
        emb = nn.Embedding(vocab_size, embed_dim)
        emb.weight = nn.Parameter(weight)

        output = emb(indices)
        loss = output.sum()
        loss.backward()

        return {
            "output": tensor_payload(output),
            "grads": [tensor_payload(emb.weight.grad)],
        }

    # Attention forward + backward (isolated)
    if op == "attention_backward":
        # inputs: q, k, v all [batch, heads, seq, head_dim]
        q = inputs[0].float().requires_grad_(True)
        k = inputs[1].float().requires_grad_(True)
        v = inputs[2].float().requires_grad_(True)
        scale = option(options, "scale", 1.0 / (q.shape[-1] ** 0.5))
        causal = option(options, "causal", False)

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) * scale

        # Optional causal mask
        if causal:
            seq_len = q.shape[-2]
            mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1)
            scores = scores.masked_fill(mask.bool(), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        output = attn @ v

        loss = output.sum()
        loss.backward()

        return {
            "output": tensor_payload(output),
            "grads": [
                tensor_payload(q.grad),
                tensor_payload(k.grad),
                tensor_payload(v.grad),
            ],
        }

    # Full attention block backward (x -> QKV projection -> attention -> output projection)
    if op == "attention_block_backward":
        x = inputs[0].float().requires_grad_(True)  # [batch, seq, embed]
        c_attn_weight = inputs[1].float().requires_grad_(True)  # [3*embed, embed]
        c_attn_bias = inputs[2].float().requires_grad_(True)  # [3*embed]
        c_proj_weight = inputs[3].float().requires_grad_(True)  # [embed, embed]
        c_proj_bias = inputs[4].float().requires_grad_(True)  # [embed]

        embed_dim = option(options, "embedDim", 64)
        num_heads = option(options, "numHeads", 2)
        head_dim = embed_dim // num_heads

        batch, seq_len, _ = x.shape

        # QKV projection
        qkv = x @ c_attn_weight.T + c_attn_bias  # [batch, seq, 3*embed]

        # Reshape and split QKV
        qkv = qkv.reshape(batch, seq_len, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = 1.0 / (head_dim ** 0.5)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # [batch, heads, seq, head_dim]

        # Concatenate heads
        out = out.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, embed_dim)

        # Output projection
        output = out @ c_proj_weight.T + c_proj_bias

        loss = output.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [
                tensor_payload(x.grad),
                tensor_payload(c_attn_weight.grad),
                tensor_payload(c_attn_bias.grad),
                tensor_payload(c_proj_weight.grad),
                tensor_payload(c_proj_bias.grad),
            ],
        }

    # Linear + attention backward
    if op == "linear_plus_attention_backward":
        x = inputs[0].float().requires_grad_(True)  # [batch, seq, embed]
        w = inputs[1].float().requires_grad_(True)  # [3*embed, embed]
        b = inputs[2].float().requires_grad_(True)  # [3*embed]
        embed_dim = option(options, "embedDim", 4)
        num_heads = option(options, "numHeads", 2)
        head_dim = embed_dim // num_heads

        batch, seq_len, _ = x.shape

        # QKV projection
        qkv = x @ w.T + b

        # PyTorch style split with head reshape
        qkv_reshaped = qkv.reshape(batch, seq_len, 3, num_heads, head_dim)
        qkv_permuted = qkv_reshaped.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv_permuted[0], qkv_permuted[1], qkv_permuted[2]

        # Attention computation
        scale = 1.0 / (head_dim ** 0.5)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        loss = out.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [
                tensor_payload(x.grad),
                tensor_payload(w.grad),
                tensor_payload(b.grad),
            ],
        }

    # Full attention backward (QKV split + attention computation)
    if op == "full_attention_backward":
        qkv = inputs[0].float().requires_grad_(True)  # [batch, seq, 3*embed]
        embed_dim = option(options, "embedDim", 4)
        num_heads = option(options, "numHeads", 2)
        head_dim = embed_dim // num_heads

        batch, seq_len, _ = qkv.shape

        # PyTorch style split with head reshape
        qkv_reshaped = qkv.reshape(batch, seq_len, 3, num_heads, head_dim)
        qkv_permuted = qkv_reshaped.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv_permuted[0], qkv_permuted[1], qkv_permuted[2]

        # Attention computation
        scale = 1.0 / (head_dim ** 0.5)
        scores = (q @ k.transpose(-2, -1)) * scale

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=qkv.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        loss = out.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(qkv.grad)],
        }

    # QKV split with head reshape backward
    if op == "qkv_head_split_backward":
        qkv = inputs[0].float().requires_grad_(True)  # [batch, seq, 3*embed]
        embed_dim = option(options, "embedDim", 4)
        num_heads = option(options, "numHeads", 2)
        head_dim = embed_dim // num_heads

        batch, seq_len, _ = qkv.shape

        # PyTorch style split with head reshape
        qkv_reshaped = qkv.reshape(batch, seq_len, 3, num_heads, head_dim)
        qkv_permuted = qkv_reshaped.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv_permuted[0], qkv_permuted[1], qkv_permuted[2]

        loss = q.sum() + k.sum() + v.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(qkv.grad)],
        }

    # QKV split backward (matching attention pattern)
    if op == "qkv_split_backward":
        qkv = inputs[0].float().requires_grad_(True)  # [batch, seq, 3*embed]
        embed_dim = option(options, "embedDim", 4)

        batch, seq_len, _ = qkv.shape

        # PyTorch style split
        qkv_reshaped = qkv.reshape(batch, seq_len, 3, embed_dim)
        qkv_permuted = qkv_reshaped.permute(2, 0, 1, 3)  # [3, batch, seq, embed]
        q, k, v = qkv_permuted[0], qkv_permuted[1], qkv_permuted[2]

        loss = q.sum() + k.sum() + v.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(qkv.grad)],
        }

    # Softmax backward
    if op == "softmax_backward":
        x = inputs[0].float().requires_grad_(True)
        dim = option(options, "dim", -1)

        out = F.softmax(x, dim=dim)
        loss = out.sum()
        loss.backward()

        return {
            "output": tensor_payload(out),
            "grads": [tensor_payload(x.grad)],
        }

    # Matmul with bias (Linear layer)
    if op == "linear_backward":
        x = inputs[0].float().requires_grad_(True)
        weight = inputs[1].float().requires_grad_(True)
        bias = inputs[2].float().requires_grad_(True) if len(inputs) > 2 else None

        output = x @ weight.T
        if bias is not None:
            output = output + bias

        loss = output.sum()
        loss.backward()

        grads = [tensor_payload(x.grad), tensor_payload(weight.grad)]
        if bias is not None:
            grads.append(tensor_payload(bias.grad))

        return {
            "output": tensor_payload(output),
            "grads": grads,
        }

    if op == "mlp_mse_backward":
        if len(inputs) < 4:
            raise ValueError("mlp_mse_backward requires 4 inputs")
        x = inputs[0].requires_grad_(True)
        w = inputs[1].requires_grad_(True)
        b = inputs[2].requires_grad_(True)
        y = inputs[3]
        pred = torch.relu(torch.matmul(x, w) + b)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        return {
            "output": tensor_payload(loss),
            "grads": [
                tensor_payload(x.grad),
                tensor_payload(w.grad),
                tensor_payload(b.grad),
            ],
        }
    if op == "mlp_mse_backward_amp":
        # Same as mlp_mse_backward but with AMP autocast
        if len(inputs) < 4:
            raise ValueError("mlp_mse_backward_amp requires 4 inputs")
        x = inputs[0].requires_grad_(True)
        w = inputs[1].requires_grad_(True)
        b = inputs[2].requires_grad_(True)
        y = inputs[3]

        # Use autocast for AMP simulation
        with torch.autocast(device_type="cpu", dtype=torch.float16, enabled=True):
            hidden = torch.matmul(x, w) + b
            pred = torch.relu(hidden)

        # Loss computation outside autocast (f32)
        loss = torch.mean((pred.float() - y) ** 2)
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "activations": {
                "hidden": tensor_payload(hidden),
                "pred": tensor_payload(pred),
            },
            "grads": [
                tensor_payload(x.grad),
                tensor_payload(w.grad),
                tensor_payload(b.grad),
            ],
        }
    if op == "two_layer_mlp_backward":
        # Two-layer MLP: x -> W1 + b1 -> relu -> W2 + b2 -> MSE
        if len(inputs) < 6:
            raise ValueError("two_layer_mlp_backward requires 6 inputs: x, w1, b1, w2, b2, target")
        x = inputs[0].requires_grad_(True)
        w1 = inputs[1].requires_grad_(True)
        b1 = inputs[2].requires_grad_(True)
        w2 = inputs[3].requires_grad_(True)
        b2 = inputs[4].requires_grad_(True)
        target = inputs[5]

        # Forward pass
        hidden1 = torch.matmul(x, w1) + b1
        act1 = torch.relu(hidden1)
        hidden2 = torch.matmul(act1, w2) + b2
        pred = hidden2  # No activation on output for regression

        # MSE loss
        loss = torch.mean((pred - target) ** 2)
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "activations": {
                "hidden1": tensor_payload(hidden1),
                "act1": tensor_payload(act1),
                "hidden2": tensor_payload(hidden2),
            },
            "grads": [
                tensor_payload(x.grad),
                tensor_payload(w1.grad),
                tensor_payload(b1.grad),
                tensor_payload(w2.grad),
                tensor_payload(b2.grad),
            ],
        }
    if op == "gpt2_forward_backward":
        # GPT-2 forward/backward with given weights
        # Options: config (vocabSize, blockSize, embedDim, numLayers, numHeads)
        # Inputs: [inputTokens, targets, wte, wpe, ...block_params..., lnF_weight, lnF_bias]
        config = options or {}
        vocab_size = config.get("vocabSize", 1000)
        block_size = config.get("blockSize", 128)
        embed_dim = config.get("embedDim", 64)
        num_layers = config.get("numLayers", 2)
        num_heads = config.get("numHeads", 2)

        # Parse inputs
        input_tokens = inputs[0].to(torch.long)  # [batch, seqLen]
        targets = inputs[1].to(torch.long)       # [batch, seqLen]

        # Create model
        model = SimpleGPT2(vocab_size, block_size, embed_dim, num_layers, num_heads)

        # Load weights from inputs (if provided beyond tokens/targets)
        if len(inputs) > 2:
            state_dict = model.state_dict()
            param_names = list(state_dict.keys())
            for i, name in enumerate(param_names):
                if i + 2 < len(inputs):
                    state_dict[name] = inputs[i + 2].reshape(state_dict[name].shape)
            model.load_state_dict(state_dict)

        # Forward pass
        logits = model(input_tokens)  # [batch, seqLen, vocabSize]

        # Cross-entropy loss
        batch, seq_len = targets.shape
        flat_logits = logits.reshape(batch * seq_len, vocab_size)
        flat_targets = targets.reshape(batch * seq_len)
        loss = F.cross_entropy(flat_logits, flat_targets)

        # Backward
        loss.backward()

        # Collect gradients
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(tensor_payload(param.grad))
            else:
                grads.append({"shape": list(param.shape), "values": [0.0] * param.numel()})

        return {
            "output": tensor_payload(loss),
            "grads": grads,
        }

    # Weight tying gradient tests - embedding only
    if op == "weight_tying_embed_grad":
        wte = inputs[0].float().requires_grad_(True)  # [vocab, embed]
        indices = inputs[1].to(torch.long)  # [batch, seq]

        # Use F.embedding for proper gradient flow
        embedded = F.embedding(indices, wte)

        loss = embedded.sum()
        loss.backward()

        return {"output": tensor_payload(wte.grad)}

    # Weight tying gradient tests - LM head only
    if op == "weight_tying_lmhead_grad":
        wte = inputs[0].float().requires_grad_(True)  # [vocab, embed]
        hidden = inputs[1].float()  # [batch, seq, embed]

        # LM head: logits = hidden @ wte.T
        logits = hidden @ wte.T

        loss = logits.sum()
        loss.backward()

        return {"output": tensor_payload(wte.grad)}

    # Weight tying gradient tests - both gather + matmul
    if op == "weight_tying_both_grad":
        wte = inputs[0].float().requires_grad_(True)  # [vocab, embed]
        indices = inputs[1].to(torch.long)  # [batch, seq]

        vocab_size, embed_dim = wte.shape

        # Step 1: Embedding via F.embedding (proper gradient flow)
        embedded = F.embedding(indices, wte)  # [batch, seq, embed]

        # Step 2: Use embedded as hidden directly
        hidden = embedded

        # Step 3: LM head (matmul with same wte)
        logits = hidden @ wte.T  # [batch, seq, vocab]

        loss = logits.sum()
        loss.backward()

        return {"output": tensor_payload(wte.grad)}

    # Full path with intermediate gradients
    if op == "attn_permute_matmul_intermediate":
        attn = inputs[0].float().requires_grad_(True)
        v = inputs[1].float().requires_grad_(True)
        w = inputs[2].float().requires_grad_(True)

        batch, heads, seq_len, head_dim = v.shape
        embed_dim = heads * head_dim

        out = attn @ v
        out.retain_grad()

        out_perm = out.permute(0, 2, 1, 3).contiguous()
        out_perm.retain_grad()

        out_flat = out_perm.reshape(batch, seq_len, embed_dim)
        out_flat.retain_grad()

        proj = out_flat @ w.T
        proj.retain_grad()

        loss = proj.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [
                tensor_payload(out.grad),  # grad at out
                tensor_payload(attn.grad),  # grad at attn
            ],
        }

    # attn @ v only
    if op == "attn_v_matmul_backward":
        attn = inputs[0].float().requires_grad_(True)  # [batch, heads, seq, seq]
        v = inputs[1].float().requires_grad_(True)      # [batch, heads, seq, head_dim]
        out = attn @ v
        loss = out.sum()
        loss.backward()
        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(attn.grad), tensor_payload(v.grad)],
        }

    # attn @ v + permute
    if op == "attn_v_matmul_permute_backward":
        attn = inputs[0].float().requires_grad_(True)  # [batch, heads, seq, seq]
        v = inputs[1].float().requires_grad_(True)      # [batch, heads, seq, head_dim]
        out = attn @ v  # [batch, heads, seq, head_dim]
        out_perm = out.permute(0, 2, 1, 3).contiguous()  # [batch, seq, heads, head_dim]
        loss = out_perm.sum()
        loss.backward()
        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(attn.grad), tensor_payload(v.grad)],
        }

    # attn @ v + permute + reshape
    if op == "attn_v_matmul_permute_reshape_backward":
        attn = inputs[0].float().requires_grad_(True)  # [batch, heads, seq, seq]
        v = inputs[1].float().requires_grad_(True)      # [batch, heads, seq, head_dim]
        batch, heads, seq_len, head_dim = v.shape
        embed_dim = heads * head_dim
        out = attn @ v  # [batch, heads, seq, head_dim]
        out_perm = out.permute(0, 2, 1, 3).contiguous()  # [batch, seq, heads, head_dim]
        out_flat = out_perm.reshape(batch, seq_len, embed_dim)  # [batch, seq, embed]
        loss = out_flat.sum()
        loss.backward()
        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(attn.grad), tensor_payload(v.grad)],
        }

    # permute -> reshape -> matmul backward
    if op == "permute_reshape_matmul_backward":
        x = inputs[0].float().requires_grad_(True)  # [batch, heads, seq, head_dim]
        w = inputs[1].float().requires_grad_(True)  # [embed, embed]

        batch, heads, seq_len, head_dim = x.shape
        embed_dim = heads * head_dim

        # x: [batch, heads, seq, head_dim] -> permute -> [batch, seq, heads, head_dim]
        x_perm = x.permute(0, 2, 1, 3).contiguous()
        # reshape -> [batch, seq, embed]
        x_flat = x_perm.reshape(batch, seq_len, embed_dim)
        # matmul with w.T
        out = x_flat @ w.T

        loss = out.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(x.grad), tensor_payload(w.grad)],
        }

    # matmul -> reshape -> permute backward
    if op == "matmul_reshape_permute_backward":
        x = inputs[0].float().requires_grad_(True)  # [batch, seq, embed]
        w = inputs[1].float().requires_grad_(True)  # [embed, embed]

        batch = option(options, "batch", 2)
        heads = option(options, "heads", 2)
        seq_len = option(options, "seqLen", 4)
        head_dim = option(options, "headDim", 4)

        # matmul first
        x_mm = x @ w.T
        # reshape to [batch, seq, heads, head_dim]
        x_reshaped = x_mm.reshape(batch, seq_len, heads, head_dim)
        # permute to [batch, heads, seq, head_dim]
        x_perm = x_reshaped.permute(0, 2, 1, 3).contiguous()

        loss = x_perm.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(x.grad), tensor_payload(w.grad)],
        }

    # 4D matmul -> permute -> reshape -> matmul backward (attention output path)
    if op == "attn_permute_matmul_backward":
        attn = inputs[0].float().requires_grad_(True)  # [batch, heads, seq, seq]
        v = inputs[1].float().requires_grad_(True)      # [batch, heads, seq, head_dim]
        w = inputs[2].float().requires_grad_(True)      # [embed, embed]

        batch, heads, seq_len, head_dim = v.shape
        embed_dim = heads * head_dim

        # attn @ v -> [batch, heads, seq, head_dim]
        attn_out = attn @ v
        # permute -> [batch, seq, heads, head_dim]
        attn_perm = attn_out.permute(0, 2, 1, 3).contiguous()
        # reshape -> [batch, seq, embed]
        attn_flat = attn_perm.reshape(batch, seq_len, embed_dim)
        # matmul with w.T
        out = attn_flat @ w.T

        loss = out.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(attn.grad), tensor_payload(v.grad), tensor_payload(w.grad)],
        }

    # 2D matmul backward
    if op == "matmul_2d_backward":
        a = inputs[0].float().requires_grad_(True)
        b = inputs[1].float().requires_grad_(True)
        c = a @ b
        loss = c.sum()
        loss.backward()
        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(a.grad), tensor_payload(b.grad)],
        }

    # 3D batched matmul backward
    if op == "matmul_3d_backward":
        a = inputs[0].float().requires_grad_(True)
        b = inputs[1].float().requires_grad_(True)
        c = a @ b
        loss = c.sum()
        loss.backward()
        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(a.grad), tensor_payload(b.grad)],
        }

    # 4D batched matmul backward
    if op == "matmul_4d_backward":
        a = inputs[0].float().requires_grad_(True)
        b = inputs[1].float().requires_grad_(True)
        c = a @ b
        loss = c.sum()
        loss.backward()
        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(a.grad), tensor_payload(b.grad)],
        }

    # 4D matmul with different inner dims backward
    if op == "matmul_4d_diff_backward":
        a = inputs[0].float().requires_grad_(True)
        b = inputs[1].float().requires_grad_(True)
        c = a @ b
        loss = c.sum()
        loss.backward()
        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(a.grad), tensor_payload(b.grad)],
        }

    # Head concat (permute + reshape) backward
    if op == "head_concat_backward":
        x = inputs[0].float().requires_grad_(True)  # [batch, heads, seq, head_dim]
        batch, heads, seq_len, head_dim = x.shape
        embed_dim = heads * head_dim

        # Permute to [batch, seq, heads, head_dim] then reshape to [batch, seq, embed]
        x_permuted = x.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_permuted.reshape(batch, seq_len, embed_dim)

        loss = x_reshaped.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(x.grad)],
        }

    # Head concat + linear backward
    if op == "head_concat_linear_backward":
        x = inputs[0].float().requires_grad_(True)  # [batch, heads, seq, head_dim]
        w = inputs[1].float().requires_grad_(True)  # [embed, embed]
        b = inputs[2].float().requires_grad_(True)  # [embed]

        batch, heads, seq_len, head_dim = x.shape
        embed_dim = heads * head_dim

        # Concat heads
        x_permuted = x.permute(0, 2, 1, 3).contiguous()
        x_reshaped = x_permuted.reshape(batch, seq_len, embed_dim)

        # Linear
        output = x_reshaped @ w.T + b

        loss = output.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(x.grad), tensor_payload(w.grad), tensor_payload(b.grad)],
        }

    # attn@v + head concat + linear backward
    if op == "attn_v_head_concat_linear_backward":
        attn = inputs[0].float().requires_grad_(True)  # [batch, heads, seq, seq]
        v = inputs[1].float().requires_grad_(True)      # [batch, heads, seq, head_dim]
        w = inputs[2].float().requires_grad_(True)      # [embed, embed]
        b = inputs[3].float().requires_grad_(True)      # [embed]

        batch, heads, seq_len, head_dim = v.shape
        embed_dim = heads * head_dim

        # attn @ v
        attn_out = attn @ v  # [batch, heads, seq, head_dim]

        # Concat heads
        attn_concat = attn_out.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, embed_dim)

        # Linear
        output = attn_concat @ w.T + b

        loss = output.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [
                tensor_payload(attn.grad),
                tensor_payload(v.grad),
                tensor_payload(w.grad),
                tensor_payload(b.grad),
            ],
        }

    # LayerNorm backward
    if op == "layernorm_backward":
        x = inputs[0].float().requires_grad_(True)
        gamma = inputs[1].float().requires_grad_(True)
        beta = inputs[2].float().requires_grad_(True)
        normalized_shape = option(options, "normalizedShape", [x.shape[-1]])

        out = F.layer_norm(x, normalized_shape, gamma, beta)
        loss = out.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(x.grad), tensor_payload(gamma.grad), tensor_payload(beta.grad)],
        }

    # LayerNorm + Linear backward
    if op == "layernorm_linear_backward":
        x = inputs[0].float().requires_grad_(True)
        gamma = inputs[1].float().requires_grad_(True)
        beta = inputs[2].float().requires_grad_(True)
        w = inputs[3].float().requires_grad_(True)
        b = inputs[4].float().requires_grad_(True)

        # LayerNorm
        ln_out = F.layer_norm(x, [x.shape[-1]], gamma, beta)
        # Linear
        qkv = ln_out @ w.T + b

        loss = qkv.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [
                tensor_payload(x.grad),
                tensor_payload(gamma.grad),
                tensor_payload(beta.grad),
                tensor_payload(w.grad),
                tensor_payload(b.grad),
            ],
        }

    # LayerNorm + Attention backward (no residual)
    if op == "layernorm_attention_backward":
        x = inputs[0].float().requires_grad_(True)
        gamma = inputs[1].float().requires_grad_(True)
        beta = inputs[2].float().requires_grad_(True)
        w = inputs[3].float().requires_grad_(True)
        b = inputs[4].float().requires_grad_(True)

        embed_dim = option(options, "embedDim", 8)
        num_heads = option(options, "numHeads", 2)
        head_dim = embed_dim // num_heads
        batch, seq_len, _ = x.shape

        # LayerNorm
        ln_out = F.layer_norm(x, [embed_dim], gamma, beta)

        # QKV projection
        qkv = ln_out @ w.T + b

        # Split Q, K, V
        qkv_reshaped = qkv.reshape(batch, seq_len, 3, embed_dim)
        qkv_permuted = qkv_reshaped.permute(2, 0, 1, 3)  # [3, batch, seq, embed]
        q, k, v = qkv_permuted[0], qkv_permuted[1], qkv_permuted[2]

        # Reshape to heads
        q = q.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # Attention
        scale = 1.0 / (head_dim ** 0.5)
        scores = q @ k.transpose(-2, -1) * scale

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1) * -1e9
        scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ v

        loss = attn_output.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [
                tensor_payload(x.grad),
                tensor_payload(gamma.grad),
                tensor_payload(beta.grad),
                tensor_payload(w.grad),
                tensor_payload(b.grad),
            ],
        }

    # LayerNorm + Attention + Residual backward
    if op == "layernorm_attention_residual_backward":
        x = inputs[0].float().requires_grad_(True)
        gamma = inputs[1].float().requires_grad_(True)
        beta = inputs[2].float().requires_grad_(True)
        w = inputs[3].float().requires_grad_(True)
        b = inputs[4].float().requires_grad_(True)
        w_proj = inputs[5].float().requires_grad_(True)
        b_proj = inputs[6].float().requires_grad_(True)

        embed_dim = option(options, "embedDim", 8)
        num_heads = option(options, "numHeads", 2)
        head_dim = embed_dim // num_heads
        batch, seq_len, _ = x.shape

        # LayerNorm
        ln_out = F.layer_norm(x, [embed_dim], gamma, beta)

        # QKV projection
        qkv = ln_out @ w.T + b

        # Split Q, K, V
        qkv_reshaped = qkv.reshape(batch, seq_len, 3, embed_dim)
        qkv_permuted = qkv_reshaped.permute(2, 0, 1, 3)
        q, k, v = qkv_permuted[0], qkv_permuted[1], qkv_permuted[2]

        # Reshape to heads
        q = q.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # Attention
        scale = 1.0 / (head_dim ** 0.5)
        scores = q @ k.transpose(-2, -1) * scale

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1) * -1e9
        scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ v

        # Concat heads
        attn_concat = attn_output.permute(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)

        # Output projection
        attn_proj_out = attn_concat @ w_proj.T + b_proj

        # Residual
        output = x + attn_proj_out

        loss = output.sum()
        loss.backward()

        return {
            "output": tensor_payload(loss),
            "grads": [
                tensor_payload(x.grad),
                tensor_payload(gamma.grad),
                tensor_payload(beta.grad),
                tensor_payload(w.grad),
                tensor_payload(b.grad),
                tensor_payload(w_proj.grad),
                tensor_payload(b_proj.grad),
            ],
        }

    # permute -> sum backward
    if op == "permute_sum_backward":
        x = inputs[0].float().requires_grad_(True)
        dims = option(options, "dims", [0, 2, 1, 3])
        x_perm = x.permute(dims).contiguous()
        loss = x_perm.sum()
        loss.backward()
        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(x.grad)],
        }

    # permute -> mul -> sum backward
    if op == "permute_mul_sum_backward":
        x = inputs[0].float().requires_grad_(True)
        w = inputs[1].float()  # weights (no grad)
        dims = option(options, "dims", [0, 2, 1, 3])
        x_perm = x.permute(dims).contiguous()
        x_mul = x_perm * w
        loss = x_mul.sum()
        loss.backward()
        return {
            "output": tensor_payload(loss),
            "grads": [tensor_payload(x.grad)],
        }

    # ========================================
    # GPT-2 Checkpoint Parity
    # ========================================

    if op == "gpt2_checkpoint_parity":
        """
        GPT-2 forward/backward with gradient checkpointing on transformer blocks.

        Inputs: [inputTokens, targets, wte, wpe, ...block_params..., lnF_weight, lnF_bias]

        Block params for each block (in order):
        - ln_1.weight, ln_1.bias
        - c_attn.weight, c_attn.bias
        - c_proj.weight, c_proj.bias
        - ln_2.weight, ln_2.bias
        - mlp.c_fc.weight, mlp.c_fc.bias
        - mlp.c_proj.weight, mlp.c_proj.bias

        Options:
        - vocabSize, blockSize, embedDim, numLayers, numHeads
        - useCheckpoint (default True)
        """
        config = options or {}
        vocab_size = config.get("vocabSize", 256)
        block_size = config.get("blockSize", 16)
        embed_dim = config.get("embedDim", 32)
        num_layers = config.get("numLayers", 2)
        num_heads = config.get("numHeads", 2)
        use_checkpoint = config.get("useCheckpoint", True)

        # Parse inputs
        input_tokens = inputs[0].to(torch.long)  # [batch, seqLen]
        targets = inputs[1].to(torch.long)       # [batch, seqLen]

        # Create model
        model = SimpleGPT2(vocab_size, block_size, embed_dim, num_layers, num_heads)

        # Load weights from inputs
        if len(inputs) > 2:
            state_dict = model.state_dict()
            param_names = list(state_dict.keys())
            for i, name in enumerate(param_names):
                if i + 2 < len(inputs):
                    state_dict[name] = inputs[i + 2].float().reshape(state_dict[name].shape)
            model.load_state_dict(state_dict)

        # Forward pass with optional checkpointing
        if use_checkpoint:
            logits = model.forward_with_checkpoint(input_tokens)
        else:
            logits = model(input_tokens)

        # Cross-entropy loss
        batch, seq_len = targets.shape
        flat_logits = logits.reshape(batch * seq_len, vocab_size)
        flat_targets = targets.reshape(batch * seq_len)
        loss = F.cross_entropy(flat_logits, flat_targets)

        # Backward
        loss.backward()

        # Collect gradients in state_dict order
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(tensor_payload(param.grad))
            else:
                grads.append({"shape": list(param.shape), "values": [0.0] * param.numel()})

        return {
            "output": tensor_payload(loss),
            "grads": grads,
        }

    # ========================================
    # Checkpoint Oracle Commands
    # ========================================

    # Checkpoint forward/backward - simple function with checkpointing
    if op == "checkpoint_forward_backward":
        x = inputs[0].float().requires_grad_(True)
        w = inputs[1].float().requires_grad_(True)
        b = inputs[2].float().requires_grad_(True) if len(inputs) > 2 else None

        num_layers = option(options, "numLayers", 2)
        use_checkpoint = option(options, "useCheckpoint", True)

        def layer_fn(x, w, b):
            """Simple layer: relu(x @ w + b)"""
            out = x @ w
            if b is not None:
                out = out + b
            return torch.relu(out)

        # Apply layers with or without checkpointing
        current = x
        for i in range(num_layers):
            if use_checkpoint:
                # Note: checkpoint requires use_reentrant=False for PyTorch 2.0+
                current = checkpoint(layer_fn, current, w, b, use_reentrant=False)
            else:
                current = layer_fn(current, w, b)

        loss = current.sum()
        loss.backward()

        grads = [tensor_payload(x.grad), tensor_payload(w.grad)]
        if b is not None:
            grads.append(tensor_payload(b.grad))

        return {
            "output": tensor_payload(loss),
            "grads": grads,
        }

    # Checkpoint with MLP layers
    if op == "checkpoint_mlp_backward":
        x = inputs[0].float().requires_grad_(True)

        # Parse layer weights: [w1, b1, w2, b2, ...]
        layer_params = []
        i = 1
        while i + 1 < len(inputs):
            w = inputs[i].float().requires_grad_(True)
            b = inputs[i + 1].float().requires_grad_(True)
            layer_params.append((w, b))
            i += 2

        use_checkpoint = option(options, "useCheckpoint", True)

        def mlp_layer(x, w, b):
            return torch.relu(x @ w + b)

        current = x
        for w, b in layer_params:
            if use_checkpoint:
                current = checkpoint(mlp_layer, current, w, b, use_reentrant=False)
            else:
                current = mlp_layer(current, w, b)

        loss = current.sum()
        loss.backward()

        grads = [tensor_payload(x.grad)]
        for w, b in layer_params:
            grads.append(tensor_payload(w.grad))
            grads.append(tensor_payload(b.grad))

        return {
            "output": tensor_payload(loss),
            "grads": grads,
        }

    # ========================================
    # AMP Oracle Commands
    # ========================================

    # AMP forward/backward - basic autocast usage
    if op == "amp_forward_backward":
        x = inputs[0].float().requires_grad_(True)
        w = inputs[1].float().requires_grad_(True)
        b = inputs[2].float().requires_grad_(True) if len(inputs) > 2 else None

        use_amp = option(options, "useAmp", True)
        device_type = option(options, "deviceType", "cpu")

        if use_amp:
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                out = x @ w
                if b is not None:
                    out = out + b
                out = torch.relu(out)
        else:
            out = x @ w
            if b is not None:
                out = out + b
            out = torch.relu(out)

        # Loss computation outside autocast (f32)
        loss = out.float().sum()
        loss.backward()

        grads = [tensor_payload(x.grad), tensor_payload(w.grad)]
        if b is not None:
            grads.append(tensor_payload(b.grad))

        return {
            "output": tensor_payload(loss),
            "activations": {
                "out_dtype": str(out.dtype),
            },
            "grads": grads,
        }

    # AMP with GradScaler
    if op == "amp_gradscaler_backward":
        x = inputs[0].float().requires_grad_(True)
        w = inputs[1].float().requires_grad_(True)
        b = inputs[2].float().requires_grad_(True) if len(inputs) > 2 else None
        target = inputs[3] if len(inputs) > 3 else None

        init_scale = option(options, "initScale", 65536.0)
        growth_factor = option(options, "growthFactor", 2.0)
        backoff_factor = option(options, "backoffFactor", 0.5)
        growth_interval = option(options, "growthInterval", 2000)
        device_type = option(options, "deviceType", "cpu")

        scaler = torch.amp.GradScaler(
            device=device_type,
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
        )

        with torch.autocast(device_type=device_type, dtype=torch.float16):
            out = x @ w
            if b is not None:
                out = out + b
            out = torch.relu(out)

            if target is not None:
                loss = F.mse_loss(out.float(), target.float())
            else:
                loss = out.float().sum()

        # Scale loss and backward
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        # Check for inf/nan in gradients
        found_inf = False
        for param in [x, w, b]:
            if param is not None and param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    found_inf = True
                    break

        # Get scale before potential update
        scale_before = scaler.get_scale()

        grads = [tensor_payload(x.grad), tensor_payload(w.grad)]
        if b is not None:
            grads.append(tensor_payload(b.grad))

        return {
            "output": tensor_payload(loss),
            "scaledOutput": tensor_payload(scaled_loss),
            "scale": scale_before,
            "foundInf": found_inf,
            "grads": grads,
        }

    # AMP with NaN/Inf injection for GradScaler testing
    if op == "amp_gradscaler_nan_test":
        x = inputs[0].float().requires_grad_(True)
        w = inputs[1].float().requires_grad_(True)

        inject_nan = option(options, "injectNan", False)
        inject_inf = option(options, "injectInf", False)
        init_scale = option(options, "initScale", 65536.0)
        device_type = option(options, "deviceType", "cpu")

        scaler = torch.amp.GradScaler(
            device=device_type,
            init_scale=init_scale,
        )

        with torch.autocast(device_type=device_type, dtype=torch.float16):
            out = x @ w
            if inject_nan:
                out = out + float('nan')
            if inject_inf:
                out = out * float('inf')

        loss = out.float().sum()

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        # Check if gradients have inf/nan
        found_inf = not torch.isfinite(x.grad).all() or not torch.isfinite(w.grad).all()

        return {
            "output": tensor_payload(loss),
            "foundInf": found_inf,
            "scale": scaler.get_scale(),
            "grads": [tensor_payload(x.grad), tensor_payload(w.grad)],
        }

    # ========================================
    # Checkpoint + AMP Combined
    # ========================================

    if op == "checkpoint_amp_forward_backward":
        x = inputs[0].float().requires_grad_(True)
        w = inputs[1].float().requires_grad_(True)
        b = inputs[2].float().requires_grad_(True) if len(inputs) > 2 else None

        num_layers = option(options, "numLayers", 2)
        use_checkpoint = option(options, "useCheckpoint", True)
        use_amp = option(options, "useAmp", True)
        device_type = option(options, "deviceType", "cpu")

        def layer_fn(x, w, b):
            out = x @ w
            if b is not None:
                out = out + b
            return torch.relu(out)

        if use_amp:
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                current = x
                for i in range(num_layers):
                    if use_checkpoint:
                        current = checkpoint(layer_fn, current, w, b, use_reentrant=False)
                    else:
                        current = layer_fn(current, w, b)
        else:
            current = x
            for i in range(num_layers):
                if use_checkpoint:
                    current = checkpoint(layer_fn, current, w, b, use_reentrant=False)
                else:
                    current = layer_fn(current, w, b)

        loss = current.float().sum()
        loss.backward()

        grads = [tensor_payload(x.grad), tensor_payload(w.grad)]
        if b is not None:
            grads.append(tensor_payload(b.grad))

        return {
            "output": tensor_payload(loss),
            "grads": grads,
        }

    # ========================================
    # Memory Trace Commands
    # ========================================

    if op == "memory_trace":
        """
        Run a function and record memory at each step.
        Returns memory snapshots after each operation.
        """
        x = inputs[0].float().requires_grad_(True)
        w = inputs[1].float().requires_grad_(True)

        num_layers = option(options, "numLayers", 3)
        use_checkpoint = option(options, "useCheckpoint", False)
        use_amp = option(options, "useAmp", False)
        device_type = option(options, "deviceType", "cpu")

        memory_snapshots = []

        def record_memory(label):
            gc.collect()
            if device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
            else:
                # For CPU, we can't easily measure memory
                allocated = 0
                reserved = 0

            memory_snapshots.append({
                "label": label,
                "allocatedBytes": allocated,
                "reservedBytes": reserved,
            })

        def layer_fn(x, w):
            return torch.relu(x @ w)

        record_memory("initial")

        if use_amp:
            with torch.autocast(device_type=device_type, dtype=torch.float16):
                current = x
                for i in range(num_layers):
                    if use_checkpoint:
                        current = checkpoint(layer_fn, current, w, use_reentrant=False)
                    else:
                        current = layer_fn(current, w)
                    record_memory(f"after_layer_{i}")
        else:
            current = x
            for i in range(num_layers):
                if use_checkpoint:
                    current = checkpoint(layer_fn, current, w, use_reentrant=False)
                else:
                    current = layer_fn(current, w)
                record_memory(f"after_layer_{i}")

        record_memory("after_forward")

        loss = current.float().sum()
        record_memory("after_loss")

        loss.backward()
        record_memory("after_backward")

        grads = [tensor_payload(x.grad), tensor_payload(w.grad)]
        return {
            "output": tensor_payload(loss),
            "grads": grads,
            "memorySnapshots": memory_snapshots,
        }

    # Memory comparison: with vs without checkpoint
    if op == "memory_comparison":
        x = inputs[0].float()
        w = inputs[1].float()

        num_layers = option(options, "numLayers", 4)
        device_type = option(options, "deviceType", "cpu")

        def layer_fn(x, w):
            return torch.relu(x @ w)

        results = {}

        # Run without checkpoint
        x1 = x.clone().requires_grad_(True)
        w1 = w.clone().requires_grad_(True)
        gc.collect()
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        current = x1
        for i in range(num_layers):
            current = layer_fn(current, w1)
        loss = current.float().sum()
        loss.backward()

        if device_type == "cuda" and torch.cuda.is_available():
            results["withoutCheckpoint"] = {
                "peakMemory": torch.cuda.max_memory_allocated(),
                "grads": [tensor_payload(x1.grad), tensor_payload(w1.grad)],
            }
        else:
            results["withoutCheckpoint"] = {
                "peakMemory": 0,
                "grads": [tensor_payload(x1.grad), tensor_payload(w1.grad)],
            }

        # Run with checkpoint
        x2 = x.clone().requires_grad_(True)
        w2 = w.clone().requires_grad_(True)
        gc.collect()
        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        current = x2
        for i in range(num_layers):
            current = checkpoint(layer_fn, current, w2, use_reentrant=False)
        loss = current.float().sum()
        loss.backward()

        if device_type == "cuda" and torch.cuda.is_available():
            results["withCheckpoint"] = {
                "peakMemory": torch.cuda.max_memory_allocated(),
                "grads": [tensor_payload(x2.grad), tensor_payload(w2.grad)],
            }
        else:
            results["withCheckpoint"] = {
                "peakMemory": 0,
                "grads": [tensor_payload(x2.grad), tensor_payload(w2.grad)],
            }

        # Verify gradients match
        grads_match = True
        for g1, g2 in zip(results["withoutCheckpoint"]["grads"], results["withCheckpoint"]["grads"]):
            t1 = torch.tensor(g1["values"]).reshape(g1["shape"])
            t2 = torch.tensor(g2["values"]).reshape(g2["shape"])
            if not torch.allclose(t1, t2, rtol=1e-5, atol=1e-6):
                grads_match = False
                break

        results["gradsMatch"] = grads_match

        # Return in expected format with output field
        return {
            "output": {"shape": [], "values": [1.0]},  # Dummy output
            "grads": [],
            "withoutCheckpoint": results["withoutCheckpoint"],
            "withCheckpoint": results["withCheckpoint"],
            "gradsMatch": results["gradsMatch"],
        }

    output = apply_op(op, inputs, options)
    return tensor_payload(output)


class SimpleGPT2(nn.Module):
    """Minimal GPT-2 implementation matching torchlette structure."""

    def __init__(self, vocab_size, block_size, embed_dim, num_layers, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Embeddings
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.wpe = nn.Embedding(block_size, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(embed_dim)

    def forward(self, idx):
        batch, seq_len = idx.shape

        # Position indices
        pos = torch.arange(seq_len, device=idx.device).unsqueeze(0)

        # Embeddings
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # LM head (weight tied with token embeddings)
        logits = x @ self.wte.weight.T

        return logits

    def forward_with_checkpoint(self, idx):
        """Forward pass with gradient checkpointing on each transformer block."""
        batch, seq_len = idx.shape

        # Position indices
        pos = torch.arange(seq_len, device=idx.device).unsqueeze(0)

        # Embeddings
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb

        # Transformer blocks with checkpointing
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False)

        # Final layer norm
        x = self.ln_f(x)

        # LM head (weight tied with token embeddings)
        logits = x @ self.wte.weight.T

        return logits


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Combined QKV projection
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        # QKV projection
        qkv = self.c_attn(x)  # [batch, seq, 3*embed]
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = (q @ k.transpose(-2, -1)) * scale  # [batch, heads, seq, seq]

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask.bool(), float('-inf'))

        # Softmax and weighted sum
        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # [batch, heads, seq, head_dim]

        # Concatenate heads
        out = out.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, self.embed_dim)

        # Output projection
        return self.c_proj(out)


class MLP(nn.Module):
    """Two-layer MLP with GELU (tanh approximation to match torchlette)."""

    def __init__(self, embed_dim):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')  # Use tanh approximation to match torchlette default
        x = self.c_proj(x)
        return x


def process_request(data):
    """Process a batch request and return {results: [...]}."""
    cases = data.get("cases", [])
    results = []

    for case in cases:
        case_name = case.get("caseName")
        try:
            output = run_case(case)
            if isinstance(output, dict) and "output" in output:
                # Build result with all fields from the output
                result = {
                    "ok": True,
                    "output": output.get("output"),
                    "grads": output.get("grads"),
                    "caseName": case_name,
                }
                # Pass through additional fields for checkpoint/AMP commands
                for key in ["memorySnapshots", "scale", "scaledOutput", "foundInf",
                            "withoutCheckpoint", "withCheckpoint", "gradsMatch",
                            "activations"]:
                    if key in output:
                        result[key] = output[key]
                results.append(result)
            else:
                results.append(
                    {"ok": True, "output": output, "caseName": case_name}
                )
        except Exception as exc:
            results.append(
                {"ok": False, "error": str(exc), "caseName": case_name}
            )

    return {"results": results}


def main():
    payload = sys.stdin.read()
    if not payload.strip():
        raise RuntimeError("No JSON payload provided to torch oracle.")

    data = json.loads(payload)
    response = process_request(data)
    print(json.dumps(response))


def server_main():
    """Persistent server mode: NDJSON over stdin/stdout."""
    sys.stdout.write('{"ready":true}\n')
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            response = process_request(data)
        except Exception as exc:
            response = {"results": [{"ok": False, "error": str(exc)}]}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
        gc.collect()


if __name__ == "__main__":
    if "--server" in sys.argv:
        server_main()
    else:
        main()
