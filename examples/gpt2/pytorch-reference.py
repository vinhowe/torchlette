#!/usr/bin/env python3
"""
PyTorch Reference Script for GPT-2 Benchmarking

This script runs GPT-2 forward and backward passes with PyTorch
and outputs JSON results for comparison with Torchlette.

Usage:
    python pytorch-reference.py <command> [args...]

Commands:
    benchmark   Run forward/backward timing benchmark
    forward     Run single forward pass and output logits
    verify      Run forward pass and output detailed values for verification
"""

import argparse
import json
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_model(config: dict, device: torch.device) -> GPT2LMHeadModel:
    """Create GPT-2 model from config."""
    gpt2_config = GPT2Config(
        vocab_size=config.get("vocabSize", 50257),
        n_positions=config.get("blockSize", 1024),
        n_embd=config.get("embedDim", 768),
        n_layer=config.get("numLayers", 12),
        n_head=config.get("numHeads", 12),
        resid_pdrop=0.0,  # Disable dropout for deterministic comparison
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(gpt2_config)
    model.to(device)
    return model


def benchmark_command(args):
    """Run forward/backward timing benchmark."""
    device = get_device()

    config = {
        "vocabSize": args.vocab_size,
        "blockSize": args.block_size,
        "embedDim": args.embed_dim,
        "numLayers": args.num_layers,
        "numHeads": args.num_heads,
    }

    model = create_model(config, device)
    model.train()

    # Create random input
    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, config["vocabSize"],
        (args.batch_size, args.seq_length),
        device=device
    )
    targets = torch.randint(
        0, config["vocabSize"],
        (args.batch_size, args.seq_length),
        device=device
    )

    # Warmup
    for _ in range(args.warmup):
        outputs = model(input_ids, labels=targets)
        loss = outputs.loss
        loss.backward()
        model.zero_grad()

    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    # Measure forward pass
    forward_times = []
    for _ in range(args.iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        outputs = model(input_ids)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        forward_times.append((time.perf_counter() - start) * 1000)

    # Measure backward pass
    backward_times = []
    for _ in range(args.iterations):
        outputs = model(input_ids, labels=targets)
        loss = outputs.loss

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        backward_times.append((time.perf_counter() - start) * 1000)
        model.zero_grad()

    # Measure full step (forward + backward)
    total_times = []
    for _ in range(args.iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        outputs = model(input_ids, labels=targets)
        loss = outputs.loss
        loss.backward()

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        total_times.append((time.perf_counter() - start) * 1000)
        model.zero_grad()

    def stats(times):
        times = sorted(times)
        return {
            "median": times[len(times) // 2],
            "mean": sum(times) / len(times),
            "min": min(times),
            "max": max(times),
        }

    result = {
        "device": str(device),
        "config": config,
        "batchSize": args.batch_size,
        "seqLength": args.seq_length,
        "warmupIterations": args.warmup,
        "measureIterations": args.iterations,
        "forwardMs": stats(forward_times),
        "backwardMs": stats(backward_times),
        "totalMs": stats(total_times),
        "paramCount": sum(p.numel() for p in model.parameters()),
    }

    print(json.dumps(result))


def forward_command(args):
    """Run single forward pass and output logits."""
    device = get_device()

    config = {
        "vocabSize": args.vocab_size,
        "blockSize": args.block_size,
        "embedDim": args.embed_dim,
        "numLayers": args.num_layers,
        "numHeads": args.num_heads,
    }

    model = create_model(config, device)
    model.eval()

    # Parse input tokens
    input_ids = torch.tensor(
        [int(x) for x in args.tokens.split(",")],
        device=device
    ).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Get logits for last position
    last_logits = logits[0, -1, :].cpu().tolist()

    # Return top-k predictions and sample of logits
    top_k = 10
    top_values, top_indices = torch.topk(logits[0, -1, :], top_k)

    result = {
        "shape": list(logits.shape),
        "topK": {
            "indices": top_indices.cpu().tolist(),
            "values": top_values.cpu().tolist(),
        },
        "sampleLogits": last_logits[:100],  # First 100 logits for verification
    }

    print(json.dumps(result))


def verify_command(args):
    """Run forward pass with detailed output for verification."""
    device = get_device()

    config = {
        "vocabSize": args.vocab_size,
        "blockSize": args.block_size,
        "embedDim": args.embed_dim,
        "numLayers": args.num_layers,
        "numHeads": args.num_heads,
    }

    # Use smaller model for verification
    if args.small:
        config = {
            "vocabSize": 1000,
            "blockSize": 128,
            "embedDim": 64,
            "numLayers": 2,
            "numHeads": 2,
        }

    model = create_model(config, device)
    model.eval()

    # Deterministic input
    torch.manual_seed(args.seed)
    input_ids = torch.randint(
        0, config["vocabSize"],
        (1, args.seq_length),
        device=device
    )

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states

    result = {
        "config": config,
        "inputShape": list(input_ids.shape),
        "inputTokens": input_ids[0, :10].cpu().tolist(),  # First 10 tokens
        "logitsShape": list(logits.shape),
        "logitsSample": logits[0, -1, :20].cpu().tolist(),  # First 20 logits of last position
        "hiddenStatesCount": len(hidden_states),
        "embeddingOutputSample": hidden_states[0][0, 0, :10].cpu().tolist(),  # First 10 values
        "finalHiddenSample": hidden_states[-1][0, -1, :10].cpu().tolist(),  # Last layer, last position
    }

    # If targets provided, compute loss
    if args.with_loss:
        targets = torch.randint(
            0, config["vocabSize"],
            (1, args.seq_length),
            device=device
        )
        outputs_with_loss = model(input_ids, labels=targets)
        result["loss"] = outputs_with_loss.loss.item()
        result["targetTokens"] = targets[0, :10].cpu().tolist()

    print(json.dumps(result))


def weights_command(args):
    """Output model weights summary and samples for verification."""
    device = torch.device("cpu")  # Use CPU for weight extraction

    config = {
        "vocabSize": args.vocab_size,
        "blockSize": args.block_size,
        "embedDim": args.embed_dim,
        "numLayers": args.num_layers,
        "numHeads": args.num_heads,
    }

    # Load pretrained weights
    if args.pretrained:
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    else:
        model = create_model(config, device)

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = {
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "sample": param.flatten()[:10].tolist(),  # First 10 values
            "mean": param.mean().item(),
            "std": param.std().item(),
        }

    result = {
        "config": config,
        "pretrained": args.pretrained,
        "paramCount": sum(p.numel() for p in model.parameters()),
        "weights": weights,
    }

    print(json.dumps(result))


def interactive_command(args):
    """Interactive mode for continuous communication."""
    device = get_device()
    model = None
    config = None

    for line in sys.stdin:
        try:
            request = json.loads(line.strip())
            command = request.get("command")

            if command == "init":
                config = request.get("config", {})
                model = create_model(config, device)
                if request.get("pretrained"):
                    model = GPT2LMHeadModel.from_pretrained("gpt2")
                    model.to(device)
                model.train() if request.get("train", True) else model.eval()
                print(json.dumps({"status": "ok", "device": str(device)}))

            elif command == "forward":
                if model is None:
                    print(json.dumps({"error": "Model not initialized"}))
                    continue

                tokens = request.get("tokens")
                input_ids = torch.tensor(tokens, device=device).unsqueeze(0)

                with torch.no_grad() if not request.get("requiresGrad", False) else torch.enable_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits

                result = {
                    "shape": list(logits.shape),
                    "sample": logits[0, -1, :50].cpu().tolist(),
                }
                print(json.dumps(result))

            elif command == "forward_backward":
                if model is None:
                    print(json.dumps({"error": "Model not initialized"}))
                    continue

                tokens = request.get("tokens")
                targets = request.get("targets")
                input_ids = torch.tensor(tokens, device=device).unsqueeze(0)
                target_ids = torch.tensor(targets, device=device).unsqueeze(0)

                start = time.perf_counter()
                outputs = model(input_ids, labels=target_ids)
                loss = outputs.loss
                loss.backward()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()

                elapsed = (time.perf_counter() - start) * 1000

                result = {
                    "loss": loss.item(),
                    "timeMs": elapsed,
                }
                print(json.dumps(result))
                model.zero_grad()

            elif command == "benchmark_step":
                if model is None:
                    print(json.dumps({"error": "Model not initialized"}))
                    continue

                warmup = request.get("warmup", 3)
                iterations = request.get("iterations", 10)
                seq_length = request.get("seqLength", 128)
                batch_size = request.get("batchSize", 1)
                vocab_size = config.get("vocabSize", 50257)

                torch.manual_seed(request.get("seed", 42))
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
                targets = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

                # Warmup
                for _ in range(warmup):
                    outputs = model(input_ids, labels=targets)
                    outputs.loss.backward()
                    model.zero_grad()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()

                # Measure
                times = []
                for _ in range(iterations):
                    start = time.perf_counter()
                    outputs = model(input_ids, labels=targets)
                    outputs.loss.backward()

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elif device.type == "mps":
                        torch.mps.synchronize()

                    times.append((time.perf_counter() - start) * 1000)
                    model.zero_grad()

                times.sort()
                result = {
                    "median": times[len(times) // 2],
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "times": times,
                }
                print(json.dumps(result))

            elif command == "quit":
                print(json.dumps({"status": "bye"}))
                break

            else:
                print(json.dumps({"error": f"Unknown command: {command}"}))

        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}))
        except Exception as e:
            print(json.dumps({"error": str(e)}))

        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="PyTorch GPT-2 Reference")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--vocab-size", type=int, default=50257)
    common.add_argument("--block-size", type=int, default=1024)
    common.add_argument("--embed-dim", type=int, default=768)
    common.add_argument("--num-layers", type=int, default=12)
    common.add_argument("--num-heads", type=int, default=12)
    common.add_argument("--seed", type=int, default=42)

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", parents=[common])
    bench_parser.add_argument("--batch-size", type=int, default=1)
    bench_parser.add_argument("--seq-length", type=int, default=1024)
    bench_parser.add_argument("--warmup", type=int, default=3)
    bench_parser.add_argument("--iterations", type=int, default=10)
    bench_parser.set_defaults(func=benchmark_command)

    # Forward command
    forward_parser = subparsers.add_parser("forward", parents=[common])
    forward_parser.add_argument("--tokens", type=str, required=True)
    forward_parser.set_defaults(func=forward_command)

    # Verify command
    verify_parser = subparsers.add_parser("verify", parents=[common])
    verify_parser.add_argument("--seq-length", type=int, default=128)
    verify_parser.add_argument("--small", action="store_true")
    verify_parser.add_argument("--with-loss", action="store_true")
    verify_parser.set_defaults(func=verify_command)

    # Weights command
    weights_parser = subparsers.add_parser("weights", parents=[common])
    weights_parser.add_argument("--pretrained", action="store_true")
    weights_parser.set_defaults(func=weights_command)

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive")
    interactive_parser.set_defaults(func=interactive_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
